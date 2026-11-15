#!/usr/bin/env python3
"""Environment probe: which factor decides the ka = 3.5 sign flip of the anchor?

Two committed artifacts of the *same* code at the *same* ``numpy`` version
disagree at ``ka = 3.5`` (1024 panels, surge/heave):

* ``band_extension/results.json`` -- metadata ``numpy 2.4.6``, host
  ``ec6ae0b42a0c``: ``A11 = -0.006802``, ``B33 = -0.000604`` (sign flipped,
  ``|err| = 104 %``), ``cond = 2.78``;
* ``evidence/numpy2_reverification.json`` -- claims ``numpy 2.4.6``:
  ``A11 = 0.16184``, ``B33 = 0.03193`` (``|err| < 3 %``), ``cond = 2.78``.

Same condition number means same matrix, so the matrix is not the explanation
and the *solution* is.  The two runs differ in the one thing neither artifact
records: the BLAS thread environment, and the entry point (the case driver
``run_cell`` vs a hand-assembled radiation + diffraction solve).  This probe
varies exactly those two factors and records, per arm, the solve-stage
residual next to the value -- because the residual is what separates "the
driver did not solve its system" (big residual) from "the driver solved a
wrong system" (small residual, wrong value, i.e. a blind spot of the E0
admissibility gate).

Output: one JSON object per line, prefixed ``PROBE``, plus a summary JSON file
when ``--out`` is given.  Nothing here writes into the case artifacts.

Usage
-----
    python benchmarks/verified/casepool_bem_hulme_hemisphere/probe_ka35_env.py \
        --mode cell --repeat 2
    python benchmarks/verified/casepool_bem_hulme_hemisphere/probe_ka35_env.py \
        --mode direct --out /tmp/direct.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

KA = 3.5
SUBDIVISIONS = 4
RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
AMPLITUDE = 1.0
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def _cell() -> dict[str, Any]:
    """Arm 1: the graded case driver (same entry point as the band run)."""
    from tensorbem.benchmarks.hulme_hemisphere_case import run_cell

    row = run_cell(KA, SUBDIVISIONS)
    return {
        "entry": "case_driver.run_cell",
        "ka": KA,
        "panels": row["panels"],
        "chief": row["chief"],
        "condition_number": row["condition_number"],
        "radiation_residual": row["radiation_relative_residual"],
        "diffraction_residual": row["diffraction_relative_residual"],
        "A11": row["A11_dimless"],
        "A33": row["A33_dimless"],
        "B11": row["B11_dimless"],
        "B33": row["B33_dimless"],
        "Fx": row["Fx_dimless"],
        "Fz": row["Fz_dimless"],
        "ref_A11": row["reference"]["A11_dimless"],
        "ref_B33": row["reference"]["B33_dimless"],
        "solve_warnings": row.get("solve_warnings", []),
    }


def _direct() -> dict[str, Any]:
    """Arm 2: hand-assembled radiation + diffraction, as the m21563 test does."""
    from tensorbem.analytic import hulme_hemisphere_radiation
    from tensorbem.free_surface_solver3d import (
        assemble_free_surface_operators_3d,
        solve_diffraction_3d,
        solve_radiation_3d,
    )
    from tensorbem.geometry3d import hemisphere_hull_mesh

    nu = KA / RADIUS
    mesh = hemisphere_hull_mesh(SUBDIVISIONS, radius=RADIUS)
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    rad = solve_radiation_3d(mesh, nu, modes=(0, 2), density=DENSITY,
                             gravity=GRAVITY, operators=ops)
    diff = solve_diffraction_3d(mesh, nu, amplitude=AMPLITUDE, wave_angle=0.0,
                                density=DENSITY, gravity=GRAVITY, operators=ops)
    vol = (2.0 / 3.0) * math.pi * RADIUS ** 3
    norm_a = DENSITY * vol
    norm_b = DENSITY * math.sqrt(GRAVITY * nu) * vol
    scale_f = DENSITY * GRAVITY * AMPLITUDE * RADIUS ** 2
    ref = hulme_hemisphere_radiation(KA, radius=RADIUS, amplitude=AMPLITUDE,
                                     density=DENSITY, gravity=GRAVITY,
                                     n_terms=144, quad_order=288)
    return {
        "entry": "direct.assemble+solve",
        "ka": KA,
        "panels": mesh.panel_count,
        "chief": True,
        "condition_number": float(ops.condition_number),
        "radiation_residual": float(rad.relative_residual),
        "diffraction_residual": float(diff.relative_residual),
        "A11": float(rad.added_mass[0, 0]) / norm_a,
        "A33": float(rad.added_mass[1, 1]) / norm_a,
        "B11": float(rad.damping[0, 0]) / norm_b,
        "B33": float(rad.damping[1, 1]) / norm_b,
        "Fx": abs(complex(diff.excitation_force[0]) / scale_f),
        "Fz": abs(complex(diff.excitation_force[2]) / scale_f),
        "ref_A11": float(ref.surge_added_mass_dimless),
        "ref_B33": float(ref.heave_damping_dimless),
        "solve_warnings": [],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("cell", "direct"), default="cell")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    fn = _cell if args.mode == "cell" else _direct
    rows: list[dict[str, Any]] = []
    for rep in range(args.repeat):
        t0 = time.time()
        row = fn()
        row.update({
            "repeat": rep,
            "wall_seconds": round(time.time() - t0, 1),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "host": platform.node(),
            "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
            "sign_flip_A11": bool(row["A11"] * row["ref_A11"] < 0),
            "sign_flip_B33": bool(row["B33"] * row["ref_B33"] < 0),
            "err_A11": abs(row["A11"] / row["ref_A11"] - 1.0),
            "err_B33": abs(row["B33"] / row["ref_B33"] - 1.0),
        })
        rows.append(row)
        print("PROBE " + json.dumps(row, sort_keys=True), flush=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"ka": KA, "rows": rows}, indent=2,
                                       sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())