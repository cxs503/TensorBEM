#!/usr/bin/env python3
"""Run one refinement rung of ``BEM-HULME-HEMISPHERE`` with a solve-health guard.

Why this exists
---------------
Dev report m20910 localised the 1024-panel cross-environment drift to the
complex least-squares stage: on the cluster node's numpy 2.0.1 build the
assembled system matrix is bit-identical to the control plane's, yet the solve
returns a solution whose relative residual is ``4.9e-1`` instead of
``1.8e-4``.  Every rung of the refinement ladder that ran in that environment
is therefore void data, and m20910 explicitly left the ``subdivisions = 5``
(4096 panel) rung unmeasured: *"该档在健康环境下仍未知，需重跑"*.

This driver measures a rung **in a healthy environment only**: it solves the
same cell twice (CHIEF-augmented least squares, the graded configuration, and
the square ``chief_points=None`` control), records the relative residual of
both, and refuses to publish the rung -- exit code 3, ``status = "unhealthy"``
-- when either residual exceeds ``RESIDUAL_GUARD``.  The rung artifact is
written after every stage, so a timeout or a crash still leaves usable
evidence on disk.

Usage
-----
    PYTHONPATH=src python scripts/hulme_scale_rung.py --ka 2.0 \
        --subdivisions 4 5 --out benchmarks/verified/\
casepool_bem_hulme_hemisphere/scale_rung_ka2.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.benchmarks.hulme_hemisphere_case import (
    DENSITY,
    GRAVITY,
    RADIUS,
    reference,
)
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

# The radiation-only subset of the case's six graded quantities (this driver
# does not solve the diffraction problem, so the excitation forces are absent).
RADIATION_QUANTITIES = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless")
RESIDUAL_GUARD = 1.0e-2
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _environment() -> dict[str, Any]:
    import tensorbem

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "tensorbem_path": str(Path(tensorbem.__file__).resolve()),
        "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def _dimensionless(rad: Any, ka: float) -> dict[str, float]:
    vol = (2.0 / 3.0) * math.pi * RADIUS**3
    norm_a = DENSITY * vol
    norm_b = DENSITY * math.sqrt(GRAVITY * (ka / RADIUS)) * vol
    return {
        "A11_dimless": float(rad.added_mass[0, 0]) / norm_a,
        "B11_dimless": float(rad.damping[0, 0]) / norm_b,
        "A33_dimless": float(rad.added_mass[1, 1]) / norm_a,
        "B33_dimless": float(rad.damping[1, 1]) / norm_b,
    }


def run_rung(ka: float, subdivisions: int, ref: dict[str, float]) -> dict[str, Any]:
    t0 = time.time()
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    nu = ka / RADIUS
    row: dict[str, Any] = {
        "ka": ka,
        "subdivisions": subdivisions,
        "panels": mesh.panel_count,
        "mesh_volume": float(mesh.volume),
        "mesh_waterplane_area": float(mesh.waterplane_area),
        "nabla_exact": (2.0 / 3.0) * math.pi * RADIUS**3,
        "reference": {k: ref[k] for k in RADIATION_QUANTITIES},
        "stage_seconds": {},
        "solves": {},
    }
    t_assemble = time.time()
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    ops_square = assemble_free_surface_operators_3d(mesh, nu, chief_points=None)
    row["condition_number"] = float(ops.condition_number)
    row["square_condition_number"] = float(ops_square.condition_number)
    row["stage_seconds"]["assemble"] = time.time() - t_assemble

    for tag, op in (("chief_lstsq", ops), ("square_solve", ops_square)):
        t_solve = time.time()
        rad = solve_radiation_3d(
            mesh, nu, modes=(0, 2), density=DENSITY, gravity=GRAVITY, operators=op
        )
        values = _dimensionless(rad, ka)
        row["solves"][tag] = {
            "values": values,
            "relative_residual": float(rad.relative_residual),
            "rel_error": {
                k: (values[k] - ref[k]) / abs(ref[k])
                for k in RADIATION_QUANTITIES
            },
            "seconds": time.time() - t_solve,
        }
        row["stage_seconds"][tag] = time.time() - t_solve

    residuals = [row["solves"][t]["relative_residual"] for t in row["solves"]]
    row["max_relative_residual"] = float(max(residuals))
    row["healthy"] = bool(max(residuals) <= RESIDUAL_GUARD)
    row["wall_seconds"] = time.time() - t0
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ka", type=float, default=2.0)
    ap.add_argument("--subdivisions", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "case_code": "BEM-HULME-HEMISPHERE",
        "purpose": (
            "healthy-environment refinement rung: the case's graded mesh "
            "(subdivisions=4) and the rung the ladder never measured "
            "(subdivisions=5), each solved twice (CHIEF least squares and "
            "square) with a residual guard so an environment with a broken "
            "complex LAPACK cannot contribute data"
        ),
        "residual_guard": RESIDUAL_GUARD,
        "ka": args.ka,
        "environment": _environment(),
        "rows": [],
    }
    ref = reference(args.ka, n_terms=144)
    exit_code = 0
    for sd in args.subdivisions:
        try:
            row = run_rung(args.ka, sd, ref)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            payload["rows"].append(
                {"subdivisions": sd, "error": f"{type(exc).__name__}: {exc}"}
            )
            out.write_text(json.dumps(payload, indent=1) + "\n")
            raise
        payload["rows"].append(row)
        out.write_text(json.dumps(payload, indent=1) + "\n")  # crash-safe
        if not row["healthy"]:
            exit_code = 3
        print(
            f"sd={sd} panels={row['panels']} cond={row['condition_number']:.6f} "
            f"resid_max={row['max_relative_residual']:.3e} "
            f"healthy={row['healthy']} wall={row['wall_seconds']:.1f}s",
            flush=True,
        )
        for tag in ("chief_lstsq", "square_solve"):
            vals = row["solves"][tag]["values"]
            print(
                "   ",
                tag,
                " ".join(
                    f"{k.split('_')[0]}={vals[k]:.6f}" for k in RADIATION_QUANTITIES
                ),
                flush=True,
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())