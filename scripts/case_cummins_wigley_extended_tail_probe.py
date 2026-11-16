#!/usr/bin/env python3
"""Certified-mesh probe of the *top* of the extended band (the nodes C9 reads).

Why this exists
---------------
Gate ``C9`` (:func:`tensorbem.benchmarks.case_bem_cummins_wigley.truncation_decidability_gate`)
asks one question about the 12 -> 20 rad/s extension: does the radiation damping of
surge / sway / yaw enter algebraic decay (measured slope ``p > 2.05``) at the new
band top, so that the omitted high-band integral in the Kramers-Kronig closure
becomes bounded?  That slope is taken over the *last few* samples of the band
(``local_tail_exponent(..., window=3)``), so the verdict is carried by the top
nodes alone.

The full certified sweep is 55 nodes x 3840 panels and belongs on the cluster
(``scripts/case_cummins_wigley_extended_band.py``).  The top four nodes are
cheap enough to solve directly (~3 min single core each), and they answer C9's
*physics* question on the **C2-admissible** mesh -- which the old probe artifacts
(``sweep_extended_probe_*.npz``, 192/480 panels) explicitly cannot, because they
violate ``k*sqrt(A_panel) <= 1`` above ~10 resp. ~12 rad/s.

So this is a probe, not a certification: ``C8`` (shared-band consistency) and
``C3a_extended`` (anchored closure over the whole 0.2..20 rad/s band) still need
the assembled full sweep.  The nodes solved here are written with the band
script's own :func:`solve_point`, same names and layout, so a later assembly
reuses them and nothing is thrown away.

Usage::

    PYTHONPATH=src python3 scripts/case_cummins_wigley_extended_tail_probe.py --indices 51,52,53,54
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402
from tensorbem.causality import (  # noqa: E402
    high_band_truncation_budget,
    local_tail_exponent,
)

OUT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
REPORT = OUT / "extended_tail_probe_certified.json"
EXTENDED_MESH = (60, 16)
EXTENDED_BAND = (0.2, 20.0)
EXTENDED_N_OMEGA = 55
#: Nodes of the pre-declared 55-node extended grid that carry C9's slope.
DEFAULT_INDICES = (51, 52, 53, 54)


def _band_module():
    """The shard solver lives in ``scripts/``; import it by path, not by name."""
    path = ROOT / "scripts" / "case_cummins_wigley_extended_band.py"
    spec = importlib.util.spec_from_file_location("extended_band", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices",
        default=",".join(str(i) for i in DEFAULT_INDICES),
        help="comma separated node indices of the extended 55-node grid",
    )
    parser.add_argument("--output", default=str(REPORT))
    args = parser.parse_args(argv)

    indices = [int(part) for part in args.indices.split(",") if part.strip()]
    grid = case.sweep_grid(EXTENDED_BAND, EXTENDED_N_OMEGA)
    band = _band_module()
    for index in indices:
        if not 0 <= index < grid.size:
            parser.error(f"index {index} outside the {grid.size}-node grid")

    started = time.time()
    omegas: list[float] = []
    damping: list[np.ndarray] = []
    added_mass: list[np.ndarray] = []
    rows = []
    panel_count = None
    panel_scale = None
    residual = float("nan")
    for index in indices:
        path = band.point_path(index)
        reused = path.exists()
        if not reused:
            solve_started = time.time()
            path = band.solve_point(index)
            solve_wall = time.time() - solve_started
        else:
            solve_wall = 0.0
        payload = np.load(path, allow_pickle=False)
        omega = float(payload["omega"])
        damper = np.asarray(payload["damping"], dtype=float)
        mass = np.asarray(payload["added_mass"], dtype=float)
        residual = float(payload["solver_relative_residual"])
        panel_count = int(payload["panel_count"])
        panel_scale = float(payload["max_panel_scale"])
        mesh = (int(payload["n_sections"]), int(payload["n_levels"]))
        if mesh != EXTENDED_MESH:
            raise SystemExit(f"node {index} was solved on {mesh}, not {EXTENDED_MESH}")
        print(
            f"node {index:3d}  omega = {omega:8.4f}  residual = {residual:.3e}  "
            f"{'reused' if reused else 'solved'}  ({time.time() - started:.0f}s)",
            flush=True,
        )
        omegas.append(omega)
        damping.append(damper)
        added_mass.append(mass)
        rows.append(
            {
                "index": int(index),
                "omega": omega,
                "reused": reused,
                "solve_wall_s": float(solve_wall),
            }
        )

    order = np.argsort(np.asarray(omegas, dtype=float))
    omegas_arr = np.asarray(omegas, dtype=float)[order]
    damping_arr = np.stack(damping, axis=0)[order]
    added_mass_arr = np.stack(added_mass, axis=0)[order]

    per_dof = []
    for index, name in enumerate(case.DOF_NAMES):
        diag = damping_arr[:, index, index]
        exponent3, _coef3 = local_tail_exponent(omegas_arr, diag, window=3)
        exponent4, _coef4 = local_tail_exponent(omegas_arr, diag, window=4)
        budget = high_band_truncation_budget(omegas_arr, diag)
        per_dof.append(
            {
                "dof": int(index),
                "dof_name": name,
                "B_at_band_top": float(diag[-1]),
                "exponent_at_20_window3": float(exponent3),
                "exponent_at_20_window4": float(exponent4),
                "budget_admissible": bool(budget.admissible),
                "budget_reason": budget.reason,
                "added_mass_at_20": float(added_mass_arr[-1, index, index]),
            }
        )

    report = {
        "probe": (
            "certified-mesh (60x16, 3840 panels) nodes at the top of the extended "
            "band; these carry gate C9's slope. C8 / C3a_extended still need the "
            "assembled full 0.2..20 rad/s sweep, so this is diagnosis, not the "
            "certification of the case."
        ),
        "mesh": {
            "n_sections": EXTENDED_MESH[0],
            "n_levels": EXTENDED_MESH[1],
            "panel_count": panel_count,
            "max_panel_scale": panel_scale,
        },
        "band": [float(omegas_arr[0]), float(omegas_arr[-1])],
        "nodes": rows,
        "max_solver_relative_residual": residual,
        "exponent_bar": 2.05,
        "per_dof": per_dof,
        "elapsed_s": float(time.time() - started),
    }
    pathlib.Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output}")
    for row in per_dof:
        print(
            f"{row['dof_name']:6s} B(20) = {row['B_at_band_top']:10.4f}  "
            f"p(20) = {row['exponent_at_20_window3']:6.3f} "
            f"(w=4 {row['exponent_at_20_window4']:6.3f})  "
            f"algebraic = {row['budget_admissible']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())