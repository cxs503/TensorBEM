#!/usr/bin/env python3
"""Board-audit m17339: independent replay of ONE BEM panel rung of CASE-BEM-CYL-DRIFT-ASYMP.

Why: the registered benchmark row (tensor_benchmark_cases.id=43) quotes
"BEM far-field (Kochin->Maruo) relative error vs closed form" numbers that were
produced by mission m17204 on branch agent/m17204 (case driver not merged).  This
audit re-derives the cheapest rung (208 panels, ka = 0.5) from **main-tree code
only** -- `tensorbem.benchmarks.drift_far_field_3d_study.run_case` (panel BEM ->
Kochin -> Maruo) plus `tensorbem.cylinder_drift` (closed form) -- and compares the
result with the delivered artifact snapshot in this directory.

Usage (from the repository root):
    PYTHONPATH=src python3 docs/board_audit/m17339/replay_bem_cell.py \
        --out docs/board_audit/m17339/replay/bem_cell_208.json

Cost: ~7-10 s (one 208-panel solve).  No cluster needed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tensorbem import cylinder_drift as cd
from tensorbem.benchmarks.drift_far_field_3d_study import run_case

HERE = Path(__file__).resolve().parent
SNAP = HERE / "snapshots" / "m17204_results.json"

DENSITY = float(cd.DEFAULT_DENSITY)
GRAVITY = float(cd.DEFAULT_GRAVITY)
KA = 0.5
MESH = (16, 6)  # 2*16*6 + 1 = 193? -> panels reported by run_case; delivered rung = 208
SCALE = DENSITY * GRAVITY * 1.0**2 * 1.0  # rho g A^2 a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "replay" / "bem_cell_208.json"))
    args = ap.parse_args()

    snap = json.loads(SNAP.read_text())
    delivered = next(r for r in snap["convergence"]["rows"] if r["panels"] == 208)

    t0 = time.time()
    r = run_case(KA, n_theta=MESH[0], n_z=MESH[1], growth=1.35,
                 density=DENSITY, gravity=GRAVITY, n_azimuth=180)
    wall = time.time() - t0

    cf = cd.far_field_cylinder_drift(KA)["mean_drift_force_x"]
    cfn = cd.near_field_cylinder_drift(KA)["mean_drift_force_x"]
    far_err = abs(r["far_field_x"] - cf) / cf
    near_err = abs(r["near_field_x"] - cfn) / cfn

    out = {
        "check": "independent replay of one BEM panel rung from main-tree code",
        "anchor_slug": "CASE-BEM-CYL-DRIFT-ASYMP",
        "audit_mission": 17339,
        "source_mission": 17204,
        "code_path": "src/tensorbem/benchmarks/drift_far_field_3d_study.py run_case -> maruo_far_field_drift",
        "ka": KA,
        "mesh_n_theta_n_z": list(MESH),
        "panels": int(r["panels"]),
        "closed_form_far_N": cf,
        "closed_form_near_N": cfn,
        "replay_far_N": float(r["far_field_x"]),
        "replay_near_N": float(r["near_field_x"]),
        "replay_far_error": far_err,
        "replay_near_error": near_err,
        "delivered_far_error": delivered["far_error"],
        "delivered_near_error": delivered["near_error"],
        "far_abs_diff": abs(far_err - delivered["far_error"]),
        "near_abs_diff": abs(near_err - delivered["near_error"]),
        "solver_relative_residual": float(r["solver_relative_residual"]),
        "energy_identity_residual": float(r["energy_residual"]),
        "wall_seconds": wall,
        "replay_matches_delivered_to_1e-12": bool(
            abs(far_err - delivered["far_error"]) < 1e-12
            and abs(near_err - delivered["near_error"]) < 1e-12
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out["replay_matches_delivered_to_1e-12"] else 1


if __name__ == "__main__":
    raise SystemExit(main())