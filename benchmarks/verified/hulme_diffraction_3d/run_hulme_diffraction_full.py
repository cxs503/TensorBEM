#!/usr/bin/env python3
"""Hulme (1982) floating-hemisphere diffraction verification — full cluster run.

Runs all ka × sub combinations including sub=4 (1024 panels).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import solve_diffraction_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
AMPLITUDE = 1.0
WAVE_ANGLE = 0.0
TOLERANCE = 0.05

KA_GRID = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
SUB_LEVELS = (2, 3, 4)  # 64, 256, 1024 panels

DOF_LABELS = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
ACTIVE_DOFS = [0, 2]

ARTIFACT_DIR = Path("benchmarks/verified/hulme_diffraction_3d")


def run_diffraction(ka: float, sub: int) -> dict[str, Any]:
    mesh = hemisphere_hull_mesh(sub, radius=RADIUS)
    res = solve_diffraction_3d(
        mesh, nu=ka, amplitude=AMPLITUDE, wave_angle=WAVE_ANGLE,
        density=DENSITY, gravity=GRAVITY,
    )
    row: dict[str, Any] = {
        "ka": ka, "subdivisions": sub, "panels": int(mesh.faces.shape[0]),
        "omega": float(res.omega), "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
    }
    for dof_idx in range(6):
        label = DOF_LABELS[dof_idx]
        f = res.excitation_force[dof_idx]
        row[f"F_{label}_mag"] = float(abs(f))
        row[f"F_{label}_phase"] = float(np.angle(f))
    return row


def run_reference(ka: float) -> dict[str, Any]:
    ref = hulme_hemisphere_radiation(ka, radius=RADIUS, density=DENSITY, gravity=GRAVITY)
    row: dict[str, Any] = {"ka": ka, "omega": float(ref.omega)}
    for dof_idx, label in enumerate(DOF_LABELS):
        if dof_idx == 0:
            f = ref.excitation_force_x
        elif dof_idx == 2:
            f = ref.excitation_force_z
        else:
            f = complex(0.0, 0.0)
        row[f"F_{label}_mag"] = float(abs(f))
        row[f"F_{label}_phase"] = float(np.angle(f))
    return row


def compare(got: dict, ref: dict) -> dict:
    entries = []
    worst = 0.0
    for dof_idx in ACTIVE_DOFS:
        label = DOF_LABELS[dof_idx]
        got_mag = got[f"F_{label}_mag"]
        ref_mag = ref[f"F_{label}_mag"]
        rel = (got_mag - ref_mag) / ref_mag if ref_mag > 1e-12 else 0.0
        entries.append({"dof": label, "got": got_mag, "ref": ref_mag, "rel_error": rel, "pass": abs(rel) < TOLERANCE})
        worst = max(worst, abs(rel))
    return {"entries": entries, "worst_rel_error": worst, "pass": worst < TOLERANCE}


def main() -> None:
    out_dir = ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Hulme 3D diffraction verification (full: sub=2,3,4)...")
    comparison_rows = []
    all_rows = []
    verdicts = []

    for ka in KA_GRID:
        ref = run_reference(ka)
        for sub in SUB_LEVELS:
            print(f"  ka={ka}, sub={sub}...", flush=True)
            got = run_diffraction(ka, sub)
            comp = compare(got, ref)
            row = {**got}
            row["rel_error_surge"] = comp["entries"][0]["rel_error"]
            row["rel_error_heave"] = comp["entries"][1]["rel_error"]
            row["pass_surge"] = comp["entries"][0]["pass"]
            row["pass_heave"] = comp["entries"][1]["pass"]
            row["worst_rel_error"] = comp["worst_rel_error"]
            row["pass"] = comp["pass"]
            all_rows.append(row)
            if sub == SUB_LEVELS[-1]:
                comparison_rows.append({
                    "ka": ka, "panels": got["panels"],
                    "surge": comp["entries"][0], "heave": comp["entries"][1],
                    "worst_rel_error": comp["worst_rel_error"], "pass": comp["pass"],
                })
                verdicts.append(comp["pass"])

    all_pass = all(verdicts)
    study = {
        "study": "hulme_diffraction_3d_full",
        "tolerance": TOLERANCE, "all_pass": all_pass,
        "comparison": comparison_rows, "full_results": all_rows,
        "verdict": "ALL PASS" if all_pass else "SOME FAIL",
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(study, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print(f"Verdict: {study['verdict']}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
