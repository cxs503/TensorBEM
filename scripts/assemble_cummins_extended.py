#!/usr/bin/env python3
"""Assemble extended-band sweep artifacts and update case_result.json with C8/C9 gates.

Usage:
    PYTHONPATH=src python3 scripts/assemble_cummins_extended.py

This script is designed to run AFTER the cluster jobs cj3824/cj3825 complete
and deposit their point_*.npz artifacts in the extended_points/ directory.

It performs:
1. Assembly: reads point_*.npz files, concatenates into sweep_extended.npz
2. Anchor: reads anchor_ladder_extended.json for the A_inf anchor
3. Gate evaluation: calls gates_from_artifact to compute C8/C9/C3a_extended
4. Update: writes updated case_result.json

If the extended sweep artifact already exists, it re-evaluates gates (idempotent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure the worktree src is on the path
WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

from tensorbem.benchmarks.case_bem_cummins_wigley import (
    CASE_SLUG,
    Sweep,
    gates_from_artifact,
    load_sweep,
    save_sweep,
)

CASE_DIR = WORKTREE / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
EXTENDED_POINTS_DIR = CASE_DIR / "extended_points"
EXTENDED_SWEEP_PATH = CASE_DIR / "sweep_extended.npz"
ANCHOR_PATH = CASE_DIR / "anchor_ladder_extended.json"
CASE_RESULT_PATH = CASE_DIR / "case_result.json"


def assemble_extended_sweep(points_dir: Path, output_path: Path) -> Sweep:
    """Assemble point_*.npz files into a single sweep_extended.npz.

    Each point_*.npz file contains the radiation solution for one frequency.
    The files are sorted by frequency and concatenated.
    """
    if not points_dir.exists():
        raise FileNotFoundError(
            f"Extended points directory not found: {points_dir}\n"
            "Wait for cluster jobs cj3824/cj3825 to complete first."
        )

    point_files = sorted(points_dir.glob("point_*.npz"))
    if not point_files:
        raise FileNotFoundError(
            f"No point_*.npz files found in {points_dir}\n"
            "Wait for cluster jobs cj3824/cj3825 to complete first."
        )

    # Load first file to get shape
    first = np.load(point_files[0], allow_pickle=True)
    n_freq = len(point_files)
    n_dof = 6

    # Read metadata from first file
    omega = first["omega"]  # scalar
    n_sections = int(first["n_sections"])
    n_levels = int(first["n_levels"])
    panel_count = int(first["panel_count"])
    max_panel_scale = float(first["max_panel_scale"])

    # Allocate arrays
    added_mass = np.zeros((n_freq, n_dof, n_dof))
    damping = np.zeros((n_freq, n_dof, n_dof))
    residuals = np.zeros(n_freq)
    omega_array = np.zeros(n_freq)

    for i, pf in enumerate(point_files):
        data = np.load(pf, allow_pickle=True)
        omega_array[i] = float(data["omega"])
        added_mass[i] = data["added_mass"]  # 6x6
        damping[i] = data["damping"]  # 6x6
        residuals[i] = float(data.get("residual", 0.0))

    sweep = Sweep(
        label="extended",
        n_sections=n_sections,
        n_levels=n_levels,
        panel_count=panel_count,
        max_panel_scale=max_panel_scale,
        omega=omega_array,
        added_mass=added_mass,
        damping=damping,
        solver_relative_residual=residuals,
        elapsed=0.0,
    )

    save_sweep(sweep, output_path)
    print(f"Assembled {n_freq} frequency points -> {output_path}")
    print(f"  Mesh: {n_sections}x{n_levels} ({panel_count} panels)")
    print(f"  Band: [{omega_array[0]:.2f}, {omega_array[-1]:.2f}] rad/s")
    return sweep


def update_case_result(case_result_path: Path, sweep_path: Path) -> dict:
    """Re-evaluate gates with the extended sweep and update case_result.json."""
    result = gates_from_artifact(str(sweep_path))

    # Read existing case_result.json to preserve metadata
    if case_result_path.exists():
        existing = json.loads(case_result_path.read_text(encoding="utf-8"))
    else:
        existing = {}

    # Update gates and verdict
    existing["gates"] = result["gates"]
    existing["verdict"] = result["verdict"]
    existing["extended"] = result.get("extended")
    existing["generated_utc"] = __import__("time").strftime(
        "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
    )
    existing["mission"] = "m22423"

    case_result_path.write_text(
        json.dumps(existing, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {case_result_path}")
    print(f"  Verdict: {result['verdict']['gates_passed']}/{result['verdict']['gates_total']} gates passed")

    # Print extended gates
    for gate_key in ["C8", "C9", "C3a_extended"]:
        gate = result["gates"].get(gate_key)
        if gate is not None:
            print(f"  {gate_key}: passed={gate['passed']}, status={gate.get('status', '?')}")
        else:
            print(f"  {gate_key}: not evaluated")

    return result


def main():
    print("=" * 60)
    print("CASE-BEM-CUMMINS-WIGLEY: Extended Band Assembly")
    print("=" * 60)

    # Step 1: Check if extended sweep already exists
    if EXTENDED_SWEEP_PATH.exists():
        print(f"Extended sweep already exists: {EXTENDED_SWEEP_PATH}")
        print("Re-evaluating gates...")
        sweep = load_sweep(EXTENDED_SWEEP_PATH)
    else:
        print("Assembling extended sweep from point files...")
        sweep = assemble_extended_sweep(EXTENDED_POINTS_DIR, EXTENDED_SWEEP_PATH)

    # Step 2: Update case_result.json
    print("\nUpdating case_result.json with C8/C9 gates...")
    result = update_case_result(CASE_RESULT_PATH, EXTENDED_SWEEP_PATH)

    print("\n" + "=" * 60)
    print("Assembly complete.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
