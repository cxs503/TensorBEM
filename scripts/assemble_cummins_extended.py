"""Assemble extended-band sweep artifacts and evaluate C8/C9 gates.

Usage (after cluster jobs cj3824/cj3825 complete)::

    PYTHONPATH=src python3 scripts/assemble_cummins_extended.py

This script:
1. Reads ``extended_points/point_*.npz`` and assembles ``sweep_extended.npz``
2. Reads ``anchor_ladder_extended.json`` for the extended-band $A_\\infty$ anchor
3. Evaluates gate C8 (extension consistency) and C9 (truncation decidability)
4. Updates ``case_result.json`` with the new gates
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
POINTS = OUT / "extended_points"
SWEEP_FILE = OUT / "sweep_extended.npz"
LADDER = OUT / "anchor_ladder_extended.json"
CASE_RESULT = OUT / "case_result.json"


def assemble_sweep() -> case.Sweep:
    """Assemble the extended sweep from per-node artifacts."""
    n_nodes = case.EXTENDED_N_OMEGA
    omegas = np.zeros(n_nodes)
    added_mass = np.zeros((n_nodes, 6, 6))
    damping = np.zeros((n_nodes, 6, 6))
    residuals = np.zeros(n_nodes)

    for k in range(n_nodes):
        p = POINTS / f"point_{k:03d}.npz"
        if not p.exists():
            raise SystemExit(f"Missing extended point {k}: {p}")
        data = np.load(p, allow_pickle=False)
        idx = int(data["index"])
        omegas[idx] = float(data["omega"])
        added_mass[idx] = np.array(data["added_mass"])
        damping[idx] = np.array(data["damping"])
        residuals[idx] = float(data["residual"])

    n_sections, n_levels = case.EXTENDED_MESH
    mesh = case.wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT,
        n_sections=n_sections, n_levels=n_levels,
    )
    max_scale = float(np.sqrt(np.max(mesh.areas)))

    sweep = case.Sweep(
        label="extended",
        n_sections=n_sections,
        n_levels=n_levels,
        panel_count=int(mesh.panels.shape[0]),
        max_panel_scale=max_scale,
        omega=omegas,
        added_mass=added_mass,
        damping=damping,
        solver_relative_residual=residuals,
        elapsed=0.0,
    )
    case.save_sweep(sweep, SWEEP_FILE)
    print(f"Assembled extended sweep: {SWEEP_FILE}")
    print(f"  omega = [{omegas[0]:.2f}, {omegas[-1]:.2f}] rad/s, {n_nodes} nodes")
    print(f"  panels = {sweep.panel_count}")
    return sweep


def evaluate_c8(committed: case.Sweep, extended: case.Sweep) -> dict:
    """Evaluate gate C8: extension consistency."""
    gate = case.extension_consistency_gate(committed, extended, identifier="C8")
    return gate


def evaluate_c9(committed: case.Sweep, extended: case.Sweep) -> dict:
    """Evaluate gate C9: truncation decidability."""
    gate = case.truncation_decidability_gate(committed, extended, identifier="C9")
    return gate


def update_case_result(c8: dict, c9: dict) -> dict:
    """Update case_result.json with C8/C9 gates."""
    if not CASE_RESULT.exists():
        raise SystemExit(f"Missing case_result.json: {CASE_RESULT}")

    with open(CASE_RESULT, "r") as f:
        result = json.load(f)

    result["gates"]["C8"] = c8
    result["gates"]["C9"] = c9

    # Re-evaluate verdict
    verdict = case.case_verdict(result["gates"])
    result["verdict"] = verdict
    result["extended_band"] = {
        "omega_range": [float(case.EXTENDED_BAND[0]), float(case.EXTENDED_BAND[1])],
        "n_omega": case.EXTENDED_N_OMEGA,
        "mesh": list(case.EXTENDED_MESH),
        "assembled_from": str(POINTS),
    }

    with open(CASE_RESULT, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Updated case_result.json: {CASE_RESULT}")
    print(f"  Verdict: {verdict['gates_passed']}/{verdict['gates_total']} gates passed")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Check artifacts exist but do not assemble")
    args = parser.parse_args(argv)

    # Check all point files exist
    n_nodes = case.EXTENDED_N_OMEGA
    missing = [k for k in range(n_nodes) if not (POINTS / f"point_{k:03d}.npz").exists()]
    if missing:
        print(f"Missing {len(missing)} extended points: {missing[:5]}...")
        return 1

    if not LADDER.exists():
        print(f"Missing anchor ladder: {LADDER}")
        return 1

    if args.dry_run:
        print(f"All {n_nodes} extended points present.")
        print(f"Anchor ladder present: {LADDER}")
        return 0

    # Load committed sweep
    committed = case.load_sweep(OUT / "sweep_medium.npz")
    print(f"Loaded committed sweep: {committed.label}, {committed.panel_count} panels")

    # Assemble extended sweep
    extended = assemble_sweep()

    # Evaluate C8
    c8 = evaluate_c8(committed, extended)
    print(f"C8: {'PASS' if c8['passed'] else 'FAIL'} (worst {c8['values']['worst_relative_difference']:.4f})")

    # Evaluate C9
    c9 = evaluate_c9(committed, extended)
    print(f"C9: {'PASS' if c9['passed'] else 'FAIL'} ({c9['status']})")

    # Update case_result.json
    result = update_case_result(c8, c9)

    # Print summary
    print("\n=== Extended Band Summary ===")
    for dof_name in case.DOF_NAMES:
        c9_row = [r for r in c9["values"]["per_dof"] if r["dof_name"] == dof_name][0]
        c3a_row = [r for r in result["gates"]["C3a"]["values"]["per_dof"] if r["dof_name"] == dof_name][0]
        print(f"  {dof_name:6s}: C3a residual={c3a_row['max_relative_residual']:.4f}, "
              f"n_decidable={c3a_row['n_decidable']}/{c3a_row['n_samples']}, "
              f"C9 budget_admissible={c9_row['budget_admissible_at_20']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
