#!/usr/bin/env python3
"""Regenerate CASE-BEM-CUMMINS-WIGLEY case_result.json from committed sweep artifacts.

This script replays all gates (including C3a, C7, C3a_coarse) from the existing
sweep_medium.npz and sweep_coarse.npz artifacts without re-solving the BEM system.
It fixes two known issues in the stale case_result.json:
1. Missing gates: C3a (zero-free-parameter KK closure), C7 (A_inf anchor consistency),
   C3a_coarse (anchored KK closure on coarse mesh)
2. Stale settings.dt: was 0.01, should be 0.002 (matching the actual gate evaluation)

Usage:
    PYTHONPATH=src python3 scripts/regenerate_cummins_wigley_verdict.py
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.case_bem_cummins_wigley import (
    CASE_SLUG,
    DOF_NAMES,
    LENGTH,
    BEAM,
    DRAFT,
    DENSITY,
    GRAVITY,
    CB_ANALYTIC,
    VOLUME_ANALYTIC,
    BAND,
    N_OMEGA,
    MESHES,
    DT,
    DURATION,
    TAU_MAX,
    TOLERANCES,
    load_sweep,
    geometry_gate,
    solver_validity_gate,
    panel_adequacy_gate,
    kk_gate,
    round_trip_gate,
    free_decay_gate,
    negative_control_gate,
    anchor_for_sweep,
    added_mass_inf_gate,
    case_verdict,
)

ARTIFACT_DIR = Path("benchmarks/verified/casepool_bem_cummins_wigley")
OUTPUT = ARTIFACT_DIR / "case_result.json"
MISSION = "m21859"
DOF_FREE_DECAY = 2  # heave


def main() -> None:
    out_path = OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load committed sweep artifacts
    sweep_medium = load_sweep(ARTIFACT_DIR / "sweep_medium.npz")
    sweep_coarse = load_sweep(ARTIFACT_DIR / "sweep_coarse.npz")

    # Build results dict with correct settings
    results: dict = {
        "case": CASE_SLUG,
        "mission": MISSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "hull": {
            "form": "mathematical Wigley hull y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)",
            "length_m": LENGTH,
            "beam_m": BEAM,
            "draft_m": DRAFT,
            "density_kg_m3": DENSITY,
            "gravity_m_s2": GRAVITY,
            "block_coefficient_analytic": CB_ANALYTIC,
            "displaced_volume_analytic_m3": VOLUME_ANALYTIC,
        },
        "settings": {
            "band_rad_s": list(BAND),
            "n_omega": N_OMEGA,
            "refine_peaks": 0,
            "meshes": {label: list(MESHES[label]) for label in MESHES},
            "chief_points": "auto",
            "tolerances": TOLERANCES,
            "free_decay_dof": DOF_NAMES[DOF_FREE_DECAY],
            "dt": DT,  # Fixed: was 0.01, now 0.002
            "duration": DURATION,
            "tau_max": TAU_MAX,
        },
    }

    # Sweeps summary
    results["sweeps"] = {
        "coarse": sweep_coarse.summary(),
        "medium": sweep_medium.summary(),
    }

    # Gates
    gates: dict = {}
    gates["C1"] = geometry_gate()

    # C0: solver validity (medium)
    gates["C0"] = solver_validity_gate(sweep_medium)
    gates["C0_coarse"] = solver_validity_gate(sweep_coarse)

    if not gates["C0"]["passed"]:
        print("WARNING: C0 failed on medium mesh — gates voided")
        for key in ("C2", "C3", "C3a", "C7", "C3_coarse", "C3a_coarse", "C4", "C5", "C6a", "C6b"):
            gates[key] = {"id": key, "passed": False, "voided": True, "voided_by": "C0"}
    else:
        gates["C2"] = panel_adequacy_gate(sweep_medium)
        gates["C3"] = kk_gate(sweep_medium)

        # C3a: zero-free-parameter KK closure (anchored A_inf)
        anchor = anchor_for_sweep(sweep_medium)
        results["added_mass_inf_anchor"] = anchor
        gates["C3a"] = kk_gate(
            sweep_medium,
            added_mass_inf=anchor["added_mass_inf_diagonal"],
            identifier="C3a",
        )

        # C7: A_inf anchor consistency
        gates["C7"] = added_mass_inf_gate(
            sweep_medium, anchor, closure_rows=gates["C3"]["values"]["per_dof"]
        )

        # Coarse mesh gates
        gates["C3_coarse"] = kk_gate(sweep_coarse)
        coarse_anchor = anchor_for_sweep(sweep_coarse)
        gates["C3a_coarse"] = kk_gate(
            sweep_coarse,
            added_mass_inf=coarse_anchor["added_mass_inf_diagonal"],
            identifier="C3a_coarse",
        )

        # C4: round trip
        gates["C4"] = round_trip_gate(sweep_medium, dof=DOF_FREE_DECAY)

        # C5: free decay
        gates["C5"] = free_decay_gate(sweep_medium, dof=DOF_FREE_DECAY)

        # C6a: negative control (zero damping)
        gates["C6a"] = negative_control_gate(sweep_medium, dof=DOF_FREE_DECAY)

        # C6b: negative control (corrupted damping)
        decidable_window = [
            all(row["decidable_mask"][k] for row in gates["C3"]["values"]["per_dof"])
            for k in range(len(gates["C3"]["values"]["per_dof"][0]["decidable_mask"]))
        ]
        gates["C6b"] = kk_gate(sweep_medium, corrupt=True, window=decidable_window)

    results["gates"] = gates
    results["verdict"] = case_verdict(gates)

    # Write
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    v = results["verdict"]
    print(f"Verdict: {v['gates_passed']}/{v['gates_total']} gates passed")
    print(f"Passed: {v['passed_list']}")
    if v['failed_list']:
        print(f"Failed: {v['failed_list']}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
