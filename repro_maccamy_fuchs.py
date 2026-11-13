#!/usr/bin/env python3
"""One-command reproducibility script for MacCamy-Fuchs benchmark.

Usage:
    PYTHONPATH=src python repro_maccamy_fuchs.py

Produces artifacts in artifacts/maccamy_fuchs/ and prints summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorbem.benchmarks.maccamy_fuchs import run

OUT = HERE / "artifacts" / "maccamy_fuchs"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=== MacCamy-Fuchs Benchmark Reproduction ===\n")

    # Run the standard benchmark (same as build_maccamy_fuchs_anchor.py)
    data = run(panel_counts=[16, 32, 64, 128])

    # Save
    with open(OUT / "repro_results.json", "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Verify key gates
    ref = data["refinement"]
    fine = ref["levels"][-1]
    force_err = fine["force_x_relative_error"]
    phi_err = fine["surface_potential_relative_l2_error"]
    last_order = ref["observed_orders"]["force_x_relative_error"][-1]

    sweep = data["frequency_sweep"]
    max_sweep_err = max(r["force_relative_error"] for r in sweep)

    irf = data["irregular_frequency_study"]
    j1 = [r for r in irf if "j11" in r["resonance"]][0]
    kappa_red = j1["unaugmented_condition_number"] / j1["chief_condition_number"]

    cm_ka01 = [r for r in sweep if abs(r["ka"] - 0.1) < 0.01][0]["computed_effective_cm"]

    gates = [
        ("G1: Force error at 128 panels < 0.05%", force_err < 5e-4, force_err),
        ("G2: Observed order > 1.9", last_order > 1.9, last_order),
        ("G3: Surface phi error at 128 panels < 0.02%", phi_err < 2e-4, phi_err),
        ("G4: Max sweep force error < 0.5%", max_sweep_err < 5e-3, max_sweep_err),
        ("G5: CHIEF kappa reduction > 100x", kappa_red > 100, kappa_red),
        ("G6: CM at ka=0.1 within 0.02 of 2.0", abs(cm_ka01 - 2.0) < 0.02, abs(cm_ka01 - 2.0)),
    ]

    all_pass = True
    for name, passed, val in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print("  [%s] %s (measured: %s)" % (status, name, val))

    print("\nOverall: %s" % ("ALL GATES PASSED" if all_pass else "SOME GATES FAILED"))
    print("Results: %s" % (OUT / "repro_results.json"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
