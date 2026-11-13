#!/usr/bin/env python3
"""One-command reproducibility script for CASE-BEM-CYL-DRIFT-ASYMP.

Cylinder mean drift force ((ka)^3 law), TensorBEM.
Ref: MacCamy-Fuchs near field + Maruo far field closed form
     F_x = (5 pi^2 / 16) rho g A^2 a (ka)^3  ~  3.08425 rho g A^2 a (ka)^3.

Usage:
    PYTHONPATH=src python3 repro_cylinder_drift_asymptotic.py

Produces:
  artifacts/cylinder_drift_asymptotic/study.json        (full run artifact)
  artifacts/cylinder_drift_asymptotic/fig_*.png         (figures)
  artifacts/cylinder_drift_asymptotic/benchmark.json    (ref vs got table)
and prints the machine-checkable verdict table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorbem.benchmarks import cylinder_drift_asymptotic as study
from tensorbem import cylinder_drift as cd

OUT = HERE / "artifacts" / "cylinder_drift_asymptotic"
OUT.mkdir(parents=True, exist_ok=True)


def _b(v):
    return bool(v)


def main() -> int:
    print("=== CASE-BEM-CYL-DRIFT-ASYMP: cylinder mean drift force ((ka)^3 law) ===\n")

    # 1. Build the full study (exact near field + far field + long-wave law + energy)
    s = study.build_study()
    s["figures"] = study.make_figures(s, OUT)
    (OUT / "study.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")

    # 2. Ref vs got benchmark table (L4)
    closed = cd.long_wave_drift_coefficient()  # 5 pi^2 / 16
    lw = s["long_wave"]
    rows = s["rows"]
    bench = {
        "case": "CASE-BEM-CYL-DRIFT-ASYMP",
        "ref_source": "MacCamy-Fuchs (1954) near field + Maruo far field closed form; "
                      "F_x = (5 pi^2/16) rho g A^2 a (ka)^3",
        "ref_closed_form_coefficient": closed,
        "tolerance_pct": 5.0,
        "comparison": [
            {
                "ka": r["ka"],
                "got_near_field": r["near_field_force"],
                "got_far_field": r["far_field_force"],
                "near_far_rel_diff": r["near_far_rel_diff"],
                "pass_5pct": _b(r["near_far_rel_diff"] < 0.05),
            }
            for r in rows
        ],
        "long_wave_law": {
            "fitted_slope": lw["fitted_slope"],
            "slope_target": 3.0,
            "slope_tolerance": 0.2,
            "slope_pass": _b(abs(lw["fitted_slope"] - 3.0) <= 0.2),
            "got_coefficient_at_smallest_ka": lw["normalised_coefficient"][0],
            "ref_coefficient": closed,
            "coefficient_rel_error": lw["coefficient_rel_error_at_smallest_ka"],
            "coefficient_pass": _b(lw["coefficient_rel_error_at_smallest_ka"] < 0.03),
        },
        "verdict": s["verdict"],
    }
    (OUT / "benchmark.json").write_text(json.dumps(bench, indent=2, sort_keys=True) + "\n")

    # 3. Print verdict table
    v = s["verdict"]
    print("  [%s] near-field vs far-field max rel diff < 5%%   (%.3e)"
          % ("PASS" if v["near_far_pass"] else "FAIL", v["near_far_max_rel_diff"]))
    print("  [%s] (ka)^3 slope = %.3f  (target 3 +/- 0.2)"
          % ("PASS" if v["long_wave_slope_pass"] else "FAIL", v["long_wave_slope"]))
    print("  [%s] coefficient vs 5pi^2/16 = %.4f  rel err %.3e  (< 3%%)"
          % ("PASS" if v["long_wave_coefficient_pass"] else "FAIL",
             closed, v["long_wave_coefficient_rel_error"]))
    print("  [%s] Kochin energy residual max = %.3e  (< 1%%)"
          % ("PASS" if v["energy_pass"] else "FAIL", v["energy_residual_max"]))
    print("  [%s] wall anchor rho g A^2/2 rel err = %.3e"
          % ("PASS" if v["wall_anchor_pass"] else "FAIL", v["wall_anchor_rel_error"]))
    print("\nOverall: %s" % ("ALL GATES PASSED" if v["all_gates_pass"] else "SOME GATES FAILED"))
    print("Artifacts: %s" % OUT)
    return 0 if v["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
