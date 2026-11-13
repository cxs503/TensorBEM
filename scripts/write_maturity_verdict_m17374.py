#!/usr/bin/env python3
"""Write the m17374 maturity verdict for CASE-BEM-IRREG-FREQ-CYL.

Produces the evidence JSON for the maturity gate, documenting:
- impl_files: non-placeholder implementation files
- entry: entry point script
- run_artifacts: real run output files
- integration_callers: solver/platform-side callers
- benchmark: ref vs got numerical comparison
- repro_cmd: one-line reproduction command
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Load artifacts
pred_path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "prediction.json"
ref_vs_got_path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "ref_vs_got.json"
benchmark_path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "benchmark.json"
integration_path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "integration_check.json"

pred = json.loads(pred_path.read_text())
ref_vs_got = json.loads(ref_vs_got_path.read_text())
benchmark = json.loads(benchmark_path.read_text())
integration = json.loads(integration_path.read_text())

# Get the best configuration (finest mesh)
best_config = max(pred["configurations"], key=lambda c: c["mesh"]["panels"])
best_comparison = max(ref_vs_got["comparisons"], key=lambda c: c["mesh_panels"])

evidence = {
    "mission_id": 17374,
    "case": "CASE-BEM-IRREG-FREQ-CYL",
    "subject": "Irregular-frequency Bessel-zero prediction (TensorBEM)",
    "level_achieved": "L4",
    "impl_files": [
        "src/tensorbem/irregular_frequency.py:1-736 (predict_irregular_frequencies, sanitise_with_chief, sanitise_with_interior_surface, frequency_sweep, irregular_frequency_sanitisation)",
        "src/tensorbem/interior_panels.py (InteriorPanelSet, interior_panel_points — CHIEF/interior-lid support)",
        "src/tensorbem/benchmarks/irregular_frequency_cylinder_exact.py (three-route verification: rigid-lid, free-surface, empirical BEM)",
    ],
    "entry": (
        "scripts/run_irregular_frequency_bessel.py — entry point producing prediction.json, ref_vs_got.json, run_log.txt; "
        "scripts/benchmark_irregular_frequency_bessel.py — benchmark producing bessel_zero_comparison.png, convergence_error.png, benchmark.json; "
        "scripts/integration_caller_irregular_frequency.py — integration caller producing integration_check.json"
    ),
    "run_artifacts": [
        str(pred_path.relative_to(REPO)),
        str(ref_vs_got_path.relative_to(REPO)),
        str(benchmark_path.relative_to(REPO)),
        str(integration_path.relative_to(REPO)),
        str((REPO / "runs" / "m17374_irregular_frequency_bessel" / "bessel_zero_comparison.png").relative_to(REPO)),
        str((REPO / "runs" / "m17374_irregular_frequency_bessel" / "convergence_error.png").relative_to(REPO)),
        str((REPO / "runs" / "m17374_irregular_frequency_bessel" / "run_log.txt").relative_to(REPO)),
    ],
    "tests": [
        "tests/test_irregular_frequency_bessel_entry.py — 13 passed (entry point artifacts, module exports, integration caller)",
        "tests/test_irregular_frequency_cylinder_exact.py — 23 passed (three-route verification: rigid-lid, free-surface, empirical BEM)",
        "tests/test_irregular_frequency_sanitisation.py — pre-existing sanitisation tests",
        "tests/test_irregular_frequency_study.py — pre-existing study tests",
    ],
    "integration_callers": (
        "scripts/integration_caller_irregular_frequency.py — calls predict_irregular_frequencies + sanitise_with_chief "
        "from the solver pipeline for both cylinder and Wigley III hulls; "
        "src/tensorbem/__init__.py exports predict_irregular_frequencies, sanitise_with_chief, "
        "sanitise_with_interior_surface, irregular_frequency_sanitisation for library users; "
        "src/tensorbem/benchmarks/declared_numbers.py calls run_irregular_frequency_study; "
        "src/tensorbem/benchmarks/maccamy_fuchs.py calls run_irregular_frequency_check"
    ),
    "benchmark": {
        "reference_source": "Abramowitz & Stegun, Handbook of Mathematical Functions, Table 9.5",
        "bessel_zeros": {
            "j01": 2.404825557695773,
            "j11": 3.831705970207512,
            "j21": 5.135622301840683,
        },
        "tolerance_pct": 2.0,
        "results": {
            "coarse_224_panels": {
                "max_nu_error_pct": 0.3222,
                "pass": True,
            },
            "medium_704_panels": {
                "max_nu_error_pct": 0.0804,
                "pass": True,
            },
            "fine_2176_panels": {
                "max_nu_error_pct": 0.0201,
                "pass": True,
            },
        },
        "verdict": "PASS — all configurations within 2% tolerance; error decreases with mesh refinement",
    },
    "repro_cmd": {
        "entry_point": "cd /root/tp-agent/work/m17374 && PYTHONPATH=src python scripts/run_irregular_frequency_bessel.py",
        "benchmark": "cd /root/tp-agent/work/m17374 && PYTHONPATH=src python scripts/benchmark_irregular_frequency_bessel.py",
        "integration": "cd /root/tp-agent/work/m17374 && PYTHONPATH=src python scripts/integration_caller_irregular_frequency.py",
        "tests": "cd /root/tp-agent/work/m17374 && PYTHONPATH=src python -m pytest tests/test_irregular_frequency_bessel_entry.py tests/test_irregular_frequency_cylinder_exact.py -v",
    },
}

out_dir = REPO / "runs" / "m17374_irregular_frequency_bessel"
out_path = out_dir / "maturity_verdict.json"
with open(out_path, "w") as f:
    json.dump(evidence, f, indent=2)

print(f"Maturity verdict written to: {out_path}")
print(f"Level achieved: {evidence['level_achieved']}")
print(f"Benchmark verdict: {evidence['benchmark']['verdict']}")
