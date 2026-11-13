"""Emit the maturity-gate evidence bundle for the MacCamy-Fuchs anchor.

The maturity gate (owner 2026-11-11 达标判断方案) re-checks every claimed
level against machine-verifiable evidence keys:

    L2  impl_files + entry + run_artifacts
    L3  + integration_callers
    L4  + benchmark (ref vs got) + repro_cmd
    L5  + external_anchor

This tool derives ``benchmarks/verified/bem_maccamy_fuchs/evidence.json``
*from the artefacts themselves* (results.json, gates.json, the committed
report, git HEAD), so the evidence cannot drift from the numbers it points
at.  Run with ``python -m tools.emit_maccamy_fuchs_evidence``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = REPO_ROOT / "benchmarks" / "verified" / "bem_maccamy_fuchs"
RESULTS_REL = "benchmarks/verified/bem_maccamy_fuchs/results.json"
GATES_REL = "benchmarks/verified/bem_maccamy_fuchs/gates.json"
REPORT_REL = "benchmarks/verified/bem_maccamy_fuchs/report.md"
SCHEMA = "tensorbem.maturity_evidence/1"

#: One-command reproduction: rebuild the anchor, re-emit the evidence, and
#: run the acceptance tests that pin the committed numbers.
REPRO_CMD = (
    "python -m build_maccamy_fuchs_anchor && "
    "python -m tools.emit_maccamy_fuchs_evidence && "
    "pytest tests/test_bem_maccamy_fuchs_verified.py "
    "tests/test_helmholtz_maccamy_fuchs.py "
    "tests/test_doc_numbers_maccamy_fuchs.py -q"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> int:
    results = json.loads((REPO_ROOT / RESULTS_REL).read_text())
    gates = json.loads((REPO_ROOT / GATES_REL).read_text())

    # ---- benchmark: ref (MacCamy & Fuchs 1954 closed form) vs got (BEM) ----
    # The reference values are the exact closed-form Bessel/Hankel numbers
    # produced by tensorbem.analytic.maccamy_fuchs_diffraction (independent
    # of the BEM discretisation); the "got" values are the BEM solver outputs
    # recorded in results.json.
    ref_levels = results["refinement"]["levels"]
    sweep = results["frequency_sweep"]
    irregular = results["irregular_frequency_study"]

    benchmark = {
        "reference": {
            "source": "tensorbem.analytic.maccamy_fuchs_diffraction (MacCamy & Fuchs 1954 closed form)",
            "kind": "CLOSED_FORM",
            "citation": (
                "MacCamy, R. C. & Fuchs, R. A. (1954). Wave forces on piles: "
                "a diffraction theory. U.S. Army Corps of Engineers, Beach "
                "Erosion Board, Technical Memorandum No. 69."
            ),
            "values": {
                "ka=1.0": {
                    "exact_force_x_magnitude_N": ref_levels[-1]["exact_force_x_magnitude"],
                    "exact_effective_cm": next(
                        r["exact_effective_cm"] for r in sweep if abs(r["ka"] - 1.0) < 0.01
                    ),
                },
                "ka=0.1": {
                    "exact_effective_cm": next(
                        r["exact_effective_cm"] for r in sweep if abs(r["ka"] - 0.1) < 0.01
                    ),
                },
                "ka=3.0": {
                    "exact_effective_cm": next(
                        r["exact_effective_cm"] for r in sweep if abs(r["ka"] - 3.0) < 0.01
                    ),
                },
            },
        },
        "got": {
            "refinement_128_panels": {
                "force_x_magnitude_N": ref_levels[-1]["force_x_magnitude"],
                "force_x_relative_error": ref_levels[-1]["force_x_relative_error"],
                "surface_potential_relative_l2_error": ref_levels[-1][
                    "surface_potential_relative_l2_error"
                ],
                "condition_number": ref_levels[-1]["condition_number"],
            },
            "frequency_sweep": [
                {
                    "ka": r["ka"],
                    "computed_effective_cm": r["computed_effective_cm"],
                    "exact_effective_cm": r["exact_effective_cm"],
                    "force_relative_error": r["force_relative_error"],
                }
                for r in sweep
            ],
            "irregular_frequency": [
                {
                    "resonance": r["resonance"],
                    "ka": r["ka"],
                    "unaugmented_condition_number": r["unaugmented_condition_number"],
                    "chief_condition_number": r["chief_condition_number"],
                    "unaugmented_force_relative_error": r["unaugmented_force_relative_error"],
                    "chief_force_relative_error": r["chief_force_relative_error"],
                }
                for r in irregular
            ],
        },
        "design_point_ka1_128panels": {
            "force_x_relative_error": ref_levels[-1]["force_x_relative_error"],
            "answer_under_3pct": ref_levels[-1]["force_x_relative_error"] < 0.03,
            "threshold_pct": 3.0,
        },
    }

    # ---- impl_files / entry / run_artifacts / integration_callers ----
    # Each artefact is listed in both path dialects the gate resolves:
    # repo-relative (audit worktree root) and eco-prefixed (ECO root).
    impl_rel = [
        "src/tensorbem/benchmarks/maccamy_fuchs.py",
        "src/tensorbem/benchmarks/maccamy_fuchs_figure.py",
        "src/tensorbem/helmholtz.py",
        "src/tensorbem/analytic.py",
        "src/tensorbem/geometry.py",
    ]
    impl_files = impl_rel + ["TensorBEM/" + r for r in impl_rel]

    entry_rel = "src/tensorbem/benchmarks/maccamy_fuchs.py"

    rel_artifacts = [
        "benchmarks/verified/bem_maccamy_fuchs/results.json",
        "benchmarks/verified/bem_maccamy_fuchs/gates.json",
        "benchmarks/verified/bem_maccamy_fuchs/report.md",
    ]
    run_artifacts = rel_artifacts + ["TensorBEM/" + r for r in rel_artifacts]

    # Integration callers: the main solver (solve_vertical_cylinder_diffraction
    # in tensorbem.helmholtz) is exercised by the benchmark driver and by the
    # acceptance tests that pin the committed numbers.
    integration_callers = [
        "tests/test_helmholtz_maccamy_fuchs.py::test_quadratic_convergence_of_force_moment_and_surface_potential",
        "tests/test_helmholtz_maccamy_fuchs.py::test_chief_suppresses_j01_and_j11_irregular_frequencies",
        "tests/test_bem_maccamy_fuchs_verified.py::TestReproducibility::test_rerun_matches_stored_results",
        "src/tensorbem/benchmarks/maccamy_fuchs.py:run -> solve_vertical_cylinder_diffraction (main solver)",
    ]

    evidence = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(),
        "mission": 17226,
        "original_mission": 17180,
        "claimed_level": 4,
        "impl_files": impl_files,
        "entry": entry_rel,
        "entry_eco": "TensorBEM/" + entry_rel,
        "entry_command": "python -m tensorbem.benchmarks.maccamy_fuchs --panels 16 32 64 128",
        "run_artifacts": run_artifacts,
        "integration_callers": integration_callers,
        "benchmark": benchmark,
        "repro_cmd": REPRO_CMD,
        "tests": [
            "tests/test_bem_maccamy_fuchs_verified.py",
            "tests/test_helmholtz_maccamy_fuchs.py",
            "tests/test_doc_numbers_maccamy_fuchs.py",
        ],
        "artefact_sha256": {
            "results.json": sha256_file(REPO_ROOT / RESULTS_REL),
            "gates.json": sha256_file(REPO_ROOT / GATES_REL),
        },
        "verdict": {
            "all_criteria_pass": gates["all_gates_passed"],
            "pass_by_id": {g["id"]: g["passed"] for g in gates["gates"]},
        },
    }
    out = OUTDIR / "evidence.json"
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
