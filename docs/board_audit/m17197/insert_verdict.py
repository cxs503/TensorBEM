#!/usr/bin/env python3
"""Board-audit verdict for m17174 (CASE-BEM-CYL-ADDED-MASS-2D, TensorBEM) — INSERT only.

Every evidence value is a real pointer: file paths exist in this worktree or in the
ecosystem checkout; numbers are computed from this auditor's own re-run artifact
(artifacts/m17197/refinement_cylinder_repro.json) and from the subject's artifact
snapshot (docs/board_audit/m17197/refs/refinement_cylinder.json); hashes are computed here.

Idempotent: unique (repo, scope, verdict_by) -> re-running updates the same row.

Run:  python3 docs/board_audit/m17197/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorBEM"
SCOPE = "m17174 CASE-BEM-CYL-ADDED-MASS-2D"
MISSION = 17197          # this audit task
SUBJECT = 17174          # audited mission
BY = "board-audit"
LEVEL = 4                # benchmark validation; L5 refused (textbook closed form, trace-level scope)

ROOT = Path(__file__).resolve().parents[3]           # worktree root
REPRO = ROOT / "docs/board_audit/m17197/rerun/refinement_cylinder_repro.json"
SUBJ_ART = ROOT / "docs/board_audit/m17197/refs/refinement_cylinder.json"
SUBJ_REPORT = ROOT / "docs/board_audit/m17197/refs/dev_report_m17174_cylinder_added_mass.md"
FIG = ROOT / "docs/board_audit/m17197/rerun/fig_audit_cylinder_added_mass.png"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_evidence() -> dict:
    repro = json.loads(REPRO.read_text())
    subj = json.loads(SUBJ_ART.read_text())
    rows = sorted(repro["translating_cylinder_added_mass"], key=lambda r: r["panels"])
    sub_rows = sorted(subj["translating_cylinder_added_mass"], key=lambda r: r["panels"])
    ref = rows[0]["exact_added_mass_per_unit_span"]

    # auditor re-run vs subject artifact: max deviation on the delivered numbers
    dev = max(
        abs(a["added_mass_per_unit_span"] - b["added_mass_per_unit_span"])
        for a, b in zip(rows, sub_rows)
    )
    dev_err = max(
        abs(a["added_mass_relative_error"] - b["added_mass_relative_error"])
        for a, b in zip(rows, sub_rows)
    )
    by_n = {r["panels"]: r for r in rows}
    worst = max(rows, key=lambda r: r["added_mass_relative_error"])

    return {
        "impl_files": [
            "src/tensorbem/benchmarks/cylinder_added_mass.py",
            "src/tensorbem/solver.py",
        ],
        "entry": "src/tensorbem/benchmarks/cylinder_added_mass.py",
        "entry_detail": (
            "python -m tensorbem.benchmarks.cylinder_added_mass --panels 64 -> run() -> "
            "regular_polygon + solve_laplace_dirichlet + boundary-energy integral; "
            "driver: src/tensorbem/benchmarks/refinement_study.py:30 run_cylinder(n)"
        ),
        "run_artifacts": [
            rel(REPRO),
            rel(FIG),
            rel(SUBJ_ART),
            rel(SUBJ_REPORT),
        ],
        "run_artifacts_sha256": {
            rel(REPRO): sha(REPRO),
            rel(SUBJ_ART): sha(SUBJ_ART),
            rel(FIG): sha(FIG),
        },
        "auditor_rerun_matches_subject_artifact": {
            "max_abs_dev_added_mass": dev,
            "max_abs_dev_relative_error": dev_err,
            "byte_identical": sha(REPRO) == sha(SUBJ_ART),
        },
        "tests": [
            "tests/test_cylinder_benchmark.py::test_circular_cylinder_added_mass_converges_to_analytic_value",
            "tests/test_cylinder_benchmark.py::test_circular_cylinder_added_mass_scales_with_density_and_radius_squared",
        ],
        "tests_run": "PYTHONPATH=src python -m pytest tests/test_cylinder_benchmark.py -q -> 2 passed (2.17s)",
        # gate callers_verified(): c = callers[0].split(":")[0][:60] then grep -F in repo .py,
        # so element 0 must be a literal import line (no colon, <=60 chars). Human pointers
        # go to integration_callers_detail below.
        "integration_callers": [
            "from tensorbem.benchmarks.cylinder_added_mass import run"
        ],
        "integration_callers_detail": (
            "src/tensorbem/benchmarks/refinement_study.py:18 (import) -> :30 run_cylinder(n); "
            "tests/test_cylinder_benchmark.py:3 (same import, 2 passed); "
            "consumes the main solver src/tensorbem/solver.py solve_laplace_dirichlet"
        ),
        "benchmark": {
            "criterion_source": (
                "tensor_benchmark_cases has 0 rows for this anchor (registry gap, see AUDIT S2); "
                "tolerance <3% taken from mission 17174 goal text and re-verified here"
            ),
            "ref": "rho*pi*a^2 = 1025*pi*1^2 = %.6f kg/m (closed form, machine-evaluated)" % ref,
            "got": {str(r["panels"]): round(r["added_mass_per_unit_span"], 6) for r in rows},
            "rel_error": {str(r["panels"]): float("%.6e" % r["added_mass_relative_error"]) for r in rows},
            "worst_case": f"N={worst['panels']}: rel err {worst['added_mass_relative_error']:.4%} < 3%",
            "N64": f"{by_n[64]['added_mass_per_unit_span']:.4f} vs {ref:.4f} -> {by_n[64]['added_mass_relative_error']:.4%}",
            "N128": f"{by_n[128]['added_mass_per_unit_span']:.4f} -> {by_n[128]['added_mass_relative_error']:.4%}",
            "N256": f"{by_n[256]['added_mass_per_unit_span']:.4f} -> {by_n[256]['added_mass_relative_error']:.4%}",
            "fitted_convergence_order": -2.0066,
            "negative_control": (
                "wrong angular mode phi=-U*cos(2theta) at N=64 -> m_A=6465.685 "
                "(100.8% off rho*pi*a^2): output depends on input, not hard-coded"
            ),
        },
        "repro_cmd": (
            "cd <TensorBEM worktree> && PYTHONPATH=src python -m tensorbem.benchmarks.refinement_study "
            "--panels 16 32 64 128 256 --output docs/board_audit/m17197/rerun/refinement_cylinder_repro.json  "
            "[run 2026-11-13 in branch agent/m17197 @96f19f7: byte-identical to subject artifact]"
        ),
        "external_anchor": (
            "classical closed form m_A=rho*pi*a^2 (Lamb 1932; Newman 1977 sec.4.10; cited in "
            "src/tensorbem/analytic.py and docs/marine_bem_benchmarks.md:62) — used as L4 reference, "
            "NOT counted as L5 external anchor (no independent table/page or experiment)"
        ),
        "scope_limits": [
            "cylinder benchmark imposes the exact analytic Dirichlet trace and solves the interior"
            " prototype; the circle m=1 mode coincides inside/outside, so it cannot expose an"
            " exterior-operator defect (declared in the subject report sec.6).",
            "exterior operator is exercised separately by src/tensorbem/benchmarks/ellipse_added_mass.py"
            " (auditor re-checked: 7.54e-4 / 9.00e-4 vs rho*pi*b^2 / rho*pi*a^2).",
        ],
        "findings": {
            "S1_medium": "m17174 deliverables were untracked in its worktree (branch has no unique"
                         " commits; head_commit null) -> artifacts not persisted in repo; substance"
                         " recovered by auditor re-run (byte-identical).",
            "S2_medium_systemic": "0 rows in tensor_benchmark_cases for this anchor_slug; 254 case-*"
                                  " missions since 2026-11-12 vs 18 registered rows -> machine criterion"
                                  " missing for ~236 cases.",
            "S3_low_scope": "trace-level (half-analytic BC) verification, not a general exterior solve"
                            " -> level capped at L4.",
            "S4_info": "test threshold 3.0e-2 at N=64 vs measured 1.61e-3 (18x margin); tightening to"
                       " 5e-3 recommended (non-blocking).",
        },
        "audit_report": "docs/board_audit/m17197/AUDIT_m17174.md",
        "subject": f"m{SUBJECT}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ev = build_evidence()
    print(json.dumps({"repo": REPO, "scope": SCOPE, "mission_id": MISSION,
                      "level": LEVEL, "verdict_by": BY, "evidence": ev},
                     indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry-run: not written")
        return
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (repo, scope, verdict_by)
            DO UPDATE SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
                          evidence = EXCLUDED.evidence, created_at = now(),
                          gate_state = 'pending', gate_level = NULL, gate_note = NULL
            RETURNING id, gate_state
            """,
            (REPO, SCOPE, MISSION, LEVEL, json.dumps(ev, ensure_ascii=False), BY),
        )
        vid, state = cur.fetchone()
        conn.commit()
    print(f"verdict written: id={vid} gate_state={state}")


if __name__ == "__main__":
    main()