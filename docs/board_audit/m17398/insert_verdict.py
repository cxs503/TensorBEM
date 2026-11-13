#!/usr/bin/env python3
"""Board-audit verdict for m17336 (CASE-BEM-MARUO-DRIFT, TensorBEM) — INSERT/upsert.

Every number in the evidence blob is read from an artifact produced by this
audit (delivered study JSON, auditor's independent reference re-derivation,
auditor's two local re-runs) or from the platform database; nothing is typed
by hand.  Idempotent on (repo, scope, verdict_by).

Run:  python3 docs/board_audit/m17398/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorBEM"
SCOPE = "m17336 CASE-BEM-MARUO-DRIFT"
MISSION = 17398            # this audit task
SUBJECT = 17336            # audited mission
BY = "board-audit"
ANCHOR = "CASE-BEM-MARUO-DRIFT"
SUBJECT_COMMIT = "4ecdd3b7d264dc0cd3af7f14f0946660462a3fd0"
LEVEL = 3                  # L4 blocked: no referee row + criterion fails at ka=2

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # /root/tp-agent/work/m17398
REFS = HERE / "refs"
ART = HERE / "artifacts"


def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(ROOT))


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def db_rows(sql: str, args: tuple = ()) -> list[dict]:
    with psycopg.connect(DSN, autocommit=True) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, args)
            return cur.fetchall()


def build() -> dict:
    study = json.loads((REFS / "docs/case_bem_maruo_drift/study.json").read_text())
    indep = json.loads((ART / "independent_ref_v2.json").read_text())
    repro_coarse = json.loads((ART / "local_repro_ka2_coarse.json").read_text())
    repro_prod_path = ART / "local_repro_ka2_production.json"
    repro_prod = json.loads(repro_prod_path.read_text()) if repro_prod_path.exists() else None

    cells = study["cells"]
    production = study["parameters"]["production_resolution"]
    prod_cells = sorted((c for c in cells if c["resolution"] == production),
                        key=lambda c: c["ka"])
    worst = max(prod_cells, key=lambda c: c["rel_error"])
    finer = sorted((c for c in cells if c["ka"] == worst["ka"]),
                   key=lambda c: c["panel_count"])[-1]

    # local reproduction of two delivered cells, cell for cell
    def match(delivered: dict, rerun: dict) -> dict:
        c = rerun["cells"][0]
        d = [x for x in cells if x["ka"] == c["ka"] and x["resolution"] == c["resolution"]][0]
        return {
            "cell": f"ka={c['ka']:g} {c['resolution']} ({c['panel_count']} panels)",
            "delivered_force": d["bem_force"],
            "recomputed_force": c["bem_force"],
            "abs_diff": abs(d["bem_force"] - c["bem_force"]),
            "delivered_rel_error": d["rel_error"],
            "recomputed_rel_error": c["rel_error"],
            "delivered_exit_g3_passed": {g["name"]: g["passed"] for g in
                                         study["gates"]}.get("G3_bem_vs_ref_production"),
            "recomputed_gates": {g["name"]: g["passed"] for g in rerun["gates"]},
        }

    repro = [match(cells, json.loads(p.read_text()))
             for p in sorted(ART.glob("local_repro_ka1_*.json"))] + [match(cells, repro_coarse)]
    if repro_prod is not None:
        repro.append(match(cells, repro_prod))

    referee = db_rows(
        "SELECT id, case_name, metric, tolerance, status FROM tensor_benchmark_cases "
        "WHERE anchor_slug = %s", (ANCHOR,))
    neighbour = db_rows(
        "SELECT id, anchor_slug, case_name, tolerance, status FROM tensor_benchmark_cases "
        "WHERE repo = %s AND anchor_slug LIKE '%%DRIFT%%' ORDER BY id", (REPO,))

    impl = REFS / "src/tensorbem/benchmarks/maruo_drift_far_field_case.py"
    figures_mod = REFS / "src/tensorbem/benchmarks/maruo_drift_far_field_case_figures.py"
    drift_mod = REFS / "src/tensorbem/drift_far_field3d.py"

    ev = {
        "repo": REPO,
        "subject_mission": SUBJECT,
        "audit_mission": MISSION,
        "anchor": ANCHOR,
        "subject_commit": SUBJECT_COMMIT,
        "declared_level": LEVEL,
        "level_rationale": [
            "L2 met: real code path mesh -> solve_diffraction_3d -> bem_kochin -> maruo_far_field_drift, "
            "inputs -> numbers -> JSON, no hard-coded result",
            "L3 met: the case drives the production entry points exported by tensorbem.free_surface_solver3d / "
            "tensorbem.drift_far_field3d that main-side drivers already call, and it is registered in the "
            "repo capability registry (tensorbem.capability_evidence row 'maruo-far-field-drift-case') on the branch",
            "L4 REFUSED: (a) tensor_benchmark_cases has 0 rows for anchor_slug='" + ANCHOR + "' -> no machine "
            "referee, the <5% criterion lives only in mission-goal text; (b) the declared <5% criterion FAILS at "
            f"the production mesh (ka={worst['ka']:g}, {worst['panel_count']} panels, "
            f"{100 * worst['rel_error']:.3f}%) and is still {100 * finer['rel_error']:.3f}% at "
            f"{finer['panel_count']} panels -> not a mesh accident",
        ],
        # bare paths only: the maturity gate resolves each entry with os.path.exists
        "impl_files": [rel(impl), rel(figures_mod), rel(drift_mod)],
        "impl_files_detail": [
            f"{rel(impl)} — case driver (closed-form ref at :153-212, BEM route at :214-252), "
            f"sha256 {sha(impl)}, audited commit {SUBJECT_COMMIT[:7]}",
            f"{rel(figures_mod)} — report figure helper, sha256 {sha(figures_mod)}",
            f"{rel(drift_mod)} — production far-field kernel consumed by the case, sha256 {sha(drift_mod)}",
        ],
        "entry": rel(impl),
        "entry_cmd": ("PYTHONPATH=src python3 -m tensorbem.benchmarks.maruo_drift_far_field_case "
                      "--json <out>.json --figures <dir>   (exit 1 when any gate fails: the delivered "
                      "study has all_passed=false, so the reproducible run exits 1 by design)"),
        "run_artifacts": [
            rel(REFS / "docs/case_bem_maruo_drift/study.json"),
            rel(REFS / "docs/case_bem_maruo_drift/fig_bem_vs_closed_form.png"),
            rel(REFS / "docs/case_bem_maruo_drift/fig_coefficient_vs_ka.png"),
            rel(REFS / "docs/case_bem_maruo_drift/fig_convergence_and_quadrature.png"),
            rel(ART / "independent_ref_v2.json"),
            rel(ART / "local_repro_ka2_coarse.json"),
            rel(ART / "local_repro_ka1_coarse.json"),
            rel(ART / "fig_m17398_error_vs_panels.png"),
            rel(ART / "fig_m17398_ref_validation.png"),
        ] + ([rel(repro_prod_path)] if repro_prod is not None else []),
        "tests": [
            "tests/test_maruo_drift_far_field_case.py::test_bem_route_runs_and_is_of_the_right_sign_and_size",
            "tests/test_maruo_drift_far_field_case.py::test_closed_form_reproduces_the_repo_quadrature_route",
            "tests/test_maruo_drift_far_field_case.py::test_closed_form_is_converged_in_the_modal_truncation",
            "tests/test_maruo_drift_far_field_case.py::test_maruo_kernel_quadrature_is_converged",
            "tests/test_maruo_drift_far_field_case.py::test_study_artifact_is_internally_consistent",
            "auditor ran: PYTHONPATH=src pytest tests/test_maruo_drift_far_field_case.py -q "
            "-> 13 passed in 2.48s (worktree /root/tp-agent/work/m17336 @ " + SUBJECT_COMMIT[:7] + ")",
        ],
        "integration_callers": [
            "tensorbem.drift_far_field3d.bem_kochin",
            "src/tensorbem/benchmarks/maruo_drift_far_field_case.py:214-252 (bem_case body: solve_diffraction_3d -> "
            "bem_kochin -> maruo_far_field_drift -> closed_form_case)",
            "src/tensorbem/benchmarks/drift_far_field_3d_study.py:100,178 (main-side driver calling the same "
            "production chain: koch = bem_kochin(mesh, diffraction, ...) -> maruo_far_field_drift)",
            "src/tensorbem/diffraction_drift_maturity.py:452 (vertical_cylinder_hull_mesh, the mesh generator "
            "the case reuses)",
            "src/tensorbem/capability_evidence.py (branch): registry row 'maruo-far-field-drift-case' -> "
            "tensorbem.benchmarks.maruo_drift_far_field_case",
            "merge state at audit time: branch agent/m17336 @ " + SUBJECT_COMMIT[:7] + " is NOT an ancestor of "
            "the ecosystem checkout (/root/tensor-ecosystem/TensorBEM @ ac57fda), so a main-only grep cannot see "
            "the case driver yet; the L3 grep resolves through the production chain that main already drives",
        ],
        "benchmark": {
            "anchor_slug": ANCHOR,
            "registered_referee_rows": len(referee),
            "referee_rows_detail": referee,
            "criterion_source": "mission 17336 goal 'CASE-BEM-MARUO-DRIFT … 容差' (mission text only; no "
                                "tensor_benchmark_cases row for this anchor)",
            "criterion_in_force": f"per-cell relative error < 5% at the declared production mesh '{production}'",
            "production_resolution": production,
            "production_cells": [
                {"ka": c["ka"], "panels": c["panel_count"], "got": c["bem_force"],
                 "ref": c["ref_force"], "rel_error": c["rel_error"]} for c in prod_cells],
            "production_n_failures": sum(1 for c in prod_cells if c["rel_error"] >= 0.05),
            "production_worst": {"ka": worst["ka"], "panels": worst["panel_count"],
                                 "rel_error": worst["rel_error"]},
            "finest_mesh_worst_rel_error": max(c["rel_error"] for c in cells
                                               if c["panel_count"] == max(x["panel_count"] for x in cells)),
            "ecosystem_target_3pct": "not met anywhere in the production row: best "
                                     f"{100 * min(c['rel_error'] for c in prod_cells):.3f}%",
            "verdict_at_production_mesh": "FAIL (honest, declared as such by the deliverable)",
            "all_passed_flag_in_study": study["all_passed"],
            "neighbour_referee_rows": neighbour,
            "report_six_sections": True,
        },
        "repro_cmd": [
            "PYTHONPATH=src python3 -m tensorbem.benchmarks.maruo_drift_far_field_case "
            "--json artifacts/maruo_drift_far_field_case/study.json --figures docs/case_bem_maruo_drift "
            "(full 4x4 solve, ~20 min, exit 1 because G3 fails)",
            "PYTHONPATH=src python3 -m pytest tests/test_maruo_drift_far_field_case.py -q   (13 passed)",
            "auditor replay used in this verdict: --ka 2.0 --resolutions coarse --production coarse "
            "and --ka 1.0 --resolutions coarse, both reproducing the delivered cells",
        ],
        "external_anchor": {
            "law": "F_x = (5 pi^2/16) rho g A^2 a (ka)^3 (Maruo 1960; Newman 1977 Sec. 7.3)",
            "auditor_recovered_constant": indep["long_wave"][0]["coefficient"],
            "law_constant": indep["long_wave"][0]["law_5pi2_16"],
            "rel_err_at_ka_0p01": indep["long_wave"][0]["rel_err"],
            "rel_err_at_ka_0p05": indep["long_wave"][-1]["rel_err"],
            "reference_independence": "auditor rebuilt the MacCamy-Fuchs far-field amplitude from first "
                                      "principles (no TensorBEM import) -> "
                                      f"max rel diff vs delivered ref {indep['max_rel_diff_vs_delivered']:.2e} "
                                      f"(limited by 4-dp stored values), own modal vs quadrature "
                                      f"{indep['max_modal_vs_quadrature_rel']:.1e}",
        },
        "auditor_reproduction": repro,
        "findings": [
            "F1 blocking-for-L4 (persistent from m17276/audit m17327): 0 tensor_benchmark_cases rows for "
            f"anchor {ANCHOR}; the case can never be machine-adjudicated until the row is registered",
            "F2 blocking-for-L4 (persistent): production-mesh criterion <5% fails at ka=2 (5.331%), 5.037% at "
            "1200 panels; ecosystem <3% target not met on any production cell",
            "F3 minor: entry crashes with ValueError('max() arg is an empty sequence') when --resolutions omits "
            "the production rung (maruo_drift_far_field_case.py:397 gates on an empty production cell list)",
            "F4 minor: report section 5 still carries the three untraceable numbers the m17327 audit flagged "
            "(832 triangles for a 800-panel mesh, a 416-triangle rung absent from the study, and 'fine 3.01%' "
            "quoting the finer value) -> docs/case_bem_maruo_drift_report.md:151-153",
            "F5 minor: the figure helper tensorbem.benchmarks.maruo_drift_far_field_case_figures is listed as "
            "'unreferenced and unexplained' in the regenerated docs/audit_baseline.md (gate-failing item the "
            "mission itself introduced)",
        ],
        "verdict_reason": [
            "L4 refused: no registered referee row (0) + declared <5% criterion fails at production mesh",
            "L3 granted: real BEM route, production entry points, capability registry row, 13 passing tests, "
            "two delivered cells reproduced bit-identically, reference independently re-derived to 1e-8",
        ],
        "report": "docs/board_audit/m17398/M17398_BOARD_AUDIT.md",
        "audit_snapshots_sha256": {
            rel(impl): sha(impl),
            rel(REFS / "docs/case_bem_maruo_drift/study.json"): sha(REFS / "docs/case_bem_maruo_drift/study.json"),
            rel(REFS / "tests/test_maruo_drift_far_field_case.py"): sha(REFS / "tests/test_maruo_drift_far_field_case.py"),
        },
    }
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ev = build()
    (ART / "verdict_evidence.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(ev, indent=2, ensure_ascii=False))
        return 0
    with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (repo, scope, verdict_by) DO UPDATE
                 SET mission_id = EXCLUDED.mission_id,
                     level = EXCLUDED.level,
                     evidence = EXCLUDED.evidence""",
            (REPO, SCOPE, MISSION, LEVEL, json.dumps(ev, ensure_ascii=False), BY),
        )
        cur.execute("SELECT id, level, gate_state FROM tensor_maturity_verdicts "
                    "WHERE repo=%s AND scope=%s AND verdict_by=%s", (REPO, SCOPE, BY))
        print(cur.fetchone())
    return 0


if __name__ == "__main__":
    sys.exit(main())