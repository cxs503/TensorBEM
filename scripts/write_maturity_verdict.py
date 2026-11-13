#!/usr/bin/env python3
"""Write the m17235 maturity verdict for CASE-BEM-HULME-HEMISPHERE.

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
with the evidence JSON built from the audit artefacts on this branch, so the
pointers (file:line, artefact paths, test names, measured numbers) cannot drift
from what was actually measured.
"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((HERE / "artifacts/audit_summary.json").read_text())
REPRO = json.loads((HERE / "artifacts/repro_check.json").read_text())

REF_WORST = SUMMARY["worst"]["ref_vs_independent_limit"]
GOT_IND = SUMMARY["worst"]["got_vs_independent_limit"]
GOT_REF = SUMMARY["worst"]["got_vs_ref"]
GEOM = SUMMARY["geometry"]

evidence = {
    "subject_mission": 17181,
    "audited_branch": "agent/m17181",
    "level_declared_by_mission": None,
    "impl_files": [
        "src/tensorbem/free_surface_solver3d.py:464 solve_radiation_3d (Kelvin/Havelock Green-function BIE, constant panels)",
        "src/tensorbem/free_surface_solver3d.py:577 solve_diffraction_3d",
        "src/tensorbem/analytic.py:1138 hulme_hemisphere_radiation (wave-free multipole series reference, n_terms=144, quad_order>=2*n_terms)",
        "src/tensorbem/geometry3d.py:581 hemisphere_hull_mesh (octahedral subdivision, 4*4^n panels)",
    ],
    "entry": (
        "branch agent/m17181 (NOT merged into origin/main): "
        "benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py "
        "--shard i --shards 4 ... --merge"
    ),
    "run_artifacts": [
        "/root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere/results.json (rows, reference, ref_ladder, verdict)",
        "/root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere/report.md (107 lines)",
        "/root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere/figure.png",
        "/root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere/reference.json",
        "/root/tp-agent/work/m17181/runs/casepool_bem_hulme_hemisphere/shard_0..3.json",
        "platform report id=16187 (tensor_reports, mission 17181)",
    ],
    "tests": [
        "branch agent/m17181 tests/test_casepool_hulme_hemisphere.py - 4 passed, incl. mutation test (injected 20% error must flip G1 to FAIL)",
        "origin/main tests/test_hulme_diffraction_verification.py, tests/test_kochin_hulme_convergence.py - pre-existing Hulme coverage in main",
        "this audit: tests/test_hulme_reference_vs_independent_limits.py - 11 passed in 61.5 s (reference vs independent capytaine limits, 3% band + truncation convergence)",
    ],
    "integration_callers": (
        "solver is in origin/main (src/tensorbem/free_surface_solver3d.py) and callable by library users/tests; "
        "the case driver is NOT: git ls-tree -r origin/main | grep casepool_bem_hulme_hemisphere -> 0 hits "
        "(only docs/feat_casepool_100_bem_expansion.md, scripts/expand_casepool_bem.py exist) -> artifacts are branch-only"
    ),
    "benchmark": {
        "platform_judge": (
            "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
            "anchor_slug='CASE-BEM-HULME-HEMISPHERE' -> 0 rows (table has 35 rows) => machine judge absent; "
            "tensor_case_pool.id=138 still status='dispatched'"
        ),
        "criterion_used": "mission goal tolerance <5% + mission pre-registered gates G1-G4",
        "G1_pass_vs_selfderived_ref": SUMMARY["mission_gates"]["G1_case_mesh_band_error_below_5pct"]["measured"],
        "G2_geometry_gate": f"FAIL: mesh volume error {GEOM['mesh_volume_error']:.6f} > 2% gate ({GEOM['gate']}), waterplane {GEOM['mesh_waterplane_error']:.6f}",
        "mission_own_verdict": SUMMARY["mission_verdict"],
        "worst_cell_vs_selfderived_ref": f"{100*GOT_REF['value']:+.2f}% ({GOT_REF['dof']} {GOT_REF['quantity']} @ka={GOT_REF['ka']}) -> PASS",
        "worst_cell_vs_independent_limit": f"{100*GOT_IND['value']:+.2f}% ({GOT_IND['dof']} {GOT_IND['quantity']} @ka={GOT_IND['ka']}) -> FAIL of the same 5% tolerance",
    },
    "repro_cmd": {
        "cmd": REPRO["repro_cmd"],
        "measured": f"{REPRO['cells_compared']}/12 band cells reproduce the frozen results.json; max |diff| = {REPRO['max_abs_diff']:.1e} (log precision)",
        "log": REPRO["rerun_log"],
    },
    "external_anchor": {
        "published_table_digits": (
            "ABSENT by the mission's own admission: results.json reference_provenance.paper_digit_transcription = "
            "'NOT AVAILABLE in this sandbox (publisher paywall); the reference is the re-derived series' "
            "-> the mission goal requirement 'digits from the paper/manual, cite original table page' is UNMET"
        ),
        "substitute_evidence": [
            "docs/capytaine_crosscheck.md + benchmarks/verified/capytaine_crosscheck/results.json: frozen third-party capytaine 3.1.dev0 at 1296 panels, worst deviation 8.51% at 256 TensorBEM panels (heave damping nu=2)",
            f"this audit: own capytaine ladder 400/784/1296/1936 faces + 3-point Richardson limit; reference vs independent limit worst {100*REF_WORST['value']:+.2f}% ({REF_WORST['dof']} {REF_WORST['quantity']} @ka={REF_WORST['ka']}) -> series credible, published digits still unverified",
        ],
    },
    "independent_measurement": {
        "artefact": "artifacts/independent_hemisphere_check.json + artifacts/audit_summary.json (this branch)",
        "script": "scripts/independent_hemisphere_check.py, scripts/summarize_audit.py",
        "cells_outside_5pct_vs_independent_limit": sum(
            1 for c in SUMMARY["cells"] if abs(c["err_got_vs_ind_limit"]) > 0.05
        ),
        "cells_total": len(SUMMARY["cells"]),
    },
    "level_rationale": (
        "L3 = integration: real solver (no MOCK/STUB), complete reproducible case with deterministic re-execution "
        "(12/12 cells), mesh/truncation/CHIEF sensitivity axes. NOT L4: the benchmark-verification leg fails under an "
        f"independent judge ({100*GOT_IND['value']:+.2f}% > 5% tolerance at {GOT_IND['dof']} {GOT_IND['quantity']} @ka={GOT_IND['ka']}), "
        "the reference has no published-table anchor, the pre-registered geometry gate G2 FAILs (2.32% > 2%), and the "
        "case artefacts are not in origin/main. NOT L2: the case is a full band study, not a demo."
    ),
    "findings": [
        "F1 blocking-for-L4: G1 PASS is reference-dependent - worst cell 4.76% vs self-derived series but 6.56% vs capytaine Richardson limit (256 constant panels carry ~6.6% discretisation error, same h^2 origin as the 2.32% mesh-volume deficit)",
        "F2 blocking-for-L4: reference is a re-derived multipole series, not Hulme (1982) table digits (provenance says NOT AVAILABLE); independent corroboration is only <=1.72%",
        "F3 doc: report.md SS1 omits the 'paper digits unavailable' caveat that results.json honestly records; SS3 says 6400/1600 panels vs actual 256/64; SS5.1 mesh-ladder table has header only; SS6 G3 measured=None",
        "F4 bookkeeping: tensor_benchmark_cases has 0 rows for the anchor and tensor_case_pool.id=138 is still 'dispatched'",
        "F5 integration: case driver/tests/doc exist only on unmerged branch agent/m17181",
        "F6 process: mission status='succeeded' (auto-approve dev_acceptance) coexists with its own verdict=FAIL",
    ],
    "checks_not_run": [
        "1024-panel ladder NOT a gap any more: this audit replayed the full 17-cell run (ladder 1024 panels worst 1.15% vs self-derived ref, chief_off 4.53%); see docs/board_audit/m17235/hulme_hemisphere_audit.md SS8 and artifacts/full_replay.json",
        "Hulme (1982) original table verification (no access to the paper)",
        "--merge document regeneration not replayed (the 17 compute cells were)",
    ],
    "audit_artifacts": [
        "docs/board_audit/m17235/hulme_hemisphere_audit.md",
        "docs/board_audit/m17235/hulme_hemisphere_errors.png",
        "artifacts/audit_summary.json",
        "artifacts/independent_hemisphere_check.json",
        "artifacts/repro_check.json",
        "artifacts/full_replay.json",
        "tests/test_hulme_reference_vs_independent_limits.py",
    ],
}

SQL = """
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (%s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT (repo, scope, verdict_by)
DO UPDATE SET level = EXCLUDED.level, evidence = EXCLUDED.evidence,
              mission_id = EXCLUDED.mission_id, created_at = now()
RETURNING id, gate_state
"""

if __name__ == "__main__":
    with psycopg.connect("dbname=tensorplatform user=postgres") as conn:
        with conn.cursor() as cur:
            cur.execute(
                SQL,
                (
                    "TensorBEM",
                    "m17181 CASE-BEM-HULME-HEMISPHERE",
                    17235,
                    3,
                    json.dumps(evidence, ensure_ascii=False),
                    "board-audit",
                ),
            )
            print("verdict row:", cur.fetchone())
        conn.commit()