#!/usr/bin/env python3
"""Board-audit verdict for m17292 (CASE-BEM-HULME-HEMISPHERE) -> tensor_maturity_verdicts.

Also registers the missing machine referee row in tensor_benchmark_cases (0 rows before
this audit) so the anchor's tolerance is judgeable without reading a mission goal text.
"""
import json
import pathlib
import subprocess

EVIDENCE = {
    "repo": "TensorBEM",
    "anchor": "CASE-BEM-HULME-HEMISPHERE",
    "audited_mission": 17292,
    "original_mission": 17181,
    "audited_commit": "agent/m17292@c687cbf (NOT an ancestor of origin/main ac57fda)",
    "level": 2,
    "level_rationale": (
        "L2 = real independent implementation (input->BEM solve->verdict, 771-line module, 6 tests, "
        "no placeholder/MOCK) exists on branch agent/m17292. L3 leg missing: nothing of the delivery is "
        "in origin/main (git ls-tree -r origin/main | grep hulme_hemisphere_case -> 0 hits) and the remote "
        "has no agent/m17292 ref, so no main-side caller/user can reach it; same cap as verdict #515 "
        "(F2_medium_unmerged_branch 'caps the level at L2'). L4 leg additionally missing: 0 registered "
        "referee rows before this audit and the external anchor is an in-repo re-derivation "
        "(paper_digits_transcribed=false)."
    ),
    "impl_files": [
        "agent/m17292@c687cbf:src/tensorbem/benchmarks/hulme_hemisphere_case.py:1-771 "
        "(13 callables: build_tasks/run_cell/reference/reference_ladder/evaluate/main/write_*; "
        "snapshot docs/board_audit/m17319/snapshot/hulme_hemisphere_case.py)",
        "agent/m17292@c687cbf:benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py:1-121 "
        "(shard driver; snapshot docs/board_audit/m17319/snapshot/run_case.py)",
        "origin/main:src/tensorbem/analytic.py:1138 hulme_hemisphere_radiation (pre-existing reference series, on main)",
        "origin/main:src/tensorbem/free_surface_solver3d.py solve_radiation_3d / solve_diffraction_3d (solver used, on main)",
    ],
    "entry": (
        "agent/m17292@c687cbf: python -m tensorbem.benchmarks.hulme_hemisphere_case "
        "--out benchmarks/verified/casepool_bem_hulme_hemisphere  |  "
        "python benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py "
        "--out benchmarks/verified/casepool_bem_hulme_hemisphere --shards 4 --parallel 4"
    ),
    "run_artifacts": [
        "docs/board_audit/m17319/snapshot/results.json (17 rows: band 256/1024 x 6 ka, ladder, chief_off)",
        "docs/board_audit/m17319/snapshot/reference.json (144-term series + 96->144 drift 6.4638e-05)",
        "docs/board_audit/m17319/snapshot/evidence.json",
        "docs/board_audit/m17319/snapshot/run.log (17 task lines + 4 gate lines x2 merges)",
        "work/m17292/benchmarks/verified/casepool_bem_hulme_hemisphere/{shard_0,shard_1,shard_2,shard_3}.json",
        "work/m17292/benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png (snapshot m17292_figure.png)",
        "docs/board_audit/m17319/M17319_hulme_audit.png (audit figure, runs/m17319_audit/make_audit_figure.py)",
    ],
    "tests": [
        "agent/m17292@c687cbf:tests/test_hulme_hemisphere_case.py (6 tests; audit ran the 4 fast ones -> 4 passed in 6.85 s; 2 slow not_run)",
        "test_geometry_precondition_holds_on_the_case_mesh",
        "test_hulme_reference_series_is_self_converged",
        "test_evaluate_passes_good_rows_and_rejects_out_of_tolerance_rows (gate can fail -> not decoration)",
        "test_committed_artefacts_reproduce_their_verdict",
    ],
    "integration_callers": [
        "agent/m17292@c687cbf:benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py:36 imports "
        "tensorbem.benchmarks.hulme_hemisphere_case and calls case.main()/case.build_tasks()",
        "agent/m17292@c687cbf:tests/test_hulme_hemisphere_case.py:30,45,54,68 call reference/run_cell/evaluate",
        "origin/main: NO caller (git ls-tree -r origin/main | grep hulme_hemisphere_case -> 0 files); "
        "remote branch agent/m17292 absent (git ls-remote --heads origin | grep -c m17292 -> 0)",
    ],
    "benchmark": {
        "registered_referee_rows_before_audit": 0,
        "registered_by_audit": "tensor_benchmark_cases row for CASE-BEM-HULME-HEMISPHERE (status pending; ref_source states printed table WITHHELD)",
        "tolerance_source_before": "mission 17292 goal text + tensor_case_pool.id=138 tol_policy=<5% (not machine-registered)",
        "case_mesh": "1024 constant panels (subdivisions 4); coarse 256",
        "max_abs_rel_error_1024": 0.011480004889853406,
        "worst_item_1024": "B33 @ ka=2 (+1.148%)",
        "gates": {"G1_band_error_below_5pct": True, "G2_geometry_precondition": True,
                  "G3_refinement_does_not_worsen": True, "G4_reference_self_converged": True},
        "geometry": {"volume_rel_error_1024": -0.005869620423973321,
                     "waterplane_rel_error_1024": -0.0016056069643816118,
                     "volume_rel_error_256_previous_round": 0.0232},
        "vs_independent_anchor": {
            "anchor": "tests/data/hulme_independent_limits.json (capytaine 3.1.dev0 Rankine BEM, 4-level Richardson, self-reported +/-1%)",
            "ka_covered": [1.0, 2.0],
            "worst_rel_diff_1024": 0.028905,
            "worst_item_1024": "B33 @ ka=2 (+2.89%)",
            "worst_rel_diff_256_previous_round": 0.065591,
            "verdict": "all 8 pinned cells <5% at 1024 panels; the 256-panel round failed this anchor at +6.56%",
        },
        "report_six_sections": True,
        "report_path": "docs/board_audit/m17319/snapshot/report.md (sections 1..6, verdict table G1-G4)",
        "tensor_report_id": 16307,
    },
    "repro_cmd": [
        "python benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py --out "
        "benchmarks/verified/casepool_bem_hulme_hemisphere --shards 4 --parallel 4",
        "python -m tensorbem.benchmarks.hulme_hemisphere_case --out benchmarks/verified/casepool_bem_hulme_hemisphere",
        "python benchmarks/verified/casepool_bem_hulme_hemisphere/evidence.py --strict",
        "python -m pytest tests/test_hulme_hemisphere_case.py -q",
    ],
    "external_anchor": (
        "Hulme (1982) J. Fluid Mech. 121:443-463 wave-free multipole formulation, re-implemented in-repo as "
        "tensorbem.analytic.hulme_hemisphere_radiation (144 terms, runtime evaluation, paper_digits_transcribed=false); "
        "printed-table digits WITHHELD (paper unavailable in sandbox); independent triangulation only via capytaine "
        "Richardson limit at ka=1,2 (tests/data/hulme_independent_limits.json, +/-1% uncertainty)"
    ),
    "findings": [
        "F1 blocking-for-L4 (high): delivery unmerged -- c687cbf not an ancestor of origin/main (ac57fda); "
        "origin/main has no hulme_hemisphere_case.py and no test_hulme_hemisphere_case.py; no remote agent/m17292 ref",
        "F2 blocking-for-L4 (high): origin/main report.md is still the m17181 FAIL round -- 'Verdict: FAIL', "
        "G2 None/FAIL, and an inconsistent parameter line 'case mesh | subdivisions 3 = 6400 constant panels' "
        "against its own '### case mesh (256 panels)' table header",
        "F3 blocking-for-L4 (medium-high): no referee row for the anchor (0 rows in tensor_benchmark_cases) and "
        "tensor_case_pool.id=138 still dispatched/mission_id=17181 (registered by this audit, status pending)",
        "F4 material (medium): mission asked for Hulme-1982 printed values; delivered reference is the in-repo "
        "analytic recomputation (disclosed honestly in report section 1 and reference.json)",
        "F5 material (low-medium): evidence.py 'recompute_vs_stored_drift=0.0 / reproduces_stored_rows=true' is a "
        "rel_error bookkeeping recompute, not a BEM solve replay (docstring discloses it); headline 'L1-L4 all met' is overstated",
        "F6 info: gates and mesh were pre-registered before the numbers; evaluate() rejects out-of-tolerance rows (gate can fail); "
        "raising 256->1024 panels is a legitimate repair of the G2 geometry gate, guarded by G3 monotonicity",
        "F7 info: run_artifacts live in the tracked benchmark dir (not gitignored runs/) but only on the unmerged branch",
    ],
    "checks_performed": [
        "temp-table read of results.json/reference.json/evidence.json/report.md, field by field (no number hand-copied)",
        "pytest 4 fast tests in the m17292 worktree -> 4 passed in 6.85 s",
        "git merge-base --is-ancestor c687cbf origin/main -> false; git ls-tree -r origin/main | grep hulme -> old 4 files only",
        "recomputed got-vs-capytaine-limit for all 8 pinned cells at 256 and 1024 panels (audit figure, panel b)",
        "programmatic report.md vs results.json value check",
    ],
    "checks_not_run": [
        "cluster single-cell replay cj1494 (ka in {0.5,1.0,2.0} x 1024 panels with main's solver + the audited driver): "
        "queued 615 ahead at audit close -> not_run, result not used",
        "tests/test_hulme_hemisphere_case.py slow tests (test_run_cell_solves_a_real_cell..., test_evidence_bundle_checks_all_pass) -> not_run",
        "Hulme 1982 printed-table comparison (source unavailable in sandbox)",
        "independent recomputation of the 144-term multipole series internals",
        "ka>2 extended band, full pytest suite, wall_seconds authenticity",
    ],
    "overstated_claims": [
        "mission headline 'L1-L4 all met' -> L3/L4 legs fail (unmerged delivery, no referee row, printed anchor WITHHELD)",
        "'ref 0 hand-copied' is literally true but must not be read as 'agrees with Hulme 1982 printed values'",
        "evidence 'recompute_vs_stored_drift=0.0' read as an independent re-solve",
    ],
    "disposition": "reopened (request changes): land agent/m17292 into main; keep the referee row honest; "
                   "obtain Hulme 1982 printed digits or downgrade the anchor wording; fix the main-side stale "
                   "FAIL report and its '6400 panels' line; state that the evidence L4 leg is table arithmetic",
    "audit_report": "docs/board_audit/m17319/M17319_BOARD_AUDIT.md",
    "snapshot_dir": "docs/board_audit/m17319/snapshot/",
}


def main() -> None:
    ev = json.dumps(EVIDENCE, ensure_ascii=False)
    assert "$$" not in ev
    sql = (
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
        "VALUES ('TensorBEM', 'm17292 CASE-BEM-HULME-HEMISPHERE', 17319, 2, $json$" + ev + "$json$::jsonb, "
        "'board-audit') ON CONFLICT (repo, scope, verdict_by) DO NOTHING;"
    )
    pathlib.Path("/tmp/verdict_m17319.sql").write_text(sql)
    r = subprocess.run(["su", "postgres", "-c",
                        "psql -d tensorplatform -v ON_ERROR_STOP=1 -f /tmp/verdict_m17319.sql"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)

    ref_src = (
        "Hulme (1982) J. Fluid Mech. 121:443-463 wave-free multipole formulation re-implemented in-repo "
        "(tensorbem.analytic.hulme_hemisphere_radiation, 144 terms, runtime evaluation; printed table WITHHELD); "
        "independent triangulation capytaine 3.1.dev0 Richardson limit +/-1% at ka=1,2 "
        "(tests/data/hulme_independent_limits.json)"
    )
    metric = (
        "6 dimensionless quantities (A11/(rho*nabla), B11/(rho*omega*nabla), A33/(rho*nabla), "
        "B33/(rho*omega*nabla), |Fx|/(rho*g*A*a^2), |Fz|/(rho*g*A*a^2)) x ka in {0.2,0.5,0.8,1,1.5,2}; "
        "relative error vs the wave-free multipole series on the 1024-panel case mesh "
        "(signed 2026-11-13 by board-audit m17319; measured max 1.148% at 1024 panels, worst vs the capytaine "
        "independent limit 2.89%)"
    )
    sql2 = (
        "INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status) "
        "SELECT 'CASE-BEM-HULME-HEMISPHERE', "
        "'Floating hemisphere (a=1 m, flat face flush with the free surface, deep water, rho=1025, g=9.80665) "
        "radiation (surge, heave) + head-on diffraction', $r$" + ref_src + "$r$, $m$" + metric + "$m$, "
        "'<5% per item per ka', 'TensorBEM', 'pending' "
        "WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-HULME-HEMISPHERE');"
    )
    pathlib.Path("/tmp/referee_m17319.sql").write_text(sql2)
    r2 = subprocess.run(["su", "postgres", "-c",
                         "psql -d tensorplatform -v ON_ERROR_STOP=1 -f /tmp/referee_m17319.sql"],
                        capture_output=True, text=True)
    print(r2.stdout, r2.stderr)


if __name__ == "__main__":
    main()