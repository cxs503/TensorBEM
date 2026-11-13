#!/usr/bin/env python3
"""Insert the board-audit verdict for mission m17278 (CASE-BEM-HULME-DIFF-VERIF) into
tensor_maturity_verdicts. Evidence values are real pointers inside this worktree."""
import json, subprocess, hashlib, pathlib

HERE = pathlib.Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256((HERE / p).read_bytes()).hexdigest()

numbers = json.loads((HERE / "audit_numbers.json").read_text())
rows = numbers["rows"]
EVIDENCE = {
    "impl_files": [
        "artifacts/m17345/m17278_raw/run_hulme_diffraction_reduced.py:1 (delivered entry; real solve, no stub)",
        "src/tensorbem/free_surface_solver3d.py:577 (solve_diffraction_3d - main 3D solver path)",
        "src/tensorbem/geometry3d.py:581 (hemisphere_hull_mesh sub=2/3/4 -> 64/256/1024 panels)",
        "src/tensorbem/analytic.py:1138 (hulme_hemisphere_radiation: spherical-harmonic multipole reference)",
    ],
    "entry": "artifacts/m17345/m17278_raw/run_hulme_diffraction_reduced.py",
    "run_artifacts": [
        "artifacts/m17345/m17278_raw/results_reduced.json (delivered 8 ka x {64,256} ladder, sha256 "
        + sha("m17278_raw/results_reduced.json") + ")",
        "artifacts/m17345/m17278_raw/report.json (delivered report payload, sha256 "
        + sha("m17278_raw/report.json") + ")",
        "artifacts/m17345/m17278_raw/fig_hulme_diffraction_magnitude.png / _error.png / _convergence.png "
        "(sha256 in artifacts/m17345/m17278_raw_sha256.txt)",
        "artifacts/m17345/audit_numbers.json (audit consolidation 64/256/1024, sha256 "
        + sha("audit_numbers.json") + ")",
        "artifacts/m17345/p1024_0.1_0.2.json, p1024_0.8_1.0.json, p1024_1.5.json, hulme_1024_probe.json "
        "(1024-panel level re-run by this audit, 7 of 8 ka)",
        "artifacts/m17345/fig_audit_error_vs_panels.png, fig_audit_excitation_dimensionless.png",
    ],
    "tests": [
        "tests/test_hulme_diffraction_verification.py (audit run 2026-11-13 in m17345 worktree, PYTHONPATH=src: 5 passed in 1.83s)",
        "test_heave_excitation_force_vs_hulme_ka1 (asserts <5%)",
        "test_surge_excitation_force_vs_hulme_ka1 (asserts <10%)",
        "test_haskind_reciprocity_with_direct_diffraction (asserts <2%)",
        "test_sway_near_zero_at_head_on (asserts <2% of |Fx|)",
        "NO test shipped by m17278 itself (finding F8) - the study is script+artifacts, not CI-covered",
    ],
    "integration_callers": [
        "artifacts/m17345/m17278_raw/run_hulme_diffraction_reduced.py:44 calls "
        "tensorbem.free_surface_solver3d.solve_diffraction_3d (src/tensorbem/free_surface_solver3d.py:577) - "
        "main solver layer, no new kernel",
        "same script:43 calls tensorbem.geometry3d.hemisphere_hull_mesh (geometry3d.py:581) - shared geometry factory",
        "same script:42 calls tensorbem.analytic.hulme_hemisphere_radiation (analytic.py:1138) - shared reference module",
    ],
    "benchmark": {
        "anchor_scope": "CASE-BEM-HULME-DIFF-VERIF (m17278 slug)",
        "criterion_source": ("tensor_benchmark_cases id=54 CASE-BEM-HULME-HEMISPHERE (registered by board-audit "
                            "m17319): 6 dimensionless quantities x ka in {0.2,0.5,0.8,1,1.5,2}, '<5% per item per "
                            "ka', mesh=1024 panels. FINDING F7: the mission's own slug CASE-BEM-HULME-DIFF-VERIF "
                            "has 0 rows in tensor_benchmark_cases (naming drift); NO duplicate row inserted by "
                            "this audit - judged against id=54."),
        "tolerance": 0.05,
        "delivered_256_panels": {
            "worst_abs_err": round(numbers["delivered_by_m17278"]["worst_abs_err_256"], 6),
            "items_pass": "16/16 items (8 ka x 2 DOF) inside 5%",
            "per_ka_err_surge_pct": {str(r["ka"]): round(r["levels"]["256"]["err_surge"] * 100, 3) for r in rows},
            "per_ka_err_heave_pct": {str(r["ka"]): round(r["levels"]["256"]["err_heave"] * 100, 3) for r in rows},
        },
        "audit_added_1024_panels": {
            "ka_covered": numbers["added_by_audit_m17345"]["ka_covered"],
            "worst_abs_err": round(numbers["added_by_audit_m17345"]["worst_abs_err_1024"], 6),
            "note": "ka=0.3 not run at 1024 (checks_not_run)",
        },
        "coarse_64_panels": "8/16 rows FAIL the 5% band (surge -5.9%..-8.1%) - contradicts the m17278 headline 'All 16 cases (8 ka x 2 mesh levels) pass <5%' (finding F0)",
        "convergence": {
            "observed_order_surge": {str(o[0]): [round(x, 3) for x in o[1]] for o in
                                     [(r["ka"], r["surge_observed_order"]) for r in rows if "surge_observed_order" in r]},
            "richardson_limit_abs_err": {str(r["ka"]): r["surge_richardson_limit_abs_err"] for r in rows
                                         if "surge_richardson_limit_abs_err" in r},
        },
        "numerics_health": "relative_residual ~2e-15, condition_number 2.18-2.72 over the 1024-panel runs",
        "reference_truncation_probe": "hulme_hemisphere_radiation n_terms in {24,32,48,64}: surge drift <=1.4e-5, heave <=1.4e-3; n_terms=48 default uncertainty <=2e-4 (audit probe 2026-11-13)",
    },
    "repro_cmd": ("mkdir -p /tmp/h && cp artifacts/m17345/m17278_raw/run_hulme_diffraction_reduced.py /tmp/h/ && "
                  "cd /tmp/h && PYTHONPATH=<worktree>/src python3 run_hulme_diffraction_reduced.py   # audit rerun "
                  "2026-11-13 in work/m17345: canonical (sort_keys) results_reduced.json IDENTICAL to the delivered "
                  "file; 1024 level: PYTHONPATH=<worktree>/src python3 artifacts/m17345/p1024.py 0.5 2.0"),
    "external_anchor": ("WITHHELD. No printed-table / experimental Hulme value was used: the 'ref' is the in-repo "
                        "multipole series tensorbem.analytic.hulme_hemisphere_radiation (analytic.py:1138), and "
                        "tests/data/hulme_independent_limits.json:provenance records 'the published Hulme table "
                        "digits were NOT AVAILABLE in the sandbox'. The mission goal demanded zero-handcopy "
                        "numbers from Hulme's original table - NOT met (finding F1). Independent external check "
                        "exists only for radiation A/B via capytaine (worst 1.72%, m17235), not for excitation "
                        "forces. Citation also wrong paper (Appl. Ocean Res. 4(4):239-244 instead of JFM "
                        "121:443-463, finding F2). Hence L5 unreachable; ref is nonetheless an independent "
                        "numerical method (panel BEM vs spherical-harmonic multipole), cross-validated by the "
                        "observed order 1.92-1.99 and Richardson residual ~1e-4."),
    "audit": {
        "verdict_doc": "docs/board_audit/m17345/AUDIT_m17278.md",
        "m17278_report_published": False,
        "m17278_report_check": "SELECT ... FROM tensor_reports WHERE mission_id=17278 -> 0 rows; mission.report_id IS NULL (finding F4)",
        "deliverables_untracked_in_m17278": "head_commit=null, git_status all '??' -> raw copies preserved byte-identically here with sha256 (finding F4)",
        "cluster_claim_check": "md claims sub=4/1024 submitted to cluster cj1476; cj1476 was still 'queued' 79 min after submission with no artifacts (finding F3) - audit re-ran the level locally",
        "checks_not_run": [
            "1024-panel ka=0.3 (7 of 8 ka covered)",
            "printed-table Hulme JFM 121:443-463 numbers (unavailable in sandbox) -> external anchor WITHHELD",
            "phase angle comparison (delivered study compares |F| only)",
            "capytaine independent cross-check of excitation forces (existing limits cover A/B only)",
            "PNG pixel content (backed by figures_digest.json + generator script)",
        ],
        "level_rationale": (
            "L4: real 3D BEM diffraction solve through the main solver path over 8 ka x 3 panel levels "
            "(1024 level supplied by this audit), per-item ref-vs-got error table (worst 2.12% at 256, 0.54% at "
            "1024), observed convergence order 1.92-1.99 with Richardson residual ~1e-4 (both numbers "
            "cross-validated), one-command repro verified byte-identical, numerics healthy (residual 2e-15, "
            "cond<2.8). Not L5: external printed/experimental anchor WITHHELD and the reference is an in-repo "
            "series (F1), citation mismatch (F2). Claim defects F0 (8/16 rows actually fail at 64 panels) and F5 "
            "(figure axis mislabelled by rho*g*A*a^2) are recorded as remediation, they do not change the "
            "functional substance for the finest delivered mesh."),
    },
}
ev = json.dumps(EVIDENCE, ensure_ascii=False)
assert "'" not in ev or True
sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
       "VALUES ('TensorBEM', 'm17278 CASE-BEM-HULME-DIFF-VERIF', 17345, 4, "
       "$json$" + ev + "$json$::jsonb, 'board-audit') RETURNING id, gate_state")
r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-c", sql],
                   capture_output=True, text=True)
print("rc:", r.returncode, "out:", r.stdout.strip(), "err:", r.stderr.strip()[:400])
