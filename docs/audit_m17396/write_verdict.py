#!/usr/bin/env python3
"""Enroll the missing platform referee row and insert the board-audit maturity verdict (m17396).

Two writes, both idempotent:
  1. tensor_benchmark_cases — the CASE-BEM-IRREG-FREQ-CYL referee row.  It did not exist at audit
     time (0 rows; carry-over finding F1 from m17304).  The metric registered here is the
     NON-tautological one (predictor vs the solver's own spurious band, Route C); the mission's
     own metric (predictor vs the identity it implements) is disclosed as non-validating.
  2. tensor_maturity_verdicts — level 3 (system integration), verdict_by='board-audit'.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = json.loads((HERE / "audit_metrics.json").read_text())
LAD = M["C2C3_ladder"]
R1 = [r for r in LAD if r["label"] == "R1_T0.5"]
fine = max(R1, key=lambda r: r["n_theta"])
RC = M["C5_route_C"]
RB = M["C4_route_A_vs_B"]
GAP_LO = min(abs(m["gap_pct"]) for m in RB)
GAP_HI = max(abs(m["gap_pct"]) for m in RB)
A = "docs/audit_m17396/"

REF_SOURCE = (
    "Route C (empirical, adjudicating): spurious band of TensorBEM's own bare radiation solve "
    "(src/tensorbem/benchmarks/irregular_frequency_cylinder_exact.py observed_spurious_band -> "
    "locate_peak on the condition-number sweep, passivity loss as signature; John 1950 / Hulme 1982 / "
    "Zhu 1993 for the interior free-surface condition). Auxiliary, NON-validating: j_{m,1}/R with "
    "j_{m,1} from Abramowitz & Stegun Table 9.5 (30-digit re-check vs mpmath, max |diff| 4.44e-16) -- "
    "the library literals ARE those constants, so that comparison carries no information about the "
    "prediction (see m17396 audit). Route B (free-surface-consistent cylinder eigenvalue, "
    "tan(kz*T)=kz/nu) sits %.1f-%.1f%% away and is deliberately NOT the registered criterion: the "
    "discrete solver follows the rigid-lid mode." % (GAP_LO, GAP_HI)
)

METRIC = (
    "|nu_pred(m=0) - nu_obs| / nu_obs, nu_obs = conditioning/passivity blow-up band of the bare BEM "
    "solve of a surface-piercing cylinder (R=1 m, T/R=1, mean-drift mode index 2, nu grid 2.1-2.8 "
    "rad/m, parabola-refined peak); nu_pred = predict_irregular_frequencies(mesh) = j_{0,1}/R_wp(mesh). "
    "Board audit m17396 re-measured on 2026-11-13: nu_obs=%.4f (264 panels, +%.2f%%) and nu_obs=%.4f "
    "(112 panels, +%.2f%%); predictor value 2.4048 rad/m. Disclosed non-validating companion number: "
    "the mission's own metric (nu vs j/R) reads %.4f%% at %d waterline panels, which equals the "
    "inscribed-polygon area deficit pi^2/(3N^2) to <1.5%% and is therefore a geometry check only "
    "(identity residual %.1e). Artifacts: %saudit_metrics.json, %sAUDIT_REPORT.md (worktree m17396)."
    % (RC[1]["nu_observed"], RC[1]["err_vs_route_A_pct"], RC[0]["nu_observed"],
       RC[0]["err_vs_route_A_pct"], 100 * fine["max_nu_err"], fine["n_theta"],
       M["C2_error_is_geometry_identity"]["max_identity_residual"], A, A)
)

TOLERANCE = (
    f"<3% on the refined mesh (264 panels, nu_obs={RC[1]['nu_observed']:.4f}). "
    f"The 112-panel rung measures +{RC[0]['err_vs_route_A_pct']:.2f}% and is recorded as a "
    "sub-criterion FAIL, not as a pass. Mission m17374's own 2% figure is NOT this criterion "
    "(it scored the identity nu = j/R against itself)."
)

CASE_NAME = (
    "BEM-IRREG-FREQ-CYL irregular wavenumber of a surface-piercing cylinder: predictor vs the "
    "solver's own spurious band (route C), with rigid-lid Bessel-zero route A and free-surface route B "
    "kept as non-adjudicating references"
)

EVIDENCE = {
    "impl_files": [
        "src/tensorbem/irregular_frequency.py",
        "src/tensorbem/diffraction_drift_maturity.py",
        "src/tensorbem/benchmarks/irregular_frequency_cylinder_exact.py",
    ],
    "impl_lines": {
        "predict_irregular_frequencies": "src/tensorbem/irregular_frequency.py:190-239",
        "bessel_literals": "src/tensorbem/irregular_frequency.py:82-86",
        "waterplane_equivalent_radius": "src/tensorbem/irregular_frequency.py:164-187",
        "route_B_exact_cylinder_modes": "src/tensorbem/benchmarks/irregular_frequency_cylinder_exact.py:237-261",
        "route_C_observed_spurious_band": "src/tensorbem/benchmarks/irregular_frequency_cylinder_exact.py:352-415",
    },
    "entry": [
        A + "ref_m17374/scripts/run_irregular_frequency_bessel.py",
        A + "ref_m17374/scripts/benchmark_irregular_frequency_bessel.py",
        A + "ref_m17374/scripts/integration_caller_irregular_frequency.py",
    ],
    "run_artifacts": [
        A + "ref_m17374/prediction.json",
        A + "ref_m17374/ref_vs_got.json",
        A + "ref_m17374/benchmark.json",
        A + "ref_m17374/integration_check.json",
        A + "ref_m17374/bessel_zero_comparison.png",
        A + "ref_m17374/convergence_error.png",
        A + "ref_m17374/report_figure.png",
        A + "ref_m17374/run_log.txt",
        A + "ref_m17374/repro_entry.log",
        A + "ref_m17374/repro_bench.log",
        A + "ref_m17374/repro_tests.log",
        A + "ref_m17374/repro_integration.log",
        A + "ref_m17374/provenance.json",
        A + "audit_metrics.json",
        A + "fig_geom_scaling.png",
        A + "fig_route_abc.png",
        A + "AUDIT_REPORT.md",
    ],
    "tests": [
        A + "ref_m17374/tests/test_irregular_frequency_bessel_entry.py (13 passed on rerun)",
        A + "ref_m17374/tests/test_irregular_frequency_cylinder_exact.py (23 passed on rerun)",
    ],
    "integration_callers": [
        "predict_irregular_frequencies",
    ],
    "benchmark": {
        "registered_row": "tensor_benchmark_cases anchor_slug='CASE-BEM-IRREG-FREQ-CYL' (enrolled by this audit)",
        "adjudicating": {
            "route_C_nu_observed_264_panels": RC[1]["nu_observed"],
            "route_C_nu_observed_112_panels": RC[0]["nu_observed"],
            "route_A_predicted": RC[0]["nu_route_A_predicted"],
            "err_pct_refined": RC[1]["err_vs_route_A_pct"],
            "err_pct_coarse": RC[0]["err_vs_route_A_pct"],
            "tolerance_pct": 3.0,
            "pass_refined": abs(RC[1]["err_vs_route_A_pct"]) < 3.0,
            "pass_coarse": abs(RC[0]["err_vs_route_A_pct"]) < 3.0,
        },
        "mission_own_metric_is_tautological": {
            "got_vs_ref_pct_by_N": {str(r["n_theta"]): 100 * r["max_nu_err"] for r in R1},
            "identity_err_equals": "a/R_wp - 1 for every mode",
            "max_identity_residual": M["C2_error_is_geometry_identity"]["max_identity_residual"],
            "polygon_model_pi2_3N2_pct": {str(r["n_theta"]): 100 * r["polygon_model_err"] for r in R1},
        },
        "route_B_gap_pct": {f"T/R={m['draft_ratio']}_m={m['m']}": round(m["gap_pct"], 2) for m in RB},
        "constants_vs_mpmath_max_abs_diff": M["C1_constants"]["max_abs_diff"],
        "mode_coverage": M["C6_modes"],
    },
    "repro_cmd": (
        "cd /root/tp-agent/work/m17374 && PYTHONPATH=src python scripts/run_irregular_frequency_bessel.py "
        "&& PYTHONPATH=src python scripts/benchmark_irregular_frequency_bessel.py "
        "&& PYTHONPATH=src python scripts/integration_caller_irregular_frequency.py "
        "&& PYTHONPATH=src python -m pytest tests/test_irregular_frequency_bessel_entry.py "
        "tests/test_irregular_frequency_cylinder_exact.py -q   # auditor reran 2026-11-13: exit 0, "
        "36 passed, artifacts bit-identical (git diff empty); logs in " + A + "ref_m17374/repro_*.log. "
        "Independent audit route: cd <worktree m17396> && python3 docs/audit_m17396/audit_irreg_freq_cyl.py"
    ),
    "external_anchor": (
        "Abramowitz & Stegun Table 9.5 (j_{0,1},j_{1,1},j_{2,1}) re-derived to 30 digits with mpmath here "
        "(max abs diff 4.44e-16); John (1950) / Hulme (1982) / Zhu (1993) for the interior free-surface "
        "condition (route B, tan(kz T)=kz/nu); the solver-side anchor is route C, the passivity-loss band "
        "of TensorBEM's own bare radiation solve."
    ),
    "findings": {
        "F1_benchmark_is_self_referential (blocking for L4)": (
            "mission's ref and got share the Bessel constants; residual == a/R_wp - 1 to 1.0e-13 and equals "
            "the inscribed-polygon deficit pi^2/(3N^2) at all 8 rungs -> the 0.0201%-0.3222% ladder measures "
            "waterplane area discretisation, not the Bessel prediction"
        ),
        "F2_referee_row_absent (blocking for L4)": (
            "tensor_benchmark_cases had 0 rows for anchor_slug='CASE-BEM-IRREG-FREQ-CYL' at audit time "
            "(carry-over from m17304 F1); enrolled by this audit with the non-tautological route-C metric"
        ),
        "F3_route_B_gap_not_disclosed": (
            "free-surface-consistent eigenvalue sits %.2f-%.2f%% away from the shipped predictor; the mission "
            "report does not mention this, so 'within 2%% of the reference' can be misread as physical accuracy"
            % (GAP_LO, GAP_HI)
        ),
        "F4_branch_not_merged": (
            "commit 4339e14 on agent/m17374 is NOT an ancestor of origin/main; the three scripts/tests are "
            "absent from the trunk checkout, so the delivered artifacts are not reproducible from main"
        ),
        "F5_no_new_functional_substance": (
            "the audited commit touches only src/tensorbem/__init__.py plus docs/scripts/tests; the predictor "
            "itself is pre-existing main code (m16777/m14563), so the mission adds evidence packaging, not capability"
        ),
        "F6_route_C_mesh_sensitivity": (
            "auditor's route-C rerun gives nu_obs=%.4f (112 panels) / %.4f (264 panels) vs the value 2.45 "
            "documented by m16777 -> the empirical anchor carries ~2%% resolution uncertainty"
            % (RC[0]["nu_observed"], RC[1]["nu_observed"])
        ),
        "F7_mode_coverage_still_capped": (
            "public API returns at most the three first-order zeros; j_{0,2}=%.6f unreachable "
            "(carry-over from m17304 F3)" % M["C6_modes"]["j02_rad_per_m"]
        ),
    },
    "checks_run": [
        "mission repro_cmd x4 (entry/benchmark/integration/pytest) — exit 0, 36 passed, artifacts bit-identical",
        "independent re-measurement audit_irreg_freq_cyl.py (C1 mpmath 30-digit, C2 identity, C3 panel ladder x8, "
        "C4 route B, C5 route C x2 meshes, C6 API coverage)",
        "git merge-base --is-ancestor 4339e14 origin/main -> false",
    ],
    "checks_not_run": [
        "route C on non-circular hulls (Wigley / Wigley III) — not_run, so no cross-geometry claim",
        "second-order radial zeros (n>=2) — not implemented by the API, cannot be run",
        "CHIEF/interior-surface passivity restoration quantified on large meshes — not_run; the mission's own "
        "integration_check.json reports passivity_restored=false for both hulls without explanation",
    ],
    "level_downgrade_reason": (
        "claimed L4. L4 requires an independent benchmark; the mission's benchmark compares the predictor with "
        "the identity it implements (ref and got share the Bessel literals), so it validates waterplane geometry "
        "only. Functional substance = real library implementation + solver-side callers + reproducible artifacts "
        "-> L3, unchanged from the m17304 verdict (#534) for the same anchor. The anchor does hold a genuine "
        "non-tautological validation (route C, +1.42% refined), but it predates this mission (m16777), so it "
        "cannot upgrade this mission's level; a successor mission should adopt it as the registered criterion."
    ),
    "prior_verdict": {"id": 534, "mission_id": 17304, "level": 3, "verdict_by": "board-audit"},
}


def esc(s: str) -> str:
    return s.replace("'", "''")


def psql(sql: str) -> str:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-c", sql],
                       capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


bench_sql = f"""
INSERT INTO tensor_benchmark_cases
    (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
SELECT 'CASE-BEM-IRREG-FREQ-CYL', '{esc(CASE_NAME)}', '{esc(REF_SOURCE)}', '{esc(METRIC)}',
       '{esc(TOLERANCE)}', 'TensorBEM', 'passed', now()
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-IRREG-FREQ-CYL')
RETURNING id;
"""

verdict_sql = f"""
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorBEM', 'm17374 CASE-BEM-IRREG-FREQ-CYL', 17396, 3,
        '{esc(json.dumps(EVIDENCE, ensure_ascii=False))}'::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
   SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id,
       gate_state = 'pending', gate_note = NULL
RETURNING id, level, gate_state;
"""

(HERE / "writes.sql").write_text(bench_sql + "\n" + verdict_sql)
print("[benchmark]", psql(bench_sql))
print("[verdict]", psql(verdict_sql))
print("[benchmark row]",
      psql("SELECT id, status, tolerance FROM tensor_benchmark_cases "
           "WHERE anchor_slug='CASE-BEM-IRREG-FREQ-CYL'"))