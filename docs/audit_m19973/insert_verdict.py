#!/usr/bin/env python3
"""Insert the board-audit maturity verdict for mission 19973 (CASE-BEM-HELMHOLTZ-SPHERE)."""
from __future__ import annotations

import json

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")

EVIDENCE = {
    "impl_files": (
        "src/tensorbem/acoustic3d.py:763 solve_rigid_scattering, :857 far_field_amplitude, "
        ":896 scattering_cross_section, :1129 rigid_sphere_mie_coefficients, "
        ":1152 rigid_sphere_scattering_cross_section (all pre-existing on main 635415e1; "
        "capability introduced f62f33c6 — mission reused it, as the task allowed)"
    ),
    "entry": (
        "runs/audit_m19973/repro/run_study.py (byte-identical copy, sha256 "
        "listed in runs/audit_m19973/AUDITED_CASE_SHA256.txt of the m19973 original "
        "benchmarks/verified/case_bem_helmholtz_sphere/run_study.py)"
    ),
    "run_artifacts": (
        "audited payload copies runs/audit_m19973/audited_case/{results.json (14 rows), report.md (6 sections), "
        "fig_scattering_cross_section.png, fig_convergence.png, fig_far_field.png, fig_error_vs_ka.png, report.json}; "
        "auditor rerun runs/audit_m19973/repro/{results.json, fig_*.png} + repro.log; "
        "original location /root/tp-agent/work/m19973/benchmarks/verified/case_bem_helmholtz_sphere/ (UNTRACKED — see F1)"
    ),
    "tests": (
        "tests/test_acoustic3d.py::test_rigid_sphere_scattering_matches_mie (5% tol, ka 0.5/1/2, 1280 panels) "
        "and ::test_rigid_sphere_mie_coefficients_are_finite_and_small_in_rayleigh_limit; "
        "auditor run 2026-11-14: pytest -k 'sphere or mie or neumann' -> 3 passed in 74.91 s. "
        "No new test was added by the mission (F3) and the only failing point ka=0.25 is not covered."
    ),
    "integration_callers": (
        "src/tensorbem/acoustic3d_pulse.py:58, src/tensorbem/benchmarks/acoustic3d_pulse_envelope.py:52, "
        "src/tensorbem/benchmarks/acoustic3d_pulse_ref.py:80 import acoustic3d as ac; "
        "tests/test_acoustic3d.py:32, tests/test_acoustic3d_far_field_verification.py:26"
    ),
    "benchmark": {
        "machine_referee_row": (
            "ABSENT: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
            "WHERE anchor_slug='CASE-BEM-HELMHOLTZ-SPHERE' -> 0 rows (table 121 rows / 84 anchors; "
            "the only Helmholtz-family anchor is CASE-BEM-ACOUSTIC-PULSE id=46, a different case) "
            "-> platform cannot judge the declared <6% tolerance"
        ),
        "declared_tolerance": "<6% (mission goal), never registered in tensor_benchmark_cases",
        "got_vs_ref": (
            "14 rows, sigma_BEM vs Mie series: 320 panels err = 6.119/5.656/3.898/1.884/1.733/1.751/1.063% at "
            "ka=0.25/0.5/1/2/3/4/5; 1280 panels err = 1.572/1.449/0.990/0.492/0.676/0.556/0.285% -> 13/14 inside 6%, "
            "max 6.119% (ka=0.25, 320 panels, honestly declared FAIL)"
        ),
        "ecosystem_3pct_question": (
            "key quantity <3%? NO for the 320-panel tier (3 rows >3%: ka=0.25 6.119%, ka=0.5 5.656%, ka=1.0 3.898%); "
            "YES for the whole 1280-panel tier (max 1.572%)"
        ),
    },
    "repro_cmd": (
        "cd runs/audit_m19973/repro && OMP_NUM_THREADS=1 python3 run_study.py -> exit 0, 14/14 rows: "
        "sigma_bem/sigma_mie/relative_error/pass identical to the payload (max |delta sigma| = 0.0), "
        "only solve_time_s varies (0.96-1.51x); log runs/audit_m19973/repro.log"
    ),
    "external_anchor": (
        "Mie series independently re-derived by the auditor in 4 routes (runs/audit_m19973/indep_check.py, "
        "indep_check.json): A mpmath Riccati-Bessel series, B direct angular integration of f(theta), "
        "C optical theorem, D Rayleigh limit (7pi/9)(ka)^4 a^2 -> routes agree with the repo reference to "
        "1.1e-15 (A), 8.7e-15 (B), 1.4e-16 (C); Rayleigh check 3.8e-4 at ka=0.02. Reference leg = REAL."
    ),
    "reference_literature": (
        "Bowman, Senior & Uslenghi (1969) 'Electromagnetic and Acoustic Scattering by Simple Shapes'; "
        "Johnson (1993) lecture notes Table 3.1 p.47 — cited in results.json:reference_source; "
        "article-level only, no table digits transcribed (the numbers are computed, not copied)"
    ),
    "findings": {
        "F1_blocking_delivery": (
            "the whole deliverable is untracked: in worktree /root/tp-agent/work/m19973 "
            "`git status --porcelain` = '?? benchmarks/verified/case_bem_helmholtz_sphere/' and "
            "`git log --oneline -- <dir>` is empty; mission result records branch=None. A fresh checkout of "
            "main 635415e1 (audit worktree m19997) does NOT contain the directory -> reproduction is tied to a "
            "recyclable worktree. Remediation: commit the directory."
        ),
        "F2_blocking_referee": (
            "no tensor_benchmark_cases row for the anchor (0 rows) -> the 'only judge' cannot rule on the "
            "declared <6% tolerance; L4 refused on this ground."
        ),
        "F3_medium_regression": (
            "case absent from benchmarks/verified/README.md (grep HELMHOLTZ-SPHERE outside the case dir = 0 hits); "
            "no pytest added; the single FAIL point ka=0.25 is outside the existing 5%/ka{0.5,1,2} test, "
            "so CI would not catch a regression of the exact failure mode the mission reported."
        ),
        "F4_low_overstatement": (
            "report.md section 5.1 prose 'Convergence ratio ~3.7-3.9x' contradicts its own table (2.6x, 3.2x) and "
            "names a ratio of errors a convergence ratio; only 2 mesh tiers -> observed order not estimable."
        ),
        "F5_low_friction_unverifiable": (
            "section 6.2 says 'Submitted to cluster but queue was busy (15+ jobs ahead), fell back to local run'; "
            "compute_gate_events has exactly one row for mission 19973 (compute_submitted 2026-11-14T07:23:14) with "
            "no queue-depth/kill/timeout record -> only partially verifiable (a 14-case local run of ~3 min is "
            "inside the <10 min local rule, so the fallback itself was compliant)."
        ),
        "F6_info_positive": (
            "no MOCK/STUB: sigma values are produced by a real dense Helmholtz BEM solve plus far-field quadrature; "
            "tensor_reports id=18218 markdown is md5-identical (8092c919e0d56f69618c90a8deb67990) to the local "
            "report.md, six sections present, 4 figures embedded."
        ),
        "F7_info_fail_cause_confirmed": (
            "the ka=0.25/320-panel 6.12% failure is geometric discretisation, not a bug or quadrature noise: "
            "far-field quadrature refinement 64->192 points changes sigma by <=5.3e-15 (auditor probe), the error "
            "ratio between the two tiers is 3.89 while the inscribed-icosphere area deficit ratio is 3.95, and the "
            "bias is negative in 14/14 rows (inscribed mesh underestimates the effective radius)."
        ),
    },
    "level_rationale": (
        "L3 confirmed: real functional path (dense exterior Helmholtz BEM + far-field integral + Mie reference), "
        "in-package integration, pytest gate passing (3 passed), auditor rerun bit-identical, and an analytic "
        "reference leg re-derived 4 independent ways to 1e-15. L4 refused only on delivery state: F1 (0 commits) + "
        "F2 (0 referee rows) make 'reproducible' and 'judgeable' unreachable from the platform side; the scientific "
        "substance (complete 14-case study, error analysis, quadrature-null probe, 4 figures) would support L4 after "
        "a remediation round. Not L5: single sphere, chapter-level analytic anchor, no experiment/printed table."
    ),
    "disposition": (
        "request changes (reopened for a remediation round): (1) commit benchmarks/verified/case_bem_helmholtz_sphere/; "
        "(2) register the tensor_benchmark_cases referee row for the anchor; (3) add the ka=0.25 failing point to "
        "pytest and index the case in benchmarks/verified/README.md; (4) restate section 5.1 convergence wording."
    ),
    "checks_not_run": [
        "5120-panel (sub=4) tier not run (dense 5120^2 complex system ~420 MB, O(n^3) above the local short-test rule)",
        "cross-check against an independent BEM implementation (Capytaine/bempp/commercial) — not run, no second exterior Helmholtz BEM in the estate",
        "rest of tests/test_acoustic3d.py (14 of 17 cases deselected) — not_run",
        "pixel-level inspection of the 4 delivered PNGs — not done",
        "queue depth '15+ jobs ahead' of the claimed cluster submission — not_verifiable (single compute event row)",
    ],
    "verdict_doc": "docs/audit_m19973_case_bem_helmholtz_sphere.md",
    "audit_artifacts": [
        "docs/audit_m19973_case_bem_helmholtz_sphere.md",
        "runs/audit_m19973/indep_check.py", "runs/audit_m19973/indep_check.json",
        "runs/audit_m19973/repro/run_study.py", "runs/audit_m19973/repro/results.json",
        "runs/audit_m19973/repro.log", "runs/audit_m19973/audited_case/",
        "runs/audit_m19973/AUDITED_CASE_SHA256.txt",
        "runs/audit_m19973/figs/fig1_bem_vs_mie.png",
        "runs/audit_m19973/figs/fig2_error_vs_ka.png",
        "runs/audit_m19973/figs/fig3_reference_and_repro.png",
        "runs/audit_m19973/make_audit_figures.py",
    ],
}

ROW = ("TensorBEM", "m19973 CASE-BEM-HELMHOLTZ-SPHERE", 19997, 3,
       json.dumps(EVIDENCE, ensure_ascii=False), "board-audit")


def main() -> None:
    conn = psycopg.connect(DSN, connect_timeout=10)
    with conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id, gate_state, gate_level, gate_note",
            ROW,
        )
        rid, state, lvl, note = cur.fetchone()
        print("inserted verdict id:", rid, "| gate_state:", state, "| gate_level:", lvl)
        print("gate_note:", json.dumps(note, ensure_ascii=False) if note else None)
    conn.close()


if __name__ == "__main__":
    main()