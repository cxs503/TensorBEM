#!/usr/bin/env python3
"""Repoint verdict 810 evidence paths from runs/ (git-ignored) to docs/audit_m19973/
and append the .gitignore finding F8.  Keeps the evidence text otherwise intact."""
from __future__ import annotations

import json

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")

NEW = {
    "entry": (
        "docs/audit_m19973/repro/run_study.py (byte-identical copy, sha256 in "
        "docs/audit_m19973/AUDITED_CASE_SHA256.txt, of the m19973 original "
        "benchmarks/verified/case_bem_helmholtz_sphere/run_study.py)"
    ),
    "run_artifacts": (
        "audited payload copies docs/audit_m19973/audited_case/{results.json (14 rows), report.md (6 sections), "
        "fig_scattering_cross_section.png, fig_convergence.png, fig_far_field.png, fig_error_vs_ka.png}; "
        "auditor rerun docs/audit_m19973/repro/results.json + docs/audit_m19973/repro.log; "
        "original location /root/tp-agent/work/m19973/benchmarks/verified/case_bem_helmholtz_sphere/ "
        "(UNTRACKED — F1)"
    ),
    "repro_cmd": (
        "cd docs/audit_m19973/repro && OMP_NUM_THREADS=1 python3 run_study.py -> exit 0, 14/14 rows: "
        "sigma_bem/sigma_mie/relative_error/pass identical to the payload (max |delta sigma| = 0.0), "
        "only solve_time_s varies (0.96-1.51x); log docs/audit_m19973/repro.log"
    ),
    "external_anchor": (
        "Mie series independently re-derived by the auditor in 4 routes (docs/audit_m19973/indep_check.py, "
        "docs/audit_m19973/indep_check.json): A mpmath Riccati-Bessel series, B direct angular integration of "
        "f(theta), C optical theorem, D Rayleigh limit (7pi/9)(ka)^4 a^2 -> routes agree with the repo reference "
        "to 1.1e-15 (A), 8.7e-15 (B), 1.4e-16 (C); Rayleigh check 3.8e-4 at ka=0.02. Reference leg = REAL."
    ),
    "audit_artifacts": [
        "docs/audit_m19973_case_bem_helmholtz_sphere.md",
        "docs/audit_m19973/indep_check.py", "docs/audit_m19973/indep_check.json",
        "docs/audit_m19973/repro/run_study.py", "docs/audit_m19973/repro/results.json",
        "docs/audit_m19973/repro.log", "docs/audit_m19973/audited_case/",
        "docs/audit_m19973/AUDITED_CASE_SHA256.txt",
        "docs/audit_m19973/figs/fig1_bem_vs_mie.png",
        "docs/audit_m19973/figs/fig2_error_vs_ka.png",
        "docs/audit_m19973/figs/fig3_reference_and_repro.png",
        "docs/audit_m19973/make_audit_figures.py",
        "docs/audit_m19973/insert_verdict.py", "docs/audit_m19973/post_report_m19997.py",
    ],
    "findings_F8": (
        "F8 medium (ecosystem): this repo's .gitignore excludes runs/ and artifacts/, so a mission that parks "
        "evidence there cannot commit it; m19973 used benchmarks/verified/ (not ignored) but never committed "
        "either. Remediation for the successor round: commit under benchmarks/verified/case_bem_helmholtz_sphere/ "
        "or docs/."
    ),
}


def main() -> None:
    conn = psycopg.connect(DSN, connect_timeout=10)
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT evidence FROM tensor_maturity_verdicts WHERE id=810")
        ev = cur.fetchone()[0]
        ev.update(NEW)
        cur.execute("UPDATE tensor_maturity_verdicts SET evidence=%s WHERE id=810",
                    (json.dumps(ev, ensure_ascii=False),))
        cur.execute("SELECT gate_state,gate_level FROM tensor_maturity_verdicts WHERE id=810")
        print("verdict 810 updated; gate:", cur.fetchone())


if __name__ == "__main__":
    main()