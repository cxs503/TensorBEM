#!/usr/bin/env python3
"""board-audit m17232: emit verdict.sql (maturity verdict + benchmark case row) and land it.

Evidence is assembled from analysis.json (artifact-derived) with repo-relative
pointers so the maturity gate can verify each file exists in this worktree.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
A = json.loads((HERE / "analysis.json").read_text())
rows = A["rows"]
best = rows[-1]
sub2 = rows[1]
audit_md = HERE / "AUDIT_m17176.md"

evidence = {
    "subject": "m17176 CASE-BEM-SPHERE-ADDED-MASS (TensorBEM); audited by m17232, verdict_by=board-audit",
    "impl_files": [
        "src/tensorbem/benchmarks/sphere_added_mass.py:34 run() (icosphere -> solve_laplace_surface(domain='exterior') -> boundary-energy added mass)",
        "src/tensorbem/benchmarks/sphere_added_mass.py:114 main() CLI (--subdivisions/--radius/--speed/--density/--output)",
        "src/tensorbem/solver3d.py (solve_laplace_surface, exterior Dirichlet operator: constant panels, c=1/2, Duffy quadrature)",
        "src/tensorbem/geometry3d.py (icosphere)",
        "src/tensorbem/analytic.py:415 ellipsoid_added_mass (closed-form reference; sphere limit alpha=2/3)",
    ],
    "entry": "PYTHONPATH=src python3 -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2 --output out.json",
    "run_artifacts": [
        f"docs/board_audit/m17232/refs/sphere_added_mass_sub{{1,2,3,4}}.json (subject artifacts, sha256 {A['sha256']['refs/sphere_added_mass_sub4.json'][:16]}... for sub4)",
        "docs/board_audit/m17232/refs/flux_panels_sub2.csv (320 rows: panel,cos_theta,q_computed,q_exact,area)",
        "docs/board_audit/m17232/refs/sensitivity_sub2.json",
        "docs/board_audit/m17232/rerun/sub{0,1,2,3}.json (auditor reruns; sub1/sub2/sub3 bit-identical to subject)",
        "docs/board_audit/m17232/fig_audit_sphere_added_mass.png",
        "docs/board_audit/m17232/analysis.json (audit-computed order table, README cross-check, findings)",
    ],
    "tests": [
        "tests/test_sphere_benchmark.py (auditor run 2026-11-13: 18 passed, 106 s)",
        "tests/test_milestone_claims.py::sphere_added_mass_{20,80,320,1280} (doc-claim pins on the same README numbers)",
        "docs/board_audit/m17232/negcontrol_sphere_added_mass.py (+ .json: wrong angular mode cos2theta -> 64.17% off ref; radius x2 -> exactly 8x)",
    ],
    "integration_callers": [
        "src/tensorbem/benchmarks/milestone_claims.py:381 (sphere_added_mass.run invoked by the doc-claim harness)",
        "src/tensorbem/benchmarks/three_dimensional_study.py (3-D study driver, same module)",
        "README.md:42 (documented CLI entry) / README.md:62-63 (declared errors 20/80/320/1280)",
        "tests/test_sphere_benchmark.py:9 (imports run)",
    ],
    "benchmark": {
        "anchor": "CASE-BEM-SPHERE-ADDED-MASS",
        "metric": "relative error of the BEM added mass vs the classical closed form m_A=0.5*rho*V (rho=1025, R=1)",
        "ref": "0.5*rho*4/3*pi*R^3 = %.10f kg (Lamb 1932 §6.10; repo closed form analytic.ellipsoid_added_mass((1,1,1),1025)=2146.7549799481294, rel diff 2.28e-12)" % best["ref_0p5_rhoV"],
        "got": {str(r["panels"]): r["m_A_subject"] for r in rows},
        "rel_error": {str(r["panels"]): r["rel_err_subject"] for r in rows},
        "worst_pass": "320 panels: %.4e" % sub2["rel_err_subject"],
        "finest": "5120 panels: %.4e (%.4f%%)" % (best["rel_err_subject"], best["rel_err_subject"] * 100),
        "all_refinements_under_3pct_from_320": True,
        "observed_order_integral": [round(o["p_log2"], 3) for o in A["integral_order"]],
        "observed_order_flux": [round(o["p_log2"], 3) for o in A["flux_order"]],
        "auditor_rerun": {str(r["panels"]): r["m_A_rerun"] for r in rows if r["m_A_rerun"] is not None},
        "rerun_bit_identical": all(r["rel_err_bit_identical"] for r in rows if r["m_A_rerun"] is not None),
        "criterion_source": "tensor_benchmark_cases had 0 rows for this anchor before this audit (F2); <3% re-derived from the mission goal and now registered by this audit",
    },
    "repro_cmd": "cd <TensorBEM worktree> && PYTHONPATH=src python3 -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2  # auditor rerun 2026-11-13: sub1/sub2/sub3 bit-identical to subject artifacts",
    "external_anchor": "classical closed form 0.5*rho*V (Lamb 1932 Hydrodynamics §6.10; Newman 1977 eq.4.132 sphere limit), machine-evaluated twice (analytic.py:415 ellipsoid_added_mass + explicit 0.5*rhow*4/3*pi) -> agreement 2.28e-12. NOT L5: no experimental or printed-table transcription.",
    "scope_limits": [
        "Dirichlet-trace verification: exact analytic u imposed, q solved -> validates the exterior single-layer/DtN operator, not a moving-body Neumann BVP (F4).",
        "flat-faceted icosphere: sub2 volume ratio 0.96616, so errors include faceting error; refinement study is the meaningful statement.",
        "no free-surface / wave-body / ship-hull physics (declared in the module note).",
        "sub4 (5120 panels) rerun was still running at delivery; sub1/sub2/sub3 reproduced bit-identically.",
    ],
    "findings": A["findings"],
    "audit": {
        "report": str(audit_md.relative_to(REPO)),
        "figure": "docs/board_audit/m17232/fig_audit_sphere_added_mass.png",
        "review_method": "read-only forensics on /root/tp-agent/work/m17176 + independent rerun in /root/tp-agent/work/m17232 (branch agent/m17232)",
        "checks_not_run": ["sub4 auditor rerun (in flight)", "cluster job cj1466 (queued)", "experimental anchor (none exists for sphere added mass)", "free-surface applicability"],
        "disposition": "approve with notes (F1 convergence-order arithmetic to be corrected; README was right)",
    },
    "level_rationale": "L4: real BEM solve with negative control + classical closed-form anchor + 4-point refinement error table + one-command repro reproduced bit-identically + tests green. Not L5: no experimental/printed-table anchor, trace-level scope, and the subject report's order analysis is wrong (F1).",
    "auditor_rerun_matches_subject_artifact": {
        "sub1": rows[0]["m_A_abs_diff"], "sub2": rows[1]["m_A_abs_diff"], "sub3": rows[2]["m_A_abs_diff"],
        "max_abs_dev_added_mass": max(r["m_A_abs_diff"] for r in rows if r["m_A_abs_diff"] is not None),
    },
    "sha256": A["sha256"],
}

sql_verdict = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)\n"
    "VALUES ('TensorBEM', 'm17176 CASE-BEM-SPHERE-ADDED-MASS', 17232, 4, $json$%s$json$::jsonb, 'board-audit')\n"
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending'\n"
    "RETURNING id, gate_state, gate_level;" % json.dumps(evidence, ensure_ascii=False)
)

metric = ("relative error of the BEM added mass of a translating sphere vs the classical closed form 0.5*rho*V; "
          "got: 80p %.6e, 320p %.6e, 1280p %.6e, 5120p %.6e (all <3%% from 320 panels, finest %.4f%%); "
          "auditor rerun sub1/sub2/sub3 bit-identical; audited by m17232"
          % (rows[0]["rel_err_subject"], rows[1]["rel_err_subject"], rows[2]["rel_err_subject"],
             best["rel_err_subject"], best["rel_err_subject"] * 100))
sql_case = (
    "INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)\n"
    "VALUES ('CASE-BEM-SPHERE-ADDED-MASS', 'Translating sphere added mass, unbounded fluid (R=1 m, rho=1025, flat-faceted icosphere 80-5120 panels)', "
    "'Lamb 1932 Hydrodynamics §6.10 / Newman 1977 eq.4.132 sphere limit m_A=0.5*rho*V (closed form, machine-evaluated in src/tensorbem/analytic.py:415)', "
    "$m$%s$m$, '<3%% (mission criterion); measured finest 0.113%%', 'TensorBEM', 'passed', now());" % metric
)
sqlfile = HERE / "insert_verdict.sql"
sqlfile.write_text(sql_verdict + "\n" + sql_case + "\n", encoding="utf-8")

p = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-v", "ON_ERROR_STOP=1", "-f", str(sqlfile)],
                   capture_output=True, text=True)
print("STDOUT:", p.stdout)
print("STDERR:", p.stderr[-800:])
q = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tAc",
                    "SELECT id, level, gate_state, gate_level, gate_note FROM tensor_maturity_verdicts "
                    "WHERE repo='TensorBEM' AND verdict_by='board-audit' ORDER BY id DESC LIMIT 2"],
                   capture_output=True, text=True)
print("GATE:", q.stdout, q.stderr[-400:])