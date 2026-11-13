"""m17302 board-audit landing script: build evidence, dry-run the maturity gate,
optionally INSERT the verdict (+ the missing tensor_benchmark_cases row).

usage: python land_verdict.py            # dry run: print gate verdict, no writes
       python land_verdict.py --commit   # INSERT benchmark row + verdict row
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

PGENV = {"PGPASSWORD": "uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz", "PATH": "/usr/bin:/bin"}
HERE = "docs/board_audit/m17302"
AUDIT_MISSION = 17302
SUBJECT_MISSION = 17265

TESTS = [
    "tests/test_cathodic_sphere_anchor.py::test_current_scale_within_ledger_tolerance",
    "tests/test_cathodic_sphere_anchor.py::test_second_order_convergence",
    "tests/test_cathodic_sphere_anchor.py::test_unscaled_current_defect_is_pinned",
    "tests/test_cathodic_sphere_anchor.py::test_nondimensional_scaling[3.5-1.0--1.0]",
    "tests/test_cathodic_sphere_anchor.py::test_nondimensional_scaling[2.0-1.0--1.0]",
    "tests/test_cathodic_sphere_anchor.py::test_nondimensional_scaling[3.5-0.5--1.0]",
    "tests/test_cathodic_sphere_anchor.py::test_two_current_routes_agree",
    "tests/test_cathodic_sphere_anchor.py::test_far_field_rejects_enclosing_radius_inside_body",
    "tests/test_cathodic_sphere_anchor.py::test_benchmark_cli_handles_single_level",
    "tests/test_cathodic_sphere_anchor.py::test_benchmark_cli_routes_block",
]

EVIDENCE = {
    "subject_mission": SUBJECT_MISSION,
    "subject_branch": "agent/m17265",
    "subject_head_commit": "912bec2",
    "impl_files": [
        f"{HERE}/snapshot/cathodic_protection_farfield_excerpt.py",
        f"{HERE}/snapshot/cathodic_sphere_dnv_b401_study.py",
        f"{HERE}/snapshot/cathodic_protection_sphere.py",
    ],
    "impl_notes": {
        "cathodic_protection_farfield_excerpt.py": "verbatim excerpt of the m17265 delta in src/tensorbem/cathodic_protection.py:1179-1352 (far_field_current + verify_conducting_sphere_routes + fibonacci_sphere_directions; no pass/NotImplemented body)",
        "cathodic_sphere_dnv_b401_study.py": "347-line study driver: convergence / two-route / scaling / decay -> results.json",
        "cathodic_protection_sphere.py": "benchmark CLI (+40 lines: --routes-subdivisions block and the single-level observed-order format fix)",
    },
    "entry": [
        f"{HERE}/snapshot/cathodic_sphere_dnv_b401_study.py",
        f"{HERE}/snapshot/cathodic_protection_sphere.py",
    ],
    "run_artifacts": [
        f"{HERE}/snapshot/results.json",
        f"{HERE}/snapshot/figures/convergence_current.png",
        f"{HERE}/snapshot/figures/exterior_decay.png",
        f"{HERE}/snapshot/figures/error_budget.png",
        f"{HERE}/analysis.json",
        f"{HERE}/fig_audit_cathodic_sphere.png",
        "artifacts/board_audit_m17302/pytest_auditor_rerun.log",
        "artifacts/board_audit_m17302/study_rerun.log",
    ],
    "run_artifacts_note": (
        "snapshot/* = byte copies (sha256 in AUDIT_m17265.md) of the audited branch; analysis.json + "
        "fig_audit_cathodic_sphere.png = the auditor's own geometry-deficit recomputation; pytest log = "
        "auditor rerun of the subject test file in a clean sandbox (/tmp/m17302_subj, PYTHONPATH=src)"
    ),
    "tests": TESTS,
    "tests_rerun": "auditor: python -m pytest tests/test_cathodic_sphere_anchor.py -q -> 10 passed in 69.39 s (first independent run, 03:49) ; log artifacts/board_audit_m17302/pytest_auditor_rerun.log",
    "integration_callers": [],
    "integration_gap": (
        "L3 refused: `git merge-base --is-ancestor 912bec2 HEAD` in the ECO main checkout (89e5a52) says the "
        "subject commit is NOT in main, and grep -rn 'far_field_current|verify_conducting_sphere_routes|"
        "cathodic_sphere_dnv_b401' over the whole main worktree returns 0 hits. The new far-field route is "
        "reachable only from the unmerged branch's own benchmark CLI and its own test file; no main-side "
        "solver/platform caller exists, and solve_cathodic_protection's production behaviour is unchanged."
    ),
    "benchmark": {
        "anchor": "CASE-BEM-CATHODIC-SPHERE",
        "metric": "relative error of the BEM total anode current I vs the DNV-RP-B401 isolated-sphere closed form I = 4*pi*sigma*a*phi_a (R = 1/(4 pi sigma a))",
        "ref": "4*pi*3.5*1.0*(-1.0) = -43.982297150257104 A ; R = 0.022736420441699334 ohm (auditor recomputed from sigma/a/phi == artifact to 0.0 absolute)",
        "got_A": {"80": -41.820245453598446, "320": -43.41020975732543, "1280": -43.83636638179202},
        "rel_error": {"80": 0.049157316, "320": 0.013007219, "1280": 0.003317943},
        "tolerance": "ledger 5% (case-pool) / internal 3%; measured finest 0.3318% -> both met; the 80-panel rung 4.9157% fails the 3% internal target (sub-criterion, disclosed)",
        "far_field_route": "extrapolated I = -43.83636653553718 A (0.3318%), surface vs far-field 3.51e-09",
        "observed_order": [1.918, 1.971],
        "auditor_extra": "facet-area-deficit probe: error/deficit = 0.686/0.691/0.696 (proportional, slope 0.6864, max resid 1.8e-03) -> the residual error is inscribed-icosphere geometry, not boundary quadrature",
        "registry": "tensor_benchmark_cases had 0 rows for this anchor_slug before this audit; registered by this audit with the audited numbers",
    },
    "repro_cmd": (
        "cd <worktree of agent/m17265> && PYTHONPATH=$PWD/src python scripts/cathodic_sphere_dnv_b401_study.py "
        "--subdiv 1 2 3 --routes-subdiv 2 3 --scaling-subdiv 2 && PYTHONPATH=$PWD/src python -m pytest "
        "tests/test_cathodic_sphere_anchor.py -q   # auditor reran the pytest half: 10 passed"
    ),
    "external_anchor": [
        f"{HERE}/snapshot/REPORT.md (§1: DNV-RP-B401 Cathodic Protection Design, isolated-sphere spreading "
        "resistance R = rho/(4 pi a) = 1/(4 pi sigma a); classical closed form, machine-evaluated in "
        "src/tensorbem/cathodic_protection.py verify_conducting_sphere)",
    ],
    "external_anchor_quality": (
        "closed form is exact and machine-evaluated (auditor recomputed independently: identical to 1e-16); "
        "but it is an analytic identity, not an experimental / printed-table transcription -> no L5, and the "
        "anchor cannot speak about polarization kinetics or the mixed Neumann/Dirichlet (holiday) branch"
    ),
    "declared_level": 2,
    "level_rationale": (
        "L2 独立功能实现：真代码（外域 Gauss 定律双路电流 far_field_current + 347 行研究驱动 + benchmark CLI 扩展），"
        "入口可执行，运行工件非空且数值可复算（审计独立重算闭式解逐位一致、10 条测试 auditor 复跑通过、几何亏损探针复现误差机理）。"
        "不授 L3：subject 分支未并入 main（git merge-base 判否），main 全树对新符号 0 引用，新路径不被任何主线求解/平台调用，"
        "solve_cathodic_protection 生产行为未变——即无系统集成面。不授 L4：L3 已是前提未过；且 5120 面板（level 4）工件缺失"
        "（集群 cj1471 至审计时仍 queued），误差表只到 1280 面板；算例池/平台机器裁判行在本次审计前不存在（已由本审计登记）。"
        "不授 L5：锚点是解析恒等式（无实验/独立实现对照），且模型无极化动力学。"
    ),
    "findings": {
        "F1_medium_benchmark_registry_gap": {
            "measured": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-CATHODIC-SPHERE' -> 0 rows (table has 41 rows; none for this anchor)",
            "impact": "the platform had no machine referee for this anchor; the numbers were only self-declared in the report/artifacts",
            "action": "registered by this audit (tensor_benchmark_cases: metric/tolerance from the case-pool criterion, values from the audited artifacts)",
            "severity": "medium (bookkeeping, not numerics)",
        },
        "F2_medium_unmerged_branch": {
            "measured": "subject commit 912bec2 not an ancestor of main 89e5a52; 0 main-side references to far_field_current / verify_conducting_sphere_routes",
            "impact": "the deliverable (tests + study + benchmark CLI extension + the far-field route) is invisible to every main-branch user; on main the anchor still has only the old single route",
            "severity": "medium -> caps the level at L2",
        },
        "F3_info_error_is_geometry_not_quadrature": {
            "measured": "error/facet-area-deficit = 0.686 / 0.691 / 0.696 across 80/320/1280 panels (slope 0.6864 through origin, max resid 1.8e-03)",
            "impact": "the 0.3318% headline is an inscribed-facet (chord-vs-arc) effect of the icosphere, and it is common to both current routes; the report's §6.3 sentence 'the boundary quadrature is certified by the closed form and the second-order convergence' is therefore OVERSTATED — on a uniformly polarised sphere the exact single-layer density is constant, so this anchor exercises the geometric measure + the sigma/4pi bookkeeping, not the Duffy/self-term quadrature accuracy that matters for a general hull",
            "severity": "info / claim-scope (numbers unaffected)",
        },
        "F4_low_sign_and_R_convention": {
            "measured": "spreading_resistance_numeric_ohm = -phi_a/I = -0.023036055 (negative) vs exact +0.022736420; the report quotes the magnitude (report tables label 'R_num (magnitude)')",
            "impact": "cosmetic but a foot-gun: a consumer comparing numeric vs exact R directly gets a sign mismatch",
            "severity": "low",
        },
        "F5_low_level4_mesh_missing": {
            "measured": "ranial: 5120-panel rung absent from results.json; cluster job cj1471 'queued' at 34 min age when audited (no artifacts)",
            "impact": "the report only says level 4 'goes to the cluster queue' and claims no number, so no false claim; but the L4 mesh rung is unverified",
            "severity": "low (honest, but error table stops at h/a=0.099)",
        },
        "F6_info_far_field_route_not_independent": {
            "measured": "far_field_current consumes the same CathodicProtectionResult.q; surface-vs-far-field agreement 3.5e-09 is a consistency check of the exterior representation + Gauss bookkeeping",
            "impact": "the subject report discloses this (report §6.3 and the test docstring) — auditor confirms the disclosure is accurate; no credit may be booked as 'independent second solver'",
            "severity": "info (correctly disclosed)",
        },
    },
    "checks_not_run": [
        "5120-panel (level 4) rung: not_run — cluster job cj1471 still queued at audit time; auditor did not resubmit",
        "mixed Neumann/Dirichlet holiday branch: not_run — outside this anchor by construction (subject discloses)",
        "polarization kinetics / measured protection potentials: not_run — linear model has no kinetics (subject discloses)",
    ],
    "auditor_independence": (
        "auditor ran the tests in a separate sandbox copy (/tmp/m17302_subj, src/tests/scripts copied from the "
        "subject worktree) with PYTHONPATH set to that copy; the closed form and the geometry-deficit probe were "
        "recomputed from sigma/a/phi with tensorbem.geometry3d's icosphere only, no subject number reused as input. "
        "Limitation: the full study re-run (1280-panel S/H assembly) was still running on the loaded control node "
        "when the verdict was filed — the pytest half and all arithmetic reproductions are complete, the study "
        "rerun log is filed even if it finished after the verdict."
    ),
}


def run_judge():
    sys.path.insert(0, "/root/tp-agent")
    import maturity_gate

    return maturity_gate.judge(2, EVIDENCE, "TensorBEM", AUDIT_MISSION)


def sql(statement: str) -> str:
    r = subprocess.run(["psql", "-h", "127.0.0.1", "-U", "foundation", "-d", "tensorplatform",
                        "-tA", "-c", statement], capture_output=True, text=True, env=PGENV)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    return r.stdout


def main() -> int:
    commit = "--commit" in sys.argv
    gl, why = run_judge()
    print(json.dumps({"gate_level": gl, "why": why}, indent=1))
    if not commit:
        return 0
    ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
    bench_note = json.dumps(EVIDENCE["benchmark"], ensure_ascii=False).replace("'", "''")
    row = sql(
        "INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at) "
        "VALUES ('CASE-BEM-CATHODIC-SPHERE', "
        "'Bare conducting sphere, uniform potential, unbounded electrolyte (a=1 m, sigma=3.5 S/m, phi_a=-1 V, flat-faceted icosphere 80/320/1280 panels)', "
        "'DNV-RP-B401 Cathodic Protection Design, isolated-sphere spreading resistance R = rho/(4 pi a) = 1/(4 pi sigma a) (classical closed form, machine-evaluated)', "
        f"'{bench_note}', "
        "'<5% (case-pool ledger); ecosystem <3%; measured finest 0.3318% at 1280 panels, 1.3007% at 320', "
        "'TensorBEM', 'passed', now()) RETURNING id")
    print("benchmark_case_id", row.strip())
    vid = sql(
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES "
        f"('TensorBEM', 'm{SUBJECT_MISSION} CASE-BEM-CATHODIC-SPHERE', {AUDIT_MISSION}, 2, '{ev}', 'board-audit') "
        "RETURNING id")
    print("verdict_id", vid.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())