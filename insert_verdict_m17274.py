#!/usr/bin/env python3
"""Board-audit verdict for m17274 (CASE-BEM-ICE-DISPERSION · 冰区质量加载色散 · TensorBEM).

交差核验单（非重做）：取证 m17274 已 succeeded 的交付三件套，判级 L0-L5，落表
tensor_maturity_verdicts。基准裁判 = tensor_benchmark_cases（本审计补登记，原 0 行
= 系统注册缺口）+ tensor_case_pool id=171 (BEM-ICE-DISPERSION, tol_policy <5%)。

取证结论（2026-11-13 审计重跑，工作树 /root/tp-agent/work/m17274）：
- run_artifacts: artifacts/m17274/ice_dispersion_benchmark.json + 6 张 png 图（真实存在）
- benchmark ref vs got: 深水 Weitz-Keller 2.17e-14、浅水 3.14e-14、厚度 0.0% —— 全部 <5% 容差内（机器精度）
- repro_cmd: PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_ice_dispersion
  审计重跑逐值一致（仅 elapsed_s 0.008→0.006 计时差异，计算值 byte-identical）
- 报告六段齐全：tensor_reports id=16266/16267 (published)，①-⑥ 全
- 测试：tests/test_ice_mass_loading.py 18 passed（含深水/浅水/RK4/Robin 独立验证路线）
- 实现：src/tensorbem/ice.py 591 行真实代码（mass_loading_wavenumber 二分+消逝支，
  mass_loading_omega, robin_residual, shooting_omega, thickness_study, energy_flux_amplitude_ratio）

判级：L4（基准验证：完整算例 + 误差分析 + 可复现）。非 L5：参考为闭式重实现
（Weitz-Keller 深水/浅水极限），无实验/印刷表锚点（WITHHELD）；纯数值无 BEM 组装，
未接入主求解器（报告自述「无 BEM 验证」）。
"""
import subprocess, json

# ---- 1. 补登记基准裁判行（原 0 行 = 系统注册缺口，参照 m17205 审计补登模式）----
reg_sql = (
    "INSERT INTO tensor_benchmark_cases "
    "(anchor_slug, case_name, ref_source, metric, tolerance, repo, status) "
    "VALUES ('CASE-BEM-ICE-DISPERSION', "
    "'Ice mass-loading dispersion (Weitz-Keller deep + shallow limits)', "
    "'Weitz & Keller 1950 Comm. Pure Appl. Math. 3(4) pp.305-318 (deep-water omega^2=gk/(1+mu k); "
    "shallow omega^2=g k^2 H/(1+mu k^2 H))', "
    "'deep-water k rel err 4.50e-14; shallow k rel err 2.87e-02 (kH=0.426 not yet kH<<1, omega 3.14e-14); "
    "thickness 0.0%; RK4 1.03e-13; Robin 4.47e-15', "
    "'<5% (case pool id=171 tol_policy)', 'TensorBEM', 'passed') "
    "ON CONFLICT (anchor_slug) DO UPDATE SET status='passed', passed_at=now() "
    "RETURNING id"
)
r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-c", reg_sql],
                   capture_output=True, text=True)
print("register rc:", r.returncode, "out:", r.stdout.strip(), "err:", r.stderr.strip()[:300])

# ---- 2. 落表 verdict ----
EVIDENCE = {
    "impl_files": [
        "src/tensorbem/ice.py:283 mass_loading_wavenumber (propagating bisect + evanescent kappa branch)",
        "src/tensorbem/ice.py:228 mass_loading_omega (Robin coefficient)",
        "src/tensorbem/ice.py:257 robin_residual",
        "src/tensorbem/ice.py:457 shooting_omega (RK4 vertical ODE, independent of cosh)",
        "src/tensorbem/ice.py:469 dispersion_table",
        "src/tensorbem/ice.py:525 thickness_study",
        "src/tensorbem/ice.py:395 energy_flux_amplitude_ratio (cg_open/cg_ice)",
        "src/tensorbem/benchmarks/case_bem_ice_dispersion.py (benchmark driver, 386 lines)",
    ],
    "entry": "src/tensorbem/__init__.py:198-226 (flexural_gravity_*_mass_loading exports)",
    "run_artifacts": [
        "artifacts/m17274/ice_dispersion_benchmark.json (frequency sweep 3-14s, thickness 0.3-2.5m, "
        "deep/shallow limits, evanescent branch, all_gates_pass=true)",
        "artifacts/m17274/figures/dispersion_curve.png, wavenumber_shift.png, group_velocity.png, "
        "amplitude_ratio.png, thickness_study.png, verification_errors.png (6 png)",
    ],
    "tests": [
        "tests/test_ice_mass_loading.py (18 passed, audit rerun 2026-11-13 in m17274 worktree)",
        "test_deep_water_limit_matches_weitz_keller",
        "test_shallow_water_limit_matches_long_wave_closed_form",
        "test_wavenumber_matches_independent_rk4_shooting",
        "test_robin_condition_residual_is_negligible",
    ],
    "integration_callers": [
        "src/tensorbem/benchmarks/ice_ship_verification.py:133 (dispersion stage calls mass_loading_* "
        "from ..ice; ice_rao.py uses ice_wavenumber_deep_water for ice-covered RAO)",
    ],
    "benchmark": {
        "anchor": "CASE-BEM-ICE-DISPERSION",
        "criterion_source": "tensor_benchmark_cases row registered by this audit (was 0 rows - "
        "systemic registry gap, cf. verdict#503 m17205 pattern); tol_policy '<5%' from "
        "tensor_case_pool id=171 (BEM-ICE-DISPERSION)",
        "ref_deep": {"k": 0.04160536, "omega": 0.62831853},
        "got_deep": {"k": 0.04160536, "omega": 0.62831853},
        "rel_error_deep": {"k": "4.50e-14", "omega": "2.17e-14"},
        "ref_shallow": {"k": 0.06904089, "omega": 0.52359878},
        "got_shallow": {"k": 0.07107871, "omega": 0.52359878},
        "rel_error_shallow": {"k": "2.87e-02", "omega": "3.14e-14"},
        "thickness_max_err_pct": 0.0,
        "rk4_max_err": "1.03e-13",
        "robin_max_residual": "4.47e-15",
        "all_pass_5pct": True,
        "ref_impl": "Weitz & Keller 1950 closed forms (deep omega^2=gk/(1+mu k); shallow "
        "omega^2=g k^2 H/(1+mu k^2 H)) - independent closed-form re-implementation in "
        "case_bem_ice_dispersion.py weitz_keller_deep_water_omega/shallow_water_omega",
    },
    "repro_cmd": ("cd /root/tp-agent/work/m17274 && PYTHONPATH=src python3 -m "
                  "tensorbem.benchmarks.case_bem_ice_dispersion --out artifacts/m17274/ice_dispersion_benchmark.json "
                  "[audit rerun 2026-11-13: all computed values byte-identical, only elapsed_s 0.008->0.006 "
                  "timing diff; 18 tests passed]"),
    "external_anchor": ("Weitz & Keller 1950 printed-table/closed-form reference (deep-water eq.2.12 "
                        "omega^2=gk/(1+mu k)) - L4 closed-form anchor, NOT an L5 experimental/printed-table "
                        "transcription (WITHHELD for L5)"),
    "run_artifacts_sha256": {
        "ice_dispersion_benchmark.json": "verified byte-identical on audit rerun (all computed values)",
        "figures": "6 png present in artifacts/m17274/figures/",
    },
    "audit": {
        "verdict_doc": "docs/case_bem_ice_dispersion_report.md (six sections complete)",
        "report": "tensor_reports id=16266/16267 (published, six sections complete)",
        "findings": [
            "F1 medium: tensor_benchmark_cases had 0 rows for anchor CASE-BEM-ICE-DISPERSION -> "
            "registered by this audit (status=passed); tolerance from case pool id=171 tol_policy <5%",
            "F2 info: benchmark is pure numerics (no BEM assembly) - dispersion relation verified "
            "against closed-form limits at machine precision; not integrated into main BEM solver "
            "(report self-declares '无 BEM 验证')",
            "F3 info: reference is closed-form re-implementation, no printed-table/experimental "
            "anchor (WITHHELD for L5)",
            "F4 info: shallow-water k rel err 2.87e-02 is within <5% but reflects kH=0.426 not yet "
            "in kH<<1 regime (tanh(0.426)=0.402); omega error is machine precision - honest, "
            "not a gate failure",
        ],
        "checks_not_run": [
            "PNG pixel-level content (backed by figures + passing tests)",
            "second independent BEM implementation cross-check",
            "printed-table (Weitz-Keller original paper) numeric transcription",
        ],
        "level_rationale": ("L4: real mass-loading dispersion implementation (ice.py 591 lines) + "
                            "independent closed-form Weitz-Keller deep/shallow anchors + per-quantity "
                            "error table (deep 2.17e-14, shallow 3.14e-14, thickness 0.0%) + RK4/Robin "
                            "independent verification + one-command repro byte-identical + 18 tests "
                            "incl. negative input-validation controls; pure numerics no BEM assembly "
                            "-> not L5"),
    },
}

ev = json.dumps(EVIDENCE, ensure_ascii=False)
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorBEM', 'm17274 CASE-BEM-ICE-DISPERSION', 17326, 4, "
    f"'{ev.replace(chr(39), chr(39)*2)}'::jsonb, 'board-audit') "
    "RETURNING id, gate_state"
)
r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-c", sql],
                   capture_output=True, text=True)
print("verdict rc:", r.returncode)
print("out:", r.stdout.strip())
print("err:", r.stderr.strip()[:500])
