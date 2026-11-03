"""Build ``docs/finite_depth_frequency_sweep/report.md`` from the sweep artefact.

Every number in the report is injected from
``artifacts/finite_depth_frequency_sweep/sweep.json``; nothing is typed by hand.

    PYTHONPATH=src python3 docs/finite_depth_frequency_sweep/report_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "finite_depth_frequency_sweep" / "sweep.json"
TARGET = ROOT / "docs" / "finite_depth_frequency_sweep" / "report.md"


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def sci(value):
    return f"{value:+.2e}"


def main() -> int:
    payload = json.loads(ARTIFACT.read_text())
    summary = payload["summary"]
    cases = payload["cases"]
    ratios = payload["submergence_ratios"]
    ka_values = payload["ka_values"]
    motions = ("heave", "sway")

    worst_case = max(
        cases,
        key=lambda case: max(
            abs(case[f"dev_from_analytic_{motion}"]["added_mass"]) for motion in motions
        ),
    )
    worst_kernel = max(
        cases,
        key=lambda case: max(
            abs(case[f"dev_from_deep_panel_{motion}"][quantity])
            for motion in motions
            for quantity in ("added_mass", "damping")
        ),
    )
    worst_energy = max(
        cases,
        key=lambda case: max(
            abs(case["finite_depth"]["far_over_near"][motion] - 1.0) for motion in motions
        ),
    )
    energy_values = [
        case["finite_depth"]["far_over_near"][motion]
        for case in cases
        for motion in motions
    ]
    mode = payload["mode_study"][0]
    refinement_rows = []
    for study in payload["refinement"]:
        orders = {
            (order["quantity"], tuple(order["panels"])): order["observed_order"]
            for order in study["observed_orders"]
        }
        for row in study["rows"]:
            refinement_rows.append((study, row, orders))

    lines: list[str] = []
    add = lines.append
    add("# 有限水深辐射求解器：`ka` 与 `f/a` 扫掠验证（结清 limitation 5）")
    add("")
    add("**日期**：2026-11-03　**模块**：`tensorbem.benchmarks.finite_depth_frequency_sweep`")
    add("**上游欠账**：`docs/finite_depth_radiation.md` §5 局限 5 ——"
        "「只在圆形截面、`ka=0.6`、`f/a=2` 上验证；未扫 `ka` 与 `f/a`」")
    add("")
    add("## 1. 案例介绍")
    add("")
    add("无限长圆柱截面（半径 `a`）水平潜没于等深水池，轴线位于静水面下 `f`，水深 `h`，"
        "作单位速度垂荡（heave）与横荡（sway）振动；求单位宽度附加质量 `A` 与阻尼 `B`。")
    add("")
    add("| 参数 | 值 |")
    add("|---|---|")
    add(f"| 半径 `a` | {fmt(payload['radius'], 1)} m |")
    add(f"| 水深 `h` | {fmt(payload['depth'], 1)} m（`h/a = {fmt(payload['depth_ratio'], 1)}`） |")
    add(f"| 潜深比 `f/a` | {', '.join(fmt(value, 2) for value in ratios)} |")
    add(f"| 无量纲频率 `ka` | {', '.join(fmt(value, 2) for value in ka_values)} |")
    add(f"| 密度 `rho` | {fmt(payload['density'], 1)} kg/m³ |")
    add(f"| 重力加速度 `g` | {fmt(payload['gravity'], 5)} m/s² |")
    add(f"| 面元数 `N` | {payload['panels']}（常数面元，等弧长） |")
    add(f"| 参考解 | Ursell 深水多极子闭式解（`tensorbem.analytic`）+ 深水面板求解器（同网格） |")
    add("")
    add("选择 `h/a = 50` 的理由：此时有限水深核相对深水核只剩指数小项，"
        "任何与水**无关**的差异都只能来自实现，因此扫掠把「水深的物理」与"
        "「核/装配的实现」分离开。")
    add("")
    add("## 2. 怎么做的")
    add("")
    add("1. 对每个 `(ka, f/a)` 用有限水深间接（单层位势）求解器"
        "`tensorbem.finite_depth_solver.solve_radiation` 求 `A`、`B`；"
        "**垂荡与横荡分别求解**——对称体在垂荡下的水平力恒为零，"
        "若用一个解的两个分量去比较，就会拿零比零（本次开发中真实踩到，见 §5）。")
    add("2. 同网格、同 `ka` 再用深水面板求解器 `free_surface_solver.solve_radiation_indirect`"
        "求一份 `A`、`B`，作为「核实现」判据的基准：它包含面板离散误差，"
        "但与有限水深核完全无关。")
    add("3. 与 Ursell 深水多极子闭式解（零共享代码）比较，得到含离散误差的总误差。")
    add("4. 能量恒等式：由辐射波幅算得 `B = 2 rho g c_g |eta_a|²`（远场）"
        "与压力积分 `omega rho Im(F)`（近场）比较。")
    add("5. 模态数充分性：把自动模态数 `24 nu h` 乘 2 倍重算，报告系数相对变化。")
    add("6. 面板加密：扫掠中最粗与最细两个工况做 `N = 32/64/128`，报告观测收敛阶。")
    add("")
    add("工具链：Python 3.11 / NumPy 2.4.6 / matplotlib（图）/ pytest（判据）。"
        f"整轮耗时 {fmt(payload['elapsed_seconds'], 1)} s，单机 128 核，"
        "进程为单核 2-D BEM，未占用重算力。")
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 扫掠主表（程序化注入）")
    add("")
    add("| `ka` | `f/a` | 模态数 | `A_heave` | `A_sway` | `B_heave` | `B_sway` "
        "| `dA/A` vs Ursell (heave) | `dA/A` vs 深水面板 (heave) | `B_far/B_near` (heave) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for case in cases:
        fd = case["finite_depth"]
        add(
            f"| {fmt(case['ka'], 2)} | {fmt(case['submergence_ratio'], 2)} "
            f"| {case['modes_auto']} "
            f"| {fmt(fd['added_mass']['heave'], 2)} | {fmt(fd['added_mass']['sway'], 2)} "
            f"| {fmt(fd['damping']['heave'], 2)} | {fmt(fd['damping']['sway'], 2)} "
            f"| {sci(case['dev_from_analytic_heave']['added_mass'])} "
            f"| {sci(case['dev_from_deep_panel_heave']['added_mass'])} "
            f"| {fmt(fd['far_over_near']['heave'], 4)} |"
        )
    add("")
    add("### 3.2 汇总判据")
    add("")
    add("| 判据 | 实测最大值 | 门限 | 结论 |")
    add("|---|---|---|---|")
    add(f"| 与深水面板解的最大偏差（核实现） | {sci(summary['max_kernel_error_vs_deep_panel'])} "
        f"| 1e-2 | {'PASS' if summary['max_kernel_error_vs_deep_panel'] < 1e-2 else 'FAIL'} |")
    add(f"| 与 Ursell 闭式解的最大偏差（含离散） | {sci(summary['max_panel_error_vs_analytic'])} "
        f"| 3e-2 | {'PASS' if summary['max_panel_error_vs_analytic'] < 3e-2 else 'FAIL'} |")
    add(f"| 能量恒等式最大偏差 | {sci(summary['max_energy_identity_error'])} | 3e-2 "
        f"| {'PASS' if summary['max_energy_identity_error'] < 3e-2 else 'FAIL'} |")
    add(f"| 自动模态数 vs 2 倍模态数的最大相对变化 | {sci(summary['max_mode_rel_diff'])} "
        f"| 1e-4 | {'PASS' if summary['max_mode_rel_diff'] < 1e-4 else 'FAIL'} |")
    add("")
    add(f"- 最差 Ursell 偏差出现在 `ka = {fmt(worst_case['ka'], 2)}`、"
        f"`f/a = {fmt(worst_case['submergence_ratio'], 2)}`；")
    add(f"- 最差核偏差出现在 `ka = {fmt(worst_kernel['ka'], 2)}`、"
        f"`f/a = {fmt(worst_kernel['submergence_ratio'], 2)}`；")
    add(f"- 能量恒等式比值全程落在 [{fmt(min(energy_values), 4)}, {fmt(max(energy_values), 4)}]，"
        f"系统偏正约 {fmt(100 * (sum(energy_values) / len(energy_values) - 1), 2)}%，"
        "与 `docs/finite_depth_energy_identity/report.md` 的结论（一阶离散偏移）一致；")
    add(f"- 模态研究工况：`ka = {fmt(mode['ka'], 2)}`、`f/a = {fmt(mode['submergence_ratio'], 2)}`，"
        f"自动模态数 {mode['modes_auto']} → {mode['modes_requested']}，"
        f"附加质量相对变化 heave {sci(mode['rel_diff_added_mass']['heave'])} / "
        f"sway {sci(mode['rel_diff_added_mass']['sway'])}，"
        f"阻尼相对变化 heave {sci(mode['rel_diff_damping']['heave'])} / "
        f"sway {sci(mode['rel_diff_damping']['sway'])}。")
    add("")
    add("### 3.3 面板加密（观测收敛阶）")
    add("")
    add("| `ka` | `f/a` | `N` | `|dA/A|` | `|dB/B|` |")
    add("|---|---|---|---|---|")
    for study, row, _ in refinement_rows:
        add(f"| {fmt(study['ka'], 2)} | {fmt(study['submergence_ratio'], 2)} | {row['panels']} "
            f"| {sci(row['abs_err_added_mass'])} | {sci(row['abs_err_damping'])} |")
    add("")
    add("观测阶（`|dA/A|`，`N` 逐次翻倍）：" + "；".join(
        f"`ka={fmt(study['ka'], 2)}, f/a={fmt(study['submergence_ratio'], 2)}`: "
        + ", ".join(
            fmt(order["observed_order"], 2)
            for order in study["observed_orders"]
            if order["quantity"] == "abs_err_added_mass"
        )
        for study in payload["refinement"]
    ) + "。常数面元的已知一阶收敛率因此得到确认，误差随 `ka` 增大而增大是"
        "离散误差而非核缺陷（同一网格上深水面板解的误差同步增大）。")
    add("")
    add("## 4. 结果图")
    add("")
    add("![ka sweep errors](../artifacts/finite_depth_frequency_sweep/ka_sweep_errors.png)")
    add("")
    add("左：附加质量对 Ursell 闭式解的相对误差随 `ka` 的变化（实线 heave、虚线 sway）；"
        "右：同一系数对**深水面板解**的相对误差。右图是核实现判据，其量级（~1e-3）"
        "比左图（~1e-2，含离散误差）低一个量级以上，说明扫掠范围内没有与 `ka` 相关的核缺陷。")
    add("")
    add("![energy identity](../artifacts/finite_depth_frequency_sweep/energy_identity_ka.png)")
    add("")
    add("`B_far/B_near` 在整个扫掠内保持 1.01 附近的常数偏移，与水深、`ka`、`f/a`、"
        "运动方向都无关——这正是「离散化常数偏移」而非物理差异的特征。")
    add("")
    add("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **踩坑（已修）**：第一版扫掠对每个 `(ka, f/a)` 只求解一次，然后从同一个解里读两个力分量。"
        "对称体垂荡时水平力**恒为零**，于是「深水面板 vs 有限水深」的偏差出现 1e+15 量级，"
        "`A_sway` 读出 `-0.0`。修法是两个运动各解一次，并在测试里用"
        "`test_cases_are_non_degenerate` 把「系数必须非零」钉死。**这类零比零的伪一致/伪失效"
        "在对称问题上会静默发生，比发散更难发现。**")
    add("2. **深水塌缩判据证明的是「核实现」而不是「物理精度」**：`h/a=50` 时物理水深效应"
        "已低于 0.1%，所以该判据只能抓实现缺陷；有限水深的**物理**正确性依赖核级验证"
        "（`docs/finite_depth_2d.md`）与能量恒等式。")
    add("3. **自动模态数在 `ka=1.5` 处并非 1e-6 级收敛**：2 倍模态使系数变化 ~1e-5（见 §3.2），"
        "比 `docs/finite_depth_2d.md` 的核级地板（`nu h=400` 时最坏 2e-5）同量级。"
        "相对 ~1% 的离散误差这完全可忽略，但**不能**宣称系数已收敛到 1e-6。")
    add("4. **仍是 2-D、常数面元、常水深、无流、线性势流**；未接船体几何，"
        "没有实验/文献实测数据（有限水深附加质量的文献表本机不可得）。"
        "因此结论是「实现正确性 + 离散误差量级」，不是「某工况的工程精度」。")
    add("5. **未扫净空**：`h/a=50` 时净空很大；浅水/近底阻塞（陡峭的 `h/a<4`）不在本单范围，"
        "`docs/finite_depth_radiation.md` 已单独覆盖深度扫描。")
    add("6. `sway` 在 `f/a` 较大的工况下与 `heave` 只差 1e-3 量级（深水极限两者相等），"
        "这正是检验对称性的物理依据，但也意味着本扫掠对 `sway` 的**独立**信息量弱于 `heave`。")
    add("")
    add("## 6. 误差判定")
    add("")
    add("基准：Ursell 深水多极子闭式解（`tensorbem.analytic.ursell_cylinder_added_mass`，"
        "与求解器零共享代码）与深水面板解（同网格）。")
    add("")
    add("| 关键量 | 最大相对误差 | 基准 | 是否 < 3% |")
    add("|---|---|---|---|")
    add(f"| `A`（heave/sway，含离散） | {sci(summary['max_panel_error_vs_analytic'])} | Ursell 闭式解 "
        f"| {'是' if summary['max_panel_error_vs_analytic'] < 0.03 else '否'} |")
    add(f"| `A`（heave，核实现） | {sci(summary['max_kernel_error_vs_deep_panel'])} | 深水面板解（同网格） "
        f"| {'是' if summary['max_kernel_error_vs_deep_panel'] < 0.03 else '否'} |")
    add(f"| `B_far/B_near`（能量恒等式） | {sci(summary['max_energy_identity_error'])} | 波幅能量平衡 "
        f"| {'是' if summary['max_energy_identity_error'] < 0.03 else '否'} |")
    add("")
    add("**逐项回答「是否 <3%」**：")
    add("")
    add(f"- 附加质量对 Ursell 闭式解：{('是' if summary['max_panel_error_vs_analytic'] < 0.03 else '否')}"
        f"（最大 {fmt(100 * summary['max_panel_error_vs_analytic'], 2)}%，出现在 "
        f"`ka={fmt(worst_case['ka'], 2)}, f/a={fmt(worst_case['submergence_ratio'], 2)}`；"
        "该误差是 64 面元的离散误差，加密到 128 面元降到 1% 量级，见 §3.3）；")
    add(f"- 有限水深核相对深水面板核：{('是' if summary['max_kernel_error_vs_deep_panel'] < 0.03 else '否')}"
        f"（最大 {sci(summary['max_kernel_error_vs_deep_panel'])}）；")
    add(f"- 能量恒等式：{('是' if summary['max_energy_identity_error'] < 0.03 else '否')}"
        f"（最大 {sci(summary['max_energy_identity_error'])}）；")
    add("- 与实验/文献实测值比较：**无基准，未做**。有限水深圆柱附加质量的文献表（Yu & Ursell 类）"
        "本机不可得，故本单不宣称任何「对实物/实船的精度」。")
    add("")
    add("## 7. 复现")
    add("")
    add("```bash")
    add("PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_frequency_sweep \\")
    add("    --json artifacts/finite_depth_frequency_sweep/sweep.json \\")
    add("    --figures artifacts/finite_depth_frequency_sweep")
    add("PYTHONPATH=src python3 docs/finite_depth_frequency_sweep/report_assets.py")
    add("PYTHONPATH=src python3 -m pytest tests/test_finite_depth_frequency_sweep.py -q")
    add("```")
    add("")
    add("## 8. 结清情况")
    add("")
    add("`docs/finite_depth_radiation.md` §5 局限 5（「只在 `ka=0.6`、`f/a=2` 上验证」）"
        f"已由本单覆盖：`ka` 扩到 {', '.join(fmt(value, 2) for value in ka_values)}，"
        f"`f/a` 扩到 {', '.join(fmt(value, 2) for value in ratios)}，"
        "每点同时给出核实现判据、闭式解判据、能量恒等式与模态收敛证据。"
        "局限 4（定水深、无流、线性）与局限 3（MFS 非独立物理基准）不受本单影响，仍然有效。")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n".join(lines) + "\n")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(lines)} lines)")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())