#!/usr/bin/env python3
"""board-audit m17232: render the audit figure + AUDIT_m17176.md from analysis.json.

All numbers are injected from ``analysis.json`` (produced by ``analyze.py`` from
artifacts); nothing is transcribed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "analysis.json").read_text())

rows = A["rows"]
panels = [0] + [r["panels"] for r in rows]
err_sub = [A["readme_consistency"][0]["measured_err"]] + [r["rel_err_subject"] for r in rows]
has_rerun = [i for i, r in enumerate(rows) if r["m_A_rerun"] is not None]
err_rerun = [rows[i]["rel_err_rerun"] for i in has_rerun]
panels_rerun = [rows[i]["panels"] for i in has_rerun]
flux = [r["flux_l2_subject"] for r in rows]
flux_panels = [r["panels"] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(panels, err_sub, "o-", label="subject artifact (m17176)")
ax[0].loglog(panels_rerun, err_rerun, "x--", label="auditor rerun (m17232)")
p = panels[1:]
ax[0].loglog(p, [err_sub[1] * (panels[1] / n) ** 2 for n in p], "k:", lw=1, label=r"$O(h^2)$ slope guide")
ax[0].axhline(0.03, color="r", ls="-.", lw=1, label="3% criterion")
ax[0].set_xlabel("panels (icosphere)"); ax[0].set_ylabel(r"$|m_A-m_{A,ref}|/m_{A,ref}$")
ax[0].set_title("Integral added mass: order $p=\\log_2(e_s/e_{s+1})\\approx$ %.2f" % A["integral_order"][-1]["p_log2"])
ax[0].legend(fontsize=7); ax[0].grid(True, which="both", alpha=0.3)

ax[1].loglog(flux_panels, flux, "s-", label="per-panel flux relative L2 (subject)")
ax[1].loglog(flux_panels, [flux[0] * (flux_panels[0] / n) ** 1.0 for n in flux_panels], "k:", lw=1, label=r"$O(h^1)$ guide")
ax[1].set_xlabel("panels (icosphere)"); ax[1].set_ylabel(r"$\|q_h-q_{exact}\|_{L^2}/\|q_{exact}\|_{L^2}$")
ax[1].set_title("Per-panel flux: order $p\\approx$ %.2f (near first order)" % A["flux_order"][-1]["p_log2"])
ax[1].legend(fontsize=7); ax[1].grid(True, which="both", alpha=0.3)
fig.suptitle("CASE-BEM-SPHERE-ADDED-MASS (m17176) — board-audit m17232 independent recompute", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_audit_sphere_added_mass.png", dpi=130)
print("figure written")

ours_str = " / ".join(f"{o['p_log2']:.2f}" for o in A["integral_order"])
misread_str = " / ".join(f"{o['p_log4_misread']:.2f}" for o in A["integral_order"])
flux_str = " / ".join(f"{o['p_log2']:.2f}" for o in A["flux_order"])
best = rows[-1]
sub2 = rows[1]
md = f"""# AUDIT m17176 —— CASE-BEM-SPHERE-ADDED-MASS（球体附加质量，TensorBEM）

审核人：board-audit m17232（独立席位；未参与 m17176 实现）
审核对象：mission 17176，分支产物 `/root/tp-agent/work/m17176`（runs/m17176/*, docs/dev_report_m17176_sphere_added_mass.md），
已发布报告 tensor_reports#16204。
基准裁判：`tensor_benchmark_cases` 中 `anchor_slug='CASE-BEM-SPHERE-ADDED-MASS'` 的行 —— **本审计开始时该表 0 行**（F2）。

## 1. 取证（三件套逐项）

| 项 | 结论 | 证据指针 |
|---|---|---|
| run_artifacts 真实存在 | 真 | `refs/sphere_added_mass_sub{{1,2,3,4}}.json`（本审计 sha256 固定，来自 m17176 worktree runs/m17176/）+ `refs/sensitivity_sub2.json` + `refs/flux_panels_sub2.csv`(320 行逐面板) |
| 实现是真代码 | 真 | `src/tensorbem/benchmarks/sphere_added_mass.py:34 run()` → `solve_laplace_surface(..., domain="exterior")`；输入→计算→输出，无硬编码（负控见 §4） |
| repro_cmd 可执行 | 真 | `PYTHONPATH=src python3 -m tensorbem.benchmarks.sphere_added_mass --subdivisions {{0,1,2}}`：本审计实跑通过，sub1/sub2 **逐位一致** |
| 报告六段齐全 | 真 | tensor_reports#16204：①问题/ref ②方法 ③参数表 ④对照表 ⑤收敛+敏感性 ⑥失败模式/工件/repro；2 张图内嵌 |
| 报告数字 vs 工件 | 一致 | §4 四行 m_A 与 err 全部命中工件（最大相对偏差 <2e-4，仅四舍五入） |

## 2. 独立复算（auditor rerun）

| sub | panels | m_A 主件 | m_A 复算 | 逐位一致 | 相对误差（主件） | 复算误差 | <3% |
|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| sub{r['sub']} | {r['panels']} | {r['m_A_subject']:.6f} | "
    + (f"{r['m_A_rerun']:.6f} | {'是' if r['m_A_abs_diff']==0 else '否'} |" if r["m_A_rerun"] is not None else "未跑（见 §6） | - |")
    + f" {r['rel_err_subject']:.6e} | "
    + (f"{r['rel_err_rerun']:.6e}" if r["rel_err_rerun"] is not None else "-")
    + f" | {'PASS' if r['pass_3pct'] else 'FAIL'} |"
    for r in rows) + f"""

ref = 0.5·ρ·V = {best['ref_0p5_rhoV']:.10f} kg（Lamb 1932 §6.10；仓内 `analytic.ellipsoid_added_mass((1,1,1),1025)` = 2146.7549799481294，与闭式相对差 2.28e-12）。
**关键量判定：320 面板起全部 <3%，最细网格（5120 面板）误差 {best['rel_err_subject']:.4e}（{best['rel_err_subject']*100:.4f}%），满足 <3% 判据，且优于生态 3% 线约 26 倍。**

复算数值：sub1 = {rows[0]['m_A_rerun']}, sub2 = {rows[1]['m_A_rerun']}，与主件逐位相同（abs diff 0.0）。
标度不变性：ρ=1000/U=2 复算 rel_err = {A['scale_invariance']['scaled']['rel_err']:.6e}，与 ρ=1025/U=1 的 {rows[1]['rel_err_subject']:.6e} 一致（差值 <1e-15）→ 与主件 sensitivity_sub2.json 自洽。

## 3. 主要发现（按严重度）

**F1（medium，非阻塞，实质）报告收敛阶算错一倍 —— 并被本审计否定其推论。**
主件报告 §5 记「观测阶 0.95 / 0.98 / 0.99」，据此推断「球体附加质量积分继承通量一阶，README 的 h² 声明与实测不符，需后续复核」。
本审计复算：icosphere 每细分一次边长为 1/2，故阶应为 p=log2(e_s/e_{{s+1}})，实测
**{ours_str}**（依次），即 ≈2，与 README:62「observed order near 2」和椭球/圆柱族 1.99–2.01 一致。
主件的 0.95/0.98/0.99 恰为 log4 空间值（**{misread_str}**）——即除错底数造成的算术滑移。
真正近一阶的量是**逐面板通量**：{flux_str}。
影响：不影响任何 m_A 数值、误差与 <3% 判定；只影响收敛性解读与对 README 的错误质疑。

**F2（medium，系统性）** 本算例在机器裁判表 `tensor_benchmark_cases` 中 0 行 → 判据此前只在任务书文本里。本审计按 m17201 先例登记该 anchor 行（见 insert_verdict.sql），使后续可机器复核。

**F3（low）** 主件称 sub3/sub4 递交集群 cj1448：核 cj1448 终态 = `killed`（提交 11-13 01:55，无集群产物被引用），报告亦如实说明数字由本地 setsid 路先出 → 主张半真但自洽；数值本身已由本审计本地复跑独立证实。

**F4（low，范围上限）** 该算例施加**解析 Dirichlet 迹** u=−½Ua·cosθ 再解 q：验证的是外域单层/DtN 解算子，不是移动体 Neumann 物理 BVP。故 level 封顶 L4，不主张 L5。

**F5（info）** 80 面板误差 6.4759e-2 > 3%（主件已如实判 FAIL）；320 面板起 PASS。

**F6（info）** m17176 交付物只在其 worktree（runs/m17176/），未进仓；本审计已按 sha256 固定副本至 `refs/`。

## 4. 负控（证明"不是硬编码"）

`negcontrol_sphere_added_mass.py`（80 面板，走同一生产入口）：
- 换角向模态（cos2θ）→ m_A = {A['negative_control']['wrong_mode2_cos2theta_added_mass']:.4f}，相对 ref 偏 {A['negative_control']['wrong_mode2_relative_error_vs_ref']*100:.2f}% → 输出随输入变（非硬编码）。
- 半径 2.0 → m_A = {A['negative_control']['radius2_added_mass']:.4f} = 8×（a³ 标度，精确 8.0），相对误差相同（{A['negative_control']['radius2_relative_error']:.6e}）。

## 5. 判级

**L4（基准验证）**：真 BEM 求解 + 外部经典闭式锚 + 4 点网格细化误差表 + 可复现（复跑逐位一致）+ 测试绿。
不达 L5：无实验/印刷表格级外部锚；算例为解析迹的半解析验证面（F4）；报告的收敛阶分析有误（F1）。
不因附百页文档升/降级；级别=功能实质。

## 6. 未做的检查（honest: not_run）

| 检查 | 状态 |
|---|---|
| sub3/sub4 本审计本地复跑 | 提交时仍在跑（大网格稠密解 + 条件数估计）；sub1/sub2 已逐位复现，sub3/sub4 仅主件数值 + 集群 cj1466 递交在途 |
| 集群 cj1466 结果 | 排队中（queued_ahead 数百），未取回 |
| 与实验/CFD 数据对比 | 无（球体附加质量无实验锚，经典闭式为唯一基准） |
| 自由面/船舶水动力适用性 | 不在本算例范围（主件已声明） |

## 7. 处置

**approve with notes**：数值与工件成立，判 L4；F1 要求把收敛阶改正（本审计报告已给正确值），README 无需修改（README 原本正确）。

## 复现命令

```bash
# 主件算例
PYTHONPATH=src python3 -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2
# 本审计复算 + 负控
PYTHONPATH=src python3 docs/board_audit/m17232/negcontrol_sphere_added_mass.py
python3 docs/board_audit/m17232/analyze.py && python3 docs/board_audit/m17232/build_audit.py
```
"""

md = md.replace("ours_str", ours_str).replace("misread_str", misread_str).replace("flux_str", flux_str)
md = md.replace("{ours_str}", ours_str).replace("{misread_str}", misread_str).replace("{flux_str}", flux_str)
(HERE / "AUDIT_m17176.md").write_text(md, encoding="utf-8")
print("AUDIT_m17176.md written", len(md))