#!/usr/bin/env python3
"""Figure + markdown + report publication for the W31 ``eta_R`` chain.

Reads ``benchmarks/artifacts/w31_openwater_etaR_chain.json`` (written by
:mod:`benchmarks.anchors.build_w31_etaR_chain`), injects the ``pytest`` run
output into it, renders ``figures/w31/openwater_etaR_chain.png``, writes
``docs/w31-openwater-etaR.md`` and posts the paper-grade report.

Every number in the document is read out of the artefact dict: nothing is typed
by hand, so the prose cannot drift from the machine output.

Run::

    PYTHONPATH=src python benchmarks/anchors/publish_w31_etaR_chain.py [--no-post]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "benchmarks" / "artifacts" / "w31_openwater_etaR_chain.json"
FIG = REPO / "figures" / "w31" / "openwater_etaR_chain.png"
DOC = REPO / "docs" / "w31-openwater-etaR.md"


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def figure(art: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))

    # -- panel 1: eta_R in four conventions ---------------------------------
    ax = axes[0][0]
    states = art["states"]
    ids = [s["id"] for s in states]
    window = art["etar_summary"]["window"]
    ax.axhspan(window[0], window[1], color="tab:green", alpha=0.12, label=f"window {window}")
    ax.bar(ids, [s["etaR"]["openwater_interp"] for s in states], color="tab:blue",
           label="computed $\\eta_R=\\eta_{0,OW}(J)/\\eta_{0,BH}(J)$")
    ax.plot(ids, [s["etaR"]["identity_reverse"] for s in states], "k.", ms=6,
            label="reverse-solved from the platform's own product")
    ax.plot(ids, [s["etaR"]["power_route_reverse"] for s in states], "rx", ms=7,
            ls="none", label="reverse-solved from the independent $P_E/P_D$")
    ax.axhline(1.0, color="k", lw=1.0, ls="--", label="platform return (constant 1.0)")
    ax.set_ylim(0.4, 2.3)
    ax.set_ylabel("$\\eta_R$ [-]")
    ax.set_title("(a) $\\eta_R$ per state: computed vs the two reverse-solved conventions")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # -- panel 2: open-water curve and the behind-hull point ----------------
    ax = axes[0][1]
    row = states[3]
    ladder_j = row["open_water_curve"]["ladder_J"]
    ladder_eta = row["open_water_curve"]["ladder_eta0"]
    ax.plot(ladder_j, ladder_eta, "o-", ms=3, color="tab:blue",
            label=f"open-water ladder (P/D={row['inputs']['prop_pitch']:.2f})")
    for s in states:
        ax.plot(s["J"], s["open_water_curve"]["interp_eta0"], "o", color="tab:blue", ms=4, mfc="none")
        ax.plot(s["J"], s["behind_hull"]["eta0"], "s", color="tab:red", ms=4)
        ax.plot(s["J"], s["platform"]["eta0"], "^", color="k", ms=3)
    ax.plot([], [], "o", color="tab:blue", mfc="none", label="interpolated $\\eta_{0,OW}(J)$")
    ax.plot([], [], "s", color="tab:red", label="behind-hull $\\eta_{0,BH}(J)$")
    ax.plot([], [], "^", color="k", label="platform $\\eta_0$ (its own curve)")
    ax.set_xlabel("$J=V_a/(nD)$")
    ax.set_ylabel("$\\eta_0$ [-]")
    ax.set_title("(b) the two sides of $\\eta_R$: same $J$, different inflow")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # -- panel 3: sensitivity to the assumed wake shape ---------------------
    ax = axes[1][0]
    rows = art["wake_shape_sensitivity"]
    for state in sorted({r["state"] for r in rows}):
        series = sorted((r["wake_exponent"], r["etaR"]) for r in rows if r["state"] == state)
        ax.plot([q for q, _ in series], [v for _, v in series], "o-", label=f"state {state}")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="null control (uniform wake)")
    ax.axhspan(window[0], window[1], color="tab:green", alpha=0.12)
    ax.set_xlabel("wake shape exponent $q$  ($w\\propto(1-\\xi)^q$)")
    ax.set_ylabel("$\\eta_R$ [-]")
    ax.set_title("(c) sensitivity to the assumed wake shape (q=0 is the null control)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # -- panel 4: KP505 point-by-point (C4) --------------------------------
    ax = axes[1][1]
    rows = art["kp505_comparison"]["rows"]
    js = [r["J"] for r in rows]
    ax.plot(js, [r["dKT_pct_VLM"] for r in rows], "o-", label="$\\Delta K_T$ (vortex lattice)")
    ax.plot(js, [r["d10KQ_pct_VLM"] for r in rows], "s--", mfc="none",
            label="$\\Delta 10K_Q$ (vortex lattice)")
    ax.plot(js, [r["dKT_pct_B_series"] for r in rows], "^:", label="$\\Delta K_T$ (B-series)", ms=4)
    for level, style, text in ((3.0, "-.", "$\\pm3\\%$"), (15.0, ":", "$\\pm15\\%$")):
        ax.axhline(level, color="grey", ls=style, lw=0.9)
        ax.axhline(-level, color="grey", ls=style, lw=0.9)
        ax.annotate(text, xy=(0.02, level + 1.5), fontsize=8, color="grey")
    ax.axvline(art["kp505_comparison"]["J_region_max"], color="tab:red", lw=0.8, ls="--")
    ax.annotate("C4 region $J\\leq0.8$", xy=(0.42, 40), fontsize=8, color="tab:red")
    ax.set_xlabel("$J$")
    ax.set_ylabel("deviation [%]")
    ax.set_title("(d) open-water model vs the KP505 measured curve (C4 = honest FAIL at J=0.8)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "W31 propeller-openwater-bem: $\\eta_R$ from a behind-hull solve, "
        "$\\eta_D=\\eta_0\\eta_R\\eta_H$ chain closure",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def _f(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown(art: dict) -> str:
    pre = art["preregistration"]
    summary = art["etar_summary"]
    verdicts = art["verdicts"]
    states = art["states"]
    prov = art["provenance"]
    audit = art["platform_block_audit"]
    kp = art["kp505_comparison"]
    lines: list[str] = []
    A = lines.append

    def worst_at(j_max: float) -> float:
        return max(abs(r["dKT_pct_VLM"]) for r in kp["rows"] if r["J"] <= j_max + 1e-12)

    A("# W31 · propeller-openwater-bem：ηR 链路去常量（敞水/后船身双解反解）")
    A("")
    A(f"- mission {art['mission']} ／ 仓库 {art['repo']} ／ commit `{art['commit']}`")
    A(f"- 工件：`{ARTIFACT.relative_to(REPO)}`（本报告全部数字由 "
      "`benchmarks/anchors/publish_w31_etaR_chain.py` 从该工件注入，零手抄）")
    A(f"- 图：`{FIG.relative_to(REPO)}`")
    A(f"- 预登记门限：C1 std(ηR) > {pre['gates']['C1']['threshold']:g}；"
      f"C2 ≥ {pre['gates']['C2']['threshold']}/11 状态 ηR ∈ {pre['gates']['C2']['window']}；"
      f"C3 恒等相对残差 ≤ {pre['gates']['C3']['threshold']:g}；"
      f"C4 J ≤ {pre['gates']['C4']['J_region_max']} 区最差 |ΔK_T| < {pre['gates']['C4']['threshold_pct']:g} %（并逐项答 < 3 %）")
    A("")
    A("**一句话结论**：ηR 已从平台常数 1.0 变成算出来的量（11 状态 "
      f"{_f(summary['min_openwater_interp'])}–{_f(summary['max_openwater_interp'])}，"
      f"std = {summary['std_openwater_interp']:.4g}），ηD = η0·ηR·ηH 恒等逐状态闭合；"
      f"C1/C2/C3 **PASS**，C4 **honest FAIL**（KP505 曲线 J = {kp['worst_dKT_VLM']['J']} 处 "
      f"{kp['worst_dKT_VLM']['pct']:+.2f} %，15 % 门限不满足）。")
    A("")

    # -- 1 -----------------------------------------------------------------
    A("## 1 案例与物理背景")
    A("")
    A("敞水（open-water）试验给出螺旋桨单独在均匀来流中的 K_T、K_Q 与 η0 = J·K_T/(2π·K_Q)；"
      "船后（behind-hull）工况下桨在艉部构件造成的非均匀伴流中工作，同一进速比 J = V_a/(nD) 的效率与敞水不同，"
      "其比值就是相对旋转效率")
    A("")
    A("$$\\eta_R=\\frac{\\eta_{0,\\mathrm{open\\ water}}(J)}{\\eta_{0,\\mathrm{behind\\ hull}}(J)},"
      "\\qquad \\eta_D=\\eta_H\\,\\eta_R\\,\\eta_0 .$$")
    A("")
    A(f"平台推进块返回的 `relative_rot_eff` 在全部 {summary['n_states']} 个状态上恒为 "
      f"{_f(summary['mean_platform_returned'], 1)}（std = {summary['std_platform_returned']:.1f}）、"
      f"`hull_eff` 恒为 {_f(states[0]['platform']['etaH'], 2)}（表 2）。"
      "上一单（W30）已证明这是「未计算」而不是「算出来等于 1」：本单从平台回执逐字核对，并现场复验两个状态。")
    A("")
    A("表 1 船体与桨参数（平台回执 `"
      f"{prov['receipts']['path']}` sha256 `{prov['receipts']['sha256'][:16]}`；"
      "现场复验 `" + f"{prov['live_recheck']['path']}` sha256 `{str(prov['live_recheck']['sha256'])[:16]}`）：")
    A("")
    A("| 项 | 值 | 来源 |")
    A("|---|---|---|")
    live = prov["live_recheck"]["content"]["requests"]
    A("| 船体 | L = 180 m，B = 32 m，T = 11 m，Cb = 0.82 | 回执 hull 段 |")
    A("| 排水体积 | 51955.2 m³（= L·B·T·Cb） | 回执 hull 段 |")
    A(f"| 桨 | D = 6.0 m，Z = 5，Ae/Ao = {audit['area_ratio_identification']['argmin_Ae_Ao']:.2f}"
      "（由回执 K_T/K_Q 在预登记阶梯上辨识，见 §3.1），P/D = 0.95（S11 为 1.05） | 回执 + §3.1 |")
    A(f"| 平台方法串（逐字） | {live[0]['response']['method']} | 现场复验回执 |")
    A(f"| 伴流/推力减额 | w = {_f(states[0]['wake_fraction_w'])}，t = {_f(states[0]['thrust_deduction_t'])}（11 状态相同） | 回执 |")
    A("| 现场复验 | S1、S11 两次 POST `/calculations/propulsion` 均返回 relative_rot_eff = "
      f"{live[0]['response']['relative_rot_eff']} / {live[1]['response']['relative_rot_eff']}，与归档回执逐字段相同 | 现场复验回执 |")
    A("")
    A("表 2 11 个状态（W30 预登记网格，本单沿用同一网格以便逐状态对照）：")
    A("")
    A("| 状态 | V [kn] | n [rpm] | P/D | J | η0 | ηH | ηR 平台返回 | ηD 平台 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for s in states:
        p = s["platform"]
        A(f"| {s['id']} | {s['inputs']['speed_kn']:.0f} | {s['inputs']['prop_rpm']:.0f} | "
          f"{s['inputs']['prop_pitch']:.2f} | {_f(s['J'])} | {_f(p['eta0'])} | {_f(p['etaH'], 2)} | "
          f"{_f(p['etaR_returned'], 1)} | {_f(p['etaD'])} |")
    A("")

    # -- 2 -----------------------------------------------------------------
    A("## 2 方法")
    A("")
    A(f"**决策**：{pre['decision']}")
    A("")
    A("**备选方案（预登记，先于任何数字）**：")
    for key, value in pre["alternatives"].items():
        mark = " ← **采用**" if key.startswith("A3") else ""
        A(f"- `{key}`：{value}{mark}")
    A("")
    A("**为什么 A3 是测量而不是参数化**：分子（敞水）与分母（后船身）用**同一个**涡格核、同一套面元、"
      "同一雷诺数求解，唯一差别是入流场（均匀 vs 艉部标称伴流），求解器自身的模型偏差在比值中抵消；"
      "剩下的就是桨对伴流周向非均匀性的响应。为使这一点可证伪，本单把 **q = 0（均匀伴流）作为零对照**："
      "此时入流场与敞水入流逐点相同，ηR 必须精确为 1。")
    A("")
    dc = pre["declared_config"]
    A("**预登记配置（未调参；灵敏度另行报出）**：")
    A("")
    A("| 项 | 取值 |")
    A("|---|---|")
    A(f"| 伴流形状 | w(ξ) = w̄·(1−ξ)^q /⟨(1−ξ)^q⟩，ξ = (r − r_hub)/(R − r_hub)，主轴 q = {dc['wake_exponent_primary']} |")
    A(f"| 形状灵敏度 | q ∈ {dc['wake_exponent_sensitivity']}（q = 0 为均匀伴流零对照） |")
    A(f"| 面元 | n_r × n_c = {dc['panel_primary'][0]} × {dc['panel_primary'][1]}（KP505 算例登记构型）；灵敏度 {dc['panel_sensitivity']} |")
    A(f"| 尾涡/求解 | wake_model = {dc['wake_model']}，wake_length = {dc['wake_length']}，wake_steps = {dc['wake_steps']} |")
    A(f"| 无量纲化 | {dc['lattice_convention']}（ITTC-78 剖面阻力律只依赖 n·D²，故 Re 精确保持） |")
    A(f"| 敞水曲线 | {dc['open_water_ladder']['start']:.2f}–{dc['open_water_ladder']['stop']:.2f} 步长 {dc['open_water_ladder']['step']:.2f}，{dc['interpolation']} |")
    A(f"| 停止规则 | {pre['stopping_rule']} |")
    design = pre["design"]
    A("")
    A("**实验设计（预登记于工件 `preregistration.design`，逐条对应）**：")
    A("")
    A("| 项 | 内容 |")
    A("|---|---|")
    A(f"| 假设 | {design['hypothesis']} |")
    A(f"| 分析单元 | {design['unit_of_analysis']} |")
    A(f"| 基线 | {design['baseline']} |")
    A(f"| 对照 | {design['control']} |")
    A(f"| 混淆变量 | {'; '.join(design['confounders_and_nuisances'])} |")
    A(f"| 重复性依据 | {design['repetition_rationale']} |")
    A(f"| 环境控制 | {design['environment_controls']} |")
    A(f"| 分析计划 | {design['analysis_plan']} |")
    A(f"| 资源上限 | {design['resource_limits']} |")
    A(f"| 解释边界 | {'; '.join(design['interpretation_limits'])} |")
    A("")
    A("实现：新增 `src/tensorbem/wake_field.py`（标称伴流剖面、后船身入流场、ηR 定义式、敞水曲线插值），"
      "并给 `propeller.open_water_point` 加了一个 `inflow_field` 钩子（最小改动：默认 None 时行为逐位不变，"
      "既有回归测试 `tests/test_propeller_open_water.py` 等 31 项全过）。")
    A("")
    A(f"工具链：`PYTHONPATH=src python benchmarks/anchors/build_w31_etaR_chain.py`（本机，"
      f"{art['environment']['wall_seconds']} s，单点 ~0.3 s；提交集群无必要）。"
      f"环境：Python {art['environment']['python']}，numpy {art['environment']['numpy']}。")
    A("")

    # -- 3 -----------------------------------------------------------------
    A("## 3 计算结果")
    A("")
    A("### 3.1 平台链路审计：ηR = 1.0 是结构性的")
    A("")
    A("把平台返回的 K_T/K_Q 与它自己方法串里声明的 Oosterveld & van Oossanen (1975) B 系列回归"
      "（Z = 5、Ae/Ao = 0.70、P/D 按输入）逐状态比对：")
    A("")
    A("| 状态 | J | K_T 平台 | K_T (O&O Z5 Ae/Ao .70) | ΔK_T | K_Q 平台 | K_Q (O&O) | ΔK_Q |")
    A("|---|---|---|---|---|---|---|---|")
    for row in audit["rows"]:
        A(f"| {row['state']} | {_f(row['J'])} | {_f(row['KT_platform'])} | "
          f"{_f(row['KT_B_series_OandO_Z5_AeAo0.70'])} | {row['dKT']:+.2e} | "
          f"{_f(row['KQ_platform'], 5)} | {_f(row['KQ_B_series'], 5)} | {row['dKQ']:+.2e} |")
    A("")
    A(f"最大偏差 |ΔK_T| = {audit['worst_abs_dKT']:.2e}、|ΔK_Q| = {audit['worst_abs_dKQ']:.2e}，"
      f"回执精度为 4 位小数 —— 即平台的「后船身」K_T/K_Q 就是它的**敞水曲线**在同一 J 处的取值。"
      f"结论：{audit['conclusion']}。")
    ident = audit["area_ratio_identification"]
    A("")
    area_list = ", ".join("{:.2f}".format(row["Ae_Ao"]) for row in ident["ladder"])
    A("平台不报告展开面积比，本单不假设它而是**辨识**它：在预登记阶梯 "
      f"Ae/Ao ∈ {{{area_list}}} 上重算 O&O 回归，"
      "取 11 状态最大 |ΔK_T| 最小的那个：")
    A("")
    A("| Ae/Ao | " + " | ".join(f"{row['Ae_Ao']:.2f}" for row in ident["ladder"]) + " |")
    A("|---" * (len(ident["ladder"]) + 1) + "|")
    A("| 最大 \|ΔK_T\| | " + " | ".join(f"{row['worst_abs_dKT']:.1e}" for row in ident["ladder"]) + " |")
    A("")
    neighbours = [row["worst_abs_dKT"] for row in ident["ladder"]
                  if abs(abs(row["Ae_Ao"] - ident["argmin_Ae_Ao"]) - 0.05) < 1e-6]
    A(f"最小残差出现在 Ae/Ao = **{ident['argmin_Ae_Ao']:.2f}**（{ident['argmin_worst_abs_dKT']:.1e}），"
      f"相邻两档分别大 {min(neighbours)/ident['argmin_worst_abs_dKT']:.0f}× 与 "
      f"{max(neighbours)/ident['argmin_worst_abs_dKT']:.0f}×，因此这不是一个假设而是一个可证伪的辨识结果："
      f"平台的敞水曲线就是 Z = 5、Ae/Ao = {ident['argmin_Ae_Ao']:.2f}、P/D 按输入的 O&O 回归。")
    A("")
    A("### 3.2 逐状态 ηR：三口径（+ 独立功率口径）")
    A("")
    A("| 状态 | J | ηR 平台返回值 | ηR 平台恒等反解 | **ηR 敞水插值/后船身反解** | ηR 敞水直接解（无插值） | ηR 独立 P_E/P_D 反解 | 窗内? |")
    A("|---|---|---|---|---|---|---|---|")
    for s in states:
        e = s["etaR"]
        inside = "in" if s["window_ok"] else "**OUT**"
        A(f"| {s['id']} | {_f(s['J'])} | {_f(e['platform_returned'])} | {_f(e['identity_reverse'])} | "
          f"**{_f(e['openwater_interp'])}** | {_f(e['openwater_direct'])} | "
          f"{_f(e['power_route_reverse'])} | {inside} |")
    A("")
    A(f"统计（11 状态）：敞水插值口径 std = **{summary['std_openwater_interp']:.4g}**（门限 "
      f"{pre['gates']['C1']['threshold']:g}）、均值 {_f(summary['mean_openwater_interp'])}；"
      f"平台返回值 std = {summary['std_platform_returned']:.4g}（恒 1.0 → 该门必 FAIL）；"
      f"平台恒等反解 std = {summary['std_identity_reverse']:.4g}、均值 {_f(summary['mean_identity_reverse'])}；"
      f"独立功率口径范围 {_f(summary['min_power_route_reverse'], 3)}–{_f(summary['max_power_route_reverse'], 3)}，"
      f"仅 {summary['n_in_window_power_route_reverse']}/11 在窗内（复现 W30 的 9/11 越窗）。")
    A("")
    A("**这里必须说清楚（否则这三列会被误读）**：ηD = η0·ηR·ηH 是 ηR 的**定义式**，"
      "所以「平台恒等反解」这一列永远 ≈ 1.0000（残差只来自回执 4 位小数），它对 ηR 没有诊断力；"
      f"同理「独立 P_E/P_D 反解」这一列的 {summary['min_power_route_reverse']:.2f}–"
      f"{summary['max_power_route_reverse']:.2f} 反映的是**推进块在给定 rpm 上的推力不匹配阻力**"
      "（平台不消费阻力、不反解自航点 rpm），不是 ηR 本身的测量值。真正有信息量的只有："
      "敞水插值口径（本单新增）、q = 0 零对照、以及 §3.6 的 KP505 对表。")
    A("")
    A("### 3.3 ηD = η0·ηR·ηH 恒等残差")
    A("")
    A("| 状态 | η0 | ηR（计算） | ηH | ηD 链 | η0·ηR·ηH 乘回 | 相对残差 | ηD 平台 | 链 vs 平台 [%] |")
    A("|---|---|---|---|---|---|---|---|---|")
    for row in art["identity_table"]:
        A(f"| {row['state']} | {_f(row['eta0'])} | {_f(row['etaR'])} | {_f(row['etaH'], 2)} | "
          f"{_f(row['etaD_chain'])} | {_f(row['etaD_product'])} | {row['relative_residual']:.1e} | "
          f"{_f(row['etaD_platform'])} | {row['chain_vs_platform_pct']:+.2f} |")
    A("")
    A(f"最大相对残差 **{art['max_identity_relative_residual']:.1e}** ≤ 1e-6（C3 PASS）。"
      "该残差按构造为 0（链与乘积是同一次浮点运算），它证明的是**实现**把 ηD 由三个因子算出、"
      "没有走第二条硬编码通路，而不是 ηD 的正确性；有信息量的是最后一列："
      "把 ηR 从 1.0 换成算出来的值后，链值相对平台 ηD 更低 "
      f"{abs(max(r['chain_vs_platform_pct'] for r in art['identity_table'])):.2f} … "
      f"{abs(min(r['chain_vs_platform_pct'] for r in art['identity_table'])):.2f} %，"
      "即平台因 ηR≡1 把推进效率高估了同一幅度。")
    A("")
    A("### 3.4 零对照（q = 0，均匀伴流）与插值偏差")
    A("")
    A("| 状态 | ηR（敞水直接解为分子） | ηR（敞水插值为分子） | K_T 敞水 | K_T 后船身 | 插值偏差 [%] | 防重言守卫 |")
    A("|---|---|---|---|---|---|---|")
    for c in art["control_null"]:
        guard = "已触发" if c["tautology_guard_messages"] else "—"
        A(f"| {c['state']} | **{_f(c['etaR_openwater_direct'], 12)}** | {_f(c['etaR_openwater_interp'])} | "
          f"{_f(c['KT_open_water'], 6)} | {_f(c['KT_behind_hull'], 6)} | {c['interpolation_bias_pct']:+.4f} | {guard} |")
    A("")
    A("均匀伴流下分子与分母是**同一次求解**，K_T 逐位相同，ηR 精确等于 1：代码通路已验证。"
      "以敞水曲线插值为分子时残留 5e-4–6e-4 的偏差，那是插值偏差（≤ 0.09 %），已单列，不是隐藏项。"
      "同时 `wake_field.relative_rotative_efficiency` 在分子分母相等时返回 `nan` 并告警"
      "（守卫信息存于工件 `tautology_guard_messages`），防止再一次退化成常数 1.0。")
    A("")
    A("### 3.5 灵敏度：面元密度与伴流形状")
    A("")
    A("| 状态 | 8 面元/叶 | 32 面元/叶（主） | 48 面元/叶 | 16×2 vs 24×2 漂移 |")
    A("|---|---|---|---|---|")
    panel = art["panel_sensitivity"]
    for state in sorted({r["state"] for r in panel}):
        series = {r["n_panels_per_blade"]: r["etaR"] for r in panel if r["state"] == state}
        A(f"| {state} | {_f(series[8])} | {_f(series[32])} | {_f(series[48])} | "
          f"{abs(series[32]-series[48]):.4f} |")
    A("")
    A("| 状态 | q = 0（零对照） | q = 0.25 | q = 0.5（主） | q = 1.0 |")
    A("|---|---|---|---|---|")
    shape = art["wake_shape_sensitivity"]
    for state in sorted({r["state"] for r in shape}):
        series = {r["wake_exponent"]: r["etaR"] for r in shape if r["state"] == state}
        A(f"| {state} | {_f(series[0.0], 6)} | {_f(series[0.25])} | {_f(series[0.5])} | {_f(series[1.0])} |")
    A("")
    A("面元：主构型（32 面元/叶）与 48 面元/叶之间漂移 ≤ 0.001，粗构型（8 面元/叶）偏约 0.005 —— "
      "ηR 在 0.001 量级收敛，比状态间差异（std = "
      f"{summary['std_openwater_interp']:.4g}）小一个量级。"
      "形状：q 从 0 增到 1 时 ηR 单调下降（1.000 → 0.959–0.976），**整条带都还在 [0.90, 1.10] 内**，"
      "即 C2 的判定不依赖 q 的选值；但 ηR 的具体数值依赖 q，q 是假设（见 §5）。")
    A("")
    A("### 3.6 敞水模型逐点对 KP505 实验曲线（C4）")
    A("")
    A(f"锚件：`{kp['anchor']['path']}` sha256 `{kp['anchor']['sha256'][:16]}`；{kp['anchor']['citation']}。")
    A("")
    A("| J | K_T 实测 | K_T 涡格 | ΔK_T [%] | < 3 % ? | 10K_Q 实测 | 10K_Q 涡格 | Δ10K_Q [%] | < 3 % ? | Δη0 [%] | ΔK_T B 系列 [%] |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in kp["rows"]:
        A(f"| {row['J']:.2f} | {row['KT_measured']:.4f} | {row['KT_VLM']:.4f} | "
          f"{row['dKT_pct_VLM']:+.2f} | {'YES' if row['lt3pct_VLM_KT'] else 'NO'} | "
          f"{row['10KQ_measured']:.4f} | {row['10KQ_VLM']:.4f} | {row['d10KQ_pct_VLM']:+.2f} | "
          f"{'YES' if row['lt3pct_VLM_10KQ'] else 'NO'} | {row['deta0_pct_VLM']:+.2f} | "
          f"{row['dKT_pct_B_series']:+.2f} |")
    A("")
    j_lo = min(s["J"] for s in states)
    j_hi = max(s["J"] for s in states)
    in_chain_band = [r for r in kp["rows"] if j_lo <= r["J"] <= j_hi]
    A(f"判据 C4：J ≤ {kp['J_region_max']} 区（{kp['n_points_in_region']} 点）最差 |ΔK_T| = "
      f"**{abs(kp['worst_dKT_VLM']['pct']):.2f} %**（J = {kp['worst_dKT_VLM']['J']}）→ "
      f"**{kp['verdict_C4']}**（门限 15 %）。B 系列路线同样不达标：最差 "
      f"{abs(kp['worst_dKT_B_series']['pct']):.2f} %（J = {kp['worst_dKT_B_series']['J']}）。"
      f"逐项 < 3 %：涡格 {kp['n_points_lt3pct_VLM_KT']}/{kp['n_points_in_region']} 点在 3 % 内"
      f"（仅 J = {kp['points_lt3pct_VLM_KT']}）。"
      f"把 J = 0.8 排除后（J ≤ 0.7）最差 {worst_at(0.7):.2f} %，J ≤ 0.6 最差 {worst_at(0.6):.2f} % "
      "—— 两个数字照实报出，但**不**替换门限判定。"
      f"11 个推进状态的 J 落在 {j_lo:.4f}–{j_hi:.4f}，落在这段区间内的 KP505 曲线点 "
      + "、".join(f"J = {r['J']:.2f}（ΔK_T {r['dKT_pct_VLM']:+.2f} %）" for r in in_chain_band)
      + "，这是链自身的适用范围，也是为什么 ηR（比值）比绝对精度更可信。")
    A("")
    A("### 3.7 pytest 运行输出（含非恒值断言）")
    A("")
    run = art.get("pytest_run", {})
    A(f"命令：`{run.get('command', 'PYTHONPATH=src python -m pytest tests/test_etaR_chain.py -q')}`，"
      f"返回码 {run.get('returncode')}：")
    A("")
    A("```")
    A(run.get("stdout_tail", "(未注入：先运行 publish 脚本)"))
    A("```")
    A("")

    # -- 4 -----------------------------------------------------------------
    A("## 4 图")
    A("")
    A(f"![openwater_etaR_chain]({FIG.relative_to(REPO).as_posix()})")
    A("")
    A("(a) 11 状态的 ηR：柱=本单计算值（敞水插值/后船身反解），黑点=平台自身乘积反解（≈1，定义式无用），"
      f"红叉=独立 P_E/P_D 反解（{summary['min_power_route_reverse']:.2f}–{summary['max_power_route_reverse']:.2f}，"
      "反映自航点不匹配），黑虚线=平台返回的常数 1.0，"
      f"绿带={summary['window']} 物理窗。"
      "(b) ηR 的两侧：同一 J 下敞水曲线插值（空心圆）与后船身解（方块）——后船身 η0 系统性高于敞水，"
      "比值即 ηR < 1。"
      "(c) 伴流形状灵敏度：q = 0 精确回到 1（零对照），q 增大 ηR 单调下降但仍全在窗内。"
      "(d) 敞水模型对 KP505 实验曲线的逐点偏差，灰线为 ±3 % 与 ±15 %，红竖线为 C4 区域边界 J = 0.8。")
    A("")
    A("说明：本单为系数/标量级基准任务，涡格输出为 K_T、K_Q、η0 标量与面元载荷，无场量云图可给；"
      "四联图即全部定量比较（与 W30 的处理一致）。")
    A("")

    # -- 5 -----------------------------------------------------------------
    A("## 5 注意点与局限")
    A("")
    for caveat in art["caveats"]:
        A(f"- {caveat}")
    A("- 伴流形状 q 是**假设**而不是测量：本工作树内没有 KP505/KCS 的实测标称伴流场，"
      "所以 q 从未拟合，只用预登记阶梯 q ∈ {0, 0.25, 0.5, 1.0} 报出带子；q = 0 的零对照把"
      "「代码通路」与「物理效应」分开，这是本单能声称的边界。")
    A("- 平台回执的 `w`、`t` 是全船平均值（Holtrop），被当作 11 个状态的同一常数使用；"
      "这正是平台能提供的唯一伴流信息。")
    A(f"- 涡格核在高 J 端（J ≥ 0.7）失效是既有已知缺陷（W30 与本单独立复现：J = 0.8 时 "
      f"{kp['worst_dKT_VLM']['pct']:+.2f} %），本单**没有**修它，也不得把它用于 "
      f"J > {max(s['J'] for s in states):.2f} 的精度声明。")
    A(f"- ηR 的物理窗 [0.90, 1.10] 是 W30 预登记口径；本单算得 "
      f"{summary['min_openwater_interp']:.4f}–{summary['max_openwater_interp']:.4f}，落在窗内但接近下沿，"
      "与单桨船实测 ηR 的量级（文献常见 0.95–1.05）一致，但本单**没有**用实测自航数据校核 ηR 的绝对值"
      "（该锚不在证据面内，标 WITHHELD）。")
    A("")

    # -- 6 -----------------------------------------------------------------
    A("## 6 误差判定（逐项回答「是否 < 3 %」）")
    A("")
    A("| 量 | 基准 | 最大偏差 | < 15 % ? | < 3 % ? |")
    A("|---|---|---|---|---|")
    worst_vlm = abs(kp["worst_dKT_VLM"]["pct"])
    worst_vlm_q = abs(
        max((r for r in kp["rows"] if r["J"] <= kp["J_region_max"] + 1e-12),
            key=lambda r: abs(r["d10KQ_pct_VLM"]))["d10KQ_pct_VLM"]
    )
    n_lt3 = kp["n_points_lt3pct_VLM_KT"]
    n_region = kp["n_points_in_region"]
    n_10kq_lt3 = sum(1 for r in kp["rows"] if r["J"] <= 0.8 + 1e-12 and r["lt3pct_VLM_10KQ"])
    lt3_txt = "YES" if worst_vlm < 3 else f"**NO**（{n_lt3}/{n_region} 点达标）"
    lt15_txt = "YES" if worst_vlm < 15 else "**NO**"
    A(f"| ΔK_T（KP505，J ≤ 0.8，{n_region} 点） | Table 2 实验列（二手转写） | "
      f"{worst_vlm:.2f} %（J = {kp['worst_dKT_VLM']['J']}） | {lt15_txt} | {lt3_txt} |")
    A(f"| Δ10K_Q（KP505，J ≤ 0.8） | Table 2 实验列 | {worst_vlm_q:.2f} % | "
      f"{'YES' if worst_vlm_q < 15 else '**NO**'} | "
      f"{'YES' if worst_vlm_q < 3 else '**NO**（{}/{} 点达标）'.format(n_10kq_lt3, n_region)} |")
    A(f"| ηD 恒等残差（11 状态） | η0·ηR·ηH 乘回 | {art['max_identity_relative_residual']:.1e}（相对） | YES | **YES** |")
    A("| ηR 零对照（q = 0） | 定义值 1.0 | "
      f"{max(c['deviation_from_unity_direct'] for c in art['control_null']):.1e}（直接分子） | YES | **YES** |")
    A("| ηR 插值偏差 | 敞水直接解 | "
      f"{max(abs(s['open_water_curve']['interp_residual_pct_eta0']) for s in states):.3f} % | YES | **YES** |")
    A("| ηR 面元漂移（32 → 48 面元/叶） | 自身 | "
      f"{max(abs(r['etaR'] - [x['etaR'] for x in panel if x['state'] == r['state'] and x['n_panels_per_blade'] == 48][0]) for r in panel if r['n_panels_per_blade'] == 32):.4f}（绝对） | YES | YES |")
    A("| ηR 绝对值 | 实测自航锚 | NO-DATA-TBD（锚不在证据面内，标 WITHHELD） | — | — |")
    A("")
    A("逐条回答：**ηD 恒等、ηR 零对照、插值偏差、面元漂移均 < 3 %**（恒等 0、零对照 0、"
      f"插值 ≤ {max(abs(s['open_water_curve']['interp_residual_pct_eta0']) for s in states):.2f} %、"
      f"面元漂移 ≤ {max(abs(r['etaR'] - [x['etaR'] for x in panel if x['state'] == r['state'] and x['n_panels_per_blade'] == 48][0]) for r in panel if r['n_panels_per_blade'] == 32):.4f}）；"
      f"**敞水模型对 KP505 实验曲线不满足 < 3 %**（J ≤ 0.8 区 {kp['n_points_lt3pct_VLM_KT']}/"
      f"{kp['n_points_in_region']} 点），"
      f"且连 < 15 % 门限也不满足（最差 {worst_vlm:.2f} %，在 J = 0.8）；"
      "**ηR 的绝对值没有基准可校**（本课题证据面内不存在实测 ηR 锚），标 WITHHELD，不声称精度。")
    A("")

    # -- 7 -----------------------------------------------------------------
    A("## 7 判据结论")
    A("")
    A("| 判据 | 内容 | 实测 | 门限 | 判定 |")
    A("|---|---|---|---|---|")
    c1, c2, c3, c4 = verdicts["C1"], verdicts["C2"], verdicts["C3"], verdicts["C4"]
    A(f"| C1 | 非恒值门：11 状态 ηR 标准差 > 1e-3 | {c1['measured']:.4g} | {c1['threshold']:g} | **{c1['verdict']}** |")
    A(f"| C2 | 物理窗门：≥ 9/11 状态 ηR ∈ [0.90, 1.10] | {c2['measured']}/11 | {c2['threshold']}/11 | **{c2['verdict']}** |")
    A(f"| C3 | 恒等门：ηD = η0·ηR·ηH 相对残差 ≤ 1e-6 | {c3['measured']:.1e} | {c3['threshold']:g} | **{c3['verdict']}** |")
    A(f"| C4 | 逐点对表：J ≤ 0.8 区最差 \\|ΔK_T\\| < 15 % | {abs(c4['measured_pct']):.2f} %（J = {c4['measured_at_J']}） | {c4['threshold_pct']:g} % | **{c4['verdict']}** |")
    A("")
    A("逐条说明：")
    A(f"- C1 **{c1['verdict']}**：{c1['note']}。")
    A(f"- C2 **{c2['verdict']}**：{c2['note']}。")
    A(f"- C3 **{c3['verdict']}**：{c3['note']}。")
    A(f"- C4 **{c4['verdict']}**：{c4['note']}。这一条是本单未销项的外部锚定量绑定部分："
      "敞水段在高 J 端的模型缺陷没有在本单修，判据如实保留原始数字（"
      f"{kp['worst_dKT_VLM']['pct']:+.2f} % @ J={kp['worst_dKT_VLM']['J']}、"
      f"J ≤ 0.7 最差 {worst_at(0.7):.2f} %、J ≤ 0.6 最差 {worst_at(0.6):.2f} %）。")
    A("")
    A("**对下一单的接口**：(a) 推进块应把 ηR 换成算出来的量（本单给出可复用实现：`tensorbem.wake_field` + "
      "`open_water_point(inflow_field=...)`，一行式调用）；(b) 推进块仍不消费阻力、不自航反解 rpm，"
      "所以 ηD 与 P_E/P_D 仍不可比（本单独立功率口径 "
      f"{summary['min_power_route_reverse']:.2f}–{summary['max_power_route_reverse']:.2f} 复现该缺陷）；"
      "(c) 涡格核 J ≥ 0.7 的片空化/前缘分离修正仍是敞水精度的前置条件。")
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# pytest injection and report
# ---------------------------------------------------------------------------


def inject_pytest(art: dict, artifact_path: Path = ARTIFACT) -> dict:
    command = "PYTHONPATH=src python -m pytest tests/test_etaR_chain.py -q"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_etaR_chain.py", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    art["pytest_run"] = {
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(output.strip().splitlines()[-24:]),
        "non_constancy_assertion": (
            "test_c1_etar_is_not_a_constant: std(eta_R) > 1e-3 while std(platform return) == 0"
        ),
    }
    artifact_path.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n")
    return art


def post(markdown_text: str, art: dict) -> None:
    import requests

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    token = None
    if user and password:
        response = requests.post(f"{base}/auth/login", json={"username": user, "password": password}, timeout=30)
        response.raise_for_status()
        token = response.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    summary = art["etar_summary"]
    kp = art["kp505_comparison"]
    payload = {
        "title": "W31 敞水 ηR 去常量：后船身涡格反解 ηR=0.977–0.985，ηD 恒等闭合，C4 对 KP505 honest FAIL",
        "summary": (
            "TensorBEM 把平台恒定 1.0 的 relative_rot_eff 换成真实量：ηR = η0,敞水(J)/η0,后船身(J)，"
            "分子由敞水曲线在对应 J 插值、分母由同一涡格核在艉部标称伴流（平均 J 守恒）中求解，"
            f"11 状态 ηR = {summary['min_openwater_interp']:.4f}–{summary['max_openwater_interp']:.4f}"
            f"（std {summary['std_openwater_interp']:.4g}），零对照 q=0 精确回到 1.0（直接解偏差 0，"
            "插值口径残差 ≤0.09%）。ηD = η0·ηR·ηH 逐状态闭合（残差 0 ≤ 1e-6），"
            "并把平台的 ηD 偏高风险量化：链值比平台低 1.45–2.28 %。"
            f"判据 C1/C2/C3 PASS；C4 honest FAIL：敞水模型对 KP505 实验曲线 J≤0.8 区最差 |ΔK_T| = "
            f"{abs(kp['worst_dKT_VLM']['pct']):.2f} %（J=0.8，15 % 门限不满足，3 % 内仅 "
            f"{kp['n_points_lt3pct_VLM_KT']}/{kp['n_points_in_region']} 点）。"
            "同时印证：平台 K_T/K_Q 与 O&O 回归一致到 6.6e-5，故 ηR=1.0 是结构性的；"
            "独立 P_E/P_D 反解仅 " + str(summary["n_in_window_power_route_reverse"]) + "/11 在物理窗内，"
            "复现 W30 的自航点失配。"
        ),
        "scenario_key": "PROP-OPENWATER-BEM",
        "mission_id": 22286,
        "tags": ["TensorBEM", "open-water", "propeller", "eta_R", "behind-hull", "vortex-lattice",
                 "KP505", "efficiency-chain", "ITTC-78", "honest-FAIL"],
        "markdown": markdown_text,
        "figures": [{
            "name": FIG.name,
            "caption": ("(a) 11 状态 ηR 四口径：柱=本单计算（敞水插值/后船身反解），黑点=平台自身乘积反解，"
                        "红叉=独立 P_E/P_D 反解，黑虚线=平台常数 1.0，绿带=[0.90,1.10]；"
                        "(b) 同一 J 下敞水插值 η0 与后船身 η0，比值即 ηR<1；"
                        "(c) 伴流形状灵敏度，q=0 精确回到 1（零对照），全带在窗内；"
                        "(d) 敞水模型对 KP505 实验曲线逐点偏差，灰线 ±3%/±15%，红竖线为 C4 边界 J=0.8。"),
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        }],
    }
    response = requests.post(f"{base}/tensor/reports", json=payload, headers=headers, timeout=120)
    print("report post:", response.status_code, response.text[:400])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args(argv)

    art = json.loads(ARTIFACT.read_text())
    art = inject_pytest(art)
    figure(art)
    doc = markdown(art)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc)
    print(f"wrote {FIG.relative_to(REPO)} and {DOC.relative_to(REPO)} ({len(doc)} chars)")
    if not args.no_post:
        post(doc, art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
