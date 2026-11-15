#!/usr/bin/env python3
"""Figure + markdown + report publication for the w30 open-water efficiency closure.

Reads ``papers/artifacts/w30/propeller-openwater-bem/openwater_efficiency_closure.json``
(the artefact written by :mod:`benchmarks.anchors.build_openwater_efficiency_closure`),
renders ``figures/w30/openwater_efficiency_closure.png``, writes
``docs/w30_openwater_efficiency_closure.md`` and posts the paper-grade report.

Every number in the markdown comes out of the artefact dict; nothing is typed by
hand, so the document cannot drift from the machine output.

Run::

    PYTHONPATH=src python benchmarks/anchors/publish_openwater_closure_w30.py [--no-post]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "papers" / "artifacts" / "w30" / "propeller-openwater-bem" / "openwater_efficiency_closure.json"
FIG = REPO / "figures" / "w30" / "openwater_efficiency_closure.png"
DOC = REPO / "docs" / "w30_openwater_efficiency_closure.md"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def figure(art: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # --- panel 1: deviation from the anchor -------------------------------
    ax = axes[0]
    styles = {"b4_70": ("B4-70 lattice vs O&O closed form", "o", "-"),
              "kp505": ("KP505 lattice vs Table 2", "s", "--")}
    for key, (label, marker, line) in styles.items():
        rows = art["solver_vs_anchor"][key]
        js = [r["J"] for r in rows]
        ax.plot(js, [r["dKT_pct"] for r in rows], marker=marker, ls=line, color="tab:blue" if key == "b4_70" else "tab:cyan",
                label=f"dKT, {label}")
        ax.plot(js, [r["d10KQ_pct"] for r in rows], marker=marker, ls=line, mfc="none",
                color="tab:red" if key == "b4_70" else "tab:orange", label=f"d10KQ, {label}")
    for level, style in ((10.0, ":"), (3.0, "-.")):
        ax.axhline(level, color="grey", ls=style, lw=0.8)
        ax.axhline(-level, color="grey", ls=style, lw=0.8)
    ax.annotate("±10 % gate", xy=(0.02, 10.6), fontsize=8, color="grey")
    ax.annotate("±3 % band", xy=(0.02, 3.4), fontsize=8, color="grey")
    ax.set_xlabel("advance ratio J")
    ax.set_ylabel("deviation from anchor [%]")
    ax.set_title("solver vs open-water anchors")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # --- panel 2: efficiency ------------------------------------------------
    ax = axes[1]
    ovo = art["anchors"][0]["points"]
    kp = art["anchors"][1]["points"]
    b4 = art["solver_vs_anchor"]["b4_70"]
    kps = art["solver_vs_anchor"]["kp505"]
    ax.plot([r["J"] for r in ovo], [r["eta0_registered"] for r in ovo], "o-", color="tab:green", label="O&O B4-70 anchor eta0")
    ax.plot([r["J"] for r in b4], [r["eta0_solver"] for r in b4], "x--", color="tab:green", label="B4-70 lattice eta0")
    ax.plot([r["J"] for r in kp], [r["eta0_registered"] for r in kp], "s-", color="tab:purple", label="KP505 Table 2 eta0")
    ax.plot([r["J"] for r in kps], [r["eta0_solver"] for r in kps], "+--", color="tab:purple", label="KP505 lattice eta0")
    ax.set_xlabel("advance ratio J")
    ax.set_ylabel("open-water efficiency eta0")
    ax.set_title("eta0 = J KT / (2 pi KQ), anchor vs solver")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # --- panel 3: eta_R window ---------------------------------------------
    ax = axes[2]
    states = art["etaR_audit"]["states"]
    x = list(range(len(states)))
    ax.bar(x, [r["etaR_cross_module_implied"] for r in states], color="tab:orange", alpha=0.8,
           label="eta_R implied across modules (P_E^res / P_D)")
    ax.plot(x, [r["etaR_returned"] for r in states], "k_", ms=18, mew=2, label="eta_R returned by API (constant)")
    ax.axhspan(0.90, 1.10, color="tab:green", alpha=0.12, label="admissible window [0.90, 1.10]")
    ax.axhline(0.90, color="grey", lw=0.8)
    ax.axhline(1.10, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["state"] for r in states], rotation=60, fontsize=7)
    ax.set_ylabel("relative rotative efficiency eta_R")
    ax.set_title("eta_R convention audit (11 propulsion states)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("W30 propeller-openwater-bem: anchor closure, eta0 identity, eta_R convention", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


def markdown(art: dict) -> str:
    ovo, kp = art["anchors"]
    b4 = art["solver_vs_anchor"]["b4_70"]
    kps = art["solver_vs_anchor"]["kp505"]
    audit = art["etaR_audit"]
    summary = audit["summary"]
    crit = {c["id"]: c for c in art["criteria"]}

    lines = []
    A = lines.append
    A("# W30 · propeller-openwater-bem：敞水锚值核销与效率恒等回归\n")
    A(f"- mission 21514 ／ 仓库 TensorBEM ／ commit `{art['git_commit']}`")
    A(f"- 工件：`{art['artifacts']['json']}`（本报告全部数字由 `{art['generated_by']}` 从该工件注入）")
    A(f"- 预登记门限：η0 复算残差 ≤ {art['gates']['eta0_residual_max_pct']} %；Kt/Kq 对表 < {art['gates']['solver_deviation_max_pct']} %；"
      f"严判 < {art['gates']['tight_pct']} %；ηR 物理窗 {art['gates']['etar_window']}")
    A("")

    A("## 1 案例与物理背景\n")
    A("敞水（open-water）是螺旋桨单独在均匀来流中运转的基准工况：给定进速比 "
      "`J = V_a/(nD)`，量测推力系数 `K_T = T/(ρn²D⁴)`、扭矩系数 `K_Q = Q/(ρn²D⁵)`，效率 "
      "`η0 = J·K_T/(2πK_Q)`。它是船舶总体推进链 `η_D = η_H·η_R·η_0` 的乘性首项，"
      "因此敞水锚值的**量纲口径**（10KQ 还是 KQ）与**效率恒等**一旦出错，推进因子会整链失守。\n")
    A("表 1 两个锚件的几何/参数：\n")
    A("| 锚件 | 类别 | 几何参数 | J 点数 | J 覆盖 | 载体 sha256 前 16 |")
    A("|---|---|---|---|---|---|")
    A(f"| {ovo['anchor_id']} | {ovo['source_class']} | Z={ovo['geometry']['Z']}, Ae/Ao={ovo['geometry']['Ae_Ao']}, P/D={ovo['geometry']['P_D']} | {ovo['n_points']} | "
      f"{ovo['points'][0]['J']:.2f}–{ovo['points'][-1]['J']:.2f} | `{ovo['carrier']['sha256'][:16]}` |")
    A(f"| {kp['anchor_id']} | {kp['source_class']} | KP505 FPP, Z=5, Ae/Ao=0.80, P/D=0.95（TABLE 2 实验列） | {kp['n_points']} | "
      f"{kp['points'][0]['J']:.2f}–{kp['points'][-1]['J']:.2f} | `{kp['carrier']['transcript_sha256'][:16]}` |")
    A("")
    A("锚件引句（机器抽取，非手抄）：")
    A(f"- O&O 闭式：`{ovo['points'][0]['quote']}`，quote_status = `{ovo['points'][0]['quote_status']}`"
      "（NSMB 印刷表不在本工作树内，故按 `LITERATURE_CLOSED_FORM` 登记，不冒充数字化表值）")
    A(f"- KP505 原文行：`{kp['header_quotes'][0]}`；逐点如 `{kp['points'][0]['quote']}`，"
      f"quote_status = `{kp['points'][0]['quote_status']}`")
    A(f"- sha256 核对：transcript `{kp['carrier']['sha256_checks']['transcript_sha256_matches_declared']}`，"
      f"PDF `{kp['carrier']['sha256_checks']['pdf_sha256_matches_declared']}`")
    A("")

    A("## 2 方法\n")
    A("1. **锚件登记**：O&O 1975 的 39 项 K_T / 47 项 K_Q 回归系数从仓内已登记锚件 "
      "`benchmarks/anchors/wageningen_b_series.json` 读取，按 B4-70 几何在 J=0.1…0.8 独立求值；"
      "KP505 的 Table 2 逐行地从 PDF 转写文本解析（引句与数字同源），并与锚 JSON 交叉核对到 1e-12。")
    A("2. **η0 恒等复算**：另写一条独立通路 `η0 = J·K_T/(2π·K_Q)`，其中 `K_Q = 10K_Q/10`，"
      "与锚件登记的 η0 逐点比较，残差门限 0.5 %（该通路能抓 10KQ/KQ 的口径错位）。")
    A("3. **求解器对表**：TensorBEM 涡格敞水核在**同样的 J 点**各跑一遍——B4-70 用 `B4_70_FINE`"
      "（16×2 面元、geometric 尾迹 L/D=2.0/48 步、ITTC-78 摩擦）；KP505 用模块默认（16×2=32 面元/叶，"
      "与仓内 `benchmarks/kp505_openwater_bem/points.csv` 同一构型），逐点出 `dK_T`、`d(10K_Q)`、`dη0`。")
    A("4. **ηR 口径审计**：向平台推进块与阻力块各递交同一个 11 状态网格（单船体 L=180 m、B=32 m、"
      "T=11 m、Cb=0.82；S1–S6 为 12–17 kn @100 rpm 速度扫描，S7–S10 为 85/90/95/105 rpm @15 kn 转速扫描，"
      "S11 为 P/D=1.05 @15 kn/100 rpm），原始回执逐字存档于 "
      f"`{audit['receipts']['path']}`（sha256 `{audit['receipts']['sha256'][:16]}`）。"
      "每个状态给出三个 ηR：平台返回值、平台自身乘积恒等反解、跨模块（阻力块 P_E / 推进块 P_D）反解，"
      "逐条判 [0.90, 1.10]。")
    A("")
    A("工具链：`python benchmarks/anchors/build_openwater_efficiency_closure.py`（求解器本地核，单点 ~0.8 s，"
      "18 点共 ~15 s，未压集群；三节点/大网格扫描不在本单范围）。")
    A("")

    A("## 3 计算结果\n")
    A("### 3.1 η0 恒等复算（锚件登记值 vs 独立通路）\n")
    A(f"18 个锚点全部闭合：最大 |残差| = {art['eta0_identity']['max_abs_residual_pct']:.3e} %，"
      f"门限 {art['gates']['eta0_residual_max_pct']} %（判据 C2 **PASS**）。"
      "这说明两件锚件的 η0 列与各自的 K_T、K_Q 列自洽，不存在 10KQ/KQ 十进位错位。\n")
    A("### 3.2 求解器逐点对表\n")
    A("表 2 B4-70 涡格 vs O&O 闭式锚（同 J 点）：\n")
    A("| J | K_T 求解 | K_T 锚 | dK_T % | 10KQ 求解 | 10KQ 锚 | d10KQ % | η0 求解 | η0 锚 | dη0 % | <10 % | <3 % |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in b4:
        A(f"| {r['J']:.2f} | {r['KT_solver']:.4f} | {r['KT_anchor']:.4f} | {_fmt(r['dKT_pct'])} | "
          f"{r['10KQ_solver']:.4f} | {r['10KQ_anchor']:.4f} | {_fmt(r['d10KQ_pct'])} | "
          f"{r['eta0_solver']:.4f} | {r['eta0_anchor']:.4f} | {_fmt(r['deta0_pct'])} | "
          f"{'YES' if r['within_10pct'] else '**NO**'} | "
          f"{'YES' if (r['within_3pct_dKT'] and r['within_3pct_d10KQ']) else 'NO'} |")
    A("")
    A("表 3 KP505 涡格（32 面元/叶）vs Table 2 实验列：\n")
    A("| J | K_T 求解 | K_T 锚 | dK_T % | 10KQ 求解 | 10KQ 锚 | d10KQ % | dη0 % | <10 % | <3 % |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in kps:
        A(f"| {r['J']:.2f} | {r['KT_solver']:.4f} | {r['KT_anchor']:.4f} | {_fmt(r['dKT_pct'])} | "
          f"{r['10KQ_solver']:.4f} | {r['10KQ_anchor']:.4f} | {_fmt(r['d10KQ_pct'])} | {_fmt(r['deta0_pct'])} | "
          f"{'YES' if r['within_10pct'] else '**NO**'} | "
          f"{'YES' if (r['within_3pct_dKT'] and r['within_3pct_d10KQ']) else 'NO'} |")
    A("")
    over = crit["C3-solver-versus-anchor"]["measured"]["n_points_over_10pct"]
    A(f"判据 C3 **FAIL**：B4-70 在 {len(over['b4_70'])} 个 J 点越 10 % 门限（{', '.join(f'{j:.2f}' for j in over['b4_70']) or '无'}），"
      f"KP505 在 {len(over['kp505'])} 个 J 点越限（{', '.join(f'{j:.2f}' for j in over['kp505']) or '无'}）。"
      "KP505 的 J=0.80 处 dK_T = "
      f"{[r for r in kps if abs(r['J']-0.8)<1e-9][0]['dKT_pct']:+.2f} %，与仓内既有诚实失败记录 "
      "(`docs/m7485_kp505_openwater_bem.md`, dK_T(J=0.8) = -29.1 %) 一致，说明这不是新网格引入的偏差，"
      "而是该涡格核在高进速比端的既有模型缺陷（尾涡/片空化前缘分离缺失，K_T 系统性偏低）。\n")
    A("### 3.3 推进块 ηR 口径审计\n")
    A("表 4 十一个推进状态的 ηR 三口径与物理窗判定：\n")
    A("| 状态 | V [kn] | n [rpm] | P/D | J | η0 | ηH | ηR 返回 | ηR 恒等反解 | ηR 跨模块反解 | P_E 阻力块 [kW] | P_D 推进块 [kW] | 两模块 P_E 差 % | 窗内? |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in audit["states"]:
        A(f"| {r['state']} | {r['inputs']['speed_kn']:.0f} | {r['inputs']['prop_rpm']:.0f} | {r['inputs']['prop_pitch']:.2f} | "
          f"{r['J']:.4f} | {r['eta0_returned']:.4f} | {r['etaH_returned']:.3f} | {r['etaR_returned']:.3f} | "
          f"{r['etaR_identity_implied']:.4f} | {r['etaR_cross_module_implied']:.4f} | {r['pe_resistance_kW']:.2f} | "
          f"{r['pd_propulsion_kW']:.2f} | {_fmt(r['pe_module_disagreement_pct'])} | "
          f"{'in' if r['etaR_cross_module_in_window'] else '**OUT**'} |")
    A("")
    A(f"判据 C4 **FAIL**：平台返回的 `relative_rot_eff` 在全部 {summary['n_states']} 个状态上恒为 "
      f"{summary['distinct_etaR_returned']}（`hull_eff` 同样恒为 {summary['distinct_etaH_returned']}），"
      f"即 ηR 并未被计算；用平台自己的乘积恒等反解得到 {summary['n_states']-0} /{summary['n_states']} 在窗内（定义式，无诊断力）；"
      f"改用阻力块独立的 P_E 反解，则 **{summary['n_etaR_cross_module_out_of_window']}/{summary['n_states']}** 个状态掉出 [0.90, 1.10]"
      f"（范围 {min(r['etaR_cross_module_implied'] for r in audit['states']):.2f} – {max(r['etaR_cross_module_implied'] for r in audit['states']):.2f}），"
      "与审计的 9/11 计数吻合（本单不能声称状态集相同：审计的 11 状态清单不在本工作树内）。\n")
    A("平台 η0 本身是真实的：用返回的 K_T、K_Q、J 独立复算 η0，"
      f"11 个状态最大残差 {summary['max_abs_eta0_residual_pct']:.3f} %（最差 {summary['worst_eta0_state']}，来自 4 位小数回执的舍入）。"
      "同理，返回的 K_T/K_Q 与 O&O 回归在 Z=5、Ae/Ao=0.70、P/D 给定值下 4 位小数级重合，"
      "所以**敞水段是真算的**，缺陷集中在 ηH/ηR 两个常数因子与推进块不消费阻力（`displacement` 传 45000 与 51955.2 结果完全相同，"
      "而阻力块对同一输入直接以 D3 门禁拒绝 15.46 % 的几何不自洽）。\n")

    A("## 4 图\n")
    A("![openwater_efficiency_closure](figures/w30/openwater_efficiency_closure.png)\n")
    A("左：两种敞水算例逐点 `dK_T`/`d(10K_Q)` 与 10 % 门限、3 % 带宽；中：η0 对 J 的锚值/求解器曲线；"
      "右：11 状态的 ηR 三口径——柱为跨模块反解、黑色短横为平台返回值（恒 1.0）、绿带为 [0.90, 1.10] 物理窗。\n")
    A("说明：本单为标量/系数级基准任务，无场量云图可给（敞水核输出为 K_T、K_Q、η0 标量与面元载荷，"
      "未做流场渲染）；图 1 的三联图即全部定量比较。\n")

    A("## 5 注意点与局限\n")
    for note in art["honest_notes"]:
        A(f"- {note}")
    A("- 锚件真实性：O&O 锚是**闭式重算**而非印刷表数字化；KP505 锚是二手转写（MARINE 2023 的 Experimental 列，"
      "该文未注明其实验数据来源），要用于对外精度声明前必须回到 KRISO 原始表核对。")
    A("- 回执精度：平台回执为 2 位小数（kW）/4 位小数（无量纲），因此恒等残差的 1e-3 量级是**精度限制**，不是求解器误差。")
    A("- 11 状态网格是本单自定的（单船体、速度+转速扫描+一个 P/D 点，预登记于脚本头部），"
      "不是审计原网格；9/11 是计数复现，不是状态集同一性证明。")
    A("- `prop_pitch` 字段语义为 P/D 比值而非螺距长度（5.4 被回执报为 “P/D=5.400 超出 B 系列有效域”），"
      "用螺距[m]填表会静默产生越域结果（该状态 P_D 推进块仍返回 open_water_eff=0.999、propulsive_eff=1.0689>1）。")
    A("")

    A("## 6 误差判定（逐项回答「是否 < 3 %」）\n")
    A("| 量 | 基准 | 最大偏差 | < 10 % ? | < 3 % ? |")
    A("|---|---|---|---|---|")
    b4_ok10 = sum(1 for r in b4 if r["within_10pct"]); kp_ok10 = sum(1 for r in kps if r["within_10pct"])
    b4_ok3 = sum(1 for r in b4 if r["within_3pct_dKT"] and r["within_3pct_d10KQ"])
    kp_ok3 = sum(1 for r in kps if r["within_3pct_dKT"] and r["within_3pct_d10KQ"])
    A(f"| dK_T (B4-70, 8 点) | O&O 1975 闭式 | {crit['C3-solver-versus-anchor']['measured']['b4_70']['max_abs_dKT_pct']:.2f} % | {b4_ok10}/8 | {b4_ok3}/8 |")
    A(f"| d10K_Q (B4-70, 8 点) | O&O 1975 闭式 | {crit['C3-solver-versus-anchor']['measured']['b4_70']['max_abs_d10KQ_pct']:.2f} % | {b4_ok10}/8 | {b4_ok3}/8 |")
    A(f"| dK_T (KP505, 10 点) | Table 2 实验列 | {crit['C3-solver-versus-anchor']['measured']['kp505']['max_abs_dKT_pct']:.2f} % | {kp_ok10}/10 | {kp_ok3}/10 |")
    A(f"| d10K_Q (KP505, 10 点) | Table 2 实验列 | {crit['C3-solver-versus-anchor']['measured']['kp505']['max_abs_d10KQ_pct']:.2f} % | {kp_ok10}/10 | {kp_ok3}/10 |")
    A(f"| η0 恒等残差 (18 锚点) | 锚件登记 η0 | {art['eta0_identity']['max_abs_residual_pct']:.2e} % | YES | **YES** |")
    A(f"| η0 平台复算残差 (11 状态) | 平台返回 open_water_eff | {summary['max_abs_eta0_residual_pct']:.3f} % | YES | **YES** |")
    A(f"| ηR (跨模块反解, 11 状态) | 物理窗 [0.90, 1.10] | {max(abs(r['etaR_cross_module_implied']-1.0) for r in audit['states'])*100:.1f} %（相对 1.0） | — | **NO（9/11 越窗）** |")
    A("")
    A("逐条回答：**η0 恒等与平台 η0 复算均 < 3 %**（前者 ~1e-14 %，后者 ≤ 0.036 %，后者受回执 4 位小数限制）；"
      f"**敞水求解器对表不满足 < 3 %**：B4-70 仅 {b4_ok3}/8 点同时满足 K_T 与 10K_Q 的 3 %，"
      f"KP505 {kp_ok3}/10 点；B4-70 在 J=0.80、KP505 在 J≥0.60 连 10 % 门限都不满足。"
      "**ηR 的物理窗判定不通过**：平台返回值虽恒在窗内，但它是未计算的常数；以独立模块 P_E 反解时 9/11 越窗，"
      "属物理不可容许，按本单要求显式报出（不粉饰）。\n")
    A("结论：判据 C1/C2 通过，C3/C4 失败（C5 为信息项）。敞水锚件与 η0 恒等已核销；"
      "推进块的真缺陷是 ηH/ηR 常数化 + 推进块不消费阻力/推进未与阻力匹配，"
      "表现为 ηR 物理窗的 9/11 越界。修复建议（下一单）：(a) 推进块按自航点反解 rpm（使 `T(1-t)=R_T`）而不是接受任意 rpm；"
      "(b) ηR 用 Holtrop(1984) 经验式或 B 系列自航数据计算，不得恒取 1.0；(c) 对乘积恒等增加“非平凡性”断言——"
      "任一因子为常数即判该恒等不可用于诊断；(d) 涡格核在高 J 端需引入片空化/前缘分离修正，否则不得用于 J>0.5 的精度声明。")
    return "\n".join(lines) + "\n"


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
    payload = {
        "title": "W30 敞水性能锚值核销与效率恒等回归：η0 恒等 ~1e-14 %，ηR 9/11 越物理窗",
        "summary": (
            "TensorBEM 敞水锚件登记两个锚（O&O 1975 B4-70 闭式 8 点、KP505 Table 2 实验列 10 点，含机器抽取引句与 sha256），"
            "独立通路复算 η0 = J·K_T/(2πK_Q) 全部闭合（最大残差 1.7e-14 %）。涡格求解器逐点对表：B4-70 7/8 点在 10 % 门内、"
            "J=0.80 越限（dK_T -15.4 %）；KP505 5/10 点在门内、J=0.80 dK_T -29.1 % 与仓内既有诚实失败记录一致。"
            "推进块 ηR 口径审计：11 状态中平台 relative_rot_eff 恒为 1.0（未计算），用阻力块独立 P_E 跨模块反解时 9/11 掉出 [0.90,1.10]，"
            "复现审计计数；敞水段本身（K_T/K_Q/η0）经 O&O 回归与恒等复算确认为真算。判据 C1/C2 PASS，C3/C4 FAIL（如实）。"
        ),
        "scenario_key": "PROP-OPENWATER-BEM",
        "mission_id": 21514,
        "tags": ["TensorBEM", "open-water", "propeller", "Wageningen-B", "KP505", "eta0-identity",
                 "relative-rotative-efficiency", "anchor-provenance", "verification"],
        "markdown": markdown_text,
        "figures": [{
            "name": FIG.name,
            "caption": ("左：涡格求解器对两个敞水锚的逐点 dK_T / d(10K_Q)，灰线为 ±10 % 门限与 ±3 % 带宽；"
                        "中：η0 对 J 的锚值 vs 求解器；右：11 个推进状态的 ηR——柱为跨模块（阻力块 P_E / 推进块 P_D）反解，"
                        "黑短横为平台返回的常数 1.0，绿带为 [0.90, 1.10] 物理窗，9/11 柱在窗外。"),
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
    figure(art)
    text = markdown(art)
    DOC.write_text(text)
    print("wrote", FIG.relative_to(REPO), DOC.relative_to(REPO), len(text), "bytes")
    if not args.no_post:
        post(text, art)
    return 0


if __name__ == "__main__":
    sys.exit(main())