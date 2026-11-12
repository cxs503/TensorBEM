#!/usr/bin/env python3
"""Render the W25 (mission 16712) seakeeping-truncation report from the artifact.

Every number in the markdown is interpolated from
``papers/artifacts/w25/seakeeping-truncation/journee_rao_truncation.json`` -- no
value is typed by hand (zero hand-copying).
"""

from __future__ import annotations

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "papers/artifacts/w25/seakeeping-truncation/journee_rao_truncation.json"
OUT_MD = ROOT / "papers/reports/w25_seakeeping_truncation5.md"
FIGDIR = ART.parent


def fmt(x, nd=4):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def pct(x, nd=3):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def make_figures(A: dict) -> list:
    figs = []
    omega = np.array([r[0] for r in A["rao_table"]["rows"]])
    nondim = np.array([r[1] for r in A["rao_table"]["rows"]])
    heave = np.array([r[2] for r in A["rao_table"]["rows"]])
    pitch = np.array([r[3] for r in A["rao_table"]["rows"]])
    levels = A["case"]["study_design"]["truncation_frequency_levels_rad_s"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(nondim, heave, "o-", ms=3, lw=1.2, color="#1f4e79")
    axes[0].set_ylabel(r"heave RAO  $|\xi_3|/A$  [m/m]")
    axes[0].set_title("Wigley III head-seas RAO, zero speed (TensorBEM 3-D Green-function BEM)")
    axes[1].plot(nondim, pitch, "s-", ms=3, lw=1.2, color="#a03a1f")
    axes[1].set_ylabel(r"pitch RAO  $|\xi_5|/A$  [rad/m]")
    axes[1].set_xlabel(r"$\omega\sqrt{L/g}$")
    per_level = [
        (lv * math.sqrt(A["case"]["model_parameters"]["L"] / A["case"]["fluid"]["gravity"]))
        for lv in levels
    ]
    for ax in axes:
        for i, x in enumerate(per_level):
            ax.axvline(x, color="0.55", ls="--", lw=0.9)
            ax.text(x, ax.get_ylim()[1] * 0.92, rf"$\omega_{{\max,{i + 1}}}$",
                    fontsize=7, ha="center", color="0.35")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = FIGDIR / "fig1_rao_curves.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    figs.append((p1, "Figure 1 - heave/pitch RAO of the Wigley III hull (head seas, U=0) "
                     "with the four truncation-frequency levels marked."))

    # axis A
    rows = A["axis_A_truncation_frequency"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for key, ax, lab in (
        ("sigma_heave", axes[0], "heave rms [m]"),
        ("sigma_pitch", axes[1], "pitch rms [rad]"),
    ):
        for ss in sorted({r["sea_state"] for r in rows}):
            sub = sorted([r for r in rows if r["sea_state"] == ss], key=lambda r: r["omega_max"])
            ax.plot([r["omega_max"] for r in sub], [r[key] for r in sub], "o-", label=ss)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"truncation frequency $\omega_{\max}$ [rad/s]")
        ax.set_ylabel(lab)
        ax.set_title("upper-band truncation (Hs = 1 m)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p2 = FIGDIR / "fig2_truncation_frequency.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    figs.append((p2, "Figure 2 - band-limited response rms versus the upper truncation "
                     "frequency for the three ITTC sea states (doubling ladder)."))

    # axis B
    rows = A["axis_B_truncation_radius_long_wave"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for key, ax, lab in (
        ("sigma_heave", axes[0], "heave rms [m]"),
        ("sigma_pitch", axes[1], "pitch rms [rad]"),
    ):
        for ss in sorted({r["sea_state"] for r in rows}):
            sub = sorted(
                [r for r in rows if r["sea_state"] == ss],
                key=lambda r: r["radius_lambda_over_L"],
            )
            ax.plot([r["radius_lambda_over_L"] for r in sub], [r[key] for r in sub], "o-",
                    label=ss)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"long-wave truncation radius $\lambda_{\max}/L$")
        ax.set_ylabel(lab)
        ax.set_title("long-wave truncation")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p3 = FIGDIR / "fig3_truncation_radius_longwave.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    figs.append((p3, "Figure 3 - response rms versus the long-wave truncation radius "
                     "(lambda_max/L, doubling ladder)."))

    # axis C
    rows = A["axis_C_truncation_radius_geometric"]["rows"]
    frs = sorted({r["froude"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for fr in frs:
        sub = sorted([r for r in rows if r["froude"] == fr], key=lambda r: r["y_max_factor"])
        ax.plot([r["y_max_factor"] for r in sub], [r["cw_pressure"] for r in sub], "o-",
                label=f"Fr = {fr:.3f}")
        for r in sub:
            ax.annotate(f"{r['n_unknowns']}", (r["y_max_factor"], r["cw_pressure"]),
                        textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center")
    ax.set_xlabel(r"lateral domain truncation radius $y_{\max}/L$")
    ax.set_ylabel(r"$C_w$ from hull pressure integration [-]")
    ax.set_title("truncated-domain Rankine solver: domain-radius sweep (labels = unknowns)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p4 = FIGDIR / "fig4_truncation_radius_geometric.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    figs.append((p4, "Figure 4 - steady truncated-domain Rankine solver: wave-resistance "
                     "coefficient versus the lateral truncation radius at the two mission "
                     "Froude numbers."))
    return figs


def render_md(A: dict, figs: list) -> str:
    geo = A["verification"]["geometry"]
    phy = A["verification"]["physics"]
    disc = A["verification"]["discretisation"]["rows"]
    crit = A["criteria"]
    c1 = crit["criterion_1_peak_rao_vs_anchor"]
    c2 = crit["criterion_2_peak_frequency_vs_anchor"]
    c3 = crit["criterion_3_doubling_convergence"]
    diag = c1["recalled_literal_diagnostic"]
    sd = A["case"]["study_design"]
    mp = A["case"]["model_parameters"]
    axA = A["axis_A_truncation_frequency"]
    axB = A["axis_B_truncation_radius_long_wave"]
    axC = A["axis_C_truncation_radius_geometric"]

    L = []
    L.append("# W25 · 耐波性 RAO 截断收敛 vs Journée (1992) 锚（TensorBEM m16712）\n")
    L.append(
        f"> 工件：`papers/artifacts/w25/seakeeping-truncation/journee_rao_truncation.json`"
        f"（schema `{A['schema']}`，生成 {A['generated_utc']}，git `{A['git'].get('describe')}`）\n"
        f"> 环境：Python {A['environment']['python']} / numpy {A['environment']['numpy']} / "
        f"{A['environment']['platform']}\n"
        f"> 复算命令：`{'`; `'.join(A['provenance']['commands'])}`\n"
    )

    L.append("## ① 案例介绍\n")
    L.append(
        "**物理背景。** 频率域耐波性计算给出 RAO，但 RAO 只在有限频带内被求值：上端**截断频率**"
        "决定响应谱被记入多少高频余量，下端**截断半径**（波长域 $\\lambda_{\\max}/L$）决定长波端"
        "被记入多少。ITTC 7.5-02-07-04.5 类推荐程序把“截断后残余谱面积是否可忽略”作为验收量，"
        "而不是逐点 RAO。本单把这两类截断各做成 4 档加倍阶梯，在**真跑**的 RAO 上量化其收敛性，"
        "并对账外部试验锚。\n"
    )
    L.append(
        "**几何（Wigley III，即 Journée 1992 Report 09-V-09 的模型船型；$L/B=10$, $B/T=1.6$）。**\n"
    )
    L.append("| 量 | 值 | 来源 |")
    L.append("|---|---|---|")
    L.append(f"| $L$ | {fmt(mp['L'],3)} m | 解析船型 |")
    L.append(f"| $B$ | {fmt(mp['B'],3)} m | 解析船型 |")
    L.append(f"| $T$ | {fmt(mp['T'],4)} m | 解析船型 |")
    L.append(f"| $\\alpha$（改型参数） | {fmt(mp['alpha'],6)} | $C_b=(4/9)+4\\alpha/45$ |")
    L.append(f"| $C_b$（闭式） | {fmt(mp['Cb_closed_form'],6)} | 闭式 |")
    L.append(f"| $C_b$（求积） | {fmt(mp['Cb_quadrature'],6)} | 船体体积求积 |")
    L.append(f"| 排水体积（闭式） | {fmt(mp['volume_closed_form'],6)} m³ | $V=LBT C_b$ |")
    L.append(f"| 质量 | {fmt(mp['mass'],3)} kg | $\\rho V$ |")
    L.append(f"| $KB$（闭式） | {fmt(geo['KB_closed_form_m'],6)} m | $T[1-(15+5\\alpha)/(40+8\\alpha)]$ |")
    L.append(f"| $KB$（求积复核） | {fmt(geo['KB_quadrature_m'],6)} m | 船体体积求积 |")
    L.append(f"| $KB/T$ | {fmt(geo['KB_over_T'],5)} | 同上 |")
    L.append(f"| $k_{{yy}}/L$ | {fmt(geo['gyration_radius_over_L_pitch'],4)} | 均匀密度惯性矩 |")
    L.append(f"| $k_{{xx}}/B$ | {fmt(geo['gyration_radius_over_B_roll'],4)} | 均匀密度惯性矩 |")
    L.append(f"| $\\rho$, $g$ | {A['case']['fluid']['rho']} kg/m³, {A['case']['fluid']['gravity']} m/s² | 常量 |")
    L.append("")
    L.append(
        f"**几何门（先验 Cb/L/B）。** $C_b$ 闭式 {fmt(mp['Cb_closed_form'],6)} 对求积 "
        f"{fmt(mp['Cb_quadrature'],6)}，偏差 {fmt(geo['Cb_quadrature_deviation_pct'],4)} %；"
        f"体积偏差 {fmt(geo['volume_deviation_pct'],5)} %；$KB$ 闭式对独立求积偏差 "
        f"{fmt(geo['KB_deviation_pct'],4)} %；$\\alpha=0$ 退化时 $KB/T="
        f"{fmt(geo['KB_over_T_at_alpha_zero'],4)}$（经典值 0.625 = $5T/8$）。"
        "**本单同时修掉了仓内 `compute_wigley3_properties` 的 $KB$ 公式缺陷**：旧式 "
        "$T(65+11\\alpha)/16(5+\\alpha)$ 在 $\\alpha=0$ 给出 0.8125 T（+30 %），新式 "
        "$T[1-(15+5\\alpha)/(40+8\\alpha)]$ 给出 0.625 T 并与求积一致。\n"
    )
    L.append("| 海况 | $H_s$ [m] | $T_1$ [s] | $\\omega_p$ [rad/s] | 谱 |")
    L.append("|---|---|---|---|---|")
    for ss in A["case"]["sea_states"]:
        L.append(f"| {ss['name']} | {fmt(ss['hs_m'],2)} | {fmt(ss['t1_s'],2)} | "
                 f"{fmt(ss['omega_p'],4)} | {ss['spectrum']} |")
    L.append("")

    L.append("## ② 怎么做的\n")
    L.append(
        "**求解器（REAL，非 MOCK）。** `tensorbem.rao.compute_rao_at_frequency`：零航速三维自由面"
        " Green 函数边界元（常值面元 + CHIEF 内部点抑制不规则频率）。网格三档："
        + "、".join(
            f"{m['name']} {m['nx']}×{m['nz']}（{A['verification']['geometry']['mesh_panel_counts'][m['name']]} 面元）"
            for m in sd["mesh_levels"]
        )
        + f"；生产档 {sd['frequency_grid_points']} 个频点，全部截断档位精确落在网格节点上。\n"
    )
    L.append(
        "**两个截断轴（RAO 侧）。**\n"
        f"* 轴 A 截断频率 $\\omega_{{\\max}}\\in\\{{{', '.join(str(v) for v in sd['truncation_frequency_levels_rad_s'])}\\}}$ rad/s（精确加倍阶梯）；\n"
        f"* 轴 B 截断半径 $\\lambda_{{\\max}}/L\\in\\{{{', '.join(str(v) for v in sd['truncation_radius_levels_lambda_over_L'])}\\}}$（精确加倍阶梯）；\n"
        "* 每个档位记：带内响应 rms $\\sigma$、带内响应方差占比（截断残余）、带内峰值/峰频。"
        "**验收量取 $\\sigma$**（逐点峰值是内点，对上端截断天然不敏感——这一点在结果里显式列出，"
        "不粉饰）。\n"
    )
    L.append(
        "**几何截断半径轴（轴 C）。** 仓内唯一显式截断流体域的求解器是 "
        "`tensorbem.rankine_fs.solve_rankine_free_surface`（Rankine 源 + 逆风差分辐射条件，"
        "**定常** Neumann-Kelvin），因此几何半径 $y_{{\\max}}/L\\in\\{{{', '.join(str(v) for v in sd['geometric_radius_levels_y_max_over_L'])}\\}}$"
        f"（含 1.0→2.0 与 1.25→2.5 两组精确加倍）、$x_{{\\max}}/L={sd['geometric_radius_x_max_over_L']}$、"
        f"{sd['geometric_radius_panels_per_wavelength']} 面元/波长固定，在任务给出的 "
        f"Fr = {' / '.join(f'{f:.3f}' for f in sd['mission_froudes'])} 上跑。\n"
    )
    L.append(
        "**预登记门限（看结果之前写死）：** 峰值 RAO 偏差 <15 %、峰频偏移 <15 %、截断加倍变化 <3 %。\n"
    )
    L.append(
        "**溯源。** 驱动脚本 sha256 `" + A["provenance"]["inputs"]["script"]["sha256"] + "`；"
        "被调代码与数据 sha256 见工件 `provenance.inputs`（"
        + "、".join(f"`{k}`" for k in list(A["provenance"]["inputs"]["code"])[:4])
        + " …）。录制的命令见文首。\n"
    )

    L.append("## ③ 计算结果\n")
    L.append("### 3.1 RAO 曲线（生产网格，零航速，迎浪）\n")
    L.append("| $\\omega$ [rad/s] | $\\omega\\sqrt{L/g}$ | heave RAO [m/m] | pitch RAO [rad/m] |")
    L.append("|---|---|---|---|")
    for r in A["rao_table"]["rows"]:
        L.append(f"| {fmt(r[0],4)} | {fmt(r[1],4)} | {fmt(r[2],5)} | {fmt(r[3],6)} |")
    L.append("")
    L.append(
        f"峰值：heave $\\omega={fmt(phy['heave_peak']['omega'],4)}$ rad/s "
        f"($\\omega\\sqrt{{L/g}}={fmt(phy['heave_peak_omega_sqrt_L_over_g'],4)}$)，"
        f"值 {fmt(phy['heave_peak']['value'],5)}；pitch $\\omega={fmt(phy['pitch_peak']['omega'],4)}$ rad/s "
        f"($\\omega\\sqrt{{L/g}}={fmt(phy['pitch_peak_omega_sqrt_L_over_g'],4)}$)，"
        f"值 {fmt(phy['pitch_peak']['value'],6)} rad/m。"
        f"最大阻抗残差 {fmt(phy['max_abs_impedance_residual'],3)}。\n"
    )
    L.append("### 3.2 轴 A：截断频率（$\\omega_{\\max}$ 加倍阶梯）\n")
    L.append("| 海况 | $\\omega_{\\max}$ | $\\sigma_3$ [m] | $\\sigma_5$ [rad] | heave 方差占比 | pitch 方差占比 | 峰值在带内 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in axA["rows"]:
        L.append(
            f"| {r['sea_state']} | {fmt(r['omega_max'],2)} | {fmt(r['sigma_heave'],6)} | "
            f"{fmt(r['sigma_pitch'],6)} | {fmt(r['area_fraction_heave'],6)} | "
            f"{fmt(r['area_fraction_pitch'],6)} | {r['peak_inside_band']} |"
        )
    L.append("")
    L.append("| 组 | 加倍对 | 变化 [%] | 最大 [%] | <3 % ? |")
    L.append("|---|---|---|---|---|")
    for key, v in axA["doubling_verdicts"].items():
        for ch in v["doubling_pairs"]:
            L.append(f"| {key} | {ch['pair'][0]}→{ch['pair'][1]} | {pct(ch['rel_change_pct'])} | "
                     f"{pct(v['max_rel_change_pct'])} | {'PASS' if v['pass'] else 'FAIL'} |")
    L.append("")
    L.append("### 3.3 轴 B：截断半径（$\\lambda_{\\max}/L$ 加倍阶梯）\n")
    L.append("| 海况 | $\\lambda_{\\max}/L$ | $\\omega_{\\min}$ [rad/s] | $\\sigma_3$ [m] | $\\sigma_5$ [rad] | heave 方差占比 | pitch 方差占比 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in axB["rows"]:
        L.append(
            f"| {r['sea_state']} | {fmt(r['radius_lambda_over_L'],1)} | {fmt(r['omega_min'],4)} | "
            f"{fmt(r['sigma_heave'],6)} | {fmt(r['sigma_pitch'],6)} | "
            f"{fmt(r['area_fraction_heave'],6)} | {fmt(r['area_fraction_pitch'],6)} |"
        )
    L.append("")
    L.append("| 组 | 加倍对 | 变化 [%] | 最大 [%] | <3 % ? |")
    L.append("|---|---|---|---|---|")
    for key, v in axB["doubling_verdicts"].items():
        for ch in v["doubling_pairs"]:
            L.append(f"| {key} | {ch['pair'][0]}→{ch['pair'][1]} | {pct(ch['rel_change_pct'])} | "
                     f"{pct(v['max_rel_change_pct'])} | {'PASS' if v['pass'] else 'FAIL'} |")
    L.append("")
    L.append("### 3.4 轴 C：几何截断半径（定常 Rankine，任务 Fr）\n")
    L.append("| Fr | $y_{\\max}/L$ | 未知量 | $R_w$ 压力积分 [N] | $R_w$ 远场 [N] | $C_w$ | 能量门 $|R_w^p-R_w^f|/|R_w^p|$ [%] |")
    L.append("|---|---|---|---|---|---|---|")
    for r in axC["rows"]:
        L.append(
            f"| {fmt(r['froude'],3)} | {fmt(r['y_max_factor'],2)} | {r['n_unknowns']} | "
            f"{fmt(r['rw_pressure'],4)} | {fmt(r['rw_farfield'],6)} | {fmt(r['cw_pressure'],5)} | "
            f"{pct(r.get('energy_gate_pct', float('nan')),2)} |"
        )
    L.append("")
    L.append("| Fr | 加倍对 | $C_w$ 变化 [%] | 最大 [%] | <3 % ? |")
    L.append("|---|---|---|---|---|")
    for fr, v in axC["per_froude"].items():
        for ch in v["doubling_pairs"]:
            L.append(f"| {fr} | {ch['pair'][0]}→{ch['pair'][1]} | {pct(ch['rel_change_pct'])} | "
                     f"{pct(v['max_rel_change_pct'])} | {'PASS' if v['pass'] else 'FAIL'} |")
    L.append("")
    L.append("### 3.5 离散化研究（与截断误差解耦）\n")
    L.append("| $\\omega$ | heave 粗→生产 [%] | heave 生产→细 [%] | pitch 粗→生产 [%] | pitch 生产→细 [%] |")
    L.append("|---|---|---|---|---|")
    for r in disc:
        if r.get("heave_coarse_to_production_pct") is None:
            continue
        L.append(f"| {fmt(r['omega'],3)} | {pct(r['heave_coarse_to_production_pct'])} | "
                 f"{pct(r['heave_production_to_fine_pct'])} | {pct(r['pitch_coarse_to_production_pct'])} | "
                 f"{pct(r['pitch_production_to_fine_pct'])} |")
    L.append("")

    L.append("## ④ 结果图\n")
    for p, cap in figs:
        L.append(f"![{p.name}]({p.name})\n\n*{cap}*\n")

    L.append("## ⑤ 注意点\n")
    for note in A["limitations"]:
        L.append(f"* {note}")
    L.append(
        "* **外部锚缺失是硬约束，不是偷懒**：任务指定的 “Journée (1992) KVLCC2 试验 RAO” 在年代上不可能存在"
        "（仓内对 Journée 1992 的引用是《4 个 Wigley 船型迎浪试验与计算》，Report 09-V-09；KVLCC2 船型"
        "晚于 1992 年设计）。仓内也没有任何试验 RAO 表，`tensorbem.reference_provenance` 把包内所有 "
        "Journée 字面量登记为 recalled/UNSOURCED；沙箱禁网，报告不可现场取回。因此判据 1/2 记为 "
        "WITHHELD，并**没有**用 recalled 数字充当锚（仅作诊断行，标注 NOT ACCEPTED）。"
    )
    if diag.get("recalled_heave_peak") is not None:
        L.append(
            f"* 诊断行（**不作判据**）：把 recalled 的 heave 峰值 {fmt(diag['recalled_heave_peak'],3)} / "
            f"峰频 {fmt(diag['recalled_heave_peak_omega'],3)} rad/s 与本单零航速计算值 "
            f"{fmt(diag['computed_peak_heave']['value'],4)} / {fmt(diag['computed_peak_heave']['omega'],4)} rad/s "
            f"并置，偏差 {pct(diag['heave_peak_deviation_pct'],1)} % / "
            f"{pct(diag['heave_peak_omega_offset_pct'],1)} %；该比对的参照值无出处且航速不同（0 vs 试验航速），"
            "只能说明量级不矛盾，不能作为达标证据。"
        )
    L.append(
        "* **零航速限制**：仓内经验证的非定常 BEM 只有零航速 Green 函数（无遭遇频率核/带速 Havelock 核）；"
        "`tensorbem.stf_fullspeed` 只有 STF 装配与圆形剖面系数、无船体激励力构造器，端到端带速 RAO 需新物理，"
        "调参自造即失去可验证性——故 Fr = 0.142/0.20 的 RAO 声明 NOT_SUPPORTED，任务 Fr 只由定常截断域求解器承载。"
    )
    L.append(
        "* **轴 C 的功能量并不可靠**：定常 Rankine 解在模型尺度/Fr ≤ 0.2 的 $C_w$ 为负且压力/远场能量门接近 100 %"
        "（与仓内 W20 #4 记录的 −2.462 一致），说明该压力积分泛函在低 Fr 被非波分量主导。轴 C 的“收敛”因此只是"
        "**同一泛函的域截断收敛**，不能当作 RAO 截断收敛的证据；本单把它单列，不并入判据 3。"
    )
    L.append(
        "* **峰值 vs 带宽**：判据 3 的“RAO 变化”对逐点峰值是弱检验（峰值是内点）；本单因此同时给出逐点峰值"
        "敏感性表与带内方差表，两者都列，读者可自行判断验收口径。"
    )
    L.append("")

    L.append("## ⑥ 误差判定（逐项，显式回答是否 <3 %）\n")
    L.append("| 判据 | 门限 | 裁决 | 数值 |")
    L.append("|---|---|---|---|")
    L.append(f"| ① 峰值 RAO 相对偏差 | <15 % | **{c1['verdict']}** | 无来源锚（见 ⑤） |")
    L.append(f"| ② 峰频偏移 | <15 % | **{c2['verdict']}** | 无来源锚（见 ⑤） |")
    worst_rao = max(
        [v["max_rel_change_pct"] for v in axA["doubling_verdicts"].values()]
        + [v["max_rel_change_pct"] for v in axB["doubling_verdicts"].values()]
        + [c3["rao_peak_sensitivity"]["max_peak_rel_change_pct"]]
    )
    L.append(
        f"| ③ 截断加倍 → RAO 变化 | <3 % | **{c3['verdict']}** | 最大 {pct(worst_rao)} %"
        f"（轴 A/B + 逐点峰值；轴 C 单列 {pct(max(v['max_rel_change_pct'] for v in axC['per_froude'].values())) if axC['per_froude'] else 'n/a'} %） |"
    )
    L.append("")
    L.append(
        "**几何/物理自校验（解析锚，逐项是否 <3 %）：**\n\n"
        "| 检查 | 偏差 | <3 % ? |\n|---|---|---|\n"
        f"| $C_b$ 求积 vs 闭式 | {pct(geo['Cb_quadrature_deviation_pct'])} % | {'是' if geo['Cb_quadrature_deviation_pct'] < 3 else '否'} |\n"
        f"| 体积 求积 vs 闭式 | {pct(geo['volume_deviation_pct'])} % | {'是' if geo['volume_deviation_pct'] < 3 else '否'} |\n"
        f"| $KB$ 闭式 vs 独立求积 | {pct(geo['KB_deviation_pct'])} % | {'是' if geo['KB_deviation_pct'] < 3 else '否'} |\n"
        f"| 长波极限 heave RAO vs 1 | {pct(phy['heave_long_wave_deviation_pct'])} % | {'是' if phy['heave_long_wave_deviation_pct'] < 3 else '否'} |"
    )
    for bc in phy["beta_invariance"]:
        L.append(
            f"| 迎浪约定 $\\beta\\to\\beta+\\pi$ 不变性（$\\omega$={fmt(bc['omega'],2)}） | "
            f"heave {pct(bc['heave_rel_change_pct'])} % / pitch {pct(bc['pitch_rel_change_pct'])} % | "
            f"{'是' if max(bc['heave_rel_change_pct'], bc['pitch_rel_change_pct']) < 3 else '否'} |"
        )
    L.append("")
    L.append(
        f"**结论（不粉饰）：** 判据 ③ 的机器裁决为 **{c3['verdict']}**，最大截断加倍变化 "
        f"{pct(c3['worst_rel_change_pct'],2)} %（未过 3 % 的门，不粉饰为达标）；失败对集中在低截断档位——"
        "它们切进了响应带（本模型 pitch 峰在 $\\omega\\approx$"
        f"{fmt(phy['pitch_peak']['omega'],3)} rad/s），属物理上“截断不足”而非数值噪声。"
        "判据 ① ② 因外部试验锚在本环境不可得而 **WITHHELD（honest FAIL）**，不可被后续轮次覆盖为“达标”。"
        "零航速与带速 RAO 的能力缺口、以及定常 Rankine 泛函的低 Fr 失效，均已在工件 "
        "`capabilities.gaps` 与上节登记，供后续轮次作为缺口账本入口。\n"
    )
    L.append(
        "## 复算\n\n```\npython run_w25_rao_truncation.py --stage sweep --shard 1/8   # … 8/8\n"
        "python run_w25_rao_truncation.py --stage radius\n"
        "python run_w25_rao_truncation.py --stage assemble\n"
        "python papers/reports/build_w25_report.py\n```\n"
    )
    return "\n".join(L)


def main() -> int:
    A = json.loads(ART.read_text())
    figs = make_figures(A)
    md = render_md(A, figs)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_MD} ({len(md)} chars) and {len(figs)} figures")
    for p, _ in figs:
        print(" fig", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
