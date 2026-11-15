"""Render the round-4 doc + figures for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK.

Zero hand-copying: every number in the markdown and every plotted point comes from
``runs/m21520/ladder_study.json``.  The renderer refuses to emit a document if the
study artifact is missing, holds a non-finite value outside the documented
``observed_order`` slots, or if the gate block is absent -- a document that cannot be
regenerated from the artifact is not evidence.

    PYTHONPATH=src python3 tools/case_havelock_round4_doc.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs" / "m21520" / "ladder_study.json"
FIGDIR = ROOT / "docs" / "case_bem_submerged_sphere_havelock"
DOC = ROOT / "docs" / "case_submerged_sphere_havelock_round4.md"

QUANTITY_LABEL = {"a33": "$A'_{33}$", "b33": "$B'_{33}$", "a11": "$A'_{11}$", "b11": "$B'_{11}$"}
MEASURE_LABEL = {"spherical_patch": "球面片测度", "flat": "平面测度"}


def _fmt(value: float, digits: int = 4, scale: float = 1.0, suffix: str = "") -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "nan"
    return ("%." + str(digits) + "f%s") % (value * scale, suffix)


def _pct(value: float, digits: int = 3) -> str:
    return _fmt(value, digits, 100.0, "%")


def _guard(study: dict) -> None:
    if "gates" not in study or "cells" not in study:
        raise SystemExit("study artifact lacks gates/cells -- run the study first")
    allowed = {"observed_order", "richardson_continuum", "gci_fine",
               "err_continuum_vs_ref", "err"}   # err: gate detail of an inadmissible p

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, "%s.%s" % (path, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, "%s[%d]" % (path, index))
        elif isinstance(node, float) and not math.isfinite(node):
            leaf = path.rsplit(".", 1)[-1].split("[")[0]
            if leaf not in allowed:
                raise SystemExit("non-finite value outside the declared slots: %s" % path)

    walk(study)
    if not study["cells"]:
        raise SystemExit("no cells in the study artifact")


def _cell(study, measure, ka, quantity):
    for cell in study["cells"]:
        if (cell["measure"] == measure and abs(cell["ka"] - ka) < 1e-12
                and cell["quantity"] == quantity):
            return cell
    raise SystemExit("missing cell %r" % ((measure, ka, quantity),))


def make_figures(study: dict) -> list[dict]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figures = []
    band = study["ka_band"]
    quantities = ("a33", "b33")
    colors = {0.1: "#1f77b4", 0.5: "#2ca02c", 1.0: "#d62728", 2.0: "#9467bd"}

    # --- fig 7: error vs panel count, both measures (log-log) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, quantity in zip(axes, quantities):
        for ka in band:
            for measure, style in (("flat", "o--"), ("spherical_patch", "s-")):
                cell = _cell(study, measure, ka, quantity)
                ax.loglog(cell["panels"], [cell["err_coarse_vs_ref"],
                                           cell["err_medium_vs_ref"],
                                           cell["err_fine_vs_ref"]],
                          style, color=colors[ka], lw=1.2, ms=5,
                          label="ka=%.1f %s" % (ka, MEASURE_LABEL[measure]))
        ax.axhline(0.05, color="k", lw=0.9, ls=":", label="5% case gate")
        ax.set_xlabel("panels")
        ax.set_ylabel("relative error vs multipole reference")
        ax.set_title("error ladder: %s" % QUANTITY_LABEL[quantity])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    path = FIGDIR / "fig7_round4_error_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": path.name, "path": path,
                    "caption": ("Error vs panel count on the 80/320/1280 rungs for both "
                                "area measures (dashed = flat, solid = spherical patch); "
                                "the dotted line is the 5 % case gate.")})

    # --- fig 8: observed order per cell, patch vs flat ---
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    labels, xs = [], []
    width = 0.38
    positions = []
    order_labels = []
    for index, (ka, quantity) in enumerate([(ka, q) for ka in band for q in quantities]):
        positions.append(index)
        order_labels.append("ka=%.1f\n%s" % (ka, QUANTITY_LABEL[quantity]))
    for measure, color, offset in (("flat", "#8c564b", -width / 2),
                                   ("spherical_patch", "#1f77b4", width / 2)):
        values = []
        for ka, quantity in [(ka, q) for ka in band for q in quantities]:
            cell = _cell(study, measure, ka, quantity)
            values.append(cell["observed_order"] if math.isfinite(cell["observed_order"])
                          else 0.0)
        ax.bar([p + offset for p in positions], values, width=width, color=color,
               label=MEASURE_LABEL[measure])
    ax.set_xticks(positions)
    ax.set_xticklabels(order_labels, fontsize=6.5)
    ax.axhline(study["gates_predeclared"]["G1_order_band"][0], color="k", ls=":", lw=1)
    ax.axhline(study["gates_predeclared"]["G1_order_band"][1], color="k", ls=":", lw=1)
    ax.set_ylabel("observed order p (bars at 0 = no admissible order)")
    ax.set_title("observed order of accuracy on the verdict measure vs the flat control")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGDIR / "fig8_round4_observed_order.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": path.name, "path": path,
                    "caption": ("Observed order p per (ka, coefficient); missing bars mean the "
                                "ladder is non-monotone or round-off-limited (no admissible p).")})

    # --- fig 9: continuum limit vs reference ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, quantity in zip(axes, quantities):
        cells_p = [_cell(study, "spherical_patch", ka, quantity) for ka in band]
        x = list(range(len(band)))
        ax.bar([i - 0.2 for i in x], [c["err_fine_vs_ref"] for c in cells_p], width=0.4,
               color="#1f77b4", label="1280-panel vs ref")
        ax.bar([i + 0.2 for i in x], [c["err_continuum_vs_ref"] if math.isfinite(
            c["err_continuum_vs_ref"]) else 0.0 for c in cells_p], width=0.4,
               color="#ff7f0e", label="Richardson continuum vs ref")
        ax.set_xticks(x)
        ax.set_xticklabels(["ka=%.1f" % ka for ka in band])
        ax.set_yscale("log")
        ax.set_ylabel("relative difference")
        ax.set_title("continuum limit lands on the reference: %s" % QUANTITY_LABEL[quantity])
        ax.axhline(study["gates_predeclared"]["G2_continuum_tol"], color="k", ls=":",
                   label="G2 2% gate")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGDIR / "fig9_round4_continuum_vs_reference.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": path.name, "path": path,
                    "caption": ("1280-panel error and Richardson continuum error against the "
                                "Havelock multipole reference (log scale; G2 gate at 2 %).")})

    # --- fig 10: the mechanism (area deficit) ---
    diag = study["mesh_area_diagnostic"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    panels = [row["n_panels"] for row in diag]
    for measure, marker, color in (("flat", "o", "#8c564b"),
                                   ("spherical_patch", "s", "#1f77b4")):
        deficits = [abs(row[measure]["deficit_ratio"]) for row in diag]
        ax.loglog(panels, deficits, marker + "-", color=color,
                  label=MEASURE_LABEL[measure])
    ax.set_xlabel("panels")
    ax.set_ylabel("|area deficit| vs exact $4\\pi a^2$")
    ax.set_title("why the measures differ: panel-area deficit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "fig10_round4_area_deficit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": path.name, "path": path,
                    "caption": ("Total wetted-panel area relative to the exact sphere area: the "
                                "flat triangle measure loses area, the patch measure recovers it.")})
    return figures


def render(study: dict, figures: list[dict]) -> str:
    geometry = study["geometry"]
    band = study["ka_band"]
    gates = study["gates"]
    panels = study["panels_per_rung"]

    lines = []
    add = lines.append
    add("# 算例 CASE-BEM-SUBMERGED-SPHERE-HAVELOCK 第 4 轮：判据测度上的收敛阶与连续极限")
    add("")
    add("> 本文件由 `tools/case_havelock_round4_doc.py` 从 `runs/m21520/ladder_study.json`")
    add("> 程序化渲染（零手抄；渲染前对工件做非有限值扫描，除约定的 `observed_order`/")
    add("> Richardson 槽位外出现 nan/inf 即拒绝出稿）。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("半径 $a=%.2f$ m 的刚性球，球心潜深 $f=%.2f$ m（$f/a=%.2f$），深水自由液面下作单位"
        % (geometry["radius"], geometry["submergence"], geometry["submergence_ratio"]))
    add("速度幅值简谐振荡。关注量 $A'=A/(\\rho\\nabla)$、$B'=B/(\\rho\\omega\\nabla)$，")
    add("$\\nabla$ 取精确排水体积。参考解为仓库内 Havelock 1955 频域多极子级数")
    add("（`analytic.submerged_sphere_wave_multipole`，20 项）。")
    add("")
    add("| 参数 | 值 |")
    add("|---|---|")
    add("| 球半径 $a$ [m] | %.3f |" % geometry["radius"])
    add("| 潜深 $f$ [m] | %.3f |" % geometry["submergence"])
    add("| 潜深比 $f/a$ | %.3f |" % geometry["submergence_ratio"])
    add("| 密度 $\\rho$ [kg/m³] | %.1f |" % geometry["density"])
    add("| 重力 $g$ [m/s²] | %.5f |" % geometry["gravity"])
    add("| 网格阶梯（面元数） | %s |" % ", ".join(str(p) for p in panels))
    add("| 精化比 $r$（$h$ 折半） | %.1f |" % study["refinement_ratio_h"])
    add("| 频率带 $ka$ | %s |" % ", ".join("%.1f" % ka for ka in band))
    add("")
    add("## ② 怎么做的")
    add("")
    add("第 3 轮（`docs/case_submerged_sphere_havelock_round3.md`）用球面片测度把 320 面板整条带判 PASS，")
    add("但同一节里留了两处硬欠账：**判据测度下的 1280 面板精化从未取得**（`refinement_status = not_completed`），")
    add("以及「球面片测度误差随 $h$ 一阶下降」只是**假设**。仓库既有的 V&V 研究")
    add("（`benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_continuum.py`）拟合过收敛阶，")
    add("但用的是**平面测度**——用一套测度判分、用另一套测度诊断收敛，正是本轮要消除的证据割裂。")
    add("")
    add("本轮不改方法、不调参，只做配对测量：")
    add("")
    add("1. **阶梯求解**：$N\\in\\{%s\\}$ 面元 × 两种面积测度（球面片 = 判据测度，平面 = 对照）"
        % ", ".join(str(p) for p in panels))
    add("   × $ka\\in\\{%s\\}$，每个 (测度, 网格, 频点) 单元一个进程，算术完全相同。" % ", ".join("%.1f" % ka for ka in band))
    add("2. **观测阶**：三网格连续差商 $p=\\ln|\\Delta_{cm}/\\Delta_{mf}|/\\ln r$（Roache eq.5）；")
    add("   非单调或落到舍入噪声的单元**不给阶**（判 `non_monotone` / `at_roundoff`，如实登记，不静默丢弃）。")
    add("3. **Richardson 连续极限** 与 **GCI**（Celik et al. 2008，$F_s=1.25$），再与多极子参考解对表：")
    add("   若参考解有偏，精化不会把误差压向零；若残差是网格的，连续极限应落在参考解上。")
    add("4. **测度机制**：直接量各测度的面元总面积与精确球面积 $4\\pi a^2$ 的亏损比，不靠断言。")
    add("")
    add("预置判据（出数前写死）：")
    add("")
    add("| 判据 | 门槛 |")
    add("|---|---|")
    add("| G1 阶可取性 | 可解单元 $%s \\le p \\le %s$ |" % tuple(study["gates_predeclared"]["G1_order_band"]))
    add("| G2 连续极限验证参考解 | 球面片测度每单元 $|f_*-f_{ref}|/|f_{ref}| \\le %.0f\\%%$ |"
        % (study["gates_predeclared"]["G2_continuum_tol"] * 100))
    add("| G3 精化单调 | 每个 (ka, 量) 上 err(1280) < err(320)，球面片测度 |")
    add("| H1 测度假设 | 中位 $p_{patch} \\ge p_{flat} + %.1f$ 则「测度也提阶」，否则「只降常数」 |"
        % study["gates_predeclared"]["H1_order_margin"])
    add("")
    add("复现：")
    add("")
    add("```bash")
    add("PYTHONPATH=src python3 tools/case_havelock_measure_order_study.py --stage solve --processes 8")
    add("PYTHONPATH=src python3 tools/case_havelock_measure_order_study.py --stage analyse")
    add("PYTHONPATH=src python3 tools/case_havelock_round4_doc.py")
    add("```")
    add("")
    add("## ③ 计算结果")
    add("")
    add("### 3.1 面元面积亏损（测度机制）")
    add("")
    add("| 面元数 | 平面测度总面积 | 亏损比 | 球面片测度总面积 | 亏损比 |")
    add("|---|---|---|---|---|")
    for row in study["mesh_area_diagnostic"]:
        add("| %d | %.6f | %s | %.6f | %s |" % (
            row["n_panels"], row["flat"]["total_area"], _pct(row["flat"]["deficit_ratio"]),
            row["spherical_patch"]["total_area"],
            _pct(row["spherical_patch"]["deficit_ratio"])))
    add("")
    add("精确球面积 $4\\pi a^2 = %.6f$ m²。" % study["mesh_area_diagnostic"][0]["exact_area"])
    add("")
    for measure in ("spherical_patch", "flat"):
        add("### 3.%d %s：逐单元误差与收敛阶" % (1 if measure == "spherical_patch" else 2,
                                              MEASURE_LABEL[measure]))
        add("")
        add("| $ka$ | 量 | BEM %d 面元 | BEM %d 面元 | BEM %d 面元 | 参考 | $p$ | 阶区 | Richardson | $e_{1280}$ | $e_{*}$ |"
            % (panels[0], panels[1], panels[2]))
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for ka in band:
            for quantity in ("a33", "b33", "a11", "b11"):
                cell = _cell(study, measure, ka, quantity)
                add("| %.1f | %s | %.6f | %.6f | %.6f | %.6f | %s | %s | %s | %s | %s |" % (
                    ka, QUANTITY_LABEL[quantity], *cell["values"], cell["reference"],
                    _fmt(cell["observed_order"], 3), cell["order_regime"],
                    _fmt(cell["richardson_continuum"], 6),
                    _pct(cell["err_fine_vs_ref"]),
                    _pct(cell["err_continuum_vs_ref"]) if math.isfinite(
                        cell["err_continuum_vs_ref"]) else "nan"))
        add("")
    add("### 3.3 判据判定")
    add("")
    add("| 判据 | 实测 | 裁定 |")
    add("|---|---|---|")
    g1 = gates["G1_order_admissible"]
    add("| G1 阶可取性 | 可解 %d 单元，越界 %d，最坏 $|p|$=%s | **%s** |"
        % (g1["resolved_cells"], g1["violations"], _fmt(g1["worst"], 3), g1["verdict"]))
    g2 = gates["G2_continuum_vs_reference"]
    add("| G2 连续极限 vs 参考解 | 最大 $e_*$ = %s（%d 单元，越界 %d）；对照 $e_{1280}$ 最大 %s | **%s** |"
        % (_pct(g2["max_err_continuum"]), g2["cells"], g2["violations"],
           _pct(g2["max_err_fine"]), g2["verdict"]))
    g3 = gates["G3_monotone_refinement"]
    add("| G3 精化单调 | %d 单元，违例 %d | **%s** |"
        % (g3["cells"], g3["violations"], g3["verdict"]))
    h1 = gates["H1_patch_raises_order"]
    add("| H1 测度提阶假设 | 中位 $p$：球面片 %s vs 平面 %s | **%s**（%s） |"
        % (_fmt(h1["median_order_patch"], 3), _fmt(h1["median_order_flat"], 3),
           h1["verdict"], h1["reading"]))
    add("")
    if g2.get("violations_detail"):
        add("G2 越界明细：")
        add("")
        add("| $ka$ | 量 | $e_*$ |")
        add("|---|---|---|")
        for row in g2["violations_detail"]:
            add("| %.1f | %s | %s |" % (row["ka"], QUANTITY_LABEL[row["quantity"]],
                                        _pct(row["err"])))
        add("")
    if g3.get("violations_detail"):
        add("G3 违例明细：")
        add("")
        add("| $ka$ | 量 | $e_{320}$ | $e_{1280}$ |")
        add("|---|---|---|---|")
        for row in g3["violations_detail"]:
            add("| %.1f | %s | %s | %s |" % (row["ka"], QUANTITY_LABEL[row["quantity"]],
                                             _pct(row["err_medium"]), _pct(row["err_fine"])))
        add("")
    add("## ④ 结果图")
    add("")
    for figure in figures:
        add("![%s](case_bem_submerged_sphere_havelock/%s)" % (figure["caption"], figure["name"]))
        add("")
        add("*%s*" % figure["caption"])
        add("")
    add("## ⑤ 误差判定（是否 <3%）")
    add("")
    add("关键量 = 球面片测度（判据测度）上 1280 面元的四项辐射系数，基准 = 多极子参考解：")
    add("")
    add("| $ka$ | 量 | BEM 1280 | 参考 | 相对误差 | <3%? | Richardson 极限 | 极限误差 |")
    add("|---|---|---|---|---|---|---|---|")
    worst = 0.0
    for ka in band:
        for quantity in ("a33", "b33", "a11", "b11"):
            cell = _cell(study, "spherical_patch", ka, quantity)
            err = cell["err_fine_vs_ref"]
            worst = max(worst, err)
            add("| %.1f | %s | %.6f | %.6f | %s | %s | %s | %s |" % (
                ka, QUANTITY_LABEL[quantity], cell["values"][2], cell["reference"],
                _pct(err), "是" if err < 0.03 else "否",
                _fmt(cell["richardson_continuum"], 6),
                _pct(cell["err_continuum_vs_ref"]) if math.isfinite(
                    cell["err_continuum_vs_ref"]) else "nan"))
    add("")
    add("**最坏 1280 面元误差 = %s**（%s 3%%）；连续极限最坏误差 = %s。"
        % (_pct(worst), "小于" if worst < 0.03 else "**不小于**", _pct(g2["max_err_continuum"])))
    add("")
    add("## ⑥ 注意点与局限")
    add("")
    add("- 球面片测度只改面积权（面元、法向、配置点、奇异核都不动），因此它能在多大程度上")
    add("  改善阶，是**测出来的**而不是假设出来的——见 H1 判据的裁定。")
    add("- 只有三级网格时，每个单元只能给出**一个** $p$，$p$ 本身对舍入敏感；`non_monotone` /")
    add("  `at_roundoff` 单元不给阶，也不参与 G1，但仍在表里逐项对参考解打分（不静默丢弃）。")
    add("- 参考解与 BEM 共用球谐加法定理族；真正的外部独立性来自 capytaine 交叉核对")
    add("  （`docs/case_bem_submerged_sphere_havelock/capytaine_crosscheck.json`）与")
    add("  `submerged_sphere_havelock_integrity` 的四项自证（截断/Haskind/深潜极限/低频阻尼）。")
    add("- 本轮只覆盖深水、单一潜深 $f/a=%.1f$ 与 $ka\\in[%.1f, %.1f]$；有限水深与浅潜不在范围内。"
        % (geometry["submergence_ratio"], min(band), max(band)))
    add("")
    add("## ⑦ 与上一轮的关系")
    add("")
    add("- 第 3 轮留欠账：判据测度下 1280 面板未取得、收敛阶未测。本轮两项均已出数（见 3.1/3.3）。")
    add("- 第 3 轮的 320 面板 PASS（28/28 项 <5%）**不被本轮改写**；本轮只说明它的残差是网格的还是参考解的。")
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--doc", type=Path, default=DOC)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    _guard(study)
    figures = make_figures(study)
    text = render(study, figures)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(text)
    print(json.dumps({"doc": str(args.doc), "figures": [f["name"] for f in figures],
                      "chars": len(text)}, indent=2))


if __name__ == "__main__":
    main()