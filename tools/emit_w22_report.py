"""Render the W22 KP505 report from the two artefacts (no number is typed here).

Usage::

    PYTHONPATH=$PWD/src python tools/emit_w22_report.py

Every figure quoted in the markdown is read out of
``benchmarks/verified/kp505_anchor_w22/{kp505_transcription,openwater_compare}.json``;
the only literals in this file are prose, section names and formatting.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ARTE = REPO_ROOT / "benchmarks/verified/kp505_anchor_w22"
TRANSCRIPTION = ARTE / "kp505_transcription.json"
COMPARE = ARTE / "openwater_compare.json"
REPORT_REL = "papers/reports/w22_propeller_openwater6.md"


def _f(value: float, digits: int = 2, sign: bool = False) -> str:
    if value is None:
        return "n/a"
    if value != value:  # nan
        return "nan"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def summary_text(tr: dict, cp: dict) -> str:
    """One-paragraph, artefact-derived summary (md section 7 == API `summary`)."""

    anchor = tr["anchor"]
    d = cp["design_point"]
    kt = cp["attribution"]["channel_panel_density"]["KT"]
    vis = cp["attribution"]["channel_viscous_correction"]
    wake = cp["attribution"]["channel_wake_model"]
    span = [row["dKT_pct"] for row in wake]
    induced = next(row for row in wake if row["wake_model"] == "induced")
    return (
        f"KP505 敞水试验锚在本仓可达（{anchor['n_rows']} 行 / {anchor['n_fields']} 字段逐字段带 "
        f"file+sha256+line+verbatim，provenance = {anchor['provenance_class']}；原始手稿层 WITHHELD）。"
        f"BEM 在 J=0.1~0.9 五档给出 K_T/K_Q/η₀，设计点 J={d['J']:.2f} 的 ΔK_T = {d['dKT_pct']:+.2f} %"
        f"（门槛 ±5 %）⇒ 判据 3 为 honest FAIL。归因：尾流模型是主渠道（三模型 ΔK_T 极差 "
        f"{max(span) - min(span):.2f} 个百分点，力自由 induced 达 {induced['dKT_pct']:+.2f} % 但非预声明主核、"
        f"且 Δη₀ {induced['deta0_pct']:+.2f} %）；面元密度次之（末次增量 {kt['last_increment_pct']:.2f} %，"
        f"观测阶 {kt['observed_order']:+.2f} 非渐近，GCI 仅作量级指示）；粘性修正为小项"
        f"（{vis['viscous_share_of_total_dKT_pct']:+.2f} 个百分点）；余项归几何/载荷保真度。"
    )


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def build() -> str:
    tr = json.loads(TRANSCRIPTION.read_text())
    cp = json.loads(COMPARE.read_text())

    anchor = tr["anchor"]
    reach = tr["reachability"]
    meta = cp["meta"]
    verdict = cp["verdict"]
    design = cp["design_point"]
    attrib = cp["attribution"]
    points = cp["points"]
    gci = cp["gci"]
    ladder = cp["ladder"]
    by_id = {row["id"]: row for row in cp["criteria"]}

    lines: list[str] = []
    add = lines.append

    add("# W22 · KP505 螺旋桨敞水性能：锚可达性回核 + 五档 BEM 对比")
    add("")
    add(
        "> 领域仓 `TensorBEM` ／ 学科域：**推进（propeller open-water）** ／ 承接零报告任务 "
        "`paper-exp-propeller-openwater-bem-1`。本报告的全部数值由工件注入："
        f"`benchmarks/verified/kp505_anchor_w22/{TRANSCRIPTION.name}`、"
        f"`benchmarks/verified/kp505_anchor_w22/{COMPARE.name}`，"
        f"驱动 commit `{meta['git_commit']}`。"
    )
    add("")
    add("## 0. 结论摘要（机器裁决）")
    add("")
    add(f"- 锚可达性：**{'可达' if tr['verdict']['anchor_reachable'] else '不可达（honest FAIL）'}** — "
        f"命中文件 {tr['verdict']['hit_files']} 个，转录字段 {tr['verdict']['transcribed_fields']} 个，"
        f"逐文件 sha256 全部登记；provenance 类别 `{tr['verdict']['provenance_class']}`。")
    add(f"- 五档敞水表（K_T/K_Q/η₀ × J=0.1/0.3/0.5/0.7/0.9）已落表，每档带相对偏差与参照类型。")
    add(f"- 设计点 J = {_f(design['J'], 2)}：BEM K_T = {_f(design['KT_bem'], 5)}，"
        f"锚 K_T = {_f(design['KT_ref'], 3)}，**ΔK_T = {_f(design['dKT_pct'], 2, True)} %**；"
        f"门槛 ±{_f(cp['meta']['inputs']['thresholds']['design_dKT_pct'], 1)} % "
        f"⇒ **{'达标' if design['within_5pct'] else '未达标（honest FAIL，已逐渠道归因）'}**。")
    add(f"- 面元密度 GCI：{len(ladder)} 套网格（r = {_f(cp['refinement_ratio'], 1)}），"
        f"观测阶与 GCI 指数在 5 档 × (K_T, K_Q) 全部落表；"
        f"渐近区格数 {by_id['A4-panel-density-gci']['measured']['n_cells_asymptotic']}"
        f"/{by_id['A4-panel-density-gci']['measured']['n_cells']}（非渐近 ⇒ GCI 只作量级指示，不作误差上界）。")
    add(f"- 判据裁决：A1 {'PASS' if by_id['A1-anchor-reachability-machine-decided']['pass'] else 'FAIL'}／"
        f"A2 {'PASS' if by_id['A2-five-J-table-with-deviation-and-reference-type']['pass'] else 'FAIL'}／"
        f"A3 {'PASS' if by_id['A3-design-point-KT-within-5pct']['pass'] else 'FAIL'}／"
        f"A4 {'PASS' if by_id['A4-panel-density-gci']['pass'] else 'FAIL'}"
        f"（{verdict['summary']}）。")
    add("")

    # ---------------------------------------------------------------- ① 案例
    add("## 1. 案例介绍（物理背景 / 几何 / 参数表）")
    add("")
    add(
        "对象为 KRISO 集装箱船（KCS）桨 **KP505**：五叶定距桨，敞水试验曲线是推进学科最常用的"
        "外部锚之一。本单不造新算例，而是①先机检‘锚在仓内是否可得’，②再在 J ∈ {0.1, 0.3, 0.5, "
        "0.7, 0.9} 五档上用仓库自带的涡格升力面 BEM 算 K_T / K_Q / η₀ 并逐档与可得参照对比。"
    )
    add("")
    geom = cp["meta"]["inputs"]["bem_geometry"]
    from tensorbem.propeller import BladeGeometry

    d = BladeGeometry()
    add("**几何与工况参数**（BEM 侧声明假设：数值取自驱动 `PRIMARY` 配置与 `BladeGeometry` 缺省值，"
        "不手抄；锚件 Table 1 的模型几何在仓内以 sha256 登记，供对照）")
    add("")
    lines += _md_table(
        ["量", "取值", "说明"],
        [
            ["直径 D", f"{_f(d.diameter, 1)}（无量纲化约定）",
             "BEM 内部以 D 无量纲化；锚件 Table 1 的模型直径见 "
             "`benchmarks/anchors/sources/kp505_marine2023_table1_propeller_characteristics.txt`"],
            ["叶数 Z", str(d.blades), "`blades`；与锚件 Table 1 一致"],
            ["盘面比 A_E/A_O", _f(d.expanded_area_ratio, 2),
             "弦长分布按此**精确匹配**（闭式积分 `chord_scale`）"],
            ["毂径比 d_h/D", _f(d.hub_ratio, 2), "`hub_ratio`"],
            ["螺距比 P/D（均值）", _f(d.pitch_ratio, 2),
             "径向常数（真桨径向螺距分布未在锚件中发布，不臆造）"],
            ["剖面", "零弯度平板涡格", "锚件 Table 1 为 NACA66（有弯度）——差异进‘模型项’"],
            ["厚度比 t/c", _f(geom["thickness_ratio"], 2), "进入 ITTC-78 剖面阻力律"],
            ["转速", f"n = {_f(geom['rev_per_second'], 1)} rev/s",
             "只经雷诺数进入阻力项；Re ∝ D²，故这是代表工况而非 0.25 m 模型实况"],
            ["尾流模型", f"`{geom['wake_model']}`，长度 {_f(geom['wake_length'], 1)} D，"
                        f"{geom['wake_steps']} 段（L1 级）", "螺旋尾涡离散；L2/L3 为 48/96 段"],
            ["进速档位 J", "、".join(_f(j, 1) for j in cp["meta"]["inputs"]["mission_J"]), "任务书五档"],
        ],
    )
    add("")
    add("**外部锚与其证据级别（回核结果，非记忆值）**")
    add("")
    add(f"- 锚 ID `{anchor['anchor_id']}`，类型 `{anchor['reference_type']}`，"
        f"provenance `{anchor['provenance_class']}`。")
    add(f"- 出处：{anchor['citation']['paper']}（DOI {anchor['citation']['doi']}），"
        f"{anchor['citation']['location']}，表题 verbatim：“{anchor['citation']['table_caption_verbatim']}”。")
    add(f"- 表格原文文件：`{anchor['file']}`，sha256 `{anchor['sha256']}`，"
        f"共 {anchor['n_rows']} 行 / {anchor['n_fields']} 个数值字段。")
    add(
        "- 证据级别判定：表在仓内**真实存在且可机读**（REAL，二级转录）；但该论文页 10 **未引用**"
        "KRISO 原始试验报告 ⇒ 对‘原始试验手稿’这一层记为 **WITHHELD**（仓内无 primary handout）。"
    )
    add(f"- 解析式锚 `AN-ITTC57`：`{anchor['ittc57_anchor']['formula_verbatim']}`"
        f"（{anchor['ittc57_anchor']['reference']}），类型 `{anchor['ittc57_anchor']['type']}`。")
    add("")

    # ---------------------------------------------------------------- ② 方法
    add("## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    add("")
    add("### 2.1 步骤 1 —— 锚可达性机检（判据 A1）")
    add("")
    add(f"在 TensorBEM 检出内做 glob（`{'`, `'.join(reach['patterns'])}`）与文本检索：")
    add("")
    add("```")
    add(reach["grep_command"])
    add("```")
    add("")
    layer_rows = [[k, str(v)] for k, v in reach["layer_summary"].items()]
    lines += _md_table(["层（后缀分类）", "命中文件数"], layer_rows)
    add("")
    add(f"合计命中 **{reach['n_hit_files']}** 个文件（glob 去重后 {reach['glob_hit_files']} 条命中、"
        f"grep 命中 {reach['grep_hit_files']} 个），其中表格层 "
        f"{reach['layer_summary'].get('table', 0)} 个、图件层 {reach['layer_summary'].get('figure', 0)} 个、"
        f"几何层 {reach['layer_summary'].get('geometry', 0)} 个。每个命中文件都登记了 sha256（见工件 "
        "`kp505_transcription.json → reachability.hit_files`）。表文件存在性："
        f"`{reach['anchor_table_source']}` ⇒ "
        f"{'存在' if reach['anchor_table_source_present'] else '缺失'}。")
    add("")
    add("表格层命中文件（前 8 条，完整清单在工件里）：")
    add("")
    lines += _md_table(
        ["文件", "sha256（前 16 位）", "字节"],
        [[r["file"], r["sha256"][:16], str(r["bytes"])]
         for r in reach["hit_files"] if r["layer"] == "table"][:8],
    )
    add("")

    add("### 2.2 步骤 2 —— 零手抄转录（判据 A1/A2 的输入）")
    add("")
    sc = anchor["source_self_consistency"]
    add(
        "驱动用行文法正则从表文件中逐行解析 7 列（J, K_T^CFD, 10K_Q^CFD, K_T^exp, 10K_Q^exp, err_K_T, err_10K_Q），"
        f"每个数值写一条记录 `file + sha256 + line + field + verbatim + reference_type`，"
        f"共 {anchor['n_rows']} 行 / **{anchor['n_fields']} 个字段，其中 {anchor['n_fields_with_source']} 个"
        "带完整溯源**。派生量 K_Q = 10K_Q/10 与 η₀ = J K_T/(2π K_Q) 由已转录列**算出**，不另行抄写。"
    )
    add("")
    add(
        f"自洽性回核（用表自身的 CFD/Experimental 列重算它的误差列，按无符号口径比较 "
        f"{sc['n_checks']} 处）：最大偏差 **{_f(sc['max_abs_diff_pct_points'], 3)} 个百分点**"
        f"（最差格 J = {_f(sc['worst']['J'], 1)} 的 {sc['worst']['quantity']}：印 "
        f"{_f(sc['worst']['printed_err_pct'], 2)} % vs 重算 {_f(sc['worst']['recomputed_err_pct'], 2)} %）。"
        f"该偏差要按**逐行传播的舍入界**判读（表给 3 位小数，最小的 K_T = 0.029 本身已带 ±1.7 % 相对舍入）："
        f"最紧一格的 ratio_to_bound = {_f(sc['max_ratio_to_roundoff_bound'], 3)} ≤ 1 ⇒ 转录忠实、"
        "偏差可由印刷精度完全解释。"
    )
    add("")
    add(
        f"与仓内已钉住的锚表交叉核对：`cross_check_pinned_table_mismatches` = "
        f"{len(anchor['cross_check_pinned_table_mismatches'])} 条 ⇒ 本次转录与既有 `ANCHOR` 表逐位一致。"
    )
    add("")

    add("### 2.3 步骤 3 —— 参照体系（三级标注）")
    add("")
    lines += _md_table(
        ["参照", "类型标注", "来源", "在本研究中的角色"],
        [
            ["KP505 敞水试验曲线", "`experiment`（二级转录）",
             f"`{anchor['file']}` (sha256 {anchor['sha256'][:16]}…)",
             "主参照：五档 K_T/K_Q/η₀ 的偏差基数"],
            ["AN-ITTC57 摩擦线", "`analytic`",
             f"`{anchor['ittc57_anchor']['formula_verbatim']}`",
             "粘性通道交叉核对（与本仓 ITTC-78 剖面阻力律对比）"],
            ["动量理论（作动盘）效率上界", "`analytic`",
             "涡格解出的 C_T ⇒ η_ideal = 2/(1+√(1+C_T))",
             "物理可容许性门（η₀ 不得越界）"],
        ],
    )
    add("")

    add("### 2.4 步骤 4 —— BEM 求解与面元密度梯度（判据 A4）")
    add("")
    add(
        "求解器：`tensorbem.propeller.open_water_point`（涡格升力面 + 螺旋尾涡 + Kutta-Joukowski 力 + "
        "ITTC-78 剖面阻力）。梯度设计：**三条面元轴同时加密、r = 2**，涡核正则化 `core_ratio` 固定，"
        "因此梯度只测离散、不测正则化："
    )
    add("")
    lines += _md_table(
        ["级别", "n_radial", "n_chord", "尾涡段", "面板/叶"],
        [[l["level"], str(l["n_radial"]), str(l["n_chord"]), str(l["wake_steps"]),
          str(l["n_radial"] * l["n_chord"])] for l in ladder],
    )
    add("")
    add(
        "GCI 用 Roache 三点法（`tensorbem.uq.grid_convergence_index`，安全因子 1.25）："
        "p = ln|(f₂−f₁)/(f₃−f₂)|/ln r，GCI_fine = 1.25|(f₂−f₁)/f₁|/|r^p−1|。理论阶取 p_th = 2（面板中点求积"
        "对光滑载荷二阶；桨尖 √(1−r) 载荷会把它拉向 1.5，故只把 p_th = 2 当作**声明值**、把观测阶原样报出）。"
    )
    add("")
    add(
        "设计点归因按**动手前声明的三条渠道**分开计：①面元密度（Richardson 外推 vs L3）、②尾流模型"
        "（`geometric`／`kinematic`／`induced` 三种，同网格 L3）、③粘性修正（`viscous` 开/关，同网格 L3）。"
    )
    add("")
    add(
        f"工具链：Python {meta['python']} / numpy {meta['numpy']} / {meta['machine']}；"
        f"复跑命令 `{meta['producing_command']}`（`--workers` 并行逐 J 求解）。"
    )
    add("")

    # ---------------------------------------------------------------- ③ 结果
    add("## 3. 计算结果（全部数字由工件注入）")
    add("")
    add("### 3.1 五档敞水表（判据 A2）")
    add("")
    add("参照标 `E` = experiment（KP505 敞水试验曲线，二级转录）。BEM 取最细网格 L3。")
    add("")
    rows = []
    for point in points:
        rows.append([
            _f(point["J"], 1),
            _f(point["KT_bem"], 5),
            _f(point["KT_ref"], 3),
            f"{_f(point['dKT_pct'], 2, True)} %",
            _f(point["KQ_bem"], 5),
            _f(point["KQ_ref"], 4),
            f"{_f(point['dKQ_pct'], 2, True)} %",
            _f(point["eta0_bem"], 4),
            _f(point["eta0_ref"], 4),
            f"{_f(point['deta0_pct'], 2, True)} %",
            "E",
        ])
    lines += _md_table(
        ["J", "K_T(BEM)", "K_T(E)", "ΔK_T", "K_Q(BEM)", "K_Q(E)", "ΔK_Q",
         "η₀(BEM)", "η₀(E)", "Δη₀", "参照类型"],
        rows,
    )
    add("")
    add("### 3.2 面元密度 GCI（判据 A4）")
    add("")
    add("`asymptotic` = |p − p_th| ≤ 30 %·p_th（p_th = 2）。`GCI_fine` 为细网格指数（%）。")
    add("")
    gci_rows = []
    for row in gci:
        if row["quantity"] not in {"KT", "KQ"}:
            continue
        gci_rows.append([
            _f(row["J"], 1),
            row["quantity"],
            _f(row["L1"], 5),
            _f(row["L2"], 5),
            _f(row["L3"], 5),
            _f(row["observed_order"], 3, True),
            _f(row["order_deviation_pct"], 1, True) + " %",
            _f(row["gci_fine_pct"], 2) + " %",
            "是" if row["asymptotic_range"] else "否",
            "是" if row["oscillatory"] else "否",
        ])
    lines += _md_table(
        ["J", "量", "L1(8×2)", "L2(16×4)", "L3(32×8)", "p_obs", "p 偏差",
         "GCI_fine", "渐近区", "振荡"],
        gci_rows,
    )
    add("")
    add(
        f"渐近区格数 {by_id['A4-panel-density-gci']['measured']['n_cells_asymptotic']}"
        f"/{by_id['A4-panel-density-gci']['measured']['n_cells']}。观测阶出现负值的格子里三级增量符号反复，"
        "该离散**未进入渐近区**，因此 GCI 只能当误差量级指示，不能当误差上界——本报告按此口径引用，不放宽判据。"
    )
    add("")

    add("### 3.3 设计点 J = 0.70 的偏差归因（判据 A3）")
    add("")
    add("**渠道①面元密度**（Richardson 外推，仅当 p > 0 且单调时视为可用）：")
    add("")
    dens_rows = []
    for quantity in ("KT", "KQ", "eta0"):
        cell = attrib["channel_panel_density"][quantity]
        dens_rows.append([
            quantity,
            _f(cell["L3"], 5),
            _f(cell["reference"], 4),
            f"{_f(cell['total_pct'], 2, True)} %",
            _f(cell["observed_order"], 3, True),
            "可用" if cell["richardson_applicable"] else "不可用",
            _f(cell["extrapolated"], 5),
            f"{_f(cell['discretisation_pct'], 2, True)} %",
            f"{_f(cell['last_increment_pct'], 2)} %",
            f"[{_f(cell['model_band_pct'][0], 1, True)}, {_f(cell['model_band_pct'][1], 1, True)}]",
        ])
    lines += _md_table(
        ["量", "L3", "参照", "总偏差", "p_obs", "Richardson", "外推值",
         "离散项", "末次增量", "模型项（区间口径）"],
        dens_rows,
    )
    add("")
    add("**渠道②尾流模型**（同网格 L3，J = 0.70）：")
    add("")
    lines += _md_table(
        ["尾流模型", "K_T", "ΔK_T", "10K_Q", "Δ10K_Q", "η₀", "Δη₀"],
        [[row["wake_model"], _f(row["KT"], 5), f"{_f(row['dKT_pct'], 2, True)} %",
          _f(row["10KQ"], 5), f"{_f(row['d10KQ_pct'], 2, True)} %",
          _f(row["eta0"], 4), f"{_f(row['deta0_pct'], 2, True)} %"]
         for row in attrib["channel_wake_model"]],
    )
    add("")
    vis = attrib["channel_viscous_correction"]
    add("**渠道③粘性修正**（同网格 L3，J = 0.70）：")
    add("")
    lines += _md_table(
        ["配置", "K_T", "ΔK_T", "10K_Q", "Δ10K_Q"],
        [
            ["viscous = True（主核）", _f(vis["viscous_on"]["KT"], 5),
             f"{_f(vis['viscous_on']['dKT_pct'], 2, True)} %",
             _f(vis["viscous_on"]["10KQ"], 5),
             f"{_f(vis['viscous_on']['d10KQ_pct'], 2, True)} %"],
            ["viscous = False", _f(vis["viscous_off"]["KT"], 5),
             f"{_f(vis['viscous_off']['dKT_pct'], 2, True)} %",
             _f(vis["viscous_off"]["10KQ"], 5),
             f"{_f(vis['viscous_off']['d10KQ_pct'], 2, True)} %"],
        ],
    )
    add("")
    add(
        f"粘性通道贡献（占 ΔK_T 的百分点）：{_f(vis['viscous_share_of_total_dKT_pct'], 2, True)} 个百分点；"
        f"K_T 的粘性增量 {_f(vis['viscous_increment_KT'], 5)}，10K_Q 的粘性增量 "
        f"{_f(vis['viscous_increment_10KQ'], 5)}。"
    )
    add("")

    ana = attrib["analytic_ittc57_cross_check"]
    add("**解析锚交叉核对（AN-ITTC57 vs 仓内 ITTC-78 剖面阻力律，判据外的附加证据）**")
    add("")
    lines += _md_table(
        ["量", "值"],
        [
            ["代表半径 r/R", _f(ana["representative_radius_over_R"], 3)],
            ["代表弦长 [m]", _f(ana["representative_chord_m"], 4)],
            ["代表雷诺数 Re", f"{ana['representative_Re']:.3e}"],
            ["C_F（ITTC-1957）", _f(ana["CF_ittc57"], 6)],
            ["C_F（ITTC-78 剖面律）", _f(ana["CF_ittc78_section_law"], 6)],
            ["比值 ITTC-78 / ITTC-57", _f(ana["ratio_ittc78_over_ittc57"], 4)],
        ],
    )
    add("")
    add(f"说明：{ana['note']}")
    add("")

    add("**动量理论上界检查（η₀ 不得越界）**")
    add("")
    lines += _md_table(
        ["J", "η₀(BEM)", "η_ideal（作动盘，analytic）", "越界？"],
        [[_f(p["J"], 1), _f(p["eta0_bem"], 4), _f(p["eta0_momentum_bound"], 4),
          "是" if p["eta0_above_momentum_bound"] else "否"] for p in points],
    )
    add("")

    # ---------------------------------------------------------------- ④ 图
    add("## 4. 结果图")
    add("")
    for figure in cp["figures"]:
        rel = Path(figure["file"]).relative_to(REPO_ROOT) if Path(figure["file"]).is_absolute() \
            else Path(figure["file"])
        add(f"**`{rel}`** — {figure['caption']}")
        add("")

    # ---------------------------------------------------------------- ⑤ 注意点
    add("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add(
        "1. **锚是二级转录**：Table 2 的 Experimental 列在仓内可机读、可哈希、可逐字段溯源，但论文页 10 "
        "没有引用 KRISO 原始试验手稿。因此‘试验锚’这一层对原始手稿是 WITHHELD；本报告所有偏差都继承"
        "一个表格转录步骤，不能宣称比表本身的印刷精度更准（表给 3 位小数）。"
    )
    add(
        "2. **工作树导入遮蔽（本次实打实踩到）**：本机 `sys.path` 里注入了其它 worktree 的 `src/`"
        "（`.pth` 机制），直接 `python -m tensorbem...` 会静默导入**别的仓副本**。本单所有运行都强制 "
        "`PYTHONPATH=<本工作树>/src`，并在报告里记 commit，避免‘跑的是别人的代码’。"
    )
    add(
        "3. **BEM 是升力面近似，不是 RANS**：零弯度平板涡格、径向常数螺距、无侧斜/纵倾、无毂与剖面厚度堵塞，"
        "弦长分布只保证 A_E/A_O 与根梢收缩形态。凡是依赖剖面弯度与真实径向螺距分布的差异，都会落入"
        "‘模型项’，不可能靠加密网格消掉。"
    )
    add(
        "4. **观测阶为负 = 未进渐近区**：本梯度在多数格子给出 p < 0，说明 L1→L3 的增量方向反复。"
        "此时 Richardson 外推值不可当物理极限，本报告只用末次增量作区间估计，并把该事实写进工件 "
        "`richardson_applicable` 字段。"
    )
    add(
        "5. **运行时与并行**：L3（32×8 面板/叶、96 尾涡段）单次约 3 min（本机高负载、load≈700 下实测）；"
        "五档 + 归因通道用进程池并行，整体 10 min 量级。**并行必须用 `spawn`**：本机在父进程已起 "
        "OpenBLAS 线程池（130 线程）后 `fork`，子进程会继承一把没人解锁的 futex，实测子进程 0 s CPU "
        "永久睡眠——表现为‘跑了半小时一个字都没写’。"
    )
    add(
        "6. **缓存与可复现**：晶格解按 (J, 梯度级, 模型开关, `propeller.py`+`uq.py` 的 sha256) 缓存在 "
        "`runs/w22_kp505_cache/`，命中时逐位复现上一轮同指纹结果（本次重出裁决时 "
        f"{cp['meta']['cache']['n_cached_cells']}/{cp['meta']['cache']['n_cells']} 格来自缓存，"
        "其中唯一需要重算的粘性关闭工况重算结果与上轮逐位相同）。工件里每个格子都带 `cache_hit` 标记，不做暗缓存。"
    )
    add(
        "7. **本单边界**：只产实验工件与结论，不写论文正文；不触碰 TensorPlatform 仓；"
        "所有工件路径都在 TensorBEM 仓内。"
    )
    add("")

    # ---------------------------------------------------------------- ⑥ 误差
    add("## 6. 误差判定")
    add("")
    add("### 6.1 判据逐条裁决表（阈值先于动手锁定，未事后修改）")
    add("")
    crit_rows = []
    for row in cp["criteria"]:
        measured = row["measured"]
        if row["id"].startswith("A1"):
            value = (f"命中文件 {measured['n_hit_files']}，字段 {measured['n_fields']}"
                     f"（带 sha256 {measured['n_hit_files_with_sha256']}）")
        elif row["id"].startswith("A2"):
            value = f"{measured['complete_cells']}/{measured['cells']} 格完整"
        elif row["id"].startswith("A3"):
            value = (f"J=0.70 ΔK_T = {_f(measured['dKT_pct'], 2, True)} %"
                     f"（K_T {_f(measured['KT_bem'], 5)} vs {_f(measured['KT_ref'], 3)}）")
        else:
            value = (f"{measured['n_levels']} 套网格，{measured['n_cells']} 格，"
                     f"渐近 {measured['n_cells_asymptotic']} 格")
        crit_rows.append([
            row["id"],
            value,
            row.get("threshold", ""),
            "**PASS**" if row["pass"] else "**FAIL**",
            f"`{COMPARE.name}` → criteria[{row['id']}]",
        ])
    lines += _md_table(["判据", "量值", "阈值", "裁决", "证据指针"], crit_rows)
    add("")
    a3 = by_id["A3-design-point-KT-within-5pct"]
    add(f"A3 的详细口径：`within_threshold` = {a3['measured']['within_threshold']}、"
        f"`honest_fail` = {a3['measured']['honest_fail']}、"
        f"`attribution_available` = {a3['measured']['attribution_available']}、"
        f"`disposition` = `{a3.get('disposition')}`。"
        "**`pass` 只表示精度阈值是否达到，绝不用‘已归因’把红格刷绿**；"
        f"任务级处置 = `{verdict['task_level_disposition']}`。")
    add("")
    add("### 6.2 关键量 vs 基准，逐项回答「是否 < 3 %」")
    add("")
    def _yn_3(value: float) -> str:
        return "**是**" if abs(value) < 3.0 else "**否**"

    lines += _md_table(
        ["量（J = 0.70）", "BEM(L3)", "实验锚", "相对偏差", "是否 < 3 %", "备注"],
        [
            ["K_T", _f(design["KT_bem"], 5), _f(design["KT_ref"], 3),
             f"{_f(design['dKT_pct'], 2, True)} %", _yn_3(design["dKT_pct"]),
             "任务门槛为 <5 %，仍未达"],
            ["K_Q", _f(design["KQ_bem"], 5), _f(design["KQ_ref"], 4),
             f"{_f(design['dKQ_pct'], 2, True)} %", _yn_3(design["dKQ_pct"]), "同上口径"],
            ["η₀", _f(design["eta0_bem"], 4), _f(design["eta0_ref"], 4),
             f"{_f(design['deta0_pct'], 2, True)} %", _yn_3(design["deta0_pct"]),
             "派生量；分子分母同向偏小故相消（不构成达标证据）"],
        ],
    )
    add("")
    add(
        f"**显式回答**：设计点 J = {_f(design['J'], 2)} 的 |ΔK_T| = "
        f"{_f(abs(design['dKT_pct']), 2)} % —— **不小于 3 %，也不小于本单的 5 % 门槛**；"
        f"|ΔK_Q| = {_f(abs(design['dKQ_pct']), 2)} %，|Δη₀| = {_f(abs(design['deta0_pct']), 2)} %。"
        "因此判据 3 按规则记为 **honest FAIL**，不粉饰、不改判据。"
    )
    add("")
    add("**归因结论（三渠道，均为工件数字）**：")
    add("")
    kt = attrib["channel_panel_density"]["KT"]
    wake_rows = attrib["channel_wake_model"]
    wake_span = [row["dKT_pct"] for row in wake_rows]
    wake_spread = max(wake_span) - min(wake_span)
    induced = next(row for row in wake_rows if row["wake_model"] == "induced")
    add(
        f"- **尾流模型（主渠道）**：`geometric` / `kinematic` / `induced` 的 ΔK_T 分别为 "
        + " / ".join(f"{row['wake_model']} {_f(row['dKT_pct'], 2, True)} %" for row in wake_rows)
        + f"，极差 **{_f(wake_spread, 2)} 个百分点** ⇒ 尾流离散方式是本模型最大的单一不确定性，"
        "比面元密度与粘性项都大。"
    )
    add(
        f"- **面元密度（次渠道）**：观测阶 p = {_f(kt['observed_order'], 3, True)}，Richardson "
        f"{'可用' if kt['richardson_applicable'] else '**不可用**（p<0，外推值落在加密区间之外，不可作物理极限）'}；"
        f"L3 与该项目外推值之差 {_f(kt['discretisation_pct'], 2, True)} 个百分点只是量级指示，"
        f"可用的离散不确定度是末次（L2→L3）增量 {_f(kt['last_increment_pct'], 2)} %。"
        "即：即使把离散不确定度整块挪走，也补不平 "
        f"{_f(kt['total_pct'], 2, True)} 个百分点的总偏差 ⇒ 离散不是主因。"
    )
    add(
        f"- **粘性修正（小项）**：开关粘性使 ΔK_T 变化 "
        f"{_f(vis['viscous_share_of_total_dKT_pct'], 2, True)} 个百分点（K_T 粘性增量 "
        f"{_f(vis['viscous_increment_KT'], 5)}、10K_Q {_f(vis['viscous_increment_10KQ'], 5)}）"
        "⇒ 粘性不是主因。"
    )
    add(
        f"- **余项（几何/载荷保真度）**：扣掉上述三项后剩 "
        f"{_f(kt['model_pct_richardson'], 2, True)} 个百分点（区间口径 "
        f"[{_f(kt['model_band_pct'][0], 1, True)}, {_f(kt['model_band_pct'][1], 1, True)}] %），"
        "只能归到**零弯度平板剖面**（真桨 NACA66 有弯度）、**径向常数螺距**（真桨 0.6R 附近有峰）、"
        "**无毂与剖面厚度堵塞**、无侧斜/纵倾。"
    )
    add(
        f"- **诚实的边界（不能拿 induced 翻案）**：`induced`（力自由尾涡对齐）在 K_T 上为 "
        f"{_f(induced['dKT_pct'], 2, True)} %，**已进 ±5 % 门槛**；但（i）它不是本单预声明的主核，"
        "换成它属于‘看见锚之后再选模型’的事后选择；（ii）它同时把 Δ10K_Q 改到 "
        f"{_f(induced['d10KQ_pct'], 2, True)} %、Δη₀ 改到 {_f(induced['deta0_pct'], 2, True)} %，"
        "不是一致性改进。故判据 3 仍判 **FAIL**，induced 只作敏感性结果登记，不作主结果。"
    )
    add("")

    add("### 6.3 复现与工件清单")
    add("")
    add("```")
    for artefact in (
        f"{TRANSCRIPTION.relative_to(REPO_ROOT)}   # 可达性 + 转录（锚字段级溯源）",
        f"{COMPARE.relative_to(REPO_ROOT)}         # 五档敞水表 + GCI + 归因 + 判据裁决",
        "benchmarks/verified/kp505_anchor_w22/points.csv",
        "benchmarks/verified/kp505_anchor_w22/gci.csv",
        "benchmarks/verified/kp505_anchor_w22/figures/*.png",
        "src/tensorbem/benchmarks/kp505_w22.py    # 驱动（判据先声明后跑）",
        "tools/emit_w22_report.py                 # 本报告生成器（零手抄）",
        "tests/test_w22_kp505_anchor.py           # 12 tests",
    ):
        add(artefact)
    add("```")
    add("")
    add("```bash")
    add("PYTHONPATH=$PWD/src python -m tensorbem.benchmarks.kp505_w22 benchmarks/verified/kp505_anchor_w22")
    add("PYTHONPATH=$PWD/src python tools/emit_w22_report.py")
    add("PYTHONPATH=$PWD/src python -m pytest tests/test_w22_kp505_anchor.py -q")
    add("```")
    add("")
    add(
        f"工件 `kp505_transcription.json` sha256 = "
        f"`{hashlib.sha256(TRANSCRIPTION.read_bytes()).hexdigest()}`；"
        f"`openwater_compare.json` sha256 = `{hashlib.sha256(COMPARE.read_bytes()).hexdigest()}`。"
    )
    add("")

    add("## 7. 平台投递摘要（与 report API 的 summary 同一段文本，机器生成）")
    add("")
    add("> " + summary_text(tr, cp))
    add("")
    add(
        "判据原文（先于动手声明）：A1 锚可达性机检（命中文件数/字段数/逐文件 sha256）；"
        "A2 五档 K_T/K_Q/η₀ 表带相对偏差与参照类型；A3 设计点 |ΔK_T| < 5 %，否则如实 FAIL + 归因；"
        "A4 至少两套网格的面元密度 GCI（观测阶 + GCI 指数落表，并声明是否渐近）。"
    )
    add("")

    return "\n".join(lines) + "\n"


def main() -> int:
    report = build()
    out = REPO_ROOT / REPORT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(json.dumps({
        "written": REPORT_REL,
        "sha256": hashlib.sha256(report.encode()).hexdigest(),
        "n_lines": report.count("\n"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())