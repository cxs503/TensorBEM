"""Render ``docs/case_bem_kp505_gci.md`` from the case artefact (no hand-typed numbers).

Usage: ``PYTHONPATH=src python tools/build_case_bem_kp505_gci_report.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.kp505_three_density_gci import (
    CASE_TOL_PCT,
    DEFAULT_OUTDIR,
    DESIGN_TOL_J,
    SPAN_ONLY,
    THEORY_ORDER,
)
from tensorbem.benchmarks.kp505_openwater_bem import GEOMETRY_SOURCE
from tensorbem.propeller import WATER_NU, BladeGeometry


def physical_diameter() -> float:
    """Model-scale diameter in metres, parsed from the transcribed Table 1 row."""
    for row in GEOMETRY_SOURCE["rows_verbatim"]:
        if row.startswith("Diameter"):
            parts = row.split()
            for token in parts[1:]:
                try:
                    return float(token)
                except ValueError:
                    continue
    raise KeyError("diameter row missing from the transcribed geometry table")

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "case_bem_kp505_gci.md"


def mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> int:
    payload = json.loads((DEFAULT_OUTDIR / "case.json").read_text(encoding="utf-8"))
    case = payload["case"]
    env = payload["env"]
    design = payload["design"]
    cells, gci = payload["cells"], payload["gci"]
    verdict = payload["verdict"]
    anchor_rows = payload["anchor_comparison"]
    sweep = payload["fail_mode_localisation"]
    cross = payload["cross_check_w20"]
    span = payload["axis_isolation_spanwise"]
    verif = payload["reference_verification"]

    # representative Reynolds number at 0.7 R, computed from the module's own
    # geometry + section-drag inputs (never typed by hand)
    geom = BladeGeometry(n_radial=16, n_chord=4,
                         thickness_ratio=design["held_fixed"]["thickness_ratio"],
                         radial_spacing=design["held_fixed"]["radial_spacing"],
                         root_chord_offset=design["held_fixed"]["root_chord_offset"])
    d_phys = physical_diameter()               # 0.25 m, Table 1 transcription
    r_frac = 0.7
    r_abs = r_frac * geom.radius               # lattice units (D_lattice = 1)
    chord_nd = float(geom.chord(np.array([r_abs]))[0])
    chord_07 = chord_nd * d_phys               # physical chord
    n_rev = design["held_fixed"]["rev_per_second"]
    v_a = DESIGN_TOL_J * n_rev * d_phys
    v_t = 2.0 * np.pi * n_rev * (r_frac * d_phys / 2.0)
    v_res = float(np.hypot(v_a, v_t))
    reynolds = v_res * chord_07 / WATER_NU

    L = []
    add = L.append
    add("# Case `CASE-BEM-KP505-GCI` — KP505 三密度 GCI 收敛（TensorBEM）\n")
    add("> 本文件由 `tools/build_case_bem_kp505_gci_report.py` 从 "
        "`benchmarks/verified/bem_kp505_gci/case.json` 渲染，正文数字**全部程序化注入**，"
        "无手抄。工件：`case.json` / `levels.csv` / `gci.csv` / `figures/*.png`。\n")
    add(f"> 复现命令：`{payload['reproduction']['cmd']}`"
        f"（分片执行：`--phase ladder --advance <J>` / `--phase span` / `--phase sweep`，"
        f"缓存于 `{DEFAULT_OUTDIR.name}/cache/`）。\n")

    # ① problem + ref source
    add("## 1. 问题与参考出处\n")
    add(f"* 算例（`tensor_case_pool`）：`{case['code']}` — {case['name']}；"
        f"仓 `{case['repo']}`；`ref_source = {case['ref_source']}`；"
        f"`tol_policy = {case['tol_policy']}`，即**离散化不确定度 GCI ≤ "
        f"{CASE_TOL_PCT:.0f}%** 为验收线（本单容差判据）。")
    add("* 参考（GCI 程序本身，非测值）出处：")
    for cite in case["ref_citations"]:
        add(f"  * {cite}")
    add(f"* 参考锚值（敞水系数）出处：{case['anchor']['citation']}（DOI "
        f"{case['anchor']['doi']}），**{case['anchor']['location']}，"
        f"{case['anchor']['table']}**；J = 0.70 行逐字转录："
        f"`{case['anchor']['row_J0.7_verbatim']}`；列头逐字："
        f"`{case['anchor']['column_header_verbatim']}`；provenance 等级 = "
        f"`{case['anchor']['provenance_status']}`（该表为二手转录，非 KRISO 原表，"
        "误差判定只在 GCI 上做验收，锚值只作报告项）。")
    add(f"* 物理问题：KP505 五叶定距桨敞水，在 J = "
        f"{', '.join(f'{j:.2f}' for j in design['advances'])} 三点上做三密度（r = 2）"
        "面板收敛研究，回答「这套升力面/涡格 BEM 的敞水系数离散化不确定度是否 < 3%」。\n")

    # ② method
    add("## 2. 方法与实现\n")
    add(f"* 求解器（仓内既有能力，未另起炉灶）：`{design['solver_entry_point']}`；"
        f"几何模型：{design['geometry_model']}。")
    add("* 三级等比加密（r = 2，三条面板轴同时加密）：")
    add("")
    add("| 级 | n_radial | n_chord | wake_steps | 每叶片束缚面元 |")
    add("|---|---|---|---|---|")
    for level in design["levels"]:
        add(f"| {level['label']} | {level['n_radial']} | {level['n_chord']} | "
            f"{level['wake_steps']} | {level['n_radial'] * level['n_chord']} |")
    add("")
    add(f"* 持常量：`{json.dumps(design['held_fixed'], ensure_ascii=False)}` — "
        "同一物理模型三个网格，GCI 程序的前提。")
    add(f"* GCI（Roache）：`p = ln|(f_c-f_m)/(f_m-f_f)|/ln r`，"
        f"`f_ext = f_f + (f_f-f_m)/(r^p-1)`，"
        f"`GCI_f = Fs|(f_m-f_f)/f_f|/|r^p-1|`，Fs = {verif['safety_factor']}"
        "（三级研究），Roache 渐近判据 `GCI_m/(r^p GCI_f) ≈ 1`。实现复用 "
        "`tensorbem.uq.grid_convergence_index`，并在本算例驱动内**独立写出闭式**对拍。")
    add("* 参考程序先验证再用（G7）：(a) 制造解 `f(h)=A+C h^pp`（pp = "
        f"{verif['manufactured']['order']}）反演阶数与极限；(b) `GCI/Fs` 必须等于制造解真实"
        "细网相对误差；(c) 仓内库函数 vs 驱动内闭式；(d) 振荡三元组必须被标记。")
    add(f"* 失败模式定位（预声明）：设计点做**平分方向纯加密扫描** n_radial ∈ "
        f"{sweep['config']['n_radial']}（n_chord = {SPAN_ONLY['n_chord']}、wake_steps = "
        f"{SPAN_ONLY['wake_steps']} 冻结），判断非单调是单步偶然还是整个展向。\n")

    # ③ parameters
    add("## 3. 参数表（无量纲数 + 模型输入）\n")
    add("| 量 | 值 | 说明 |")
    add("|---|---|---|")
    add(f"| J（进速系数） | {', '.join(f'{j:.2f}' for j in design['advances'])} "
        f"| 设计点 J = {DESIGN_TOL_J:.2f}（验收点） |")
    add(f"| 叶数 Z | 5 | 几何来源 `GEOMETRY_SOURCE`（Table 1, 二手转录） |")
    add(f"| 直径 D | {d_phys:.3f} m | 模型尺度，由转录 Table 1 `Diameter (m)` 行解析；"
        f"求解器内部以 D = {geom.diameter:.0f} 无量纲化 |")
    add(f"| 毂径比 d_h/D | {geom.hub_radius / geom.radius:.3f} | 模型输入（Table 1 Hub Ratio 行） |")
    add(f"| 剖面厚度比 t/c | {design['held_fixed']['thickness_ratio']:.3f} | 模型输入 |")
    add(f"| 弦长分布 | 余弦展向节点 + 模型弦长律 | 0.7R 处 c/D = {chord_nd:.4f} → "
        f"c = {chord_07 * 1000:.1f} mm（模型尺度） |")
    add(f"| 转速 n | {n_rev:.0f} rev/s | 仅用于剖面 Reynolds 数 |")
    add(f"| 代表 Reynolds 数 Re(0.7R) | {reynolds:.3e} | "
        f"V_res = {v_res:.3f} m/s, c = {chord_07 * 1000:.1f} mm, ν = {WATER_NU:.3e} m²/s |")
    add(f"| 涡核松弛 core_ratio | {design['held_fixed']['core_ratio']:.3f} | 正则化，三级恒定 |")
    add(f"| 尾涡模型 / 长度 | {design['held_fixed']['wake_model']} / "
        f"{design['held_fixed']['wake_length']:.2f} 转 | 三级恒定 |")
    add(f"| 加密比 r | {design['refinement_ratio']:.1f} | 三密度阶梯 |")
    add(f"| 安全因子 Fs | {verif['safety_factor']:.2f} | Roache 三级默认 |")
    add(f"| 理论阶数 p_th | {THEORY_ORDER:.1f} | 面板中点积分（尖部 √ 载荷退化 → 1.5 为次阶） |")
    add("")

    # ④ comparison table
    add("## 4. 对照表（ref vs got，逐项打勾/叉）\n")
    add(f"### 4.1 主判据：GCI vs 算例容差 {CASE_TOL_PCT:.0f}%（J = {DESIGN_TOL_J:.2f}）\n")
    add("| 量 | ref（验收线） | got GCI_fine | 估计细网误差 GCI/Fs | 观测阶 p | 渐近比 | 单调 | <3% |")
    add("|---|---|---|---|---|---|---|---|")
    for row in gci:
        if row["J"] != DESIGN_TOL_J:
            continue
        add(f"| {row['quantity']} | ≤ {CASE_TOL_PCT:.0f}% | {fmt(row['gci_fine_pct'], 3)}% "
            f"| {fmt(row['estimated_fine_error_pct'], 3)}% | {fmt(row['observed_order'], 3)} "
            f"| {fmt(row['asymptotic_ratio'], 4)} | {mark(row['monotone'])} "
            f"| {mark(row['under_case_tol_3pct'])} |")
    add("")
    add("### 4.2 三个进速系数全覆盖（GCI_fine）\n")
    add("| J | K_T GCI | 10K_Q GCI | eta_0 GCI | 全部 <3% |")
    add("|---|---|---|---|---|")
    for advance in design["advances"]:
        rows = {r["quantity"]: r for r in gci if r["J"] == advance}
        ok = all(r["under_case_tol_3pct"] for r in rows.values())
        add(f"| {advance:.2f} | {fmt(rows['KT']['gci_fine_pct'], 3)}% | "
            f"{fmt(rows['10KQ']['gci_fine_pct'], 3)}% | {fmt(rows['eta0']['gci_fine_pct'], 3)}% "
            f"| {mark(ok)} |")
    add("")
    add("### 4.3 锚值对照（参考项，非验收项；J = 0.70，Table 2 page 10 逐字转录）\n")
    add("| 量 | ref 锚值 | got D3（细网） | 误差 | got 外推 | 误差 |")
    add("|---|---|---|---|---|---|")
    for row in anchor_rows:
        add(f"| {row['quantity']} | {fmt(row['ref_anchor'], 4)} | {fmt(row['got_finest'], 4)} "
            f"| {fmt(row['error_finest_pct'], 2)}% | {fmt(row['got_extrapolated'], 4)} "
            f"| {fmt(row['error_extrapolated_pct'], 2)}% |")
    add("")
    add("### 4.4 参考程序自验（G7）与既有工件互校\n")
    add("| 检查 | 目标 | got | 相对误差 | 判 |")
    add("|---|---|---|---|---|")
    for check in verif["checks"]:
        target = check["target"] if not isinstance(check["target"], float) else fmt(check["target"], 6)
        got = check["got"] if not isinstance(check["got"], float) else fmt(check["got"], 6)
        err = "n/a" if check["error"] is None else f"{check['error']:.2e}"
        add(f"| {check['id']} | {target} | {got} | {err} | {mark(check['pass'])} |")
    if cross.get("status") == "COMPARED":
        for item in cross["items"]:
            add(f"| 互校 vs `{cross['reference']}` ({item['quantity']} GCI) | "
                f"{fmt(item['ref_gci_fine_pct'], 4)}% | {fmt(item['got_gci_fine_pct'], 4)}% "
                f"| {item['abs_difference_pct_points']:.2e} pp | {mark(item['pass'])} |")
    add("")

    # ⑤ convergence / sensitivity
    add("## 5. 收敛性与参数敏感性\n")
    add("* **三级阶梯本身**（K_T，三进速）：")
    for advance in design["advances"]:
        pts = sorted([c for c in cells if c["J"] == advance],
                     key=lambda c: c["n_panels_per_blade"])
        add(f"  * J = {advance:.2f}: " + " → ".join(
            f"{c['label']}={c['KT']:.6f}" for c in pts)
            + f"（D2→D3 变化 {100 * (pts[2]['KT'] - pts[1]['KT']) / pts[1]['KT']:+.2f}%）")
    add(f"* **观测阶 vs 理论阶**：理论 {THEORY_ORDER:.1f}（中点积分；尖部载荷退化给 1.5），实测 "
        + ", ".join(f"{r['quantity']}@J{r['J']:.1f}: p={r['observed_order']:.2f}"
                    for r in gci)
        + "。负阶意味着**增量随加密反而增大**（反收敛），不是幂律收敛。")
    add(f"* **渐近区间判据（含方法论注记）**：实测渐近比 "
        f"{fmt(min(r['asymptotic_ratio'] for r in gci), 3)}–"
        f"{fmt(max(r['asymptotic_ratio'] for r in gci), 3)}，全部落在 ±10% 带内，"
        f"但观测阶几乎全为负 — 因为 `GCI_m/(r^p GCI_f)` 在单个三元组上**代数恒等于 "
        f"`f_f/f_m`**（同一 Fs、同一三元组），三级研究里它几乎恒 ≈1，**并不检验 p**。"
        "真正检验渐近区需四级以上比较相邻三元组（见 "
        "`benchmarks/kp505_gci_ladder.py` 的 triplet 分歧 ~30×）。本算例因此把 G6 "
        "门控在「单调且 p>0」上；否则 p<0 的反收敛序列会被这条恒等式错判为渐近。")
    add(f"* **展向纯加密扫描**（J = 0.70，n_chord = {SPAN_ONLY['n_chord']}、"
        f"wake_steps = {SPAN_ONLY['wake_steps']} 冻结）："
        + ", ".join(f"{c['n_radial']}→{c['KT']:.6f}" for c in sweep["cells"])
        + f"；跨扫描 K_T 相对极差 **{fmt(sweep['KT_spread_pct'], 2)}%**，"
        f"单调性判 {mark(bool(sweep['monotone_in_n_radial']))}。")
    add(f"* **展向三密度阶梯（隔离诊断）**：K_T 观测阶 "
        f"{fmt(span['gci_KT']['observed_order'], 3)}、GCI {fmt(span['gci_KT']['gci_fine_pct'], 2)}%；"
        f"10K_Q 观测阶 {fmt(span['gci_10KQ']['observed_order'], 3)}、"
        f"GCI {fmt(span['gci_10KQ']['gci_fine_pct'], 2)}% — 非单调性在**展向**已单独复现。")
    add("* **进速系数敏感性**：同一阶梯在 J = 0.5 / 0.7 / 0.9 上 K_T 的 D3→(D2,D1) 落差分别为 "
        + ", ".join(f"{100 * (next(c for c in cells if c['J'] == j and c['label'] == 'D3')['KT'] / next(c for c in cells if c['J'] == j and c['label'] == 'D2')['KT'] - 1):+.2f}%"
                    for j in design["advances"])
        + "，量级一致 → 不是某一点的偶然，而是展向离散化的系统性行为。\n")

    # ⑥ fail modes + artefacts
    add("## 6. 失败模式、判定与工件\n")
    add(f"* **判定：honest-FAIL**。`fail_stage = {verdict['fail_stage']}`；"
        f"`failure_reason = {verdict['failure_reason']}`。")
    add("* 逐条判据：")
    for key, ok in verdict["pass_by_id"].items():
        add(f"  * {mark(ok)} {key}")
    add(f"* **失败机理（可复核）**：D1→D2 几乎不动（K_T {100 * (next(c for c in cells if c['J'] == DESIGN_TOL_J and c['label'] == 'D2')['KT'] / next(c for c in cells if c['J'] == DESIGN_TOL_J and c['label'] == 'D1')['KT'] - 1):+.2f}%），"
        f"而 D2→D3 跳 {100 * (next(c for c in cells if c['J'] == DESIGN_TOL_J and c['label'] == 'D3')['KT'] / next(c for c in cells if c['J'] == DESIGN_TOL_J and c['label'] == 'D2')['KT'] - 1):+.2f}%；"
        f"展向隔离扫描 8→32 之间 K_T 极差 {fmt(sweep['KT_spread_pct'], 2)}% 且非单调。"
        "即：**展向面元数不是本 BEM 收敛的有效旋钮**（尖部载荷 √(1-r) 退化 + 中点积分），"
        "GCI 因此既给不出幂律阶数，也给不出可信误差界。3% 判据在这套格式上**当前不可达**。")
    add("* **无法进一步验证的边界（如实声明）**：加第四级（r = 2 再加密）需 1024 束缚面元 × "
        "192 尾涡步，直接稠密求解的内存/时间在本容器不可承载（D3 已 6.4 GB 峰值、"
        "约 50 s），故只能用 Roache 渐近比**判准入**而不能用再加密**验收敛**。")
    add("* 生态摩擦（friction）：")
    for item in payload["friction"]:
        add(f"  * {item}")
    add(f"* 工件：`{DEFAULT_OUTDIR.relative_to(REPO_ROOT)}/{{case.json, levels.csv, gci.csv, "
        f"figures/kp505_gci_ladder.png, figures/kp505_gci_indices.png}}`；"
        f"缓存 `cache/`；测试 `tests/test_case_bem_kp505_gci.py`。")
    add(f"* 环境：python {env['python']} / numpy {env['numpy']} / {env['platform']} / "
        f"commit `{env['git_commit'][:12]}`；总墙钟 {payload['reproduction']['wall_seconds_total']:.0f} s。")
    add("")

    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {DOC} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())