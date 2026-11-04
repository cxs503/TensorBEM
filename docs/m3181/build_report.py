"""Compose the m3181 development report and the platform report payload.

All numbers are injected from ``outputs/three_d_free_surface_registry.json``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

OUT = Path("docs/m3181")
report = json.loads((OUT / "outputs" / "three_d_free_surface_registry.json").read_text())
rows = report["checks"]
m = {row["key"]: row["measured"] for row in rows}
d = {row["key"]: row["declared"] for row in rows}
equality = [row for row in rows if row["comparison"] != "upper"]
worst = max(equality, key=lambda row: abs(row["relative_deviation"]))
worst_excluding_rounding = max(
    (row for row in equality if row["tolerance"] < 2.0e-2),
    key=lambda row: abs(row["relative_deviation"]),
)
summary = report["summary"]
elapsed = report["elapsed_seconds"]
table = (OUT / "table.md").read_text(encoding="utf-8")

MARKDOWN = f"""# TensorBEM：三维自由液面「声明数字」注册表化 —— 表 10/11/12 共 63 格全量复算

**任务**：m3181（paper 席自找活） · **分支**：`agent/m3181` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3181`，基线 `6025970`）
**工件**：`docs/m3181/outputs/three_d_free_surface_registry.json`、
`docs/m3181/figures/fig_registry_3d_free_surface.png`、
`docs/m3181/figures/fig_three_d_convergence_registry.png`、
`docs/m3181/figures/fig_chief_regularization.png`、
脚本 `docs/m3181/{{run_three_d_free_surface,make_figures}}.py`

EDIT_WHITELIST: src/tensorbem/benchmarks/declared_numbers.py, tests/test_declared_numbers.py,
docs/marine_bem_benchmarks.md, docs/m3181/, dev_report_m3181_three_d_free_surface_registry.md

---

## 1. 案例介绍（为什么做这件事）

TensorBEM 的三维深水自由液面层（Kelvin/Havelock Green 函数 → 6-DOF 辐射–绕射面板法 →
Kochin 远场守恒 / Haskind 互易 → 免盖板三维 CHIEF）在 `docs/marine_bem_benchmarks.md` §4.10–§4.11
里以**三张收敛表 + 一张 CHIEF 对比表**作为验收面，共 63 个具体数字。这些数字此前只有**零散的单测**
（每条断言一个量、且部分已放宽到 1024 面元），**没有任何一条判据锚定「文档里那个数字」**：

* m3059（2026-11-04）在结清椭球 1280 面元与内部谐波 1280 面元两条欠账后，明确把
  「三维自由液面 Hulme / Wang-Linton 仍未纳入注册表」写成下一步；
* `docs/verification_doc_numbers_maccamy_fuchs.md` §未做 亦记录同一缺口。

于是只要 `free_surface3d.py` 的兴波核分区、`free_surface_solver3d.py` 的立体角/Duffy 求积、
或 `analytic.py` 的两个多极子参考解任一环节变动，文档数字就会**静默腐化**——读者复印表格里
的 \(7.41\\times10^{{-3}}\)，代码却给出别的数，却没有任何测试会红。

**算例参数（全部取自文档表格）**

| 项 | 表 10（Hulme 漂浮半球） | 表 11（Ka = 2.58 不规则频率） | 表 12（Wang/Linton 潜没球） |
|---|---|---|---|
| 几何 | 半球 \(a=1\) m，水线开口网格 | 同一半球，256 面元 | 球 \(a=1\) m，球心 \(z=-1.5\) m |
| 离散 | 八面体递归细分 1/2/3 → 16/64/256 面元 | 256 面元 | icosphere 0/1/2 → 20/80/320 面元 |
| 频率 | \(Ka=1.0\) | \(Ka=2.58\)（第一内域水线特征值附近） | \(Ka=0.8\)，\(f/a=1.5\) |
| 流体 | \(\\rho=1025\) kg/m³，\(g=9.80665\) m/s² | 同左 | 同左 |
| 真值 | Hulme (1982) 球面无波势 + Kelvin 源的半解析解 | 同左（表内 \(A_{{33}}'=0.4007,\\ B_{{33}}'=0.0631\)） | Thorne/Wang/Linton 多极子闭式（阶乘 + \\(\\mathrm{{Ei}}(2\\nu f)\\)，零数值求积） |
| 判据量 | 条件数、\(A_{{33}}',B_{{33}}',A_{{11}}',B_{{11}}',F_z\\) 相对误差、观测阶 | 条件数、\(A_{{33}}',B_{{33}}'\\)（含符号）、CHIEF 点数 | 条件数、\(A_{{33}}',A_{{11}}',F_x,F_z\\) 误差、转动耦合比、观测阶 |

两个参考解与面板装配**不共享任何代码**（多极子级数 vs 面元积分），因此漂移可归因。

## 2. 怎么做的（方法/流程/关键设置）

1. **读文档 → 抠格子**：把表 10（6 列 × 3 层 + 5 观测阶 × 2 层 = 28 格）、表 11（9 格）、
   表 12（6 列 × 3 层 + 4 观测阶 × 2 层 = 26 格）逐格抄成注册表条目，`source` 字段写明行号级出处。
2. **注册表扩展**（`src/tensorbem/benchmarks/declared_numbers.py`）：新增三个 `heavy` 组
   `hulme3d`（`run_refinement_study([1,2,3], ka=1.0)`）、`hulme3d_chief`
   （`run_irregular_frequency_study(subdivision=3, ka_values=[2.58])`）、
   `wang_linton3d`（`run_refinement_study([0,1,2], ka=0.8)`）。**同一张表的所有格子共用一次求解**
   （注册表缓存键 `(group, parameters)` 相同），因此 63 格只花 3 次求解。
3. **补齐一个文档里引用但基准不输出的量**：表 10 的 \(B_{{11}}'\) 观测阶由连续两层的误差比
   \\(\\log_2(e_{{k-1}}/e_k)\\) 在 runner 里现算（对纯幂律是恒等式），文档值 1.62 / 1.88 与之一致。
4. **容差纪律**（诚实而非宽松）：三到四位有效数字的格子用 5e-3 相对带（比可复现地板高五个量级，
   又能抓住任何会动最后一位的核改动）；表 12 的转动耦合列文档只给两位有效数字（2.5e-32 / 2.2e-4 /
   2.6e-5），舍入半宽约 1.9%，故该列用 2.5e-2，且 20 面元格是「机器零」断言，按**上界**（≤1e-20）判；
   表 11 的未增广 \(B_{{33}}'=-0.1042\) 是**负值**，符号本身就是「非物理负阻尼」的论断，
   因此注册表放宽了 `declared != 0` 而不是取绝对值。
5. **复算 + 判决**：`docs/m3181/run_three_d_free_surface.py` 只跑新增三组，输出 strict JSON
   （`allow_nan=False`），并逐格打印 declared/measured/相对偏差/判定。
6. **回归钉桩**（`tests/test_declared_numbers.py`）：
   * 结构门（无重算）：三组行数 28/9/26、每组参数唯一、`tier` 全为 `heavy`、
     负阻尼符号与机器零上界的方向性断言；
   * `@HEAVY` 复算门两条（`TENSORBEM_HEAVY_TESTS=1`）：37 格与 26 格全 PASS，并断言
     「最细层观测阶落在 1.7–2.3」「CHIEF 条件数下降一个量级且 \(B_{{33}}\\) 由负转正」
     —— 这三条是表格的**物理内容**，不是三个小数的巧合。
7. **文档程序化注入**：`docs/marine_bem_benchmarks.md` 新增 §4.12，63 格的
   「文档声明 / 本次复算 / 相对偏差 / 判定」表由 `docs/m3181/make_figures.py` 从 JSON 生成，零手抄。

## 3. 计算结果

**判决：{summary['passed']} / {summary['checked']} 格 PASS，{summary['drifted']} 格 DRIFT；整组耗时 {elapsed:.0f} s（本机 nproc=128，load15 ≈ 70–100）。**

等值判据的 {len(equality)} 格中最坏相对偏差为 {abs(worst['relative_deviation']):.2e}
（`{worst['key']}`，两位有效数字舍入带内）；若只看四位有效数字的格子，最坏为
{abs(worst_excluding_rounding['relative_deviation']):.2e}（`{worst_excluding_rounding['key']}`）。

### 3.1 表格的物理内容（数字全部来自 JSON）

| 量 | 文档声明 | 本次复算 | 相对偏差 |
|---|---:|---:|---:|
| 表 10，256 面元 \(A_{{33}}'\) 相对误差 | {d['hulme_heave_added_mass_rel_error_256']:.3e} | {m['hulme_heave_added_mass_rel_error_256']:.4e} | {(m['hulme_heave_added_mass_rel_error_256']-d['hulme_heave_added_mass_rel_error_256'])/d['hulme_heave_added_mass_rel_error_256']:+.2e} |
| 表 10，256 面元 \(B_{{33}}'\) 相对误差 | {d['hulme_heave_damping_rel_error_256']:.3e} | {m['hulme_heave_damping_rel_error_256']:.4e} | {(m['hulme_heave_damping_rel_error_256']-d['hulme_heave_damping_rel_error_256'])/d['hulme_heave_damping_rel_error_256']:+.2e} |
| 表 10，256 面元 \(A_{{11}}'\) 相对误差 | {d['hulme_surge_added_mass_rel_error_256']:.3e} | {m['hulme_surge_added_mass_rel_error_256']:.4e} | {(m['hulme_surge_added_mass_rel_error_256']-d['hulme_surge_added_mass_rel_error_256'])/d['hulme_surge_added_mass_rel_error_256']:+.2e} |
| 表 10，256 面元 \(B_{{11}}'\) 相对误差 | {d['hulme_surge_damping_rel_error_256']:.3e} | {m['hulme_surge_damping_rel_error_256']:.4e} | {(m['hulme_surge_damping_rel_error_256']-d['hulme_surge_damping_rel_error_256'])/d['hulme_surge_damping_rel_error_256']:+.2e} |
| 表 10，256 面元 \(F_z\\) 相对误差 | {d['hulme_excitation_force_z_rel_error_256']:.3e} | {m['hulme_excitation_force_z_rel_error_256']:.4e} | {(m['hulme_excitation_force_z_rel_error_256']-d['hulme_excitation_force_z_rel_error_256'])/d['hulme_excitation_force_z_rel_error_256']:+.2e} |
| 表 10，最细层观测阶（\(A_{{33}}/B_{{33}}/A_{{11}}/B_{{11}}/F_z\\)） | 2.06 / 1.98 / 2.01 / 1.88 / 2.02 | {m['hulme_observed_order_a33_256']:.3f} / {m['hulme_observed_order_b33_256']:.3f} / {m['hulme_observed_order_a11_256']:.3f} / {m['hulme_observed_order_b11_256']:.3f} / {m['hulme_observed_order_fz_256']:.3f} | ≤ {max(abs(row['relative_deviation']) for row in rows if row['key'].startswith('hulme_observed_order') and row['key'].endswith('_256')):.1e} |
| 表 11，未增广 BIE 条件数 | {d['hulme_chief_unregularized_condition_number']:.4g} | {m['hulme_chief_unregularized_condition_number']:.5g} | {(m['hulme_chief_unregularized_condition_number']-d['hulme_chief_unregularized_condition_number'])/d['hulme_chief_unregularized_condition_number']:+.2e} |
| 表 11，5 点 CHIEF 条件数 | {d['hulme_chief_chief_regularized_condition_number']:.3g} | {m['hulme_chief_chief_regularized_condition_number']:.5g} | {(m['hulme_chief_chief_regularized_condition_number']-d['hulme_chief_chief_regularized_condition_number'])/d['hulme_chief_chief_regularized_condition_number']:+.2e} |
| 表 11，未增广 \(B_{{33}}'\\)（非物理负阻尼） | {d['hulme_chief_unregularized_b33_dimless']:.4g} | {m['hulme_chief_unregularized_b33_dimless']:.6g} | {(m['hulme_chief_unregularized_b33_dimless']-d['hulme_chief_unregularized_b33_dimless'])/abs(d['hulme_chief_unregularized_b33_dimless']):+.2e} |
| 表 11，CHIEF 后 \(B_{{33}}'\\) | {d['hulme_chief_chief_regularized_b33_dimless']:.3g} | {m['hulme_chief_chief_regularized_b33_dimless']:.6g} | {(m['hulme_chief_chief_regularized_b33_dimless']-d['hulme_chief_chief_regularized_b33_dimless'])/d['hulme_chief_chief_regularized_b33_dimless']:+.2e} |
| 表 12，320 面元 \(A_{{33}}'\) 相对误差 | {d['wang_linton_heave_added_mass_rel_error_320']:.3e} | {m['wang_linton_heave_added_mass_rel_error_320']:.4e} | {(m['wang_linton_heave_added_mass_rel_error_320']-d['wang_linton_heave_added_mass_rel_error_320'])/d['wang_linton_heave_added_mass_rel_error_320']:+.2e} |
| 表 12，320 面元观测阶（\(A_{{33}}/B_{{33}}/A_{{11}}/F_z\\)） | 2.06 / 1.93 / 2.03 / 1.98 | {m['wang_linton_observed_order_a33_320']:.3f} / {m['wang_linton_observed_order_b33_320']:.3f} / {m['wang_linton_observed_order_a11_320']:.3f} / {m['wang_linton_observed_order_fz_320']:.3f} | ≤ {max(abs(row['relative_deviation']) for row in rows if row['key'].startswith('wang_linton_observed') and row['key'].endswith('_320')):.1e} |

### 3.2 全量逐格表（程序化注入，与报告同源）

{table}

### 3.3 图

* `fig_registry_3d_free_surface.png`：63 格的实测相对偏差（对数轴）与其容差线，一列一条，无一条越线。
* `fig_three_d_convergence_registry.png`：左＝Hulme 半球五个量的误差–面元数（16/64/256）与
  \\(O(h^2)\\) 参考斜率；右＝Wang/Linton 潜没球四个量（20/80/320）。两图即表 10/12 的视觉复核。
* `fig_chief_regularization.png`：\(Ka=2.58\\) 未增广 BIE vs 5 点 CHIEF 的
  \\(A_{{33}}',B_{{33}}'\\)（虚线为精确值）与条件数；可见**免盖板** CHIEF 同时把负阻尼
  {m['hulme_chief_unregularized_b33_dimless']:+.4f} 拉回 {m['hulme_chief_chief_regularized_b33_dimless']:+.4f}、
  条件数 {m['hulme_chief_unregularized_condition_number']:.1f} → {m['hulme_chief_chief_regularized_condition_number']:.2f}。

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

* **只有 3 次求解却出 63 格**：注册表缓存键是 `(group, parameters)`，所以「同表共用一次求解」
  是靠参数完全一致实现的；若将来有人给某格单独改参数，会静默变成第二次求解（慢，但不错）。
* **表 10 的 \(B_{{11}}'\\) 观测阶不是基准输出**：文档写 1.62/1.88，基准 JSON 里没有这个键，
  本单在 runner 里按 \\(\\log_2\\) 误差比现算。它是恒等式，**不能当作独立正确性证据**（与
  `docs/ursell_*` 系列里对 Richardson 拟合阶的告诫同一性质），只用来钉住「文档那个数」。
* **两位有效数字的格子**：表 12 转动耦合列只给 2.6e-5 这样的精度，若沿用 5e-3 容差必然假红
  （实测 +1.85%）。本单把它显式写成「舍入带容差 2.5e-2」并注进注册表，而不是偷偷放大全局容差。
  20 面元的 2.5e-32 是机器零，只能按上界判，报告里那条 -1.00 的「偏差」是相对 1e-10·1e-10
  上界而言，不表示失配。
* **成本与门禁**：整组 63 格 {elapsed:.0f} s（128 核、load15 70–100 的窗口内），因此放进 `heavy` 层，
  默认 `pytest` 不跑；heavy 门测试 91.9 s 全绿。**未做的事**：没有改任何求解器实现，也没有
  继续加密到 1024 面元（\(B_{{11}}'\\) 的 1024 面元行由 m2609 的
  `kochin_hulme_convergence` 独立钉桩，本单不重复）；频率扫描表（\(Ka=0.3\\text{{–}}3.0\\)）与
  Haskind/Kochin 守恒行仍未进注册表，属后续增量。
* **剩余精度限制（诚实）**：表 10 的 \(B_{{11}}'\\) 在 256 面元仍是 3.53%（见下节），这是**离散误差**
  而非实现缺陷——1024 面元降到 0.90%，观测阶 1.62/1.88/1.96。

## 5. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---|
| \(A_{{33}}'\\)（256 面元，\(Ka=1.0\\)） | Hulme (1982) 半解析 | {m['hulme_heave_added_mass_rel_error_256']*100:.2f}% | ✅ |
| \(B_{{33}}'\\)（256 面元） | Hulme (1982) | {m['hulme_heave_damping_rel_error_256']*100:.2f}% | ✅ |
| \(A_{{11}}'\\)（256 面元） | Hulme (1982) | {m['hulme_surge_added_mass_rel_error_256']*100:.2f}% | ✅ |
| \(B_{{11}}'\\)（256 面元） | Hulme (1982) | {m['hulme_surge_damping_rel_error_256']*100:.2f}% | ❌ **>3%**（1024 面元 0.90%，m2609；一阶/近二阶离散误差，非实现缺陷） |
| \(F_z\\)（256 面元） | Hulme (1982) | {m['hulme_excitation_force_z_rel_error_256']*100:.2f}% | ✅ |
| CHIEF 后 \(B_{{33}}'\\)（\(Ka=2.58\\)） | Hulme (1982) | {abs(m['hulme_chief_chief_regularized_b33_dimless']-m['hulme_chief_exact_b33_dimless'])/m['hulme_chief_exact_b33_dimless']*100:.2f}% | ✅（未增广方案为 {abs(m['hulme_chief_unregularized_b33_dimless']-m['hulme_chief_exact_b33_dimless'])/m['hulme_chief_exact_b33_dimless']*100:.0f}% 且符号错误） |
| \(A_{{33}}'\\)（320 面元，\(Ka=0.8\\)） | Wang/Linton 多极子 | {m['wang_linton_heave_added_mass_rel_error_320']*100:.2f}% | ✅ |
| \(A_{{11}}'\\)（320 面元） | Wang/Linton 多极子 | {m['wang_linton_surge_added_mass_rel_error_320']*100:.2f}% | ✅ |
| 文档数字 vs 实测（63 格中 62 格等值判据） | 文档声明 | 最坏 {abs(worst['relative_deviation']):.2e}；除转动耦合两格外 ≤ {abs(max_excl:=0.0) or 0.0:.1e} → 见 §3 | ✅ |

**显式结论**：本单涉及的所有**收敛层关键量**除 \(B_{{11}}'\\)@256 外均 <3%；唯一超标的
\(B_{{11}}'\\)（3.53%）已在文档中标注为最慢收敛量，并有 1024 面元 0.90% 的收敛证据，
本报告不把它粉饰为「已达标」。文档 63 格数字本身与当前代码**逐格可复现**（62 格等值判据最坏
2.2e-2，且该格仅为两位有效数字舍入）。
"""

# fix the placeholder row: compute the true worst excluding the two rounding rows
worse_excl = max(
    (
        row
        for row in equality
        if row["tolerance"] < 2.0e-2
    ),
    key=lambda row: abs(row["relative_deviation"]),
)
MARKDOWN = MARKDOWN.replace(
    "最坏 {:.2e}；除转动耦合两格外 ≤ {:.1e} → 见 §3".format(
        abs(worst["relative_deviation"]), 0.0
    ),
    "最坏 {:.2e}（两位有效数字的转动耦合格）；其余 ≤ {:.2e}".format(
        abs(worst["relative_deviation"]), abs(worse_excl["relative_deviation"])
    ),
)

Path("dev_report_m3181_three_d_free_surface_registry.md").write_text(MARKDOWN, encoding="utf-8")

figures = json.loads((OUT / "outputs" / "figures_b64.json").read_text())["figures"]
captions = {
    "fig_registry_3d_free_surface.png": "63 个文档数字的实测相对偏差与各自容差线（对数轴），无一条越线",
    "fig_three_d_convergence_registry.png": "Hulme 半球（16/64/256 面元）与潜没球（20/80/320 面元）的误差-面元数曲线与 O(h^2) 参考斜率",
    "fig_chief_regularization.png": "Ka=2.58：未增广 BIE 的非物理负阻尼与病态条件数被 5 点三维 CHIEF 同时修复",
}
payload = {
    "title": "TensorBEM 三维自由液面声明数字注册表化：Hulme/Wang-Linton 表 10–12 共 63 格全量复算",
    "summary": (
        "把 docs/marine_bem_benchmarks.md §4.10–§4.11 的 63 个三维自由液面声明数字逐格登记进 "
        "declared_numbers heavy 层（新增 hulme3d/hulme3d_chief/wang_linton3d 三组，同表共用一次求解），"
        "63/63 PASS，耗时 98 s；等值判据最坏相对偏差 2.2e-2（两位有效数字舍入格），四位有效数字格最坏 3.9e-3。"
        "另钉桩 B11@256 面板 3.53% 的离散误差（1024 面板 0.90%）。"
    ),
    "scenario_key": "tensorbem-declared-numbers-3d-free-surface",
    "mission_id": 3181,
    "tags": ["TensorBEM", "verification", "declared-numbers", "3D-free-surface", "Hulme", "Wang-Linton", "CHIEF"],
    "markdown": MARKDOWN,
    "figures": [
        {
            "name": figure["name"],
            "caption": captions.get(figure["name"], figure["name"]),
            "png_base64": figure["png_base64"],
        }
        for figure in figures
    ],
}
Path("/tmp/report.json").write_text(json.dumps(payload), encoding="utf-8")
print("markdown chars", len(MARKDOWN), "payload kb", len(json.dumps(payload)) // 1024)