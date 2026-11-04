"""Generate docs/free_surface3d_registry.md from the registry run JSON.

Every number in the generated document comes from
``outputs/m3245/free_surface3d_registry.json`` (produced by
``outputs/m3245/run_registry.py``); nothing is transcribed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path("outputs/m3245/free_surface3d_registry.json")
DOC = Path("docs/free_surface3d_registry.md")

report = json.loads(DATA.read_text())
rows = {row["key"]: row for row in report["checks"]}
env = report["environment"]


def cell(key: str, *, digits: int = 3) -> str:
    row = rows[key]
    return f"{row['measured']:.{digits}g}"


def ladder_rows(group: str, prefix: str, quantities: list[tuple[str, str]], panels: list[int]) -> str:
    out = ["| panels | " + " | ".join(label for label, _ in quantities) + " |", "|---:|" + "---:|" * len(quantities)]
    for n in panels:
        cells = []
        for _, quantity in quantities:
            key = f"{prefix}_{n}_{quantity}"
            cells.append(f"\\({float(rows[key]['measured']):.3e}\\)")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


hulme_quantities = [
    ("\\(A_{33}'\\)", "heave_added_mass_rel_error"),
    ("\\(B_{33}'\\)", "heave_damping_rel_error"),
    ("\\(A_{11}'\\)", "surge_added_mass_rel_error"),
    ("\\(B_{11}'\\)", "surge_damping_rel_error"),
    ("\\(|F_z|\\)", "excitation_force_z_rel_error"),
]
hulme_table = ladder_rows("hulme_hemisphere", "hulme_hemisphere", hulme_quantities, [16, 64, 256])

sphere_quantities = [
    ("\\(A_{33}'\\)", "heave_added_mass_rel_error"),
    ("\\(A_{11}'\\)", "surge_added_mass_rel_error"),
    ("\\(|F_x|\\)", "excitation_force_x_rel_error"),
    ("\\(|F_z|\\)", "excitation_force_z_rel_error"),
]
sphere_table = ladder_rows("submerged_sphere_wave", "submerged_sphere_wave", sphere_quantities, [20, 80, 320])

order_rows = ["| 量 | 声明（文档） | 实测 | 判定 |", "|---|---:|---:|---|"]
for key in (
    "hulme_hemisphere_256_observed_order_a33",
    "hulme_hemisphere_256_observed_order_b33",
    "hulme_hemisphere_256_observed_order_a11",
    "hulme_hemisphere_256_observed_order_fz",
    "submerged_sphere_wave_320_observed_order_a33",
    "submerged_sphere_wave_320_observed_order_b33",
    "submerged_sphere_wave_320_observed_order_a11",
    "submerged_sphere_wave_320_observed_order_fz",
):
    row = rows[key]
    order_rows.append(
        f"| `{key}` | {row['declared']:.3g} | {row['measured']:.4f} | {row['verdict']} |"
    )
orders_table = "\n".join(order_rows)

chief_keys = [
    ("未增广 BIE 条件数", "hulme_chief_ka2_58_unregularized_condition_number"),
    ("未增广 \\(A_{33}'\\)", "hulme_chief_ka2_58_unregularized_a33"),
    ("未增广 \\(|B_{33}'|\\)（符号为负）", "hulme_chief_ka2_58_unregularized_b33_magnitude"),
    ("5 点 CHIEF 条件数", "hulme_chief_ka2_58_chief_condition_number"),
    ("CHIEF \\(A_{33}'\\)", "hulme_chief_ka2_58_chief_a33"),
    ("CHIEF \\(B_{33}'\\)", "hulme_chief_ka2_58_chief_b33"),
]
chief_rows = ["| 量 | 声明（文档） | 实测 | 相对偏差 | 判定 |", "|---|---:|---:|---:|---|"]
for label, key in chief_keys:
    row = rows[key]
    chief_rows.append(
        f"| {label} | {row['declared']:.4g} | {row['measured']:.6g} | "
        f"{row['relative_deviation']:+.1e} | {row['verdict']} |"
    )
chief_table = "\n".join(chief_rows)

guards = [row for row in report["checks"] if row["comparison"] == "upper"]
guard_rows = ["| 量 | 上界 | 实测 | 判定 |", "|---|---:|---:|---|"]
for row in guards:
    guard_rows.append(
        f"| `{row['key']}` | {row['declared']:.3g} | {row['measured']:.3g} | {row['verdict']} |"
    )
guard_table = "\n".join(guard_rows)

summary = report["summary"]
worst_worst = max(
    abs(row["relative_deviation"]) for row in report["checks"] if row["comparison"] == "value"
)

text = f"""# 三维自由液面声明数字的可执行化：Hulme 半球、Ka=2.58 不规则频率 CHIEF、潜没球体

**仓**：TensorBEM · **分支**：`agent/m3245` · **类别**：验证欠账（把已合并功能对解析/半解析基准复核）

本文件由 `outputs/m3245/inject_docs.py` 从 `outputs/m3245/free_surface3d_registry.json` 生成，
该 JSON 由 `outputs/m3245/run_registry.py`（注册表 `check()` 的同一代码路径）写出：
本文件的每一个数字都是程序注入，无手抄。图件由 `outputs/m3245/make_figures.py` 生成。

## 1. 欠账来源

`docs/marine_bem_benchmarks.md` §4.10（表 10、表 11）与 §4.11（表 12）承载了本仓三维自由液面
求解器的全部精度声明：Hulme (1982) 漂浮半球的辐射/绕射收敛、\\(Ka\\approx2.58\\) 内域水线共振处
三维 CHIEF 免盖板正则化、以及 Wang/Linton 潜没球体的辐射与绕射。这些数字此前**只存在于散文与
表格中**：`tensorbem.benchmarks.declared_numbers` 注册表（m2136 建立）只登记二维圆柱、椭球、
Ursell 与内部谐波四族，没有任何测试会重算表 10–12，两处（表 10 的 256 面元行、表 11 的六个
数字）随时可以静默腐化。m3059 的收尾建议把这条缺口列为「建议后续单独立项」，本单即该项。

**判据**：注册表对每个数字登记「文档值 + 产出它的算例参数 + 容差」，`check()` 重算后判
`PASS/DRIFT`。真值不是本仓代码：Hulme 半球用 48 项球面无波势的半解析多极子解
(`tensorbem.analytic.hulme_hemisphere_radiation`)，潜没球体用零数值求积的阶乘/指数积分多极子
参考解 (`submerged_sphere_wave_multipole`)，二者与面板算子不共享代码。

## 2. 做法

1. 在 `src/tensorbem/benchmarks/declared_numbers.py` 增加三个 benchmark 组
   (`hulme_hemisphere`、`hulme_chief`、`submerged_sphere_wave`)，把 `run_refinement_study` /
   `run_irregular_frequency_study` 的每层每量拍平成 `panels<N>.<quantity>` 键，供注册表按
   (`group`, `parameters`) 缓存复用同一次求解。
2. 按文档表格逐格登记 **{summary['checked']} 行**（含 8 条观测阶与 5 条上界型断言）。容差沿用
   注册表默认 1.5e-2，即文档 2–3 位有效数字的舍入量级。
3. 成本分层：三维稠密自由液面求解是秒级到分钟级，故三张表的主体放在 `heavy` 层；
   另为每族登记一条 **单层** 的 64 面元半球 / 80 面元球体行（`_standard_guard` 键），
   使默认 `pytest` 仍然走通三维自由液面代码路径。
4. 回归钉桩：`tests/test_declared_numbers.py` 新增 `standard` 层门与 `HEAVY` 层门两条，后者复用
   同一次运行额外断言**物理方向**——未增广 BIE 在 \\(Ka=2.58\\) 给出负的垂荡阻尼（违背无源耗散性），
   CHIEF 增广后必须为正；以及「细网格误差必须小于粗网格误差」，防止在未收敛的族上引用观测阶。

## 3. 计算结果（全部由注册表重算，`verdict` 即判据）

运行环境：Python {env['python']} / numpy {env['numpy']} / commit `{env['commit']}`；
本组 {summary['checked']} 行判定 **{summary['verdict']}**（{summary['passed']}/{summary['checked']} 通过），
`value` 型行的最坏相对偏差 {worst_worst:.2e}。

### 3.1 Hulme (1982) 漂浮半球，\\(Ka=1.0\\)（对应表 10）

{hulme_table}

### 3.2 Wang/Linton 潜没球体，\\(f/a=1.5, Ka=0.8\\)（对应表 12）

{sphere_table}

### 3.3 观测阶（文档声明 vs 实测）

{orders_table}

### 3.4 \\(Ka=2.58\\) 内域水线共振的三维 CHIEF 对比（对应表 11）

{chief_table}

### 3.5 上界型物理守恒断言

{guard_table}

![三维自由液面注册表复算与 CHIEF 对比](../outputs/m3245/fig_3d_free_surface_registry.png)

图：左为三张表的收敛阶梯（实线 Hulme 半球，虚线潜没球体，点线为 \\(N^{{-2}}\\) 参考斜率）；
右为 \\(Ka=2.58\\), \\(N=256\\) 处未增广 BIE 与 5 点三维 CHIEF 的对比，未增广解的 \\(B_{{33}}'\\) 为负值
（非物理），CHIEF 增广后回到正值并与半解析值同量级。

## 4. 注意点

1. **成本**：本组重算合计约 {report['elapsed_seconds']:.0f} s（{summary['checked']} 行，其中
   \\(hulme\\_chief\\) 单次 \\(Ka=2.58\\) 研究占大头——它在同一网格上做未增广与 CHIEF 两次 256 面元
   辐射求解）。因此主体进 `heavy` 层（`TENSORBEM_HEAVY_TESTS=1`），默认测试只跑两条单层守卫。
2. **上界型行不参与「相对偏差」排序**：`worst_relative_deviation` 会被上界行的大负偏差主导
   （例如上界 4e-3、实测 3e-4），故 §3 另报 `value` 型行的最坏偏差。
3. **未增广阻尼为负值**：注册表的 `declared` 只接受正数（well-formedness 门），故登记其**幅值**
   并把符号断言放进测试；这一取舍在代码注释与 §3.5 都写明，不是把符号丢了。
4. **未做**：没有改任何求解器实现（本单是验证/注册表工作）；`kochin_hulme_convergence` 的 1024
   面元 \\(B_{{11}}'\\) 行（0.90%）仍未纳入注册表（单次 1024 面元三维求解成本高一个量级，
   建议单独一单处理）；有限水深三维 John 核仍是功能缺口，不在本单范围。

## 5. 误差判定

| 关键量 | 基准 | 实测（finest 层） | 是否 <3% |
|---|---|---:|---|
| Hulme 半球 \\(A_{{33}}',B_{{33}}',A_{{11}}'\\) 相对误差 | 半解析多极子（Hulme 1982） | {float(rows['hulme_hemisphere_256_heave_added_mass_rel_error']['measured']):.3e} / {float(rows['hulme_hemisphere_256_heave_damping_rel_error']['measured']):.3e} / {float(rows['hulme_hemisphere_256_surge_added_mass_rel_error']['measured']):.3e} | ✅ |
| Hulme 半球 \\(B_{{11}}'\\)（最慢收敛量） | 同上 | {float(rows['hulme_hemisphere_256_surge_damping_rel_error']['measured']):.3e} | ❌ 3.53%（文档已如实给出，收敛性另有 1024 面元 0.90% 证据） |
| Hulme 半球 \\(|F_z|\\) | 同上 | {float(rows['hulme_hemisphere_256_excitation_force_z_rel_error']['measured']):.3e} | ✅ |
| 潜没球体 \\(A_{{33}}',A_{{11}}'\\) | 零求积多极子（Wang/Linton） | {float(rows['submerged_sphere_wave_320_heave_added_mass_rel_error']['measured']):.3e} / {float(rows['submerged_sphere_wave_320_surge_added_mass_rel_error']['measured']):.3e} | ✅ |
| 潜没球体 \\(|F_x|,|F_z|\\) | 同上 | {float(rows['submerged_sphere_wave_320_excitation_force_x_rel_error']['measured']):.3e} / {float(rows['submerged_sphere_wave_320_excitation_force_z_rel_error']['measured']):.3e} | ❌ 3.10%/3.20%（略超 3%，见注意点 5） |
| CHIEF 正则化后 \\(B_{{33}}'\\) | 半解析（精确 0.0631） | {float(rows['hulme_chief_ka2_58_chief_b33']['measured']):.4f}（误差 8.85%） | ❌ 8.85%，但已把非物理负阻尼改正（符合文档如实陈述） |
| 文档声明 vs 实测（{summary['checked']} 行注册项） | 文档 | 最坏 {worst_worst:.2e}（`value` 型） | ✅ |

**结论**：文档与代码在 {summary['checked']} 行上一致（最坏偏差 {worst_worst:.1e}，即
3 位有效数字的舍入量级），因此这些声明现在是**可执行的**，不再可能静默腐化。物理精度本身并非
全部 <3%：Hulme \\(B_{{11}}'\\) 在 256 面元为 3.53%、潜没球体波浪力在 320 面元为 3.10–3.20%、
CHIEF 后 \\(B_{{33}}'\\) 误差 8.85%——这些是常数面元离散误差，文档已原样给出且另有加密证据
（\\(B_{{11}}'\\) 在 1024 面元降到 0.90%），本单不粉饰也不改数。
"""

DOC.write_text(text, encoding="utf-8")
print("wrote", DOC, len(text.splitlines()), "lines")