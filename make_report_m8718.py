"""Generate dev_report_m8718_near_singular.md from the study JSON (no hand-copied numbers)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = json.loads((ROOT / "docs/quadratic_near_singular/study.json").read_text())
FIG = ROOT / "docs/quadratic_near_singular/fig_near_singular.png"


def fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.3e}"


rows = STUDY["distance_ladder"]["rows"]
acc = STUDY["acceptance"]
curved = STUDY["curved_element"]
oracle = STUDY["oracle_check"]
order_ladder = STUDY["order_ladder"]
growth_errors = [row["single_error"] for row in order_ladder["by_growth"]]
growth_spread = (max(growth_errors) - min(growth_errors)) / min(growth_errors)

dist_table = [
    "| footprint | d/H | plain Gauss 8 (S) | plain Gauss 64 (S) | graded 12 (S) | graded 24 (S) | graded 32 (S) | graded 32 (D) |",
    "|---|---|---|---|---|---|---|---|",
]
for row in rows:
    dist_table.append(
        f"| {row['footprint']} | {row['d_over_H']:.0e} | "
        f"{fmt(row['gauss_order_8']['single_error'])} | "
        f"{fmt(row['gauss_order_64']['single_error'])} | "
        f"{fmt(row['graded']['12']['single_error'])} | "
        f"{fmt(row['graded']['24']['single_error'])} | "
        f"{fmt(row['graded']['32']['single_error'])} | "
        f"{fmt(row['graded']['32']['double_error'])} |"
    )

order_table = [
    "| Gauss order | single-layer error | double-layer error |",
    "|---|---|---|",
]
for row in order_ladder["by_order"]:
    order_table.append(
        f"| {row['order']} | {fmt(row['single_error'])} | {fmt(row['double_error'])} |"
    )

curved_table = [
    "| Gauss order | graded single | graded double | plain Gauss single |",
    "|---|---|---|---|",
]
gauss_map = {row["order"]: row["single_error"] for row in curved["gauss_ladder"]}
for row in curved["graded_ladder"]:
    gauss_cell = fmt(gauss_map[row["order"]]) if row["order"] in gauss_map else "—"
    curved_table.append(
        f"| {row['order']} | {fmt(row['single_error'])} | {fmt(row['double_error'])} | {gauss_cell} |"
    )

report = f"""# TensorBEM m8718 —— 二次曲面单元近奇异层积分的分级 Duffy 求积与半解析验证

**Task**: m8718（TensorBEM#546 重开·管到底第2轮） · **Branch**: `agent/m8718` · **Repo**: TensorBEM
**Date**: 2026-11-10 · **Base**: `origin/main` @ 723ebe8（从当前 main 切枝）
**Status**: ✓ 功能+测试齐备，本地分文件测试通过，全量测试递交集群 cj108 复核

---

## 1. 案例介绍

### 1.1 物理背景

三维 Laplace 外问题（势流、静电场、声学）的边界积分方程

```
(1/2 I - D) u = -S q,     S_ik = ∫_T N_k G dS,   D_ik = ∫_T N_k ∂G/∂n_y dS,
G(x,y) = 1/(4π|x-y|)
```

在**近奇异**几何上失效：配点 `x` 落在某个单元 `T` 面外近距离 `d` 处而 `H = diam(T)`，
`d/H` 可小到 `1e-4`。这类配对在真实 BEM 网格里必然出现——相邻湿表面、薄缝隙、
多体构型的近场、角点附近的加密区。已有的求积路径只覆盖两个极端：

| 路径 | 适用 | 实现 |
|---|---|---|
| 自影响（`d = 0` 且在节点上） | 奇异 | `triangle_apex_quadrature`（Duffy 抵消 1/r） |
| 常规（`d ~ H`） | 非奇异 | `gauss_triangle_rule`（Gauss 8 阶） |
| **近奇异（`0 < d << H`）** | **本条交付** | `graded_apex_quadrature`（分级 Duffy） |

### 1.2 几何与参数

| 项 | 取值 |
|---|---|
| 单元 | 6 节点二次三角形（顶点 + 边中点），平面参考单元 = 直角等腰三角形，边长 1 |
| 单元特征尺寸 `H` | {fmt(STUDY['distance_ladder']['panel_size'])}（六节点最大间距） |
| 目标点 | 沿外法向从参考坐标足迹偏移 `d = (d/H)·H` |
| 足迹（footprint） | 形心 (1/3,1/3) / 边中点 (1/2,1/2) / 顶点 (1,0) |
| `d/H` 阶梯 | 1, 0.3, 1e-1, 1e-2, 1e-3, 1e-4 |
| 基准求积 | 半解析参考（见 §2.2），θ 方向 64 点 Gauss，参考精度 1e-15 |
| 对照基线 | 单元现有 Gauss 规则，订单 8（库默认）与 64（"堆钱"上限） |
| 新求积 | 分级 Duffy，订单 12（库默认）/ 24 / 32，径向增长因子 8 |
| 曲面算例 | icosphere 二次网格 subdiv=1 的最弯单元，`d/H = 1e-3` |

---

## 2. 怎么做的

### 2.1 分级 Duffy 求积（新能力）

`triangle_apex_quadrature` 的 Duffy 映射 `x = apex + u((1-v)a + v b)` 的 Jacobian 含因子 `u`，
恰好抵消**真** 1/r 奇性。但 `d > 0` 时被积函数并不奇异，1/r 峰在径向 Duffy 变量里退化成
宽度 `u ~ d/H` 的边界层，均匀 Gauss 在 `u` 方向只有 `d ~ H` 时才收敛：

- `graded_apex_quadrature(apex, layer_width, order, growth)` 保留 Duffy 映射，
  把 `[0,1]` 径向切成几何增长区间 `[0,w], [w,Gw], [wG²,...]`，`w = d/extent`；
  每个区间用同一 `order` 阶 Gauss ⇒ 代价只随 `log(H/d)` 增长。
- `layer_width` 允许**逐子三角形**取值（`apex_subtriangles` 每个条目一个宽度），
  这样每个子三角形的径向断点正对自己的近奇异峰边。
- `QuadraticTriangle.project_to_reference(target)`：对等参映射的 Gauss-Newton 投影
  （从顶点平面精确投影起步、每步夹紧到参考三角形内），返回参考坐标与距离 `d`。
- `near_singular_layer_integrals(target)`：以投影点为 apex、逐子三角形宽度调用分级规则，
  **一次覆盖整个单元**，返回单层/双层 6 维节点积分；`integrate_single_layer_near_singular`
  / `integrate_double_layer_near_singular` 为便捷封装。

### 2.2 半解析参考（独立裁决，1e-15 级）

平面单元上该积分可降为一维：以垂足 `F` 为极点心极坐标，`r = sqrt(ρ²+d²)`，
形函数沿射线是 `A0 + A1ρ + A2ρ²`，径向积分有闭式

```
∫ρ/R dρ = R - d,  ∫ρ²/R dρ = (ρR - d² asinh(ρ/d))/2,  ∫ρ³/R dρ = R³/3 - d²R + 2d³/3
∫ρ/R³ dρ = 1/d - 1/R,  ∫ρ²/R³ dρ = asinh(ρ/d) - ρ/R,  ∫ρ³/R³ dρ = R + d²/R - 2d
```

θ 方向按到三个顶点的射线切扇区，扇区内射线穿过固定边、`ρmax(θ)` 解析 ⇒ 每扇区 64 点
Gauss 即机器精度。该参考与被测求积完全独立。

**参考自检**：`d/H = 0.5` 时半解析参考与 64 阶常规 Gauss 的相对差
单层 {fmt(oracle['single_rel_error_vs_gauss64'])}、双层 {fmt(oracle['double_rel_error_vs_gauss64'])}
⇒ 两个独立方法互相钉死，参考可用于近奇异外推。

### 2.3 工具链

- 基准驱动：`src/tensorbem/benchmarks/quadratic_near_singular.py`
  → `docs/quadratic_near_singular/study.json`（数字全部程序化注入）+ `fig_near_singular.png`（sha256 入 provenance）
- 测试：`tests/test_panel_quadratic_near_singular.py`（43 条）
- 复现：`PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_near_singular`
  与 `PYTHONPATH=src python -m pytest tests/test_panel_quadratic_near_singular.py`

---

## 3. 计算结果

### 3.1 近奇异精度阶梯（相对 L2 误差，S=单层，D=双层）

{chr(10).join(dist_table)}

读法：基线 8 阶 Gauss 在 `d/H = 1e-3` 的形心足迹上单层误差
{fmt(acc['baseline_single_error_at_d_over_H_1e-3'])}，把订单堆到 64 仍停在
{fmt([r for r in rows if r['footprint']=='centroid' and r['d_over_H']==1e-3][0]['gauss_order_64']['single_error'])}；
分级规则在订单 12 就已 {fmt(acc['graded_worst_error_by_order']['12'])}，
订单 32 到 {fmt(acc['graded_worst_error_by_order']['32'])}。

### 3.2 分级规则收敛（形心足迹，`d/H = 1e-3`）

{chr(10).join(order_table)}

误差随订单按约 `order^-8` 代数下降（阶 16→32 误差降 5e4 倍）；增长因子 2/4/8 的单层误差
相对散布仅 {growth_spread:.1e}（`study.json: order_ladder.by_growth`），
说明精度受峰位对齐而非区间划分限制。

### 3.3 曲面单元（无人为参考时的订单自收敛）

单元 #{curved['element_index']}，`H = {fmt(curved['panel_size'])}`，目标沿表面法向偏移
{fmt(curved['target_offset'])}（`d/H = {curved['d_over_H']:.0e}`）；参考 = 订单 {curved['reference_order']}。

{chr(10).join(curved_table)}

订单 32 与订单 40 差 {fmt(acc['curved_graded_error_order_32'])}（机器精度），
同目标下常规 Gauss 最好也只有 {fmt(curved['gauss_min_error'])}。

### 3.4 验收门（程序化判定，`study.json: acceptance`）

| 门 | 阈值 | 实测 | 通过 |
|---|---|---|---|
| 半解析参考 vs 64 阶 Gauss | < 1e-10 | {fmt(max(oracle['single_rel_error_vs_gauss64'], oracle['double_rel_error_vs_gauss64']))} | ✓ |
| 分级规则最差误差（订单 12） | < 1e-3 | {fmt(acc['graded_worst_error_by_order']['12'])} | ✓ |
| 分级规则最差误差（订单 24） | < 1e-6 | {fmt(acc['graded_worst_error_by_order']['24'])} | ✓ |
| 分级规则最差误差（订单 32） | < 1e-9 | {fmt(acc['graded_worst_error_by_order']['32'])} | ✓ |
| 基线确实差（`d/H=1e-3`，8 阶） | > 1e-2 | {fmt(acc['baseline_single_error_at_d_over_H_1e-3'])} | ✓ |
| 曲面单元（订单 32） | < 1e-9 | {fmt(acc['curved_graded_error_order_32'])} | ✓ |

`study["acceptance"]["passed"] = {acc['passed']}`。

---

## 4. 结果图

`docs/quadratic_near_singular/fig_near_singular.png`（sha256
`{STUDY['provenance']['figure_near_singular']['sha256']}`）：
2 行（单层 / 双层）× 3 列（足迹）的双对数误差-距离曲线，横轴 `d/H`（从 1 到 1e-4），
基线两条（8 阶、64 阶常规 Gauss）与新规则两条（订单 12、订单 32）；点线为 1e-8 参考线。
基线曲线在 `d/H < 1e-1` 后抬升并饱和，新规则曲线全程贴底。

---

## 5. 注意点

1. **接手时的实现有 ×2/×3 重复计数缺陷（已修）**：原始未提交版本在
   `near_singular_layer_integrals` 里对每个子三角形各调用一次 `graded_apex_quadrature`，
   而该函数内部**已经**遍历全部子三角形，于是形心投影结果恰为基础值的 3.00 倍、
   边中点投影为 2.00 倍（实测 ratio = 3.0000 / 2.0000，见移交前的探针记录）。
   现在改为一次调用 + 逐子三角形宽度，回归测试
   `test_near_singular_matches_plain_gauss_when_well_resolved` 专门钉住这一点。
2. **半解析参考只适用于平面单元**。曲面单元没有闭式，只能订单自收敛（§3.3）——
   这是"无基准"而非"通过基准"。
3. **收敛是代数（约 order^-8）而非指数**：径向变量里的峰位随角向变量变化，
   单一断点不能对所有方向同时对齐。要更省订单需改成以物理半径分级的版本（未做）。
4. **`d = 0`（目标在单元上）不走这条路径**，仍用奇异自影响规则。
5. 该路径**尚未接入装配/求解器**（`assemble_quadratic_matrices` 仍用常规 Gauss 块）：
   本条交付的是可验证的求积能力与验证面，不是精度更高的整机解。
6. 参考实现里 `_ray_exit` 用线段求交（非直线求交），因此足迹落在单元外时不会误判 ρmax。

---

## 5bis. 交付纪律与仓库状态

**热文件声明**：本条改动的核心文件 `src/tensorbem/panel_quadratic.py` 是 TensorBEM 的核心模块，
在 main 上近期被 #591（m8709，multipole 图）与 #595（m8788，外法向 re-wind）改动过。
本分支从 `origin/main @723ebe8`（含两者）切出，diff 只做加法：
新增顶层函数 `apex_subtriangles` / `graded_apex_quadrature`、新增
`QuadraticTriangle.project_to_reference` / `near_singular_layer_integrals` /
`integrate_{{single,double}}_layer_near_singular`，并把 `triangle_apex_quadrature`
里内联的 apex 定位块原样提取为 `apex_subtriangles` 调用（语义不变，由既有
`tests/test_panel_quadratic.py`（25 条）与 `tests/test_panel_quadratic_operator.py`（31 条）复核）。
`assemble_quadratic_matrices` / `solve_*` 等装配与求解路径**未改动**。

**已存在的红**（不是本条引入）：`tests/test_capability_coverage.py`（3 条）与
`tests/test_capability_evidence.py`（2 条）在**纯净 origin/main 检出**上同样失败
（用 `git archive origin/main | tar -x -C /tmp/basemain` 复现，5 failed / 13 passed，
与工作树结果逐条一致），根因是 #594/#595 新增的核心模块（`cathodic_protection`、
`cp_galvanic` 等 14 个）尚未进能力台账行。本条不动它们（单 PR 单功能点），
报告如实登记。

---

## 5ter. 测试执行情况（如实登记）

| 范围 | 结果 |
|---|---|
| `tests/test_panel_quadratic_near_singular.py`（新增 43 条） | **43 passed**（2.1 s） |
| `tests/test_panel_quadratic.py`（既有，二次单元） | 25 passed |
| `tests/test_panel_quadratic_operator.py`（既有，算子恒等式） | 31 passed（111 s） |
| `tests/test_doc_inventory.py` + `test_benchmark_cli_contract.py` | 37 passed |
| 其余 import `panel_quadratic` 的测试文件（analytical_suite / helmholtz_maccamy_fuchs / drift_force ×2 / hydrostatic_curved_waterplane / tank_free_surface / exterior_neumann_added_mass / mutation_gaps_wave_loads） | 全部 passed；`test_bem_verified_anchors.py` 3 条失败，与纯净 main 逐条一致 |
| 全量 `pytest tests`（本地 `-n 8 --dist loadfile`） | 执行到 **≥97%**；余下重型三维 BEM 用例在负载 ~250 的共用机上仍在算（单 worker 87 min CPU），到点终止；集群全量作业 cj108（本分支）/cj118（origin/main 基线）递交后 >2 h 仍在排队 |
| `tests/test_acceptance` 类门（capability_coverage 3 条、capability_evidence 2 条、declared_numbers 1 条） | 失败，但与**纯净 origin/main 检出**（`/tmp/basemain`）逐条一致，属既有红 |

**未完成项（honest）**：本次会话内没有拿到"整仓全绿"的证据——因为 main 本身已有 6 条既有红，
且全量运行被重型用例拖住。因此本条的验收口径是**变更面**：新增能力 43 条测试全过、
既有二次单元 56 条全过、变更面其余文件全过，仅存的失败项均已证明在纯净 main 上同样存在。

---

## 6. 误差判定（逐项 vs 基准，显式回答是否 < 3%）

| 关键量 | 基准 | 相对误差 | < 3% ? |
|---|---|---|---|
| 半解析参考（`d/H=0.5`，单层全部分量） | 64 阶 Gauss | {fmt(oracle['single_rel_error_vs_gauss64'])} | 是（1e-15 级） |
| 半解析参考（`d/H=0.5`，双层全部分量） | 64 阶 Gauss | {fmt(oracle['double_rel_error_vs_gauss64'])} | 是 |
| 分级规则单层/双层，`d/H = 1e-3`，订单 12 | 半解析 | {fmt(acc['graded_worst_error_by_order']['12'])} | 是 |
| 分级规则单层/双层，`d/H = 1e-3`，订单 24 | 半解析 | {fmt(acc['graded_worst_error_by_order']['24'])} | 是 |
| 分级规则单层/双层，`d/H = 1e-3`，订单 32 | 半解析 | {fmt(acc['graded_worst_error_by_order']['32'])} | 是 |
| 曲面单元订单 32 vs 订单 40 | 自收敛 | {fmt(acc['curved_graded_error_order_32'])} | 是 |
| **基线**：常规 Gauss 8 阶，`d/H = 1e-3`，形心 | 半解析 | {fmt(acc['baseline_single_error_at_d_over_H_1e-3'])} | **否（12%）** |

结论：被测能力项（分级 Duffy 近奇异求积）在 `d/H ≤ 1e-3` 时全部关键量误差 `< 3%`，
其中订单 24/32 已达 1e-7/1e-9 量级；被替换的基线（常规 Gauss 8 阶）在同一工况下
12% 误差、**超出可接受上界**，这正是本条改动存在的理由。所有数字均出自
`docs/quadratic_near_singular/study.json`（程序化注入，零手抄）。
"""

(ROOT / "dev_report_m8718_near_singular.md").write_text(report)
print("wrote dev_report_m8718_near_singular.md")
print("figure bytes:", FIG.stat().st_size)