# CASE-BEM-LAPLACE-CYLINDER：近壁判据复核（验证欠账）与两条被实测否证的前置声明

**锚点** `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，ref 解析复势，容差 `<3%`）
· **仓** TensorBEM · **分支** `agent/m22884` · **状态**：锚点 `open`，本轮**不**认领完成
（锚点验收量由路线 D/N/G 给出，与本文三条近壁判据无关，见 §5）。

本文回答 m22521 报告里唯一 `not_run` 的欠账：m22235 的重生成（cj3969）报出的
3 条近壁失败判据到底是不是真缺陷。结论：**1 条不可复现（代码已修）**、
**2 条是判据本身写错（方向/口径错），实测数字推翻了判据里的物理断言**。两条错判据
已按实测改写为可测且方向正确的不变量，否证数字原样留档在下面第 3、4 节。

## 1. 复核对象与证据身份

| 项 | 值 |
| --- | --- |
| 被复核的记录 | `runs/m22235/cj3969_results.json`（commit `655f6ce7`），`overall=FAIL`，49 条判据 6 条失败 |
| 其中本文负责的 3 条 | `near_wall_guard_rejects_inside_window`、`near_wall_linear_window_extends_to_half_panel`、`near_wall_exclusion_floor_is_faceting` |
| 复核代码 | 本工作树 `src/tensorbem/benchmarks/case_cylinder_stream.py`（main `cc889c2b` + 本轮改动） |
| 复核方式 | 直接调用模块内的近壁子研究（不触发完整阶梯），单层稠密解为秒级；见每节的复现命令 |
| 未复核项 | 完整阶梯端到端重生成（队列常态 70+ 位次，见 §6） |

另外 3 条 m22235 失败判据属远场多极拟合，已由 m22521 复核为**不可复现**（纯度实测 1e-14，
要求 <1e-9），与本文结论一致，不再重复。

## 2. `near_wall_guard_rejects_inside_window`：不可复现（现码通过）

判据：`minimum_clearance_factor=4` 的守门探针必须拒绝 `δ = 1 h_max` 的目标。
按现码对三层各建内接正 N 边形、解 Neumann、在 `r = a + h_max` 处求值：

| N | `h_max` | 守门结论 |
| --- | --- | --- |
| 32 | 0.19603 | `ValueError`（拒绝） |
| 64 | 0.09808 | `ValueError`（拒绝） |
| 128 | 0.04906 | `ValueError`（拒绝） |

`mesh.boundary_clearance` 在 N=32 上量得 0.20085 > 0（探针确实在域外），阈值 4·h_max = 0.7841
——拒绝是 `tensorbem/solver.py::_screen_exterior_targets` 的既有行为，本轮未改该函数。
**判定**：cj3969 的失败来自那一份工作树（早于 m22521 的入口点/门禁改动），不是 main 缺陷；
判据保留不变。

复现：

```bash
PYTHONPATH=src python3 - <<'EOF'
import numpy as np
from tensorbem.geometry import regular_polygon
from tensorbem.solver import solve_laplace_exterior_neumann, evaluate_exterior_field
for N in (32, 64, 128):
    m = regular_polygon(N, 1.0); c = m.centers
    r = np.hypot(c[:,0], c[:,1]); nd = (c[:,0]/r)/r**2
    sol = solve_laplace_exterior_neumann(m, nd); h = float(np.max(m.lengths))
    th = np.arctan2(c[:,1], c[:,0])
    p = np.column_stack(((1+h)*np.cos(th), (1+h)*np.sin(th)))
    try:
        evaluate_exterior_field(m, sol, p, minimum_clearance_factor=4.0); print(N, "NO RAISE")
    except ValueError:
        print(N, "rejected")
EOF
```

## 3. `near_wall_linear_window_extends_to_half_panel`：**否证**

原判据断言：一阶矩（线性密度）基把 1% 窗口从 `δ = 2 h_max`（常值面板）收到 `δ = 0.5 h_max`，
且以**跨层最差**口径判定（`usable_clearance_over_h_max` 取三层里最差的方向、最差的层）。

实测（径向面板中点法向探针，最差方向相对速度误差；两基同一网格同一右端项）：

| N | `h_max` | 常值 @2h | 常值 @1h | 常值 @0.5h | 线性 @2h | 线性 @1h | 线性 @0.5h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.19603 | 6.609e-03 | 1.433e-02 | 3.689e-02 | 8.138e-03 | **1.579e-02** | **1.639e-02** |
| 64 | 0.09808 | 3.600e-03 | 7.851e-03 | 2.696e-02 | 4.465e-03 | 8.239e-03 | 3.920e-03 |
| 128 | 0.04906 | 1.888e-03 | 4.387e-03 | 2.212e-02 | 2.349e-03 | 4.123e-03 | 2.669e-03 |

1% 容差下的可用窗口（从远到近扫描、遇到第一个越界即停）：

- **跨层最差**：常值 `2 h_max`，线性 `2 h_max` —— **没有变宽**（原断言被否证）；
- **最细层（N=128）单独**：常值 `1 h_max`，线性 `0.5 h_max` —— 原断言的数字只在单层成立。

粗层之所以不跟随：线性基的弧长斜率把密度外推到顶点处的不连续上，N=32 时该外推误差
（1.6e-2 量级）压过了它相对常值基的增益（同点常值 1.4e-2→3.7e-2 之间）。
因此「窗口收到 0.5 h_max」是一条**最细网格结论**，写成跨层断言不成立。

**处置**（不删结论，拆成两条可测判据）：

- `near_wall_linear_window_extends_the_finest_window`：最细层窗口 `≤ 0.5 h_max`（实测 0.5，通过）；
- `near_wall_linear_window_does_not_widen_the_worst_case_window`：跨层窗口 `≤` 常值跨层窗口
  （实测 2.0 ≤ 2.0，通过），docstring 里写清原断言与否证数字。

## 4. `near_wall_exclusion_floor_is_faceting`：**方向写反**

原判据断言：把探针参考面从圆周改到内接折面后，「误差必须塌缩大部分」，
判据式 `surface_referenced ≤ circle_referenced / 3`。

几何：折面面板中点在半径 `a cos(π/N) < a`，即**在圆周内侧**，偏移
`a(1-cos(π/N)) ≈ h²/8`。同一名义 `δ = 0.1 h_max` 从折面量起时真实间隙**变小**
（少了该偏移），误差只能**变大**。实测（表示误差，迹取解析值以隔离求解器）：

| N | 偏移/h_max | 偏移/δ | 常值 圆→折面 | 线性 圆→折面 | 线性放大 |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.02457 | 0.2456 | 0.1799 → 0.1984 | 0.1045 → 0.1223 | 1.170 |
| 64 | 0.01228 | 0.1227 | 0.1773 → 0.1863 | 0.1189 → 0.1278 | 1.075 |
| 128 | 0.00615 | 0.0614 | 0.1757 → 0.1801 | 0.1261 → 0.1306 | 1.036 |

放大方向与几何一致（全部 >1），而原判据要求 <1/3，**几何符号被写反**；cj3969 实测
1.04×（反向）正是这一点的表现，不是缺陷。

折面几何真正能解释的是**增长率**：单位偏移的放大 `(放大-1)/(偏移/δ)` 分别为
0.694 / 0.614 / 0.590（粗→细），**与层几乎无关**，且放大量本身随 `h` 单调下降
（1.170 → 1.036）——这正是「偏移是 `O(h)` 的一阶量」的定量形式。

**处置**（保留归因，改判据）：
`near_wall_exclusion_floor_is_faceting` 现断言
(a) 每层 `surface > circle`（放大 >1），
(b) `(放大-1)/(偏移/δ) ∈ [0.3, 1.2]`（实测 0.59–0.69，带很宽仍能抓错方向），
(c) 放大随 `h` 单调下降。三条都取自同一批实测数字。

## 5. 与锚点验收的关系（不粉饰）

锚点验收量是**外域速度场相对解析复势的误差 <3%**，由路线 D/N/G 在最细层的 `δ = 0.25 a`
（多面板之外）给出；本文三条判据约束的是 `δ ≤ 2 h_max` 的近壁可用性与误差归因，
**既不构成也不阻断锚点验收**。锚点在总账中仍为 `open`：本仓库的提交记录
（`benchmarks/verified/case_bem_laplace_cylinder/results.json`）仍停在 38 条判据、无
`provenance`，端到端重生成未在本单窗口内落盘（§6）。

## 6. 注意点与未完成项

- **未完成（not_run）**：带本轮判据的完整阶梯端到端重生成。已递交 `cj4455`（旧码，复现用途）
  与 `cj4460`（本工作树新码），递交时队列位次 69/71，在飞 85+ 作业；本机 `run_shell`
  单次 120 s 硬顶而完整阶梯 300 s 量级，本机不可替代。**因此提交记录未更新，记录陈旧这笔账仍然开着。**
- 本机可复核的部分（近壁子研究在 `ladder=(32,64)` 上跑通）：新字段全部按预期产出
  （`usable_linear=2.0`、`usable_const=2.0`、`finest_linear=0.5`、放大 1.170/1.075 且单调下降、
  单位偏移放大 0.694/0.614 落在带内、守门三层全拒绝、3% 边界三层全部定位、窗口余量 4.23×）。
- 常量含碳：判据改写只**放宽**了被否证的那部分（跨层 0.5 h_max、塌缩 3 倍），
  同时**新增**了三条原来没有的实测不变量（最细层窗口、跨层窗口不宽于常值、放大率与偏移的定量关系），
  净效果是把「未测过的期望」换成「测过的断言」，不是把红改绿。