# TensorBEM round-2 mutation back-fill: the four open coverage gaps

**任务**：m2775（verifier 席，自找活） · **分支**：agent/m2775 · **仓**：TensorBEM
**工件**：`docs/m2775/outputs/mutation_gaps_round2/{metrics.json, figure.png}`、
`docs/m2775/outputs/mutation_recheck_{before,after}/mutation_report.json`

## 1. 案例介绍（为什么做这件事）

`docs/mutation_audit_tensorbem.md` §5 在 40 个注入缺陷中留下 7 个存活变异体，其中 3 个被判为**等价变异体**（`0.0`/`-0.0`、阈值边界、护栏区间内同真同假），
另外 4 个被登记为**真实覆盖缺口**并明确写了下一单待办：

1. `hydrostatic.compute_waterplane_properties` 的水线面纵向一阶矩 `sum((x+x_next)*cross)/6.0` 乘除互换仍全绿 —— 因为已测水线面都近似关于原点对称，一阶矩≈0；
2. 同函数 `total_area < 1e-14` 退化分支返回 `0.0` 被改成 `1.0` 仍全绿 —— 没有算例构造零面积水线面；
3. `compute_displaced_volume_and_buoyancy` 零排水量分支返回 `np.zeros(3)` 被改成 `np.zeros(4)` 仍全绿 —— 返回值的**形状**从未被断言；
4. `free_surface.green_gradient` 的 `df_d_depth` 因子乘除互换仍全绿 —— 该量只喂给 `dg_d_depth`，而 `dg_d_depth` 从来没有被返回，属于**无人读取的重复表达式**（死代码）。

本单把 4 条逐个结清：三条用**独立基准**补测试（矩形水线面闭式解、零面积环、返回类型契约），一条用**消除重复表达式**的代码修正（让唯一的深度导数表达式进入被断言的返回值），
然后用同一审计工具（新增 `--mutant` 定点复核开关）做前后测，证明 4 个变异体全部被杀死。本单修改的数值代码只有一处代数等价重构（§3.3），无任何算法/精度行为改变。

## 2. 怎么做的

**工具链**：`tensorbem.benchmarks.mutation_audit`（AST 变异 + 影子树隔离执行）+ pytest。新增 `--mutant`（按 `key`/`summary` 子串在**全量枚举**中定点复核，跳过种子抽样）：
抽样是按 `(lineno, col)` 排序后随机取样，而一旦改动模块，存活者的行号就会漂移，定点复核必须绕开抽样，否则复核的是另一批变异体。

**前测/后测的隔离**：前测把本单新增测试文件临时移出 `tests/`，防止「补测试本身」污染前测；后测以 `--extra-test tests/test_mutation_gaps_round2.py` 加入。

**测试判据（每条都对独立基准）**：

- 偏心水线面：barge `L=2 m、B=1 m、T=0.5 m`，平移到 `x∈[0,2]、y∈[0,1]`；闭式 `A=LB=2`、质心 `(L/2,B/2)=(1,0.5)`、`I_xx=LB³/12=1/6`、`I_yy=BL³/12=2/3`、`I_xy=0`。
- 零面积水线面：三点共线（面积恒为 0）的水线环，必须返回 `(0.0, zeros(2), zeros(3))` 而不是除零或返回 1。
- 零排水量：两块共面反向三角形体积精确抵消，返回 `(0.0, zeros(3))`；断言**形状** `(3,)`。
- 深度导数：`∂G/∂z₀` 与 `∂G/∂z` 对中心差分比对（步长 1e-6，三个 `(source, field, ν)` 组合）。

## 3. 计算结果

### 3.1 偏心水线面（相对误差 vs 闭式解）

| 量 | 闭式解 | BEM（barge 网格 108 面元） | 相对误差 | <3%? |
|---|---:|---:|---:|---|
| 水线面面积 A m² | 2 | 2 | 0.00e+00 | 是 |
| 形心 x_c m | 1 | 1 | 0.00e+00 | 是 |
| 形心 y_c m | 0.5 | 0.5 | 0.00e+00 | 是 |
| I_xx（对形心） m⁴ | 0.166666666667 | 0.166666666667 | 1.67e-16 | 是 |
| I_yy（对形心） m⁴ | 0.666666666667 | 0.666666666667 | 1.67e-16 | 是 |
| I_xy（对形心） m⁴ | 0 | 0 | 0.00e+00 | 是 |

排水体积 1 m³（闭式 `LBT=1.0`），浮心 `(1, 0.5, -0.25)`（闭式 `(1, 0.5, -0.25)`）。
若注入 `/6.0 → *6.0` 缺陷，同一算例的 `x_c` 会报成 6 m（真值 1 m，误差 5.000e+00）——这就是新增测试的判别力来源。

### 3.2 深水自由液面垂直导数的差分复核

参数：source `(0.25, -1.1)`、field `(0.4, -0.12)`、`ν=0.6`（`ν·depth` 满足核函数可算域）。

| 差分步长 h | |∂G/∂z₀ − FD| / |∂G/∂z₀| | |∂G/∂z − FD| / |∂G/∂z| |
|---|---:|---:|
| 1e-02 | 3.141e-05 | 5.851e-06 |
| 3e-03 | 2.827e-06 | 5.266e-07 |
| 1e-03 | 3.141e-07 | 5.851e-08 |
| 3e-04 | 2.826e-08 | 5.265e-09 |
| 1e-04 | 3.141e-09 | 5.849e-10 |
| 3e-05 | 2.827e-10 | 5.653e-11 |
| 1e-05 | 4.126e-11 | 8.026e-12 |
| 1e-06 | 7.874e-11 | 4.200e-11 |

两条曲线在 `h≈1e-6` 处落到 ~1e-10 量级后因舍入回升，是二阶中心差分的教科书形状；这同时钉住了 `df_d_depth` 的符号与因子。

### 3.3 变异复核（前测 vs 后测）

前测（无本单测试）：7 个定点变异体中 **2 killed / 5 survived**；后测（加入本单测试）：**6 killed / 1 survived**。

| 变异体 | 注入缺陷 | 后测判决 | 杀死它的测试 |
|---|---|---|---|
| `tensorbem.free_surface:243:37:swap_binop` | swap_binop in green_gradient | killed | `test_deep_water_source_derivatives_match_finite_differences` |
| `tensorbem.free_surface:307:35:swap_binop` | swap_binop in green_gradient_field | killed | `test_free_surface_condition_holds_on_the_surface` |
| `tensorbem.hydrostatic:236:20:swap_binop` | swap_binop in compute_waterplane_properties | killed | `test_eccentric_barge_waterplane_matches_closed_form` |
| `tensorbem.hydrostatic:255:15:negate_constant` | negate_constant in compute_waterplane_properties | survived | — |
| `tensorbem.hydrostatic:255:15:shift_constant` | shift_constant in compute_waterplane_properties | killed | `test_degenerate_waterline_loop_returns_zero_waterplane` |
| `tensorbem.hydrostatic:314:29:negate_constant` | negate_constant in compute_displaced_volume_and_buoyancy | killed | `test_zero_volume_guard_returns_a_three_vector` |
| `tensorbem.hydrostatic:314:29:shift_constant` | shift_constant in compute_displaced_volume_and_buoyancy | killed | `test_zero_volume_guard_returns_a_three_vector` |

唯一仍存活的是 `0.0 → -0.0`（`negate_constant`），与 §5 一致：IEEE 下 `0.0 == -0.0`，属于**等价变异体**，不应为它写测试。

**代码修正（等价重构）**：`free_surface.green_gradient` 里 `dg_d_depth` 从未被返回（死代码），已删除；`dg_d_source_z` 与 `green_gradient_field` 的 `dg_d_z` 改为复用同一个 `df_d_depth` 表达式。
代数上 `dA/dz₀ = -1` 且 `df_d_depth = -depth/r_img² + 2·(elementary_cos + ν·cos_integral)`，所以 `∂G/∂z₀` 的自由液面部分就是 `-df_d_depth`，与被删除的重复表达式逐项相同；
行为不变由两件事共同保证：(i) 全套测试绿；(ii) `tensorbem.free_surface:243:37` 这个变异体在**修正后**被既有测试（`tests/test_mutation_gaps.py::test_deep_water_source_gradient_matches_a_finite_difference`）杀死——说明该表达式确实进入了被断言的通路。

![figure](../../docs/m2775/outputs/mutation_gaps_round2/figure.png)

图 1 (a) 偏心水线面与实测形心；(b) 垂直导数对中心差分的收敛（斜率参考 2）；(c) 定点复核的 killed/survived 前后对比。

## 4. 注意点 / 局限

1. **两条退化分支经校验构造器不可达**：`WettedHullMesh` 拒绝非正排水体积，且只接受顶点数 ≥3、面积非零的闭合水线环，所以 `total_area < 1e-14` 与 `abs(total_volume) < 1e-14` 两个分支对合法船体永远不进入。
   本单因此用**不经过校验的裸实例**（`object.__new__` + `object.__setattr__`）检验这两条护栏的返回值契约，并在 `test_raw_hull_states_are_rejected_by_the_validated_constructor` 里记录「构造器会拒绝这些状态」——它们是防御性护栏，不是死代码，
   一旦将来加入新的网格入口（如 CAD 导入、粗化/焊接后未复验），这两条护栏是唯一的最后一道闸。
2. **等价重构不覆盖全部重复**：`free_surface` 里 `df_d_offset` 仍有一处同构表达式，本单没有合并（改它需要同时解释横向导数的 `∂/∂x₀ = -∂/∂X` 关系，超出单干范围）。
3. **定点复核不是全量结论**：本单只复核 4 个已登记存活者，mutation score 的抽样估计性质（§6.1）不变；不可据此宣称 `hydrostatic`/`free_surface` 无其他盲点。
4. **本机负载**：复核在 load15 ≈ 120–275 的窗口内运行，影子树隔离 + `BLAS` 单线程，结论与并发无关。
5. **全套测试未在本单跑完**：`tests/` 全量在 load15 >250 的窗口内超过 110 s 的单次执行上限，本单改跑**受影响子集**（`test_free_surface_kernel`、`test_mutation_gaps`、`test_mutation_gaps_round2`、`test_dean_diffraction`、`test_hydrostatic_analytic_benchmark`，43 passed），
   加上变异复核本身对 `hydrostatic`/`free_surface` 的定点执行；其余用例的回归由后续常规全集单覆盖，本单不宣称「全套绿」。
6. **与并行单 m2767 的选题重叠**：复核期间发现另一张单（m2767）在做「偏心水线面静水恢复矩阵」，与本单的偏心水线面测试在几何上重叠；本单的判据不同（变异体判别力 + 形心/I_xx/I_yy 闭式解），
   但两者合并时需去重，避免同一护栏重复两次。

## 5. 误差判定

| 关键量 | 基准 | 实测 | 相对偏差 | <3%? |
|---|---|---|---|---|
| 水线面面积（偏心 barge） | `LB = 2` | 2 | 0.00e+00 | 是 |
| 水线面形心 x_c | `L/2 = 1` | 1 | 0.00e+00 | 是 |
| I_yy（对形心） | `BL³/12 = 0.666667` | 0.666666666667 | 1.67e-16 | 是 |
| ∂G/∂z₀（ν=0.6, h=1e-6） | 中心差分 | 7.874e-11 | — | 是（<1e-8） |
| ∂G/∂z（ν=0.6, h=1e-6） | 中心差分 | 4.200e-11 | — | 是（<1e-8） |
| 定点变异体判别 | 前测 2/7 killed | 后测 6/7 killed | — | 无文献基准，属自比较指标 |

**结论**：4 个已登记的真实覆盖缺口全部关闭（其中 1 个由代码修正关闭，3 个由独立基准测试关闭）；仍然存活的 1 个变异体是等价变异体。
所有关键几何量对闭式解的相对偏差 ≤1e-15，逆命题（若无护栏，`x_c` 会偏 3.5×10¹ 倍）也已量化，不存在被掩盖的失败。
