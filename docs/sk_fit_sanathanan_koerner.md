# Sanathanan-Koerner 阻尼有理拟合：从「有名字没迭代」到「极点重定位达 3 % 门」

模块 `tensorbem.sk_fit` · 工件 `docs/sk_fit_study/` · 本轮任务 m17837（PR#692 重开第 2 轮）
逐轮开发记录见 `docs/dev_report_m16042_sk_iteration.md`。

## 1 案例介绍

状态空间 / 时域链路要的阻尼基是实衰减率指数和：

`K(t) = Σ_k c_k e^{-α_k t}  ⇔  B(ω) = d + Σ_k c_k α_k/(α_k² + ω²)`

其中 `B(ω)` 是频域辐射阻尼。TensorBEM 的 `state_space` / `time_domain` 链路
（状态空间实现、自由衰减、记忆核两向闭合）都吃这个基。`tensorbem.sk_fit`
负责用 Sanathanan–Koerner（SK, 1963）迭代把一条 `B(ω)` 拟合成这组 `(α_k, c_k, d)`。

**为什么要查它（两轮）**：

* **第 1 轮**：能力证据门 `tests/test_capability_evidence.py` 报出 `tensorbem.sk_fit`
  为「无能力行、亦无豁免理由」的未认领模块；代码事实是极点固定在
  `logspace(0.3ω_min, 3ω_max, n_poles)`、只解一次加权线性最小二乘、`max_iter`/`tol`
  接收后**从未使用**、返回 `n_iter=1, converged=True`——函数名与文献署名指向一个
  迭代法，实现里没有迭代。
* **第 2 轮（本轮）**：第 1 轮把 SK 迭代补上后，在**实测** Hulme 半球曲线上仍停在
  峰值的 17–49 %，当时写作「实极点基的模型类下限」。本轮把这句话当假设去测：若极点
  位置本身没被优化，则 17–49 % 是**方法**误差而非模型类误差。测量结果支持后者——
  允许极点位置移动后，同数据同阶次降到 2.0e-04（峰值 0.02 %），3 % 门从 6 阶起满足。
  第 1 轮的 honest-FAIL 因此**撤回**（不是粉饰，而是被新的测量推翻，见 §4）。

**两条被测曲线（参数见 §2 与工件 `truth` 字段）**

| 记号 | 曲线 | 是否可被实衰减率基表征 | 用途 |
|---|---|---|---|
| A `representable` | 合成：`α=(0.5,1.0,2.5)` rad/s、`c=(1.0,−1.2,0.6)`、`d=0.05`，ω∈[0.02,50] 300 点 | **是**（构造上精确） | 反问题可解性判据 |
| B `hulme_hemisphere` | 浮半球 heave 实测参考阻尼曲线（Hulme 1982 解析解，仓库夹具 `tests/data/time_domain_hemisphere_sweep.json`，201 点，B 跨 1.7e3 : 2.1e-2 四个量级） | **否**（自由面记忆核有代数尾，非有限实指数和） | 精度门判据、方法 vs 模型类误差分离 |

A 的衰减率刻意取在基线对数网格**不会**落点的位置（网格 0.006 / 0.075 / 0.95 / 12 / 150 …），
因此 A 上的误差是方法误差，不是网格运气。

## 2 怎么做的

1. **SK 迭代**（`tensorbem.sk_fit._sk_iterate`）。在 `u = ω²` 上写
   `B(u) ≈ A(u)/D(u)`，`D(u) = 1 + d₁u + … + d_M u^M`；每个频率点 `j` 解线性化方程
   `A(u_j) − B_j·(D(u_j) − 1) = B_j`，权重 = 数据相对权重 × `1/|D_{i−1}(u_j)|`，
   解出后更新分母并迭代，直到分母相对变化小于 `tol` 或加权残差相对变化小于 1e-12。
   列归一化 + `lstsq`（设计矩阵条件数随阶数平方增长，不归一化会直接失稳）。
   迭代记录**最优 iterate**（加权残差最小者）：实测在刚性真实曲线上 SK 会 2-周期极限环。
2. **极点投影到物理基**（`_real_pole_projection`）：`D` 的实负根 `u_k = −α_k²/u_ref`
   精确映射为衰减率；共轭复根无法表示，投影到其模（`α=|u|^½`），并把复根对数记进报告，
   让「模型类误差」与「拟合误差」分开。**这一步正是第 2 轮找到的瓶颈**：共轭对被投影成
   *重合* 的实极点，SK 根在复平面上很好，投影后落点很差。
3. **物理基极点重定位（本轮新增，`_refine_poles`）**：对 `log(α_k)` 做有界下山单纯形
   （Nelder–Mead，无随机状态，同种子同迭代），目标函数就是 `_solve_linear` 已在对残基
   最小化的**同一个加权残差平方和**，因此残基永远是线性最小二乘解、`d ≥ 0` 的物理约束
   仍然生效，只有 `n_poles` 个变量在动。种子两个：SK 投影极点、改前固定网格；各带
   1 次热重启（`_REFINE_RESTARTS=2`），预算 `refine_maxfev=1500`（每次单纯形）。
   搜索区间 `α ∈ [0.05 ω_min, 50 ω_max]` 外一律 clip，避免极点跑到数据带外去拟合噪声。
4. **候选集选择（无回归不变量）**：候选 = {SK 投影极点, 固定网格, 两套重定位极点}，
   全部用同一约束最小二乘解残基，按**实测** `max|ΔB|/max|B|` 取优。返回值因此
   在扫描的每一阶上都不劣于任一候选，`pole_source` 如实记录赢家与种子
   （`refine_start`）。`refine_maxfev=0` 关闭第 3 步，逐位复现第 1 轮结果。
5. **收敛「测得」而不是「断言」**：单纯形本身在本类平坦目标上永远不满足自己的
   simplex-size 判据（`success=False`），所以本模块**不转发**这个标志；改为在返回极点上
   做单极点 ±5 % 对数探针（`_probe_local_optimality`），把最大可改进量记成
   `refine_probe_gain`，`refine_converged = (probe_gain ≤ 1e-3)`。即「收敛」=
   「不存在使加权残差下降超过 0.1 % 的单极点移动」，是一条可复算的测量。
6. **基线保留为可测函数** `fixed_grid_damping_fit`（改前行为逐字保留），
   使「改进了多少」成为可复算的数字而非声明。
7. 工具链：`python -m tensorbem.benchmarks.sk_fit_verification --out-root docs/sk_fit_study`
   （`--write-doc` 直接重写本表）；`pytest tests/test_sk_fit.py` 重跑全 study 并逐行比对
   工件与本表。依赖 numpy + scipy.optimize（+ matplotlib 出图）。

## 3 计算结果（程序化注入 `docs/sk_fit_study/results.json`）

### 3.1 阶次扫描（三个估计器 + 精度门）

`returned` = 本模块返回值（默认 `refine_maxfev=1500`）；`SK unrefined` = 关闭重定位的
第 1 轮行为；`fixed-grid baseline` = 改写前行为。

<!-- sk-fit-table:begin -->
| case | order | returned max err | fixed-grid baseline | SK unrefined | SK + pole refinement | gain vs unrefined | pole source | refine seed | probe gain | <3 %? |
|---|---|---|---|---|---|---|---|---|---|---|
| representable | 2 | 5.548e-02 | 8.894e-01 | 7.839e-01 | 5.548e-02 | 14.1x | refined | sk | 0.000e+00 | no |
| representable | 3 | 1.225e-15 | 8.072e-01 | 1.225e-15 | 1.225e-15 | 1.0x | sk | none | 0.000e+00 | yes |
| representable | 4 | 4.084e-16 | 2.395e-01 | 3.472e-15 | 4.084e-16 | 8.5x | refined | sk | 0.000e+00 | yes |
| representable | 6 | 6.127e-16 | 1.534e-01 | 6.127e-16 | 6.127e-16 | 1.0x | sk | none | 0.000e+00 | yes |
| representable | 8 | 4.084e-16 | 1.447e-01 | 2.042e-15 | 4.084e-16 | 5.0x | refined | sk | 0.000e+00 | yes |
| representable | 10 | 6.127e-16 | 8.509e-02 | 1.838e-15 | 6.127e-16 | 3.0x | refined | sk | 0.000e+00 | yes |
| representable | 12 | 6.127e-16 | 4.434e-02 | 3.165e-15 | 6.127e-16 | 5.2x | refined | sk | 0.000e+00 | yes |
| hulme_hemisphere | 2 | 9.805e-01 | 9.997e-01 | 9.976e-01 | 9.805e-01 | 1.0x | refined | sk | 0.000e+00 | no |
| hulme_hemisphere | 3 | 6.491e-01 | 9.902e-01 | 9.902e-01 | 6.491e-01 | 1.5x | refined | grid | 0.000e+00 | no |
| hulme_hemisphere | 4 | 3.802e-01 | 9.785e-01 | 9.785e-01 | 3.802e-01 | 2.6x | refined | grid | 0.000e+00 | no |
| hulme_hemisphere | 6 | 1.487e-02 | 9.230e-01 | 9.023e-01 | 1.487e-02 | 60.7x | refined | sk | 1.506e-04 | yes |
| hulme_hemisphere | 8 | 2.418e-03 | 4.921e-01 | 4.921e-01 | 2.418e-03 | 203.5x | refined | sk | 1.001e-04 | yes |
| hulme_hemisphere | 10 | 5.081e-04 | 2.241e-01 | 2.241e-01 | 5.081e-04 | 441.1x | refined | sk | 4.558e-02 | yes |
| hulme_hemisphere | 12 | 2.046e-04 | 1.684e-01 | 1.684e-01 | 2.046e-04 | 822.8x | refined | sk | 1.170e-02 | yes |
<!-- sk-fit-table:end -->

要点：

* **A 曲线、3 阶**：SK 用 3 次迭代把三个衰减率、三个残基与渐近值全部复原，
  误差 `1.225e-15`；同阶改前基线 `8.072e-01`（峰值 81 %）。差别只在有没有极点重定位。
* A 曲线 2 阶（模型太小、刻意留的**不达标点**）：重定位仍把误差从 `7.839e-01` 拉到
  `5.548e-02`（14.1×），但 5.5 % 仍 > 3 %，表里如实记 `no`——重定位是精度杠杆，
  不能替代足够的阶数。
* B 曲线（实测）：**6/8/10/12 阶全部进 3 % 门**，6 阶 `1.487e-02`（60.7×）、
  8 阶 `2.418e-03`（203.5×）、10 阶 `5.081e-04`（441.1×）、12 阶 `2.046e-04`（822.8×）；
  同一行 `SK unrefined` 是 `9.023e-01 / 4.921e-01 / 2.241e-01 / 1.684e-01`。
* B 曲线 2/3/4 阶仍不达标（`9.805e-01 / 6.491e-01 / 3.802e-01`）：4 阶以内实极点基
  连曲线的量级结构都装不下，重定位只能改善 1.5–2.6×。这是阶数不足，不是算法失效。
* `regression_free = true`、`never_worse_than_unrefined = true`：两个可机检不变量，
  由 `tests/test_sk_fit.py::test_never_worse_than_the_fixed_grid` 钉住。

### 3.2 迭代与重定位本身的行为

| 观察 | 数字（工件内逐行可查） |
|---|---|
| A 曲线 3 阶 SK 收敛所需迭代 | 3 次 |
| B 曲线 8/10/12 阶 SK | 200 次上限内未满足停止判据（`sk_converged = no`），返回残差最小 iterate |
| `max_iter=1` | `n_iter=1`、`converged=False`，误差劣于多次迭代（参数不是装饰） |
| 单极点 ±5 % 探针（`refine_probe_gain`） | B 6/8 阶 `1.5e-04 / 1.0e-04`（局部最优）；B 10/12 阶 `4.6e-02 / 1.2e-02`（**未**局部最优，如实记 `refine_converged=no`） |
| 重定位预算 | 每次 fit 消耗 `4 × 1500 = 6000` 次加权残差评估（两种子 × 两轮），单次 fit 约 2 s（B 曲线 12 阶） |
| 确定性 | 无 RNG；`test_refinement_is_deterministic` 逐位比对两次调用的极点与残基 |
| 关闭开关 | `refine_maxfev=0` → `pole_source ∈ {sk, grid}`、`refine_evaluations=0`、`refined_pole_rel_error = nan`，逐位复现第 1 轮 |

SK 在 B 上的极限环是文献已知病态（SK 对强刚性/宽带数据无收敛保证），本模块不粉饰：
`sk_converged` 报停止判据是否满足；重定位的收敛则用探针测量，两者都进报告。

### 3.3 图

`docs/sk_fit_study/sk_fit_verification.png`：左右两栏（A、B），横轴阶数、纵轴
`max|ΔB|/max|B|`，四条线 = 本模块返回值（SK + 重定位）、SK 不重定位（第 1 轮）、
改前固定网格基线、无约束有理模型上限，另加 3 % 精度门横线。B 栏里第 1 轮曲线与
本轮返回值之间 2–3 个量级的落差，就是极点重定位本身的贡献。

## 4 误差判定（是否 <3 %？）

| 关键量 | 基准 | 本实现 | 相对误差 | <3 %？ |
|---|---|---|---|---|
| A 曲线 3 阶衰减率 | (0.5, 1.0, 2.5) rad/s | 复原 | 3.6e-16 | ✅ |
| A 曲线 3 阶残基 | (1.0, −1.2, 0.6) | 复原 | <1e-8（绝对） | ✅ |
| A 曲线 3 阶渐近值 | 0.05 | 复原 | <1e-10（绝对） | ✅ |
| A 曲线 3 阶拟合误差 | 数据本身 | 1.225e-15 | 1.2e-13 % | ✅ |
| A 曲线 2 阶拟合误差 | 数据本身 | 5.548e-02 | 5.5 % | ❌（阶数不足） |
| A 曲线同阶改前基线（3 阶） | 数据本身 | 8.072e-01 | 81 % | ❌（改前不达标） |
| B 曲线 12 阶 **第 1 轮**实极点 | 数据本身 | 1.684e-01 | 16.8 % | ❌（第 1 轮结论） |
| B 曲线 12 阶 **本轮**（重定位） | 数据本身 | 2.046e-04 | 0.020 % | ✅ |
| B 曲线 8 阶 本轮 | 数据本身 | 2.418e-03 | 0.24 % | ✅ |
| B 曲线 6 阶 本轮 | 数据本身 | 1.487e-02 | 1.5 % | ✅ |
| B 曲线 4 阶 本轮 | 数据本身 | 3.802e-01 | 38 % | ❌（阶数不足） |
| B 曲线 12 阶无约束有理上限 | 数据本身 | 1.25e-06 | 1.3e-4 % | ✅（在实极点基之外） |

**honest 结论（四条，含一条撤回、两条局限）**：

1. 对**可被实衰减率基表征**的曲线，SK 迭代精确到机器零，改前基线不达标（81 %）——
   能力修复，判据清晰（A 曲线，3 阶）。
2. 对**真实三维辐射阻尼实测曲线**，第 1 轮「17–49 % 是模型类下限」的结论**撤回**：
   同数据、同阶次，只允许极点位置移动，误差从 `1.684e-01` 降到 `2.046e-04`
   （822.8×），6 阶起满足 3 % 门。原因是 `_real_pole_projection` 把共轭对投影成
   重合实极点——那是**方法**误差，不是基的表达能力上限。撤回依据是本表两列
   （`SK unrefined` vs `SK + pole refinement`）与 `tests/test_sk_fit.py` 的 4 个阶次 pin。
3. 仍存在的局限（不粉饰）：4 阶及以下在 B 曲线上不达标（38 % @4 阶），需要 ≥6 个
   实衰减率才能表征该曲线的量级结构；重定位在 10/12 阶**未**达到单极点探针局部最优
   （`refine_converged = no`，探针仍显示 1.2–4.6 % 的可改进量），即 2.0e-04 还不是
   该阶次实极点基的下限；与无约束有理上限（1.25e-06）仍有 163× 的差距，这部分
   才真正属于模型类误差。
4. 适用范围：本模块给出的是**实极点物理基**上的最优拟合，可直接喂给
   `state_space` / `time_domain`；若只要频域精度、不需要实极点，应改用无约束有理模型
   （误差上限列），但那样得到的极点在复平面上，不能进实数状态空间实现。

## 5 注意点（坑与局限）

1. **SK 不保证收敛**：真实刚性曲线上会 2-周期极限环（B 曲线 8/10/12 阶）；
   阻尼/松弛更新在本数据上反而使 15 阶发散，故用「最优 iterate + 诚实 `sk_converged`」。
2. **列归一化必须做**：`A(u)/D(u)` 的线性化设计矩阵含 `u^k` 与 `B·u^k`，
   未归一化时阶数一高就秩亏，`lstsq` 给出无意义极点。
3. **共轭复根投影是第 1 轮的隐性瓶颈**：一旦数据不是有限实指数和，`D` 必有共轭复根，
   投影到模会造成极点重合（B 曲线原来 12 阶得到 5 对复根 → 10 个重合极点）；
   第 2 轮的重定位就是修这个落点。判别信号：`complex_pair_count > 0` 且
   `rational_rel_error << sk_pole_rel_error`。
4. **单纯形的 `success` 标志不可当收敛证据**：本类平坦目标上它永不满足 simplex-size
   判据；要么报假「未收敛」，要么报假「已收敛」。本模块用单极点 ±5 % 探针**测**局部
   最优性（`refine_probe_gain`），而不是抄 `scipy` 的标志。探针本身也有盲区：单极点
   移动无改进不等于多极点联合移动无改进，故探针只作为「必要条件」上报。
5. **重定位不能替代阶数**：B 曲线 2/3/4 阶重定位后仍 98 %/65 %/38 %；4 参数装不下
   四量级的曲线，这是模型容量问题。
6. **`d ≥ 0` 约束来自仓库既有实现**（`state_space._solve_linear`），本单沿用，
   保证高频渐近阻尼非负；重定位过程中残基始终由该函数解出，所以约束不被绕过。
7. **阶数与点数**：SK 线性系统有 `2M+1` 个未知量，点数不足时 `ValueError` 明确拒绝。
8. **成本**：默认预算下单次 fit ≈0.4–3 s（阶数相关）；批量扫描/参数研究应递交集群，
   或把 `refine_maxfev` 降到 0/500 换取速度（返回值仍不劣于候选集中的最优者）。

## 6 能力台账

* 行 `damping-rational-fit`（模块 `tensorbem.sk_fit` + `tensorbem.state_space`，
  证据 = `tests/test_sk_fit.py`，文档 = 本文）claim 已随本轮更新：从「实测曲线 12 阶
  下限 1.7e-01」改为「极点重定位后同阶 2.0e-04，6 阶起进 3 % 门；4 阶及以下仍不达标」。
* 本轮新增证据：`tests/test_sk_fit.py::test_pole_refinement_meets_the_accuracy_bar_on_the_measured_curve`
  （4 个阶次）、`::test_pole_refinement_round_two_numbers`、`::test_refine_budget_zero_reproduces_round_one`、
  `::test_refinement_is_deterministic`、`::test_refinement_convergence_is_measured_not_asserted`、
  `::test_unrefined_pole_projection_stalls_on_the_measured_curve`。