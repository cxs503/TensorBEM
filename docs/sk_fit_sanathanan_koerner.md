# TensorBEM: Sanathanan–Koerner 阻尼拟合——把「有迭代」坐实，并量化真实曲线的模型类下限

**任务** m16021（tp-merge-4 自主工时，TensorBEM）**分支** `agent/m16021`
**判据等级**：REAL（可表征曲线 1e-15 级恢复）；真实测量曲线可达性 = honest 上限（见 §4）
**工件** `docs/sk_fit_study/results.json`（零手抄，表格由
`python -m tensorbem.benchmarks.sk_fit_verification --write-doc` 直接渲染进本文）

---

## 1 案例介绍

**物理背景**：Cummins 时域运动方程把辐射记忆核写成有限个实衰减率之和

  `K(t) = Σ_k c_k e^{-α_k t}  ⇔  B(ω) = d + Σ_k c_k α_k/(α_k² + ω²)`

其中 `B(ω)` 是频域辐射阻尼。TensorBEM 的 `state_space` / `time_domain` 链路
（状态空间实现、自由衰减、记忆核两向闭合）都吃这个基。`tensorbem.sk_fit`
负责用 Sanathanan–Koerner（SK, 1963）迭代把一条 `B(ω)` 拟合成这组 `(α_k, c_k, d)`。

**为什么要查它**：能力证据门 `tests/test_capability_evidence.py` 在本分支上是红的，
报出 `tensorbem.sk_fit` 为「无能力行、亦无豁免理由」的未认领模块；而
`docs/m15632_freq_time_bridge_fix.md` §注意点 1 早已自承这一欠账：
「SK 迭代模块 `sk_fit.py` 是固定极点变体，不包含完整的极点重定位步骤」。

**代码事实（改前）**：极点固定在 `logspace(0.3ω_min, 3ω_max, n_poles)`，
只解一次加权线性最小二乘，`max_iter` / `tol` 接收后**从未使用**，返回
`n_iter=1, converged=True`。即：函数名与文献署名指向一个迭代法，实现里没有迭代。

**两条被测曲线（参数见 §2 与工件 `truth` 字段）**

| 记号 | 曲线 | 是否可被实衰减率基表征 | 用途 |
|---|---|---|---|
| A `representable` | 合成：`α=(0.5,1.0,2.5)` rad/s、`c=(1.0,−1.2,0.6)`、`d=0.05`，ω∈[0.02,50] 300 点 | **是**（构造上精确） | 反问题可解性判据 |
| B `hulme_hemisphere` | 浮半球 heave 实测参考阻尼曲线（Hulme 1982 解析解，仓库夹具 `tests/data/time_domain_hemisphere_sweep.json`，201 点，B 跨 1.7e3 : 2.1e-2 四个量级） | **否**（自由面记忆核有代数尾，非有限实指数和） | 模型类下限、无回归判据 |

A 的衰减率刻意取在基线对数网格**不会**落点的位置（网格 0.006 / 0.075 / 0.95 / 12 / 150 …），
因此 A 上的误差是方法误差，不是网格运气。

## 2 怎么做的

1. **实现真正的 SK 迭代**（`tensorbem.sk_fit._sk_iterate`）。在 `u = ω²` 上写
   `B(u) ≈ A(u)/D(u)`，`D(u) = 1 + d₁u + … + d_M u^M`；每个频率点 `j` 解线性化方程
   `A(u_j) − B_j·(D(u_j) − 1) = B_j`，权重 = 数据相对权重 × `1/|D_{i−1}(u_j)|`，
   解出后更新分母并迭代，直到分母相对变化小于 `tol` 或加权残差相对变化小于 1e-12。
   列归一化 + 最小二乘 `lstsq`（设计矩阵条件数随阶数平方增长，不归一化会直接失稳）。
   迭代记录**最优iterate**（加权残差最小者），因为实测在刚性真实曲线上 SK 会 2-周期
   极限环（见 §3.2 `converged=no` 的行）。
2. **极点投影到物理基**（`_real_pole_projection`）：`D` 的实负根 `u_k = −α_k²/u_ref`
   精确映射为衰减率；共轭复根无法表示，投影到其模（`α=|u|^½`），并把复根对数
   记进报告，让「模型类误差」与「拟合误差」分开。
3. **无回归的极点集选择**：候选集 = {SK 投影极点} ∪ {基线对数网格}，
   两者都用仓库原有的约束最小二乘 `state_space._solve_linear`（`d ≥ 0`，防空载阻尼）
   解残基，按**实测加权误差**取优，`pole_source` 如实记录赢家。
4. **基线保留为可测函数** `fixed_grid_damping_fit`（改前行为逐字保留），
   使"改进了多少"成为可复算的数字而非声明。
5. 工具链：`python -m tensorbem.benchmarks.sk_fit_verification --out-root docs/sk_fit_study`；
   `pytest tests/test_sk_fit.py` 重跑全study并逐行比对工件与本表。
   仅用 numpy（+ matplotlib 出图）。

## 3 计算结果（程序化注入 `docs/sk_fit_study/results.json`）

### 3.1 阶次扫描（`returned` = 本模块返回值；`baseline` = 改前固定网格）

<!-- sk-fit-table:begin -->
| case | order | returned max err | baseline max err | SK wins? | pole source | SK iters | converged |
|---|---|---|---|---|---|---|---|
| representable | 2 | 7.839e-01 | 8.894e-01 | yes | sk | 200 | no |
| representable | 3 | 1.225e-15 | 8.072e-01 | yes | sk | 3 | yes |
| representable | 4 | 3.472e-15 | 2.395e-01 | yes | sk | 8 | yes |
| representable | 6 | 6.127e-16 | 1.534e-01 | yes | sk | 8 | yes |
| representable | 8 | 2.042e-15 | 1.447e-01 | yes | sk | 8 | yes |
| representable | 10 | 1.838e-15 | 8.509e-02 | yes | sk | 8 | yes |
| representable | 12 | 3.165e-15 | 4.434e-02 | yes | sk | 8 | yes |
| hulme_hemisphere | 2 | 9.976e-01 | 9.997e-01 | yes | sk | 119 | yes |
| hulme_hemisphere | 3 | 9.902e-01 | 9.902e-01 | no | grid | 200 | no |
| hulme_hemisphere | 4 | 9.785e-01 | 9.785e-01 | no | grid | 26 | yes |
| hulme_hemisphere | 6 | 9.023e-01 | 9.230e-01 | yes | sk | 53 | yes |
| hulme_hemisphere | 8 | 4.921e-01 | 4.921e-01 | no | grid | 200 | no |
| hulme_hemisphere | 10 | 2.241e-01 | 2.241e-01 | no | grid | 200 | no |
| hulme_hemisphere | 12 | 1.684e-01 | 1.684e-01 | no | grid | 200 | no |
<!-- sk-fit-table:end -->

要点：

* **A 曲线、3 阶**：SK 用 3 次迭代把三个衰减率、三个残基与渐近值全部复原，
  误差 `1.225e-15`；同一阶的改前基线是 `8.072e-01`（峰值的 81 %）。这是本单
  主张的核心数字——同一阶、同一数据、同一后处理，差别只在于有没有极点重定位。
* A 曲线 4 阶及以上：SK 收敛到 1e-15 量级并稳在机器零；基线随着网格变密才缓慢
  从 `2.4e-01` 降到 `4.4e-02`，说明固定网格的误差由**极点落点**决定，加密阶数
  并不能替代重定位。
* B 曲线：`regression_free = true` —— 扫描的每一阶上返回值都不劣于改前基线
  （输的是 `grid`，赢的记 `sk`，在工件里逐行可查）。这是"能力扩展不得造成回归"
  的可机检不变量，由 `tests/test_sk_fit.py::test_never_worse_than_the_fixed_grid` 钉住。

### 3.2 迭代本身的行为

| 观察 | 数字（工件内） |
|---|---|
| A 曲线 3 阶收敛所需迭代 | 3 次 |
| B 曲线 8/10/12 阶 | 200 次上限内未满足停止判据（`converged = no`），返回的是残差最小的 iterate |
| `max_iter=1` | `n_iter=1`、`converged=False`，且误差**劣于**多次迭代结果（参数不再是装饰） |
| 残差历史长度 | 与 `sk_iterations` 相等（每次迭代一条） |

SK 在 B 上的极限环是文献已知的病态（SK 对强刚性/宽带数据无收敛保证），本模块不粉饰：
`converged` 字段报的是停止判据是否满足，而不是 `True` 常量。

### 3.3 图

`docs/sk_fit_study/sk_fit_verification.png`：左右两栏（A、B），横轴阶数、
纵轴 `max|ΔB|/max|B|`，三条线 = 本模块返回值、改前固定网格基线、
**无约束有理模型的误差上限**。B 栏里"上限"与两条实极点线之间近 5 个量级的缺口
就是模型类误差（见 §4）。

## 4 误差判定（是否 <3 %？）

| 关键量 | 基准 | 本实现 | 相对误差 | <3 %？ |
|---|---|---|---|---|
| A 曲线 3 阶衰减率 | (0.5, 1.0, 2.5) rad/s | 复原 | 3.6e-16 | ✅ |
| A 曲线 3 阶残基 | (1.0, −1.2, 0.6) | 复原 | <1e-8（绝对） | ✅ |
| A 曲线 3 阶渐近值 | 0.05 | 复原 | <1e-10（绝对） | ✅ |
| A 曲线 3 阶拟合误差 | 数据本身 | 1.225e-15 | 1.2e-13 % | ✅ |
| A 曲线同阶改前基线 | 数据本身 | 8.072e-01 | 81 % | ❌（改前不达标） |
| B 曲线 12 阶实极点最优 | 数据本身 | 1.684e-01 | 17 % | ❌ **honest FAIL** |
| B 曲线 12 阶无约束有理上限 | 数据本身 | 1.25e-06 | 1.3e-4 % | ✅（但落在实极点基之外） |

**honest 结论（三条，含负面）**：

1. 对**可被实衰减率基表征**的曲线（Cummins 核截断、由 `(α,c)` 合成的频响），
   SK 迭代现在精确到机器零，改前基线不达标——这是能力修复，判据清晰。
2. 对**真实三维辐射阻尼实测曲线**，本基在 12 阶只能到 17 %，**达不到 3 % 门限**：
   浮体辐射记忆核有代数尾，不是有限实指数和。无约束有理模型在同阶能到 1.25e-6，
   说明残余误差几乎全部来自"实极点基"这一模型类，而非拟合算法。该结论是负面结论，
   已写进能力声明与报告，不以提高阶数掩盖。
3. 因此本模块的适用范围 = 「已知/可截断为有限实衰减率」的阻尼曲线；对实测曲线
   应改用 `state_space.vector_fit_damping`（可分变量投影，极点连续优化）或提高阶数
   并接受模型类误差。`sk_fit` 不会再比改前基线给出更差的拟合（已机检）。

## 5 注意点（坑与局限）

1. **SK 不保证收敛**：真实刚性曲线上会 2-周期极限环（本单实测 B 曲线 8/10/12 阶）；
   阻尼/松弛更新在本数据上反而使 15 阶发散，故未采用，改用"最优 iterate + 诚实 `converged`"。
2. **列归一化必须做**：`A(u)/D(u)` 的线性化设计矩阵含 `u^k` 与 `B·u^k`，
   未归一化时阶数一高就秩亏，`lstsq` 给出无意义极点。
3. **复根无法回避**：一旦数据不是有限实指数和，`D` 必有共轭复根；投影到模会
   造成极点重合（本单 B 曲线可见），故报告里 `complex_pair_count` 逐行可见。
4. **`d ≥ 0` 约束来自仓库既有实现**（`state_space._solve_linear`），本单沿用，
   保证高频渐近阻尼非负；否则拟合会在尾部给出"主动系统"。
5. **阶数与点数**：SK 线性系统有 `2M+1` 个未知量，点数不足时 `ValueError` 明确拒绝，
   不静默给出过拟合结果。

## 6 能力台账

* 新增行 `damping-rational-fit`（模块 `tensorbem.sk_fit` + `tensorbem.state_space`，
  证据 = `tests/test_sk_fit.py`，文档 = 本文），`tensorbem.sk_fit` 从
  「未认领模块」变为被行级证据认领；
* 顺带清掉证据门的两处红：`tensorbem.benchmarks.kp505_w20_gci` 的豁免理由已过期
  （该模块现已被测试引用）、`docs/audit_baseline.md` §4 发布块与树漂移，
  二者由 `python -m tensorbem.capability_evidence --write` 重新生成。