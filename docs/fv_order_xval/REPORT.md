# XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR — 三网格 log-log 斜率估计单元中心有限体积观测阶（第 2 轮：调研结合 + 实现）

任务 m18742 · 仓库 TensorBEM · 分支 `agent/m18742` · 判据（owner 精度标准）：观测阶误差 **> 15% 不可接受**；本单同时报 `<3%` 线。

> 生成方式：本文所有数字由 `scripts/m18742_fv_order_xval.py run` 从 `artifacts/fv_order_xval/results.json` **程序化注入**，无手抄；`scripts/m18742_fv_order_xval.py verify` 只读该 JSON 重判每条判据（机器可裁）。

## ① 问题设定与基准出处

**被验证的对象**：单元中心有限体积（cell-centred FV）对二维 Poisson 方程

$$ -\nabla^2 u = f \;\text{in}\;\Omega=[0,1]^2,\qquad u = u_{exact}\;\text{on}\;\partial\Omega $$

的离散解的**观测阶**。这是纯验证（verification）问题：物理模型误差为 0，唯一被检验的量是离散化阶数，因此它可以作为任何 FVM 求解器的“仪器标定”。

**制造解（external anchor）**：

| 算例 | 解析解 | 右端项 | 出处 |
|---|---|---|---|
| `sin_sin` | $sin(pi x) sin(pi y)$ | $2 pi^2 sin(pi x) sin(pi y)$ | method of manufactured solutions, eigenfunction of both operators (Ferziger & Peric 2002, Sec. 3.3 'manufactured solutions / order verification'); the double sine is the standard Dirichlet eigenfunction on the unit square |
| `exp_sin` | $exp(x) sin(pi y)$ | $(pi^2 - 1) exp(x) sin(pi y)$ | second manufactured solution, non-symmetric and exponentially stretched in x, so the observed order is not an artefact of the problem symmetry (Ferziger & Peric 2002, Sec. 3.3) |

**基准（ref）出处**：

1. **解析解本身**：上述制造解是精确解，离散误差可以逐点评估 —— 这是最强的基准，不需要任何实验或文献数值（误差为 0 的 ref）。
2. **阶数的理论值** `p_exact`：由截断误差分析给出（Ferziger & Perić, *Computational Methods for Fluid Dynamics*, 3rd ed., Springer 2002, §3.3 阶数验证与 §4.3-4.4 单元中心 FV 面通量/源项记账；Versteeg & Malalasekera, *An Introduction to CFD: The Finite Volume Method*, 2nd ed., Pearson 2007, §4.2-4.3 扩散通量平衡、§5 边界处理）。规则网格 + 中点源项 + 半格 Dirichlet 幽灵→ `p_exact = 2`；源项退化为角点矩形公式 → `p_exact = 1`；边界通量取全格距离 → `p_exact = 1`。
3. **三网格观测阶公式**：Roache, *J. Fluids Eng.* 116(3):405-413 (1994) 与 *Verification and Validation in Computational Science and Engineering*, Hermosa 1998（三网格 `Fs = 1.25`、渐近区间比值判据）；Celik et al., *J. Fluids Eng.* 130(7):078001 (2008) / ASME V&V 20-2009（`p` 由三重解差分给出）。本仓既有实现 `tensorbem.uq.grid_convergence_index` 用的就是后者，本单把“误差型 log-log 斜率”与“泛函型 Roache 阶”当两条**互相独立的路由**来交叉验证。
4. **独立实现（OSS 横向调研）**：scikit-fem 12.0.2——同一制造解的 P1 / P2 Galerkin 有限元，同一网格、同一采样点、同一离散范数。调研结论见 §②。

**调研来源（第 2 轮要求）**：

* OSS 目录（实读）：`/root/oss-cache` 本租户缓存里只有 3 个仓——`pylbm`（194 个 py）、`Maarten-vd-Sande_lbm`（7 个 py）、`hschwane`（0 个 py），全是 LBM 系，**没有**单元中心 FV Poisson 求解器，也没有任何 GCI / 观测阶工具链。`pylbm` 的验证范式是“与解析 Poiseuille 剖面对比”（`pylbm/demo/2D/Poiseuille_vec.py`、`pylbm/demo/3D/poiseuille.py` 文件头给出解析解 `ux = umax(1-4(y/L)^2)`），只有单个网格 → 只能给误差、给不出阶。这正是本单要补的一层：**把“对比解析解”升级成“阶梯上的 log-log 斜率”**。
* OSS 目录（文档侧）：本仓冻结语料 `docs/rci-pilot200-r2/frozen/corpus/TensorFVM/docs/oss-inventory-fvm/gap-proposals.md` 列了 FVM 生态 8 条能力缺口与参考实现（OpenFOAM / Basilisk / Clawpack / SU2 / PeleC 等 T1/T2 及许可证提示）；该清单里同样**没有**“观测阶验证工具链”条目。集群侧 `~/oss-cache` 的 SQL 检索（作业 cj2009）递交后 12 min 仍在队列（提交时队列 139），本单不等它，如实记录。
* OSS 源码（唯一有的现成方法，直接学结构）：scikit-fem 12.0.2 的 `ElementTriP1`/`ElementTriP2` 组装与按点插值（`skfem/assembly/form/bilinear_form.py`、`skfem/element/element_h1.py`）被直接调用而非重写；其 `MeshTri` 结构化剖分在 `src/tensorbem/benchmarks/fv_order_xval.py:skfem_ladder` 里生成（用 numpy 向量化，避免按单元 Python 循环）。
* 前轮教训（负知识 `negk:loop:18722`，keywords 检索命中）：m18722 在本锚点 honest-FAIL 的直接原因是 **result 未提供 `checks.reason`**，即交付记账缺字段，而非算法结论被推翻。本单据此把判据、工件路径、复现命令写进 JSON 与 finish 的 `checks`（含 reason），并让 `verify` 子命令独立重判。

## ② 方法与实现

**离散格式**（`src/tensorbem/benchmarks/fv_order_xval.py:solve_fv_poisson`）：对每个控制体 `P`（体积 `h²`、面积 `h` 的四个面）积分得

$$ \sum_{f} h\,(\partial u/\partial n)_f = f(x_P)\,h^2 ,\qquad \text{interior: } (\partial u/\partial n)_f = (u_N-u_P)/h ,\qquad \text{boundary: } (\partial u/\partial n)_f = (u_b-u_P)/(h/2) $$

半格边界距离（等价于一次幽灵单元延拓）与中点源项的组合是教科书二阶格式。线性系统用 `scipy.sparse.linalg.spsolve` 直接求解，**没有迭代收敛容差**，所以结果里不含求解器噪声（可重复性见 §⑥ 的 sha256）。

**降解变体**（同一段组装代码，只改一个关键字参数，作为负对照）：

| 变体 | 改动 | 理论阶 |
|---|---|---|
| `fv2` | —（基准格式） | 2 |
| `fv1_src` | 源项中点公式 → 控制体左下角矩形公式（`O(h)` 求积缺陷） | 1 |
| `fv1_bc` | 边界通量距离 `h/2 → h`（边界通量整体缩放 1/2） | 1 |

**三个观测阶估计器**：

* `loglog_slope(hs, errs)`：`log e` 对 `log h` 的最小二乘斜率（三网格 → 一次多余自由度，报 `r²` 与成对斜率）；
* `pairwise_slopes`：相邻两级的 `p = ln(e_k/e_{k+1}) / ln(h_k/h_{k+1})`；
* `three_grid_slope` / `tensorbem.uq.grid_convergence_index`：**不需要精确解**的泛函型 Roache 阶（用解的均值序列差分），附带 GCI 与渐近区间比值。

**独立路由**（`skfem_ladder`）：同一制造解、同一网格梯度的 P1/P2 Galerkin 有限元，Dirichlet 用显式消元（不依赖 skfem 版本相关的 `condense` 关键字），解在**同样的单元中心点**上用 `basis.interpolator` 采样，用**同样的离散 L2** 度量。

## ③ 参数表

本问题是纯扩散验证问题，无 Reynolds/Froude 等物理无量纲数；下表是本单实际起作用的“无量纲/离散参数”。

| 参数 | 取值 | 说明 |
|---|---|---|
| 网格数 N | ['r2_coarse', 'r2_fine', 'r15'] | `r2_coarse`=32/64/128，`r2_fine`=64/128/256，`r15`=40/60/90 |
| 加密比 r | 2（`r2_coarse`,`r2_fine`）、1.5（`r15`） | 非整数比用于检验斜率与 r 无关 |
| h | 1/N | 均匀网格 |
| 制造解 | `['sin_sin', 'exp_sin']` | 见 §① |
| GCI 安全因子 Fs | 1.25 | Roache 三网格取值 |
| owner 容差 | 15.0000% | 观测阶相对误差 |
| 报告线 | 3.0000% | 更严的一档 |
| 渐近区间容差 | 10.0000% | `|GCI_medium/(r^p GCI_fine) - 1|` |
| 线性求解 | 直接（`spsolve`） | 无迭代容差 |
| 环境 | Python 3.11.6, numpy 2.4.6, scipy 1.16.0, scikit-fem 12.0.2 | loongarch64 |

## ④ 对照表（ref vs got，逐项打勾）

### 4.1 主判据：`fv2` 的观测阶 vs 解析理论阶 `p_exact = 2`

| 算例 | 梯度 | 范数 | ref (`p_exact`) | got (`p_obs`) | 误差 | <15% | <3% |
|---|---|---|---|---|---|---|---|
| `sin_sin` | `r2_coarse` | l2 | 2.000 | 2.000326 | 0.0163% | ✅ | ✅ |
| `sin_sin` | `r2_coarse` | l1 | 2.000 | 2.000869 | 0.0435% | ✅ | ✅ |
| `sin_sin` | `r2_coarse` | linf | 2.000 | 1.998696 | 0.0652% | ✅ | ✅ |
| `sin_sin` | `r2_fine` | l2 | 2.000 | 2.000081 | 0.0041% | ✅ | ✅ |
| `sin_sin` | `r2_fine` | l1 | 2.000 | 2.000217 | 0.0109% | ✅ | ✅ |
| `sin_sin` | `r2_fine` | linf | 2.000 | 1.999674 | 0.0163% | ✅ | ✅ |
| `sin_sin` | `r15` | l2 | 2.000 | 2.000305 | 0.0153% | ✅ | ✅ |
| `sin_sin` | `r15` | l1 | 2.000 | 2.000814 | 0.0407% | ✅ | ✅ |
| `sin_sin` | `r15` | linf | 2.000 | 1.998779 | 0.0611% | ✅ | ✅ |
| `exp_sin` | `r2_coarse` | l2 | 2.000 | 2.000419 | 0.0209% | ✅ | ✅ |
| `exp_sin` | `r2_coarse` | l1 | 2.000 | 2.001150 | 0.0575% | ✅ | ✅ |
| `exp_sin` | `r2_coarse` | linf | 2.000 | 1.999648 | 0.0176% | ✅ | ✅ |
| `exp_sin` | `r2_fine` | l2 | 2.000 | 2.000104 | 0.0052% | ✅ | ✅ |
| `exp_sin` | `r2_fine` | l1 | 2.000 | 2.000126 | 0.0063% | ✅ | ✅ |
| `exp_sin` | `r2_fine` | linf | 2.000 | 1.999766 | 0.0117% | ✅ | ✅ |
| `exp_sin` | `r15` | l2 | 2.000 | 2.000391 | 0.0195% | ✅ | ✅ |
| `exp_sin` | `r15` | l1 | 2.000 | 1.998692 | 0.0654% | ✅ | ✅ |
| `exp_sin` | `r15` | linf | 2.000 | 1.999374 | 0.0313% | ✅ | ✅ |

worst L2 cell: `exp_sin` / `r2_coarse` → 0.0209%。

### 4.2 估计器一致性：误差型 log-log 斜率 vs 泛函型 Roache 阶（同一 `fv2` 解）

| 算例 | 梯度 | 误差型 `p` | Roache 泛函型 `p` | 相对差 | <15% |
|---|---|---|---|---|---|
| `sin_sin` | `r2_coarse` | 2.000326 | 2.001195 | 0.0435% | ✅ |
| `sin_sin` | `r2_fine` | 2.000081 | 2.000299 | 0.0109% | ✅ |
| `sin_sin` | `r15` | 2.000305 | 2.001119 | 0.0407% | ✅ |
| `exp_sin` | `r2_coarse` | 2.000419 | 2.002076 | 0.0828% | ✅ |
| `exp_sin` | `r2_fine` | 2.000104 | 2.000520 | 0.0208% | ✅ |
| `exp_sin` | `r15` | 2.000391 | 2.001945 | 0.0777% | ✅ |

### 4.3 跨实现：TensorBEM 自研 FV vs scikit-fem P1/P2（OSS）

| 路由 | 梯度 | ref `p_exact` | got `p_obs` | 误差 | <15% |
|---|---|---|---|---|---|
| cell-centred FV `fv2` | 32/64/128 | 2.000 | 2.000326 | 0.0163% | ✅ |
| skfem P1 Galerkin | 32/64/128 | 2.000 | 1.998794 | 0.0603% | ✅ |
| skfem P2 Galerkin | 32/64/128 | 4.000 | 3.997200 | 0.0700% | ✅ |

FV↔P1 观测阶相对差 = 0.0766%（判据 <15%：✅）。两条完全不同的离散家族给出同一个 2，说明 `p_obs ≈ 2` 不是某一套代码的记账偶然。

### 4.4 判别力（负对照）：退化变体必须返回 `p = 1`

| 变体 | ref `p_exact` | got `p_obs` | vs 自身目标 | vs 主目标 2 | Roache 泛函型 |
|---|---|---|---|---|---|
| `fv1_src:sin_sin` | 1 | 1.003420 | 0.3420% ✅ | 49.8290% ✅ | 2.000869 |
| `fv1_src:exp_sin` | 1 | 1.002644 | 0.2644% ✅ | 49.8678% ✅ | 0.960481 |
| `fv1_bc:sin_sin` | 1 | 0.998478 | 0.1522% ✅ | 50.0761% ✅ | 1.009022 |
| `fv1_bc:exp_sin` | 1 | 0.989383 | 1.0617% ✅ | 50.5308% ✅ | 1.023238 |

**这张表是本单最重要的方法学发现**（描述性观察，非判据）：负对照的点值解确定是一阶的（误差型 `p_obs` 全部在 1 的 1.1% 以内），但**泛函型** Roache 阶并不总是跟着降级——

* `fv1_src:sin_sin`：误差型 1.0034，泛函型 2.0009 → **泛盲**（泛函仍报 2）
* `fv1_src:exp_sin`：误差型 1.0026，泛函型 0.9605 → 未被蒙蔽
* `fv1_bc:sin_sin`：误差型 0.9985，泛函型 1.0090 → 未被蒙蔽
* `fv1_bc:exp_sin`：误差型 0.9894，泛函型 1.0232 → 未被蒙蔽

共 1/4 个负对照被泛函路由漏掉。原因：矩形求积的首阶误差是有符号的，在对称算例（`sin_sin`，解关于 (0.5,0.5) 对称）上做区域平均时**抵消**，泛函发生超收敛；换到非对称的 `exp_sin` 就露出来。所以泛函盲区是**算例相关**的：只在一个对称算例上做泛函型 GCI 的验证链，可能对一阶求积缺陷完全无感。误差型 log-log 斜率才是能分辨阶数的仪器——这也是本锚点名为 `LOGLOG-SLOPE` 的原因；判据 C4 要求两条路由在**主格式**上一致，而 §4.4 说明为什么不能只留一条。

### 4.5 判据清单（机器复核）

| 判据 | 含义 | 结果 |
|---|---|---|
| `C10_procedure_verification` | 估计器在合成幂律上 1e-10 精度 | ✅ PASS |
| `C1_finiteness` | 所有级别/估计器有限 | ✅ PASS |
| `C2_primary_order_15pct` | 主格式观测阶 vs 理论阶 <15%（owner 标准） | ✅ PASS |
| `C3_primary_order_3pct` | 同项 <3%（报告线） | ✅ PASS |
| `C4_estimator_agreement` | 误差型 vs 泛函型估计器一致 <15% | ✅ PASS |
| `C5_refinement_ratio_independence` | r=2 vs r=1.5 观测阶一致 <15% | ✅ PASS |
| `C6_asymptotic_range` | 32/64/128 vs 64/128/256 + 渐近区间比值 ≤10% | ✅ PASS |
| `C7_cross_implementation` | FV vs skfem P1 一致 <15% | ✅ PASS |
| `C8_discrimination` | 负对照返回 p=1 且漏掉 p=2 逾 15% | ✅ PASS |
| `C9_problem_independence` | 两个制造解一致 <15% | ✅ PASS |

**总判定**：10/10 条通过；honest note：every predeclared criterion met

## ⑤ 收敛性、渐近区间与参数敏感性

### 5.1 逐级误差（`fv2`，`sin_sin`）

| 梯度 | N | h | L1 | L2 | Linf |
|---|---|---|---|---|---|
| `r2_coarse` | 32 | 0.031250 | 3.259395e-04 | 4.017888e-04 | 8.016430e-04 |
| `r2_coarse` | 64 | 0.015625 | 8.140636e-05 | 1.004109e-04 | 2.007009e-04 |
| `r2_coarse` | 128 | 0.007812 | 2.034669e-05 | 2.510046e-05 | 5.019336e-05 |
| `r2_fine` | 64 | 0.015625 | 8.140636e-05 | 1.004109e-04 | 2.007009e-04 |
| `r2_fine` | 128 | 0.007812 | 2.034669e-05 | 2.510046e-05 | 5.019336e-05 |
| `r2_fine` | 256 | 0.003906 | 5.086365e-06 | 6.274973e-06 | 1.254947e-05 |
| `r15` | 40 | 0.025000 | 2.085048e-04 | 2.571002e-04 | 5.134079e-04 |
| `r15` | 60 | 0.016667 | 9.262645e-05 | 1.142472e-04 | 2.283378e-04 |
| `r15` | 90 | 0.011111 | 4.115895e-05 | 5.077266e-05 | 1.015144e-04 |

误差每级缩小约 4 倍（r=2、p=2 → 期望 4.000），即三个数量级以上的干净渐近行为。

### 5.2 两分辨率/两参数点的敏感性

| 检查 | 结果 | 判据 |
|---|---|---|
| 加密比 r=2 vs r=1.5（`sin_sin`） | 2.000326 vs 2.000305 → 0.0010% | <15% ✅ |
| 加密比 r=2 vs r=1.5（`exp_sin`） | 2.000419 vs 2.000391 → 0.0014% | <15% ✅ |
| 渐近区间（`sin_sin`） | p(32/64/128)=2.000326, p(64/128/256)=2.000081 → 0.0122%；渐近比值=0.9999 | <15% & ≤10% ✅ |
| 渐近区间（`exp_sin`） | p(32/64/128)=2.000419, p(64/128/256)=2.000104 → 0.0157%；渐近比值=1.0000 | <15% & ≤10% ✅ |
| 制造解无关性 | p(sin_sin)=2.000326, p(exp_sin)=2.000419 → 0.0046% | <15% ✅ |

### 5.3 仪器标定（用之前先验仪器）

合成幂律 `e(h) = 3.0 h^1.7, h = [1.0, 0.5, 0.25]`：log-log 斜率 `1.700000000000`（绝对误差 6.66e-16）、成对斜率 ['1.700000000000', '1.700000000000']、Roache 三重差分 `2.22e-16` 绝对误差，容差 1e-10 → ✅。仪器在 1e-10 精度上被标定之后才用于求解器。

## ⑥ 失败模式、复现命令与工件

**失败模式（按严重度）**：

1. **泛函型 GCI 可能对一阶缺陷盲，且盲区与算例有关**（§4.4`fv1_src:sin_sin`：误差型 1.0034，泛函型 2.0009；同一变体在 `exp_sin` 上泛函型掉到 0.9605）。任何只用“某个工程量收敛”来宣称二阶的验证链都可能被这条骗过。缓解：本单同时报误差型与泛函型，并把两者的一致性做成判据 C4。
2. **边界距离错用**（`fv1_bc`）：把半格距离写成整格会使边界通量整体小一倍，此时**泛函型**估计器仍能捕捉到（1.009），但绝对误差放大 40 倍量级（L2 error 从 4.0e-4 到 3.7e-2，见 §4.4 与 fig. 1）。缓解：把“绝对误差量级”与“阶”一起报，阶正确不等于精度正确。
3. **梯度窗口过短**：三网格只有一次多余自由度，`r²` 天然接近 1，单看 `r²` 不能证明渐近区间成立。缓解：加 `r2_fine`（64/128/256）与 Roache 渐近区间比值（判据 C6）。
4. **未标定仪器**：若估计器本身有 bug（例如用 `log10` 与 `ln` 混用、或把 h 用成 N），任何求解器结果都会被误判。缓解：C10 在 1e-10 精度上标定。
5. **本轮未做的部分（诚实声明）**：
   * 只做二维、均匀笛卡尔网格。非结构/多面体网格上的单元中心 FV 阶数（非正交修正项）未在本单覆盖——那是 TensorFVM 的域，本仓无该实现，故不虚报。
   * 只验 Poisson（纯扩散）。对流占优时的阶数还受限制器/格式影响，未覆盖。
   * `oss_solutions` 表在本租户的 API 面（`/api/v1` 1324 条路由）中**不存在可写端点**（本地检索 `oss|solution|inventory` 无匹配）；因此方法选型沉淀改为仓内可复算结构件 `docs/fv_order_xval/oss_solutions.json`（已带 `feature_slug` 字段），并在本文记录该限制。集群 `~/oss-cache` 的 SQL 查询作业已递交（cj2009），在队列中未及时返回，本单以仓内冻结的 FVM OSS 清单为横向调研依据。

**复现命令**：

```bash
# 全量研究（本地 ~2 min，无需集群）
PYTHONPATH=src python3 scripts/m18742_fv_order_xval.py run --outdir docs/fv_order_xval
# 只读工件重判判据（机器复核，退出码 0 = 全通过）
PYTHONPATH=src python3 scripts/m18742_fv_order_xval.py verify --outdir docs/fv_order_xval
```

**工件**：

| 路径 | 内容 |
|---|---|
| `src/tensorbem/benchmarks/fv_order_xval.py` | 求解器 + 三个阶估计器 + 独立路由 |
| `scripts/m18742_fv_order_xval.py` | 入口（`run`/`verify`/`report`），预声明判据 |
| `scripts/m18742_fv_report.py` | 图 + 本报告的程序化生成 |
| `tests/test_fv_order_xval.py` | 单元/回归测试 |
| `docs/fv_order_xval/results.json` | 全部数字（含判据与结论） |
| `docs/fv_order_xval/fv_order_loglog.png` | 误差-网格 log-log 曲线与拟合斜率 |
| `docs/fv_order_xval/fv_observed_order.png` | 各路由观测阶 vs 理论阶 ±15% 带 |
| `docs/fv_order_xval/fv_field_error.png` | 解场 + 误差云图（N=128） |
| `benchmarks/verified/fv_cc_three_grid_orr/case.json` | 基准用例登记 |
| `docs/fv_order_xval/oss_solutions.json` | 方法选型沉淀（feature_slug 见文件名/字段） |

结果 JSON 的 sha256 记录在 `results.sha256`，`verify` 子命令可复算判据。git commit（生成时）：`20cbd2220ed481bf62945fbb8fe05a51d3aadf1f`。

**误差判定（显式回答）**：主判据为观测阶相对误差，worst L2 单元为 0.0209%，**<15%：✅**；同一数字对照更严的 3% 线：**<3%：✅**（worst 单元 0.0209%）。跨实现 FV↔P1 的相对差 0.0766%。

