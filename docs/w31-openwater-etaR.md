# W31 · propeller-openwater-bem：ηR 链路去常量（敞水/后船身双解反解）

- mission 22286 ／ 仓库 TensorBEM ／ commit `5a2764db5e50ea220bd6a185a6281f1f82f7d9c7+dirty`
- 工件：`benchmarks/artifacts/w31_openwater_etaR_chain.json`（本报告全部数字由 `benchmarks/anchors/publish_w31_etaR_chain.py` 从该工件注入，零手抄）
- 图：`figures/w31/openwater_etaR_chain.png`
- 预登记门限：C1 std(ηR) > 0.001；C2 ≥ 9/11 状态 ηR ∈ [0.9, 1.1]；C3 恒等相对残差 ≤ 1e-06；C4 J ≤ 0.8 区最差 |ΔK_T| < 15 %（并逐项答 < 3 %）

**一句话结论**：ηR 已从平台常数 1.0 变成算出来的量（11 状态 0.9773–0.9854，std = 0.002521），ηD = η0·ηR·ηH 恒等逐状态闭合；C1/C2/C3 **PASS**，C4 **honest FAIL**（KP505 曲线 J = 0.8 处 -29.13 %，15 % 门限不满足）。

## 1 案例与物理背景

敞水（open-water）试验给出螺旋桨单独在均匀来流中的 K_T、K_Q 与 η0 = J·K_T/(2π·K_Q)；船后（behind-hull）工况下桨在艉部构件造成的非均匀伴流中工作，同一进速比 J = V_a/(nD) 的效率与敞水不同，其比值就是相对旋转效率

$$\eta_R=\frac{\eta_{0,\mathrm{open\ water}}(J)}{\eta_{0,\mathrm{behind\ hull}}(J)},\qquad \eta_D=\eta_H\,\eta_R\,\eta_0 .$$

平台推进块返回的 `relative_rot_eff` 在全部 11 个状态上恒为 1.0（std = 0.0）、`hull_eff` 恒为 1.07（表 2）。上一单（W30）已证明这是「未计算」而不是「算出来等于 1」：本单从平台回执逐字核对，并现场复验两个状态。

表 1 船体与桨参数（平台回执 `data/w30/propulsion_states/api_receipts.json` sha256 `37a26be1ba93c5c4`；现场复验 `data/w31/propulsion_states/live_recheck.json` sha256 `42c7700ce27e87a0`）：

| 项 | 值 | 来源 |
|---|---|---|
| 船体 | L = 180 m，B = 32 m，T = 11 m，Cb = 0.82 | 回执 hull 段 |
| 排水体积 | 51955.2 m³（= L·B·T·Cb） | 回执 hull 段 |
| 桨 | D = 6.0 m，Z = 5，Ae/Ao = 0.70（由回执 K_T/K_Q 在预登记阶梯上辨识，见 §3.1），P/D = 0.95（S11 为 1.05） | 回执 + §3.1 |
| 平台方法串（逐字） | Wageningen B 系列 (Oosterveld & van Oossanen 1975) + Holtrop (1984) 伴流/推力减额 + 功率链闭合 P_E = eta_D*P_D + sigma_0.7R 空泡数 | 现场复验回执 |
| 伴流/推力减额 | w = 0.3038，t = 0.2551（11 状态相同） | 回执 |
| 现场复验 | S1、S11 两次 POST `/calculations/propulsion` 均返回 relative_rot_eff = 1.0 / 1.0，与归档回执逐字段相同 | 现场复验回执 |

表 2 11 个状态（W30 预登记网格，本单沿用同一网格以便逐状态对照）：

| 状态 | V [kn] | n [rpm] | P/D | J | η0 | ηH | ηR 平台返回 | ηD 平台 |
|---|---|---|---|---|---|---|---|---|
| S1 | 12 | 100 | 0.95 | 0.4298 | 0.4565 | 1.07 | 1.0 | 0.4884 |
| S2 | 13 | 100 | 0.95 | 0.4656 | 0.4892 | 1.07 | 1.0 | 0.5234 |
| S3 | 14 | 100 | 0.95 | 0.5014 | 0.5205 | 1.07 | 1.0 | 0.5569 |
| S4 | 15 | 100 | 0.95 | 0.5372 | 0.5503 | 1.07 | 1.0 | 0.5888 |
| S5 | 16 | 100 | 0.95 | 0.5731 | 0.5783 | 1.07 | 1.0 | 0.6187 |
| S6 | 17 | 100 | 0.95 | 0.6089 | 0.6042 | 1.07 | 1.0 | 0.6464 |
| S7 | 15 | 85 | 0.95 | 0.6320 | 0.6196 | 1.07 | 1.0 | 0.6630 |
| S8 | 15 | 90 | 0.95 | 0.5969 | 0.5958 | 1.07 | 1.0 | 0.6375 |
| S9 | 15 | 95 | 0.95 | 0.5655 | 0.5726 | 1.07 | 1.0 | 0.6126 |
| S10 | 15 | 105 | 0.95 | 0.5117 | 0.5292 | 1.07 | 1.0 | 0.5662 |
| S11 | 15 | 100 | 1.05 | 0.5372 | 0.5166 | 1.07 | 1.0 | 0.5527 |

## 2 方法

**决策**：Is eta_R deliverable as a computed state-dependent quantity from the behind-hull K_T/K_Q and the open-water curve read at the same J, with eta_D = eta_0 eta_R eta_H closing as a chain?

**备选方案（预登记，先于任何数字）**：
- `A1_platform_constant`：eta_R = 1.0 as returned by the propulsion block
- `A2_empirical_regression`：Holtrop(1984)-type closed form (near-constant, kept as fallback)
- `A3_behind_hull_ratio`：eta_0,open-water(J) / eta_0,behind-hull(J) from the same lattice ← **采用**

**为什么 A3 是测量而不是参数化**：分子（敞水）与分母（后船身）用**同一个**涡格核、同一套面元、同一雷诺数求解，唯一差别是入流场（均匀 vs 艉部标称伴流），求解器自身的模型偏差在比值中抵消；剩下的就是桨对伴流周向非均匀性的响应。为使这一点可证伪，本单把 **q = 0（均匀伴流）作为零对照**：此时入流场与敞水入流逐点相同，ηR 必须精确为 1。

**预登记配置（未调参；灵敏度另行报出）**：

| 项 | 取值 |
|---|---|
| 伴流形状 | w(ξ) = w̄·(1−ξ)^q /⟨(1−ξ)^q⟩，ξ = (r − r_hub)/(R − r_hub)，主轴 q = 0.5 |
| 形状灵敏度 | q ∈ [0.0, 0.25, 0.5, 1.0]（q = 0 为均匀伴流零对照） |
| 面元 | n_r × n_c = 16 × 2（KP505 算例登记构型）；灵敏度 [[8, 1], [16, 2], [24, 2]] |
| 尾涡/求解 | wake_model = geometric，wake_length = 1.0，wake_steps = 24 |
| 无量纲化 | non-dimensional D = 1; Re preserved by n_eff = n_phys D_phys^2（ITTC-78 剖面阻力律只依赖 n·D²，故 Re 精确保持） |
| 敞水曲线 | 0.10–0.75 步长 0.05，piecewise linear in J on the ladder |
| 停止规则 | one pass of the full state set at the primary configuration, one null control, one panel ladder and one wake-shape ladder; no iteration on thresholds or on the shape exponent after seeing eta_R |

**实验设计（预登记于工件 `preregistration.design`，逐条对应）**：

| 项 | 内容 |
|---|---|
| 假设 | H1: the propeller responds to the radial non-uniformity of the afterbody wake, so eta_R = eta_0,OW(J)/eta_0,BH(J) differs from 1 by an amount that depends on J, i.e. eta_R is a computed quantity and not a constant. H0 (the platform's implicit hypothesis): eta_0,BH == eta_0,OW at the same J, so eta_R == 1. |
| 分析单元 | one propulsion state (11 states), not one solver run: the gate is a property of the series |
| 基线 | the platform's constant eta_R = 1.0 on the identical 11 states |
| 对照 | q = 0 uniform wake: the inflow field then equals the open-water inflow point by point, so eta_R must be exactly 1. Any departure would be a code defect, and the control is what separates a code defect from a physical effect. |
| 混淆变量 | panel density (n_radial x n_chord): declared ladder (8,1)/(16,2)/(24,2); wake shape exponent q: declared ladder 0/0.25/0.5/1.0, monotonicity checked; open-water curve interpolation error: measured against a direct solve per state; receipt rounding (4 dp non-dimensional): bounds identity residuals at ~1e-3; Reynolds number: preserved by n_eff = n_phys D^2 so it is not a free variable; kernel scale-invariance defect at D=6: avoided by the D=1 convention, recorded |
| 重复性依据 | the solver is deterministic (no random seed, one dense linear solve); repeating a state can only reproduce it, so repetitions are replaced by (a) an explicit determinism re-solve recorded in the artefact and (b) the panel/shape ladders, which vary the two nuisance variables that actually move the number |
| 环境控制 | single machine, one process, same numpy/Python as recorded; wall time and commit recorded in the artefact; the run is short (~40 s) so it is not submitted to the cluster and no cross-node variability enters |
| 分析计划 | gate arithmetic is computed by the builder, never typed; the doc reads every number from the artefact; the KP505 comparison keeps all 10 anchor points (no selection) and reports the region-restricted (J<=0.8) and the stricter (J<=0.7, J<=0.6) worst deviations side by side |
| 资源上限 | 8 cores unused (single-threaded numpy), < 60 s wall, < 4 GB RAM; no cluster job |
| 解释边界 | eta_R here is a nominal-axial-wake eta_R: no swirl, no measured wake field; the absolute value of eta_R has no experimental anchor inside the evidence surface (WITHHELD); only its non-constancy, window and sensitivity are claimed; the eta_D chain is algebraic; it cannot validate eta_R (see caveats) |

实现：新增 `src/tensorbem/wake_field.py`（标称伴流剖面、后船身入流场、ηR 定义式、敞水曲线插值），并给 `propeller.open_water_point` 加了一个 `inflow_field` 钩子（最小改动：默认 None 时行为逐位不变，既有回归测试 `tests/test_propeller_open_water.py` 等 31 项全过）。

工具链：`PYTHONPATH=src python benchmarks/anchors/build_w31_etaR_chain.py`（本机，35.21 s，单点 ~0.3 s；提交集群无必要）。环境：Python 3.11.6，numpy 2.4.6。

## 3 计算结果

### 3.1 平台链路审计：ηR = 1.0 是结构性的

把平台返回的 K_T/K_Q 与它自己方法串里声明的 Oosterveld & van Oossanen (1975) B 系列回归（Z = 5、Ae/Ao = 0.70、P/D 按输入）逐状态比对：

| 状态 | J | K_T 平台 | K_T (O&O Z5 Ae/Ao .70) | ΔK_T | K_Q 平台 | K_Q (O&O) | ΔK_Q |
|---|---|---|---|---|---|---|---|
| S1 | 0.4298 | 0.2902 | 0.2902 | -4.32e-05 | 0.04348 | 0.04351 | +2.75e-05 |
| S2 | 0.4656 | 0.2746 | 0.2746 | +2.61e-05 | 0.04160 | 0.04163 | +2.95e-05 |
| S3 | 0.5014 | 0.2587 | 0.2587 | -7.45e-06 | 0.03966 | 0.03969 | +2.90e-05 |
| S4 | 0.5372 | 0.2424 | 0.2424 | -2.30e-05 | 0.03766 | 0.03769 | +2.66e-05 |
| S5 | 0.5731 | 0.2257 | 0.2257 | -4.67e-05 | 0.03559 | 0.03562 | +2.67e-05 |
| S6 | 0.6089 | 0.2086 | 0.2086 | +3.53e-05 | 0.03347 | 0.03349 | +2.14e-05 |
| S7 | 0.6320 | 0.1975 | 0.1975 | -1.65e-05 | 0.03206 | 0.03209 | +2.77e-05 |
| S8 | 0.5969 | 0.2144 | 0.2144 | -2.37e-05 | 0.03418 | 0.03421 | +3.06e-05 |
| S9 | 0.5655 | 0.2292 | 0.2292 | +2.30e-05 | 0.03603 | 0.03606 | +3.00e-05 |
| S10 | 0.5117 | 0.2541 | 0.2540 | -6.33e-05 | 0.03909 | 0.03912 | +2.92e-05 |
| S11 | 0.5372 | 0.2943 | 0.2944 | +6.64e-05 | 0.04872 | 0.04877 | +5.46e-05 |

最大偏差 |ΔK_T| = 6.64e-05、|ΔK_Q| = 5.46e-05，回执精度为 4 位小数 —— 即平台的「后船身」K_T/K_Q 就是它的**敞水曲线**在同一 J 处的取值。结论：the platform's behind-hull K_T/K_Q reproduce its open-water regression to receipt precision, so eta_R = 1.0 is structural, not computed。

平台不报告展开面积比，本单不假设它而是**辨识**它：在预登记阶梯 Ae/Ao ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90} 上重算 O&O 回归，取 11 状态最大 |ΔK_T| 最小的那个：

| Ae/Ao | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|---|---|
| 最大 \|ΔK_T\| | 5.8e-03 | 3.7e-03 | 1.7e-03 | 6.6e-05 | 1.5e-03 | 2.7e-03 | 3.7e-03 | 4.4e-03 |

最小残差出现在 Ae/Ao = **0.70**（6.6e-05），相邻两档分别大 22× 与 26×，因此这不是一个假设而是一个可证伪的辨识结果：平台的敞水曲线就是 Z = 5、Ae/Ao = 0.70、P/D 按输入的 O&O 回归。

### 3.2 逐状态 ηR：三口径（+ 独立功率口径）

| 状态 | J | ηR 平台返回值 | ηR 平台恒等反解 | **ηR 敞水插值/后船身反解** | ηR 敞水直接解（无插值） | ηR 独立 P_E/P_D 反解 | 窗内? |
|---|---|---|---|---|---|---|---|
| S1 | 0.4298 | 1.0000 | 0.9999 | **0.9854** | 0.9861 | 0.5199 | in |
| S2 | 0.4656 | 1.0000 | 0.9999 | **0.9843** | 0.9849 | 0.6697 | in |
| S3 | 0.5014 | 1.0000 | 0.9999 | **0.9836** | 0.9836 | 0.8771 | in |
| S4 | 0.5372 | 1.0000 | 1.0000 | **0.9818** | 0.9823 | 1.1715 | in |
| S5 | 0.5731 | 1.0000 | 0.9999 | **0.9801** | 0.9808 | 1.5961 | in |
| S6 | 0.6089 | 1.0000 | 0.9999 | **0.9787** | 0.9792 | 2.2109 | in |
| S7 | 0.6320 | 1.0000 | 1.0000 | **0.9773** | 0.9781 | 1.9901 | in |
| S8 | 0.5969 | 1.0000 | 1.0000 | **0.9796** | 0.9798 | 1.6352 | in |
| S9 | 0.5655 | 1.0000 | 0.9999 | **0.9805** | 0.9812 | 1.3724 | in |
| S10 | 0.5117 | 1.0000 | 0.9999 | **0.9828** | 0.9832 | 1.0136 | in |
| S11 | 0.5372 | 1.0000 | 0.9999 | **0.9845** | 0.9848 | 0.9645 | in |

统计（11 状态）：敞水插值口径 std = **0.002521**（门限 0.001）、均值 0.9817；平台返回值 std = 0（恒 1.0 → 该门必 FAIL）；平台恒等反解 std = 5.563e-05、均值 0.9999；独立功率口径范围 0.520–2.211，仅 2/11 在窗内（复现 W30 的 9/11 越窗）。

**这里必须说清楚（否则这三列会被误读）**：ηD = η0·ηR·ηH 是 ηR 的**定义式**，所以「平台恒等反解」这一列永远 ≈ 1.0000（残差只来自回执 4 位小数），它对 ηR 没有诊断力；同理「独立 P_E/P_D 反解」这一列的 0.52–2.21 反映的是**推进块在给定 rpm 上的推力不匹配阻力**（平台不消费阻力、不反解自航点 rpm），不是 ηR 本身的测量值。真正有信息量的只有：敞水插值口径（本单新增）、q = 0 零对照、以及 §3.6 的 KP505 对表。

### 3.3 ηD = η0·ηR·ηH 恒等残差

| 状态 | η0 | ηR（计算） | ηH | ηD 链 | η0·ηR·ηH 乘回 | 相对残差 | ηD 平台 | 链 vs 平台 [%] |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.4565 | 0.9854 | 1.07 | 0.4813 | 0.4813 | 0.0e+00 | 0.4884 | -1.45 |
| S2 | 0.4892 | 0.9843 | 1.07 | 0.5152 | 0.5152 | 0.0e+00 | 0.5234 | -1.56 |
| S3 | 0.5205 | 0.9836 | 1.07 | 0.5478 | 0.5478 | 0.0e+00 | 0.5569 | -1.64 |
| S4 | 0.5503 | 0.9818 | 1.07 | 0.5781 | 0.5781 | 0.0e+00 | 0.5888 | -1.82 |
| S5 | 0.5783 | 0.9801 | 1.07 | 0.6065 | 0.6065 | 0.0e+00 | 0.6187 | -1.98 |
| S6 | 0.6042 | 0.9787 | 1.07 | 0.6327 | 0.6327 | 0.0e+00 | 0.6464 | -2.11 |
| S7 | 0.6196 | 0.9773 | 1.07 | 0.6479 | 0.6479 | 0.0e+00 | 0.6630 | -2.28 |
| S8 | 0.5958 | 0.9796 | 1.07 | 0.6245 | 0.6245 | 0.0e+00 | 0.6375 | -2.03 |
| S9 | 0.5726 | 0.9805 | 1.07 | 0.6008 | 0.6008 | 0.0e+00 | 0.6126 | -1.93 |
| S10 | 0.5292 | 0.9828 | 1.07 | 0.5565 | 0.5565 | 0.0e+00 | 0.5662 | -1.71 |
| S11 | 0.5166 | 0.9845 | 1.07 | 0.5442 | 0.5442 | 0.0e+00 | 0.5527 | -1.54 |

最大相对残差 **0.0e+00** ≤ 1e-6（C3 PASS）。该残差按构造为 0（链与乘积是同一次浮点运算），它证明的是**实现**把 ηD 由三个因子算出、没有走第二条硬编码通路，而不是 ηD 的正确性；有信息量的是最后一列：把 ηR 从 1.0 换成算出来的值后，链值相对平台 ηD 更低 1.45 … 2.28 %，即平台因 ηR≡1 把推进效率高估了同一幅度。

### 3.4 零对照（q = 0，均匀伴流）与插值偏差

| 状态 | ηR（敞水直接解为分子） | ηR（敞水插值为分子） | K_T 敞水 | K_T 后船身 | 插值偏差 [%] | 防重言守卫 |
|---|---|---|---|---|---|---|
| S1 | **1.000000000000** | 0.9994 | 0.321663 | 0.321663 | -0.0628 | 已触发 |
| S6 | **1.000000000000** | 0.9995 | 0.215691 | 0.215691 | -0.0530 | 已触发 |

均匀伴流下分子与分母是**同一次求解**，K_T 逐位相同，ηR 精确等于 1：代码通路已验证。以敞水曲线插值为分子时残留 5e-4–6e-4 的偏差，那是插值偏差（≤ 0.09 %），已单列，不是隐藏项。同时 `wake_field.relative_rotative_efficiency` 在分子分母相等时返回 `nan` 并告警（守卫信息存于工件 `tautology_guard_messages`），防止再一次退化成常数 1.0。

### 3.5 灵敏度：面元密度与伴流形状

| 状态 | 8 面元/叶 | 32 面元/叶（主） | 48 面元/叶 | 16×2 vs 24×2 漂移 |
|---|---|---|---|---|
| S1 | 0.9906 | 0.9854 | 0.9844 | 0.0010 |
| S4 | 0.9878 | 0.9818 | 0.9809 | 0.0009 |
| S7 | 0.9839 | 0.9773 | 0.9770 | 0.0003 |

| 状态 | q = 0（零对照） | q = 0.25 | q = 0.5（主） | q = 1.0 |
|---|---|---|---|---|
| S1 | 0.999372 | 0.9917 | 0.9854 | 0.9764 |
| S4 | 0.999491 | 0.9901 | 0.9818 | 0.9685 |
| S7 | 0.999134 | 0.9878 | 0.9773 | 0.9586 |

面元：主构型（32 面元/叶）与 48 面元/叶之间漂移 ≤ 0.001，粗构型（8 面元/叶）偏约 0.005 —— ηR 在 0.001 量级收敛，比状态间差异（std = 0.002521）小一个量级。形状：q 从 0 增到 1 时 ηR 单调下降（1.000 → 0.959–0.976），**整条带都还在 [0.90, 1.10] 内**，即 C2 的判定不依赖 q 的选值；但 ηR 的具体数值依赖 q，q 是假设（见 §5）。

### 3.6 敞水模型逐点对 KP505 实验曲线（C4）

锚件：`benchmarks/anchors/kp505_openwater.json` sha256 `9771a40b9a24e3f2`；Iftikhar & Takinaci (2023), Table 2 experimental columns (SECONDARY_REPRODUCTION)。

| J | K_T 实测 | K_T 涡格 | ΔK_T [%] | < 3 % ? | 10K_Q 实测 | 10K_Q 涡格 | Δ10K_Q [%] | < 3 % ? | Δη0 [%] | ΔK_T B 系列 [%] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.4940 | 0.5286 | +7.00 | NO | 0.6830 | 0.6529 | -4.41 | NO | +11.93 | -14.71 |
| 0.20 | 0.4470 | 0.4734 | +5.91 | NO | 0.6360 | 0.6104 | -4.03 | NO | +10.36 | -13.41 |
| 0.30 | 0.3920 | 0.4163 | +6.19 | NO | 0.5850 | 0.5599 | -4.29 | NO | +10.95 | -11.10 |
| 0.40 | 0.3430 | 0.3570 | +4.10 | NO | 0.5300 | 0.5014 | -5.39 | NO | +10.03 | -10.75 |
| 0.50 | 0.2900 | 0.2958 | +2.01 | YES | 0.4710 | 0.4349 | -7.66 | NO | +10.46 | -10.17 |
| 0.60 | 0.2410 | 0.2326 | -3.51 | NO | 0.4090 | 0.3605 | -11.86 | NO | +9.48 | -11.98 |
| 0.70 | 0.1930 | 0.1673 | -13.34 | NO | 0.3420 | 0.2781 | -18.70 | NO | +6.59 | -16.33 |
| 0.80 | 0.1410 | 0.0999 | -29.13 | NO | 0.2700 | 0.1876 | -30.50 | NO | +1.98 | -22.61 |
| 0.90 | 0.0870 | 0.0306 | -64.85 | NO | 0.1920 | 0.0892 | -53.52 | NO | -24.38 | -36.17 |
| 1.00 | 0.0290 | -0.0408 | -240.71 | NO | 0.1080 | 0.0171 | -84.13 | NO | -986.67 | -95.82 |

判据 C4：J ≤ 0.8 区（8 点）最差 |ΔK_T| = **29.13 %**（J = 0.8）→ **honest FAIL**（门限 15 %）。B 系列路线同样不达标：最差 22.61 %（J = 0.8）。逐项 < 3 %：涡格 1/8 点在 3 % 内（仅 J = [0.5]）。把 J = 0.8 排除后（J ≤ 0.7）最差 13.34 %，J ≤ 0.6 最差 7.00 % —— 两个数字照实报出，但**不**替换门限判定。11 个推进状态的 J 落在 0.4298–0.6320，落在这段区间内的 KP505 曲线点 J = 0.50（ΔK_T +2.01 %）、J = 0.60（ΔK_T -3.51 %），这是链自身的适用范围，也是为什么 ηR（比值）比绝对精度更可信。

### 3.7 pytest 运行输出（含非恒值断言）

命令：`PYTHONPATH=src python -m pytest tests/test_etaR_chain.py -q`，返回码 0：

```
........x......                                                          [100%]
=============================== warnings summary ===============================
../../../../usr/lib/python3.11/site-packages/anyio/pytest_plugin.py:14
  /usr/lib/python3.11/site-packages/anyio/pytest_plugin.py:14: PytestRemovedIn10Warning: _pytest.python.CallSpec2 has been renamed to CallSpec.
  The CallSpec2 alias will be removed in pytest 10.
  Update imports to use CallSpec instead.
  See https://docs.pytest.org/en/stable/deprecations.html#callspec2-renamed
    from _pytest.python import CallSpec2

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
XFAIL tests/test_etaR_chain.py::test_c4_point_by_point_against_kp505 - C4 is an honest FAIL: worst |dK_T| vs the KP505 measured curve over J<=0.8 is 29.13 % at J=0.8 (gate 15 %). Remove this xfail only together with a regenerated artefact and an updated doc.
14 passed, 1 xfailed, 1 warning in 6.43s
```

## 4 图

![openwater_etaR_chain](figures/w31/openwater_etaR_chain.png)

(a) 11 状态的 ηR：柱=本单计算值（敞水插值/后船身反解），黑点=平台自身乘积反解（≈1，定义式无用），红叉=独立 P_E/P_D 反解（0.52–2.21，反映自航点不匹配），黑虚线=平台返回的常数 1.0，绿带=[0.9, 1.1] 物理窗。(b) ηR 的两侧：同一 J 下敞水曲线插值（空心圆）与后船身解（方块）——后船身 η0 系统性高于敞水，比值即 ηR < 1。(c) 伴流形状灵敏度：q = 0 精确回到 1（零对照），q 增大 ηR 单调下降但仍全在窗内。(d) 敞水模型对 KP505 实验曲线的逐点偏差，灰线为 ±3 % 与 ±15 %，红竖线为 C4 区域边界 J = 0.8。

说明：本单为系数/标量级基准任务，涡格输出为 K_T、K_Q、η0 标量与面元载荷，无场量云图可给；四联图即全部定量比较（与 W30 的处理一致）。

## 5 注意点与局限

- the wake is axial only: no swirl and no radial velocity, because the platform reports no swirl. A wake swirl changes K_Q directly, so the eta_R above is a 'nominal axial wake' eta_R, not a full behind-hull eta_R.
- the radial wake shape exponent q is an ASSUMED shape (the measured KP505/KCS nominal wake is not available offline in this worktree); the declared q ladder and the q=0 null control bound what is and is not a measurement here.
- the lattice is non-dimensional (D=1) because the kernel's horseshoe influence matrix loses rank at D=6 in this build; the Reynolds number is preserved by n_eff = n_phys D_phys^2, and this is a kernel scale-invariance defect recorded rather than fixed here.
- the kernel returns torque with the bound-vorticity sign convention; K_Q is taken positive here (same convention flip as the registered KP505 case).
- receipts carry 4 dp on non-dimensional quantities and 2 dp on kW, which bounds any identity residual computed from receipts at ~1e-3 relative.
- the eta_D chain closes with the platform's eta_0 and eta_H; the eta_0 of the chain is therefore the platform's open-water value, not the lattice value, so C3 closes an algebraic identity and is not evidence that eta_D is right.
- 伴流形状 q 是**假设**而不是测量：本工作树内没有 KP505/KCS 的实测标称伴流场，所以 q 从未拟合，只用预登记阶梯 q ∈ {0, 0.25, 0.5, 1.0} 报出带子；q = 0 的零对照把「代码通路」与「物理效应」分开，这是本单能声称的边界。
- 平台回执的 `w`、`t` 是全船平均值（Holtrop），被当作 11 个状态的同一常数使用；这正是平台能提供的唯一伴流信息。
- 涡格核在高 J 端（J ≥ 0.7）失效是既有已知缺陷（W30 与本单独立复现：J = 0.8 时 -29.13 %），本单**没有**修它，也不得把它用于 J > 0.63 的精度声明。
- ηR 的物理窗 [0.90, 1.10] 是 W30 预登记口径；本单算得 0.9773–0.9854，落在窗内但接近下沿，与单桨船实测 ηR 的量级（文献常见 0.95–1.05）一致，但本单**没有**用实测自航数据校核 ηR 的绝对值（该锚不在证据面内，标 WITHHELD）。

## 6 误差判定（逐项回答「是否 < 3 %」）

| 量 | 基准 | 最大偏差 | < 15 % ? | < 3 % ? |
|---|---|---|---|---|
| ΔK_T（KP505，J ≤ 0.8，8 点） | Table 2 实验列（二手转写） | 29.13 %（J = 0.8） | **NO** | **NO**（1/8 点达标） |
| Δ10K_Q（KP505，J ≤ 0.8） | Table 2 实验列 | 30.50 % | **NO** | **NO**（0/8 点达标） |
| ηD 恒等残差（11 状态） | η0·ηR·ηH 乘回 | 0.0e+00（相对） | YES | **YES** |
| ηR 零对照（q = 0） | 定义值 1.0 | 0.0e+00（直接分子） | YES | **YES** |
| ηR 插值偏差 | 敞水直接解 | 0.087 % | YES | **YES** |
| ηR 面元漂移（32 → 48 面元/叶） | 自身 | 0.0010（绝对） | YES | YES |
| ηR 绝对值 | 实测自航锚 | NO-DATA-TBD（锚不在证据面内，标 WITHHELD） | — | — |

逐条回答：**ηD 恒等、ηR 零对照、插值偏差、面元漂移均 < 3 %**（恒等 0、零对照 0、插值 ≤ 0.09 %、面元漂移 ≤ 0.0010）；**敞水模型对 KP505 实验曲线不满足 < 3 %**（J ≤ 0.8 区 1/8 点），且连 < 15 % 门限也不满足（最差 29.13 %，在 J = 0.8）；**ηR 的绝对值没有基准可校**（本课题证据面内不存在实测 ηR 锚），标 WITHHELD，不声称精度。

## 7 判据结论

| 判据 | 内容 | 实测 | 门限 | 判定 |
|---|---|---|---|---|
| C1 | 非恒值门：11 状态 ηR 标准差 > 1e-3 | 0.002521 | 0.001 | **PASS** |
| C2 | 物理窗门：≥ 9/11 状态 ηR ∈ [0.90, 1.10] | 11/11 | 9/11 | **PASS** |
| C3 | 恒等门：ηD = η0·ηR·ηH 相对残差 ≤ 1e-6 | 0.0e+00 | 1e-06 | **PASS** |
| C4 | 逐点对表：J ≤ 0.8 区最差 \|ΔK_T\| < 15 % | 29.13 %（J = 0.8） | 15 % | **honest FAIL** |

逐条说明：
- C1 **PASS**：the platform's own eta_R series has std = 0, i.e. the gate FAILS on the current platform output -- that is what the gate is for。
- C2 **PASS**：window count for the computed (openwater_interp) eta_R; the platform's constant 1.0 also sits in the window, which is exactly why a window test alone cannot detect a constant -- C1 and C2 are reported together。
- C3 **PASS**：this closes the implemented chain and is algebraically exact by construction; the informative number is chain_vs_platform_pct, which is the amount by which the platform's eta_D differs from the chain with a computed eta_R。
- C4 **honest FAIL**：gate as stated over J <= 0.8; the same ladder restricted to J <= 0.7 gives 13.34 % (reported, not substituted for the verdict)。这一条是本单未销项的外部锚定量绑定部分：敞水段在高 J 端的模型缺陷没有在本单修，判据如实保留原始数字（-29.13 % @ J=0.8、J ≤ 0.7 最差 13.34 %、J ≤ 0.6 最差 7.00 %）。

**对下一单的接口**：(a) 推进块应把 ηR 换成算出来的量（本单给出可复用实现：`tensorbem.wake_field` + `open_water_point(inflow_field=...)`，一行式调用）；(b) 推进块仍不消费阻力、不自航反解 rpm，所以 ηD 与 P_E/P_D 仍不可比（本单独立功率口径 0.52–2.21 复现该缺陷）；(c) 涡格核 J ≥ 0.7 的片空化/前缘分离修正仍是敞水精度的前置条件。
