# 审计：XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE 的转动块为何先前不可判定，以及现在的真判定

**仓库** TensorBEM · **锚点** `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE` · **前置** m19943 / m20251（R2 原判 FAIL，最差 active 偏差 9859.64%）

> 本文由 `src/tensorbem/benchmarks/capytaine_crosscheck_r2.render_degeneracy_document()` 从 `benchmarks/verified/capytaine_crosscheck_r2/results.json` **程序化生成**（零手抄）：表中每个数字都可由工件重算，`tests/test_capytaine_r2_audit_doc_pins.py` 断言本文件与重渲染结果逐字节一致，因此文档不会随研究细化而失同步。

## 1. 案例介绍

R2 的目标是把 TensorBEM 的 **完整 6x6 辐射矩阵**（含转动块与平动-转动耦合）与开源面元法 code capytaine 对齐，验收标准：对标误差 <15%（owner 精度标准）。R2 用了四个本体：

| 本体 | 几何 / 转动中心 | 精确零自由度 | TensorBEM 面板 | capytaine 面板 |
|---|---|---|---:|---:|
| hemisphere_r1 | surface-piercing hemisphere, radius 1.0 m, centre (0, 0, 0)，转动中心 (0, 0, 0) | roll/pitch/yaw | 1024 | 784 |
| hemisphere_r1_cz-0.5 | surface-piercing hemisphere, radius 1.0 m, centre (0, 0, 0), rotation centre (0, 0, -0.5)，转动中心 (0, 0, -0.5) | yaw | 1024 | 784 |
| sphere_r1_z-0.5 | floating sphere, radius 1.0 m, centre (0, 0, -0.5)，转动中心 (0, 0, -0.5) | roll/pitch/yaw | 1026 | 1120 |
| sphere_r1_z-0.5_c0 | floating sphere, radius 1.0 m, centre (0, 0, -0.5), rotation centre at the waterline origin (0, 0, 0)，转动中心 (0, 0, 0) | yaw | 1026 | 1120 |

前两个本体把转动中心恰好放在**球面曲率中心**上；后两个保持几何不变、把转动中心移开，使 roll/pitch 成为真实非零量。流体 rho = 1025 kg/m^3, g = 9.80665 m/s^2，无限水深，nu = 0.5, 1, 2 m^-1。capytaine 侧：3.1.dev0（capytaine.BEMSolver() default settings, no lid mesh, no special quadrature override，Python 3.11.6）。

## 2. 怎么做的

1. **解析判定**：球面是其自身的曲率球，`(r - centre) x n ≡ 0` 逐点成立，故转动中心 = 球心时 三个转动自由度与全部平动-转动耦合项的精确目标是 **0**。对精确零做相对偏差，测到的是零的离散残差。
2. **细化阶梯实测**：`degeneracy_audit()` 在 hemisphere_r1 @ nu=1: 64 / 256 / 1024, sphere_r1_z-0.5 @ nu=1: 270 / 1026 面板上测残差随网格的走向（正确装配 → 残差衰减；力臂取自原点或漏掉 `(r - centre)` → 收敛到 O(1)）。
3. **不依赖外部程序的恒等式**：`centre_transfer_invariant()` 验证 `A(c+d) = T^T A(c) T`，`T = [[I, +[d]x], [0, I]]`（`V_c = V_c' + [d]x Omega`）。该式对离散算子亦精确成立，容差 1e-06。
4. **补做可判定的算例**：`hemisphere_r1_cz-0.5`、`sphere_r1_z-0.5_c0` 的转动中心偏离球心，roll/pitch 成为真实非零量；capytaine 侧参考数据由 `scripts/capytaine_crosscheck_r2_ref.py` 现场重算（第一轮冻结参考复现最差相对差 1.9e-14，见 3.4）。
5. **分开计分**：`compare()` 的登记带宽（对角 5%、耦合 10%）原样保留；`invariant_scoring()` 只把精确零项从相对评分中摘出，改按 3% x rho L^k 的 sanity 界限检查（结构量，不是收敛目标）。

## 3. 计算结果

### 3.1 关键结论一：不变量轴（精确零残差衰减 + 转动中心恒等式）

| 本体 | 面板阶梯 | 残差衰减比 | 最细网格残差 / rho L^k | 判定 |
|---|---|---|---:|---|
| hemisphere_r1 | 64 / 256 / 1024 | 3.47、3.84 | 1.1e-03 | PASS |
| sphere_r1_z-0.5 | 270 / 1026 | 3.36 | 2.8e-03 | PASS |
| 转动中心变换恒等式 | 64 | — | 最大相对残差 1.1e-16 | PASS |

不变量轴总判 **PASS**：转动残差随网格单调趋零、面积加权模态范数同步按 O(h) 收缩 → 这是**正确装配的离散残差**，不是收敛到非零极限的实现错误。旧的 9859.64% 是把两个数值零相减再除以其中一个。

### 3.2 关键结论二：可判定算例的真偏差（分开计分，诚实判定）

| case | 转动中心 | 零模态 dof | active | 结构零 | 精确零一致 | 精确零越界 | 最差计分偏差 | 物理判 | 不变量判 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| hemisphere_r1 | (0, 0, 0) | roll/pitch/yaw | 18 | 36 | 162 | 0 | 5.82% | FAIL | PASS |
| hemisphere_r1_cz-0.5 | (0, 0, -0.5) | yaw | 54 | 96 | 66 | 0 | 5.82% | FAIL | PASS |
| sphere_r1_z-0.5 | (0, 0, -0.5) | roll/pitch/yaw | 17 | 37 | 162 | 0 | 6.52% | FAIL | PASS |
| sphere_r1_z-0.5_c0 | (0, 0, 0) | yaw | 53 | 97 | 66 | 0 | 6.71% | FAIL | PASS |

计分失败项（8 条，逐项列出，不合并）：

* `hemisphere_r1 nu=2 damping A[heave,heave]: +5.82% (band 5%)`
* `hemisphere_r1_cz-0.5 nu=2 damping A[heave,heave]: +5.82% (band 5%)`
* `sphere_r1_z-0.5 nu=2 added_mass A[surge,surge]: -6.49% (band 5%)`
* `sphere_r1_z-0.5 nu=2 added_mass A[sway,sway]: -6.52% (band 5%)`
* `sphere_r1_z-0.5_c0 nu=2 added_mass A[surge,surge]: -6.49% (band 5%)`
* `sphere_r1_z-0.5_c0 nu=2 added_mass A[sway,sway]: -6.52% (band 5%)`
* `sphere_r1_z-0.5_c0 nu=2 added_mass A[roll,roll]: -6.71% (band 5%)`
* `sphere_r1_z-0.5_c0 nu=2 added_mass A[pitch,pitch]: -6.54% (band 5%)`

精确零界限（3% x rho L^k）在四个本体上无一越界（越界数合计 0），因此 3.2 的 FAIL 全部来自**可判定的物理项**，不再是退化伪影.

### 3.3 关键结论三：残差随 TensorBEM 网格单调收缩（本审计的解释轴）

| case | 层 | TensorBEM 面板 | capytaine 面板 | 最差对角计分偏差 | 最差耦合计分偏差 |
|---|---:|---:|---:|---:|---:|
| hemisphere_r1 | 2 | 64 | 784 | 23.08% | 0.00% |
| hemisphere_r1 | 3 | 256 | 784 | 9.38% | 0.00% |
| hemisphere_r1 | 4 | 1024 | 784 | 5.82% | 0.00% |
| hemisphere_r1_cz-0.5 | 2 | 64 | 784 | 23.08% | 15.06% |
| hemisphere_r1_cz-0.5 | 3 | 256 | 784 | 9.38% | 6.52% |
| hemisphere_r1_cz-0.5 | 4 | 1024 | 784 | 5.82% | 4.24% |
| sphere_r1_z-0.5 | 2 | 270 | 1120 | 8.75% | 0.00% |
| sphere_r1_z-0.5 | 3 | 1026 | 1120 | 6.52% | 0.00% |
| sphere_r1_z-0.5_c0 | 2 | 270 | 1120 | 8.90% | 8.15% |
| sphere_r1_z-0.5_c0 | 3 | 1026 | 1120 | 6.71% | 6.62% |

* `hemisphere_r1`：对角最差偏差 23.08% → 9.38% → 5.82% → 逐层下降（首末比 3.97x），说明剩下的偏差随 h 收缩；capytaine 侧面板数固定为 784，所以该量收敛到**两套网格之间的离散差**而不是零。
* `hemisphere_r1_cz-0.5`：对角最差偏差 23.08% → 9.38% → 5.82% → 逐层下降（首末比 3.97x），说明剩下的偏差随 h 收缩；capytaine 侧面板数固定为 784，所以该量收敛到**两套网格之间的离散差**而不是零。
* `sphere_r1_z-0.5`：对角最差偏差 8.75% → 6.52% → 逐层下降（首末比 1.34x），说明剩下的偏差随 h 收缩；capytaine 侧面板数固定为 1120，所以该量收敛到**两套网格之间的离散差**而不是零。
* `sphere_r1_z-0.5_c0`：对角最差偏差 8.90% → 6.71% → 逐层下降（首末比 1.33x），说明剩下的偏差随 h 收缩；capytaine 侧面板数固定为 1120，所以该量收敛到**两套网格之间的离散差**而不是零。

### 3.4 分层：raw 判定 / 分开计分 / 验收口径

| 口径 | 判据 | 结果 | 判定 |
|---|---|---|---|
| `compare()`（登记带宽，原样） | 对角 5% / 耦合 10% | 最差 active 6.71%，失败项 8 条 | FAIL |
| `invariant_scoring()`·物理项 | 同上 | 最差 6.71%，失败 8 条 | FAIL |
| `invariant_scoring()`·精确零界限 | 3% x rho L^k | 越界 0 格 | PASS |
| 锚点验收（owner 15%） | worst active relative deviation of the 6x6 radiation matrices <= 15% | 6.71% | PASS |

第一轮冻结参考的现场复现最差相对差 1.9e-14（18 项），说明参考数据本身没有被本轮改动漂移。

## 4. 结果图

* `docs/fig_capytaine_r2_matrix.png` — 四个本体 x 附加质量/阻尼的逐格偏差（NaN 为结构零）
* `docs/fig_capytaine_r2_convergence.png` — 最差偏差随面板数（即 3.3 的阶梯）
* `docs/fig_capytaine_r2_deep.png` — 深潜解析轴（Lamb 1932 极限）

图由 `render_figures()` 程序化生成并随 results.json 一并落盘。

## 5. 注意点

* 旧的 PASS/FAIL 都不可引用：转动块在中心 = 球心时目标恒为零，任何相对偏差都在测离散残差；`comparison` 里的 `worst_relative_any_entry = 1.8e+26` 就是这种退化相对量的产物，不可作为精度指标。
* 剩余偏差主要来自**网格分辨率不对等**：TensorBEM 1024 / 1026 面板 vs capytaine 784 / 1120 面板。3.3 的单调收缩说明 TensorBEM 侧仍在收敛前段；本轮没有把 TensorBEM 侧推到同等面板数（算力限制，如实记录）。
* 精确零界限（3% x rho L^k）是 **sanity bound**，不是物理判定；本轮不调宽它，也不因它通过而改判物理 FAIL。
* 带宽（对角 5% / 耦合 10%）事前登记、事后未放宽；owner 的 15% 验收口径与内部带宽分开记账。
* `deep_submergence` 一节沿用 m20251 已提交的计算结果（该代码路径本轮未改）。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 量 | 基准 | 测得最差偏差 | 是否 <3% | 是否 <5% 登记带 | 是否 <15% 验收 |
|---|---|---:|---|---|---|
| hemisphere_r1 最差计分项 | capytaine 784 面板 | 5.82% | 否 | 否 | 是 |
| hemisphere_r1_cz-0.5 最差计分项 | capytaine 784 面板 | 5.82% | 否 | 否 | 是 |
| sphere_r1_z-0.5 最差计分项 | capytaine 1120 面板 | 6.52% | 否 | 否 | 是 |
| sphere_r1_z-0.5_c0 最差计分项 | capytaine 1120 面板 | 6.71% | 否 | 否 | 是 |
| 转动中心变换恒等式 | 解析恒等 | 残差 1.1e-16 | 是（机器精度） | 是 | 是 |
| 退化残差（最细网格） | 0（解析） | 1.1e-03 x rho L^k | 不适用（目标为零） | 不适用 | 不适用 |
| 深潜球体 surge 附加质量 | Lamb (1932) 2146.8 kg | 0.60% | 是 | 是 | 是 |

**结论（诚实）**：不变量轴（退化残差、转动中心变换、深潜解析极限）全部通过，量级为机器精度 / 离散残差 / 解析极限的 0.60%；**可判定的物理项目前未通过 <3%，也未通过登记的 5% 对角带**，最差 6.71%，但满足 owner 的 15% 验收口径（满足）。这不是旧的伪影结论，而是把 R2 从「不可判定」变成「可判定且未达内部带宽」的实质进展；下一步须做等面板数重判（见第 5 节）。
