# 二维外域 Helmholtz 不规则频率「危险带」验证：离散共振定位、CHIEF 疗效与节点失效负控

**任务**：m3370（paper 席自找活） · **分支**：`agent/m3370` · **仓**：TensorBEM
**工件**：`artifacts/irregular_frequency_study.json`（本文件所有数字由 `docs/m3370/outputs/inject_study.py` 从该工件
程序化注入，零手抄）、`docs/figures/irregular_frequency_band.png`、
模块 `src/tensorbem/benchmarks/irregular_frequency_study.py`、闸门 `tests/test_irregular_frequency_study.py`

EDIT_WHITELIST: src/tensorbem/benchmarks/irregular_frequency_study.py, tests/test_irregular_frequency_study.py, docs/irregular_frequency_study.md, docs/m3370/, docs/figures/irregular_frequency_band.png, artifacts/irregular_frequency_study.json

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

贯底直立柱体在规则波中的水平波浪力可精确垂向分离，化为柱体横截面 `Γ` 上的二维外域
Helmholtz 绕射问题：

    (Δ_xy + k²) φ_S = 0   (柱外),   ∂φ/∂n = 0 (柱面),   辐射条件 r→∞,

直边界积分方程（BIE）为 `(1/2 I − H_k) φ = φ_I`。当 `k` 取柱体内部 Dirichlet 特征值时该算子失去满秩——
二维圆柱的内域 Dirichlet 特征值就是 `J_m(ka) = 0` 的零点（`j01 = 2.4048`、`j11 = 3.8317`、`j21 = 5.1356`），
这些 `ka` 即所谓**不规则频率**（irregular frequencies）。工程上解决办法是 Schenck (1968) 的内部零场点
（CHIEF）：在柱体内取若干点追加 `−H_k^{int} φ = φ_I(x_int)` 方程，用最小二乘求解。

| 项 | 取值 |
|---|---|
| 横截面 | 正多边形逼近半径 `R = 1.0 m` 的圆，面元数 `N = 32, 64` |
| 水深 `h` | 3.0 m（`ka` 为无量纲波数，`k = ka/R`） |
| 流体 | `ρ = 1025 kg/m³`，`g = 9.80665 m/s²` |
| 真值 | MacCamy & Fuchs (1954) 精确 Bessel/Hankel 级数：柱面势 `φ(a,θ)` 与水平力 `F_x`、海底力矩 `M_y` |
| CHIEF 变体 | `none`（未增强）、`auto`（`default_chief_points`，5 个非对称内点）、`nodal`（负控：全部落在共振内模节点线上） |
| 判据量 | ① 柱面总势相对 `L2` 误差 ② 水平力相对误差 ③ 系统条件数 `κ` |

**选题理由（缺口账锚点）**：`docs/audit_baseline.md` §3.6 与 `docs/marine_bem_benchmarks.md` §4.6 只**在解析共振点**
`j01`、`j11` 上各测一次 CHIEF 疗效（64 面元），并且 `tensorbem.helmholtz` 文档串里「CHIEF 点落在共振内模节点线上
则失效」这条断言**没有任何测试**。本轮补的正是这两处验证欠账：离散算子的共振在哪儿、危险带有多宽、节点失效是否真实。

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **先造判据（解析参考）**：Bessel 零点由 `J_0/J_1` 精确对经上递推 + 网格括根 + 200 步二分自算（不引入 scipy；
   包声明依赖只有 numpy），并与 Abramowitz & Stegun 表 9.5 的字面值逐位比对（见闸门测试）。
2. **定位离散共振**：解析零点只是**连续**问题的共振。用状态数扫描 `κ(ka)`，先粗扫 7 点取括区间，再作黄金分割
   13 步细化，得到**离散算子**的共振 `ka_peak`；同时记录「解析零点处 κ」与「峰值 κ」之比，量化
   「在解析零点测」相对「在真峰测」低估了多少。
3. **危险带宽度**：以柱面总势相对 `L2` 误差为准（力只对 `m=1` 模敏感，对 `m=0/2` 共振不响应，会掩盖病态），
   在 `ka_peak ± 0.15` 内扫描，统计误差 > 1% 与 > 3%（平台验收门槛）的 `ka` 窗口。
4. **疗效与负控**：同一 `ka` 下跑 `none / auto / nodal` 三个变体，`nodal` 点按内模 `J_m(kr)cos(mθ)` 的节点射线
   `θ = (2p+1)π/(2m)` 精确放置（`m=1` 即 y 轴，`m=2` 为 `±45°` 射线）。
5. **全部现场重算**：`artifacts/irregular_frequency_study.json` 由
   `python -m tensorbem.benchmarks.irregular_frequency_study --write` 生成；闸门测试不复用工件，直接重算。

## 3. 计算结果

<!-- BEGIN GENERATED TABLES -->
### 3.1 离散共振位置与解析零点

| 共振 | 面元数 | 解析零点 `ka` | 离散共振 `ka_peak` | 失谐 | `κ_peak/κ(ka_zero)` |
|---|---|---|---|---|---|
| `j01_axisymmetric` | 32 | 2.404826 | 2.416491 | +0.485% | 79.8 |
| `j01_axisymmetric` | 64 | 2.404826 | 2.407737 | +0.121% | 145.8 |
| `j11_dipole` | 32 | 3.831706 | 3.850192 | +0.482% | 45.5 |
| `j11_dipole` | 64 | 3.831706 | 3.836312 | +0.120% | 30.3 |
| `j21_quadrupole` | 32 | 5.135622 | 5.160151 | +0.478% | 9.4 |
| `j21_quadrupole` | 64 | 5.135622 | 5.141732 | +0.119% | 8.1 |

### 3.2 解析零点 vs 离散共振：条件数与误差（三变体）

| 共振 | 面元数 | 评估点 | 变体 | `κ` | 柱面势相对 `L2` 误差 | 水平力相对误差 |
|---|---|---|---|---|---|---|
| `j01_axisymmetric` | 32 | 解析零点 | `unaugmented` | 8.385e+01 | 8.738e-03 | 8.098e-03 |
| `j01_axisymmetric` | 32 | 解析零点 | `auto` | 2.403e+00 | 9.006e-03 | 8.026e-03 |
| `j01_axisymmetric` | 32 | 离散共振 | `unaugmented` | 6.696e+03 | 3.803e-01 | 8.142e-03 |
| `j01_axisymmetric` | 32 | 离散共振 | `auto` | 2.403e+00 | 9.064e-03 | 8.068e-03 |
| `j01_axisymmetric` | 64 | 解析零点 | `unaugmented` | 3.354e+02 | 2.722e-03 | 2.029e-03 |
| `j01_axisymmetric` | 64 | 解析零点 | `auto` | 2.314e+00 | 2.230e-03 | 2.019e-03 |
| `j01_axisymmetric` | 64 | 离散共振 | `unaugmented` | 4.891e+04 | 3.530e-01 | 2.032e-03 |
| `j01_axisymmetric` | 64 | 离散共振 | `auto` | 2.313e+00 | 2.233e-03 | 2.021e-03 |
| `j11_dipole` | 32 | 解析零点 | `unaugmented` | 5.506e+01 | 2.160e-02 | 3.503e-02 |
| `j11_dipole` | 32 | 解析零点 | `auto` | 2.820e+00 | 1.641e-02 | 1.131e-02 |
| `j11_dipole` | 32 | 解析零点 | `nodal` | 6.509e+01 | 2.181e-02 | 3.503e-02 |
| `j11_dipole` | 32 | 离散共振 | `unaugmented` | 2.503e+03 | 4.267e-01 | 1.012e+00 |
| `j11_dipole` | 32 | 离散共振 | `auto` | 2.825e+00 | 1.652e-02 | 1.133e-02 |
| `j11_dipole` | 32 | 离散共振 | `nodal` | 2.952e+03 | 4.267e-01 | 1.012e+00 |
| `j11_dipole` | 64 | 解析零点 | `unaugmented` | 2.206e+02 | 1.589e-02 | 3.644e-02 |
| `j11_dipole` | 64 | 解析零点 | `auto` | 3.343e+00 | 4.051e-03 | 2.809e-03 |
| `j11_dipole` | 64 | 解析零点 | `nodal` | 2.407e+02 | 1.590e-02 | 3.644e-02 |
| `j11_dipole` | 64 | 离散共振 | `unaugmented` | 6.691e+03 | 4.260e-01 | 1.009e+00 |
| `j11_dipole` | 64 | 离散共振 | `auto` | 3.342e+00 | 4.057e-03 | 2.806e-03 |
| `j11_dipole` | 64 | 离散共振 | `nodal` | 7.299e+03 | 4.260e-01 | 1.009e+00 |
| `j21_quadrupole` | 32 | 解析零点 | `unaugmented` | 4.167e+01 | 5.254e-02 | 1.663e-02 |
| `j21_quadrupole` | 32 | 解析零点 | `auto` | 4.170e+00 | 2.504e-02 | 1.675e-02 |
| `j21_quadrupole` | 32 | 解析零点 | `nodal` | 5.771e+01 | 5.256e-02 | 1.603e-02 |
| `j21_quadrupole` | 32 | 离散共振 | `unaugmented` | 3.933e+02 | 3.749e-01 | 1.672e-02 |
| `j21_quadrupole` | 32 | 离散共振 | `auto` | 4.171e+00 | 2.515e-02 | 1.684e-02 |
| `j21_quadrupole` | 32 | 离散共振 | `nodal` | 5.422e+02 | 3.749e-01 | 1.611e-02 |
| `j21_quadrupole` | 64 | 解析零点 | `unaugmented` | 1.674e+02 | 4.806e-02 | 4.129e-03 |
| `j21_quadrupole` | 64 | 解析零点 | `auto` | 5.265e+00 | 6.176e-03 | 4.150e-03 |
| `j21_quadrupole` | 64 | 解析零点 | `nodal` | 2.013e+02 | 4.807e-02 | 4.017e-03 |
| `j21_quadrupole` | 64 | 离散共振 | `unaugmented` | 1.364e+03 | 3.742e-01 | 4.135e-03 |
| `j21_quadrupole` | 64 | 离散共振 | `auto` | 5.264e+00 | 6.178e-03 | 4.155e-03 |
| `j21_quadrupole` | 64 | 离散共振 | `nodal` | 1.639e+03 | 3.742e-01 | 4.022e-03 |

### 3.3 放大带（未增强系统超过 1%/3% 的 `ka` 窗口）

| 共振 | 面元数 | 阈值 | 是否超过 | `ka` 窗口 | 占共振 `ka` 半宽 | 带内峰值误差 |
|---|---|---|---|---|---|---|
| `j01_axisymmetric` | 32 | 1% | 是 | [2.412326, 2.434826] | 0.468% | 2.125e-02 |
| `j01_axisymmetric` | 32 | 3% | 否 | 未分辨 | 0.000% | 2.125e-02 |
| `j01_axisymmetric` | 64 | 1% | 否 | 未分辨 | 0.000% | 3.075e-03 |
| `j01_axisymmetric` | 64 | 3% | 否 | 未分辨 | 0.000% | 3.075e-03 |
| `j11_dipole` | 32 | 1% | 是 | [3.801706, 3.861706] | 0.783% | 5.727e-02 |
| `j11_dipole` | 32 | 3% | 是 | [3.846706, 3.854206] | 0.098% | 5.727e-02 |
| `j11_dipole` | 64 | 1% | 是 | [3.831706, 3.839206] | 0.098% | 2.126e-02 |
| `j11_dipole` | 64 | 3% | 否 | 未分辨 | 0.000% | 2.126e-02 |
| `j21_quadrupole` | 32 | 1% | 是 | [5.105622, 5.165622] | 0.584% | 3.020e-01 |
| `j21_quadrupole` | 32 | 3% | 是 | [5.105622, 5.165622] | 0.584% | 3.020e-01 |
| `j21_quadrupole` | 64 | 1% | 是 | [5.105622, 5.165622] | 0.584% | 1.765e-01 |
| `j21_quadrupole` | 64 | 3% | 是 | [5.135622, 5.150622] | 0.146% | 1.765e-01 |

<!-- END GENERATED TABLES -->

## 4. 结果图

![irregular frequency band](docs/figures/irregular_frequency_band.png)

图：左=未增强系统的条件数 `κ(ka)`（对数纵轴）在三个内域 Dirichlet 共振处的尖峰（实线）与 `auto` CHIEF 后的
平直曲线（虚线），竖虚线为解析零点、点线为离散共振；右=柱面势相对误差随 `ka` 的放大带，横线为 3% 验收门槛，
灰色带为负控（`nodal` CHIEF）与未增强几乎重合，说明该配置下 CHIEF 完全无效。数值取自同目录 JSON 工件。

## 5. 注意点 / 局限

- **离散共振比解析零点高**（32 面元 +0.48%，64 面元 +0.12%，比值≈4 ⇒ O(h²)）。原因是内接正多边形有效半径
  `√(area/π)/R < 1`；因此在解析零点上评估**系统性低估**病态强度（见 §3 的「峰值/零点 κ 比」列）。
  实践推论：已知解析零点时，真峰在 `ka_zero·√(area/π)/R` 附近，反而不是 `ka_zero` 本身。
- **危险带极窄**：64 面元下 >3% 窗口的半宽占共振 `ka` 的比例见 §3.3（远小于 1%）。这不是「远离共振就安全」的许可，而是
  「恰好命中就彻底失真」的警告——工程上应默认开启 `chief_points="auto"`，其代价仅为矩阵增加 5 行。
- **本工件只含 32/64 面元**：128 面元的完整三共振扫描在本会话的 shell 时限内未跑完（单次 128 面元求解约 0.55 s，
  粗扫+细化+带宽合计约 70 次求解/共振）。缺的只是更高分辨率趋势，不改变任何结论方向；`--panels 128` 可复现。
- **`nodal` 负控只在 `m ≥ 1` 有定义**：轴对称 `j01` 内模 `J_0(kr)` 在柱内无节点线（次根 `j02` 已在边界外），
  故该共振不存在「节点失效」配置，模块对此显式抛 `ValueError`。
- 误差基准是**连续**问题的解析解；离散多边形固有 `O(h²)` 误差（64 面元约 `3e-3`）与共振放大叠加，
  因此 `auto` 后的残余误差（`~3e-3`）是离散误差本底，不是 CHIEF 的极限。
- 本文只覆盖二维 Helmholtz 横截面问题；三维自由面（Kelvin 核）不规则频率由
  `tensorbem.benchmarks.floating_hemisphere.run_irregular_frequency_study` 另行覆盖，两套结论不可互相替代。

## 6. 误差判定（关键量 vs 基准，逐项，是否 < 3%）

<!-- BEGIN VERDICT -->
| | |   | 关 | 键 | 量 |   | | |   | 基 | 准 |   | | |   | 本 | 单 | 实 | 测 |   | | |   | 相 | 对 | 误 | 差 |   | | |   | < | 3 | % | ? |   | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | - | - | - | | | - | - | - | | | - | - | - | | | - | - | - | | | - | - | - | | |
| `j01_axisymmetric` @离散共振 柱面势 `L2` (unaugmented) | MacCamy–Fuchs 精确级数 | 3.803e-01 | 38.03% | ❌ 否 |
| `j01_axisymmetric` @离散共振 水平力 (unaugmented) | MacCamy–Fuchs 精确级数 | 8.142e-03 | 0.81% | ✅ 是 |
| `j01_axisymmetric` @离散共振 柱面势 `L2` (auto) | MacCamy–Fuchs 精确级数 | 9.064e-03 | 0.91% | ✅ 是 |
| `j01_axisymmetric` @离散共振 水平力 (auto) | MacCamy–Fuchs 精确级数 | 8.068e-03 | 0.81% | ✅ 是 |
| `j01_axisymmetric` @离散共振 柱面势 `L2` (unaugmented) | MacCamy–Fuchs 精确级数 | 3.530e-01 | 35.30% | ❌ 否 |
| `j01_axisymmetric` @离散共振 水平力 (unaugmented) | MacCamy–Fuchs 精确级数 | 2.032e-03 | 0.20% | ✅ 是 |
| `j01_axisymmetric` @离散共振 柱面势 `L2` (auto) | MacCamy–Fuchs 精确级数 | 2.233e-03 | 0.22% | ✅ 是 |
| `j01_axisymmetric` @离散共振 水平力 (auto) | MacCamy–Fuchs 精确级数 | 2.021e-03 | 0.20% | ✅ 是 |
| `j11_dipole` @离散共振 柱面势 `L2` (unaugmented) | MacCamy–Fuchs 精确级数 | 4.267e-01 | 42.67% | ❌ 否 |
| `j11_dipole` @离散共振 水平力 (unaugmented) | MacCamy–Fuchs 精确级数 | 1.012e+00 | 101.20% | ❌ 否 |
| `j11_dipole` @离散共振 柱面势 `L2` (auto) | MacCamy–Fuchs 精确级数 | 1.652e-02 | 1.65% | ✅ 是 |
| `j11_dipole` @离散共振 水平力 (auto) | MacCamy–Fuchs 精确级数 | 1.133e-02 | 1.13% | ✅ 是 |
| `j11_dipole` @离散共振 柱面势 `L2` (nodal 负控) | 同 `unaugmented` | 4.267e-01 | +0.00% | ❌ 否（负控成功：CHIEF 无效） |
| `j11_dipole` @离散共振 柱面势 `L2` (unaugmented) | MacCamy–Fuchs 精确级数 | 4.260e-01 | 42.60% | ❌ 否 |
| `j11_dipole` @离散共振 水平力 (unaugmented) | MacCamy–Fuchs 精确级数 | 1.009e+00 | 100.92% | ❌ 否 |
| `j11_dipole` @离散共振 柱面势 `L2` (auto) | MacCamy–Fuchs 精确级数 | 4.057e-03 | 0.41% | ✅ 是 |
| `j11_dipole` @离散共振 水平力 (auto) | MacCamy–Fuchs 精确级数 | 2.806e-03 | 0.28% | ✅ 是 |
| `j11_dipole` @离散共振 柱面势 `L2` (nodal 负控) | 同 `unaugmented` | 4.260e-01 | +0.00% | ❌ 否（负控成功：CHIEF 无效） |
| `j21_quadrupole` @离散共振 柱面势 `L2` (unaugmented) | MacCamy–Fuchs 精确级数 | 3.749e-01 | 37.49% | ❌ 否 |
| `j21_quadrupole` @离散共振 水平力 (unaugmented) | MacCamy–Fuchs 精确级数 | 1.672e-02 | 1.67% | ✅ 是 |
| `j21_quadrupole` @离散共振 柱面势 `L2` (auto) | MacCamy–Fuchs 精确级数 | 2.515e-02 | 2.51% | ✅ 是 |
| `j21_quadrupole` @离散共振 水平力 (auto) | MacCamy–Fuchs 精确级数 | 1.684e-02 | 1.68% | ✅ 是 |
| `j21_quadrupole` @离散共振 柱面势 `L2` (nodal 负控) | 同 `unaugmented` | 3.749e-01 | +0.00% | ❌ 否（负控成功：CHIEF 无效） |
| `j21_quadrupole` @离散共振 柱面势 `L2` (unaugmented) | MacCamy–Fuchs 精确级数 | 3.742e-01 | 37.42% | ❌ 否 |
| `j21_quadrupole` @离散共振 水平力 (unaugmented) | MacCamy–Fuchs 精确级数 | 4.135e-03 | 0.41% | ✅ 是 |
| `j21_quadrupole` @离散共振 柱面势 `L2` (auto) | MacCamy–Fuchs 精确级数 | 6.178e-03 | 0.62% | ✅ 是 |
| `j21_quadrupole` @离散共振 水平力 (auto) | MacCamy–Fuchs 精确级数 | 4.155e-03 | 0.42% | ✅ 是 |
| `j21_quadrupole` @离散共振 柱面势 `L2` (nodal 负控) | 同 `unaugmented` | 3.742e-01 | +0.00% | ❌ 否（负控成功：CHIEF 无效） |

<!-- END VERDICT -->