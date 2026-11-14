# CASE-BEM-HULME-SURGE-LOWFREQ: 浮半球水平（surge）长波极限 `A11(ka->0)` 锚定

> 锚点 `CASE-BEM-HULME-HEMISPHERE`（算例池 id 138，容差 `5%`）的水平分量验证欠账。
> 本文件由 `benchmarks/verified/casepool_bem_hulme_surge_lowfreq/limits.py` 从
> `results.json` 程序化渲染，正文不手抄任何数字。

## 1. 案例介绍

- 几何：half-submerged hemisphere, flat face flush with the mean free surface，a = 1.000 m，排水体积 ∇ = 2.094395 m³，水线面 πa² = 3.141593 m²
- 流体：ρ = 1025.0 kg/m³，g = 9.80665 m/s²，无限水深；ρ∇ = 2146.755 kg
- 模态与量纲：surge, unit velocity amplitude, exp(-i*omega*t)；A11/(rho*nabla), B11/(omega*rho*nabla)
- 极限语义：ka -> 0 is the rigid-wall (omega -> 0) limit of the free surface

### 为什么需要这个包

锚点算例在 `ka ∈ [0.2, 2.0]` 带内已闭环（辐射/绕射、面元与参考项阶梯、capytaine 第三方路线、Haskind 互易、长波激励力极限）。**heave** 列另有精确低频锚 `A33(ka→0) = 0.8309572 ρ∇`（`bem_rigid_lid_hemisphere_ka0`；滑移/平移之辨见其 erratum），**surge** 列此前没有任何带外锚——带内所有对照都在 `ka ≥ 0.2`，参考级数若在长波段整体偏移仍可全绿。本包用精确闭式把该端钉死。

### 精确锚的推导

**A11(ka->0) = A_sph/2 = rho*nabla/2 (exact)**

1. omega -> 0: dphi/dz - nu*phi = 0 degenerates to the rigid wall dphi/dz = 0 on z = 0 (r > a)
2. even continuation z -> -z preserves the horizontal Neumann data u.n = U*n_x
3. the mirrored problem is a full sphere translating in x in unbounded fluid
4. Kirchhoff: A_sph = rho*(4/3)*pi*a^3/2
5. T_doubled = 2*T_hemisphere at equal U  =>  A_hemi = A_sph/2 = rho*nabla/2
6. normalising by rho*nabla with nabla = (2/3)*pi*a^3 gives exactly 1/2

| 量 | 值 |
|---|---|
| 镜像球体积 | 4.188790 m³ |
| 半球排水体积 ∇ | 2.094395 m³ |
| Kirchhoff 球附连质量 A_sph | 2146.7550 kg |
| 半球附连质量 A11(0) = A_sph/2 | 1073.3775 kg |
| 无量纲 A11(0)/(ρ∇) | 0.500000 |

## 2. 怎么做的（四条独立路线）

| 路线 | 实现 | 独立性 |
|---|---|---|
| R1 闭式 | Kirchhoff 球 + 镜像能量减半 | 手推解析值，不依赖任何代码 |
| R2 谱展开 | `tensorbem.benchmarks.hemisphere_zero_frequency_envelope.spectral_added_mass('translation', n_terms=60)` | 数值复核镜像球常数（0.5ρVol_sph） |
| R3 参考级数 | `tensorbem.analytic.hulme_hemisphere_radiation`，ka = 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5，项数阶梯 [48, 96, 144]，quad_order = 512 | 半解析级数，与面元求解器无关 |
| R4 生产求解器 | `tensorbem.free_surface_solver3d.solve_radiation_3d` on `tensorbem.geometry3d.hemisphere_hull_mesh`，ka = 0.0010 面元阶梯 [2, 3, 4]，ka 阶梯 [0.001, 0.01, 0.03, 0.1]（sub = 3） | 3-D Kelvin 格林函数 BEM，与 R3 方法完全不同 |

关键设置：单位速度幅值、`exp(-iωt)`；参考项数阶梯 (48/96/144) 把「级数截断」与「物理极限差」分开；求解器在 `ka = 1e-3` 时 ω = 0.099 rad/s，自由液面条件 `∂φ/∂z = νφ`（ν = 1e-3）已进入刚壁极限。

## 3. 计算结果

### 3.1 R3 参考级数低频阶梯（144 项）

| ka | A11/(ρ∇) | A11 − 1/2 | (A11 − 1/2)/ka | B11/(ωρ∇) | A33/(ρ∇) |
|---|---|---|---|---|---|
| 0.002 | 0.500376 | 0.000376 | 0.188111 | 0.000000 | 0.838011 |
| 0.005 | 0.500945 | 0.000945 | 0.189051 | 0.000000 | 0.845035 |
| 0.010 | 0.501907 | 0.001907 | 0.190664 | 0.000001 | 0.853590 |
| 0.020 | 0.503880 | 0.003880 | 0.194018 | 0.000009 | 0.864813 |
| 0.050 | 0.510235 | 0.010235 | 0.204701 | 0.000142 | 0.876370 |
| 0.100 | 0.522322 | 0.022322 | 0.223219 | 0.001095 | 0.862736 |
| 0.200 | 0.551533 | 0.051533 | 0.257666 | 0.008153 | 0.793834 |
| 0.500 | 0.643897 | 0.143897 | 0.287794 | 0.098670 | 0.586085 |

级数截断漂移（48 vs 144 项）最大 2.63e-08（相对），比判据小四个量级以上，说明上表的差别是物理的、不是截断误差。

### 3.2 外推与极限

- 外推：ordinary least squares A11 = a0 + a1*ka over the 3 smallest ka；ka = 0.002, 0.005, 0.010
- 拟合截距 A11(0) = 0.499991（斜率 0.1914 /ka，最大残差 3.18e-06）
- 与精确值 1/2 的相对差：**-0.0017%**
- 最小 ka = 0.002 处直接值 0.500376，相对精确值 +0.0752%
- R2 谱展开：整球平移闭式 1.000000000（ρ∇ 单位），下半球积分（即半球附连质量）0.500000000，应为 1/2

### 3.3 R4 生产求解器（surge，面元阶梯 @ ka = 1e-3）

| sub | 面元 | A11/(ρ∇) | vs 精确 1/2 | 条件数 | 相对残差 | 用时 s |
|---|---|---|---|---|---|---|
| 2 | 64 | 0.473480 | -5.304% | 2.177 | 3.73e-16 | 0.1 |
| 3 | 256 | 0.493330 | -1.334% | 2.102 | 7.63e-16 | 1.1 |
| 4 | 1024 | 0.498475 | -0.305% | 2.062 | 1.25e-15 | 19.0 |

| ka | 面元 | A11/(ρ∇) | vs 精确 1/2 | B11/(ωρ∇) |
|---|---|---|---|---|
| 0.0010 | 256 | 0.493330 | -1.334% | 0.000000 |
| 0.0100 | 256 | 0.494990 | -1.002% | 0.000001 |
| 0.0300 | 256 | 0.498869 | -0.226% | 0.000030 |
| 0.1000 | 256 | 0.514692 | +2.938% | 0.001054 |

## 4. 结果图

`figure.png`（三面板，由 `write_figure` 从本 JSON 生成）：

1. 左：参考级数 A11/(ρ∇) vs ka（对数横轴）+ 精确渐近线 1/2 + 生产求解器阶梯点；
2. 中：(A11 − 1/2)/ka vs ka，显示超标量在窗口内近似常数（极限是有限值而非数值平台）；
3. 右：B11/(ωρ∇) 与求解器面元收敛的绝对相对误差 vs 面元数（观测收敛阶）。

## 5. 注意点

1. **滑移 vs 平移**：heave 的刚壁镜像把半球映成「对滑」两半（Neumann 数据变号），surge 的镜像则是**整球平移**（水平速度在 z→−z 下不变）。混用会得到 0.5 vs 0.831 的错误锚，这正是 m16802 erratum 的坑；本包的 R1 显式写出这一不对称性。
2. **极限只在 ka→0 成立**：带内 ka = 0.2 的参考值已比极限高 10.3%，所以带内 gate 仍必须以参考级数逐点为准，不能拿 1/2 去判带内数值。
3. **低频求解器行为**：ka = 1e-3 下系统仍可解（条件数、残差随表给出），误差主项是面元离散；因此面元阶梯是这一端的主要不确定性来源，而不是格林函数消波项。
4. **参考解仍非 Hulme 原文印刷表格**，而是本仓重推的级数；本包给它补的是**与级数无关**的低频端精确锚，这是相对带内 capytaine/Haskind 对照的增量。
5. **B11 的判据是定性的**：只要求它随 ka→0 单调趋零，没有解析基准（未做，标为未验证项）。

## 6. 误差判定（逐项 vs 基准，显式回答是否 < 3%）

基准 = 精确闭式 `A11(0)/(ρ∇) = 1/2`（R1，独立于代码与级数）。

| 关键量 | 计算值 | 基准 | 相对误差 | < 3%? |
|---|---|---|---|---|
| A11(0) 级数线性外推 | 0.499991432 | 1/2 | -0.0017% | 是 |
| A11(ka=2e-3) 级数直接值 | 0.500376223 | 1/2 | +0.0752% | 是 |
| A11(ka=1e-3) 求解器 1024 面元 | 0.498474813 | 1/2 | -0.3050% | 是 |
| R2 谱展开下半球（基准 1/2） | 0.500000000 | 1/2 | -0.0000% | 是 |

**总判定：PASS**（全部 gate 通过）

| gate | 判据 | 实测 | 结论 |
|---|---|---|---|
| G1_exact_closed_form | closed form A11(0)/(rho*nabla) = 1/2 to 1e-12 | 0.500000 | PASS |
| G2_doubled_body_self_test | independent spectral solve of the translating-sphere BC, integrated over the lower hemisphere, returns 1/2 (rho*nabla units) to 1e-6 | 0.500000 | PASS |
| G3_reference_at_min_ka_within_0p5pct | |A11(ka=2e-3)/0.5 - 1| < 0.5 % | 0.075245 | PASS |
| G4_extrapolated_limit_within_0p2pct | |linear extrapolation to ka=0 / 0.5 - 1| < 0.2 % | -0.001714 | PASS |
| G5_monotone_approach_from_above | excess A11-1/2 strictly increasing with ka (decreasing as ka falls), positive at the smallest ka | 3.762e-04, 9.453e-04, 1.907e-03, 3.880e-03, 1.024e-02, 2.232e-02, 5.153e-02, 1.439e-01 | PASS |
| G6_reference_term_truncation | |A11(144 terms) - A11(48 terms)| / A11 < 1e-4 at every ka | 0.000000 | PASS |
| G7_solver_finest_within_case_tolerance | |A11(ka=1e-3, 1024 panels)/(0.5 rho*nabla) - 1| < 5 % (case tolerance) | -0.305037 | PASS |
| G8_solver_mesh_monotone | |error| decreases 64 -> 256 -> 1024 panels | 5.304e+00, 1.334e+00, 3.050e-01 | PASS |
| G9_damping_vanishes | B11(ka=1e-3) < 1e-3 and B11 increases with ka over the ladder | min_ka=0.001, b11_dimless=1.133425038139812e-09 | PASS |
| G10_heave_crosslink_1pct | reference A33 at ka=2e-3 within 1 % of the settled rigid-lid heave anchor 0.8309572 (harness consistency) | 0.848864 | PASS |

## 7. 复现

```bash
PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_surge_lowfreq/run_case.py
PYTHONPATH=src pytest -q tests/test_casepool_hulme_surge_lowfreq.py
```

运行环境：Python 3.11.6，numpy 2.4.6，Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38；本包运行 0.5 s。
