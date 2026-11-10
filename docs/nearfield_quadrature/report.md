# 近边界近奇异面元求积的可信域 + 半球垂荡辐射 Hulme 锚（IN50-W4-N02）

**分支**：`agent/m10370` · **仓**：TensorBEM · **节点**：IN50-W4-N02（TensorBEM 近场求积可信域成熟化）
**工件**：`artifacts/nearfield_quadrature/`（`gates_verified.json`、`a33_b33.csv`、`quadrature_error_law.csv`、
`quadrature_convergence.csv`、`energy_identity.csv`、`bem_energy_identity.csv`、`figures/`）
**用例**：`tests/test_nearfield_quadrature.py`（15 项；其中 2 项读取发布工件）
**复算**：

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.nearfield_quadrature --stage quadrature --artifacts artifacts/nearfield_quadrature
PYTHONPATH=src python3 -m tensorbem.benchmarks.nearfield_quadrature --stage energy    --artifacts artifacts/nearfield_quadrature
PYTHONPATH=src python3 -m tensorbem.benchmarks.nearfield_quadrature --stage bem_energy --artifacts artifacts/nearfield_quadrature --subdivisions 2 3 4
PYTHONPATH=src python3 -m tensorbem.benchmarks.nearfield_quadrature --stage hemisphere --artifacts artifacts/nearfield_quadrature --subdivisions 3 --ka-list 0.1 ... 2.0
PYTHONPATH=src python3 -m tensorbem.benchmarks.nearfield_quadrature --stage hemisphere --artifacts artifacts/nearfield_quadrature --subdivisions 4 --ka-list 0.1 0.25 ... 2.0
PYTHONPATH=src python3 -m tensorbem.benchmarks.nearfield_quadrature --stage gates --artifacts artifacts/nearfield_quadrature
PYTHONPATH=src python3 -m pytest tests/test_nearfield_quadrature.py -q
PYTHONPATH=src python3 docs/nearfield_quadrature/build_tables.py --inject
```

本文所有数字由 `build_tables.py` 从工件注入（零手抄）。

---

## 1. 案例介绍

### 1.1 物理背景

三维 Laplace 外问题的边界积分方程

```
(1/2 I - D) phi = -S q,   S_ik = int_{T_k} N_i G dS,   D_ik = int_{T_k} N_i dG/dn dS,
G(x,y) = 1/(4 pi |x-y|),  dG/dn_y = -n_y . (x - y) / (4 pi |x-y|^3)
```

在**近边界场点**上有一个被忽视的失效模式：当配点 `x` 到面元 `T` 的距离 `eps` 远小于面元尺寸 `h`
（离体表面压力、自由面上方的波面升高、薄缝隙与多体间隙、近壁速度），被积函数虽仍有界，但其峰宽
只有 `eps`。任何固定在参考单元上的 Gauss/Duffy 规则都在这个宽度上没有任何节点，误差随 `eps -> 0`
发散。求积误差因此直接变成"近场结果不可信"的先天缺陷，而不是可由加密网格消除的离散误差。

本单把这条链路推到**数学可信域可证**：两条互相独立的求积路线 + 一个半解析裁决者 + 一个必须失败的
负控，给出 `e(eps)` 与 `e(N)` 的实测误差律；再用半球垂荡辐射把它接到**解析基准**（Hulme 1982）与
**能量互证**（远场 Kochin 能量积分 vs 近场压力积分）上。

### 1.2 几何与参数

| 项 | 取值 |
|---|---|
| 求积面板 | 平面三角形 `(0,0,0) (1,0,0) (0,1,0)`，特征尺寸 `h = sqrt(2) = 1.41421` |
| 场点足迹（footprint） | 形心 `(1/3,1/3)` / 内部点 `(0.5,0.25)` / 近顶点 `(0.1,0.1)` |
| 场点位置 | 沿法向从足迹偏移 `eps = (eps/h) * h` |
| `eps/h` 阶梯 | `1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001` |
| 核 | 单层 `G = 1/(4 pi r)`、双层 `dG/dn_y = -d/(4 pi r^3)`（平面单元上 `n.(x-y) = d` 为常数） |
| 裁决者 | 极坐标 + 径向闭式积分（`int rho drho/r = R - eps`，`int rho drho/r^3 = 1/eps - 1/R`），θ 向 160 点 Gauss |
| 求解器 | `tensorbem.free_surface_solver3d`（Kelvin 深水 Green 函数、常数面元、`rankine_order=5`） |
| 半球网格 | 正八面体下半球递归四分：`subdivisions 2/3/4 -> 64/256/1024` 面元，`R = 1 m` |
| 频率 | `Ka = 0.1 ... 2.0`（`Ka = omega^2 R / g`，`nu = Ka/R`） |
| 流体 | `rho = 1025 kg/m^3`，`g = 9.80665 m/s^2`，时间因子 `exp(-i omega t)` |
| 解析基准 | Hulme (1982) 半球辐射/绕射半解析级数（`tensorbem.analytic.hulme_hemisphere_radiation`，`n_terms=96, quad_order=192`） |
| 归一化 | `A33/(rho R^3)`、`B33/(rho omega R^3)`（Hulme 原表按 `rho*V`、`V = 2 pi R^3/3` 归一，本文乘 `V/R^3 = 2 pi/3`） |

---

## 2. 怎么做的

### 2.1 路线 A：自适应三角形细分（`adaptive_triangle_quadrature`）

递归按三边中点四分面元；子三角形一旦满足"良分离"判据
`distance(x, sub) >= split_ratio * size(sub)`（本单 `split_ratio = 1`）就用 8 阶 Duffy 规则积分，
否则继续四分，直到层数预算 `max_level`。只有包含峰的那条链被细分，代价随 `log(h/eps)` 增长。
产出：两层积分值、核函数求值次数（= 工作量 `N`）、实际达到的层数。

### 2.2 路线 B：分级极坐标（`polar_triangle_quadrature`）

以场点在面板平面上的**垂足** `F` 为极点，`r = sqrt(rho^2 + eps^2)`，取

```
rho = eps * sinh(t)  =>  rho drho / r      = eps * sinh(t) dt,
                          rho drho / r**3  = (1/eps) * sinh(t) / cosh(t)^2 dt
```

两条被积函数都**解析且无峰**，峰宽 `eps` 被解析地吸收进变换，误差与 `eps` 无关。角向按
"垂足—顶点"连线切成 3 个扇区，扇区内射线只打在该扇区对应的三角形边上，`rho_max(theta)`
是射线与边段的精确交点（闭式），因此极坐标域被三角形**精确**覆盖、不重不漏。

### 2.3 独立裁决者与负控

* **裁决者**（`semi_analytic_reference`）：同样扇区分解，但径向用闭式积分，只剩 θ 向一维
  Gauss（160 点）。与 A/B 两条路线**不共享任何求积路径**，故三方对比能把求积误差和建模误差分开。
* **负控**（`plain_duffy_quadrature`）：常数面元 BEM 的默认 Duffy 规则（`kernels3d.duffy_quadrature`，
  8 阶）。它必须在 `eps -> 0` 上**失败**——否则上面两条路线"通过"只说明闸门松。

### 2.4 半球垂荡：解析场的独立重建 + 能量互证

为了把"近场 vs 远场"的互证做到 `1e-6` 以下，必须让两个读数属于**同一个辐射场**且该场严格满足
Laplace + 自由面条件 + 辐射条件。做法：

1. **独立重建 Hulme 垂荡势**（`hulme_heave_coefficients` / `hulme_heave_potential`）：湿表面
   `r = R, z <= 0` 上 `phi(mu) = sum_n c_n [P_{2n}(mu) + (Ka/2n) P_{2n-1}(mu)]` 加 Kelvin 源项
   `c_0 [2 + 2 Ka W(Ka s, Ka mu)]`，`mu = -z/R`；系数由体条件 `dphi/dr = -mu` 在奇次 Legendre
   试验函数上的 Galerkin 投影解出。`analytic.hulme_hemisphere_radiation` 的内部不暴露势函数，
   故此处重写同一级数；两者在同一 `n_terms` 上互钉到机器精度（见 §3.1）。
2. **近场压力积分**：`B33 = -rho omega Im int phi n_z dS`，即
   `-3 int_0^1 phi (-mu) dmu`（按 `rho omega V` 归一），600 点 Gauss。
3. **远场 Kochin 能量积分**：`K(theta) = int_S [-dphi/dn + nu phi (n_z - i(n_x cos + n_y sin))]
   exp(nu z - i nu (x cos + y sin)) dS`，μ 向 600 点 Gauss × 周向 96 点梯形 × 8 个 `theta`；
   然后 `B33_far = (rho omega k / 4 pi) int |K|^2 dtheta`。自由面平均能流为零，故两者在
   连续意义上**必须**相等，残差只剩"级数截断导致体条件只被满足到截断精度"这一项。
4. **结构自检**：轴对称垂荡的 `K(theta)` 必须与 `theta` 无关，且 `|K| = 4 pi |c_0|`（源项振幅）。

### 2.5 半球垂荡 BEM 对标（常数面元）

`hemisphere_hull_mesh(subdivisions)` + `solve_radiation_3d(modes=[2])` 得 `A33, B33`；
`far_field_damping_matrix_3d(n_azimuth=72)` 给远场读数。`Ka <= 2` 低于首个内部水线共振
（`Ka ~ 2.58`），故不加 CHIEF 行；每个工况记录矩阵条件数与线性解残差。

---

## 3. 计算结果

### 3.1 头条（程序化注入）

<!-- BEGIN:HEADLINE -->
* gates passing: **10/10** (``all_gates_pass = True``)
* quadrature worst relative error, adaptive route: 4.84e-11
* quadrature worst relative error, graded polar route: 1.35e-12
* negative control (plain Duffy) worst relative error: 9.79e-01
* hemisphere heave vs Hulme (1982), Ka = 0.1 .. 2.0: max |A33 err| = 5.28e-03, max |B33 err| = 1.12e-02
* energy identity (near-field pressure vs far-field Kochin), worst residual: 5.78e-07 at Ka = 1.0
<!-- END:HEADLINE -->

### 3.2 闸门表

<!-- BEGIN:GATES -->
| gate | claim | measured | threshold | verdict |
|---|---|---|---|---|
| Q1-adaptive-uniform | adaptive subdivision: relative error <= 1e-9 for every eps/h in [1e-3, 1] | 4.84e-11 | 1e-09 | PASS |
| Q2-polar-uniform | graded polar: relative error <= 1e-6 for every eps/h in [1e-3, 1] | 1.35e-12 | 1e-06 | PASS |
| Q3-convergence-order | both certified routes converge with a negative log-log slope of error vs kernel evaluations | adaptive=-2.59e+00, plain_duffy=-8.75e-03, polar=-6.18e+00 | < 0 | PASS |
| Q4-negative-control | untransformed Duffy rule (negative control) does not resolve the clearance: error at eps/h=1e-3 >= 1e-1 and >= 100x the eps/h=1 error | degradation_factor=3.31e+10, error_at_clearance_1=3.03e-12, error_at_clearance_1e-3=1.00e-01 | 0.1 | PASS |
| H1-heave-added-mass | hemisphere heave A33/(rho R^3) vs Hulme (1982) within 3% over the certified 1024-panel grid | 5.28e-03 | 0.03 | PASS |
| H2-heave-damping | hemisphere heave B33/(rho omega R^3) vs Hulme (1982) within 3% over the certified 1024-panel grid | 1.12e-02 | 0.03 | PASS |
| H3-frequency-sweep | sweep covers Ka = 0.1 .. 2 on a common mesh | [0.1, 2.0] | [0.1, 2.0] | PASS |
| E1-near-far-identity | far-field Kochin energy integral vs near-field pressure integral of the same radiating field: residual < 1e-6 | 5.78e-07 | 1e-06 | PASS |
| E2-kochin-axisymmetry | Kochin function of the axisymmetric heave field is theta-independent (relative spread < 1e-10) | 1.69e-15 | 1e-10 | PASS |
| E3-bem-near-far | BEM near-field vs far-field damping residual converges with panel refinement (finest mesh < 1e-3, positive order) | orders_in_h=[2.2061614801277085, 1.9256350380774288], residual_at_finest=1.44e-04 | 0.001 | PASS |
<!-- END:GATES -->

### 3.3 求积：`e(eps)` 与 `e(N)`

三路求积在**同一**半解析裁决者下的最坏相对误差与 `e(N)` 对数斜率（`eps/h = 1e-3` 处）：

<!-- BEGIN:QUAD_SUMMARY -->
| route | worst rel. error over eps/h in [1e-3, 1] | e(N) log-log slope |
|---|---|---|
| adaptive subdivision | 4.84e-11 | -2.59 |
| graded polar (sinh) | 1.35e-12 | -6.18 |
| plain Duffy (negative control) | 9.79e-01 | -0.009 |
Negative control: error at ``eps/h = 1`` is 3.03e-12, at ``eps/h = 1e-3`` it is 1.00e-01, i.e. a degradation factor of 3.31e+10.
<!-- END:QUAD_SUMMARY -->

`e(eps)`（固定分辨率，逐足迹、逐 `eps/h`）：

<!-- BEGIN:QUAD_LAW -->
| route | footprint | eps/h | single rel. err. | double rel. err. | kernel evals |
|---|---|---|---|---|---|
| adaptive | centroid | 1e+00 | 3.03e-12 | 4.84e-11 | 64 |
| adaptive | centroid | 3e-01 | 2.54e-15 | 3.43e-14 | 1024 |
| adaptive | centroid | 1e-01 | 2.96e-14 | 5.81e-13 | 6208 |
| adaptive | centroid | 3e-02 | 2.04e-14 | 6.16e-13 | 11392 |
| adaptive | centroid | 1e-02 | 1.61e-14 | 8.18e-13 | 19264 |
| adaptive | centroid | 3e-03 | 1.33e-14 | 7.42e-13 | 27136 |
| adaptive | centroid | 1e-03 | 1.32e-14 | 1.06e-12 | 32704 |
| adaptive | interior | 1e+00 | 8.31e-13 | 1.81e-11 | 64 |
| adaptive | interior | 3e-01 | 1.32e-15 | 1.19e-13 | 1024 |
| adaptive | interior | 1e-01 | 1.60e-14 | 1.07e-13 | 4864 |
| adaptive | interior | 3e-02 | 5.43e-14 | 1.59e-12 | 12160 |
| adaptive | interior | 1e-02 | 7.44e-14 | 6.65e-13 | 18688 |
| adaptive | interior | 3e-03 | 9.12e-14 | 7.83e-13 | 24064 |
| adaptive | interior | 1e-03 | 9.82e-14 | 2.01e-12 | 30208 |
| adaptive | near_vertex | 1e+00 | 1.28e-12 | 1.92e-11 | 64 |
| adaptive | near_vertex | 3e-01 | 3.12e-14 | 8.49e-13 | 1024 |
| adaptive | near_vertex | 1e-01 | 8.01e-14 | 1.72e-12 | 3712 |
| adaptive | near_vertex | 3e-02 | 9.75e-14 | 2.97e-12 | 9280 |
| adaptive | near_vertex | 1e-02 | 6.24e-14 | 4.82e-13 | 15808 |
| adaptive | near_vertex | 3e-03 | 6.52e-14 | 2.94e-13 | 23872 |
| adaptive | near_vertex | 1e-03 | 6.37e-14 | 6.35e-13 | 28096 |
| polar | centroid | 1e+00 | 0 | 3.77e-16 | 6144 |
| polar | centroid | 3e-01 | 7.27e-16 | 4.39e-16 | 6144 |
| polar | centroid | 1e-01 | 1.24e-15 | 8.74e-16 | 6144 |
| polar | centroid | 3e-02 | 1.13e-15 | 1.01e-15 | 6144 |
| polar | centroid | 1e-02 | 9.02e-16 | 1.04e-15 | 6144 |
| polar | centroid | 3e-03 | 1.03e-15 | 6.74e-16 | 6144 |
| polar | centroid | 1e-03 | 1.31e-15 | 7.80e-16 | 6144 |
| polar | interior | 1e+00 | 1.53e-15 | 3.85e-16 | 6144 |
| polar | interior | 3e-01 | 5.66e-16 | 4.77e-16 | 6144 |
| polar | interior | 1e-01 | 6.56e-16 | 7.49e-16 | 6144 |
| polar | interior | 3e-02 | 6.83e-16 | 7.72e-16 | 6144 |
| polar | interior | 1e-02 | 9.50e-16 | 5.82e-16 | 6144 |
| polar | interior | 3e-03 | 9.25e-16 | 6.76e-16 | 6144 |
| polar | interior | 1e-03 | 9.17e-16 | 8.92e-16 | 6144 |
| polar | near_vertex | 1e+00 | 1.35e-12 | 1.09e-13 | 6144 |
| polar | near_vertex | 3e-01 | 5.38e-13 | 6.01e-16 | 6144 |
| polar | near_vertex | 1e-01 | 3.38e-13 | 7.53e-16 | 6144 |
| polar | near_vertex | 3e-02 | 2.66e-13 | 8.70e-16 | 6144 |
| polar | near_vertex | 1e-02 | 2.45e-13 | 9.68e-16 | 6144 |
| polar | near_vertex | 3e-03 | 2.37e-13 | 7.97e-16 | 6144 |
| polar | near_vertex | 1e-03 | 2.35e-13 | 8.96e-16 | 6144 |
| plain_duffy | centroid | 1e+00 | 3.03e-12 | 4.84e-11 | 64 |
| plain_duffy | centroid | 3e-01 | 3.34e-06 | 4.27e-05 | 64 |
| plain_duffy | centroid | 1e-01 | 1.97e-03 | 2.48e-02 | 64 |
| plain_duffy | centroid | 3e-02 | 3.44e-02 | 4.16e-01 | 64 |
| plain_duffy | centroid | 1e-02 | 7.38e-02 | 7.87e-01 | 64 |
| plain_duffy | centroid | 3e-03 | 9.39e-02 | 9.37e-01 | 64 |
| plain_duffy | centroid | 1e-03 | 1.00e-01 | 9.79e-01 | 64 |
| plain_duffy | interior | 1e+00 | 8.31e-13 | 1.81e-11 | 64 |
| plain_duffy | interior | 3e-01 | 2.98e-06 | 3.56e-05 | 64 |
| plain_duffy | interior | 1e-01 | 8.19e-04 | 1.55e-02 | 64 |
| plain_duffy | interior | 3e-02 | 2.91e-02 | 3.79e-01 | 64 |
| plain_duffy | interior | 1e-02 | 6.82e-02 | 7.68e-01 | 64 |
| plain_duffy | interior | 3e-03 | 8.90e-02 | 9.31e-01 | 64 |
| plain_duffy | interior | 1e-03 | 9.57e-02 | 9.77e-01 | 64 |
| plain_duffy | near_vertex | 1e+00 | 1.28e-12 | 1.92e-11 | 64 |
| plain_duffy | near_vertex | 3e-01 | 9.51e-08 | 9.20e-07 | 64 |
| plain_duffy | near_vertex | 1e-01 | 7.78e-05 | 1.61e-03 | 64 |
| plain_duffy | near_vertex | 3e-02 | 1.76e-03 | 1.21e-02 | 64 |
| plain_duffy | near_vertex | 1e-02 | 1.16e-02 | 4.43e-01 | 64 |
| plain_duffy | near_vertex | 3e-03 | 3.05e-02 | 8.21e-01 | 64 |
| plain_duffy | near_vertex | 1e-03 | 3.82e-02 | 9.41e-01 | 64 |
<!-- END:QUAD_LAW -->

`e(N)`（固定 `eps/h = 1e-3`，看工作量而非阶数参数——两条路线的"参数量"不可直接比较）：

<!-- BEGIN:QUAD_CONV -->
| route | resolution | kernel evals | single rel. err. | double rel. err. |
|---|---|---|---|---|
| adaptive | max_level=0 | 64 | 1.00e-01 | 9.79e-01 |
| adaptive | max_level=1 | 256 | 4.30e-02 | 9.48e-01 |
| adaptive | max_level=2 | 1024 | 1.98e-02 | 8.92e-01 |
| adaptive | max_level=3 | 3712 | 9.66e-03 | 8.06e-01 |
| adaptive | max_level=4 | 8128 | 2.92e-03 | 5.83e-01 |
| adaptive | max_level=5 | 12544 | 6.78e-04 | 2.91e-01 |
| adaptive | max_level=6 | 16960 | 1.05e-04 | 8.18e-02 |
| adaptive | max_level=7 | 21376 | 1.16e-06 | 2.15e-03 |
| adaptive | max_level=8 | 25792 | 1.78e-09 | 2.09e-06 |
| adaptive | max_level=9 | 29440 | 1.24e-11 | 4.04e-08 |
| adaptive | max_level=10 | 32704 | 1.32e-14 | 1.06e-12 |
| polar | radial=angular=2 | 24 | 1.82e-01 | 4.00e-01 |
| polar | radial=angular=3 | 54 | 2.16e-02 | 9.88e-02 |
| polar | radial=angular=4 | 96 | 2.84e-03 | 2.82e-02 |
| polar | radial=angular=6 | 216 | 1.31e-04 | 7.79e-04 |
| polar | radial=angular=8 | 384 | 7.72e-06 | 5.72e-05 |
| polar | radial=angular=12 | 864 | 2.78e-08 | 9.58e-07 |
| polar | radial=angular=16 | 1536 | 1.01e-10 | 3.29e-09 |
| polar | radial=angular=24 | 3456 | 1.16e-15 | 4.28e-14 |
| polar | radial=angular=32 | 6144 | 1.31e-15 | 7.80e-16 |
| plain_duffy | order=2 | 4 | 3.75e-01 | 9.97e-01 |
| plain_duffy | order=3 | 9 | 4.02e-02 | 9.78e-01 |
| plain_duffy | order=4 | 16 | 1.85e-01 | 9.90e-01 |
| plain_duffy | order=6 | 36 | 1.23e-01 | 9.83e-01 |
| plain_duffy | order=8 | 64 | 1.00e-01 | 9.79e-01 |
| plain_duffy | order=12 | 144 | 6.51e-02 | 9.66e-01 |
| plain_duffy | order=16 | 256 | 4.81e-02 | 9.54e-01 |
<!-- END:QUAD_CONV -->

图：`figures/quadrature_error_vs_clearance.png`（`e(eps)`，单层/双层两个子图）、
`figures/quadrature_convergence.png`（`e(N)`）。

### 3.4 半球垂荡 `A33 / B33` 与 Hulme (1982)

<!-- BEGIN:A33B33 -->
| Ka | panels | A33/(rho R^3) BEM | A33/(rho R^3) Hulme | err | B33/(rho w R^3) BEM | B33/(rho w R^3) Hulme | err | BEM near/far resid. | certified |
|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 64 | 1.75282 | 1.80693 | -2.994% | 0.36458 | 0.38038 | -4.153% | 3.94e-04 |  |
| 0.10 | 256 | 1.79356 | 1.80693 | -0.740% | 0.37640 | 0.38038 | -1.046% | 1.24e-04 |  |
| 0.10 | 1024 | 1.80359 | 1.80693 | -0.185% | 0.37938 | 0.38038 | -0.264% | 3.27e-05 | yes |
| 0.20 | 256 | 1.65219 | 1.66262 | -0.628% | 0.58021 | 0.58498 | -0.815% | 2.27e-04 |  |
| 0.20 | 1024 | 1.66002 | 1.66262 | -0.156% | 0.58378 | 0.58498 | -0.205% | 5.98e-05 | yes |
| 0.25 | 64 | 1.54237 | 1.57992 | -2.377% | 0.62514 | 0.64344 | -2.845% | 8.19e-04 |  |
| 0.25 | 1024 | 1.57764 | 1.57992 | -0.144% | 0.64230 | 0.64344 | -0.177% | 7.08e-05 | yes |
| 0.30 | 256 | 1.49090 | 1.49893 | -0.536% | 0.67744 | 0.68144 | -0.587% | 3.02e-04 |  |
| 0.40 | 256 | 1.34491 | 1.35131 | -0.474% | 0.71167 | 0.71423 | -0.358% | 3.45e-04 |  |
| 0.50 | 64 | 1.20476 | 1.22751 | -1.853% | 0.70555 | 0.71013 | -0.645% | 9.34e-04 |  |
| 0.50 | 256 | 1.22206 | 1.22751 | -0.444% | 0.70925 | 0.71013 | -0.124% | 3.48e-04 |  |
| 0.50 | 1024 | 1.22615 | 1.22751 | -0.110% | 0.70991 | 0.71013 | -0.031% | 9.25e-05 | yes |
| 0.60 | 256 | 1.12198 | 1.12702 | -0.447% | 0.68595 | 0.68514 | +0.118% | 3.04e-04 |  |
| 0.70 | 256 | 1.04190 | 1.04694 | -0.481% | 0.65127 | 0.64887 | +0.369% | 2.04e-04 |  |
| 0.75 | 64 | 0.99243 | 1.01351 | -2.080% | 0.63942 | 0.62844 | +1.747% | 1.28e-05 |  |
| 0.75 | 1024 | 1.01222 | 1.01351 | -0.127% | 0.62923 | 0.62844 | +0.125% | 3.57e-05 | yes |
| 0.80 | 256 | 0.97861 | 0.98394 | -0.542% | 0.61100 | 0.60717 | +0.631% | 3.77e-05 |  |
| 0.90 | 256 | 0.92909 | 0.93496 | -0.627% | 0.56867 | 0.56359 | +0.903% | 2.08e-04 |  |
| 1.00 | 64 | 0.87124 | 0.89732 | -2.907% | 0.54322 | 0.52027 | +4.412% | 2.53e-03 |  |
| 1.00 | 256 | 0.89077 | 0.89732 | -0.730% | 0.52644 | 0.52027 | +1.186% | 5.49e-04 |  |
| 1.00 | 1024 | 0.89568 | 0.89732 | -0.183% | 0.52181 | 0.52027 | +0.297% | 1.44e-04 | yes |
| 1.10 | 256 | 0.86148 | 0.86883 | -0.846% | 0.48556 | 0.47849 | +1.479% | 1.00e-03 |  |
| 1.20 | 256 | 0.83944 | 0.84768 | -0.971% | 0.44678 | 0.43895 | +1.784% | 1.60e-03 |  |
| 1.25 | 64 | 0.80529 | 0.83937 | -4.061% | 0.45109 | 0.42015 | +7.363% | 7.41e-03 |  |
| 1.25 | 1024 | 0.83718 | 0.83937 | -0.261% | 0.42219 | 0.42015 | +0.485% | 5.21e-04 | yes |
| 1.30 | 256 | 0.82321 | 0.83238 | -1.101% | 0.41047 | 0.40203 | +2.098% | 2.36e-03 |  |
| 1.40 | 256 | 0.81161 | 0.82175 | -1.234% | 0.37676 | 0.36785 | +2.422% | 3.33e-03 |  |
| 1.50 | 64 | 0.77151 | 0.81482 | -5.316% | 0.37202 | 0.33641 | +10.584% | 1.60e-02 |  |
| 1.50 | 256 | 0.80369 | 0.81482 | -1.367% | 0.34569 | 0.33641 | +2.756% | 4.56e-03 |  |
| 1.50 | 1024 | 0.81201 | 0.81482 | -0.345% | 0.33873 | 0.33641 | +0.688% | 1.23e-03 | yes |
| 1.60 | 256 | 0.79867 | 0.81083 | -1.500% | 0.31716 | 0.30763 | +3.098% | 6.12e-03 |  |
| 1.70 | 256 | 0.79592 | 0.80915 | -1.635% | 0.29105 | 0.28135 | +3.447% | 8.13e-03 |  |
| 1.75 | 64 | 0.75578 | 0.80902 | -6.580% | 0.30687 | 0.26910 | +14.036% | 3.09e-02 |  |
| 1.75 | 1024 | 0.80553 | 0.80902 | -0.431% | 0.27153 | 0.26910 | +0.902% | 2.55e-03 | yes |
| 1.80 | 256 | 0.79494 | 0.80928 | -1.772% | 0.26721 | 0.25742 | +3.803% | 1.07e-02 |  |
| 1.90 | 256 | 0.79527 | 0.81081 | -1.917% | 0.24546 | 0.23565 | +4.164% | 1.42e-02 |  |
| 2.00 | 64 | 0.74894 | 0.81342 | -7.927% | 0.25391 | 0.21587 | +17.620% | 5.81e-02 |  |
| 2.00 | 256 | 0.79653 | 0.81342 | -2.076% | 0.22564 | 0.21587 | +4.526% | 1.89e-02 |  |
| 2.00 | 1024 | 0.80912 | 0.81342 | -0.528% | 0.21830 | 0.21587 | +1.124% | 5.24e-03 | yes |
<!-- END:A33B33 -->

网格加密的观测阶（`h ~ panels^{-1/2}`）：

<!-- BEGIN:REFINEMENT -->
| Ka | panels | A33 rel. error | B33 rel. error |
|---|---|---|---|
| 0.1 | 64 | -2.994% | -4.153% |
| 0.1 | 256 | -0.740% | -1.046% |
| 0.1 | 1024 | -0.185% | -0.264% |
| 0.2 | 256 | -0.628% | -0.815% |
| 0.2 | 1024 | -0.156% | -0.205% |
| 0.25 | 64 | -2.377% | -2.845% |
| 0.25 | 1024 | -0.144% | -0.177% |
| 0.5 | 64 | -1.853% | -0.645% |
| 0.5 | 256 | -0.444% | -0.124% |
| 0.5 | 1024 | -0.110% | -0.031% |
| 0.75 | 64 | -2.080% | +1.747% |
| 0.75 | 1024 | -0.127% | +0.125% |
| 1 | 64 | -2.907% | +4.412% |
| 1 | 256 | -0.730% | +1.186% |
| 1 | 1024 | -0.183% | +0.297% |
| 1.25 | 64 | -4.061% | +7.363% |
| 1.25 | 1024 | -0.261% | +0.485% |
| 1.5 | 64 | -5.316% | +10.584% |
| 1.5 | 256 | -1.367% | +2.756% |
| 1.5 | 1024 | -0.345% | +0.688% |
| 1.75 | 64 | -6.580% | +14.036% |
| 1.75 | 1024 | -0.431% | +0.902% |
| 2 | 64 | -7.927% | +17.620% |
| 2 | 256 | -2.076% | +4.526% |
| 2 | 1024 | -0.528% | +1.124% |
<!-- END:REFINEMENT -->

认证口径：**只有 1024 面元那一行是"认证读数"**（表最后一列 `certified`）。64/256 面元的行是收敛曲线，
不参与精度声明——256 面元在 `Ka > 1.5` 已经超 3%（`Ka = 2` 处 `B33 +4.5%`），这是"每个波长 256 面元不够"
的网格需求，不是核函数误差。1024 面元认证网格覆盖 `Ka = 0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0`
（10 点，跨 `0.1 .. 2`），逐点误差见上表 `err` 列。

图：`figures/a33_b33_vs_ka.png`（左 `A33/(rho R^3)`，右 `B33/(rho omega R^3)`；黑线 = Hulme 解析）。

### 3.5 能量互证：近场压力积分 vs 远场 Kochin 能量积分

解析辐射场（同一势函数的两个读数）：

<!-- BEGIN:ENERGY -->
| Ka | n_terms | B33 near (rho w V) | B33 far (rho w V) | residual | K(theta) spread |
|---|---|---|---|---|---|
| 0.10 | 1024 | 0.181614 | 0.181614 | 2.60e-07 | 6.61e-16 |
| 0.20 | 1024 | 0.279301 | 0.279300 | 2.85e-07 | 8.61e-16 |
| 0.30 | 1024 | 0.325356 | 0.325356 | 3.15e-07 | 4.23e-16 |
| 0.40 | 1024 | 0.341010 | 0.341009 | 3.50e-07 | 5.12e-16 |
| 0.50 | 1024 | 0.339053 | 0.339053 | 3.90e-07 | 4.73e-16 |
| 0.60 | 1024 | 0.327120 | 0.327120 | 4.34e-07 | 1.05e-15 |
| 0.70 | 1024 | 0.309804 | 0.309804 | 4.78e-07 | 5.19e-16 |
| 0.80 | 1024 | 0.289892 | 0.289892 | 5.19e-07 | 9.19e-16 |
| 0.90 | 1024 | 0.269081 | 0.269081 | 5.54e-07 | 8.33e-16 |
| 1.00 | 1024 | 0.248398 | 0.248398 | 5.78e-07 | 6.35e-16 |
| 1.20 | 1024 | 0.209573 | 0.209573 | 5.73e-07 | 7.54e-16 |
| 1.40 | 1024 | 0.175625 | 0.175624 | 4.73e-07 | 9.97e-16 |
| 1.60 | 1024 | 0.146869 | 0.146869 | 2.62e-07 | 5.10e-16 |
| 1.80 | 1024 | 0.122895 | 0.122895 | 4.79e-08 | 7.08e-16 |
| 2.00 | 1024 | 0.103060 | 0.103060 | 4.16e-07 | 1.69e-15 |
<!-- END:ENERGY -->

截断研究（残差随级数项数下降）：

<!-- BEGIN:ENERGY_TRUNCATION -->
| Ka | n_terms | near/far residual |
|---|---|---|
| 0.5 | 32 | 3.78e-04 |
| 0.5 | 64 | 9.73e-05 |
| 0.5 | 128 | 2.47e-05 |
| 0.5 | 256 | 6.21e-06 |
| 0.5 | 512 | 1.56e-06 |
| 0.5 | 1024 | 3.90e-07 |
| 1.0 | 32 | 5.61e-04 |
| 1.0 | 64 | 1.44e-04 |
| 1.0 | 128 | 3.65e-05 |
| 1.0 | 256 | 9.20e-06 |
| 1.0 | 512 | 2.31e-06 |
| 1.0 | 1024 | 5.78e-07 |
| 2.0 | 32 | 4.09e-04 |
| 2.0 | 64 | 1.04e-04 |
| 2.0 | 128 | 2.63e-05 |
| 2.0 | 256 | 6.62e-06 |
| 2.0 | 512 | 1.66e-06 |
| 2.0 | 1024 | 4.16e-07 |
<!-- END:ENERGY_TRUNCATION -->

常数面元 BEM 解的自洽性（同一数值解的两个读数，`Ka = 1`）：

<!-- BEGIN:BEM_ENERGY -->
| panels | B33 near (N s/m) | B33 far (N s/m) | near/far residual | B33 error vs Hulme | cond |
|---|---|---|---|---|---|
| 64 | 1743.66 | 1739.24 | 2.53e-03 | +4.412% | 2.86 |
| 256 | 1689.78 | 1688.85 | 5.49e-04 | +1.186% | 2.73 |
| 1024 | 1674.94 | 1674.70 | 1.44e-04 | +0.297% | 2.68 |

Observed order of the residual in ``h``: 2.21, 1.93
<!-- END:BEM_ENERGY -->

图：`figures/energy_identity.png`。

---

## 4. 结果图

| 图 | 内容 |
|---|---|
| `figures/quadrature_error_vs_clearance.png` | 单层/双层核相对误差 vs `eps/h`（三条路线；负控必须在 `eps/h -> 0` 上抬升） |
| `figures/quadrature_convergence.png` | `eps/h = 1e-3` 时相对误差 vs 每面元核函数求值次数（`e(N)`） |
| `figures/a33_b33_vs_ka.png` | `A33/(rho R^3)`、`B33/(rho omega R^3)` vs `Ka`：Hulme 曲线 + 64/256/1024 面元 BEM 点 |
| `figures/energy_identity.png` | 近场/远场残差 vs 级数项数（解析场）与 vs 面元数（BEM），含 `1e-6` 闸门线 |

---

## 5. 注意点（踩过的坑与适用范围）

1. **极坐标路线第一版错了，是裁决者抓到的**。把 `rho drho/r` 误写成 `(rho drho)/(4 pi r)` 的
   等价形式时，"`rho drho = eps^2 sinh cosh dt` 再除以 `r`"与"直接用 `eps sinh t dt`"差了
   因子 `eps cosh t`，误差达 `10^0 ~ 10^2`；自适应路线与半解析裁决者仍然自洽（`1e-13`），
   所以**单靠自洽性检查发现不了这类错误**——必须有不共享代码路径的第三方裁决者。
2. **`numpy.polynomial.legendre.leggauss` 是 `O(n^3)`**（伴矩阵特征值），`n = 2560` 需 72 s，
   而本单的级数项数要上到 1024（高斯点 2048）。模块内改用 Newton 迭代的
   `gauss_legendre`（`O(n^2)`，`n = 2560` 约 0.33 s，与 `leggauss` 差 `1e-16`）。
3. **`eps -> 0` 的真正奇点是另一类问题**：本单只覆盖"目标在面板平面外"（`eps > 0`）；
   `eps = 0` 的自影响（双层积分的立体角）由现有 `solid_angle_triangle` 闭式处理，不在本单范围。
4. **"良分离"判据的代价是分辨率**：`split_ratio = 1` 时，`eps/h = 1` 整块面板直接被接受（无细分），
   精度由 8 阶 Duffy 自己决定（~1e-11）；`eps/h = 1e-3` 需 `max_level >= 12` 才到机器精度
   （`max_level = 8` 时双层仍有 `2e-6`）。文档里给的是 `max_level = 16` 的默认值。
5. **`1e-6` 的近场/远场残差是"解析场"的性质，不是常数面元的性质**。解析级数在 `n_terms = 1024`
   下体条件满足到 `~5e-7`，所以两个读数差 `<1e-6`；同一网格上的常数面元解只有 `O(h)` 精度
   （1024 面元时 `1.4e-4`）。两者都写进工件，不拿解析场的 `1e-6` 冒充求解器精度。
6. **本单不做低频/高频外推**：`Ka < 0.1` 时垂荡阻尼本身趋于零（`B33 ~ Ka^3` 量级），相对残差的
   分母会塌；`Ka > 2` 之后半球出现首个内部水线共振（`Ka ~ 2.58`），需 CHIEF/内部水面盖板，
   属另一条能力线。
7. **级数收敛慢是水线接触线的代数奇性**：`n_terms` 从 96 加到 384 只把残差从 `1e-5` 压到
   `1e-6`（约 `n^{-2}`），不是实现缺陷。

---

## 6. 误差判定（逐项 vs 基准，显式回答"是否 <3%"）

| 关键量 | 基准 | 本文取值 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 近场求积，单层核，`eps/h in [1e-3,1]` | 半解析闭式径向 | 见 §3.3 最坏值（自适应 `<1e-9`，极坐标 `<1e-6`） | `<1e-9` / `<1e-6` | 是（远优于） |
| 近场求积，双层核，同上 | 同上 | 同上 | `<1e-9` / `<1e-6` | 是（远优于） |
| 负控：默认 Duffy 规则，`eps/h = 1e-3` | 同上 | 见 §3.3 负控行（`~1e-1`，且不随阶数收敛） | `~10%` 且不收敛 | **否（这是设计意图：负控必须失败）** |
| 半球垂荡 `A33/(rho R^3)`，`Ka = 0.1 ... 2` | Hulme (1982) | 见 §3.4（认证网格） | 见 `max_abs_a33_rel_error` | 是 |
| 半球垂荡 `B33/(rho omega R^3)`，`Ka = 0.1 ... 2` | Hulme (1982) | 见 §3.4（认证网格） | 见 `max_abs_b33_rel_error` | 是 |
| 能量互证（解析场近场 vs 远场） | 能量守恒（同一场） | 见 §3.5 | `<= 1e-6` 闸门，实测最坏值见工件 | 是 |
| 能量互证（1024 面元常数面元解） | 同一数值解的两个读数 | 见 §3.5 BEM 表 | `~1e-4`，且随 `h` 收敛 | 是（但**不**声称 `1e-6`） |

**诚实声明**：三项作业里唯一"未达到 `<1e-6`"的是常数面元 BEM 的近场/远场互证（`1.4e-4` @1024 面元），
原因是离散势只把体条件满足到 `O(h)`；`<1e-6` 已在解析辐射场上达成并单独列闸（E1）。不把两者混为一谈。