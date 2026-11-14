# CASE-BEM-LAPLACE-CYLINDER 第 3 轮 — 给定环量（旋转圆柱 / Magnus）路线与 Kutta–Joukowski 升力验证

锚点 `CASE-BEM-LAPLACE-CYLINDER`（池码 `BEM-LAPLACE-CYLINDER`），域
ship-total-performance/resistance-potential-flow，容差 `< 3%`。
本文件的每个数字都由 `docs/build_m20511_report.py` 从
`benchmarks/verified/case_bem_laplace_cylinder/results.json` 程序化注入（`hand_typed_numbers = 0`）。

## 1. 案例介绍

半径 $a=1$ 的圆柱在无界、无粘、不可压流中以 $U=1$ 沿 $+x$ 运动，并绕自身轴以环量
$\Gamma$ 旋转（Magnus 效应）。复势

$$W(z)=U\left(z+\frac{a^2}{z}\right)-\frac{i\Gamma}{2\pi}\ln\frac{z}{a},\qquad
\frac{dW}{dz}=u-iv=U\left(1-\frac{a^2}{z^2}\right)-\frac{i\Gamma}{2\pi z}$$

物面切向速度 $v_\theta=\Gamma/(2\pi a)-2U\sin\theta$、$C_p=1-(v_\theta/U)^2$、
驻点 $\sin\theta_s=\Gamma/(4\pi Ua)$、升力 $L'=-\rho U\Gamma$（Kutta–Joukowski）、阻力 0（d'Alembert）。
参考出处：Lamb《Hydrodynamics》6th ed. 1932、Milne-Thomson《Theoretical Hydrodynamics》5th ed. 1968
（`docs/classical_benchmarks.md` ref.[1]/[3]）；**页码 WITHHELD**（容器内不可达原文），
改以闭式自洽性反查（调和性 / 无旋性）与逐点闭式对照替代。

| 参数 | 值 |
| --- | --- |
| 半径 $a$ / 来流 $U$ | 1 / 1（全部量按 $a$、$U$ 无量纲）|
| 环量 $\Gamma/(Ua)$ | 3.0（两个物面驻点的亚临界区 $|\Gamma|<4\pi Ua\approx12.57$）|
| 环量扫描 | [1.0, 2.0, 3.0] |
| 面元阶梯 $N$ | [32, 64, 128]（预算 $N=128$）|
| 外部测点 | 64 点，环 $r/a=[1.25, 1.5, 2.0, 3.0]$ |
| 容差 | 3% |
| 环境 | Python 3.11.6 / NumPy 2.4.6 / loongarch64 |

## 2. 怎么做的

新增模块 `src/tensorbem/lifting_cylinder.py`（本单实现，非占位）：

1. **闭式参考**：`exact_velocity`、`exact_surface_speed`、`exact_pressure_coefficient`、
   `stagnation_angles`、`joukowski_lift`，全部在同一点现算；
2. **BEM 求解** `solve_rotating_cylinder`：把总扰动拆成单值偶极子
   $\phi_d=Ua^2\cos\theta/r$（Neumann 数据 $q_e=\nabla\phi_d\cdot n_e$）与单位点涡响应
   （数据 $q_e=v_{unit}\cdot n_{body}$，消去点涡在壁面的法向速度），各解一次
   `solver.solve_laplace_exterior_neumann`（多值部分 $\Gamma\theta/2\pi$ 以解析速度回加，
   因为外部 Green 表示只对单值迹成立）；总速度
   $U\hat x+\nabla\phi_d+\Gamma(v_{unit}+\nabla\phi_c)$，法向分量取所给数据、切向分量取
   解的迹的弧长中心差分；
3. **后处理**：$C_p=1-|v|^2/U^2$，压力积分 $F'=-\oint p\,n_{body}\,ds$ 给升力与阻力；
   物面切向速度的零点是驻点（相邻面板中点线性插值，二阶）；
4. **算例集成**：`src/tensorbem/benchmarks/case_cylinder_stream.py` 新增路线 $\Gamma$
   （`route_circulation` / `circulation_sweep`），并入 `run_case` 工件、29 项预置判据、
   `results.json` / `report.md` / `case.json` / `figures/circulation.png`；
   平台算例池 `case_pool_registry` 的 `CASE-BEM-LAPLACE-CYLINDER` 条目即调度这条链。

**诚实的边界（实测发现）**：内接正多面体的面板中点法向恰为径向、点涡在该点纯切向，
故点涡响应数据恒为零（$\sim10^{-16}$，`tests/test_lifting_cylinder.py` 已钉住该零，
并在椭圆网格上验证其非零 1e-2 量级）。所以圆柱阶梯上**环量是解析给定的**，
BEM 数值部分=折面体上的偶极子 Neumann 问题；本路线验证的是"带环量的外部 Neumann 问题 +
物面速度 + 压力积分 + 驻点定位"整条链，不验证 Kutta 条件（后者是
`docs/lifting_kutta_closure.md` 的 lifting 家族）。

工具链：Python/NumPy + 常值面元配点（Hess & Smith 1962；Katz & Plotkin 闭式影响系数）。

## 3. 计算结果

路线 $\Gamma$（$\Gamma=3.0\,Ua$，预算 $N=128$）：

| N | 环量（实测/给定） | 物面速度 L2 | $Cp$ L2 | 外部速度 L2 | 升力 $L'$ | $L'_{ref}$ | 升力相对误差 | 阻力系数 | 驻点误差(deg) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 3.009676 / 3.0 | 3.421e-03 | 8.971e-03 | 2.479e-03 | -3.019334 | -3.000000 | 6.445e-03 | 1.9e-16 | 0.070 |
| 64 | 3.002412 / 3.0 | 8.523e-04 | 2.235e-03 | 5.926e-04 | -3.004819 | -3.000000 | 1.606e-03 | -3.6e-16 | 0.008 |
| 128 | 3.000603 / 3.0 | 2.131e-04 | 5.587e-04 | 1.481e-04 | -3.001204 | -3.000000 | 4.014e-04 | 8.0e-17 | 0.006 |

环量线性（$N=128$，比值应恒为 1）：

| $\Gamma/(Ua)$ | $L'$ | $-\rho U\Gamma$ | 比值 | 相对误差 | $Cp_{min}$ | $Cp_{min,ref}$ | 驻点误差(deg) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | -1.000401 | -1.000000 | 1.000401 | 4.014e-04 | -3.66129 | -3.65935 | 0.001 |
| 2.0 | -2.000803 | -2.000000 | 1.000401 | 4.014e-04 | -4.37407 | -4.37177 | 0.003 |
| 3.0 | -3.001204 | -3.000000 | 1.000401 | 4.014e-04 | -5.13755 | -5.13485 | 0.006 |

收敛（相对 $L_2$，$h_{mean}$ 为尺度）：

| N | $h_{mean}/a$ | 物面速度 | $Cp$ | 外部速度 | 升力相对误差 |
| --- | --- | --- | --- | --- | --- |
| 32 | 1.960e-01 | 3.421e-03 | 8.971e-03 | 2.479e-03 | 6.445e-03 |
| 64 | 9.814e-02 | 8.523e-04 | 2.235e-03 | 5.926e-04 | 1.606e-03 |
| 128 | 4.908e-02 | 2.131e-04 | 5.587e-04 | 1.481e-04 | 4.014e-04 |

实测收敛阶：物面速度 [2.0084326626949522, 2.0009051001603915]、升力误差 [2.007921397294378, 2.0013703647305294]（面元 $h$ 二阶）。

预算面板（$N=128$）关键量：

| 量 | got | ref | 相对误差 |
| --- | --- | --- | --- |
| 物面速度 $\|v\|$ | BEM | $v_\theta=\Gamma/(2\pi a)-2U\sin\theta$ | 2.131e-04 |
| $Cp$ | BEM | $1-(v_\theta/U)^2$ | 5.587e-04 |
| 外部速度场 | BEM | $dW/dz$ | 1.481e-04 |
| 升力 $L'$ | -3.001204 | -3.000000 | 4.014e-04 |
| 阻力系数 | — | 0 | 8.0e-17（绝对值）|
| 驻点角 | [13.817, 166.183] deg | [13.812, 166.188] deg | 0.006 deg |
| 环量 | 3.000603 | 3.0 | 2.008e-04 |

几何/参考自洽门（同一工件）：顶点半径偏差
1.11e-16、面积相对偏差
2.83e-16、参考 harmonicity
7.77e-08、irrotationality
2.57e-08；路线 D/N 的既有结论不变
（D 外部速度 1.366e-07、N 物面速度
1.008e-04）。

近壁适用窗（第 2 轮成果，本轮随同一工件重跑；$\delta$ 以 $h_{max}$ 为单位）：

| N | $h_{max}/a$ | $\delta=2h$ | $1h$ | $0.5h$ | $0.25h$ | $0.1h$ |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.960e-01 | 6.609e-03 | 1.433e-02 | 3.689e-02 | 9.086e-02 | 1.795e-01 |
| 64 | 9.814e-02 | 3.600e-03 | 7.851e-03 | 2.696e-02 | 8.830e-02 | 1.772e-01 |
| 128 | 4.908e-02 | 1.888e-03 | 4.387e-03 | 2.212e-02 | 8.732e-02 | 1.757e-01 |

## 4. 结果图

![circulation](benchmarks/verified/case_bem_laplace_cylinder/figures/circulation.png)

`figures/circulation.png`：左=$Cp(\theta)$（BEM 面板 vs 闭式，红虚线为闭式驻点）；
右=升力随 $\Gamma$ 的线性（数值 vs $-\rho U\Gamma$）与相对误差（对数右轴）。

![velocity field](benchmarks/verified/case_bem_laplace_cylinder/figures/field.png)

`figures/field.png`：无环量基准场的解析 $|v|/U$ 云图 + BEM 箭头（$N=64$ 演示网格）与逐点误差。

![convergence](benchmarks/verified/case_bem_laplace_cylinder/figures/convergence.png)

`figures/convergence.png`：三条路线的误差随 $h_{mean}$ 的二阶收敛与 $h^2$/$h^4$ 参考斜率。

## 5. 注意点

- **环量是给定的**（旋转圆柱），不是解出来的；点涡响应数据在内接多面体阶梯上实测恒为零，
  故不要把"升力吻合"读成"BEM 解出了环量"。Kutta 条件另属 lifting 家族；
- 面元密度分片常数，近壁（$\delta\lesssim0.5h_{max}$）速度场误差可达 2%–18% 且加密不改善
  （见上表），故测点从 $r/a=1.25$ 起；
- 驻点定位依赖相邻面板中点的线性插值，二阶；$\Gamma\to4\pi Ua$ 时两驻点合并于顶点，
  亚临界前提必须显式检查（$|\Gamma|\ge4\pi Ua$ 直接拒绝）；
- 势流极限：无粘、无分离，真实旋转圆柱的 Magnus 升力在高 $Re$ 下会被分离显著削弱，
  本算例是算子/实现验证，不是工程升力预报；
- 复势页码 WITHHELD（原文不可达），参考以闭式现算 + 自洽性反查替代。

## 6. 误差判定

预置判据 29 项，失败 0 项，判定 **PASS**：

| 判据 | 实测 | 阈值 | 判定 |
| --- | --- | --- | --- |
| geometry_vertices_on_circle | 1.110e-16 | <= 1e-12 a | PASS |
| geometry_area_matches_inscribed_closed_form | 2.828e-16 | <= 1e-12 | PASS |
| geometry_perimeter_matches_inscribed_closed_form | 0.000e+00 | <= 1e-12 | PASS |
| reference_is_harmonic | 7.772e-08 | <= 1e-06 U/a | PASS |
| reference_is_irrotational | 2.571e-08 | <= 1e-06 U/a | PASS |
| field_gradient_matches_finite_difference | 1.465e-10 | <= 1e-05 | PASS |
| dirichlet_velocity_relative_l2 | 1.366e-07 | < 0.03 | PASS |
| dirichlet_potential_relative_l2 | 3.578e-07 | < 0.03 | PASS |
| neumann_surface_velocity_relative_l2 | 1.008e-04 | < 0.03 | PASS |
| neumann_velocity_relative_l2 | 1.532e-04 | < 0.03 | PASS |
| neumann_normal_velocity_residual | 6.025e-04 | <= 0.01 U | PASS |
| neumann_cp_relative_l2 | 2.849e-04 | < 0.03 | PASS |
| neumann_surface_velocity_observed_order | [2.0165907313184706, 2.0065676588343466] | >= 1.5 | PASS |
| near_wall_window_error | 6.609e-03 | <= 0.01 | PASS |
| near_wall_window_error_scales_with_h | {'observed_order_in_h': 0.9046978473113044, 'monotone_decreasing': True} | decreasing with order >= 0.5 | PASS |
| near_wall_exclusion_zone_is_real | 1.757e-01 | >= 0.03 | PASS |
| near_wall_error_is_panel_relative | 1.022e+00 | in [0.5, 2.0] | PASS |
| near_wall_guard_rejects_inside_window | True | True on every level | PASS |
| circulation_surface_velocity_relative_l2 | 2.131e-04 | < 0.03 | PASS |
| circulation_cp_relative_l2 | 5.587e-04 | < 0.03 | PASS |
| circulation_velocity_relative_l2 | 1.481e-04 | < 0.03 | PASS |
| circulation_measured_matches_prescribed | 2.008e-04 | <= 0.01 | PASS |
| circulation_lift_matches_joukowski | 4.014e-04 | < 0.03 | PASS |
| circulation_drag_vanishes | 7.980e-17 | <= 0.01 | PASS |
| circulation_stagnation_angle_matches_closed_form | 5.582e-03 | <= 1 deg | PASS |
| circulation_observed_order | [2.0084326626949522, 2.0009051001603915] | >= 1.5 | PASS |
| circulation_sweep_lift_is_linear | 4.014e-04 | < 0.03 | PASS |
| wrong_reference_is_rejected | 3.952e+06 | >= 10 | PASS |
| far_field_matches_closed_form | 5.591e-09 | <= 0.01 U | PASS |

**是否 <3%**：是 —— 预算面板下
物面速度 2.131e-04、$Cp$ 5.587e-04、外部速度 1.481e-04、
升力 4.014e-04、环量 2.008e-04，全部 `3%` 以内，且误差随加密按二阶下降
（物面速度阶 [2.0084326626949522, 2.0009051001603915]），不是靠放宽阈值通过的。近壁适用窗外（$\delta=2h_{max}$）
最大相对误差 1.888e-03 也已在 1% 以内。
