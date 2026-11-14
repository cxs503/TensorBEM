# 圆柱/正多边形绕流：把算例误差拆成「面板离散」与「几何多面体」两部分

- 研究编号：`BEM-LAPLACE-CYLINDER-POLYGON-EXACT`，锚点算例 `CASE-BEM-LAPLACE-CYLINDER`
- 参数：$a=1.0$，$U=1.0$，面板梯 $[8, 16, 32, 64]$，预算 $N=128$，容差 0.03

## 1. 方法
对半径 $a$ 的正 $n$ 边形，单位圆外域 Schwarz-Christoffel 映射 $z(w)=c\int^w(1-t^{-n})^{2/n}dt$，$c=a\sin(\pi/n)/(\alpha 2^{\alpha}I(\alpha))$，$\alpha=2/n$，复势 $W=Uc(w+1/w)$，复速度 $dW/dz=U(1-w^{-2})/(1-w^{-n})^{2/n}$；边中点处 $w^{-n}=-1$，故壁面速度有闭式 $|v|=2U|\sin\theta_w|/2^{2/n}$，无需数值积分。

## 2. 结果
| $n$ | $h_{mean}/a$ | 面板误差(壁面速度) | 面板误差(Cp) | 面板误差(外场速度) | 锚点指标(BEM vs 圆) | 多面体几何误差 | 多面体/面板 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 7.6537e-01 | 1.576e-01 | 5.441e-01 | 3.495e-02 | 2.859e-02 | 2.382e-01 | 8.3 |
| 16 | 3.9018e-01 | 8.335e-02 | 2.609e-01 | 5.704e-03 | 6.686e-03 | 1.026e-01 | 15.3 |
| 32 | 1.9603e-01 | 4.258e-02 | 1.267e-01 | 1.260e-03 | 1.633e-03 | 4.727e-02 | 28.9 |
| 64 | 9.8135e-02 | 2.148e-02 | 6.233e-02 | 2.986e-04 | 4.046e-04 | 2.264e-02 | 56.0 |

## 3. 判据
| 判据 | 值 | 阈值 | 结果 |
| --- | --- | --- | --- |
| geometry_gate_all_levels | 2.6479e-07 | <= 1e-6 | PASS |
| exact_solution_no_flux | 1.7208e-15 | <= 1e-12 U | PASS |
| exact_solution_mirror_symmetry | 0.0000e+00 | <= 1e-12 U | PASS |
| exact_solution_far_field | 1.2763e-06 | <= 1e-5 U | PASS |
| exact_neumann_data_matches_closed_form_velocity | 8.6042e-16 | <= 1e-5 U | PASS |
| neumann_surface_velocity_relative_l2_budget | 2.1484e-02 | < 0.03 | PASS |
| neumann_cp_relative_l2_budget | 6.2325e-02 | < 0.03 | FAIL |
| neumann_velocity_relative_l2_budget | 2.9859e-04 | < 0.03 | PASS |
| neumann_potential_relative_l2_up_to_constant_budget | 8.0029e-01 | < 0.03 | FAIL |
| dirichlet_velocity_relative_l2_budget | 1.1579e-06 | < 0.03 | PASS |
| dirichlet_potential_relative_l2_budget | 8.0044e-01 | < 0.03 | FAIL |
| neumann_no_flux_at_budget | 8.3267e-16 | <= 0.01 U | PASS |
| neumann_surface_velocity_observed_order | [0.9453886323676782, 0.9759942451184551, 0.9884990685923164] | >= 1.5 | FAIL |
| dirichlet_velocity_observed_order | [2.402483675910428, 4.956725679186195, 7.444173161893415] | >= 1.5 | PASS |
| faceting_dominates_anchor_surface_metric | 8.3326e+00 | > 1 on every ladder level | PASS |
| triangle_inequality_consistent | 3.6724e-01 | >= 0 | PASS |
| wrong_reference_control | 3.6523e-02 | >= 0.03 | PASS |

总判定：**FAIL**
