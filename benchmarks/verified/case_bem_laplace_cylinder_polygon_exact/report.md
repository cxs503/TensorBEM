# 圆柱/正多边形绕流：把算例误差拆成「面板离散」与「几何多面体」两部分

- 研究编号：`BEM-LAPLACE-CYLINDER-POLYGON-EXACT`，锚点算例 `CASE-BEM-LAPLACE-CYLINDER`
- 参数：$a=1.0$，$U=1.0$，面板梯 $[8, 16, 32, 64, 128]$，预算 $N=128$，容差 0.03

## 1. 方法
对半径 $a$ 的正 $n$ 边形，单位圆外域 Schwarz-Christoffel 映射 $z(w)=c\int^w(1-t^{-n})^{2/n}dt$，$c=a\sin(\pi/n)/(\alpha 2^{\alpha}I(\alpha))$，$\alpha=2/n$，复势 $W=Uc(w+1/w)$，复速度 $dW/dz=U(1-w^{-2})/(1-w^{-n})^{2/n}$；边中点处 $w^{-n}=-1$，故壁面速度有闭式 $|v|=2U|\sin\theta_w|/2^{2/n}$，无需数值积分。

对边界积分方程沿直面元求切向导数，联立求解各面元中点的壁面势斜率。法向速度使用原 Neumann 数据。外场扰动势对照解析总势减去均匀来流势 Ux。

## 2. 结果
| $n$ | $h_{mean}/a$ | 面板误差(壁面速度) | 面板误差(Cp) | 面板误差(外场速度) | 锚点指标(BEM vs 圆) | 多面体几何误差 | 几何误差/锚点指标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 7.6537e-01 | 2.348e-02 | 7.425e-02 | 3.495e-02 | 2.859e-02 | 2.382e-01 | 8.3 |
| 16 | 3.9018e-01 | 4.898e-03 | 1.468e-02 | 5.704e-03 | 6.686e-03 | 1.026e-01 | 15.3 |
| 32 | 1.9603e-01 | 1.030e-03 | 2.999e-03 | 1.260e-03 | 1.633e-03 | 4.727e-02 | 28.9 |
| 64 | 9.8135e-02 | 2.293e-04 | 6.582e-04 | 2.986e-04 | 4.046e-04 | 2.264e-02 | 56.0 |
| 128 | 4.9082e-02 | 5.360e-05 | 1.527e-04 | 7.563e-05 | 1.008e-04 | 1.107e-02 | 109.9 |

## 3. 判据
| 判据 | 值 | 阈值 | 结果 |
| --- | --- | --- | --- |
| geometry_gate_all_levels | 2.6479e-07 | <= 1e-6 | PASS |
| exact_solution_no_flux | 3.7192e-15 | <= 1e-12 U | PASS |
| exact_solution_mirror_symmetry | 0.0000e+00 | <= 1e-12 U | PASS |
| exact_solution_far_field | 1.2778e-06 | <= 1e-5 U | PASS |
| exact_neumann_data_matches_closed_form_velocity | 1.8735e-15 | <= 1e-5 U | PASS |
| neumann_surface_velocity_relative_l2_budget | 5.3599e-05 | < 0.03 | PASS |
| neumann_cp_relative_l2_budget | 1.5270e-04 | < 0.03 | PASS |
| neumann_velocity_relative_l2_budget | 7.5631e-05 | < 0.03 | PASS |
| neumann_potential_relative_l2_up_to_constant_budget | 1.9823e-04 | < 0.03 | PASS |
| dirichlet_velocity_relative_l2_budget | 1.3258e-07 | < 0.03 | PASS |
| dirichlet_potential_relative_l2_budget | 3.4750e-07 | < 0.03 | PASS |
| neumann_no_flux_at_budget | 1.8319e-15 | <= 0.01 U | PASS |
| neumann_surface_velocity_observed_order | [2.3261778465247613, 2.2652924260441596, 2.170801392787399, 2.098122769243469] | >= 1.5 | PASS |
| dirichlet_velocity_observed_order | [2.4024836759104247, 4.956725679186057, 7.444173161915501, 3.127871618900014] | >= 1.5 | PASS |
| faceting_dominates_anchor_surface_metric | 8.3326e+00 | > 1 on every ladder level | PASS |
| triangle_inequality_consistent | 1.9520e-08 | >= 0 | PASS |
| wrong_reference_control | 3.4706e-02 | >= 0.03 | PASS |

总判定：**PASS**
