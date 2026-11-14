# CASE-BEM-RIGID-BODY-MASS — 装配方程的谱绑定（自然频率路线）

本文件由 `src/tensorbem/benchmarks/case_rigid_body_mass_spectrum_binding.py` 从 `spectrum_binding.json` 程序注入，无手抄数字。

## ① 为什么补这一层

锚点的既有证据全部是**矩阵级**的（闭式块形式 R / 四面体积分 M / capytaine X / 动量定义 P）加上**调用点回放**：它们把出厂矩阵与同一约定的另一次转写相比，或证明求解器确实吃进了注入的矩阵。
没有任何一条把 `M_RB` 绑到装配方程的**物理可观测量**上。本单用自然频率补这一层：频率不需要约定即可推出。

## ② 参数

| 参数 | 数值 |
|---|---|
| L x B x T | 20.0 x 8.0 x 3.0 m |
| rho / g | 1025.0 kg/m3 / 9.81 m/s2 |
| 排水量 / 质量 | 4.920000e+05 kg |
| KG / KM / GM | 2.6 / 3.2778 / +0.6778 m |
| I_xx / I_yy / I_zz (关于质心) | 2.9930e+06 / 1.6769e+07 / 1.9024e+07 kg m2 |
| 水线面面积 A_wp | 160.0 m2 |

## ③ 结果

**(a) 解耦垂荡 vs 闭式**：`omega^2 = rho g A_wp / m`，判据 < 3%。

实测 omega_heave = **1.808314 rad/s**，闭式 1.808314 rad/s，相对误差 **1.2279e-16**。

**(b) 参考点平移下谱不变**（物理不变量；判据 1e-09）

| 平移 (m) | 谱 (rad/s) | 标签 | 最大相对变化 |
|---|---|---|---|
| [0.0, 0.0, 0.0] | 1.808314, 1.874740, 2.154231 | heave, pitch, roll | 0.0000e+00 |
| [5.0, -3.0, 0.0] | 1.808314, 1.874740, 2.154231 | heave, pitch, roll | 6.1844e-16 |
| [-7.2, 4.4, 0.0] | 1.808314, 1.874740, 2.154231 | heave, pitch, roll | 1.2369e-15 |

**(c) 交叉表述**：把出厂矩阵经 twist 映射压到质心系，得到 `M_cg`；它与教材 `diag(m I3, I_g)` 的相对差 **0.0000e+00**，其谱与原点系谱的相对差 **0.0000e+00**（判据 1e-09）。

**(d) 对照（证明判据有牙齿）**

| 对照 | 结果 | 结论 |
|---|---|---|
| 历史 `inertia=None` 回退 | 最小谱变化 4.3855e-01 | 拒绝 |
| `-m S(r_g)` 耦合块符号取反（平移后） | 谱变化 1.7892e+00 | 拒绝 |
| 同上，垂荡频率 | 相对误差 1.2279e-16 | 看不见（这正是矩阵级判据的盲区） |

## ④ 判据汇总

| 判据 | 通过 |
|---|---|
| `heave_matches_closed_form` | ✅ |
| `spectrum_invariant_under_reference_shift` | ✅ |
| `cg_form_reproduces_shipped_spectrum` | ✅ |
| `cg_form_is_textbook_diagonal_inertia` | ✅ |
| `legacy_inertia_fallback_rejected` | ✅ |
| `coupling_sign_flip_rejected` | ✅ |
| `coupling_sign_flip_invisible_to_heave` | ✅ |

## ⑤ 注意点 / 明确不声称

* **不含水动力附加质量**：参照是真空谱。加进 `A` 会把 BEM 自身的离散误差带进参照，从而掩盖约定错误；`A_ij` 的验证在别处（Hulme/椭球/协变恒等式）。
* **不钉经典横摇周期** `2 pi r / sqrt(g GM)`：出厂的 `C[3,3]` 用 Journee & Massie 的水线原点形式 `rho g [I_xx + nabla (z_B - z_G)]`，与 `rho g nabla GM` 只在无耦合极限一致；两者之差需要二阶静水复原力矩独立仲裁，已登记为开放问题（`not_pinned`），本单不作断言。
* 判据容差：闭式比较用锚点 3%；协变/交叉表述用圆整极限 1e-09；对照必须 > 10% 才算被拒绝。
* 环境：Python 3.11.6 · NumPy 2.4.6 · SciPy 1.16.0。
