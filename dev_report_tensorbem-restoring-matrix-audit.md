# TensorBEM 水动力恢复矩阵审计：对角阵缺陷、完整 6×6 推导、实现与验证

任务：m2424 ｜ 分支：`agent/m2424` ｜ 仓库：TensorBEM
交付物：`src/tensorbem/hydrostatic.py`（新增完整矩阵，旧接口不变）、
`src/tensorbem/benchmarks/hydrostatic_coupling.py`（可复现验证/影响评估工具）、
`tests/test_hydrostatic_restoring_coupling.py`（12 项测试）、
`docs/m2424/`（入库：研究 JSON + 5 图 + 程序化表格）、`artifacts/m2424/`（同内容，脚本生成目录，被 .gitignore 排除）。

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| 缺陷是否成立 | **成立**。`compute_hydrostatic_restoring_matrix()` 只填 C[3,3]、C[4,4]、C[5,5]（0 基序号 2/3/4），忽略 3 个耦合对（荡-横/荡-纵/横-纵）与横/纵恢复的水线平移轴修正。代码确实已算水线形心与惯性积（`waterplane_center`、`waterplane_moments[2]`），但两者**都没有进入矩阵**。 |
| 对角阵适用域 | **三重条件**：① 水线形心落在过机动原点的垂线上（x_c = y_c = 0，相对原点）；② 水线对过原点的轴无惯性积（I_xy,O = 0，任何一条过原点的镜像对称面即可）；③ 浮态平衡（m = ρ∇）。三条同时满足时耦合项**恰好为零**，且对角项与完整矩阵**逐位相同**（已用逐位相等断言证明）。 |
| 新增实现 | `compute_hydrostatic_restoring_matrix_full()` 返回完整 6×6 + 对角参考阵 + 耦合修正量 + 适用域诊断；旧函数**一行未改**其数值路径。 |
| 验证 | 非对称算例：与独立有限差分"预言机"（对湿表面多面体做 z≤0 精确裁剪后直接积分，不使用任何线性化公式）逐元吻合到 **1.4e-10 相对误差**；与闭式解析解吻合到 **3.1e-16**（机器精度）。对称算例：与旧版**逐位一致**（`np.array_equal` 断言）。 |
| RAO 链影响 | 非对称浮体（偏心驳船）里最大耦合项达对角刚度的 **88%**；RAO 链的横摇/纵摇幅值变化可达数十到上百个百分点（驳船横摇阻尼近零，共振区极敏感）。 |

误差判定（详见 §8）：与解析解/预言机的相对误差 **1e-16 / 1e-10，远小于 3%**；
旧对角阵在其适用域外与正确刚度的相对偏差最高 **94.6%**（偏心算例横摇刚度），**远超 15% 不可接受线**——这正是本次修复的动因。

---

## 1. 审计发现（现状复现）

`compute_hydrostatic_restoring_matrix()` 的现状逻辑：

```python
c[2, 2] = density * gravity * wp_area                     # 荡
c[3, 3] = density * gravity * volume * gm_t               # 横摇：GM_T = KB + I_T/∇ − KG
c[4, 4] = density * gravity * volume * gm_l               # 纵摇
```

三个已存在但未使用的量：

1. `waterplane_center`（水线形心 x_c, y_c）——只作为返回值输出，未进矩阵；
2. `waterplane_moments[2]` = I_xy（水线惯性积，绕**形心**）——只作为返回值输出；
3. `mass` 形参——**函数体内从未使用**：`gm_t = kb + I_c/∇ − kg` 隐含假设 `mass = ρ∇`（浮态平衡）。若调用方传入的 mass 与排水量不符，恢复力被系统性错误缩放。

此外，横/纵恢复用的是**绕形心轴**的水线惯性矩 `I_c`（`compute_waterplane_properties()` 已用平行轴定理把惯性矩移到了形心），而恢复矩阵的定义要求这些惯性矩**绕机动原点 O 的轴**（BEM/RAO 链里 6 自由度幅值都是以原点为参考点定义的）。两者相差平行轴项 `A·y_c²`、`A·x_c²`。

量化（偏心矩形驳船 L=4 m, B=2 m, T=1 m，水线形心 (3.0, 1.8) m，ρ=1025）：

| 量 | 旧对角阵 | 完整矩阵 | 相对偏差 |
|---|---|---|---|
| C[荡,荡] | 8.041e+04 | 8.041e+04 | 0 |
| C[荡,横] | 0 | 1.447e+05 | 100% |
| C[荡,纵] | 0 | −2.412e+05 | 100% |
| C[横,横] | 1.474e+04 | 2.753e+05 | **94.6%** |
| C[横,纵] | 0 | −4.342e+05 | 100% |
| C[纵,纵] | 9.516e+04 | 8.189e+05 | **88.4%** |

（数字见 `artifacts/m2424/tables.md`，程序化生成，未手抄。）

---

## 2. 完整 6×6 恢复矩阵的推导

### 2.1 坐标系、平衡姿态与线性化约定

* 机体坐标系 Oxyz 固连船体；**O 位于静水面上**（`z = 0`），z 轴向上；运动幅值 ξ 定义为 O 点的平移与绕 O 的转动，即 BEM/RAO 链使用的同一约定，同序 `(纵荡, 横荡, 垂荡, 横摇, 纵摇, 艏摇)`。
* 静平衡为线性化点：`mass = ρ∇`、重心与浮心在同一铅垂线上（xg→xb、yg→yb）。此时广义恢复力 `Q = −C ξ`，`C = ∂²V/∂ξ∂ξ`（V = 重量势能 + 被排开流体势能），因此 **C 必为对称阵**（Hessian）。
* 小位移刚性运动：`X(r) = R r + t`，`t = (ξ1, ξ2, ξ3)`，`R = Rz(ξ6)Ry(ξ5)Rx(ξ4)`。精确到二阶，机体点 r=(x,y,z) 的绝对垂向坐标为

```
Z(r) = ξ3 − ξ5·x + ξ4·y + z·(1 − (ξ4² + ξ5²)/2)                       (1)
```

一阶部分 `δ(r) = ξ3 + ξ4·y − ξ5·x` 决定水线处的"局部沉浮"，**二次项** `−z(ξ4²+ξ5²)/2` 才是 KB/KG 项的来源（这一步是常见推导错误的根源：只保留一阶项会漏掉 `∇z_B − mg z_G`）。

### 2.2 势能展开

重量势能：`V_w = m g Z(r_G)`，二阶展开

```
V_w = m g [ ξ3 + ξ4 y_G − ξ5 x_G − (ξ4² + ξ5²) z_G / 2 ] + const       (2)
```

浮力（被排开流体）势能：`V_b = −ρ g ∫_{Z≤0} Z dV`。以机体坐标对水线多边形 A 逐柱积分，精确到二阶得

```
V_b = ρ g [ (1/2) ∫_A δ² dA + ( (ξ4²+ξ5²)/2 )·∇ z_B ] + const          (3)
```

其中用到 `∫_A z_min² dA = −2∇z_B`（z_min 为船底垂向坐标），这一步把"底部形状"消掉，只留下浮心高度 z_B —— 与经典 `BM + KB − KG` 完全一致。

### 2.3 逐项结果与物理来源

由 `(1/2)∫_A δ² dA`（δ = ξ3 + ξ4 y − ξ5 x）的二次型系数直接读出：`A = ∫dA`、`A y_c = ∫y dA`、`A x_c = ∫x dA`、`I_xx,O = ∫y²dA`、`I_yy,O = ∫x²dA`、`I_xy,O = ∫xy dA`（**绕原点 O**）。汇总：

| 元素 | 表达式 | 物理来源 |
|---|---|---|
| C[2,2] | ρ g A | 垂荡改变排水体积：水线面积 × 下沉量 |
| C[2,3] = C[3,2] | ρ g A y_c | 垂荡增减的浮力作用在**偏心的**水线形心上，对横摇轴产生力矩（横倾-垂荡耦合） |
| C[2,4] = C[4,2] | −ρ g A x_c | 同上，对纵摇轴（纵倾-垂荡耦合），符号来自 δ 中 `−ξ5·x` |
| C[3,3] | ρ g (I_xx,O + ∇ z_B) − m g z_G | 三项：水线横向惯性矩（BM_T）、浮心处浮力向量转动（KB）、重心垂向偏移（KG） |
| C[3,4] = C[4,3] | −ρ g I_xy,O | 横摇与纵摇各自绕 x/y 轴改变水线力矩，混合二阶矩把它们线性耦合（水线惯性积非零时才存在） |
| C[4,4] | ρ g (I_yy,O + ∇ z_B) − m g z_G | 纵向对应项（BM_L + KB − KG） |
| 纵荡/横荡/艏摇各行与各列 | **0** | 势能只依赖垂向位置；平移与绕 z 轴转动不改变任何点的 z（精确到二阶），故无静水恢复。这也说明"艏摇-横摇耦合"在本线性化里不存在（文献里看到的 C[横,艏] 类项来自质量阵的 m·S 项，不是恢复项） |

于是

```
        ⎡ 0      0      0            0            0         0 ⎤
        ⎢ 0      0      0            0            0         0 ⎥
  C  =  ⎢ 0      0     ρgA        ρgA·y_c     −ρgA·x_c      0 ⎥
        ⎢ 0      0    ρgA·y_c      S_44        −ρg·I_xy,O   0 ⎥
        ⎢ 0      0   −ρgA·x_c    −ρg·I_xy,O     S_55       0 ⎥
        ⎣ 0      0      0            0            0         0 ⎦
```

`S_44 = ρg(I_xx,O + ∇ z_B) − m g z_G`，`S_55 = ρg(I_yy,O + ∇ z_B) − m g z_G`。

### 2.4 当前对角阵的适用域（本任务第 2 问）

耦合项为零 ⟺ **同时**满足：

1. `y_c = 0`（水线形心在横摇轴上）⇒ C[2,3] = 0 且 I_xx,O = I_xx,c，横摇对角项也自动正确；
2. `x_c = 0`（水线形心在纵摇轴上）⇒ C[2,4] = 0 且 I_yy,O = I_yy,c；
3. `I_xy,O = 0`（水线对过原点的某条轴镜像对称即足，通常 1 已蕴含）⇒ C[3,4] = 0；
4. `m = ρ∇`（浮态平衡）⇒ 对角项与重量项自洽。

满足时对角阵 = 完整阵**逐位相同**（不是"近似"）。反例：常规船舶水线关于中纵剖面对称（y_c = 0），但水线形心通常在原点（如舯剖面）**之前/之后** ⇒ `x_c ≠ 0` ⇒ **C[荡,纵] ≠ 0、C[纵,纵] 缺平行轴项**；即使母型对称，只要参考原点不在水线形心铅垂线上，旧对角阵就在适用范围之外。非对称水线（破舱、双体单侧、非对称附体、横倾姿态）则三条全破：本报告梯形/三角水线算例的横/纵刚度偏差分别为 2.8%/12.5% 与 68.0%/85.7%（水线形心离原点越远越糟）。

---

## 3. 实现

### 3.1 旧接口（不变）

`compute_hydrostatic_restoring_matrix(hull, mass, center_of_gravity, density, gravity)` 的签名、返回值与数值路径**完全保留**（仅模块 docstring 由"off-diagonal terms are zero"更正为"仅在 §2.4 的适用域内为零"，并列出完整形式）。

### 3.2 新增完整模式

```python
from tensorbem import compute_hydrostatic_restoring_matrix_full

props = compute_hydrostatic_restoring_matrix_full(hull, mass, cg, density=1025.0)
props.restoring_matrix            # 完整 6×6（F_rest = −C ξ）
props.diagonal_matrix             # 旧对角阵（逐位等于旧函数返回值）
props.coupling_terms              # 完整 − 对角（逐项可读的修正量）
props.waterplane_moments_about_origin   # [I_xx,O, I_yy,O, I_xy,O]
props.equilibrium_relative_mismatch     # |m − ρ∇| / ρ∇
props.diagonal_adequate / props.notes   # 适用域判定 + 逐条人话说明
props.as_hydrostatic_properties()       # 直接喂给 RAO 链
```

设计要点：

* **位一致性**：对角项按 `对角参考 + 修正量` 组装，修正量由 `ρg·A·y_c²`、`ρg·A·x_c²`、`(ρ∇−m) g z_G` 构成。在适用域内这些修正量**精确为 0.0**，所以完整阵与旧阵逐位相同（`np.array_equal` 断言）；域外则修正量本身即是"偏差有多大"的度量，便于审计。
* **不破坏调用方**：`FullRestoringProperties.as_hydrostatic_properties()` 返回原有 `HydrostaticProperties`，因此 RAO 链（`compute_rao_at_frequency(..., hydrostatic_props=...)`）无需改动即可用上完整阵。
* **物理来源自文档化**：`RESTORING_MATRIX_TERM_SOURCES` 把每个可能非零元素的公式与物理来源登记成表，测试断言"矩阵非零元素集合 ⊆ 登记表"且"登记表元素在非对称算例中确实非零"，避免公式与文档漂移。
* `waterplane_second_moments_about_origin()` 公开化平行轴移轴（形心 → 原点），供外部复用/核对。

---

## 4. 验证（三层，全部可复跑）

### 4.1 第一层：闭式解析解（非对称/偏心算例）

偏心矩形驳船（L=4, B=2, T=1；水线矩形左下角 (1.0, 0.8)）与三角形、梯形水线：把 `A, x_c, y_c, I_xx,c = LB³/12, I_yy,c = BL³/12, I_xy,c = 0, ∇ = LBT, z_B = −T/2` 直接代入 §2.3 的闭式，与实现输出比对。

### 4.2 第二层：独立有限差分"预言机"（不使用任何线性化公式）

`src/tensorbem/benchmarks/hydrostatic_coupling.py`：

1. 以水线多边形 + 吃水 + 干舷构造完整凸多面体（`prism_hull` 只给湿表面，BC 需要整船，否则零干舷驳船在垂荡时甲板入水会让刚度不连续——见 §7）；
2. 施加**精确**刚体运动 `X = R r + t`（不是小角度近似）；
3. 用半空间 `z ≤ 0` 对每条面做精确多边形裁剪（Sutherland–Hodgman + 水线封盖），得到湿多面体；
4. 散度定理积分出排水体积与浮心，得浮力/重力及其对原点的力矩；
5. 中心差分 `C[i,j] = −∂Q_i/∂ξ_j`（步长 2e-5）。

这条路径与推导完全解耦（不含 `1/2 ∫δ²dA`、不含 `GM`），是"实现是否真对"的判据。

### 4.3 第三层：负控（zero-coupling control）

对称驳船（原点即水线形心）：预言机必须给出 `C[*,0..1] = C[*,5] = 0`、`C[荡,横] = C[横,纵] = 0`，且与实现逐元吻合。若预言机在对称算例上给出非零耦合，说明预言机自身有偏（历史上就踩过：非平衡重心会让艏摇角带动常值力矩不平衡，见 §7）。

### 4.4 数值结果（`artifacts/m2424/tables.md`，程序化生成）

以下表格由 `python -m tensorbem.benchmarks.hydrostatic_coupling --output artifacts/m2424` 直接写出，未经手抄：

**表 1｜算例几何与静水力**

| case | A [m²] | ∇ [m³] | x_c [m] | y_c [m] | I_xx,O [m⁴] | I_xy,O [m⁴] | z_B [m] | z_G [m] | mass mismatch | diagonal adequate |
|---|---|---|---|---|---|---|---|---|---|---|
| `symmetric_barge` | 8.0000 | 8.0000 | +0.0000 | +0.0000 | 2.6667 | +0.0000 | -0.5000 | -0.3500 | 0.0e+00 | True |
| `eccentric_barge` | 8.0000 | 8.0000 | +3.0000 | +1.8000 | 28.5867 | +43.2000 | -0.5000 | -0.3500 | 0.0e+00 | False |
| `trapezoid_waterplane` | 3.5550 | 2.8440 | +0.2624 | +0.0840 | 0.7558 | +0.2097 | -0.4000 | -0.4500 | 1.6e-16 | False |
| `triangle_waterplane` | 1.9650 | 1.7685 | +1.3667 | +0.6333 | 1.0709 | +1.8029 | -0.4500 | -0.5000 | 0.0e+00 | False |

**表 2｜完整矩阵的三重验证**

| case | 最大相对误差 vs 闭式解 | 最大相对误差 vs 有限差分预言机 | max \|耦合\| / max \|对角\| |
|---|---|---|---|
| `symmetric_barge` | 3.70e-16 | 2.94e-10 | 0.000 |
| `eccentric_barge` | 2.84e-16 | 1.40e-10 | 0.884 |
| `trapezoid_waterplane` | 2.02e-16 | 1.37e-10 | 0.262 |
| `triangle_waterplane` | 3.12e-16 | 1.35e-10 | 0.857 |

**表 3｜偏心驳船逐元对比**（N/m, N/rad, N·m/m, N·m/rad）

| entry | complete | legacy (diagonal) | coupling | legacy rel. error |
|---|---|---|---|---|
| C[heave,heave] | 8.041453e+04 | 8.041453e+04 | +0.000000e+00 | 0.0000 |
| C[heave,roll] | 1.447462e+05 | 0.000000e+00 | +1.447462e+05 | 1.0000 |
| C[heave,pitch] | -2.412436e+05 | 0.000000e+00 | -2.412436e+05 | 1.0000 |
| C[roll,roll] | 2.752857e+05 | 1.474266e+04 | +2.605431e+05 | 0.9464 |
| C[roll,pitch] | -4.342385e+05 | 0.000000e+00 | -4.342385e+05 | 1.0000 |
| C[pitch,pitch] | 8.188880e+05 | 9.515719e+04 | +7.237308e+05 | 0.8838 |

（`C[roll,heave]`、`C[pitch,heave]`、`C[pitch,roll]` 与其转置逐位相等，此处省略；对称算例三个对角项与旧版完全相同，全部耦合项为 0。）

（`trapezoid_waterplane`、`triangle_waterplane` 的逐元表见 `artifacts/m2424/tables.md`。）

### 4.5 回归（判据"对称算例逐位一致"）

* `test_full_matrix_is_bit_identical_to_legacy_for_symmetric_hull`：`np.array_equal(full.restoring_matrix, legacy.restoring_matrix)` ✅
* `test_full_matrix_is_bit_identical_to_legacy_for_offset_along_x_only`：只偏 x 时 `C[横,横]` 仍与旧版**逐位相等**，`C[荡,横] = C[横,纵] = 0`，而 `C[荡,纵] ≠ 0` ✅
* `test_rao_chain_is_unchanged_for_a_hull_inside_the_diagonal_domain`：对称算例整条 RAO 复向量 `np.array_equal` ✅

### 4.6 全量验证记录（本地，`PYTHONPATH=src python -m pytest`）

在本分支上实际跑了以下模块（按模块分组、`-n` 并行；机器上与其它任务共享 CPU，慢模块单独标注）：

| 模块组 | 结果 |
|---|---|
| `test_hydrostatic_restoring_coupling.py`（本任务新增） | **12 passed** |
| `test_hydrostatic_rao.py` + `test_waterplane_multi_loop.py` + 新增文件 | **31 passed** |
| `test_hydrostatic_convergence.py` + `test_rao_verification.py` | **48 passed** |
| `test_finite_depth` / `test_helmholtz_maccamy_fuchs` / `test_free_surface_kernel` / `test_exterior_solver` / `test_solver` / `test_solver3d` / `test_kochin_validation` | 69 passed，1 failed（`test_kochin_validation::test_surge_damping_vs_hulme`，**pristine origin/main 上同样失败**，已核） |
| `test_ursell_benchmark` / `test_ursell_convergence` / `test_ursell_quadrature_convergence` / `test_ursell_reproducibility` / `test_ursell_doc_reproducibility` / `test_wigley_wave_resistance` | 59 passed，6 failed（全部为 ursell 文档/产物固定项，**pristine origin/main 上完全相同的 6 项失败**，已核） |
| `test_analytical_suite`(24) / `test_clearance`(3) / `test_conditioning`(10) / `test_cylinder_benchmark`(2) / `test_dean_diffraction`(6) / `test_declared_numbers`(12+1skip) / `test_diagnostics_report`(8) / `test_doc_numbers_maccamy_fuchs`(5) / `test_ellipse_benchmark`(7) / `test_geometry3d`(14) / `test_mesh_quality`(10) / `test_refinement_study`(1) | 全部 **passed** |
| `test_doc_claims.py` | 3 failed（缺 `artifacts/ursell_doc_claims.json`，**pristine origin/main 同**） |
| `test_documented_benchmarks.py` | 2 failed（**pristine origin/main 同**：同上表） |
| `test_hydrostatic_verification.py` | **收集失败**（`ImportError: cannot import name 'box_hull_mesh' from 'tensorbem.geometry3d'`），pristine origin/main 同 |

**未在本次时间预算内跑完**（机器被多任务共享，单模块 >100 s，且与本次改动无交集——改动是**纯新增**函数/模块 + 模块 docstring 更正）：`test_analytic_references`、`test_ellipsoid_benchmark`、`test_exterior_neumann_added_mass`、`test_sphere_velocity`、`test_sphere_flow_benchmark`、`test_sphere_benchmark`、`test_submerged_sphere_wave_3d`、`test_surface_velocity_recovery`、`test_surface_velocity_study`、`test_free_surface3d`、`test_floating_hemisphere_3d`、`test_kernels`、`test_kernels3d`、`test_kernels3d_gradients`、`test_pressure`、`test_quadrature_diagnostics(_2d)`、`test_interior_harmonics`、`test_published_reference_columns`、`test_unit_disk_benchmark`。这属于**未完成项**，如实记录，未以任何方式改判据凑绿。

基线红色结论（用 `git archive HEAD` 解出 pristine 树在 `/tmp` 独立复核）：`test_hydrostatic_verification.py`（收集期 ImportError）、`test_doc_claims.py`(3)、`test_documented_benchmarks.py`(2)、`test_ursell_reproducibility.py`+`test_ursell_doc_reproducibility.py`(6)、`test_kochin_validation.py`(1) 在**未打本次补丁的 origin/main 上即已失败**，与本任务无关。

---

## 5. 图

见 `docs/m2424/`（脚本生成；`artifacts/m2424/` 为同内容生成目录）：

1. `restoring_matrix_maps.png`：梯形水线算例的旧对角阵 / 完整阵 / 耦合修正量三张矩阵云图（归一化），直观看到耦合项的量级与对称性；
2. `restoring_matrix_oracle_parity.png`：四个水线形状、36 个元素，实现 vs 有限差分预言机的 parity 图（symlog）；
3. `legacy_diagonal_error.png`：逐元素"旧阵相对误差"柱状图（左：适用域内全 0；右：域外最高 ~100%）；
4. `rao_impact_coupling.png`：两个算例的垂荡/横摇/纵摇 RAO 曲线（旧阵 vs 完整阵）；
5. `waterplane_geometry.png`：四个水线平面 + 原点到形心的耦合力臂示意。

---

## 6. RAO 链影响评估

RAO 链求解 `[−ω²(M+A) + iωB + C] ξ = F_exc`，C 是唯一直接受本缺陷影响的项。用偏心驳船与梯形水线算例扫频（同一 BEM 网格、同一质量阵，只换 C）：

**表 4｜偏心驳船（浪向 90°），旧阵 → 完整阵**

| ω [rad/s] | 垂荡 legacy | 垂荡 complete | Δ | 横摇 legacy | 横摇 complete | Δ | 纵摇 legacy | 纵摇 complete | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 8.7919e-01 | 5.5665e-01 | 57.94% | 2.0180e+00 | 1.9351e-01 | 942.86% | 1.6709e-01 | 1.6771e-02 | 896.28% |
| 1.00 | 9.1051e-01 | 5.6428e-01 | 61.36% | 1.3473e+00 | 2.1967e-01 | 513.34% | 5.1501e-02 | 2.0799e-02 | 147.61% |
| 1.20 | 9.5671e-01 | 5.6976e-01 | 67.91% | 1.1486e+00 | 3.0366e-01 | 278.26% | 2.4018e-01 | 7.8995e-02 | 204.05% |
| 1.40 | 1.0404e+00 | 6.4439e-01 | 61.46% | 1.6949e+00 | 8.0467e-01 | 110.63% | 7.8818e-01 | 4.0275e-01 | 95.70% |
| 1.80 | 1.3657e+00 | 4.3543e-01 | 213.65% | 1.1939e-01 | 1.7694e-01 | 32.53% | 2.3537e-01 | 2.0861e-01 | 12.82% |
| 2.40 | 1.2076e+00 | 5.2564e-01 | 129.75% | 3.1064e-02 | 2.8790e-02 | 7.90% | 2.3385e-02 | 1.2467e-01 | 81.24% |

（完整 8 个频点见 `artifacts/m2424/tables.md`；梯形水线算例最大偏差 2556.60%，出现在 ω=0.8 的横摇。）

解读与限定：

* **量级来源是刚度本身，不是数值噪声**：偏心算例的横摇刚度被低估 94.6%（旧阵/完整阵 = 1.474e+04/2.753e+05 ≈ 0.054，相差 18.7 倍），横摇固有频率因此被低估 `sqrt(18.7) ≈ 4.3` 倍，共振位置整体移动，RAO 峰值自然大幅错位。驳船横摇阻尼近零，共振区相对偏差可达数百个百分点——这类算例是"最坏情形"，不代表所有船舶。
* **结论**：旧对角阵在适用域外会使 RAO 定量失真到不可用（>15% 线），且失真同时出现在与耦合相关的三个自由度（垂荡、横摇、纵摇）上；这也是"非对称水线/偏心参考点必须用完整阵"的直接证据。
* **RAO 链无需改代码**：`hydrostatic_props=props.as_hydrostatic_properties()` 即可切换；本任务未改 `rao.py`，避免影响既有基准测试的可比性。

---

## 7. 注意点 / 局限 / 适用范围

1. **零干舷驳船的垂荡刚度不连续**（推导期踩到）：湿表面网格只有 `z ≤ 0` 部分，若把"甲板 = 水线面"当作整船，船舶下沉一点点就整船没入水下，排水体积不再随垂荡变化，有限差分给出的是 0 与 ρgA 的对称平均。做预言机时必须给出**干舷**（本工具默认 0.5 m）。实际船舶有干舷，但**甲板入水/出水**时确实要分段处理，本实现未覆盖。
2. **恢复矩阵只在平衡点有意义**：若重心不在浮心铅垂线上，常值力矩不平衡会随艏摇角转动，使"矩阵"依赖于参数化约定（历史调试中预言机曾在非平衡重心下给出伪 C[横,艏]）。因此 `compute_hydrostatic_restoring_matrix_full()` 明确报告 `equilibrium_relative_mismatch` 并在域外给出 note；**不要**用非平衡输入做稳定性结论。
3. **`mass ≠ ρ∇`**：旧函数完全忽略 `mass`（隐含平衡假设）；完整函数把 `(ρ∇ − m) g z_G` 显式加进横/纵对角项并报警。是否要"允许不平衡输入"取决于上层业务；本任务选择**如实计入并报告**。
4. **未涵盖的物理**：非线性恢复（大角度稳性 / GZ 曲线）、自由液面修正（FSC）、横倾姿态下的水线变化、甲板浸没、锚链/系泊约束、多体（双体）间的水动力相互作用。双体水线多环路已被现有 `compute_waterplane_properties` 支持（形心/惯性矩按并集合并），完整矩阵沿用同一并集结果。
5. **艏摇/纵荡/横荡恢复恒为零**是线性静水力的严格结论（势能与这些自由度无关）；不要把质量阵里的 CG 偏心项误记为恢复项。
6. **本仓库基线（origin/main）本来就是红的**：`tests/test_hydrostatic_verification.py` 在 pristine HEAD 上即因 `ImportError: cannot import name 'box_hull_mesh' from 'tensorbem.geometry3d'` 收集失败（`benchmarks/hydrostatic_benchmarks.py`、`hydrostatic_verification.py` 引用了从未进入本仓的几何构造器，疑为历史门禁过滤遗留）；`test_doc_claims.py` 亦因缺失 `artifacts/ursell_doc_claims.json` 在上游即 3 项失败。本任务未触碰这些文件，也未"凑绿"：审计报告如实记录，作为独立欠账。

---

## 8. 误差判定（关键量 vs 基准，逐项回答"是否 <3%"）

| 关键量 | 基准 | 相对误差 | <3% ? |
|---|---|---|---|
| 完整矩阵 C[2,2], C[2,3], C[2,4], C[3,3], C[3,4], C[4,4]（4 个水线算例） | 独立有限差分预言机（精确裁剪 + 散度定理，无线性化） | ≤ 1.40e-10 | ✅ 是 |
| 同上 | 闭式解析解（矩形/梯形/三角形水线） | ≤ 3.12e-16 | ✅ 是 |
| 对称算例完整阵 vs 旧对角阵 | 旧版实现（回归基准） | 0（逐位相等） | ✅ 是 |
| 横/纵恢复的平行轴修正 `ρg A y_c²`、`ρg A x_c²` | 闭式（与实现独立推导） | ≤ 1e-12（pytest 断言） | ✅ 是 |
| **旧对角阵在域外** C[3,3]、C[4,4] | 完整矩阵 | 94.6% / 88.4%（偏心驳船）；三角水线 68.0% / 85.7%；梯形水线 2.8% / 12.5% | ❌ **否（除梯形横摇外远超 15% 线）** |
| RAO 幅值（域外算例，旧 vs 完整） | 完整矩阵解 | 1.8% ~ 2556.6% | ❌ 否（最坏情形数百 %） |

结论：**新增实现本身误差在机器精度量级（≪3%）**；被审计的旧实现只在适用域内合格，域外误差远超验收线——修复必要。

---

## 9. 复现

```bash
cd TensorBEM            # worktree, PYTHONPATH=src
export PYTHONPATH=$PWD/src

# 1) 单元测试（12 项：位一致回归 / 闭式解 / 预言机 / 负控 / 诊断 / RAO 链）
python -m pytest -q tests/test_hydrostatic_restoring_coupling.py

# 2) 全量研究 + 图 + 程序化表格
python -m tensorbem.benchmarks.hydrostatic_coupling --output artifacts/m2424
```

---

## 10. 变更清单

| 文件 | 变更 |
|---|---|
| `src/tensorbem/hydrostatic.py` | 模块 docstring 更正（对角阵仅适用域内成立）；新增 `compute_hydrostatic_restoring_matrix_full()`、`FullRestoringProperties`、`waterplane_second_moments_about_origin()`、`RESTORING_MATRIX_TERM_SOURCES`；旧函数数值路径未改 |
| `src/tensorbem/__init__.py` | 导出上述新公开 API |
| `src/tensorbem/benchmarks/hydrostatic_coupling.py` | 新增：水线算例构造、独立有限差分预言机、耦合/RAO 影响研究、图与 Markdown 表格生成 |
| `tests/test_hydrostatic_restoring_coupling.py` | 新增 12 项测试 |
| `docs/m2424/` | 研究 JSON、5 张图、`tables.md`（报告表格来源，程序化生成）；`artifacts/m2424/` 为脚本默认输出目录（.gitignore 排除） |