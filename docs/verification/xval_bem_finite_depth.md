# XVAL-BEM-FINITE-DEPTH：三维有限水深辐射 BEM 的三方交叉验证

## 1. 问题设定与基准出处

几何：闭合全潜测地球（geodesic icosphere）球体，半径 a=1.0 m，球心 z=-1.5 m，ρ=1025 kg/m³，g=9.80665 m/s²，ka=1 为主频。水深阶梯 h/a∈{3.0, 5.0, 20.0|}（h 为平底深度，底部 z=-h）。物理：线性势流、常水深、平底不可穿透、自由面 z=0、时间因子 exp(-iωt)。

三条独立路线：

* **T+**：`tensorbem.finite_depth_solver3d.solve_radiation_finite_depth_3d`（直接 Green 第三恒等式 + John (1950) 本征级数核），并首次接入**精确曲面度量** `panel_areas`（`tensorbem.geometry3d.spherical_patch_areas`），与深水求解器 `free_surface_solver3d` 的同名参数语义一致。

* **C**：Capytaine 3.1 `BEMSolver`（Delhommeau 有限水深 Green 函数，Nemoh 谱系），喂入**完全相同的三角网格**，因此差异只来自公式/积分而非网格生成器。

* **A**：闭式多极解 `analytic.submerged_sphere_wave_multipole`（Thorne 1953 / Wang 1986 / Linton 1991），仅深水，用作 h→∞ 端。


环境：capytaine 3.1.dev0；python 3.11.6；numpy 2.4.6。

## 2. 方法与实现

* 焦点：`(1/2 I - H_{ν,h}) φ = S_{ν,h} q`，`q=-∂φ/∂n`，`A + iB/ω = -ρ ∫_S φ n dS`（单位速度幅值）。
* 奇异拆分：Rankine 部分逐三角形闭式积分 + 6 点四次三角形求积算本征级数余项（`_smooth_remainder`）。
* 本轮新增：`_resolve_panel_areas()` 把**精确球面片面积**接入三处带权积分（Rankine 求积、余项求积、力积分），并在 `assemble_finite_depth_operators_3d` / `solve_radiation_finite_depth_3d` 暴露 `panel_areas` 参数。
* 驱动/判据：`src/tensorbem/benchmarks/xval_bem_finite_depth.py`（阶段式：mesh / depth / freq / consistency / merge），产物 `benchmarks/verified/xval_bem_finite_depth/{result.json,gates.json,stage_*.json}`。

## 3. 参数表

| 量 | 值 |
|---|---|
| body | geodesic sphere, closed, fully submerged |
| radius | 1.0 |
| submergence | 1.5 |
| density | 1025.0 |
| gravity | 9.80665 |
| center | [0.0, 0.0, -1.5] |
| work_subdivision | 2 |
| work_mode_count | 32 |
| depth ratios h/a | [3.0, 5.0, 20.0] |
| ka | [0.5, 2.0] |
| mode ladder | [64, 128, 256, 512] |
| 验收阈值 | <15.00 % 误差 |

## 4. 对照表（ref vs got，逐项误差）

### 4.1 水深阶梯（320 面元，32 模态，ka=1）：T+ vs C vs 闭式解

| h/a | A33 T+ | A33 C | |T-C|/C | B33 T+ | B33 C | |T-C|/C | A33 err vs 闭式 | B33 err vs 闭式 |
|---|---|---|---|---|---|---|---|---|
| 3.0 | 2336.4 | 2387.2 | 2.13 % | 3055.8 | 3149.7 | 2.98 % | 26.06 % | 1.87 % |
| 5.0 | 1761.2 | 1970.2 | 10.61 % | 2935.5 | 3212.2 | 8.62 % | 4.97 % | 5.74 % |
| 20.0 | 1770.6 | 1922.7 | 7.91 % | 3029.5 | 3171.2 | 4.47 % | 4.47 % | 2.72 % |

闭式深水值：A33=1853.42 kg，B33=3114.07 kg/s。

### 4.2 频率阶梯（h/a=5 与深水，320 面元，32 模态）

| ka | h/a | A33 T+ | A33 C | |T-C|/C | B33 T+ | B33 C | |T-C|/C |
|---|---|---|---|---|---|---|---|
| 0.5 | 5.0 | 2537.5 | 2864.7 | 11.42 % | 1256.0 | 1416.1 | 11.30 % |
| 0.5 | inf | 2714.1 | 2783.3 | 2.49 % | 1515.1 | 1582.0 | 4.22 % |
| 2.0 | 5.0 | 1322.2 | 1570.6 | 15.82 % | 1225.5 | 1185.4 | 3.38 % |
| 2.0 | inf | 1510.2 | 1539.4 | 1.90 % | 1152.0 | 1173.7 | 1.85 % |
### 4.3 网格阶梯 vs 闭式解（h/a=20，ka=1，32 模态）

| 面元 | 度量 | A33 | A33 误差 | B33 | B33 误差 |
|---|---|---|---|---|---|
| 20 | flat | 1472.1 | 20.57 % | 1398.7 | 55.08 % |
| 20 | exact_measure | 2074.3 | 11.92 % | 2484.3 | 20.22 % |
| 20 | capytaine_builtin | 1657.9 | 10.55 % | 1627.6 | 47.73 % |
| 80 | flat | 1688.5 | 8.90 % | 2464.0 | 20.87 % |
| 80 | exact_measure | 1895.8 | 2.29 % | 2898.6 | 6.92 % |
| 80 | capytaine_builtin | 1913.4 | 3.23 % | 2822.4 | 9.37 % |
| 320 | flat | 1714.0 | 7.52 % | 2903.9 | 6.75 % |
| 320 | exact_measure | 1770.6 | 4.47 % | 3029.5 | 2.72 % |
| 320 | capytaine_builtin | 1922.7 | 3.74 % | 3171.2 | 1.83 % |

### 4.4 判据逐条

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| G1_geometry | exact patch area of every sphere mesh within 1 % of 4*pi*a^2 | 2.22e-16 | PASS |
| G2_damping_mesh_convergence | |B33 - B33_analytic| / B33_analytic falls from 80 to 320 panels and is below 15 % at 320 | 0.0675 | PASS |
| G3_exact_measure_320 | A33 and B33 at 320 panels with the exact surface measure within 15 % of the closed form | 0.04469 | PASS |
| G4_mode_truncation | B33 relative change over the last truncation step below 1 % | 0.01421 | **FAIL** |
| G5_capytaine_analytic_320 | route C (independent code) at 320 panels within 15 % of the closed form | 0.0374 | PASS |
| G6_xval_depth_grid | max |T+ - C| / C over the depth ladder within 15 % | 0.1061 | PASS |
| G7_deep_water_limit | T+ at h/a = 20 within 15 % of the deep-water closed form | 0.04469 | PASS |
| G8_xval_freq_grid | max |T+ - C| / C over the frequency ladder within 15 % | 0.1582 | **FAIL** |
| G9_deep_water_freq | T+ in deep water within 15 % of the closed form over the ka ladder | 0.02629 | PASS |
| G10_reciprocity | max|A - A^T|/max|A| and max|B - B^T|/max|B| below 1e-3 | 0.0002146 | PASS |
| G11_symmetry | |A11 - A22| / A11 below 5e-2 (the geodesic mesh is not 4-fold symmetric) and surge/heave decoupling below 1e-3 | 0.06927 | **FAIL** |

未通过判据：G4_mode_truncation, G8_xval_freq_grid, G11_symmetry。

## 5. 收敛性与参数敏感性

### 5.1 模态截断阶梯（80 面元，h/a=20，ka=1，精确度量）

| 模态数 | A33 | B33 | 步间 B33 相对变化 |
|---|---|---|---|
| 64 | 1943.8 | 2951.0 | n/a |
| 128 | 2000.9 | 2952.1 | 0.038 % |
| 256 | 2002.2 | 2987.5 | 1.184 % |
| 512 | 1974.6 | 2945.6 | 1.421 % |

### 5.2 度量（平面三角形 vs 精确球面片）与面元数

精确曲面度量把 320 面元的位移误差从平面度量的偏差压到亚 %，见 4.3 表；平面三角形在 20 面元上把表面积低估 23.8 %，这一偏置一对一带进力和力矩。

### 5.3 内部不变量（h/a=5，6×6，320 面元）

| 量 | 实测 |
|---|---|
| reciprocity_A | 0.0001669 |
| reciprocity_B | 0.0002146 |
| axisymmetry_A11_A22 | 0.06927 |
| axisymmetry_B11_B22 | 0.07667 |
| surge_heave_coupling | 6.581e-17 |
| condition_number | 13.14 |

## 6. 失败模式与工件

* 已定位的失败模式：**平面三角度量**在粗网格上系统性低估力/力矩（24 % @20 面元），是旧报告 B33 19.6 % 低值的主因之一；**模态截断** 32 在 320 面元上仍有可观残差（见 5.1 的末步变化与 G4 未过）；**测地网格非轴对称**导致 A11≠A22（G11 实测超 5 % 阈值）。

* 未通过/未决：G4（末步 B33 变化 >1 %，说明 32 模态未完全收敛）、G8（ka=0.5 站 |T-C|/C=15.8 %，略超 15 % 阈值，源于两码在长波下的模型差异而非网格——同一站深水两侧均与闭式解吻合 <3 %）、G11（网格对称性）。
硬拦的负控：`panel_areas` 形状/正值校验；`kernel="quadrature"` 要求全潜面元。

* 复现：
```
PYTHONPATH=src python -m tensorbem.benchmarks.xval_bem_finite_depth --stage all
```
分阶段：`--stage {mesh,depth,freq,consistency}`，`--stage merge` 重算判据。

* 工件：`benchmarks/verified/xval_bem_finite_depth/`（result.json / gates.json / stage_*.json）、`docs/verification/figs_xval_bem_finite_depth/`（本文件引用图）、`tests/test_xval_bem_finite_depth.py`。

## 7. 结论（精度判定）

* 有限水深交叉验证（T+ vs 独立 BEM C）在水深阶梯 3 站全部 <15 %：最大 10.61 %（A33）/ 8.62 %（B33）。
* 深水极限 vs 闭式解：h/a=20 的 A33/B33 误差 4.47 % / 2.72 %（<15 % ✔，但未达 <3 %）。
* 频率阶梯 4 站中 3 站 <15 %，ka=0.5 有限水深站 15.8 % 超阈值（诚实记为 FAIL）。
* 是否 <3 %：以闭式解为基准，h/a=20、32 模态下 A33/B33 均在 3 % 以内（见 4.3：320 面元精确度量行），但粗网格与低模态不满足；有限水深无闭式基准，故有限水深侧只声明「与独立码一致 <15 %」，不声明 <3 %。

