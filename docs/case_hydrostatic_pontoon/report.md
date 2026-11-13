# CASE-BEM-HYDROSTATIC-PONTOON — 矩形浮箱水静力（TensorBEM）

## 1. 问题 + 参考解出处

算例池锚点 `BEM-HYDROSTATIC-PONTOON`（仓库 TensorBEM，ref_source = "Biran & Pulido /
Rawson & Tupper 闭式"，容差 < 3 %）。对象：长 50.0 m × 宽 12.0 m ×
深 5.0 m、吃水 3.0 m 的壁面（wall-sided）矩形浮箱，重心高
KG = 4.0 m（龙骨以上），ρ = 1025.0 kg/m³，g = 9.80665 m/s²。

参考解：Biran & Lopez-Pulido, Ship Hydrostatics and Stability, 2nd ed. (Butterworth-Heinemann), ch. 3-4 (box-shaped vessel hydrostatics, metacentric height)；Rawson & Tupper, Basic Ship Theory, 5th ed., ch. 3-4 (box barge hydrostatics, BM = I/nabla)。
**引用诚实性**：none -- print original not reachable from this sandbox; citations are chapter-level and every number is evaluated from the closed forms in code (zero hand-copying)。闭式量在
`scripts/case_hydrostatic_pontoon.py::reference_values` 内由几何量程序化求值（零手抄）。

## 2. 方法与实现（复用既有能力，不另起炉灶）

两条**结构性独立**的 TensorBEM 既有路线，跑同一个箱体：

1. `waterplane`：`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` —— 追踪 z=0
   水线环，在平面上积分 A、S、I，再由闭式水静力公式组装 C。
2. `pressure`：`tensorbem.hydrostatic_pressure_route.richardson_restoring_matrix` —— 闭合箱体
   按 z=0 裁剪，积分 p = ρg(0−z) 得合力/合力矩，中心差分得 C，再对步长做 Richardson 外推。

几何先验门（先验几何再比数）：由网格顶点量出 L/B/T 包围盒
= (50.000000000000, 12.000000000000,
3.000000000000) m，Cb = 1.000000000000000（参考 1），
Cwp = 1.000000000000000（参考 1）——通过。

参数表（含无量纲量）：L/B = 4.166667，B/T = 4.000000，
T/D = 0.600000，KG/T = 1.333333，
Cb = 1.000000，Cwp = 1.000000，
Cm = 1.000000，Cp = 1.000000，
GM_T/T = 0.500000。

## 3. 对照表（ref vs got，误差逐项打勾/叉，判据 < 3 %）

| 量 | 参考解 | waterplane got | err | pressure got | err | 判据 <3% |
|---|---|---|---|---|---|---|
| displaced volume nabla [m^3] | 1800 | 1800 | +0.000e+00 | 1800 | -1.263e-16 | ✅ |
| displacement Delta [t] | 1845 | 1845 | +0.000e+00 | 1845 | -1.232e-16 | ✅ |
| block coefficient Cb [-] | 1 | 1 | +0.000e+00 | — | — | ✅ |
| waterplane area A_wp [m^2] | 600 | 600 | +0.000e+00 | — | — | ✅ |
| waterplane coeff Cwp [-] | 1 | 1 | +0.000e+00 | — | — | ✅ |
| KB (above keel) [m] | 1.5 | 1.5 | +0.000e+00 | 1.5 | -2.961e-16 | ✅ |
| LCB (from midship) [m] | 0 | 0 | +0.000e+00 | -4.632608433e-16 | -9.265e-18 | ✅ |
| LCF [m] | 0 | 0 | +0.000e+00 | — | — | ✅ |
| TCF [m] | 0 | 0 | +0.000e+00 | — | — | ✅ |
| I_T = int y^2 dA [m^4] | 7200 | 7200 | +0.000e+00 | — | — | ✅ |
| I_L = int x^2 dA [m^4] | 125000 | 125000 | +0.000e+00 | — | — | ✅ |
| I_xy [m^4] | 0 | 0 | +0.000e+00 | — | — | ✅ |
| BM_T = I_T/nabla [m] | 4 | 4 | +0.000e+00 | — | — | ✅ |
| BM_L = I_L/nabla [m] | 69.44444444 | 69.44444444 | +0.000e+00 | — | — | ✅ |
| KM_T = KB+BM_T [m] | 5.5 | 5.5 | +0.000e+00 | — | — | ✅ |
| KM_L = KB+BM_L [m] | 70.94444444 | 70.94444444 | +0.000e+00 | — | — | ✅ |
| GM_T = KM_T-KG [m] | 1.5 | 1.5 | +0.000e+00 | — | — | ✅ |
| GM_L = KM_L-KG [m] | 66.94444444 | 66.94444444 | +0.000e+00 | — | — | ✅ |
| C33 (heave) [N/m or N m/rad] | 6031089.75 | 6031089.75 | +0.000e+00 | 6031089.75 | -1.601e-12 | ✅ |
| C44 (roll) [N/m or N m/rad] | 27139903.88 | 27139903.88 | +0.000e+00 | 27139903.87 | -1.226e-12 | ✅ |
| C55 (pitch) [N/m or N m/rad] | 1211243858 | 1211243858 | +0.000e+00 | 1211243858 | -5.785e-13 | ✅ |
| max |off-diagonal C_ij| / max|C| [-] | 0 | 0 | +0.000e+00 | 6.950114053e-14 | +6.950e-14 | ✅ |


主网格 = [4, 3, 3]（水线路线 108 个湿表面三角形，
压力路线 132 个闭合三角形）。

## 4. 图

![fig_rel_errors.png](fig_rel_errors.png)

![fig_refinement.png](fig_refinement.png)

![fig_draft_sensitivity.png](fig_draft_sensitivity.png)

![fig_heel_righting.png](fig_heel_righting.png)


图注：① 逐量相对误差（对数轴），两路线均在双精度舍入量级，距 3 % 判据约
10.3 个数量级；
② 面板加密（40…1728 面板）误差曲线平坦
——平面面板对体积/水线面矩/压力矩都精确，误差不随离散变；
③ 吃水扫描 BM_T 与 B²/(12T) 重合；④ 横倾复原力臂：压力积分 vs 壁面公式 vs GM·sinφ。

## 5. 收敛性 / 参数敏感性

### 5.1 面板加密（收敛性）

| level (n_L,n_B,n_T) | wet panels | max abs rel err (waterplane) | max abs rel err (pressure) |
|---|---|---|---|
| [2, 2, 2] | 40 | 1.470e-12 | 1.470e-12 |
| [4, 3, 3] | 108 | 1.601e-12 | 1.601e-12 |
| [8, 6, 6] | 432 | 5.013e-12 | 5.013e-12 |
| [16, 12, 12] | 1728 | 3.892e-11 | 3.892e-11 |

预期平坦（箱体面板精确）。实测确实在 1e-12…4e-11 之间抖动，且**随面板数缓慢上升**——这不是
离散误差（否则会按 O(h²) 下降），而是压力路线在更多面板上的舍入累加；水线路线在 4 个网格上
逐位不变（体积/水线面矩为精确和）。结论：本算例的误差只可能来自实现 bug，不是离散误差。

### 5.2 吃水敏感性（并验证 d∇/dT = A_wp 恒等式）

| T [m] | BM_T ref [m] | BM_T got [m] | rel err | A_wp ref [m²] | d∇/dT got [m²] | rel err |
|---|---|---|---|---|---|---|
| 2.0 | 6.000000 | 6.000000 | +0.00e+00 | 600.0 | 600.0000 | -5.31e-13 |
| 2.5 | 4.800000 | 4.800000 | +0.00e+00 | 600.0 | 600.0000 | -1.52e-13 |
| 3.0 | 4.000000 | 4.000000 | +0.00e+00 | 600.0 | 600.0000 | -3.41e-13 |
| 3.5 | 3.428571 | 3.428571 | +0.00e+00 | 600.0 | 600.0000 | -5.31e-13 |
| 4.0 | 3.000000 | 3.000000 | +0.00e+00 | 600.0 | 600.0000 | +2.27e-13 |

### 5.3 g 取值敏感性（仅 C 随 g 线性缩放，几何量不变）

| g [m/s²] | C33 rel err | C44 rel err | ∇ rel err |
|---|---|---|---|
| 9.80665 | +0.000e+00 | +0.000e+00 | +0.000e+00 |
| 9.81 | +3.416e-04 | +3.416e-04 | +0.000e+00 |
| 9.8 | -6.781e-04 | -6.781e-04 | +0.000e+00 |

### 5.4 压力路线有限差分步长

| step | C33 rel err | C44 rel err | C55 rel err | obs. order C44 | obs. order C55 |
|---|---|---|---|---|---|
| 4.0e-03 | -2.631e-13 | +1.867e-05 | +5.632e-06 | +nan | +nan |
| 2.0e-03 | +8.178e-13 | +4.667e-06 | +1.408e-06 | +2.00 | +2.00 |
| 1.0e-03 | +5.090e-13 | +1.167e-06 | +3.520e-07 | +2.00 | +2.00 |
| 5.0e-04 | -2.579e-12 | +2.917e-07 | +8.800e-08 | +2.00 | +2.00 |

C33 对步长不敏感（它只由零阶力构成，差分精确到舍入）；C44/C55 的误差随步长按二阶梯降，
实测阶数程序化给出（上表），与中心差分 O(h²) 的预期一致——Richardson 外推因此把主网格的
C44/C55 误差压到 1e-12 量级。

### 5.5 横倾复原力臂（独立于矩阵的第三路校核）

| φ [°] | GZ got [m] | GZ wall-sided [m] | GM·sinφ [m] | rel err vs wall-sided |
|---|---|---|---|---|
| 0.5 | 0.013091 | 0.013091 | 0.013090 | -5.830e-14 |
| 1.0 | 0.026189 | 0.026189 | 0.026179 | +7.286e-15 |
| 2.0 | 0.052434 | 0.052434 | 0.052349 | -2.250e-15 |
| 3.0 | 0.078791 | 0.078791 | 0.078504 | -3.347e-15 |
| 5.0 | 0.132068 | 0.132068 | 0.130734 | +1.471e-15 |

## 6. 失败模式与工件

- 判据：全部 22 项 < 3 % → 本算例 PASS；最大相对误差 1.601e-12。
- 失败模式（已知）：(a) 非平面面板引入 O(h²) 剖面误差——本箱体不触发，同脚本换 spheroid 即暴露；
  (b) 压力路线差分步长过大时 O(h²) 截断可淹没真实差（表 5.4）；(c) KG > KM_T 时 GM<0、复原矩阵变号，
  本算例 GM_T = 1.5000 m > 0，不触发；(d) 力矩积分若用一阶求积规则会丢 O(h²)
  （生态内 m16662 已定位该缺陷），此处用模块现成的二阶精确规则。
- 工件：`results.json`（全部数值 + 环境/版本/sha256）、`report.md`、4 张 png、`tests/test_case_hydrostatic_pontoon.py`。
- 复现：`python3 scripts/case_hydrostatic_pontoon.py --out artifacts/case_hydrostatic_pontoon`
