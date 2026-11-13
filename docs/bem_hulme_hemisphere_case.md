# CASE-BEM-HULME-HEMISPHERE：浮半球辐射/绕射（3D 频域 BEM 对标 Hulme 1982）

> 算例码 `BEM-HULME-HEMISPHERE`（算例池 id 138，任务 17292，仓库 TensorBEM，容差 <5%）。本文件由 `benchmarks/verified/casepool_bem_hulme_hemisphere/make_doc.py` 从 `results.json`/`evidence.json` 程序化渲染，文档内不手抄任何数字。

**机器裁定：PASS**

## ① 问题与参考解出处

半径 a = 1 m 的浮半球（平面沿平均自由面 z = 0，曲面入水）在无限深水中作单位幅值强迫振荡并受迎浪单位波幅入射。求 6 自由度中与来流平面内/а轴的 surge（k 沿 x）与 heave 两个模态的无量纲附加质量、辐射阻尼，以及绕射激励力的幅值与相位。物理背景：这是频域势流 BEM 的经典验证算例——半球是少数几个存在半解析解的有限吃水浮体，能同时检验自由面 Green 函数、水线积分与 Kochin/Haskind 远场关系。

- 参考解文献：Hulme, A. (1982) The wave forces acting on a floating hemisphere undergoing forced periodic oscillations. J. Fluid Mech. 121:443-463.
- 参考解方法：spherical wave-free multipole expansion (Hulme's semi-analytical solution)
- 参考解实现：`tensorbem.analytic.hulme_hemisphere_radiation`，取 144 项（截断阶梯 48/96/144）
- 参考解自收敛（96→144 项最大相对漂移）：6.464e-05

**关于「零手抄」的如实说明**：

- NOT AVAILABLE (publisher paywall, no copy in the sandbox): no digit of Hulme's printed table is transcribed or hand-copied anywhere in this repository. Every reference number in results.json is computed at run time by the wave-free multipole series (recomputed and cross-checked by evidence.py), so 'ref' here means 'Hulme's semi-analytical method, re-derived and self-converged', not 'Hulme's printed digits' -- see docs/bem_hulme_hemisphere_case.md sections 1 and 6.
- 共享内核风险（如实登记）：the reference series and the solver share tensorbem.free_surface3d.wave_function_3d, so agreement alone cannot catch a defect in that kernel; the independent axes are the third-party BEM crosscheck (capytaine, docs/capytaine_crosscheck.md) and the ka->0 rigid-lid even-mode series.
- 独立校验轴：`docs/capytaine_crosscheck.md (capytaine, a third-party BEM code, same body and frequencies)`；`benchmarks/verified/bem_rigid_lid_hemisphere/summary.md (ka -> 0 rigid-lid even-mode series 0.830954)`

## ② 方法与实现

1. 几何：`tensorbem.geometry3d.hemisphere_hull_mesh(subdivisions)`——八面体四个下侧面四分并径向投影到 r = a、z ≤ 0，面板数 4·4^n，水线顶点严格落在 z = 0；
2. 自由面算子：`tensorbem.free_surface_solver3d.assemble_free_surface_operators_3d`（Kelvin/Havelock/Delhommeau 深水 Green 函数，Rankine 直项与镜像项分别用Van Oosterom–Strackee 立体角与 polar/Duffy 近奇积分）；
3. 内部 CHIEF 空场方程（`chief_points='auto'`）抑制不规则频率；
4. 辐射解：`solve_radiation_3d`，surge（mode 0）与 heave（mode 2），单位速度幅值；
5. 绕射解：`solve_diffraction_3d`，wave_angle = 0（迎浪），取复数激励力，与 Haskind 远场关系一致；
6. 归一化：A/(ρ∇)、B/(ρω∇)，∇ = 2πa³/3 = 2.094395 m³（精确值，不用网格体积，避免把几何误差混进系数）；力按 ρgAa² 归一；时间约定 exp(−iωt)。
7. 判据（动手前预注册，事后不放宽）：`G1_case_mesh_band_error_below_5pct` = |rel err| < 5% for every quantity and every ka on the case mesh；`G2_geometry_precondition` = |nabla_mesh/nabla-1| < 2% and |A_wp/(pi a^2)-1| < 2% on the case mesh；`G3_refinement_does_not_worsen` = band-max error at 1024 panels <= band-max error at 256 panels；`G4_reference_self_converged` = reference 96->144 term drift at least 20x below the 5 % gate

## ③ 参数表

| 项目 | 取值 |
|---|---|
| 几何 | 浮半球 a = 1.0 m，平面与液面齐平 |
| ∇ / A_wp | 2.094395 m³ / 3.141593 m² |
| ρ / g / A | 1025.0 kg/m³ / 9.80665 m/s² / 1.0 m |
| ka 频带（无量纲波数） | 0.2, 0.5, 0.8, 1, 1.5, 2 |
| 主网格 | subdivisions 4 = 1024 常数面元 |
| 粗网格 | subdivisions 3 = 256 常数面元 |
| 网格阶梯 | 1024 面板（ka = 0.5, 1, 2） |
| 参考截断 | 48/96/144 项 |
| 容差 | 每个量、每个 ka 相对误差 < 5% |
| 几何前置条件 | |∇_mesh/∇−1| < 2%、|A_wp/(πa²)−1| < 2% |

派生无量纲数：波长 λ/a = 2π/ka（ka=0.2 时 31.4，ka=2 时 3.14）；频率 ω√(a/g) = √ka；雷诺数/傅汝德数对本问题不适用（线性势流、无黏、无航速）。

## ④ 参考 vs 计算（逐项误差）

### 主网格（subdivisions 4 = 1024 常数面元）

| ka | A11/(ρ∇) ref | got | err | B11/(ρω∇) ref | got | err | A33/(ρ∇) ref | got | err | B33/(ρω∇) ref | got | err | |Fx|/(ρgAa²) ref | got | err | |Fz|/(ρgAa²) ref | got | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.55153 | 0.54936 | -0.39% ✓ | 0.00815 | 0.00807 | -0.96% ✓ | 0.79383 | 0.79260 | -0.16% ✓ | 0.27930 | 0.27873 | -0.20% ✓ | 0.58438 | 0.58136 | -0.52% ✓ | 2.41864 | 2.41624 | -0.10% ✓ |
| 0.5 | 0.64390 | 0.64091 | -0.46% ✓ | 0.09867 | 0.09770 | -0.99% ✓ | 0.58608 | 0.58544 | -0.11% ✓ | 0.33906 | 0.33896 | -0.03% ✓ | 1.28578 | 1.27889 | -0.54% ✓ | 1.68539 | 1.68525 | -0.01% ✓ |
| 0.8 | 0.64212 | 0.63939 | -0.42% ✓ | 0.26531 | 0.26276 | -0.96% ✓ | 0.46979 | 0.46916 | -0.13% ✓ | 0.28990 | 0.29036 | +0.16% ✓ | 1.66682 | 1.65800 | -0.53% ✓ | 1.23204 | 1.23304 | +0.08% ✓ |
| 1 | 0.57395 | 0.57206 | -0.33% ✓ | 0.35349 | 0.35030 | -0.90% ✓ | 0.42844 | 0.42765 | -0.18% ✓ | 0.24840 | 0.24915 | +0.30% ✓ | 1.72087 | 1.71213 | -0.51% ✓ | 1.02007 | 1.02143 | +0.13% ✓ |
| 1.5 | 0.36832 | 0.36829 | -0.01% ✓ | 0.40135 | 0.39843 | -0.73% ✓ | 0.38905 | 0.38771 | -0.34% ✓ | 0.16062 | 0.16174 | +0.69% ✓ | 1.49719 | 1.49063 | -0.44% ✓ | 0.66974 | 0.67139 | +0.25% ✓ |
| 2 | 0.24928 | 0.24985 | +0.23% ✓ | 0.34241 | 0.34043 | -0.58% ✓ | 0.38838 | 0.38634 | -0.53% ✓ | 0.10307 | 0.10425 | +1.15% ✓ | 1.19762 | 1.19289 | -0.40% ✓ | 0.46462 | 0.46688 | +0.49% ✓ |

相位对照（度，exp(−iωt) 约定）：

| ka | phase Fx ref (°) | got (°) | Δ (°) | phase Fz ref (°) | got (°) | Δ (°) |
|---|---|---|---|---|---|---|
| 0.2 | -89.714 | -89.715 | -0.001 | -2.793 | -2.790 | +0.003 |
| 0.5 | -87.004 | -87.019 | -0.015 | -12.639 | -12.627 | +0.012 |
| 0.8 | -82.999 | -83.038 | -0.038 | -25.185 | -25.163 | +0.023 |
| 1 | -81.757 | -81.806 | -0.050 | -34.289 | -34.258 | +0.031 |
| 1.5 | -87.842 | -87.887 | -0.045 | -58.538 | -58.509 | +0.029 |
| 2 | -103.921 | -103.915 | +0.006 | -84.126 | -84.129 | -0.002 |

### 粗网格（subdivisions 3 = 256 常数面元，用于 h 收敛对照）

| ka | A11/(ρ∇) ref | got | err | B11/(ρω∇) ref | got | err | A33/(ρ∇) ref | got | err | B33/(ρω∇) ref | got | err | |Fx|/(ρgAa²) ref | got | err | |Fz|/(ρgAa²) ref | got | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.55153 | 0.54287 | -1.57% ✓ | 0.00815 | 0.00784 | -3.78% ✓ | 0.79383 | 0.78884 | -0.63% ✓ | 0.27930 | 0.27703 | -0.81% ✓ | 0.58438 | 0.57243 | -2.05% ✓ | 2.41864 | 2.40909 | -0.39% ✓ |
| 0.5 | 0.64390 | 0.63208 | -1.84% ✓ | 0.09867 | 0.09485 | -3.87% ✓ | 0.58608 | 0.58347 | -0.45% ✓ | 0.33906 | 0.33865 | -0.12% ✓ | 1.28578 | 1.25857 | -2.12% ✓ | 1.68539 | 1.68485 | -0.03% ✓ |
| 0.8 | 0.64212 | 0.63135 | -1.68% ✓ | 0.26531 | 0.25535 | -3.75% ✓ | 0.46979 | 0.46723 | -0.55% ✓ | 0.28990 | 0.29175 | +0.64% ✓ | 1.66682 | 1.63203 | -2.09% ✓ | 1.23204 | 1.23604 | +0.32% ✓ |
| 1 | 0.57395 | 0.56647 | -1.30% ✓ | 0.35349 | 0.34102 | -3.53% ✓ | 0.42844 | 0.42529 | -0.73% ✓ | 0.24840 | 0.25138 | +1.20% ✓ | 1.72087 | 1.68646 | -2.00% ✓ | 1.02007 | 1.02558 | +0.54% ✓ |
| 1.5 | 0.36832 | 0.36812 | -0.05% ✓ | 0.40135 | 0.38994 | -2.84% ✓ | 0.38905 | 0.38371 | -1.37% ✓ | 0.16062 | 0.16512 | +2.80% ✓ | 1.49719 | 1.47136 | -1.73% ✓ | 0.66974 | 0.67649 | +1.01% ✓ |
| 2 | 0.24928 | 0.25141 | +0.86% ✓ | 0.34241 | 0.33473 | -2.24% ✓ | 0.38838 | 0.38044 | -2.05% ✓ | 0.10307 | 0.10797 | +4.76% ✓ | 1.19762 | 1.17900 | -1.55% ✓ | 0.46462 | 0.47349 | +1.91% ✓ |

## ⑤ 收敛性与参数敏感性

### 5.1 网格阶梯（h 收敛）

| ka | panels | A11 err | B11 err | A33 err | B33 err | |Fx| err | |Fz| err | cond |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1024 | -0.46% | -0.99% | -0.11% | -0.03% | -0.54% | -0.01% | 2.62 |
| 1 | 1024 | -0.33% | -0.90% | -0.18% | +0.30% | -0.51% | +0.13% | 2.69 |
| 2 | 1024 | +0.23% | -0.58% | -0.53% | +1.15% | -0.40% | +0.49% | 2.40 |


### 5.2 参考解截断阶梯（参考自身的敏感性）

| ka | 48→96 项漂移 | 96→144 项漂移 | 尾部比（A33） |
|---|---|---|---|
| 0.2 | 8.438e-05 | 1.596e-05 | 1.892e-01 |
| 0.5 | 1.005e-04 | 1.898e-05 | 1.890e-01 |
| 0.8 | 1.257e-04 | 2.374e-05 | 1.889e-01 |
| 1 | 1.476e-04 | 2.784e-05 | 1.888e-01 |
| 1.5 | 2.219e-04 | 4.181e-05 | 1.887e-01 |
| 2 | 3.436e-04 | 6.464e-05 | 1.889e-01 |

### 5.3 CHIEF 内部空场方程开关（不规则频率控制）

| ka | chief | cond | A33 err | B33 err | B33/(ρω∇) |
|---|---|---|---|---|---|
| 1 | on | 2.69 | -0.18% | +0.30% | 0.24915 |
| 2 | on | 2.40 | -0.53% | +1.15% | 0.10425 |
| 1 | off | 2.68 | -0.18% | +0.30% | 0.24915 |
| 2 | off | 2.42 | -0.53% | +1.13% | 0.10423 |

（频带止于 ka = 2，刻意避开该水线首个不规则频率 ka ≈ 2.58，避免把不规则频率假解误读成离散误差。）

## ⑥ 失败模式、工件与生态摩擦

### 6.1 判据裁定

| 判据 | 内容 | 实测 | 裁定 |
|---|---|---|---|
| `G1_case_mesh_band_error_below_5pct` | |rel err| < 5% for every quantity and every ka on the case mesh | 1.1480e-02 | PASS |
| `G2_geometry_precondition` | |nabla_mesh/nabla-1| < 2% and |A_wp/(pi a^2)-1| < 2% on the case mesh | 体积 0.587%，水线 0.161% | PASS |
| `G3_refinement_does_not_worsen` | band-max error at 1024 panels <= band-max error at 256 panels | None | PASS |
| `G4_reference_self_converged` | reference 96->144 term drift at least 20x below the 5 % gate | 6.4638e-05 | PASS |

**总判定：PASS**

### 6.2 失败模式与已知局限

- **几何前置条件是本算例的第一道门**：256 面板（subdivisions 3）的内接多面体体积比精确 ∇ 低 2.32%，已越过 2% 前置门——这不是系数误差，而是「体都没摆对就别看表」的直接证据；上一轮（m17181）正是在这里 honest-FAIL。本轮把主网格提到 1024 面板（体积误差 0.59%），前置门与系数门同时通过，判据本身不放宽。
- **参考解与求解器共享自由面核**（`wave_function_3d`）：两者一致不能排除核本身有误；独立轴只有第三方 BEM（capytaine）与 ka→0 刚盖偶模级数两个。
- **纸面数字不可得**：Hulme (1982) 原文表格在本环境无法获取，因此「ref」= Hulme 方法的重新推导级数（自收敛 3e-5 量级），不是印刷数字；证据等级按 REAL（半解析方法，可复算），印刷值对照记 WITHHELD。
- **线性势流假设**：无黏、无航速、无限水深；不能用于强非线性/碎击/浅水工况。

### 6.3 工件与复现

- `benchmarks/verified/casepool_bem_hulme_hemisphere/results.json`
- `benchmarks/verified/casepool_bem_hulme_hemisphere/reference.json`
- `benchmarks/verified/casepool_bem_hulme_hemisphere/environment.json`
- `benchmarks/verified/casepool_bem_hulme_hemisphere/run.log`
- `benchmarks/verified/casepool_bem_hulme_hemisphere/report.md`
- `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png`
- `benchmarks/verified/casepool_bem_hulme_hemisphere/evidence.json`

```bash
# 1) 全量跑（4 分片并行；约 15 分钟，1024 面板/频点为主耗时）
python benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py --out benchmarks/verified/casepool_bem_hulme_hemisphere --shards 4 --parallel 4
# 2) 机器核验（工件存在性 + 非占位审计 + 调用方 grep + ref/got 复算 + 几何前置）
python benchmarks/verified/casepool_bem_hulme_hemisphere/evidence.py --strict
# 3) 测试
python -m pytest tests/test_hulme_hemisphere_case.py -q
# 4) 重新渲染本文档
python benchmarks/verified/casepool_bem_hulme_hemisphere/make_doc.py --results benchmarks/verified/casepool_bem_hulme_hemisphere/results.json
```

运行环境：Python 3.11.6，numpy 2.4.6，host 36cb7828dbd5，arch loongarch64。

### 6.4 达标层级（机器核验，非自声明）

| 层级 | 内容 | 证据 | 裁定 |
|---|---|---|---|
| L1 | 算例工件（报告含裁定、数值可溯源） | `report.md`/`figure.png` | PASS |
| L2 | 非占位实现 + 入口 + 真实运行工件 | 4 个实现文件、2 个入口、10 个工件（含 sha256） | PASS |
| L3 | 集成调用方（驱动 + 测试 + 证据复算） | 4/4 处 grep 命中 | PASS |
| L4 | benchmark ref vs got + repro_cmd | 48 行数值对照，带内最大相对误差 1.15% | PASS |

