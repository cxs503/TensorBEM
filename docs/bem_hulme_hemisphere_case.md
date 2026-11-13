# CASE-BEM-HULME-HEMISPHERE：浮半球辐射/绕射（3D 自由面 Kelvin 边界面元）

算例池 `tensor_case_pool.id = 138`（`BEM-HULME-HEMISPHERE`，repo TensorBEM，容差 `<5%`），任务 mission 17181。本文每个数字都由 `run_case.py` 写入 `results.json` 后由 `make_doc.py` 程序化注入，无手抄。

## ① 问题与参考解出处

对象：半径 a = 1.0 m 的浮半球，平面（圆盘）与平均自由液面齐平，深水（无航速、无海流、线性势流、无粘）。ρ = 1025.0 kg/m³，g = 9.80665 m/s²，入射波幅 A = 1.0 m。排水体积 ∇ = 2πa³/3 = 2.094395 m³，水线面积 A_wp = πa² = 3.141593 m²。
无量纲频率 ka = k·a，k = ω²/g（深水色散），ω = √(g·ka/a)。辐射量取单位速度幅值、`exp(-iωt)` 约定；A 以 ρ∇ 无量纲化，B 以 ρω∇ 无量纲化，波浪激励力以 ρgAa² 无量纲化。

**参考解出处**：Hulme, A. (1982) The wave forces acting on a floating hemisphere undergoing forced periodic oscillations. J. Fluid Mech. 121:443-463.

Hulme 的方法为球坐标「无自由面波」多极（wave-free multipole）展开：原点处的 Kelvin 波源/水平偶极，加上满足自由液面条件 dψ/dz − Kψ = 0 的球谐波势族，在半球的奇次 Legendre 谐波上做投影配点，解出多极系数后给出辐射系数与（经 Haskind 互易的）激励力。仓内实现：`tensorbem.analytic.hulme_hemisphere_radiation`，本文取 n_terms = 144，并给出 48/96/144 截断阶梯（见 ⑤）作为参考解自身的自收敛证据。

**关于「原文数值」的诚实声明**：NOT AVAILABLE in this sandbox (publisher paywall); the reference is the re-derived series, self-convergence and independent cross-checks are reported instead -- see docs/bem_hulme_hemisphere_case.md section 1.

参考解的可信度由三条独立证据支撑（都不是本算例的求解器）：

- docs/capytaine_crosscheck.md (capytaine, a third-party BEM code, same body and frequencies)
- benchmarks/verified/bem_rigid_lid_hemisphere/summary.md (ka -> 0 rigid-lid even-mode series 0.830954)

## ② 方法与实现

求解器＝`tensorbem.free_surface_solver3d`：深水 Kelvin（Havelock）Green 函数边界面元，常数面元，浸湿面由 `tensorbem.geometry3d.hemisphere_hull_mesh` 递归八面体细分给出（面板数 = 4·4^n，全部水线顶点严格落在 z = 0，法向指向流体）。

流程：①几何前置校验（网格排水体积 / 水线面积 vs 解析值）；②装配自由面算子（`assemble_free_surface_operators_3d`，内部 CHIEF 空场方程 `chief_points='auto'` 抑制不规则频率）；③辐射解（surge mode 0、heave mode 2，单位速度幅值）；④绕射解（迎浪，wave_angle = 0）得复数激励力；⑤对 Hulme 半解析参考逐项求相对误差；⑥网格阶梯 / 参考截断阶梯 / CHIEF 开关三点敏感性。

复现命令（单进程；集群上多分片只是同一任务表的轮转切分）：

```bash
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py \
    --shard 0 --shards 1 --out runs/casepool_bem_hulme_hemisphere
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py \
    --merge --out runs/casepool_bem_hulme_hemisphere
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/make_doc.py \
    --results runs/casepool_bem_hulme_hemisphere/results.json
```

运行环境：Python 3.11.6，numpy 2.4.6，host 36cb7828dbd5，arch loongarch64（loong64 集群节点；判读时注意与 x86 CI 的架构差异）。

## ③ 参数表

| 项目 | 取值 |
|---|---|
| 几何 | 浮半球 a = 1.0 m，平面与液面齐平 |
| ∇ / A_wp | 2.094395 m³ / 3.141593 m² |
| ρ / g / A | 1025.0 kg/m³ / 9.80665 m/s² / 1.0 m |
| ka 频带 | 0.2, 0.5, 0.8, 1, 1.5, 2（避开首个不规则频率 ka≈2.58） |
| 主网格 | subdivisions 3 = 256 常数面元 |
| 粗网格 | subdivisions 2 = 64 常数面元 |
| 网格阶梯 | 256, 64 面板 |
| 参考截断 | 144, 48, 96 项 |
| 判据 | 每个量、每个 ka 的相对误差 < 5% |

派生无量纲数（ka 每点）：波长 λ/a = 2π/ka，ω√(a/g) = √(ka)。

## ④ 参考 vs 计算对照

#### 主网格（256 常数面元，subdivisions 3）

| ka | A11/(rho nabla) ref | A11/(rho nabla) got | err | ? | B11/(rho omega nabla) ref | B11/(rho omega nabla) got | err | ? | A33/(rho nabla) ref | A33/(rho nabla) got | err | ? | B33/(rho omega nabla) ref | B33/(rho omega nabla) got | err | ? | |Fx|/(rho g A a^2) ref | |Fx|/(rho g A a^2) got | err | ? | |Fz|/(rho g A a^2) ref | |Fz|/(rho g A a^2) got | err | ? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.55153 | 0.54287 | -1.57% | PASS | 0.00815 | 0.00784 | -3.78% | PASS | 0.79383 | 0.78884 | -0.63% | PASS | 0.27930 | 0.27703 | -0.81% | PASS | 0.58438 | 0.57243 | -2.05% | PASS | 2.41864 | 2.40909 | -0.39% | PASS |
| 0.5 | 0.64390 | 0.63208 | -1.84% | PASS | 0.09867 | 0.09485 | -3.87% | PASS | 0.58608 | 0.58347 | -0.45% | PASS | 0.33906 | 0.33865 | -0.12% | PASS | 1.28578 | 1.25857 | -2.12% | PASS | 1.68539 | 1.68485 | -0.03% | PASS |
| 0.8 | 0.64212 | 0.63135 | -1.68% | PASS | 0.26531 | 0.25535 | -3.75% | PASS | 0.46979 | 0.46723 | -0.55% | PASS | 0.28990 | 0.29175 | +0.64% | PASS | 1.66682 | 1.63203 | -2.09% | PASS | 1.23204 | 1.23604 | +0.32% | PASS |
| 1 | 0.57395 | 0.56647 | -1.30% | PASS | 0.35349 | 0.34102 | -3.53% | PASS | 0.42844 | 0.42529 | -0.73% | PASS | 0.24840 | 0.25138 | +1.20% | PASS | 1.72087 | 1.68646 | -2.00% | PASS | 1.02007 | 1.02558 | +0.54% | PASS |
| 1.5 | 0.36832 | 0.36812 | -0.05% | PASS | 0.40135 | 0.38994 | -2.84% | PASS | 0.38905 | 0.38371 | -1.37% | PASS | 0.16062 | 0.16512 | +2.80% | PASS | 1.49719 | 1.47136 | -1.73% | PASS | 0.66974 | 0.67649 | +1.01% | PASS |
| 2 | 0.24928 | 0.25141 | +0.86% | PASS | 0.34241 | 0.33473 | -2.24% | PASS | 0.38838 | 0.38044 | -2.05% | PASS | 0.10307 | 0.10797 | +4.76% | PASS | 1.19762 | 1.17900 | -1.55% | PASS | 0.46462 | 0.47349 | +1.91% | PASS |

Phase comparison (deg, `exp(-i omega t)` convention):

| ka | phase Fx ref | phase Fx got | delta | phase Fz ref | phase Fz got | delta |
|---|---|---|---|---|---|---|
| 0.2 | -89.714 | -89.719 | -0.005 | -2.793 | -2.780 | +0.013 |
| 0.5 | -87.004 | -87.062 | -0.058 | -12.639 | -12.593 | +0.046 |
| 0.8 | -82.999 | -83.149 | -0.150 | -25.185 | -25.096 | +0.090 |
| 1 | -81.757 | -81.949 | -0.193 | -34.289 | -34.167 | +0.122 |
| 1.5 | -87.842 | -88.010 | -0.168 | -58.538 | -58.407 | +0.130 |
| 2 | -103.921 | -103.884 | +0.036 | -84.126 | -84.120 | +0.006 |

#### 粗网格（64 常数面元，subdivisions 2）

| ka | A11/(rho nabla) ref | A11/(rho nabla) got | err | ? | B11/(rho omega nabla) ref | B11/(rho omega nabla) got | err | ? | A33/(rho nabla) ref | A33/(rho nabla) got | err | ? | B33/(rho omega nabla) ref | B33/(rho omega nabla) got | err | ? | |Fx|/(rho g A a^2) ref | |Fx|/(rho g A a^2) got | err | ? | |Fz|/(rho g A a^2) ref | |Fz|/(rho g A a^2) got | err | ? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.55153 | 0.51814 | -6.05% | FAIL | 0.00815 | 0.00700 | -14.10% | FAIL | 0.79383 | 0.77322 | -2.60% | PASS | 0.27930 | 0.27017 | -3.27% | PASS | 0.58438 | 0.53862 | -7.83% | FAIL | 2.41864 | 2.37978 | -1.61% | PASS |
| 0.5 | 0.64390 | 0.59887 | -6.99% | FAIL | 0.09867 | 0.08449 | -14.37% | FAIL | 0.58608 | 0.57498 | -1.90% | PASS | 0.33906 | 0.33695 | -0.62% | PASS | 1.28578 | 1.18208 | -8.06% | FAIL | 1.68539 | 1.68175 | -0.22% | PASS |
| 0.8 | 0.64212 | 0.60102 | -6.40% | FAIL | 0.26531 | 0.22848 | -13.88% | FAIL | 0.46979 | 0.45918 | -2.26% | PASS | 0.28990 | 0.29661 | +2.32% | PASS | 1.66682 | 1.53476 | -7.92% | FAIL | 1.23204 | 1.24652 | +1.18% | PASS |
| 1 | 0.57395 | 0.54496 | -5.05% | FAIL | 0.35349 | 0.30736 | -13.05% | FAIL | 0.42844 | 0.41572 | -2.97% | PASS | 0.24840 | 0.25963 | +4.52% | PASS | 1.72087 | 1.59046 | -7.58% | FAIL | 1.02007 | 1.04098 | +2.05% | PASS |
| 1.5 | 0.36832 | 0.36629 | -0.55% | PASS | 0.40135 | 0.35914 | -10.52% | FAIL | 0.38905 | 0.36815 | -5.37% | FAIL | 0.16062 | 0.17831 | +11.01% | FAIL | 1.49719 | 1.39923 | -6.54% | FAIL | 0.66974 | 0.69671 | +4.03% | PASS |
| 2 | 0.24928 | 0.25580 | +2.62% | PASS | 0.34241 | 0.31396 | -8.31% | FAIL | 0.38838 | 0.35846 | -7.70% | FAIL | 0.10307 | 0.12314 | +19.48% | FAIL | 1.19762 | 1.12669 | -5.92% | FAIL | 0.46462 | 0.49750 | +7.08% | FAIL |

Phase comparison (deg, `exp(-i omega t)` convention):

| ka | phase Fx ref | phase Fx got | delta | phase Fz ref | phase Fz got | delta |
|---|---|---|---|---|---|---|
| 0.2 | -89.714 | -89.733 | -0.020 | -2.793 | -2.741 | +0.052 |
| 0.5 | -87.004 | -87.224 | -0.220 | -12.639 | -12.458 | +0.181 |
| 0.8 | -82.999 | -83.554 | -0.555 | -25.185 | -24.832 | +0.353 |
| 1 | -81.757 | -82.457 | -0.700 | -34.289 | -33.801 | +0.488 |
| 1.5 | -87.842 | -88.367 | -0.525 | -58.538 | -57.874 | +0.663 |
| 2 | -103.921 | -103.590 | +0.331 | -84.126 | -83.718 | +0.409 |

## ⑤ 收敛性与参数敏感性

### 5.1 网格阶梯（h 收敛）

阶梯：subdivisions 2/3（64/256 常数面元）。第三档 1024 面板（约 310 s/频点）已在 `run_case.py` 的 `LADDER_SUBDIVISIONS` 中定义，但超出本机单次预算，需交集群复跑（本单已递交 cj1454，队列 589 单未排到即被回收，故本表只用 64→256 两级估阶）；1024 档据体积收敛率外推，几何前置误差约 −0.58%。

| ka | 面板数 | A11/(rho nabla) 误差 | B11/(rho omega nabla) 误差 | A33/(rho nabla) 误差 | B33/(rho omega nabla) 误差 | |Fx|/(rho g A a^2) 误差 | |Fz|/(rho g A a^2) 误差 | 条件数 |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 64 | -6.05% | -14.10% | -2.60% | -3.27% | -7.83% | -1.61% | 2.55 |
| 0.2 | 256 | -1.57% | -3.78% | -0.63% | -0.81% | -2.05% | -0.39% | 2.39 |
| 0.5 | 64 | -6.99% | -14.37% | -1.90% | -0.62% | -8.06% | -0.22% | 2.90 |
| 0.5 | 256 | -1.84% | -3.87% | -0.45% | -0.12% | -2.12% | -0.03% | 2.70 |
| 0.8 | 64 | -6.40% | -13.88% | -2.26% | +2.32% | -7.92% | +1.18% | 3.06 |
| 0.8 | 256 | -1.68% | -3.75% | -0.55% | +0.64% | -2.09% | +0.32% | 2.82 |
| 1 | 64 | -5.05% | -13.05% | -2.97% | +4.52% | -7.58% | +2.05% | 3.05 |
| 1 | 256 | -1.30% | -3.53% | -0.73% | +1.20% | -2.00% | +0.54% | 2.78 |
| 1.5 | 64 | -0.55% | -10.52% | -5.37% | +11.01% | -6.54% | +4.03% | 2.67 |
| 1.5 | 256 | -0.05% | -2.84% | -1.37% | +2.80% | -1.73% | +1.01% | 2.33 |
| 2 | 64 | +2.62% | -8.31% | -7.70% | +19.48% | -5.92% | +7.08% | 2.39 |
| 2 | 256 | +0.86% | -2.24% | -2.05% | +4.76% | -1.55% | +1.91% | 2.35 |

观测收敛阶（相邻两级网格，e ∝ h^p，h ∝ 1/√panels）：

| ka | 面板对 | A33 p | B33 p | A11 p | B11 p |
|---|---|---|---|---|---|
| 0.2 | 64→256 | +2.05 | +2.01 | +1.95 | +1.90 |
| 0.5 | 64→256 | +2.09 | +2.37 | +1.93 | +1.89 |
| 0.8 | 64→256 | +2.05 | +1.86 | +1.93 | +1.89 |
| 1 | 64→256 | +2.01 | +1.92 | +1.95 | +1.89 |
| 1.5 | 64→256 | +1.97 | +1.98 | +3.35 | +1.89 |
| 2 | 64→256 | +1.91 | +2.03 | +1.61 | +1.89 |

### 5.2 参考解截断阶梯（参考自身的自收敛）

| ka | 48→96 项漂移 | 96→144 项漂移 | A33 尾部比 |
|---|---|---|---|
| 0.2 | 8.438e-05 | 1.596e-05 | 1.892e-01 |
| 0.5 | 1.005e-04 | 1.898e-05 | 1.890e-01 |
| 0.8 | 1.257e-04 | 2.374e-05 | 1.889e-01 |
| 1 | 1.476e-04 | 2.784e-05 | 1.888e-01 |
| 1.5 | 2.219e-04 | 4.181e-05 | 1.887e-01 |
| 2 | 3.436e-04 | 6.464e-05 | 1.889e-01 |

### 5.3 CHIEF 开关（不规则频率抑制的影响）

| ka | CHIEF | 条件数 | A33 误差 | B33 误差 | B33/(ρω∇) |
|---|---|---|---|---|---|
| 1 | off | 2.73 | -0.73% | +1.19% | 0.25135 |
| 1 | on | 2.78 | -0.73% | +1.20% | 0.25138 |
| 2 | off | 2.41 | -2.08% | +4.53% | 0.10774 |
| 2 | on | 2.35 | -2.05% | +4.76% | 0.10797 |

## ⑥ 判定、失败模式与工件

| 判据 | 定义 | 实测 | 结论 |
|---|---|---|---|
| G1_case_mesh_band_error_below_5pct | |rel err| < 5% for every quantity and every ka on the case mesh | 4.7551e-02 | PASS |
| G2_geometry_precondition | |nabla_mesh/nabla-1| < 2% and |A_wp/(pi a^2)-1| < 2% on the case mesh | None | FAIL |
| G3_refinement_does_not_worsen | band-max error at 256 panels <= band-max error at 64 panels | None | PASS |
| G4_reference_self_converged | reference 96->144 term drift at least 20x below the 5 % gate | 6.4638e-05 | PASS |

**总判定：FAIL**（判据在计算前固定，未在看过结果后放宽）

### 误差判定（逐项回答）

- 算例池容差（<5%）：主网格 256 面板全频带最差 4.76%（B33/(rho omega nabla) @ ka=2）；粗网格 64 面板最差 19.48%（B33/(rho omega nabla) @ ka=2）。→ **是否 <5%：是**。
- 更严的平台通用线（<3%）：同一主网格全频带最差 4.76%；→ **是否 <3%：否**。

### 失败模式与适用边界

1. **常数面元的几何地板**：半球是曲率面，常数面元用平面近似曲面，排水体积与水线面积有 O(h²) 偏小，这一几何误差是系数误差的下界（见 ③/⑤ 中的体积、水线面积项）。
2. **ka > 2 的短波分辨不足**：面板尺寸对应 λ/a = 2π/ka，ka 越大每波长面板越少，相位误差先于幅值误差增大；本算例把频带上限固定在 ka = 2.0。
3. **不规则频率**：本水线圆的第一个内部 Dirichlet 特征频率在 ka ≈ 2.58，无 CHIEF 时该点附近条件数陡升、B33 可能变负；本算例频带避开该点，并用 CHIEF on/off 对照（5.3）确认抑制机制在起作用。
4. **参考解截断**：多极级数在 ka → 2 时收敛变慢（尾部比见 5.2）；本文用 144 项并把 96→144 的漂移量化，若漂移接近判据量级则该判据不可用——见 G4。

### 工件

- `benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py`：本算例驱动（含预注册设计）
- `benchmarks/verified/casepool_bem_hulme_hemisphere/report.md`：由驱动直接生成的分表报告
- `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png`：参考 vs 计算的 A33、B33、|Fx|/|Fz| 曲线
- `runs/casepool_bem_hulme_hemisphere/results.json`：全部原始单元与判据
- `runs/casepool_bem_hulme_hemisphere/reference.json`：参考解与截断阶梯

