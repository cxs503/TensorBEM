# CASE-BEM-HULME-HEMISPHERE — 第 2 轮：频带扩展与不规则频率控制

*本文件由 `benchmarks/verified/casepool_bem_hulme_hemisphere/make_band_doc.py` 从 `artifacts/hulme_band_extension/results.json` 渲染；文中每个数字均为程序注入，无手抄。*

**总判定：FAIL**（判据 E1–E5，预注册于 `band_extension.py` 文件头，事后未放宽）

## ① 问题设定与基准出处

浮半球（平面与平均自由面齐平，半径 a = 1.0 m，无限水深，ρ = 1025 kg/m³，g = 9.80665 m/s²）的辐射（surge/heave，单位速度幅值）与迎浪绕射。无量纲波数 ka = k·a，时间约定 exp(−iωt)。

基准 = Hulme (1982) *J. Fluid Mech.* **121**:443–463 的球谐波-free 多极级数（`tensorbem.analytic.hulme_hemisphere_radiation`，spherical wave-free potential 满足自由面条件 ∂ψ/∂z − Kψ = 0，在半球的奇次 Legendre 谐波上投影）。该级数在 ka = 4 处的自收敛（96/144/192 项阶梯）本轮同测，见 ④。

**为什么第 2 轮要扩频带**：已入库的算例（`results.json`, case-pool id 138）把 ka 频带上限刻意停在 2，文件里写明是为了避开该水线首个不规则频率 ka ≈ 2.58。于是「<5% 达标」只在 ka ≤ 2 有证据，ka > 2 是 WITHHELD——这不是失败，但是知识缺口。本轮把缺口补成实测数据。

## ② 方法与实现

**求解侧**：深水 Kelvin/Havelock Green 函数边界积分（`tensorbem.free_surface_solver3d.assemble_free_surface_operators_3d`，常数面元；立体角 + polar/Duffy 近奇积分），内部 CHIEF 空场方程 (`chief_points='auto'`) 抑制不规则频率；辐射 `solve_radiation_3d`、绕射 `solve_diffraction_3d`。

**网格**：`geometry3d.hemisphere_hull_mesh(subdivisions=4)` = 1024 常数面元（水线顶点严格落在 z = 0）；对照网格 subdivisions=3 = 256 面元。

**归一化**：A/(ρ∇)、B/(ρω∇)，∇ = 2πa³/3 = 2.094395 m³（精确值，不用网格体积）；力按 ρgAa² 归一。

**本轮新增代码**：`benchmarks/verified/casepool_bem_hulme_hemisphere/band_extension.py`（单元列表 / 参考截断阶梯 / E1–E5 判据 / 报告与图渲染）、`tests/test_hulme_band_extension.py`（9 项，含「判据必须能翻 FAIL」的反向测试与工件复算）。

**判据（预注册）**：

| 判据 | 内容 |
|---|---|
| `E1_extension_band_error_below_5pct` | |rel err| < 5% for all six magnitudes, every ka of [2.2, 2.58, 3.0, 3.5, 4.0] on the 1024-panel mesh |
| `E2_geometry_precondition` | |nabla_mesh/nabla-1| < 2% and |A_wp/(pi a^2)-1| < 2% on the extension mesh |
| `E3_refinement_does_not_worsen_beyond_ka2` | band-max error at 1024 panels <= band-max error at 256 panels for ka = 3 and ka = 4 |
| `E4_reference_self_converged_up_to_ka4` | reference 144->192 term drift at ka = 4 at least 20x below the 5 % gate (< 2.5e-3) |
| `E5_chief_keeps_irregular_frequency_cell_inside_gate` | at ka = 2.58 the CHIEF-on cell is finite and inside 5 % on every magnitude; the CHIEF-off cell is the control |

## ③ 参数表

| 项目 | 取值 |
|---|---|
| 几何 | 浮半球 a = 1.0 m，平面与自由面齐平 |
| ∇ / A_wp | 2.094395 m³ / 3.141593 m² |
| ρ / g / A | 1025.0 kg/m³ / 9.80665 m/s² / 1.0 m |
| ka 扩展频带 | 2.2, 2.58, 3, 3.5, 4 |
| 主网格 | subdivisions 4 = 1024 常数面元 |
| 对照网格 | subdivisions 3 = 256 常数面元 |
| 参考截断阶梯 | 96 / 144 / 192 项 |
| 容差 | 5%（每个量、每个 ka） |
| 几何前置 | 体积与水线面积误差各 < 2% |
| 派生无量纲量 | λ/a = 2π/ka ∈ [1.57, 2.86]；ω√(a/g) = √ka；Froude/Reynolds 不适用（线性势流、无黏、无航速） |

## ④ 参考 vs 计算（逐项误差，基准判据逐条打勾/打叉）

### 扩展频带，1024 面元

| ka | A11 ref | got | err | B11 ref | got | err | A33 ref | got | err | B33 ref | got | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.2 | 0.22220 | 0.20423 | -8.09% ✗ | 0.31511 | 0.32041 | +1.68% ✓ | 0.39195 | 0.35580 | -9.22% ✗ | 0.08670 | 0.07321 | -15.56% ✗ |
| 2.58 | 0.18989 | 0.18839 | -0.79% ✓ | 0.26748 | 0.26780 | +0.12% ✓ | 0.40074 | 0.37558 | -6.28% ✗ | 0.06308 | 0.05336 | -15.41% ✗ |
| 3 | 0.17195 | 0.16138 | -6.15% ✗ | 0.22367 | 0.23616 | +5.58% ✗ | 0.41109 | 0.40433 | -1.64% ✓ | 0.04516 | 0.04037 | -10.61% ✗ |
| 3.5 | 0.16344 | -0.00680 | -104.16% ✗ | 0.18264 | 0.15678 | -14.16% ✗ | 0.42246 | 0.17198 | -59.29% ✗ | 0.03106 | -0.00060 | -101.95% ✗ |
| 4 | 0.16204 | 0.09818 | -39.41% ✗ | 0.15108 | 0.17517 | +15.94% ✗ | 0.43221 | 0.25954 | -39.95% ✗ | 0.02189 | 0.02811 | +28.38% ✗ |

| ka | Fx ref | got | err | Fz ref | got | err | Δphase Fx (°) | Δphase Fz (°) | cond | 墙钟 (s) |
|---|---|---|---|---|---|---|---|---|---|
| 2.2 | 1.09542 | 1.10035 | +0.45% ✓ | 0.40631 | 0.36610 | -9.90% ✗ | +2.169 | +3.746 | 3.09 | 23.1 |
| 2.58 | 0.93196 | 0.88576 | -4.96% ✓ | 0.32004 | 0.26082 | -18.50% ✗ | +2.787 | -1.140 | 8.29 | 33.6 |
| 3 | 0.79032 | 0.77785 | -1.58% ✓ | 0.25111 | 0.26986 | +7.46% ✗ | -4.844 | -3.140 | 2.73 | 56.7 |
| 3.5 | 0.66118 | 0.41849 | -36.71% ✗ | 0.19282 | 0.09193 | -52.32% ✗ | +20.168 | +18.377 | 2.78 | 92.2 |
| 4 | 0.56252 | 0.50545 | -10.15% ✗ | 0.15143 | 0.18620 | +22.97% ✗ | -7.353 | +10.981 | 14.80 | 112.5 |

### 判据裁定

| 判据 | 实测 | 裁定 |
|---|---|---|
| `E1_extension_band_error_below_5pct` | 带内最大 |rel err| = 1.0416（最差量 A11_dimless @ ka = 3.5） | FAIL |
| `E2_geometry_precondition` | 体积 0.587%、水线 0.161% | PASS |
| `E3_refinement_does_not_worsen_beyond_ka2` | ka=3: 1024 面板 10.61% vs 256 面板 22.94%; ka=4: 1024 面板 39.95% vs 256 面板 32.67% | FAIL |
| `E4_reference_self_converged_up_to_ka4` | 144→192 项漂移 7.422e-05（门限余量 134.7×） | PASS |
| `E5_chief_keeps_irregular_frequency_cell_inside_gate` | CHIEF on: 带内最大 18.50%, cond 8.29；CHIEF off: 6.12%, cond 65.32 | FAIL |

## ⑤ 收敛性与参数敏感性

### 5.1 网格阶梯（同一 ka，256 → 1024 面元）

| ka | 256 面板 带内最大 | 1024 面板 带内最大 |
|---|---|---|
| 3 | 22.94% | 10.61% |
| 4 | 32.67% | 39.95% |

### 5.2 参考截断阶梯（参考自身敏感性，扩展到 ka = 4）

| ka | 96→144 项漂移 | 144→192 项漂移 |
|---|---|---|
| 2.58 | 9.838e-05 | 3.462e-05 |
| 4 | 2.112e-04 | 7.422e-05 |

### 5.3 CHIEF 开关（跨不规则频率 ka = 2.58）

| ka | chief | cond | 带内最大 err | A33 | B33 |
|---|---|---|---|---|---|
| 2.58 | on | 8.29 | 18.50% | 0.37558 | 0.05336 |
| 2.58 | off | 65.32 | 6.12% | 0.41959 | 0.06259 |
| 3 | on | 2.73 | 10.61% | 0.40433 | 0.04037 |
| 3 | off | 2.82 | 2.24% | 0.40990 | 0.04617 |

### 5.4 两个参数点（同一网格，ka 两端）

- ka = 2.2：带内最大 15.56%，最差量 B33_dimless
- ka = 4：带内最大 39.95%，最差量 A33_dimless

## ⑥ 失败模式与工件

**失败模式**：

- 面元尺寸 vs 波长：ka 越大，1024 面元的 λ/a 覆盖越差——这是扩展频带的主误差源，判据 E3 就是它的对照臂。
- 不规则频率：ka ≈ 2.58 处算子条件数会恶化，CHIEF 空场方程是唯一的抑制手段；E5 把 on/off 两臂并列，读者可直接看到差值。
- 参考侧风险：多极级数在 ka 增大时收敛变慢，E4 用 144→192 项漂移证明「不是级数截断」。
- 共享内核：求解器与参考解都调用 `free_surface3d.wave_function_3d`；独立第三方轴为 capytaine 3.1.dev0（见 `docs/case_bem_hulme_independent_axis.md`，带内最差 3.36%）。**ka > 2 段该独立轴尚未覆盖**，是本文档明确声明的局限。

**已入库算例判据（回引，非本轮数字）**：`G1_case_mesh_band_error_below_5pct` = 0.0115；`G2` 体积 0.5870%；总判定 PASS。

**工件**：

- `benchmarks/verified/casepool_bem_hulme_hemisphere/band_extension/results.json`（cells + 参考阶梯 + 判据；`artifacts/` 被 git-ignore，故工件直接落在算例目录下）
- `.../band_extension/report.md`（程序渲染的数表）
- `.../band_extension/figure.png`（系数对照 / 误差 / 条件数）
- `.../band_extension/cells/*.json`（逐单元原始输出）
- `.../band_extension/oss_solutions.json`（方法选型与迁移方案，`feature_slug = CASE-BEM-HULME-HEMISPHERE`）

```bash
# 复现（7 个单元进程，集群递交同一命令即可）
python benchmarks/verified/casepool_bem_hulme_hemisphere/band_extension.py --out /root/tp-agent/work/m20034/benchmarks/verified/casepool_bem_hulme_hemisphere/band_extension --parallel 7
# 单元测试
PYTHONPATH=src python -m pytest tests/test_hulme_band_extension.py -q
# 重新渲染本文档
python benchmarks/verified/casepool_bem_hulme_hemisphere/make_band_doc.py \
    --results artifacts/hulme_band_extension/results.json \
    --out-doc docs/hulme_band_extension.md
```

运行环境：python 3.11.6，numpy 2.4.6，host ec6ae0b42a0c，arch loongarch64，git 56f0649d9b29db3da8c8f4170efa05593c1dc85f。
