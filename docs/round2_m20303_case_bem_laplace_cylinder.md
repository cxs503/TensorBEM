# CASE-BEM-LAPLACE-CYLINDER 第 2 轮（调研结合）— TensorBEM m20303

锚点 `CASE-BEM-LAPLACE-CYLINDER`，算例池 `BEM-LAPLACE-CYLINDER`，基准=解析复势
`W(z) = U(z + a^2/z)`，容差 <3%。本报告所有数字由 `results.json` /
`verify_m20303_independent.json` 程序化注入（脚本 `docs/build_round2_m20303_report.py`）。

## 1. 问题设定与基准出处

半径 $a$ 的圆柱在无界无粘无旋均匀流 $U\hat x$ 中（$Re\to\infty$ 的势流极限）。
解析基准：$W(z)=U(z+a^2/z)$，$dW/dz=U(1-a^2/z^2)$，物面速度 $2U|\sin\theta|$，
$C_p=1-4\sin^2\theta$。出处：Lamb《Hydrodynamics》6th ed. 1932、
Milne-Thomson《Theoretical Hydrodynamics》5th ed. 1968（`docs/classical_benchmarks.md`
ref.[1]/[3]）；**页码 WITHHELD**（容器内不可达原文），故基准自洽性用调和性门
（Laplacian 7.772e-08、curl 2.571e-08）与我方独立复算（探针表 vs 复势闭式，
最大差 0.0e+00 / 0.0e+00）替代。

## 2. 方法与实现

**调研（开源横向）**：`oss_inventory` 33 行全部属 TensorLBM 域（LBM 求解器），
按 `CYLINDER`/BEM 检索 **0 条**可用同类实现；`~/oss-cache` 仅 lbm/pylbm/hschwabe 三项。
同族参照改为仓内已有的 Capytaine 交叉验证测试（`tests/test_capytaine_crosscheck*.py`，
频域 Rankine/自由面 BEM）与 `docs/classical_benchmarks.md` 的古典算例谱系。

**方法谱系**：常值面元 + 配点（Hess & Smith 1962；Katz & Plotkin《Low-Speed
Aerodynamics》常强度面元影响系数为闭式）+ 外部 Green 表示。选定两条互不依赖的路线：

- 路线 D（Dirichlet）：物面给解析扰动势 $\phi'=Ua^2\cos\theta/r$，解第一类外部问题，
  再用本轮修复的 `evaluate_exterior_field`（`src/tensorbem/solver.py`）取**势与速度**；
- 路线 N（Neumann）：物面给解析 $q_e=Ua^2\cos\theta/r^2$（零法向流动），解第二类外部问题，
  比物面速度 / $C_p$ / 外部场。

几何先验证：内接正 $N$ 边形，顶点半径偏差 1.11e-16，面积/周长闭式偏差 2.83e-16 / 0.00e+00。

**本轮（第 2 轮）新增**：① 主线交付 `CASE-BEM-LAPLACE-CYLINDER` 与其新 API 已在 main 缺失
（前序内容仅存在于未并入的历史分支），本轮在 main 基线上补齐并以 20 项判据复核；
② 新增**近壁适用窗**研究：常值面板密度分片常数，近壁速度场精度未被任何判据守门，
本轮量化 $\delta/h_{max}$ 扫描、守门拒绝、饱和比，并写进 `evaluate_exterior_field` docstring。

## 3. 参数表

| 量 | 值 |
| --- | --- |
| 半径 $a$ / 来流 $U$ | 1.0 / 1.0（无量纲：全部长度按 $a$、速度按 $U$）|
| Reynolds / Froude | $\infty$ / 不适用（无自由面、无粘）|
| 面元阶梯 $N$ | [32, 64, 128]（预算 $N=128$）|
| 外部测点 | 64 点 = 环 [1.25, 1.5, 2.0, 3.0] $\times$ 16 角向 |
| 远场检验 | $r/a=8.0$ |
| 近壁扫描 | $\delta/h_{max}\in$[2.0, 1.0, 0.5, 0.25, 0.1]，三层网格 |
| 容差 | 3%（算例池判据），近壁窗 1% |
| 环境 | 3.11.6 / numpy 2.4.6 / loongarch64 |

## 4. 对照表（ref vs got）

| 关键量（$N=128$）| got | ref（解析复势）| 误差 | <3% |
| --- | --- | --- | --- | --- |
| 路线 D 外部速度 $L_2$ | — | $dW/dz$ | 1.366e-07 | 是 |
| 路线 D 扰动势 $L_2$ | — | $Ua^2\cos\theta/r$ | 3.578e-07 | 是 |
| 路线 N 物面速度 $L_2$ | — | $2U|\sin\theta|$ | 1.008e-04 | 是 |
| 路线 N 外部速度 $L_2$ | — | $dW/dz$ | 1.532e-04 | 是 |
| 路线 N $C_p$ $L_2$ | — | $1-4\sin^2\theta$ | 2.849e-04 | 是 |
| 零法向流动 $\max|v\cdot n|/U$ | 6.025e-04 | 0 | 同上 | 是 |
| $C_p$ 极小值 | -2.999194 | -3.000000 | 8.060e-04 | 是 |
| 远场 $r=8a$ 速度 | — | $dW/dz$ | 5.591e-09 | 是 |

20 项预置判据（机器复核 `--check`）：

| # | 判据 | 实测 | 阈值 | 判定 |
| --- | --- | --- | --- | --- |
| 1 | `geometry_vertices_on_circle` | 1.1102e-16 | <= 1e-12 a | PASS |
| 2 | `geometry_area_matches_inscribed_closed_form` | 2.8283e-16 | <= 1e-12 | PASS |
| 3 | `geometry_perimeter_matches_inscribed_closed_form` | 0.0000e+00 | <= 1e-12 | PASS |
| 4 | `reference_is_harmonic` | 7.7716e-08 | <= 1e-06 U/a | PASS |
| 5 | `reference_is_irrotational` | 2.5711e-08 | <= 1e-06 U/a | PASS |
| 6 | `field_gradient_matches_finite_difference` | 1.4647e-10 | <= 1e-05 | PASS |
| 7 | `dirichlet_velocity_relative_l2` | 1.3656e-07 | < 0.03 | PASS |
| 8 | `dirichlet_potential_relative_l2` | 3.5782e-07 | < 0.03 | PASS |
| 9 | `neumann_surface_velocity_relative_l2` | 1.0076e-04 | < 0.03 | PASS |
| 10 | `neumann_velocity_relative_l2` | 1.5320e-04 | < 0.03 | PASS |
| 11 | `neumann_normal_velocity_residual` | 6.0245e-04 | <= 0.01 U | PASS |
| 12 | `neumann_cp_relative_l2` | 2.8492e-04 | < 0.03 | PASS |
| 13 | `neumann_surface_velocity_observed_order` | [2.0165907313184706, 2.0065676588343466] | >= 1.5 | PASS |
| 14 | `near_wall_window_error` | 6.6090e-03 | <= 0.01 | PASS |
| 15 | `near_wall_window_error_scales_with_h` | {'observed_order_in_h': 0.9046978473113044, 'monotone_decreasing': True} | decreasing with order >= 0.5 | PASS |
| 16 | `near_wall_exclusion_zone_is_real` | 1.7567e-01 | >= 0.03 | PASS |
| 17 | `near_wall_error_is_panel_relative` | 1.0216e+00 | in [0.5, 2.0] | PASS |
| 18 | `near_wall_guard_rejects_inside_window` | True | True on every level | PASS |
| 19 | `wrong_reference_is_rejected` | 3.9523e+06 | >= 10 | PASS |
| 20 | `far_field_matches_closed_form` | 5.5909e-09 | <= 0.01 U | PASS |

## 5. 收敛性与参数敏感性

总线（相对 $L_2$，$h_{mean}$ 为尺度）：

| $N$ | $h_{mean}/a$ | D 速度 | D 势 | N 物面速度 | N 外部速度 | N $C_p$ | cond$(\frac12 I-H)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.960e-01 | 2.266e-04 | 2.334e-05 | 1.633e-03 | 2.565e-03 | 4.601e-03 | 2.103e+00 |
| 64 | 9.814e-02 | 1.229e-06 | 2.998e-06 | 4.046e-04 | 6.131e-04 | 1.143e-03 | 2.050e+00 |
| 128 | 4.908e-02 | 1.366e-07 | 3.578e-07 | 1.008e-04 | 1.532e-04 | 2.849e-04 | 2.025e+00 |

实测收敛阶：路线 N 物面 [2.0165907313184706, 2.0065676588343466]、路线 N 外部 [2.068248030773628, 2.0015453338106735]、路线 D 外部 [7.53951965123384, 3.1711289098675235]。

**近壁敏感性（第 2 轮新增，三层网格 × 五个 clearance）**，表内为最大相对速度误差：

| $N$ | $h_{max}/a$ | $\delta=2h$ | $\delta=1h$ | $\delta=0.5h$ | $\delta=0.25h$ | $\delta=0.1h$ |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.960e-01 | 6.609e-03 | 1.433e-02 | 3.689e-02 | 9.086e-02 | 1.795e-01 |
| 64 | 9.814e-02 | 3.600e-03 | 7.851e-03 | 2.696e-02 | 8.830e-02 | 1.772e-01 |
| 128 | 4.908e-02 | 1.888e-03 | 4.387e-03 | 2.212e-02 | 8.732e-02 | 1.757e-01 |

适用窗：$\delta\ge2h_{max}$ 各层误差上限 6.609e-03（≤0.01），随 $h$ 实测阶 0.90；排除区 $\delta=0.1h_{max}$ 误差 1.757e-01，粗/细网格比 1.022（加密不改善 ⇒ 常值密度表示固有误差）。

**独立复核（不 import 算例模块）**：以外部多极最小二乘（无面元、无 Green 函数，6 基函数，cond 1.002，拟合残差 1.9e-16）在同一多边形边界数据上求同一外域
Dirichlet 问题，其在所有 clearance 处的最大相对速度误差为 9.0e-15——
即圆柱解的几何/离散残差可忽略，近壁 10%–20% 误差确系**常值面板密度**所致，
不是线性求解器或几何问题。路线 D/N 指标与近壁表由独立脚本重算，与提交件差 0.0e+00 / 0.0e+00（逐位一致）。

## 6. 失败模式与工件

- **近壁失效**（已守门/已量化）：$\delta\lesssim0.1h_{max}$ 时相对速度误差达 17.6%，
  算例池测点实际从 $r/a=1.25$ 起；本轮把适用窗 $2h_{max}$、排除区 $0.1h_{max}$、
  守门拒绝（`minimum_clearance_factor`）写成 5 项判据，并写进 docstring。
- **首轮审计建议被证伪**：建议默认 clearance 取 $\sim0.05h_{max}$ 不足（该处误差仍十余 %）。
- **多面体化残差**：内接多边形 ≠ 圆，路线 N 物面误差按 $h^2$ 收敛到零；路线 D 因数据与
  多边形相容而超收敛；两条路线都只报告实测阶，不做收敛性外推。
- **势的常数不定性**：外部 Neumann 只定到常数，比较前减常数（速度不受影响）。
- 不在范围：粘性尾流/分离/自由面/三维；本算例是势流算子链的验证，不是阻力预报。

工件与复现：

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md
PYTHONPATH=src python3 docs/verify_m20303_independent.py
PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py -q
```

- `benchmarks/verified/case_bem_laplace_cylinder/results.json`（20 判据，verdict PASS）
- `benchmarks/verified/case_bem_laplace_cylinder/{case.json,report.md,figures/{field,convergence,near_wall}.png}`
- `docs/CASE-BEM-LAPLACE-CYLINDER.md`（算例文档）
- `docs/verify_m20303_independent.{py,json}`（独立复核）
- 运行环境与耗时：2026-11-14T11:44:36Z，wall 40.9 s
