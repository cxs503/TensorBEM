# 频域绕射/辐射与二阶漂移力：完整验证面成熟化（m14571 / p41w3-n01）

**节点** `p41w3-n01-bem-diffraction-drift-maturity` · **仓** TensorBEM ·
**分支** `agent/m14571` · **返修轮 1**（原单 m7374 经 maturity_gate 判 L0：evidence 缺 `tests`）。

本文件由 `tools/make_report_m14571.py` 从 `runs/m14571/study.json` 程序化生成，
文档内所有数字均来自该 JSON（零手抄）。生成命令见 §7。

## 0. 任务书逐项对照（机器可核）

| 任务书条目 | 交付状态 | 判据 / 证据 |
|---|---|---|
| (a) Wigley III 波浪激励力 + RAO，覆盖 6 个遭遇频率 | BEM 侧交付：6 频三路互证（近场压力积分 / 边界 Haskind / 远场 Kochin 互易） | G2/G2b/G3/G3b，§3.2–3.3 |
| (a) 与实验数据对比 <10% | **WITHHELD**：仓内无数字化实验表，不键入记忆数字 | G10（`evaluable=false`），§5.6 |
| (b) 竖直圆柱二阶漂移力 vs `(ka)^3` 渐近解析解 + Maruo 远场双式互证 | 交付：双式互差 2.8e-15；`(ka)^3` 斜率与闭式系数 | G6/G7/G7b，§3.4 |
| (b) 半径外推收敛 <3% | **不可评估**：无二阶势时有限半径通量公式无收敛率（实测散度 O(1)）；替代判据=约化 Maruo 核无半径依赖 | G9（实测）、§5.5 |
| (c) Kochin 函数能量守恒残差 <1% | 交付：近/远场辐射阻尼差、光程定理残差 | G4/G5，§3.3 |
| (d) pytest ≥ 15 全绿 | 交付：29 例全绿 | §7 |
| API `POST /api/v1/calculations/bem/diffraction`、`/bem/drift_force` | **不在本仓**：TensorBEM 无 HTTP 服务层（全仓无 fastapi/flask），入口以模块 CLI 交付；HTTP 侧属 `ship-performance-platform`，跨仓耦合如实登记为未做 | §7.1 entry |

## 1. 案例介绍（物理背景 / 几何 / 参数表）

**物理问题**：无航速、深水、线性频域。两半：

1. **一阶**：顶浪入射下 Wigley III 抛物型船体的波浪激励力 $F_3,F_5$ 与
   heave/pitch RAO。几何 $L=3.000$ m,
   $B=0.300$ m, $T=0.1875$ m，
   $C_b=4/9$（解析），排水体积 $\nabla=0.0750$ m$^3$，
   水线面 $A_{wp}=0.6000$ m$^2$。
2. **二阶**：直立圆柱的平均波浪漂移力。以 MacCamy–Fuchs 精确解（Bessel 级数）
   作为解析参照，与**离散 BEM 解**互证；`(ka)^3` 长波律对标闭式系数
   $5\pi^2/16=3.08425$。

流体：$\rho=1025.0$ kg/m$^3$，
$g=9.80665$ m/s$^2$；无量纲频率按深水色散
$k=\omega^2/g$ 换算。频率扫描 6 个遭遇频率
$\omega\in\{1.20, 1.80, 2.40, 3.00, 3.90, 4.80\}$ rad/s
（$kL=0.441\ldots7.048$，
即波长从 14.26 L 到 0.89 L），覆盖长波极限→heave 共振→短波区。

质量/恢复矩阵不用拟合：$m=\rho\nabla$，纵向惯量由船体体积分闭式
$I_{yy}=\rho TBL^3/45$ 给出（$k_{yy}=L/\sqrt{20}$ 再扣除 $z_G$ 平移），
恢复系数取 `wigley_hydrostatic_reference` 的解析 $C_{33},C_{55}$。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

* 求解器：`tensorbem.free_surface_solver3d`（Kelvin 源 + 自由面秩一修正 BEM）。
  每个频率**只装配一次算子**（`assemble_free_surface_operators_3d`），
  绕射与辐射共用，节省一次 O(N²) 装配。
* **激励力三路互证**：
  (i) 总势近场压力积分（`Diffraction3DResult.excitation_force`）；
  (ii) 辐射势上的边界型 Haskind 积分；
  (iii) 远场 Kochin 互易式 $F_j=\rho g A K_j(\beta+\pi)$。
* **Kochin 能量守恒**：近场辐射阻尼 $B_{ij}$ vs 远场 Kochin 阻尼
  （`far_field_damping_matrix_3d`）；另加光程定理残差
  $\int|K|^2d\theta = (4\pi/k)(gA/\omega)\,\mathrm{Re}\,K(0)$
  （`drift_far_field3d.optical_theorem_residual`）。
* **运动方程**：$[-\omega^2(M+A)+i\omega B+C]\xi=F$，2×2 heave/pitch 块
  （`rao_2dof`），并报告线性求解残差。
* **圆柱**：
  - 精确参照：`cylinder_drift`（MacCamy–Fuchs 精确场；近场动量积分 + 约化
    Maruo 远场；`(ka)^3` 律）。
  - 离散解：`vertical_cylinder_hull_mesh` 生成**自由面加密环**的直立圆柱
    （$z_j=-H(j/n_l)^{2.5}$，$H=16a$，故 $e^{-2kH}\le e^{-16}$ 底影响不可见），
    单次 `solve_diffraction_3d` → `bem_kochin` → `maruo_far_field_drift`
    + `radius_scan` + 光程定理。
* 机器：`pytest tests/test_diffraction_drift_maturity.py`；
  计算驱动 `python -m tensorbem.benchmarks.bem_diffraction_drift_maturity`。

## 3. 计算结果（数值表；全部由 study.json 注入）

### 3.1 几何前置门（网格必须复现 $C_b$、$\nabla$、$A_{wp}$）

| quantity | panelised mesh | closed form | relative error |
|---|---|---|---|
| panel count | 2048 | -- | -- |
| displaced volume [m^3] | 0.0749 | 0.0750 | 0.0012 |
| waterplane area [m^2] | 0.5994 | 0.6000 | 9.7656e-04 |
| block coefficient $C_b$ | 0.4439 | 0.4444 | 0.0012 |

（`WettedHullMesh` 的面元体积用中点求积，故 $\nabla$ 系统性偏低且 $O(h^2)$ 收敛：
8×4 网格 1.93%、32×16 网格 0.122%（比值 15.8≈16=4²）——本节点判据取 0.2%，
实测值始终显式报告，不用"通过"掩盖。）

### 3.2 Wigley III 顶浪：激励力与 RAO

| $\omega$ [rad/s] | $kL$ | $\lambda/L$ | $|F_3|$ [N] | $|F_5|$ [N m] | heave RAO | pitch RAO $|\theta|/(kA)$ | phase $\xi_3$/deg | phase $\xi_5$/deg |
|---|---|---|---|---|---|---|---|---|
| 1.20 | 0.441 | 14.26 | 5685.3328 | 383.2794 | 0.9945 | 0.9880 | -0.7 | -90.0 |
| 1.80 | 0.991 | 6.34 | 5213.8448 | 814.5222 | 0.9765 | 0.9768 | -3.2 | -90.0 |
| 2.40 | 1.762 | 3.57 | 4546.3538 | 1303.4566 | 0.9286 | 0.9455 | -9.0 | -90.6 |
| 3.00 | 2.753 | 2.28 | 3719.9442 | 1704.0838 | 0.8303 | 0.8787 | -18.3 | -93.1 |
| 3.90 | 4.653 | 1.35 | 2229.7972 | 1843.0539 | 0.5548 | 0.6805 | -37.1 | -105.7 |
| 4.80 | 7.048 | 0.89 | 743.4728 | 1338.3752 | 0.2078 | 0.3755 | -70.1 | -129.4 |

RAO 定义：heave $|\xi_3|/A$，pitch $|\xi_5|/(kA)$；$A=1$ m 单位入射波幅。
低频谱端 $kL=0.441$ 的 heave RAO 应为 1（质量-恢复一致性），
实测 $0.9945 $。

### 3.3 一阶互证与能量守恒（逐频率）

| $kL$ | rel diff near vs boundary Haskind | rel diff near vs Kochin reciprocity | near/far damping | optical theorem | Kochin azimuth |
|---|---|---|---|---|---|
| 0.441 | 1.4001e-05 | 2.4661e-05 | 2.2180e-05 | 1.3516e-04 | 6.3032e-06 |
| 0.991 | 1.5153e-05 | 5.3333e-05 | 4.2511e-05 | 1.0056e-04 | 1.9901e-05 |
| 1.762 | 1.6738e-05 | 1.1948e-04 | 5.8251e-05 | 3.4450e-05 | 7.7867e-05 |
| 2.753 | 1.6387e-05 | 2.4550e-04 | 6.1900e-05 | 5.7910e-05 | 1.6324e-04 |
| 4.653 | 3.3432e-05 | 5.9035e-04 | 1.0504e-04 | 9.8589e-05 | 3.4901e-04 |
| 7.048 | 4.3479e-04 | 0.0012 | 2.8301e-04 | 5.2644e-05 | 7.1457e-04 |

### 3.4 圆柱漂移力：精确 near/far 双式与 `(ka)^3` 律

| $ka$ | $F_x$ [N] (exact far field) | $F_x/(\rho g a^3 (ka)^3)$ |
|---|---|---|
| 0.020 | 0.2481 | 3.0839 |
| 0.030 | 0.8371 | 3.0832 |
| 0.050 | 3.8721 | 3.0806 |
| 0.080 | 15.8216 | 3.0732 |
| 0.100 | 30.8252 | 3.0656 |
| 0.150 | 103.0450 | 3.0364 |
| 0.200 | 240.5907 | 2.9909 |
| 0.300 | 771.5428 | 2.8419 |

拟合斜率 $2.9768$（目标 $3\pm0.2$），
$ka=0.020$ 处系数误差
$1.1682e-04$（对标 $5\pi^2/16$）。

### 3.5 圆柱漂移力：离散 BEM vs 精确解

| $ka$ | exact near field [N] | exact far field [N] | near/far rel diff | BEM [N] | BEM vs exact | control-radius spread |
|---|---|---|---|---|---|---|
| 0.50 | 2874.6750 | 2874.6750 | 4.7457e-16 | 2738.4494 | 0.0474 | 11.6034 |
| 1.00 | 6683.7103 | 6683.7103 | 5.4431e-16 | 6329.9619 | 0.0529 | 4.1489 |
| 2.00 | 6304.2445 | 6304.2445 | 1.0099e-15 | 5691.9850 | 0.0971 | 2.3203 |

### 3.6 网格收敛（$ka=1.00$）

| total panels | $F_x$ [N] | rel error vs exact |
|---|---|---|
| 684 | 6327.3465 | 0.0533 |
| 912 | 6329.9619 | 0.0529 |

## 4. 结果图

| 图 | 内容 |
|---|---|
| `fig_wigley_rao.png` | heave/pitch RAO vs $\omega$（实验叠图 **WITHHELD**，见 §5） |
| `fig_wigley_routes.png` | 三路激励力互差、近/远场阻尼差、光程定理残差（对数纵轴） |
| `fig_cylinder_law.png` | 归一化漂移力 vs $ka$ 与闭式 $5\pi^2/16$ |
| `fig_cylinder_drift.png` | 精确近场/远场与 BEM 漂移力 vs $ka$ |
| `fig_radius_scan.png` | 控制圆半径扫描（BEM Kochin）相对偏差 |

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **Kochin 幅值约定是跨模块陷阱（本节点首次给出数值桥）**：`cylinder_drift`
   的 `kochin_3d` 建立在无量纲散射幅值上，省略了入射势幅值因子 $gA/\omega$
   （`drift_far_field3d.AMPLITUDE_NORMALISATION` 的警告）。因力对幅值二次，
   直接喂 `maruo_far_field_drift` 会低报 $g/k$ 倍。本节点把它做成可执行断言
   （`test_maruo_reduced_kernel_amplitude_convention`，$ka=0.5/1/2$ 相对误差 <1e-9），
   并把解析推导写进 §6.2，避免下一个人再花一轮。
2. **`wigley_hydrostatic_reference` 的 C44/C55 键与 roll/pitch 约定互换（本节点实测放大 64 倍）**：
   该函数把 $\rho g\nabla GM_L$ 放在 `C44`、$\rho g\nabla GM_T$ 放在 `C55`，
   而 `compute_hydrostatic_restoring_matrix` 的矩阵（`tests/test_wigley_hydrostatics.py` 钉住）
   是 roll↔$GM_T$、pitch↔$GM_L$。直接用它的 `C55` 当纵摇恢复，等于用 $GM_T$ 代替 $GM_L$
   （本例比值 $56.1$），
   纵摇 RAO 被放大到 $0.9880$；改用 $\rho g\nabla GM_L$
   （$2731.6596$ N·m/rad）后回到
   $0.9880$，
   长波极限 $\to 1$（船体跟随波面斜率）。**这是本节点发现并绕开的一个真实缺陷**：
   `wigley_hydrostatics` 未改动（其键名被既有测试钉住），修复方式是在使用方按物理量取值，
   并加 `test_pitch_restoring_uses_the_longitudinal_metacentric_height` 钉死。
3. **RAO 的参考点必须是水线原点**：BEM 的 $A,B$ 与激励力矩、以及 $C55$ 都以水线原点为参考；
   刚体块因此取 $I_{55}=\rho TBL^3/45$（原点纵摇惯量），而不是 $I_{55}-mz_G^2$
   （重心纵摇惯量）。二者差 $(z_G/(L/\sqrt{20}))^2=0.0195$，
   取错会把纵摇方程的系统性偏差带进 RAO（本节点由 `tools/recompute_rao_m14571.py` 从已存
   $A,B,F$ 重算，未重跑 BEM）。
4. **光程定理的相位约定**：`optical_theorem_residual` 用 $\mathrm{Re}\,K(0)$，
   这是 BEM Kochin 的相位约定；MacCamy–Fuchs 的 2D 散射幅值约定（带 $-i$）
   必须与 $\mathrm{Im}\,K(0)$ 收缩，否则一个**精确**场会给出 20%~150% 的
   "残差"（`test_optical_theorem_uses_imaginary_forward_amplitude_for_this_convention`）。
   本节点的所有光程定理判据都在 BEM 约定下评估。
5. **控制圆半径外推在无二阶势时不可评估**：`radius_scan` 测得的散度
   $\sim O(1)$（本例 $11.6034$），
   与 `drift_far_field3d.RADIUS_SCAN_NOTE` 的推导一致：有限半径辐射应力公式
   漏掉的竖直柱面压力/动量通量与被求的漂移力同阶，$\varphi_2$ 缺失使
   $R\to\infty$ 的 Richardson 外推没有物理收敛率。**任务书 (b) 的
   "半径外推 <3%" 由此判为不可评估（honest not-evaluable）**；替代判据是
   **约化 Maruo 核无半径依赖**（同一 `ka` 下残余 <1e-14），已作为 G6 判据。
6. **Wigley 实验对标 WITHHELD**：本仓（及本生态）无任何数字化拖曳水池 RAO/附加质量
   数据表。凭记忆键入实验数字=造假，故任务书 (a) 的"vs 实验 <10%"如实登记
   `evaluable=false`。已交付的是**可执行的对标框架**：把一份带出处的
   $\omega$–RAO 表放进 `data/` 即可把 G10 从 WITHHELD 变为实评。
7. **面元体积偏低**：`WettedHullMesh.volume` 用面元中点求积，$C_b$ 系统性偏低
   （8×4 网格 1.9%，32×16 网格见 §3.1），是几何离散误差不是物理误差。
8. 深水、无航速、固定/系泊体；不含二阶势、不含前向速度相关的
   波阻/漂移力修正；$ka>2$ 短波区精确解与长波律相差数十倍，长波律只适用于
   $ka\lesssim0.1$。

## 6. 误差判定（关键量 vs 基准，逐项）

### 6.1 判据表（机器裁）

| gate | threshold | measured | passed | evaluable |
|---|---|---|---|---|
| `G1_geometry` | Cb rel err <= 2e-3, volume rel err <= 2e-3 | 0.0012 | yes | yes |
| `G2_excitation` | < 5% | 0.0012 | yes | yes |
| `G2b_excitation_boundary` | < 5% | 4.3479e-04 | yes | yes |
| `G3_rao_lowfreq` | |RAO-1| <= 0.05 | 0.0055 | yes | yes |
| `G3b_rao_pitch_lowfreq` | |RAO-1| <= 0.10 | 0.0120 | yes | yes |
| `G3c_rao_residual` | < 1e-10 | 1.9822e-16 | yes | yes |
| `G4_damping_energy` | < 5% | 2.8301e-04 | yes | yes |
| `G5_optical` | < 1% | 1.3516e-04 | yes | yes |
| `G5b_kochin_azimuth` | < 5% | 7.1457e-04 | yes | yes |
| `G10_wigley_vs_experiment` | < 10% amplitude error (brief item a) | n/a | no | no |
| `G6_drift_two_route` | < 5% | 2.8352e-15 | yes | yes |
| `G7_drift_law_slope` | 3 +- 0.2 | 2.9768 | yes | yes |
| `G7b_drift_law_coefficient` | < 3% | 1.1682e-04 | yes | yes |
| `G8_drift_bem` | < 10% | 0.0971 | yes | yes |
| `G9_radius_scan` | < 3% (brief item b) | 11.6034 | no | no |
| `G9b_optical_bem` | < 1% | 0.0293 | no | yes |
| `G9c_energy_radius` | < 1% | 1.7436e-15 | yes | yes |
| `G9d_energy_radius_independence` | < 1e-3 | 8.7180e-16 | yes | yes |
| `G8b_panel_convergence` | error(max panels) <= error(min panels) | 0.0529 | yes | yes |

**未通过 / 不可评估判据（honest，逐条列出，不覆盖）：**

* `G10_wigley_vs_experiment`（阈值 < 10% amplitude error (brief item a)）：实测 n/a，**不可评估**。原因：No digitised experimental RAO/added-mass table ships with this repository; typing values from memory would be fabrication. Registered WITHHELD; the BEM side of the comparison is delivered and internally cross-verified.
* `G9_radius_scan`（阈值 < 3% (brief item b)）：实测 11.6034，**不可评估**。原因：Measured spread 11.6: the finite-radius radiation-stress route omits the vertical control-cylinder pressure/momentum flux, which is the same order as the drift force itself, so no power-law extrapolation exists (drift_far_field3d.RADIUS_SCAN_NOTE, m9286). Reported as NOT EVALUABLE, not as a pass; the radius-free reduced Maruo kernel is used instead (G6).
* `G9b_optical_bem`（阈值 < 1%）：实测 0.0293，**未通过**

### 6.2 关于"是否 <3%"

* **`(ka)^3` 系数（对标闭式 $5\pi^2/16$）**：误差
  $1.1682e-04 → **<3% 达标**。
* **圆柱近场 vs 远场双式**：最大相对差
  $2.8352e-15 → **<3% 达标**（实为机器精度）。
* **BEM 漂移力 vs 精确解**：最差
  $0.0971
  → 见判据 G8（阈值 10%，短波 $ka=2$ 受方位分辨率限制）。
* **一阶激励力三路互差**：
  $0.0012
  → 见判据 G2（阈值 5%）。
* **Kochin 能量守恒（近/远场阻尼）**：
  $2.8301e-04
  → 见判据 G4（阈值 5%）；光程定理残差
  $1.3516e-04
  → 见判据 G5（阈值 1%）。
* **控制圆半径外推 <3%**：**不可评估**（§5.5），不以任何拟合数字冒充达标。
* **Wigley RAO vs 实验 <10%**：**WITHHELD**（§5.6）。

### 6.3 幅值约定的解析推导（§5.1/§5.2 的依据）

MacCamy–Fuchs 远场：$F_x=\dfrac{\rho g A^2}{2\pi\nu}\int|s(\theta)|^2(1-\cos\theta)\,d\theta$，
`s` 为无量纲散射幅值。约化 Maruo：$F_x=\dfrac{\rho k^2}{8\pi}\int|K|^2(1-\cos\theta)\,d\theta$。
`kochin_3d` $K=-2is/(ka)$ 代入得 $F_{\rm maruo}=\dfrac{\rho}{2\pi a^2}\int|s|^2(1-\cos\theta)d\theta$，
两式相除 $F_{\rm exact}/F_{\rm maruo}=gA^2/k$ —— 即缺失的势幅值因子
$(gA/\omega)^2=g/k$。数值：$ka=0.5/1/2$ 处比值分别为 19.62/9.81/4.905，与 $g/k$ 完全一致。

## 7. 复现

```
# 6 个频率分片（每片一次装配 + 绕射 + 2 模态辐射），再合并
for i in 1 2 3 4 5 6; do PYTHONPATH=src python3 -m tensorbem.benchmarks.bem_diffraction_drift_maturity \
    --skip-cylinder --sections 32 --levels 16 --omegas <w_i> --json runs/m14571/shard_w$i.json; done
PYTHONPATH=src python3 -m tensorbem.benchmarks.bem_diffraction_drift_maturity \
    --skip-wigley --cylinder-ka 0.5,1.0,2.0 --cylinder-panels 36 \
    --convergence-panels 24,36 --json runs/m14571/shard_cylinder.json
PYTHONPATH=src python3 -m tensorbem.benchmarks.bem_diffraction_drift_maturity \
    --merge runs/m14571/shard_w*.json runs/m14571/shard_cylinder.json \
    --json runs/m14571/study.json --figures runs/m14571
PYTHONPATH=src python3 -m pytest tests/test_diffraction_drift_maturity.py -q
```

实测：`29 passed`（本文件全部用例）。驱动与库的加入**未引入新的回归**：
`tests/test_capability_evidence.py` 与 `tests/test_doc_claim_audit.py`
在 `HEAD`（干净检出）与本分支上各有**同样 5 条失败**（基线漂移：
13 个既有 benchmark 模块未登记 + `docs/audit_baseline.md` 生成块过期），
本节点新增的两个模块不在其列（新增用例 `test_driver_merge_recomputes_gates_from_committed_shards`
把驱动模块纳入 capability-evidence 的覆盖引用）。

### 7.1 机器可核判据与产物路径

* 判据 JSON：`runs/m14571/study.json`（`verdicts` 数组，逐条 `threshold`/`value`/`passed`/`evaluable`）
* 图：`runs/m14571/fig_*.png`
* 测试：`tests/test_diffraction_drift_maturity.py`（29 例）
* 实现：`src/tensorbem/diffraction_drift_maturity.py`、
  `src/tensorbem/benchmarks/bem_diffraction_drift_maturity.py`
* 入口：`python -m tensorbem.benchmarks.bem_diffraction_drift_maturity`（`main()`，见 §7 命令）
* 集成调用方：`solve_diffraction_3d` / `solve_radiation_3d` / `kochin_function_3d`
  （`free_surface_solver3d`，本节点新驱动真实调用）、`maruo_far_field_drift` /
  `radius_scan` / `optical_theorem_residual`（`drift_far_field3d`）、
  `near_field_cylinder_drift` / `far_field_cylinder_drift`（`cylinder_drift`）
