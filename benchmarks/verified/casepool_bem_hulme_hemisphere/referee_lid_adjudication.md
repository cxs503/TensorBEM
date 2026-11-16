# CASE-BEM-HULME-HEMISPHERE — 用 lid 闭合的裁判重新裁定三格 `referee_unresolved`

生成：`referee_lid_adjudication.py`（辐射四量） + `referee_lid_excitation.py`（激励两量） + 本文件 `make_referee_lid_doc.py`（数字零手抄）。环境：python 3.11.6 / numpy 2.4.6 / capytaine 3.1.dev0 / loongarch64。

**总裁定：辐射臂 `FAIL`**（唯一 FAIL 是 `G2` 裁判自收敛门槛，见 §3.3 与 §5.1）；**激励臂 `PASS`**。核心问题 `G3_no_referee_unresolved` = **PASS**——三格不再是 `referee_unresolved`。

## ① 案例介绍

半径 $a=1$ m 的半球，平面浸没于平均自由面，无限水深；绕射/辐射问题由 TensorBEM 常值面元 BEM（对称性 + CHIEF）求解，无量纲化 $A/(\rho\nabla)$, $B/(\rho\omega\nabla)$, $\nabla=\tfrac{2}{3}\pi a^3$，力按 $\rho g A a^2$。参考序列为 Hulme 球谐级数（144 项）。

本案的评分带原本只到 $ka\le2$；扩展臂 `band_extension/` 把频带推到 4.0，而三方交叉验证 `radiation_capytaine_band_xval.json` 在 `ka = 2.58 / 3.0 / 4.0` 三格记 `referee_unresolved`，并因此整体 **FAIL**：capytaine 自身 1024→4096 自漂移在 2.58 达 37 %，裁判无法投票。本页把裁判换成 lid 闭合的 capytaine（水面加 Nemoh 圆盘盖，4096 湿面元 + 960 盖面元），重新给这三格一个裁定。**不改动任何既有记录**，只是补上当时缺的那一票。

| 项 | 值 |
|---|---|
| 案例容差 | 5 % / 格 |
| 待裁定格 | 2.58, 3, 4 |
| 参考 | Hulme 级数，1.0 m 半径，144 项 |
| 裁判 | capytaine 3.1.dev0，4096 湿面元，lid 开/关两态 |
| 被测 | TensorBEM 常值面元，1024（扩展带）/4096（$ka\ge3.5$）|

## ② 怎么做的

1. **不规则频率是怎么来的**：常值面元 BEM 在水面开口体上，内部 Dirichlet 问题有离散本征值；当 $ka$ 越过第一个本征值（该网格 $ka\approx2.56$，`first_irregular_wavenumber` = 4.72 对应 $k$），外部解被污染，表现为 $B_{33}$ 在 2.54（−68 %）与 2.58（+108 %）之间变号。Nemoh/CHIEF 的标准解法是用水面盖盘把内部问题闭合成非奇异问题，本页沿用同一做法。

2. **辐射四量**：不新跑任何求解器——把 `referee_robustness_part_lid_band_*.json` （lid 臂已有的 1024/4096 × 开/盖 结果）与 `band_extension/cells/*.json`、`band_edge_refinement/cells/*.json`（TensorBEM）按同一条 Hulme 参考序列对齐，形成「TensorBEM 误差 / lid 裁判误差 / 开口裁判误差 / 两码互差」四列。参考序列一致性 `reference_worst_gap` = 1.21e-14（机器断言，防止两列误差打在两条不同参考上）。输入文件 sha256 逐个人账。

3. **激励两量**：lid 臂此前只算辐射，激励通道仍由开口裁判投票——而开口裁判正是问题所在。本页补跑 `DiffractionProblem`（与水/盖同体网格，4096 + 960），总激励力 = `froude_krylov_force(problem) + DiffractionResult.forces`（capytaine ≥3 中后者只含散射力），按 $\rho g A a^2$ 无量纲化，与 Hulme 序列、TensorBEM 的 `Fx_dimless/Fz_dimless` 对比。共 5 个 $ka$ × 开/盖 = 10 次求解，单核约 1 分钟。

4. **判据先声明后算**：`G3/G4/G5`、`E1–E4` 写死在脚本的 `GATES` 字典里，先于互差数字形成；`G1/G2`（裁判精度与自收敛）在 lid 臂数据产生之后才写下，本页明确标为**事后判据**，不计入预注册检验。

## ③ 计算结果

### 3.1 辐射四量：三格的两任裁判

| ka | 角色 | TensorBEM vs Hulme | lid 裁判 vs Hulme | 开口裁判 vs Hulme | TensorBEM vs lid 裁判 | 裁定 |
|---:|---|---:|---:|---:|---:|---|
| 1 | in_band_control | 0.90 % | 1.15 % | 1.14 % | 1.65 % | `agree` |
| 2 | in_band_control | 1.13 % | 1.56 % | 1.98 % | 1.75 % | `agree` |
| 2.2 | extension_control | 1.35 % | 1.76 % | 3.79 % | 1.83 % | `agree` |
| 2.58 | unresolved_in_xval | 2.25 % | 2.06 % | 108.02 % | 1.80 % | `agree` |
| 3 | unresolved_in_xval | 2.31 % | 2.25 % | 5.81 % | 2.23 % | `agree` |
| 3.5 | extension_control | 0.68 % | 2.31 % | 3.51 % | 2.51 % | `agree` |
| 4 | unresolved_in_xval | 2.18 % | 2.26 % | 9.19 % | 1.18 % | `agree` |

（每列取四个量 $A_{11},B_{11},A_{33},B_{33}$ 的最坏值；TensorBEM 取该 $ka$ 最细可用网格，即 2.2–3.0 用 1024、3.5/4.0 用 4096。）

### 3.2 激励两量：新补的裁判票

| ka | 角色 | 开口裁判 $|F_x|$ | 开口裁判 $|F_z|$ | lid 裁判 $|F_x|$ | lid 裁判 $|F_z|$ | 互差 $|F_x|$ | 互差 $|F_z|$ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | in_band_control | 0.26 % | 1.54 % | 0.33 % | 0.19 % | — | — |
| 2.58 | unresolved_in_xval | 0.40 % | 47.13 % | 0.50 % | 1.00 % | 0.05 % | 0.05 % |
| 3 | unresolved_in_xval | 0.56 % | 2.58 % | 0.42 % | 1.40 % | 0.15 % | 0.00 % |
| 3.5 | extension_control | 1.09 % | 1.39 % | 0.07 % | 1.40 % | 0.07 % | 0.85 % |
| 4 | unresolved_in_xval | 4.40 % | 0.92 % | 0.38 % | 0.81 % | 0.91 % | 0.34 % |

$ka=2.58$ 处开口裁判的 $|F_z|$ 误差 47.13 %、加盖后 1.00 %：极点同时污染辐射与激励，且同一枚盖把两个通道一起治好。

### 3.3 裁判自身的可信度

| ka | lid 裁判 1024→4096 自漂移 | lid 相对开口的改动 |
|---:|---:|---:|
| 1 | 1.12 % | 0.38 % |
| 2 | 2.46 % | 2.57 % |
| 2.2 | 2.89 % | 4.54 % |
| 2.58 | 3.51 % | 51.61 % |
| 3 | 3.88 % | 4.76 % |
| 3.5 | 4.27 % | 2.51 % |
| 4 | 5.50 % | 8.04 % |

最坏自漂移 5.50 %（$ka=4.0$，$B_33$）——这正是 `G2` 判 FAIL 的地方：lid 裁判的 1024 档在 $ka=4$ 还不够细，所以三格一律只用它的 4096 档。作为对照，**开口**裁判的同口径自漂移是 37.42 %（`radiation_capytaine_band_xval.json` 的 `G2_referee_self_converged.worst`，机器读出）。

### 3.4 判据

| gate | criterion | measured | verdict | pre-registered? |
|---|---|---|---|---|
| `G1_lid_referee_accuracy` | worst |lid(4096) - reference| / |reference| over TARGET_KA <= 3% | `2.31 %` | **PASS** | no (post-hoc) |
| `G2_lid_self_converged` | worst |lid(4096) - lid(1024)| / |lid(4096)| over TARGET_KA <= 5% | `5.50 %` | **FAIL** | no (post-hoc) |
| `G3_no_referee_unresolved` | all of ka = 2.58, 3.0, 4.0 get a verdict other than referee_unresolved / reference_disputed | `{"2.58": "agree", "3.00": "agree", "4.00": "agree"}` | **PASS** | yes |
| `G4_solver_gate_at_best_rung` | for every cell of UNRESOLVED_CELLS, the TensorBEM error on the finest available rung is below the 5% case tolerance, and any coarser rung that breaks the gate is repaired by refinement | `{}` | **PASS** | yes |
| `G5_lid_leaves_in_band_physics_alone` | at the in-band controls ka = 1.0, 2.0: |lid - open| / |open| at 4096 panels <= 3% for all four magnitudes | `{"worst_lid_vs_open": "0.02572"}` | **PASS** | yes |

| gate | criterion | measured | verdict | pre-registered? |
|---|---|---|---|---|
| `E1_open_referee_excitation_polluted` | without the lid, the referee's |F| error at ka = 2.58 exceeds the 5% case tolerance (the irregular frequency spoils the exciting force as well as the radiation coefficients) | `{"ka": "2.58", "worst": "0.4713"}` | **PASS** | yes |
| `E2_lid_referee_excitation_accuracy` | worst lid-closed referee |F| error against the Hulme series over TARGET_KA <= 5% (the case tolerance) | `{"worst": "0.01402"}` | **PASS** | yes |
| `E3_cross_code_excitation_defined` | at ka = 2.58, 3.0, 4.0 the TensorBEM |F| agrees with the lid-closed referee within 5%, so the attribution is defined | `{}` | **PASS** | yes |
| `E4_lid_leaves_in_band_force_alone` | at ka = 2.0: |lid - open| / |open| <= 3% on both |Fx| and |Fz| | `{"worst": "0.01756"}` | **PASS** | yes |

### 3.5 三格的裁定

| ka | 原记录 | 本页裁定 | 依据 |
|---:|---|---|---|
| 2.58 | `referee_unresolved` | `agree` | TensorBEM 与 lid 裁判互差 1.80 %（$\le$ 5 %），两者对同一条 Hulme 序列的误差分别 2.25 % / 2.06 %；开口裁判此处误差 108.02 % |
| 3 | `referee_unresolved` | `agree` | TensorBEM 与 lid 裁判互差 2.23 %（$\le$ 5 %），两者对同一条 Hulme 序列的误差分别 2.31 % / 2.25 %；开口裁判此处误差 5.81 % |
| 4 | `referee_unresolved` | `agree` | TensorBEM 与 lid 裁判互差 1.18 %（$\le$ 5 %），两者对同一条 Hulme 序列的误差分别 2.18 % / 2.26 %；开口裁判此处误差 9.19 % |

## ④ 结果图

* `fig_referee_lid_error_vs_ka.png` — 左：辐射四量最坏误差 vs $ka$，开口裁判（红）/ lid 裁判（绿）/ TensorBEM（紫），灰带为原未裁定区；右：激励两量的同一比较。
* `fig_referee_lid_b33_pole.png` — $B_{33}$ 误差随 $ka$ 的扫频：开口裁判两档网格的极点（2.54→2.58 变号）与加盖后的平坦曲线。
* `fig_referee_lid_crosscode.png` — 三格上 TensorBEM 与 lid 裁判的六量互差柱状图，虚线为 5 % 容差。

## ⑤ 注意点 / 局限

1. **`G2` 是 honest FAIL，没有事后调阈值。** lid 裁判在 $ka=4$ 的 1024 档自漂移 5.5 % （$B_{33}$）超出 5 % 门槛。结论因此限定为「**4096 档** lid 裁判可用」，而不是「lid 裁判处处可用」。
2. **`G1`/`G2` 是事后判据**（lid 臂数据先于本页存在），只能作描述性证据；`G3/G4/G5` 与 `E1–E4` 在对应数字形成前已写入脚本。
3. **盖盘不是物理边界。** 它只把内部 Dirichlet 问题闭合，水面上的数学表述改变、辐射/绕射解不变；`G5`（辐射）与 `E4`（激励）在带内控制格 $ka=1.0/2.0$（辐射）与 $ka=2.0$（激励）上分别给出 2.57 % / 1.76 %，即盖盘在带内确实什么都没动。
4. **本页不重写历史记录。** `radiation_capytaine_band_xval.json` 的 `G3_attribution_defined = FAIL` 保持原样（它当时确实无法裁定）；本页是补票，不是翻案。
5. **`R1`/`R2`/`L1b` 三条探针设计问题照旧记账**（见 `referee_robustness.md` §6）：那条线的阈值与实测结构不符，不作为证据，也不在本页被悄悄放宽。
6. **$ka=4.0$ 的 TensorBEM 仍受网格限制**：1024 档 $A_{11}$ 误差 8.94 % 超容差，4096 档降到 2.18 %（`G4` 的唯一「粗档破门、细档修复」格）。2.58/3.0 目前只有 1024 档 —— 该档已在容差内（$\le2.31$ %），但**没有** 4096 档实测，因此「这两格收敛」是推断不是测量。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）

| 关键量 | 值 | 基准 | 是否 <3 % |
|---|---:|---|---|
| 三格 TensorBEM vs Hulme（最坏，最细可用网格） | 2.31 % | Hulme 144 项 | 是 |
| 三格 lid 裁判 vs Hulme（最坏，4096） | 2.31 % | Hulme 144 项 | 是 |
| 三格 TensorBEM vs lid 裁判（互差，六量最坏中辐射四量） | 2.23 % | 独立第三方码 | 是 |
| 三格 TensorBEM vs lid 裁判（互差，激励两量） | 0.91 % | 独立第三方码 | 是 |
| lid 裁判激励 vs Hulme（最坏） | 1.40 % | Hulme 144 项 | 是 |
| 带内控制格 TensorBEM vs Hulme（最坏） | 1.13 % | Hulme 144 项 | 是 |
| 开口裁判在 2.58 处 vs Hulme（对照，说明极点） | 108.02 % | Hulme 144 项 | 否（这就是要修的裁判） |

**是否 <3 %：是。** 三格上 TensorBEM、lid 裁判、两码互差、激励两量，五个关键数字全部落在 3 % 以内；唯一超过 3 % 的是**开口**裁判在极点附近的108.02 %，即本页要移除的那个伪影。另需诚实标注：lid 裁判自身的 1024→4096 自漂移在 $ka=4$ 达 5.50 %（>5 % 门槛），故裁判只用 4096 档。

---

复现：`PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/referee_lid_adjudication.py`；`PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/referee_lid_excitation.py`；本页与本目录三张 png 由 `make_referee_lid_doc.py` 重出。

