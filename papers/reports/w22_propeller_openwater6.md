# W22 · KP505 螺旋桨敞水性能：锚可达性回核 + 五档 BEM 对比

> 领域仓 `TensorBEM` ／ 学科域：**推进（propeller open-water）** ／ 承接零报告任务 `paper-exp-propeller-openwater-bem-1`。本报告的全部数值由工件注入：`benchmarks/verified/kp505_anchor_w22/kp505_transcription.json`、`benchmarks/verified/kp505_anchor_w22/openwater_compare.json`，驱动 commit `481be532a1c50c5ecfc1bf2b3088699003de55d7`。

## 0. 结论摘要（机器裁决）

- 锚可达性：**可达** — 命中文件 39 个，转录字段 70 个，逐文件 sha256 全部登记；provenance 类别 `SECONDARY_REPRODUCTION`。
- 五档敞水表（K_T/K_Q/η₀ × J=0.1/0.3/0.5/0.7/0.9）已落表，每档带相对偏差与参照类型。
- 设计点 J = 0.70：BEM K_T = 0.15824，锚 K_T = 0.193，**ΔK_T = -18.01 %**；门槛 ±5.0 % ⇒ **未达标（honest FAIL，已逐渠道归因）**。
- 面元密度 GCI：3 套网格（r = 2.0），观测阶与 GCI 指数在 5 档 × (K_T, K_Q) 全部落表；渐近区格数 0/10（非渐近 ⇒ GCI 只作量级指示，不作误差上界）。
- 判据裁决：A1 PASS／A2 PASS／A3 FAIL／A4 PASS（J=0.70 dKT = -18.01 % (threshold +/-5 %): HONEST FAIL; panel-density GCI ladder of 3 levels on all five J）。

## 1. 案例介绍（物理背景 / 几何 / 参数表）

对象为 KRISO 集装箱船（KCS）桨 **KP505**：五叶定距桨，敞水试验曲线是推进学科最常用的外部锚之一。本单不造新算例，而是①先机检‘锚在仓内是否可得’，②再在 J ∈ {0.1, 0.3, 0.5, 0.7, 0.9} 五档上用仓库自带的涡格升力面 BEM 算 K_T / K_Q / η₀ 并逐档与可得参照对比。

**几何与工况参数**（BEM 侧声明假设：数值取自驱动 `PRIMARY` 配置与 `BladeGeometry` 缺省值，不手抄；锚件 Table 1 的模型几何在仓内以 sha256 登记，供对照）

| 量 | 取值 | 说明 |
|---|---|---|
| 直径 D | 1.0（无量纲化约定） | BEM 内部以 D 无量纲化；锚件 Table 1 的模型直径见 `benchmarks/anchors/sources/kp505_marine2023_table1_propeller_characteristics.txt` |
| 叶数 Z | 5 | `blades`；与锚件 Table 1 一致 |
| 盘面比 A_E/A_O | 0.80 | 弦长分布按此**精确匹配**（闭式积分 `chord_scale`） |
| 毂径比 d_h/D | 0.18 | `hub_ratio` |
| 螺距比 P/D（均值） | 0.95 | 径向常数（真桨径向螺距分布未在锚件中发布，不臆造） |
| 剖面 | 零弯度平板涡格 | 锚件 Table 1 为 NACA66（有弯度）——差异进‘模型项’ |
| 厚度比 t/c | 0.06 | 进入 ITTC-78 剖面阻力律 |
| 转速 | n = 10.0 rev/s | 只经雷诺数进入阻力项；Re ∝ D²，故这是代表工况而非 0.25 m 模型实况 |
| 尾流模型 | `geometric`，长度 1.0 D，24 段（L1 级） | 螺旋尾涡离散；L2/L3 为 48/96 段 |
| 进速档位 J | 0.1、0.3、0.5、0.7、0.9 | 任务书五档 |

**外部锚与其证据级别（回核结果，非记忆值）**

- 锚 ID `AN-KP505`，类型 `experiment`，provenance `SECONDARY_REPRODUCTION`。
- 出处：Iftikhar, H. & Takinaci, A.C. (2023). Performance Prediction of KP505 Propeller Using Lifting Line Theory and RANS. X International Conference on Computational Methods in Marine Engineering (MARINE 2023).（DOI 10.23967/marine.2023.120），Table 2, page 10，表题 verbatim：“Table 2. Comparison of CFD and Experimental Results”。
- 表格原文文件：`benchmarks/anchors/sources/kp505_marine2023_p10_table2.txt`，sha256 `07a7ebabe67f98eacf96ce9fe2d2dfeb57aae13c9d5ef7ea695a963b08b2a987`，共 10 行 / 70 个数值字段。
- 证据级别判定：表在仓内**真实存在且可机读**（REAL，二级转录）；但该论文页 10 **未引用**KRISO 原始试验报告 ⇒ 对‘原始试验手稿’这一层记为 **WITHHELD**（仓内无 primary handout）。
- 解析式锚 `AN-ITTC57`：`C_F = 0.075 / (log10(Re) - 2)^2`（ITTC Recommended Procedure 7.5-02-03-01.4, 1957 friction line (AN-ITTC57 anchor register, mission W22)），类型 `analytic`。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

### 2.1 步骤 1 —— 锚可达性机检（判据 A1）

在 TensorBEM 检出内做 glob（`*kp505*`, `*KP505*`, `*kp_505*`, `*kp-505*`, `*Kp505*`）与文本检索：

```
grep -ril --include=* --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.venv --exclude-dir=node_modules --exclude-dir=runs --exclude-dir=kp505_anchor_w22 --exclude-dir=w22_kp505_cache kp505 /root/tp-agent/work/m14506
```

| 层（后缀分类） | 命中文件数 |
|---|---|
| figure | 1 |
| geometry | 1 |
| table | 12 |
| text | 25 |

合计命中 **39** 个文件（glob 去重后 27 条命中、grep 命中 36 个），其中表格层 12 个、图件层 1 个、几何层 1 个。每个命中文件都登记了 sha256（见工件 `kp505_transcription.json → reachability.hit_files`）。表文件存在性：`benchmarks/anchors/sources/kp505_marine2023_p10_table2.txt` ⇒ 存在。

表格层命中文件（前 8 条，完整清单在工件里）：

| 文件 | sha256（前 16 位） | 字节 |
|---|---|---|
| benchmarks/anchors/kp505_openwater.json | 9771a40b9a24e3f2 | 9917 |
| benchmarks/anchors/sources/kp505_marine2023_p10_table2.txt | 07a7ebabe67f98ea | 1106 |
| benchmarks/anchors/sources/kp505_marine2023_table1_propeller_characteristics.txt | b94913826b91e65f | 1311 |
| benchmarks/kp505_openwater_bem/verdict.json | c0eaf703c5979750 | 53124 |
| benchmarks/verified/bem_kp505_openwater_w20/gci.json | 7f9967068dfb0de5 | 85086 |
| benchmarks/verified/propeller_openwater/kp505_gci_ladder.json | 0ca5f86e6e97ef42 | 155774 |
| benchmarks/verified/propeller_openwater/kp505_gci_ladder_decomposition.csv | bcdedae9596edc3a | 1821 |
| benchmarks/verified/propeller_openwater/kp505_gci_ladder_gci.csv | 40b9895a9c6da6b4 | 5797 |

### 2.2 步骤 2 —— 零手抄转录（判据 A1/A2 的输入）

驱动用行文法正则从表文件中逐行解析 7 列（J, K_T^CFD, 10K_Q^CFD, K_T^exp, 10K_Q^exp, err_K_T, err_10K_Q），每个数值写一条记录 `file + sha256 + line + field + verbatim + reference_type`，共 10 行 / **70 个字段，其中 70 个带完整溯源**。派生量 K_Q = 10K_Q/10 与 η₀ = J K_T/(2π K_Q) 由已转录列**算出**，不另行抄写。

自洽性回核（用表自身的 CFD/Experimental 列重算它的误差列，按无符号口径比较 20 处）：最大偏差 **1.178 个百分点**（最差格 J = 1.0 的 KT：印 2.27 % vs 重算 3.45 %）。该偏差要按**逐行传播的舍入界**判读（表给 3 位小数，最小的 K_T = 0.029 本身已带 ±1.7 % 相对舍入）：最紧一格的 ratio_to_bound = 0.497 ≤ 1 ⇒ 转录忠实、偏差可由印刷精度完全解释。

与仓内已钉住的锚表交叉核对：`cross_check_pinned_table_mismatches` = 0 条 ⇒ 本次转录与既有 `ANCHOR` 表逐位一致。

### 2.3 步骤 3 —— 参照体系（三级标注）

| 参照 | 类型标注 | 来源 | 在本研究中的角色 |
|---|---|---|---|
| KP505 敞水试验曲线 | `experiment`（二级转录） | `benchmarks/anchors/sources/kp505_marine2023_p10_table2.txt` (sha256 07a7ebabe67f98ea…) | 主参照：五档 K_T/K_Q/η₀ 的偏差基数 |
| AN-ITTC57 摩擦线 | `analytic` | `C_F = 0.075 / (log10(Re) - 2)^2` | 粘性通道交叉核对（与本仓 ITTC-78 剖面阻力律对比） |
| 动量理论（作动盘）效率上界 | `analytic` | 涡格解出的 C_T ⇒ η_ideal = 2/(1+√(1+C_T)) | 物理可容许性门（η₀ 不得越界） |

### 2.4 步骤 4 —— BEM 求解与面元密度梯度（判据 A4）

求解器：`tensorbem.propeller.open_water_point`（涡格升力面 + 螺旋尾涡 + Kutta-Joukowski 力 + ITTC-78 剖面阻力）。梯度设计：**三条面元轴同时加密、r = 2**，涡核正则化 `core_ratio` 固定，因此梯度只测离散、不测正则化：

| 级别 | n_radial | n_chord | 尾涡段 | 面板/叶 |
|---|---|---|---|---|
| L1 | 8 | 2 | 24 | 16 |
| L2 | 16 | 4 | 48 | 64 |
| L3 | 32 | 8 | 96 | 256 |

GCI 用 Roache 三点法（`tensorbem.uq.grid_convergence_index`，安全因子 1.25）：p = ln|(f₂−f₁)/(f₃−f₂)|/ln r，GCI_fine = 1.25|(f₂−f₁)/f₁|/|r^p−1|。理论阶取 p_th = 2（面板中点求积对光滑载荷二阶；桨尖 √(1−r) 载荷会把它拉向 1.5，故只把 p_th = 2 当作**声明值**、把观测阶原样报出）。

设计点归因按**动手前声明的三条渠道**分开计：①面元密度（Richardson 外推 vs L3）、②尾流模型（`geometric`／`kinematic`／`induced` 三种，同网格 L3）、③粘性修正（`viscous` 开/关，同网格 L3）。

工具链：Python 3.11.6 / numpy 2.4.6 / loongarch64；复跑命令 `python -m tensorbem.benchmarks.kp505_w22 benchmarks/verified/kp505_anchor_w22`（`--workers` 并行逐 J 求解）。

## 3. 计算结果（全部数字由工件注入）

### 3.1 五档敞水表（判据 A2）

参照标 `E` = experiment（KP505 敞水试验曲线，二级转录）。BEM 取最细网格 L3。

| J | K_T(BEM) | K_T(E) | ΔK_T | K_Q(BEM) | K_Q(E) | ΔK_Q | η₀(BEM) | η₀(E) | Δη₀ | 参照类型 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.50166 | 0.494 | +1.55 % | 0.06571 | 0.0683 | -3.79 % | 0.1215 | 0.1151 | +5.55 % | E |
| 0.3 | 0.39503 | 0.392 | +0.77 % | 0.05567 | 0.0585 | -4.84 % | 0.3388 | 0.3199 | +5.90 % | E |
| 0.5 | 0.28056 | 0.290 | -3.26 % | 0.04292 | 0.0471 | -8.88 % | 0.5202 | 0.4900 | +6.17 % | E |
| 0.7 | 0.15824 | 0.193 | -18.01 % | 0.02746 | 0.0342 | -19.69 % | 0.6419 | 0.6287 | +2.10 % | E |
| 0.9 | 0.02809 | 0.087 | -67.71 % | 0.00931 | 0.0192 | -51.53 % | 0.4324 | 0.6491 | -33.38 % | E |

### 3.2 面元密度 GCI（判据 A4）

`asymptotic` = |p − p_th| ≤ 30 %·p_th（p_th = 2）。`GCI_fine` 为细网格指数（%）。

| J | 量 | L1(8×2) | L2(16×4) | L3(32×8) | p_obs | p 偏差 | GCI_fine | 渐近区 | 振荡 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | KT | 0.54256 | 0.53996 | 0.50166 | -3.881 | -294.1 % | 10.24 % | 否 | 否 |
| 0.1 | KQ | 0.06296 | 0.06573 | 0.06571 | +7.206 | +260.3 % | 0.00 % | 否 | 是 |
| 0.3 | KT | 0.42659 | 0.42476 | 0.39503 | -4.018 | -300.9 % | 10.03 % | 否 | 否 |
| 0.3 | KQ | 0.05497 | 0.05666 | 0.05567 | +0.772 | -61.4 % | 3.15 % | 否 | 是 |
| 0.5 | KT | 0.30272 | 0.30150 | 0.28056 | -4.098 | -304.9 % | 9.91 % | 否 | 否 |
| 0.5 | KQ | 0.04332 | 0.04424 | 0.04292 | -0.512 | -125.6 % | 12.90 % | 否 | 是 |
| 0.7 | KT | 0.17094 | 0.17018 | 0.15824 | -3.967 | -298.3 % | 10.07 % | 否 | 否 |
| 0.7 | KQ | 0.02801 | 0.02847 | 0.02746 | -1.125 | -156.3 % | 8.47 % | 否 | 是 |
| 0.9 | KT | 0.03126 | 0.03081 | 0.02809 | -2.576 | -228.8 % | 14.54 % | 否 | 否 |
| 0.9 | KQ | 0.00905 | 0.00935 | 0.00931 | +2.770 | +38.5 % | 0.10 % | 否 | 是 |

渐近区格数 0/10。观测阶出现负值的格子里三级增量符号反复，该离散**未进入渐近区**，因此 GCI 只能当误差量级指示，不能当误差上界——本报告按此口径引用，不放宽判据。

### 3.3 设计点 J = 0.70 的偏差归因（判据 A3）

**渠道①面元密度**（Richardson 外推，仅当 p > 0 且单调时视为可用）：

| 量 | L3 | 参照 | 总偏差 | p_obs | Richardson | 外推值 | 离散项 | 末次增量 | 模型项（区间口径） |
|---|---|---|---|---|---|---|---|---|---|
| KT | 0.15824 | 0.1930 | -18.01 % | -3.967 | 不可用 | 0.17100 | -7.46 % | 7.54 % | [-33.1, -2.9] |
| KQ | 0.02746 | 0.0342 | -19.69 % | -1.125 | 不可用 | 0.02933 | -6.35 % | 3.67 % | [-27.0, -12.4] |
| eta0 | 0.64190 | 0.6287 | +2.10 % | -0.774 | 不可用 | 0.69967 | -8.26 % | 3.74 % | [-5.4, +9.6] |

**渠道②尾流模型**（同网格 L3，J = 0.70）：

| 尾流模型 | K_T | ΔK_T | 10K_Q | Δ10K_Q | η₀ | Δη₀ |
|---|---|---|---|---|---|---|
| geometric | 0.15824 | -18.01 % | 0.27465 | -19.69 % | 0.6419 | +2.10 % |
| kinematic | 0.13804 | -28.48 % | 0.24397 | -28.66 % | 0.6304 | +0.27 % |
| induced | 0.18927 | -1.93 % | 0.30980 | -9.42 % | 0.6806 | +8.26 % |

**渠道③粘性修正**（同网格 L3，J = 0.70）：

| 配置 | K_T | ΔK_T | 10K_Q | Δ10K_Q |
|---|---|---|---|---|
| viscous = True（主核） | 0.15824 | -18.01 % | 0.27465 | -19.69 % |
| viscous = False | 0.16322 | -15.43 % | 0.23469 | -31.38 % |

粘性通道贡献（占 ΔK_T 的百分点）：-2.58 个百分点；K_T 的粘性增量 0.00498，10K_Q 的粘性增量 -0.03996。

**解析锚交叉核对（AN-ITTC57 vs 仓内 ITTC-78 剖面阻力律，判据外的附加证据）**

| 量 | 值 |
|---|---|
| 代表半径 r/R | 0.597 |
| 代表弦长 [m] | 0.3263 |
| 代表雷诺数 Re | 3.350e+06 |
| C_F（ITTC-1957） | 0.003663 |
| C_F（ITTC-78 剖面律） | 0.003374 |
| 比值 ITTC-78 / ITTC-57 | 0.9211 |

说明：the lattice's section-drag law is the ITTC-78 form 2(1+2t/c)(0.044/Re^(1/6) - 5/Re^(2/3)); AN-ITTC57 gives the skin-friction level to check it against.  This is a formula evaluation (analytic), not test data.

**动量理论上界检查（η₀ 不得越界）**

| J | η₀(BEM) | η_ideal（作动盘，analytic） | 越界？ |
|---|---|---|---|
| 0.1 | 0.1215 | 0.1620 | 否 |
| 0.3 | 0.3388 | 0.4455 | 否 |
| 0.5 | 0.5202 | 0.6747 | 否 |
| 0.7 | 0.6419 | 0.8511 | 否 |
| 0.9 | 0.4324 | 0.9788 | 否 |

## 4. 结果图

**`benchmarks/verified/kp505_anchor_w22/figures/kp505_curves.png`** — K_T, 10K_Q and eta_0 vs J: BEM ladder vs the transcribed KP505 anchor.

**`benchmarks/verified/kp505_anchor_w22/figures/kp505_deviations.png`** — Relative deviation of K_T, K_Q and eta_0 from the anchor; the predeclared +/-5 % design-point band is drawn.

**`benchmarks/verified/kp505_anchor_w22/figures/kp505_panel_ladder.png`** — K_T and K_Q against panels per blade on the three-level ladder for every mission advance ratio.

**`benchmarks/verified/kp505_anchor_w22/figures/kp505_gci.png`** — Observed order p and fine-grid GCI per (J, quantity); negative p marks a non-asymptotic triple.

**`benchmarks/verified/kp505_anchor_w22/figures/kp505_attribution.png`** — Design-point K_T miss split into panel-density (Richardson) and model contributions.

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **锚是二级转录**：Table 2 的 Experimental 列在仓内可机读、可哈希、可逐字段溯源，但论文页 10 没有引用 KRISO 原始试验手稿。因此‘试验锚’这一层对原始手稿是 WITHHELD；本报告所有偏差都继承一个表格转录步骤，不能宣称比表本身的印刷精度更准（表给 3 位小数）。
2. **工作树导入遮蔽（本次实打实踩到）**：本机 `sys.path` 里注入了其它 worktree 的 `src/`（`.pth` 机制），直接 `python -m tensorbem...` 会静默导入**别的仓副本**。本单所有运行都强制 `PYTHONPATH=<本工作树>/src`，并在报告里记 commit，避免‘跑的是别人的代码’。
3. **BEM 是升力面近似，不是 RANS**：零弯度平板涡格、径向常数螺距、无侧斜/纵倾、无毂与剖面厚度堵塞，弦长分布只保证 A_E/A_O 与根梢收缩形态。凡是依赖剖面弯度与真实径向螺距分布的差异，都会落入‘模型项’，不可能靠加密网格消掉。
4. **观测阶为负 = 未进渐近区**：本梯度在多数格子给出 p < 0，说明 L1→L3 的增量方向反复。此时 Richardson 外推值不可当物理极限，本报告只用末次增量作区间估计，并把该事实写进工件 `richardson_applicable` 字段。
5. **运行时与并行**：L3（32×8 面板/叶、96 尾涡段）单次约 3 min（本机高负载、load≈700 下实测）；五档 + 归因通道用进程池并行，整体 10 min 量级。**并行必须用 `spawn`**：本机在父进程已起 OpenBLAS 线程池（130 线程）后 `fork`，子进程会继承一把没人解锁的 futex，实测子进程 0 s CPU 永久睡眠——表现为‘跑了半小时一个字都没写’。
6. **缓存与可复现**：晶格解按 (J, 梯度级, 模型开关, `propeller.py`+`uq.py` 的 sha256) 缓存在 `runs/w22_kp505_cache/`，命中时逐位复现上一轮同指纹结果（本次重出裁决时 18/19 格来自缓存，其中唯一需要重算的粘性关闭工况重算结果与上轮逐位相同）。工件里每个格子都带 `cache_hit` 标记，不做暗缓存。
7. **本单边界**：只产实验工件与结论，不写论文正文；不触碰 TensorPlatform 仓；所有工件路径都在 TensorBEM 仓内。

## 6. 误差判定

### 6.1 判据逐条裁决表（阈值先于动手锁定，未事后修改）

| 判据 | 量值 | 阈值 | 裁决 | 证据指针 |
|---|---|---|---|---|
| A1-anchor-reachability-machine-decided | 命中文件 39，字段 70（带 sha256 39） | n_hit_files > 0, n_fields > 0, every hit file sha256'd | **PASS** | `openwater_compare.json` → criteria[A1-anchor-reachability-machine-decided] |
| A2-five-J-table-with-deviation-and-reference-type | 15/15 格完整 | 15/15 cells finite with a declared reference type | **PASS** | `openwater_compare.json` → criteria[A2-five-J-table-with-deviation-and-reference-type] |
| A3-design-point-KT-within-5pct | J=0.70 ΔK_T = -18.01 %（K_T 0.15824 vs 0.193） | |dK_T| < 5.0 % | **FAIL** | `openwater_compare.json` → criteria[A3-design-point-KT-within-5pct] |
| A4-panel-density-gci | 3 套网格，10 格，渐近 0 格 | >= 2 levels, 10/10 (J, quantity) cells tabulated, asymptotic flag present | **PASS** | `openwater_compare.json` → criteria[A4-panel-density-gci] |

A3 的详细口径：`within_threshold` = False、`honest_fail` = True、`attribution_available` = True、`disposition` = `HONEST_FAIL_WITH_ATTRIBUTION`。**`pass` 只表示精度阈值是否达到，绝不用‘已归因’把红格刷绿**；任务级处置 = `HONEST_FAIL_WITH_ATTRIBUTION`。

### 6.2 关键量 vs 基准，逐项回答「是否 < 3 %」

| 量（J = 0.70） | BEM(L3) | 实验锚 | 相对偏差 | 是否 < 3 % | 备注 |
|---|---|---|---|---|---|
| K_T | 0.15824 | 0.193 | -18.01 % | **否** | 任务门槛为 <5 %，仍未达 |
| K_Q | 0.02746 | 0.0342 | -19.69 % | **否** | 同上口径 |
| η₀ | 0.6419 | 0.6287 | +2.10 % | **是** | 派生量；分子分母同向偏小故相消（不构成达标证据） |

**显式回答**：设计点 J = 0.70 的 |ΔK_T| = 18.01 % —— **不小于 3 %，也不小于本单的 5 % 门槛**；|ΔK_Q| = 19.69 %，|Δη₀| = 2.10 %。因此判据 3 按规则记为 **honest FAIL**，不粉饰、不改判据。

**归因结论（三渠道，均为工件数字）**：

- **尾流模型（主渠道）**：`geometric` / `kinematic` / `induced` 的 ΔK_T 分别为 geometric -18.01 % / kinematic -28.48 % / induced -1.93 %，极差 **26.54 个百分点** ⇒ 尾流离散方式是本模型最大的单一不确定性，比面元密度与粘性项都大。
- **面元密度（次渠道）**：观测阶 p = -3.967，Richardson **不可用**（p<0，外推值落在加密区间之外，不可作物理极限）；L3 与该项目外推值之差 -7.46 个百分点只是量级指示，可用的离散不确定度是末次（L2→L3）增量 7.54 %。即：即使把离散不确定度整块挪走，也补不平 -18.01 个百分点的总偏差 ⇒ 离散不是主因。
- **粘性修正（小项）**：开关粘性使 ΔK_T 变化 -2.58 个百分点（K_T 粘性增量 0.00498、10K_Q -0.03996）⇒ 粘性不是主因。
- **余项（几何/载荷保真度）**：扣掉上述三项后剩 -10.55 个百分点（区间口径 [-33.1, -2.9] %），只能归到**零弯度平板剖面**（真桨 NACA66 有弯度）、**径向常数螺距**（真桨 0.6R 附近有峰）、**无毂与剖面厚度堵塞**、无侧斜/纵倾。
- **诚实的边界（不能拿 induced 翻案）**：`induced`（力自由尾涡对齐）在 K_T 上为 -1.93 %，**已进 ±5 % 门槛**；但（i）它不是本单预声明的主核，换成它属于‘看见锚之后再选模型’的事后选择；（ii）它同时把 Δ10K_Q 改到 -9.42 %、Δη₀ 改到 +8.26 %，不是一致性改进。故判据 3 仍判 **FAIL**，induced 只作敏感性结果登记，不作主结果。

### 6.3 复现与工件清单

```
benchmarks/verified/kp505_anchor_w22/kp505_transcription.json   # 可达性 + 转录（锚字段级溯源）
benchmarks/verified/kp505_anchor_w22/openwater_compare.json         # 五档敞水表 + GCI + 归因 + 判据裁决
benchmarks/verified/kp505_anchor_w22/points.csv
benchmarks/verified/kp505_anchor_w22/gci.csv
benchmarks/verified/kp505_anchor_w22/figures/*.png
src/tensorbem/benchmarks/kp505_w22.py    # 驱动（判据先声明后跑）
tools/emit_w22_report.py                 # 本报告生成器（零手抄）
tests/test_w22_kp505_anchor.py           # 12 tests
```

```bash
PYTHONPATH=$PWD/src python -m tensorbem.benchmarks.kp505_w22 benchmarks/verified/kp505_anchor_w22
PYTHONPATH=$PWD/src python tools/emit_w22_report.py
PYTHONPATH=$PWD/src python -m pytest tests/test_w22_kp505_anchor.py -q
```

工件 `kp505_transcription.json` sha256 = `25421d1ad115b937e9a77158b70536f0434e72383e87a617fca217290383cf4f`；`openwater_compare.json` sha256 = `ee7e410d77490af467a1ac0f474565a53a02ae701837d52bafd441a4ca407310`。

## 7. 平台投递摘要（与 report API 的 summary 同一段文本，机器生成）

> KP505 敞水试验锚在本仓可达（10 行 / 70 字段逐字段带 file+sha256+line+verbatim，provenance = SECONDARY_REPRODUCTION；原始手稿层 WITHHELD）。BEM 在 J=0.1~0.9 五档给出 K_T/K_Q/η₀，设计点 J=0.70 的 ΔK_T = -18.01 %（门槛 ±5 %）⇒ 判据 3 为 honest FAIL。归因：尾流模型是主渠道（三模型 ΔK_T 极差 26.54 个百分点，力自由 induced 达 -1.93 % 但非预声明主核、且 Δη₀ +8.26 %）；面元密度次之（末次增量 7.54 %，观测阶 -3.97 非渐近，GCI 仅作量级指示）；粘性修正为小项（-2.58 个百分点）；余项归几何/载荷保真度。

判据原文（先于动手声明）：A1 锚可达性机检（命中文件数/字段数/逐文件 sha256）；A2 五档 K_T/K_Q/η₀ 表带相对偏差与参照类型；A3 设计点 |ΔK_T| < 5 %，否则如实 FAIL + 归因；A4 至少两套网格的面元密度 GCI（观测阶 + GCI 指数落表，并声明是否渐近）。

