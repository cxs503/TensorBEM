# CASE-BEM-KP505-OPENWATER：KP505 螺旋桨敞水（TensorBEM 涡格升力面）

任务 `m17207`（第 1 轮·任务榜·算例池）。本文件由 `src/tensorbem/benchmarks/kp505_openwater_case.py` 程序化渲染，
所有数字来自 `runs/m17207/kp505_case.json`，**无一手抄**。

## ① 问题与参考出处

**物理问题**。KP505 是 KRISO 为 KCS 集装箱船设计的五叶定距桨（Tokyo 2015 敞水基准桨）。敞水系数：

    K_T = T/(rho n^2 D^4),   K_Q = Q/(rho n^2 D^5),   eta_0 = J K_T/(2 pi K_Q),   J = V_a/(n D)

**目标数值出处（零手抄：锚行以原文整行形式存于代码，本脚本 import 使用）**
- 文献：Iftikhar, H. & Takinaci, A.C. (2023). Performance Prediction of KP505 Propeller Using Lifting Line Theory and RANS. X International Conference on Computational Methods in Marine Engineering (MARINE 2023).；DOI `10.23967/marine.2023.120`
- 表：Table 2, page 10，表题原文 `Table 2. Comparison of CFD and Experimental Results`
- 列头原文：`CFD Experimental Error / J KT 10KQ KT 10KQ KT 10KQ`
- 取用列：**Experimental 列**（K_T、10K_Q）；论文自己的 CFD 列**不**作为参考
- 溯源深度（如实声明）：secondary tabulation: the paper prints the experimental columns of the KP505 open-water test, but page 10 does not cite the primary experimental report and the paper's reference list (p. 15) contains no KRISO/NMRI experimental entry. The numbers therefore enter this study through one tabulation step, which is declared rather than hidden.
- 几何出处：Table 1, page 6，表题原文 `Table 1. Propeller Characteristics`；未发表项：chord distribution, radial pitch distribution, skew, rake, section camber and thickness law are not in Table 1; they are modelled below and their effect is reported as sensitivity, not hidden
- 设计 J（程序化推导 = 锚 eta_0 峰值所在 J）：`0.8`

**容差判据（动手前固定）**：5.0%；C1 = 全 10 点三项全达标（无排除）；
C2 = 仅对通过数值可容许门（|K_T| ≤ 1、
|K_Q| ≤ 0.2、且 eta0 <= 2/(1+sqrt(1+C_T))）的点达标。

## ② 方法与实现

复用仓库既有能力，**不新起炉灶**：`tensorbem.propeller.open_water_point`（涡格升力面 BEM，
horseshoe 涡格 + Kutta-Joukowski 力 + ITTC-1978 剖面阻力量级耦合），面元由
`tensorbem.propeller.BladeGeometry` 生成（平剖面，坐落在平均螺距 P/D 的螺旋线弦线上，弦长分布使 Ae/Ao 精确匹配）。

本单新增的只是**算例驱动**（`src/tensorbem/benchmarks/kp505_openwater_case.py`）：锚表 import、
逐点打分、两轴 GCI 收敛阶梯、螺距假设敏感性、尾涡模型稳定性探针、工件与报告程序化渲染。

复现命令：

    PYTHONPATH=src python -m tensorbem.benchmarks.kp505_openwater_case runs/m17207

环境：python 3.11.6 / numpy 2.4.6 /
`Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38` / git `96f19f7ced7b`

## ③ 参数表（无量纲量完整）

| 量 | 值 | 出处 |
|---|---|---|
| 桨型 / 叶数 Z | FPP / 5 | 表 1 p.6（原文 `Propeller Type FPP Ae/Ao 0.800` / `No. of Blades 5 Rotation Right Hand`） |
| 直径 D | 0.25 m（计算取 D=1 无量纲） | 同上 `Diameter (m) 0.25 Hub Ratio 0.18` |
| 毂径比 | 0.18 | 同上 |
| 平均 P/D | 0.950 | 同上 `P/D (mean) 0.950 Section NACA66` |
| 伸张面积比 Ae/Ao | 0.800 | 同上 |
| 剖面 | NACA66（厚度比作为已声明假设） | 同上 |
| 前进比 J 网格 | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 | 锚表列 |
| 转速 n | 10 rev/s（仅用于剖面 Re） | 既有配置 |
| 尾涡模型 / 长度 / 步数 | geometric / 1 D / 24 | 既有 PRIMARY 配置 |
| 面元网格 | n_radial=16, n_chord=2, 余弦径向加密 | 同上 |
| 涡核正则比 | 0.05 | 同上 |
| 面元数 | 每叶 32（= n_radial × n_chord），全桨 160 | 运行时 |

## ④ 对照表（ref vs got，逐项误差 + 判据打勾/叉）

| J | K_T got | K_T ref | ΔK_T % | 5%? | 10K_Q got | 10K_Q ref | Δ10K_Q % | 5%? | η0 got | η0 ref | Δη0 % | 5%? | 数值可容许 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.5286 | 0.494 | +7.00 | ✗ | 0.6529 | 0.683 | -4.41 | ✓ | 0.1288 | 0.1151 | +11.93 | ✗ | 是 |
| 0.2 | 0.4734 | 0.447 | +5.91 | ✗ | 0.6104 | 0.636 | -4.03 | ✓ | 0.2469 | 0.2237 | +10.36 | ✗ | 是 |
| 0.3 | 0.4163 | 0.392 | +6.19 | ✗ | 0.5599 | 0.585 | -4.29 | ✓ | 0.3550 | 0.3199 | +10.95 | ✗ | 是 |
| 0.4 | 0.3570 | 0.343 | +4.10 | ✓ | 0.5014 | 0.530 | -5.39 | ✗ | 0.4533 | 0.4120 | +10.03 | ✗ | 是 |
| 0.5 | 0.2958 | 0.290 | +2.01 | ✓ | 0.4349 | 0.471 | -7.66 | ✗ | 0.5412 | 0.4900 | +10.46 | ✗ | 是 |
| 0.6 | 0.2326 | 0.241 | -3.51 | ✓ | 0.3605 | 0.409 | -11.86 | ✗ | 0.6160 | 0.5627 | +9.48 | ✗ | 是 |
| 0.7 | 0.1673 | 0.193 | -13.34 | ✗ | 0.2781 | 0.342 | -18.70 | ✗ | 0.6701 | 0.6287 | +6.59 | ✗ | 是 |
| 0.8 | 0.0999 | 0.141 | -29.13 | ✗ | 0.1876 | 0.270 | -30.50 | ✗ | 0.6781 | 0.6649 | +1.98 | ✓ | 是 |
| 0.9 | 0.0306 | 0.087 | -64.85 | ✗ | 0.0892 | 0.192 | -53.52 | ✗ | 0.4908 | 0.6491 | -24.38 | ✗ | 是 |
| 1 | -0.0408 | 0.029 | -240.71 | ✗ | -0.0171 | 0.108 | -115.87 | ✗ | 3.7893 | 0.4274 | +786.67 | ✗ | 否 |

逐量达标点数（5% 带内，全部 10 点）：K_T **3/10**、
10K_Q **3/10**、
η0 **1/10**。

汇总（全部 10 点，未做任何排除）：三项全达标点数 **0/10**；
可容许点（9 点）中三项全达标 **0/9**。
最差：ΔK_T = -240.71%（J=1），
Δ10K_Q = -115.87%（J=1），
Δη0 = +786.67%（J=1）。

**判定：FAIL**（C1 = False，
C2 = False）。
fail_stage = `canonical vortex-lattice open-water solve vs Table 2 (p.10)`；failure_reason = the repository's flat-section lifting-surface lattice does not reach 5% on the KP505 open-water grid: see offenders/vs anchor

## ⑤ 收敛性 / 参数敏感性

### 5.1 两轴细化阶梯 + Roache GCI（r = 2）

**n_radial** 阶梯：[8, 16, 32]

| 量 | J | coarse | medium | fine | p | GCI % | 渐近(p>0) |
|---|---|---|---|---|---|---|---|
| KT | 0.4 | 0.36564 | 0.35705 | 0.33276 | -1.499 | 14.121 | False |
| KT | 0.7 | 0.17094 | 0.16726 | 0.15634 | -1.566 | 13.179 | False |
| KT | 0.9 | 0.03126 | 0.03058 | 0.02842 | -1.656 | 13.914 | False |
| 10KQ | 0.4 | 0.49598 | 0.50141 | 0.49240 | -0.730 | 5.758 | False |
| 10KQ | 0.7 | 0.28011 | 0.27806 | 0.26719 | -2.412 | 6.266 | False |
| 10KQ | 0.9 | 0.09050 | 0.08925 | 0.08607 | -1.336 | 7.633 | False |
| eta0 | 0.4 | 0.46933 | 0.45333 | 0.43022 | -0.531 | 21.809 | False |
| eta0 | 0.7 | 0.67990 | 0.67013 | 0.65189 | -0.900 | 7.535 | False |
| eta0 | 0.9 | 0.49481 | 0.49078 | 0.47294 | -2.147 | 6.091 | False |

**n_chord** 阶梯：[1, 2, 4]

| 量 | J | coarse | medium | fine | p | GCI % | 渐近(p>0) |
|---|---|---|---|---|---|---|---|
| KT | 0.4 | 0.35649 | 0.35705 | 0.35769 | -0.191 | 1.807 | False |
| KT | 0.7 | 0.16759 | 0.16726 | 0.16734 | 2.002 | 0.021 | False |
| KT | 0.9 | 0.03104 | 0.03058 | 0.03025 | 0.505 | 3.228 | True |
| 10KQ | 0.4 | 0.49927 | 0.50141 | 0.49998 | 0.579 | 0.723 | False |
| 10KQ | 0.7 | 0.27538 | 0.27806 | 0.28064 | 0.063 | 25.880 | True |
| 10KQ | 0.9 | 0.08601 | 0.08925 | 0.09266 | -0.075 | 90.384 | False |
| eta0 | 0.4 | 0.45455 | 0.45333 | 0.45545 | -0.793 | 1.370 | False |
| eta0 | 0.7 | 0.67801 | 0.67013 | 0.66431 | 0.437 | 3.093 | True |
| eta0 | 0.9 | 0.51700 | 0.49078 | 0.46764 | 0.180 | 46.490 | True |

### 5.2 尾涡模型稳定性探针（同网格，全 J 保留）

| 尾涡模型 | 可容许点数 | 不可容许 J | max\|K_T\| | kernel 预警点数 |
|---|---|---|---|---|
| geometric | 9/10 | [1.0] | 5.286e-01 | 0 |
| kinematic | 5/10 | [0.1, 0.2, 0.3, 0.4, 1.0] | 1.049e+07 | 3 |
| induced | 6/10 | [0.1, 0.3, 0.4, 1.0] | 6.549e+01 | 3 |

kernel 预警点数 ≤ 不可容许点数：可容许门除物理包络外还检查作动盘动量上界
（`eta_0 <= 2/(1+sqrt(1+C_T))`，J=1.0 这类"数值正常但物理越界"的点由动量门捕获，
kernel 的物理包络预警不触发），两个门互补，故这里两个计数不必相等。

### 5.3 未发表几何假设的敏感性（径向螺距分布 → 以平均 P/D 水平代理）

| P/D | 三项全达标点数 | 最差 ΔK_T % | 最差 Δη0 % |
|---|---|---|---|
| 0.9 | 0/10 | -368.61 | +319.96 |
| 0.95 | 0/10 | -240.71 | +786.67 |
| 1.0 | 0/10 | -117.82 | -151.58 |

### 5.4 偏差来源诊断：K_T(J) 仿射拟合（J ≤ 0.9，仅诊断不用于修正）

| 来源 | 斜率 dK_T/dJ | 截距 | 零推力 J |
|---|---|---|---|
| 本模型 | -0.6225 | +0.6003 | 0.9643 |
| 参考 | -0.5077 | +0.5458 | 1.0752 |

斜率比（本模型/参考）= 1.226。拟合范围：J <= 0.9 (the J=1.0 point violates the momentum bound)。
斜率差异 ⇒ 不是"整体平移"（缺拱度型偏差），而是**速度场/来流角**层面的模型形式缺口；
这与 5.3 的结果一致：只把平均 P/D 抬高能改善高 J 区、却把低 J 区推坏，说明单一标定救不回全 J 域。

## ⑥ 误差判定、失败模式、局限与工件

### 6.0 误差判定（逐项，显式回答）

| 量 | 本算例判据 5% 达标点数 | 3% 严格线达标点数 | 最差误差 |
|---|---|---|---|
| K_T | 3/10 | 1/10 | -240.71%（J=1） |
| 10K_Q | 3/10 | 0/10 | -115.87%（J=1） |
| η0 | 1/10 | 1/10 | +786.67%（J=1） |

**是否 <3%？——否。** K_T / 10K_Q / η0 分别只有 1、
0、1（共 10 点）进入 3% 带内。
**是否满足本算例 5% 判据？——否**（三项全达标 0/10；可容许点 0/9）。
证据分级：本算例当前为 **honest FAIL**——数字由真实算法产生、有收敛与敏感性支撑，
既不是 MOCK（非硬编码/随机）也不是 STUB（非空壳）；失败点在模型形式，不在管线。

**失败模式（实测，非推测）**

1. **低 J 区偏高、高 J 区偏低且偏差随 J 放大**：模型 K_T(J) 斜率比参考陡（5.4，比值
   1.226），导致 J ≥ 0.7 三项全面超标，J = 1.0 处 K_T 变负
   （参考为 +0.029）。**离散不确定性本身已超判据**：5.1 中最细三重组的 spanwise GCI 在 K_T 上为
   13～14%，且观测阶 p < 0（非渐近区），chordwise 轴在 η0(J=0.9) 上 GCI 达 46% ——
   因此"仅靠加密网格"不可能把本算例做进 5%，缺口是模型形式（平剖面/常螺距/无拱度）而非离散。
2. **近奇异解（数值崩溃）**：`wake_model` 取非几何尾涡（kinematic / induced）时，
   低前进比出现 1e2..1e9 量级的 K_T（5.2 表），本轮驱动在运行前即以可容许门标注，
   崩塌点**保留在工件里**而不是丢弃。仓库既有 `b4_70_error_surface` 已记录同一现象。
3. **几何缺口**：公开表只给 6 个标量（Z、D、毂径比、平均 P/D、Ae/Ao、剖面族），
   弦长分布/径向螺距分布/拱度/厚度律/侧斜与纵倾均未发表 → 本模型取平剖面、常螺距、零侧斜，
   5.3 表明只动平均 P/D 水平救不回高 J 区（这是模型形式缺口，不是标定缺口）。

**局限**：判据只针对本算例的这 10 个 J 点；参考列本身是二次转录（论文未引一次实验报告），
误差基准因此带一个未知的转录环节。**未做**：不加拱度、不做几何反演、不对参数做任何向锚表的标定。

**工件**

- `runs/m17207/kp505_case.json`：全部数字、判据、配置、环境、GCI、探针、判定
- `runs/m17207/kp505_case_table.csv`：④ 表机读版
- `runs/m17207/kp505_case_figures/fig1_curves.png`、`fig2_errors.png`、`fig3_ladders.png`
- 一键复现：见 ② 的 repro_cmd

## ⑦ 生态摩擦（本单实测，供平台/仓库修正）

1. **桨叶载荷/流场图出不来**：`open_water_point` 只返回积分系数（K_T/K_Q/η0），不返回
   `LatticeSolution`（环量分布）也不返回展向/弦向载荷 —— 因此本算例只能给"系数 vs J"比较曲线，
   给不出"桨叶载荷分布"这类场量图。修复方向：`OpenWaterResult` 增加可选 `solution` 字段或
   `section_loading()` 访问器。
2. **配置字典混层**：既有 `kp505_openwater_bem.PRIMARY` 把几何键（`n_radial`/`n_chord`/…）与
   求解器键（`wake_model`/`wake_steps`/…）放在同一 dict，而 `open_water_point` 不接受 `n_radial`
   —— 直接 `open_water_point(J, **PRIMARY)` 立刻 `TypeError`。本驱动因此显式分层
   （`GEOMETRY_KEYS` / `SOLVER_KEYS`）。修复方向：把配置拆成 `GeometryConfig` / `SolverConfig`。
3. **API 默认项是坑**：`open_water_point` 的默认 `wake_model="kinematic"` 在默认几何上给出
   `K_T = -1.58e5`（近奇异解），此前静默返回，读者会把它当成"模型误差 1e7 %"。本单在核内加了
   物理包络 + 作动盘动量上界的 `RuntimeWarning` 与回归测试（`tests/test_kp505_openwater_case.py`）。
4. **仓库红测试（与本次改动无关）**：`tests/test_propeller_wageningen_crosscheck.py::
   TestBEMvsWageningenOrderOfMagnitude` 6 例在未改动的工作树上即为 FAIL（BEM 侧 K_T = 0.0000），
   属既有技术债，本单未修（超出算例范围）。
5. **算力排队**：全量回归 4067 例递交集群 `cj1456`，排队 588 位（"不指定节点"常态令下的正常等待）；
   本单的算例本身在主控机 44 s 跑完。
6. **锚表一度转录**：参考列本身是二次转录（论文 p.10 未引一次实验报告），误差基准带一个未知转录环节；
   与本地另一份转录逐位一致（转录交叉核对，非独立证据）。
