# W20 / 计划项 5：KP505 敞水锚回核 + 三工况 GCI 与偏差分解

*仓库*: TensorBEM  ·  *分支*: `agent/m14569`  ·  *mission*: 14569（返修轮 1，原单 m9641 经 maturity_gate 复核 L0 后重开）  ·  *生成 UTC*: 2026-11-11T10:29:52+00:00
*git commit（工件生成时）*: `f586616401e51bf252d550017cde78db5f430ca0`

## 0. 工件与溯源（先落工件，后写报告）

| 工件 | sha256 |
|---|---|
| `papers/data/w20-anchors/kp505_transcription.json` | `c97281bfa8e48147ea866c8e9e2f7a965259ca98d180962d77f88b94c77e2e00` |
| `benchmarks/verified/bem_kp505_openwater_w20/gci.json` | `8686bd182db5aec2c02706a05f92001d10282cfe50b7689e521e1d327a1eb8d2` |
| `benchmarks/verified/bem_kp505_openwater_w20/levels.csv` | `a505506bc3f81ad50797707cccd8c118c8ed3bfc47722da2239ab739451e4702` |
| `benchmarks/verified/bem_kp505_openwater_w20/gci_table.csv` | `298c495f83288c4b94661b5c20a624f1a93762977f663b9edc32e72a3575e8ef` |
| `benchmarks/verified/bem_kp505_openwater_w20/figures/w20_kt_kq_eta_vs_anchor.png` | `686ecbfadd132dee1fe716c2d06206e316e1f0f38472ee14cc86881dbc3e9084` |
| `benchmarks/verified/bem_kp505_openwater_w20/figures/w20_order_and_gci.png` | `b30b604149bb421fef669aabf2ae40f3ca93eb702fec5b7bcb8d06e016b26d01` |
| `benchmarks/verified/bem_kp505_openwater_w20/figures/w20_deviation_split.png` | `0b804effebe6b9015565cf29aa1b7f463b422a2ec1a9603dd1e1e88ad5cb0a2a` |
| `papers/data/w20-anchors/sources/lbm_bench_propeller_kp505_excerpt.txt` | `da2de70df401dfbf18ac1f0cf8378d3c2e76c6d69339244cb9567405ba226ec8` |

生成命令（可重跑，缓存使中断可续）：

```
python -m tools.build_w20_kp505_transcription
python -m tensorbem.benchmarks.kp505_w20_gci
python -m tools.emit_w20_report
python -m tools.emit_w20_evidence
pytest tests/test_w20_kp505_transcription.py tests/test_w20_kp505_gci.py tests/test_w20_kp505_integration.py -q
```

## 1. 案例介绍（物理背景 / 几何 / 参数）

**物理背景**：KP505 是 KRISO 集装箱船 KCS 的五叶定距桨，其敞水试验数据是推进域最常用的公开锚。本单要回答两件事：(a) 生态里流传的 KP505 系数是否为原始表值；(b) TensorBEM 的升力面/涡格 BEM 在该锚上的偏差，有多少是面板离散误差、多少是模型误差。

**锚来源（分级：SECONDARY_REPRODUCTION）**：`H. Iftikhar, A. C. Takinaci, 'Performance Prediction of KP505 Propeller Using Lifting Line Theory and RANS', X International Conference on Computational Methods in Marine Engineering (MARINE 2023), CIMNE, DOI 10.23967/marine.2023.120`，DOI `10.23967/marine.2023.120`，第 10 页 “Table 2. Comparison of CFD and Experimental Results” 表的 Experimental 列。PDF sha256 `161ef62673c89e81…`，转写文本 `benchmarks/anchors/sources/kp505_marine2023_p10_table2.txt` sha256 `07a7ebabe67f98ea…`。该论文未引用原始 KRISO 报告，故锚是二手转录；本条差距在工件中显式声明，未隐藏。

**几何与参数（本研究实际使用的设置）**：D = 0.25 m（官方 IGES 成员名 250 mm）、Z = 5、Ae/Ao = 0.80、轮毂比 0.18、平均 P/D = 0.95、NACA66 剖面（源文 Table 1）；BEM 计算取 `wake_model = geometric`、`wake_length = 1.0`、`rev_per_second = 10.0`、t/c = 0.06`、黏性 = True（ITTC-78 剖面阻力）。

**锚表（10 个工作点，程序化注入）**：

| J | K_T | 10K_Q | η₀ = 5·J·K_T/(π·10K_Q) | 源行 |
|---|---|---|---|---|
| 0.1 | 0.494 | 0.683 | 0.1151 | L15 |
| 0.2 | 0.447 | 0.636 | 0.2237 | L16 |
| 0.3 | 0.392 | 0.585 | 0.3199 | L17 |
| 0.4 | 0.343 | 0.530 | 0.4120 | L18 |
| 0.5 | 0.290 | 0.471 | 0.4900 | L19 |
| 0.6 | 0.241 | 0.409 | 0.5627 | L20 |
| 0.7 | 0.193 | 0.342 | 0.6287 | L21 |
| 0.8 | 0.141 | 0.270 | 0.6649 | L22 |
| 0.9 | 0.087 | 0.192 | 0.6491 | L23 |
| 1 | 0.029 | 0.108 | 0.4274 | L24 |

## 2. 怎么做的

**(a) 锚回核（判据 1）**：`tools/build_w20_kp505_transcription.py` 把表 2 的 70 个数字（10 行 × 7 列）逐个拆成带 provenance 的记录：文件 + sha256 + 1-based 行号 + 列字段 + 逐字源行；η₀ 由闭式公式派生并记录两个操作数的源行；源文自带的误差列用其自身 CFD/Experimental 列复算，容差由印刷小数位推出（半 ulp 传播），不是拍的。跨仓回核：TensorLBM `benchmarks/bench_propeller.py` 的 `_KP505_REFERENCE` 六个常数逐条转录、哈希并与锚表逐项比较（该常量块逐字复制进仓内，`papers/data/w20-anchors/sources/lbm_bench_propeller_kp505_excerpt.txt`，因此回核不依赖外部工作树）。

**(b) GCI（判据 2）**：三套面板密度同时细化（r = 2，三轴同时加密）：L1 = 8×2 面板/叶 + 24 尾涡方位段，L2 = 16×4 + 48，L3 = 32×8 + 96；J = 0.5/0.7/0.9 三工况（外加 J = 0.8 的历史对照点）。Roache GCI 由 `src/tensorbem/uq.py:grid_convergence_index` 计算（安全系数 1.25），理论阶**预先声明** p_th = 2.0（中点求积的常设计阶；叶尖载荷 √(1−r) 行为使 1.5 成为备选，两套偏差都在表里）。涡核策略预声明两套：`fixed`（core_ratio 三级不变 = 主）与 `similarity`（core_ratio ∝ 1/n_radial）。

**(c) 偏差分解（判据 3）**：d_total = 100(f_fine − f_anchor)/f_anchor；d_disc = 100(f_fine − f_ext)/f_anchor；d_model = 100(f_ext − f_anchor)/f_anchor，f_ext 取 Richardson 外推；三项恒等式闭合残差 ≤ 1e-9 个百分点（机检）。因为观测阶可能非法（见 §3），另给一条不依赖 p 的稳健归属：把最细一次加密增量 |f_fine − f_medium|/f_anchor 作为残余离散误差的尺度，得模型项区间 d_model ∈ [d_total − 2Δ, d_total + 2Δ]。

## 3. 计算结果（全部数字由工件注入）

### 3.1 判据 1：锚转录件可机检回读（PASS）

| 判据 | 内容 | 结果 |
|---|---|---|
| T1-every-number-has-source | every table number carries file + sha256 + line + field + verbatim line | PASS |
| T2-machine-reread | the artefact can be re-read against the sources and reproduces every number | PASS |
| T3-source-self-consistency | the source's own error columns reproduce from its own CFD/Experimental columns | PASS |
| T4-design-point-present | the design advance ratio J = 0.70 is in the transcribed set | PASS |
| T5-anchor-table-agreement | the transcribed Experimental columns equal the pinned anchor table | PASS |
| T6-circulating-constants-are-source-values | the KP505 constants circulating in TensorLBM are the values of the source table | FAIL |
| T7-circulating-J-grid-is-source-grid | every advance ratio of the circulating table is an advance ratio of the source table | FAIL |

数字总数 70，全部带 file+sha256+line+field+verbatim；源文自洽最大原始偏差 1.178 个百分点（J = 1.0 的 K_T 误差列，落在 3 位小数的舍入界内；相对舍入界最坏比 0.51）。

**跨仓回核结论（negative finding，已入判据 T6/T7 = FAIL）**：

| J（流传表） | K_T 流传值 | K_T 锚值 | ΔK_T% | 第二列流传值 | 10K_Q 锚值 | Δ% (当 10K_Q) | Δ% (当 K_Q×10) |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.450 | 0.494 | -8.91 | 0.065 | 0.683 | -90.5 | -4.8 |
| 0.3 | 0.420 | 0.392 | +7.14 | 0.061 | 0.585 | -89.6 | +4.3 |
| 0.5 | 0.370 | 0.290 | +27.59 | 0.055 | 0.471 | -88.3 | +16.8 |
| 0.7 | 0.290 | 0.193 | +50.26 | 0.047 | 0.342 | -86.3 | +37.4 |
| 0.9 | 0.170 | 0.087 | +95.40 | 0.033 | 0.192 | -82.8 | +71.9 |
| 1.1 | 0.040 | 0.029 | +37.93 | 0.015 | 0.108 | -86.1 | +38.9 |

流传表自称出处：`# KP505 reference data (Fujisawa et al. 2000, SIMMAN 2008)`；其 J 网格 [0.1, 0.3, 0.5, 0.7, 0.9, 1.1] 含 J = 1.1，而源表只到 J = 1.0（T7 FAIL）。结论：**TensorLBM 的 KP505 常数不是本锚表值**，K_T 最大偏离 95.4%，第二列即便按 K_Q（×10）读也偏离 71.9%。该仓以 30% 容差对这些常数做“通过/不通过”判定，其结论不构成对 KP505 的证据。

### 3.2 判据 2：三工况三密度 GCI（观测阶与 GCI 落表）

| J | 量 | L1 | L2 | L3 | 观测阶 p | GCI_fine % | p 与 p_th=2 偏差 % | p_th=1.5 偏差 % | 3% 门槛 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | KT | 0.30272 | 0.30150 | 0.28056 | -4.098 | 9.91 | 304.9 | 373.2 | FAIL |
| 0.5 | 10KQ | 0.43316 | 0.44244 | 0.42920 | -0.512 | 12.90 | 125.6 | 134.2 | FAIL |
| 0.7 | KT | 0.17094 | 0.17018 | 0.15824 | -3.967 | 10.07 | 298.3 | 364.4 | FAIL |
| 0.7 | 10KQ | 0.28011 | 0.28473 | 0.27465 | -1.125 | 8.47 | 156.3 | 175.0 | FAIL |
| 0.9 | KT | 0.03126 | 0.03081 | 0.02809 | -2.576 | 14.54 | 228.8 | 271.7 | FAIL |
| 0.9 | 10KQ | 0.09050 | 0.09349 | 0.09306 | +2.770 | 0.10 | 38.5 | 84.6 | PASS |

判据 2 裁定：**FAIL**（W3）。6 个格子（3 工况 × 2 量）合格 0 个；最接近理论阶的是 10KQ@J=0.9，p = +2.770，与 p_th = 2 偏差 38.5%（仍在 30% 带外）。其余格子观测阶为负——三级加密的增量符号反复，说明该离散**未进入渐近区**，GCI 不是误差上界，只是量级指示。判据按原始 30% 门槛裁 FAIL，如实上报，未以“阶已接近”为由放宽。

### 3.3 判据 3：设计点 |ΔK_T|、|ΔK_Q| 与 3% 问答

| 量 | BEM(L3) | 锚 | Δ% | GCI_fine % | 是否 < 3% |
|---|---|---|---|---|---|
| K_T | 0.15824 | 0.193 | -18.01 | 10.07 | 否 |
| 10K_Q | 0.27465 | 0.342 | -19.69 | 8.47 | 否 |

**显式回答：J = 0.70 设计点 |ΔK_T| = -18.01%（判定 > 3%），|ΔK_Q| = -19.69%（判定 > 3%）。**两项都不满足 3% 门槛，判据 3 = FAIL（W6）**。历史 29.1% 未过，本单禁下调阈值，故此处照实判负，并把失败拆到 §3.4。

### 3.4 偏差分解：29.1% 里有多少是离散、多少是模型

历史值复现（先复现再分解）：

- 报告 8822 的原配置（16×2 面板/叶、24 尾涡段、core_ratio 0.05）在 J = 0.8：K_T = 0.09993（锚 0.141）⇒ ΔK_T = -29.13%，与记载的 -29.13% 相差 0.0043 个百分点 ⇒ 复现成立。
- 同一工况在 L3 细网格：ΔK_T = -33.23%（离散项吃掉约 4.1 个百分点）。

三工况分解表（主核策略 `fixed`，Richardson 分裂 + 不依赖 p 的区间归属并列）：

| J | 量 | Δ_total % | 离散项 % | 模型项 % | 闭合残差 | 末次增量 Δ% | 模型项区间 % | 模型项出带？ |
|---|---|---|---|---|---|---|---|---|
| 0.5 | KT | -3.26 | -7.67 | +4.41 | 4.4e-16 | 7.22 | [-17.7, +11.2] | 否 |
| 0.5 | 10KQ | -8.88 | -9.41 | +0.53 | 0.0e+00 | 2.81 | [-14.5, -3.3] | 是 |
| 0.7 | KT | -18.01 | -6.61 | -11.40 | 3.6e-15 | 6.19 | [-30.4, -5.6] | 是 |
| 0.7 | 10KQ | -19.69 | -5.44 | -14.25 | 0.0e+00 | 2.95 | [-25.6, -13.8] | 是 |
| 0.9 | KT | -67.71 | -3.76 | -63.96 | 0.0e+00 | 3.13 | [-74.0, -61.5] | 是 |
| 0.9 | 10KQ | -51.53 | +0.04 | -51.57 | 7.1e-15 | 0.23 | [-52.0, -51.1] | 是 |

读法与诚实边界：Richardson 分裂在 p 为负的格子里**不可作物理归属**（外推值落在加密区间之外）。可用的结论只有区间口径：

- J = 0.5：ΔK_T = -3.26%，末次增量 7.22%（区间 -17.7…+11.2%）；ΔK_Q = -8.88%，末次增量 2.81%（区间 -14.5…-3.3%）。
- J = 0.7：ΔK_T = -18.01%，末次增量 6.19%（区间 -30.4…-5.6%）；ΔK_Q = -19.69%，末次增量 2.95%（区间 -25.6…-13.8%）。
- J = 0.9：ΔK_T = -67.71%，末次增量 3.13%（区间 -74.0…-61.5%）；ΔK_Q = -51.53%，末次增量 0.23%（区间 -52.0…-51.1%）。

即：**J = 0.5 的 |ΔK_T| 落在离散带内、不可归因；J = 0.7 与 J = 0.9 的偏差在两次末增量之外，以模型项为主**。J = 0.8 的历史 29.1% 同理：末次增量约 5% 量级，总偏差 −29%～−33%，差值的绝对主体（≥20 个百分点）只能来自模型/几何保真度（叶片弦长分布与径向螺距为声明假设、无弯度、无侧斜/纵倾、无毂与厚度堵塞），而不是面板数。

### 3.5 附带发现：预声明的“涡核策略”在本研究的密度下是死参数（W7 = FAIL，不可测）

| 密度 | core_ratio 扫描 | K_T | 是否逐位相同 |
|---|---|---|---|
| L1-coarse (8×2) | 0.025 | 0.17094361 | — |
| L1-coarse (8×2) | 0.050 | 0.17094361 | — |
| L1-coarse (8×2) | 0.100 | 0.17094361 | — |
| L1-coarse (8×2) | 0.400 | 0.20251842 | — |
| L1-coarse (8×2) | 0.800 | 0.24695186 | — |
| L3-fine (32×8) | 0.025 | 0.15824276 | — |
| L3-fine (32×8) | 0.050 | 0.15824276 | — |
| L3-fine (32×8) | 0.100 | 0.15824276 | — |
| L3-fine (32×8) | 0.400 | 0.16003856 | — |

policy 区间 0.025/0.05/0.10 下 K_T 逐位相同（`policy_range_is_inert = True`）；首个能改变结果的比值：L1 0.4、L3 0.4，L1 在 0.025→0.8 全扫描下最大变化 44.5%。机制：`solve_lifting_surface` 把段核钳到 `max(core_ratio·|segment|, 0.5·条带宽)`，在这些密度下条带宽下限压过了 core_ratio 项。后果有两面：（i）W7 的稳健性检验在本研究密度下**不可测**，按失败关闭记为 FAIL；（ii）任何以 core_ratio = 0.05…0.4 作“核敏感性”轴的后续研究都可能实际只扫到平台段。

## 4. 结果图

![K_T/10K_Q/η₀ 对锚](../../benchmarks/verified/bem_kp505_openwater_w20/figures/w20_kt_kq_eta_vs_anchor.png)

![观测阶与 GCI](../../benchmarks/verified/bem_kp505_openwater_w20/figures/w20_order_and_gci.png)

![偏差分解](../../benchmarks/verified/bem_kp505_openwater_w20/figures/w20_deviation_split.png)

图 1 `benchmarks/verified/bem_kp505_openwater_w20/figures/w20_kt_kq_eta_vs_anchor.png`：细网格 L3 与转录锚的 K_T(J)、10K_Q(J)、η₀(J) 对比曲线；锚为 10 个工作点，BEM 只算三个工况（另加历史对照点 J = 0.8）。

图 2 `benchmarks/verified/bem_kp505_openwater_w20/figures/w20_order_and_gci.png`：观测阶 p（带状 ±30% 与 p_th = 2 的虚线）与细网格 GCI（3% 验收线与 1% 参考线）逐工况柱状图。

图 3 `benchmarks/verified/bem_kp505_openwater_w20/figures/w20_deviation_split.png`：偏差分解堆叠柱（灰 = 离散项，绿 = 模型项，黑点 = 对锚总偏差）。
**说明**：本研究的场所量是积分量的扫掠曲线，不是三维流场，故未提供场量云图——BEM 求解器只返回面板环量/载荷，面元场分布图对判据无增量，按“无图必须写明理由”的例外如实声明。

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **锚是二手转录**：源文未引原始 KRISO 报告，表值经一次转录步骤；任何基于它的精度断言都继承该步骤。primary handout 在本环境不可达（详见 `docs/kp505_openwater_anchor.md` §1）。
2. **三级 GCI 无法自证渐近区**：本单 6 个格子中 0 个观测阶合格；仓内四级单轴梯形研究（`benchmarks/verified/propeller_openwater/kp505_gci_ladder.json`）已记录同一条梯子的两个三重组可差 30 倍。三级结果只能当量级指示。
3. **`core_ratio` 在本研究密度下被条带宽下限钳住**（§3.5）：不要把 W7 当作“核无关”的证据。
4. **任务书点名的 `examples/propeller_openwater_d3q27.py` 在 TensorLBM 中不存在**（仓内 `examples/` 无该文件，`benchmarks/` 内亦无 propeller_openwater 命名的脚本）；可回核的只有 `benchmarks/bench_propeller.py` 的 `_KP505_REFERENCE`，已在 §3.1 逐条比对。本项按实况记录，未虚构该文件的转录。
5. **BEM 的模型误差来源是可枚举的**：叶片弦长分布与径向螺距非公开（声明假设）、无弯度、无侧斜/纵倾、无毂与厚度堵塞、尾涡为几何螺距无收缩；J 越大进速比越接近零推力，这些假设的相对代价越大（J = 0.9 达 −67%）。
6. **适用范围**：结论只对“D = 1 无量纲、n = 1 rev/s、Re 由 rev_per_second = 10 映射”的该实现成立；换 Re 会移动 ITTC-78 黏性项。

## 6. 误差判定（逐项，含是否 < 3% 的显式回答）

| 判据 | 要求 | 实测 | 裁定 |
|---|---|---|---|
| 1) 锚转录件每数带源且可机检回读 | 100% 数字带 file+字段/原表，verify() 0 mismatch | 70 个数字全部带源；verify 通过；跨仓回核 T6/T7 = FAIL（负结论） | PASS |
| 2) 三工况 GCI：p 与 GCI 落表，p 与理论阶偏差 < 30% | 六格全部 < 30% | 0/6 格合格；最接近理论阶的是 10KQ@J=0.9，偏差 38.5%（仍在 30% 带外） | FAIL |
| 3) 设计点 |ΔK_T|、|ΔK_Q| 逐项给出并回答 < 3% | 两项都 < 3% | |ΔK_T| = 18.01%，|ΔK_Q| = 19.69%，两项都 > 3% | FAIL |

**对“是否 < 3%”的显式回答：否。** J = 0.70 设计点的 K_T 偏 -18.01%，10K_Q 偏 -19.69%，均超出 3% 门槛一个量级；阈值未下调，历史 29.1% 的欠账在本单得到的是**分解与归因**，不是“过关”。

判据总表（程序化注入）：

| 判据 id | 内容 | 结果 |
|---|---|---|
| W1-coverage | three mission advance ratios x three panel densities x K_T/10K_Q, all finite | PASS |
| W2-physical-admissibility | every level yields K_T > 0 and resisting K_Q > 0 at the mission advance ratios | PASS |
| W3-observed-order-vs-theory | observed order within 30 % of p_th = 2 at every mission J | FAIL |
| W4-gci-tabulated-and-tested | fine-grid GCI is tabulated for both quantities at every mission J and the 3 % acceptance is answered | PASS |
| W5-decomposition-closure | discretisation + model term equals the total anchor deviation | PASS |
| W6-design-point-3pct-question | |dK_T| and |dK_Q| at J = 0.70 are reported and the 3 % question is answered explicitly | FAIL |
| W7-core-policy-agreement | the observed-order verdict does not depend on the vortex-core policy | FAIL |
| W8-band-attribution | a decomposition that does not require a valid observed order is reported for every cell | PASS |

## 7. 交付清单与复跑

```
papers/data/w20-anchors/kp505_transcription.json        # 判据 1
papers/data/w20-anchors/sources/lbm_bench_propeller_kp505_excerpt.txt
benchmarks/verified/bem_kp505_openwater_w20/gci.json    # 判据 2、3
benchmarks/verified/bem_kp505_openwater_w20/levels.csv
benchmarks/verified/bem_kp505_openwater_w20/gci_table.csv
benchmarks/verified/bem_kp505_openwater_w20/figures/*.png
benchmarks/verified/bem_kp505_openwater_w20/evidence.json  # 达标判断方案 evidence 束
runs/w20_kp505_gci_cache/*.json                        # 原始逐格求解输出（运行工件）
papers/reports/w20_propeller_openwater_bem5.md          # 本报告
src/tensorbem/benchmarks/kp505_w20_gci.py               # 驱动（含预热声明判据）
tools/build_w20_kp505_transcription.py                  # 转录构建 + verify()
tools/emit_w20_report.py                                # 本文件生成器
tests/test_w20_kp505_transcription.py                   # 10 tests
tests/test_w20_kp505_gci.py                             # 9 tests
tests/test_w20_kp505_integration.py                     # 5 tests（集成调用方）
tools/emit_w20_evidence.py                              # evidence 束生成器
```

复跑求解时间为：L3 单元（32×8 面板/叶、96 尾涡段）单次约 66 s/工况；全量（4 J × 2 核策略 × 3 密度 + 9 次探针）在本机 loongarch64 上约 12 min，缓存位于 `runs/w20_kp505_gci_cache/`，中断可续。

