# W20 耐波性域截断 item 4（anchor-repair）：Wigley 仓内锚件逐字段抄录 + 域截断/辐射能量门判别

> 任务 mission 9647 ｜ 仓 TensorBEM ｜ 学科域：耐波 ｜ 分支 `agent/m9647` ｜ 基线 commit `7f5e068f5c23bc34b0de3f57b094475d9fb93187`
> 工件 1：`papers/data/w20-anchors/wigley_repo_benchmark_transcription.json`（sha256 `e53d1415b064ded9b7681c5c5c968d595d3f0a009dc1932e85ac679bd762055d`）
> 工件 2：`papers/data/w20-anchors/wigley_truncation_comparison.json`（sha256 `0bf3a74a8c65a6c09af5077b830b17caef39961456d4c15132f23677afeab29d`）
> 输入锚件：`benchmarks/verified/bem_wigley_wave_resistance/results.json`（sha256 `244cc29b1a50f5cec9b3869f586e611db855406f3616c5039db2133577a8792b`，report sha256 `dd7e1974c532f7701892e6876dddb79c1d6d67885e0e35b4dcf7bf9ebf02e74b`）、
> 　　　　　`benchmarks/verified/bem_wigley_hydrostatics/results.json`（sha256 `a4f0d43dd8937d84afb5ff523944bbe4d39e1947592771d690845b8c80c199c1`）
> 生成（UTC）：2026-11-10T13:48:48+00:00 ｜ 环境：Python 3.11.6 / numpy 2.4.6 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38
> 本报告由 `papers/reports/build_w20_report4.py` 从上述工件**程序化渲染**（零手抄）。

## 1. 案例介绍

**物理背景。** Wigley III 抛物型船型是兴波/耐波数值方法最常用的解析船型：水线半宽
$y(x)=\tfrac{B}{2}[1-(2x/L)^2]$，因此 $C_b=4/9$、$KB=5T/8$、$A_{wp}=2LB/3$ 全有闭式解，
几何本身不引入误差——与它的任何偏差都归给方法。W18 exp#1 试图抄录**模型试验** RAO 表
（Wigley 1942 / ITTC 耐波基准），自报 honest FAIL（源不可达、无表）。本单改走仓内路径：
把仓内两份已验证基准逐字段抄录，并把域截断实验跑到**与锚件同口径**，最后明确区分
"仓内 BEM 自算 / 数值参照 / 解析参照 / 试验锚（缺失）"四类。

**几何与流体（逐字段取自锚件，指针见括号）。**

| 量 | 值 | 单位 | 指针 |
|---|---|---|---|
| $L$ | 10 | m | `/geometry/L` |
| $B$ | 1 | m | `/geometry/B` |
| $T$ | 0.625 | m | `/geometry/T` |
| $L/B$ | 10 | – | `/geometry/L_over_B` |
| $B/T$ | 1.6 | – | `/geometry/B_over_T` |
| $\rho$ | 1025 | kg/m³ | `/fluid/rho` |
| $g$ | 9.80665 | m/s² | `/fluid/g` |
| $C_b$（闭式） | 0.444444 | – | `/reference/Cb` |
| 排水体积（闭式） | 2.777778 | m³ | `/reference/volume` |
| $KB$（闭式） | 0.390625 | m | `/reference/KB` |
| 水线面积（闭式） | 6.666667 | m² | `/reference/waterplane_area` |
| 湿表面积（闭式求积） | 14.879063 | m² | `/wetted_area_reference` |

**Froude 数工况（`/cases/<i>/*`，与兴波阻力锚件同口径；$C_w=R_w/(0.5\rho U^2LT)$）。**

| i | Fn | U [m/s] | $R_w$ 参照 [N] | $C_w$ 参照 | 参照类型 |
|---|---|---|---|---|---|
| 0 | 0.20 | 1.980571 | 2.575166 | 2.04951e-04 | numerical (Michell) |
| 1 | 0.30 | 2.970856 | 12.292634 | 4.34818e-04 | numerical (Michell) |
| 2 | 0.40 | 3.961141 | 26.323870 | 5.23763e-04 | numerical (Michell) |

**四类参照的界线（本单核心纪律）。**

| 参照量 | 类型 | 是什么 | 可当试验锚？ |
|---|---|---|---|
| `Rw_reference` / `Cw_reference` | **numerical** | Michell 薄船积分，480×20 复合 Gauss-Legendre 收敛求积（`/method/reference`） | ❌ |
| `Rw_production` / `Cw_production` | numerical | Michell 薄船积分，256 节点生产路径（`/method/production`） | ❌ |
| hydrostatics `/reference/*` | **analytic** | Wigley 闭式解（$V=4LBT/9$ 等） | ❌（但是最严基准） |
| hydrostatics `/levels/*` | numerical | 面元散度定理 / 水线矩积分 | ❌ |
| 模型试验 RAO、阻力 | **experiment** | Wigley (1942)/ITTC 试验表 | ⚠️ **WITHHELD — 仓内不存在** |

锚件自身写明不做试验对比：`/evidence_level` = `"REAL (converged Michell theory reference; two independent quadratures)"`，兴波阻力 report 原文
“This verification does **not** compare with experimental data.”。故本报告**没有任何一行被标为
`experiment`**（判据 2 已机器化：`experiment_rows = 0`）。

## 2. 怎么做的

**① 字段级抄录（零手抄）。** `papers/data/w20-anchors/build_wigley_transcription.py` 递归枚举两份
`results.json` 的**全部标量叶节点**，为每叶生成 RFC-6901 指针并 `resolve(pointer)` 回读比对（值+类型）；
再按 9 类"可用锚必备字段"（几何/静水参考/流体/工况/物理量/网格记录/收敛记录/验收判据/证据等级）逐类核查。
叶节点总数 220、抄录 220、完整率 100.00%、
回读失败 0、必备类全在 = `True`。
**下游脚本里没有任何锚件数值字面量**——全部经指针取值。

**② 域截断实验（同口径重跑，而非从 exp#2/#3 换算）。** W18 exp#2/#3 的截断工件在本工作树内检索不到
（`papers/`、`docs/`、`benchmarks/`、`tools/`、git 全量历史），因此**不引用任何 exp#2/#3 数字**，
而是用仓内机制在**锚件同口径**（L = 10 m、$\rho$ = 1025 kg/m³、g = 9.80665 m/s²、
同一 $C_w$ 归一化）重新执行：

* **求解器**：`tensorbem.rankine_fs.solve_rankine_free_surface` —— Rankine 源自由面面元法，自由面只铺在
  **截断**矩形域 $x\in[-L,\,x_{max}L]$、$y\in[0,\,y_{max}L]$，辐射条件用自由面条件的**逆风（后向）差分**数值施加；
* **档位**：横向 $y_{max}/L \in \{1.0, 1.5, 2.0, 2.5\}$（下游固定 3.5L）；下游 $x_{max}/L \in \{1.5, 2.5, 3.5\}$
  （横向固定 2.5L）；参考域 = $x_{max}=3.5L$、$y_{max}=2.5L$；
* **G3 辐射能量门** = $|R_w^{p}-R_w^{f}|/|R_w^{p}|$，$R_w^{f}$ 由 `farfield_wave_resistance`
  在下游波切上取动量通量；
* **消波区**：本仓**没有**海绵层/消波区参数（`src/tensorbem/free_surface.py` 原文
  “truncated fluid domain, no sponge layer and no free-surface mesh”）→ 裁决 **NOT_SUPPORTED**；
  只把逆风格式 1→2 阶作为**替代变量**并显式标注"不是消波区长度"；
* **G5 辐射边界条件（解析核）**：`tensorbem.free_surface3d.verify_radiation_condition_3d` 的
  $\sqrt{R}\,|\partial G/\partial R - i\nu G|$（深水 Green 函数 Sommerfeld 残差），预登记斜率 −1.5±0.10；
* **G7 参考量自身截断**：`rankine_wave.free_surface_truncation_study` 把 Michell 积分波数范围截到
  $k_{max}/k_0\in\{10,20,30,50\}$；
* **离散化固定**：每个 Fn 内 ppw 固定（Fn 0.20 → 4，
  Fn 0.30 → 12，Fn 0.40 → 12；
  船体 2×24×8），**每次扫描只变域大小**，截断效应与离散误差不混叠；
* **求解成本**：稠密 $O(N^3)$，实测 3–57 s/点（未知量 836–5212），
  全部结果由 `papers/data/w20-anchors/.solve_cache.json` 记忆化，可复算。

**③ 预登记门限（看结果之前写死，见工件 `predeclared_gates`）：** G1 横向 ≤ 2%、
G2 下游 ≤ 2%、G3 能量门 ≤ 5%、
G4 精度 ≤ 5%（平台带 ≤ 15%）、
G5 斜率 −1.5±0.1、G7 ≤ 0.5%。

**④ 溯源。** 脚本 sha256：transcription `d712f2e37f47e76d5478b12bb7d5a946…`、
truncation `105049cbe2cc1e54672e048d3038aab2…`；被调代码 sha256（见工件 `inputs`）：
`rankine_fs.py` `d5c7aba083fb3289…`、
`free_surface3d.py` `abb7f9dd34b2d9e1…`、
`rankine_wave.py` `48fca885b0014be1…`、
`free_surface.py` `4832e5da7ceabdca…`。

## 3. 计算结果

### 3.1 对比表 A：兴波阻力系数 $C_w$（同口径，参照 = 数值）

| Fn | 仓内截断域自算 $C_w$ | 参照 $C_w$ | 参照类型 | 参照指针 | 相对偏差 [%] | ≤5% | ≤15% |
|---|---|---|---|---|---|---|---|
| 0.20 | -2.462 | 2.050e-04 | numerical | `/cases/0/Cw_reference` | 1.201e+06 | FAIL | FAIL |
| 0.30 | 0.3343 | 4.348e-04 | numerical | `/cases/1/Cw_reference` | 7.677e+04 | FAIL | FAIL |
| 0.40 | 0.3493 | 5.238e-04 | numerical | `/cases/2/Cw_reference` | 6.659e+04 | FAIL | FAIL |

### 3.2 对比表 B：静水力（参照 = 解析闭式）

| 网格 | $C_b$ [%] | 体积 [%] | $KB$ [%] | 水线面积 [%] | 湿表面积 [%] | 最大 [%] | ≤15% |
|---|---|---|---|---|---|---|---|
| Wigley hydrostatics, mesh 16x8 (512 panels) | 0.487 | 0.487 | 0.058 | 0.391 | 0.0457 | 0.487 | PASS |
| Wigley hydrostatics, mesh 32x16 (2048 panels) | 0.122 | 0.122 | 0.0146 | 0.0977 | 0.0114 | 0.122 | PASS |
| Wigley hydrostatics, mesh 64x32 (8192 panels) | 0.0305 | 0.0305 | 0.00366 | 0.0244 | 0.00284 | 0.0305 | PASS |
| Wigley hydrostatics, mesh 128x64 (32768 panels) | 0.00763 | 0.00763 | 9.15e-04 | 0.0061 | 7.10e-04 | 0.00763 | PASS |

### 3.3 域截断逐档落表 — G1 横向

**Fn = 0.20**（U = 1.9806 m/s，λ = 2.513 m，λ/L = 0.251，ppw = 4，船体面元 2×24×8，$x_{max}=3.5L$）

| $y_{max}/L$ | $y_{max}$ [m] | 未知量 | $R_w$ 压力积分 [N] | $R_w$ 远场 [N] | 能量门 [%] | $\Delta R_w/R_w$ vs 参考域 [%] | 墙钟 [s] |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 1334 | -30931.4 | 8.01e-05 | 100 | 0.00257 | 0.0 |
| 1.5 | 15 | 1910 | -30930.7 | 9.85e-05 | 100 | 3.54e-04 | 0.0 |
| 2 | 20 | 2486 | -30930.6 | 1.08e-04 | 100 | 3.66e-05 | 0.0 |
| 2.5 | 25 | 3062 | -30930.6 | 1.12e-04 | 100 | 0 | 0.0 |
**Fn = 0.30**（U = 2.9709 m/s，λ = 5.655 m，λ/L = 0.565，ppw = 12，船体面元 2×24×8，$x_{max}=3.5L$）

| $y_{max}/L$ | $y_{max}$ [m] | 未知量 | $R_w$ 压力积分 [N] | $R_w$ 远场 [N] | 能量门 [%] | $\Delta R_w/R_w$ vs 参考域 [%] | 墙钟 [s] |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 2172 | 9454.38 | 5.06e-07 | 100 | 0.0485 | 0.0 |
| 1.5 | 15 | 3217 | 9451.25 | 9.94e-07 | 100 | 0.0154 | 0.0 |
| 2 | 20 | 4167 | 9450.27 | 1.16e-06 | 100 | 0.00506 | 0.0 |
| 2.5 | 25 | 5212 | 9449.8 | 1.25e-06 | 100 | 0 | 0.0 |
**Fn = 0.40**（U = 3.9611 m/s，λ = 10.053 m，λ/L = 1.005，ppw = 12，船体面元 2×24×8，$x_{max}=3.5L$）

| $y_{max}/L$ | $y_{max}$ [m] | 未知量 | $R_w$ 压力积分 [N] | $R_w$ 远场 [N] | 能量门 [%] | $\Delta R_w/R_w$ vs 参考域 [%] | 墙钟 [s] |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 836 | 17538.3 | 4.42e-04 | 100 | 0.0963 | 0.0 |
| 1.5 | 15 | 1160 | 17552.3 | 5.03e-04 | 100 | 0.0165 | 0.0 |
| 2 | 20 | 1484 | 17554.5 | 5.34e-04 | 100 | 0.00382 | 0.0 |
| 2.5 | 25 | 1808 | 17555.2 | 5.55e-04 | 100 | 0 | 0.0 |

### 3.4 域截断逐档落表 — G2 下游

**Fn = 0.20**（$y_{max}=2.5L$）

| $x_{max}/L$ | $x_{max}$ [m] | 未知量 | $R_w$ 压力积分 [N] | 能量门 [%] | $\Delta R_w/R_w$ vs 参考域 [%] |
|---|---|---|---|---|---|
| 1.5 | 15 | 1782 | -30935.2 | 100 | 0.0149 |
| 2.5 | 25 | 2422 | -30931 | 100 | 0.00143 |
| 3.5 | 35 | 3062 | -30930.6 | 100 | 0 |
**Fn = 0.30**（$y_{max}=2.5L$）

| $x_{max}/L$ | $x_{max}$ [m] | 未知量 | $R_w$ 压力积分 [N] | 能量门 [%] | $\Delta R_w/R_w$ vs 参考域 [%] |
|---|---|---|---|---|---|
| 1.5 | 15 | 2986 | 9450.94 | 100 | 0.0121 |
| 2.5 | 25 | 4099 | 9450.01 | 100 | 0.00226 |
| 3.5 | 35 | 5212 | 9449.8 | 100 | 0 |
**Fn = 0.40**（$y_{max}=2.5L$）

| $x_{max}/L$ | $x_{max}$ [m] | 未知量 | $R_w$ 压力积分 [N] | 能量门 [%] | $\Delta R_w/R_w$ vs 参考域 [%] |
|---|---|---|---|---|---|
| 1.5 | 15 | 1088 | 17545.7 | 100 | 0.0544 |
| 2.5 | 25 | 1448 | 17553.3 | 100 | 0.0109 |
| 3.5 | 35 | 1808 | 17555.2 | 100 | 0 |

### 3.5 辐射能量门 — G3

| Fn | $|R_w^p-R_w^f|/|R_w^p|$ [%] | 门限 [%] | 裁决 |
|---|---|---|---|
| 0.20 | 100 | 5 | FAIL |
| 0.30 | 100 | 5 | FAIL |
| 0.40 | 100 | 5 | FAIL |

远场阻力比压力积分小 **10–16 个数量级**（见 3.3 远场列）：压力积分与远场动量通量描述的**不是同一个流动**，
即自由面离散没有产生向下的辐射波系。这是结构性失败，不是"门限设太紧"。

### 3.6 消波区 — G6：NOT_SUPPORTED 及替代变量

仓内无海绵层/消波区实现（证据见第 2 节、工件 `damping_zone`）。下表只是**辐射条件离散化**（逆风阶数）的敏感性，
**不是消波区长度敏感性**，不得作为 ITTC 7.5-02-07-02.1 消波区条款的证据：

| Fn | $R_w^p$ 逆风 1 阶 [N] | $R_w^p$ 逆风 2 阶 [N] | 相对变化 [%] | 能量门 1 阶 [%] | 能量门 2 阶 [%] |
|---|---|---|---|---|---|
| 0.20 | -30930.6 | 2.57306e+05 | 112 | 100 | 100 |
| 0.30 | 9449.8 | 81127.5 | 88.4 | 100 | 99.9 |
| 0.40 | 17555.2 | 7.09391e+05 | 97.5 | 100 | 93.1 |

### 3.7 解析辐射边界条件（G5，**FAIL**）与参考量自身截断（G7，**FAIL**）

**G5。** $\nu=1.1111$ m⁻¹、源 $z=-T=-0.625$ m、$R\in[0.2,100]$ m：
标度残差**非单调**（`False`），对数斜率 **-0.8158**（预登记 −1.5±0.10）→ **FAIL**。
诊断（见 `diagnosis_of_failure`）：预登记拟合把 $\nu R\in[0.22,111]$ 的近场与远场混在一起，且传入核的是
**水平半径**而非三维距离（源在 $z=-T$），残差本不该在全程服从 $R^{-1.5}$。
**事后探索性**（明确标注非门限、不改变判决）：只取 $\nu R\ge 5$ 的子集，斜率 = **-0.9999**、单调 = `True` ——
仍不等于 −1.5，说明该检查的残差标度与教科书的深水理论预期**不一致，原因未查明**，作为遗留问题上报（不宣称通过）。

**G7。** Michell 积分波数截断（逐档 $R_w$ [N] 与相对最大档变化 [%]）：

| Fn | $k_{max}/k_0$ | $R_w$ [N] | $\Delta R_w/R_w$ [%] | 与本锚件 `Rw_reference` 偏差 [%] |
|---|---|---|---|---|
| 0.20 | 10 / 20 / 30 / 50 | 239.68 / 239.6 / 238.92 / 238.09 | 0.671 / 0.635 / 0.348 / 0 | 9145 |
| 0.30 | 10 / 20 / 30 / 50 | 300.8 / 302.48 / 302.32 / 301.62 | 0.272 / 0.284 / 0.234 / 0 | 2354 |
| 0.40 | 10 / 20 / 30 / 50 | 239.19 / 242.66 / 242.87 / 242.35 | 1.3 / 0.131 / 0.216 / 0 | 820.6 |

最大相对变化 1.3% > 0.5% → **FAIL**（诚实 FAIL，不调门限）。
注意 the study runs inside rankine_wave._michell_with_kmax, a separate and much cruder Michell implementation: its Rw already differs from the anchor reference by the factors listed in rel_deviation_vs_anchor_pct, so G7 quantifies ONLY the integration-range sensitivity of that kernel and must not be read as the truncation error of the anchor reference。

### 3.8 标度不变性与离散化归因（三问三答）

| 问题 | 做法 | 结果 | 结论 |
|---|---|---|---|
| 历史 L=2 m 记录能否复现？ | 完全按其文档化设置（hull 10×4、ppw 12、2.0L/1.0L、L=2 m）重跑 | $R_w$ = 2.854475588 N vs 记录 2.854475588 N（指针 `/free_surface_study/levels/2/rw_pressure_N`）→ 复现 = **True**（未知量 1369） | 调用路径正确，后续结论不是"用错 API"造成的 |
| 方法是否标度不变？ | 同一 Fn/ppw/域因子，L=2 m vs L=10 m | $C_w$ 比 = **0.999999999998**（1% 内 = `True`） | **标度不变**：量纲化正确，问题不在几何尺度 |
| 那 26 倍摆动来自哪里？ | 域**完全固定** 2.0L/1.0L（L=10 m），只换船体面元 10×4 → 24×8 | $C_w$：0.012621 → 0.33443，比 **26.5**；$C_w/C_w^{Michell}$：29.03 → 769.1 | **船体面元离散化**主导（域只贡献 ≤0.1%，标度贡献 1e-12）→ 该方法在船体面元上不收敛 |

历史记录本身也写着 `convergent = false`、$C_w/C_w^{Michell}$ 在 ppw 5/8/12 下为 9.52 / −32.62 / 29.03（**变号**），
与本单的归因一致：**"29×" 与 "769×" 都只是未收敛方法在不同离散下的取值，不能任选其一当结果**。

### 3.9 判据裁决（逐条量化）

| 判据 | 裁决 | 量化依据 |
|---|---|---|
| 1) 锚件抄录完整率 100%（字段级 JSON 指针） | **PASS** | 220/220 叶节点 = 100.00%；每叶 RFC-6901 回读值+类型比对，失败 0 项；9 类必备字段全在 = `True` |
| 2) 对比表逐项相对偏差 + 参照类型标注（禁自算当试验） | **PASS** | 7 行全部带相对偏差与参照类型；`experiment` 行 = 0；试验锚 = **WITHHELD** |
| 3a) 域截断逐档落表（交付要求） | **PASS** | 横向 4 档 × 3 Fn、下游 3 档 × 3 Fn、辐射条件替代变量 2 档 × 3 Fn 全部落表并各带门限裁决 |
| 3b) 截断门限的物理裁决 | **FAIL** | G1 PASS（0.0963% ≤ 2%）、G2 PASS（0.0544% ≤ 2%）、**G7 FAIL**（1.3% > 0.5%）；能量门 **FAIL**（100% >> 5%）；精度 **FAIL**（最好一档偏差 6.66e+04%）；消波区 **NOT_SUPPORTED** |

## 4. 结果图

![Fig. 1](figures/w20_wigley_truncation_domain.png)

*Fig. 1  Wigley III at L = 10 m: relative change of the truncated-domain wave resistance against the largest domain, lateral (left) and downstream (right); dashed line = pre-declared 2 % gate.*

![Fig. 2](figures/w20_wigley_energy_and_accuracy.png)

*Fig. 2  Left: radiation-energy gate |Rw_p - Rw_f|/|Rw_p| at the reference domain (5 % gate, grossly failed at all three Froude numbers). Right: Cw of the truncated-domain Rankine solve against the Michell reference at identical hull/fluid caliber; annotations are relative deviations.*

![Fig. 3](figures/w20_wigley_gates_and_mesh.png)

*Fig. 3  Left: scaled Sommerfeld residual of the deep-water Green function (analytic radiation condition, G5). Centre: truncation of the Michell integration range (G7). Right: mesh convergence of the Wigley hydrostatics against closed-form references (analytic).*

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **"仓内参考"≠"试验锚"。** 兴波阻力锚件的 `Rw_reference` 是 Michell 薄船理论收敛求积（numerical），
   静水力 `/reference/*` 是闭式解（analytic）。W18 exp#1 缺的是**试验**表，本单没有补上，只把它显式标为
   WITHHELD——诚实欠账不粉饰。
2. **exp#2/#3 不可得 → 重跑而非换算。** 工作树内检索不到 W18 exp#2/#3 工件，故同口径对比用仓内机制**重算**；
   工件与报告中不出现任何 exp#2/#3 数字。若日后取到，只需换算成 $C_w=R_w/(0.5\rho U^2LT)$ 后套用本工件
   的 `predeclared_gates` 与 tier 结构即可直接比对。
3. **"域够大"绝不等于"算对了"。** G1/G2 通过（0.10%/0.05%）而 G3 能量门 100%、G4 偏差 ≥6.66e+04%：
   截断收敛只是必要条件。这是本单最重要的教训，也是对"收敛性检查=正确性"这类自报的直接反驳。
4. **先复现再归因。** 第一版归因错了方向：我最初用默认 hull 24×8 复跑历史配置（未按其 hull 10×4），
   得出"方法不标度不变（差 26 倍）"的错误假设；按文档化设置重跑后复现到 1e-9 一致，再做**域固定、只换面元**
   的归因实验，才定位到离散化。教训：**归因前必须先复现基线**，否则会把配置差异当成物理缺陷。
5. **消波区在本仓不存在。** 任何"消波区长度敏感性"表述都是虚构；仓内只有截断域 + 逆风辐射条件。
   若论文需要 ITTC 消波区/域长波长比条款，须在 TensorBEM 内**新实现**海绵层（未做，后续项）。
6. **稠密求解限制档位。** $O(N^3)$ 无迭代加速，各 Fn 的 ppw 不同（4/12/12），
   跨 Fn 不可比绝对量，只能比**同一 Fn 内**的截断变化率。
7. **G5 检查本身可能有问题。** 预登记判 FAIL；事后子集斜率仍非 −1.5，但"为什么"未查明——不作为通过申报，
   也不在本文中臆测原因（可能是核的定义含镜像项，或残差标度定义与水平半径混用）。

## 6. 误差判定（关键量 vs 基准，逐项回答"是否 < 3%"）

| 关键量 | 基准 | 参照类型 | 相对偏差 | < 3%？ |
|---|---|---|---|---|
| $C_b$（生产网格 64×32） | 闭式 4/9 | analytic | 0.0305% | **是** |
| 排水体积（64×32） | $4LBT/9$ | analytic | 0.0305% | **是** |
| $KB$（64×32） | $5T/8$ | analytic | 0.00366% | **是** |
| 湿表面积（64×32） | 闭式求积 | analytic | 0.00284% | **是** |
| 域截断 G1/G2 | 最大域（自洽） | 自洽 | 0.0963% / 0.0544% | **是** |
| 参考量积分范围 G7 | 最大 $k_{max}$（自洽） | 自洽 | 1.3% | **否**（> 0.5%） |
| $C_w$ @ Fn 0.20/0.30/0.40 | Michell 收敛求积 | numerical | 1.201e+06% / 7.677e+04% / 6.659e+04% | **否**（4–6 个量级，Fn 0.20 变号） |
| 辐射能量门 G3 | 能量守恒（pressure ≡ far field） | 自洽 | 100% | **否** |
| 标度不变性 | 自身 L=2 m vs L=10 m | 自洽 | abs(Cw比−1) = 1.51e-12 | **是** |
| Sommerfeld 斜率 G5 | −1.5 | analytic | -0.8158（偏差 45.61%） | **否** |

静水力与截断收敛在 3% 内（解析/自洽基准）；**截断域 Rankine 求解器的精度与辐射能量守恒均不达标**，
属结构性能力缺口，不能靠调门限或换窗口掩盖。

## 7. 结论与后续

* **判据 1 PASS**：220/220 字段级抄录（RFC-6901 指针 + 回读校验 + 9 类必备字段）。
* **判据 2 PASS**：7 行对比全部带相对偏差与参照类型；`experiment` 行 = 0，试验锚 = WITHHELD。
* **判据 3**：**落表要求 PASS**（3a），**物理裁决 FAIL**（3b：G1/G2 PASS，G7/G3/G4 FAIL，消波区 NOT_SUPPORTED）。
* **能力缺口（建议后续项）**：(i) 消波区/海绵层实现（ITTC 7.5-02-07-02.1）；(ii) 自由面辐射条件的稳定离散
  （当前逆风格式在船体面元 10×4→24×8 下 $C_w$ 变 26.5× 且能量门 100%）；(iii) G5 残差标度的定义核查；
  (iv) 取得 Wigley 试验 RAO/阻力表以补齐真试验锚。

**诚实声明（未完成部分，均已在工件中机器可读登记）。** ① 试验锚仍缺（源不可达，原样 WITHHELD）；
② W18 exp#2/#3 工件未取到，未做换算对比，改为同口径重算；③ 消波区敏感性因仓内无实现而 NOT_SUPPORTED，
只给替代变量；④ G7 为诚实 FAIL（1.30% > 0.5% 门限），未调门限；⑤ G5 FAIL 且原因未查明，未宣称通过。
