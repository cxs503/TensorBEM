# Wigley 波阻双路径审计：Kochin 路径不是力，压力路径给出负阻力

**任务**：m3125（selfstart-*，infrastructure 席自主工时） · **分支**：agent/m3125 · **仓**：TensorBEM
**基线**：`1e40b09`（origin/main，含 PR#110–#115）
**工件**：`src/tensorbem/benchmarks/wigley_normalization_audit.py`、
`tests/test_wigley_normalization_audit.py`（12 项全部通过）、
`docs/m3125/outputs/wigley_normalization_audit.json`、
`docs/m3125/outputs/tables.md`（本文所有表格由脚本生成，零手抄）、
`docs/m3125/figures/{fig_dimensional_response,fig_pressure_mesh}.png`

---

## 1. 案例介绍（为什么挑这一件）

`docs/wigley_wave_resistance.md` 里躺着一条**尚未结清的 honest FAIL**：

> ❌ Two-path agreement within 5% (honest FAIL: 103-284% discrepancy over the mesh ladder)

以及一行自我辩解：「文档表格无法裁决，因为混用了两个参考面积」。这两句话的共同点是——
它们都只在**比数字**（收敛、网格、参考面积），而从来没有问过**这两个路径输出的量纲是什么**。
本单把这条欠账按「验证欠账」处理：不去调参救数字，而是检验已合并功能是否满足最基本的
**量纲齐次性**与**物理符号**约束。判据先于动手，且两条都与任何参考值无关：

* **判据 A（量纲）**：波阻是力。一个公式是量纲齐次的，当且仅当它在新单位制下按
  `kg m s^-2` 的比例响应。这与参考值无关，纯属实现性质。
* **判据 B（符号）**：稳速前进的船体对波场做功，故 `Rw ≥ 0`。负波阻是物理不可能。

**物理背景与几何**：数学 Wigley 船型（薄船），半宽
`y(x,z) = (B/2)(1-(2x/L)²)(1-(z/T)²)`，`x∈[-L/2,L/2]`、`z∈[-T,0]`，对称于 `y=0`。
两条路径同源于 Michell 薄船理论：Kochin 远场路径积 `|H(k₀λ)|²/(λ²√(λ²-1))`（`k₀=g/U²`），
压力路径在船体表面 ∫p·n_x dS。二者共享 Kochin 函数 `H(k)=2∬Y_x e^{kz+ikx}dxdz`。

| 参数 | 值 |
|---|---|
| `L, B, T` | 2.0 m, 0.2 m, 0.125 m（`L/B=10`, `B/T=1.6`，标准 Wigley） |
| `ρ, g` | 1025 kg/m³, 9.80665 m/s² |
| 湿表面积 `S_wet` | 0.5949 m²（由网格程序化计算） |
| Froude 阶梯 | 0.20 / 0.25 / 0.30 / 0.35 / 0.40 |
| 压力路径网格 | 128 / 512 / 1152 / 2048 / 4608 面元 |

---

## 2. 怎么做的（方法与关键设置）

1. **单位制重标定（判据 A 的实现）**：质量单位 ×5、长度单位 ×2、时间单位 ×3，于是输入变为
   `ρ→ρ·5/2³`、`U→U·2/3`、`g→g·2/3²`、`L→L·2`；力应响应 `5·2/3² = 1.1111`。
   分别对三条单位轴做单轴重标定，解出输出的 `(kg, m, s)` 指数。**不用局部对数导数**：
   `U` 与 `g` 以 `k₀=g/U²` 进入 `H` 的自变量，局部导数会混入 Froude 依赖，
   实测 `d ln Rw/d ln U = -0.69` 而非预因子的 `-2`——这一点第一版实现踩过（见 §5 坑 1）。
2. **几何相似测试**：船体放大 `s=2`、`U→U√2`（同 Fr 同流体），力应涨 `s³`。这条**两条路径都通过**
   （1e-15），恰好解释了为什么历年收敛研究发现了问题。
3. **Froude 阶梯**：记录两路径 `Rw`、`Cw`（两种参考面积）与量纲强制的修正带 `g²/C`。
4. **压力路径网格阶梯**：`128→4608` 面元，检验是否有收敛值。
5. **把结论写成可执行判据**：`tests/test_wigley_normalization_audit.py` 12 项，
   分别钉住两路径的量纲指数、相似性、符号、网格非收敛、以及「文档 `Cw` 低于最小可采信值」。
   一旦有人修好缺陷，`test_kochin_path_is_not_homogeneous` 会**故意变红**，逼文档同步。

工具链：`numpy` + 仓内 `tensorbem.wigley`（未改动）+ `matplotlib`（Agg）；全程约 26 s，
属轻量（入场 load15=115 ≥100，按容量纪律不起重算力）。

---

## 3. 计算结果（表格全部由 `docs/m3125/make_figure_and_tables.py` 程序化生成）

### T1. Unit-rescaling test (the decisive measurement)

| path | measured response `sm^a sl^b st^c` | force requires | homogeneous? |
|---|---|---|---|
| kochin | `sm^1 sl^-1 st^2` = kg m^-1 s^2 | `sm sl st^-2` = kg m s^-2 | **NO** |
| pressure | `sm^1 sl^1 st^-2` = kg m s^-2 | `sm sl st^-2` = kg m s^-2 | yes |

unit change used: mass x5, length x2, time x3; Kochin ratio = 22.5 vs required 1.11111 (ratio of ratios = 20.25 = $g^2$-scaled), pressure ratio = 1.11111.

### T2. Geometric-similarity test (why no convergence study could see it)

| path | $R_w$ base [N] | $R_w$ at $s=2$, $U\to U\sqrt2$ [N] | measured ratio | required $s^3$ | rel. error |
|---|---|---|---|---|---|
| kochin | 0.0983411 | 0.786728 | 8.0000000000 | 8.0 | 4.44e-16 |
| pressure | -0.0203003 | -0.162402 | 8.0000000000 | 8.0 | 6.00e-15 |

Both paths pass: geometric similarity is blind to the defect.

### T3. Froude ladder ($\rho=1025$, $g=9.80665$, $L=2$, $B=0.2$, $T=0.125$, $S_{wet}=0.5949$ m$^2$)

| $Fr$ | $U$ [m/s] | $R_w$ Kochin [N] | $R_w$ pressure [N] | pressure/Kochin | $C_w$ Kochin (doc) | $C_w$ pressure |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.8857 | 2.0601e-02 | -4.9616e-03 | -0.241 | 8.613e-05 | -2.074e-05 |
| 0.25 | 1.1072 | 2.8774e-02 | -1.1411e-02 | -0.397 | 7.699e-05 | -3.053e-05 |
| 0.30 | 1.3286 | 9.8341e-02 | -2.0300e-02 | -0.206 | 1.827e-04 | -3.772e-05 |
| 0.35 | 1.5500 | 3.7499e-02 | -2.9020e-02 | -0.774 | 5.119e-05 | -3.962e-05 |
| 0.40 | 1.7715 | 2.1059e-01 | -4.1521e-02 | -0.197 | 2.201e-04 | -4.340e-05 |

### T4. Dimensionally forced correction band for $C_w$ (wetted area)

| $Fr$ | $C_w$ as implemented | $g^2/C$, $C=4\pi$ | $C=2\pi$ | $C=\pi$ | $C=4$ | $C=2$ | $C=1$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 8.613e-05 | 6.592e-04 | 1.318e-03 | 2.637e-03 | 2.071e-03 | 4.142e-03 | 8.283e-03 |
| 0.25 | 7.699e-05 | 5.892e-04 | 1.178e-03 | 2.357e-03 | 1.851e-03 | 3.702e-03 | 7.404e-03 |
| 0.30 | 1.827e-04 | 1.398e-03 | 2.797e-03 | 5.594e-03 | 4.393e-03 | 8.787e-03 | 1.757e-02 |
| 0.35 | 5.119e-05 | 3.918e-04 | 7.835e-04 | 1.567e-03 | 1.231e-03 | 2.462e-03 | 4.923e-03 |
| 0.40 | 2.201e-04 | 1.685e-03 | 3.369e-03 | 6.738e-03 | 5.292e-03 | 1.058e-02 | 2.117e-02 |

### T5. Pressure-path mesh ladder ($Fr=0.3$)

| $n_x\times n_z$ | panels | $R_w$ pressure [N] | $R_w$ Kochin [N] | pressure/Kochin |
|---|---:|---:|---:|---:|
| 8$\times$4 | 128 | -4.6375e-02 | 9.8341e-02 | -0.4716 |
| 16$\times$8 | 512 | -2.0300e-02 | 9.8341e-02 | -0.2064 |
| 24$\times$12 | 1152 | -8.0619e-03 | 9.8341e-02 | -0.0820 |
| 32$\times$16 | 2048 | -7.1713e-03 | 9.8341e-02 | -0.0729 |
| 48$\times$24 | 4608 | -5.7314e-03 | 9.8341e-02 | -0.0583 |


---

## 4. 结果图

**图 1** `fig_dimensional_response.png` — 左：文档 `Cw`（红）与量纲强制修正带 `g²/C, C∈[1,4]`
（蓝色阴影）；文档值整体落在带的下方一个数量级以上。右：单位重标定测得的三轴指数，
黑虚线为力的目标 `(1,1,-2)`；Kochin 路径（红）落在 `(1,-1,2)`。
**图 2** `fig_pressure_mesh.png` — 压力路径 `Rw` 随面元数：始终为负，绝对值单调下降而无极限，
Kochin 参考线（红虚线）在零线另一侧。

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **坑：局部对数导数不等于预因子指数。** 第一版实现用 `d ln Rw/d ln x` 直接读指数，测出
   `U^-0.69`、`L^3.35`，与预因子 `U^-2 L^4` 不符——因为 `k₀=g/U²` 把 Froude 依赖带进了
   `H` 的自变量。判据必须用**单位制重标定**（`g` 一起变），这时 Froude 数是不变量，测量才干净。
   已把这段写进模块 docstring。
2. **几何相似测试是盲的。** 缩放 + 同 Fr 下两个路径都精确给 `s³`：U 幂次错误（`U^-2` vs `U^+2`）
   恰好被 `L²` 的同次缩放掩盖。**任何只做几何缩放或网格收敛的研究都不可能发现本缺陷**——
   这是本次审计存在意义的核心理由。
3. **缺失因子只能是 `g²/C`。** 量纲账：实现预因子 `4ρ/U²` 的输出量纲 `kg s² m⁻¹`，
   需要 `kg m s⁻²`，比值量纲 `m² s⁻⁴`；在 `(ρ, U, g)` 中只有 `g²` 具有该量纲。
   故缺失因子 = `g²/C`，`C` 无量纲。数值上 `g²=96.17`，`C∈[1,4]` 给出 **24–96 倍**的低估。
   注意：这是**量纲强制**的结论，不依赖任何文献值。
4. **`C` 本仓钉不住。** 唯一可能独立定 `C` 的是压力路径，但它符号错且不收敛（T5），
   不能用来定标；本次也没有从一次文献核到 Wigley `Cw` 的权威数值，因此**不宣称绝对量**（WITHHELD）。
5. **适用范围**：本审计只针对薄船 Michell/Neumann-Kelvin 层；结论与网格、积分点数、参考面积无关，
   因为这些都改变不了量纲。
6. **未做 / 后续**：未修 `wave_resistance_pressure` 的符号与谱表示（薄船近场需完整 Green 函数，
   只保留横向波实 `k` 积分不成立），未改 `michell_wave_resistance` 的预因子（改数须先定 `C`），
   未做文献 `Cw` 裁决。这三项已在本单登记为后续线索，**不以调参方式凑合**。

---

## 6. 误差判定（关键量 vs 基准，逐项显式回答）

| 关键量 | 基准 | 实测 | 误差 | 是否 <3%？ |
|---|---|---|---|---|
| Kochin 路径输出量纲 | 力 `kg m s^-2` | `kg s^2 m^-1` | **量纲不等**（不可用百分比表达） | **否（hard fail）** |
| Kochin 单位响应比 | `sm sl st^-2` = 1.1111 | 22.5 | 20.25×（= `sl^-2 st^4`） | **否** |
| 压力路径输出量纲 | 力 `kg m s^-2` | `kg m s^-2` | 0（精确，1e-15） | 是 |
| 压力路径符号 | `Rw ≥ 0` | `Rw < 0`（5/5 Fr，且 128–4608 面元全负） | 符号错 | **否（hard fail）** |
| 压力路径网格收敛 | 有收敛值 | `|Rw|`: 4.64e-2→5.73e-3，末两级仍差 20% | 无极限 | **否** |
| 文档 `Cw`(Fr=0.30, `S_wet`) | 量纲强制最小可采信 1.40e-3 | 1.83e-4 | 至少 7.7× 偏低（`g²/C`） | **否** |
| 几何相似 `Rw` 比值 | `s³`=8 | Kochin 8.0000000000 / 压力 8.0000000000 | 4e-16 / 6e-15 | 是（但**盲**，见坑 2） |

**判定**：本单不是「文档数字偏差」类的软失败，而是两条**硬失败**——
Kochin 路径输出不是力（量纲不齐），压力路径输出是负力（符号错）。
因此 `docs/wigley_wave_resistance.md` 中 103–284% 的「双路径差异」**不能当作收敛判据**：
它的绝对值运算把符号错抹掉了（`|0.0983-(-0.0203)|/0.0983 = 121%`，正落在该带内）。
绝对 `Cw` 一律 **WITHHELD**，直到 `C` 被独立基线钉住。

---

## 7. 复现

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.wigley_normalization_audit --outdir docs/m3125/outputs
PYTHONPATH=src python3 docs/m3125/make_figure_and_tables.py
PYTHONPATH=src python3 -m pytest tests/test_wigley_normalization_audit.py -q
```