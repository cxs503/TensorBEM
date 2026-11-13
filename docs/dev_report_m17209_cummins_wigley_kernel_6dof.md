# CASE-BEM-CUMMINS-WIGLEY：Wigley III 船 6 自由度 Cummins 记忆核（m17209）

**仓** TensorBEM · **算例** CASE-BEM-CUMMINS-WIGLEY（算例池） · **判定** `FAIL:R1,R3,R4,R5`
**工件** `artifacts/cummins_wigley_kernel_6dof/results.json`、`sweep_coarse.json`、`sweep_medium.json`
**复现**
```
PYTHONPATH=src python scripts/m17209_cummins_wigley_kernel_6dof.py --stage sweep   # BEM 扫频（分钟级）
PYTHONPATH=src python scripts/m17209_cummins_wigley_kernel_6dof.py --stage gates   # 五道闸门（秒级）
PYTHONPATH=src python -m pytest tests/test_cummins_wigley_kernel_6dof.py -q        # 4 passed
```

## ① 问题 + ref 出处

**问题**：对一艘有航速修形的船型（水面穿透、非流线型截面）而言，TensorBEM 的频域辐射解
A(ω)、B(ω) 经 Cummins 方程转到时域后，记忆核 K(t) 与自由衰减是否在 6 个自由度上都成立？

$$
\left(M + A_\infty\right)\ddot{x} + \int_0^t K(t-s)\,\dot{x}(s)\,ds + Cx = F(t),
\qquad K(t) = \frac{2}{\pi}\int_0^\infty B(\omega)\cos(\omega t)\,d\omega
$$

**参考值（全部运行时读入，零手抄）**

| 编号 | 参考 | 来源（仓内可核） |
|---|---|---|
| R1 | 独立求解器 Capytaine 3.1 的 A_ii、B_ii（ω=1 rad/s，6 DOF，三级网格） | `benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`（内含 python 3.11.6 / numpy 2.4.6 / capytaine 3.1.dev0 / loongarch64 / ρ=1025 / g=9.80665 / 水深远洋 / 旋转中心 (0,0,−0.09375) 的完整环境记录） |
| R2 | Cummins 恒等式 `K(0)=(2/π)∫B dω` | 上式本身；程序内以同一 B(ω) 的梯形积分对照采样核值 |
| R3 | Kramers-Kronig 因果 `A(ω)−A(∞)=Hilbert[B](ω)` | `tensorbem.causality.kkr_added_mass`（主值积分，取扫频中点） |
| R4 | 时域自由衰减阻尼比 vs 频域半功率阻尼比 | 同一 A、B 的两种独立估计（Cummins 卷积 / 频响半功率） |
| R5 | 往返闭环 B→K→B | `tensorbem.time_domain.fourier_round_trip`（自带有效频带） |

**诚实声明**：仓内**没有任何数字化发表的 K(t) 表或 Wigley III 辐射系数表**；且仓的出处闸门
`tensorbem.reference_provenance` 把"回忆值"一律记为债务（不算证据，见
`docs/dev_report_m14725_reference_provenance.md`，其中 15 个文献常数 14 个为回忆值）。因此本算例
**不引用任何"原文表格页码"**——编造页码比没有页码更坏。R1 是独立求解器锚点（唯一能发现频域
解的**系统性**偏差的参考，但只覆盖 ω=1 rad/s 单频点）；R2–R5 是 Cummins/KK 形式体系自带的精确
恒等式，**只能发现实现与采样误差，不能发现 BEM 本身的系统偏差**——这一局限写进 §⑤。

## ② 方法与实现

1. **几何**：Wigley III 数学船型（`tensorbem.wigley3_freq.wigley3_wetted_hull_mesh`），
   L=3.0 m，B=0.3 m，T=0.1875 m，c=1.1088，**Cb=0.5430**，
   ∇=0.09163125 m³，L/B=10.0，B/T=1.6，ρ=1025 kg/m³，g=9.80665 m/s²。
   几何闸门（Cb、∇ 与 L·B·T·Cb 一致性）在 `wigley3_hydrostatic_reference` 内已由既有测试守护。
2. **频域**：3-D 面元 BEM（Kelvin 格林函数，`solve_radiation_3d`，`chief_points="auto"`），
   ω ∈ {0.5, 0.7, 1, 1.5, 2.2, 3.2, 4.2, 6} rad/s（无量纲
   ω√(L/g) ∈ [0.277, 3.319]），
   一次求解同时给出 6×6 的 A、B。
3. **时域**：`radiation_memory_kernel` 由 B(ω) 的余弦变换给出 K(t)（dt=0.02 s，
   τ_max=20.0 s）；`cummins_free_decay` 解 Cummins 方程自由衰减（heave，
   质量 = ρ∇、恢复力 C33 = ρgA_wp 均由几何算出，**无经验回转半径**）；`half_power_damping`
   由同一 A、B 给出频域阻尼比。
4. **闸门**：`cummins_wigley_kernel_6dof.run_case` 产出 R1–R5，预声明容差见
   `GATES`：R1/R3/R4 = **10 %（本算例容差）**，R2 = 2 %，R5 = 5 %。

## ③ 参数表（无量纲数完整）

| 量 | 值 |
|---|---|
| L / B / T | 3.0 / 0.3 / 0.1875 m |
| L/B，B/T | 10.0，1.60 |
| Cb（方块系数） | 0.5430 |
| ∇ | 0.09163125 m³ |
| ρ，g | 1025 kg/m³，9.80665 m/s² |
| 排水质量 | 93.9220 kg |
| C33（heave 恢复力） | 7368.5 N/m |
| A_wp | 0.733050 m² |
| 频率节点数 | 8 |
| ω 范围 / ω√(L/g) | 0.5–6 rad/s / 0.277–3.319 |
| 网格（门禁档） | n_sections=20,n_levels=10，800 面元 |
| 核采样 | dt=0.02 s，τ_max=20.0 s |

## ④ 对照表（ref vs got + 误差打勾）

**R1 独立求解器对照（ω = 1 rad/s，n_sections=20,n_levels=10）**

| DOF | A_ref (Capytaine) | A_got (TensorBEM) | 误差 | B_ref | B_got | 误差 |
|---|---|---|---|---|---|---|
| surge | 2.585 | 2.52 | -2.52 % ✓ | 0.002354 | 0.002304 | -2.12 % ✓ |
| sway | 138.1 | 136.7 | -1.02 % ✓ | 0.01367 | 0.01352 | -1.09 % ✓ |
| heave | 197.4 | 194.7 | -1.36 % ✓ | 25.95 | 25.63 | -1.23 % ✓ |
| roll | 0.67 | 0.2982 | -55.50 % ✗ | 6.049e-05 | 1.056e-05 | -82.54 % ✗ |
| pitch | 53.04 | 50.86 | -4.12 % ✓ | 0.04218 | 0.03993 | -5.34 % ✓ |
| yaw | 91.12 | 91.74 | +0.68 % ✓ | 1.353e-05 | 1.376e-05 | +1.72 % ✓ |

**五道闸门（逐 DOF 明细）**

| DOF | R1 A 误差 | R1 B 误差 | R2 K(0) 误差 | R3 KK 误差 | R5 往返误差 | A_∞ |
|---|---|---|---|---|---|---|
| surge | -2.52 % | -2.12 % | +0.00 % | +42.13 % | +43.92 % | 2.0169 |
| sway | -1.02 % | -1.09 % | +0.00 % | +224.26 % | +242.72 % | 156.57 |
| heave | -1.36 % | -1.23 % | +0.03 % | +42.28 % | +35.07 % | 37.653 |
| roll | -55.50 % | -82.54 % | +0.00 % | +169.89 % | +181.42 % | 0.32175 |
| pitch | -4.12 % | -5.34 % | +0.00 % | +37.67 % | +15.77 % | 34.45 |
| yaw | +0.68 % | +1.72 % | +0.00 % | +141.45 % | +150.05 % | 118.24 |

| 闸门 | 名称 | 值 | 容差 | 判定 |
|---|---|---|---|---|
| R1 | R1_cross_solver | 0.555 | 0.10 | ✗ |
| R2 | R2_kernel_at_zero | 0.0003179 | 0.02 | ✓ |
| R3 | R3_kramers_kronig | 2.243 | 0.10 | ✗ |
| R4 | R4_free_decay | nan | 0.10 | ✗ |
| R5 | R5_round_trip | 2.427 | 0.05 | ✗ |

**R4 自由衰减（heave）**：ζ_decay = 0.00439，ζ_half-power = nan，
相对误差 n/a ✗；
周期 0.7720 s，衰减率 0.03575 1/s，
对数幅值拟合 R² = 0.99875。

**最差 DOF**：A_ii 相对 roll（+55.50 %）。总判定 `FAIL:R1,R3,R4,R5`。

## ⑤ 收敛性与参数敏感性（两点）

1. **网格阶梯**（同一 ω=1 rad/s 的 A_ii 对角）：

| 档 | 标签 | 面元 | A_11 | A_22 | A_33 | A_44 | A_55 | A_66 |
|---|---|---|---|---|---|---|---|---|
| medium | n_sections=20,n_levels=10 | 400 | 2.52 | 136.7 | 194.7 | 0.2982 | 50.86 | 91.74 |

2. **核采样敏感性**（K(t) 是 B(ω) 的积分，采样步长与记忆窗长决定分辨率）：
   dt 0.02 → 0.01 s 时 K(0) 相对变化
   +0.00 %；τ_max 20 →
   10 s 时 K(0) 相对变化
   +0.00 %；
   往返闭环误差 +35.07 % → +35.07 %
   （dt 减半）。**频带截断不可测**：`validate_radiation_sweep` 要求 ≥8 个频率节点，
   8 节点扫频砍掉高频尾巴即被拒（记为生态摩擦）。

**局限**：R2–R5 用的是同一份 BEM 扫频，彼此独立的是"数学路径"而不是"物理模型"；能发现频域解的
系统偏差的只有 R1，而 R1 只有 ω=1 rad/s 单频点、且 Capytaine 与该 BEM 都是势流面元法。
A_∞ 由尾部外插给出（尾部分数 None），
这与 ω_max 直接耦合，是本算例最大的单点不确定源。

## ⑥ 失败模式与工件

* **honest 记录**：最差 A_ii 相对误差 +55.50 %（roll）；判定 `FAIL:R1,R3,R4,R5`。
  逐 DOF 的失败项见上表，未做任何覆盖或剔除。
* **已知陷阱**（本单踩到，已写进模块 docstring）：
  `kkr_added_mass` 要求采样点**不与节点重合**且**不在扫频端点**，否则主值积分抛
  `ValueError`——R3 因此在扫频中点评估，并对实测 A 做线性插值比较；
  往返闭环的相对误差在 A(ω)−A(∞) 过零点处无定义，R5 改用按量级归一化并只在
  `fourier_round_trip` 自报的有效频带内统计。
* **工件**：`artifacts/cummins_wigley_kernel_6dof/results.json`（全部数字的唯一来源）、
  `sweep_coarse.json` / `sweep_medium.json`（BEM 原始输出）、
  `docs/fig_m17209_cummins_kernels_6dof.png`、`docs/fig_m17209_cross_solver_6dof.png`、
  `docs/fig_m17209_gates_6dof.png`。
* **生态摩擦**：①仓内无任何数字化文献表（Journée/WAMIT 均为回忆值且已被证伪，见
  `benchmarks/verified/wigley3_recalled_ref/results.json` 的 `adjudication` 中
  `verdict: REFUTED`），算例池要"原文表格"参考时无路可走——需要数字化手册工件注入通道；
  ②`kkr_added_mass` 的端点/节点限制未在文档中声明，使用者必须先踩一次异常；
  ③平台 knowledge_base 的 `/knowledge/search` 对本域关键词（cummins/kramers/wigley/BEM）
  命中 0 条（`artifacts/cummins_wigley_kernel_6dof/kb_probe.json`），开工三查实际上查不到东西。

## 图

![fig_m17209_cummins_kernels_6dof.png](fig_m17209_cummins_kernels_6dof.png)

*Normalised Cummins memory kernels K_ii(tau)/K_ii(0) for the six DOF of the Wigley III hull, from K(t)=(2/pi)int B(w)cos(wt)dw.*

![fig_m17209_cross_solver_6dof.png](fig_m17209_cross_solver_6dof.png)

*TensorBEM vs Capytaine 3.1 diagonal radiation coefficients at omega = 1 rad/s (log scale); values loaded from the committed reference file.*

![fig_m17209_gates_6dof.png](fig_m17209_gates_6dof.png)

*Left: Cummins-equation heave free decay. Right: per-DOF gate residuals (R1 independent solver, R3 Kramers-Kronig, R5 band-limited round trip) against the 10% case tolerance.*

