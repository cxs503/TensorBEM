# CASE-BEM-CYL-DRIFT-ASYMP — 圆柱平均漂移力与 (ka)³ 律

## 1. 案例介绍（问题 + ref 出处）

**锚点 `CASE-BEM-CYL-DRIFT-ASYMP`**（算例池，owner 2026-11-12 令）。物理背景：深水（`ω²=gk`）中**坐底固定竖直圆柱**（半径 `a`，吃水到水底、穿出自由面）在单色入射波正横（β=0）下的**二阶平均漂移力** `F̄_x`。无量纲目标量

`C(ka) = F̄_x /(ρ g A² a)`，长波极限律 `F̄_x = (5π²/16) ρ g A² a (ka)³`，其中 `5π²/16 = 3.0842513753`。

**ref 出处（逐条可查，PDF 由本单抓取并记 sha256，抽取文本存工件）**

| 参考 | 出处 | 位置 | 抓取凭据 |
|---|---|---|---|
| MacCamy-Fuchs 衍射势（`A_m = -i^m J_m'(ka)/H_m'^(1)(ka)`） | MacCamy, R.C. & Fuchs, R.A. (1954) Wave forces on a pile: a diffraction theory. Technical Memorandum No. 69, Beach Erosion Board, US Army, 17 pp.（原文）；本单逐字核对用的是复印本：Chakrabarti, S.K. & Gupta, A. (2007) Steady wave drift force on basic objects of symmetry. WIT Transactions on the Built Environment, Vol. 92 (Fluid Structure Interaction and Moving Boundary Problems IV), WIT Press, pp. 1-11. | 4.1 Vertical cylinder, Eq. (3)，PDF 第 3 页 | `sha256=c8506effb013fcd8…`，URL `http://www.witpress.com/Secure/elibrary/papers/FSI07/FSI07001FU1.pdf` |
| Maruo 远场漂移力（约化、与辐射半径无关） | Maruo, H. (1960) On the increase of the resistance of a ship in rough seas (the drift force). J. Soc. Nav. Archit. Japan 107, 59-67.（原文）；逐字核对复印本：McIver, P. (1986) Mean drift forces on arrays of bodies due to incident long waves. Brunel University, Department of Mathematics, Report TR/13/86. | 5 The mean drift force, Eqs. (5.1)-(5.2)，PDF 第 15 页 | `sha256=d24ff37ec34abb78…`，URL `http://bura.brunel.ac.uk/bitstream/2438/2010/1/TR_13_86.pdf` |
| 长波闭式系数 `5π²/16` | 见下方"文献缺口"声明 | — | 表达式 `5*pi**2/16`，代码内求值 |

**文献缺口（诚实声明）**：`5π²/16` 这一**系数**未能在本沙箱内取得可逐字引用的原文版面（Maruo 1960 在 J. Soc. Nav. Archit. Japan 107，出版社全文不可达；Semantic Scholar 429、CORE 403、DuckDuckGo/Google 不可达、Bing 查询被截断为单词）。因此该系数在本单按**推导量**处理，并用三条互相独立的路径验证同一个数（第 2、4 节）：(i) 截断模渐近 + 解析核积分，(ii) 壁面动量积分（含水线项），(iii) `ka=0.01…0.4` 精确数值极限。**未完成项**：逐字引文与页码，留给第二轮（两轮法接棒）。

## 2. 怎么做的

三条互不共享代码的路线，全部复用 TensorBEM 现有能力，未另起炉灶：

1. **近场（MacCamy-Fuchs 动量积分）** — `cylinder_drift.near_field_cylinder_drift`：二阶平均压力积分 `F_i = ¼ρ∫|∇Φ̂|² n_i dS − ¼ρg∫|ζ̂|² n_i dl`，柱壁法向 `n=(cosθ,sinθ,0)`，深度积分解析 `∫_{-∞}^{0} e^{2kz}dz = 1/(2k)`，模幅 `A_m` 取上表 MF 闭式。作为符号/归一化锚，同一公式作用在"全反射直立壁"上必须精确返回 `ρgA²/2`（仓内测试 `tests/test_cylinder_drift_asymptotic.py::test_wall_anchor_reproduces_rho_g_A2_over_2`）。
2. **远场（Maruo 约化核）** — `cylinder_drift.far_field_cylinder_drift`：`F_x = (ρgA²/(2πk))∫₀^{2π}|S(θ)|²(1−cosθ)dθ`，`|S|` 由 MF 模解析求和；约化后**不含辐射半径**，因此不存在 m5242 记录的半径振荡失败模式。
3. **真实求解器路线（本单的"跑该算例"部分）** — `free_surface_solver3d.solve_diffraction_3d` 在**离散面板**圆柱上解线性绕射 → `drift_far_field3d.bem_kochin` 取散射 Kochin 函数 → `drift_far_field3d.maruo_far_field_drift` 得远场漂移力；同一 BEM 解上再算近场压力/水线积分。网格 `depth = 5/k`（`kd=5`，深度衰减 `e⁻⁵ = 0.7 %`），面板族 `2·n_θ·n_z + 1`。

工具链：`PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cyl_drift_asymptotic --json artifacts/case_cyl_drift_asymptotic/results.json --figures artifacts/case_cyl_drift_asymptotic`（集群作业 `cj1462`，单核，`timeout 5400 s`）。provenance 见工件 `run` 段（git `96f19f7ced`、python `3.11.6`、numpy `2.4.6`、host `36cb7828dbd5`、wall `750 s`）。

## 3. 参数表（无量纲数完整）

| 量 | 值 | 单位/说明 |
|---|---|---|
| ρ | 1025.0 | kg/m³（海水） |
| g | 9.81 | m/s² |
| a（圆柱半径） | 1.0 | m |
| A（入射波幅） | 1.0 | m |
| ka（无量纲波数） | 长波梯 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30；宽域梯 0.20, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00 | 无量纲 |
| kd（BEM 水深） | 5 | 无量纲（深水近似 `e⁻⁵=0.7 %`） |
| 波向 β | 0 | rad（正横，轴对称体所以与方向无关） |
| BEM 主力网格 | 32×12 → 800 面板 | 面板数 |
| Kochin 方位点数 | 180（敏感性到 360） | 方位格 |
| 容差判据 | 5 % | 每一项闸门 |

## 4. 对照表（ref vs got + 误差打勾叉）

**(a) 长波律**（ref = `5π²/16·ρgA²a(ka)³`；got = MF+Maruo 闭式精确求值）

| ka | ref（律）[N] | got（闭式）[N] | 误差 | 归一系数 `F̄/(ρgA²a(ka)³)` | <5 % |
|---|---|---|---|---|---|
| 0.02 | 0.248103 | 0.248074 | +0.012 % | 3.083891 | ✓ |
| 0.05 | 3.876615 | 3.872054 | +0.118 % | 3.080622 | ✓ |
| 0.10 | 31.012919 | 30.825211 | +0.605 % | 3.065584 | ✓ |
| 0.15 | 104.668600 | 103.045038 | +1.551 % | 3.036410 | ✓ |
| 0.20 | 248.103349 | 240.590682 | +3.028 % | 2.990859 | ✓ |
| 0.25 | 484.576854 | 459.820784 | +5.109 % | 2.926683 | ✗ |
| 0.30 | 837.348803 | 771.542829 | +7.859 % | 2.841865 | ✗ |

律的适用边界（程序扫 `ka=0.01…0.40` 找最后一个满足 <5 % 的点）：**`ka ≤ 0.24`**；`ka=0.30` 处已超差（7.86 %），仓内测试 `test_long_wave_law_breaks_above_band` 反向锁定该边界，不让它成为自由参数。

**(b) 真实 BEM 求解器 vs 闭式**（ref = 闭式；got = 面板 BEM）

| ka | 面板 | ref [N] | got 远场 [N] | 误差 | <5 % | got 近场 [N] | 误差 | <5 % |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 800 | 2875.657 | 2817.327 | +2.028 % | ✓ | 2700.769 | +6.082 % | ✗ |
| 1.00 | 800 | 6685.994 | 6614.598 | +1.068 % | ✓ | 6410.460 | +4.121 % | ✓ |
| 2.00 | 800 | 6306.398 | 6223.327 | +1.317 % | ✓ | 6000.597 | +4.849 % | ✓ |

**(c) 两条独立二阶公式互校**（同一闭式，两个不同的二阶公式）

| ka | 近场 [N] | 远场 [N] | 相对差 | `C(ka)` | 长波律偏差 |
|---|---|---|---|---|---|
| 0.20 | 240.590682 | 240.590682 | 2.84e-13 % | 0.023927 | +3.03 % |
| 0.50 | 2875.656993 | 2875.656993 | 3.16e-14 % | 0.285986 | +25.82 % |
| 0.70 | 5296.273595 | 5296.273595 | 1.72e-14 % | 0.526717 | +50.21 % |
| 1.00 | 6685.993519 | 6685.993519 | 4.08e-14 % | 0.664926 | +78.44 % |
| 1.50 | 6037.260100 | 6037.260100 | 0.00e+00 % | 0.600409 | +94.23 % |
| 2.00 | 6306.398054 | 6306.398054 | 8.65e-14 % | 0.627175 | +97.46 % |
| 3.00 | 6364.148578 | 6364.148578 | 4.29e-14 % | 0.632918 | +99.24 % |

**(d) 闸门汇总**

| 闸门 | 内容 | got | 阈值 | 通过 |
|---|---|---|---|---|
| G1_law_at_ka002 | exact closed form vs published long-wave law at ka = 0.02 | 0.000116824 | 0.05 | ✓ |
| G2_law_band | widest ka at which the law holds to < 5 % | 0.24 | 0.2 | ✓ |
| G3_coefficient_identity | 5 pi^2/16 (literature expression) vs module long_wave_drift_coefficient() | 0 | 1e-12 | ✓ |
| G4_slope | log-log exponent of F(ka), ka <= 0.1 | 2.99643 | 3 | ✓ |
| G5_near_vs_far | two independent closed-form routes, ka = 0.2 .. 3 | 2.83519e-15 | 1e-06 | ✓ |
| G6_amplitude_square | F(2A)/F(A) = 4 (second order) | 4 | 4 | ✓ |
| G7_bem_far_vs_closed_form | real BEM solve -> Kochin -> Maruo vs closed form, 800 panels, ka = 0.5/1/2 | 0.020284 | 0.05 | ✓ |
| G8_bem_near_converged | BEM near-field pressure integral at the finest panel rung (ka = 0.5) | 0.0212334 | 0.05 | ✓ |
| G9_convergence_monotone | both BEM routes converge monotonically under panel refinement | {"far": true, "near": true} | 1 | ✓ |
| G10_azimuth | BEM Kochin azimuth resolution 180 -> 360 | 6.16534e-05 | 0.01 | ✓ |

`all_pass = True`（程序化注入）。

## 5. 收敛性与参数敏感性（≥2 点，实测）

**(i) 面板收敛**（ka=0.5）

| 面板 | n_θ×n_z | 远场误差 | 近场误差 | 近远场互差 | 用时 |
|---|---|---|---|---|---|
| 208 | 16×6 | 7.396 % | 30.202 % | 24.628 % | 8 s |
| 456 | 24×9 | 3.179 % | 13.350 % | 10.505 % | 31 s |
| 800 | 32×12 | 2.028 % | 6.082 % | 4.137 % | 69 s |
| 1776 | 48×18 | 1.406 % | 2.123 % | 0.727 % | 296 s |

观测阶（`h∝panels^{-1/2}`）：far 208→456 面板 p=2.15, near 208→456 面板 p=2.08, far 456→800 面板 p=1.60, near 456→800 面板 p=2.80, far 800→1776 面板 p=0.92, near 800→1776 面板 p=2.64。单调性：远场 `True`、近场 `True`。**近场（压力积分）路线是求积受限的**：800 面板时 ka=0.5 仍有 6.08 %，最细 1776 面板降到 2.12 %；远场（约化 Maruo）800 面板即 2.03 %。这是能力边界，不是被覆盖的失败。

**(ii) Kochin 方位分辨率**：`n_azimuth 180→360`，远场变化 `0.0062 %`（闸门 <1 %）→ ✓。

**(iii) 二阶幅值律**（解析）：`F(2A)/F(A) = 4.000000000000`（目标 4）→ ✓。

**(iv) 律的适用域**（上表 a）：`ka ≤ 0.24` 内 <5 %；`ka=1` 处律高估 `78.4 %`，`ka=3` 高估 `99.2 %`——长波律不能外推到工程频段。

## 6. 失败模式与工件

**失败模式（如实记录）**

1. **BEM 近场压力积分求积受限**：粗网格下误差可达数十个百分点（本单最粗一档 30.2 % @ 208 面板），必须靠面板加密解决，不能靠调参；因此本算例的**主闸门放在远场约化 Maruo 路线**，近场只作收敛性证据。
2. **未约化的辐射应力（控制半径）路线不发散地不收敛**：m5242 已记 honest FAIL，本单**不重开、不覆盖**，只用约化核。
3. **文献逐字引文缺口**：`5π²/16` 系数无原文版面可引（第 1 节声明的三条独立推导替代）；Maruo (1960) 与 McIver (1986) 的公式版面在 PDF 抽取中分式错序（工件 `literature.*.transcription.extraction` 已记），故只做"公式级"核对。
4. **平台摩擦**：knowledge_base 检索引擎离线（`/tensor/knowledge/search` 返回 `engine_offline`，engine `127.0.0.1:8021` connection refused），开工三查只能用 `negative_knowledge_ledger` + 仓内 grep 替代；Web 检索能力退化（Google/DuckDuckGo 不可达、Bing 查询被截断为单词、出版商 PDF 403）——这正是第 3 条缺口的环境根因。
5. 本算例是**深水 + 固定 + 刚性 + 无二阶势**（不含自由面二阶势、不含流、不含系泊、不含有限水深）；有限水深与截断圆柱不在覆盖内。

**工件（worktree 内路径）**

| 路径 | 内容 |
|---|---|
| `artifacts/case_cyl_drift_asymptotic/results.json` | 全部数字 + 闸门 + provenance + 文献抽取凭据（sha256/paged/snippet） |
| `artifacts/case_cyl_drift_asymptotic/cyl_drift_coefficient.png` | `C(ka)`：闭式曲线 + 长波律 + BEM 点 + 律失效区 |
| `artifacts/case_cyl_drift_asymptotic/cyl_drift_long_wave_plateau.png` | `(ka)³` 归一平台 `3.084251` 与 ±5 % 带 |
| `artifacts/case_cyl_drift_asymptotic/cyl_drift_bem_convergence.png` | 面板收敛 + 5 % 闸门线 |
| `src/tensorbem/benchmarks/case_cyl_drift_asymptotic.py` | 算例驱动（一键复现） |
| `tests/test_case_cyl_drift_asymptotic.py` | 解析闸门常跑；BEM 闸门读工件 |
| `docs/case_cyl_drift_asymptotic.md` | 本算例文档（与本节内容同源生成） |

**误差判定（显式回答）**：本算例判据是 **<5 %**（非 3 %）。长波律 `ka=0.02` 误差 `+0.0117 %` → ✓；BEM 远场最差 `2.028 %` → ✓；BEM 近场（最细网格）`2.123 %` → ✓；近远场互校 `2.8e-13 %` → ✓。
