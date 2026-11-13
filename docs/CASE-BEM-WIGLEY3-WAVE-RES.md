# CASE-BEM-WIGLEY3-WAVE-RES — Wigley III 兴波阻力 vs Michell 薄船理论（Kochin 远场）

* 算例 ID：`CASE-BEM-WIGLEY3-WAVE-RES` ｜ 仓：`TensorBEM` ｜ 域：`ship-total-performance/wave-resistance`
* 能力：Forward-speed wave resistance (Wigley, Michell/Kochin)
* 工件：`benchmarks/verified/case_bem_wigley3_wave_res/`（`results.json` / `report.md` / `figures` / `repro.sh`）
* 一键复现：`PYTHONPATH=src python3 -m tensorbem.benchmarks.case_wigley3_wave_resistance --output benchmarks/verified/case_bem_wigley3_wave_res`  （本机 92 s）
* 判据：`ref vs got` 相对误差 < 5 %；几何门 Cb/L/B 先过；参考解数值收敛 < 1e-3。

## 0. 结论
**全部判据 通过**。两条 got 路线对 Michell 远场参考的最大相对误差：
kochin（生产路径，canonical 谱对）= 3.673e-04；中心面面元（800 面元）= 3.622e-03；面元最细（12800）= 1.721e-04。
**但绝对量级与侧数约定仍 WITHHELD**（见 §5），本算例主张的是「代码实现对已发表的 Michell 积分形式成立」，不是「与实验/手册表一致」。

## 1. 几何与参数（先验证 Cb/L/B）
| 量 | 闭式 | 实算 | 相对误差 |
|---|---|---|---|
| L_over_B | 10 | 10 | 0.00e+00 |
| B_over_T | 1.6 | 1.6 | 1.39e-16 |
| Cb | 0.444444444444 | 0.444444444444 | 1.25e-16 |
| displaced_volume | 0.075 | 0.075 | 0.00e+00 |

L = 3.0 m, B = 0.3 m, T = 0.1875 m（L/B = 10.0, B/T = 1.6）；ρ = 1025.0 kg/m³, g = 9.80665 m/s²；湿表面积 S_w = 1.339116 m²，L·T = 0.562500 m²。
Cw 归一化两种面积都在工件里给出，引用时必须写明用哪一个。

## 2. ref 出处（零手抄）
* ref = Michell (1898) 远场积分（波角形式）`Rw = (4 ρ g²/(π U²)) ∫₀^{π/2} sec³θ |A(θ)|² dθ`，A 为 Wigley 抛物线型的**闭式** Kočin 幅值（中心面源片），θ 方向分段 Gauss 收敛到 GCI < 1e-5。
* Michell, J.H. (1898) The wave resistance of a ship, Phil. Mag. 45(5) 106-123 (far-field integral, sec^3 theta weight)
* Tuck, E.O. (1989) The wave resistance formula of J.H. Michell (1898) ..., J. Austral. Math. Soc. Ser. B 30, 426-445 (Kochin form, Wigley test body)
* 零手抄：工件内每个 ref 数值都由 `tensorbem.thin_ship_michell.wave_resistance_theta_form` 算出并记录，无人工键入。
* **手册/原表页状态：WITHHELD** — a tabulated Wigley Cw(Fr) (ITTC / towing tank / printed Michell table) with page citation, stated L/B/T and stated Cw reference area is not available offline in this work tree, and network egress is blocked; not reconstructed from memory。
* 绝对量级：WITHHELD（docs/michell_unit_arbitration.md section 6）。
* 约定：both sides of the centreplane sheet (|H|^2 carries the factor 4 of the one-side amplitude |A|^2). A one-side convention scales Rw by 1/4 in *both* routes, so the 5% criterion is convention-invariant while the quoted absolute Cw is not; see docs/michell_unit_arbitration.md section 6 ('Sides convention').

## 3. 方法
* ref：`Michell (1898) far field in the wave-angle form, closed-form Wigley amplitude, graded composite Gauss in theta (tensorbem.thin_ship_michell.wave_resistance_theta_form)`，层级 [75, 150, 300, 600, 1200]（最细为参考值）。
* got-A `route_kochin`：`tensorbem.wigley.michell_wave_resistance_canonical`，生产 n_quad = 256（λ = 1+s² 去尖点）。
* got-B `route_panels`：`tensorbem.thin_ship_michell.wigley_panels + wave_resistance_from_amplitude`，中心面元密度 [(20, 10), (40, 20), (80, 40), (160, 80)]，密度逐层加倍。
* 两条路线与 ref 的积分族、幅值求值方式均不同，构成三路互证。

## 4. 对照表（ref vs got）
| Fr | Rw ref [N] | Rw kochin [N] | err | Rw panels(160×80) [N] | err | <5% |
|---|---|---|---|---|---|---|
| 0.20 | 7.168236e-01 | 7.165603e-01 | 0.0367 % | 7.167596e-01 | 0.0089 % | ✅ |
| 0.25 | 1.342588e+00 | 1.342420e+00 | 0.0126 % | 1.342819e+00 | 0.0172 % | ✅ |
| 0.30 | 3.891780e+00 | 3.891664e+00 | 0.0030 % | 3.891858e+00 | 0.0020 % | ✅ |
| 0.35 | 3.086580e+00 | 3.086493e+00 | 0.0028 % | 3.086534e+00 | 0.0015 % | ✅ |
| 0.40 | 8.832036e+00 | 8.831973e+00 | 0.0007 % | 8.832016e+00 | 0.0002 % | ✅ |
| 0.45 | 1.698526e+01 | 1.698521e+01 | 0.0003 % | 1.698524e+01 | 0.0001 % | ✅ |

无量纲（两种面积都列）：
| Fr | Cw ref (L·T) | Cw kochin (L·T) | Cw ref (S_w) | Cw kochin (S_w) |
|---|---|---|---|---|
| 0.20 | 2.112973e-03 | 2.112197e-03 | 8.875613e-04 | 8.872353e-04 |
| 0.25 | 2.532821e-03 | 2.532503e-03 | 1.063920e-03 | 1.063786e-03 |
| 0.30 | 5.098560e-03 | 5.098406e-03 | 2.141667e-03 | 2.141603e-03 |
| 0.35 | 2.970866e-03 | 2.970783e-03 | 1.247922e-03 | 1.247887e-03 |
| 0.40 | 6.508524e-03 | 6.508478e-03 | 2.733927e-03 | 2.733908e-03 |
| 0.45 | 9.889833e-03 | 9.889803e-03 | 4.154257e-03 | 4.154245e-03 |

## 5. 收敛性与敏感性
| Fr | ref GCI | 面元观测阶 | 面元 GCI | λ_max=30 截断漂移 |
|---|---|---|---|---|
| 0.20 | 1.545e-06 | 3.97 | 1.884e-05 | 8.027e-07 |
| 0.25 | 5.247e-06 | 7.05 | 3.901e-08 | 1.589e-06 |
| 0.30 | 5.198e-06 | 非单调(n/a) | 8.987e-06 | 1.865e-06 |
| 0.35 | 6.614e-06 | 非单调(n/a) | 1.204e-05 | 5.332e-06 |
| 0.40 | 8.545e-06 | 1.69 | 2.992e-06 | 3.754e-06 |
| 0.45 | 2.171e-06 | 1.86 | 1.791e-06 | 4.191e-06 |

面元路线误差（按密度）：
| Fr | 200 面元 | 800 面元 | 3200 面元 | 12800 面元 |
|---|---|---|---|---|
| 0.20 | 4.969e-02 | 3.622e-03 | 1.328e-04 | 8.916e-05 |
| 0.25 | 1.220e-02 | 7.190e-04 | 1.762e-04 | 1.721e-04 |
| 0.30 | 2.044e-03 | 8.523e-05 | 1.265e-05 | 1.984e-05 |
| 0.35 | 1.218e-03 | 2.435e-05 | 2.448e-05 | 1.485e-05 |
| 0.40 | 1.684e-04 | 2.466e-05 | 7.560e-06 | 2.248e-06 |
| 0.45 | 6.731e-06 | 1.870e-05 | 5.100e-06 | 1.346e-06 |

截断敏感性（仅报告）：把角积分截在 secθ = 8，参考值最多变化 9.702e-04；生产/面元路线用的 λ_max = 30 在门内。
非单调三元组（数值已进入舍入/振荡区）记 `panels_gci.monotone = false`，GCI 以最细层差为上界，不拟合阶数。

## 6. 失败模式 / 局限 / 生态摩擦
1. **绝对量级 WITHHELD**：`docs/michell_unit_arbitration.md` §6 已登记：无实验/第一性原理 Wigley Cw 随仓发布，且「单侧 vs 双侧」常数（因子 4）未由能量流推导裁定。两条路线同用双侧约定 ⇒ 因子 4 在对照中抵消，<5% 判据对约定不变；但**任何绝对值引用都必须带此注**。
2. **legacy 默认函数仍是缺陷**：`wigley.michell_wave_resistance` 采样 k = α = k0λ（违反定常自由面条件），返回 kg s² m⁻¹，本网格低 10.3–24.4 倍；`benchmarks/verified/bem_wigley_wave_resistance/results.json` 与多个 driver 仍读它（该工件自洽但物理量级错），本算例故意不用。
3. **厚船自由面面元 BEM 仍无可用兴波阻力数**：`docs/m10003/REPORT.md` honest FAIL（近场压力积分被噪声主导、远场前因子未标定）。本算例的离散路线是**中心面薄船源片**，不是自由面面元解 —— 生态缺口保留。
4. Fr ≥ 0.45 超出 B/L = 0.1 的薄船可信域，仅报告不作验收；验收带 Fr = 0.20–0.40。
5. 平台三查（knowledge_base / negative_knowledge_ledger / oss_solutions，关键词 wigley、michell、TensorBEM、wave resistance）本波全部 0 命中 —— 同域无既有能力条目、无否定性知识、无 OSS 方案可复用；本算例是首个 case-pool 条目。

## 7. 判据表
| criterion | measured | limit | status |
|---|---|---|---|
| geometry_gate | 1.388e-16 | 1.000e-12 | ✅ |
| kochin_vs_reference_resistance | 3.673e-04 | 5.000e-02 | ✅ |
| panels_vs_reference_resistance | 1.721e-04 | 5.000e-02 | ✅ |
| panels_800_vs_reference_resistance | 3.622e-03 | 5.000e-02 | ✅ |
| reference_quadrature_gci | 8.545e-06 | 1.000e-03 | ✅ |
| reference_truncation_drift | 5.332e-06 | 1.000e-03 | ✅ |
| panels_gci | 1.884e-05 | 5.000e-02 | ✅ |
| reference_truncation_probe_lam8 | 9.702e-04 | report-only | ✅ |

**all_passed = True**

