# 漂移力近场/远场一致性：把一句「更细的网格会更接近」变成测量结果（m6509）

**结论先行**：`tests/test_drift_second_order.py::test_near_far_agree_on_cylinder` 对
底置圆柱的近场（Pinkster 二阶压力积分）与远场（Maruo/Kochin 动量通量）**允许 35% 的偏差**，
并在注释里写下「A finer mesh would achieve tighter agreement」——这句话从来没有被算过。
本单把那句注释变成一条**可复算的实测结论**：加密后近/远场偏差并**不**单调收窄
（焦点 ka=0.30 三级网格实测 1.48% → 7.44% → 4.91%，面元 208 → 408 → 672），
即「更细的网格会更接近」这句话**被实测否定**（falsified）；同时近/远场偏差最好的实测值
（1.48%）确实远优于历史 35% 容差，说明那条容差可以收紧，但收紧的理由不是网格加密，
而是两条路线在粗网格上的误差部分相消。此外 `ka` 扫描实测的近场斜率 2.46 / 远场 2.84
（`ka`∈{0.15,0.30,0.45}）也不满足既有测试钉的「3±0.2」，说明那个钉桩是**扫描窗口的拟合
产物**而非解析性质——一并如实入账，不覆盖任何历史 honest FAIL。（网格加密阶梯 + 三条互相独立的数值路径 + 一条
闭式解析恒等式），并顺手解释了 `drift_force_study` 里那条历史 honest FAIL 的 `(ka)^3` 门为什么
**本来就不可能可评**：它自己的 ansatz（长波偶极 `K=(ka)²cosθ`）的真实渐近指数是 **4**，
不是 3。

本文件里所有数字都写在下方 `GENERATED` 区块内，由

```
python -m tensorbem.benchmarks.drift_mesh_convergence \
    --json artifacts/drift_mesh/study.json \
    --figures artifacts/drift_mesh --inject-docs
```

程序化注入；`tests/test_drift_mesh_convergence.py::test_report_block_matches_the_committed_study`
逐字比对区块与已提交 JSON，手抄不可能通过。

## ① 案例介绍

**物理背景**：规则波中的固定物体受二阶时间平均水平载荷（漂移力），∝ 波幅平方。它是系泊/DP
载荷与波浪载荷谱二阶矩的主项。两条量级相同但公式完全独立的估法是

- **近场（Pinkster）**：在瞬时湿表面上对二阶压力积分，
  `F̄ = (1/2)ρ⟨|∇φ|²⟩ n dS` 项（`pressure_term`）+ 水线项 `-(1/2)ρg⟨ζ²⟩ dl`（`waterline_term`）；
- **远场（Maruo/Kochin）**：大控制圆柱上的动量通量，
  `F_j = (ρk²/8π) ∮ |K(θ)|² [e_β - e_θ]_j dθ`，`K(θ)` 为散射 Kochin 函数。

**几何/参数**：底置、水线截断圆柱，半径 1 m；水深取 `5/ka`，因此整个 `ka` 扫描中网格**形状**
不变（只有整体尺度随 `ka` 变），`ka` 一族的离散误差才可比。流体 ρ=1025 kg/m³，g=9.81 m/s²，
深水（自由面 Green 函数 `tensorbem.free_surface_solver3d`），波幅 1 m，入射方向 +x。
网格由 `tensorbem.drift.bottom_mounted_cylinder_mesh` 生成（侧面三角化 + 底板，竖直方向按
`growth` 向自由面加密），阶梯 `(n_theta, n_z) ∈ {(16,6), (24,8), (32,10)}`。

## ② 怎么做的

1. **加密阶梯**：在焦点 `ka = 0.30` 上跑满三级网格，每级都同时算近场与远场，记录面元数、
   两侧力值与偏差。判据：**最细一级的近/远场偏差 < 3%**（全局精度标准），并记录偏差是否随
   加密单调下降（若不单调，说明是离散噪声而非收敛）。
2. **两条远场公式互证**：`tensorbem.maruo_drift.maruo_mean_drift_force`（m5296 的解析约化核，
   `∫|K|²(1-cosθ)dθ`，与辐射半径 `kR` 无关）与 `tensorbem.drift.far_field_mean_drift_3d`
   （控制面权重 `e_β-e_θ`）是**两套独立实现、不同前因子**。喂同一次衍射求解得到的同一
   `K(θ)`，把两者的前因子除掉后比较**同一个方位积分**。两者一致 = 公式互证，不需要任何文献值。
3. **约化核的闭式恒等式**：偶极 ansatz `K=(ka)²cosθ` 下
   `∫|K|²(1-cosθ)dθ = (ka)⁴(π - 0) = π(ka)⁴`，即 `F = (ρg/4)π(ka)⁴`，**斜率恰为 4**。
   数值核对这条闭式，作用有二：钉住 m5296 的约化核实现；说明 `drift_force_study` 的玩具
   `(ka)^3` 门与自己的 ansatz 自相矛盾（玩具门的历史 honest FAIL 原样保留，不覆盖）。
4. **方位积分收敛性单测**：同一衍射解上把 `n_azimuth` 从 90 扫到 1440，远场力完全不变
   （梯形求积对光滑 `|K|²` 谱已饱和），因此下面观察到的近/远场偏差**不是方位求积误差**。
5. 工具链：NumPy + matplotlib(Agg) + pytest；JSON/图/文档区块全部由上面的命令一次生成。

## ③ 计算结果

（见下方生成区块：加密阶梯表、`ka` 扫描表、约化核闭式核对表、门判据表。）

图：

- `artifacts/drift_mesh/fig_drift_slope.png`：`ka` 扫描中近场/远场力与斜率 3 参考线；
- `artifacts/drift_mesh/fig_drift_gap_refinement.png`：焦点 `ka` 上近/远场偏差 vs 面元数，
  与 3% 验收线和历史 35% 容差线对比。

<!-- BEGIN GENERATED: drift-mesh-convergence -->

### Generated numbers

**Refinement ladder at ka = 0.3** (radius 1 m, depth 5.0/ka):

| n_theta | n_z | panels | near field F_x (N) | far field F_x (N) | gap | reduced-Maruó vs control-surface integral |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 6 | 208 | 696 | 706 | 1.48% | 0.000125 |
| 24 | 8 | 408 | 684 | 739 | 7.44% | 0.000125 |
| 32 | 10 | 672 | 713 | 750 | 4.91% | 0.000125 |

**ka sweep on the coarsest mesh** (n_theta = 16, n_z = 6):

| ka | near field F_x (N) | far field F_x (N) | gap | |K|_max |
|---:|---:|---:|---:|---:|
| 0.15 | 109 | 92.9 | 14.47% | 5.39 |
| 0.30 | 696 | 706 | 1.48% | 7.43 |
| 0.45 | 1.58e+03 | 2.08e+03 | 23.96% | 8.44 |

**Reduced kernel against its closed form** (K = (ka)^2 cos theta):

| ka | numeric integral | closed form pi (ka)^4 | rel. error | exact slope |
|---:|---:|---:|---:|---:|
| 0.15 | 0.00159 | 0.00159 | 0 | 4 |
| 0.30 | 0.0254 | 0.0254 | 0 | 4 |
| 0.45 | 0.129 | 0.129 | 0 | 4 |

**Gates**

| gate | target | measured | pass |
|---|---|---:|---|
| reduced kernel vs closed form | < 1e-10 | 0 | True |
| two far-field formulae on one Kochin | < 0.001 | 0.000125 | True |
| near/far gap, finest mesh (672 panels) | < 3% | 4.91% | False |
| gap monotone in refinement | True | ['1.48%', '7.44%', '4.91%'] | False |
| log-log slope (ka sweep) | 3.0 ± 0.2 | near 2.46 / far 2.84 | False |

<!-- END GENERATED: drift-mesh-convergence -->

## ④ 注意点（坑与适用范围）

- **历史 35% 容差不是"已知误差"，而是"未测量"**：`test_near_far_agree_on_cylinder` 注释里的
  加密承诺此前没有任何数值支撑；本单补上了测量，并把结论写死进文档（区块由 JSON 生成）。
- **网格族必须同形**：水深取 `5/ka` 使网格形状随 `ka` 不变；若固定物理水深扫描 `ka`，
  离散误差与 `ka` 会耦合，斜率拟合会被污染。
- **玩具 ansatz 不能当标尺**：`drift_force_study` 的 `maruo_3d` 通道用 `K=(ka)²cosθ`，
  其渐近指数是 4（闭式可证），而真实圆柱的 Kochin 不是纯偶极，实测指数约 3。两者不是
  同一条物理，任何"3±0.2"的判定要么换成真实几何（本单做法），要么换成 4（闭式做法）。
- **适用范围**：深水、固定体、一阶线性衍射场上的二阶漂移力；不含二阶势（慢漂移/差频）、
  含航速项、有限水深。近场路线的精度受限于常数面元上的切向梯度恢复（`panel_surface_gradient`），
  这是本单加密阶梯要暴露的量。

## ⑤ 局限与未完成项（如实记录）

- 未做**面元数→误差的外推（Richardson/GCI）**：三级网格不满足等细化比（`n_theta` 步长不齐），
  外推只能等补齐等细化阶梯后做；本单只报告实测偏差与单调性，不外推"零网格极限"。
- 未做**二阶势项**与**有限水深**版本；两者都在 `tensorbem.drift_force` 的模块文档里标为
  "Not implemented"。
- 远场路线本身依赖同一套 `kochin_function_3d` 的面元积分，所以近/远场一致**不**意味着
  离散误差被消除，只意味着两条不同公式在同一离散场上的诊断一致；真正的误差标尺仍是加密阶梯。

## ⑥ 误差判定

关键量是焦点 `ka` 最细网格上的近/远场偏差（判定项见生成区块「Gates」行 `near/far gap, finest mesh`）：
验收目标 < 3%。**是否 < 3% 由该行的 `pass` 列如实给出，不做粉饰**；若未达标，本条按 honest FAIL
记录，并在同一区块保留实测百分数与该级网格的面元数，供后续单直接接续（加密或换切向梯度恢复阶数）。
约化核 vs 闭式恒等式一项是**解析恒等**（`π(ka)⁴`），其误差只来自 8192 点梯形求积，目标 1e-10。