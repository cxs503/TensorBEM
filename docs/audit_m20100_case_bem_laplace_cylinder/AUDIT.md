# 交差审计 m20100 — CASE-BEM-LAPLACE-CYLINDER（任务 m20029，TensorBEM）

**结论：L4（基准验证）确认，`tensor_maturity_verdicts` id=826, level=4, verdict_by=board-audit。**
基准裁判行此前缺失，本次审计补登 `tensor_benchmark_cases` id=130（anchor_slug=CASE-BEM-LAPLACE-CYLINDER, status=passed）。
被审对象：m20029 @ `337fca8b`（分支 agent/m20029，尚未并入 main）。

## 1. 取证：三件套逐项核验

| 项 | 判据 | 结果 |
| --- | --- | --- |
| run_artifacts 文件 | `results.json`/`case.json`/`report.md`/`figures/*.png` 在提交内 | 存在，337fca8b 提交内 5 个文件 + doc；report.md 六段齐全（1–6），平台报告 tensor_reports.id=18254 六段 + 2 内嵌图 |
| benchmark ref vs got | 对照裁判表逐项核 got 是否在容差内 | 裁判行**原缺**（`SELECT ... WHERE anchor_slug='CASE-BEM-LAPLACE-CYLINDER'` 返回 0 行）→ 本次补登 id=130；预算 N=128 全部指标实测远优于 3%，最差 2.849e-04 |
| repro_cmd 可执行 | 实跑并与提交件比对 | 在 m20029 @337fca8b 实跑 `PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir /tmp/repro_m20029 ...`，14.6 s，exit 0，verdict PASS，failed_checks []；与提交 `results.json` 逐键比对 589 个数值键中 **587 个完全一致**，差异仅 `generated_at_utc`、`wall_seconds` |
| 测试 | 单元/集成测试实跑 | `pytest tests/test_case_cylinder_stream.py -q` → **26 passed, 2 skipped**（2 个 skip 为参数化落在对数奇异面板/面板本身上，属主值情形，带显式理由） |
| 级别判据 | 功能实质 | 新能力走 **tensorbem.solver 公开 API**（`solve_laplace_exterior_dirichlet/neumann` + 新增 `evaluate_exterior_field`/`evaluate_exterior_surface_velocity`，`src/tensorbem/__init__.py:181-182,286-287` 导出），算例模块 `case_cylinder_stream.py` 直接调用求解器（180/221/225/284/368/410/1012 行），非独立重写 → L4 |

## 2. 独立复核（不 import 被审模块的部分）

| 独立复核 | 方法 | 结果 |
| --- | --- | --- |
| 参考表是否真解析 | 用纯 numpy 复势 W(z)=U(z+a²/z) 重算 64 行参考表 | 与 mission 参考表最大差 **2.22e-16**（u/v 分量 0.0，势 2.2e-16）→ 参考表确为闭式解析，无存数 |
| 路线 D 的极限解 | 外域多极最小二乘（与面板 BEM 完全不同的离散化） | 内接多边形上给定同一 Dirichlet 数据，精确解就是圆柱解（外域 Dirichlet 唯一性），多极解与圆柱解差 **6.94e-16** → 路线 D 的**多面体化误差恰为 0**，报告的 1.366e-07 是纯离散误差，不是被几何残差掩盖 |
| 路线 N 的物面速度量级 | 外域多极 LS（连续型，逐点法向） | N=32/64/128 相对 L2 = 4.850e-03 / 1.207e-03 / 3.013e-04，二阶收敛（与 mission 的 1.633e-03/4.046e-04/1.008e-04 同阶同量级）→ 报告的多面体化残差 O(h²) 属数据性质，非求解器假象 |
| 核函数选择不变性 | 改 `reference_length`=2/3/5（cond(S) 76→177） | 路线 D 速度误差 1.365622e-07 / 1.365622e-07 / 1.365622e-07；路线 N 1.532016e-04 / 1.532016e-04 → 结果与核平移无关 |
| 场求值精度随距离 | 审计自己按 N=128 重算外域网格点误差 | r/a∈[1.002,1.05): max **1.225e+00**；[1.05,1.25): 9.126e-04；[1.25,1.6): 5.504e-07；[1.6,3.2): 1.998e-07 |

## 3. 计算结果（mission artifact，预算 N=128）

路线 D（解析扰动势作 Dirichlet 数据 → 外域势+速度）：

| N | cond(S) | 相对残差 | φ′ 相对 L2 | 速度相对 L2 | 速度最大偏差/U |
| --- | --- | --- | --- | --- | --- |
| 32 | 1.917e+01 | 1.703e-16 | 2.334e-05 | 2.266e-04 | 6.712e-04 |
| 64 | 3.811e+01 | 2.272e-16 | 2.998e-06 | 1.229e-06 | 2.420e-06 |
| 128 | 7.611e+01 | 3.881e-16 | 3.578e-07 | 1.366e-07 | 2.290e-07 |

路线 N（解析 q_e 作 Neumann 数据 → 物面速度/Cp/外域速度）：

| N | cond(½I−H) | 物面速度相对 L2 | |v·n|/U | Cp 相对 L2 | 外域速度相对 L2 |
| --- | --- | --- | --- | --- | --- |
| 32 | 2.103e+00 | 1.633e-03 | 9.654e-03 | 4.601e-03 | 2.565e-03 |
| 64 | 2.050e+00 | 4.046e-04 | 2.411e-03 | 1.143e-03 | 6.131e-04 |
| 128 | 2.025e+00 | 1.008e-04 | 6.025e-04 | 2.849e-04 | 1.532e-04 |

收敛阶（以 h_mean 为尺度）：路线 N 物面 [2.0165907313184706, 2.0065676588343466]、外域 [2.068248030773628, 2.0015453338106735]；路线 D 外域 [7.53951965123384, 3.1711289098675235]（超收敛：数据与多边形相容，多面体化误差为 0）。

15 项预置判据（mission 自报，本次逐项复核 ok）：

| 判据 | 实测 | 阈值 | 判定 |
| --- | --- | --- | --- |
| geometry_vertices_on_circle | 1.1102e-16 | <= 1e-12 a | PASS |
| geometry_area_matches_inscribed_closed_form | 2.8283e-16 | <= 1e-12 | PASS |
| geometry_perimeter_matches_inscribed_closed_form | 0.0000e+00 | <= 1e-12 | PASS |
| reference_is_harmonic | 7.7716e-08 | <= 1e-06 U/a | PASS |
| reference_is_irrotational | 2.5711e-08 | <= 1e-06 U/a | PASS |
| field_gradient_matches_finite_difference | 1.4647e-10 | <= 1e-05 | PASS |
| dirichlet_velocity_relative_l2 | 1.3656e-07 | < 0.03 | PASS |
| dirichlet_potential_relative_l2 | 3.5782e-07 | < 0.03 | PASS |
| neumann_surface_velocity_relative_l2 | 1.0076e-04 | < 0.03 | PASS |
| neumann_velocity_relative_l2 | 1.5320e-04 | < 0.03 | PASS |
| neumann_normal_velocity_residual | 6.0245e-04 | <= 0.01 U | PASS |
| neumann_cp_relative_l2 | 2.8492e-04 | < 0.03 | PASS |
| neumann_surface_velocity_observed_order | [2.0165907313184706, 2.0065676588343466] | >= 1.5 | PASS |
| wrong_reference_is_rejected | 3.9523e+06 | >= 10 | PASS |
| far_field_matches_closed_form | 5.5909e-09 | <= 0.01 U | PASS |

## 4. 结果图

见 `figures/fig_field.png`（解析 |v|/U 云图+BEM 速度箭头+逐点相对误差）、`figures/fig_convergence.png`（误差梯队与 h²、h⁴ 斜率、审计独立路线点）、`figures/fig_verdict.png`（11 项数值判据的容差裕度）。

## 5. 注意点 / 缺陷分级

**非阻塞（建议改进）**

1. **近壁精度未守门**（严重度：中，影响面：外域场求值的下游用户）。`evaluate_exterior_field` 有 `minimum_clearance_factor` 参数但默认 0.0，r−a=0.002a 处相对速度误差可达 122.5%；算例只探到 r/a≥1.25，故近壁regime未被覆盖。建议文档明示可用区间或把默认 clearance 设为 ~0.05 h_max 并在结果里带误差估计。
2. **路线 D 的单层算子条件数随加密增长**（19.2→38.1→76.1），参考长度 2.0 对 a=1 圆接近退化点；算例已逐级报告 cond(S)，审计改 reference_length 后结果不变，故仅属可扩展性限制（N≳512 前需注意）。
3. **赛题摘要措辞两处略超实测**（严重度：低）：平台摘要写"速度场误差 ≤1.5e-4"，实测路线 N 外域速度 1.5320e-04（超 2%）；摘要写"收敛阶≈2"仅对路线 N 成立（路线 D 为 [7.53951965123384, 3.1711289098675235]）。mission 自身 report.md 数值正确，属摘要级泛化。

**已核销（不构成缺陷）**

- 裁判行缺失：已补登 id=130。
- `page_or_article_number: WITHHELD`：如实声明（Lamb 1932 / Milne-Thomson 1968 不可达），替代判据为解析式自洽（调和 7.77e-08、无旋 2.57e-08）+ 审计独立复算 2.2e-16。
- 2 个 skip：主值/奇异参数化，带理由，合理。
- 分支未并入 main：属平台推送流程，非任务缺陷。

## 6. 误差判定（关键量 vs 基准，逐项是否 <3%）

| 关键量（N=128） | 实测 | 基准 | 是否 <3% |
| --- | --- | --- | --- |
| 路线 D 外域速度 L2 | 1.3656e-07 | dW/dz 解析 | 是 |
| 路线 D 扰动势 L2 | 3.5782e-07 | U a²cosθ/r | 是 |
| 路线 N 物面速度 L2 | 1.0076e-04 | 2U|sinθ| | 是 |
| 路线 N 外域速度 L2 | 1.5320e-04 | dW/dz 解析 | 是 |
| 路线 N Cp L2 | 2.8492e-04 | 1−4sin²θ | 是 |
| 远场 r=8a 速度 | 5.59e-09 | dW/dz 解析 | 是 |
| 错参考判别比 | 3.952e+06 | ≥10 倍 | 是（反向对照通过） |

六段报告齐全、数值全部程序化从 `results.json` 注入（`hand_typed_numbers: 0`），审计未发现 MOCK/STUB 迹象：
`evaluate_exterior_field`/`evaluate_exterior_surface_velocity` 为闭式面板积分实现，测试用中心差分独立校核核函数梯度（1.46e-10）。

## 7. 未执行的检查（如实声明）

- 未做三维/自由面/粘性扩展（超出本算例声明范围）。
- 未在 loong64 集群重跑（本机 96 s 级，单机即可；mission 环境的 numpy 版本与审计机不同，但结果逐位一致）。
- 未审计分支推送后的 main 合并状态（待平台推送）。
