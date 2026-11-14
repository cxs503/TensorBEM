# m20977 · CASE-BEM-RIGID-BODY-MASS：外部裁判路线 route T（trimesh）

**锚点** `CASE-BEM-RIGID-BODY-MASS`（算例池·刚体质量矩阵，ref=闭式块形式，容差 3%）｜**机判** `PASS`（17 项，not_run=[]）｜最差接受逐元素误差 1.4151%

本文所有数字由 `make_report_m20977.py` 从 `docs/case_bem_rigid_body_mass/results.json` 读出注入，无人工转录。

## ① 案例介绍

频域运动方程 `(M_RB + A)ẍ + Bẋ + Cx = F` 中，刚体惯量矩阵 `M_RB` 由质量 `m`、质心 `r_g`与惯量张量 `I_g` 组成闭式块形式 `[[m I3, -m S(r_g)], [m S(r_g), I_g + m S(r_g)ᵀS(r_g)]]`。算例体为三轴椭球 `a,b,c = [60.0, 12.0, 8.0]` m、密度 1025 kg/m³、质心偏置 [5.0, 0.0, -3.0] m，另有姿态旋转体与 Wigley 船型体。

本锚点原有四层证据：闭式块形式（R）、TensorBEM 带符号四面体积分（M）、动量定义（P）、capytaine 装配（X，只涉及附加质量）。**四层里体积积分全部出自 TensorBEM 自己的代码** —— 积分器或网格生成器若有偏，参照会与被测对象一起漂移，整条链看不见。

本单补的是第五层 route T：把「离散」与「积分」两层都换成第三方实现（trimesh 的 Mirtich 1996 投影算法 + `uv_sphere` 经纬参数化），再用它回头裁判 TensorBEM。

## ② 怎么做的

| 层 | 文件 / 入口 | 作用 |
|---|---|---|
| 新路线 | `src/tensorbem/benchmarks/rigid_body_mass_external_route.py` | 外部质量特性积分、外部离散阶梯、船型体复核、拓扑守卫外部裁判 |
| 算例入口 | `src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py` | 把 route T 折进 `results.json` / `verdict.json` / `CASE.md`，新增 `--no-external` |
| 测试 | `tests/test_rigid_body_mass_external_route.py` | 10 项门控（自校准、圆整一致性、接受层级、判别力、船型体、守卫裁判、焊接策略、not_run 语义、落地工件） |
| 依赖 | `pyproject.toml` `[project.optional-dependencies] external-verification` | `trimesh` 只用于验证，求解器不 import；缺库时本段严格报 `not_run`，判定不得 PASS |

四问，逐层加码：

1. **裁判先自校准**：立方体 `I = m L²/6` 无人需要积分的数，trimesh 必须复现到圆整；
2. **分层归因**：同一套三角形上，两个积分器必须一致到圆整 —— 这样接受层级的误差才能归给「面板化」而不是「积分」；
3. **独立路线**：外部离散 + 外部积分对闭式块形式必须进 3%；
4. **外部裁判守卫**：`trimesh.is_watertight` 对人为损坏的网格必须与 TensorBEM 守卫同意（或至少不比守卫更宽松），否则上一轮 ADR 的补救就是自证。

## ③ 参数与结果

**裁判自校准（trimesh 先证明自己）与分层归因**

| 项 | 结果 |
|---|---|
| 立方体（边长 3.0 m，12 面元）体积误差 | 0.0000e+00 |
| 立方体惯量误差 vs `m L²/6` | 0.0000e+00 |
| 共享网格上两积分器 6×6 最差差异（4 层最大） | 1.6061e-11（限值 1.0000e-09） |
| 两积分器各自对闭式形式的距离之差（误差预算差） | 4.0246e-16 |

**外部离散阶梯（uv_sphere + trimesh，参照 = 闭式块形式）**

| uv 计数 | 面元 | 体积误差 | 惯量误差 | 6×6 最差元素 | 最差位置 | 旧耦合对照 | 接受 |
|---|---|---|---|---|---|---|---|
| 16 | 1792 | 1.2514% | 2.1267% | 2.0985% | yaw-yaw | 198.75% | — |
| 32 | 7680 | 0.2966% | 0.5070% | 0.5002% | yaw-yaw | 199.70% | ✅ |
| 64 | 31744 | 0.0722% | 0.1236% | 0.1220% | yaw-yaw | 199.93% | ✅ |
| 128 | 129024 | 0.0178% | 0.0305% | 0.0301% | yaw-yaw | 199.98% | ✅ |

接受层级（uv ≥ 32）最差 0.5002%；拟合收敛阶 -0.993；最先达标的层级 uv = 16（误差 ~ 面元数^-0.99）。

**船型体（Wigley，`Cb = 4/9`）**

| 项 | 参照（闭式积分） | 外部裁判（trimesh） | 误差 |
|---|---|---|---|
| 排水体积 | 10240.000 m³ | 10234.446 m³ | 0.0542% |
| 质量 | 10496000.0 kg | 10490307.5 kg | 0.0542% |
| 形心 | [0.0, 0.0, -3.0] | [-1.895808813669034e-15, -1.0967232001987803e-17, -3.000325132027163] | 3.25e-04 m |
| 惯量张量（在设计质心） | — | — | 0.0395% |
| **6×6 最差元素**（4704 面元，最差位置 roll-roll） | 闭式块形式 | 外部路线 | **0.1248%** |

分块最差：平动-平动 0.0542%、平动-转动 0.0542%、转动-转动 0.1248%（转动块最大，与薄壁船体 `I_zz` 主要来自水线面分布一致）。

**外部裁判对拓扑守卫的逐变体裁决**

| 变体 | 守卫接受 | 外部接受 | 体积误差（外部积分） | 直接采信外部的 6×6 最差 |
|---|---|---|---|---|
| closed (control) | 是 | 是 | 3.3839% | 5.501% |
| open patch (every 7th face removed) | 否 | 否 | 18.1530% | 385276.270% |
| single flipped face | 否 | 是 | 5.4248% | 330514.616% |
| duplicated face | 否 | 否 | 3.1575% | 39795.751% |
| inward winding (all faces) | 否 | 否 | 196.6161% | 未定义（质量≤0） |

**净增量在第 3 行**：单个翻转面对 trimesh 的边关联判据不可见（仍算 watertight，它会直接采信并积出一个错体积），而 TensorBEM 守卫按有向半边判定、予以拒绝。外部漏检清单 = ['single flipped face']。结论：拓扑校验不能外包给第三方库。

焊接策略差异（全部顶点拆开的同一闭合面）：守卫接受=True、trimesh 原样 watertight=False、`merge_vertices` 后=True —— 差异只是焊接策略，对齐后一致。

## ④ 图

- `docs/case_bem_rigid_body_mass/fig_external_route.png`：左=外部离散阶梯（误差 vs 面元数、3% 判据线、旧耦合对照）；右=守卫与外部判据对五个变体的接受矩阵。

## ⑤ 注意点

- **独立性有边界**：route T 换掉了离散与积分，但仍共用闭式参照、`rigid_body_mass_matrix` 装配与逐元素误差口径。块约定由 route P 负责，route T 只证「质量特性输入」。
- **收敛阶是 1 阶而非 2 阶**：uv_sphere 的板元是内接平面三角形，观测阶 -0.99（面积/体积型量的典型值），与 icosphere 的经验一致；外推不能当解析值用。
- **粗层级已达标**：uv=16（1792 面元）就已在 3% 内，声明接受层级取 uv ≥ 32 是留裕度，不是「刚好通过」；`results.json` 同时记录最先达标层级以免读者误判裕度。
- **trimesh 是可选依赖**：缺库/`--no-external` 时本段 `not_run`，机判变 `INCOMPLETE`，绝不落成 PASS。
- **三角网格公共盲区**：两库都积分三角化表面，所有三角网格共有的几何误差（如半轴标错）不在本锚点覆盖内 —— 那是几何锚点（Cb/L/B）的活。
- **既有红项（与本单无关）**：`tests/test_case_pool_registry.py` 有两个用例在本工作树里因其他锚点缺图（HAVELOCK / ANNULUS / LAPLACE-CYLINDER / ELLIPSE-2D 的 figures）而失败；RIGID-BODY-MASS 的路径全部存在。未在本单修（越界）。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 外部离散 6×6 最差（接受层级） | 闭式块形式 | 0.5002% | 是 |
| Wigley 船型体 6×6 最差 | 闭式积分 | 0.1248% | 是 |
| Wigley 排水体积 | 闭式积分 | 0.0542% | 是 |
| 两积分器一致性（共享网格） | 逐元素、限值 1.0000e-09 | 1.6061e-11 | 是（圆整量级） |
| 旧耦合约定对照（必错） | 闭式块形式 | 199.98% | 否（预期，判别力证据） |
| 原锚点最差接受（不变） | 闭式块形式 | 1.4151% | 是 |
| route P 动量定义 vs 出厂装配（不变） | 动量定义 | 1.1477e-16 | 是（圆整） |

**结论**：全部关键量 vs 基准误差均 < 3%（最差的 6×6 外部路线 0.500%、船型体 0.125%）；一致性项与动量项在圆整量级。无「无基准」项。

机判检查项（`verdict.json`）：`{"rao_caller_verified": true, "rao_caller_replay_roundoff": true, "rao_caller_legacy_coupling_material": true, "worst_accepted_entry_within_tolerance": true, "momentum_route_roundoff": true, "momentum_route_rejects_flipped_coupling": true, "momentum_route_beats_energy_blind_spot": true, "base_case_passed": true, "orientation_case_passed": true, "external_integrator_roundoff": true, "external_mesh_route_within_tolerance": true, "external_route_rejects_legacy_coupling": true, "external_ship_shape_within_tolerance": true, "external_topology_guard_agrees": true, "external_guard_is_load_bearing": true, "legacy_coupling_rejected": true, "orientation_negative_control_rejected": true}`

