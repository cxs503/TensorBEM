# CASE-BEM-RIGID-BODY-MASS — 6×6 刚体质量矩阵的第三条独立路线（动量定义）与机器注册

**判定：PASS** — 接受层级最大逐元素相对误差 1.4151%（判据 3%）；动量定义路线（route P）与出厂装配在 4 种配置上一致到 1.1477e-16（圆整极限 1.0000e-09）；route P 拒绝两种错误约定；端到端回放 0.0000e+00、旧约定令 RAO 变 93.69%。
本文所有数字由 `src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py` 从 `results.json` 注入，无人工转录。

## ① 案例介绍

运动方程 `(M_RB + A)ẍ + Bẋ + Cx = F` 中的刚体惯量矩阵 `M_RB` 是 TensorBEM 频域全链（辐射/绕射 → RAO → 短期预报）最底层的输入。广义坐标取在参考点（`(n, r×n)` 的取矩点）而质心偏离该点时，`M_RB` 呈闭式块形式 `[[m I3, -m S(r_g)], [m S(r_g), I_g + m S(r_g)ᵀ S(r_g)]]`。

本锚点此前已由闭式块形式（route R）、面元带符号四面体积分（route M）与 capytaine 独立装配（route X）验证。本单补的是两件此前缺失的东西：**(a) 机器注册**——锚点有报告无注册表条目，`scripts/case_pool_run.py` 无法运行它；**(b) route P**——前三条路线全部以块形式或其转写为参照，没有任何一条从牛顿-欧拉动量定义反推符号约定。

## ② 怎么做的

| 层 | 文件 / 入口 | 作用 |
|---|---|---|
| 新路线 | `src/tensorbem/rigid_body_mass.py::momentum_map` / `mass_matrix_from_momentum` / `momentum_residual` | `p = m(v_O + ω×r_g)`、`h_O = I_g ω + r_g×p`；单位 twist 的广义动量为 `M_RB` 的列 |
| 算例入口 | `src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py` | 基例 + 姿态扩展 + route P + 两项对照，落 `results.json`/`verdict.json`/`CASE.md` |
| 注册表 | `src/tensorbem/benchmarks/case_pool_registry.py` | 锚点 → 驱动 → 工件 → 文档 的机器映射 |
| 测试 | `tests/test_rigid_body_mass_momentum.py` | route P 一致性与判别力；注册表条目完整性 |

## ③ 参数表

| 参数 | 数值 |
|---|---|
| 半轴 a,b,c | [60.0, 12.0, 8.0] m |
| 密度 ρ | 1025.0 kg/m³ |
| 质心偏置 r_g | [5.0, 0.0, -3.0] m |
| 旋转（yaw, pitch, roll） | [20.0, 15.0, 25.0] deg |
| 面元细分 | [1, 2, 4] |
| 接受层级自 | 细分 3 |
| 判据 | 逐元素 < 3% |
| 模式顺序 | surge, sway, heave, roll, pitch, yaw |

## ④ 计算结果

**route P 一致性（4 种配置，出厂装配 vs 动量定义）**

| 配置 | 最大逐元素相对误差 | 最差位置 | 矩阵尺度 |
|---|---|---|---|
| cg-offset (axis-aligned ellipsoid, diagonal inertia) | 0.0000e+00 | surge–surge | 1.914e+10 |
| reference-point offset o = (2, -1, 4) m | 1.1477e-16 | roll–yaw | 1.956e+10 |
| rotated body (yaw 20 / pitch 15 / roll 25 deg, full inertia tensor) | 0.0000e+00 | surge–surge | 1.774e+10 |
| two-component box body (payload station x = 30.0 m) | 1.1472e-16 | roll–yaw | 1.772e+10 |

**基例面元收敛阶梯（椭球裁判配置，闭式 vs 四面体积分）**

| 面元 | 体积误差 | 惯量张量误差 | 6×6 最差元素 | 旧约定对照 |
|---|---|---|---|---|
| 20 | 39.4539% | 56.1884% | 55.6477% | 160.55% |
| 80 | 12.6547% | 20.1245% | 19.8832% | 187.35% |
| 320 | 3.3839% | 5.5712% | 5.5005% | 196.62% |
| 1280 | 0.8606% | 1.4299% | 1.4115% | 199.14% |
| 5120 | 0.2161% | 0.3599% | 0.3552% | 199.78% |

**旋转体（满惯量张量）收敛阶梯**

| 细分 | 面元 | 6×6 最差元素 | 接受 |
|---|---|---|---|
| 1 | 80 | 19.9305% | ❌ |
| 2 | 320 | 5.5144% | ❌ |
| 3 | 1280 | 1.4151% | ✅ |
| 4 | 5120 | 0.3561% | ✅ |

**端到端调用者回放（`tensorbem.rao.compute_rao_at_frequency`，ω = 0.600 rad/s，108 面元）**

| 模态 | \|RAO\| 出厂装配 | \|RAO\| 注入旧耦合约定 | 相对变化 |
|---|---|---|---|
| surge | 2.085462e-01 | 3.207771e-01 | 53.8159% |
| sway | 2.921468e-02 | 1.992523e-02 | 31.7972% |
| heave | 4.251956e-01 | 4.245664e-01 | 0.1480% |
| roll | 3.607652e-03 | 6.987810e-03 | 93.6941% |
| pitch | 2.501301e-02 | 2.510888e-02 | 0.3833% |
| yaw | 5.949149e-04 | 5.652105e-04 | 4.9930% |

回放判据：把闭式块形式注入求解器唯一的质量矩阵调用点后，RAO 复现到 0.0000e+00（圆整极限 1.0000e-09）——证明求解器读的就是这一矩阵；旧耦合约定使 roll 的 RAO 改变 93.69%，远超锚点判据 3%，端到端可分辨。

**对照（判别力）**

| 对照 | 能量恒等式（只测 sym(M)） | route P | 判决 |
|---|---|---|---|
| 耦合块整体翻转（仍对称：True） | 17.548% | 200.0% | 拒绝 ✅ |
| 反对称污染 1e+06 kg·m² | 2.14e-16（盲） | 5225.6% | 拒绝 ✅ |

**外部裁判路线（route T，trimesh Mirtich 积分 + uv_sphere 离散，可选依赖）**

| 层 | 外部实现 | 结果 |
|---|---|---|
| 裁判自校准（边长 3.0 m 立方体 vs `I=m L²/6`） | trimesh | 体积误差 0.0000e+00、惯量误差 0.0000e+00 |
| 同一网格两积分器一致性（4 层，最差） | trimesh vs 四面体 | 1.6061e-11（限值 1.0000e-09） |
| 船型体（Wigley，4704 面元）vs 闭式积分 | trimesh | 6×6 最差 0.1248%（判据 3%，最差位置 roll-roll） |
| 拓扑守卫外部复核（5 个变体） | trimesh `is_watertight` | 守卫从不更宽松：True；外部漏检：single flipped face |

外部路线自身的收敛阶梯（椭球，uv_sphere + trimesh，参照=闭式块形式）

| uv 计数 | 面元 | 6×6 最差元素 | 最差位置 | 旧耦合约定对照 | 接受 |
|---|---|---|---|---|---|
| 16 | 1792 | 2.0985% | yaw-yaw | 198.75% | — |
| 32 | 7680 | 0.5002% | yaw-yaw | 199.70% | ✅ |
| 64 | 31744 | 0.1220% | yaw-yaw | 199.93% | ✅ |
| 128 | 129024 | 0.0301% | yaw-yaw | 199.98% | ✅ |

接受层级（uv ≥ 32）最差 0.5002%，拟合收敛阶 -0.993（误差 ~ 面元数^-0.99）。

外部裁判对拓扑守卫的逐变体裁决

| 变体 | 守卫接受 | 外部接受 | 体积误差 | 直接采信外部的 6×6 最差元素 |
|---|---|---|---|---|
| closed (control) | 是 | 是 | 3.3839% | 5.501% |
| open patch (every 7th face removed) | 否 | 否 | 18.1530% | 385276.270% |
| single flipped face | 否 | 是 | 5.4248% | 330514.616% |
| duplicated face | 否 | 否 | 3.1575% | 39795.751% |
| inward winding (all faces) | 否 | 否 | 196.6161% | 未定义（质量≤0） |

第 3 行是净增量：**单个翻转面**在 trimesh 的边关联判据下仍算 watertight（外部实现会直接采信它并积出一个偏 2% 的体积），而 TensorBEM 守卫按有向半边判定并拒绝。外部实现比守卫**更宽松**，因此守卫不能被外部库替代。

## ⑤ 注意点

- **能量恒等式的盲区**：`0.5 xᵀ M x` 只看到 `sym(M)`，任意反对称污染对它完全不可见（上表实测能量残差为 0 量级）。route P 逐元素定义式的比较补上了这一盲区，这是它相对既有检查的净增量。
- **块符号翻转的归一化陷阱**：若以「整矩阵最大元素」归一，耦合块符号错误（~`m|r_g|`）会被惯性对角项（~`I_O`）稀释到 1%量级，恰好落在 3% 判据以内。故 route P 采用**逐元素**相对误差（`max(|exact_ij|, 1e-6·scale)`），与基例口径一致。
- **端到端那一层只钉到调用点**：回放注入的位置是 `tensorbem/rao.py` 里唯一的 `compute_mass_matrix(...)` 调用，因此它证明的是「求解器读这一矩阵、且只读它」，不是「`M_RB` 的块约定已与外部 BEM 商软对齐」。后者由 `docs/case_rigid_body_mass.md` 的 Sensitivity C 与 capytaine 交叉装配（本锚点 orientation 例）覆盖。
- **假设与适用范围**：均匀密度刚体、参考点为广义坐标取矩点；不覆盖附加质量 `A` 的约定（那是辐射问题的责任）；`M_RB` 本身在求解器里的落地见本文 ④ 的端到端调用者回放。
- **回放残差恰为 0 的含义**：出厂 `compute_mass_matrix` 与用于注入的闭式块形式是同一函数，故残差是精确 0 而非 1e-16 量级。这一项证明的是「调用者除该函数外不再给矩阵加任何东西」（例如额外的质心平移）；它的非平凡性由同组的旧耦合约定对照提供——若调用点被绕过，注入旧约定就不会移动 RAO（实测 roll 变化 93.69%）。

- **route T 与前四条路线不共享的层**：外部裁判换掉了离散（uv_sphere 而非 icosphere）与积分（Mirtich 投影而非带符号四面体），但仍共用闭式参照、`rigid_body_mass_matrix` 的装配与逐元素误差口径；因此它证伪的是「质量特性输入」，块约定仍由 route P 负责。
- **外部判据比守卫更宽松**：trimesh 的 `is_watertight` 只看边关联次数，不看绕向，单个翻转面对它不可见（见 ④ 表格第 3 行）；把拓扑校验外包给第三方库会引入盲区。
- **三角网格的公共盲区**：trimesh 与 TensorBEM 都积分三角化表面，任何「所有三角网格共有」的几何误差（例如半轴标错）不在本锚点覆盖范围内。
## ⑥ 误差判定

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 6×6 最差元素（接受层级 = 细分 ≥ 3） | 闭式块形式 | 1.4151% | 是 |
| RAO 回放（注入闭式块形式） | 出厂 RAO | 0.0000e+00 | 是（<1e-9 圆整） |
| 端到端判别力（旧耦合约定 → RAO 变化） | 锚点判据 3% | 93.69% | 是（超判据 ⇒ 可分辨） |
| route P vs 出厂装配（4 配置最差） | 动量定义 | 1.1477e-16 | 是（远小于，属圆整） |
| route T：外部网格+外部积分（接受层级最差） | 闭式块形式 | 0.5002% | 是 |
| route T：Wigley 船型体 6×6（4704 面元） | 闭式积分 | 0.1248% | 是 |
| route T：两积分器在共享网格上的一致性 | 逐元素（限值 1.0000e-09） | 1.6061e-11 | 是（圆整量级） |
| 旧耦合约定对照（几何上必错） | 闭式块形式 | 200.00% | 否（预期，判别力证据） |

检查项：`{"rao_caller_verified": true, "rao_caller_replay_roundoff": true, "rao_caller_legacy_coupling_material": true, "worst_accepted_entry_within_tolerance": true, "momentum_route_roundoff": true, "momentum_route_rejects_flipped_coupling": true, "momentum_route_beats_energy_blind_spot": true, "base_case_passed": true, "orientation_case_passed": true, "external_integrator_roundoff": true, "external_mesh_route_within_tolerance": true, "external_route_rejects_legacy_coupling": true, "external_ship_shape_within_tolerance": true, "external_topology_guard_agrees": true, "external_guard_is_load_bearing": true, "legacy_coupling_rejected": true, "orientation_negative_control_rejected": true}`

