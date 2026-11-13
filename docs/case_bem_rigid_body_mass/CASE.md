# CASE-BEM-RIGID-BODY-MASS — 6×6 刚体质量矩阵的第三条独立路线（动量定义）与机器注册

**判定：PASS** — 接受层级最大逐元素相对误差 1.4151%（判据 3%）；动量定义路线（route P）与出厂装配在 4 种配置上一致到 1.1477e-16（圆整极限 1.0000e-09）；route P 拒绝两种错误约定。本文所有数字由 `src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py` 从 `results.json` 注入，无人工转录。

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

**对照（判别力）**

| 对照 | 能量恒等式（只测 sym(M)） | route P | 判决 |
|---|---|---|---|
| 耦合块整体翻转（仍对称：True） | 17.548% | 200.0% | 拒绝 ✅ |
| 反对称污染 1e+06 kg·m² | 2.14e-16（盲） | 5225.6% | 拒绝 ✅ |

## ⑤ 注意点

- **能量恒等式的盲区**：`0.5 xᵀ M x` 只看到 `sym(M)`，任意反对称污染对它完全不可见（上表实测能量残差为 0 量级）。route P 逐元素定义式的比较补上了这一盲区，这是它相对既有检查的净增量。
- **块符号翻转的归一化陷阱**：若以「整矩阵最大元素」归一，耦合块符号错误（~`m|r_g|`）会被惯性对角项（~`I_O`）稀释到 1%量级，恰好落在 3% 判据以内。故 route P 采用**逐元素**相对误差（`max(|exact_ij|, 1e-6·scale)`），与基例口径一致。
- **假设与适用范围**：均匀密度刚体、参考点为广义坐标取矩点；不覆盖附加质量 `A` 的约定（那是辐射问题的责任，见 `docs/case_bem_rigid_body_mass/` 中 RAO caller 回放一节）。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 6×6 最差元素（接受层级 = 细分 ≥ 3） | 闭式块形式 | 1.4151% | 是 |
| route P vs 出厂装配（4 配置最差） | 动量定义 | 1.1477e-16 | 是（远小于，属圆整） |
| 旧耦合约定对照（几何上必错） | 闭式块形式 | 200.00% | 否（预期，判别力证据） |

检查项：`{"worst_accepted_entry_within_tolerance": true, "momentum_route_roundoff": true, "momentum_route_rejects_flipped_coupling": true, "momentum_route_beats_energy_blind_spot": true, "base_case_passed": true, "orientation_case_passed": true, "legacy_coupling_rejected": true, "orientation_negative_control_rejected": true}`

