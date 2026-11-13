# CASE-BEM-RIGID-BODY-MASS 扩展：姿态协变 + 外部实现锚点

**判定：PASS** — 旋转刚体的 6×6 与闭式块形式逐元素最大相对误差 1.4151%（判据 3%）；姿态协变恒等式 `M(R·b) = A M(b) A^T` 残差 1.08e-16（机器零）；独立实现 capytaine 3.1.dev0 的刚体惯量在质量块/耦合块上与 TensorBEM 一致到 1.2e-15 / 8.8e-15。

## ① 案例介绍

运动方程 `(M_RB + A)ẍ + Bẋ + Cx = F` 中的刚体惯量矩阵 `M_RB` 是 BEM 全链最底层的输入。基础算例 `docs/case_rigid_body_mass.md` 已在**轴对齐**椭球上验证闭式块形式，但轴对齐体的惯量张量是对角的：惯性积 `I_xy, I_xz, I_yz` 全为零，roll-pitch-roll 之间的交叉项按对称性为零。于是两条真实船舶必经的通道**未被覆盖**：(a) 通用惯量张量（含惯性积）；(b) 姿态（R 群）的协变性——基础算例只测了参考点平移（T 群）那一半。本扩展把同一实心椭球绕参考点刚性旋转 `R = Rz(yaw)·Ry(pitch)·Rx(roll)`，使 `I_g` 变为满张量，并引入**独立实现锚点**（capytaine 3.x 的 `compute_rigid_body_inertia`）来钉住耦合块符号约定。

物理不变量：对任意 twist `x_ref = A x_body`，`A = blockdiag(R,R)`，必须有

```
M(旋转后物体) = A M(物体) A^T
```

| 参数 | 数值 | 说明 |
|---|---|---|
| 半轴 a,b,c | [60.0, 12.0, 8.0] m | L/B/T = 120.0/24.0/16.0 m |
| 密度 ρ | 1025.0 kg/m³ | 排水体积 24127.43 m³ |
| 质心偏置 r_g | [5.0, 0.0, -3.0] m | 未旋转系 |
| 欧拉角 | yaw 20.00°, pitch 15.00°, roll 25.00° | R = Rz·Ry·Rx |
| 面元层级 | [1, 2, 3, 4] | icosphere 细分 |
| 判据 | 逐元素相对误差 < 0.03 | 零参照项以矩阵尺度归一 |

## ② 怎么做

| 层 | 文件/入口 | 作用 |
|---|---|---|
| 解析路线 | `rotated_reference()` | 闭式块形式作用于 `R r_g`, `R I_g R^T`；核对协变恒等式 |
| 面元路线 | `mesh_level()` | 旋转+平移面元网格 → 带符号四面体积分 → 装配 6×6 |
| 负对照 | `legacy_convention()` | 旧平台约定 `[[mI,-mS^T],[-mS, I_O+mSS^T]]`（仍对称） |
| 外部锚点 | `external_capytaine()` | capytaine 独立装配，逐块比较 |
| 回归修复 | `mesh_mass_properties()` | 二阶矩改为对 `reference_point` 取矩（见 ⑤） |
| 测试 | `tests/test_rigid_body_mass_orientation.py`（9 项） | 平移不变性、协变、负对照、外部锚点 |

## ③ 计算结果

**旋转后质量特性（解析，机器求值）**

| 量 | 数值 |
|---|---|
| I_xy | -5.171590e+09 kg·m² |
| I_xz | 4.142154e+09 kg·m² |
| I_yz | 1.351832e+09 kg·m² |
| max|I_off|/max|I_diag| | 0.2993 |
| M[roll,pitch] | -5.393220e+09 kg·m² |
| M[sway,yaw] | -6.436246e+07 |
| 协变残差 | 1.08e-16 |
| 块形式残差 | 0.00e+00 |
| 动能恒等式残差 | 4.76e-16 |

**面元收敛阶梯（rotated，逐元素）**

| 细分 | 面元 | 体积误差 | 惯量张量误差 | 6×6 最差元素 | 最差元素位置 | 动能残差 | <3% |
|---|---|---|---|---|---|---|---|
| 1 | 80 | 12.6547% | 20.1245% | 19.9305% | yaw-yaw | 3.5e-16 | ❌ |
| 2 | 320 | 3.3839% | 5.5712% | 5.5144% | yaw-yaw | 3.4e-16 | ❌ |
| 3 | 1280 | 0.8606% | 1.4299% | 1.4151% | yaw-yaw | 2.6e-16 | ✅ |
| 4 | 5120 | 0.2161% | 0.3599% | 0.3561% | yaw-yaw | 2.1e-16 | ✅ |

**姿态扫描（3 细分）**

| yaw | pitch | roll | max|I_off|/diag | 协变残差 | 6×6 最差元素 | <3% |
|---|---|---|---|---|---|---|
| 20.0° | 15.0° | 25.0° | 0.299 | 1.1e-16 | 1.4151% | ✅ |
| -35.0° | 10.0° | -12.0° | 0.434 | 2.1e-16 | 1.4148% | ✅ |
| 70.0° | -20.0° | 40.0° | 0.322 | 1.1e-16 | 1.4370% | ✅ |
| 90.0° | -0.0° | 90.0° | 0.000 | 1.6e-32 | 1.4115% | ✅ |
| -110.0° | 30.0° | -55.0° | 0.420 | 1.1e-16 | 1.4122% | ✅ |

**外部实现锚点（capytaine 3.1.dev0，5120 面元）**

| 块 | capytaine vs 闭式 | capytaine vs TensorBEM |
|---|---|---|
| translation_translation | 2.161e-03 | 1.208e-15 |
| translation_rotation | 2.161e-03 | 8.780e-15 |
| rotation_rotation | 4.028e-03 | 4.685e-04 |

capytaine 与 TensorBEM 用同一份面元网格：质量块与耦合块一致到机器零，说明两者用的是**同一个**Fossen 符号约定（耦合块 `-m S(r_g)`）；转动块差异 4.685e-04 来自 capytaine 用面心/面积的面积分近似（一阶/近似二阶），而 TensorBEM 用精确带符号四面体多项式积分——两者对闭式解的误差分别为 4.028e-03 与 3.561e-03。

## ④ 图

图 1 `fig_rotation_error_map.png`：闭式矩阵、面元矩阵与逐元素相对误差（对数色标）。图 2 `fig_rotation_convergence.png`：左=面元收敛（6×6 最差元素/惯量张量/体积 vs 面元数，含 3% 判据线）；右=五个互不相关姿态在 3 细分下的最差元素误差。

## ⑤ 注意点

1. **发现并修复一个真实缺陷**：`mesh_mass_properties` 的二阶矩积分用绝对坐标而把 `reference_point` 只用于体积/一阶矩，导致返回的 `inertia_cg` 依赖网格的绝对位置。把同一物体连同 `reference_point` 一起平移 (0,0,−20) m，`I_cg` 相对偏差达 **60.3%**（修复前）；修复后平移不变性达 4.2e-16。基础算例全部使用 `reference_point=(0,0,0)`，因此该缺陷从未被发现。
2. **平台主路径约定显式化**：`hydrostatic.compute_mass_matrix` 现支持 `inertia_about={"cg","origin"}`（默认 `"cg"`，与 `rao.py`/`ice_rao.py` 调用一致）；过去同一调用按不同假设使用会重复计入平行轴项，本算例已显式声明 `origin`。
3. **判据只在细分 ≥ 3 有意义**：80 面元时最差元素 19.931%，未过线；这是内接多面体 O(h²) 误差，不是算法错误。
4. **负对照**：旧约定的矩阵仍然对称，逐元素误差 200.00%、协变残差 0.038——对称性检查（旧测试全部依赖它）无法发现该缺陷，能量/协变仲裁才能。
5. **未含流体**：本算例只验证刚体惯量；附连质量 A(ω) 属 CASE-BEM-ELLIPSOID-ADDED-MASS 等。

## ⑥ 误差判定

| 关键量 | 参照 | 结果 | 是否 <3% |
|---|---|---|---|
| 旋转体 6×6 最差元素（3 面元） | 闭式块形式 | 1.4151% | ✅ |
| 惯量张量（4 面元） | 闭式 `R I_g R^T` | 0.3599% | ✅ |
| 姿态协变 | `A M A^T` | 1.08e-16 | ✅（机器零） |
| capytaine 全阵 vs 闭式（5120 面元） | capytaine 独立实现 | 0.4028% | ✅ |
| capytaine 耦合块 vs TensorBEM | 两实现互比 | 8.8e-15 | ✅（机器零） |

**是否 <3%：是**（最差元素 1.4151%，容差 3%）。协变恒等式与外部锚点均达机器零。

## 复现与工件

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.rigid_body_mass_orientation \
    --out docs/cases/case_bem_rigid_body_mass/rotation/run.json
PYTHONPATH=src python3 docs/cases/case_bem_rigid_body_mass/rotation/make_report.py
PYTHONPATH=src python -m pytest tests/test_rigid_body_mass_orientation.py -q
```

| 工件 | sha256 |
|---|---|
| `docs/cases/case_bem_rigid_body_mass/rotation/run.json` | `b94dcc1b88be0b16858f01509a6026f42fedb729907c433831962139a4c41445` |
| `docs/cases/case_bem_rigid_body_mass/rotation/fig_rotation_error_map.png` | `bbd349020f55686da9c87852841fc45e9300a9bdab51e9204ebc9a427a251344` |
| `docs/cases/case_bem_rigid_body_mass/rotation/fig_rotation_convergence.png` | `07180459dd1517ac5c09ae06c1d07082cef57d81a6a5cb4ad0e89602b6686d20` |
| `docs/cases/case_bem_rigid_body_mass/run.json` | `fe1772ed8dfebb9d6d53ec7dca3f549c6428054e6165ad1f231ab7aaeacb4340` |

环境：Python 3.11.6，numpy 2.4.6，capytaine 3.1.dev0，loongarch64。

