# CASE-BEM-RIGID-BODY-MASS — 6-DOF 刚体质量矩阵闭式块形式验证

**判定：PASS** — 在 1280 / 5120 面元两个层级上，6×6 矩阵逐元素相对误差 ≤ 1.412 % < 3 %；装配器与闭式块形式之间的残差为机器零（0）；同时发现并修复了平台主求解路径 `hydrostatic.compute_mass_matrix` 的耦合块符号错误 + 平行轴重复计数缺陷（该项在修复前能量恒等式误差 30.4 %）。

## ① 问题与参考出处

船舶/海洋结构 6 自由度运动方程 `(M_RB + A) ẍ + B ẋ + C x = F` 中的刚体惯量矩阵 `M_RB` 是整条 BEM 求解链（辐射/绕射 → RAO → 短期预报）最底层的输入。当广义坐标取在**参考点**（body-fixed 原点，`(n, r × n)` 的取矩点）而质心偏离该点时，`M_RB` 呈闭式块结构：

```
M_RB = [[ m I3 ,      -m S(r_g)                    ],
        [ m S(r_g) ,   I_g + m S(r_g)^T S(r_g)      ]]
```

其中 `S(a) b = a × b`，`I_g` 为**过质心**的惯量张量，下右块由平行轴定理给出 `I_O`。

- **ref 出处**：Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion Control, Sec. 3.3 'Rigid-Body Dynamics', eq. (3.44): M_RB = [[m I3, -m S(r_g)], [m S(r_g), I_b + m S(r_g)^T S(r_g)]]; solid-ellipsoid second moments I_cg = m(b^2+c^2, a^2+c^2, a^2+b^2)/5 (Lamb 1932 Ch. VI). All numbers machine-evaluated from these closed forms (solid_ellipsoid_mass_properties / rigid_body_mass_matrix); no table copied.
- **零手抄做法**：所有 ref 数值由 `tensorbem.rigid_body_mass.solid_ellipsoid_mass_properties` + `rigid_body_mass_matrix` 在机器算术中**求值**得到（`docs/cases/case_bem_rigid_body_mass/run.json` 的 `reference` 段即其输出），报告与本文件中的数字全部由 `make_report.py` 从 `run.json` 注入，无人工转录。
- **诚实声明**：本工作树中没有 Fossen (2011) 的纸质/电子副本，因此**未做页级核对**（不声称 L5 外部锚点）。可独立核对的部分用两条互不依赖的路线完成：闭式解路线与面元四面体积分路线（见 ②），并附加与参考系无关的动能恒等式仲裁（见 ⑤/⑥）。

## ② 方法与实现

| 层 | 文件 / 入口 | 作用 |
|---|---|---|
| 能力实现 | `src/tensorbem/rigid_body_mass.py` | 闭式块形式装配 `rigid_body_mass_matrix`、平行轴/逆平行轴 `inertia_about_point` / `inertia_about_centroid`、闭式椭球质量特性 `solid_ellipsoid_mass_properties`、闭合面元四面体积分 `mesh_mass_properties`、物理仲裁 `energy_residual` / `block_form_residual` |
| 算例入口 | `src/tensorbem/benchmarks/rigid_body_mass_case.py` (`python -m ...`) | 面板收敛 + 装配器形式校验 + 负对照 + 两项敏感性 + 平台装配对照，落 `run.json` |
| 平台集成 | `src/tensorbem/hydrostatic.py::compute_mass_matrix` | 委托到同一装配器（`inertia_about="origin"` 语义保留）；调用方 `tensorbem/rao.py:434`、`tensorbem/ice_rao.py:413` |
| 测试 | `tests/test_rigid_body_mass.py`（17 项） | 块形式、平行轴、面元积分、动能恒等式、负对照、平台集成 |

**两条互不依赖的实测路线**：(a) *装配路线*——把解析质量特性喂给装配器，检验块形式本身（应为机器零）；(b) *几何路线*——用 TensorBEM 自带 `ellipsoid_mesh` 面板化，把闭合三角网格分解为带符号四面体做精确多项式积分（体积/一阶矩/二阶矩，`V/20·(Σ v_k v_k + (Σ v_k)(Σ v_k))`），得到体积、质心与惯量张量，再装配 6×6。后者的离散误差正是 3 % 判据所度量的量。

**物理仲裁（与参考系无关）**：对任意 twist `x = (v, ω)`，

```
0.5 x^T M_RB x  ==  0.5 m |v + ω × r_g|^2 + 0.5 ω · I_g · ω
```

该恒等式对偶块符号错误与平行轴重复计数**都以 O(1) 相对误差**暴露；而仅做对称性检查无法发现（对称性在符号同时翻转时仍成立——这正是平台旧实现的漏洞，见 ⑥）。

## ③ 参数表（无量纲量完整）

| 参数 | 符号 | 数值 | 无量纲化 |
|---|---|---|---|
| 长 / 宽 / 深（全长） | L, B, T | 120 / 24 / 16 m | L/B = 5.000, B/T = 1.500 |
| 半轴 | a, b, c | 60 / 12 / 8 m | a:b:c = 1 : 0.2000 : 0.1333 |
| 排水体积 | ∇ | 24127.4 m³ | ∇/L³ = 0.0140 |
| 质量 | m = ρ∇ | 24730617 kg | — |
| 密度 | ρ | 1025 kg/m³ | — |
| 质心偏置 | r_g | (5, 0, -3) m | &#124;r_g&#124;/L = 0.0486 |
| 回转半径（过质心） | k_x, k_y, k_z | 6.45, 27.07, 27.36 m | k_x/L = 0.0537, k_y/L = 0.2256, k_z/L = 0.2280 |
| 惯量（过质心） | I_xx, I_yy, I_zz | 1.02879e+09, 1.81226e+10, 1.85183e+10 kg·m² | I_xx/(m a²) = 0.0116, I_yy/(m a²) = 0.2036, I_zz/(m a²) = 0.2080 |
| 参考点（广义坐标取矩点） | o | (0, 0, 0) m | 与质心同坐标系 |
| 面元层级 | N | 20 / 80 / 320 / 1280 / 5120 | icosphere 细分 0–4 |
| 判据 | — | 逐元素相对误差 < 3% | 收敛判据；机器复核 |

模式顺序（全局约定，`tensorbem.rigid_body_mass.MODES`）：`surge`, `sway`, `heave`, `roll`, `pitch`, `yaw`。

## ④ 对照表（ref vs got，逐项打勾）

**基础质量特性**（接受层级 5120 面元，细分 4）：

| 量 | ref（闭式） | got（面元积分） | 相对误差 | < 3 % |
|---|---|---|---|---|
| 排水体积 ∇ | 24127.432 m³ | 24075.296 m³ | 0.216 % | ✅ |
| 质量 m | 24730617 kg | 24677179 kg | 0.216 % | ✅ |
| I_xx（过质心） | 1.0287937e+09 kg·m² | 1.0250915e+09 kg·m² | 0.360 % | ✅ |
| I_yy（过质心） | 1.8122596e+10 kg·m² | 1.8057381e+10 kg·m² | 0.360 % | ✅ |
| I_zz（过质心） | 1.8518286e+10 kg·m² | 1.8451647e+10 kg·m² | 0.360 % | ✅ |
| 质心位置 | (5, 0, -3) m | (5, -3.53179e-15, -3) m | 绝对误差 1.33e-14 m | ✅ |

**6×6 矩阵逐元素对照**（5120 面元；ref = 闭式块形式）：

| i \ j | **surge** | **sway** | **heave** | **roll** | **pitch** | **yaw** |
|---|---|---|---|---|---|---|
| **surge** | 2.46772e+07 (0.22 %) ✅ | 0 (0.00 %) ✅ | 0 (0.00 %) ✅ | -0 (0.00 %) ✅ | -7.40315e+07 (0.22 %) ✅ | 8.71546e-08 (0.00 %) ✅ |
| **sway** | 0 (0.00 %) ✅ | 2.46772e+07 (0.22 %) ✅ | 0 (0.00 %) ✅ | 7.40315e+07 (0.22 %) ✅ | -0 (0.00 %) ✅ | 1.23386e+08 (0.22 %) ✅ |
| **heave** | 0 (0.00 %) ✅ | 0 (0.00 %) ✅ | 2.46772e+07 (0.22 %) ✅ | -8.71546e-08 (0.00 %) ✅ | -1.23386e+08 (0.22 %) ✅ | -0 (0.00 %) ✅ |
| **roll** | 0 (0.00 %) ✅ | 7.40315e+07 (0.22 %) ✅ | -8.71546e-08 (0.00 %) ✅ | 1.24719e+09 (0.33 %) ✅ | -3.34433e-07 (0.00 %) ✅ | 3.70158e+08 (0.22 %) ✅ |
| **pitch** | -7.40315e+07 (0.22 %) ✅ | 0 (0.00 %) ✅ | -1.23386e+08 (0.22 %) ✅ | -3.34433e-07 (0.00 %) ✅ | 1.88964e+10 (0.35 %) ✅ | -1.57346e-07 (0.00 %) ✅ |
| **yaw** | 8.71546e-08 (0.00 %) ✅ | 1.23386e+08 (0.22 %) ✅ | 0 (0.00 %) ✅ | 3.70158e+08 (0.22 %) ✅ | -1.57346e-07 (0.00 %) ✅ | 1.90686e+10 (0.36 %) ✅ |

ref 对角线（闭式）：surge = 24730617, sway = 24730617, heave = 24730617, roll = 1.2513692e+09, pitch = 1.8963437e+10, yaw = 1.9136552e+10。

最差元素：`3` 细分下 max 逐元素误差 **1.412 %** （判据 3 %，通过）。

**装配器层面（解析质量特性直接装配，检验块形式本身）**：

| 检查 | 结果 |
|---|---|
| 与闭式块形式的最大相对残差 | 0 |
| 与手写块形式（独立复推，不调装配器）的相对差 | 0 |
| 对称性误差 | 0 |
| 动能恒等式残差 | 2.14e-16 |
| 正定性（最小特征值） | 1.97214e+07（> 0：True）|

**平台主求解路径（`hydrostatic.compute_mass_matrix`，被 `rao.py` / `ice_rao.py` 调用）**：

| 检查 | 结果 |
|---|---|
| 与闭式块形式的最大逐元素误差 | 0 |
| 块形式残差 | 0 |
| 动能恒等式残差 | 2.14e-16 |
| 对称性误差 | 0 |

## ⑤ 收敛性与参数敏感性

### (1) 面元收敛（细分 0→4，20→5120 面元）

| 细分 | 面元 | ∇ 误差 | I_g 误差（最差主轴） | 6×6 最差元素误差 | < 3 % | 能量恒等式残差 | 墙钟 (s) |
|---|---|---|---|---|---|---|---|
| 0 | 20 | 39.454 % | 56.188 % | 55.648 % | ❌ | 4.2e-16 | 0.00 |
| 1 | 80 | 12.655 % | 20.125 % | 19.883 % | ❌ | 3.2e-16 | 0.00 |
| 2 | 320 | 3.384 % | 5.571 % | 5.501 % | ❌ | 2.1e-16 | 0.01 |
| 3 | 1280 | 0.861 % | 1.430 % | 1.412 % | ✅ | 4.5e-16 | 0.02 |
| 4 | 5120 | 0.216 % | 0.360 % | 0.355 % | ✅ | 3.5e-16 | 0.14 |

观测收敛阶（对面板数取对数）：体积 +0.82, +0.95, +0.99, +1.00；6×6 最差元素 +0.74, +0.93, +0.98, +1.00。
细化比 4（每级面元 ×4）下误差按 h² 律下降 4 倍左右，与内接多面体体积/二阶矩的 O(h²) 误差一致；因此 3 % 判据是**有意义的判据**：320 面元（细分 2）时仍然 5.501 % 未过线，1280 面元才通过。

### (2) 质心偏置敏感性（|r_g| 扫描）

| r_g 缩放 | &#124;r_g&#124; (m) | 耦合块 max&#124;M&#124; (kg·m) | max&#124;M_ur&#124;/(m&#124;r_g&#124;) | 转动块 vs 平行轴 | 能量残差 |
|---|---|---|---|---|---|
| 0.0 | 0 | 0 | — | 0 | 2e-16 |
| 0.5 | 2.9155 | 6.18265e+07 | 0.857493 | 0 | 2.2e-16 |
| 1.0 | 5.831 | 1.23653e+08 | 0.857493 | 0 | 2.1e-16 |
| 2.0 | 11.662 | 2.47306e+08 | 0.857493 | 0 | 2.6e-16 |

拟合指数：耦合块范数 ∝ &#124;r_g&#124;^**1.000000**（期望 1，即 `m S(r_g)` 对 r_g 线性）；转动块与平行轴定理 `I_g + m S^T S` 的残差为机器零。这说明耦合块不是拟合/硬编码，而是随质心偏置线性增长的结构项。

### (3) 参考点协变性（同一物体换取矩点）

| 检查 | 结果 |
|---|---|
| 参考点 | [0.0, 0.0, 0.0] → [10.0, 0.0, -2.0] m |
| `M_2 - A^T M_1 A` 相对残差（A 为 twist 变换） | 1.99e-16 |
| 原参考点动能残差 | 2.14e-16 |
| 新参考点动能残差 | 2.98e-16 |
| 转动块对角（原 → 新） | [1251369238.874378, 18963437398.594326, 19136551720.17774] → [1053524299.921907, 18765592459.641857, 19136551720.17774] |

## ⑥ 失败模式、缺陷与工件

### (a) 本轮发现并修复的平台缺陷（最重的一条）

`tensorbem.hydrostatic.compute_mass_matrix` 修复前装配的是 `[[m I3, -m S^T], [-m S, I_O + m S S^T]]`（docstring 声明 `J` 为过原点惯量）。该形式**仍然对称**，所以既有测试（对称性 + 与同一错误形式自比）全绿，缺陷被掩盖；但它与闭式块形式不符：

| 量（本算例本体） | 修复前 | 修复后 |
|---|---|---|
| 耦合块（translation-rotation）相对误差 | 200.0 % | 0.000 % |
| 转动块相对误差 | 4.394 % | 0.000 % |
| 动能恒等式残差 | 30.4 % | 2.1e-16 |
| 是否对称 | True | True |

根因两条：(i) 两个非对角块符号同时取反（等于取转置）——`-m S^T` 与 `-m S` 互换；(ii) 下右块在“过原点惯量”上**再加一次** `m S S^T`，即平行轴重复计数。影响面：`tensorbem/rao.py:434`（频域 RAO 主链）与 `tensorbem/ice_rao.py:413`（冰区 RAO）都用它做质量阵；对有质心偏置的浮体，耦合项符号错误 + 转动惯量被放大。修复后装配委托给 `rigid_body_mass.rigid_body_mass_matrix`（默认 `inertia_about="origin"` 保留原有语义），并同步修正了两处把错误约定当“精确解”的既有校验（`benchmarks/hydrostatic_rao_verification.run_mass_matrix_check`、`tests/test_mutation_gaps.py`），同时把 `benchmarks/hydrostatic_verification.run_mass_matrix_energy_check` 的 CG 语义参数显式化——该物理仲裁此前**无任何测试断言**，因此缺陷长期未被发现。

### (b) 其他失败模式（已实测/已知边界）

- 粗网格必然不过判据：20 面元时 6×6 最差元素误差 55.648 %，320 面元仍 5.501 %；判据只在细分 ≥ 3（1280 面元）以上成立。
- 点质量退化：`I_g = 0` 时块形式奇异（存在无动能 twist `ω × r_g = -v`），`tests/test_rigid_body_mass.py::test_symmetric_and_positive_definite` 用该极限把正定性边界钉住。
- 网格朝向/闭合：`mesh_mass_properties` 对非闭合或内向网格抛 `ValueError`（体积非正），不接受“大致闭合”的输入。
- 未含流体：本算例是**刚体**惯量，不含附连质量 A(ω)；两者在运动方程里相加，A 的验证属另外的算例（CASE-BEM-ELLIPSOID-ADDED-MASS 等）。
- 参考源等级：未做 Fossen 页级核对（见 ①），故本单不主张 L5 外部锚点；但结论不依赖该核对——装配器形式、面元积分路线与动能恒等式三者互为独立仲裁。

### (c) 工件与复现

| 工件 | sha256 |
|---|---|
| `docs/cases/case_bem_rigid_body_mass/run.json` | `fe1772ed8dfebb9d6d53ec7dca3f549c6428054e6165ad1f231ab7aaeacb4340` |
| `docs/cases/case_bem_rigid_body_mass/fig_convergence.png` | `1915e96c9642971b6eb612b11b0ad15f78fbdf1f78853afdbc4a29feb7c1ed0b` |
| `docs/cases/case_bem_rigid_body_mass/fig_block_matrix.png` | `4a622ed21704e6646162f06e2b19280bbc6a33f0f6db21e116778c77eb13703e` |
| `docs/cases/case_bem_rigid_body_mass/fig_energy_identity.png` | `178acee1155af2b67e244b22d17150e688371cb74f35331e49a043b45c1a91aa` |

```bash
# 一键复现（本工作树）
cd <TensorBEM checkout>
PYTHONPATH=src python -m tensorbem.benchmarks.rigid_body_mass_case \
    --subdivisions 0 1 2 3 4 \
    --out docs/cases/case_bem_rigid_body_mass/run.json
PYTHONPATH=src python3 docs/cases/case_bem_rigid_body_mass/make_figures.py
PYTHONPATH=src python3 docs/cases/case_bem_rigid_body_mass/make_report.py
PYTHONPATH=src python -m pytest tests/test_rigid_body_mass.py -q
```

图 1 `fig_convergence.png`：体积/惯量/6×6 最差元素误差 vs 面元数（含 3 % 判据线）。
图 2 `fig_block_matrix.png`：闭式矩阵、面元路线矩阵与逐元素相对偏差（对数色标）。
图 3 `fig_energy_identity.png`：正确装配 / 平台装配 / 两种受控扰动（符号翻转、修复前公式）的动能恒等式残差。

### (d) 生态摩擦（如实记）

- `artifacts/` 与 `runs/` 在 `.gitignore` 中（且平台推送门会丢弃 `docs/` 外的 `*.png`），所以本算例工件落在 `docs/cases/case_bem_rigid_body_mass/`；这是仓库既有约定，不是新发明。
- 仓库里**已有的** `run_mass_matrix_check` 把错误形式当作“精确解”自比，属于“测试只断言自洽、不断言不变量”的典型：绿灯 ≠ 物理正确。已修正并在注释中留痕。
- `tests/test_doc_claim_audit.py` 在本单基线与本单工作树上均 3 项失败（52 个历史文档未登记 baseline / 工件过期），与本次改动无关，未在本单范围内处理。

### (e) 环境

Python 3.11.6，numpy 2.4.6，Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38（loongarch64）。总运行 0.18 s（本地 CPU，无需集群）。

