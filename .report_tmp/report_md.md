# TensorBEM#1193 管到底第 3 轮：C10 外部锚点研究工件的可移植性修复（CASE-BEM-CUMMINS-WIGLEY）

## 1 案例介绍

**物理背景**：CASE-BEM-CUMMINS-WIGLEY 是 TensorBEM 船舶总体性能平台的 6 自由度（surge/sway/heave/roll/pitch/yaw）深水自由面辐射基准算例。该算例的全部自持闸门（C3a Kramers-Kronig 自洽、C4 Cummins 变换往返、C5 时域衰减、C7 镜像解仲裁 A_inf）都是**自证**陈述——共享的建模错误（自由面核、ω 归一化、密度/重力约定、旋转中心符号）对它们全部不可见。因此算例声明了外部（L5）锚点：**闸门 C10**——把同一套面板（同一顶点、同一三角形）交给独立实现的自由面 BEM（capytaine）重解 6×6 辐射矩阵，逐格比对。同网格 ⇒ 残差是**方法差**，不是离散差，这是本算例第一个不能由 TensorBEM 自己与自己一致而产生的证据。

**几何与参数**：

| 项 | 值 |
| --- | --- |
| 船型 | Wigley，L=3.0 m，B=0.3 m，T=0.1875 m，Cb=4/9 |
| 排水体积 | 0.075 m³（解析 (4/9)LBT） |
| ρ / g | 1025 kg/m³ / 9.80665 m/s² |
| 水深 | 无限 |
| 频带 | 0.2–12.0 rad/s，41 节点 |
| 网格 | coarse 192 / medium 480 / fine 1080 面板 |
| 旋转中心 | (0, 0, 0) |
| 外来程序 | capytaine 3.1.dev0（loongarch64，numpy 2.4.6，Python 3.11.6） |
| 闸门判据 | 最大格归一化偏差 ≤ 10%（算例判据）；3% 为 owner 精度线，单列不混同 |

## 2 怎么做的

- **切枝**：本 PR 从当前 main（f595d7d5）切出（worktree 先 fast-forward 到 origin/main 再落改动）。
- **问题**：已提交的 `docs/case_bem_cummins_wigley_xval/xval_study.json` 中 `values.reference` 字段记录的是**生产者 worktree 的绝对路径**（`/srv/tensor/agent/work/m21839/benchmarks/...`）。该工件会在多台机器上被读取；绝对路径泄露生产者机器布局，使工件机器相关、不可复现。
- **修复**（最小改动，3 个文件，单一功能点）：
  1. `src/tensorbem/benchmarks/case_bem_cummins_wigley_xval.py`：`load_reference` 把参考位置记录为**仓库相对路径**（仅当参考文件在仓库外时回退到解析后的绝对路径）；
  2. `docs/case_bem_cummins_wigley_xval/xval_study.json`：用工具链（`run_study` + `write_outputs`）重新生成，`reference` 字段变为 `benchmarks/verified/casepool_bem_cummins_wigley/capytaine_ref.json`；已验证工件与 `run_study` 输出逐字节一致（`study == committed: True`），README 与 `render_document` 渲染一致；
  3. `tests/test_case_bem_cummins_wigley_xval.py`：新增 `test_reference_path_is_repo_relative_and_portable`——记录路径必须非绝对、必须指向仓库内冻结的 capytaine 参考、且必须实际解析到该文件。
- **验证**：xval 套件 13/13 本地通过（含新测试）；全套测试（456 个测试文件）按 CI 同款命令 `pytest -q -n auto --maxprocesses 8 --dist loadfile` 递交 外协方（cj6422）执行。
- **热文件声明**：本 PR 仅触及基准算例模块 `case_bem_cummins_wigley_xval.py`（非 TensorBEM 核心模块）、其提交工件 `xval_study.json` 与其测试文件，共 3 个文件（≤10），聚焦单一功能点（工件可移植性），与其他开 PR 无路径重叠。

## 3 计算结果（程序化注入，零手抄）

**C10 闸门**（medium/lid，480 面板，identical_panels=True）：

| 项 | 值 |
| --- | --- |
| 闸门 | C10 **passed**（判据 10%） |
| 最差格 | heave-heave 阻尼，**6.09%**（绝对 19.78，尺度 324.62） |
| 最差频率 | 10.8325 rad/s |
| 计分格 | 20（52 个结构零格排除，单列绝对偏差） |
| ≤3% 格数 | **16/20** |
| 附加质量最差格 | roll-roll 4.89% |
| 阻尼最差格 | heave-heave 6.09% |
| capytaine 自身对称化误差 | A 8.21% / B 32.09%（记录于参考文件） |

**面板阶梯**（最差格偏差，同网格方法差视角）：coarse(192) 7.79% → medium(480) 6.09% → fine(1080) 9.59%。

**可移植性修复前后**：

| 项 | 修复前 | 修复后 |
| --- | --- | --- |
| `xval_study.json` `values.reference` | `/srv/tensor/agent/work/m21839/benchmarks/verified/casepool_bem_cummins_wigley/capytaine_ref.json`（绝对路径，泄露生产者 worktree） | `benchmarks/verified/casepool_bem_cummins_wigley/capytaine_ref.json`（仓库相对） |
| 不变量测试 | 无 | `test_reference_path_is_repo_relative_and_portable`（新增，通过） |

## 4 结果图

- 图 1：逐频率最差格偏差曲线（TensorBEM vs capytaine，medium/lid），叠加 10% 闸门线与 3% 精度线；
- 图 2：6×6 相对偏差热图（附加质量 A / 阻尼 B，结构零格记 0）。

## 5 注意点

1. **同网格是闸门前提**：参考文件存几何 sha256，`geometry_matches` 每次重算比对，偏差不含网格差；
2. **结构零格不计分**：幅值 < 同 DOF 对角幅值 1% 的格排除出闸门（用数值零除自身幅值不是证据），其绝对偏差单列；
3. **capytaine 矩阵不正对称**（逐列直接 BEM），对称化误差记录于参考文件（A 8.2%、B 32.1%），TensorBEM 侧按需对称化；
4. **lid 归因在 primary 层级不可测**：medium 级 lid_panel_count=0（`testable=false`），lid 效应仅在 coarse/fine 级可观测；
5. **L5 边界**：这是独立程序锚点，不是实验锚点——排除共享实现错误，不能排除两程序共有的建模假设（线性势流、无限水深、无粘）；
6. **阶梯非单调**：fine 级最差偏差（9.59%）高于 medium（6.09%），频带上端（ω→12 rad/s）是两程序核处理差异最大的区域；因同网格，该差异是方法差而非离散差。

## 6 误差判定（逐项）

本算例的基准是**独立程序**（capytaine），非解析解/实验；"误差"为两程序间逐格偏差。owner 精度线 3% 与算例池闸门判据 10% 分列：

| 关键量 | 偏差 | 是否 <3% |
| --- | --- | --- |
| 最差格（heave-heave 阻尼） | 6.09% | **否**（超 3% 线，在 10% 算例判据内） |
| 附加质量最差格（roll-roll） | 4.89% | 否 |
| 阻尼最差格（heave-heave） | 6.09% | 否 |
| 计分格中 ≤3% 的份额 | 16/20（80%） | 80% 的格满足 3% 线 |
| C10 闸门（10% 判据） | 6.09% ≤ 10% | **PASS** |

**显式回答**：最差格**不是** <3%（6.09%）；16/20 计分格 <3%；闸门按 10% 算例判据通过。基准为独立程序（capytaine）而非解析解/实验，故此处"误差"是程序间方法差，不是对真值的误差；3% 线（owner 精度线）与 10% 判据（算例池闸门）分列报告、不混同。本单修复的工件可移植性缺陷（绝对路径泄露）已消除，并有不变量测试钉住。
