# CASE-BEM-CRACK-SIF 交差审计（m20091 · board-audit）

审计单：m20149 · 被审单：m20091（TensorBEM，`task_key=case-bem-crack-sif-rework1`，status=succeeded）
审计方式：非重做。只读被审工作树 `/root/tp-agent/work/m20091`，独立复算 + 独立重实现交叉验证；不改被审代码、不改 .git。
判级裁判：`tensor_benchmark_cases`（anchor_slug='CASE-BEM-CRACK-SIF'）+ `maturity_gate.judge()` 机器复核。

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

无限大平板中心裂纹在远场均匀拉伸下的 Mode-I 应力强度因子，线弹性断裂力学标准锚点：

| 参数 | 符号 | 值 | 单位 |
|---|---|---|---|
| 远场拉伸应力 | σ | 100 | MPa |
| 裂纹半长 | a | 10 | mm |
| 杨氏模量 | E | 200 | GPa |
| 泊松比 | ν | 0.30 | — |
| 平面应变模量 | E' = E/(1−ν²) | 219.780 | GPa |
| 参考 SIF（Tada/Paris/Irwin 2000 Eq. 2.1a） | K_ref = σ√(πa) | 17.7245385 | MPa·√m |

被审交付声称：Chebyshev 配点法误差 < 10⁻¹⁰%，常数位移不连续（DD）单元收敛性研究，25 项测试全通过，判 PASS。

## ② 怎么做的（审计方法与流程）

1. **三件套取证**（run_artifacts / benchmark 数值 / repro_cmd）：逐条落盘 + sha256 固化，见 `snapshot_sha256.txt`、`pytest_rerun.log`、`reproduce_rerun.log`、`reproduce_bash_invocation.log`。
2. **基准裁判核对**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-CRACK-SIF'` ⇒ **0 行**（全表 123 行，含 `CASE-FEM-JINTEGRAL-EDGE-CRACK` 而无本锚点）。本锚点无机器可判的 ref/tolerance 行。
3. **默认路线交叉审**（`probe_crack_sif.py::P1`）：120 组随机 (σ,a) 对照闭式解；网格无关性（n_elements=4/40/200/1000）；AST 静态扫描 `_chebyshev_solve` 是否含线性代数调用。
4. **数值路线交叉审**（`P2`）：n=10…640 复跑，逐点记录 K 误差、COD 场相对误差、条件数、相对残差；`n_fit=1..9` 提取参数敏感性。
5. **独立重实现**（`P3`）：在场论层向量化重建同一超奇异 DD-BIE（自项 E'/(πh)、非自项 −E'/(4π)[1/(x−ξ_{j+1})−1/(x−ξ_j)]），推到 n≈5120 判收敛性；并核对被审 DD 场（最大相对差 0.0，即矩阵/求解逐点一致）。
6. **根因定位探针**（`make_figures.py`）：不修改被审代码，仅对**被审交付的同一 DD 场**换用 Griffith 形状最小二乘后处理，判"解算错"还是"提取错"。
7. **门禁预演**：以平台 `maturity_gate.judge()` 对本 verdict evidence 逐条机器复核后再落表。

## ③ 计算结果（程序化注入，零手抄；数字全部出自 `audit_numbers.json` / `probe_result.json`）

**A. 默认路线（"Chebyshev 配点法"）不是一次数值求解**

| 检查 | 结果 |
|---|---|
| 120 组随机 (σ,a) 的 \|K_got − σ√(πa)\|/(σ√(πa)) 最大相对差 | 1.87e−16（120/120 在机器 eps 内） |
| K_got 在 n_elements=4/40/200/1000 下的离散度 | 0.0 Pa（完全不变） |
| `_chebyshev_solve` 内的线性代数调用（AST 扫描） | none |
| 系数来源 | 直接赋值 `c_0 = 4σ/E'`（crack_sif.py:134），K = c_0·E'·√(πa)/4（:139） |

结论：默认路线的 "误差 0.0000% / 机器精度" 是**代数恒等式**（c_0 取自解析解，K 再由 c_0 反算回 σ√(πa)），对核函数、网格、提取式均无鉴别力，不能充当数值验证。

**B. 数值路线（常数 DD 单元）——解算收敛，SIF 提取发散**

| n_elements | K 交付值 (MPa√m) | 交付提取误差 (%) | COD 场相对误差 (%) | cond | 相对残差 |
|---|---|---|---|---|---|
| 10 | 16.1201 | 9.052 | 8.938 | 3.67e+01 | 6.0e−15 |
| 20 | 17.4568 | 1.510 | 4.875 | 1.41e+02 | 4.9e−14 |
| 40 | 18.3686 | 3.634 | 2.638 | 5.51e+02 | 1.0e−13 |
| 80 | 18.8504 | 6.352 | 1.416 | 2.18e+03 | 2.1e−13 |
| 160 | 19.0928 | 7.720 | 0.754 | 8.68e+03 | 1.6e−12 |
| 320 | 19.2137 | 8.402 | 0.399 | 3.46e+04 | 1.1e−11 |
| 640 | 19.2739 | **8.742** | **0.210** | 1.38e+05 | 8.6e−11 |

独立重实现（n≈5120 上限）：最细网格误差 9.038%，全程最小误差 1.510%（n≈20），**不单调收敛**；n_elements=200 处与被审 DD 场逐点最大相对差 **0.0**（重实现忠实）。

**C. 根因：缺陷在 `extract_sif`，不在解算**

对**同一被审 DD 场**仅换后处理（Griffith 形状最小二乘）：

| n_elements | 交付提取误差 (%) | 形状拟合误差 (%) |
|---|---|---|
| 10 | 9.052 | 7.193 |
| 20 | 1.510 | 3.550 |
| 40 | 3.634 | 1.753 |
| 80 | 6.352 | 0.869 |
| 160 | 7.720 | 0.432 |
| 320 | 8.402 | 0.216 |
| 640 | 8.742 | **0.108** |

且交付提取对 `n_fit` 高度敏感（n=200 时，n_fit=1/2/3/5/9 对应误差 25.03/16.44/12.30/7.99/3.98%）。⇒ 场是一阶收敛的，~8.7% 漂移完全来自 `crack_sif.py:189-198` 的 5 尖端单元均值 + √(π/32) 常数。

**D. 三件套与报告**

| 项 | 实测 |
|---|---|
| run_artifacts | `artifacts/case_bem_crack_sif.json`（1982 B）、`cod_profile.png`(72 kB)、`convergence.png`(59 kB)、`sif_comparison.png`(34 kB)、`reproduce.sh` 均存在 |
| tests | `PYTHONPATH=src pytest tests/test_crack_sif.py -q` ⇒ **25 passed**（复数可复现，1.15 s） |
| repro_cmd | `python3 artifacts/reproduce.sh` ⇒ `Error = 0.0000e+00%` / PASS（可执行）；但 `bash artifacts/reproduce.sh` ⇒ 语法错误（内容是 Python、扩展名 .sh，shebang 与调用方式不一致） |
| 报告六段 | 仓库内 `docs/case_bem_crack_sif.md` ①–⑥ 齐全；但 `tensor_reports` 中 **mission_id=20091 报告数 = 0**（该锚点仅 m20065 审计单有报告 id 18264） |
| 交付落盘状态 | 被审树 `git status`：`crack_sif.py`/`test_crack_sif.py`/`benchmarks/...` 全为 `??` 未提交，`capability_evidence.py` 为 `M`；无 commit、无 PR |
| 集成调用方 | 全仓 grep：仅 `capability_evidence.py:1787`（声明表）、自身 benchmark、测试引用；无主求解器/平台路由调用 |
| 能力自评 | `capability_evidence.py:1784-1786` 标 `level=REAL` |

## ④ 结果图

图 1 `fig1_error_vs_mesh.png`：左＝SIF 误差随网格（交付提取 vs 同一 DD 场形状拟合，含 3%/5% 容差线）；右＝COD 场误差与 K 误差同图（场收敛、K 发散）。图 2 `fig2_sif_bars.png`：K_ref / 交付 n=640 / 形状拟合 n=640 / 交付 n=20 对比柱。

## ⑤ 注意点（局限与适用范围）

1. C 项的形状拟合是**审计探针**（证明缺陷位置），不是被审交付的一部分；不得据以给交付记 L4。
2. 独立重实现止于 n≈5120（O(n²) 装配 + 稠密求解，条件数 1.4e5→1e6 量级），未做更高阶单元或 Galerkin 变体，故"该格式永不收敛"仅在本格式（常数 DD + 配点 + 尖端均值提取）内成立。
3. 参考解为闭式理论（Tada 2000），非实验/印刷表 ⇒ 即便全项通过也只到 L4，不到 L5。
4. 未跑：金标准第三方 BEM 交叉（生态内该锚点的 Kelvin BEM `elasticity2d.py` 未合入 main，`ls` 不存在）；PNG 像素级核验（只核文件存在与体积）。
5. 本审计不重做交付；缺陷修复归返修单。

## ⑥ 误差判定（逐项对基准，显式回答是否 <3%）

| 关键量 | 基准 | 交付值 | 相对误差 | <3% ? |
|---|---|---|---|---|
| K_I（默认路线宣称） | 17.7245 MPa√m | 17.7245 | 0.0000%（代数恒等式，非计算） | **不成立**（不计分） |
| K_I（数值路线，最细网格 n=640） | 17.7245 MPa√m | 19.2739 | **8.742%** | **否** |
| K_I（数值路线，独立推到 n≈5120） | 17.7245 MPa√m | 19.3257 | **9.038%** | **否** |
| K_I（数值路线最好点 n=20，偶然） | 17.7245 MPa√m | 17.4568 | 1.510% | 是（但他网格反向恶化，非收敛值） |
| COD 场（n=640） | 解析 4σ/E'·√(a²−x²) | — | 0.210% | 是（证明解算正确） |

**结论（honest）**：交付的"PASS / 0.0000%"不能作为基准验证；其唯一真实数值路线的收敛态误差 ≈8.7–9.0%，**不满足平台 3% 判据**，也不满足台账 5%。报告把该漂移归因于"常数单元离散化方法局限"，经 §③C 反证为**提取式缺陷**（同一 DD 场换后处理即 0.108%）——归因错误，属实质性误判。

## ⑦ 判级（L0–L5）

| 级 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元验证 | 公式与单元级校验 | 满足（但默认路线只到这一层：恒等式） |
| L2 独立功能实现（真代码：输入→计算→输出） | impl_files 非占位 + entry + run_artifacts | **满足**：`crack_sif.py:152-206` 真装配超奇异影响矩阵（:180 自项、:187 numpy 求解）、输入 σ/a/E/ν → 解 → 输出 K 与 COD 场；25 测试过；工件齐 |
| L3 系统集成 | 主求解器/平台真调用 | 不满足：无 integration_callers（仅 capability_evidence 声明表与自身 benchmark） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现且达容差 | 不满足：数值路线收敛态 8.7–9.0% ≫ 3%；宣称的 0.0% 是恒等式；锚点在 `tensor_benchmark_cases` 无登记行 |
| L5 科研工程 | 实验/印刷表外部锚点 | 不满足：仅闭式理论 |

**verdict level = L2**（功能实质：有一条真实的、可复现的 DD-BIE 实现，但精度/集成/基准三项均未达下一级）。交付自称的"Chebyshev 机器精度 + PASS"应按 L4 口径视为**虚标**：其主指标不来自任何计算。

## ⑧ 建议返修（最小改动，供 gate 返修单使用）

1. `extract_sif` 换 Griffith 形状最小二乘（或固定物理窗口 r/a≤0.1 的 √r 拟合），并把 n_fit 从"可调旋钮"改为收敛判据（误差随 n 单调降）。探针实测 n=640 ⇒ 0.108%。
2. 默认 `method` 改为真实数值路线；把恒等式路线更名为 `closed_form_check`，不得计入基准判据。
3. 向 `tensor_benchmark_cases` 登记 `CASE-BEM-CRACK-SIF`（metric=K_I 中心裂纹 a/W→0，tolerance=<3%），否则门禁无裁判行。
4. 把 `capability_evidence.py:1784` 的 `level=REAL` 描述与真实证据对齐；报告 §⑥ 须显式回答 <3%。
5. `reproduce.sh` 改为 `#!/usr/bin/env python3` 可直接执行（或改名 .py）；补 `tensor_reports` 报告；交付需 commit（当前全为未跟踪文件）。

---

**工件清单**
- `probe_crack_sif.py` / `probe_result.json` — 交叉审 + 独立重实现
- `make_figures.py` / `audit_numbers.json` / `fig1_error_vs_mesh.png` / `fig2_sif_bars.png`
- `auditee_snapshot/` + `snapshot_sha256.txt` — 被审源码/工件/报告字节快照
- `pytest_rerun.log`、`reproduce_rerun.log`、`reproduce_bash_invocation.log`
- `land_verdict.py` — 门禁预演 + 落表
---

## ⑨ 落表与裁定（本单产物）

| 项 | 落地 |
|---|---|
| verdict | `tensor_maturity_verdicts` id=**839**（repo=TensorBEM, scope=`m20091 CASE-BEM-CRACK-SIF`, mission_id=20149, level=**2**, verdict_by=board-audit, gate_state=pending） |
| 门禁预演 | `maturity_gate.judge(2, evidence, 'TensorBEM', 20149)` → `gate_level=2, reasons=[]` ⇒ 预期 **confirmed**（L2 消账；落表脚本 `land_verdict.py` 内置预演，预演不过则拒绝写库） |
| 裁判行补登记 | `tensor_benchmark_cases` id=**132**（anchor=CASE-BEM-CRACK-SIF, metric=K_I vs σ√(πa), tolerance `<3%/<5%`, status=**pending**——交付未达标，不置 passed） |
| 负知识入库 | `negative_knowledge_ledger.entry_id=audit:m20149:crack-sif-tautology`（恒等式冒充验证 + 提取式缺陷误归因离散化） |
| 报告 | `tensor_reports` id=**18353**（六段 + 2 图内嵌） |
