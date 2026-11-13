# M17398 · 交差审计：m17336 CASE-BEM-MARUO-DRIFT（TensorBEM）

**审计对象** m17336（succeeded，branch `agent/m17336` @ `4ecdd3b`）
**锚点** `CASE-BEM-MARUO-DRIFT` · **判级** **L3** · **判级人** `board-audit`
**本审计单** m17398 · 证据快照 `docs/board_audit/m17398/`（`refs/` = 被审 commit 的只读副本，`artifacts/` = 审计自产工件）

---

## ① 交差对象与判据（唯一裁判=基准台账）

| 项 | 实测值 | 来源 |
|---|---|---|
| 判据来源 | mission 17336 goal 'CASE-BEM-MARUO-DRIFT … 容差' (mission text only; no tensor_benchmark_cases row for this anchor) | mission 文本 |
| `tensor_benchmark_cases` 该锚点登记行数 | **0** | 本审计脚本直连 DB 查询 |
| 生效判据（交付自定） | per-cell relative error < 5% at the declared production mesh 'fine' | study `parameters.production_resolution='fine'` |
| 生产网格失败格数 | **1 / 4** | study.json `cells` |
| 生产网格最差 | ka=2.0, 800 panels, 5.331% | study.json |
| 最密网格最差 | 5.037%（1200 panels，仍 >5%） | study.json |
| 生态 <3% 目标 | not met anywhere in the production row: best 3.032% | study.json 生产行 |
| study 自报 `all_passed` | `False`（诚实 FAIL，CLI 退出码 1） | study.json |
| 邻近已登记锚点（同仓 drift 类） | #43 CASE-BEM-CYL-DRIFT-ASYMP (passed), #53 CASE-BEM-QTF-DRIFT (failed) | DB |

**结论 1（阻断 L4，承 m17276/audit m17327 未变的阻断项）**：本锚点在基准台账中**没有任何登记行**，
即「基准判据是唯一裁判」这一点在本单上仍无机器裁判——容差只活在 mission 文本里。
**结论 2（阻断 L4，同样未变）**：交付自定的 `<5%` 在生产网格上 **不成立**（ka=2.0，5.331%），
且加密到 1200 面板仍 5.037%，说明这是**模型/离散系统性偏差**而非网格偶然。

## ② 怎么核的（方法/流程/工具链）

1. **先读代码**：`src/tensorbem/benchmarks/maruo_drift_far_field_case.py`（闭式 ref `:153-212`、
   被测 BEM 路线 `bem_case` `:214-252`）、生产核 `src/tensorbem/drift_far_field3d.py`（`bem_kochin`
   只消费 `solve_diffraction_3d` 的散射势与 `-∂φ_I/∂n`，无解析注入）、测试 `tests/test_maruo_drift_far_field_case.py`。
2. **独立性检验（防「自证自」）**：审计方**不 import TensorBEM**，从第一性原理重建 MacCamy–Fuchs 远场散射幅值
   （入射展开 → 物面条件 → 出射 Hankel 辐射条件 → `c_m`），再走两条互不相同的角向积分（Brute-force 梯形求积 / 模态恒等式），
   得到独立的 `F_x`，与交付 ref 对比；并在 `ka→0` 比对长波律 `5π²/16`。
   脚本：`docs/board_audit/m17398/independent_ref_check.py`。**注**：v1 版本因负阶次折叠错误整体偏 20%，
   由「自身两条路线不自洽」自检出后修正（详见 ⑤-F6），v2 结果见下表。
3. **本地复现（不是重做全算例）**：在 `/root/tp-agent/work/m17336`（被审 commit 检出）重跑两格——
   `ka=2.0 coarse`（命中生产网格判据之外的对照格）与 `ka=1.0 coarse`，与交付 study 逐位比对。
4. **集群复现（完整 4×4 网格）**：递交 `run_cluster` 作业 **cj1530**（stage_dir=被审工作树，单命令跑 module 入口）；
   报告时状态：**cluster replay: NOT RUN at report time (job cj1530 still queued)**。
5. **测试**：`PYTHONPATH=src pytest tests/test_maruo_drift_far_field_case.py -q` → **13 passed in 2.48s**。
6. **可执行性探测**：`--resolutions` 只给非生产档位时入口崩（见 ⑤-F3），其余参数组合可跑。

## ③ 三件套核验（工件 / 数值 / 复现命令）

### ③-1 run_artifacts 是否真存在

| 工件 | 位置 | 状态 |
|---|---|---|
| 全量数字（16 格 × 11 字段） | `docs/case_bem_maruo_drift/study.json`（tracked，sha256 `89f382eef3cc79d7…`） | 真实、非空 |
| 报告三图 | `docs/case_bem_maruo_drift/fig_*.png`（3 个，tracked） | 真实 |
| `artifacts/maruo_drift_far_field_case/*.json` | `.gitignore:10` 忽略 `artifacts/` | 工作树有、**入库无**（见 ⑤-F7：m17276 的 verdict 曾把它们当工件指针，gate 判「工件缺失」） |

### ③-2 benchmark：ref vs got 逐格（生产网格加粗）

| ka | resolution | panels | got F_BEM [N] | ref F_closed [N] | rel err | <5% | <3% |
|---|---|---|---|---|---|---|---|
| 0.2 | coarse | 312 | 222.7147 | 240.5907 | 7.430% | FAIL | FAIL |
| 0.2 | medium | 544 | 229.4834 | 240.5907 | 4.617% | PASS | FAIL |
| 0.2 | fine **(prod)** | 800 | 233.2949 | 240.5907 | 3.032% | PASS | FAIL |
| 0.2 | finer | 1200 | 233.9049 | 240.5907 | 2.779% | PASS | PASS |
| 0.5 | coarse | 312 | 2580.7903 | 2875.6570 | 10.254% | FAIL | FAIL |
| 0.5 | medium | 544 | 2701.6146 | 2875.6570 | 6.052% | FAIL | FAIL |
| 0.5 | fine **(prod)** | 800 | 2779.9799 | 2875.6570 | 3.327% | PASS | FAIL |
| 0.5 | finer | 1200 | 2788.3032 | 2875.6570 | 3.038% | PASS | FAIL |
| 1 | coarse | 312 | 5919.1530 | 6685.9935 | 11.469% | FAIL | FAIL |
| 1 | medium | 544 | 6239.4142 | 6685.9935 | 6.679% | FAIL | FAIL |
| 1 | fine **(prod)** | 800 | 6480.7722 | 6685.9935 | 3.069% | PASS | FAIL |
| 1 | finer | 1200 | 6485.0329 | 6685.9935 | 3.006% | PASS | FAIL |
| 2 | coarse | 312 | 4884.7166 | 6306.3981 | 22.543% | FAIL | FAIL |
| 2 | medium | 544 | 5509.1819 | 6306.3981 | 12.641% | FAIL | FAIL |
| 2 | fine **(prod)** | 800 | 5970.2151 | 6306.3981 | 5.331% | FAIL | FAIL |
| 2 | finer | 1200 | 5988.7699 | 6306.3981 | 5.037% | FAIL | FAIL |

### ③-3 ref 的独立性（审计方重建）

| ka | auditor ref (first principles) [N] | delivered ref [N] | rel diff | auditor modal vs quadrature |
|---|---|---|---|---|
| 0.2 | 240.590682 | 240.5907 | 7.34e-08 | 2.95e-13 |
| 0.5 | 2875.656993 | 2875.6570 | 2.34e-09 | 4.27e-13 |
| 1 | 6685.993519 | 6685.9935 | 2.88e-09 | 7.63e-13 |
| 2 | 6306.398054 | 6306.3981 | 7.30e-09 | 2.23e-12 |

长波锚点（`ka→0`）：

| ka | (5 pi^2/16) | auditor recovery | rel err |
|---|---|---|---|
| 0.01 | 3.084251375 | 3.084203600 | 1.549e-05 |
| 0.02 | 3.084251375 | 3.083891060 | 1.168e-04 |
| 0.05 | 3.084251375 | 3.080622456 | 1.177e-03 |

→ 交付 ref 被独立重建复现到 **7.3e-08**（受交付值 4 位小数存储限制；同精度全量值比对为 5e-16），
且长波律恢复到 1.2e-03 —— **参考值不是自证，可信**。

### ③-4 repro_cmd 可执行性（本地逐位复现）

| cell | delivered got [N] | auditor recomputed [N] | abs diff | rel err (delivered/recomputed) | G3 |
|---|---|---|---|---|---|
| ka=1 coarse (312 panels) | 5919.1530 | 5919.1530 | 0.00e+00 | 11.469% / 11.469% | G3=False |
| ka=2 coarse (312 panels) | 4884.7166 | 4884.7166 | 0.00e+00 | 22.543% / 22.543% | G3=False |

## ④ 图

| 图 | 说明 |
|---|---|
| `artifacts/fig_m17398_error_vs_panels.png` | 4 个 ka × 4 档网格的误差阶梯，叠 5%（本算例判据）与 3%（生态目标）两条线；ka=2 在 800/1200 面板均落线之上 |
| `artifacts/fig_m17398_ref_validation.png` | 左：生产网格 got vs ref 直方对比；右：审计方独立第一性原理重建 vs 交付 ref 的相对差（~1e-8，量化噪声级） |

## ⑤ 判级与缺陷清单（按严重度）

### 逐级判定

| 级 | 判据 | 本单事实 | 判 |
|---|---|---|---|
| L0 | 文档成果 | 有报告+文档 | 过（但非本单实质） |
| L1 | 数学/单元验证 | 闭式与仓内独立求积 2.36e-16；模态截断收敛；长波律 1.2e-4 | 过 |
| L2 | 独立功能实现（输入→计算→输出） | `bem_case`：网格→绕射解→Kochin→约化 Maruo 核→F_x，真实数值，无硬编码 | 过 |
| L3 | 系统集成 | 消费 `solve_diffraction_3d`/`bem_kochin`/`maruo_far_field_drift`/`vertical_cylinder_hull_mesh`（生态主侧已有驱动 `drift_far_field_3d_study.py:100,178` 走同链）；分支上登记 `capability_evidence` 行 `maruo-far-field-drift-case` | **达** |
| L4 | 基准验证（完整算例+误差分析+可复现） | 完整算例 ✅、误差分析 ✅、可复现 ✅，**但** ①无机器裁判行 ②生产网格判据 ka=2 未过 | **拒** |
| L5 | 科研工程成果 | 需 L4 + 独立外部锚整体达标 | 不适用 |

**判级 = L3**（与 m17276 判级一致；m17336 是 re-land + 能力登记，未触及两项 L4 阻断）。

### 缺陷（严重度排序）

- **F1 blocking-for-L4**：`tensor_benchmark_cases` 无 `CASE-BEM-MARUO-DRIFT` 行（实测 0 行）。
  修法：把该算例登记进台账（case_name/metric/tolerance/ref_source），tolerance 按交付自定 5% 或生态 3% 明示。
- **F2 blocking-for-L4**：生产网格判据 KA=2.0 失败 5.331%（1200 面板 5.037%）。
  修法：或加密（远未收敛，性价比低）、或换高阶元/奇异性处理、或把 ka 适用范围显式限为 ≤1 并申请判据带。
- **F3 minor（可用性）**：`--resolutions` 不含生产档位时 `ValueError: max() arg is an empty sequence`
  （`maruo_drift_far_field_case.py:396-397` 对空 `production_cells` 取 max）。一行守卫即可（无生产档位时报可读错误或跳过 G3）。
- **F4 minor（报告溯源）**：`docs/case_bem_maruo_drift_report.md:151-153` 仍保留 m17327 已指出的三处不可溯源数字
  （800 面板网格写成「832 三角形」、study 中不存在的「416 三角形」档、「fine 的 3.01%」实为 finer 的 3.006%）。
  正文表格（④）全对，仅 ⑤ 散文有瑕疵 → 不改判等级，但建议订正。
- **F5 minor（生态摩擦）**：m17336 重生成的 `docs/audit_baseline.md` 把本算例的图助手
  `tensorbem.benchmarks.maruo_drift_far_field_case_figures` 列进「Unreferenced and unexplained（fails the gate）」——
  本单自己引入的 gate 失败项。
- **F6 审计方自身更正（诚实记录）**：独立重建 v1 用负阶次折叠捷径，全局偏 20%（长波系数 π²/4 vs 5π²/16），
  靠「模态 vs 求积不自洽」自检出，改为全整数阶直接求和后收敛到 1e-12。**这不是交付缺陷**，是本审计的方法学教训。
- **F7 记账（可复用教训）**：`artifacts/` 被 `.gitignore` 忽略，verdict 的 `run_artifacts` 若指向它，gate 必判「工件缺失」
  （m17276 即如此被 reopen）。本单改为指向 tracked 的 `docs/case_bem_maruo_drift/` 与审计自产 `artifacts/`（后者在审计工作树内可解析）；本审计自产工件本身也被 `.gitignore` 覆盖，故用 `git add -f` 强制入库，
  保证 verdict 指针在克隆后的仓里仍可解析。

## ⑥ 误差判定（逐项显式回答）

| 关键量 | 基准 | 误差 | <3%? | <5%? | <15%（生态不可接受线）? |
|---|---|---|---|---|---|
| F_x, ka=0.2 生产网格 | 闭式（审计方独立复现） | 3.032% | 否 | 是 | 是 |
| F_x, ka=0.5 生产网格 | 同上 | 3.327% | 否 | 是 | 是 |
| F_x, ka=1.0 生产网格 | 同上 | 3.069% | 否 | 是 | 是 |
| F_x, ka=2.0 生产网格 | 同上 | 5.331% | 否 | **否** | 是 |
| 长波系数（ka=0.02） | 5π²/16 | 1.17e-04 | 是 | 是 | 是 |
| 闭式 ref vs 仓内独立求积 | 两条独立路线 | 2.36e-16 | 是 | 是 | 是 |

**显式回答：「是否 <3%」= 否**（生产网格最好 3.032%，无一格 <3%；
甚至最密网格最好也只有 2.779%）。
**「是否 <5%」= 部分**：ka ∈ {0.2,0.5,1.0} 成立，ka=2.0 不成立，故整体判 FAIL（交付方亦如此自报，未粉饰）。

### 未完成 / 未运行（如实声明）

- 集群完整 4×4 网格复现：**cluster replay: NOT RUN at report time (job cj1530 still queued)**
- 报告中的 `docs/case_bem_maruo_drift_report.md` 正文表格未逐格重算（只重算 2 格 + 全表内部一致性检查）。
- 无外网：未逐页核对 Maruo 1957/1960 原始印刷表，ref 为机器求值的解析闭式 + 长波律双锚（交付方亦这样申报）。

## 附：判决

**L3 成立，L4 拒绝**；`gate_state` 由 maturity_gate 自动裁定（本审计已用 gate 的 `judge()` 预演：返回 `(3, [])`）。
两项 L4 阻断项（台账无裁判行 / ka=2 判据未过）**自 m17276 起未变**，m17336 只做了 re-land 与能力登记，
因此本单交差结论是：**消账到 L3，L4 需另开修复单**（登记台账行 + 修 ka=2 精度或收窄适用范围）。
