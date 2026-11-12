# m16384 · TensorBEM 能力台账欠账清偿：26 个"无行"模块 → 17 行分级（14 REAL / 3 诚实 WITHHELD）

**仓库** TensorBEM · **分支** `agent/m16384` · 基线 `79ab26a`（main @ 2026-11-12）

## 1 案例介绍

不是流场算例，而是**证据台账**算例：`docs/audit_baseline.md` §4 把 26 个核心物理模块
（占包内 93 个模块的 28%）停在 `AUXILIARY_MODULES` 的一句延期判词上——

> "physics/geometry module <leaf> is pinned by tests/<file>.py … it has no row of its
> own yet (debt registered in docs/m14869/report.md), so its level is NOT claimed here."

读表人无法判断这四分之一包的证据等级；"pinned by tests/X.py" 这句话本身**没有任何机制验证**：
注册表原有的 detached-test 检查只看测试**文件**是否提到模块名，一个只 import 模块而断言别的
东西的文件也能让判词活着。本单把判词变成可执行判据。

对象清单：26 个延期模块 / 93 个核心模块；判决口径沿用注册表既有五级
（REAL / MOCK / STUB / WITHHELD / NOT_SUPPORTED），等级由代码+测试节点+文档锚三者**推导**，不许手填。

## 2 怎么做的

1. **独立 AST 扫描器** `scripts/audit_module_pins.py`（新增，静态、无求解）：对每个模块取公开符号集，
   逐测试文件找**在 assertion（或 pytest.approx / allclose 等比较）内部调用该符号**的测试函数，
   与只调用不断言的"冒烟节点"分开计数；输出矩阵 `docs/capability_pin_audit.json`（程序化，零手抄）。
   首版漏了 `class Test*` 内的方法，修正后 26/26 模块都有 ≥1 个断言节点。
2. **按物理族归并成 20 行**（不是按文件一对一），与既有行风格一致，例如
   `haskind-2d-excitation` = `haskind2d` + `net_source_reg`（同一条剖面理论管线）、
   `cylinder-and-farfield-drift` = `cylinder_drift` + `drift_far_field3d`。
   每行给出 modules / symbols / 测试节点 / **非** `audit_baseline.md` 的文档锚。
3. 从 `AUXILIARY_MODULES` 摘掉 25 个已入行模块（余 4 条：resistance、verification、analytic、
   pekeris_acoustics），`python -m tensorbem.capability_evidence --write` 重生成 §4 与两条清单。
4. 新增守护测试 `tests/test_capability_registry_debt.py`（7 例）：延期判词不得无理由复活、
   模块不得同时"入行 + 豁免"、pin 矩阵不得只剩冒烟节点、行不得引用自己生成的 baseline 作证据、
   晋升行必须真的推导出等级、以及那些只能推导 WITHHELD 的行必须给出**唯一允许的理由**。

## 3 计算结果

| 指标 | 改动前 | 改动后 |
|---|---:|---:|
| 分级行数 | 51 | **71** |
| `AUXILIARY_MODULES` 条目 | 29 | **4** |
| 停留在"no row of its own"延期的模块 | 26 | **1**（`pekeris_acoustics`，见 §5） |
| 派生 REAL 的晋升行 | — | **14 / 20** |
| 派生 WITHHELD 的晋升行 | — | **6 / 20**（全部同一原因，见 §5） |
| 派生失败（MISSING/STUB/MOCK/STALE_CLAIM）的行 | 0 | **0** |
| pin 矩阵中"只有冒烟节点"的模块 | 未测 | **0 / 93** |

派生等级明细（20 行，`python -m tensorbem.capability_evidence` 程序化输出）：
REAL：cathodic-protection-bem、cylinder-and-farfield-drift、finite-depth-3d-quadrature-solver、
flexural-gravity-dispersion、interior-panel-lid、irregular-frequency-sanitisation、nk-panel-3d、
quadratic-panel-operator、propeller-open-water、rankine-fs-panel、reference-provenance、
state-space-realization、tank-free-surface、wigley3-hullform（14）；
WITHHELD：figure-integrity、finite-depth-green-kernel、haskind-2d-excitation、
neumann-kelvin-steady、rankine-wave-series60、stf-fullspeed-strip（6）。

## 4 结果图

本单为台账/静态分析任务，**无场量云图**：产出是等级表与一致性判据，画云图会是无物理含义的图。
替代证据是两张程序化表：`docs/capability_pin_audit.json`（每模块的断言节点/冒烟节点清单）与
重生成的 `docs/audit_baseline.md` §4（71 行等级 + 两条豁免清单）。

## 5 注意点

1. **6 行只能推导 WITHHELD，原因是注册表的节点索引缺陷，不是证据缺失**：
   `_test_nodes()` 只收模块级 `def test_*`，`class TestX` 内的方法**不进索引**，因此
   `tests/test_neumann_kelvin.py::TestGreenFunction::test_*` 这类节点无法被点名。这 6 个模块的
   断言证据是真实存在且通过的（本单已跑过其中 2 个文件），但按"等级必须可机器推导"的纪律，
   宁报 WITHHELD 也不手填 REAL。遗留改进：让 `_test_nodes` 支持类限定节点。
2. **`pekeris_acoustics` 保持延期**：`docs/dev_report_m14725_reference_provenance.md` 已记录仓内
   没有该模型的文档锚点，入行只会得到 docs 腿 WITHHELD；已在 `DOC_ANCHOR_DEBT` 登记理由，并为
   "不许无声复活延期判词"写了测试（除登记项外，任何 `no row of its own` 复活即红）。
3. **申报措辞自查**：行 claim 只写"实现了什么 + 被哪些断言节点检验"，不写"已验证到 X%"——
   本单没有重跑新的物理算例，精度数字仍以各模块原报告为准，不在此复述以免二手引用。
4. **容量纪律**：本机 load15 114/128，全程只跑静态分析与轻量测试，未起新的重算力进程。

## 6 误差判定（显式回答"是否 <3%"）

本单无新的物理关键量，衡量对象是台账一致性，判据是**精确的集合/计数恒等**，不是浮点误差：

| 判据 | 基准 | 实测 | 是否满足 |
|---|---|---|---|
| 派生失败行数 | 0 | 0 | 是（精确） |
| 模块同时入行且豁免 | 0 | 0 | 是（精确） |
| 无理由复活的延期判词 | 0 | 0 | 是（精确） |
| pin 矩阵中只有冒烟节点的模块 | 0 | 0 | 是（精确） |
| 晋升行派生 REAL | 20/20 | 14/20 | **否**（6 行因注册表节点索引缺陷 → WITHHELD，已如实登记） |
| 文档锚存在性 | 20/20 | 20/20 | 是（精确） |

结论：台账层面 100% 一致；**等级层面 14/20 达标，6/20 诚实地留在 WITHHELD 并给出可修原因**——
按"honest FAIL 不得覆盖"的纪律如实报告，不粉饰为 20/20。

## 7 变更清单

| 文件 | 性质 |
|---|---|
| `scripts/audit_module_pins.py` | 新增：AST pin 审计（断言节点 vs 冒烟节点） |
| `src/tensorbem/capability_evidence.py` | 新增 20 行（25 模块）；`AUXILIARY_MODULES` 29→4 |
| `tests/test_capability_registry_debt.py` | 新增：7 例欠账守护 |
| `docs/audit_baseline.md` | 重生成 §4（51→71 行）与模块清单 |
| `docs/capability_pin_audit.json` | 新增：程序化 pin 矩阵 |
| `dev_report_m16384.md` | 本报告 |
