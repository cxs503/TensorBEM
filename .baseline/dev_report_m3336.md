# TensorBEM：审计基线的能力-证据矩阵从手写改为机器推导（含 4 条失效声明的更正）

**任务**：m3336（platform 席自找活） · **分支**：`agent/m3336` · **日期**：2026-11-04
**仓**：TensorBEM · **工件**：`src/tensorbem/capability_evidence.py`、`tests/test_capability_evidence.py`、
`tests/test_benchmark_cli_contract.py`、`docs/audit_baseline.md`§4、`docs/m3336/outputs/{capability_matrix.json,capability_matrix.png}`

EDIT_WHITELIST: docs/, src/tensorbem/capability_evidence.py, tests/, docs/m3336/outputs/, dev_report_m3336.md

## 1. 欠账是什么

`docs/audit_baseline.md`§4 的能力表是本仓对外唯一的「实现了什么、证据到哪一级」清单。它由 m308（2026-10-29）**手写**，此后
从未重新推导。到 2026-11-04，它仍把 **有限水深核（2-D/3-D）、有航速兴波阻力（Wigley）、Kutta 条件升力** 判为
`NOT_SUPPORTED`，而 `tensorbem/finite_depth.py`、`finite_depth3d.py`、`wigley.py`、`lifting.py` 早已随各自测试套件合入。
也就是说：**四条已发布声明与代码事实相反**，而套件里没有任何断言能发现它——这是典型的「已合并功能对证据的复核欠账」，
且复核对象是文档而非物理量，代价低、判据硬。

附带发现（同一次扫描得到）：包内 9 个模块**没有任何测试文件引用**（`finite_depth_kernel`、`hulme_diffraction_verification`、
`hydrostatic_verification`、`kkr_causality`、`lifting_foil`、`lifting_solver`、`symmetry_reduction`、`helmholtz`、`mutation_audit`），
其中前七个正是报告里让人「复现这张表」的 benchmark 驱动——入口坏掉不会被任何测试发现。

## 2. 怎么做的

1. **把每一行变成机器可判的声明**：新增 `src/tensorbem/capability_evidence.py`，登记 28 条能力，每条形如
   `modules / symbols / tests(file::node) / docs(path, anchor)`。
2. **证据等级由程序推导，不由人填**：模块与符号必须可导入解析；符号体不得是 `pass`/`...`/`raise NotImplementedError`（→STUB）；
   无种子随机或常量返回→MOCK（**有种子**的 `np.random.default_rng(seed=42)` 不算 MOCK，conditioning 的幂迭代起点即属此类）；
   命名测试节点必须在 `tests/` 文件中真实存在、且该文件确实提到被声明的模块或符号（防「随便配一个测试」→detached→WITHHELD）；
   文档锚点必须存在。**声明为 `NOT_SUPPORTED` 但代码+测试都在场 → `STALE_CLAIM`，闸门直接红**。
3. **重生成 §4**：`python -m tensorbem.capability_evidence --write` 把表格写进 `docs/audit_baseline.md` 的 marker 块内，
   §4 附「与 2026-10-29 基线相比的更正日志」与「未测模块清单」。
4. **漂移闸门**：`tests/test_capability_evidence.py` 断言（a）无任何失败等级；（b）发布块 == 生成块（手改文档即红）；
   （c）四条被更正的能力必须 `REAL` 且 `published_2026_10_29 == NOT_SUPPORTED`（回归钉桩）；（d）未测模块必须各有理由。
5. **负控（证明闸门会失败）**：临时造一个微型包：正常模块（应 REAL）、声明 NOT_SUPPORTED 但有码有测（应 STALE_CLAIM）、
   `raise NotImplementedError`（应 STUB）、无种子 `np.random`（应 MOCK）、常量返回（应 MOCK）、缺模块（应 MISSING）、
   无测试/测试不相关/无文档锚点（均应 WITHHELD）、有种子随机（**不得**判 MOCK）。全部按预期分级才算通过。
6. **入口契约测试**：`tests/test_benchmark_cli_contract.py` 对 6 个 argparse 驱动跑 `python -m ... --help`（必须 0 退出且打印 usage），
   对无 argparse 的 `lifting_foil` 在进程内验证 `analytic_reference` 两条解析路径（压力积分 vs 环量）互洽 <1%。

## 3. 计算结果（程序化注入）

| 指标 | 2026-10-29 基线（手写） | 本单（推导） |
|---|---:|---:|
| 被分级的能力行 | 15 | **28** |
| 失效的 `NOT_SUPPORTED` 声明 | 4 | **0** |
| 无解释的未测模块 | 9 | **0**（另有 2 个带理由） |
| 派生等级分布 | — | REAL 28、WITHHELD 0、STUB 0、MOCK 0、MISSING 0、STALE_CLAIM 0 |

被更正的 4 条（现均 `REAL`，各带 ≥2 个真实测试节点）：2-D 有限水深核（`finite_depth`+`finite_depth_solver`）、
3-D 有限水深核（`finite_depth3d`，含 `bessel_k0`/模式正交）、Wigley 兴波阻力（`wigley`+`wigley_verification`）、
Kutta 升力（`lifting`+`lifting_foil`）。

**新增测试**：`tests/test_capability_evidence.py` 11 条 + `tests/test_benchmark_cli_contract.py` 14 条 = **25 条，25 passed in 4.71 s**
（含 6 次子进程 `--help` 探针，实测 2.4 s 内跑完 14 条契约测试）。

![capability matrix](docs/m3336/outputs/capability_matrix.png)

图：左=发布声明与代码事实的差距（红=手写基线，蓝=本单推导：行数 15→28、失效声明 4→0、无解释未测模块 9→0）；
右=本单推导出的证据等级分布（28 行全 REAL，因为每个非 REAL 都被当作待修项处理掉了）。

## 4. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 本单实测 | 误差 | <3%? |
|---|---|---|---|---|
| 失效 `NOT_SUPPORTED` 声明数 | 代码事实（模块可导入+测试在场） | 0 | 0（判据为布尔，不是数值） | ✅ 是（精确 0） |
| 无解释未测模块数 | 理由表 + 引用索引 | 0 | 0 | ✅ 是（精确 0） |
| 发布块 vs 生成块 | `document_drift()` | `None`（无 diff） | 0 行差异 | ✅ 是 |
| 分级器的判别力 | 9 类合成负控 | 9/9 按预期 | — | ✅ 是 |

**诚实声明**：本单是**静态证据审计**，不产生新的物理基准数值，故「物理量 vs 解析解/文献误差」一项**不适用**（NOT_MEASURED）；
四条被更正能力的物理精度仍由它们各自既有测试与历史报告钉住，本单未重新跑其全量收敛研究。

## 5. 注意点 / 局限

- 分级器是**静态**的：无法发现「测试在场但不判数」的软断言，也无法发现导入链深处的伪造（子程序级 MOCK）。真正的判别力仍要靠变异测试（m2582/m3328 路线）。
- 「未测模块」的引用索引对**动态导入**会假阳性；本单以「模块必须被引用或带书面理由」的 fail-closed 方式处理，新增未测模块会立刻变红。
- 尝试把 12 个测试文件放进一次 pytest 调用取运行时间证据时**超过 shell 120 s 上限被杀**，故运行证据只覆盖新增的 25 条；`docs/m3336/outputs/capability_matrix.json` 是完整机器可读矩阵。
- 文档 §4 现在是生成物：**不要手改**，改登记表后 `--write`。
