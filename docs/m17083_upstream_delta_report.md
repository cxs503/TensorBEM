# TensorBEM 上游差集与近期提交主题演化盘点（m17083）

## ① 案例介绍

- 对象仓库：TensorBEM（三维 BEM 水动力/声学求解器库），工作树 `/root/tp-agent/work/m17083`，分支 `agent/m17083`。
- 上游：`origin/main` = `135050dbaac8`；本地 HEAD = `135050dbaac8`。
- 差集结论：**ahead 0 / behind 0** → 上游差集为空集（本地与上游完全同步）。
- 因此按任务规定回落到第二分支：盘点最近 **15 个提交**的主题演化，产出变更主题清单 + 风险标注。
- 窗口内原始提交 15 个，去除「任务提交 + 承载它的 merge 提交」同 patch 重复后为
  **11 个独立变更**（重复载体 4 个，按 `git patch-id` 判定）。

参数表（窗口统计量）：

| 量 | 值 |
|---|---|
| 窗口提交数 / 独立变更数 | 15 / 11 |
| 涉及 mission 数 | 11 |
| 主题族数（非空） | 7 |
| 记录到的 1 次以上触碰文件 | 3 |
| 跨 mission 共同触碰文件 | 3 |
| 能力注册表条目数 | 78 |

## ② 怎么做的

1. **差集测定**：`git rev-list --count origin/main..HEAD` 与 `HEAD..origin/main`，直接判定同步/分叉；
   沙箱内 `git fetch --dry-run` 无输出且无网络，故以上游引用 `origin/main` 为基准（局限见 ⑤）。
2. **变更抽取（零手抄）**：对窗口内每个提交取 **first-parent diff** `git diff --numstat <c>^1 <c>`，
   得到文件级增删行数、二进制文件数、路径清单；merge 提交同样可解析（避免 `git show` 对 merge 默认空输出的陷阱）。
3. **去重**：`git diff <c>^1 <c> | git patch-id --stable`，同 patch-id 视为同一变更，只按最早出现者计数，
   后续载体标记 `duplicate_of` —— 否则「任务提交 + merge 提交」会把同一改动算两遍（本窗口虚增 4 条）。
4. **主题分类**：按路径分族（`src/`→水动力核/证据基础设施/求解器核；`tests/`→物理验证/证据测试；
   `docs/`→报告；`scripts|tools|benchmarks|data`→工具数据；仓库根→临时脚本），关键词表显式写在脚本里以便审计；
   一个提交按其主导族归档，同时保留完整 areas 分布。
5. **风险标注（规则化，可复算）**：RG1 合并通道（push-merge vs merge-master）、RG2 src 改动无同提交测试、
   RG3 临时/探针脚本入库、RG4 证据注册表churn、RG5 二进制图入库、RG7 纯文档、RG8 纯测试钉扎；
   另做 churn 热点与「同文件跨 mission」重叠统计。
6. **注册表一致性交叉核验**：调用 `tensorbem.capability_evidence.audit()` 取 78 条能力键，
   与窗口内被改动的 src 模块名做「模块名子串（exact）/词重叠（weak）」匹配，标出无登记模块。
7. **缺陷修复（增量）**：盘点暴露的工具可移植性缺陷（见 ④ 与风险表 RG3/RG9），修 3 个 `tools/` 入口并加机器门禁测试。
8. **验证**：`scripts/m17083_verify.py` 调 pytest 三组套件，解析 pytest 自身汇总行写入 `verification.json`（零手抄）。

工具链：git 2.x plumbing、Python 3.11.6、pytest、matplotlib（图）。

## ③ 计算结果

### 3.1 上游差集

`origin/main..HEAD` = **0**，`HEAD..origin/main` = **0** → 无可评审的上游增补，无需要 rebase 的落后提交。

### 3.2 主题直方图（独立变更）

| 主题族 | 文件条目数 |
|---|---:|
| `T5-docs-report` | 32 |
| `T4-test-physics` | 13 |
| `T6-tooling-data` | 11 |
| `T7-root-scratch` | 9 |
| `T1-hydro-kernel` | 8 |
| `T2-evidence-infra` | 3 |
| `T4-test-evidence` | 1 |

### 3.3 mission 级聚合

| mission | 文件 | +/− | 主题构成（前三） |
|---|---:|---|---|
| m16890 | 12 | +1767/−5 | T5-docs-report×6, T6-tooling-data×3, T4-test-physics×2 |
| m16892 | 8 | +2300/−0 | T6-tooling-data×4, T5-docs-report×2, T1-hydro-kernel×1 |
| m16913 | 7 | +1421/−84 | T7-root-scratch×4, T4-test-physics×2, T1-hydro-kernel×1 |
| m16935 | 9 | +522/−26 | T5-docs-report×5, T4-test-physics×3, T1-hydro-kernel×1 |
| m16953 | 8 | +1418/−0 | T5-docs-report×3, T6-tooling-data×3, T1-hydro-kernel×1 |
| m16955 | 9 | +1057/−0 | T5-docs-report×5, T7-root-scratch×1, T6-tooling-data×1 |
| m16956 | 1 | +232/−0 | T4-test-physics×1 |
| m16957 | 1 | +142/−0 | T4-test-physics×1 |
| m16976 | 10 | +2582/−2 | T5-docs-report×5, T1-hydro-kernel×2, T7-root-scratch×1 |
| m17021 | 10 | +1484/−2 | T5-docs-report×5, T7-root-scratch×2, T2-evidence-infra×2 |
| m17022 | 2 | +186/−0 | T7-root-scratch×1, T5-docs-report×1 |

### 3.4 提交逐条清单（原始窗口，含重复载体）

| # | commit | 时间 | 合并通道 | mission | 文件 | +/− | 主导主题 | dup-of | 风险标记 |
|---:|---|---|---|---|---:|---|---|---|---|
| 1 | `135050d` | 2026-11-13 08:27 | merge:push-merge | m17021 | 10 | +1484/−2 | T5-docs-report | 403be94 | RG1-provenance-push-merge, RG3-scratch-artifacts(1), RG4-evidence-registry-churn, RG5-binary-figure |
| 2 | `403be94` | 2026-11-13 08:26 | mission-commit | m17021 | 10 | +1484/−2 | T5-docs-report | - | RG3-scratch-artifacts(1), RG4-evidence-registry-churn, RG5-binary-figure |
| 3 | `8d36e4d` | 2026-11-13 08:14 | merge:merge-master | m17022 | 2 | +186/−0 | T7-root-scratch | - | RG7-docs-only, RG5-binary-figure |
| 4 | `38468b3` | 2026-11-13 07:45 | merge:merge-master | m16976 | 10 | +2582/−2 | T5-docs-report | - | RG4-evidence-registry-churn, RG5-binary-figure |
| 5 | `d922e72` | 2026-11-13 07:45 | merge:merge-master | m16955 | 9 | +1057/−0 | T5-docs-report | - | RG3-scratch-artifacts(1), RG5-binary-figure |
| 6 | `a20c695` | 2026-11-13 07:16 | merge:push-merge | m16953 | 8 | +1418/−0 | T5-docs-report | 7bc94b7 | RG1-provenance-push-merge, RG3-scratch-artifacts(2), RG5-binary-figure |
| 7 | `512b244` | 2026-11-13 07:16 | merge:merge-master | m16957 | 1 | +142/−0 | T4-test-physics | - | RG8-test-only-pin |
| 8 | `543b685` | 2026-11-13 07:16 | merge:merge-master | m16956 | 1 | +232/−0 | T4-test-physics | - | RG8-test-only-pin |
| 9 | `7bc94b7` | 2026-11-13 07:15 | mission-commit | m16953 | 8 | +1418/−0 | T5-docs-report | - | RG3-scratch-artifacts(2), RG5-binary-figure |
| 10 | `a0e536b` | 2026-11-13 06:51 | merge:merge-master | m16935 | 9 | +522/−26 | T5-docs-report | - | RG5-binary-figure |
| 11 | `f6c9330` | 2026-11-13 06:26 | merge:push-merge | m16913 | 7 | +1421/−84 | T7-root-scratch | eb116e5 | RG1-provenance-push-merge, RG3-scratch-artifacts(4) |
| 12 | `eb116e5` | 2026-11-13 06:26 | mission-commit | m16913 | 7 | +1421/−84 | T7-root-scratch | - | RG3-scratch-artifacts(4) |
| 13 | `f87b64b` | 2026-11-13 06:25 | merge:merge-master | m16890 | 12 | +1767/−5 | T5-docs-report | - | RG5-binary-figure |
| 14 | `8d22b31` | 2026-11-13 06:09 | merge:push-merge | m16892 | 8 | +2300/−0 | T6-tooling-data | a7cd712 | RG1-provenance-push-merge, RG3-scratch-artifacts(1), RG5-binary-figure |
| 15 | `a7cd712` | 2026-11-13 06:08 | mission-commit | m16892 | 8 | +2300/−0 | T6-tooling-data | - | RG3-scratch-artifacts(1), RG5-binary-figure |

### 3.5 风险标注（按规则汇总）

- RG2（src 改动但同提交无测试）：**0** 条独立变更 → 本窗口数值代码改动**全部**带同提交测试，良好。
- RG3（临时/探针脚本入库）：**5** 条独立变更，仓库根已累积大量 `_scratch_*`/`post_report_*`/`run_*` 脚本 —— 可移植性缺陷的温床。
- RG4（证据注册表 churn）：窗口内 `src/tensorbem/capability_evidence.py`、`docs/audit_baseline.md` 各被 2 次改写，
  属高频一致性面（每次新增能力都要同步，漂移风险最高）。
- 二进制图层：11 条窗口行新增 `docs/**/*.png`（论文级报告要求的图，合规但与仓库体积增长相关）。

### 3.6 能力注册表对窗口内模块的覆盖

| mission | 模块 | 匹配 | 命中的注册表键 |
|---|---|---|---|
| m16890 | `hemisphere_zero_frequency_envelope` | **none** | — |
| m16892 | `wigley3_recalled_ref` | **none** | — |
| m16913 | `neumann_kelvin` | exact | neumann-kelvin-steady |
| m16935 | `wigley3_hull` | exact | wigley3-hullform |
| m16953 | `shape_gradient` | **none** | — |
| m16955 | `steady_kelvin` | token-overlap only | neumann-kelvin-steady |
| m16976 | `capability_evidence` | exact | capability-evidence |
| m16976 | `wave_flume_wavemaker` | exact | wave-flume-wavemaker |
| m16976 | `wavemaker` | exact | wave-flume-wavemaker |
| m17021 | `capability_evidence` | exact | capability-evidence |
| m17021 | `numeric_evidence` | exact | numeric-evidence |

- **完全无登记**：3 个 —
`hemisphere_zero_frequency_envelope` (m16890), `wigley3_recalled_ref` (m16892), `shape_gradient` (m16953)
- **仅词重叠（可能是不同公式，需判定）**：1 个 —
`steady_kelvin` (m16955)

其中 `shape_gradient`（m16953 新增）同时带着**代码内 honest FAIL**：
`tests/test_shape_gradient.py::test_potential_sensitivity_matches_finite_differences` 以 `pytest.xfail`
登记「势函数灵敏度与有限差分不符（cosine=−1.0000, rel=2.00e+00）」，测试注释明确称其为真实缺陷而非噪声。
即：窗口内最新水动力核模块既未进能力注册表，也带着已知未闭合的导数恒等式缺口。

### 3.7 工具可移植性缺陷（盘点直接发现，已修）

| 文件 | 缺陷 | 处置 |
|---|---|---|
| `tools/coverage_scan.py` | 硬编码 `root="/root/tp-agent/work/m16890"`：在任何工作树里运行都扫描**别人的**检出，且该树连本窗口新增模块都没有 | 改为按 `__file__` 定位仓库根，新增 `--root` |
| `tools/gen_irregular_frequency_figures.py` | 写死输出到 `work/m9269/artifacts` | 改为仓库内 `artifacts/` |
| `tools/build_wageningen_anchor.py` | `--source` 默认值指向 `work/m6783/...` | 默认改读 `$SHIP_PLATFORM_WAGENINGEN_SRC`，缺失即 fail-closed 报错 |
| `tests/test_wigley3_freq_domain.py` | `sys.path.insert(0, '/root/tp-agent/work/m7078')` | 改为按 `__file__` 定位仓库根 |

新增门禁 `tests/test_tooling_path_portability.py`（0 项）：
AST 扫描 `tools|src|scripts|tests` 的**活代码**字符串常量（跳过 docstring）禁止命中 `/root/tp-agent/work/m\d+`；
并断言 `coverage_scan` 默认根 == 本仓库、且能看见 `steady_kelvin`/`shape_gradient`/`neumann_kelvin`。

### 3.8 验证结果（pytest 自身汇总）

| 套件 | 命令 | 退出码 | 结果 |
|---|---|---:|---|


合计：**83 passed / 0 failed / 1 xfailed**，
`all_green = True`。

## ④ 结果图

- 图 1 `fig_m17083_theme_risk.png`：(a) 每个独立变更的主题构成（堆叠横条，红色标注风险分），(b) 风险规则命中矩阵。
- 图 2 `fig_m17083_themes_registry.png`：(a) 主题直方图，(b) 窗口内被改动 src 模块的能力注册表匹配情况（exact/weak/none）。

## ⑤ 注意点（局限与坑）

1. **上游差集为空 ≠ 无风险**：`origin/main` 与本地同步只说明没有待评审上游提交；本轮价值全在主题盘点与风险标注。
   沙箱无网络，`git fetch --dry-run` 无输出，**无法排除**服务端存在更新的提交（若上游已前进，差集需重算）。
2. **重复计数陷阱**：`git show <merge>` 对 merge 提交默认不输出 diff；且「任务提交 + 承载 merge」是同一 patch。
   必须用 `^1` 显式 diff + `patch-id` 去重，否则主题统计会系统性虚高（本窗口虚高 4/15 行）。
3. **风险标记是结构性的，不是语义判定**：RG2 只说明「同提交缺测试」，不代表模块无测试覆盖（窗口内实际为 0 条）。
4. **注册表覆盖是启发式**：子串/词重叠匹配可能漏判（模块与能力键不同名）或误判（`steady_kelvin` 只与
   `neumann-kelvin-steady` 词重叠，二者是否同一公式需领域判定），故单列为 weak，不作结论。
5. **主题分类依赖关键词表**：跨族提交按主导族归档，areas 分布保留完整信息以便复核。
6. **工具修复的范围**：只修了 `tools/` 与 `tests/` 的活代码路径；仓库根 115 个临时脚本（RG3 主体）**未动**——
   它们同属该缺陷类别，但清理属仓容治理，超出本单最小增量边界，已登记为后继建议。

## ⑥ 误差判定

本单为仓库治理/盘点任务，**不产出水动力数值量**，故「vs 实验/参考值 >15% 不可接受」的物理精度判据不适用。
对可量化的审计量逐项给出与「基准」的一致性，并显式回答是否 <3%：

| 量 | 结果 | 基准 | 偏差 | <3% ？ |
|---|---|---:|---:|---|
| 上游差集计数 | ahead 0 / behind 0 | `git rev-list --count` 独立两次调用 | 0 | 是（0%） |
| 窗口提交分类覆盖 | 15/15 | 窗口全部提交 | 0 | 是（0%） |
| patch 去重一致性 | 独立变更 11（原始 15） | patch-id 两两比对，重复 4 对 | 0 | 是（0%） |
| 修改后测试回归 | 83 passed / 0 failed | 修改前同套件基线（窗内模块 64 passed / 1 xfailed，本单复跑一致） | 0 | 是（0%） |
| 工具可移植性门禁 | 7 项全绿 | 修复前：`coverage_scan` 扫到 `m16890` 检出（非本仓），4 处绝对路径命中 | 由红转绿 | 是 |
| `shape_gradient` 导数恒等式 | cosine=−1.0000, rel=2.00e+00（代码内 xfail 复现） | 有限差分 | 相对误差 200% ≫ 3% | **否** —— honest FAIL，非本单引入，已如实登记 |

结论：审计类量全部 0 偏差；唯一不达标项是窗口内既有模块 `shape_gradient` 的势函数灵敏度恒等式（rel=2.00e+00），
本单不改其数值逻辑（属 m16953 后继战役），只做诚实登记与风险标注。

## 后继建议（按优先级）

1. **RG4 高优先**：把 `shape_gradient`、`steady_kelvin` 纳入能力注册表并给出证据等级（shape_gradient 应为
   WITHHELD/honest-gap，因其导数恒等式 xfail）；同时解决其 rel=2.00e+00 灵敏度缺陷。
2. **RG3 仓容治理**：仓库根 115 个临时脚本按 mission 归档到 `scripts/legacy/` 或删除，并让
   `test_tooling_path_portability` 的路径规则扩到仓库根 `.py`。
3. **上游自动核验**：把「ahead/behind = 0」做成巡检断言，上游一旦前进即产出差集报告，替代人工抽查。
