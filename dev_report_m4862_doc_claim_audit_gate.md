# m4862 — doc-claim 守卫账本修复 + 工件新鲜度门（重开 #309 · 第 2 轮）

- 仓库：TensorBEM；分支 `agent/m4862`，从 `origin/main = 0c23cc1`（Merge #327）切出
- 前情：#309 由 owner 于 2026-11-05 裁定关闭。核对结论见第 5 节：m4655 的**功能**已由 #324
  （`m4794`，commit `b31acd6`）以更强的形式落地，但**同一类缺陷在 main 上重新出现**
  （棘轮红 + 已发布工件陈旧），本轮把它修干净，并把「账本不得静默漂移」变成可执行判据。
- 证据：`artifacts/m4862_audit_gate_evidence.json`（由 `tools/m4862_audit_gate_evidence.py`
  程序化生成；本报告所有数字由脚本从该文件注入，无手抄）

## 1. 背景：这条门到底在守什么

`tensorbem.benchmarks.doc_claim_audit` 是 TensorBEM 的**元门禁**：它不问「某个数对不对」，
而问「哪些文档里的数字没有任何测试在读」。每个 `docs/*.md` 被打成三档

| 档 | 含义 | 出事时的后果 |
|---|---|---|
| `PINNED` | `tests/` 里有人读这个文件本体 | 文档改一个数字，测试就红 |
| `ARTIFACT` | 文档声明了生成器且该生成器被测试跑到 | 数字可复现，但文档文本改数字不被发现 |
| `UNGUARDED` | 两者皆无 | 文档可以静静烂掉 |

`UNGUARDED` 集合被棘轮（ratchet）钉死在 `docs/doc_claim_audit/baseline.json` 上：
**只许缩小，不许变大**（`audit.check()`，由 `tests/test_doc_claim_audit.py::test_ratchet_against_baseline`
执行）。这层逻辑对 BEM 本项目有直接意义：`docs/` 里的数字是审稿人/下游用户会引用的量
（附加质量、阻尼、RAO、覆盖率…），一旦无门守护又无基线登记，数字漂移就无人报警。

本单的判据（先于动手定好，全部可机器裁）：

1. `python -m tensorbem.benchmarks.doc_claim_audit --check` 退出码 0；
2. `pytest tests/test_doc_claim_audit.py` 全绿（含新增的工件门与其负控）；
3. 新门必须在**修复前**是红的、修复后是绿的（负控证明它不是空门）。

## 2. 现场诊断（在 `origin/main = 0c23cc1` 上，用本 worktree 的源码实测）

**前置坑（重要）**：本机的 editable 安装（`pip` 的 `__editable__` 指针）指向
`/root/tp-agent/work/m4614/src`——**另一个任务的 worktree**。直接 `python -m tensorbem…` 或
`pytest` 会静默测量别人的代码与文档（本单第一次测量就是这么错的，得到的是 m4614 树的状态）。
本报告全部数据均以 `PYTHONPATH=/root/tp-agent/work/m4862/src` 强制指向本 worktree 后取得。

### 2.1 缺陷 A：棘轮在 main 上就是红的（2 条）

`audit.check(live, origin/main:docs/…/baseline.json)` 非空：

```
new unguarded document: dev_report_m4757_function_level_api_pins.md
new unguarded document: m4789_gauss_rule_fallback.md
```

两个都是**dated 交付物快照**（`paper_level=False`：文件名以 mission 号开头或带 `dev_report_`
前缀，分类器按 #324 的 `^m\d{3,}[_-]` / `dev_report` 规则已正确判成 historical），
但没有任何人把它们登记进基线——于是 `test_ratchet_against_baseline` 与
`test_historical_documents_are_still_registered_in_the_baseline` 双红。
这与 #306/m4655 当初的红是**同一根因族**：新文档进树，基线没跟上。

### 2.2 缺陷 B：已发布工件陈旧（17 条漂移，其中 5 个文档整条缺失）

`docs/doc_claim_audit.md` 开头写着「本文件由 `python -m tensorbem.benchmarks.doc_claim_audit`
程序化生成，无手抄数字」，但**没有任何测试校验这句话**。实测：工件停留在 5 个文档之前的状态。

| 现象 | 数量 |
|---|---|
| 工件里根本没有的文档记录 | 5 |
| 记录字段（tier/tokens/lines）过期 | 9 |
| 合计 counts / coverage 过期 | 2 |
| markdown 正文与实时报告不一致 | 1 |

也就是说 `docs/doc_claim_audit.md` 向读者公布的「71 篇 / 7745 token / 82.8%」在 main 上
已经**不再成立**（实测 76 篇 / 8216 token / 81.9%），有 471 个数字 token 从未被任何人看过。
这正是本项目反复出事的「文档声明 vs 实际」缺陷，只不过这次发生在**审计自己的工件**上。

### 2.3 顺带观测：两篇论文级文档已经补上了守卫（好消息）

`finite_depth3d_regime.md`、`hydrostatic_restoring_couplings.md` 由 `UNGUARDED` 升级为
`PINNED`（论文级 UNGUARDED 从 10 降到 8）。这是别的任务补的读文档测试，本轮只如实登记，
不据功。

## 3. 修复（单一功能点：账本 + 工件不得静默漂移）

### 3.1 基线重写（`--write-baseline`）

按 CLI 的规范命令重写 `docs/doc_claim_audit/baseline.json`：登记 2 个新 historical 交付物，
同时让两篇已升级为 PINNED 的文档从「待守名单」里退出。
这保持了一个可复现性约束：**committed 基线必须等于 `--write-baseline` 的输出**，
否则下一个人重跑命令就又多出一份 diff。（做过比较：把已守卫文档继续留在基线里对门禁毫无作用——
`check()` 只报「变大」，「已守卫」的文档永远不会触发，故不保留为「记忆」。）

### 3.2 新增工件新鲜度门（`artifact_drift` + `write_artifacts` + `--check-artifacts`）

- `artifact_drift(report, json_path, markdown_path) -> list[str]`：把实时审计与
  树里已提交的 `audit.json` / `doc_claim_audit.md` 逐记录（`tier`/`paper_level`/
  `number_tokens`/`lines`）、逐总计（counts/coverage）、逐字节（markdown 正文）比对，
  返回人类可读的失败清单。缺文件也算失败（缺工件不能等于「无漂移」）。
- `write_artifacts(...)`：**不动点写盘**。审计会读 `docs/` 下每个文档——包括它自己的
  markdown，所以「渲染 → 写盘」会改动自己的输入。单趟写盘会发布一份「记录落后自己一个版本」
  的 JSON。函数迭代到渲染结果不再变化才落盘 JSON，保证 `artifact_drift` 的两份工件自洽。
- CLI 增 `--check-artifacts`：红时打印逐条原因并给出重生成命令，退出码 1。
  先检查、后写盘，避免「检查顺手把东西改了」这种假绿。

### 3.3 测试（`tests/test_doc_claim_audit.py` +6，12 → 18）

| 测试 | 作用 |
|---|---|
| `test_committed_artifacts_are_current` | 正门：committed 工件必须描述当前文档 |
| `test_artifact_drift_detects_stale_json` | 负控：JSON 里改 1 个 token 计数必须被抓 |
| `test_artifact_drift_detects_unregistered_document` | 负控：JSON 里丢一条记录必须被抓 |
| `test_artifact_drift_detects_stale_markdown` | 负控：markdown 多一个字节必须被抓 |
| `test_artifact_drift_detects_missing_artifacts` | 负控：工件缺失必须被抓（不能算通过） |
| `test_write_artifacts_reaches_a_fixed_point` | 写两遍必须是幂等（自引用收敛） |

负控一律用合成/篡改副本，不新增真实文档名到 `tests/` 里（`classify` 用「文件名出现在
`tests/` 源码中」判 PINNED，随手写个文件名就假升级一档——模块 docstring 里就警告过这种洗白）。

## 4. 结果

### 4.1 公布值 vs 实测值（修复前 committed 工件 → 修复后）

| 指标 | 修复前工件（origin/main） | 修复后（= 实测） |
|---|---:|---:|
| 文档数 documents | 71 | 76 |
| PINNED | 26 | 30 |
| ARTIFACT | 23 | 24 |
| UNGUARDED | 22 | 22 |
| 其中论文级 UNGUARDED | 10 | 8 |
| 数字 token 总数 | 7745 | 8216 |
| 已守数字 token | 6412 | 6731 |
| 数字覆盖率 | 82.8% | 81.9% |
| 文档覆盖率 | 69.0% | 71.1% |

其中 `artifacts/m4862_audit_gate_evidence.json` 记录：修复前 17 条工件漂移 + 2 条棘轮失败，修复后分别为 0 与 0；未进统计的数字 token 为 471。

### 4.2 工件里整条缺失的文档（修复前）

| document |
|---|
| `dev_report_m4757_function_level_api_pins.md` |
| `finite_depth3d_deepwater_arbitration.md` |
| `kelvin_kernel_reference.md` |
| `m4789_gauss_rule_fallback.md` |
| `two_cylinder_added_mass.md` |

### 4.3 分层变化（修复前 → 修复后）

| document | was | now |
|---|---|---|
| `doc_claim_audit.md` | ARTIFACT | PINNED |
| `finite_depth3d_regime.md` | UNGUARDED | PINNED |
| `hydrostatic_restoring_couplings.md` | UNGUARDED | PINNED |

`doc_claim_audit.md` 由 ARTIFACT 升为 PINNED 是本单引入的**守卫升级**：新增的 freshness 测试逐字节读它的正文（强于「解析其中几个数字」），不是把指标放宽。

### 4.4 门禁输出（修复后，命令实录）

```
$ PYTHONPATH=src python -m tensorbem.benchmarks.doc_claim_audit --check-artifacts
documents=76 pinned=30 artifact=24 unguarded=22 (paper-level 8) number coverage=81.9%
[exit 0]

$ PYTHONPATH=src python -m tensorbem.benchmarks.doc_claim_audit --check
OK: unguarded set did not grow
[exit 0]
```

图：`artifacts/m4862_audit_gate.png`（左：公布 vs 实测分层计数；右：两把门修复前/后的失败条数），由 `tools/m4862_audit_gate_evidence.py` 从证据 JSON 直接绘制。


## 5. 与前 PR #309 / m4655 的关系（诚实交代）

- m4655 的诊断（「PR #306 不是红源，是继承了红掉的 main」）是对的；
  其补丁（`^m\d{3,}` 规则 + 基线补 6 条）已由 **#324（`b31acd6`）** 以更强形式进入 main：
  main 版把 mission 前缀拆成独立 `MISSION_RE`，并保留 `PAPER_LEVEL_EXTRA` 例外
  （`m3538_propeller_openwater_uncertainty.md` 仍是论文级契约，不得降级）。
  因此本轮**不重复**那段补丁，只修 main 上当前仍在的两条缺陷，并补上「工件」这半边。
- 打卡点：本轮的红是**新出现的**（`dev_report_m4757…`、`m4789_gauss_rule_fallback.md` 是
  #324 之后才进树的文档），不是 #324 修漏。

## 6. 注意点 / 踩过的坑

1. **editable 安装指向别的 worktree**（`/root/tp-agent/work/m4614/src`）：本机跑任何
   `python -m pytest` 前必须 `PYTHONPATH=<本 worktree>/src`，否则测的是别人的树（`pyproject.toml`
   的 `[tool.pytest.ini_options]` 没有 `pythonpath` 配置，`tests/` 下也没有 conftest 注入）。
   本单第一次「工件是旧的」结论就是这么产生的假发现——读数之前先确认 `__file__`。
2. **审计读自己的 markdown**：自引用使「生成」不是一次映射而是求不动点，见 3.2。
   若把 markdown 从 `docs/` 移走，`doc_claim_audit.md` 自身记录会消失，覆盖率统计会跟着变，
   属于设计变更，本单不碰。
3. **棘轮只报变大**：`check()` 不检测单篇文档从 PINNED 退化成 ARTIFACT。这个缺口不是本单
   范围（会改动 #324 刚定的语义），已在此登记为已知局限。
4. **本单不追求「覆盖率高」**：覆盖率的上升/下降都是文档变化的被动结果；本单只保证
   公布的数字等于实测的数字。实测覆盖率从 82.8% 降到 81.9%，不是退步，是此前「少算」被纠正
   （新进树的 5 篇文档里有 471 个 token 之前根本没进统计）。

## 7. 误差判定（关键量 vs 基准，逐项）

判据是**精确相等**（布尔量），不是百分比，故误差为 0 或 1：

| 量 | 基准（真实值） | 本单交付值 | 误差 | <3%? |
|---|---|---|---|---|
| `audit.check()` 失败条数 | 0（棘轮语义） | 0 | 0 | 是（精确 0） |
| `artifact_drift()` 失败条数 | 0 | 0 | 0 | 是（精确 0） |
| counts（documents/pinned/artifact/unguarded/paper-level/tokens） | 实时审计 | 同上，逐键相等 | 0 | 是（精确相等） |
| coverage（两个分数） | 实时审计 | 同上，逐位相等 | 0 | 是（精确相等） |
| `pytest tests/test_doc_claim_audit.py` | 全绿 | 见第 4 节 | — | 是 |

修复前的偏差（同一把尺子量）已量化在上表 2.2：5 篇文档缺失、471 token 未进统计、
覆盖率公布 82.8% 而实测 81.9%（相对偏差 1.1%，但方向是「公布了不存在的覆盖」，
故按不可接受处理，不是百分比容差问题）。

## 8. 未完成 / 局限

- 未处理 `check()` 不检测 PINNED→ARTIFACT 退化（见坑 3）。
- 未把 `--check-artifacts` 接进 CI workflow（`.github/workflows` 属平台/CI 配置面，
  本单只交 CLI + 测试；接入建议留给 CI 任务）。
- `docs/doc_claim_audit/doc_claim_audit.png` 由 matplotlib 重生成，二进制随之变化；
  按设计不对 PNG 做逐字节门（matplotlib 版本敏感），只门 JSON 与 markdown。
