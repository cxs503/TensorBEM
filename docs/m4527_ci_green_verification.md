# PR #279（`agent/m4293`）CI 复红的定位与修复证据 —— mission m4527

**结论**：CI 的 `pytest=failure` 不是分支内容的问题，而是 **GitHub PR 事件检查的是
`merge(PR head, 当前 main)`**，而 `main` 在本分支切出后前进了 14 个提交
（`5364459..efec0d7`），其中 `m3878` 新增文档 `docs/merge_m3878_pr104_resolution.md`
（分支树里没有、合并树里有、且无测试读它）→ 触发 doc-claim 棘轮门
`test_ratchet_against_baseline`。修法 = 合入 main + 用文档化工具重切基线（差异恰好 1 条）。

## 1. 复现（本地等价于 CI）

| 树 | `tests/test_doc_claim_audit.py` |
|---|---|
| PR head 单树（`6a2534a`） | 9 passed（绿 —— 所以"分支绿"不足以判 PR 绿） |
| `merge(PR head, origin/main)` | **1 failed**: `['new unguarded document: merge_m3878_pr104_resolution.md'] == []` |

```
git worktree add ../m4527_merge -B m4527-merge-test agent/m4293
git -C ../m4527_merge merge --no-edit origin/main      # clean
PYTHONPATH=src python -m pytest tests/test_doc_claim_audit.py -q
```

## 2. 修复

1. `git merge origin/main`（clean，无冲突）—— 合并树即 CI 检查树；
2. `python -m tensorbem.benchmarks.doc_claim_audit --write-baseline`：`baseline.json` 差异
   **恰好一条**（`+ "merge_m3878_pr104_resolution.md"`），论文级未守集合 10 份不变；
   该文档的证据件 `docs/m3878/kernel_merge_ledger.json` 已提交但无测试读、无生成器声明，
   故按同一判据记为显式欠账，**不**临时加文件名把它伪装成 PINNED；
3. 重新发布审计证据：`docs/doc_claim_audit.md` / `audit.json` 与合并树对齐
   （documents 60→64，pinned 23→25，artifact 21→22，unguarded 16→17，数字覆盖 83.7 %）。

## 3. 验证（loong64，本工作树）

| 检查 | 结果 |
|---|---|
| `test_doc_claim_audit.py` + `test_skip_gates_are_tracked.py` + `test_multicomponent.py` | **44 passed**（44 s） |
| 全套 `pytest tests -n 10`（未跑完，宿主被他单压满） | 已完成 **833 / 1247** 项，**0 failed**、`s` 为既有 skip |
| CI 证据链步骤差异 | 未在本地重放（CI 的 14 条 benchmark 步骤依赖 x86 轮/空闲算力），据 §1 判据红只来自合并树基线漂移 |

未完成部分（如实记录）：全套 1247 项在单机 120 s 壳位/被压满的宿主上跑不完，故只给出
833 项零失败 + 目标模块 44 项全绿的证据；`tests/test_finite_depth3d_pins.py` 的 4 个
roundoff 级 pin 在 loong64 上预期红、x86 CI 上绿（见 `docs/m4293_ci_red_fix.md` §4），
与本次修复无关。
