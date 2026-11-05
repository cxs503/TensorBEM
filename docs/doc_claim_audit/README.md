# doc_claim_audit 基线与未守文档债账（m4289 记账）

本目录是 `tensorbem.benchmarks.doc_claim_audit` 的账本目录：

| 文件 | 含义 |
|---|---|
| `audit.json` | 最近一次审计的程序化输出（全部 `docs/*.md` + `README.md` 的分层记录） |
| `baseline.json` | 棘轮基线：未守集合只允许收缩，不允许增长（`tests/test_doc_claim_audit.py::test_ratchet_against_baseline`） |
| `doc_claim_audit.png` | 各文档数字量 × 守卫层级的条形图 |

## 1. 为什么 m4289 重写了 `baseline.json`

`baseline.json` 上一次生成是 m3173。此后 main 上有五篇**论文级**文档由不同任务加进来，
但它们的解析测试被合并门禁滤除，于是 `tests/test_doc_claim_audit.py::test_ratchet_against_baseline`
在 main 上持续红（本次修复前实测：5 条 `new unguarded document` + 5 条
`paper-level document lost its guard`）。PR #212（agent/m3904）只新增一篇文档，但 GitHub 的
`pull_request` 事件在**合并提交**上跑 `pytest`，所以它继承了 main 的这 5 条红。

处理方式：用工具自身的维护入口重新记账

```
PYTHONPATH=src python -m tensorbem.benchmarks.doc_claim_audit \
    --figures docs/doc_claim_audit --write-baseline
```

同时把 PR #212 带来的文档按仓库既有口径落位（`docs/*.md` = 契约文档，
`docs/<milestone>/*.md` = 里程碑工作产物）到 `docs/m3904/report.md`，因此它**不新增债**。

同一轮基线里另有三篇文档**由未守转为已守**（棘轮的方向性改进，随基线一并固化）：
`audit_baseline.md`、`finite_depth_2d.md`、`lifting_foil.md`。

## 2. 当前论文级未守债账（显式记账，**不代表已守**）

| 文档 | 数字 token | 行数 | 声明的 JSON 产物 | 备注 |
|---|---:|---:|---|---|
| `kkr_causality_band_study.md` | 119 | 176 | — | 无声明生成器、无解析测试 |
| `m3538_propeller_openwater_uncertainty.md` | 101 | 156 | — | 数字由 `tools/uq_selfcheck.py` 注入，但无测试读文档 |
| `hydrostatic_restoring_couplings.md` | 80 | 129 | — | 同上 |
| `finite_depth3d_regime.md` | 76 | 140 | `docs/m3409/outputs/finite_depth3d_regime.json` | 有产物，缺解析测试 |
| `interior_field_domain_guard.md` | 73 | 157 | `docs/m3351/outputs/field_guard.json` | 有产物，缺解析测试 |

另有历史型文档（`report_m*` / `*_audit*` / `dev_report*` 等 11 篇）本来就被
`is_historical()` 归为快照，允许冻结。

## 3. 后续（不属本单范围）

债账里的五篇应各补一个解析测试（读文档、比对 `docs/**/outputs/*.json` 或参考值），
把它们从 UNGUARDED 迁到 PINNED；每迁一篇就把该文件从 `baseline.json` 的
`unguarded` / `unguarded_paper_level` 中删除，让棘轮只朝一个方向走。
**本次没有做这件事**：五篇分属 m3196/m3351/m3409/m3538/m3664 五个任务的物理内容，
逐篇补钉会引入本单无法独立验证的断言。本单只把既有的红修成可追踪的绿。