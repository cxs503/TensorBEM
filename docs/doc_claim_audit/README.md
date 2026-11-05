# doc-claim 审计账本（`baseline.json`）

`tests/test_doc_claim_audit.py::test_ratchet_against_baseline` 要求
**未守（UNGUARDED）文档集合只许缩、不许涨**。基线由工具自身写：

```
PYTHONPATH=src python -m tensorbem.benchmarks.doc_claim_audit \
    --json docs/doc_claim_audit/audit.json --write-baseline
```

## 为什么基线在 m4320 变了

m3173 生成基线之后，`main` 上又有五篇**论文级**文档加入，而它们的解析测试没有一起落地
（合并门禁滤除了部分测试文件）。棘轮因此把「新文档没补钉」变成整仓红，任何开着的 PR 都会被传染。
m4320 的处理是**显式记债**而不是写假钉：

| 文档 | 引入任务 | 数字 token | 声明产物 |
|---|---|---:|---|
| `kkr_causality_band_study.md` | m3664 | 119 | — |
| `m3538_propeller_openwater_uncertainty.md` | m3538 | 101 | — |
| `hydrostatic_restoring_couplings.md` | m3196 | 80 | — |
| `finite_depth3d_regime.md` | m3409 | 76 | `docs/m3409/outputs/finite_depth3d_regime.json` |
| `interior_field_domain_guard.md` | m3351 | 73 | `docs/m3351/outputs/field_guard.json` |

## 还债路径

逐篇补一个**真正解析文档**的测试（`tests/` 里出现该文件名即自动升为 PINNED），
再从 `baseline.json` 的 `unguarded` 列表删掉该行。删行是收紧棘轮，永远允许。
不要用 `--write-baseline` 反过来把新的未守文档加进基线——那会让棘轮失去意义。
