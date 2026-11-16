# 开发报告 m23256：把 XVAL-BEM-R2 的退化审计文档接回归档数据

**仓库** TensorBEM · **锚点** `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE` · **分支** `agent/m23256`
**性质** 缺陷修复（文档与归档数据分叉）+ 防复发门禁

## 1. 背景与问题

PR#1089 的交付物（`results.json`、`docs/capytaine_crosscheck_r2.md`、`capytaine_crosscheck_r2.py`、测试）已于 Merge #923（m20476）等到 Merge #1136 之间进入 main；本单按「管到底」原则**不重做正文**，只核验 main 上的实际状态。

核验办法与结论（先把判定写死再看结果）：

| 检查 | 判据 | 实测 |
|---|---|---|
| 主文档与归档数据一致 | `docs/capytaine_crosscheck_r2.md` 逐字节等于 `render_document(results.json)` | 通过（已由 `tests/test_capytaine_crosscheck_r2.py` 把守） |
| 归档自洽 | `results.json` 内 142 个 active 项、266 个结构零、456 个不变零的计数与主文档表 3.1 逐格相等 | 通过 |
| 退化轴 | 残差随网格单调衰减、中心变换残差达机器精度 | 通过（衰减比 3.47/3.84、3.36；中心变换 1.1e-16） |
| 测试 | R2 相关测试在本工作树全绿 | 通过（35 passed） |
| **审计文档与归档数据一致** | `docs/audit_xval_r2_degeneracy.md` 的每个数字须能由 `results.json` 复现 | **不通过** ← 本单的缺陷 |

缺陷内容：`docs/audit_xval_r2_degeneracy.md` 最后一次修改是 Merge #923（m20476，首测轮），此后 m22034 把四个本体从 256/270 面板细化到 1024/1026 面板并替换了 `results.json`，**审计文档没有跟着更新**，于是它对同一锚点报出：

| 量 | 审计文档（陈旧） | 归档实际 |
|---|---:|---:|
| 最差计分偏差 | 87.93 % | 6.71 % |
| 精确零界限越界数 | 16 | 0 |
| `compare()` 失败项 | 137 | 8 |
| 退化残差（中心变换） | 1.7e-16 | 1.1e-16 |
| 建议 | 「须细化到 1024+ 面板重判」 | 归档已经是 1024/1026 面板 |

后果不是数值错，而是**结论错**：读者会得到「球本体仍有 87.93 % 的未解缺口、建议尚未执行」的印象，而归档的真实状态是「4 个算例全部落在 15 % 验收线内、最差 6.71 %、内部 5 % 对角带仍未满足」。陈旧数字之所以能存活，是因为唯一读它的地方是人的眼睛——没有任何测试绑它。

## 2. 修法（结构性，不做手工改数）

1. 新增 `src/tensorbem/benchmarks/xval_r2_degeneracy_audit.py`：把审计文档变成**派生文件**。`summarise(results.json)` 抽出可比数字，`render_document(current, history)` 渲染全文，`--write` 落盘、`--check` 在漂移时非零退出。
2. 新增 `benchmarks/verified/capytaine_crosscheck_r2/degeneracy_round_m20476.json`：首测轮不再被静默覆盖，而是冻结为 4.5 KB 摘要（含 `source_revision` 4123e585、`source_path` 与取代说明），由 `--write-history-from <该轮 results.json>` 生成，历史列同样不是手抄。
3. 文档改为**两轮并列**：首测轮（256/270 面板，最差 87.93 %、16 处越界）与当前归档（1024/1026 面板，最差 6.71 %、0 处越界）同表呈现，并显式写出三个口径的关系（15 % 验收线 / 5 %·10 % 内部登记带 / 3 % 目标）。
4. 新增 `tests/test_xval_r2_degeneracy_audit_doc.py`：要求文档与渲染结果**逐字节相同**，并把「陈旧数字不得出现在当前轮标签下」「历史轮必须仍是粗网格且更差」「历史轮零越界数必须仍为正（否则叙述要重写）」写成断言。文档因此从 PINNED 之外变成被读的文档（doc-claim 审计把它归类为 `PINNED`，reading test = 本测试文件）。

## 3. 结果

- `python -m tensorbem.benchmarks.xval_r2_degeneracy_audit --check` → 文档与归档一致。
- `pytest tests/test_xval_r2_degeneracy_audit_doc.py tests/test_capytaine_crosscheck_r2.py tests/test_capytaine_r2_invariant_residual.py` → 45 passed。
- 归档数值、带宽、判定**未改动一个字节**：本单只消除文档与数据的陈述分叉。

## 4. 注意点与局限

* 本次不触碰求解器、`results.json`、主文档与任何带宽；`comparison.verdict` 仍是 FAIL（内部带），`ledger_acceptance.satisfied` 仍是 true（15 % 验收线）——两者并存是设计，现在文档把这一点写清楚了。
* 面板数仍不对等（1024/1026 对 784/1120），且两码面元离散不同，**面板数相等不等于离散误差相等**；剩余 6.71 % 不能只归给分辨率，等误差水平验证仍未做。
* 4 个算例最差值 5.82/5.82/6.52/6.71 % 均 **未 <3 %**；测试里显式断言「若将来出现 <3 % 的算例，必须重写诚实 FAIL 段落」，防止文档自动变成过度乐观。
* `tests/test_doc_claim_audit.py` 有 3 个用例红（未登记的 `UNGUARDED` 文档、陈旧审计工件）。**在 HEAD 的干净解包上同样红**，属既有欠账，本单未动；本单的文档反而被该审计归为 `PINNED`。

## 5. 复现

```
PYTHONPATH=src python -m tensorbem.benchmarks.xval_r2_degeneracy_audit --check
PYTHONPATH=src python -m tensorbem.benchmarks.xval_r2_degeneracy_audit --write
pytest tests/test_xval_r2_degeneracy_audit_doc.py -q
```