# m4614 — main CI 红修复：doc-claim 棘轮（TensorBEM）

**日期**：2026-11-06  **基线 commit**：`a261e23`（main HEAD）
**失败测试**：`tests/test_doc_claim_audit.py::test_ratchet_against_baseline`
**判定**：REAL 缺陷（真实的门禁退化），非随机/flaky。

## 1. 复现（不是猜的）

按 `.github/workflows/tests.yml` 的顺序本地复刻：先跑 19 个 benchmark 证据步骤生成
`artifacts/*.json`，再跑 `pytest`。串行运行时第一个红就是本项：

```
FAILED tests/test_doc_claim_audit.py::test_ratchet_against_baseline
1 failed, 158 passed, 6 skipped in 463.95s
```

`audit.check(report, baseline)` 返回 9 条失败，分两类：

| # | 类型 | 文档 |
|---|---|---|
| 1 | new unguarded | `dev_report_m4291_pr215_ci_red.md` |
| 2 | new unguarded | `merge_m3878_pr104_resolution.md` |
| 3 | new unguarded | `merge_report_m3929_pr134.md` |
| 4 | new unguarded | `m3977_pr109_merge_healing_report.md` |
| 5 | new unguarded | `m4283_pr186_ci_triage.md` |
| 6 | new unguarded | `m4290_ci_fix_report.md` |
| 4–6 | 同时被判「paper-level 失去守护」 | 同上三份 |

## 2. 根因

`docs/doc_claim_audit/baseline.json` 是棘轮下限，最近若干“CI 抢修/合并”任务的产报
（`m4291`、`m4283`、`m4290`、`m3977`、`m3878`、`m4292`）陆续进了 `docs/`，但没同步记账，
`unguarded` 集合从 16 涨到 22，门禁按设计报警（m4285 已立过同样的判例）。

另外三份被判成 **paper-level** 是**分类器的命名盲区**，不是真的“论文级契约失效”：
`HISTORICAL_RE` 认得 `_m\d{3,}`（id 带下划线后缀）、`dev_report_`、`report_m\d`，
但认不出任务 id 挪到名字最前面的写法（`m4290_ci_fix_report.md`）。
同样是“某一次运行的时间快照”，只因拼写不同就被当成逐字契约，棘轮因此多报 3 条假警。

## 3. 修法（最小改动，不削弱门禁）

1. `src/tensorbem/benchmarks/doc_claim_audit.py`
   * 新增 `MISSION_RE = ^m\d{3,}[_-]`：任务 id 前置的产报与既有的 `_m\d{3,}` 同等待遇（historical）。
   * 新增 `PAPER_LEVEL_EXTRA = {m3538_propeller_openwater_uncertainty.md}`：该文档名字像任务产报，
     但它是论文级研究笔记，必须留在 paper-level 门禁里——不让新规则把它悄悄放过。
   * 模块 docstring 补一条陷阱说明：`PINNED` 只按文件名出现与否判定，因此在测试里
     **提到**某个报告的文件名（哪怕只是 docstring）就等于宣称“它的数字被解析了”，
     会把未守护文档洗成 PINNED。本任务自身就踩过这个坑（见 §5）。
2. `docs/doc_claim_audit/baseline.json`：按测试自带的官方补救路径重生成
   （`python -m tensorbem.benchmarks.doc_claim_audit --write-baseline`），把 6 份已存在的
   未守护文档显式记账。**`unguarded_paper_level` 一字未增**——paper-level 的 10 份仍是原来 10 份。
3. `tests/test_doc_claim_audit.py`：新增 3 个测试
   * `test_mission_prefixed_reports_are_historical`：任务 id 前置 → historical（用合成名，避免钉住真文档）；
   * `test_paper_level_exemptions_survive_the_mission_prefix_rule`：豁免名单里的文档仍留在 paper-level；
   * `test_historical_documents_are_still_registered_in_the_baseline`：负控——historical 只免除
     paper-level 半边棘轮，`UNGUARDED` 半边照样要求记账，新报告不登记仍然红。

## 4. 结果

| 量 | 修前 | 修后 |
|---|---|---|
| `test_ratchet_against_baseline` | FAILED（9 条） | PASSED（0 条） |
| `tests/test_doc_claim_audit.py` | 8 passed + 1 failed | **12 passed** |
| `audit --check` 退出码 | 1 | **0**（`OK: unguarded set did not grow`） |
| documents / pinned / artifact / unguarded | 70 / 26 / 22 / 22 | 70 / 26 / 22 / 22 |
| paper-level unguarded | 10（基线）→ 报出 13 | **10（与基线一致）** |
| 数字覆盖率 | 82.8% | 82.8% |
| baseline.json | 16 条 unguarded | 22 条 unguarded，paper-level 仍 10 条 |

（上表是修后树 `documents=70` 的读数；本报告自身入仓后重跑审计为 `documents=71 pinned=26 artifact=23 unguarded=22 (paper-level 10)`，因为报告声明了 `python -m tensorbem.benchmarks.doc_claim_audit` 而被归入 ARTIFACT，未增加任何未守护债；`--check` 仍返回 0。）

棘轮机制本身没有变化：新增未守护文档、或某文档丢掉守护，仍然会红（有负控测试为证）。

## 5. 注意点 / 踩坑

* **文件名即证据**：`classify()` 用 `doc_path.name in test_source` 判 PINNED。我第一版测试里
  把三份报告的真实文件名写进断言，结果它们全被洗成 PINNED（`unguarded` 22→17），
  门禁变绿得毫无道理。改用合成名（`m9999_...`）并把这条陷阱写进模块 docstring。
* **别用并行跑羸弱证据**：`pytest -n` 下多个模块并发写 `artifacts/*.json`，
  会出现与本改动无关的 F/E，位置随 worker 调度变（`-n 24` 与 `-n 8` 红点不同）。
  判据以串行为准。
* 记账 ≠ 根治：这 6 份产报的根治路径是让它们声明 `python -m tensorbem...` 生成器（回到 ARTIFACT），
  或干脆不进 `docs/`。本轮只做显式记账，债留痕于此（同 m4285 判例）。
* `benchmarks.classical_study` 在本机（共享 128 核、多智能体争抢）单步 >30 min，
  全量证据链串行复刻未能在本任务预算内跑完；无任何测试读 `classical-study.json`（已 grep 确认）。

## 6. 验证边界（honest）

* 已验证：本项红 → 绿（串行、CI 顺序下的证据链），doc 门禁层（`test_doc_claim_audit.py` 12/12）。
* 未完整验证：全仓 2000+ 测试的串行全绿。并行全量跑（`-n 8`）在 33% 处出现 4 个与本改动
  无关模块的 F（并发写 artifacts 的竞赛），未在预算内定位；因此**不宣称**“main CI 全绿”，
  只宣称“本单确认的红已修，且改动落在 doc 门禁这一层、层内全绿”。