# m4614 · doc-claim ratchet: mission-prefixed reports are work products, not contracts

Status: fix + regression tests, gate green on current `main`.
Scope: `src/tensorbem/benchmarks/doc_claim_audit.py`, `docs/doc_claim_audit/baseline.json`,
`tests/test_doc_claim_audit.py`, regenerated `docs/doc_claim_audit.md` /
`docs/doc_claim_audit/audit.json` artifacts.

## 1. The defect

`doc_claim_audit` splits every number-bearing markdown file into two orthogonal
axes:

* **tier** — `PINNED` / `ARTIFACT` / `UNGUARDED` (does anything read the numbers?);
* **paper level** — `True` for a contract with a reader, `False` for a dated
  snapshot of one run (dev reports, audits, reviews, incident logs).

The ratchet in `check()` uses both halves: the `UNGUARDED` set may never grow,
and a *paper-level* document may never leave the guard set. The second half is
the strict one — it is what forces a new contract document to be pinned or
generated before it lands.

Membership in "historical" was decided by `HISTORICAL_RE`, which knows the
spellings

| spelling | example |
|---|---|
| `dev_report` | `dev_report_tensorbem_declared_numbers.md` |
| `report_m\d` | `report_m96.md` |
| `_m\d{3,}` | `m3977_pr109_merge_healing_report.md`? — **no** |
| `audit` / `review` / `incident` | `tensorbem_health_audit_m1621.md` |

Mission reports have migrated to *mission id first* (`m4290_ci_fix_report.md`,
`m4283_pr186_ci_triage.md`, `m3977_pr109_merge_healing_report.md`). That
spelling matches none of the patterns: the id is not behind a prefix and there
is no leading underscore. The result was an accounting error, not a numerical
one — three frozen snapshots of past runs were counted as paper-level contracts
that "lost their guard":

```
unguarded_paper_level: 13   <- of which 3 are dated mission reports
```

Every consequence of that error is wrong in the expensive direction:

* `check()` cannot distinguish "a contract regressed" from "the baseline still
  lists an old report", so the only way to keep the gate green was to keep
  hand-editing the paper-level half of the baseline for work products that are
  *supposed* to be frozen;
* the audit's headline metric (`unguarded_paper_level`, and the "paper-level
  UNGUARDED" table in `docs/doc_claim_audit.md`) over-reported risk, which is
  the failure mode a ratchet is least able to tolerate.

## 2. Fix

`HISTORICAL_RE` is a substring match, so extending it is unsafe: a document that
merely *mentions* `m4290_` inside a name would flip. The fix is a separate,
anchored pattern plus an explicit exemption list:

```python
MISSION_RE = re.compile(r"^m\d{3,}[_-]")          # id leads the name: m4290_..., m3538-...
PAPER_LEVEL_EXTRA = frozenset({"m3538_propeller_openwater_uncertainty.md"})

def is_historical(name):
    if name in PAPER_LEVEL_EXTRA:   # contracts that only look like work products
        return False
    if name in HISTORICAL_EXTRA:    # existing English/Chinese singles
        return True
    return bool(HISTORICAL_RE.search(name) or MISSION_RE.search(name))
```

Design points:

1. **Anchored, separator-required.** `^m\d{3,}[_-]` needs the id at the start
   *and* a separator after the digits, so `mid3_report.md` stays paper-level.
2. **Freezing is not exempting.** A frozen report still reaches the ratchet
   through the `UNGUARDED` half: it must be registered in
   `baseline["unguarded"]`. The mission rule only moves a document out of the
   *paper-level* half.
3. **Exemptions are explicit.** `m3538_propeller_openwater_uncertainty.md` is a
   mission-prefixed name that is nevertheless a reader-facing benchmark note, so
   it keeps the paper-level gate. Making that a named frozenset (instead of a
   regex exception) means `test_paper_level_exemptions_survive_the_mission_prefix_rule`
   can assert the two invariants together.
4. **A naming trap is now documented.** Tier assignment is a plain filename
   match, so quoting a report's *file name* in a test docstring silently marks
   that report `PINNED` ("tier laundering"). The module docstring says so, and
   the new tests use synthetic `m9999_*.md` names to avoid it.

## 3. Baseline accounting

`baseline.json` is regenerated (`--write-baseline`) so both halves describe the
same classification the code now computes. Only the paper-level half moves:

| key | before | after |
|---|---:|---:|
| `unguarded` | 22 | 22 (unchanged — frozen reports stay registered) |
| `unguarded_paper_level` | 13 | 10 |
| removed from paper level | `m3977_pr109_merge_healing_report.md`, `m4283_pr186_ci_triage.md`, `m4290_ci_fix_report.md` | |

Fleet-wide audit after the change (regenerated programmatically, no hand-copied
digits — `docs/doc_claim_audit/audit.json` is the source):

| metric | value |
|---|---:|
| audited documents | 70 |
| `PINNED` | 26 |
| `ARTIFACT` | 22 |
| `UNGUARDED` | 22 |
| of which paper-level | 10 |
| numeric claim tokens | 7744 |
| guarded tokens | 6411 |
| numeric coverage | 82.8 % |

## 4. Tests

`tests/test_doc_claim_audit.py` gains three tests (12 total, all green):

* `test_mission_prefixed_reports_are_historical` — positive cases plus the
  negative control `mid3_report.md`; uses synthetic names so it cannot pin a real
  document.
* `test_paper_level_exemptions_survive_the_mission_prefix_rule` — every name in
  `PAPER_LEVEL_EXTRA` must match `MISSION_RE` (so the exemption is meaningful)
  and must *not* be historical.
* `test_historical_documents_are_still_registered_in_the_baseline` — the negative
  control for point 2 above: the set of unguarded *and* historical documents must
  be a subset of the registered baseline, so "freeze it" can never become
  "forget it".

Reproduce:

```bash
export PYTHONPATH=src
python -m tensorbem.benchmarks.doc_claim_audit --write-baseline --check
python -m pytest tests/test_doc_claim_audit.py -q
```

Honest notes / limitations:

* The classification is a heuristic over file names and test sources; it cannot
  see a number that a test pins through a different document, nor a report whose
  numbers are correct but superseded. It answers "is this file read?", not "is
  this file right?".
* `PAPER_LEVEL_EXTRA` is a manual list by design. Adding a name is a policy
  decision and should be reviewed, not bulk-generated.
* This change does not reduce the 22 unguarded documents; it only stops three of
  them from being miscounted. Guarding them is the ratchet's next round of work.
# m4614 — main CI 红修复：doc-claim 棘轮（TensorBEM）
**日期**：2026-11-06  **基线 commit**：`a261e23`（main HEAD）
**失败测试**：`tests/test_doc_claim_audit.py::test_ratchet_against_baseline`
**判定**：REAL 缺陷（真实的门禁退化），非随机/flaky。
## 1. 复现（不是猜的）
按 `.github/workflows/tests.yml` 的顺序本地复刻：先跑 19 个 benchmark 证据步骤生成
`artifacts/*.json`，再跑 `pytest`。串行运行时第一个红就是本项：
FAILED tests/test_doc_claim_audit.py::test_ratchet_against_baseline
1 failed, 158 passed, 6 skipped in 463.95s
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
## 7. 重开落仓（m4807，2026-11-06，PR#308 关闭后补开）
原 PR#308 于 2026-11-05 被 owner 裁定关闭，本轮从当时的 `main`（`fa868b6`）切枝重开，
内容按上 §3 原样重放（`src` + `tests` 两处代码逐字一致），只对 `baseline.json` 重新记账。
与关闭时的差别有两处，都在 `docs/doc_claim_audit/baseline.json`：
* `unguarded` 半边的 6 条（`dev_report_m4291`、`m3977`、`m4283`、`m4290`、`merge_m3878`、
  `merge_report_m3929`）在本轮改动前已经由 main 上的并行任务记账，故本轮 `unguarded`
  不动（仍 22 条）；
* `unguarded_paper_level` 半边在 main 上仍带着 `m3977`、`m4283`、`m4290` 三条（13 条）——
  这正是 §2 的命名盲区在基线里的残留。`MISSION_RE` 生效后这三份被判 historical，
  基线按 `--write-baseline` 重生成，`unguarded_paper_level` 回到 **10 条**（与 §4 结论一致）。
  棘轮语义因此更严不变松：paper-level 门槛仍是 10 份，未新增任何豁免。
### 7.1 复验读数（本轮，`PYTHONPATH=<worktree>/src` 钉死后串行）
documents=71 pinned=26 artifact=23 unguarded=22 (paper-level 10) number coverage=82.8%
OK: unguarded set did not grow            # --check 退出码 0
tests/test_doc_claim_audit.py            12 passed
doc 门禁层（audit + doc_claims + inventory + prose_claims） 66 passed
无基准可比，故误差判定只落在「棘轮是否响应」这一类**布尔判据**上：注入一条新的未守护
文档 → `check` 返回 `new unguarded document`；把某份 paper-level 文档摘出 → 返回
`paper-level document lost its guard`（两条负控由 `test_ratchet_detects_regression`
与 `test_historical_documents_are_still_registered_in_the_baseline` 覆盖），本轮二者均为红，
即门禁仍然咬得住。
### 7.2 新增注意点：解释器指向别的 worktree（本轮踩到）
共享主机上 `site-packages/__editable__.tensorbem-*.pth` 指向的是**上一个任务**
（`work/m4614/src`），因此 `pytest` 不加 `PYTHONPATH` 时跑的是别的工作树：本树新增的测试
根本不会被执行，且 `test_doc_claims.py::test_repo_root_is_discovered_and_holds_the_documents`
会因 `repo_root()` 指到别的目录而假红。判据：跑测试前先
`python -c "import tensorbem; print(tensorbem.__file__)"`，必要时
`PYTHONPATH=$PWD/src` 钉死；本次所有读数均在该前提下取得。
### 7.3 遗留债（如实）
* `docs/doc_claim_audit/audit.json` 与同名 png 是上一次公开快照（`documents=60`），
  已落后于 main（现 71）。本轮**未**重生成：重生成会把 11 份与本次改动无关的文档
  差异混进同一个 PR，违反单 PR 单一功能点纪律；重生成命令即
  `python -m tensorbem.benchmarks.doc_claim_audit --figures docs/doc_claim_audit`。
* §4 的 `classical_study` 串行证据缺口与 §6 的“全仓全绿未完整复刻”结论不变。
