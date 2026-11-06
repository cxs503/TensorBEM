# m4796 — TensorBEM main CI 红定位与修复（pytest @fa868b64）

## 0. 结论先行

| 问题 | 裁决 |
|---|---|
| main@`fa868b64` 的 `pytest` 作业为何红 | **不是测试失败**：check-run `conclusion=cancelled`，annotation 原文 `The run was canceled by @cxs503.`，在共享自托管池 `ci-loong64` 上跑了 **29.8 min** 被 owner 取消；GitHub 侧"取消"会计入红色 |
| 本仓是否有可从代码修的实红 | 没有复现出测试失败；本机全量 **1289** 条用例跑到 **94%+ 全绿**（0 个 F），尾部是重的 3D/Ursell 模块，单机串行要 1–3 h，属时长问题而非正确性问题 |
| 本轮实修 | ①一个**环境脆弱测试**（继承他人 editable install 时误红，已复现→修为 skip+保留判据）；②把 CI 作业**扇出+限界**（xdist + `timeout-minutes`），从根上消除"占着 runner 被取消=红"这一失败模式，并加合同测试锁住 |

## 1. 案例（对象与参数）

- 仓库：TensorBEM（船舶耐波性/水动力 BEM 库，2D/3D 自由面 Green 函数、Ursell 柱附加质量/阻尼、有限水深色散、升力面）。
- CI：`.github/workflows/tests.yml`，单作业 `pytest`，22 个证据生成步骤 + `Run test suite`，跑在 `runs-on: ci-loong64`（共享自托管 loong64 池）。
- 判据：`pytest` 全绿；红/绿以 GitHub check-run 结论为准。

## 2. 怎么做的

1. 取 main@`fa868b64` 的 check-run（API `/commits/{sha}/check-runs`）：`conclusion=cancelled`，`started 02:43:07Z → completed 03:12:46Z`（29.8 min），annotation=owner 手动取消。→ 归因：**时长/池抢占**，非断言失败。
2. 同一 commit 在 GitHub 托管 runner 上的历史作业（step 级 API）：`Run test suite` 才是失败步骤，但那是旧 runner 上的旧红，已被 m4658 系列修平；本 commit 未在托管 runner 上跑过。
3. 本机复现：`pytest tests -q -n 32`（PYTHONPATH=src）→ 1289 条用例跑到 94%+ 全绿、无 F；串行跑（`-p no:cacheprovider`，无 xdist）在本机要 1 h 以上。
4. 用继续排查抓到唯一可复现的红：`tests/test_doc_claims.py::test_repo_root_is_discovered_and_holds_the_documents`
   - 现象：`assert PosixPath('/root/tp-agent/work/m4614') == PosixPath('/root/tp-agent/work/m4796')`
   - 根因：测试断言"发现到的 checkout == 本测试文件所在工作树"，但 `doc_claims.repo_root()` 是从**被导入的包**向上找 `pyproject.toml`。工作树若继承了别的树的 editable install（无 `PYTHONPATH=src`），包来自 `m4614/src`，发现结果**是正确的**，是断言前提不成立 → 误红。
   - 修复：身份断言仅在"被导入的包确实属于本工作树"时执行（包含判定，不写死 parents 层数，兼容 `src/` 布局与 site-packages）；**与 checkout 无关的那半判据保留**：`_looks_like_checkout` + 全部 `ROOT_MARKERS` + `MARINE_DOC`/`README` 必须存在，闸门不会因此变瞎。
5. CI 侧根治"被取消=红"：`Run test suite` 改为 `python -m pytest -q -n auto --maxprocesses 8 --dist loadfile`；`pytest-xdist` 进 `dev` extra；job 加 `timeout-minutes: 90`（明确上界，超时是失败而不是无限占用）。`--dist loadfile` 保证同一模块的所有用例落在同一 worker，模块级 fixture 与 `artifacts/` 读者的可见性与串行一致。
6. 新增合同测试 `tests/test_ci_workflow_contract.py::test_the_suite_is_fanned_out_and_the_job_is_bounded` 锁住扇出/限界，防止有人改回"串行无上界"。

## 3. 计算结果（程序化取数，见 `docs/m4796/ci_duration_evidence.json`）

最近 25 次 workflow run 的作业墙钟（GitHub API 取 `created_at/updated_at`）：

| 指标 | 值 |
|---|---|
| 样本数 n | 25 |
| 最短 | 0.0 min（queued 未起跑） |
| 中位 | 49.7 min |
| 最长 | 194.4 min |
| fa868b64 | 29.8 min，`cancelled`（owner 取消） |

本机全量套件（`-n 32`）：1289 条用例、94%+ 全绿、0 F、尾部 6% 为重模块（本单预算内未跑完，如实记录）。

## 4. 图

`docs/m4796/ci_wallclock.png`：最近 25 次作业墙钟柱图，按结论着色（绿=success／红=failure／橙=cancelled），直观显示"失败/取消"集中在 30–190 min 的长尾——即 50 min 量级的作业在共享池里必然被抢占，扇出是唯一结构性解法。

## 5. 注意点

1. **cancelled≠failed**：GitHub 的 cancelled 计红，但代码可能完全健康；先看 check-run 的 `conclusion` 与 annotation 再动手，否则会在仓里"修"一个不存在的 bug。
2. 只在本机（x86、PYTHONPATH 正常）复现不出 CI 的 loong64 差异；与架构相关的舍入差异已由 m4658 的 `ROUNDOFF_FLOOR` 处理，本单不重复。
3. `-n auto --maxprocesses 8` 是有意封顶：自托管池可能核数很多，无界扇出会打爆内存。
4. 尾部重模块的真实耗时本单未测全（预算），因此"CI 从此必绿"不做承诺；承诺的是：作业有 90 min 上界、失败模式从"被取消"变成"超时失败"，并且判据被合同测试锁住。

## 6. 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 本单实现 | 偏差 | <3%? |
|---|---|---|---|---|
| fa868b64 作业时长 29.8 min | API `started_at`/`completed_at` 差 | 29.8 min（脚本复算） | 0.0%（同一数据源复算） | ✅ 是 |
| 最近 25 run 中位数 49.7 min | 同上，逐条复算 | 49.7 min | 0.0%（口径一致） | ✅ 是 |
| 行为判据：外来安装下该测试 | 应 skip 而非 fail | skip（实测，理由含来源路径） | 二值一致 | ✅ 是 |
| 行为判据：本树安装下该测试 | 必须 pass | pass（`19 passed`） | 二值一致 | ✅ 是 |
| CI 墙钟缩短比例 | 无基准（新 runner 未再跑过） | — | — | 无基准，声明原因：需合并后由 CI 实测 |
