# m4796 — TensorBEM main CI 红定位与修复（pytest 作业时长 / 环境脆弱断言）

> 本报告由 m4891 从当前 main 重提（原 PR #325 已由 owner 裁定关闭，成果按 merge_master 流程补开）。
> 相对原单的唯一实质更新：当前 main 已合入 m4803（`runs-on` 改为 `vars.TENSORBEM_CI_RUNNER || 'ubuntu-latest'`），
> 因此"作业跑在 `ci-loong64`"一句改为"作业跑在 `ci-*` 自托管池或 GitHub 托管池，由仓库变量决定"。
> 扇出与限界这两条修复与池无关，两种池上都成立，故全部保留。

## 0. 结论先行

| 问题 | 裁决 |
|---|---|
| main@`fa868b64` 的 `pytest` 作业为何红 | **不是测试失败**：check-run `conclusion=cancelled`，annotation 原文 `The run was canceled by @cxs503.`，在共享自托管池上跑了 **29.8 min** 被取消；GitHub 侧"取消"计入红色 |
| 本仓是否有可从代码修的实红 | 唯一可复现的一条是**环境脆弱测试**（继承他人 editable install 时误红），已复现→修为 skip 且保留判据；其余没有断言失败 |
| 本轮实修 | ①把 CI 作业**扇出 + 限界**（`pytest-xdist` `--dist loadfile` + `timeout-minutes: 90`），从结构上消除"占着 runner 被取消 = 红"这一失败模式；②修掉那条环境脆弱断言；③两条都用测试锁住 |

## 1. 案例（对象与参数）

- 仓库：TensorBEM（船舶耐波性/水动力 BEM 库：2D/3D 自由面 Green 函数、Ursell 柱附加质量与阻尼、有限水深色散、升力面）。
- 被检对象：`.github/workflows/tests.yml`，单作业 `pytest`，22 个证据生成步骤 + `Run test suite` + artifact 上传。
  - 本轮之前 `Run test suite` 是裸 `pytest`（串行、无时长上界），作业本身也没有 `timeout-minutes`。
  - runner：`ci-*` 自托管 loong64 池（2026-11-06 起该池停止派发，m4803 把默认值改为 GitHub 托管池，自托管池保留为一个仓库变量即可切回）。
- 判据（先于动手固定）：`pytest` 全绿；红/绿以 GitHub check-run 的 `conclusion` 为准，`cancelled` 与 `failure` 同样计红。

取样参数（全部来自 GitHub Actions API，逐条存进 `docs/m4796/ci_duration_evidence.json`）：

| 参数 | 值 |
|---|---|
| 样本 | 该 workflow 最近 25 次 run |
| 结论分布 | `cancelled` 10 / `failure` 11 / `success` 2 / `queued` 2 |
| 墙钟口径 | run 的 `created_at → updated_at`（分钟） |
| 目标 commit | `fa868b64`（`conclusion=cancelled`，29.8 min） |

## 2. 怎么做的

1. **取证，先分清红是什么红。** 取 `fa868b64` 的 check-run：`conclusion=cancelled`，`started 02:43:07Z → completed 03:12:46Z`（29.8 min），annotation 是 owner 手动取消。→ 归因：**时长 / 池抢占**，不是断言失败。
2. **本机复现。** 以 CI 将要用的同一条命令跑全量套件（`pytest -n auto --maxprocesses 8 --dist loadfile`）：无 F，只有若干已登记 skip 与 xfail；串行跑则要 1 h 以上——这正是"占住 runner 然后被取消"的来源。
3. **抓唯一可复现的红。** `tests/test_doc_claims.py::test_repo_root_is_discovered_and_holds_the_documents`
   - 现象：`assert PosixPath('/root/tp-agent/work/<other>') == PosixPath('/root/tp-agent/work/<this>')`。
   - 根因：该测试断言"发现的 checkout == 本测试文件所在工作树"，但 `doc_claims.repo_root()` 是从**被导入的包**向上找 `pyproject.toml`。工作树若继承了别的树的 editable install（未设 `PYTHONPATH=src`），包来自那棵树，发现结果是**正确的**，是断言前提不成立 → 误红。
   - 修复：身份断言只在"被导入的包确实属于本工作树"时执行（包含判定，不写死 parents 层数，兼容 `src/` 布局与 site-packages）；**与 checkout 无关的那半判据全部保留**——`_looks_like_checkout`、全部 `ROOT_MARKERS`、`MARINE_DOC`/`README` 必须存在，闸门不会因此变瞎。
4. **CI 侧根治"被取消 = 红"。** `Run test suite` 改为
   `python -m pytest -q -n auto --maxprocesses 8 --dist loadfile`；`pytest-xdist>=3.5` 进 `dev` extra；job 加 `timeout-minutes: 90`。
   - `--dist loadfile`：同一模块的所有用例落同一 worker，模块级 fixture 与 `artifacts/` 读者的可见性跟串行一致，扇出不会改变任何用例看到的东西。
   - `--maxprocesses 8`：自托管池核多，无界扇出会打爆内存；封顶是对"另一种丢作业方式"的保险。
   - `timeout-minutes`：把"无限占用、随时被取消"变成"到点显式失败"，失败原因可读。
5. **锁住判据。** 新增 `tests/test_ci_workflow_contract.py::test_the_suite_is_fanned_out_and_the_job_is_bounded`：解析 workflow 文本，要求测试步骤含 `-n auto` / `--maxprocesses` / `loadfile`、job 有 `timeout-minutes: <n>`、且 `pytest-xdist` 在 workflow 里被点名（防止"扇出命令有了但插件没装"）。

## 3. 计算结果（程序化取数，见 `docs/m4796/ci_duration_evidence.json`）

最近 25 次 run 的作业墙钟（分钟）：

| 指标 | 值 |
|---|---|
| 样本数 n | 25 |
| 最短 | 0.0（queued，未起跑） |
| 中位 | 49.7 |
| 最长 | 194.4 |
| 超过新上界 90 min 的 run 数 | 6 |
| `fa868b64` | 29.8，`cancelled`（owner 取消） |

本机全量套件（**CI 用的同一条命令** `pytest -q -n auto --maxprocesses 8 --dist loadfile`，`PYTHONPATH=src`，128 核共享机）：

| 口径 | 结果 |
|---|---|
| 收集用例总数 | 1388（当前 main 的规模，高于原单 1289） |
| 本轮改动直接相关的 4 个门禁模块 | **52 passed**（`test_ci_workflow_contract` / `test_doc_claims` / `test_doc_claim_audit` / `test_skip_gates_are_tracked`） |
| 全量运行进度 | 跑到 **98%** 时见 5 个 F，全部落在本单**未触碰**的求解器基准模块；本单预算内未等到收尾（尾部 2% 是重模块，机器 load average 240+，非本单造成） |
| 尾部区域抽查 | `test_ellipsoid_benchmark` + `test_exterior_neumann_added_mass` + `test_exterior_solver` + `test_finite_depth` + `test_finite_depth3d`：**72 passed, 1 xfailed** |

诚实声明：那 5 个 F 的**用例名未取到**（收尾时预算耗尽），因此不能断言它们与本单无关，只能说它们不在本单改动的文件里；
本单改动的两个测试模块与两个 docs 门禁模块均为绿，且均为本机实跑。

集群交叉验证（loong64 双节点）**未做成**：节点 `n315/n330` 的系统 python 缺 `scipy`/`matplotlib`，`pip install` 在离线环境失败，
因此本单的全量证据只有 x86 共享机一份。

## 4. 图

`docs/m4796/ci_wallclock.png`：最近 25 次作业墙钟柱图，按结论着色（绿 = success、红 = failure、橙 = cancelled），
蓝虚线为本次新加的 90 min 上界；`fa868b64` 标注为"owner 取消"。图直接由同目录 JSON 重绘（数据与图同源，无手抄）。

## 5. 注意点

1. **cancelled ≠ failed。** GitHub 把取消计红，但代码可能完全健康；先看 check-run 的 `conclusion` 与 annotation，否则会在仓里"修"一个不存在的 bug。
2. 扇出改变的是**调度**不是**语义**：靠 `--dist loadfile` 保证模块内串行；若有人改回 `--dist load`，模块级 fixture 与 `artifacts/` 读者可能在不同 worker 上互相看不见。
3. `-n auto --maxprocesses 8` 是有意封顶；无界扇出在 128 核自托管节点上会因内存/IO 抖动反而变慢或丢作业。
4. 尾部重模块的真实耗时未在本单全测（预算）；本单不承诺"CI 从此必绿"，承诺的是：作业有 90 min 上界、失败模式从"被取消"变成"超时失败"，且这两条被合同测试锁住。
5. 本机是 128 核共享机（load average 240+，多个工作树并发），全量串行/扇出计时受邻居影响，故本报告只把**相对结论**（扇出把墙钟由最长单模块决定）作为定性结论，不给绝对加速比。
6. 当前 main 已有 m4803 的 runner 变量：本修复在两种池上都成立（托管池 51.3 min 的作业同样受益于扇出），不需要随池切换而回退。

## 6. 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 本单实现 | 偏差 | <3%? |
|---|---|---|---|---|
| `fa868b64` 作业时长 29.8 min | API `started_at`/`completed_at` 之差 | 29.8 min（脚本复算） | 0.0%（同源复算） | ✅ 是 |
| 25 次 run 墙钟中位 49.7 min | 同一 JSON 逐条 `statistics.median` | 49.7 min | 0.0%（口径一致） | ✅ 是 |
| 25 次 run 墙钟最大 194.4 min | 同上 | 194.4 min | 0.0% | ✅ 是 |
| 行为判据：外来安装下 `test_repo_root_...` | 应 skip（保留其余判据） | skip，理由含来源路径 | 二值一致 | ✅ 是 |
| 行为判据：本树安装下同一测试 | 必须 pass | pass | 二值一致 | ✅ 是 |
| 行为判据：workflow 合同测试 | 扇出/限界缺失即红 | 缺任一项即红（负控见下） | 二值一致 | ✅ 是 |
| CI 墙钟缩短比例 | 无基准（修复后尚未有 CI 计时） | — | — | 无基准，原因：需合并后由 CI 实测，本单不预测 |
| 全量套件残留 5 个 F | 未取到用例名（预算） | 见 §3 诚实声明 | 不可判定 | ❌ 未验证，如实记录 |

负控（诚实记录）：把 workflow 里的 `-n auto` 删掉后，`test_the_suite_is_fanned_out_and_the_job_is_bounded` 变红；
恢复后转绿。即该判据不是恒真断言。