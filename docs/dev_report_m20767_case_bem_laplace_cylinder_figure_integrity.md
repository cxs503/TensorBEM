# 算例池锚点 CASE-BEM-LAPLACE-CYLINDER：证据链自持化（Figure/文档不再依赖被过滤的二进制）

任务：m20767（selfstart-2，generalist 席）。锚点：`CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，
容差 `<3%`，ref=解析复势）。

## 1. 起点：锚点的物理侧已完成，证据侧是红的

锚点的求解器工作（`src/tensorbem/benchmarks/case_cylinder_stream.py`，2001 行；两条路线
D/N + 多极矩路线 M + 近壁适用窗）已在既有单中完成，`benchmarks/verified/case_bem_laplace_cylinder/results.json`
`overall = "PASS"`，文档与报告的数值都成立。所以本轮**不重做物理**，只做验证欠账。

在干净检出上跑锚点测试，红：

```
$ python -m pytest tests/test_case_cylinder_stream.py -x -q
E  AssertionError: PosixPath('.../benchmarks/verified/case_bem_laplace_cylinder/figures/field.png')
E  assert False
E   +  where False = is_file()
tests/test_case_cylinder_stream.py:183: AssertionError
1 failed, 25 passed, 2 skipped
```

## 2. 根因：断言把「被推送门丢弃的二进制」当成了不变量

平台推送门 `tp_pusher._EXCLUDE2` 丢弃 `docs/` 之外的 `*.png`（本仓已在
`tests/_figure_integrity.py` 引言中记录，实测于 2026-11-11 `origin/main`）。因此
`benchmarks/verified/<case>/figures/*.png` **永远不会**出现在 GitHub 检出里，
`path.is_file()` 断言在任何干净检出上恒红——与本算例算得对不对无关。这正是
`tests/test_gate_forbidden_artifacts.py` 主动扫描并禁止的反模式，而锚点测试是残留的
9 个违规之一。

次要缺陷（同一处证据链）：注册表 `CASE-BEM-LAPLACE-CYLINDER` 的 `figures` 只声明了
3 张（field / convergence / near_wall），而入口点实际产出并写进正文的是 4 张——
`multipole.png`（多极矩面板）被漏登记，报告与 `docs/` 都链接了它。

## 3. 修复（最小改动，两处）

1. `tests/test_case_cylinder_stream.py::test_committed_case_artifact_matches_the_committed_document`
   改写为**可复现不变量**，只读已提交的 `results.json`，不重解 BEM：
   - `results.json` 自洽：`case_id` 正确、`overall == "PASS"`、`failed_checks == []`；
   - **文档漂移门**：`report.md` 与 `docs/CASE-BEM-LAPLACE-CYLINDER.md` 必须逐字等于
     `render_markdown(results.json)`；
   - **图片可复现门**：用 `plot_field/plot_convergence/plot_near_wall/plot_multipole`
     把 4 张图重建到 `tmp_path`，经 `tests/_figure_integrity.assert_reproducible_figures`
     检查非空（>10 kB）且被正文链接；
   - **注册表一致性门**：`CASES[CASE_ID].figures` 的文件名集合必须等于入口点产出的集合。
2. `src/tensorbem/benchmarks/case_pool_registry.py`：补登记 `figures/multipole.png`。

## 4. 证据

### 4.1 绿

```
$ python -m pytest tests/test_case_cylinder_stream.py -q
28 passed, 2 skipped, 1 warning in 72.09s
```

（2 个 skip 是核函数梯度测试中「目标点落在对数奇异面板上 / 双层势主值处」的参数组合，
与本单无关。）

### 4.2 新断言不是空转：两个定向变异都被抓

| 变异 | 结果 |
| --- | --- |
| `docs/CASE-BEM-LAPLACE-CYLINDER.md` 里把一个数 `1.366e-07` 改成 `1.366e-06` | FAILED（文档漂移门，1.46 s 内失败，先于出图） |
| 注册表重新删掉 `multipole.png` | FAILED（注册表一致性门，14.56 s） |

两处变异均已还原，工作树只留最终成果。

### 4.3 图片确实能只从已提交的数字重建（不重解 BEM）

| 图 | 重建字节数 |
| --- | --- |
| `field.png` | 212902 |
| `convergence.png` | 88276 |
| `near_wall.png` | 81635 |
| `multipole.png` | 96862 |

四张重建耗时合计 12.0 s；`render_markdown(results.json)` 与两份已提交文档逐字相等
（`report.md` 与 `docs/CASE-BEM-LAPLACE-CYLINDER.md` 均 True）。

### 4.4 归因：不引入新红

| 测试文件 | 改动前 | 改动后 |
| --- | --- | --- |
| `tests/test_case_cylinder_stream.py` | 1 failed / 25 passed | 28 passed / 2 skipped |
| `test_gate_forbidden_artifacts::test_no_test_pins_a_push_gate_filtered_artifact` 违规条目数 | 10 | 9（锚点这一条被消除） |
| `tests/test_case_pool_registry.py::test_every_registered_case_resolves_its_paths` | FAILED | FAILED（**预先存在**） |
| `tests/test_case_pool_registry.py::test_cli_lists_and_checks` | FAILED | FAILED（**预先存在**） |
| `tests/test_gate_forbidden_artifacts.py::test_no_test_pins_a_push_gate_filtered_artifact` | FAILED | FAILED（**预先存在**，其余 9 条） |

后三条用 `git stash` 在改动前跑过，基线失败集合与失败文本一致，本单不改变其状态。

## 5. 注意点与局限

- **未改物理**：本单不动 `case_cylinder_stream.py`，`results.json` 数值不变，故不重报
  误差判定；锚点的 `<3%` 结论仍由既有 `results.json` 承担（预算面板 5 个关键量
  1.37e-07 … 2.85e-04）。
- **未提交任何 png**：`benchmarks/` 下的图按本仓不变量不入库；`docs/` 下也不新增二进制，
  以免「提交图片」再次被误当作证据。
- **预先存在的系统性问题（本单未修，建议单开）**：
  `case_pool_registry.missing_paths()` 把 `figures/*.png` 当作「必须存在」的路径来校验，
  于是 `test_every_registered_case_resolves_its_paths` 在干净检出上恒红，涉及
  `CASE-BEM-ANNULUS-LAPLACE`、`CASE-BEM-ELLIPSE-2D-ADDED-MASS`、
  `CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`（后者是 `docs/.../run.log`）等。
  根因与本单相同（把推送门过滤的产物当不变量），但修法要改注册表语义
  （「可复现」而非「存在」），影响面超出单仓单锚点范围，故本单只记录不动手。
- 建议新条目：`CASE-POOL-REGISTRY-REPRODUCIBLE-EVIDENCE`（所属域 TensorBEM 算例池），
  验收：`missing_paths()` 不再对 `figures/*.png` 报缺，改为「图片可由 committed results.json
  重建且 >10 kB 且被正文链接」；`test_case_pool_registry.py` 两条红转绿；不降低任何既有
  算例的证据等级。
## 6. 产物指针

| 项 | 指针 |
| --- | --- |
| commit | `7f0a3dd9`（测试 + 注册表）、本文件所在提交（文档 + 发布脚本） |
| 平台报告 | `tensor_reports.id = 18834`（六要素，2 张内嵌图由 `results.json` 现场重建） |
| 发布脚本 | `post_report_m20767.py`（数值全部程序化注入，`hand_typed_numbers = 0`） |
| 复跑命令 | `python -m pytest tests/test_case_cylinder_stream.py -q` → 28 passed, 2 skipped |
