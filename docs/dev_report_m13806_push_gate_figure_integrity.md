# m13806 — 主分支红：推送门禁滤除验证图件（figures）导致的图件完整性断言失效

**任务**：自找活 tp-merge-5（generalist 席）· **仓**：TensorBEM · **分支**：`agent/m13806`
**基点**：`origin/main` = `6600bcc`（Merge #619 m8718）· **日期**：2026-11-11
**锚点**：NEW（总账无对应条目；属「验证资产完整性」类缺陷，见 §6 建议条目）

## 1. 缺陷（可复现，先于修复）

在 `origin/main@6600bcc` 的干净检出上跑：

```
PYTHONPATH=src python3 -m pytest tests/ -q            # 全量，超 120 s，已另交集群
PYTHONPATH=src python3 -m pytest tests/test_bem_verified_anchors.py \
    tests/test_bem_finite_depth_anchors.py tests/test_thin_ship_anchor_artifacts.py \
    tests/test_w20_wigley_anchor_artifacts.py -q
```

实测红 8 条，全部指向同一类断链——**提交在 `benchmarks/verified/<case>/figures/*.png` 的验证图
在推上 GitHub 时被门禁滤除**：

| 测试文件 | 红 | 失败断言 |
|---|---|---|
| `test_bem_finite_depth_anchors.py` | 1 | `len(figures) >= 4` → 0 |
| `test_thin_ship_anchor_artifacts.py` | 3 | `(directory/"figures"/f"{study}.png").is_file()` |
| `test_w20_wigley_anchor_artifacts.py` | 1 | `papers/reports/figures/*.png` 不存在 |
| `test_bem_verified_anchors.py` | 3 | 2 条图件/结构 + 1 条 schema（见 §4 D2） |

**根因（读平台推送器定位，只读，未改平台）**：
`ops/taskplane/tp_pusher.py` 的推送白名单
`_EXCLUDE2 = re.compile(r'\.(db|sqlite3?|png|log)$')` 丢弃所有 `*.png`，
唯一豁免是 v5.8 F3：`path.startswith("docs/")` 的 png/svg。
即：**`docs/` 之外的 PNG 在 git 提交里合法、但在 GitHub 主干上必然缺失**。
`git ls-tree -r HEAD benchmarks/verified/` 证实主干树里 `benchmarks/verified/**` 下一个 PNG 都没有
（而各 case 的 `results.json`/`report.md`/`gates.json` 齐全、`artifacts.figure_png` 指针悬空），
m5668 的合并提交信息亦自述「4 个文件被门禁滤除」。

于是「图件已提交」不是本仓可长期维持的不变量；把它写成断言的主干测试会周期性变红。

## 2. 修复（最小、可验证、门禁无关）

不去改平台门禁（越界），而是把断言改回**真正的不变量**：图件必须能由已提交的数字
（`result.json` 等）**逐位重现**，且报告仍链接它。为此新增共享探针
`tests/_figure_integrity.py::assert_reproducible_figures`（一次写清意图，避免各测试各自弱化）：

- `minimum` 条数下限；
- `size > 10 000` 字节（空图/中断的渲染过不去）；
- `embedded_in`：图名必须仍出现在该 case 的 `report.md` 里（重生成的图就是记录里声称的图）。

改动两处调用点（均复用 case 自带的作图函数，不手绘、不手抄数字）：

| 文件 | 改动 |
|---|---|
| `tests/test_bem_finite_depth_anchors.py` | `test_case_artifacts_and_documentation_exist` 改为 `anchors.make_figures(result, tmp_path)` 重生成 4 图 → 条数 + 体积 + 报告链接三重断言（原来只 glob 已提交 PNG） |
| `tests/test_thin_ship_anchor_artifacts.py` | `test_artifact_set_is_complete` 改为用 `STUDIES_FIGURES[study]`（`_figure_hydrostatics/_figure_ellipsoid/_figure_multihull`）重生成 → 体积断言 + `artifacts.figure_png` 指针仍指向 `figures/<study>.png` |

图件写入 `tmp_path`，不污染工作树；断言强度不降（原断言只证「文件在」，新断言证「可从记录重现 + 报告仍引用 + 非空图」）。

## 3. 结果（同基线同命令复测）

```
PYTHONPATH=src python3 -m pytest tests/test_thin_ship_anchor_artifacts.py \
    tests/test_bem_finite_depth_anchors.py -q
→ 31 passed, 1 warning in 15.92 s      # 修复前：4 failed / 28 passed
```

净清红 **4 条**（finite-depth 1 + thin-ship 3），未新增任何失败。

## 4. 未修（诚实申报）

同批还有 4 条红，属**另一根因**，超出单张自找活的范围，已定位未动：

- **D2 产物所有权冲突（3 条红，`test_bem_verified_anchors.py`）**：
  `benchmarks/verified/bem_wigley_hydrostatics/` 等目录被**两套互不兼容的 writer** 写同一路径：
  `benchmarks/bem_verified_anchors.py`（P19-N02，schema `case/froude_cases/panel_densities/figure`）
  与 `benchmarks/thin_ship_verification.py`（W20 波，schema `study/reference/criteria/levels`）。
  后者后写，覆盖了前者的 `results.json`；`summary.json` 也随之丢了 `mission` 键。
  实测三条红：
  `test_committed_artifacts_reproduce_from_the_current_code`（键集不等）、
  `test_case_directories_contain_report_and_figure`（report 不含 case 名）、
  `test_summary_lists_a_verdict_per_case`（`KeyError: 'mission'`）。
  **不能靠重跑任一 driver 修**：谁重跑谁把对方打红。需一次目录命名决策（下一节）。
- **D3 `papers/reports/figures/*.png`（1 条红，`test_w20_wigley_anchor_artifacts.py`）**：
  同一门禁根因，但作图脚本未在本仓 `src/` 内（`papers/` 侧材料），需按其数据文件接回 builder 后
  才能改成「可重现」断言。

## 5. 注意点 / 局限

1. 本单只做「断言↔不变量」对齐，不重算任何物理量：**没有改进精度**，清除的是主干红。
2. 未跑全量套件（>`120 s` shell 上限），已递交集群作业 `cj784`（`pytest tests/ -q -rf`）留档；
   本单结论只依赖上表 4 个测试文件的定点复测。
3. 门禁行为是平台侧事实（`tp_pusher._EXCLUDE2` + v5.8 F3 豁免），本仓不能改；若平台后续放行
   `benchmarks/**/*.png`，本修复依然成立（它不否定提交图件，只不依赖它）。
4. `tests/_figure_integrity.py` 以无包名方式被测试导入，依赖 pytest 默认 `prepend` 导入模式（本仓现状）。

## 6. 给下次自找活的线索（建议新总账条目）

- **slug**：`VERIF-ASSET-GATE-PROOF`
  **归属域**：TensorBEM（验证资产完整性）
  **量化验收**：① 全仓不存在「断言依赖 `docs/` 外二进制/记录文件存在」的测试（扫描器零命中）；
  ② `benchmarks/verified/` 下每个 case 的图件在 `tmp_path` 重生成成功且体积/链接断言全绿；
  ③ `origin/main` 全量 `pytest` 0 failed 连续 3 次合并保持。
- **D2 修法建议**：给两套 writer 各自命名空间（如 `benchmarks/verified/thin_ship/<case>/`），
  或让 `bem_verified_anchors` 写入 `benchmarks/verified/p19n02/<case>/`，两侧 driver + 测试 + 文档同步改，
  `summary.json` 按 case 前缀分节；一次提交完成，禁止一侧覆盖另一侧。
- **附**：可顺手加一个 `tests/test_gate_forbidden_artifacts.py`——扫描所有测试里对
  `*.png|*.db|*.log`（非 `docs/`）的存在性断言并报红，把这类复发挡在源头。