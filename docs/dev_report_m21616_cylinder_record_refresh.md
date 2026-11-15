# CASE-BEM-LAPLACE-CYLINDER：记录陈旧与方位角守门测试失效（m21616）

**锚点** CASE-BEM-LAPLACE-CYLINDER（算例池·圆柱绕势流速度场，ref 解析复势，容差 <3%）· **仓** TensorBEM · **分支** agent/m21616

## ① 本轮定位到什么（都是可复现的欠账，不是新功能）

开工先核对总账：`feature_ledger.slug='CASE-BEM-LAPLACE-CYLINDER'` 仍为 `open`，
算例模块 `src/tensorbem/benchmarks/case_cylinder_stream.py`（3366 行）已含路线 D/N/Γ/M、
常值/一阶矩两种密度、刻面残差归因、方位角扫描与远场多极展开，并在 `criteria()` 里
预置了 **44 条**判据（含 2 条对照解，`--no-controls` 时 42 条）。数值侧（上一轮记录）
最差误差 2.849e-04，远优于 3% 容差。

但仓库内的**证据链与测试有两处脱节**，本轮把它们钉死：

### 1.1 已提交记录落后于代码（验证欠账）

`benchmarks/verified/case_bem_laplace_cylinder/results.json`：

| 项 | 已提交记录 | 当前模块 |
| --- | --- | --- |
| `criteria` 条数 | **38** | **44**（+一阶矩近壁 2 条、刻面 1 条、方位角 4 条… 见 `criteria()`） |
| `provenance.generator` | **缺失**（键不存在） | `generator_provenance()` 已提供 8 个源文件 sha256 + combined |
| 缺失的数据块 | `near_wall.linear`、`near_wall.faceting`、`near_wall.azimuth` | 全部计算并输出 |

后果：`tests/test_case_cylinder_stream.py::test_committed_record_was_rendered_from_the_current_case_module`
只能 `skip`（记录没有 provenance，无法机检 staleness），而
`test_committed_case_artifact_matches_the_committed_document` 在干净检出上 **红**——
`docs/CASE-BEM-LAPLACE-CYLINDER.md` 与 `report.md` 是按陈旧记录渲染的，与
`render_markdown(results.json)` 已不一致。也就是说：模块 docstring 里「近壁可用窗」「方位角
无关性」指向的证据并不在已提交记录中。

本地复现（<120 s，不需集群）：

```
PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py -q -k "not quick_ladder"
# 2 failed, 27 passed, 3 skipped
```

### 1.2 方位角守门测试打错入口（测试失效）

`test_near_wall_azimuth_sweep_is_measured_not_midpoint_only` 报
`KeyError: 'midpoint_cross_check'`。

根因不是算例缺功能，而是 m21112 的重构：帧一致性交叉检验需要「径向阶梯」的中点列
（`near_wall["window"]["levels"]`）才能比较，重构把这段从 `near_wall_azimuth_study()`
上移到

```python
attach_azimuth_study(window, radius, speed, ladder)   # azimuth["midpoint_cross_check"] = {...}
```

并由 `near_wall_study()` 内部调用（`near_wall["azimuth"]`）。测试仍直接调
`near_wall_azimuth_study()`，拿到的是**没有帧比较**的扫描结果，于是本来用于守门的断言
（`max_relative_difference <= NEAR_WALL_AZIMUTH_FRAME_TOL = 0.10`）从未真正执行过——
判据 `near_wall_azimuth_midpoint_cross_check` 在 `criteria()` 里存在，但测试路径绕开了它。

## ② 改了什么（最小改动）

`tests/test_case_cylinder_stream.py`：把入口改为模块真实使用的公共路径

```python
near = case.near_wall_study(ladder=(32, 64))   # 内部调用 attach_azimuth_study
azimuth = near["azimuth"]                       # 含 midpoint_cross_check
```

其余 10 条断言逐字未动（窗内 1% 无关性、排除区顶点惩罚 ≥2、线性基不得宣称加宽窗口、
跨面板取样代表全面板最大值）。这是**修复测试**，不是放宽测试：断言强度不变，改动只是让它
走判据真实所在的入口。

验证：

```
PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py -q -k azimuth
# 1 passed in 55.04 s  （此前 KeyError 立即失败）
```

## ③ 记录刷新（已递交调度，未在本单内落地）

刷新整份记录要跑完整阶梯（N=32/64/128 × 常值/一阶矩 × 常值密度近壁 + 刻面 + 方位角 +
环量扫描 + 对照解），本地单次 >10 min，按容量纪律递交计算门：

```
PYTHONPATH=src MPLBACKEND=Agg python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir benchmarks/verified/case_bem_laplace_cylinder \
  --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check
# run_compute submit -> cj3565（4 核，stage_dir=本工作树），完成为止仍在排队
```

作业完成后应满足：`overall=PASS`、`acceptance.n_checks=44`、`failed_checks=[]`、
`provenance.generator.files["benchmarks/case_cylinder_stream.py"]` 与当前源码一致、
`docs/CASE-BEM-LAPLACE-CYLINDER.md` 与 `report.md` 均由新记录渲染（含一阶矩近壁表、
刻面归因、方位角表）。届时 staleness 测试 `skip → pass`，另一条从红转绿。

### ③.1 调度结果（诚实记录）

`cj3565` 在 n315 上 **终态 rc=0**。`--check` 的语义是「任一预声明判据不满足即非零退出」，
因此 rc=0 本身就是一条机检证据：**当前模块的 44 条判据在完整阶梯（N=32/64/128 全路线）
上无一失败**。但取回工件（fetch，63.9 MB tar）里 `benchmarks/verified/case_bem_laplace_cylinder/`
的三个文件仍是作业前的旧记录（mtime 与 `generated_at_utc=2026-11-14T19:08:10Z` 均未变，
`criteria` 仍 38 条、仍无 `provenance`），只有 `.pytest_cache` 等未跟踪路径被刷新——即**作业
写入的跟踪文件没有回到取回包里**。所以「判据全过」有证据，「记录已刷新」**没有**：
本单不宣称记录已刷新，落地刷新需另派一单（`run_compute submit` + 确认取回路径）。

## ④ 未完成 / 风险

- **记录刷新未落地**（cj3565 排队 51 min 未开跑）。因此本单提交的是测试缺陷修复 +
  欠账定位，**不是**「锚点证据已刷新」；`docs/CASE-BEM-LAPLACE-CYLINDER.md` 与
  `results.json` 仍是旧内容，`overall=PASS` 仍只覆盖 38 条判据。诚实标注，不粉饰。
- 记录内 `criteria` 条数没有独立守门项：provenance 摘要只能保证「模块没变」，不能保证
  「记录没被手改小」。若后续再有人手工裁剪记录，staleness 测试也发现不了；建议增加
  「记录条数 == 记录自述 n_checks == 模块在 quick 阶梯上的条数」的一致性测试。
- 方位角扫描的跨面板取样（`NEAR_WALL_AZIMUTH_PANELS=6` 条面板）在测试里只对 N=32 的
  窗内一点做了「抽样 ≥ 0.5×全面板最大值」的比较，未对三层网格逐level 做；如需更强结论，
  应对每层都做全面板最大值对照（成本 ×N/6）。

## ⑤ 给下次自找活的线索

1. 先 `run_compute ps cj3565`；若已终态，取回产物落地并跑
   `pytest -q -k "committed"`，把本条欠账一次结清（这是锚点验收的最后一块）。
2. 上面 ④ 第 2 条（记录条数一致性守门）是低成本、可机检的真缺口。
3. 同类欠账可能在别的算例池条目上重复出现：全仓 grep 一次
   `"provenance" not in payload` 与 `pytest.mark.skip` 的记录相关测试，看还有哪些算例
   的 committed record 缺 provenance。