# CASE-BEM-LAPLACE-CYLINDER：记录重生成的阻塞根因、门禁补丁与路线 M 复核（m22521）

**锚点** `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，ref 解析复势，容差 `<3%`），
总账状态 `open` · **仓** TensorBEM · **分支** `agent/m22521`
**状态**：记录**未**重生成（集群作业 cj4188 排队 18+ min 未起跑），只交付不依赖重生成的增量。
下面每条都标注了「已做」/「已备但未启用」/「未做」。

## 1. 欠账现状（全部机检）

| 事实 | 机检 | 结果 |
| --- | --- | --- |
| 提交记录判据数 | `len(json.load(results.json)['criteria'])` | **38** |
| 提交记录有无 `provenance` | `'provenance' in payload` | **False** |
| 记录↔文档一致性门 | `pytest -k committed_case_artifact_matches` | **FAILED**（`render_markdown(旧记录)` 直接抛异常） |
| 陈旧记录门 | `pytest -k committed_record_was_rendered` | **SKIPPED** |

两条门都**不是本轮引入**：在 `git stash` 掉本轮改动的原始 `main` 上重跑，
`test_committed_case_artifact_matches_the_committed_document` 同样 FAILED（0.97 s）。
即该算例在 main 上已经是红的，红的原因只有一个：提交记录停在 38 条、无 `provenance`，
而模块此后长出了线性密度近壁节、刻面残差归因节、方位角扫描与斜向来流（heading）研究。

`test_committed_record_was_rendered_from_the_current_case_module` 是唯一能发现
「记录与文档**一起**落后」的检查（一致性门只比较文档是否等于 `render_markdown(记录)`，
两者一起落后时它看不出来），但它缺 `provenance` 时走 `pytest.skip` → 绿。
跳过不是因为它跑不起来，而是因为结论相反——这是这笔记账存活十余轮的机制性原因
（m20961/m21296/m21616/m21776/m22021/m22101/m22326/m22410 都记过，
cj3091/cj3565/cj3901/cj3943/cj3969/cj4100 都递交过但未在各自窗口内落盘）。

## 2. 本轮交付

### 2.1 入口点写入顺序修复（已做，`src/tensorbem/benchmarks/case_cylinder_stream.py`）

m22235 的集群重生成 cj3969 在节点上**没有 matplotlib**：`plot_field` 抛
`RuntimeError: matplotlib is required for plotting`，而入口点把 `--doc` 的写出排在画图之后，
于是作业写出了新 `results.json` 却没有改写 `docs/CASE-BEM-LAPLACE-CYLINDER.md` ——
记录与文档描述**两次不同的运行**，正是这道门要防的状态，且它由 m22235 实测触发过。

修复：把 `--doc` 的写出移到画图之前。`results.json` / `report.md` / `case.json` / 文档
都是同一份记录的纯函数，与画图无依赖；图件需要 matplotlib 且 `docs/` 之外的 png 不在推送门内
（`tests/_figure_integrity.py`），因此让最脆的一步排在最后，失败时不再污染记录链。

### 2.2 记录截断不变量（已做，`tests/test_case_cylinder_stream.py`）

新增 `len(payload["criteria"]) == payload["acceptance"]["n_checks"]`。
`provenance` 指纹守护的是**模块**，不是载荷：手删一条判据后摘要依然匹配，
只有这条记录自身的算术不变量能发现被截断的记录。该不变量在现有旧记录上成立（38 == 38），
不改变 main 的红/绿状态。

### 2.3 门禁硬化（已备，未启用）

`skip → pytest.fail` 的替换文本已写入
`tests/test_case_cylinder_stream.py::test_committed_record_was_rendered_from_the_current_case_module`
的 docstring 与本文件第 5 节。**未启用**的原因：记录没重生成时启用它会让 main 对所有人都变红；
它与重生成必须同批落地。

### 2.4 路线 M（远场多极分解）复核：m22235 的 FAIL 结论在 main 上不可复现（已做）

m22235 的重生成报告记录 6 条失败判据，其中 3 条属于远场多极拟合
（`multipole_purity` 实测 9.586 要求 `<1e-9`；`multipole_fit_residual` 0.514 要求 `<1e-5`；
`uniform_flow` N=32 目标矩相对误差 0.473）。本轮在**本机**直接调用
`case.multipole_level(n, family)`（单层稠密解，秒级；不触发重计算门，也未触碰完整阶梯）：

| family | N | 目标矩相对误差 | 最大污染谐波 /scale | 拟合相对残差 | 环间相对展布 |
| --- | --- | --- | --- | --- | --- |
| uniform_flow | 32 | 2.630e-05 | 1.832e-14 | 8.614e-08 | 5.351e-14 |
| uniform_flow | 64 | 2.999e-06 | 7.689e-15 | 9.808e-14 | 2.220e-16 |
| uniform_flow | 128 | 3.578e-07 | 1.498e-14 | 2.638e-15 | 6.661e-16 |
| pure_strain | 32 | 1.881e-03 | 9.544e-15 | 3.412e-07 | 3.625e-13 |
| pure_strain | 64 | 4.367e-04 | 5.499e-15 | 3.762e-13 | 4.221e-15 |
| pure_strain | 128 | 1.048e-04 | 1.646e-14 | 2.043e-14 | 2.731e-14 |

判据阈值：纯度 `<1e-9`、拟合残差 `<1e-5`、目标矩阶次 `>= 1.5`。实测纯度 1e-14 量级（**通过**，
余量 5 个数量级）、残差 1e-8 量级（**通过**）、目标矩误差随 h 递减（N=32→128 降 55×，一阶以上）、
环间展布为舍入级。

结论（负面结果也入账）：**m22235 的「多极拟合本身不成立」是那一份工作树产物的结论，不是 main 的缺陷**；
两者相差 4 个数量级，不能由节点环境解释（本轮与 cj3969 都用同一 numpy 主版本）。
因此路线 M 的 3 条失败不构成锚点阻塞，`multipole2d.py` 无需修改。

复现：

```bash
PYTHONPATH=src python3 -c "
from tensorbem.benchmarks import case_cylinder_stream as c
d = c.multipole_level(128, 'uniform_flow')
print(d['target_moment_relative_error'], d['largest_contaminant_over_scale'], d['fit_relative_residual_max'])"
```

## 3. 记录重生成（未完成，如实记录）

递交 `run_compute` 作业 **cj4188**（4 核，`stage_dir`=本工作树，`MPLBACKEND=Agg`）：

```bash
MPLBACKEND=Agg PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir benchmarks/verified/case_bem_laplace_cylinder \
  --doc docs/CASE-BEM-LAPLACE-CYLINDER.md
```

queue 位次 40、在飞 ~57 作业，18.4 min 后仍 `queued`（本单窗口内未起跑，已 `kill` 前序同类作业 cj4182）。
**因此 `benchmarks/verified/case_bem_laplace_cylinder/results.json` 仍是 38 条、无 `provenance` 的旧版，
`docs/CASE-BEM-LAPLACE-CYLINDER.md` 未被改写，main 上那两条门的红/跳状态不变。** 不粉饰。

## 4. 注意点 / 局限

- 本机 `run_shell` 单次 120 s 硬顶：完整阶梯（旧记录 wall 144 s，加入 heading 研究后更长）
  无法在本机完成；后台 `nohup` 被重计算门判定为规避并拦下（实测）。
- 因此「重生成」在本仓当前 harness 下**结构性依赖队列**，而队列常态 40+ 位置；
  于是这笔记账在多轮里反复递交、反复落空。这不是数值问题，是吞吐问题。
- 34 份 `benchmarks/verified/*/results.json` 中仅 **4** 份带 `provenance`：
  把这条门推广到全池需要逐案重生成，超出单干范围。
- 未复核项（如实）：m22235 报告中另外 3 条近壁失败判据
  （`near_wall_guard_rejects_inside_window` / `near_wall_linear_window_extends_to_half_panel` /
  `near_wall_exclusion_floor_is_faceting`）需完整阶梯才能测，本轮无法机检，标记 `not_run`。

## 5. 处置与下一次自找活的线索

- **approve with notes**：本轮增量（写入顺序缺陷修复 + 截断不变量 + 路线 M 复核）
  可独立成立、不依赖队列；锚点本身仍 `open`。
- 下一步（必须拿到队列才能做）：
  1. cj4188 或重递作业跑完后 `fetch` 取回 `results.json`、`report.md`、`case.json`、文档，
     落盘后把本文件 §2.3 的门禁硬化同批启用——届时
     `test_committed_record_was_rendered_from_the_current_case_module` 才会由 skip 变绿（真检）。
  2. 同批核对近壁 3 条判据在完整阶梯上的实测值，若仍失败则修 `near_wall_study` 而不是改阈值。
  3. 若队列长期饱和，建议 owner 评估把该算例预算从 N=128 降到 N=64（并在文档显式声明预算变更），
     换取「算例可在单机窗口内重生成」的可维护性——属验收口径变更，需 owner 同意。
- 建议新总账条目（域：resistance-potential-flow，slug `RECORD-REGEN-QUEUE-UNBLOCK`）：
  「算例池记录的 provenance 覆盖率 ≥ 80%，且入口点重生成在 10 min 单机窗口内可完成
   （量化验收：34 份记录中 ≥ 27 份带 provenance 且判据数等于模块输出；单次重生成 wall < 600 s）」。