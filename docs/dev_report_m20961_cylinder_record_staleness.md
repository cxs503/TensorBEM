# CASE-BEM-LAPLACE-CYLINDER：main 上的算例记录落后于算例代码（验证欠账，m20961）

**锚点** CASE-BEM-LAPLACE-CYLINDER（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** agent/m20961

## ① 问题（缺陷，非新功能）

main HEAD（`dd98f505`，Merge #942）上，锚点的**已提交记录**与**算例模块**不同步：

| 项 | 已提交记录 `benchmarks/verified/case_bem_laplace_cylinder/results.json` | main HEAD 的算例模块 |
| --- | --- | --- |
| `generated_at_utc` | `2026-11-14T19:08:10Z`（早于 #942 合并的 21:20 文件时间） | — |
| 判据条数 | **38** | `criteria()` 预置 **41** |
| `near_wall` 节 | `window/exclusion/guard/ladder/factors/levels/probe_directions/note` | 额外 `faceting` / `linear` / `attribution` 三节 |
| `level.sweep[*]` 键 | 4 个（仅常值密度） | 8 个（含 `*_linear`、`representation_error_*`） |
| `docs/CASE-BEM-LAPLACE-CYLINDER.md` | 无「一阶矩（线性密度）基」章节 | `render_markdown` 会输出该章 |

后果链：`src/tensorbem/solver.py:575` 的 `evaluate_exterior_field` docstring 明写
「`linear_density=True` 消除一阶限制……`CASE-BEM-LAPLACE-CYLINDER` 在 `near_wall` 里并排报告两种模式」，
但被指向的记录里**没有**这个研究 —— 这是「docstring 引用了不存在的证据」。
同时 3 条已合并判据（`near_wall_linear_window_extends_to_half_panel`、
`near_wall_linear_beats_constant_basis`、`near_wall_exclusion_floor_is_faceting`）在 main 的记录里
完全没有出现，等于**新判据从未在主线上被评估过**。

## ② 根因

`tests/test_case_cylinder_stream.py::test_committed_case_artifact_matches_the_committed_document`
只断言 `report.md` / `docs/...md` 等于 `render_markdown(results.json)`：它只检查两份产物互相一致，
不问「记录是否由当前代码生成」，代码-记录之间没有机器可见的链接。

而且这条测试在 main HEAD 上是**红**的，原因是另一个更硬的缺陷（见 ③0）：
`render_markdown` 引用未绑定变量 `linear_near`（`src/tensorbem/benchmarks/case_cylinder_stream.py:2111`，
`linear_near` 只在 `criteria()` 的局部作用域里定义过，第 1630 行），于是**任何**记录渲染都抛
`NameError: name 'linear_near' is not defined`。修掉该 NameError 之后再渲已提交记录，则抛
`KeyError: 'linear'` —— 因为记录里没有 `near_wall.linear` 节。两者叠加说明：
**main 上这个锚点的记录既渲染不出来，也无法被当前代码重新生成之外的方式救活，重生成是唯一出路。**

对照审计 `docs/audit_m20383_case_bem_laplace_cylinder.md`（L0 裁定）的失败模式
（「分支未合并即视为已交付」），这是同一族的第二种形态：「代码已合并即视为记录已更新」。

## ③ 本轮改动

0. **修复阻塞缺陷**：`render_markdown` 里 `linear_near` 未绑定（NameError，第 2111 行）——
   在函数内绑定 `linear_near = near["linear"]`（与 `criteria()` 同源），使渲染路径不再崩溃。
   复现：`PYTHONPATH=src python3 -c "from tensorbem.benchmarks import case_cylinder_stream as c;
   import json; c.render_markdown(json.load(open('benchmarks/verified/case_bem_laplace_cylinder/results.json')))"`
   —— 修前 NameError，修后 KeyError 'linear'（记录缺节，属 ④ 未完成项）。

1. `case_cylinder_stream.generator_provenance()`：记录产生该记录的源码 sha256
   （`benchmarks/case_cylinder_stream.py` 为受守护文件，`solver/kernels/geometry/multipole2d/
   lifting_cylinder/near_singular/lifting` 仅存档溯源），写入 `artifact["provenance"]["generator"]`。
2. `tests/test_case_cylinder_stream.py` 新增
   `test_committed_record_was_rendered_from_the_current_case_module`：毫秒级比对受守护文件摘要，
   代码改了而记录没重跑 → 红，并在断言消息里给出重跑命令。
   受守护文件之外的源码只记录不断言（避免无关改动把 CI 打红）。
3. 该测试对**当前**已提交记录显式 `skip` 并说明原因与重跑命令：记录生成于 provenance 机制存在之前，
   属**已知未偿还**项，不粉饰成 pass。

## ④ 未完成（如实声明）

**记录重生成未在本单内落地。** 完整算例（`--check`，N=32/64/128 + 环量扫描 + 多极 + 近壁三节）
在本机单核超过 120 s 前台时限、属重计算，已递交调度（`cj3091`，8 核，stage_dir=本工作树），
作业报 `succeeded` 但取回工件与已提交件逐位相同（`generated_at_utc` 未变），
即在节点上**未真正改写**记录文件——据此判定该次递交不可作为证据，本轮不把它写成「已重跑」。

重跑命令（唯一权威复现路径）：
```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir benchmarks/verified/case_bem_laplace_cylinder \
  --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check
```
重跑后应得：`overall=PASS`、41 条判据、`near_wall.linear`/`faceting`/`attribution` 三节、
`provenance.generator` 存在且与当前源码一致，且新增测试由 `skip` 转为 `pass`。

## ⑤ 误差判定

本单不产生新的数值结果，故无「vs 基准误差」可比（原因：记录未重生成，见 ④）。
已提交记录的既有判据（38 条，预算 N=128）全部 `<3%`（最差 `neumann_cp_relative_l2` 2.849e-04），
这一点不因本单改动而改变：改动只增记录字段与测试，不触碰求解数值路径。

## ⑥ 结论

- 缺陷确认：main 上的算例记录落后于算例代码，docstring 引用了记录中不存在的证据；
- 已修：代码→记录的可检测链接（provenance + 测试），并把现存欠账显式标成 skip 而非 pass；
- 未修完：记录重生成（需调度作业真正跑通），留作下一单的第一件事。
