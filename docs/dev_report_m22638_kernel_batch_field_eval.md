# CASE-BEM-LAPLACE-CYLINDER 第 8 轮：场求值器的目标轴批量化（记录重生成的真正瓶颈）

**锚点** CASE-BEM-LAPLACE-CYLINDER（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** agent/m22638
**总账状态核对**（开工时）：`open`（未被他单完成）

## ① 本轮问题：为什么前 7 轮的「记录重生成」一直没落地

第 7 轮（m22556）留下的欠账是：`benchmarks/verified/case_bem_laplace_cylinder/results.json`
仍是旧记录 —— 38 条判据、`provenance` 缺失、`figures/` 目录不存在，导致
`test_committed_record_was_rendered_from_the_current_case_module` 只能 `pytest.skip`。
本轮查明**这不是流程问题而是算力问题**（本机只准 <10 min / <8 核的单次测试，而实测
入口点单次运行超过 120 s 的执行窗口），因此把「重生成为什么贵」量化并直接拆掉其中最大的一块。

分阶段实测（本机，单进程；`case_cylinder_stream` 各 stage 单独调用）：

| stage | 改前 | 改后 |
| --- | --- | --- |
| `multipole_study()`（路线 M，2 族 × 3 层） | **73.16 s** | **5.99 s** |
| `multipole_level(128, uniform_flow)` | 20.05 s | 1.83 s |
| `route_dirichlet(32 / 64)` | 0.45 / 1.05 s | 不变 |
| `route_neumann(32 / 64)` | 0.44 / 1.04 s | 不变 |
| `route_circulation(32 / 64)` | 0.89 / 2.08 s | 不变 |
| `heading_study(32 / 64)` | 2.11 / 5.41 s | 不变 |
| `near_wall_study(ladder=(32,))` | 9.54 s | 不变（线性密度路径仍是标量） |
| `near_wall_study(ladder=(64,))` | 33.94 s | 不变（同上） |

定位：`tensorbem.solver.evaluate_exterior_field` 把
「每个目标点 × 每块面板」写成 python 双层循环，每次调用一个**标量**闭式核
（`integrate_*_panel*`，形状 (2,) 的入参、`math.log/math.atan`）。路线 M 在 N=128 时
每层 768 个目标 × 128 块面板 × 4 个核 = 39 万次标量调用，单层 20 s，无关于 BEM 求解本身
（同一层的稠密解只要 1.63 s）。

## ② 做法（最小、可验证）

1. `src/tensorbem/kernels.py`：**新增** 4 个批量化闭式核，目标轴向量化，标量版本一字未动：
   `integrate_single_layer_panel_many` / `integrate_double_layer_panel_many` /
   `integrate_single_layer_panel_gradient_many` / `integrate_double_layer_panel_gradient_many`，
   外加 `_log_distance_primitive_many`（保留标量版在 `|eta| <= 1e-14` 时的稳定分支）。
   特殊分支逐条镜像：对数原函数的小法向距离分支、双层势 `|eta| <= 1e-14 -> 0`、
   单层梯度 `eta == 0 -> across = 0`、两个梯度在面板端点 `r2 <= 1e-28` 的拒绝。
2. `src/tensorbem/solver.py`：`evaluate_exterior_field` 的**常值密度分支**与
   `evaluate_interior_field` 的势循环改为「一层面板一次 numpy 调用」，**每个目标点对面板
   的累加顺序仍是 j 递增**；线性密度（一阶矩）分支保持标量原样 —— 因为一阶矩核本轮未批量化，
   贸然替换会悄悄改变近壁研究的数值。
3. `tests/test_kernels_batch.py`（新增，20 条用例）：4 个核 × 4 组随机面板 × 48 目标
   与标量参考逐点比对（相对差 `<= 1e-14`）、面板共线目标的分支、端点拒绝、
   入参校验、以及线性密度标量路径对自身梯度的中心差分自检（钉住"未改动"这一事实）。

## ③ 结果

- **等价性**：60 组随机面板 × 40 目标 × 4 个核，批量与标量**逐位相同**（最大相对差 `0.0`）。
- **路线 M 数值不变**：`multipole_level(128, uniform_flow)['target_moment_relative_error']`
  = `3.5781652596433844e-07`，与已提交记录 `results.json` 中的同名值**完全一致**
  （该量对应解析矩 `A_1 = U a^2`，相对误差 3.58e-07 ≪ 3%）。
- **加速**：`multipole_study` 73.16 s → 5.99 s（12.2×）；微基准
  「128 面板 × 768 目标 × 单个梯度核」0.029 s。
- **测试**：`tests/test_kernels_batch.py` 20 passed；`tests/test_solver.py` 10 passed；
  `tests/test_exterior_solver.py` 12 passed；`tests/test_case_cylinder_stream.py`
  中与场求值相关的 27 条（外场/梯度/物面速度/镜像）passed、2 skipped（既有的几何跳过）。

## ④ 注意点与仍未完成的部分（honest）

- **记录重生成仍未落地**。本轮删掉的是路线 M 这块 73 s，剩下的最大项是
  `near_wall_study`（线性密度/一阶矩路径，仍是标量）：实测 `(32,)` 9.54 s、`(64,)` 33.94 s，
  按 4×/加倍的趋势 `(32,64,128)` 全阶梯约 175 s —— 完整入口点仍 > 120 s 的执行窗口，
  所以**本机仍不能一键重生成**，`provenance` 依旧缺失，那条门禁测试依旧 `skip`。
- 集群作业 `cj4269`（本轮递交，`--outdir benchmarks/verified/... --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check`，
  4 核 / 2400 s）在排队 60 分钟后仍未调度（队列 in-flight ~60），工件未取回。
  后续轮次应 `run_compute ps/fetch cj4269`，若已终态则直接落盘记录并把门禁由
  `skip` 改为 `fail`（替换文本见 `docs/dev_report_m22521_cylinder_record_gate_and_routeM_recheck.md` §5）。
- **下一次的直接线索**：把一阶矩的 4 个核（`integrate_*_panel_first_moment*`）按同样方式批量化，
  即可把 `near_wall_study` 从 ~175 s 压到秒级，完整重生成进入 120 s 窗口 —— 那时记录欠账、
  门禁硬化、图件提交可以一次做完（建议新条目 slug `CASE-BEM-LAPLACE-CYLINDER-REGEN-UNBLOCK`，
  验收：`near_wall_study()` 全阶梯 wall < 15 s 且一阶矩核批量/标量逐位等价、入口点 `--check` 在
  120 s 内 exit=0 且 `results.json` 带 `provenance` 与 ≥ 53 条判据）。
- 适用范围/局限：本轮的加速只作用于**常值密度**的场求值；近壁适用窗结论
  （1% 误差窗 2·h_max、线性密度在窗内反而差 ~1.2×）是把标量核与面板积分本身的性质，
  与本次的循环重写无关，未做任何阈值或口径变更。

## ⑤ 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 本轮实测 | 误差 | <3% |
| --- | --- | --- | --- | --- |
| 批量核 vs 标量核（4 核 × 60 面板 × 40 目标） | 标量闭式（同一解析式） | 逐位相同 | 0.0 | 是（≤1e-14 硬断言） |
| 路线 M 目标矩 `A_1/(U a^2)` | 解析 `A_1 = U a^2`（Lamb） | 3.5781652596433844e-07 相对误差 | 3.58e-07 | 是 |
| 场求值改变对已提交记录的影响 | 已提交 `results.json` 同名值 | 完全一致 | 0.0 | 是 |

**结论：本轮改动是纯性能改动，数值不变（逐位一致）；锚点 `CASE-BEM-LAPLACE-CYLINDER` 仍 `open`，未完成的是记录重生成本身。**