# 审计：CASE-BEM-LAPLACE-CYLINDER 记录重生成的实测结果（m22235）

**锚点** CASE-BEM-LAPLACE-CYLINDER（算例池·圆柱绕势流速度场，ref 解析复势，容差 <3%）· **仓** TensorBEM · **分支** agent/m22235
**计算作业** `cj3969`（node n315，8 核，rc=0，wall 351.6 s）· **工件** `runs/m22235/cj3969_results.json`（sha256 `b8e4c1d37d6e01ed29c12ef8705b89e88cca2ab48df83555973aa002156dce6e`）

## 1. 案例介绍

均匀来流 U=1 绕半径 a=1 圆柱的二维势流，复势 W(z)=U(z+a^2/z)，物面离散为内接正 N 边形
（N=32/64/128，预算 N=128）。算例模块 `src/tensorbem/benchmarks/case_cylinder_stream.py` 预置路线：
D（解析扰动势作 Dirichlet 数据 → 外域势/梯度）、N（零法向流动 → 物面速度/Cp）、
G（给定环量旋转圆柱 → 升力 vs Kutta-Joukowski）、M（远场多极谐波分解），以及近壁可用窗
（常值/一阶矩密度、刻面归因、方位角扫描）。

## 2. 怎么做的

1. 认领总账：`feature_ledger.slug='CASE-BEM-LAPLACE-CYLINDER'` 状态 `open`，本单认领。
2. 机检欠账：把**已提交记录**喂给当前 `criteria()` 立即 `KeyError: 'floor'`
   （记录里没有 `near_wall.floor` 节）——记录落后于代码，可复现。
3. 完整阶梯重生成按容量纪律递交计算门（本机 120 s 硬顶）：
   `run_compute submit repo=TensorBEM stage_dir=本工作树`，cmd =
   `PYTHONPATH=src MPLBACKEND=Agg python3 -m tensorbem.benchmarks.case_cylinder_stream
   --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check`
   → 作业 `cj3969`（n315）。
4. 取回：作业目录 tar `/srv/tensor/agent/cluster/fetch/cj3969_n315.tar.gz` 内
   `cj3969/repo/benchmarks/verified/case_bem_laplace_cylinder/results.json` 逐位拷入
   `runs/m22235/cj3969_results.json`。前两轮（cj3091/cj3565）「取回件仍是旧记录」的问题本轮
   **没有复现**：该文件确实是本次运行产物（generated_at_utc=
2026-11-15T18:54:08Z，`provenance.generator` 的受守护文件摘要与当前源码一致）。

## 3. 计算结果：重生成之后**不是全绿**

`overall = FAIL`，判据 49 条，失败 6 条（已提交记录只有 38 条且无 provenance）。

| 失败判据 | 实测值 | 要求 |
| --- | --- | --- |
| `near_wall_guard_rejects_inside_window` | false | "True on every level" |
| `near_wall_linear_window_extends_to_half_panel` | {"usable_clearance_over_h_max": 2.0, "levels": [{"panels": 32, "h_max_over_radius": 0.19603428065912168, "max_relative_velocity_er | "<= 0.5 at <= 0.01 error" |
| `near_wall_exclusion_floor_is_faceting` | {"circle_referenced": 0.12614820457455073, "surface_referenced": 0.13063732547576687, "faceting_offset_over_h_max": 0.006136231189 | "surface_referenced <= circle_referenced / 3" |
| `multipole_pure_strain_moment_observed_order` | [-1.0629542040393827, 4.807929320086197] | ">= 1.5" |
| `multipole_purity` | 9.585580445759657 | "< 1e-09" |
| `multipole_fit_residual` | 0.5141655741824956 | "< 1e-05" |

对照：**锚点本身的验收量全部远优于 3%**（N=128 预算面板）：

| 关键量 | 相对误差 | 容差 | 是否 <3% |
| --- | --- | --- | --- |
| 路线 D 外部速度 L2 | 1.3656e-07 | 3% | 是 |
| 路线 N 物面速度 L2 | 1.0076e-04 | 3% | 是 |
| 路线 N Cp L2 | 2.8492e-04 | 3% | 是 |
| 路线 G 物面速度 L2 | 2.1308e-04 | 3% | 是 |
| 路线 G 升力相对误差 | 4.0141e-04 | 3% | 是 |

即圆柱绕势流**速度场**这条验收标准是满足的（最差 4.01e-04），但模块自己预置的 6 条辅助判据
在完整阶梯上不满足——本轮如实把 `overall=FAIL` 记入证据工件，不改写记录、不宣称锚点已完成。

## 4. 结果图

![criteria](fig_criteria.png)

左：49 条判据逐条通过/失败（红=失败）；右：锚点五个关键量 vs 3% 容差线（对数轴）。

![multipole](fig_multipole.png)

远场多极（路线 M）实测：目标矩相对误差与最大污染谐波随 h_mean/a。污染本应为舍入级（<1e-9），
实测达 O(1)，拟合相对残差约 0.5——**拟合本身不成立**。

| 族 | N | h_mean/a | 目标矩相对误差 | 最大污染谐波 | 拟合相对残差 |
| --- | --- | --- | --- | --- | --- |
| uniform_flow | 32 | 1.9603e-01 | 4.7331e-01 | 8.8294e+00 | 5.1417e-01 |
| uniform_flow | 64 | 9.8135e-02 | 7.8588e-02 | 4.7854e+00 | 3.3273e-01 |
| uniform_flow | 128 | 4.9082e-02 | 2.0751e-02 | 9.5856e+00 | 4.6277e-01 |
| pure_strain | 32 | 1.9603e-01 | 2.2396e-01 | 2.0397e+00 | 3.9761e-01 |
| pure_strain | 64 | 9.8135e-02 | 4.6729e-01 | 1.0791e+00 | 4.7066e-01 |
| pure_strain | 128 | 4.9082e-02 | 1.6707e-02 | 6.7430e-01 | 1.1621e-01 |

## 5. 注意点（踩到的坑与局限）

- **节点上没有 matplotlib**：`plot_field` 抛 `RuntimeError: matplotlib is required for plotting`，
  作业在写 doc/图件前中止。因此本次 `results.json / report.md / case.json` 是完整的新记录，
  但 `docs/CASE-BEM-LAPLACE-CYLINDER.md` 与 `figures/` 未被该作业改写（doc 仍按旧记录渲染）。
- **`rc=0` 不能当「判据全过」**：`main()` 里 `--check` 的退出码在绘图异常时根本走不到；
  历史上「作业 rc=0」被误读过。本单改为直接读 `results.json` 的 `acceptance.failed_checks`。
- 路线 M 的失败在 N=32 就存在（目标矩误差 0.47），与加密无关：是取样/拟合实现缺陷，不是收敛慢。
- 近壁三族失败是**测量结论不达标**，阈值是预置判据的一部分，不能用调阈值消掉。

## 6. 误差判定（显式回答）

锚点关键量 vs 解析复势：路线 D 外部速度 1.37e-07、路线 N 物面速度 1.01e-04、路线 G 升力 vs
Kutta-Joukowski 4.01e-04——**逐项均 <3%**。但「算例是否可判通过」的答案是 **否**：49 条预置判据
中 6 条失败，故本单结论为 **honest FAIL**：锚点未完成，已提交记录保持原样、不被本单覆盖。

## 7. 给下一单的线索

1. 修路线 M 取样/拟合（先看 N=32 的 uniform_flow 族：目标矩 0.527、污染 8.8、残差 0.51，是坐标/
   归一化错配而非精度）；修好后 3 条 multipole_* 判据应转绿。
2. 修近壁三族：`minimum_clearance_factor` 守门未拒绝窗内点（`near_wall_guard_rejects_inside_window`
   为 False），以及一阶矩 1% 窗、刻面地板两条测量判据。
3. 给算例加 `--no-figures` 降级开关，使 `--check` 能在无 matplotlib 的节点上端到端跑完并写 doc。
4. 记录条数一致性守门（records 条数 == 模块判据条数）已在测试里提出方案，建议随后续单落地。
