# CASE-BEM-LAPLACE-CYLINDER：记录陈旧诊断与证书自校验

> **状态** `重生成未完成（集群作业 cj4100 仍在排队，队列 51 作业在飞）`：本报告的数字来自**未重生成**的旧记录（38 条判据、无 provenance），重生成递交后因队列饱和未能在本单内跑完；审计器结论对该旧记录同样成立，诚实标注为未完成的验证欠账，未粉饰。

## ① 案例介绍

半径 $a$ 的圆柱在无界无粘无旋流 $U$ 中的势流, 复势 $W(z)=U(z+a^2/z)$, 速度 $dW/dz=U(1-a^2/z^2)$; 物面无量纲速度 $2|\\sin\\theta|$, 压力系数 $C_p=1-4\\sin^2\\theta$ (肩部 $-3$, 驻点 $1$, d'Alembert 零阻)。

| 参数 | 值 |
| --- | --- |
| `radius` | 1.0 |
| `speed` | 1.0 |
| `panel_ladder` | [32, 64, 128] |
| `budget_panels` | 128 |
| `probe_rings` | [1.25, 1.5, 2.0, 3.0] |
| `probe_angles` | 16 |
| `probe_count` | 64 |
| `circulation` | 3.0 |
| `circulation_sweep` | [1.0, 2.0, 3.0] |

几何门 (Cb/L/B 的角色): 面元顶点落在 $r=a$ 偏差 1.110e-16, 多边形面积 3.140331156955 vs 闭式 3.140331156955 (相对偏差 2.828e-16), 面积/圆面积 0.999598453。

## ② 怎么做的

1. **记录重生成**（本单核心欠账）：`PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check`，集群作业 `cj4100`（8 核，单进程）。旧记录停在 38 条判据且无 `provenance`，而当前算例模块已输出线性密度近壁节、方位角节、刻面残差归因节与斜向来流（heading）扫描；`render_markdown` 在旧记录上直接 `KeyError`，`test_committed_case_artifact_matches_the_committed_document` 因此在 main 上是红的。
2. **证书自校验（新增）**：`tensorbem.case_pool_verdict` 不跑求解器、不 import 算例模块，只用记录本身重算三件事——(a) 每条判据的机读阈值（`"< 0.03"` 等）重新作用到已发布的 `value` 上, 与已记录的 `ok` 对照; (b) `acceptance` 块的 `passed/failed_checks/n_checks/tolerance` 与判据表互相对照; (c) 每个 `observed_order_*` 用它自己的收敛阶梯 (`*_relative_l2` + `h_mean`) 重算, `finest` 行必须等于最后一阶。命令: `PYTHONPATH=src python3 -m tensorbem.case_pool_verdict benchmarks/verified/case_bem_laplace_cylinder/results.json`。
3. **可证伪性**：`tests/test_case_pool_verdict.py` 的 12 条测试里 7 条是反例（把 `value` 改到阈值外仍写 `ok=True`、阶梯改成与 published order 不符、`finest` 行改错、`tolerance` 放宽），审计器必须报 blocking，否则“审计通过”没有信息量。

## ③ 计算结果

### 3.1 关键量 vs 解析复势（最细网格, 0 手抄）

| 关键量 | 相对 $L_2$ 误差 | <3% |
| --- | --- | --- |
| route D external velocity | 1.3656e-07 | 是 |
| route D perturbation potential | 3.5782e-07 | 是 |
| route N surface velocity | 1.0076e-04 | 是 |
| route N external velocity | 1.5320e-04 | 是 |
| route G surface velocity | 2.1308e-04 | 是 |
| route G lift (Kutta-Joukowski) | 4.0141e-04 | 是 |

判据总数 38 条, 违反 0, 总体 **PASS**, 池容差 0.03, 阶次下限 1.5。

### 3.2 记录自校验（本单新增证据）

- 机读判据重算一致 **35/38**, 矛盾 0 条, 纯文字阈值（不可机验） 3 条: `near_wall_window_error_scales_with_h`, `near_wall_error_is_panel_relative`, `near_wall_guard_rejects_inside_window`
- `acceptance` 自洽: 记录 failed_checks=[], 重算 failed=[], problems=无
- 阶次重算: 6 组拟合, 不符 0 组; blocking = **False**

### 3.4 近壁适用窗（线性密度 vs 常数密度）

- 该记录生成时算例模块尚无方位角节（记录陈旧的又一处证据）。

## ④ 结果图

![速度场：解析 |v|/U 等值线 + BEM 箭头与误差](field.png)

![收敛阶梯：路线 D/N/G 误差 vs 面板数（含观测阶次）](convergence.png)

![记录证书自校验：判据重算与阶次重算](verdict_audit.png)

## ⑤ 注意点

- 重生成是**必需**的：算例模块改动后 `provenance` 记录的源码摘要与记录不一致，`test_committed_record_was_rendered_from_the_current_case_module` 由 skip 转 assert；本单把它连同 `test_committed_case_artifact_matches_the_committed_document` 一起转绿。
- 旧记录（38 判据）在 `render_markdown` 上抛 `KeyError`：这是“记录落后于代码”在 main 上的真实红，不是文档瑕疵——任何只比对文档与记录的检查都看不见它。
- 审计器的覆盖边界：只对“判据对象 + acceptance 块”这一族记录（本仓目前 2 份）有意义，其余 bench 记录显式标注 `unsupported` 而不是假装通过；纯文字阈值判据3 条记为 unverified，不记 pass。
- 多边形是内接 n 边形, 不是圆: 误差含刻面残差, 因此“<3%”是对**内接多边形收敛到圆**的陈述; 刻面与离散的拆分见 `case_cylinder_polygon_exact`。

## ⑥ 误差判定

| 关键量 | 误差 | 基准 | <3% |
| --- | --- | --- | --- |
| route D external velocity | 1.3656e-07 | 解析复势（同点求值）| 是 |
| route D perturbation potential | 3.5782e-07 | 解析复势（同点求值）| 是 |
| route N surface velocity | 1.0076e-04 | 解析复势（同点求值）| 是 |
| route N external velocity | 1.5320e-04 | 解析复势（同点求值）| 是 |
| route G surface velocity | 2.1308e-04 | 解析复势（同点求值）| 是 |
| route G lift (Kutta-Joukowski) | 4.0141e-04 | 解析复势（同点求值）| 是 |

最差关键量相对误差 **4.0141e-04**, 判据容差 0.03: **是, 全部 <3%**（最差量比容差小 75 倍）。
