# CASE-BEM-LAPLACE-CYLINDER 第 4 轮：`near_wall_study` KeyError 阻塞修复 + loong64 `cond=inf` 阻塞记录

**锚点** `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** agent/m21296
**任务单** m21296（selfstart 自主工时，generalist 席）

## ① 本轮结论（先写给验收人）

| 项 | 状态 |
| --- | --- |
| 已提交记录 `results.json` 陈旧（38 判据、无 `provenance`） | **未闭合** —— 记录重生成被下述缺陷 ② 阻塞 |
| `near_wall_study()` 崩溃（`KeyError: 'h_max_over_radius'`） | **已修**（本单） |
| 回归测试 | **已加**：`tests/test_case_cylinder_stream.py::test_near_wall_half_panel_rows_carry_the_panel_size` |
| 集群节点 `numpy.linalg.cond` 返回 `inf` 导致算例跑不完 | **已定位、未修**（外部环境缺陷，见 ③） |
| 解析基准（复势）复核 | 复核通过，最差误差 2.85e-04 ≪ 3%（既有记录数字，见 ④） |

## ② 缺陷一（已修）：算例模块在 `main` 上无法跑完

`near_wall_study()` 构造半面板比较行时漏了面板尺度：

```python
linear_half = [{"panels": level["panels"], "max_relative_velocity_error": ...} for level in levels]
...
half_h = [level["h_max_over_radius"] for level in linear_half]   # KeyError
```

后果：`run_case()` 在 `artifact["near_wall"] = near_wall_study(...)` 处抛
`KeyError: 'h_max_over_radius'`，**整个算例（含既有 38 条判据）无法重生成**，
这也是自 m20961 起"记录陈旧"一直没被刷新的直接原因。
修复：`linear_half` / `constant_half` 两处行字典补 `"h_max_over_radius"`（6 行改动，见 `git diff`）。
修后集群复跑 `near_wall_study` 输出 `near_wall ok`（作业 cj3333 诊断段）。

## ③ 缺陷二（未修，阻塞记录重生成）：节点 `np.linalg.cond` 返回 `inf`

| 作业 | 节点 | 环境 | 失败点 |
| --- | --- | --- | --- |
| cj3321 | n315 | `OMP_NUM_THREADS=8` | `solver._solve`：`single-layer matrix numerically singular (condition number inf)` |
| cj3333 | n315 | `OMP_NUM_THREADS=8` | `multipole2d.fit_ring`：`harmonic basis numerically rank deficient (condition number inf > 1e10)` |
| cj3346 | n315 | `OMP_NUM_THREADS=1` | 同上 `fit_ring`（单线程仍复现） |

对同一节点同一源码、绕开 guard 直接调用装配函数，矩阵**完全正常**：

```
16 finite True cond 9.808948675692804  maxabs 0.20663127822045585
32 finite True cond 19.16922421394064  maxabs 0.12529101132710493
64 finite True cond 38.11334353855526  maxabs 0.07352830691639846
128 finite True cond 76.11357941479741 maxabs 0.042187545056315875
```

本机（x86_64, NumPy 2.4.6，与节点同版本）同一调用：`fit_ring` 正常，`cond(128 panels) = 76.11357941479734`。
结论：**节点 loongarch64 上 `numpy.linalg.cond`（SVD）会给出 `inf`，而被测矩阵条件数实际 ~1（`fit_ring`）
与 ~76（128 面板单层阵）**。把 `cond` 当作硬门限的 guard（`solver._solve`、`multipole2d.fit_ring`）
因此把正常问题判死，算例在集群上永远跑不到 `results.json` 落盘。
既有提交记录（`generated_at_utc=2026-11-14T19:08:10Z`，`machine=loongarch64`，`multipole` 段
`condition_number=1.0015`）说明该 guard 在更早的节点/线程配置下能偶然通过 —— 这是**间歇性**节点数值缺陷，
不是算例或求解器的数学缺陷。

建议（未在本单实施，属跨仓共性机制、需独立验收）：
1. `multipole2d.fit_ring` 与 `solver._solve` 的秩判据改为**不依赖 SVD**的量（如 Gram 阵 Cholesky/LU 的
   pivots + 残差），或对 `cond` 结果加 `isfinite` 兜底后改用残差判定；
2. 或在集群执行层给 numpy 关掉可疑的多线程 LAPACK 路径并加节点数值自检（提交前跑一个小矩阵 SVD 探针）。

**诚实边界**：本单没有把 `results.json` 重生成，锚点的"记录新鲜度"欠账仍在。原因就是 ③：
唯一可用的计算面（外协方 loong64 双节点）上算例必然在 `fit_ring` guard 处中止。本机单条命令
120 s 超时 + 重计算门禁，无法在本机完成 145 s 的全量重生成。

## ④ 解析基准复核（用既有记录，未新算）

| 关键量 | 记录值 | 解析复势基准 | 相对误差 | <3% |
| --- | --- | --- | --- | --- |
| 路线 D 外部速度 $L_2$ (N=128) | 1.366e-07 | $\mathrm{d}W/\mathrm{d}z$ | 1.37e-07 | 是 |
| 路线 D 扰动势 $L_2$ | 3.578e-07 | $Ua^2\cos\theta/r$ | 3.58e-07 | 是 |
| 路线 N 物面速度 $L_2$ | 1.008e-04 | $2U|\sin\theta|$ | 1.01e-04 | 是 |
| 路线 N 外部速度 $L_2$ | 1.532e-04 | $\mathrm{d}W/\mathrm{d}z$ | 1.53e-04 | 是 |
| 路线 N $C_p$ $L_2$ | 2.849e-04 | $1-4\sin^2\theta$ | 2.85e-04 | 是 |
| 独立多极（无面板）复核速度 $L_2$ | 5.34e-16 | 同上 | 5.3e-16 | 是 |

独立复核脚本 `docs/verify_m20303_independent.py` 在节点上跑通（`INDEP_RC=0`，cj3321），
`route_metrics_committed_vs_recomputed_max_abs_diff = 2.6e-17`，
`near_wall_committed_vs_recomputed_max_abs_diff = 2.5e-15` —— 已提交记录的误差指标可被**独立重算复现**，
不是自证。

## ⑤ 注意点 / 局限

- 本单修的是**让算例能跑完**的崩溃点；真正把 41 条判据写进记录仍待 ③ 解决后重跑。
- 节点 `cond=inf` 是间歇的（cj3321 落在 `_solve`，cj3333/cj3346 落在 `fit_ring`），单次复现不能证明
  特定线程数触发；报告已按"观察到的事实 + 本机对照"表述。
- 测试 `test_exterior_field_refuses_points_inside_the_body_or_on_the_boundary` 在节点上失败：
  探针半径 0.9954 相对 32 边形内切半径 0.9952 只有 2e-4 余量，节点浮点差异把"刚体外"判成"体内"，
  报错信息因此不是预期的 `too close`。这是**测试余量过窄**（建议把该探针抬高 ~1e-2），不是守卫逻辑错误。

## ⑥ 复现

```bash
# 崩溃复现（修前）
PYTHONPATH=src python3 -c "from tensorbem.benchmarks import case_cylinder_stream as c; c.near_wall_study(1.0,1.0)"
# 修复验证
PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py -q -k half_panel_rows
# 节点数值缺陷复现
PYTHONPATH=src python3 -c "import numpy as np; from tensorbem.geometry import regular_polygon; from tensorbem import solver; m=regular_polygon(128,1.0); s,_=solver.assemble_matrices(m,reference_length=2.0); print(np.linalg.cond(s))"
```
