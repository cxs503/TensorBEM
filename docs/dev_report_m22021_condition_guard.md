# m22021 · CASE-BEM-LAPLACE-CYLINDER：条件数硬门限在 SVD 失效节点的误杀修复 + 记录复核

**锚点** `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** agent/m22021 · **任务单** m22021
**席位** verifier（tp-ver-6，自找活）

## ① 本单结论（先写给验收人）

| 项 | 状态 |
| --- | --- |
| 锚点解析基准复核（复势 W=U(z+a²/z)，N=128 预算） | **通过**：已提交记录 38/38 判据 ok，关键量最差 2.849e-04 ≪ 3% |
| 缺陷：`np.linalg.cond` 非有限时被当作"数值奇异"硬拒 | **已修**（本单，`solver._solve` / `multipole2d.fit_ring`） |
| 修复回归测试 | **已加 5 条**（`tests/test_conditioning.py` / `test_solver.py` / `test_multipole2d.py`），全绿 |
| 已提交 `results.json` 陈旧（38 判据、无 `provenance`，缺 near_wall floor/azimuth 两段） | **未闭合**：重生成作业 cj3853 本单递交后仍在队列（见 ⑤） |

## ② 怎么做的（方法与证据链）

1. 只读扫描：总账锚点 `CASE-BEM-LAPLACE-CYLINDER` 状态 `open`；仓库 `benchmarks/verified/case_bem_laplace_cylinder/` 有工件，
   `docs/CASE-BEM-LAPLACE-CYLINDER.md` 有渲染文档，但 `results.json` 的 `generated_at_utc=2026-11-14T19:08:10Z`、
   判据 38 条、**无 `provenance` 段**，`near_wall` 缺 floor/azimuth 两段 → 记录落后于模块（m20961/m21296 记的同一笔欠账）。
2. 追查重生成阻塞点：`docs/dev_report_m21296_cylinder_near_wall_keyerror.md` 记录 loongarch64 节点 `numpy.linalg.cond` 对
   良态矩阵返回 `inf`，被 `solver._solve`（`condition > _MAX_CONDITION` 硬拒）与 `multipole2d.fit_ring` 当作秩亏。
3. 独立复算同一批矩阵（本机 x86_64，绕过 guard 直接装配）：单层阵 cond 在 16/32/64/128 面板为
   9.81/19.17/38.11/76.11，**全部有限**；`fit_ring` 基阵 cond ≈ 1.0 → 节点上的 `inf` 是 LAPACK/SVD 路径缺陷，不是数学奇异。
4. 修复（最小、复用既有能力）：`conditioning.condition_number_with_fallback` — SVD 结果有限则原样返回（`source="svd"`），
   非有限则退回**既有** `estimate_condition_number(method="power_iteration")`（LU 反迭代，不碰 SVD）；
   长方形最小二乘基阵经 Gram 阵估计并用 `cond(A)=sqrt(cond(AᵀA))` 换算。`_solve` 与 `fit_ring` 改用它，
   结果 dataclass 新增 `condition_number_source` 如实记录估计来源；**阈值语义不变**（真奇异仍抛 `ValueError`）。
5. 顺带加固 `_power_iteration_smallest`：迭代中出现非有限 `sigma` 直接返回 0（→ cond=inf），不再产生 `nan` 除零警告。

## ③ 计算结果（程序化注入，零手抄）

已提交记录（db 判据 38/38 ok，预算 N=128）关键量：

| 关键量 | 面板 | 相对误差 | 是否 <3% |
| --- | --- | --- | --- |
| D 外部速度场 | 128 | 1.366e-07 | 是 |
| D 外部扰动势 | 128 | 3.578e-07 | 是 |
| N 物面速度 | 128 | 1.008e-04 | 是 |
| N 外部速度场 | 128 | 1.532e-04 | 是 |
| N Cp | 128 | 2.849e-04 | 是 |
| Γ 物面速度 | 128 | 2.131e-04 | 是 |
| Γ 外部速度场 | 128 | 1.481e-04 | 是 |

环量路线升力 vs Kutta-Joukowski 相对误差 `circulation_lift_matches_joukowski` = 4.014e-04。

修复行为证据（新增测试，本机实跑）：

| 场景 | 期望 | 实测 |
| --- | --- | --- |
| `np.linalg.cond → inf` + 32 面板 Dirichlet 解 | 解不变、`source="power_iteration"`、残差 <1e-10 | 通过 |
| `np.linalg.cond → inf` + 真奇异阵 | 仍判奇异（异常） | 通过 |
| `np.linalg.cond → inf` + `fit_ring`（32×5 基阵） | 系数与残差与 SVD 路径一致到 1e-12 | 通过 |

## ④ 结果图

![record](docs/figures/m22021_laplace_record.png)

`fig_laplace_record.png`：按已提交 `results.json` 三条路线（D/N/Γ）的误差-面板尺寸关系重绘（本单程序化生成，
非手抄）；圆柱场量云图由算例模块生成（`figures/field.png` 等），需完整重生成运行产出，本单队列未清（见 ⑤）。

## ⑤ 注意点（局限与未闭合项）

- **记录仍陈旧**：重生成命令
  `PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check`
  已作为 cj3853（8 核，stage_dir=本工作树）递交，排队 19 分钟未起跑；**未跑完就不把记录刷新算完成**。
- 节点缺陷只被"绕过"未被根治：本单让 guard 不再依赖 SVD 结果，但 `np.linalg.cond` 在 loongarch64 上返回 `inf` 的
  根因（LAPACK 构建 / 线程化 SVD 路径）仍属集群执行层问题，建议加节点数值自检探针。
- 数值不变性：修复只在 SVD 结果非有限时生效，x86_64 上所有既有数字保持不变（route D/N/Γ 未重算）。
- 圆柱离散体是内接正多边形，误差按实测报告；只验证势流算子链，不能当阻力预报。

## ⑥ 误差判定

| 关键量（N=128） | 相对误差 | <3% |
| --- | --- | --- |
| D 外部速度场 | 128 | 1.366e-07 | 是 |
| D 外部扰动势 | 128 | 3.578e-07 | 是 |
| N 物面速度 | 128 | 1.008e-04 | 是 |
| N 外部速度场 | 128 | 1.532e-04 | 是 |
| N Cp | 128 | 2.849e-04 | 是 |
| Γ 物面速度 | 128 | 2.131e-04 | 是 |
| Γ 外部速度场 | 128 | 1.481e-04 | 是 |

显式回答：**是**，已提交记录中全部关键量均 <3%（最差 2.849e-04，预算 3 级网格），
且实测收敛阶 ~2 与离散化一致，不是放宽阈值通过。
