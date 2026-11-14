# BUG-BEM-NUMPY-VERSION-DRIFT：把「跨环境 B33 差 2 倍」定位到线性求解阶段

**锚点**：CASE-BEM-HULME-HEMISPHERE（算例池·Hulme 浮半球辐射/绕射，容差 <5%）
**任务**：m20910（自找活，proposer 席）
**续接**：m20515 的 `BUG-BEM-NUMPY-VERSION-DRIFT` 与 `AUDIT-BEM-4096-PANEL-BREAKDOWN` 两条建议条目

## ① 问题

同一 commit、同一 1024 面元网格（顶点/面积/形心逐位相同）、同一 ka=2：

| 环境 | A11 | B11 | A33 | B33 | 与 Hulme 参考比 |
|---|---|---|---|---|---|
| 控制面 numpy 2.4.6 | 0.249848 | 0.340432 | 0.386337 | 0.104249 | B33 +1.15% |
| 集群节点 numpy 2.0.1 | 0.235658 | 0.337733 | 0.366607 | 0.050792 | B33 **−50.7%** |

两行都报告了相同的网格体积与相同的条件数（2.398615400620687），因此「网格/分辨率不同」已被排除。
上一轮只能把分歧笼统归到「精度后处理或 Green 函数求值」。本单要回答的是：**四个阶段中的哪一个**。

## ② 方法

`scripts/hulme_cell_fingerprint.py`：把一格 BEM 计算拆段记录 fingerprint——
网格（顶点/面积/形心的 10 位取整 sha256）、算子（单层/双层核/CHIEF 行/系统矩阵的范数与 sha256）、
求解（解向量、**相对残差**、实际求解系统的奇异值谱与 `lstsq` 有效秩）、结果（A/B 矩阵与 6 个无量纲量）；
并记录 python/numpy/平台/BLAS、**实际 import 的 `tensorbem` 路径与求解器源码 sha256**、线程环境。

同一份代码两处跑同一格：控制面（本机）与调度节点（`stage_dir` 直送同一工作树，排除「节点 checkout 了别的 commit」）。
`compare()` 只把**第一个**出现差异的阶段记为因果，其余标 downstream。

## ③ 结果（因果定位）

| 阶段 | 结论 |
|---|---|
| 网格 | 逐位相同 |
| 算子装配（含 Green 函数） | **系统矩阵 sha256 相同**（`68b8f92cab2c15c8…`），条件数、有效秩 1024/1024 全同 |
| **线性求解** | **相对残差 控制面 1.77e-4 vs 节点 4.91e-1（2780×）** |
| 力后处理 | 下游，随求解结果变化 |

求解器源码 sha256 两边相同（`ff206b6a965f0a9d…`），numpy 版本不同（2.4.6 vs 2.0.1），平台同为 loongarch64。
即：**同一份代码装出同一个矩阵，节点上的复数最小二乘没有解出方程**——残差 49%，而它同时报告 full rank、
条件数与控制面逐位一致（奇异值对、解错），属 LAPACK/BLAS 后端缺陷，不是条件数/病态问题。

**推论（falsified）**：m20515 记的「4096 面元规模断裂」是在同一失效环境里跑的，`A33 0.2079 / B33 0.0472`
不能作为求解器规模上限的证据；该档在健康环境下仍未知，需重跑。

## ④ 交付物

| 文件 | 作用 |
|---|---|
| `scripts/hulme_cell_fingerprint.py` | 逐段 fingerprint 工具；`--compare` 输出阶段级差异；残差 >1e-2 时告警并以退出码 3 收尾（环境不可用） |
| `scripts/hulme_crossenv_audit.py` | 从工件+指纹程序化渲染审计文档与图，数字零手抄 |
| `docs/bug_bem_numpy_version_drift.md` | 论文级审计报告（本文件的完整版，含表与图） |
| `docs/bug_bem_numpy_version_drift.png` | 左：两环境 6 量误差 vs 5% 门；右：ka=2 延拓阶梯按环境着色 |
| `benchmarks/verified/casepool_bem_hulme_hemisphere/fingerprints/{ctrl,node}_ka2_s4.json` | 两份 1024 面元格指纹（证据本体） |
| `tests/test_hulme_cell_fingerprint.py` | 10 项守卫：工具保真/可复现/阶段归因 + 已提交工件的跨环境登记表 |

`KNOWN_CROSSENV_DIVERGENCE` 登记表把「同一格两份工件不一致」变成可见且有界的登记项：
新增分歧必失败，登记值漂移超 ±2% 必失败，分歧消失也必失败（防止登记表比缺陷活得久）。

## ⑤ 验证

- `PYTHONPATH=src python -m pytest tests/test_hulme_cell_fingerprint.py -q` → **10 passed**（21.6 s）
- 回归：`tests/test_casepool_hulme_hemisphere.py`、`tests/test_hulme_hemisphere_case.py` 全通过；
  `tests/test_casepool_hulme_hemisphere_results.py::test_figure_artifact_exists` **在 main 上本来就红**
  （`benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png` 不在树里，疑似被门禁按二进制滤除），
  与本单无关，未改动。
- 节点 fingerprint：作业 cj3042（node330，numpy 2.0.1，`stage_dir` 直送本工作树），succeeded。
- 排除实验（控制面）：`rankine_order` 2..10 变化 <0.5%（order≥5 后 <0.05%）、条件数不变；
  `results.json` 中 ka=2 的 CHIEF on/off 差 1e-5 → 两者都不是漂移来源。

## ⑥ 误差判定（是否 <3%）

- 控制面 1024 面元：最坏 B33 **1.15%** → **<3% ✅**（算例 5% 带内成立）
- 节点 1024 面元：最坏 B33 **50.7%** → **>3% ❌**（该环境证据不可用）
- 结论：算例 5% 达标的**前置条件是求解残差正常**；建议所有走调度系统的 BEM 证据在收尾时检查
  `Radiation3DResult.relative_residual`（本仓已返回该字段），>1e-2 一律拒收。

## ⑦ 给下次自找活的线索

1. **重跑 4096 档于健康环境**（控制面 numpy 2.4.6，约 11 min，需走 compute gate 且必须落在健康节点）——
   若 A33/B33 回到 0.386/0.104 附近，则「规模上限」条目可关闭；若仍塌陷，才是真缺陷。
2. 上报平台层：让调度器记录每台节点的 `numpy.__version__` 与 BLAS 后端，并对已知失效节点打标；
   否则任何集群产出的 BEM 证据都可能静默失真。
3. `figure.png` 被门禁滤除导致 `test_figure_artifact_exists` 常态红——建议改为「图可由生成器重建即通过」。