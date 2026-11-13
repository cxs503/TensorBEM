# m18196 — FMM 远场比例诊断：`far_fraction` 的可审计化（重开 #697）

> 仓：TensorBEM ｜ 分支：`agent/m18196` ｜ 起始 main：`178de43`
> 产物：`src/tensorbem/fmm.py`（`PairPartition` / `pair_partition` / `_pair_components`）、
> `scripts/fmm_far_fraction_diagnostic.py`、
> `docs/fmm_far_fraction_diagnostic/{README.md,tables.md,evidence.json,fig_far_fraction.png}`、
> `tests/test_fmm_diagnostic_evidence.py`（16 例）、
> `docs/fmm_operator_linearity/evidence.json`（修正一处陈旧数字）

## ① 案例介绍

`tensorbem.fmm.FMMOperator` 是 3-D Laplace 单层算子的 O(N) 快速求和（八叉树 + 实球谐多极），
`far_fraction()` 是它的效率诊断量：报告有多大比例的 (靶点, 源叶) 对用多极展开。

m16000（线性算子修复）在报告的 §⑤-2 留下一条**自相矛盾**的观察：
同一几何（200 源点高斯团 + 3 个远场靶点，`max_leaf=25`）上 `far_fraction()` 返回 **0.0**，
但该几何上「旧电荷多极缓存」造成的误差是 **2.46**——只可能来自远场路径，两者不能同时成立。
0.0 的直接原因已由 m16109 定位（bench 在**从未 `setup` 的算子**上调用，旧实现 fail-open 返回 0.0，
`matvec` 同样静默返回全零），并加了 fail-closed 守卫。

本单（重开 #697）回答剩下的、更本质的三问，并把诊断量本身做成可审计的：
`far_fraction` 报的数**是不是** `matvec` 真正用多极展开的配对数？近场集合的**成分**是什么？
那条保守余量**花了多少加速、买了多少精度**？

几何与参数（全部由脚本生成）：

| 几何 | 源点 | 靶点 | `max_leaf` | 说明 |
|---|---|---|---|---|
| `surface` | 320（icosphere(2) 面元中心） | 同一批点 ×1.02 | 32 | BEM 均配点场景，靶点在源点面上 |
| `shell` | 320 | 同一批点 ×3.0 | 32 | 每个靶点离**所有**源点都远 |
| `separated` | 200（N(0,0.5²) 高斯团） | 3 个评价点 | 25 | m16000 原始几何 |
| `surface_fine` | 1280（icosphere(3)） | 同批点 ×1.02 | 8…2048 | 叶数扫描 |

## ② 怎么做的

1. **先读代码定位分歧点**：`matvec` 与 `far_fraction` 都用 `_near_mask`，
   而 `_near_mask = (~admissible) | _leaf_near[routed_leaf]`。后一项是「靶点**路由到的那片叶子**」
   的盒邻域判据——对源点云外部的靶点，路由（最近叶心）与配对本身无关。
   于是把近场集合拆成**互斥且完备**的三桶：
   `far`（多极）/ `near_geometric`（靶点离该叶源点球太近，膨胀不收敛，正当直和）/
   `near_box`（只因路由叶是盒邻域而直和，保守余量）。
2. **单一事实来源**：新增 `FMMOperator._pair_components()`，`_near_mask` 与
   `pair_partition()` 都从它取，保证 `near == (~admissible) | box_near` 与 `matvec` 逐位相同；
   新增冻结 dataclass `PairPartition`（三桶计数 + 源点加权质量 + 两个分数）。
   `far_fraction()` 重建为 `pair_partition().far_pair_fraction`（语义、空集语义、
   fail-closed 报错文本均不变）。
3. **回路校验**：`_CountingOperator` 在 `_far_contributions` 上计数，
   断言「实测送入多极路径的配对数」== `pair_partition().far_pairs`（3 几何 × p=2/4/6，9/9 行相等）。
4. **A/B 单变量对照**：`_AdmissibilityOnlyOperator` 只覆盖 `_pair_components`（去掉盒邻域项，
   纯可容许判据），其余代码完全相同 → 差异只能来自那条保守余量。
5. **回归**：`_pair_components` 重构前后，在 3 组随机几何上 `matvec` 输出 `np.array_equal`
   为**真**（逐位不变）；`pytest -k "fmm or FMM"` 137 例全绿。
6. **证据管线**：脚本生成 `evidence.json` / `tables.md` / `README.md` / `fig_far_fraction.png`，
   README 与 tables 的数字全部由 evidence 注入（零手抄）；
   测试用 `_row()` 重算 committed 行并逐字段比对，手改证据文件即红。

## ③ 计算结果

三桶分解与精度（p=4，基准＝稠密直接求和）：

| 几何 | far 配对分数 | far 源点加权分数 | 几何近场对 | 盒邻域近场对 | vs 稠密误差 |
|---|---|---|---|---|---|
| surface | 0.7411 | 0.7280 | 1556 | 3084 | 8.713e-05 |
| shell | 0.7893 | 0.7857 | 0 | 3776 | 5.932e-06 |
| separated | 0.1739 | 0.0933 | 0 | 57 | 1.333e-06 |

A/B（唯一变量＝近/远判据）：

| 几何（p=4） | far 分数 现行 → 纯可容许 | 工作量代理 现行 → 纯可容许 | 误差 现行 → 纯可容许 |
|---|---|---|---|
| surface | 0.7411 → 0.9132 | 359848 → 418924 | 8.713e-05 → 8.365e-04 |
| shell | 0.7893 → 1.0000 | 375544 → 448000 | 5.932e-06 → 1.177e-05 |
| separated | 0.1739 → 1.0000 | 844 → 1725 | 1.333e-06 → 9.713e-05 |

阶数收敛（surface，现行判据）：p=2 `1.123e-03` → p=4 `8.713e-05` → p=6 `9.427e-06`
→ p=8 `1.203e-06`。

叶数扫描（`surface_fine`，p=4）：

| `max_leaf` | 叶数 | far 分数（现行） | 误差（现行） | far 分数（纯可容许） | 误差（纯可容许） |
|---|---|---|---|---|---|
| 8 / 16 | 248 | 0.9365 | 1.945e-04 | 0.9777 | 1.141e-03 |
| 32 / 64 / 128 | 56 | 0.7421 | 1.475e-04 | 0.8829 | 1.162e-03 |
| 2048 | 1 | 0.0000 | 1.900e-15 | 0.0000 | 1.900e-15 |

三条可复核结论：

1. **`far_fraction` 是配对数，不是工作量**：`separated` 上配对分数 0.1739、源点加权只有 0.0933。
2. **源点云外靶点的近场集合 100% 是路由伪影**：`shell`/`separated` 的 `near_geometric_pairs`
   都是 0，直和全部来自盒邻域余量；去掉后 far 分数升到 1.0000。也就是说多极路径的**上限**
   由可容许判据决定，**实际使用率**却由与靶点真实位置无关的路由决定。
3. **盒邻域余量是精度投资**：去掉它误差上升 1 个量级（surface 8.7e-05→8.4e-04），
   而在靶点极少/叶极小的 `separated` 上工作量代理反而**变差**（844→1725，多极每对 `(p+1)²`
   谐波开销压过几十个源点直和）。「far 分数高就更快」是错的。

## ④ 结果图

`docs/fmm_far_fraction_diagnostic/fig_far_fraction.png`（脚本生成，89 kB）：
左图＝三种几何在 p=2/4 下的 far 配对分数（现行 vs 纯可容许，分组柱）；
右图＝`surface_fine` 上 far 分数随 `max_leaf` 的变化（现行 vs 纯可容许，最右端为单叶退化点）。
图上的每个点都对应 `evidence.json` 里的一行，测试校验 tables/README 与 evidence 一致。

## ⑤ 注意点 / 局限

1. **本单是诊断能力，不是算法改进**：`matvec` 数值路径逐位未变（已用 `np.array_equal` 验证），
   新增的是记账 API + 证据管线。故不应把本单记成精度或性能提升。
2. **严格说是树码**：单层多极、无 M2L，far 分数再高也不代表 O(N)。
3. **工作量代理 ≠ 墙钟**：`直和源点对数 + far_pairs·(p+1)²`，不含常数因子；未做端到端 BEM 求解
   （`solve_laplace_surface*`）计时，如实登记为未覆盖项。
4. **叶数被几何量化**：八叉树按点分布分裂，`surface_fine` 的 6 个 `max_leaf` 档只落到 3 个不同叶数。
5. **单叶退化是对的**：`max_leaf=2048` 时叶数 1、far 分数 0、误差 1.9e-15（等价直和），
   不是缺陷——与 m16000 的「未 `setup` 返回 0.0」是两回事。
6. **修了一处陈旧证据**：`docs/fmm_operator_linearity/evidence.json` 的
   `error_versus_order.far_fraction` 曾是 **0.0**（未 `setup` 的旧调用留下的值），
   本单用脚本在同一几何上重算为 **0.1739** 并写回该字段；
   `TestLinearityEvidenceConsistency` 钉住该字段必须等于当前代码复现值，防止再腐化。
   该文件其余行（p=1..8 的误差表）重算后仅末位 ULP 级差异（BLAS 求和序），未改动。

## ⑥ 误差判定

| 关键量 | 基准 | 本单实测 | 是否 <3% |
|---|---|---|---|
| surface p=4 matvec | 稠密直接求和 | 8.713e-05 | ✅ |
| surface p=8 matvec | 稠密直接求和 | 1.203e-06 | ✅ |
| shell p=4 matvec | 稠密直接求和 | 5.932e-06 | ✅ |
| separated p=4 matvec | 稠密直接求和 | 1.333e-06 | ✅ |
| 单叶退化 matvec | 稠密直接求和 | 1.900e-15 | ✅（等价直和） |
| far 配对数 vs 实测 matvec 远场对数 | 逐行相等 | 9/9 行，偏差 0 | ✅（0%） |
| `_pair_components` 重构前后 matvec | HEAD 版实现 | 3 组几何 `array_equal` | ✅（逐位相同） |

结论：所有被引用的数值误差都远小于 3%；本单的判别性指标是**配对计数完全一致（0 偏差）**，
精度量只用于说明保守判据的取舍。

## 验证清单

- `PYTHONPATH=src python -m pytest tests/test_fmm_diagnostic_evidence.py -q` → 16 passed
- `PYTHONPATH=src python -m pytest -q -k "fmm or FMM" tests` → 137 passed
- `PYTHONPATH=src python -m pytest tests/test_fmm.py tests/test_fmm_cost_audit.py
  tests/test_fmm_setup_contract.py tests/test_fmm_geometry_contract.py
  tests/test_fmm_operator_linearity.py -q` → 74 passed
- `PYTHONPATH=src python scripts/fmm_far_fraction_diagnostic.py` → 生成全部证据（~12 s）
- 全域套件（CI 同序：先跑 17 个 benchmark 证据步骤再 `pytest -n 8 --dist loadfile`）递交集群
  作业 `cj1826`（stage_dir = 本工作树）。**截至本报告定稿该作业仍在排队（队列 306 位），未取回结果**——
  如实登记为未完成项，域内已由 `-k "fmm or FMM"` 的 137 例 + 5 个 fmm 模块 74 例覆盖
  （`fmm` 模块的调用面只有 `src/tensorbem/benchmarks/fmm_cost_audit.py` 与上述测试）。
- 两处与本次改动无关的 **main 既有红**（已在 HEAD 的干净导出上复现，非本单引入）：
  `tests/test_gate_forbidden_artifacts.py::test_no_test_pins_a_push_gate_filtered_artifact`
  与 `tests/test_skip_gates_are_tracked.py`（3 例）——前者指向
  `test_casepool_hulme_*` / `test_rigid_lid_*` 等未触及文件，后者是未注册的 artifact 跳过门清单。