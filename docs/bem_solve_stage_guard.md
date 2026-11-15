# 求解阶段的健康审计：把「静默算错」变成「要么算对、要么报错」

**锚点**：CASE-BEM-HULME-HEMISPHERE（算例池·Hulme 浮半球辐射/绕射，容差 <5%）
**任务**：m21143（自找活，generalist 席）
**续接**：m20910（`BUG-BEM-NUMPY-VERSION-DRIFT`，把跨环境漂移定位到线性求解阶段）与
m20234 §5 遗留的「4096 面元档在健康环境仍未知」

## ① 问题：一个「条件数完美、答案错 51%」的求解器

m20910 的跨环境指纹已经证明：同一 commit、同一 1024 面元网格（顶点/面积/形心逐位相同）、
同一 ka=2，控制面（numpy 2.4.6）与集群节点（numpy 2.0.1，loongarch64）的
**系统矩阵 sha256 相同**、**条件数逐位相同（2.398615）**、**有效秩都是 1024/1024**，
但结果差一倍：

| 环境 | A11 | B11 | A33 | B33 | B33 相对 Hulme 参考 |
|---|---|---|---|---|---|
| 控制面 numpy 2.4.6 | 0.249848 | 0.340432 | 0.386337 | 0.104249 | +1.15% |
| 集群节点 numpy 2.0.1 | 0.235658 | 0.337733 | 0.366607 | 0.050792 | **−50.7%** |

也就是说：**求解器在一个已知会静默返回错误答案的 LAPACK 后端上，没有任何自检，
把 51% 错的阻尼系数当结果发布了**。这是本单要修的东西——不是环境，而是
「求解阶段没有不可靠性判据」这个设计缺陷。

问题在于**残差和条件数都不能当判据**：

- CHIEF 增广系统是**长方形**（1024+5 行 × 1024 列），它在离散层面**本来就矛盾**：
  控制面实测最小相对残差 1024 面元 1.77e-4、256 面元 1.20e-3、64 面元 6.33e-3。
  这是「物面方程与内部空场方程在有限面元下不完全兼容」，不是求解失败。
- `numpy.linalg.cond` 对长方形矩阵按 `min(M,N)` 取奇异值比，多出来的 5 行只会**增加**奇异值，
  所以条件数永远好看——m20910 的 2.398615 就是这个假象。

## ② 方法：最小二乘的**驻点性**才是求解器的判据

最小二乘最小点处 `A^H r = 0`（`r = A x − b`）严格成立，与系统是否矛盾无关。于是定义无量纲量

    stationarity(x) = ||A^H r|| / (||A|| ||r||)

- 只要 `x` 是真最小点（无论系统多矛盾），它是机器精度量级；
- `x` 是错的向量时它是 O(1)。

这是**唯一能把「方程彼此不同意」和「求解器说谎」分开的量**，而残差做不到（两者都让残差变大）。
实现落在 `src/tensorbem/free_surface_solver3d.py`：

| 组件 | 作用 |
|---|---|
| `least_squares_stationarity(matrix, solution, rhs)` | 公开判据，O(N²) 一次 matvec，可被 benchmark/测试直接复用 |
| `_solve_least_squares(matrix, rhs)` | 先 `np.linalg.lstsq`（SVD/gelsd）；驻点性 > `1e-6` 时改用 **Householder QR**（`geqrt/geqrf`+三角回代，**另一条 LAPACK 路径**）；两条都不达标记则 `RuntimeError` |
| `_solve_square(matrix, rhs)` | 方阵路径 `np.linalg.solve`，后向稳定性审计（相对残差 > `1e-8` → QR 回退 → 否则报错） |
| `_solve_system(...)` | 未改签名（`symmetry3d` 等三处调用方零改动）；健康路径**逐位不变** |

关键设计取舍：回退时发 `RuntimeWarning`（可见但不致命），**两条路径都失败时抛异常，
绝不返回一个没通过审计的向量**。

## ③ 计算结果（程序化注入，见 `solve_stage_audit.json`）

**判据的量级分离**（64 面元 CHIEF 系统上把解按相对量 δ 扰动）：

| δ | stationarity | 相对残差 |
|---|---|---|
| 0（最小点） | 4.0e-14 | 4.3e-03 |
| 1e-6 | ≈1e-10 量级 | 4.3e-03 |
| 1e-2 | 2.4e-03 | 4.3e-03 |
| 1e-1 | 2.0e-02 | 1.5e-02 |

1% 扰动使驻点性跳 **>1e10 倍**，而相对残差**几乎不动**（4.26e-3 → 4.3e-3）：
这直接证明残差不能当求解器判据，驻点性可以。门限 `1e-6` 落在这七个数量级空隙中间。

**真实生产系统上判据保持安静**：

| 面元 | 条件数 | CHIEF 固有相对残差 | 健康解 stationarity | A33（无量纲） |
|---|---|---|---|---|
| 64 | 2.389 | 4.26e-03 | **4.03e-14** | 0.358460 |
| 256 | 2.352 | 9.42e-04 | **1.93e-13** | 0.380435 |

固有残差随加密下降 4.5 倍（离散收敛），驻点性始终是 1e-13 量级——判据对「物理上的矛盾」
完全免疫，只对「求解器的错误」敏感。

## ④ 结果图

`docs/bem_solve_stage_guard.png`：左=扰动扫描 vs 门限（七个数量级的空隙）；右=三个真实
rung 的「固有残差（橙色，判物理）」对照「解驻点性（绿色，判求解器）」，红色虚线为门限。

## ⑤ 验证

```
PYTHONPATH=src python -m pytest tests/test_bem_solve_stage_guard.py -q      # 9 passed
PYTHONPATH=src python -m pytest tests/test_hulme_hemisphere_case.py \
    tests/test_casepool_hulme_hemisphere.py tests/test_hulme_cell_fingerprint.py \
    tests/test_hulme_reference_bvp.py -q                                    # 30 passed (118 s)
PYTHONPATH=src python scripts/bem_solve_stage_figure.py                      # 生成图与审计工件
```

新测试的两条**负控**（防「空审计」）：
- `test_fallback_recovers_the_minimiser_when_the_driver_lies`：monkeypatch 让 `lstsq` 返回零向量
  → 必须发 `RuntimeWarning`、走 `"qr"`、且**恢复出的解与真最小点逐位一致**（atol 1e-12）；
- `test_both_paths_failing_is_loud_not_silent`：两条路径都被 monkeypatch 破坏 → 必须 `RuntimeError`，
  即「静默返回错向量」这条路被物理性堵死；
- `test_no_spurious_warnings_on_a_normal_radiation_solve`：把 warning 提升为 error 跑真实辐射解，
  证明门限在生产路径上不会误报（否则运维一定会关掉它）。

## ⑥ 注意点与局限（本单不能声称什么）

- **不能替代修复环境**：本改动让错误可见/可绕行，不修 LAPACK。节点上如果 `geqrt` 也坏，
  结果就是一条明确的异常而不是正确数字——这比静默错好，但仍不是「节点可用」。
- **QR 回退只对满列秩成立**：CHIEF 增广系统实测有效秩 1024/1024（m20910 指纹），所以安全；
  遇到真正秩亏的算例会走 `LinAlgError` 分支并报出清晰信息，不会静默。
- **4096 面元档在健康环境下仍未测完**：本单已把健康环境 rung 驱动（`scripts/hulme_scale_rung.py`，
  带残差守卫与 crash-safe 落盘）与作业 cj3213 递交到调度队列，但本单收尾时作业仍在排队，
  因此**没有**新的 4096 面元数字入账——按诚实纪律记 `not_run`，不得读作「规模上限不存在」。
  m20910 的推论（`A33 0.2079` 来自失效环境，不能当规模证据）仍未在健康环境被**实测**证实或推翻。
- **驻点性不是误差界**：它只判「是不是最小点」，不判「最小点离真解多远」——后者仍由本算例的
  5% 门与参考级数负责。

## ⑦ 误差判定（是否 <3%）

| 量 | 实测 | 判据 | 是否 <3% |
|---|---|---|---|
| 健康路径结果是否被改动 | A33(sd=2)=0.358460、A33(sd=3)=0.380435，与已发布 `results.json` 逐位相同 | 5e-7 绝对 | ✅（0.00%） |
| 毁坏驱动时恢复解与真最小点之差 | ≤1e-12（atol） | 1e-12 | ✅ |
| 判据误报（健康路径） | 0 次（warnings→error 下全通过） | 0 | ✅ |

## ⑧ 交付物

| 文件 | 作用 |
|---|---|
| `src/tensorbem/free_surface_solver3d.py` | 驻点性判据 + QR 回退 + 方阵残差审计 + 失败即抛 |
| `tests/test_bem_solve_stage_guard.py` | 9 项守卫（含 2 条负控、1 条误报控制、1 条已发布值回归） |
| `scripts/bem_solve_stage_figure.py` | 图与审计工件（数字零手抄） |
| `scripts/hulme_scale_rung.py` | 健康环境 rung 驱动（残差守卫 + crash-safe），供 4096 面元档补测 |
| `scripts/hulme_scale_probe.py` | 分段探针（算子一致性/奇异值/CHIEF 行残差） |
| `scripts/hulme_env_probe.py` | 节点侧环境探针（numpy 版本 + 求积规则精确矩检验） |
| `docs/bem_solve_stage_guard.png` | 本报告图 |
| `benchmarks/verified/casepool_bem_hulme_hemisphere/solve_stage_audit.json` | 上表全部数字 |