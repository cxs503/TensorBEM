# CASE-BEM-HULME-HEMISPHERE：4096 面元档断裂与环境依赖（honest FAIL）

**锚点**：CASE-BEM-HULME-HEMISPHERE（算例池·Hulme 浮半球辐射/绕射，容差 <5%）

## ① 做了什么

上一单（m20234）留了两笔债：本仓求解器阶梯的 4096 面元档在节点上以 `malloc(): invalid size` 崩溃、工件因此没落盘，且它打印的 ka=2/1024 面元 A33/B33 = 0.3666/0.0508 与已发布 results.json 的 0.38634/0.10425 不符「未及核实」。本单把这两笔债一起结清：

1. `scripts/hulme_hemisphere_inhouse_ladder.py` 改为**逐档落盘**（原子替换）、`--resume` 复用已完成档、逐档记录峰值 RSS 与线程环境，失败档写进 `failures` 后仍以非零退出码结束；新增 `--merge` 把不同进程（不同主机）产出的单档工件合成一条阶梯，并保留每档 provenance。
2. 控制面跑 64/256/1024 三档，调度系统（节点 n330，作业 cj2750）跑 4096 档，合并为 `inhouse_ladder_ka2.json`（4 档）。
3. 判据硬化：`E7_inhouse_model_credible` 原先只要求「三参数模型复现最后三档」——**三档恰好决定三参数模型，残差恒为零，这条判据在任何数据上都通过**。现要求档数 ≥ 4 且复现*全部*档；并新增 `E9_inhouse_order_and_limit_stable`（去掉最粗档后观测阶与极限的稳定性）。

## ② 计算结果（ka=2，A/(ρ∇)、B/(ρω∇)，∇=2πa³/3）

| 面板数 | A33 | B33 | 相对参考 A33 | 相对参考 B33 |
|---|---|---|---|---|
| 64 | 0.35846 | 0.12314 | -7.70% | +19.48% |
| 256 | 0.38044 | 0.10797 | -2.05% | +4.76% |
| 1024 | 0.38634 | 0.10425 | -0.53% | +1.15% |
| 4096 | 0.20786 | 0.04718 | -46.48% | -54.23% |
| 1024（节点 n315，numpy 2.0.1） | 0.36661 | 0.05079 | -5.61% | -50.72% |
| 参考（Hulme 级数 144 项） | 0.38838 | 0.10307 | — | — |

## ③ 两个发现

**发现 1（可复现性，阻断级）**：同一 commit、同一网格（`condition_number` 在控制面与节点上逐位相同：2.398615400620687）、同一 ka，1024 面元档在 **numpy 2.4.6**（控制面）给出 A33=0.386337 / B33=0.104249，在 **numpy 2.0.1**（节点）给出 A33=0.366607 / B33=0.050792——B33 差 **2.05×**，A33 差 5.1%。矩阵装配（几何、条件数）完全一致，说明分歧在系数**后处理**（绕射/辐射力的虚部）或 Green 函数求值路径上对 numpy 版本的依赖。已发布算例判定 G1（PASS, 1.15%）只在本控制面环境成立；在 numpy 2.0.1 上同一网格的 B33 相对误差是 51%，**判据会翻成 FAIL**。上一单那条「未核实」的 0.3666/0.0508 由此得到解释：它不是进程崩溃污染，而是节点环境（numpy 2.0.1）下的真实输出。

**发现 2（规模断裂）**：4096 面元档在节点上**跑通**（654 s，峰值 RSS 1.4 GB，无崩溃）——上一单的 `malloc(): invalid size` 不是求解器的规模上限，而是那次驱动（fork 进程池 + BLAS）的问题。但 4096 档的数值不可用：A33 由 0.3863 掉到 0.2079（离参考 46%），B33 由 0.1042 掉到 0.0472（离参考 54%），**不是收敛方向**，且与本单发现 1 的量级同源（同为约 2× 的虚部塌陷）。故阶梯在 4096 档断裂，`v_inf` 不可用，`E7/E9/E8` 三项诚实判 FAIL。

## ④ 判据裁定（本轮）

| 判据 | 实测 | 裁定 |
|---|---|---|
| `E1_convergence_model_credible` | 0.077% | PASS |
| `E2_observed_order_identified` | 878.461% | FAIL |
| `E3_extrapolation_stable` | 18.294% | FAIL |
| `E4_frozen_level_residual` | 21.021% | FAIL |
| `E5_adjudicated_spread` | 16.335% | FAIL |
| `E6_extrapolation_moves_toward_ref` | n/a | FAIL |
| `E7_inhouse_model_credible` | 54.825% | FAIL |
| `E8_inhouse_extrapolated_error` | 517.497% | FAIL |
| `E9_inhouse_order_and_limit_stable` | 144.966% | FAIL |

## ⑤ 误差判定（是否 < 3%）

- 1024 面元档（控制面 numpy 2.4.6）最坏 B33 相对参考 **1.15%** → **< 3% ✅**（算例 5% 带内）。
- 1024 面元档（节点 numpy 2.0.1）B33 相对参考 **50.7%** → **> 3% ❌**。
- 4096 面元档 A33/B33 相对参考 **46.5% / 54.2%** → **> 3% ❌**（阶梯断裂，不可用）。
- 结论：算例的 5% 达标**只在 numpy 2.4.6 控制面环境下成立**，跨环境不可复现——这是本单的阻断级发现。

## ⑥ 复现与工件

- 工件：`inhouse_ladder_ka2.json`（4 档 + provenance）、`inhouse_ladder_ka2_rung5.json`（cj2750/n330）、`inhouse_ladder_ka2_rung4_node.json`（cj2767/n315，numpy 2.0.1）、`evidence/independent_convergence.json`。
- 图：`docs/case_bem_hulme_ladder_env_figure.png`（左：阶梯误差 vs 面板；右：同格换 numpy 后 B33 偏 2×）。
- 判据：`python -m tensorbem.benchmarks.hemisphere_independent_convergence`（含新 E7/E9）；`pytest tests/test_casepool_hulme_independent_convergence.py`（11 passed，含「三档不得通过 E7」的反例测试）。
- 建议后续条目：`BUG-BEM-NUMPY-VERSION-DRIFT`（同一网格系数随 numpy 版本 2× 漂移，量化到具体 matplotlib-free 数值路径）与 `AUDIT-BEM-4096-PANEL-BREAKDOWN`（4096 档虚部塌陷的根因：Green 函数求值分块 / 双精度累加阶）。
