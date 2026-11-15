# 锚点 CASE-BEM-HULME-HEMISPHERE：逐档证据台账（E0 可审计化）+ 第四档未到账的诚实记录

## ① 案例介绍
浮半球（a=1.0 m，平面位于平均自由面 z=0）在无限水深中的辐射/绕射。排体积 ∇=2πa³/3=2.094395 m³，
ρ=1025 kg/m³、g=9.80665 m/s²，频带 ka=0.2…2，参考解为 Hulme 1982 多极级数（144 项，截断漂移 ~6e-5）。
算例容差 5%。

## ② 怎么做的
1. **E0 判据（求解健康准入）从“标量摘要”升级为“逐档台账”**：`hemisphere_independent_convergence`
   新增 `rung_ledger`（每档：面板数、记录残差、host、numpy、求解阶段警告）与 `mixed_provenance`
   （是否跨主机、多少档来源未知）。理由：把一列阶梯压成一个最大值后，便宜档跑控制面、贵的档跑
   集群节点（该节点的线性驱动器曾返回满秩却错误的向量，m20910）这类混合阶梯在工件里不可见。
2. **阶梯驱动器补齐逐行来源**：`scripts/hulme_hemisphere_inhouse_ladder.py` 每档写入 `host`/`numpy`；
   `--merge` 合并旧档时从文件级 environment 补 `host` 并标 `host_source="file_environment"`
   （此前 merge 的 docstring 声称逐行带 host，实现里并没有——这是文档与实现不符的缺陷）。
3. **第四档（4096 面板）续跑**：cj3371 在 n330 上 rc=1（`ModuleNotFoundError`，提交侧没有把 `src`
   放进 sys.path），cj3395（已修正 `PYTHONPATH=src`）在本会话收尾时仍在排队。故阶梯止于 1024 档。

## ③ 计算结果（程序化注入，来自已提交工件）
| 面板数 | A33 | B33 | 记录求解残差 | 相对 Hulme 参考 |
|---|---|---|---|---|
| 64 | 0.358460 | 0.123144 | 9.62e-03 | +7.70% / +19.48% |
| 256 | 0.380435 | 0.107966 | 2.12e-03 | +2.05% / +4.76% |
| 1024 | 0.386337 | 0.104249 | 3.24e-04 | +0.53% / +1.15% |

判据裁定（`evidence/independent_convergence.json`，判定 **FAIL**）：
| 判据 | 实测 | 裁定 |
|---|---|---|
| `E0_rungs_solver_healthy` | 0.009622410465603701 | PASS |
| `E1_convergence_model_credible` | 0.0007675610745442769 | PASS |
| `E2_observed_order_identified` | 8.784605514128511 | FAIL |
| `E3_extrapolation_stable` | 0.18293626839633226 | FAIL |
| `E4_frozen_level_residual` | 0.21021106577786686 | FAIL |
| `E5_adjudicated_spread` | 0.1633457215519194 | FAIL |
| `E6_extrapolation_moves_toward_ref` | null | FAIL |
| `E7_inhouse_model_credible` | 6.280659934122025e-09 | FAIL |
| `E8_inhouse_extrapolated_error` | 0.0011998792327345686 | PASS |
| `E9_inhouse_order_and_limit_stable` | null | FAIL |

E0 台账：最坏记录残差 0.00962，3 档有记录、3 档入拟合、
剔除 0 档；来源未知 3/3 档（本工件早于逐行来源，不冒认主机）。

## ④ 结果图
左：ka=2 的内场阶梯 A33/B33 与 Hulme 参考；右：逐档记录的求解残差与 E0 门槛（1e-2）。

## ⑤ 注意点
- E0 只封住“驱动器没解方程却照样入账”这一条已知通道，不证明残差之外无缺陷。
- 门槛 1e-2 是数量级判据（健康 1.8e-4 vs 失效 4.9e-1），不是精细容差。
- 三档时 E7/E9 必然 FAIL 且这是**正确**的：三参数模型被三档恰好决定，残差恒为零，不构成证据。
  不因“看起来在收敛”而放行。
- 图由提交的数字重建（交付门禁过滤二进制），测试已改为“图可由 results.json 重建”这一不变量。

## ⑥ 误差判定（是否 <3%）
| 关键量 | 值 | 基准 | 误差 | <3%？ |
|---|---|---|---|---|
| A33（1024 档，算例判据） | 0.386337 | 0.388378 | 0.53% | 是 |
| B33（1024 档，算例判据） | 0.104249 | 0.103065 | 1.15% | 是 |
| 求解残差（最坏档） | 9.62e-03 | E0 门槛 1e-2 | 低 1 个量级 | 是 |

收敛证据（E7/E9）**未通过**——第四档未到账，honest FAIL，不粉饰。
