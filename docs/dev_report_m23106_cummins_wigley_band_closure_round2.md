# CASE-BEM-CUMMINS-WIGLEY 频带闭合 round 2（重开 #1088）：求解器可采性闸门 G0

**Mission** m23106 · **锚点** `CASE-BEM-CUMMINS-WIGLEY` · **仓** TensorBEM · 分支 `agent/m23106`
上游交接：`docs/dev_report_m20462_cummins_wigley_band_closure.md`（round 5，扩频带管线交付，锚点仍 open）。

## 1 案例介绍

Wigley III 数学船型（$L=3.0$ m，$B=0.3$ m，$T=0.1875$ m，$C_b=4/9$，排水体积 0.075 m³），$\rho=1025$ kg/m³，$g=9.80665$ m/s²，无限水深，旋转中心 $(0,0,-T/2)$。6 自由度刚性辐射，判据来自 Cummins (1962) 时域卷积与 Kramers-Kronig 因果对：

```
(M+A∞) x'' + ∫₀ᵗ K(t−s) x'(s) ds + C x = F(t),   K(t) = (2/π)∫₀^∞ B(ω) cos(ωt) dω
B(ω) = −(2/π) PV∫₀^∞ s²(A(s)−A∞)/(s²−ω²) ds      A(ω) = A∞ + (2/π) PV∫₀^∞ B(s)/(s²−ω²) ds
```

算例已预声明五门一致性闸门（G1 交叉求解器 10 %、G2 $K(0)$ 2 %、G3 Hilbert 对 10 %、G4 自由衰减 $\zeta$ vs 半功率 $\zeta$ 10 %、G5 往返变换 5 %）。本轮补上第六门 **G0 求解器可采性 1 %**——它不是一致性陈述，而是"这些节点到底是不是收敛解"的前置问题；门限取算例池 `C0` 门同一数值（相对残差 < 1 %），预先声明、不看数字调参。

## 2 怎么做的

**发现的缺陷（本单的功能点）。** 五门一致性闸门只读 $A(\omega),B(\omega)$ 系数，结构上无法看见"某个节点的线性求解根本没收敛"：系数是有限数、积分是精确的、判决照常给出。这不是假想——在树工件 `docs/m20021/sweep_medium.npz`（扩频带证据簇提交物）41/41 节点残差越 1 %（最坏 1.245），其垂荡阻尼在 $\omega=5.29116$ rad/s 为 $B_{33}=-129.9$（负阻尼 = 凭空造能）。此前 round-5 管线的行为是：`_solve_one` 无条件落盘缓存，`gates_from_sweep` 直接读系数——负阻尼直到深层 `time_domain.radiation_memory_kernel` 才抛 `ValueError`（无节点/残差归属），而"错的但仍是正的"坏节点则会一路静默进入判决。

**改动（4 个文件，单一功能点）。**

1. `src/tensorbem/benchmarks/cummins_wigley_kk_causality.py`：新增 `SOLVER_RESIDUAL_TOL = 0.01`、`GATES["G0_solver_validity"]`、`residual_admissibility()`（给出越限节点列表、最坏残差及其频率；无残差记录时 `passed=None` 即"跳过"，绝不算通过），并把它作为 `gates_from_sweep` 的第一门接入；不通过时仍汇总各门数字但置 `interpretable=False`，且 `admissible=False` 写进返回体。
2. `scripts/run_cummins_wigley_closure_full.py`：`_solve_one` 残差不达标即**不缓存**（坏解落 `rejected/` 留档后报错退出）；已缓存节点若坏也拒绝复用；`run_gate_stage` 在评估前先跑 G0，缓存里存在坏节点则拒绝发布判决；`--residual-tol` 暴露但默认取预声明值。
3. `scripts/audit_cummins_closure_admissibility.py`（新）：只读缓存工件的审计器，输出 `docs/m23106/closure_admissibility.json`。
4. `tests/test_cummins_wigley_residual_admissibility.py`（新）：7 项测试，含真实工件回归。

**工具链与设置。** 全部计算只读已提交 `sweep_*.npz`（无 BEM 求解、无插值）：5 份工件 × G0 + 两份可采带的 G0–G5 ≈ 6 s/带，本机短测即可；报告数字由 `scripts/make_m23106_cummins_admissibility_evidence.py` 从 JSON 程序化渲染注入，零手抄。

## 3 计算结果

<!-- BEGIN AUTOTABLES -->
### 3.1 G0 审计（在树 5 份 sweep 工件，阈值 1% = 算例池 C0）

| sweep | 网格 | 节点 | 越限节点 | 最坏残差 | 最坏 ω [rad/s] | G0 |
|---|---|---|---|---|---|---|
| `committed_medium` | medium | 41 | 0 | 0.002447 | 10.832469 | PASS |
| `committed_coarse` | coarse | 41 | 0 | 0.006392 | 10.832469 | PASS |
| `m20021_medium` | medium | 41 | 41 | 1.2451 | 5.291161 | FAIL |
| `m20021_coarse` | coarse | 41 | 41 | 1.2534 | 10.832469 | FAIL |
| `n315_coarse` | coarse | 41 | 41 | 1.2534 | 10.832469 | FAIL |

### 3.2 G0–G5 逐门（可采性通过的带）

| band | G0 | G1 | G2 | G3 | G4 | G5 | verdict |
|---|---|---|---|---|---|---|---|
| `committed_medium` | 0.00245 ✓ | skip | 4.95e-16 ✓ | 0.518 ✗ | 0.397 ✗ | 0.480 ✗ | `FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip` |
| `committed_coarse` | 0.00639 ✓ | skip | 3.35e-16 ✓ | 0.517 ✗ | 0.404 ✗ | 0.481 ✗ | `FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip` |

门限：G0 1 %、G1 10 %、G2 2 %、G3 10 %、G4 10 %、G5 5 %。G1 为 `skip`：该 41 节点带不含 ω=1 rad/s，交叉求解器参考只在该频率冻结，不插值、不臆造。

### 3.3 分歧工件的可判定性（G1–G5 根本跑不动）

| sweep | 评估结果 |
|---|---|
| `m20021_medium` | `ValueError`：negative radiation damping is not physical (energy would be created, not radiated): min damping = -129.917 at omega = 5.29116; B(omega) >= 0 is required |
| `m20021_coarse` | `ValueError`：negative radiation damping is not physical (energy would be created, not radiated): min damping = -0.94709 at omega = 5.86145; B(omega) >= 0 is required |
| `n315_coarse` | `ValueError`：negative radiation damping is not physical (energy would be created, not radiated): min damping = -0.94709 at omega = 5.86145; B(omega) >= 0 is required |

### 3.4 单个坏节点的静默性（committed medium，1 节点扰动量级扫描）

| heave 阻尼缩放 | G0（含门） | 无 G0 的 verdict | G4 | G3 | G5 | 是否可解释 |
|---|---|---|---|---|---|---|
| 1.00（基线） | PASS | `FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip` | 0.3970 | 0.5178 | 0.4804 | 是 |
| 1.00（阻尼不变，仅残差坏） | FAIL (residual 1.245) | `FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip` | 0.3970 | 0.5178 | 0.4804 | 否 |
| 0.80 | FAIL (residual 1.245) | `FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip` | 0.4609 | 0.5178 | 0.4804 | 否 |
| 0.60 | FAIL (residual 1.245) | `FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip` | 0.7009 | 0.5178 | 0.4804 | 否 |
| 0.40 | FAIL (residual 1.245) | `FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip` | 0.7521 | 0.5178 | 0.4804 | 否 |
| 0.20 | FAIL (residual 1.245) | `FAIL:G3_kramers_kronig,G4_free_decay,G5_round_trip` | 0.7577 | 0.5178 | 0.4804 | 否 |

扰动节点 ω = 5.291 rad/s，注入残差 1.245（实测 m20021 medium 最坏值）；缩放后 B33 仍为正，因此能通过核函数唯一的一条物理检查（B ≥ 0）。

<!-- END AUTOTABLES -->

* **G0 的判别力**（§3.1）：两份 committed 带 0/41 越限（最坏 0.245 % / 0.639 %），三份扩频带证据簇工件 41/41 越限（最坏 1.245 / 1.253）——量级差三个 10 倍档，门限落在中间，不是拟合出来的。
* **两份坏工件根本进不了门阵**（§3.3）：`radiation_memory_kernel` 直接以负阻尼拒绝，`m20021_medium` 的 $-129.917$ 在 $\omega=5.29116$ 被抓到——这正是历史上被当成"不规则频率残留"的那颗非物理负阻尼点，本轮确认其身份是坏节点求解（残差 1.245）。
* **G0 与前五门不冗余**（§3.4）：单节点垂荡阻尼缩放到 0.8/0.6/0.4/0.2 时 G4 从 39.7 % 走到 46.1/70.1/75.2/75.8 %（最高 +36 pp），而 verdict 字符串**一次都没变**（该带本就 FAIL：G3/G4/G5），五门没有任何一门能指出这颗节点；只有 G0 报出 `residual=1.245 @ ω=5.291` 并把整带置 `interpretable=False`。

## 4 结果图

图 1（`docs/m23106/fig_residual_and_damping.png`）：(a) 残差–频率，committed medium 全在 1 % 线下、m20021 medium 全在线上，量级分离一眼可辨；(b) 垂荡阻尼 $B_{33}(\omega)$，m20021 在 $\omega\approx5.29$ 处翻负到 $-129.9$，标注即坏节点。

图 2（`docs/m23106/fig_gate_sensitivity.png`）：(a) committed 带逐 DOF 的 G3/G5 与门限（尾截断造成的超限，非求解器回归）；(b) 单节点毒化扫描下 G4 的移动曲线 vs 干净基线——判决字符串不变，说明"看不见"是结构性的而不是数值偶然。

## 5 注意点（坑 / 局限 / 适用范围）

* **G0 不认证物理**。它只回答"节点收敛了吗"。收敛但建模错的带仍会被 G0 放行，需要 G1（外部 capytaine 锚点）等继续兜底。
* **committed 带不是预声明闭合网格**。§3.2 的 G3 51.8 % / G4 39.7 % / G5 48.0 % 是在已提交的 41 节点 [0.2, 12.0] rad/s 带上做的**诊断**，不是 round-5 声明的闭合网格 [0.10, 20] rad/s + 共振簇上的判决；两者不可混引。扩频带仍未解完，锚点状态**保持 open**（本单不改变这一点）。
* **G1 在这些带上是 `skip`**：41 节点带不含 $\omega=1$ rad/s，冻结参考只在该频率定义，插值会伪造证据。
* **两份坏工件的身份**：`m20021_*` 与 `evidence_cluster_n315_sweep_coarse.npz` 是负控/失败证据（README 早已记为 FAIL 39.43 % roll），本单**不删不改**它们，只让管线不再生产同类节点、并让门阵不再被其毒化。
* **`ref=main` 不可用于本算例**（round-5 已记）——集群节点检出的 main 曾缺 `tensorbem.benchmarks.cummins_wigley_kk_causality`；本单未做集群求解，不涉及。
* **热文件声明**：`src/tensorbem/benchmarks/cummins_wigley_kk_causality.py` 是与 round-5/6/7/8/9 及 #1123 等开 PR 共享的核心模块。本单对其改动是**纯增量**：新增 1 个常量、1 个 `GATES` 条目、1 个函数、`gates_from_sweep` 两个关键字参数（默认值 = 预声明值），不修改任何既有函数签名/语义；默认行为对"无残差记录"的旧工件保持向后兼容（`passed=None` 跳过）。

## 6 误差判定（逐项 vs 基准，显式回答是否 <3 %）

| 关键量 | 基准 | 实测 | 误差 | <3 %？ |
|---|---|---|---|---|
| G0 门限 | 算例池 `C0` 相对残差限 1 % | `SOLVER_RESIDUAL_TOL` = 1 % | 0（定义同一） | 是（逐位相同） |
| committed medium 最坏节点残差 | 1 % | 0.245 % | 0.245 % | 是 |
| committed coarse 最坏节点残差 | 1 % | 0.639 % | 0.639 % | 是 |
| G3（committed medium，诊断带） | 10 % | 51.8 % | +41.8 pp | **否（honest FAIL）** |
| G4（committed medium，诊断带） | 10 % | 39.7 % | +29.7 pp | **否（honest FAIL）** |
| G5（committed medium，诊断带） | 5 % | 48.0 % | +43.0 pp | **否（honest FAIL）** |
| 扩频带闭合网格 G1–G5 | 预声明五门 | **无数据**（扩频带未解完） | 不可判 | **无法回答（声明未完成）** |

结论：本单交付的是"证据真伪的前置判别"，其自身精度即门限本身（1 %，与算例池 C0 逐位一致，<3 % 满足）。**它不改善、也不宣称改善** G3/G4/G5 在物理带上的超限；按 round-5 交接，锚点的通过判定仍待扩频带（[0.10, 20] rad/s + 共振簇）求解完成。

## 7 复现

```
# G0 审计（只读缓存工件，~10 s）
PYTHONPATH=src python3 scripts/audit_cummins_closure_admissibility.py
# 图 + 自动表注入
PYTHONPATH=src python3 scripts/make_m23106_cummins_admissibility_evidence.py
# 测试
PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_residual_admissibility.py -q   # 7 passed
PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_kk_causality.py \
    tests/test_cummins_wigley_closure_pipeline.py tests/test_cummins_wigley_band_edges.py \
    tests/test_cummins_verdict_closure.py tests/test_cummins_wigley_g4.py \
    tests/test_cummins_wigley_g5_attribution.py -q     # 43 passed, 8 skipped
```

## 8 未完成 / 下一步

1. **扩频带闭合仍待求解**：`scripts/run_cummins_wigley_closure_full.py --stage solve --grid extended|closure` 分片递交（现在若节点分歧会在递交时就被 G0 拦下，不再污染缓存）。
2. §3.2 的 G3/G5 超限是**带截断**结论，需在扩频带上复核；若扩频带仍 FAIL，则应立项 G3 归一化（round-5 §5 已记 `CASE-BEM-CUMMINS-WIGLEY-G3-SCALE`）。
3. G0 目前只覆盖 CUMMINS-WIGLEY 锚点的闭合管线；是否推广到算例池其它 BEM 扫频（凡 `sweep_*.npz` 都带 `solver_relative_residual`）应作为独立条目评估。