# m23281：Hulme 浮半球本仓延拓阶梯的逐档自证来源（E0 provenance 升级）

**锚点**：`CASE-BEM-HULME-HEMISPHERE`（算例池·Hulme 浮半球辐射/绕射，容差 <5%）
**本单**：m23281（重开 TensorBEM#1094 第 1 轮；承接 m20515 / m21334 的 E7/E9 欠账）

## ① 案例介绍与欠账

浮半球（a = 1.0 m，平面与自由面齐平）在无限深水中的辐射/绕射。参考解为 Hulme 1982 的
无波多极级数（144 项，截断漂移 ~6e-5），算例容差 5%。算例池对**本仓求解器自身的离散误差**
的裁定依赖 ka=2 处的延拓阶梯 `inhouse_ladder_ka2_healthy.json`：三参数收敛模型
`v(N) = v_inf + C·N^-p` 需要 **≥4 档**才有拟合自由度（E7），并在去掉最粗档后要求观测阶与
极限稳定（E9）。m21334 已把阶梯做到 3 档并把第 4 档（4096 面元）的缺失写进 `absence_reason`，
因此 E7/E9 保持诚实 FAIL。

本单要结的是同一处欠账的**前置条件**：已提交的 3 档只有文件级 `environment`，
**没有任何一档能自证是哪台机器、哪个求解驱动、以什么残差产出的**。而本算例恰好被证明对环境敏感
（同一网格在 numpy 2.0.1 节点上 `lstsq` 返回满秩且条件数「健康」的错误解，残差 4.9e-1，B33 偏 2 倍；
`docs/bug_bem_numpy_version_drift.md`）。在这样的算例上，一档来源不明的数据与一档来自失效节点的数据，
在工件里长得一模一样。

## ② 怎么做的

1. **重跑、不重算**：用当前阶梯驱动 `scripts/hulme_hemisphere_inhouse_ladder.py` 在同一控制面
（`ec6ae0b42a0c`，python 3.11.6，numpy 2.4.6
，OPENBLAS_NUM_THREADS=4）重跑 subdivisions 2/3/4 = 64/256/1024 面元，落 `inhouse_ladder_ka2_ctrl_rerun.json`。
驱动为每档写入 `host` / `numpy` / `radiation_solve_method` / `radiation_stationarity` / 求解残差 / `solve_warnings`。
2. **逐位一致性核验（这是控制实验，不是新物理）**：重跑的三档六量必须与已提交档**逐位相同**，否则
就不是「补来源」而是「换数据」，必须停下来查原因。
3. **工件换代**：`inhouse_ladder_ka2_healthy.json` 的 rows 换成带逐档来源的重跑行，新增 `provenance`
块指名贡献文件；`evidence/independent_convergence.json` 与 `docs/case_bem_hulme_independent_convergence.md`
由 `python -m tensorbem.benchmarks.hemisphere_independent_convergence` **重新渲染**（数字零手抄）。
4. **回归测试**：新增 `test_committed_healthy_ladder_carries_per_row_provenance`，锁住「每档必须自证来源」
与「E0 的 unknown-host 计数必须为 0」。

## ③ 计算结果

### 3.1 重跑档 vs 上一版 healthy 行（逐位比对，六量）

| 面板数 | 量 | 重跑值 | 上一版 healthy 值 | 逐位相同（`==`） |
|---|---|---|---|---|
| 64 | A11 | 0.255801957 | 0.255801957 | 是 |
| 64 | B11 | 0.313958412 | 0.313958412 | 是 |
| 64 | A33 | 0.358459837 | 0.358459837 | 是 |
| 64 | B33 | 0.123144254 | 0.123144254 | 是 |
| 64 | Fx | 1.126694022 | 1.126694022 | 是 |
| 64 | Fz | 0.497503798 | 0.497503798 | 是 |
| 256 | A11 | 0.251414111 | 0.251414111 | 是 |
| 256 | B11 | 0.334732491 | 0.334732491 | 是 |
| 256 | A33 | 0.380435291 | 0.380435291 | 是 |
| 256 | B33 | 0.107966231 | 0.107966231 | 是 |
| 256 | Fx | 1.178996067 | 1.178996067 | 是 |
| 256 | Fz | 0.473493450 | 0.473493450 | 是 |
| 1024 | A11 | 0.249848452 | 0.249848452 | 是 |
| 1024 | B11 | 0.340431591 | 0.340431591 | 是 |
| 1024 | A33 | 0.386336589 | 0.386336589 | 是 |
| 1024 | B33 | 0.104248555 | 0.104248555 | 是 |
| 1024 | Fx | 1.192885382 | 1.192885382 | 是 |
| 1024 | Fz | 0.466884989 | 0.466884989 | 是 |

### 3.2 每档的自证来源（E0 台账）

| 面板数 | host | numpy | 求解驱动 | 平稳性 ||A^H r||/(||A||||r||) | 最坏求解残差 | solve_warnings |
|---|---|---|---|---|---|---|
| 64 | ec6ae0b42a0c | 2.4.6 | lstsq | 5.38e-14 | 9.622e-03 | 0 条 |
| 256 | ec6ae0b42a0c | 2.4.6 | lstsq | 1.15e-13 | 2.116e-03 | 0 条 |
| 1024 | ec6ae0b42a0c | 2.4.6 | lstsq | 4.36e-13 | 3.240e-04 | 0 条 |

E0：`pass = True`，记录到残差的档数 3/3，被剔除 0，unknown-host 档数 **0**（升级前为 3）。

### 3.3 只差第 4 档：E7/E9 的当前读数（honest FAIL 保持不变）

| 判据 | 实测 | 裁定 |
|---|---|---|
| `E0_rungs_solver_healthy` | 0.962% | PASS |
| `E8_inhouse_extrapolated_error` | 0.120% | PASS |
| `E7_inhouse_model_credible` | 0.000% | FAIL |
| `E9_inhouse_order_and_limit_stable` | n/a | FAIL |

E7 的 `n_rungs = 3`，`fit_identified = False`：三档恰好决定三参数模型，
残差恒为零，**不构成证据**——这一点本单没有放行，E7/E9 维持 FAIL。

## ④ 与参考解的逐量误差（三档 + 外推极限）

| 量 | v(64) | v(256) | v(1024) 误差（算例判据） | v_inf 误差 | 观测阶 p |
|---|---|---|---|---|---|
| A11 | 0.255802 | 0.251414 | +0.23% | +0.12% | 0.743 |
| B11 | 0.313958 | 0.334732 | +0.58% | +0.05% | 0.933 |
| A33 | 0.358460 | 0.380435 | +0.53% | +0.03% | 0.948 |
| B33 | 0.123144 | 0.107966 | +1.15% | +0.02% | 1.015 |
| Fx | 1.126694 | 1.178996 | +0.40% | +0.02% | 0.956 |
| Fz | 0.497504 | 0.473493 | +0.49% | +0.05% | 0.931 |

（`graded_error` = 1024 面元档相对 Hulme 级数参考；`extrapolated_error` = 三档外推极限相对参考。）

## ⑤ 注意点与局限

- **本单没有增加档数**：4096 面元档的重跑已递交计算门（作业 `cj4738`），但本单会话内该作业始终处于
  `queued`（队列 152 个在跑作业），未取回载荷。按 m21334 的纪律，E7/E9 不因「看起来快收敛了」而放行。
- **逐位一致不等于物理正确**：它只证明「同一环境下同一档可复现」，正是 m20515 发现 2× 漂移时缺失的那一层。
- **平稳性判据只排除一类缺陷**：它证明解是所给方程组的最小二乘驻点，不证明方程组本身对。
- **来源仍不跨环境**：三档全部来自控制面；节点侧的 4096 档（若到达）会以 `host` 不同的样子出现在台账里，
  E0 会把混合来源显式披露（`mixed_provenance.is_mixed`），不会被平滑掉。

## ⑥ 误差判定（是否 <3%）

| 关键量 | 值 | 相对参考误差 | <3%? |
|---|---|---|---|
| B33（1024 面元，算例判据所用） | 0.104249 | +1.15% | ✅ |
| A11（三档外推极限） | 0.248980 | +0.12% | ✅ |

两行均 **< 3%**：算例 5% 带内，且外推极限与参考解一致到 0.12% 以内——本仓求解器在 ka=2 无系统偏置。
第 4 档缺失只影响 E7/E9 的**模型可辨识性**，不影响已发布判据 G1 的数值结论。

## ⑦ 复现

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 python scripts/hulme_hemisphere_inhouse_ladder.py \
    --ka 2.0 --subdivisions 2 3 4 \
    --out benchmarks/verified/casepool_bem_hulme_hemisphere/inhouse_ladder_ka2_ctrl_rerun.json
PYTHONPATH=src python -m tensorbem.benchmarks.hemisphere_independent_convergence
PYTHONPATH=src python -m pytest tests/test_casepool_hulme_independent_convergence.py -q   # 18 passed
```

