# 有限水深交叉验证的**可判性**契约与参考侧真值审计（m23144）

**锚点**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`（capytaine 的 `finite_depth_crossval_plan` 做法）· 仓 TensorBEM · 分支 `agent/m23144`

## 1. 案例介绍

三维有限水深辐射问题：全潜球 `a = 1.0 m`、球心 `z = -1.5 m`（h/a=3 时离底 0.5 m），水深网格 `h/a in [3.0, 5.0, 10.0, 20.0]` × `k0 a in [0.5, 1.0, 2.0]`，`nu` 由 `k0 tanh(k0 h) = nu` 逐格反解。两条数值路线（TensorBEM 本征级数 BEM / capytaine 波数积分 BEM）互为参考。

本单要回答的不是「偏差多大」，而是**这些偏差在任何容差下是否可判**。

## 2. 怎么做的

### 2.1 可判性契约（新模块）

`tensorbem.benchmarks.crossval_decidability`：每个被判量给三值判决 `PASS / FAIL / UNDECIDABLE`，规则逐条落码——

| 规则 | 内容 |
|---|---|
| 1 | 幂律梯 A-posteriori 界 `u = |Δ末档|/(r^p − 1)`（无需额外计算）|
| 2 | 外推**不免费**：四档 drop-coarsest 实测带可替代模型界；无带则沿用末档界 |
| 3 | 阶数缺失或低于预登记下限 → 仅用末档变化，并标注 basis（弱结论，不是小结论）|
| 4 | 预算 = 容差的 1/3，两侧**线性**相加（保守）|
| 5 | 未知≠零：任一侧界缺失 ⇒ `UNDECIDABLE`，永不 PASS；但下界已超容差时 FAIL 仍稳健 |
| 6 | 模型界乘 GCI 安全因子 1.25；实测量不乘 |

消费端：`compare()` 为每个 `cell×dof×quantity` 输出 `decidability` 与 `decidability_owner_limit`，并在 `comparison['decidability']` 汇总裁决；`scripts/rebuild_finite_depth_verdict.py` 零求解器重判。

### 2.2 参考侧真值审计（新证据）

`scripts/finite_depth_reference_truth_audit.py`：在海底影响可忽略的格上，深水 Thorne/Wang/Linton 多极子解是**精确**答案（零数值积分，Haskind 恒等式闭合到 `~1e-16`，截断阶 `n_terms=24 vs 32` 一致到 `1e-10` 已校核）。准入判据用**离底**指数而非 `k0 h`：`2 k0 (h − f − a) ≥ 4`。

这一步把「末档变化是参考误差保守上界」的假设变成可测量的量。

## 3. 计算结果

### 3.1 参考侧实测误差（对精确多极子锚，程序化注入）

| 格 `h/a` | `k0 a` | 离底指数 | heave.added_mass | heave.damping | surge.added_mass | surge.damping |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.00 | 5.0 | +3.82 % | +3.03 % | +1.78 % | +2.34 % |
| 5 | 2.00 | 10.0 | +3.49 % | +2.03 % | +1.92 % | +1.71 % |
| 10 | 0.50 | 7.5 | +1.82 % | +2.21 % | +1.89 % | +2.46 % |
| 10 | 1.00 | 15.0 | +1.44 % | +1.84 % | +1.75 % | +2.26 % |
| 10 | 2.00 | 30.0 | +1.56 % | +1.14 % | +1.77 % | +1.71 % |
| 20 | 0.50 | 17.5 | +1.55 % | +2.13 % | +1.88 % | +2.51 % |
| 20 | 1.00 | 35.0 | +1.26 % | +1.73 % | +1.75 % | +2.26 % |
| 20 | 2.00 | 70.0 | +1.39 % | +1.05 % | +1.76 % | +1.71 % |

准入 8 格 / 排除 4 格（`h_over_a=3.0|ka=0.50`, `h_over_a=3.0|ka=1.00`, `h_over_a=3.0|ka=2.00`, `h_over_a=5.0|ka=0.50`）。

### 3.2 「末档变化」代理的乐观程度

| 统计 | 值 |
|---|---:|
| 代理低估真值的比例 | 100 % （32/32）|
| 最大乐观倍数 | 3.71× |
| 中位乐观倍数 | 1.39× |
| 最大实测参考误差 | 3.82 % （`h_over_a=5.0|ka=1.00`, `heave.added_mass`）|

即 m23018 用「末档变化」当保守上界**在 32/32 个量上都偏乐观**（原始界 1.6–1.8 % vs 真值 1.05–3.82 %）。参考梯不是渐近收敛，误差在 1.5–2 % 形成平台（球面网格多面体化主导），因此本单把模型界统一乘 GCI 因子并允许实测误差**替代**模型界。

### 3.3 冻结工件（PARTIAL）的三值裁决

| 容差 | 判决计数 | 可判? |
|---|---|---|
| 预登记 3 % | PASS 0 / FAIL 0 / UNDECIDABLE 4 | **否** |
| 预登记 5 % | PASS 0 / FAIL 0 / UNDECIDABLE 4 | **否** |
| owner 15 % | PASS 0 / FAIL 0 / UNDECIDABLE 8 | **否** |

`execution_status = PARTIAL: 11 covered cells missing rungs ['0', '1', '2']; 4 confirmation cells missing the 3-subdivision rung`

原因：TensorBEM 侧仅 3 档（20/80/320 面板），末档界 3.7–12.7 %，无第四档实测带，外推值不可信用；参考侧模型界（含安全因子）同样超预算。**两侧都不足**，所以当前工件在任何容差下都不可判——不是「不达标」，而是「不可判」。

### 3.4 可判所需的分辨率（可执行处方）

| 侧 | 现界 | 目标 | 需要面板倍数 |
|---|---:|---:|---:|
| capytaine 参考（实测最差 3.82 %） | 3.82 % | ≤1.00 %（3 % 容差预算） | 未测阶数下无法外推；按 `p=0.24` 需 ≈×12 面板 |
| TensorBEM | 3.7–12.7 % | ≤1.00 % | 靠第 4 档 `drop-coarsest` 实测带（在飞 `cj4422`） |

结论性处方：**3 %/5 % 预登记带对当前 capytaine 参考不可达**（参考自身误差下限 1.05 % > 3 % 容差的 1/3 预算 1 %）；可达路径是把参考换成已就位的**精确多极子锚**（本单 `reference_truth` 通道已接好，`compare(..., reference_truth=...)` 落地即用），或在 `h/a ≥ 5` 以外的格子改用几何修正的 capytaine。

## 4. 注意点（踩过的坑）

1. **准入判据踩过一次坑**：最初按 `k0 h ≥ 6` 判「等效深水」，会把 `h/a=3, k0a=2` 当准入格——该格离底仅 0.5 m，镜像场指数 `2 k0 (h−f−a) = 2`（13 % 量级），深水锚根本不等价。改用离底指数后该格被正确排除。

2. **「保守上界」必须先测再信**：同一份冻结参考，末档变化在 32/32 个量上都小于真值。

3. **外推带是判决的钥匙**，不是装饰：没有第四档，TensorBEM 侧界只能取末档界，三值裁决就永远是 `UNDECIDABLE`。

4. 本单不重复在飞计算（`cj4422` 12 格梯 + 4 个确认格；`cj4526` 参考分辨率探针 18432 面板）。

## 5. 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 基准 | 结果 | <3 %? | <15 %? |
|---|---|---|---|---|
| 参考侧自身误差（8 格 × 4 量） | 精确多极子 | 1.05 % … 3.82 % | ❌ 无一个量 ≤1 % 预算 | ✅（均 <15 %）|
| TensorBEM vs capytaine 偏差（冻结格 `h/a=3, k0a=0.5`） | capytaine | −0.03 % / +3.38 % | **不可判**（两侧界 3.7–12.7 %） | **不可判** |

**显式回答**：本锚当前**没有任何一个量能判 <3%**——不是因为偏差大，而是因为两侧离散不确定度（参考 ≥1.05 %，求解侧 3.7–12.7 %）吃光了 3 % 容差的 1/3 预算。同样**不能判 <15%**（求解侧界超 5 % 预算）。诚实结论：`UNDECIDABLE`，而非 PASS/FAIL；这与 owner「>15 % 不可接受」并不冲突——15 % 线**尚未被检验**，而不是被违反。

## 6. 复现

```bash
# 参考侧真值审计（零求解器，秒级）
PYTHONPATH=src python3 scripts/finite_depth_reference_truth_audit.py
# 重判冻结工件（零求解器）
PYTHONPATH=src python3 scripts/rebuild_finite_depth_verdict.py
# 契约与消费端测试
PYTHONPATH=src:scripts python3 -m pytest tests/test_crossval_decidability.py tests/test_capytaine_finite_depth_crossval.py -q   # 40 passed, 6 xfailed
```

## 7. 给下一次自找活的线索

1. `cj4422` 落地后，用 `compare(..., reference_truth=...)` 在 8 个准入格上直接对**精确锚**判定 3 % 带——那时只剩 TensorBEM 侧的实测带一个未知量。
2. 同一可判性契约可搬到 `capytaine_crosscheck*` / `barge_crosscheck` / `cross_solver_regression`：它们的预登记带是否也不可达，是一个可批量回答的问题。
3. capytaine 参考在 1.5–2 % 处的平台更像**球面网格多面体化**而非 Green 函数误差：用 `mesh_sphere(..., radial_correction)` 或几何精确面积做一次对照，可把参考侧界压到 1 % 以内。
