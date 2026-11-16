# XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN — 参考侧截断误差与「预登记阶数下限」契约修复（m23018）

**锚点**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`（capytaine 的 `finite_depth_crossval_plan` 做法）· 仓 TensorBEM · 分支 `agent/m23018`
**日期**: 2026-11-16
**开工核对**: `feature_ledger` 该锚 `status=open`，`acceptance=<15%`；前序单（m22103/m21322/m22843/m22902）已完成计划模块、消费端、参考冻结、负控腿。**12 格 TensorBEM 梯仍在飞（`cj4422`，11-16 03:32 递交，shard 模式）**——本单不重复该计算，只动**参考侧（capytaine）与判定规则**。

## 1. 为什么还欠账：冻结结果里有一处自相矛盾的证据链

`benchmarks/verified/capytaine_finite_depth/results.json` 的 gates 里，
`reference_ladder_in_asymptotic_range`（限 0.3）判 **FAIL**，同时 `bundle` 的
headline 偏差用的却是**外推值**。计划模块把规则写死过：

```python
# finite_depth_crossval_plan.py
#: Minimum sensible observed order; below this the Richardson estimate is not
#: extrapolated ... the cell is reported through its finest-level difference only.
MIN_OBSERVED_ORDER = 0.3
```

而 `richardson()` 只要三点单调就无条件外推，`MIN_OBSERVED_ORDER` **只**用在 gate 里。
于是「判据说梯级不在渐近区」和「headline 用了该梯级的外推值」同时出现在同一份
结果里——这是契约违背，不是口径问题。

**量化（只用冻结参考，零求解器调用，48 个量）**：

| 统计 | 值 |
|---|---:|
| 参考量总数 | 48 |
| 测得阶数 < 0.3（预登记下限） | **11 / 48** |
| 测得阶数范围 | 0.151 … 0.367 |
| 被禁止的外推步长（外推值 vs 最细档）范围 | 0 … 5.25 % |
| 最大「外推步长 / 末档变化」比 | **4.50×** |

最后一行是关键：外推把值挪到末档实测变化的 4.5 倍以外，是典型的**非渐近区拟合放大**。
最干净的反例（`h/a=3.0, k0a=0.50`, `surge.added_mass`，阶数 0.241）：

| 口径 | TensorBEM | capytaine | 相对差 |
|---|---:|---:|---:|
| 无条件外推（旧 headline） | 2545.37 | 2461.59 | **+3.40 %**（>3 % 线，假 FAIL） |
| 预登记口径（参考不外推，用最细档） | 2545.37 | 2566.49 | **−0.82 %**（<3 %，PASS） |

同一个格、同两个求解器，**符号翻转**并跨界 3 % 预登记带：被禁止的那一步恰好是判据的
决定项。这就是「阶数下限」必须落到代码里、而不是留在注释里的原因。

## 2. 本单做了什么（4 项，全部可验证）

1. **`richardson()` 执行预登记阶数下限**（`min_order=plan.MIN_OBSERVED_ORDER`）：
   测得阶数低于下限时**不外推**，取值回落到最细档，`extrapolated=False`，
   并把被拒绝的拟合作为**可审计的 provenance** 保留：
   `blocked_by_order` / `blocked_value` / `blocked_shift` / `last_rung_change`。
2. **`compare()` 增加 `relative_basis`** 与 `relative_if_extrapolated`：
   headline `relative` 走预登记口径；旧的无条件外推数字仍保留，便于审计，但不再当判据。
3. **新增门 `reference_truncation_within_share`**：用「参考梯末档变化」作为参考自身
   剩余离散误差的**保守上界**（对幂律单调收敛，残余误差 = 末档变化/(r^p−1) < 末档变化），
   与 `plan.extrapolation_limit(quantity)`（容差的 1/3）比较。冻结参考实测：
   | 量 | 末档变化（参考截断上界） | 允许份额 | 结论 |
   |---|---:|---:|---|
   | added_mass | 1.623 % | 1.000 % | 超 |
   | damping | **1.808 %** | 1.667 % | 超 |
   → **参考梯（288/1152/4608 面板）本身已吃掉容差份额之外的部分**：3 %/5 % 这两条预登记带
   在当前参考分辨率下**不可判**；owner 15 % 线仍有充裕余量（1.8 % ≪ 15 %）。
4. **两处 PARTIAL 态崩溃 / 假 PASS 修复**（冻结件可自判）：
   - `compare()` 对「有键无档」（分片梯的初始态）抛 `IndexError` → 现记录
     `cells_without_ladder` 并跳过；`write_figures()` 同理不再对空档求值。
   - `added_mass_agreement`/`damping_agreement` 在**零判定格**时曾按
     `max(0.0, 0.0)=0.0 ≤ 限值` 记 **PASS**（空研究被发布成「一致」）→ 现为 `None/False`。
   - 新增可复用工具 `scripts/rebuild_finite_depth_verdict.py`：**不调用求解器**，
     用冻结的 `capytaine_ref.json` + `tensor_ladder.json` 重算 verdict/gates/页面/图，
     `--verify-only` 可与 `results.json` 对账。冻结 verdict 因此可随时自判。

## 3. 重算后的冻结判决（程序化，零手抄）

`PYTHONPATH=src python scripts/rebuild_finite_depth_verdict.py`

`execution_status = PARTIAL: 11 covered cells missing rungs ['0','1','2']; 4 confirmation cells missing the 3-subdivision rung`

| gate | 实测 | 限 | 判定 |
|---|---:|---:|---|
| `added_mass_agreement` | 0.0901 | 0.03 | FAIL（PARTIAL：另一格仅 2 档） |
| `damping_agreement` | 0.2205 | 0.05 | FAIL（同上） |
| `negative_control_separates_depth` | 0.1705 | 0.02 | PASS |
| `reference_ladder_in_asymptotic_range` | 0.2412 | 0.3 | **FAIL**（参考梯非渐近） |
| `reference_truncation_within_share` | 0.0181 | 0.0167 | **FAIL**（参考截断占超份额） |
| `tensorbem_ladder_in_asymptotic_range` | 0.6141 | 0.3 | PASS（唯一三级齐格，阶数 0.61–0.78） |
| `extrapolation_self_consistent` | n/a | 1.0 | FAIL（无四档，band 未知≠0） |
| `linear_solve_residuals` | 4.2e-16 | 1e-8 | PASS |

## 4. 误差判定（逐项，显式回答「是否 <3%」）

| 关键量（预登记口径，`h/a=3.0, k0a=0.50`，TensorBEM 三级外推） | TensorBEM | capytaine（最细档 4608） | 相对差 | <3%? | <15%? |
|---|---:|---:|---:|---|---|
| surge added mass | 2545.37 | 2566.49 | −0.82 % | ✅ | ✅ |
| heave added mass | 2998.96 | 2999.91 | −0.03 % | ✅ | ✅ |
| surge damping | 814.38 | 797.41 | +2.13 % | ✅ | ✅ |
| heave damping | 765.98 | 740.90 | +3.38 % | ❌ | ✅ |

**显式回答**：唯一三级齐格的 4 个量里 **3/4 < 3 %**，heave damping +3.38 % 超 3 % 但在
owner 15 % 线内；**但这一结论仍不是本锚的最终判定**——参考侧截断（1.6–1.8 %）与
4.50× 的非渐近外推史说明 3 % 带目前**不可判**，只能判 15 % 线。第二格
（`h/a=3.0, k0a=1.00`）只有 2 档 TensorBEM，−6.2 %/−21.7 % 这类数字是**单档差**，
不构成判定（`_partial_run` 的 6 个 XFAIL 保持，不覆盖）。

## 5. 注意点与局限

1. **外推不是免费的**：低阶梯上做 Richardson 会把值推到末档变化的数倍之外且能翻转符号。
   本单把「预登记阶数下限」变成代码行为，旧数字仍可审计（`relative_if_extrapolated`）。
2. **参考侧也需要误差预算**：crossval 的判定力受限于**参考自身**的截断。用末档变化作保守
   上界（幂律单调收敛下严格成立）比「外推残差」更稳，因为它不依赖阶数估计。
3. **参考能否补救是测量问题**：按测得阶数 p≈0.24 反推，把末档变化从 1.8 % 压到 1.0 % 需要
   面板数 ×(1.8)^(1/0.24) ≈ ×11（即 ~5 万面板，稠密 LU 不可行）。已递交 4 档探针
   `cj4526`（`scripts/capytaine_finite_depth_resolution_probe.py`，(96,192)=18432 面板，
   单格 `h/a=3.0, k0a=0.50`），用于**直接测量**阶数是否随加密上升；本单交付时其在排队。
   结果回填后应回答：参考梯可救（阶数上升→加第 4 档）或不可救（→ 改用解析/半解析有限水深
   锚，或把本锚的判定线明确钉在 owner 15 %）。
4. **本单没有动 TensorBEM 求解侧**：`cj4422` 在飞的 12 格梯回填后，用
   `scripts/rebuild_finite_depth_verdict.py` 一条命令即可重判，无需重跑求解器。

## 6. 复现

```bash
# 判定重算（不调求解器，<1 s）
PYTHONPATH=src python3 scripts/rebuild_finite_depth_verdict.py
# 与本单交付的 results.json 对账
PYTHONPATH=src python3 scripts/rebuild_finite_depth_verdict.py --verify-only
# 测试（含新增 6 条：阶数下限、口径记录、PARTIAL 不崩、截断门）
PYTHONPATH=src python3 -m pytest tests/test_capytaine_finite_depth_crossval.py -q
#  -> 24 passed, 6 xfailed
```

## 7. 给下一次自找活的线索

1. **`cj4526` 探针回填**：若 (96,192) 阶数仍 <0.3，则本锚的参考策略应换成
   「有限水深多极子解析锚」（仓内目前只有深水多极子，`bem_finite_depth_anchors.py` 仅 h/a=20 近似深水），
   这是该锚真正的收口路径。
2. **把 `reference_truncation_within_share` 推广**：其他 crossval（`capytaine_crosscheck*`、
   `barge_crosscheck`、`cross_solver_regression`）是否也缺参考侧误差预算——同一门可直接搬。
3. **判定可判性（decidability）应成为 crossval 的显式输出**：本单只加了单门；
   更彻底的做法是让 `compare()` 每格输出 `decidable_at(tolerance)`。