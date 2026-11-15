# m22360 — CASE-BEM-CUMMINS-WIGLEY：把「被污染的证据」变成 C0 的永久负控，并让重判不再抛异常

**锚点**：feature ledger `CASE-BEM-CUMMINS-WIGLEY`（算例池 id=145，Wigley 船 Cummins 记忆核 6DOF，tol <10%）。
**任务来源**：TensorBEM#1040 已裁定关闭后重开第 2 轮（管到底）；功能要点为 m20021 的成熟交付物。
**本单定位**：主功能（medium 网格、`save_sweep`/`load_sweep`/`gates_from_artifact`/`--replay`）已在 main；
本单不重做，而是**核验其可重判性**，并把核验中发现的两处真实缺陷修掉。

## 1 案例介绍（物理背景 / 几何 / 参数）

Wigley 数学船型（抛物线型），`Cb = 4/9`、`L/B = 10`、`B/T = 1.6`，`L=10 m`、`B=1 m`、`T=0.625 m`。
频域辐射问题给出每自由度 `A(ω)`、`B(ω)`，Cummins 时域记忆核 `K(t) = (2/π)∫₀^∞ B(ω)cos(ωt)dω`，
因果性要求 `A(ω) − A_∞` 是 `B(ω)` 的 Kramers–Kronig 变换。本算例把「几何 → 面板分辨率 →
KK 闭合 → B→K→B 往返 → 自由衰减」串成可机裁的闸门链 C1..C7（+ C5b/C5c/C6a/C6b/C10）。

| 参数 | 值 |
| --- | --- |
| 网格 coarse / medium | 12×4 = 192 面元 / 20×6 = 480 面元 |
| 频带 / 节点 | 0.2–12 rad/s，41 个对数节点 |
| ρ / g | 1025 kg/m³ / 9.80665 m/s² |
| 不规则频率抑制 | CHIEF `chief_points="auto"` |
| 自由衰减 | heave，dt = 0.002 s，40 s，τ_max = 25 s |

## 2 怎么做的（方法 / 流程 / 关键设置 / 工具链）

本单的决定是：**先判定「哪些 committed 扫频是证据」**，再谈精度。因为本算例的判据全部是
「扫频系数 + 船体常数」的函数，同一算例在树里存在两份**网格身份与频带完全相同**的工件：

* `benchmarks/verified/casepool_bem_cummins_wigley/sweep_{coarse,medium}.npz` —— 发布判据所用（m21960）；
* `docs/m20021/sweep_{coarse,medium}.npz` —— m20021 medium 网格尝试，报告记录 `min B = −129.9`。

1. `docs/m22360/build_evidence.py` 把两份工件逐频率对拍，算出残差/负阻尼/逐点偏差，写入
   `provenance_audit.json`，并画 `fig_medium_evidence.png`（三面板，程序化绘制，零手抄）。
2. 两份工件都过 `gates_from_artifact()` 重判（BEM 不再解一次）。
3. `reproduction_spotcheck.json`：在当前树上用 `_solve_one` 独立重解 8 个节点（两网格 × 4 频率），
   与 committed 健康工件逐位比对，验证「工件可复现」而不是「报告可信」。

**发现并修复的两处缺陷（本单代码增量）**

* **缺陷 1（重判路径抛异常）**：`gates_from_artifact` 无条件进入 C2..C5c，被负阻尼守卫
  `validate_radiation_sweep` 抛 `ValueError` 打断 —— 即**被污染工件根本无法被重判**，恰恰摧毁了
  「证据可重判」这个能力本身。修复：把 `run_case` 已有的 C0 政策提取为
  `C0_DEPENDENT_GATES` + `_voided_gate()`，重判路径与运行路径**共用**；C0 FAIL 时下游闸门记为
  `voided`（`voided_by = "C0"` + 原因），返回 `verdict` FAIL，不抛异常。
* **缺陷 2（失败运行写不出文档）**：`render_markdown` 假定 `C3.values.per_dof` 存在，C0 FAIL 时
  `KeyError: 'per_dof'` 崩在 `CASE.md` 写出之前 —— 这正是 m20021「驱动中止，没有 case_result.json /
  CASE.md」的机制性原因（留下两个 npz、没有一个判据）。修复：`_sweep_voided()` 判定 +
  `_voided_lines()` 渲染「C0 未通过、下游闸门作废」的第三节，判据与复现节照常输出
  （判据/复现两节抽成 `_verdict_and_reproduction_lines()` 供两条路径共用，不复制文本）。
* **附带**：`benchmarks/verified/.../CASE.md` 与其 `case_result.json` 已漂移（文档写 15/15，JSON 是
  16/16，缺 C5c 行）。用同一 `render_markdown` 从 JSON 重新生成，并新增测试把「CASE.md == render(JSON)」
  钉住，避免再生出第二份事实源。

## 3 计算结果

### 3.1 两份工件的对拍（`provenance_audit.json`）

| 网格 | 工件 | 残差中位 | 残差最大 | 残差 > 1e-2 的节点 | min diag B | 负对角节点 | min B / 峰值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coarse | healthy | 2.19e-4 | 6.39e-3 | 0/41 | +2.70e-13 | 0 | +2.9e-16 |
| coarse | m20021 | 0.610 | 1.25 | 41/41 | −1.463 | 39 | −3.5e-3 |
| medium | healthy | 4.52e-5 | 2.45e-3 | 0/41 | +2.53e-13 | 0 | +2.8e-16 |
| medium | m20021 | 0.546 | 1.245 | 41/41 | −129.917 | 26 | −0.236 |

网格身份与频带**完全一致**（`panel_count` 480、`max_panel_scale` 0.0645、41 节点全等）：
差别只在求解结果。最坏偏差在 ω = 5.2912 rad/s，`B33` 由 +305.19 变成 −129.92（相对偏差 142.6%，
符号翻转）；中位偏差 2.46%。→ 结论：`docs/m20021/sweep_*.npz` **不是**物理证据，是失败的线性求解。

### 3.2 重判（`gates_from_artifact`，两条路径同一政策）

| 工件 | C0 | 下游闸门 | 通过 / 总数 |
| --- | --- | --- | --- |
| healthy medium | PASS（残差 2.45e-3 ≤ 1e-2，min B ≥ 0） | 全部评分 | **13 / 13 PASS** |
| m20021 medium | **FAIL**（残差 1.245，min B = −129.9） | C2,C3,C3a,C7,C4,C5,C6a,C6b,C5b,C5c 全部 voided | 1 / 12（仅 C1 几何） |

发布判据（运行路径，`case_result.json`，含 C10 capytaine 外部锚点与 C8/C9 扩展带）：**16 / 16 PASS**。

### 3.3 关键闸门数值（healthy medium，重判得到）

| 闸门 | 量 | 值 | 判据 |
| --- | --- | --- | --- |
| C0 | max 求解相对残差 | 2.447e-3 | ≤ 1e-2 |
| C2 | k·√A_panel,max @12 rad/s | 0.9471 | ≤ 1 |
| C3 | KK 闭合最差自由度相对残差 | 3.598% | ≤ 10% |
| C3a | 同一闭合、零自由参数 | 8.853% | ≤ 10% |
| C5 | heave 自由衰减阻尼比 vs 频域 | 3.960%（ζ = 0.13460） | ≤ 5% |
| C5c | roll 多窗一致 ζ | 0.463% | ≤ 5% |
| C6b | 高频段故意破坏后 C3 | 20.64% | 必须 > 10%（负控咬得住） |

### 3.4 可复现性（`reproduction_spotcheck.json`）

8 个节点（coarse/medium × ω = 1.03/2.86/5.29/10.83）在当前树、BLAS 单线程下重解，与 committed
健康工件**逐位相同**：`max |ΔB33|/|B33| = 0.0`、`max |Δdiag|/scale = 0.0`、残差亦相同。

### 3.5 图

`docs/m22360/fig_medium_evidence.png`（`build_evidence.py` 生成）：
左 = medium 网格 heave 阻尼 `B33(ω)`，健康 vs m20021（symlog，可见 5.29 rad/s 处的负瓣）；
中 = 每频率线性系统相对残差（含 C0 判据线 1e-2，可见 m20021 全频段 O(1)、健康全频段低两个量级）；
右 = m20021 对健康的逐点相对偏差（log）。

## 4 注意点 / 局限

* 污染是**不可在本机复现**的：同一代码、同一网格/频带，本机（BLAS 1 线程）给出 2.4e-3 的残差。
  因此本单给的是**判定与围栏**（C0 + voiding + 负控测试），不是根因修复；触发条件是那个节点上的
  求解路径（LAPACK/BLAS 线程化），本单没有能力定位到库级。
* `docs/m20021/*.npz` 保留不动，作为 C0 的**永久负控**（测试
  `test_committed_contaminated_sweep_is_kept_as_the_c0_negative_control` 现在会读它）。它的价值在于
  证明 C0 在真实工件上会咬，而不只是在合成数据上。
* C10（capytaine 外部锚点）与 C8/C9（扩展带）依赖另一批 committed 参考工件，本单未重跑；
  发布判据 16/16 引用 m21960 的 `case_result.json`。
* 本单**没有**重跑 82 个频域解来重新生成 `benchmarks/verified/` 下的工件：spot check 已证明逐位可复现，
  重跑只会重复同样的字节（算力留给了更需要的任务）。

## 5 误差判定（逐项回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%？ |
| --- | --- | --- | --- |
| committed 扫频复现（8 节点，B33/diag/残差） | 同一代码 • committed 工件 | **0.0%** | **是** |
| C5c roll 自由衰减 ζ（多窗一致） | 频域 `B/(2(M+A)ω_n)` | 0.463% | **是** |
| C3 KK 闭合最差自由度 | 直接 BEM `A(ω)` | **3.598%** | **否（>3%，但 < 算例判据 10%）** |
| C3a 零自由参数闭合 | 同上（A_∞ 由双层镜像锚点定） | **8.853%** | **否** |
| C5 heave 自由衰减 ζ | 频域阻尼比 | **3.960%** | **否** |
| C2 面板分辨率 k·√A | ≤ 1 的结构判据（非误差量） | 0.947 vs 1（余量 5.3%） | — |

结论：**算例判据（<10%）全绿，但 3% 这一更严的尺子没有全过**：KK 闭合最差自由度 3.60%、
零自由参数形式 8.85%、heave 衰减 3.96% 都在 3% 之外。这三项都对「频带上端截断 + 面板分辨率
余量仅 5.3%」敏感，属已知局限，不作粉饰。参考值是**另一条数值路线**（直接 BEM、capytaine），
Wigley 辐射系数无闭式解、树内亦无可引用实验表，故 L5 外部实验锚点仍记 WITHHELD。

## 6 复现

```
# 证据对拍 + 图（无需 BEM 解，本机 < 2 min）
PYTHONPATH=src python3 docs/m22360/build_evidence.py

# 重判 committed 健康工件（不重解 BEM）
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_cummins_wigley \
  --replay benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz

# 重判 m20021 污染工件（应得 C0 FAIL + 下游 voided，不得抛异常）
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_cummins_wigley \
  --replay docs/m20021/sweep_medium.npz

# 测试（本单相关用例）
PYTHONPATH=src python3 -m pytest tests/test_case_bem_cummins_wigley.py -q
```

**测试执行记录**：本单在工作树内跑了 `tests/test_case_bem_cummins_wigley.py` 的 61/63 条：
47 条 + 8 条 + 2 条（roll/committed_roll）+ 新增 5 条均通过；剩下 2 条「生产工件重判」用例单条需 ~62 s
（含双层镜像 A_∞ 锚点求解），在本机 120 s 上限下成对超时，已递交集群 `cj4075`（该文件 +
`tests/test_time_domain.py` 全量）与 `cj4076`（`--out docs/m22360/case_run` 全流程重跑）排队执行。
等价的数值已在 `build_evidence.py` 内独立跑通（healthy 重判 13/13 PASS）。