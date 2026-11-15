# CASE-BEM-CUMMINS-WIGLEY — C5c 闸门独立审计与失效注入补齐（m22091）

**锚点**：`CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核（6DOF））
**仓**：TensorBEM ｜ **审计对象**：`src/tensorbem/benchmarks/case_bem_cummins_wigley.py` 的 C5c 闸门
（m22005 引入）与其提交工件 `benchmarks/verified/casepool_bem_cummins_wigley/case_result.json`
**判据来源**：Cummins (1962) 时域卷积 + Kramers-Kronig 因果对，容差 <10%（C5c 内部用 5%）

本单是**验证欠账**类工作：C5c 是上一轮为了「救回」roll 模态而新增的闸门，整个算例的
16/16 闭合结论直接依赖它。一个只被观察到 pass 的闸门与图章无法区分，因此本单对该闸门
做独立审计 + 失效注入（negative control），并把「文档声明的判据」与「代码实际执行的判据」
对齐。

---

## 1 审计范围与独立性限制

**做了**：独立重放执行路径、逐分支失效注入、对已提交工件复算、mutation check 证明测试有牙。
**未做（诚实声明）**：

- 未调用外部独立求解器（Capytaine）重算 `B(ω)`。C5c 的两条路线都源自**同一组** `B(ω)`
  系数，因此 C5c 是**自洽性**检查，不是对 `B(ω)` 精度的独立验证。本审计无法、也未试图
  证明 `B(ω)` 本身准确。
- 未做网格/频带收敛重算（属重计算，本机 load15 已 ~78–99，全部递交集群）。
- 未复核 C5b/C5 之外的闸门判据（沿用前轮）。

**独立性限制**：本单作者同时是修复者，存在「作者自证」偏置；缓解手段是全部结论都绑定到
可复算的工件数字与 mutation check（见 §4），而非文字推断。

---

## 2 执行路径重建（C5c 到底算了什么）

`robust_decay_gate(sweep, dof=3)`：

1. `oscillator_properties(3)` → roll 转动惯量 `I = 0.81163 kg·m²`、恢复刚度 `k = 48.686 N·m/rad`。
2. `kk_added_mass_closure` → `A∞ = 0.38492 kg·m²`。
3. `radiation_memory_kernel(ω, B[3,3])` → 记忆核 `K(t)`（由 `B(ω)` 余弦变换）。
4. `cummins_free_decay(K, mass=I, restoring=k, added_mass_inf=A∞)` → 时域衰减记录。
5. `_decay_zeta_multi_window` → 在 1% / 0.1% / 0.01% 三个振幅地板上各取对数衰减率，取中位数。
6. 频域路线：在共识自然频率 `ω_n` 处 `ζ_fd = B(ω_n) / [2 (I + A(ω_n)) ω_n]`（对扫频网格插值）。
7. **判定**：`in_band` 且 `|ζ_consensus − ζ_fd| / ζ_fd ≤ 5%` 且合格窗口数 ≥ 2 → `decided`。

关键结构性事实：第 4 步的 `K(t)` 与第 6 步的 `B(ω_n)` **来自同一个数组**
`sweep.damping[:, 3, 3]`。所以两路一致 = 两条不同**归约**（时域卷积 / 频域阻抗）自洽，
不构成对输入系数的外部校准。这一点在 `TOLERANCES["free_decay"]` 的注释里有披露，但在
讲「0.46% 误差」的报告叙事里容易被读成精度证明。

---

## 3 发现的缺陷（按严重度）

### F1（高·已修）文档声明的判据 ≠ 代码执行的判据

`robust_decay_gate` 的 docstring 原文声明模态 `decided` 需同时满足：

> - at least two floors yield >= 5 peaks and finite zeta, AND
> - **the spread of zeta across qualifying floors <= the case bar (5 %)**, AND
> - the consensus zeta agrees with the frequency-domain estimate within the bar.

而实际判定代码（修复前）只要求 `in_band and relative <= bar and n_qualifying >= 2`，
**spread 从未参与判定**。也就是说：任何人按 docstring 审计代码，都会相信存在一条收敛性判据，
而它并不存在。这正是「判据被悄悄放宽」的典型形态——即使放宽在物理上是可辩护的，
**未披露的放宽**本身就是缺陷。

m22005 的报告确实写了「spread 作为诊断量报告，不作为判据」，但那份说明只存在于 700+ 份
dev_report 中的一份里，不在代码契约里。

**修复**：把 docstring 改写为如实描述**实际执行**的判据，并补充物理论证（为何不强制
spread），同时在闸门 values 中加两个显式披露字段：
`convergence_claimed`、`non_exponential_decay`，使 pass 结论**无法**被读成「窗口收敛」。

### F2（中·已修）偶数窗口数时「中位数」取的是上分位

修复前：

```python
sorted_z = sorted(qualifying)
median_z = sorted_z[len(sorted_z) // 2]      # 2 个样本时 = 较大的那个
```

当恰好两个窗口合格时，所谓「共识」等于**较大的** ζ。对单调漂移的窗口序列（本算例正是如此），
这系统性**上偏**。修复为真中位数（`statistics.median`，偶数取两中值均值）。

本算例 roll 有 3 个合格窗口，故该修复不改变已提交数字——但这是一条潜伏缺陷。

### F3（中·已修）单个合格窗口被报告为 `converged=True, spread=0.0`

修复前 `len(qualifying) == 1` 时返回 `converged=True, spread=0.0`，即**声称**窗口一致。
只有一个窗口时不存在「一致」，`spread=0.0` 是假陈述。

外层判定有 `n_qualifying >= 2` 兜底，故不会误判 pass；但该 dict 是公开返回值，
任何下游消费者读 `converged`/`spread` 都会被误导。修复为 `converged=False, spread=nan`。

### F4（限制·记录，非缺陷）C5c 是自洽性检查，不是精度证明

见 §2。建议今后报告把 C5c 的 0.46% 表述为「两条归约路线自洽（self-consistency）」，
独立的精度证据仍只能来自 C10（Capytaine 外部锚点）与 C3（KK 因果闭合）。

### F5（观察·无需动作）roll 窗口是单调漂移，不是散点噪声

已提交工件：

| 振幅地板 | ζ | 峰数 |
|---|---|---|
| 1 % | 0.09012 | 8 |
| 0.1 % | 0.07115 | 15 |
| 0.01 % | 0.05005 | 28 |

spread = 56.3%，单调递减。这是**非单指数衰减**（记忆核卷积）的物理签名：看得越长，
表观衰减率越低。因此「共识 = 三者的中位数」在统计上就是「趋势的中点」，其与 ζ_fd 的
0.46% 吻合是**两条归约在该 ω_n 处相交**，不应当被解释为估计器收敛。闸门现在通过
`convergence_claimed=False` 显式披露这一点。

> 一条留给下次的线索（本单未做）：既然衰减非单指数，更物理的 decay-side 估计量是在
> **最长窗口**上做 Prony/矩阵铅笔拟合并取最靠近虚轴的极点，而不是对窗口取中位数；
> 那将给出可与 ζ_fd 独立比较的量，并可直接检验「0.050 还是 0.0711 更接近真值」。

---

## 4 证据与复现

**修复后复算已提交工件**（`sweep_medium.npz`，dof=3），逐字段与提交值比对：

| 量 | 修复前（提交值） | 修复后复算 | 是否变化 |
|---|---|---|---|
| `zeta_decay_consensus` | 0.07114851188893904 | 同值 | 否 |
| `zeta_frequency_domain` | 0.0714875… | 同值 | 否 |
| `relative_error` | 0.0046… | 同值 | 否 |
| `status` | `decided` | `decided` | 否 |
| `multi_window.converged` | `false` | `false` | 否 |
| `multi_window.spread` | 0.5632464372466132 | 同值 | 否 |
| `convergence_claimed` | （不存在） | `false` | **新增披露** |
| `non_exponential_decay` | （不存在） | `true` | **新增披露** |

复算脚本断言了上表全部数值审计前后严格相等（`==`，非近似），仅追加披露字段。

**mutation check**（证明新测试有牙）：把聚合逻辑回退到修复前语义后，
`test_multi_window_even_count_uses_true_median` 与
`test_multi_window_single_window_is_not_a_consensus` **均 FAILED**；恢复修复版后**均 passed**。

**失效注入覆盖**（`tests/test_case_bem_cummins_wigley.py`，全部 <1.5 s，无需 BEM 求解）：

| 分支 | 注入 | 期望 |
|---|---|---|
| 正常 | in_band, rel=0, 3 窗口 | `decided` / pass |
| 边界 | rel == bar | pass（闭区间） |
| 越界 | rel = bar×(1+1e−9) | `beyond_bar` / fail |
| 不一致 | rel = 0.30 | `beyond_bar` / fail |
| 带外 | in_band=False | `undecidable: … outside computed band` / fail |
| 单窗口 | n_qualifying=1 | `undecidable: fewer than 2 …` / fail |
| 非有限 | rel = inf / nan | fail |

即：C5c 的**七条分支全部被证伪过一次**，它不是一个恒真的图章。

---

## 5 结论与处置

**disposition：approve with notes（闸门判据本身可辩护，但契约与实现不一致，已修复）。**

- 算例 16/16 的闭合结论**数值上不变**：修复不动已提交的 roll 数字与判定。
- 但闭合结论的**可审计性**改变：现在从代码 docstring、闸门 values、测试三处都能读到
  「跨窗口未收敛而判定仍为 pass」这一事实，不依赖某一份 dev_report 的叙事。
- **不宣称**本单提升了对 `B(ω)` 精度的信心；C5c 的 0.46% 是自洽性数字。

**未跑的检查（标 unknown，不标 pass）**：外部求解器交叉验证、网格/频带收敛重算、
Prony/矩阵铅笔 decay-side 估计量。