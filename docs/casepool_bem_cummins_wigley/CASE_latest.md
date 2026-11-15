# CASE-BEM-CUMMINS-WIGLEY — 当前判据（m21100 提交件，10/10 全过）

## 1. 结论（一句话）

算例池锚点 `CASE-BEM-CUMMINS-WIGLEY`（Wigley 6DOF Cummins 记忆核，判据 Kramers-Kronig 闭合 <10%）
的**当前提交件**（`benchmarks/verified/casepool_bem_cummins_wigley/case_result.json`，mission m21100，
band 0.2–12 rad/s / 41 节点 / medium 20x6 生产网格）**10/10 闸门全过**。本文件是这份提交件的
最新渲染（由 `tools/rerender_casepool_bem_cummins_wigley.py` 从已提交 `sweep_*.npz` 重画图并重渲染，
不重解 BEM）。

## 2. 与历史欠账的关系（m20419 复算有效性）

* m20419 曾记录：提交件与代码不同步（旧 `case_result.json` 是 13 节点 / band 0.15–8 / coarse 网格），
  且节点 n315 对同一输入给出**未收敛解**（max residual 1.253、负阻尼）。本单补上有效性闸门 C0
  （`max residual <= 1e-2` 且 `min B >= 0` 才允许 C2–C6 发言）。
* **当前提交件已通过 C0**：`C0 solver residual max = 2.45e-3`（≤1e-2），`min B >= 0`（无负阻尼）。
  因此 C2–C6 的判据是**有效解**上的判据，不是把未收敛解当数据。
* C5（自由衰减 vs 频域阻尼比）在旧提交件（dt=0.01）上判 FAIL（相对误差 7.09%）；当前提交件
  用 **dt=0.002** 的时域积分重估，`relative_error = 3.96%`（≤5%）→ **PASS**。这是同一物理量
  在更细时间步上的收敛，不是换判据。

## 3. 当前闸门（全部 PASS）

| gate | criterion | threshold | passed |
| --- | --- | --- | --- |
| C1 | geometry precondition (Cb, L/B, B/T, displaced volume) | Cb within 1% of 4/9; volume within 2%; L/B=10; B/T=1.6 | PASS |
| C0 | solver validity: converged BIE solves and non-negative damping | max solver relative residual <= 0.01 and min B >= 0 | PASS |
| C0_coarse | solver validity (coarse mesh) | max solver relative residual <= 0.01 and min B >= 0 | PASS |
| C2 | panel resolution at the top of the band (medium mesh) | k*sqrt(A_panel,max) <= 1 | PASS |
| C3 | Kramers-Kronig closure of the added mass (6 DOF) | max over decidable samples and DOF |A_BEM-A_KK|/max|A_BEM| <= 10% | PASS |
| C3_coarse | KK closure (coarse mesh) | same as C3 | PASS |
| C4 | Cummins round trip B->K->B (heave) | max pointwise relative error <= 2% | PASS |
| C5 | Cummins free decay vs frequency-domain damping ratio (heave) | |zeta_decay - zeta_fd|/zeta_fd <= 5% | **PASS** |
| C6a | negative control: zero damping gives zero decay (heave) | max|K| <= 1e-12, zeta_decay <= 1e-3 | PASS |
| C6b | negative control: corrupted damping shape must fail the KK gate | max |A_BEM-A_KK|/max|A_BEM| <= 10% | PASS |

## 4. 关键量（C5 自由衰减，当前提交件）

| quantity | value |
| --- | --- |
| dof | 2 (heave) |
| zeta_decay | 0.1346 |
| zeta_frequency_domain | 0.1295 |
| relative_error | **3.96%** (≤5% → PASS) |
| natural_frequency_rad_s | 7.790 |
| added_mass_at_natural | 28.86 |
| damping_at_natural | 213.3 |
| restoring_n_per_m | 6016 |
| displaced_mass_kg | 76.88 |
| added_mass_inf_used | 50.93 (KK closure) |
| dt | **0.002** |
| duration | 40 s |

## 5. 误差判定（vs 基准）

* **C3（算例判据）**：逐 DOF KK 闭合残差 surge 2.62% / sway 2.81% / heave 0.30% / roll 0.72% /
  pitch 0.48% / yaw 3.60%，全部 **< 10%**（判据容差）。基准=直接 BEM 解（另一条路线），
  非实验/解析表（Wigley 辐射系数无闭式解，L5 外部锚点 WITHHELD）。
* **C4**：B->K->B 往返阻尼相对误差 0.087%（≤2%）。
* **C5**：自由衰减 vs 频域阻尼比相对误差 3.96%（≤5%）。
* **C6b 负控**：人为破坏 B 高频段形状（乘 1.3）后 C3 判 FAIL（worst 20.6% > 10%），证明闸门咬得住。

## 6. 复现

```
PYTHONPATH=src python -m tensorbem.benchmarks.case_bem_cummins_wigley --out benchmarks/verified/casepool_bem_cummins_wigley --jobs 1
```

（本文件由 `tools/rerender_casepool_bem_cummins_wigley.py` 从已提交 `sweep_*.npz` 重渲染，不重解 BEM。）
