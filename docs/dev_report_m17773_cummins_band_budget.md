# CASE-BEM-CUMMINS-WIGLEY round 3 — the half-power window defect, and measuring (not asserting) the KK band floor

Anchor: `CASE-BEM-CUMMINS-WIGLEY` (算例池·Wigley 船 Cummins 记忆核 6DOF; ref = Cummins equation + Kramers-Kronig 因果; tol < 10 %). Mission m17773, branch `agent/m17773`, worktree `/root/tp-agent/work/m17773`.

Round 2 left G3 (Kramers-Kronig, 53-81 %), G4 (free decay, `IndexError`), G5 (round trip, 22-53 %) failing and *explained* the residual in prose ("band/scale effect"). Round 3 does not claim those gates pass. It (a) removes a real defect that made one gate undecidable, and (b) replaces the prose with a measurement tool plus first numbers on an existing cached sweep.

## 1 案例介绍（物理背景/几何/参数）

Wigley III 船型（L=3.0, B=0.3, T=0.1875, Cb=0.543），密度 1025 kg/m³，g=9.80665 m/s²，无限水深，旋转中心 (0,0,−T/2)。Cummins 方程

```
(M + A_inf) x'' + ∫_0^t K(t-s) x'(s) ds + C x = F(t),   K(t) = (2/pi) ∫_0^inf B(w) cos(w t) dw
```

与 Hilbert 对（Kotik & Mangulis 1962; Newman 1977 ch.6）

```
B(w) = -(2/pi) PV ∫_0^inf s^2 (A(s)-A_inf)/(s^2-w^2) ds
A(w) = A_inf + (2/pi) PV ∫_0^inf B(s)/(s^2-w^2) ds
```

五个预声明门：G1 交叉求解器 10 %、G2 K(0)=(2/pi)∫B dw 2 %、G3 上述 Hilbert 对 10 %、G4 自由衰减 ζ vs 半功率 ζ 10 %、G5 B→K→(B,A) 5 %。

本单使用的数据（**无新求解**，见 §5）：仓库内已冻结的 `benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz`——192 面板粗网格、13 个频率节点 [0.15, 8.0] rad/s、6×6 附加质量/阻尼矩阵（由更早的算例池任务写入）。本机 `/proc/loadavg` load15 ≈ 101（nproc=128），按容量纪律不起本机重算；扩展频带扫描已递交集群但**未在预算内运行**（§5）。

## 2 怎么做的（方法/流程/关键设置/工具链）

1. **缺陷定位（G4 不可判）**：`tensorbem.time_domain._half_power_crossing` 在扫描窗内找不到半功率交点时，下降支 `np.nonzero(excess < 0)[0][0]` 取空数组索引 → `IndexError: index 0 is out of bounds for axis 0 with size 0`（round-2 逐字记录的 G4 判定）；上升支则把窗外的交点**静默钳到 `scan[0]`**，给出系统性偏小的 ζ。修法（`src/tensorbem/time_domain.py`）：
   * `_half_power_crossing` 现在只在**交点被窗内括号夹住**时返回插值，否则抛带窗口诊断的 `ValueError`；
   * `half_power_damping` 以 ×4 逐次加宽扫描窗（上限 0.9·ω_n 与 0.9·(ω_max−ω_n)）直到两交点都被夹住，并在返回值中记录 `scan_widenings` / `scan_attempts`——加宽次数本身就是"网格对阻尼峰的分辨率有多差"的量化指标；连最宽窗都夹不住时抛 `ValueError`（不再是 `IndexError`，也不再有钳位数字）。
2. **KK 残差的"地板"测量（新工具）**：`src/tensorbem/benchmarks/kk_band_floor.py`
   * `kk_residual`：与算例 G3 **同一归一化**（除以 max|A−A_inf|）、同一求解器（`causality.kkr_added_mass`）的残差，逐中点返回；
   * `estimator_floor`：把**同一份数据**用保形插值（PCHIP，无自由参数、不制造新极值）重表达到 601 点后重跑同一估计器——离散化误差会消失、频带截断不会；若提供同一 DOF 的更宽频带数据，则把 `band_attributable = refined(narrow) − refined(wide)` 变成数字。
   * 曾先用"指数和记忆核（28 个松弛率）最小二乘拟合实测扫频"的做法做同一件事——它把 13 节点的扫频**精确拟合**（拟合误差 ~0），地板恒等于实测、加密外推发散（roll 达 227 %），即**自由参数多于数据点必然过拟合**；该做法已弃用并写进模块 docstring 作为记录在案的陷阱。
3. **驱动**：`scripts/cummins_wigley_band_budget.py`（`--npz` 读缓存扫频、`--points-dir` 读逐频 JSON；后者在扩展频带缓存齐全时自动把 case 频带 / 扩展频带 / legacy 频带并排评估），输出 `artifacts/case_bem_cummins_wigley_budget/band_budget_coarse192.json`、`g3_edge_analysis.json`、`fig_kk_profile.png`。
4. **测试**：`tests/test_half_power_scan_window.py`（7 项）、`tests/test_kk_band_floor.py`（4 项）。

## 3 计算结果（全部由脚本从缓存工件读出，零手抄）

缓存粗网格扫频 [0.15, 8.0] rad/s，13 节点，G3 残差（max / rms / 去掉最外区间后的内部 max）：

| DOF | coarse max | coarse rms | coarse interior max | refined(601) max | refined rms | argmax coarse→refined |
|---|---|---|---|---|---|---|
| surge | 0.4021 | 0.2856 | 0.3809 | 1.1214 | 0.3404 | 6.87 → 7.99 |
| sway  | 0.3256 | 0.1095 | 0.0675 | 2.0054 | 0.2930 | 6.87 → 7.99 |
| heave | 0.3725 | 0.3304 | 0.3628 | 0.6568 | 0.3428 | 6.87 → 7.99 |
| roll  | 0.3222 | 0.1361 | 0.1137 | 2.2654 | 0.3754 | 6.87 → 7.99 |
| pitch | 0.4346 | 0.3531 | 0.3887 | 0.8824 | 0.3718 | 6.87 → 7.99 |
| yaw   | 0.3867 | 0.3545 | 0.3866 | 3.0141 | 0.4422 | 0.18 → 7.99 |

G2 = 3.3e-16（恒等式，PASS）；G1 跳过（该网格不在 capytaine 冻结参考的网格阶梯上，`include_cross_solver=False` 明确记为 skipped，不计入失败）；G3 = 0.4623（FAIL）；G5 = 0.4551（FAIL）。

G4 修复后的判决（同一缓存扫频）：`ValueError: the half-power points are not bracketed even with the widest admissible window 0.236492 rad/s around omega_n=7.73723 (band [0.15, 8]); attempts=[{'scan_half_width': 0.23649222503476458, 'error': 'the lower half-power point lies below the scan window: the first sample 0.0005740...'}]`——G4 仍然**不可判**，但原因现在是可读的、可定位的：13 节点 [0.15,8] 的网格根本没有分辨出 B33 的峰（峰宽 ~ζω_n，ζ≈几×10⁻³，而网格间距 ~0.5 rad/s），所以半功率宽度无从谈起。

图 `fig_kk_profile.png`：左=6×4 DOF 的 KK 残差逐中点曲线（对数纵轴，10 % 门线）；右=缓存节点与其分段线性重表达下的 B33——峰被压平，正是 G4 与 G3 共用的病灶。

## 4 结果判定（关键量 vs 基准，逐项回答是否 <3%）

* 本单**没有**通过 G3/G5 门：缓存粗网格上 G3 = 32-43 %（max）、11-35 %（rms），相对 10 % 容差与 3 % 的严格线**均不满足**。honest FAIL 如实记录，不粉饰。
* 新增的定量结论（本轮真正的产出）：
  1. **G3 的 max-over-midpoints 统计量在频带上边缘不稳定**：同一份数据经保形加密后，argmax 全部跑到 7.99 rad/s（带边最后一个中点），max 从 32-43 % 涨到 66-301 %；而 rms 基本不动（0.11-0.35 → 0.29-0.44）。也就是说 round-2 报告的"53-81 %"里有一部分是**统计量定义**（max 落在带边、PK 积分在带边条件数差）而非物理非因果性。→ G3 的判据应改为 rms 或"排除最外区间"，或对带边单独声明。
  2. **不能说它是纯频带效应**：加密后 rms 仍有 0.29-0.44，内部 max 0.07-0.39，量级与实测相当，说明在 [0.15,8] 这段带内、按现有闭端处理，估计器对该数据仍给出数十百分点的残差——round-2 的"band/scale effect"解释**未获支持**，需要扩展频带实测（§5）才能定案。
  3. 缺陷层面的确定性结论：G4 的 `IndexError` 是代码缺陷（已修，7 项测试锁定），且在缓存网格上 G4 的不可判来自网格分辨率，不是物理。

## 5 注意点（坑 / 局限 / 适用范围）

* **未运行**：扩展频带 [0.10, 20.0] rad/s 28 节点的中/粗网格扫描已递交集群（`cj1658` medium、`cj1659` coarse，各 8 分片），递交时队列前方 323 个作业，等待 19 min 仍未派发；本机 load15 ≈ 101 触发容量纪律（只做轻量活），因此本轮**没有**扩展频带的任何数字。`band_attributable` 在本轮全部为 `null`——工具已就绪，数据未到。
* **陷阱（记录在案）**：指数和 surrogate 拟合是错的解法（自由参数 28 ≫ 数据 13），任何"地板"都会退化成实测值；已改用零参数保形重表达。
* **PCHIP 加密不是新数据**：它只去掉估计器输入里的分段线性模型，因此 refined 残差**不能**当作"更好的答案"，只能当作"粗采样额外带来的那一部分"。
* G2 的 3.3e-16 说明同一扫频上的正交恒等式成立到机器精度——即"估计器本身没写错"，与 G3 的残差并不矛盾（G2 是求积恒等式，G3 是 Hilbert 对）。
* 缓存扫频来自 192 面板粗网格（12×4），不在 capytaine 参考的网格阶梯上，故 G1 只能跳过；任何"通过"都不应基于此网格。

## 6 复现

```
PYTHONPATH=src python3 scripts/cummins_wigley_band_budget.py \
  --npz benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz \
  --label coarse192_13nodes --out artifacts/case_bem_cummins_wigley_budget/band_budget_coarse192.json
PYTHONPATH=src python3 -m pytest tests/test_half_power_scan_window.py tests/test_kk_band_floor.py -q
```

扩展频带（未完成）：
```
PYTHONPATH=src python3 scripts/run_cummins_wigley_kk_causality.py --mode sweep --level medium \
  --shard i --shards 8 --outdir artifacts/case_bem_cummins_wigley_ext \
  --omega-list 0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.65,0.80,1.00,1.20,1.50,1.85,2.30,2.80,3.40,4.00,4.70,5.40,6.20,7.00,7.80,8.60,9.60,11.00,12.50,15.00,20.00
PYTHONPATH=src python3 scripts/cummins_wigley_band_budget.py --points-dir artifacts/case_bem_cummins_wigley_ext/points --level medium
```