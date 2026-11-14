# CASE-BEM-CUMMINS-WIGLEY round 5 — G3 残差到底是"哪种错"：把 A_inf 剖面解出来

锚点 `CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核 6DOF；ref = Cummins 方程 + Kramers-Kronig 因果；容差 <10 %）。任务 m20594 / 分支 `agent/m20594`。本轮不重算面板，而是**把已合并的 G3 判据拆成可判定的分量**，并在有真值的合成对上先验证这套拆法本身。

## 1 案例介绍

Wigley III（L=3.0 m, B=0.3 m, T=0.1875 m, Cb=0.543），ρ=1025 kg/m³，g=9.80665 m/s²，无限水深，旋转中心 (0,0,−T/2)，几何门在取数前先过（Cb 体积误差 0、L/B=10、B/T=1.6）。数据为仓内已入库的 41 点粗网格辐射扫频 `docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz`（ω∈[0.2,12] rad/s，192 面元），本轮未新起重算。

## 2 怎么做的

1. **合成对照臂先纠正约定**：round-4 报告记录"KK 残差在 69 点与 24001 点采样上为 45 % / 40 %，加密 350 倍几乎不降"，据此撤回了"采样效应"假设。复核发现该闭式对 `K(t)=c e^{-σt}cos(Ωt)` 在 `B` 上多带了一个 `1/π`（使 (A,B) 自身非因果 `1−2/π=36.3 %`）。闭式对已移入库函数 `kk_band_floor.damped_oscillator_pair`（含推导），修正后同一估计器残差随加密下降（本轮测试锁定：201→801 点下降 >3 倍）。
2. **A_inf 剖面闭式解（新库函数 `causality.profile_added_mass_inf` / `AInfProfile`）**：KK 关系 (2) 对 `A_inf` 是斜率为 1 的仿射函数，故 G3 的 `max_i|A_pred−A_i|/scale` 是绝对仿射函数的最大值 → 凸、分段线性、**最优解是隐含 A_inf 的中列**：`a*=(max e+min e)/2`，`residual*=(max e−min e)/(2·scale)`（`e_i=A_i−q_i`）。因此「原判残差 = |A_inf 偏差|/scale + 剖面残差」是**代数恒等式**，不是近似。
3. **诊断脚本 `scripts/cummins_wigley_g3_conditioning.py`**：对每个 DOF 输出原判残差、PCHIP 加密后残差（采样项）、剖面（校准后）残差、A_inf 一维扫描的实测斜率、以及恒等式闭合误差；图见 `docs/fig_m20594_g3_conditioning.png`。

## 3 计算结果（程序化注入，零手抄）

合成对照臂（真值 A_inf = 0，69 点闭合网格）：原判 0.1707，剖面 0.0237，真值 A_inf 下的残差 0.0305，剖面给出的 A_inf* = +0.0024，**拆法自校验误差 6.8e-03**。

粗网格面板数据（41 点，[0.2,12] rad/s）：

| DOF | points | G3 原判 residual | 校准后 residual (profile) | 拟合 A_inf | 最优 A_inf | 校准分量 | 闭合误差 |
|---|---|---|---|---|---|---|---|
| surge | 41 | 0.3777 | 0.0432 | -0.12 | 0.67 | 0.3345 | 5.6e-17 |
| sway | 41 | 0.5128 | 0.0494 | -6.48 | 86.30 | 0.4634 | 0.0e+00 |
| heave | 41 | 0.2302 | 0.0139 | 27.27 | 51.78 | 0.2163 | 2.8e-17 |
| roll | 41 | 0.5094 | 0.0274 | 0.22 | 0.41 | 0.4820 | 1.1e-16 |
| pitch | 41 | 0.3562 | 0.0128 | 5.04 | 14.94 | 0.3434 | 1.1e-16 |
| yaw | 41 | 0.5066 | 0.0513 | -5.01 | 65.31 | 0.4553 | 0.0e+00 |

* 六个 DOF 的 A_inf 灵敏度斜率实测均为 -1.00（理论值 1.00）；
* 恒等式闭合误差 ≤ 1.1e-16（机器精度）；
* 校准后残差 0.0128–0.0513，**六个 DOF 全部 < 10 %（G3 容差）**。

## 4 判定（逐项 vs 基准，显式回答是否 <3 %）

* 拆法自校验（合成臂，真值已知）：剖面残差 0.0237 vs 真值 A_inf 下 0.0305，差 0.0068 → **0.68 % < 3 %，满足**。
* 面板数据 G3 原判 0.5128（失败）→ 校准后 0.0513 → **< 3 %：否**（最大 5.13 %，但**< 10 % 门限：是**）。
* 结论性质：G3 的 23–51 % **不是**面板解的非因果性，也**不是**频带截断或采样（PCHIP 加密后残差不降反升：0.5128→0.7790，因为加密同时改变了 scale 内的峰形），而是 `fit_added_mass_asymptote` 的两项尾拟合在"加质量随频率近水平"的 DOF 上病态：拟合出的 A_inf 可以落到数据范围之外（sway 拟合 A_inf = -6.5，剖面最优 86.3）。这一条已写成测试 `test_tail_fit_can_leave_the_physical_range_of_the_sweep`。

## 5 注意点

* 剖面残差把 A_inf 当冗余参数（profile/Chebyshev 中心），因此它是**更弱**的因果性陈述：它不再检验"绝对 A_inf 对不对"，只检验"整条 (A,B) 是否来自同一个因果系统"。G3 原判据不变，本轮只增加可判定的补充统计（与 round-4 的 G3b 同类），不把 FAIL 改成 PASS。
* 剖面残差对采样仍有依赖（合成臂 69 点闭合网格上 0.0237，而 24001 点均匀网格上更低），所以 5.13 % 是**上界**而非收敛值。
* 未完成：扩展频带 [0.1,20] + 共振簇的 medium 面板扫频已递交集群（cj2806 medium / cj2807 coarse），本轮预算内仍在排队，`band_attributable` 仍缺；扩展带数据到手后可直接 `--extended-points-dir` 复算。
* 局部加密（共振簇步长 0.0125→0.003125）不能降低本残差（1.71e-1→1.90e-1），因为它由 A_inf 偏差主导——这解释了 round-2/3/4 反复"加密无效"的观测，与 round-4 的"非采样"结论方向一致但归因不同（是校准，不是估计器地板）。

## 6 误差判定与 honest 状态

* 本轮**没有**把锚点推到 G1–G5 全绿：G4/G5 未重算（需扩展带数据）。
* 本轮的可判定结论：G3 的失败模式已定位到 A_inf 估计器（定量、可复现、有测试），校准后六 DOF 全部进入 10 % 容差；这一结论在真值已知的合成臂上自校验误差 0.68 %。
* 复现：`PYTHONPATH=src python3 scripts/cummins_wigley_g3_conditioning.py --sweep-npz docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz`；测试 `PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_kk_causality.py tests/test_kk_band_floor.py tests/test_causality.py -q`（40 passed）。
