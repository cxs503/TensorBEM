# m14767 — TensorBEM 状态空间降阶（Cummins 记忆核）拟合与实现的缺陷修复与验证

**仓**：TensorBEM · **分支**：agent/m14767 · **锚点**=NEW（总账无 state_space 条目：`docs/audit_m14488/audit_report.md` §3 表第 10 行把该能力判 L1，理由是「自带精度门不通过：`tests/test_state_space.py` 5 failed（拟合误差 6.15e-01 vs 0.15）」——本单把这条欠账做实）

## 1. 案例介绍（物理背景/几何/参数表）

物理本体：浮体垂荡（heave）的 Cummins 时域方程

    (M + A_inf) z'' + ∫_0^t K(t-s) z'(s) ds + C z = F(t),   K(τ) = Σ_k c_k e^{-α_k τ}

时域卷积是耐波性仿真的算力瓶颈；若记忆核可写成指数和，卷积可用 N 个一阶辅助 ODE
（状态空间）替代，单步代价与时历长度无关。从「表列 B(ω)」到「指数和」是**有理逼近**问题：
B(ω) = d + Σ_k c_k α_k/(α_k² + ω²)（对 ω² 的有理函数，极点为衰减率 α_k）。

算例（仓内已入仓数据 `tests/data/time_domain_hemisphere_sweep.json`）：Hulme 潜没半球垂荡。

| 量 | 值 |
|---|---|
| 半径 R | 1.0 m |
| 密度 ρ | 1025 kg/m³ |
| 排水体积 | 2.0944 m³ |
| 质量 M | 2146.755 kg |
| 恢复力系数 C | 31578.712 N/m |
| A_inf | 1073.377 kg |
| 频率带 | 0.14 – 29.7 rad/s（201 点） |
| B 动态范围 | 1745.297 : 0.02096（**四个数量级**） |

## 2. 怎么做的（方法/流程/关键设置/工具链）

### 2.1 诊断（先读代码再改）

对 `src/tensorbem/state_space.py` 的 3 处独立缺陷逐一定位，全部给出可复现数字：

1. **拟合器**：`vector_fit_damping` 名为「vector fitting」（文件头引 Gustavsen & Semlyen 1999），
   实为**未加权的绝对残差** L-BFGS-B 非线性最小二乘，且残差约束 c_k ≥ 0、α_k ≥ 1e-6。
   后果：目标函数被峰值（1745）主导，四个数量级的曲线两端完全失配。实测（修复前）：
   `relative_error_max = 0.6153`，**5 个极点被压在 α = 1e-6 下界上、残差 ~7.5e3**，
   B(ω_min) = **1075.2** 而实测 **1.4086**（差 763 倍），且 d = **−226.6**（高频负阻尼，非物理）。
2. **实现（realization）**：`state_space_from_fit` 只取 poles/residues，**静默丢弃拟合常数 d**。
   物理上 d 是核里的 δ(τ) 项（F_rad = Σ y_k + d·x'），丢掉后时域模型比它所声称代表的拟合「少辐射」。
3. **时间推进**：`cummins_state_space` 用裸 RK4，步长无条件限制。极点最快者 α_max 时
   α_max·dt > 2.78 即发散——半球数据在 dt=0.01 下直接 overflow 成 NaN（修复前实测警告
   `overflow encountered in matmul`，`test_full_workflow_hemisphere` 拿到 nan）。

另外查出两处**验证面伪证据**（比失败更危险）：

4. `validate_state_space` 的 C4（双路径一致性）把时域值直接赋成频域值、相对误差写死 0.0、`c4_pass = True`；
   C5（因果性）写死 `c5_K_negative_time_max = 0.0` 后据此判 pass。两条判据是**构造出来的通过**，
   且全仓无任何测试调用 `validate_state_space`（只有 import）。
5. 测试侧两处**阈值/符号本身写错**：`test_kramers_kronig_consistency` 把残差符号翻了两次
   （`zip(-fit.residues, ...)` 再取 `-c/...`），断言 A = A_inf + Σc/(α²+ω²)——任何正确实现都过不了
   （实测该式残差 **2.279**，正确式 **8.7e-8**）；`test_full_workflow` 要求 20 s 后幅值 < 初值 1%，
   而两条独立路径都给出 **6.3%**（物理上 ζ_eff·ω_n ≈ 0.058 /s）。

### 2.2 修复（最小、可验证）

* **拟合器 → 可分离加权最小二乘（variable projection）**：外层只优化 log α_k（`n_poles` 维，
  界在 [1e-3ω_min, 1e3ω_max]），内层对 (d, c) 精确解线性最小二乘；残差按
  `1/max(|B|, floor·max|B|)`（floor=1e-3）加权——这正是四数量级曲线可拟合的关键。
  d 保持 d ≥ 0（高频阻尼不得为负）；残差符号**不加约束**，并在 docstring 里写明理由与实测依据
  （c_k ≥ 0 时任何阶数都停在 ~100% 峰值误差，见证据表阶梯）；三起点确定性多启。参数与返回类型不变。
  新增字段 `relative_error_max_pointwise`（floor 保护的逐点相对误差）与 `weight_floor`，
  原有 `relative_error_max` 定义（峰值归一）**保持不变**，门禁没有被放宽。
* **实现**：`StateSpaceRealization` 增 `D_ss`（默认 0），`state_space_from_fit` 取 `fit.constant`，
  `cummins_state_space` 用 F_rad = Σy + D·x'（含输出 `radiation_force`）。
* **时间推进**：按最快速率 `max(α_max, ω_n)` 自动内部子步（RK4 稳定界取 2.0 留余量），
  子步数写入结果的 `substeps`；子步数超 20000 时显式报错要求减小 dt（不再静默输出垃圾）。
* **判据 C4/C5 去伪**：新增 `radiation_response`（频域复数响应 G(iω) = d + Σ c_k/(α_k+iω)，
  实部即拟合 B、虚部即 ω(A−A_inf)）与 `_harmonic_memory_response`（在**同一**辅助 ODE 上以
  规定谐速度 v=v0 sin ωt 积分、取后 4 周期复 Fourier 系数）。C4 = 两者幅值/相位互差（真双路径）；
  C5 = 拟合 B 经 Kramers-Kronig 反推的 A 与实测 A 的最大相对残差 + 极点正性。写死的 0.0/True 全部删除。
* **测试侧**：修正 KK 符号（注明原因与前后数字）；`test_full_workflow` 的不可满足阈值换成
  **两路径不变量**（SS vs 由表列 B 积分出的 K(t) 卷积，周期 <2%、衰减率 <5%）——判据更强而不是更松。

工具链：numpy/scipy（lstsq，无新增依赖）+ pytest + matplotlib；全部数字由
`scripts/m14767_state_space_study.py` 程序化写入 `docs/m14767/evidence.json` 与
`docs/m14767/numbers.md`（零手抄），图 `fig1_fit.png` / `fig2_decay.png` 同源生成。

## 3. 计算结果（程序化注入，见 docs/m14767/numbers.md）

下图 1：B(ω) 数据 vs 10/15 极点拟合（双对数）与峰值归一残差；图 2：自由衰减 SS 路径 vs 独立 IRF 卷积路径。

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 拟合 `relative_error_max`（10 极点，门禁 0.15） | **0.6153** | **0.00361**（15 极点 5.2e-4） |
| 极点下界堆积数（α = 1e-6） | 5（共 10） | 0（α ∈ [0.157, 14.9]） |
| B(ω_min) 拟合 / 实测 | 1075.2 / 1.4086（×763） | 1.409 / 1.409（0.0%） |
| 拟合常数 d | −226.6（非物理） | 0（≥0 约束） |
| 半球自由衰减（dt=0.01, 30 s） | **NaN**（RK4 发散） | 有限；子步 1；末幅 0.009% 初值 |
| SS vs IRF 周期 / 衰减率 | 无（nan） | 差 0.055% / 0.618% |
| `validate_state_space` C4 | 写死 pass=True（相对误差 0.0） | 幅值 0.000%、相位 2.7e-6 rad（真算） |
| `validate_state_space` C5 | 写死 0.0 → pass | KK 残差 2.969%、min α > 0 |
| `pytest tests/test_state_space.py` | **5 failed / 11 passed** | **16 passed** |
| 新增回归钉 `tests/test_state_space_fit_invariants.py` | —（不存在） | **6 passed** |

### 3.1 程序化数字表（原样摘自 docs/m14767/numbers.md）

## 1. 拟合阶梯

| n_poles | peak-normalised max err | pointwise floor err | d (feedthrough) | alpha range | B(w_min) data | B(w_min) fit |
|---|---|---|---|---|---|---|
| 1 | 99.816 % | 116.658 % | 0 | 9.727 … 9.727 | 1.409 | 3.445 |
| 3 | 64.896 % | 127.057 % | 2.486 | 3.25 … 3.263 | 1.409 | 1.041 |
| 5 | 8.016 % | 10.910 % | 0 | 3.125 … 3.346 | 1.409 | 1.398 |
| 8 | 0.847 % | 0.848 % | 0.1076 | 0.1604 … 5.164 | 1.409 | 1.409 |
| 10 | 0.361 % | 0.811 % | 0 | 0.1565 … 14.9 | 1.409 | 1.409 |
| 12 | 0.060 % | 0.114 % | 0.02408 | 0.1224 … 11.91 | 1.409 | 1.409 |
| 15 | 0.052 % | 0.072 % | 0 | 0.1511 … 23.71 | 1.409 | 1.409 |

## 2. 实现（realization）双路径 C4

| omega (rad/s) | |G| analytic | |G| integrated | amp rel err | phase err |
|---|---|---|---|---|
| 0.3 | 189.735 | 189.735 | 0.000 % | 1.55e-06 rad |
| 1 | 861.192 | 861.164 | 0.003 % | 6.73e-06 rad |
| 3 | 1750.22 | 1750.23 | 0.000 % | 1.96e-06 rad |
| 10 | 482.284 | 482.283 | 0.000 % | 3.84e-07 rad |
| 25 | 162.709 | 162.709 | 0.000 % | 1.74e-07 rad |

### feedthrough D 项审计（合成曲线，真值 d = 50）

拟合得到 d = 50，peak err = 0.000 %

| omega (rad/s) | amp rel err (含 D) | amp rel err (丢掉 D = m14767 之前的行为) |
|---|---|---|
| 0.5 | 0.000 % | 2.814 % |
| 2 | 0.001 % | 3.647 % |
| 4 | 0.001 % | 4.566 % |

## 3. 自由衰减 SS vs IRF

| 量 | SS 辅助 ODE | IRF 卷积 (K(t) 由列表 B 积分) | 相对差 |
|---|---|---|---|
| 固有周期 (s) | 1.94433 | 1.94539 | 0.055 % |
| 衰减率 (1/s) | 0.30636 | 0.30827 | 0.618 % |
| 30 s 后幅值 / 初值 | — | — | SS 0.009 % |
| 两条路径最大位移差 (m) | — | — | 0.0001963 (峰值 0.05) |
| RK4 内部子步数 | 1 | 1 | — |

## 4. 判据 C1–C6（10 极点，Hulme 半球）

| 判据 | 量 | 值 | pass |
|---|---|---|---|
| C1 fitting | peak-normalised max err | 0.361 % | True |
| C2 stability | min pole | -14.9 | True |
| C3 free decay | period rel err | 0.645 % | True |
| C3 free decay | decay-rate rel err | 0.044 % | True |
| C4 dual path | memory amp rel err | 0.000 % | True |
| C4 dual path | memory phase err | 2.72e-06 rad | True |
| C5 Kramers-Kronig | max rel A residual | 2.969 % | True |
| C6 negative control | order-1 peak err | 99.816 % | True |

all_pass = **True**

## 4. 结果图

* `docs/m14767/fig1_fit.png` — 上图：Hulme 半球垂荡 B(ω) 数据（点）与 10/15 极点拟合（线，双对数，
  四数量级全覆盖）；下图：峰值归一残差 vs 15% 门禁线。
* `docs/m14767/fig2_decay.png` — 自由衰减：状态空间（辅助 ODE）与独立 IRF 卷积（K(t) 由表列 B 积分）
  两条路径叠加，标题即两路径周期/衰减率相对差（程序化写入）。

## 5. 注意点（踩过的坑/局限/适用范围）

1. **比例尺就是算法**：四数量级曲线上，未加权的绝对残差 = 只拟合峰值。加权 + 可分离线性内解是
   唯一让它可解的路径；但对 B 接近 0 的尾部，`weight_floor=1e-3` 仍会放行「floor 级」误差
   （逐点相对误差如实另报在 `relative_error_max_pointwise`）。**不能**用 peak-normalised 指标
   宣称尾部相对精度。
2. **残差符号不可加正约束**：实测 c_k ≥ 0 时任何阶数都停在 ~100% 峰值误差（浮体记忆核会变号）。
   这是本次把 bounds 中 `c_k >= 0` 删掉的依据；但代价是拟合不再保证 positive-real（严格无源性），
   只在频带内核查 B(ω) ≥ 0（测试 `test_positive_residues_cannot_represent_the_curve` 覆盖）。
3. **d（feedthrough）是物理项不是数值装饰**：它对应核里的 δ(τ)，等于高频极限阻尼。
   丢掉它时域模型少辐射 2.8–4.6%（合成 d=50 的负控实测），而半球曲线 d≈0 时该缺陷**完全隐形**——
   这正是它此前没被发现的原因。
4. **显式 RK4 的条件稳定被自动子步兜住**，但子步只是稳定性，不是精度：`substeps` 应作为诊断量读；
   若用户的外力含高频成分，dt 仍需自行按该频率选取。
5. **C3 的频域参照仍是半带宽估计**（原实现），未换成独立 RAO 求解；C4 的比较对象是「实现」与
   「拟合」的一致性，**不是**与实验/外部 BEM 码的对照。半球垂荡阻尼无仓内实验基准，
   本条只能作内部一致性证据。
6. **两处测试是错的，不是代码错**：KK 符号（残差 2.279 → 8.7e-8）与 20 s 衰减阈值（真值 6.3% vs
   要求 1%）。改测试而非改代码，理由与前后数字都写进了测试注释——这是本次唯一的「改判据」动作，
   且都以两条独立路径的数字为凭。
7. **未做（honest）**：`docs/audit_m14488` 指出的「`state_space` 与 `memory_kernel_matrix` 功能重叠」
   未合并；`adaptive_order_fit` 仍是逐阶穷举（阶数选择未做信息准则）；仓内 4 项
   `tests/test_capability_evidence.py` 失败是**既有问题**（在未改动的 HEAD 上同样失败 4 项，已用
   `git stash` 复核），本单未触碰。

## 6. 误差判定（关键量 vs 基准，逐项回答是否 <3%）

| 关键量 | 基准来源 | 误差 | 是否 <3% |
|---|---|---|---|
| 拟合 B(ω)（10 极点） | 表列 BEM 扫频曲线（峰值归一） | 0.361% | **是** |
| 拟合 B(ω)（15 极点） | 同上 | 0.052% | **是** |
| B(ω_min) 低频端 | 实测 1.4086 | 0.0% | **是** |
| 自由衰减固有周期 | 独立 IRF 卷积路径 | 0.055% | **是** |
| 自由衰减衰减率 | 独立 IRF 卷积路径 | 0.618% | **是** |
| 实现双路径 C4（幅值） | 频域 G(iω) 闭式 | 0.000%（5 个频率最差） | **是** |
| 实现双路径 C4（相位） | 同上 | 2.7e-6 rad | **是** |
| KK 反推 A vs 实测 A | 同一 BEM 扫频的 A(ω) | 2.969% | **是（临界）** |
| 严格无源性（positive-real） | — | 未满足（残差含负值） | 无基准，属已知限制 |

关键量全部 <3%；其中 C4/自由衰减是**同源两路径**一致性（能抓实现错误，不能抓物理建模错误），
KK 是 10 极点 B-only 拟合的必然残差，2.969% 贴线，不作独立物理验证使用。
