# m20021 — CASE-BEM-CUMMINS-WIGLEY：中等网格扫频被一条非物理负阻尼挡住（+ 证据可重判能力）

**锚点**：feature ledger `CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核（6DOF），
ref = Cummins 方程 + Kramers-Kronig 因果，tol < 10%）。开工时状态 `open`（未被他单完成）。
**前序**：m17375 建算例、m18249 定位「两个失败闸门是量测装置的数值问题」并把 coarse 网格做到 6/7
（C2 在 coarse 上按设计不通过）。**本单目标**：把同一算例推到 medium 网格（480 面元，C2 可达），
补齐「已合并功能对基准复核」的验证欠账。

## 1 交付物

| 类型 | 内容 |
| --- | --- |
| 代码 | `src/tensorbem/benchmarks/case_bem_cummins_wigley.py`：新增 `save_sweep` / `load_sweep` / `gates_from_artifact` / `case_verdict`；`--replay NPZ`、`--mission` 两个 CLI 开关；C3 每自由度新增 `added_mass_inf_cross_check`（闭合解 A_inf vs `A_inf + c/ω²` 尾拟合的独立对照） |
| 测试 | `tests/test_case_bem_cummins_wigley.py` 新增 5 条：save/load 往返、缺身份字段拒载、形状不匹配拒绝、verdict 记账、生产证据重判（缺证据时 skip） |
| 证据 | `docs/m20021/sweep_coarse.npz`、`sweep_medium.npz`（41 节点 × 6DOF × 6DOF 辐射系数）、`sweep_defect_scan.json`、`solver_residual_scan.json`、`fig_medium_negative_damping.png` |
| 浮点算力 | 集群作业 `cj2446`（node n330，16 核，rc=0，`stage_dir=/root/tp-agent/work/m20021`） |

`gates_from_artifact` 让本算例的每个数字都能从**committed sweep 工件**重判，不需要再解一次 BEM
（全部闸门都是「扫频系数 + 船体常数」的函数）。这是本单的可复用增量：CI 里可以用
`sweep_medium.npz` 复现判据，而不必信任报告正文。

## 2 方法与关键设置

* 网格：coarse (12×4, 192 面元) / medium (20×6, 480 面元)；频带 0.2–12 rad/s，41 个对数节点；
  ρ=1025 kg/m³，g=9.80665 m/s²，CHIEF `chief_points="auto"`；自由衰减自由度 heave；
  dt=0.01 s、40 s 记录、记忆核截断 TAU_MAX=25 s。
* 全部 82 个频率解（2 网格 × 41）在集群上 16 核并行完成，落盘 `sweep_{coarse,medium}.npz`。

## 3 计算结果

### 3.1 C2（面板分辨率）在 medium 上达标

| 网格 | 面元数 | max panel √A [m] | k·√A_max @12 rad/s | 判据 ≤1 |
| --- | --- | --- | --- | --- |
| coarse | 192 | 0.1008 | 1.474 | 不通过（m18249 已记录「按设计」） |
| medium | 480 | 0.0645 | **0.947** | **通过** |

### 3.2 阻断项：medium 网格 heave 阻尼在 ω=5.291 rad/s 出现非物理负值

| 网格 | DOF | min B_ii | 出现 ω [rad/s] | B<0 的节点数 | 说明 |
| --- | --- | --- | --- | --- | --- |
| coarse | heave | +0.1249 | 0.2 | 0 | 干净 |
| medium | heave | **−129.917** | **5.291** | 1 | 相对 heave 阻尼峰值 (~200) 是 −65%，非物理 |
| coarse | surge/pitch/roll/yaw/sway | −0.95 / −1.46 / −8.6e-4 / −6.8e-4 / −0.50 | — | 3–13 | 量级远小于该 DOF 峰值，属离散噪声（非阻断） |
| medium | surge/sway/pitch/roll/yaw | 0 / 0 / 0 / −2.7e-4 / −1e-5 | — | 0–17 | 同上 |

后果是**硬失败而非精度问题**：`time_domain.validate_radiation_sweep` 拒绝负阻尼（能量守恒守卫，
`ValueError: negative radiation damping is not physical ... min damping = -129.917 at omega = 5.29116`），
因此 C4（B→K→B 往返）在 medium 上**根本无法评估**，算例驱动在写完两个 sweep 工件后中止，
`case_result.json` / `CASE.md` 未生成。这是诚实记录：本单**没有**把锚点做到全绿。

原因指向 medium 网格在该单点上的**不规则频率残留**（表面穿透船体外场 BIE 的特征值问题；CHIEF 在该
频率没有压住），而不是物理。判定依据：coarse 同频率 heave 阻尼为正、其余 40 个 medium 节点全正、
只有一个节点为负。**未做**的验证（下一次要做的）：该节点换 CHIEF 内点数/位置重解，
或改用 `omega=5.291` 邻域加密 + 不动点检查，确认是孤点伪解。

### 3.3 另一个证据质量问题：`solver_relative_residual` 在两个网格上都很大

| 网格 | 中位数 | 最大 | >1% 的节点数 |
| --- | --- | --- | --- |
| coarse | 0.610 | 1.253 (@10.83 rad/s) | 41/41 |
| medium | 0.546 | 1.245 (@5.291 rad/s) | 41/41 |

孤点抽查 ω=1.0 rad/s、medium、6 模全解时该字段为 4.14e-05，说明它**不是**「残差/右端范数」的
寻常定义（否则不会 41/41 都 O(1)）。该字段此前被原样写进工件（`solver_relative_residual_max`）
而无人质疑。本单只做记录，不给结论：需要读 `solve_radiation_3d` 的残差定义才能判定这些解是否可信。
在弄清楚之前，3.2 的负阻尼也不能仅当作「后处理噪声」。

## 4 图

`fig_medium_negative_damping.png`：左图 heave B33(ω) 两网格对比（symlog 纵轴，可见 medium 在 5.29
处的负瓣），右图每频率线性系统残差。两图均由 `docs/m20021/sweep_*.npz` 程序化绘制，零手抄。

## 5 注意点 / 局限

* 本单**未**产出可判 PASS 的算例结论；锚点仍应保持 `open`。
* C3（KK 闭合）、C5（自由衰减）在 medium 上的数值**没有**得到：驱动在 C4 就中止了。用
  `gates_from_artifact(docs/m20021/sweep_medium.npz)` 可以在修好负阻尼前先看 C3/C2 的数值，
  但 C4/C5 会被负阻尼守卫拒绝——这是设计上的正确行为，不是缺陷。
* A_inf 交叉对照（新增字段）在本单没有取值可报：C3 的结果在本单未跑出。
* 频带 0.2–12 rad/s 覆盖本模型全部刚体固有频率（heave ≈ 7.8 rad/s），但覆盖到 12 rad/s 的代价是
  medium 网格的面元分辨率刚好卡在 C2 阈值 0.947（余量仅 5%），再加密频带上端就必须加密网格。

## 6 误差判定

本单**没有可比的关键量**：驱动在生成 `case_result.json` 之前中止，因此没有任何闸门数值可以
对基准做误差判定，「是否 <3%」**无法回答**（不是「未达标」，而是「未测到」）。已测到的只有几何类
量与网格离散量：C2 的 k·√A_max = 0.947（阈值 1.0）、medium 网格体积相对误差 < 1%（C1 在
`gates_from_artifact` 中仍可通过）。如实记为 WITHHELD。

## 7 复现

```
# 集群（不要在 99 控制面跑）
run_compute submit repo=TensorBEM stage_dir=<worktree> cores=16 \
  cmd='cd . && PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_cummins_wigley \
       --out artifacts/case_bem_cummins_wigley --jobs 16 --n-omega 41 --meshes coarse medium'
# 从工件重判（无需 BEM 解）
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_cummins_wigley \
  --replay docs/m20021/sweep_coarse.npz
# 测试
PYTHONPATH=src python3 -m pytest tests/test_case_bem_cummins_wigley.py -q   # 23 passed, 1 skipped
```