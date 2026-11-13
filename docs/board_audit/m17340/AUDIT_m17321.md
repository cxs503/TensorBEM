# 交差审计 — m17321 / CASE-BEM-CUMMINS-WIGLEY（TensorBEM）

审计单 m17340 · verdict_by = `board-audit` · 对象 = m17321（status=succeeded，round 2，deepseek-v4-flash）
审计性质：**交差核验，非重做**。所有数字由 `docs/board_audit/m17340/audit_independent.py` 独立重算，
原始输出 `artifacts/audit_m17340/audit_checks.json`（随仓副本 `docs/board_audit/m17340/audit_checks.json`；主体 results.json 副本 `subject_results.json`）（全量 JSON，本文所有数字的机器来源）。

## 0 结论（一句话）

真实代码 + 真工件 + 真外部参照，**但没有通过锚点判据**：5 门中 G1/G2 过、G3/G4/G5 不过（G3 39.8 %–81.6 %、
G5 35.3 %–59.1 %、G4 41.5 %/未定义），与锚点「Cummins 方程 + Kramers-Kronig 因果 <10 %」不符。
判 **L3（系统集成）**，不给 L4。主体报告把失败如实标为 honest-partial —— 诚实性成立；
但报告 §6 把 G3 残差归因于「低频带截断」是**未被实验支持的**（见 F4）。

## 1 验收三件套是否存在（逐项实测）

| 件 | 判据 | 实测 | 结论 |
|---|---|---|---|
| run_artifacts | 真实运行工件 | `artifacts/case_bem_cummins_wigley/results.json` + `points/` 40 个单频解 + `figures/fig_sweep.png`(84 KB)；每个 point 带 `relative_residual`（如 medium ω=1：7.04e-05）与 `wall_seconds`（201 s） | 存在且为真解 |
| benchmark ref vs got | 有独立参照 + got 数值 | 参照 = capytaine 3.1.dev0 冻结件 `benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`；got = 同一 wetted mesh 上的 TensorBEM 3D 辐射解 | 存在（G1 行 6 自由度 × A/B 全列） |
| repro_cmd | 一键可执行 | `PYTHONPATH=src python3 scripts/run_cummins_wigley_kk_causality.py --mode gates --levels coarse,medium`；`pytest tests/test_cummins_wigley_kk_causality.py -q` | 审计方实跑：**9 passed**；gates 模式从缓存重算成功（见 §2） |

## 2 基准裁判：谁在裁

- `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-CUMMINS-WIGLEY'`
  → **0 行**（该表全表 51 行，本锚点未登记）。即：本锚点**没有机器可读的 benchmark case 行**，
  裁判落回总账 acceptance：`算例池|BEM-CUMMINS-WIGLEY｜ref: Cummins 方程 + Kramers-Kronig 因果｜容差 <10%`。
- 主体自设五门并把容差写进模块 docstring（G1 10 %、G2 2 %、G3 10 %、G4 10 %、G5 5 %），
  属于**预先声明**（写在 `GATES` 常量 + docstring，非事后回填），可作为本锚点的判据替身。

## 3 数值核验（审计方独立重算）

### 3.1 可复现性

| 档 | 磁盘 point 数 | results.json 记录 | 五门值逐项一致？ | 裁定 |
|---|---|---|---|---|
| coarse | 22/22 | cached 22, missing 0 | **是**（G1 6.2878 %、G2 3.89e-16、G3 53.23 %、G4 41.51 %、G5 46.33 % 全等） | 完全可复现 |
| medium | **18**/22 | cached **16**, missing 6 | **否**：G3 81.62 %→**85.99 %**、G5 53.37 %→**59.12 %**、G2 1.70e-16→3.50e-16；判定不变（仍 FAIL） | 见 F6（工件与 results.json 不同步） |

### 3.2 几何门（先验，独立算法）

解析 ∇=0.09163125 m³、Cb=0.543（L/B=10、B/T=1.6 全对）；审计方用散度定理自算网格体积：
coarse 0.0897211746（−2.085 %）、medium 0.091149534（−0.526 %），与主体 `sweep_signature` **逐位相同**。

### 3.3 G1 跨求解器（唯一外部锚）

审计方**不怕被比**：直接重跑 `scripts/wigley3_recalled_ref_capytaine.py`（capytaine 3.1.dev0 本机可导入），
把冻结参照整个重建：

| level | 网格签名 | panels | 重现误差（A/B 全 6 自由度） |
|---|---|---|---|
| n_sections=10,n_levels=5 | 相同 | 200 | worst 2.95e-12 |
| n_sections=20,n_levels=10 | 相同 | 800 | worst 5.19e-13 |

⇒ 参照**不是复制 TensorBEM 输出**，是独立代码真跑出来的；G1 的 ref vs got 差异是真正的求解器间差：
coarse 6.29 %、medium 3.26 %（<10 % PASS）。**报告 §4 的 6 行 per-DOF 数字与 results.json 逐位一致**（零手抄成立）。

### 3.4 G3/G4/G5（锚点核心：因果性）

G3 用**与主体完全不同的路线**独立重算：K(τ)=(2/π)∫B cos ωτ dω，A(ω)=A∞−ω⁻¹∫K(τ) sin ωτ dτ：

| 档 | DOF | 主体 G3 | 审计方（Cummins 符号） | 审计方（反号对照） | <10 %? |
|---|---|---|---|---|---|
| coarse | surge | 44.26 % | 44.21 % | 166.8 % | 否 |
| coarse | heave | 14.13 % | 14.12 % | 187.0 % | 否 |
| coarse | roll | 49.44 % | 49.40 % | 161.8 % | 否 |
| coarse | yaw | 53.23 % | 53.19 % | 154.8 % | 否 |
| medium | surge | 70.96 % | 70.98 % | 156.0 % | 否 |
| medium | heave | 37.27 % | 37.28 % | 172.5 % | 否 |

两条数值路线差 ≤0.05 pp，反号对照差 3–4 倍 ⇒ **符号约定正确、G3 FAIL 是真结论，不是门实现 bug**。
G4：coarse ζ_decay 0.09797 vs ζ_half-power 0.16749 → 41.51 %（FAIL）；medium 精确复现主体的
`IndexError: index 0 is out of bounds`（半功率交点在扫描窗外），主体如实记录为 nan，未编数。
G5：coarse 46.33 %、medium 53.37 %/59.12 %，6 自由度全超标（tol 5 %）。

## 4 findings（按严重度）

- **F1（blocking，锚点未达）** G3 因果残差 14.1 %–81.6 %（tol 10 %），G5 22.3 %–59.1 %（tol 5 %），
  G4 41.5 %/nan ⇒ 锚点 acceptance 未满足。审计方独立路线复核一致，**非度量作弊、非门 bug**。
- **F2（blocking，证据口径）** 锚点在 `tensor_benchmark_cases` **无登记行**，L4 的「基准验证」没有机器裁判；
  现有裁判是主体自设五门。建议把五门（含容差与 ref_source）登记进表，才谈得上 L4。
- **F3（medium）** G2「K(0) = (2/π)∫B dω」在实现上是**同义反复**：kernel 的前向腿就是 B 的梯形积分
  （`analysis_gates` 注释自承 "the forward leg is nothing but the exact piecewise-linear integral of B"），
  残差 1e-16 只证明代码路径自洽，**不含物理信息**。报告 §4 把它与 G1 并列呈现，读者会高估验证强度。
- **F4（medium，过度归因）** 报告 §6 把 G3 残差归因于低频带截断（"needs a denser low-frequency sweep 0.1–0.3"）。
  审计方做带截断对照：coarse yaw G3 在 [0.30,12.5] 为 53.23 %，砍掉最低两节点([0.50,12.5])后仍 **50.96 %**（仅降 2.3 pp）
  ⇒ 低频带不是主因，该诊断**未被实验支持**，属 overstated claim（不改变 FAIL 结论，但会误导下一轮返修方向）。
- **F5（medium，侧发现，不属于本单主体）** 同一参照文件里 `wigley3_hull` 段**不可复现**：重建网格签名不同、
  体积 0.08834853→0.08835325、系数最大漂移 **110 %**（Pitch damping）。该段被其它 study 引用时不可信；
  主体本单只用 `wigley3_freq` 段（重现 1e-12），不影响本 verdict。
- **F6（low，取证一致性）** medium 档磁盘 18 个 point、results.json 记 16 个：`results.json` 未随补算重生成，
  数值有 4.4 pp（G3）/5.7 pp（G5）漂移。方向对主体不利（重算更差），不构成美化，但工件-结论不同步。
- **F7（low）** 报告§5 的收敛性只有"coarse vs medium 共振频率 7.3931 vs 7.3785 rad/s"一句，
  G3/G5 无一阶收敛证据（coarse 53 %→medium 82 % 实为随网格变差），"收敛性"段名不副实。

## 5 判级

| 级别 | 是否成立 | 依据 |
|---|---|---|
| L0 文档 | 成立（但不止） | dev_report + 平台报告 16328 六段齐 |
| L1 数学/单元 | 成立 | 9 tests（含 KK 负控、旋转中心双钉）实跑全过 |
| L2 独立功能实现 | 成立 | 真实代码链：网格→3D BEM 辐射扫频→时间域核→五门裁决→results.json |
| **L3 系统集成** | **成立（取此级）** | 走仓内主求解器 `free_surface_solver3d.solve_radiation_3d` + `time_domain`/`causality` 生产线；驱动/测试/工件/平台报告齐 |
| L4 基准验证 | **不成立** | 锚点核心判据（KK 因果 <10 %）FAIL，且无 benchmark case 登记行；报告好看不改变级别 |
| L5 科研工程 | 不成立 | 无新方法结论，且核心物理量未达容差 |

## 6 处置（disposition）

**approve with notes — as honest partial delivery at L3；L4 拒绝授予；锚点应保持 open（返修）。**
返修最小目标（按证据排序，非本单执行）：
1. 查清 G3 的真实来源：先对 A∞ 估计做敏感性（yaw 的 A∞ 拟出 −14.25，量级可疑），再用两档分辨率做
   KK 残差的网格收敛（F7）——**不要**再按"加低频点"的假设投入（F4）。
2. G4 medium 的半功率扫描窗要与实际频带自洽（当前 IndexError 是估计器窗宽 bug，不是"D 不存在"）。
3. medium 档补满 22 点后**重生成 results.json**（F6），并把五门登记进 `tensor_benchmark_cases`（F2）。
4. 修 `capytaine_ref.json` 的 `wigley3_hull` 段或标注 deprecated（F5）。

## 7 复现命令

```
PYTHONPATH=src python3 docs/board_audit/m17340/audit_independent.py   # 本审计全部数字（10.4 s）
PYTHONPATH=src python3 scripts/wigley3_recalled_ref_capytaine.py --out /tmp/ref_audit.json
cd docs/board_audit/m17340/subject_snapshot && PYTHONPATH=$OLDPWD/src python3 -m pytest test_cummins_wigley_kk_causality.py -q
```