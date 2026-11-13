# 交差核验审计 — m17204 `CASE-BEM-CYL-DRIFT-ASYMP`（圆柱平均漂移力 (ka)³ 律）

审计单：m17263 · 审计对象：m17204（`succeeded`）· 仓：TensorBEM · 判定：**L4**（不升 L5）
审计时间：2026-11-13 · 审计者：board-audit（独立于 m17204 作者）

本单不是重做，是**交差核验**：按任务榜令逐项核 `run_artifacts` / `benchmark ref vs got` / `repro_cmd`
三件套是否真存在，基准以 `tensor_benchmark_cases` 为唯一裁判，级别只看功能实质。

---

## 0. 结论摘要

| 项 | 结论 |
|---|---|
| L 级 | **L4**（真实面板 BEM + 两条独立闭式 + 误差表 + 面板收敛梯 + 一键复现） |
| 不升 L5 的理由 | 参照物是**解析闭式**（MacCamy-Fuchs+Maruo）的仓内实现，无印刷数值表/实验锚点；`5π²/16` 无逐字文献页可引（如实 WITHHELD） |
| gate 裁定 | verdict id=509 落表（`gate_state=pending`，由 maturity_gate 机器复核） |
| 关键阻断项 | `tensor_benchmark_cases` 对本锚点 **原为 0 行**（机器裁判无对照），本审计补登 row id=43（tol `<5%`，来自 case pool id=140） |
| 精度 | 主闸门（远场）最差 **2.028 %** < 5 %，且 < 3 % 生态线；800 面板近场 6.082 %（> 5 %，已披露，1776 面板 2.123 %） |

---

## 1. 核验对象（reviewed commit / artifacts）

- 提交：`3fe683d7944eb032fed18573f6126fc0e6a29143`（分支 `agent/m17204`，父 `96f19f7ced`）
  8 文件 +1641 行：算例驱动 612 行、测试 96 行、案例文档 138 行、`results.json` 531 行、3 张 PNG、报送脚本。
- 工件：`/root/tp-agent/work/m17204/artifacts/case_cyl_drift_asymptotic/`
  - `results.json` sha256 `ac14e4e5…`（本审计复算，见下）
  - `cyl_drift_coefficient.png` sha256 `de26d65d…`
  - `cyl_drift_long_wave_plateau.png` sha256 `d91b73da…`
  - `cyl_drift_bem_convergence.png` sha256 `52b2ab9e…`
- 报告：`tensor_reports` id=16239（六段齐全，3 图内嵌）
- 基准位：`tensor_case_pool` id=140（tol `<5%`）；`tensor_benchmark_cases` **本审计前 0 行 → 补登 id=43**

## 2. 独立性限制（本人不能声称的）

1. 本审计与 m17204 作者共用同一台 99 控制面与同一份仓/环境变量；**没有**第二套 BEM 求解器。
2. 我没有重跑 750 s 全量链路；提交了集群作业 `cj1470`（同一 `repro_cmd`），审计收尾时仍在排队 →
   标 **not_run**，不用「应当一致」冒充通过。
3. 我是通过**重算关键量**与**重跑粗档**取证，不是逐行代码复审全部 612 行。

## 3. 做过的检查（scope / checks performed）

| # | 检查 | 方法 | 结果 |
|---|---|---|---|
| C1 | 工件存在性 | `ls` + `sha256sum` 四件 | 全部存在，sha256 已入 verdict evidence |
| C2 | 算例测试 | `PYTHONPATH=src pytest tests/test_case_cyl_drift_asymptotic.py`（m17204 树内） | **9 passed in 2.87 s** |
| C3 | 长波系数 `5π²/16` 独立复算 | 自写 Maruo 远场积分（自写 `S(θ)`、自选求积，仅共用 scipy Bessel），`ka=1e-4…5e-2` | `ka=1e-4` → **3.0842513989**，对 `5π²/16=3.0842513753` 相对差 **7.65e-9** |
| C4 | BEM 面板梯粗档复跑 | `bem_case(ka=0.5, mesh=16×6 / 24×9)` 经仓内驱动 | 208 面板 far **7.3956 %** / near **30.2024 %**；456 面板 far **3.1792 %** / near **13.3502 %**；与工件差 **≤1.1e-15**（数值同位） |
| C5 | ref vs got 逐项对表 | 工件 `bem`/`long_wave`/`convergence`/`azimuth` 与 `docs/case_cyl_drift_asymptotic.md` 表 4、5 对读 | 逐项一致（far 2.028/1.068/1.317 %；law@ka=0.02 = 0.011684 %；收敛梯 7.396/3.179/2.028/1.406 %） |
| C6 | 闸门语义 | 读 10 个 gate 定义与阈值 | 阈值在算例内**预先声明**（`TOL=5%`），非事后调参；G2 用「律在 `ka=0.30` 必须已超差」的反向测试锁边界 |
| C7 | 集成调用链 | 追踪 `drift_far_field_3d_study.py:97/100/104` 与 `cylinder_drift.py:206/277/322/340` | 真实走在主 3D 绕射求解器 + Kochin + Maruo 核上，未另起炉灶 |
| C8 | 2026-11-13 邻近算例对照 | 同级审计（m17237/m17232 等）已把 BEM 算例判到 L4 的先例 | 本算例证据面**不弱于**已判 L4 的兄弟算例（多一条解析系数独立复算） |

## 4. 分级判定（L0–L5）

- L1 数学/单元：解析闸门（长波律、斜率 2.9964、系数恒等、近场/远场互校 2.8e-15、幅值² 律 4.000000）✓
- L2 独立功能：真代码，输入（`ka`、网格、`ρ,g,A,a`）→ 计算 → 输出（N、无量纲系数）✓
- L3 系统集成：调用 `solve_diffraction_3d` / `bem_kochin` / `maruo_far_field_drift` 主求解器层 ✓
- L4 基准验证：完整算例 + ref/got 误差表 + 4 档收敛梯 + 方位敏感性 + 一键复现 + 9 测试（含负向对照）✓
- L5 科研工程成果：**未达**——参照物是解析闭式的仓内实现，无印刷数值表/实验；`5π²/16` 无逐字文献页（WITHHELD）✗

## 5. 严重度排序的发现（findings）

**F1（high，系统级）基准登记缺行。** `SELECT … WHERE anchor_slug='CASE-BEM-CYL-DRIFT-ASYMP'` 在本审计前返回 **0 行**；
按任务令「基准判据是唯一裁判」，机器裁判当时**无对照可比**，报告再好看也不能自证。
复现：`runuser -u postgres -- psql -d tensorplatform -Atc "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-CYL-DRIFT-ASYMP'"`。
处置：本审计按 case pool id=140 的 `tol_policy='<5%'` 与 mission 17204 判据补登 **row id=43**（status=passed，
metric 内含 got 数值与本审计复算值）。与 m17237 在 m17205 上记的 F1 同型 → 属**系统性登记缺口**，非本算例独有。

**F2（medium，交付链）成果只在分支，未进 main。** 提交 `3fe683d` 仅存在于 `agent/m17204`；`origin/main` HEAD `0516776`
不含算例驱动/测试/文档/四个工件（本审计在 m17263 树内 `ls` 全为 missing，`git merge-base --is-ancestor` = no）。
影响：**从干净 main 检出无法复现**。处置：按平台约定（验收通过后自动推送分支）等待推送；本审计不 push、不改 `.git`。
证据指针因此同时给「分支内提交路径 + 本审计 worktree 实路径」。

**F3（medium，能力边界）800 面板近场压力积分 6.082 % > 5 % 判据。** 只有 1776 面板档（2.123 %）达标；
即 gate `G8_bem_near_converged` 是**网格外推陈述**而非主网格结果。检查：工件 `convergence.rows` + 本审计 C4 复跑。
判断：**不构成虚标**——案例文档表 4(b) 明确打 ✗、第 5 节明说「近场是求积受限，主闸门放远场」，主闸门 G7 用的是远场。
属如实披露的能力边界，建议下轮把 `G8` 的 `what` 字段改写为「1776 面板外推值」以免误读。

**F4（low，文献）`5π²/16` 无逐字来源。** Maruo 1960 全文不可达（出版方 403），案例声明为「推导量」并三路验证。
本审计追加**第四条独立路径**（自写远场积分，7.65e-9）→ 数验证到位，引文缺口仍在（故 L5 关闭）。

**F5（low，方法学混杂）BEM 用 `kd=5` 有限水深，闭式是深水。** 收敛梯到 1776 面板仍残 ~1.4 %，未报 Richardson 极限，
故该残差混了离散误差与 `kd=5` 截断（`e⁻⁵≈0.7 %`），无法单独归因。不影响 <5 % 闸门结论，但「收敛到 0」的主张不被支持。

**未发现的问题（明确记录）**：文档引用的 `tests/test_cylinder_drift_asymptotic.py::test_wall_anchor_reproduces_rho_g_A2_over_2`
**真实存在**（`tests/test_cylinder_drift_asymptotic.py:49`），非引用幻觉；`results.json` 的 `gates.all_pass=true` 与逐项数值自洽；
`plan`/`figures` 三张 PNG 皆有 sha256 与重绘路径。

## 6. 未做的检查（checks not run）

- 750 s 全量端到端复跑：集群 `cj1470` 已递交（`stage_dir=/root/tp-agent/work/m17204`，同一 `repro_cmd`），审计收尾时仍在排队 → **not_run**。
- 印刷数值表（Maruo/Havelock 的 `C(ka)` 表）对照：沙箱内不可得 → L5 维持关闭。
- PNG 像素级内容：只验 sha256 与重绘入口。

## 7. 是否夸大（overstated claims）

| 声称 | 核验 | 判定 |
|---|---|---|
| 「远场误差 2.03 %」 | 工件 0.020284；粗档独立复跑到 1e-15 | 成立 |
| 「长波系数 0.012 %」 | 工件 1.168e-4；独立复算 1.168e-4（ka=0.02） | 成立 |
| 「近/远场互差 2.8e-13 %」 | 工件 2.835e-15（相对） | 成立（注：两条路线共用同一 `A_m` 展开，互校强度弱于与 BEM 的互校） |
| 「律的边界 ka≤0.24」 | 程序扫 + 反向测试锁 ka=0.30 超差 | 成立 |
| 报告六段 / 3 图 | id=16239 核读 | 成立 |

## 8. 处置（disposition）

**approve with notes**：L4 成立、基准行已补登（id=43）、verdict 已落表（id=509，gate 机器复核）。
notes = F2（待平台推送分支后从 main 可复现）+ F3（G8 表述建议改为 1776 面板外推）+ F4/F5（L5 关闭理由与残差归因）。

---

## 附：本审计自产工件

| 路径（m17263 worktree） | 内容 |
|---|---|
| `audit/m17204/repro_independent_checks.py` | 独立远场积分复算 + BEM 粗档复跑（自包含，可复跑） |
| `artifacts/m17263_audit/independent_checks.json` | C3/C4 全部数字 |
| `artifacts/m17263_audit/coefficient_check.json` | 仅解析部分（~3 s 版） |
| `audit/m17204/register_benchmark_and_verdict.py` | 幂等登记脚本（benchmark row + verdict） |
| `docs/board_audit/m17263/AUDIT_m17204.md` | 本文件 |