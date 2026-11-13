# 交差核验单 m17310 — m17273 / CASE-BEM-ELLIPSOID-KOCHIN（TensorBEM 算例池）

审计者：board-audit（任务 m17310）· 被审对象：mission m17273 · 仓 TensorBEM · 被审 commit `c121bee`
（`case-pool: add CASE-BEM-ELLIPSOID-KOCHIN (ellipsoid Kochin far field)`，分支 `agent/m17273`）
审计时间：2026-11-13 03:42–03:50 UTC+8 · 平台 main 快照：`89e5a52 Merge #773`；ECO 快照：`96f19f7 Merge #758`

**性质声明：本单为交差核验，非重做。** 交付物一字未改；审计者做的唯一"计算"是**把被审 commit 原样检出来自己重跑一遍**，
以确认申报数字是算出来的、且能复现。审计者不修复被审代码。

---

## ① 案例与验收合同（判据先于动手）

| 项 | 值 |
|---|---|
| 锚点 | `CASE-BEM-ELLIPSOID-KOCHIN`（算例池 row id=170，`tol_policy=<5%`，`mission_id=17273`） |
| 物理 | 半浸薄椭球 `x²/a²+y²/b²+z²/c²≤1, z≤0`，a/b/c = 5.0/1.0/0.5 m（L/B/T = 10/2/0.5 m） |
| 被测 | 远场 Kochin 函数 `H(k)` 与经 Kelvin 积分的兴波阻力 `R_W`（Fr = 0.20/0.30/0.50） |
| 参考 | `H(k)` 闭式 + 长波极限 `H(k→0) = -i k V` + 远场/近场（Michell 压力积分）能量恒等 |
| 容差判据 | 池判据 **5 %**（面元路线，预声明包络 `k·a ≤ 10`） |
| 生态更严线 | <3 %（本单逐项看，见 ④） |

**基准裁判体检（第一枪就打这里）**

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
WHERE anchor_slug='CASE-BEM-ELLIPSOID-KOCHIN';   -- → 0 rows
```

审计时 `tensor_benchmark_cases` **没有本锚点的行**（同仓其余 7 个 BEM 锚点全有行：MACCAMY-FUCHS / BARGE /
SPHERE-ADDED-MASS / ELLIPSOID-ADDED-MASS / TWO-CYL-MUTUAL / HYDROSTATIC-PONTOON / CYL-DRIFT-ASYMP）。
即：**机器裁判缺席**，m17273 的"算例池条目"只落在分支工件里，没进裁判台账。平台自己的 L4 前置检查也这么判：

```
GET /api/v1/tensor/bench/l4-verdict/17273
→ {"status":"CLAIMED_NOT_PROVEN","l4_benchmark_pass":false,"rows":0,
   "failed_checks":["execution_success","source_cluster_job","cluster_job_bound",
                    "ref_got_error_present","within_tolerance"],
   "reason":"no tensorbench_results row for this mission: the L4 benchmark claim is self-declared, not measured"}
```

本审计按通行做法（同 m17267/m17233 在先）**代裁判补登行**，判据沿用任务原判据 5 %，**未放宽**（见 ⑤ F1）。

## ② 复现方法与审计独立性

```bash
git -C TensorBEM archive c121bee | tar -x -C runs/m17310/repro          # 被审 commit 原样检出（不改一行）
cd runs/m17310/repro
PYTHONPATH=src python3 -m tensorbem.benchmarks.ellipsoid_kochin_case --outdir runs_audit
# 退出码 0（全判据 PASS），wall 30 s（同机 512 任务并发，单核）
PYTHONPATH=src python3 -m pytest tests/test_ellipsoid_kochin_case.py -q   # 16 passed in 37.01 s
python3 runs/m17310/compare_m17273.py                                     # 逐数字 diff（见下）
```

局限（如实）：审计者**没有**独立第三实现去重写 Kochin 闭式，因此"参考值本身对不对"只由被审交付自带的
三条机器可核事实背书（两条独立实现一致 6.62e-14；长波一阶极限斜率 1.027；远/近场恒等 8.83e-14）。
三条中前两条共享同一 HULL 几何、第三条与被测量同源，**不是完全独立的外部锚**——若要 L5 级外部锚，需文献表值或实验值（本沙箱不可达）。

## ③ 复现结果：逐数字对照（零手抄，程序 diff）

`compare_m17273.py` 递归遍历两份 `results.json` 的全部数值叶：

| 项 | 结果 |
|---|---|
| 数值叶总数 | 264 |
| **逐位相同** | **263** |
| 唯一不同 | `runtime_seconds` 12.791265487670898（交付） vs 25.77441906929016（审计复跑）——计时，非物理量 |
| 非计时量最大相对差 | **0**（bit-identical） |
| sha256 交付 results.json | `55476266d2cae561fc599a4844b522c71f029fe4b20f4a9a0865c8202de6c3c1` |
| sha256 审计复跑 results.json | `6d874ce883f5058e369f0830c46cf83fe1ff12cc69331998531fff74393565c4` |

图 `figures/bem_ellipsoid_kochin.png` 复跑重建 94 510 B（可由 `results.json` 重建，非手绘图）。
→ **交付数字是算出来的，并且可零漂移复现**；`repro_cmd` 可执行（唯一坑：`repro_cmd` 未带 `PYTHONPATH=src`，
直接照抄会 ImportError，审计者补 `PYTHONPATH=src` 后通过——见 F4）。

## ④ 判据逐项核（复跑值，来自 `snapshot/rerun_results.json.criteria`）

| 判据 | 复跑值 | 限 | 判定 |
|---|---|---|---|
| geometry_cb_half_ellipsoid（几何门 Cb vs π/6） | 2.120370e-16 | ≤1e-12 | PASS |
| closed_form_vs_independent（两条独立实现） | 6.617957e-14 | ≤1e-9 | PASS |
| long_wave_asymptotic_order（`H→-ikV` 一阶斜率） | 1.026595 | ≥0.9 | PASS |
| long_wave_finest（k=1e-4 残差） | 1.877475e-05 | ≤1e-3 | PASS |
| kochin_panels_integrated（面元，包络内最坏） | **1.727781e-02** | ≤5e-2 | PASS |
| kochin_panels_ordinate（面元，包络内最坏） | **2.098326e-02** | ≤5e-2 | PASS |
| haskind_far_near_identity（远/近场能量恒等） | 8.830727e-14 | ≤1e-9 | PASS |
| haskind_far_panels（远场 vs 面元） | 9.554812e-03 | ≤5e-2 | PASS |
| convergence_finest（最细层面元） | 5.016805e-03 | ≤5e-2 | PASS |
| convergence_order（观测收敛阶） | 2.003790 | ≥0.5 | PASS |
| reference_parameter_sensitivity | 1.287129e-13 | ≤1e-9 | PASS |
| **[diagnostic]** long_wave_limit_three_point | 2.122090e-03 | ≤1e-3 | **FAIL（未计分，交付已自报）** |

11/11 计分判据 PASS；唯一 FAIL 是被交付**显式降级为 diagnostic** 且报告里如实写了（"round-1 长波判据 2.12e-3 vs 1e-3 的 FAIL 仍在 results.json.diagnostics"）——
**诚实 FAIL 不覆盖、不粉饰，判 PASS 无问题**。

**误差判定（生态 <3 % 线）**：面元路线最坏 2.098 %（ordinate，k·a=10）< 3 % ✔；远场 vs 面元 0.955 % ✔；最细层面元 0.502 % ✔。
**面元包络外（k·a > 10）不计分**，交付预声明并打 `in_envelope=false`——这点合规（避免"把超包络的 16 % 混进成功率"）。
未计分路线 `panels_source` 15.4 % 已被交付显式排除并写明理由（龙骨线 `y_x` 奇异不可点采样），不作为失败计入。

## ⑤ 交差三件套 + 裁判台账体检

| 三件套 | 真伪 | 指针 |
|---|---|---|
| run_artifacts | **真**：`results.json`(19 349 B)/`report.md`(10 133 B)/`figures/*.png`(94 510 B)，审计者复跑重建 | `benchmarks/verified/bem_ellipsoid_kochin/`（c121bee） |
| ref vs got 数值 | **真**：264 数值叶 263 逐位相同；11 判据 PASS，最坏 2.098 % < 5 % | `snapshot/rerun_results.json` |
| repro_cmd 可执行 | **真（需补 PYTHONPATH=src）**：审计者实跑，退出码 0，30 s | `snapshot/compare_stdout.txt` |
| impl 真伪（MOCK 扫描） | **真代码**：驱动 1 007 行 + 库 572 行；`grep -nE "1\.17(4|7)|0\.3044|0\.17803|expected|mock|stub|TODO"` → 0 命中；参考值全由 `kochin_analytic` 现算（零手抄） | `src/tensorbem/benchmarks/ellipsoid_kochin_case.py`、`src/tensorbem/ellipsoid_kochin.py` |
| 六段报告 | **齐**：①背景/几何/参数表 ②方法/工具链 ③数字表 ④收敛/敏感性（面元阶梯 + 参考参数敏感性）⑤长波判据 FAIL 的诚实保留 ⑥误差判定 | `docs/dev_report_m17273_ellipsoid_kochin_case.md` |

**发现清单（按严重度）**

- **F1 高（裁判缺席，已由本审计补）**：`tensor_benchmark_cases` 0 行 → L4 机器门 `CLAIMED_NOT_PROVEN`。
  本审计按 5 % 原判据补登行（`metric` 内含实测值 + `audited by m17310`），**未放宽**。
- **F2 高（对接未闭合）**：算例池 `tensor_case_pool.id=170` 仍是 `dispatched`（同期 ELLIPSOID-ADDED-MASS=134 已 `passed`）；
  且 `c121bee` **未并入 main**（`git archive c121bee` 才能检到；main `89e5a52` 与 ECO `96f19f7` 均无此驱动）。
  后果：装好的 TensorBEM 跑不出这条算例。
- **F3 中（集成只有仓内一环）**：被审 commit 内确有主线挂点——`src/tensorbem/figure_integrity.py:122-128` 把
  `bem_ellipsoid_kochin` 注册进案例注册表（module + builder），`tests/test_ellipsoid_kochin_case.py` 16 项测试覆盖。
  但 ECO 侧 `grep -rl --include=*.py -F ellipsoid_kochin_case /root/tensor-ecosystem/TensorBEM` = **0 命中**（未合并所致），
  故门方案要求的主求解器/平台侧 `integration_callers` **无法核真**。
- **F4 低（复现卫生）**：`results.json.repro_cmd` 无 `PYTHONPATH=src`，照抄跑 `python -m tensorbem...` 会 ImportError；`--outdir` 为仓内相对路径（好）。
- **F5 低（外部锚强度）**：文献只到卷/页级（Michell 1898、Havelock 1932、Newman 1977 ch.6），沙箱无原文，
  `provenance.page_verified=false` 交付如实标注；三条机器可核事实替代，但都不是独立外部数据。

## ⑥ 判级与裁定请求

**判级 = 3（系统集成在仓内成立：figure_integrity 注册表 + 16 项测试；但生态侧对接未闭合到 main，故不授 4）。**

理由分解（按 gate 判据方案逐层核）：
- L1（tests）：`tests/test_ellipsoid_kochin_case.py` 16 passed（审计者实跑 37.01 s）✔
- L2（impl_files/entry/run_artifacts）：1 007 行真实现 + `python -m` 入口 + 三件工件非空 ✔
- L3（integration_callers）：**仓内**挂点为 `src/tensorbem/figure_integrity.py:122-128`（注册 module+builder）与测试 ✔；
  但 ECO/main **0 引用**（未合并），故若门按 ECO grep 复核会降级，此处如实标"部分成立"。
- L4（benchmark+repro_cmd）：数值对照 11 判据 PASS、最坏 2.098 % < 5 %、复跑逐位一致、repro_cmd 实跑可执行——
  **内容面上达到 L4 实质**，但缺"行在 main 里"的对接，故级别按阶梯取上限 3，不虚标 4。

**请求（remediation）**：
1. 合并 `agent/m17273`(c121bee) 进 main，并在 `tensor_case_pool.id=170` 置终态（本审计已补裁判行）；
2. 合并后重跑 `GET /tensor/bench/l4-verdict/17273` 应转 PASS（届时可升 L4，以机器重算为准，不是本审计的结论）。

**明确未做（not_run，不得记为通过）**：独立第三实现重写 Kochin 闭式（未做）；文献原表原页核对（沙箱不可达）；
集群重算 / 提交行 `tensorbench_results`（未本审计补登，只补了 `tensor_benchmark_cases` 裁判行）。

## 工件

```
docs/board_audit/m17310/AUDIT_m17273_CASE_BEM_ELLIPSOID_KOCHIN.md   本报告
docs/board_audit/m17310/compare_m17273.py                            逐数字 diff 脚本（零手抄）
docs/board_audit/m17310/snapshot/rerun_results.json                  审计复跑 results.json
docs/board_audit/m17310/snapshot/subject_results_c121bee.json        被审交付 results.json（原样）
docs/board_audit/m17310/snapshot/rerun_report.md                     审计复跑 report.md
docs/board_audit/m17310/snapshot/bem_ellipsoid_kochin.png            审计复跑重建图
docs/board_audit/m17310/snapshot/compare_stdout.txt                  diff 实跑输出
docs/board_audit/m17310/snapshot/SHA256SUMS.txt                      全部工件与源码 sha256
runs/m17310/repro/                                                   c121bee 原样检出 + runs_audit/ 复跑输出
```