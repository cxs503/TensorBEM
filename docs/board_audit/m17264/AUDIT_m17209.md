# 交差审计：m17209 CASE-BEM-CUMMINS-WIGLEY（Wigley III 6-DOF Cummins 记忆核，TensorBEM）

*审计席位 board-audit · 审计任务 m17264 · 对象任务 m17209（status=succeeded）· 对象分支 `agent/m17209`
@ `f8a5442c7a8282ea66bc583c79d42ad8313a9a0b`（仅在该分支，未进 origin/main@0516776）*
*审计判级：**L3**（不是 L4）· 处置：**reopen 返修**（基准未过；且 R1 失败系算例自身 bug，非求解器缺陷）*
*独立性局限：审计者(=本席位)未参与 m17209 实现；除 `artifacts/audit_m17264/center_check_20x10.json`
中那一次 400 面元重解外，所有数字均从对象自己的工件重算/重绑定，未重跑 467 s 的整段扫频。*

---

## 0 结论先行

| 问题 | 机器判据 | 结果 |
|---|---|---|
| 三件套是否真存在 | run_artifacts / ref-vs-got / repro_cmd | **存在且可核**（C1–C7 全 PASS，见 §2） |
| got 是否在容差内 | 算例池 tol `<10%`（`tensor_case_pool` case_code=`BEM-CUMMINS-WIGLEY`） | **不在**：R1=0.5550、R3=2.2426、R4=nan、R5=2.4272 四门 FAIL |
| 报告六段是否齐 | 六要素 | **齐**（①–⑥ + 3 图，数字可逐项追到 `results.json`） |
| 失败归因是否成立 | 报告称「TensorBEM 对 roll 的 A_ii 偏差 55.5 %」 | **归因错误**：该 55.5 % 是算例漏传旋转中心（默认 z=0）而参考件用 z=−T/2 的口径错配；补上 `center=(0,0,−T/2)` 后同一网格 A44=−2.61 %、B44=−3.25 %，**R1 由 FAIL 转 PASS** |
| 级别 | 功能实质 | **L3**：真代码+真 3-D BEM 扫频+工件+确定性复现，但基准未验证（4/5 门 FAIL），且交付未合入 main |

---

## 1 取证（三件套逐项）

**run_artifacts（真存在，非空壳）**
`artifacts/cummins_wigley_kernel_6dof/` 三个文件（已快照到 `docs/board_audit/m17264/subject_artifacts/`，
sha256 记入 §5）：`results.json`（30 KB，含 gates/per_dof/kernels/free_decay/cross_solver/sensitivity）、
`sweep_coarse.json`（100 面元×8 频率，wall 40.2 s）、`sweep_medium.json`（400 面元×8 频率，wall 467.5 s）。
**benchmark ref vs got（真数值，零手抄）** R1 的参考来自仓内冻结件
`benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`（capytaine 3.1.dev0 / loongarch64 /
`rotation_center=[0,0,−0.09375]` / 水深远洋 / ρ=1025 / g=9.80665），模型在运行时读入，报告表格与工件逐项一致。
**repro_cmd（可执行）** 审计实跑：`--stage gates` 4 s 内重算出 `verdict: FAIL:R1,R3,R4,R5`
且五门数值与 `results.json` **逐位相同**；`pytest tests/test_cummins_wigley_kernel_6dof.py -q` → **4 passed, 0.89 s**。

**基准裁判（唯一裁判）**：`tensor_benchmark_cases` 中 `anchor_slug='CASE-BEM-CUMMINS-WIGLEY'` **0 行**
（全表 39 行，TensorBEM 已有 5 个 anchor 在册）——判据只存在于 `tensor_case_pool`
（`case_code=BEM-CUMMINS-WIGLEY`，`ref_source='Cummins 方程 + Kramers-Kronig 因果'`，`tol_policy='<10%'`，
`status='dispatched'`，`mission_id=17209`）。**机器裁判缺位**（见 F4）。

---

## 2 复算与绑定（`docs/board_audit/m17264/audit_evidence_binding.py`，8 项全跑，秒级）

| 检查 | 内容 | 结果 |
|---|---|---|
| C1 | 工件存在且可解析 | PASS |
| C2 | 用对象自己的 `run_case` + 缓存扫频**重算五门** vs `results.json` | PASS，逐位相同（R1 0.554954 / R2 0.000317862 / R3 2.24263 / R4 nan / R5 2.4272） |
| C3 | verdict 与门表、容差自洽（无「叙述式通过」） | PASS：`FAIL:R1,R3,R4,R5` 与逐门复核一致 |
| C4 | 几何先验证 | PASS：Cb=0.543，∇=0.09163125 m³ = L·B·T·Cb = 0.09163125，L/B=10.0，B/T=1.6 |
| C5 | 报告 §4 R1 表 6 行（ref/got/误差）绑定 `cross_solver.rows` | PASS（3–4 位有效数字内一致） |
| C6 | 报告门表 + 自由衰减 ζ=0.00439 绑定工件 | PASS |
| C7 | 参考件为仓内冻结件、6 DOF ref 值逐位相同 | PASS（tool/version/platform/rotation centre 齐全） |
| C8 | roll 亏损是否网格分辨率极限 | **系统效应，非分辨率**：A44 ref=0.6700，coarse=0.3353（−49.96 %），medium=0.2982（−55.50 %）——加密**远离**参考，比值 2.247 |

脚本退出码 0（C1–C7 全 PASS）；`artifacts/audit_m17264/binding.json` 为机读结果。

---

## 3 发现（按严重度）

**F1｜blocking｜R1 失败是算例 bug，不是求解器缺陷（且报告把它写成了求解器偏差）**
`src/tensorbem/benchmarks/cummins_wigley_kernel_6dof.py:171` 调 `solve_radiation_3d(mesh, ν, density, gravity, chief_points="auto")`
**未传 `center=`**，故 roll/pitch 绕求解器默认 `(0,0,0)`（水线）求矩；而同仓早前研究
`src/tensorbem/benchmarks/wigley3_recalled_ref.py:166-172` 与参考冻结件都用 `center=(0,0,−T/2)`。
审计在**同一 400 面元、ω=1** 上按参考口径重解（56.8 s，单核）：A44 **0.29819→0.65249**（−55.50 %→**−2.61 %**），
B44 **1.056e-5→5.852e-5**（−82.54 %→**−3.25 %**），pitch A 由 −4.12 % 变 −1.45 %，
surge/sway/heave/yaw **不变**（刚体运动学要求）。R1 门值 **0.55495 → 0.02613 = PASS**。
⇒ m17209 报告 §④/§⑥ 的「最差 A_ii 相对 roll +55.50 %」是对 **TensorBEM 的假阴性**；
该结论若进台账会把一个正确求解器记成缺陷。数据：`artifacts/audit_m17264/center_check_20x10.json`，
探针：`docs/board_audit/m17264/audit_rotation_center.py`。

**F2｜blocking｜基准仍未过（R3/R4/R5 结构性不可能通过，属算例设计缺陷）**
- R4=nan：heave 固有频率 8.14 rad/s 落在扫频带 [0.5, 6] **之外**，半功率带宽法无共振点
  （`free_decay_heave.half_power.error='no resonance inside the tabulated band: Re kappa does not change sign'`）；
- R3=2.2426、R5=2.4272：KK 主值积分/往返闭环要求覆盖 ω→0 与 ω→∞，8 点 [0.5, 6] 截断尾巴，
  而同一模块的自测用 0.3–8 rad/s、41 点、B 两端归零时误差 <2 %——**门本身有效，是算例频带选得不够**。
⇒ 该算例当前配置下**不可能**给出 PASS；要 L4 必须重设频带（含 ≥8.14 rad/s）与更密频率节点后重跑。

**F3｜medium｜交付未合入 main**：`cummins_wigley_kernel_6dof.py` 等 5 个文件只在 `agent/m17209`
（`f8a5442`），origin/main@0516776 无此路径 → CI 无人守护、后继无法复用；报告 `repro_cmd` 指向可回收工作树绝对路径。

**F4｜systemic｜机器裁判缺位**：`tensor_benchmark_cases` 对 `CASE-BEM-CUMMINS-WIGLEY` **0 行**，
`tensor_dev_missions.result.evidence.server_quality_audit = {gate_valid:false, key_present:false, gate_missing_warning:true}`
——算例池派单不写 benchmark 行，benchmark 门控键缺失，任务仍被 `TP_AUTO_APPROVE_DEV_ACCEPTANCE` 自动批准。
本轮审计已把 got/ref/容差/复现命令写进 verdict evidence 以补登记位。

**F5｜low｜报告内部表述瑕疵（不影响数值溯源）**：§③「网格（门禁档）…800 面元」用的是参考件的
三角形计数，而对象自己的扫频是 **400 四边形面元**（§⑤ 表也是 400）；两者是同一网格的两种计数，宜统一单位。

**F6｜credit｜该给的分**：几何闸门先过（Cb/∇/L·B/T 一致到 1e-12）；报告六段齐全、FAIL 未粉饰、
逐 DOF 失败项全列；`--stage gates` 4 s 确定性重算 + 4 个单测（含 KK 负控：把 A 破坏为 0.5×(A−A∞) 后
R3 由 <2 % 升到 32 %，证明门会咬人）——**方法学与证据纪律是合格的，错在口径与频带**。

---

## 4 判级与失分项

- **L3**：真代码（536 行 benchmark 模块 + 123 行驱动 + 89 行测试）、真 3-D BEM 扫频（400 面元 × 8 频率，
  467.5 s）、真工件、确定性复现、接入仓内既有求解栈（`solve_radiation_3d` → `radiation_memory_kernel`
  → `cummins_free_decay` → `kkr_added_mass`/`fourier_round_trip`）→ 满足 L2/L3 实质。
- **不给 L4**：基准判据未过（R3/R4/R5 结构性 FAIL；R1 即便修口径后 PASS，四门仍不通过），
  且 F2 属算例设计而非求解器能力；L4 要求「完整算例+误差分析+可复现」**并通过容差**。
- **不给 L5**：无外部实验/文献锚点（`ref_source` 是 Cummins/KK 自洽关系），R2–R5 是同源 BEM 扫频的
  数学路径互检，只有 R1（单频点）能发现系统偏差。

**返修清单（reopen 建议）**：①`radiation_sweep` 传 `center=(0,0,−T/2)` 并加单测钉住口径；
②扫频带扩到 ≥10 rad/s、节点加密到 ≥16（含 heave 固有频率 8.14 rad/s）后重算 R3/R4/R5；
③把 `CASE-BEM-CUMMINS-WIGLEY` 写入 `tensor_benchmark_cases`（ref/got/tol 逐项）；④合入 main；
⑤统一「面元/三角形」计数口径。

---

## 5 证据索引（sha256，供 gate 机器复核）

```
subject_snapshot/cummins_wigley_kernel_6dof.py   6cd0ca07b9e0d5b48e830833f0205fba10a8abca56722c0d05771cd45c70eb9b
subject_snapshot/m17209_cummins_wigley_kernel_6dof.py 3a6185e54ccd92b2607de2d248afb187562d08f32d27fbb0733f5f968aff5c2c
subject_snapshot/test_cummins_wigley_kernel_6dof.py   4db51de5ce94530f5db13ed57d41b1fa4edb64f0592412d144e6086e57536413
subject_artifacts/results.json                    b1cb219920e2b0c8ec161c03621d320bbf8c867fbf18c723e92f4e31cfee58da
subject_artifacts/sweep_coarse.json               8ba5d1094a78362ae4be6889bc89871ae2aa9b197db93ce23ae4a007e631fc55
subject_artifacts/sweep_medium.json               c4fb512039d29f6da81254f0c47515dcf4b5efba82d0b3a4599dba0ee1386fe6
subject_report/dev_report_m17209_…md              214152f1922f8a6e0dc7241d0a582b77b6c7397391be6f6a4acae66277d84e96
figures/fig_m17209_cummins_kernels_6dof.png       0a11749d56d2c98cd11cce89ab6a1d44cc4365050958e7f8cf5f8510b8b1e738
figures/fig_m17209_cross_solver_6dof.png          bbd81e5fabbe166bcec673a8bd566b761890e59e66136f69e97529b40b2a4d5d
figures/fig_m17209_gates_6dof.png                 2627297b645cf584263b810b28334619e74f72e13a74823f6e10b8e17acf3172
```

**未跑/不可得**：①参考件最细档 40×20 在参考口径下的重解（集群作业 cj1472 排队中，未在预算内完成）；
②同侧 400 面元 default-center 重解未重跑（直接用对象 sweep_medium 工件，C2 已证逐位一致）；
③对象声称的 `--stage sweep` 全链重跑（467.5 s）未复跑，只复算 gates 阶段。