# 交差核验 m20079 · CASE-BEM-HELMHOLTZ-SPHERE（TensorBEM）

**判级 L3（系统集成）** · maturity_gate 自动裁定见 verdicts 行 · 处置：request-changes on delivery,
L4 拒绝。

> 本报告的全部数字由 `docs/audit_m20147/make_audit_artifacts.py` 从被审工件的 CSV/JSON 与本审计
> 自己写的第三实现（`independent_mie.py`）程序化注入，无手抄。被审对象 = mission m20079 的工作树
> `/root/tp-agent/work/m20079`，其交付**未提交**（见 §5 F1）。

## 1. 案例介绍

刚性（声硬）球，半径 a = 1 m，单位平面波 e^{ikx}（时间因子 e^{-iωt}）入射，边界条件
∂(φ_inc+φ_sc)/∂n = 0。被比对的量是无量纲总散射截面

  σ̂ = σ_sc/(π a²),  σ_sc = ∫|f(θ)|² dΩ,  f(θ) = (−i/k) Σ_n (2n+1) A_n P_n(cos θ),  A_n = −j_n′(ka)/h_n^(1)′(ka)

唯一无量纲群是 ka，故基准曲线就是 σ̂(ka)。判据（案例自己预设）：ka 扫描上 max|σ_BEM − σ_ref|/σ_ref < 6 %。

| 参数 | 值 |
| --- | --- |
| 半径 a | 1.0 m |
| ka 扫描（committed） | 0.50 … 2.00（3 点） |
| 网格 | icosphere(subdiv=2) = 320 面板，单网格 |
| 单位波长面板 | 16.0 – 64.0 |
| 远场积分 | Gauss–Legendre n_θ = n_φ ∈ {32, 64} |
| 内点 CHIEF 行 | 5（ka = π 处） |
| 预设容差 | 6%（编译期常量 `TOLERANCE`） |
| 参考 | 解析分波（Rayleigh/Mie）级数，mpmath dps=30 + scipy 双实现 |

## 2. 怎么做的（审计方法与工具链）

1. **读工件**：把 `docs/case_bem_helmholtz_sphere/`（CASE.md / NOTES.md / numbers.json / verdict.json /
   8 张表 / 5 张图 / run.log / integration/integration_call.json）逐项打开，核对「三件套」——
   run_artifacts 文件是否存在且非空、ref vs got 数值、repro_cmd 能否真跑。
2. **独立第三实现**：审计者不 import tensorbem，仅用 scipy `spherical_jn/yn` 自写 Riccati–Bessel 分波级数
   （`independent_mie.py`），重算 σ̂ 并与被审的 `sigma_hat_ref` 列逐一比较；同时核两个解析极限 7/9（ka→0）
   与 2（ka→∞）。
3. **复跑**：`run_case.py --stage cheap` 在审计环境下重跑一次，把 8 张表与 numbers.json 与被审版本做
   逐字段 diff（只允许 wall_seconds 不同）。
4. **测试**：在被审工作树里跑 `pytest tests/test_case_helmholtz_sphere.py -q`。
5. **裁判行**：直接查库 `tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-HELMHOLTZ-SPHERE'`
   与 `tensor_case_pool`，核有没有官方裁判/池状态是否推进。
6. **交付身份**：`git status --porcelain` / `git log --all --grep=m20079` / 与 main(066cf3c5) 和
   `/root/tensor-ecosystem/TensorBEM` 比对，判断交付是否入库。
7. **机器判据**：按 `maturity_gate.SCHEME`（L1 tests / L2 impl+entry+artifacts / L3 +callers /
   L4 +benchmark+repro / L5 +external_anchor）给证据指针，级别由机制裁定，不由本报告语气裁定。

工具链：Python 3.11 + numpy/scipy/mpmath + matplotlib + psycopg + pytest + git；集群未用（本单是核验单，
最重一步 42.9 s 单进程即可完成，属 <8 核 <10 min 的本机短测试预算）。

## 3. 计算结果

### 3.1 got vs 参考 vs 审计者第三实现（决定性表）

| ka | 面板 | 面板/波长 | σ̂_BEM (got) | σ̂_ref (case) | 对 case ref 误差 | 对审计者级数误差 | < 6% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 320 | 64.0 | 0.036552 | 0.038744 | 5.656 % | 5.656 % | OK |
| 1.00 | 320 | 32.0 | 0.309091 | 0.321629 | 3.898 % | 3.898 % | OK |
| 2.00 | 320 | 16.0 | 0.766779 | 0.781500 | 1.884 % | 1.884 % | OK |

- 最坏点 **5.656 %**（ka = 0.5, 320 面板）；
  用审计者自己的级数做分母时最坏 5.656 %——两条分母答案一致到 1e-3 %。
- 参考列可信：case 内两实现最大相对差 **8.52e-16**，
  审计者第三实现与 case 参考列最大相对差 **1.07e-15**。
  解析极限：审计者算得 ka=0.02 时 σ̂/(ka)⁴ = 0.777485
  （精确 7/9 = 0.777778），ka=100 时 σ̂ = 1.91582（精确 2）。

### 3.2 网格阶梯（ka = 1，只有两档）

| 面板 | 边/波长 | 相对误差 | 观测阶（对边长的阶） |
| --- | --- | --- | --- |
| 80 | 0.0608 | 14.670 % | - |
| 320 | 0.0312 | 3.898 % | 1.99 |

最细网格（320 面板）误差 3.898 %，
而案例预设的阶梯判据是 **< 1 %** → `passed=false`（blocking=false）。
即：**案例自己预设的收敛判据没有达到**，PASS 只靠 6 % 主判据。

### 3.3 其余敏感性

| 项 | 数值 | 预设限 | 判定 |
| --- | --- | --- | --- |
| 远场求积（n=32 vs 64）散布 | 0.000e+00 % | 0.5% | PASS |
| CHIEF 开/关（ka = π） | 1.249 % / 8.865 % | 6 % | CHIEF 承重（cond 65.965） |
| 角向远场 f(θ)（ka = 2, 73 角） | 4.915 % | 6 % | PASS |

CHIEF 的边界残差 4.02e-03 相对无 CHIEF 的 1.28e-15 大 12 个量级
（多约束行的正常代价），但误差从 8.86 % 降到 1.25 %，
证明内点行确实在起作用，不是摆设。

### 3.4 复跑可复现性

`--stage cheap` 重跑耗时 real	0m42.906s，
8 张表 + numbers.json 与被审版本逐字段对比：**只有 wall_seconds 不同**
（numbers/verdict/table_sweep/table_ladder 的物理量值完全一致，table_quadrature/table_chief/table_pattern/table_reference 文件 byte-identical）。

## 4. 结果图

图 1：σ̂(ka) —— BEM 点 vs case 参考列 vs 审计者独立级数（密网格曲线）。
图 2：左＝逐 ka 误差与 3 %/6 % 线（两套分母），右＝网格阶梯与预设 1 % 判据线（可见 3.898 % 越线）。
图 3：ka = 2 的远场角向 |f(θ)|，BEM vs 参考。

（图内数字与 §3 表格同源，均由脚本注入。）

## 5. 注意点（坑、局限、发现）

**F1【blocking · 交付】** 交付完全未入库。被审工作树 `git status --porcelain`：

```
M src/tensorbem/benchmarks/case_pool_registry.py
?? benchmarks/verified/casepool_bem_helmholtz_sphere/
?? docs/case_bem_helmholtz_sphere/
?? scripts/integration_caller_helmholtz_sphere.py
?? src/tensorbem/benchmarks/case_helmholtz_sphere.py
?? tests/test_case_helmholtz_sphere.py
```

分支头 `43e59824`（早于最后两次 master 合并），`git log --all --grep=m20079` 命中 0 条；
`docs/case_bem_helmholtz_sphere/` 与 `case_helmholtz_sphere.py` 在 main(066cf3c5) 与
`/root/tensor-ecosystem/TensorBEM` 里都不存在。**mission 标记 succeeded，但零 commit，无法 push/merge，
审计也无法把结论绑定到 commit 身份。**（对比：同期正常单会有 "mXXXXX: matured mission deliverables" 提交。）

**F0【lineage｜返修链】** mission 回放：本锚点上一轮审计（verdict#810, mission 19997）声明 L3 被 gate 机器复核降为
L1（`impl_files/entry/run_artifacts/tests` 指针全部指不到工件），gate 据此开返修单 **m20079** —— 即本单。
也就是说：本单是**返修交付**，而它把「交付未入库」这个原始缺陷原样重复了一遍（F1）。这说明返修单没有把
「commit 落库」当作验收前置条件，建议返修单模板里加一条机器可核的门：`git status --porcelain` 为空。

**F2【blocking · 裁判】** `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE
anchor_slug='CASE-BEM-HELMHOLTZ-SPHERE'` → **0 行**（全表 123 行；含 helmholtz 关键字的只有
CASE-BEM-ACOUSTIC-PULSE，脉冲球、边界条件不同）。上一轮审计 m19997（report 18232）就以「裁判行缺失」为由
拒绝 L4，本轮复检该行仍未登记。`tensor_case_pool` 里 `BEM-HELMHOLTZ-SPHERE` 状态仍是 `dispatched`、
绑定 mission 19973，未随 m20079 推进。→ 「基准判据是唯一裁判」这条链路目前**断着**。

**F3【material · 算例完整性】** committed 的是驱动器的 `cheap`（CI 冒烟）档：3 个 ka、单网格 320 面板、
阶梯 2 档。驱动器自己声明的 `full` 生产档（9 个 ka、1280 面板、阶梯到 5120、求积 32..96）**不在 numbers.json**
里；`NOTES.md` 自述 cj2466 的产物因 `runs/` 被 gitignore 而丢失、cj2494 还在排队。
因此「完整算例」不成立，且 PASS 余量只有 6 % − 5.656 % ≈
0.34 pp。

**F4【material · 披露】** `verdict.json` 与 `run.log` 都打印 `failed_checks: []`，但 checks 里
`mesh_ladder_converged` 的 `passed=false`。原因在
`case_helmholtz_sphere.py:652`：`failed = [c for c in checks if c["blocking"] and not c["passed"]]`
只留 blocking 失败。后果：任何读 `failed_checks` 的下游（审计/看板）得到**假的全清**；CASE.md §1 只写
"Verdict: PASS"，§4.1 给了 3.898 % 的表但没说「预设 <1 % 判据未达」。

**F5【material · 溯源】** 随数字一起提交的 `run.log` 记录的是 `stage: render`（driver_wall 3.24 s）——
它**不是产生这些数字那次运行的日志**，其 `repro:` 行是 `--stage render`（只重建 CASE.md/图，不重算）。
CASE.md §7 给的 `--stage all` 又会用未提交的 full 配置**覆盖**这些数字。真正能 bit-identical 复现
committed 数字的命令是 `--stage cheap`（本审计已跑，见 §3.4），它只写在
`integration/integration_call.json` 里。

**F6【info · 正面】** 无 MOCK/STUB：σ 来自真实稠密 Helmholtz 边界积分求解（边界残差
2.00e-05–1.79e-04）+ 64×64 远场积分；CHIEF 行在 ka=π
承重（8.865 % → 1.249 %）。

**F7【info · 正面】** 参考确属独立：case 内 mpmath/scipy 两条码路一致到 8.5e-16，审计者第三条实现一致到
1.07e-15，并被 7/9 与 2 两个精确极限钉住（0.777485 /
1.91582）。

**F8【info · 正面】** 8/8 单测通过（9.6 s，见 `pytest_case_helmholtz_sphere.log`），其中
`test_chief_rows_are_load_bearing_at_the_irregular_frequency` 是不变式而非灵敏度研究；报告 18331 六段齐全、
4 图内嵌、且**如实**写「not <3 %」。

**局限**：知识的检索侧当时离线（`three_checks.json`: engine_offline / Errno 111），本审计无法用知识库交叉印证，
只能靠解析解 + 自写第三实现；未做实验/文献值对照（本算例的物理可对照量就是解析解，故影响有限）。

## 6. 误差判定（显式回答 <3 %？）

| 关键量 | 参考 | got | 相对误差 | <3 % ? |
| --- | --- | --- | --- | --- |
| σ̂(ka=0.50) | 0.038744 | 0.036552 | 5.656 % | 否 |
| σ̂(ka=1.00) | 0.321629 | 0.309091 | 3.898 % | 否 |
| σ̂(ka=2.00) | 0.781500 | 0.766779 | 1.884 % | 是 |
| 阶梯最细网格 ka=1 | 同 ref | 0.309091 | 3.898 % | 否 |
| 角向 f(θ) ka=2 | 参考曲线 | 见 table_pattern | 4.915 % | 否 |
| 参考列本身（两实现） | 解析级数 | 解析级数 | 8.52e-16 | 是 |

**结论：不满足 <3 %。** 最坏 5.656 %（ka=0.5，320 面板），满足预设 6 %，但预设本身
宽于 3 %；平台 15 % 的不可接受线则远离。若要 <3 %，需生产配置（1280 面板）——`NOTES.md` 引用
`docs/acoustic3d_ices209/core_results.json` 的 0.49 %–1.57 % 支持这一点，但那是**脉动球（辐射）**算例，
与本算例（刚性球散射）边界条件不同，不能直接顶替本算例的生产档证据。

## 7. 判级与处置

| 级 | 判据（maturity_gate.SCHEME） | 本单证据 | 结论 |
| --- | --- | --- | --- |
| L1 | tests | 8 个测试名 + 审计者实跑 8 passed | PASS |
| L2 | impl_files + entry + run_artifacts | 真代码（BIE 解→远场积分→σ），审计者复跑 bit-identical | PASS |
| L3 | + integration_callers | 算例调用 main 里的 `tensorbem.acoustic3d.solve_rigid_scattering` 等公共 API；main 的 `scripts/run_acoustic3d_ices209.py`、`tests/test_acoustic3d.py` 走同一条路 | PASS |
| L4 | + benchmark + repro_cmd | 裁判行 0、生产档缺失、预设收敛判据未达、无 commit 身份 | **拒绝** |
| L5 | + external_anchor | 解析解非实验/文献；无独立实验锚 | 未主张 |

- 声明级别：**L3**；L4 四条拒绝理由见 `insert_verdict.py::veto_reasons_for_L4`。
- 处置：**request changes on delivery** —— ① `git add` 并提交全部 6 个路径（含 registry 改动）后重跑
  `--stage cheap` 使 run.log 就是产出日志；② 登记 `tensor_benchmark_cases` 行（anchor_slug、
  metric=σ̂/(πa²)、tolerance=0.06、ref_source=Mie 级数）；③ 提交 full 档工件（或把预设 1 % 阶梯判据
  改为 blocking 并给出达到它的配置）；④ `failed_checks` 应包含非 blocking 失败或改名，CASE.md 明写判据未达；
  ⑤ 推进 `tensor_case_pool` 到 passed 并绑定本 mission。
- 未跑项（unknown / not_run，不算通过）：full 档生产重算（需 1280/5120 面板，未跑）；集群侧 cj2466/cj2494
  产物取证（节点上已不存在，`runs/` 被 gitignore）；实验/文献锚（无）。
