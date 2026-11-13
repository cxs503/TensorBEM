# 交差审计 m17277 — CASE-BEM-QTF-DRIFT（二阶差频 QTF 漂移力，TensorBEM）

**审计单**: m17328（board-audit）· **被审单**: m17277（succeeded，auto-approve dev_acceptance）
**锚点**: CASE-BEM-QTF-DRIFT · **算例池**: tensor_case_pool id=176 (`BEM-QTF-DRIFT`, tol_policy `<10%`, status=dispatched)
**被审交付**: commit `3cd2c8d`（仅存在于 `agent/m17277`，未进 origin/main）· **报告**: tensor_reports id=16291
**日期**: 2026-11-13 · **审计人**: agent/m17328

## 0. 结论

**判级 L3（系统集成）**；三件套 **真存在**；**算例判据 FAIL（honest）**，非虚标。

| 验收件 | 状态 | 证据（本节数字全部程序化取自 study.json / 审计复算，零手抄） |
|---|---|---|
| run_artifacts | ✅ 真存在 | `docs/case_qtf_drift_closed_form/`：study.json (19104 B) + tables.md + report.md + 4 PNG，sha256 见 §2 |
| benchmark ref vs got | ❌ **超容差（honest FAIL）** | 最细可用网格 medium（696 面板）：非对角逐对相对误差 中位 20.62%、最大 32.07% > 10% 判据；对角 4.56% |
| repro_cmd 可执行 | ✅ 可执行（有两点瑕疵） | 审计独立重跑：coarse 级 9/9 逐对数字与交付件一致；medium 级见 §3.3（跑完后回填） |
| 测试 | ✅ | `pytest tests/test_cylinder_qtf.py -q` 在审计侧 **19 passed, 3.71 s** |
| 报告六段 | ✅ 齐全 | report.md 第 1–6 节 + 注意点/复现两节；tensor_reports id=16291 带 4 张内嵌图 |

**判级理由（L0→L5 逐级核）**
- L1（数学/单元验证）：成立且超常——闭式在 G1 对角恒等 4.25e-15、G2 Maruo 远场 1.42e-15、G4 Hermitian 2.45e-14、G3 长波律 0.118%（ka=0.05）上全过；审计用**自己的 scipy MacCamy–Fuchs 级数**独立复算，见 §4。
- L2（独立功能实现）：成立——`src/tensorbem/cylinder_qtf.py`（225 行，输入 ka → PairIntegrals/DriftQTF 输出），benchmark 入口 591 行，19 条测试；无 placeholder。
- L3（系统集成）：成立——算例入口驱动主求解器栈（`solve_diffraction_3d` → `quadratic_qtf` → `bottom_mounted_cylinder_mesh`），闭式返回主求解器同一个 `DriftQTF` 契约；**但新模块只有 harness 侧调用方，无主求解器反向调用**（见 §5 F6）。
- L4（基准验证）：**不成立**——算例判据在唯一注册口径（<10%，最细网格非对角）上 FAIL（20.62%/32.07%），且非对角误差**不随加密下降**（20.41%→20.62%）；参考侧无外部印刷表锚（WITHHELD，§5 F3）。完整算例+误差表+可复现三件在结构上齐，但"基准验证"的实质是判据通过，此处没有通过，故不上 L4。
- L5：不成立——无文献/实验锚，闭式系自洽推导。

> 与先例一致：m17242（SUBMERGED-SPHERE-HAVELOCK，生产网格 5.59% 超 5%）判 L3；m17205（判据全过）判 L4。本单按同一口径判 L3。

## 1. 基准裁判（唯一裁判=基准判据）

任务令指定裁判行 `tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-QTF-DRIFT'`：**查得 0 行**（与 verdict#490/m17237 F1 同类的系统性登记缺口）。本审计按先例**补登记裁判行**，容差口径回退到算例池行 id=176 `tol_policy='<10%'` + 被审单 goal 明文 `<10%`，登记后 status 按实测结果置 **failed**（不粉饰）。

| 项 | 内容 |
|---|---|
| metric | 非对角频率对 \|T_xx(ω_i,ω_j)\| 相对闭式的最大误差（最细可用网格 medium，696 面板，ka=0.4/0.6/0.8，头浪，kH=12） |
| tolerance | `<10%`（算例池 id=176 + 被审单 goal） |
| 实测 | 非对角 median 20.62% / max **32.07%**（对 (0,2)=(0.8,0.4) 与 (2,0)）；对角 median 4.56% / max 4.58% |
| 判定 | **FAIL（honest）** |

逐对核（medium，取自 study.json `levels[1].pairs`，零手抄）：

| (i,j) | ka_i | ka_j | BEM \|T\| [N/m²] | 闭式 \|T\| [N/m²] | rel err | <10% |
|---|---|---|---|---|---|---|
| (0,0) | 0.40 | 0.40 | 1.7535e+03 | 1.6770e+03 | 4.56% | ✅ |
| (0,1) | 0.40 | 0.60 | 3.4907e+03 | 2.8940e+03 | 20.62% | ❌ |
| (0,2) | 0.40 | 0.80 | 5.1107e+03 | 3.8696e+03 | 32.07% | ❌ |
| (1,0) | 0.60 | 0.40 | 3.4907e+03 | 2.8940e+03 | 20.62% | ❌ |
| (1,1) | 0.60 | 0.60 | 4.3310e+03 | 4.1622e+03 | 4.06% | ✅ |
| (1,2) | 0.60 | 0.80 | 5.5928e+03 | 5.1636e+03 | 8.31% | ✅ |
| (2,0) | 0.80 | 0.40 | 5.1107e+03 | 3.8696e+03 | 32.07% | ❌ |
| (2,1) | 0.80 | 0.60 | 5.5928e+03 | 5.1636e+03 | 8.31% | ✅ |
| (2,2) | 0.80 | 0.80 | 6.3901e+03 | 6.1101e+03 | 4.58% | ✅ |

收敛性（G6）：coarse 336 面板 → medium 696 面板，对角 6.59%→4.56%（降），非对角 20.41%→20.62%（**不降**），即非对角误差不被单一 h^p 律主导；`gates_result.G6_monotone_refinement.pass=false`。这与交付件 §6 的 self-report 完全一致——**报告没有粉饰，核验通过**。

## 2. 三件套取证（sha256 = 本次审计实测）

| 工件 | 字节 | sha256 |
|---|---|---|
| study.json | 19104 | 6318f5216f70adc55be61b6afa0cd74cfd879e3172421cea07d57218a2107973 |
| tables.md | 1700 | d7c636e3f4fd5945524bc84abe1ad0d477699411a702d379be8ac283ae031196 |
| report.md | 12603 | b480990d0301a46aa65ca454a82ce145eb6429fe2edaa219d56b1b0aa3c505a3 |
| qtf_bem_vs_closed_form.png | 35357 | a9fe40213078a9377e04ae0a78f7c23a754e595c17049562ce9f63175434621a |
| qtf_closed_form_curves.png | 49588 | 41ed9da924fd7069684c703628fb9a67e57d7b0d389f631bb402059ab92bee4e |
| qtf_closed_form_map.png | 41565 | feca2d0f54929bd57fb145863c6198dade2800515b39a89fa851f990259421c4 |
| qtf_mesh_convergence.png | 53628 | 92e48e7a82b6acd87b55340a4fd853d10a9e580a00bf84dfad2766ba1e17a2fc |

- 全部 8 个文件在 `3cd2c8d` 的 tree 内（`git show --stat 3cd2c8d`：12 files, +2077）。
- tables.md 与"从 study.json 重新生成"**逐字节相同**（审计脚本 `markdown_table()+pair_table()` 复算：True，1700 B）——数字表不是手写。
- git 状态：commit 仅在 `agent/m17277`（`git branch --contains 3cd2c8d` 只列该分支），`origin/main` 无该文件；按平台机制待推送，非作者缺陷（同 m17246 F3 口径）。

## 3. 复现（审计独立重跑）

审计机实测单次 336 面板圆柱绕射求解 **26.8 s**（当前负载 ~300/128），与交付件 §7"600 面板约 70 s"同量级。

### 3.1 coarse 级

`PYTHONPATH=src python3 -m tensorbem.benchmarks.cylinder_qtf_closed_form --levels coarse medium --json ...`（审计侧输出到 /tmp，不动被审工作树）：
coarse 级 9/9 逐对数字与交付 study.json 一致（bem_abs 1.7876e+03 / 3.4847e+03 / 4.9100e+03 / 4.0975e+03 / 5.1143e+03 / 5.6418e+03 等）。

### 3.2 审计侧复现命令（可执行性）

文档 §8 的命令可直接执行成功（本条即实跑命令，仅改了输出路径）：
```
PYTHONPATH=src python3 -m tensorbem.benchmarks.cylinder_qtf_closed_form \
    --json <out>/study.json --figures <out> --levels coarse medium --tables <out>/tables.md
```

### 3.3 medium 级与整体判定

（见 §3.4 回填段）审计重跑完整两级的耗时由 6 次真实求解主导（≈27–80 s/次），与本机 120 s shell 上限冲突——交付件自己在 §7 记了这条生态摩擦，并用 `scripts/warm_qtf_case_cache.py` 预热缓存规避。

### 3.4 回填：重跑结果（程序化比对，`rerun_compare.json`）

审计重跑完整跑完（含 depth sweep）并落盘 `/tmp/audit_m17328/study_rerun.json`（18909 B，5 图）：coarse 与 medium 两级
**18/18 逐对 bem_abs 与交付件一致**（max rel delta 见 rerun_compare.json，medium 9 对 = 1753.489/3490.704/5110.736/
3490.704/4330.962/5592.78/5110.736/5592.78/6390.102，与交付 study.json 相同）。故"可复现"成立（数值级），
F4 的瑕疵只是工件溯源（缓存/超集），不影响判据结论。重跑额外得到交付件§5 声称未跑的吃水扫描（见新版 F8）。

## 4. 独立复核（审计自己的 MacCamy–Fuchs 级数，不 import `cylinder_drift`）

审计脚本：`docs/board_audit/m17328/audit_independent_qtf.py`（scipy `jv/jvp/hankel1/h1vp` 重建 MF 模态，自建 4096 点方位求积）。要点（全部为审计侧数值）：

| 检查 | 结果 | 判读 |
|---|---|---|
| 圆柱壁无通量 BC `∂ψ/∂r`（相对） | ka=0.2/0.4/0.6/0.8/1.0/1.4 全部 ≤1.1e-16 | 一阶场真实满足 MF 模态 |
| 审计场 vs 交付模块 `cylinder_wall_field` | max|Δψ| ≤7.1e-16，max|Δ∂θψ| ≤9.2e-16 | 交付模块一阶层是同一真实现，非桩 |
| 方位积分 I1/I2（(0.4,0.4)…(0.8,0.8)） | rel diff ≤3.7e-16 | 闭式求积精确（有限 Fourier 级数） |
| 对角 vs 审计自建近场精确解 | rel diff ≤1.7e-15（ka=0.2…1.4） | 对角归一化（1/4、1/(k_i+k_j)、水线项符号）钉死 |
| 对角 vs 审计自建简化 Maruo 远场 | rel diff ≤1.8e-15 | **另一条独立公式路线**同时钉死 |
| 长波渐近系数 | ka=0.002: 3.084251 vs 5π²/16=3.084251（6.7e-7）；ka=0.005: 4.2e-7 | 经典立方律确认，非拟合 |
| 非对角可用的独立锚 | **无**：近场与 Maruo 两条路线都只产对角；只能查 Hermitian（≤8.9e-16）与连续性 | 见 §5 F2（诚实缺口，非作者隐瞒） |

结论：**参考侧（对角+归一化+长波律）是真机器精度构造**，审计独立实现复现；**非对角绝对值无外部/独立锚**，故"缺口在 BEM 侧"是推断而非证明（F2）。

## 5. 发现（按严重度）

**F1 — high（判据）：算例在唯一注册口径上 FAIL，且非对角误差不随加密下降。**
medium（696 面板）非对角 median 20.62% / max 32.07% > 10%；coarse→medium 不降（20.41%→20.62%）而同时对角降（6.59%→4.56%）。交付件已按 honest-FAIL 记 `fail_stage=G5`、`failure_reason`、`next_round`，报告标题即写"非对角误差中位 20.6%，10% 判据未通过"。审计复核**自报与实测一致，无粉饰**。裁判行登记为 failed。

**F2 — medium（归因）：`failure_reason` 说"缺口在 BEM 侧，不在参考侧"，这是推断，不是证明。**
统计上：对角（有两条独立精确锚）随加密下降，非对角（无锚）不降——这支持"面板离散误差主导"，但不排除闭式非对角项本身有偏差。审计侧只能给 Hermitian（8.9e-16）与 ka 连续性，**无法独立验证非对角绝对值**。建议第二轮引入第三方 Rankine 面板码（capytaine，生态内已有先例 m17303 用于 Havelock 椭球）做非对角三方对照。范围声明同侧同缺项：两侧都只含二次压力项（不含二阶势项），故对比同范围成立；绝对 QTF 仍是下界（交付件 §7.1 已声明，审计确认声明准确）。

**F3 — medium（验收口径）：被审单 goal 要求"ref 原文/手册查得（零手抄，报告引用原表原页）"，交付的 ref 是自算闭式，无印刷表锚。**
报告 §1 明写"没有从文献原表抄任何数字……因此 ref 值是闭式在机器精度下算出来的"，并给出三条独立锚。这是**如实声明的方法偏差**，不是造假；但"外部锚"这一半要求未达成 → L5 不成立、L4 的外部可信度打折。本审计在 verdict 的 `external_anchor` 键按实登记：仅经典长波律 5π²/16（Maruo/Vugts 渐近）可用，印刷表数值 **WITHHELD**。

**F4 — low（可复现性/工件溯源）：文档 §8 的 repro_cmd 不能逐字节重交所交付的 study.json。**
（a）命令未带 `--cache`：交付件 `levels[*].solve_seconds_total` 分别 0.235 s / 0.603 s，实为**缓存重载时间**（`/tmp/qcache` 7 个 npz 时间戳与 study.json 生成时间吻合），审计实测真实求解 26.8 s/次——"求解耗时"字段名与语义不符，且工件未记录"由缓存产出"；（b）命令未带 `--no-depth-sweep`：交付 study.json 无 `depth_sensitivity` 键（§5 也写"本次未跑吃水扫描"），但默认命令会跑该段，重跑结果因此是交付件的**超集**，不是同一工件。数值层面（§3.1 已证 coarse、§3.4 见 medium）逐对一致，故不影响判据结论，属溯源瑕疵。

**F5 — low（工件溯源）：study.json 无 `git_head`/BLAS/线程数/平台字段**，只有 `generated_at` 类隐含信息；平台侧复算者无法把工件钉到具体 commit/环境（对照 m17246 的 numbers.json 有 git_head 的先例）。

**F6 — info（集成方向）：新模块 `tensorbem/cylinder_qtf.py` 只有 harness 侧调用方（benchmark 模块 + 测试），无主求解器反向调用；且未进 origin/main。** 本审计按 m17205 同类口径（"模块调用主求解器层"即算集成）判 L3；若平台在推送前跑 grep，`integration_callers` 会命中失败而降 L2（同 verdict 503 的现象），属进程性降级、非实质缺陷。

**F7 — info（既有接口不一致，未修）：** BEM 侧 `DriftQTF.matrix` 是 `(n,n,n_components)`，闭式侧是 `(n,n)` 标量；交付件用 `_component()` 在算例内吸收（§7.5 声明）。这是公共类契约不一致，作者选择零回归不改签名——合理，登记为待修债务。

**F8 — medium（敏感性·审计重跑实测，修正原判读）：交付件未跑的吃水扫描，审计重跑补出，且结果与交付件的 e^{-kH} 论证不一致。**
coarse/ka=0.6 实测：kH=6 → 12.20%，kH=12 → **12.07%**，kH=24 → 6.44%（对闭式对角 4162.16 N/m²）。但同一网格级在本算例实际吃水
（depth=30 m，ka=0.6 时 kH=18）下对角误差只有 1.55%——三点**非单调**，且 kH=24（更长柱、n_z 仍 10）反而比 kH=18 差，说明该
扫描把**底盖/吃水截断**与**固定 n_z=10 的 z 向离散**混在一起（柱越长每段越粗），因此它既不能证实也不能否证 e^{-kH}=6.1e-6 的解析界。
判读修正：交付件"未跑吃水扫描、改引解析界"的选择本身可辩护（解析界对*物理*截断成立），但 §5 若被引用为"已量化截断影响"会失真；
审计原稿"接受该解析界"改为"解析界只在物理层面成立，本扫描实测被离散混淆，须以匹配面板尺度的吃水扫描复验"。

## 6. 未做的检查（如实）

- 第三方面板码（capytaine）非对角 QTF 三方对照 —— 需另立节点（生态内 3D QTF 对比面尚无先例外推）。
- 印刷表/手册（如 Newman、Molins）原页数值锚 —— 本执行环境无外网，WITHHELD。
- fine 级（34×20）网格 —— 单次 ≈ 18 s/求解 × 3 且本机 shell 上限，未跑；交付件亦按预注册停止规则停在 medium。
- 吃水扫描（6/12/24）—— 未跑（交付件同样未跑，见 F8）。
- PNG 像素级内容 —— 仅核尺寸/sha256/生成脚本，未逐像素复核。

## 7. 处置（disposition）

**approve with notes**：交付实质为 L3；算例判据 FAIL 已如实入账（裁判行 status=failed、verdict evidence 记 FAIL），不视为造假；F4/F5 为可复现性瑕疵，建议（不影响本轮结论）后续工件带 `--cache`/`--no-depth-sweep` 的显式注明与 git_head 字段。返修方向见算例 `next_round`（n_theta≥32 免混叠、I1/I2 误差分解、频差更小频率对、capytaine 三方对照）。