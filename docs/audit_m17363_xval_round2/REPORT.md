# m17363 交差审计 — XVAL-BEM-ROUND2-METHOD-SELECTION（TensorBEM）

审计单 m17442 · 被审单 m17363（status=succeeded）· 审计对象 commit `8bf9345`
（"m17363: XVAL round-2 method selection for the exterior-Laplace CP BEM"）
· 审计员席位 `board-audit` · 裁决级别 **L3**（L4 明确否决，理由见 §5/§6）

本单是**交差核验**，不含任何重做：所有断言都指向被审单的真实文件与其自身产出的数字。
唯一新增物 = 本审计的可复算脚本 `scripts/m17442_audit_m17363.py` + `artifacts/audit_m17363_xval_round2/`。

---

## 0. 一句话结论

m17363 的**实质成果为真**：一个新的 Laplace 外部 BIE 交叉对标模块（607 行真实代码）+ 一个
在产线上真正生效的**双层位符号缺陷修复**（`evaluate_exterior_potential`），其运行工件
`results.json` 我在干净快照上**逐位复现（sha256 完全一致）**，解析电流误差 0.332% < 3%。
但**证据层不达标**：anchor 在 `tensor_benchmark_cases` 里 **0 行登记**、随提交的
`tests/test_laplace_sphere_xval.py` **4/9 红**、文档里的 `repro_cmd` 与产出工件的配置不一致。
故判 **L3（真实现在主方程 + 有真实调用方）**，**L4 不成立**。

---

## 1. 取证：三件套逐项核验

| 件 | 指针（真实路径） | 状态 |
|---|---|---|
| impl_files | `src/tensorbem/benchmarks/laplace_sphere_xval.py`（607 行，sha256 `9e2d3f96…`）<br>`src/tensorbem/cathodic_protection.py`（缺陷修复，sha256 `9a1a31c6…`） | ✅ 真代码，无 `NotImplementedError`/`pass` 占位 |
| entry | `scripts/m17363_xval_run.py`（525 行，`run` / `report` 两个子命令，sha256 `c214f54f…`） | ✅ 可执行 |
| run_artifacts | `artifacts/xval_laplace_sphere/results.json`（13978 B，sha256 `1efd46de…`）<br>`docs/xval_laplace_sphere/REPORT.md` + `xval_convergence.png` | ✅ 存在且非空；⚠️ `artifacts/` 在 `.gitignore` 内 → 未入库（见 F7） |
| tests | `tests/test_laplace_sphere_xval.py`（9 test，sha256 `67108b63…`） | ❌ **4 failed / 5 passed**（见 F2） |
| benchmark | 解析 Legendre 级数 + capytaine 3.1.dev0（Rankine 极限影响矩阵）+ scikit-fem 12.0.2（轴对称 P2 Galerkin） | ⚠️ 数值对照齐备，但 `tensor_benchmark_cases` **无该 anchor 行**（见 F1） |
| repro_cmd | `XVAL_MIN=1 PYTHONPATH=src python3 scripts/m17363_xval_run.py run` | ✅ 我实跑复现，**sha256 与产出完全一致**；⚠️ 文档写的是去掉 `XVAL_MIN=1` 的命令（见 F3） |

### 1.1 复现（独立、干净快照）

```
git -C work/m17363 archive 8bf9345 | tar -x -C /tmp/xv          # 纯 commit 快照，无工作区残留
cd /tmp/xv && XVAL_MIN=1 PYTHONPATH=src python3 scripts/m17363_xval_run.py run
sha256sum artifacts/xval_laplace_sphere/results.json
  1efd46debaecab4614b309d00b81b6a913936fad117cbc7f5d7ae1448cb1e241   ← 与 m17363 工件同哈希
```

逐字段数值比对（`artifacts/audit_m17363_xval_round2/audit_evidence.json`）：
`max_rel_diff = 0.0`，`n_fields_differing = 0`，键集合完全一致 → **确定性复现，无挑选**。

### 1.2 基准裁判（`tensor_benchmark_cases`）

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-BEM-ROUND2-METHOD-SELECTION';
-- (0 rows)
```

该 anchor **未登记任何基准用例**。因此本次「基准」只能以被审单自报的解析解 + 两个 OSS 实现为准，
属**自声明基准**，不构成 L4 所需的机器可裁基准。我按其自报判据逐个复核（见 §2），数字本身站得住。

---

## 2. 关键数值复核（全部来自 `results.json`，非手抄）

| 量 | 参考值 | got | 相对误差 | <3%? | <15%? |
|---|---|---|---|---|---|
| 总电流 I（TensorBEM，N=1280） | 4πσaφ_a = **-43.98230 A** | **-43.83637 A** | **0.332%** | ✅ | ✅ |
| 总电流 I（capytaine 矩阵驱动同一 BIE，N=1280） | 同上 | -43.8105 A | 0.391% | ✅ | ✅ |
| 总电流 I（scikit-fem 轴对称 P2，R/a=4，48×32） | 同上 | -44.01834 A | 0.072% | ✅ | ✅ |
| 单层矩阵 A vs capytaine（同网格，Frobenius） | capytaine | — | 1.84e-4（N=1280） | ✅ | ✅ |
| 双层矩阵 D vs capytaine（Frobenius） | capytaine | — | **2.15e-3**（N=1280；N=320 时 9.15e-4） | ⚠️ 退化 | ✅ |
| 面密度 q 的 L∞ | 解析 φ_a(1+2μ)/a | — | 1.082%（N=1280） | ✅ | ✅ |
| 外部电位 φ 的 L2 | 解析级数 | — | 0.540%（N=1280） | ✅ | ✅ |

收敛级（TensorBEM）：4.916% (N=80) → 1.301% (N=320) → 0.332% (N=1280)；
工件自报观测阶：电流 **0.96 / 0.99**，密度 L∞ 0.44 / 0.42，外部电位 L2 0.95 / 0.98 → **一阶，非二阶**（见 F5）。

### 2.1 独立验证符号修复（不看被审单的测试）

把**解析** Dirichlet 数据与**解析**密度注入表示式，两种符号各算一遍（1280 面板，9 个探针，
`audit_evidence.json / independent_sign_check`）：

| 表示式 | 对解析外部电位的最大相对误差 |
|---|---|
| `∫Gq − ∫φ ∂G/∂n`（修复前） | **26.23%** |
| `∫Gq + ∫φ ∂G/∂n`（修复后，commit 8bf9345） | **0.675%** |

→ 符号修复是**真实且实质**的：修复前该算例外部电位错 26%，修复后落到球面分面几何误差量级。
且 `tests/test_cathodic_protection.py` 在修复后 **30 passed**（日志 `pytest_cp_regression.log`，
159.57 s）→ 无回归。

### 2.2 Cb/L/B 几何前置

本算例是**解析球面**（单位球 icosphere，无船体型线），不适用 Cb/L/B 校验；
几何自校验由"分面球顶点严格落在 r=a、面元质心 r/a∈[0.982, 1]"承担（`separate_test_bugs.test_exact_series`）。

---

## 3. 六段报告齐全性

`docs/xval_laplace_sphere/REPORT.md`（235 行）与平台报告 id=16374（13673 字符）：
案例介绍 ✅（几何/参数表/无量纲控制量）· 方法 ✅（三路线 + 失败模式）· 结果数字表 ✅（§3.1-3.7）
· 图 ⚠️（`xval_convergence.png` 有；§6 声称的 `xval_potential_profile.png` **不存在**，F6）
· 注意点 ✅（§5 caveats）· 误差判定 ✅§3.1（<15%/<3% 逐项显式回答）。
**六段齐全**，但两处内部数字断言与自身工件矛盾（F4/F5）。

---

## 4. 判级（L0–L5）

| 级 | 判据 | 结论 |
|---|---|---|
| L0 文档 | — | 超越 |
| L1 数学/单元验证 | 解析级数 + 单元测试 | ✅ 达成（但有 4 个红测试，见 F2） |
| L2 独立功能实现 | 真代码 输入→计算→输出 + 运行工件 | ✅ **达成**（607 行模块 + 驱动器 + 与 1280 面板 BEM 一致的工件） |
| L3 系统集成 | L2 + 主求解器/平台侧真实调用方 | ✅ **达成**：修复落在产线模块 `cathodic_protection.py` 的 `evaluate_exterior_potential`，模块内 4 处调用（`:692` 调和性检查、`:710` 六点 stencil、`:891`、`:1250` 远场 Gauss 电流路线），并被 `tensorbem/verification/corrosion/cathodic_protection/__init__.py` 公开导出、被 `scripts/cathodic_sphere_dnv_b401_study.py` 使用；30 个既有 CP 测试全绿 |
| L4 基准验证 | L3 + 机器可裁基准 + 完整算例 + 可复现 | ❌ **否决**：①anchor 在 `tensor_benchmark_cases` 0 行；②随提交测试 4/9 红；③文档 `repro_cmd` 缺 `XVAL_MIN=1`（默认配置产出不同数字，且默认档从未跑完）；④"Richardson 0.000%" 是代数恒等式而非独立证据；⑤完整档（subdivisions 1..4 / Duffy 4,6,8,12 / R/a 2,4,8 / 96×64）未运行 |
| L5 科研工程 | L4 + 外部锚 | ❌ 不成立（L4 已否） |

**裁定：L3。** 增量价值为真（生产代码缺陷修复 + 可复现的 XVAL 能力），但不足以称"基准验证完成"。

---

## 5. 发现（按严重度；每条带指针与复现）

**F1 · 阻塞 L4 · 基准未登记**
`SELECT … WHERE anchor_slug='XVAL-BEM-ROUND2-METHOD-SELECTION'` → 0 行。anchor_slug 只写在 `results.json`
（`mission=17363`）里，机器无从裁定。补法：把 §2 三行登记进 `tensor_benchmark_cases`（ref_source=解析级数 /
capytaine / scikit-fem，metric=tot_current_rel_error，tolerance=3%）。

**F2 · 高 · 随提交测试红（4/9）**
`PYTHONPATH=src pytest tests/test_laplace_sphere_xval.py -q`（commit 8bf9345 原样）→
`4 failed, 5 passed in 52.00s`，日志 `pytest_xval_commit8bf9345.log`：
* `test_capytaine_assembly_agreement` → `AttributeError: module 'tensorbem.benchmarks.laplace_sphere_xval'
  has no attribute 'solver3d'` —— 该测试**从未跑通过**；
* `test_exact_series_reproduces_boundary_data` → 用 r<a 的面元质心处解析值去比 r=a 的边界数据
  （`centroid_r_min=0.9822`，最大偏差 0.0687）；
* `test_exterior_potential_dipole_sign` → 参考值扣单极项扣了 `1.0` 而非 `a/r`，导致
  "误差 100%"；改用正确参考后误差 4.38%（`separate_test_bugs` 已复算）→ **测试侧约定错，非内核错**；
* `test_density_and_potential_converge` → 阈值 1% 卡在 1.082%（密度 L∞），量级合理但阈值与工件不符。
> 结论：内核与修复可用，**测试是欠修的**（mission 自述"tests 写好但未跑 pytest"已构成证据真空）。

**F3 · 高 · repro_cmd 与产出工件配置不一致**
工件由 `XVAL_MIN=1`（subdivisions=(1,2,3)、Duffy=(4,8)、FEM R/a=(2,4)、网格 (24,16)/(48,32)）生成；
而 `REPORT.md:234` 与 `scripts/m17363_xval_run.py:465` 给的是**不带** `XVAL_MIN=1` 的默认命令
（subdivisions 1..4、Duffy 4/6/8/12、R/a 2/4/8、96×64，见驱动 `cmd_run()` 三分支）。
按文档命令跑出来的是另一组数字。**补法**：文档改成 `XVAL_MIN=1 … run`（或把工件换成默认档）。

**F4 · 中 · Richardson 0.000% 是恒等式，不是证据**
`route_skfem` 用**同一对误差**估 p 再拿 p 去外推：`I_ext = I_f + (I_f − I_c)/(2^p − 1)`，
`p = log2(e_c/e_f)` → 代数上 `I_ext ≡ I_exact`（故两条记录都给出 `richardson_rel_error = 0.0`，
且 p 分别为 1.92 / 1.75 而非 2）。§3.4"R/a 无关到机器精度"因此是同一恒等式的回声。
**补法**：用理论阶 p=2（或三档网格估计）外推，报非零误差。

**F5 · 中 · "clean second-order trend" 与自身工件矛盾**
`REPORT.md §3.5` 称修复后外部电位"0.540% with a clean second-order trend"；`results.json`
`tensorbem_observed_order.probe_l2 = [0.9456, 0.9833]`、`current = [0.9590, 0.9855]`、
`density_linf = [0.4397, 0.4221]` → **一阶**（密度更低）。属实质性过度表述。

**F6 · 低 · 文档承诺的图不存在 / 表头误标**
§6 列 `xval_potential_profile.png`，`docs/xval_laplace_sphere/` 只有 `xval_convergence.png`；
§3.6 表头 "two finest levels" 但 `sensitivity_subdivisions=(2,)` 只有一档（N=320）。

**F7 · 低 · 工件未入库 + 平台报告摘要夸大**
`artifacts/` 在 `.gitignore` 内，`results.json` 只活在 worktree（本审计已登记其 sha256 并另存副本
`results_mission.json` / `results_repro.json`）。平台报告 id=16374 的 summary 称与 capytaine
"element-wise to ~1e-4"，而其自身 §3.3 表中 1280 面板双层矩阵为 **2.15e-3**（比 N=320 的 9.15e-4 **退化**）；
"~1e-4" 只对单层成立。

**非发现（澄清，避免误伤）**：capytaine / scikit-fem 路线**不是 mock** —— 版本从包内读出
（3.1.dev0 / 12.0.2），capytaine 用 `DefaultMatrixEngine.build_matrices(free_surface=inf, wavenumber=0)`
真装影响矩阵并驱动同一 BIE，缺包时按 `available:false` 降级而非抛异常。

---

## 6. 未执行的检查（如实申报）

* 未跑默认档（subdivisions 1..4 + 96×64）：时间预算外，且与工件配置无关；
* 未复核 capytaine 内部 Delhommeau 实现的正确性（超出本单范围：只核"能否被独立实现复现"）；
* 未跑全仓 pytest 回归（只跑 `tests/test_cathodic_protection.py` 与 XVAL 测试文件）；
* 未评估该修复对**船体尺度** CP 算例的数值影响（本算例是解析球，非船体）。

## 7. 处置（disposition）

**裁决 L3（confirmed），证据层"要求修改"**：F1/F2/F3 属**证据链缺陷**（不推翻功能实质，但使 L4 不成立），
应在返修轮内修掉：登记基准用例、修复 4 个红测试、校正 `repro_cmd`；F4/F5 属表述纠偏（改报告文本或改算法）。
本审计不修改被审单的交付代码，只在本文与 `artifacts/` 留下可复算证据。