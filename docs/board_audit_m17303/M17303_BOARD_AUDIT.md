# m17242 交差核验：CASE-BEM-SUBMERGED-SPHERE-HAVELOCK（TensorBEM）

> 交差审计（非重做）。被审交付：`1ae83e19337e` @ 分支 `agent/m17242`（未并入 origin/main）
> 审计工作树 `m17303`；本页与 `artifacts/audit_m17303/*.json` 全部程序化生成（`scripts/audit/m17303_board_audit.py`）。

**裁定：L3（verdict_by=board-audit）**

## 1. 三件套取证

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts | 存在；主工件**未入库**（`runs/` 被 `.gitignore` 覆盖） | `runs/m17242/case_havelock_submerged_sphere.json`（30 KB）+ 4 张 png 已入库；sha256 见 `artifacts/audit_m17303/branch_1ae83e1/MANIFEST.json` |
| benchmark ref vs got | 53 条逐项判定，逐条可由工件复算 | `judgement.all_items` n_items=53 / n_failures=19；生产网格 28 项 / 违例 7 项 |
| repro_cmd | 可执行且**实跑复现** | §4：`PYTHONPATH=src python3 -m tensorbem.benchmarks.case_havelock_submerged_sphere --output runs/audit/case.json` |

交付 commit 内文件 12 个；白名单核对的交付件 11 个全部存在、缺失 0 个。

## 2. 基准裁判（唯一裁判）

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug = 'CASE-BEM-SUBMERGED-SPHERE-HAVELOCK';
-- rows = 0
```

**登记裁判不存在**：`tensor_benchmark_cases` 全表 46 行，本锚 **0 行**；TensorBEM 下已登记的球体类锚只有 CASE-BEM-SPHERE-ADDED-MASS、CASE-BEM-CATHODIC-SPHERE、CASE-BEM-ICE-SPHERE-HEAVE。所以 `<5%` 判据只写在 mission 17242 正文里，不是机器可查的登记行。

裁判栏位对账（任务单 vs 实交）：

- 任务单要求 ref = 「Havelock 1955 频域多极解析解——目标数值从 ref 原文/手册查得（零手抄，报告引用原表原页）」
- 实交 ref = 仓库内闭式多极子 `tensorbem.analytic.submerged_sphere_wave_multipole` **求值**（阶乘 + 指数积分 + 小线性系统，无求积、不碰面元核）
- 报告 ⑥.4 自曝：Havelock 1955 原表是**半潜浮球**垂荡，全潜球多极子系数出自 Thorne 1953 / Wang 1986 / Linton 1991；环境外网受限，**未使用任何文献表格数字**
- 裁定：**ref 出处与任务单标注不符**（诚实披露，未粉饰）；「零手抄」以「零抄表」达成，「引用原表原页」未达成（环境事实，不是造假）

## 3. 判据复算（不信报告，信 got/ref）

从工件里每一对 got/ref 独立重算相对误差，再按预置 5% 重判：

| 范围 | 项数 | 违例 | 判定 | 最差项 |
|---|---:|---:|---|---|
| `production_mesh` | 28 | 7 | **FAIL** | sweep/ka=0.5/b33 5.59% |
| `all_items` | 53 | 19 | **FAIL** | refinement/sub1/b11 19.69% |

重算 53 项 / 违例 19 项；生产网格重算违例 7 项，与模块自报违例清单逐项一致：**是**。

生产网格（320 面元、f/a=2）违例清单 —— 全部是**阻尼**：

| 项 | 量 | got | ref | 相对误差 | 判 |
|---|---|---:|---:|---:|:--:|
| sweep/ka=0.1 | b33 | 0.003138 | 0.003322 | **5.55%** | ✗ |
| sweep/ka=0.1 | b11 | 0.001531 | 0.001620 | **5.49%** | ✗ |
| sweep/ka=0.5 | b33 | 0.079970 | 0.084706 | **5.59%** | ✗ |
| sweep/ka=0.5 | b11 | 0.038918 | 0.041192 | **5.52%** | ✗ |
| sweep/ka=1 | b33 | 0.076360 | 0.080582 | **5.24%** | ✗ |
| sweep/ka=1 | b11 | 0.039511 | 0.041738 | **5.34%** | ✗ |
| sweep/ka=2 | b11 | 0.005495 | 0.005787 | **5.05%** | ✗ |

最细网格（1280 面元）最差相对误差 **1.39%**（<5% 通过，亦 <3%）；收敛阶 2.03（理论二阶）；内接多面体面积亏损 99.52%（sub3），误差/亏损比 a33 = 0.964、b33 = 2.826（三级近恒定 → 系统偏差，非随机噪声）。

## 4. 复现（repro_cmd 实跑）

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_havelock_submerged_sphere --output runs/audit/case.json
```

在交付 commit 的 `src/tests/tools` 副本（`/tmp/a17303`）里独立重跑，与交付工件逐叶对比（排除 `environment`）：共同叶 743，差异叶 0，最大绝对差 **0**，`judgement` 块逐字节一致。

即：报告里每个数字都能由交付代码重算出来 —— **可复现性成立**（本单最硬的一块）。

## 5. 独立性（got 与 ref 是否同源）

| 检查 | 结果 |
|---|---|
| `got_path_calls_assemble` | 是 |
| `got_path_calls_solve_radiation` | 是 |
| `got_path_imports_solver` | 是 |
| `ref_path_imports_multipole` | 是 |
| `ref_uses_quadrature` | 否 |
| `ref_uses_factorials_and_Ei` | 是 |
| `ref_uses_linear_solve` | 是 |
| `analytic_imports_bem_kernel` | 否 |

`got` 走三维自由液面面元求解器（Green 函数 + 常值面元）；`ref` 走多极子闭式解（阶乘 + Ei + 小线性系统，无求积、不 import 面元核）—— 两条路径无共享代码，方法学独立性成立；ref 自身另有三重检查（截断 20→40 变化 0、Haskind 残差 8.8e-16、深潜回到 Lamb 0.5 且超额衰减阶 2.97）。

## 6. 外部第三轴（capytaine，独立 Rankine 面元码）

额外引入**不属于 TensorBEM** 的第三方码：capytaine 3.1.dev0（Nemoh 后继），对同一几何（a=1 m、球心 z=−2 m、深水、ka=1、ρ=1025）独立求解，网格阶梯 128/288/512/1152/2048/4608 面元（`scripts/audit/m17303_capytaine_independent.py`）：

| 量 | ref（多极子） | got（TensorBEM 320） | capytaine 最细 | ref vs capy 最细 | ref vs capy 1/N 外推 |
|---|---:|---:|---:|---:|---:|
| a33 | 0.448073 | 0.440890 | 0.454379 | 1.41% | 2.19% |
| b33 | 0.080582 | 0.076360 | 0.081969 | 1.72% | 2.87% |
| a11 | 0.475009 | 0.466268 | 0.483570 | 1.80% | 2.69% |
| b11 | 0.041738 | 0.039511 | 0.042659 | 2.21% | 3.54% |

读法：capytaine 自身收敛慢（低阶 Rankine + 全潜球，±1.5% 漂移未完），最细 4608 面元与 ref 最大差 **2.21%** —— 外部码独立支持 ref 在 2% 量级可信；同时 TensorBEM 320 面元与 capytaine 最细最大差 7.97%，方向（TensorBEM 偏低）与面积亏损一致，证实 5.2–5.6% 阻尼偏差是**离散偏差**而非 ref 错。1/N 线性外推给出 3.54%，比最细网格差更大，说明 capytaine 不是干净的 1/N 收敛，**外推值不作证据**，只用最细网格直接对比。

局限（如实记）：capytaine 的网格阶梯 128→4608 面元仍在漂移（heave A′ 0.4650→0.4544，surge A′ 0.5100→0.4836），所以这条第三轴是「同量级互证」而非「高精度裁判」；它足以否证「ref 与 got 共同的系统性错误」，但不足以把 ref 精度钉到 0.5% 以内。此外 capytaine 与 ref 的 1.4–2.2%% 差额本身也是第三方低阶面元的离散误差。

## 7. 报告六段与数字注入核对

| 段 | 存在 |
|---|---|
| 1_problem_and_ref | ✓ |
| 2_method | ✓ |
| 3_parameters | ✓ |
| 4_comparison | ✓ |
| 5_convergence | ✓ |
| 6_failures | ✓ |

| 头号数字 | 报告出现 | 工件复算 |
|---|---|---|
| `n_failures_7_of_28` | ✓ | `7 / 28` |
| `worst_damping_5.59pct` | ✓ | `5.59%` |
| `finest_mesh_1.39pct` | ✓ | `19.69%` |
| `convergence_order_2.03` | ✓ | `2.03` |
| `deficit_ratio_a33_0.964` | ✓ | `0.964` |
| `deficit_ratio_b33_2.826` | ✓ | `2.826` |
| `haskind_8.803e-16` | ✓ | `8.803e-16` |
| `low_freq_1.60pct` | ✓ | `1.60%` |

六段齐全 **是**；头号数字可由工件复算 **是**；报告显式 honest FAIL **是**；自曝 ref 出处不符 **是**。

## 8. 测试

```
PYTHONPATH=src python3 -m pytest tests/test_case_havelock_submerged_sphere.py -q
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
7 passed, 2 warnings in 42.84s
```

7 个用例覆盖：多极子截断无关性、Haskind 恒等式、深潜 Lamb 极限与衰减阶、低频阻尼趋零、判据函数、端到端 smoke（断言 320 面元、系数落在物理区间）、面积亏损比值。**没有**断言生产网格「通过」判据 —— 本算例诚实 FAIL，断言通过等于造假，取舍正确；代价是日后若有人把 `production_mesh` 悄悄改成 1280 面元刷绿，测试不会拦。建议补一条 pin 住「320 面元阻尼误差 5.2–5.6%」的回归断言。

## 9. 判级（L0–L5）与理由

- L2 过：got 由生产三维自由液面面元求解器现算，ref 由闭式多极子现算，两条路径无共享代码；7/7 交付测试通过
- L3 过：算例模块 tensorbem.benchmarks.case_havelock_submerged_sphere 直接调用主求解器 API（free_surface_solver3d），测试随仓
- L4 未达（裁判缺席）：tensor_benchmark_cases 对该锚 0 行，5% 判据只写在 mission 17242 正文，不是登记裁判
- L4 未达（诚实 FAIL）：生产网格 320 面元 7/28 项超 5%，最差 5.59%（sweep/ka=0.5/b33）；仅 1280 面元全项 1.39% 通过
- 可复现性成立：重跑逐叶复现判定块（最大叶 |diff| = 0）

| 级别 | 门槛 | 本单 |
|---|---|---|
| L0 | 文档成果 | 过 |
| L1 | 数学/单元验证 | 过 |
| L2 | 独立功能实现（输入→计算→输出） | 过（真面元求解 + 真闭式 ref） |
| L3 | 系统集成（接入主求解器/平台） | **过**（直调生产求解器 API + 测试入仓） |
| L4 | 基准验证（完整算例 + 误差分析 + 可复现） | **不过**：无登记裁判行；生产网格 7/28 项超 5% |
| L5 | 科研工程成果 | 不过 |

**结论：L3。** 不是「重做不合格」，而是「交差口径没到顶」：功能、误差分析、复现、独立性都成立，卡在（a）算例未登记进 `tensor_benchmark_cases`（无机器裁判）、（b）预置 5% 判据在生产网格（320 面元）上诚实 FAIL（7 项阻尼 5.05–5.59%）。返修路径：登记裁判行（tolerance 取生产网格可达值，或把生产网格改为 1280 面元）+ 补回归断言 pin 住当前失败图案，即可重报 L4。

## 10. 发现清单（按严重度）

| 级别 | 发现 | 证据 |
|---|---|---|
| blocking(L4) | 无登记裁判行：`tensor_benchmark_cases` 0 行 | §2 |
| blocking(L4) | 预置 5% 判据在生产网格诚实 FAIL（7/28，最差 5.59%） | §3 |
| material | ref 出处与任务单标注不符（Havelock 1955 原表=半潜浮球；实交内建多极子） | §2 对账 |
| material | run_artifacts 未入库（`runs/` gitignore），工件只能靠重跑重建 | §1 |
| minor | 无回归断言 pin 住生产网格失败图案（改判据不会被测试拦住） | §8 |
| minor | `ka≥4` 时 `B'~1e-6`，相对判据接近离散噪声底；报告另列 scale-aware 量且不改判定 | §3 |

## 11. 未做 / 未知（不冒充通过）

- 未取得 Havelock 1955 原表页（外网受限），故**无法**核对「原表数值」这一路；本审计用 capytaine 外部码替代第三轴，属替代证据，不等于文献原表核对。
- 1280 面元的 dashboard 级复跑只在审计重跑（同一 commit）中出现一次，未做跨机器/跨数量级复算。
- 未评审 `tools/case_havelock_submerged_sphere_report.py` 的注入正确性逐行（只核了头号数字 vs 工件）。

*工件：`artifacts/audit_m17303/{referee,report_check,repro,independence,external_anchor,tests,finding}.json`*
