# TensorBEM 质量审计报告（AUDIT-ASSET-12REPO · m14488 · 2026-11-11）

**对象**：`cxs503/TensorBEM`，分支 `agent/m14488`，审计 HEAD = `481be53`（Merge #625），
窗口基线 = `b17b169`（Merge #599）之后的 **26 个已并 agent PR（#600–#625）**。
**审计席位**：audit2-ab（独立复核；按 adversarial-review / codebase-investigation 两份技能流程执行）。
**唯一核心标准**：可验证的功能交付（不是任务数、PR 数、行数）。

**复现工件（全部在 worktree 内）**

| 工件 | 内容 |
|---|---|
| `docs/audit_m14488/inventory_scan.py` | 资产盘点 + 接线（AST import）+ 占位符扫描，只读，可复跑 |
| `docs/audit_m14488/inventory.json` | 26 个 PR 的逐文件分类/行数 + 154 个模块的接线层 + stub 扫描结果 |
| `docs/audit_m14488/probes.py` | 三个独立数值复核探针（P1/P2/P3），自建参考实现，不复用被审代码 |
| `docs/audit_m14488/probe_results.json`, `probe_results_p1.json` | 探针原始输出 |
| `docs/audit_m14488/fig_audit.png` | 图 1（求积复核误差阶梯）+ 图 2（flexural 双实现分歧） |
| 本文件 | 分类清单 + 六问表 + L 判级 + top 问题 + 膨胀清单 |

---

## 0. 方法与先决陷阱（先做，否则证据全废）

### 0.1 环境陷阱：本机 `import tensorbem` 默认指向**别的 worktree**

```
$ cat /usr/lib/python3.11/site-packages/__editable__.tensorbem-0.5.0.pth
/root/tp-agent/work/m14010/src
$ python -c "import tensorbem; print(tensorbem.__file__)"
/root/tp-agent/work/m14010/src/tensorbem/__init__.py
```

本机 editable 安装 + `sys.path` 中还挂着 8 个其它 worktree（m14461 / m12604 / m10906 / m13963 / m14464 / m10660 …），
`solve-sim` 出来的 `tensorbem` 是 **m14010（HEAD `6eb42b5`，FMM 加速分支）**，不是被审提交。
因此**任何不带 `PYTHONPATH=<本 worktree>/src` 的 pytest / python 运行，其"绿"不能作为本 commit 的证据**。
本报告所有运行证据一律用 `export PYTHONPATH=/root/tp-agent/work/m14488/src`，并在运行时打印
`tensorbem.__file__` == `/root/tp-agent/work/m14488/src/tensorbem/__init__.py` 断言（见 §4 命令）。
**这是平台级证据污染源**，不只是本仓问题：本机任何 agent 的"测试通过"都可能是别的工作树的测试通过。

### 0.2 复核原则

1. 先读代码/工件，再动手；
2. 用**自建参考实现**独立复核（不复用被审模块的求积/几何/核函数）；
3. 报告里声称的数字，先用仓内 shipped 函数复现，**复现不出来不等于造假**——先排除自己用错参数（§4.3 就是这样一个自我纠错的实例）；
4. 无法验证一律记 WITHHELD / L0–L1，不因报告篇幅升级。

---

## 1. 资产盘点（26 个已并 PR，200 个文件，56,411 新增行）

分类口径（`inventory_scan.py: CATEGORY_RULES`）：`core_module` = `src/tensorbem/*.py` 且不在
`benchmarks/` 下；`benchmark_module` = `src/tensorbem/benchmarks/*`；`tool` = `scripts/ tools/ examples/`；
`test` = `tests/`；`doc` = `docs/ papers/ *.md`；`artifact` = `artifacts/ benchmarks/verified/ *.json/png/csv`；
`other` = 仓根散落脚本等。

| 类别 | 新增行 | 占比 |
|---|---|---|
| 文档/报告（docs/ papers/ *.md） | 19,743 | 35.0% |
| 核心算法模块（src/tensorbem/*.py） | 8,281 | 14.7% |
| 基准/研究驱动（src/tensorbem/benchmarks/*） | 7,758 | 13.8% |
| 测试代码（tests/） | 5,805 | 10.3% |
| 工具脚本（scripts/ tools/ examples/） | 5,376 | 9.5% |
| 工件/图片/JSON（含 papers/data 转写） | 7,339 | 13.0% |
| 仓根散落（post_report_*.py / run_*.py / .m*_openpr.py 等） | 2,109 | 3.7% |
| **合计** | **56,411** | 100% |

**逐 PR**（行数为新增行；`core/bench/test/doc/tool/art` 为对应类别行数）：

| PR | mission | 文件 | 新增 | core | bench | test | doc | tool | art/other |
|---|---|---|---|---|---|---|---|---|---|
| #600 | m8799 | 8 | 610 | 0 | 68 | 114 | 244 | 0 | 184 |
| #601 | m8718 | 9 | 2702 | 290 | 720 | 229 | 993 | 0 | 470 |
| #602 | m9147 | 3 | 1429 | 761 | 315 | 353 | 0 | 0 | 0 |
| #603 | m9162 | 17 | 6245 | 1781 | 1343 | 318 | 2577 | 0 | 226 |
| #604 | m9204 | 4 | 870 | 0 | 431 | 249 | 0 | 0 | 190 |
| #605 | m9161 | 10 | 7665 | 11 | 566 | 168 | 1350 | 646 | 4924 |
| #606 | m9269 | 3 | 1359 | 736 | 0 | 476 | 0 | 147 | 0 |
| #607 | m9285 | 3 | 1027 | 0 | 453 | 425 | 149 | 0 | 0 |
| #608 | m9286 | 5 | 1914 | 536 | 642 | 346 | 253 | 0 | 137 |
| #609 | m9388 | 7 | 1466 | 1244 | 0 | 222 | 0 | 0 | 0 |
| #610 | m9641 | 12 | 7645 | 0 | 1056 | 351 | 2306 | 1305 | 2627 |
| #611 | m9647 | 8 | 4715 | 0 | 0 | 142 | 4571 | 0 | 2 |
| #612 | m9696 | 13 | 3230 | 140 | 1068 | 180 | 1842 | 0 | 0 |
| #613 | m9712 | 10 | 888 | 0 | 114 | 143 | 370 | 0 | 261 |
| #614 | m10003 | 9 | 1871 | 901 | 0 | 402 | 124 | 444 | 0 |
| #615 | m10370 | 8 | 2853 | 639 | 836 | 300 | 807 | 0 | 271 |
| #616 | m11067 | 5 | 206 | 0 | 146 | 59 | 1 | 0 | 0 |
| #617 | m11546 | 13 | 1123 | 0 | 0 | 0 | 399 | 724 | 0 |
| #618 | m12052 | 12 | 2222 | 562 | 0 | 472 | 761 | 427 | 0 |
| #619 | m8718 | 0 | 0 | —（空合并，无 diff） | | | | | |
| #620 | m13784 | 1 | 52 | 0 | 0 | 0 | 52 | 0 | 0 |
| #621 | m13779 | 10 | 1806 | 424 | 0 | 218 | 445 | 719 | 0 |
| #622 | m13787 | 2 | 258 | 0 | 0 | 147 | 111 | 0 | 0 |
| #623 | m13806 | 6 | 328 | 0 | 0 | 83 | 106 | 139 | 0 |
| #624 | m13811 | 10 | 1016 | 22 | 0 | 106 | 762 | 126 | 0 |
| #625 | m13836 | 12 | 2911 | 234 | 0 | 302 | 1520 | 699 | 156 |

### 1.1 占位实现扫描

AST 扫描全部 `src/tensorbem/**/*.py`：`pass` 空体 / `...` / `raise NotImplementedError` 命中 **1 处**——
`src/tensorbem/benchmarks/mixed_boundary.py:110,113` 的抽象基类方法（`NotImplementedError` 为接口契约，非占位实现）。
`TODO` / `FIXME` = **0**。**结论：本窗口不存在空壳模块**，问题不在"假代码"，而在"真代码但无人调用 / 验证只是自证"（§2、§5）。

### 1.2 测试规模

`tests/` 205 个文件、2,409 个测试函数、5,699 条断言（断言/测试 ≈ 2.37）。
本窗口新增测试 5,805 行 / 200 文件里 60 个测试文件——**测试是真写的，不是空 test 占位**。
但"绿"只证明自洽，不证明对外精度（§4）。

### 1.3 本次实跑的测试证据（`PYTHONPATH=<本 worktree>/src`，仅本审计跑了这些文件）

| 测试文件 | 结果 | 备注 |
|---|---|---|
| `tests/test_nk_panel3d.py` | **38 passed** | #614 |
| `tests/test_finite_depth_solver3d.py` | **16 passed** | #621 |
| `tests/test_nearfield_quadrature.py` | **14 passed, 2 skipped** | 2 项跳过因 `artifacts/` 未生成（gitignore） |
| `tests/test_flexural_gravity.py` | **53 passed** | #618 |
| `tests/test_flexural_gravity_mass_loading.py` + `test_capability_evidence.py` | 见下 | — |
| `tests/test_flexural_gravity_mass_loading.py` | **23 passed**（并入上行的 73 passed 批次） | #625 |
| `tests/test_memory_kernel_matrix.py` | **13 passed** | #601–#603 |
| `tests/test_ellipsoid_kochin_three_route.py` | **1 failed**（`test_panels_source_bounded`：k=4.0 相对误差 **65.2%** vs 自设 20% 界） | **#622 携带红测试合并** |
| `tests/test_state_space.py` | **11 passed, 5 FAILED**（`test_vector_fit_hemisphere`…：拟合最大相对误差 **6.15e-01** vs 判据 0.15；另见 NaN `matmul` 警告） | **#602 自带红测试** |
| `tests/test_capability_evidence.py` | **2 failed**：`test_uncapability_baseline_has_no_untested_module_without_a_reason`（25+ benchmark 模块无测试无理由）、`test_published_block_matches_the_tree`（`docs/audit_baseline.md` §4 与树不一致） | **仓自身审计门是红的** |
| `tests/test_drift_far_field3d.py`、`test_irregular_frequency_sanitisation.py`、`test_seakeeping_verification.py` | **未跑完**（>2 min/文件，共享 99 上与其他 agent 作业竞争；集群作业 `cj834` 排队 40 min 未派发） | 如实记 "not_run" |

**结论**：窗口内 PR 报告里"X passed"全部是**局部**口径；**仓级门在 HEAD 上是红的**（至少 8 项失败，横跨 3 个模块 + 2 条治理门）。

---

## 2. 结构接线审计：窗口内新增核心模块 **0 个**进入库主面

`inventory_scan.py` 用 AST 解析 import（含 `from tensorbem import X as Y`、相对导入），把每个
`src/tensorbem/<m>.py` 的消费者分成 `core`（被库内非 benchmarks 模块 import）/ `bench`（被
`benchmarks/` import）/ `tool`（被 scripts/tools import）/ `tests_only` / `orphan`。

| 新增核心模块（本窗口） | 行数 | 接线层 | 消费者 |
|---|---|---|---|
| `nk_panel3d` (#614) | 900 | tool | 仅 `scripts/nk_panel3d_study.py` |
| `finite_depth_solver3d` (#621) | 423 | tool | 仅 `scripts/verify|report_finite_depth_solver3d.py` |
| `flexural_gravity` (#618/#625) | 794 | tool | 仅 `scripts/verify_flexural_gravity*.py` |
| `near_field_quadrature` (#615) | 606 | bench | `benchmarks/nearfield_quadrature.py` |
| `irregular_frequency` (#606) | 736 | tool | 仅 `tools/gen_irregular_frequency_figures.py` |
| `drift_far_field3d` (#608) | 536 | bench | `benchmarks/drift_far_field_3d_study.py` |
| `memory_kernel_matrix` (#600–#603) | 1745 | bench | `benchmarks/cummins_matrix_closure.py` |
| `state_space` (#602) | 761 | bench | `benchmarks/freq_time_bridge.py` |

**0/8 被 `core` 层 import。** 也就是说：本窗口交付的每一个新核心模块，**都没有成为库（`solver3d` /
`free_surface_solver3d` / `rao` / `wigley` / `time_domain` 等）的能力**，它们只被同批 PR 自带的研究脚本/基准脚本消费。
按 owner 六级定义，这直接封死 L3（系统集成）——**无论报告写得多长**。

对比：真正在 `core` 层被消费的模块（`solver3d`、`kernels3d`、`near_singular`、`finite_depth3d`、
`recovery`、`hydrostatic`、`rao` …）都是窗口之前就存在的。

### 2.1 重复实现（与既有模块功能重叠）

| 新模块 | 既有模块 | 重叠内容 | 证据 |
|---|---|---|---|
| `near_field_quadrature`（近奇异三角形求积，三分支） | `near_singular`（近奇异三角形求积，自适应二分） | 同一物理/数学功能；`solver3d` 只 import `near_singular`（`solver3d.py:38`），新模块无人用 | `grep -n "near_singular\|near_field_quadrature" src/tensorbem/solver3d.py` |
| `flexural_gravity`（冰板色散/冰缘反射） | `hydroelastic.flexural_gravity_omega` + `ice.mass_loading_omega` | 同一色散关系的第三份实现 | §4.2 数值分歧 0.17%→14.2% |
| `nk_panel3d`（NK 稳态兴波 BEM） | `neumann_kelvin`（NK 验证面，432 行） | 同一 NK 线性化+兴波阻力目标 | 两者均只被 benchmark/script 消费 |
| `state_space`（Cummins 记忆核指数拟合） | `memory_kernel_matrix`（K_ij(t) 矩阵核，同窗口） | 时域卷积降阶的近邻能力 | 两者互不引用，各自 bench-only |

---

## 3. 功能真实性六问表（窗口内核心功能逐条）

六问 = ①代码入口在哪 ②是否真的执行计算 ③是否接入主程序 ④是否存在实际输入输出 ⑤是否有完整运行案例
⑥是否有独立基准验证。`判定` 列引证据文件:行号 / 工件 / 测试名。

| # | 功能（PR） | ①入口 | ②真算? | ③接入主程序? | ④实际 I/O | ⑤运行案例 | ⑥独立基准 | 判定 |
|---|---|---|---|---|---|---|---|---|
| 1 | NK 三维面板法 (#614) | `nk_panel3d.py: solve_nk_panel`, `nk_wave_kernel`, `green_kernel` | 是：自由面 Green 函数（Ei 渐近+PV）+ 稠密影响矩阵求解，`pytest tests/test_nk_panel3d.py` 38 passed | **否**（仅 `scripts/nk_panel3d_study.py`） | `NKPanelResult` dataclass；脚本写 `artifacts/nk_panel3d_study.json`（**被 .gitignore 忽略，仓内不存在**） | 有脚本，但工件不入仓 | 核函数门：我独立复现 4.53e-7 残差 / 3.02 方向比（§4.3）；兴波阻力 vs Michell **FAIL（−60×…+47×）** | **L2** |
| 2 | 三维有限水深辐射 BEM (#621) | `finite_depth_solver3d.py: solve_radiation_finite_depth_3d` | 是：John 本征级数核 + 直接 Green 恒等式 + 自适应模态截断 | **否**（仅 scripts） | `FiniteDepth3DResult`；`docs/finite_depth_solver3d/results.json` 入仓 | 有 `scripts/verify_…py` + `results.json` 入仓 | vs 仓内多极解析解：A33 3.77%、**B33 19.6%（80 面元，最细已发布网格）**；脚本文档承诺的 A4 320 面元阶梯 **未出现在 results.json** | **L2** |
| 3 | 近场近奇异求积 (#615) | `near_field_quadrature.py: adaptive_triangle_quadrature / polar_triangle_quadrature / semi_analytic_reference` | 是：三分支求积 + 能量恒等式 + Hulme 锚 | **否**（bench-only；`solver3d` 仍用 `near_singular`） | 返回 dict；脚本写 `artifacts/nearfield_quadrature/`（**gitignore，仓内不存在 → 15 项测试中 2 项 SKIP**） | 有 `python -m tensorbem.benchmarks.nearfield_quadrature --stage …` | vs 我自建暴力求积参考：单层 1.4e-13 … 1.7e-4（受参考自身收敛限制，§4.1） | **L2** |
| 4 | flexural-gravity 冰板色散/冰缘 (#618) | `flexural_gravity.py: flexural_gravity_omega / ice_edge_coefficients / wedge_ice_reflection` | 是：色散闭式 + 冰缘 R/T 匹配解 | **否**（仅 scripts） | `IceEdgeResult`；`docs/flexural_gravity_verification/*.json` 入仓 | 有 `scripts/verify_flexural_gravity.py` | 仅自洽极限（k→0 退开敞水、k→∞ 退弯曲波、|R|²+|T|²=1）与参数单调性，**无实验/文献系数对照** | **L1** |
| 5 | 统一 flexural+mass-loading (#625) | `flexural_gravity.py: flexural_gravity_omega2_mass_loading` 等 8 个新符号 | 是：闭式解析式 | **否** | `docs/flexural_gravity_mass_loading/verification_report.json` 入仓 | 有 `scripts/verify_flexural_gravity_mass_loading.py`（V1–V10 全 PASS） | **V1–V10 全部是同一闭式的代数极限/往返比对**（阈值 1e-9…1e-14）——数学上必然成立，**不构成独立基准**；无实验、无 BEM 耦合算例 | **L1** |
| 6 | 椭球 Kochin 三路径交叉验证 (#622) | 仅 `tests/test_ellipsoid_kochin_three_route.py`（147 行），无新 src | 是（调用既有 `ellipsoid_kochin`） | 不适用（纯测试 PR） | 测试断言 | 测试即案例 | analytic vs quadrature：两法同源于同一闭式 → <0.1% 属**同族自证**；panels 路线自认 4% 建模误差 | **L1** |
| 7 | 不规则频率预测与压制 (#606) | `irregular_frequency.py: predict_irregular_frequencies / sanitise_with_chief / sanitise_with_interior_surface` | 是：CHIEF 内点 + 内水面 lid 增广最小二乘 | **否**（仅 tools） | 返回 dataclass；测试含 GCI 网格收敛 | 有测试算例（tiny_mesh） | 混合：既有 `bare 伪共振负控`、CHIEF 条件数、GCI 收敛（真验证），也有**自证**条目——`test_prediction_error_vs_analytic_less_than_2pct` 的 docstring 自认“the prediction **IS** this analytical estimate” | **L1**（跑完可升 L2） |
| 8 | 三维远场平均漂移力 (#608) | `drift_far_field3d.py: maruo_far_field_drift / control_circle_drift / radius_scan` | 是：Kochin 耦合 + 半径扫描 + 能量恒等式 | **否**（bench-only） | 返回 dict/array；`benchmarks/drift_far_field_3d_study.py` | 有 study 脚本 | 与 `drift.far_field_mean_drift_3d` 权重一致性（同式比对）+ 能量恒等式；**本审计仅跑到 26 passed+1 skip 未出总结** | **L1**（跑完可升 L2） |
| 9 | 多自由度记忆核矩阵 (#601–#603) + Cummins 闭合 (#617) | `memory_kernel_matrix.py: K_ij(t) / complex_resonance` | 是：频域 B_ij(ω)→时域核 →复数共振闭式 | **否**（bench-only） | `benchmarks/cummins_matrix_closure.py`；`docs/cummins_memory_closure/m11546/*.json` | 有生成脚本 + 报告 | 核 vs 闭式（`test_kernel_matches_closed_form`）、KK 因果性、自由衰减 vs 核根（真验证链，13 passed） | **L2**（核矩阵本体；#617 闭合研究脚本本次未重跑 → 单列 L1） |
| 10 | 状态空间降阶 (#602) | `state_space.py: state_space_from_fit / cummins_state_space / validate_state_space` | 是：指数和拟合 + 辅助 ODE | **否**（bench-only） | `benchmarks/freq_time_bridge.py` | 有 | 频时往返一致性；**自带精度门不通过：`tests/test_state_space.py` 5 failed（拟合误差 6.15e-01 vs 0.15）** | **L1** |
| 11 | 耐波性验证工具包 (#609) | `verification/seakeeping/{benchmark_runner,hemisphere_anchor,wigley_rao,mesh_convergence,fk_excitation}.py` | 是（但它是验证基础设施，不是求解器） | 否（库内验证面，非求解路径） | 报告 JSON + 图 | `run_full_verification()` | 半球 0 频 A33 vs Hulme <5% 锚（`benchmark_runner.py:239`）；**本审计未跑完该测试文件** | **L1**（工具链；跑完可升 L2） |
| 12 | W20 锚转写/截断对比 (#610/#611) | `papers/data/w20-anchors/*.py` | 转写+对比计算，非求解器 | 否 | `benchmarks/verified/bem_kp505_openwater_w20/gci.json` 等入仓 | 有 | 锚数据来自文献转写，本 PR 不产生新解 | **L0** |
| 13 | 有限水深离轴求积修复 (#624) | `finite_depth3d_quadrature.py`（**core**） | 是：修复非仿射离轴面元的求积拆分 | **是**（被 `finite_depth3d` 消费，后者被 core 消费） | 修复前后对照 JSON | `scripts/m13811_…offaxis_fix.py` | 修复前/后误差对照 + 负控 | **L3**（集成路径上的缺陷修复） |
| 14 | 图片完整性门 (#623) | `tests/_figure_integrity.py` | 静态校验 PNG/markdown 引用 | 否（CI 门） | 测试 | 测试即案例 | 无（治理门，非物理） | **L1**（治理） |
| 15 | 阴极保护验证 (#604) | `benchmarks/cp_verification_m9204.py` | 是 | 否（bench-only） | 测试 + 报告 | 有 | 未深查（本轮未展开，见 §7） | **WITHHELD** |

### 3.1 判级分布（15 个被审对象）

| L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| 1 (#610/#611 W20 转写) | 9 (#618, #625, #622, #623, #602, #606, #608, #617, #609) | 4 (#614, #621, #615, #601–#603) | 1 (#624) | **0** | **0** |

**L4 = 0，L5 = 0**：窗口内没有任何一个对象同时具备「完整算例 + 收敛/守恒 + 误差分析 + 可复现基准对照」四要素。
最接近 L4 的 #621 在 80 面元处阻尼误差 19.6%，且承诺的 320 面元阶梯缺失（§5 I-2）。
Verdict 全部落表 `tensor_maturity_verdicts`（见 §6，16 行：15 个对象 + 1 条 HEAD 门状态快照）。

---

## 4. 独立复核实验（自建参考，不复用被审代码）

### 4.1 P1 — 近场求积第三方复核（`probe_results_p1.json`）

方法：对单位直角三角形，目标点在形心上方 `h`、近顶点上方 `h`；**自建**均匀细分（n_sub=40/56/72）
+ 6 点四次规则（权重已归一，`sum(w)=1`）作参考，与被审三条路线（自适应二分 / 分级极坐标 / 模块内
semi-analytic 仲裁）比对单层核 `1/(4πr)`：

| 案例 | 参考自身误差 | adaptive | polar | semi-analytic |
|---|---|---|---|---|
| h=1e-1 形心 | 3.4e-09 | 1.4e-13 | 1.3e-10 | 4.3e-14 |
| h=1e-2 形心 | 6.2e-05 | 1.2e-05 | 1.2e-05 | 1.2e-05 |
| h=1e-3 形心 | 6.7e-04 | 1.7e-04 | 1.7e-04 | 1.7e-04 |
| h=1e-2 近顶点 | 4.4e-05 | 9.3e-06 | 3.6e-04 | 9.3e-06 |

结论：三条路线在第三方参考的收敛极限内一致（差异量级 = 我参考自身的离散误差），
`near_field_quadrature` 的求积**是真实的、精度可复现的**（L2 的 ⑥ 项成立）。
**未定论**：双层核 `-h/(4πr³)` 我的参考在 h≤1e-3 未收敛（自身误差主导），故双层对比不作结论（honest）。
另：本次首次运行得到"三条路线齐差 17.1%"的假警报，根因是**我自己**抄错 Dunavant 权重（权重和 0.8536）；
归一化后消失——记录在此作为"先自查参考再指控"的实例。

### 4.2 P2 — flexural 色散三实现分歧（`probe_results.json: p2`）

同一冰板（h=1 m, E=5 GPa, ν=0.334, H=200 m）：

| k (1/m) | `flexural_gravity` ω | `hydroelastic` ω | 相对差 |
|---|---|---|---|
| 0.005 | 0.1932 | 0.1929 | 0.17% |
| 0.05 | 0.7958 | 0.7786 | 2.16% |
| 0.1 | 2.3571 | 2.2583 | 4.19% |
| 0.2 | 12.1810 | 11.2186 | 7.90% |
| 0.4 | 68.4777 | 58.7655 | **14.18%** |

同时 `flexural_gravity_group_velocity` 与其自身 ω(k) 的中心差分吻合到 1e-10…1e-11（解析导数正确，**但这是同式自证**）。
含义：仓内三份"冰板色散"实现（外加 `ice.mass_loading_omega`）在高波数端相差 14%，没有任何一处文档
说明三者各自的模型项（刚性 / 冰质量惯性 / 有限水深）导致差异——**跨模块数值不一致风险**，落地到冰区 RAO 时会直接串味。

### 4.3 P3 — #614 报告数字复现（自我纠错实例）

先按默认参数（`radiation='downstream'`）调 `verify_radiation_direction`，得方向比 **0.986**（≈1，无方向性），
与报告宣称的 "3.0" 冲突；**但报告写的是 halfplane 模式**（`docs/m10003/REPORT.md:51` 表头 + 测试
`tests/test_nk_panel3d.py:164-171` 用 `"halfplane"`）。改用 halfplane 后：

```
halfplane sign=-1 d=8 : up=0.0377 down=0.1140 ratio=3.020   ← 与报告 3.0（后 0.114 / 前 0.038）一致
halfplane sign=+1 d=8 : up=0.1156 down=0.0369 ratio=0.319   ← 与报告 0.32 反证一致
verify_free_surface_condition: max_relative_residual = 4.53e-07（报告 7.53e-06，更保守）
verify_laplace: laplacian = 4.16e-07（报告 2.9e-07，同量级）
```

结论：**#614 报告的自报数字可复现**，其"核函数通过 / 阻力 honest FAIL"的定性判断成立——这是本窗口
诚实纪律的正面样本（对照 §5 的膨胀清单）。

---

## 5. Top 问题（按严重度）

### I-1（高）**新增核心模块全部未接线，L3/L4 结构性不可达**
8 个新核心模块（4,801 行）`core` 消费者 = 0（§2）。平台侧看到的是"TensorBEM 交付了有限水深 3D 求解器 /
NK 面板法 / 近场求积…"，而库的实际能力面（`solver3d` / `free_surface_solver3d` / `rao`）**一行未变**。
复现：`PYTHONPATH=src python3 docs/audit_m14488/inventory_scan.py | grep orphan/wiring`。
危害：任何"能力增量"口径若只数模块名，会系统性高估；下游（TensorPhysics 冰区流程 / 平台 RAO 接口）拿不到这些能力。

### I-2（高）**#621 承诺的 320 面元阶梯缺失，最细已发布网格阻尼误差 19.6%**
`scripts/verify_finite_depth_solver3d.py:17` 声明 A4 = `20/80/320` 面元阶梯 vs 解析多极解；
`docs/finite_depth_solver3d/results.json` 中 **A4 键不存在**（`grep -c A4` = 0），报告 §6 只引用 20 面元的
17.21% / 53.78%，并把"趋势"寄托于"深水仓既有 320 面元基准"。而 80 面元行的真实数字是：
A33 误差 **3.77%**（>3% 判据），B33 误差 **19.6%**（**>15% 不可接受线**）。报告未在误差判定节披露 80 面元行。
复现：`python -c "import json;d=json.load(open('docs/finite_depth_solver3d/results.json'));print(d['A1']['ladder'][1])"`。

### I-3（中高）**"验证完成"= 同一闭式的代数极限（#625 V1–V10）**
`docs/flexural_gravity_mass_loading/verification_report.json` 10 条判据全 PASS，阈值 1e-9…1e-14：
V1 惯性→0 极限、V2 短波渐近、V3 浅水渐近、V4 与 `ice.py` 一致性、V5 交叉波长（同式二分）、V6 群速度（同式差分）、
V7 波数往返、V8 冰缘能量、V9/V10 参数扫描。**全部可由闭式在纸面推出**，不涉及任何独立数值方法或实验。
这类"验证"只能证代码抄对了公式（L1），不能证物理（L4）。全仓 `docs/audit_baseline.md` 的能力表把此类条目
统一标 `REAL`（判据是"模块+触及测试+文档锚点三者可解析"，属结构性判据而非数值判据）——**这是全仓级别的口径膨胀源**。

### I-4（中高）**已发布证据不入仓：`artifacts/` 被 gitignore**
`.gitignore:10: artifacts/`。#614 的 `artifacts/nk_panel3d_study.json`、#615 的 `artifacts/nearfield_quadrature/*`
都是报告数字的唯一来源，却不在仓内；fresh checkout 下 `tests/test_nearfield_quadrature.py` **2 项直接 SKIP**
（我实测：`14 passed, 2 skipped`，skip 理由 "artifacts not generated in this checkout"）。
→ 报告不可复现、验证面在 CI 上自动缩水。

### I-5（中）**重复实现 + 数值分歧无仲裁（§2.1、§4.2）**
`near_field_quadrature` vs `near_singular`；`flexural_gravity` vs `hydroelastic` vs `ice`（高波数端差 14%）；
`nk_panel3d` vs `neumann_kelvin`。仓内无"哪个是权威实现"的登记，后续调用方会挑一个用。

### I-6（中）**自证式测试混在真验证里（#606）**
`tests/test_irregular_frequency_sanitisation.py:111-129` 两条用例把"解析估计"与"解析估计"比，
docstring 自认 "the prediction **IS** this analytical estimate"。同文件其余 19 条（伪共振负控、CHIEF 抑制、
GCI 收敛）是真验证 —— 问题不是造假，而是**一个文件里 L1 与 L2 混排，读报告的人无法区分**。

### I-7（低-中）**仓根散落过程脚本 2,109 行**
窗口内新增 `post_report_m*.py`（12 个）、`run_cp_*.py`、`make_report_*.py`、`.m8799_openpr.py`
直接落在仓根（`inventory.json: merges[*].files[category=other]`）。无密钥硬编码（`debug_auth*.py` 读 env，
已核），但属过程遗留，污染仓结构；建议统一挪到 `tools/reporting/`。

### I-9（高）**最新合并 PR #622 自带红测试：panels 路线在 k=4.0 误差 65.2% > 自设 20% 界**
`tests/test_ellipsoid_kochin_three_route.py::TestThinShipPanelsBounded::test_panels_source_bounded`
在 HEAD 实测 `AssertionError: k=4.0: rel_err=6.5222e-01`。PR 报告口径"panels 路线有界（<20%）"在短波段不成立，
测试与报告同时入仓但**互相矛盾**，合并门未拦住。
复现：`PYTHONPATH=src python -m pytest -q tests/test_ellipsoid_kochin_three_route.py` → `1 failed, 12 passed`。

### I-10（高）**#602 `state_space` 5/16 测试失败：矢量拟合精度 61.5% vs 判据 15%**
`tests/test_state_space.py::test_vector_fit_hemisphere` 实测 `Max error 6.153e-01`（拟合 `converged=True`、`n_iter=959`，
极点出现 3 个 1e-6 简并极），另有 `state_space.py:501/511/515` 的 `invalid value encountered` NaN 警告 ⇒ 时域积分散架。
`git log --oneline -- src/tensorbem/state_space.py` 只有 `50d2ca3 Merge #602` 一条，即**代码自合并起未变**，
故不是后续回归，而是"合并即红"（本机 numpy 2.4.6 / scipy 1.16.0；CI 环境可能不同，需在 CI 复核，见 §8）。
按纪律：该模块判 L1（有自己的精度门且**不通过**）。

### I-11（中高）**仓自身的能力台账门是红的：`docs/audit_baseline.md` 与树不一致**
`tests/test_capability_evidence.py` 两条治理门失败：
① `25+` 个 `tensorbem.benchmarks.*` 模块既无触及测试也无 `UNTESTED_MODULES` 理由；
② §4 生成块与树漂移（缺 `tensorbem.cylinder_drift`、`tensorbem.drift_far_field3d`、`tensorbem.nk_panel3d` 等）。
即：平台若按 `docs/audit_baseline.md` 读"能力"，读到的是**过期表**；且 §5-I-3 已指出该表的 `REAL` 判据是结构性的。

### I-8（低）**证据环境不可信：本机 editable `.pth` 指向别的 worktree（§0.1）**
若 CI/agent 未显式设置 `PYTHONPATH`，`pytest` 通过的是 m14010 的代码。建议仓内加 `tests/conftest.py`
断言 `tensorbem.__file__` 位于本仓 `src/`（唯一能机器化堵住此漏洞的位置）。

---

## 6. 成熟度判级 → `tensor_maturity_verdicts`（16 行已 INSERT，mission_id=14488，verdict_by='audit2-ab'）

判级严格按 owner 定义；L≥2 必须实际执行证据（本报告 §1.2 测试实跑 + §4 探针），L≥4 必须基准对照数据——
本窗口 **无一对象满足 L4**，故最高写到 L3。

| scope | level | 关键证据指针 |
|---|---|---|
| `nk_panel3d#614` | 2 | `tests/test_nk_panel3d.py` **38 passed**；探针复现残差 4.53e-7 / 方向比 3.02（`probe_results.json:p3`）；`docs/m10003/REPORT.md` 自报阻力 honest FAIL；wiring=tool（未集成 → 非 L3） |
| `finite_depth_solver3d#621` | 2 | `tests/test_finite_depth_solver3d.py` **16 passed**；`results.json:A1.ladder[1]` rel_err_A33=0.0377 / rel_err_B33=0.1956；A4（320 面元）缺失；wiring=tool |
| `near_field_quadrature#615` | 2 | `tests/test_nearfield_quadrature.py` **14 passed 2 skipped**（工件 gitignore）；`probe_results_p1.json` 第三方参考 1.4e-13…1.7e-4；wiring=bench |
| `memory_kernel_matrix#601-603` | 2 | `tests/test_memory_kernel_matrix.py` **13 passed**；闭式核 / KK 因果 / 自由衰减三链；wiring=bench（未接 `time_domain` 主路径 → 非 L3） |
| `flexural_gravity#618` | 1 | `tests/test_flexural_gravity.py` **53 passed**，但判据仅自洽极限与单调性；`docs/flexural_gravity_verification/verification_report.json`；无外部基准 |
| `flexural_gravity_mass_loading#625` | 1 | `tests/test_flexural_gravity_mass_loading.py` **23 passed**；`verification_report.json:criteria` V1–V10 全为同一闭式的代数极限（阈值 1e-9…1e-14） |
| `ellipsoid_kochin_three_route#622` | 1 | `tests/test_ellipsoid_kochin_three_route.py` **1 failed 12 passed**：`test_panels_source_bounded` k=4.0 rel_err=6.52e-01（自设 20% 界）；analytic↔quadrature 同族 <0.1% |
| `irregular_frequency#606` | 1 | 代码 + 21 项测试存在（CHIEF/lid、GCI、负控），**本审计未跑完**（>2 min，`cj834` 未派发）→ 按纪律不升 L2；另 `tests/...:111-129` 两条为同式自证 |
| `drift_far_field3d#608` | 1 | `tests/test_drift_far_field3d.py` 本审计仅跑到 26 passed + 1 skip 未出总结；代码/能量恒等式存在；wiring=bench |
| `cummins_memory_closure#617` | 1 | `docs/cummins_memory_closure/m11546/verification_report_m11546.json` 频时往返 + KK；脚本未在本次审计重跑 |
| `state_space#602` | 1 | **`tests/test_state_space.py` 5 failed 11 passed**（`test_vector_fit_hemisphere`：6.153e-01 vs 0.15；NaN 警告 `state_space.py:501/511/515`）⇒ 自带精度门不通过 |
| `verification_seakeeping#609` | 1 | 工具链代码 + Hulme 锚（`benchmark_runner.py:239` <5% 门）；`tests/test_seakeeping_verification.py` **本审计未跑完** |
| `finite_depth3d_offaxis_fix#624` | 3 | `tests/test_finite_depth3d_quadrature_offaxis_fix.py` **7 passed**；`finite_depth3d_quadrature.py` 在 core 消费路径上；`docs/m13811/outputs/m13811/fix_verification.json` 修复前后对照 |
| `w20_anchor_transcription#610-611` | 0 | `papers/data/w20-anchors/*.json` 文献转写 + 对比报告；本 PR 不产生新解（L0 = 文档成果，不计功能） |
| `figure_integrity_gate#623` | 1 | `tests/_figure_integrity.py`；治理门，无物理算例 |

（分布：L0=1，L1=9，L2=4，L3=1，L4=0，L5=0。L≥2 均有本审计实跑证据；未跑完的一律压到 L1 并写明"跑完可升"。）

## 7. 「曾声称为功能、实为文档/测试/同式自证」膨胀清单

| # | 声明（报告/PR） | 实况 | 证据 | 应有口径 |
|---|---|---|---|---|
| E-1 | #625「统一 flexural+mass-loading 色散 **实现与验证**，10 判据全 PASS」 | 10 判据全是同一闭式的代数极限/往返，阈值 1e-9…1e-14；无独立方法/实验 | `docs/flexural_gravity_mass_loading/verification_report.json:criteria` | L1（公式抄对），非"物理验证" |
| E-2 | #621「A1–A6 六组锚点验证」 | A4（20/80/320 面元阶梯）**未产出**；报告误差判定只引 20 面元行，隐去 80 面元 B33=19.6% | `scripts/verify_finite_depth_solver3d.py:17` vs `results.json` 无 A4；`report.md:104` | 附 80 面元行；B33 标注 >15% 不可接受 |
| E-3 | #622「三路径交叉验证」 | 无新代码，纯测试；两条"精确"路线同源同式（<0.1% 属自证），第三条自认 4% 建模误差 | `tests/test_ellipsoid_kochin_three_route.py:1-20` | L1；"交叉验证"应写明同族自证 |
| E-4 | #615「近边界近奇异求积可信域 + Hulme 锚」 | 求积本身真（我第三方复核通过），但**发布工件 gitignore**，2/15 测试在 fresh checkout 直接 SKIP | `.gitignore:10`；`pytest` 输出 `14 passed, 2 skipped` | 工件入仓或 CI 生成，否则证据面不可复现 |
| E-5 | 全仓 `docs/audit_baseline.md` 32 行能力表全 `REAL` | 判据 ="模块+触及测试+文档锚点三者可解析"，结构性，非数值 | `src/tensorbem/capability_evidence.py:1-40` | `REAL` 应拆 `REAL(结构)` / `VERIFIED(数值)` |
| E-6 | #606「不规则频率预测 vs 解析解误差 <2%」 | 预测器**就是**该解析估计（docstring 自认），<2% 是同义反复 | `tests/test_irregular_frequency_sanitisation.py:111-129` | 删该判据或改为 vs 2D Helmholtz 数值特征值 |
| E-7 | #617「Status: ✅ PASS」 | 该项确为频时往返闭合（真验证），但 memory kernel 侧全程 bench-only，未接 `time_domain` 主路径 | `inventory.json: wiring.memory_kernel_matrix` | 标注"验证通过但未集成" |

（对照：**未**列入膨胀清单的 #614 是正面样本——自报 honest FAIL 且数字可复现，§4.3。）

---

## 8. 未做 / 局限（honest）

1. **未跑全量测试套件**：`pytest tests/` 在共享 99 上与其他 agent 作业竞争，约 6 分钟仅推进 ~25/2409 项；
   已递交集群作业 `cj834`（排队中，258 位在前）。因此本报告的测试证据是**按模块定向实跑**（共 9 个测试文件、
   约 270 项，见 §1.2 与各 verdict），**不是**全量绿；全量结果以 `cj834` 为准，本报告不宣称全绿。
2. **双层核（1/r³）第三方复核未定论**（§4.1），我的参考在该奇性下未收敛，作 honest 未结论处理。
3. **#604 阴极保护验证、#616 沉球 Havelock 基准、#612 三维辐射绕射锚、#607 wigley3_freq** 只做了接线与测试存在性核查，
   未展开数值复核 → verdict 标 WITHHELD 或按证据下限给级，不拔高。
4. **精度判据**：本报告只对能取到基准的量给误差（#621 A1 行、#614 核函数门）；其余对象**无基准**，
   按纪律声明原因而不含糊过关。
5. 未做性能/内存/并行审计；未审计窗口之外的既有模块（窗口外老债另案）。

---

## 9. 复现命令（本 worktree，注意 PYTHONPATH）

```bash
cd /root/tp-agent/work/m14488
export PYTHONPATH=/root/tp-agent/work/m14488/src          # §0.1：否则读到 m14010 的代码
python -c "import tensorbem; print(tensorbem.__file__)"   # 必须打印本 worktree 路径
python docs/audit_m14488/inventory_scan.py --base-ref b17b169 --out docs/audit_m14488/inventory.json
python docs/audit_m14488/probes.py --only p2,p3 --out docs/audit_m14488/probe_results.json
python docs/audit_m14488/probes.py --only p1 --out docs/audit_m14488/probe_results_p1.json
python docs/audit_m14488/make_figures.py                   # 生成 fig_audit.png
python -m pytest -q tests/test_nk_panel3d.py tests/test_finite_depth_solver3d.py \
       tests/test_nearfield_quadrature.py tests/test_flexural_gravity.py \
       tests/test_flexural_gravity_mass_loading.py tests/test_ellipsoid_kochin_three_route.py
```

**误差判定（§关键量 vs 基准，逐项显式回答）**

| 量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| #615 单层求积（h=1e-1 形心） | 自建暴力求积 | 1.4e-13 | **是** |
| #615 单层求积（h=1e-3 形心） | 自建暴力求积 | 1.7e-4（受参考自身误差 6.7e-4 限制） | 是（且在参考不确定度内） |
| #614 自由面条件残差 | 解析（G_xx+κG_z=0） | 4.53e-7 | **是** |
| #614 辐射方向比 | 物理判据（尾迹在船后） | 3.02（>1 定向）/反证 0.32 | 定性成立 |
| #614 兴波阻力 vs Michell | Michell 薄船 | −60×…+47×，随网格变号 | **否（honest FAIL）** |
| #621 A33（80 面元，深水极限） | 仓内多极解析解 | 3.77% | 否（略超 3%） |
| #621 B33（80 面元，深水极限） | 仓内多极解析解 | 19.6% | **否（>15% 不可接受）** |
| #625 V1–V10 | 同一闭式代数极限 | 1e-11…1e-16 | "是"但**无独立基准**，不计功 |
| #618 冰板色散 vs `hydroelastic` | 仓内另一实现 | 0.17%…14.18%（k=0.005…0.4） | 低波数 是，高波数 **否**（且分歧无仲裁） |
