# CASE-BEM-RIGID-BODY-MASS —— 机器门与算例池接线

**结论：PASS** — 已提交工件 `docs/cases/case_bem_rigid_body_mass/run.json`
（sha256 `fe1772ed8dfebb9d6d53ec7dca3f549c6428054e6165ad1f231ab7aaeacb4340`）的 <3% 判据由**独立代码路径**
`tensorbem.benchmarks.rigid_body_mass_gate` 从存储矩阵重算；接受层级
（细分 3，1280 面元）最差逐元素相对误差
**1.4115%** < 3%。

本文件由 `tools/emit_case_bem_rigid_body_mass_gate_doc.py` 从工件生成，数字零手抄。

## ① 为什么需要这道门

算例 `docs/cases/case_bem_rigid_body_mass/REPORT.md` 已给出 PASS 判定与 1.412% 的误差，
但 `run.json` 是**自述式**工件：仓库里没有任何代码从存储矩阵重算那个数字，也没有东西在
装配器改动后因工件过期而变红。于是「这个锚点跑哪个入口、上次判定如何」无法机器回答
（`tensorbem.benchmarks.case_pool_registry` 里根本没有这条算例）。

本门只做一件事：**不信工件自述，从存储数据重推判定**。

1. **参考块重生成**：用工件记录的几何（semi_axes / density / c_g 偏置 / 参考点）经
   `solid_ellipsoid_mass_properties` + `rigid_body_mass_matrix` 重建闭式参考块。
   手改或过期 → 红。实测重生成差：矩阵 0.0e+00、
   惯量 0.0e+00、质量 0.0e+00
   （阈值 1e-10）。
2. **判据重算**：从 `levels[i].matrix` 与重生成参考逐元素重算误差，既与工件记录的
   `max_entry_rel_error` 对比（可复现性），也在接受层级上与 `parameters.tolerance`
   对比（真判定）。
3. **不变量复核**：对称性、正定性，以及用各层级**自身**质量特性的动能恒等式
   （正确的块形式为机器零，平台修复前的装配为 O(1)）。

**诚实声明（作用域）**：第 1 步用的是与算例相同的闭式，因此它是**新鲜度/防篡改**检查，
不是第二条物理路线；独立物理路线（面元四面体积分）在算例本体里。本门新增的是：
散文里引用的 <3% 数字现在由另一条代码路径从已提交数据重算，且过期/被改的工件不能继续挂 PASS。

## ② 判据重算结果（逐层）

| 细分 | 面元 | 工件记录 max 误差 | 重算 max 误差 | <3% | 是否计入判定 | 动能恒等式残差 |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.556477 | 0.556477 | FAIL | 不判 | 4.16e-16 |
| 1 | 80 | 0.198832 | 0.198832 | FAIL | 不判 | 3.24e-16 |
| 2 | 320 | 0.055005 | 0.055005 | FAIL | 不判 | 2.11e-16 |
| 3 | 1280 | 0.014115 | 0.014115 | PASS | 接受 | 4.50e-16 |
| 4 | 5120 | 0.003552 | 0.003552 | PASS | 接受 | 3.53e-16 |

重生成参考对角（kg / kg·m²）= 2.47306e+07, 2.47306e+07, 2.47306e+07, 1.25137e+09, 1.89634e+10, 1.91366e+10。

## ③ 门自检清单

| 检查 | 结果 |
|---|---|
| `expected_block_matches_recorded_geometry` | PASS |
| `kinetic_energy_identity_all_levels` | PASS |
| `reference_matches_recorded_geometry` | PASS |
| `reported_invariants_within_tolerance` | PASS |
| `stored_errors_reproducible` | PASS |
| `stored_matrices_positive_definite` | PASS |
| `stored_matrices_symmetric` | PASS |
| `tolerance_respected_on_accepted_levels` | PASS |
| `within_tolerance_flags_reproducible` | PASS |

全部通过 → `verdict = PASS`（失败项：无）。

## ④ 负对照：门是不是空转

`tests/test_rigid_body_mass_gate.py` 用四类真实腐化证明门有牙齿：

| 注入的腐化 | 期望 | 门捕获的检查 |
|---|---|---|
| 参考块一个元素手改 1e-6 相对量 | FAIL | `reference_matches_recorded_geometry` |
| 判据收紧到 1e-4 | FAIL | `tolerance_respected_on_accepted_levels` |
| 声称 320 面元（5.501%）可接受 | FAIL | `tolerance_respected_on_accepted_levels` |
| 平台修复前的块形式（仍**对称**） | FAIL | `kinetic_energy_identity_all_levels` |

最后一行是重点：旧缺陷**对称且正定**，只查对称性的测试对它全绿——这正是它长期存活的原因。

## ⑤ 算例池接线（本单新增）

- 注册表新增 `CASE-BEM-RIGID-BODY-MASS`：`tolerance=<3%`，
  entry=`benchmarks/verified/casepool_bem_rigid_body_mass/run_case.py`，artifacts=['docs/cases/case_bem_rigid_body_mass/verdict.json', 'docs/cases/case_bem_rigid_body_mass/run.json']，doc=`docs/cases/case_bem_rigid_body_mass/REPORT.md`。
- `scripts/case_pool_run.py --check CASE-BEM-RIGID-BODY-MASS` →
  `status=OK, verdict=PASS`。
- 一次复现：`PYTHONPATH=src python3 benchmarks/verified/casepool_bem_rigid_body_mass/run_case.py --refresh`（`--refresh` 先重跑算例再判门，约 0.3 s 单核）。

## ⑥ 顺带修掉的仓库级腐化（本单发现，非本算例）

注册表在 main 上是**红的**：三条 entry 把「按交付策略必然被丢弃的构建产物」登记成了必需件
（非 `docs/` 的 `*.png` 被推送门过滤、日志被 gitignore），`tests/test_case_pool_registry.py`
因此 4 项失败（其中一项是 `--check` 退出码 1）。处置：`CaseEntry` 新增 `regenerable`
字段——声明式区分「策略上不提交」与「丢失」；`missing()` 仍是被要求的不变量，
`--check` 新增 `regenerable_missing` 字段报告会被 `--run` 重建的产物。
`__post_init__` 拒绝把 entry/工件/文档声明成 regenerable（防止用改名掩盖腐化）。
现状：6 条 entry 全部 `status=OK`、`missing=[]`。

## ⑦ 可复现性的一个如实发现

`--refresh` 重跑算例后 `run.json` 的 sha256 **会变**：自动化递归比对显示差异只有
`levels[i].wall_time_s` 五个墙钟值，矩阵/质量/惯量/误差全部逐位相同。所以
(a) 判据数字是确定性的；(b) 工件不是字节可复现的，故门的新鲜度检查只针对几何与数值，
不针对墙钟；`verdict.json` 本身不含时间戳，可增量 diff。

## ⑧ 局限

- 未做 Fossen (2011) 页级核对（工作树无原书），故不主张 L5 外部锚点。
- 门重算的是算例已声明的判据，不新增物理路线；`assembly` / `platform_assembly` 两块在工件里
  只存了残差标量（未存矩阵），因此只能**按阈值门控**，不能重推（`verdict.json` 中标为 reported）。
- `CASE-BEM-ANNULUS-LAPLACE`、`CASE-BEM-NEUMANN-KELVIN` 两条 entry 的工件仍无法压平成标量判定
  （`see artifact`），未在本单范围内处理。
