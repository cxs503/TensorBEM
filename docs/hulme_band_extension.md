# CASE-BEM-HULME-HEMISPHERE — 第 2 轮：调研选型 + 频带扩展研究台（数值未跑出，honest 部分分）

**状态：honest 部分分。** 本轮交付的是①开源/文献横向调研结论、②方法选型与迁移方案（`oss_solutions` 载荷已生成）、
③一套预注册、可复现、判据能翻 FAIL 的频带扩展研究台（代码 + 9 项测试）。
**扩展频带（ka 2.2–4.0）的数值本轮没有跑出来**，因此本文不含任何扩展段 ref-vs-got 数字；
失败段位与原因见 §⑥，不建议把本文当作 ka > 2 的达标证据。

---

## ① 问题设定与基准出处

浮半球（平面与平均自由面齐平，半径 a = 1.0 m，无限水深，ρ = 1025 kg/m³，g = 9.80665 m/s²）
的辐射（surge / heave，单位速度幅值）与迎浪绕射。无量纲波数 ka = k·a，时间约定 exp(−iωt)。

基准 = **Hulme, G. E. D. (1982), “The wave forces acting on a floating hemisphere undergoing forced
periodic oscillations”, _J. Fluid Mech._ 121:443–463** —— 球谐波-free 多极级数
（spherical wave-free potential 满足 ∂ψ/∂z − Kψ = 0 恒等，投影到半球的奇次 Legendre 谐波；
激励力由原点 Kochin 幅值经 Haskind 关系给出）。本仓实现：
`src/tensorbem/analytic.py::hulme_hemisphere_radiation`（条款：n_terms / quad_order）。

**已入库算例（case-pool id 138，`BEM-HULME-HEMISPHERE`）的既得数字（回引，非本轮）**：
1024 常数面元、ka ∈ {0.2, 0.5, 0.8, 1, 1.5, 2}，带内最大相对误差 **1.1480e-02**（门 5%），
几何前置体积 **+0.587%**、水线 **+0.161%**（门 2%），参考 96→144 项漂移 **6.4638e-05**，
总判定 PASS。工件：`benchmarks/verified/casepool_bem_hulme_hemisphere/results.json`、
`report.md`、`figure.png`。

**缺口（本轮要打的目标）**：该算例把频带上限**刻意停在 ka = 2**，理由写在 `results.json` 的判据里——
避开该水线首个不规则频率 ka ≈ 2.58。于是「<5% 达标」在 ka > 2 属于 WITHHELD。本轮把研究台搭好，
把这个缺口变成一次可执行、判据先于数字的实验。

## ② 方法与实现

### 2.1 求解侧（既有）
`tensorbem/free_surface_solver3d.py::assemble_free_surface_operators_3d`（深水 Kelvin/Havelock
Green 函数常数面元 BIE，Van Oosterom–Strackee 立体角 + polar/Duffy 近奇积分），
内部 CHIEF 空场方程 `chief_points='auto'` 抑制不规则频率；`solve_radiation_3d`、
`solve_diffraction_3d`。网格 `geometry3d.hemisphere_hull_mesh`（subdivisions 4 → 1024 面元，
水线顶点严格落在 z = 0）。归一化 A/(ρ∇)、B/(ρω∇)，∇ = 2πa³/3 精确值；力按 ρgAa²。

### 2.2 本轮新增（研究台）
`benchmarks/verified/casepool_bem_hulme_hemisphere/band_extension.py`

- 单元列表 `cell_spec()`：扩展带 ka ∈ (2.2, **2.58**, 3.0, 3.5, 4.0) @ 1024 面元；
  网格对照臂 ka ∈ (3.0, 4.0) @ 256 面元；CHIEF off 对照臂 ka ∈ (2.58, 3.0)。
- 参考截断阶梯 `reference_ladder()`：96 / 144 / 192 项（在 ka = 2.58 与 ka = 4 两处）。
- 判据 `evaluate()`，**预注册、事后不放宽**：
  - **E1** 扩展带每一 ka、六个量 |rel err| < 5%（1024 面元）
  - **E2** 几何前置：|∇_mesh/∇ − 1| < 2% 且 |A_wp/(πa²) − 1| < 2%
  - **E3** ka = 3、4 处 1024 面元带内最大误差 ≤ 256 面元（加密不得变差）
  - **E4** ka = 4 参考 144→192 项漂移 × 20 < 5%（参考自身收敛）
  - **E5** ka = 2.58（首个不规则频率）CHIEF-on 单元有限且六个量全部 < 5%，CHIEF-off 作对照
- 并行策略：**每单元独立子进程**（`--cell`），不用 fork 进程池——见 §⑥ 的踩坑记录。
- 渲染器 `make_band_doc.py`：从 `results.json` 程序注入全部数字，产出本文的同构版本、
  以及 `oss_solutions` 载荷（列集与平台表一致）。

### 2.3 测试
`tests/test_hulme_band_extension.py`，**8 passed / 1 skipped**（skip = 工件尚未生成）：

- `test_predeclared_grid_covers_irregular_frequency_and_band_top`
- `test_cell_list_is_deterministic_and_shardable_by_construction`
- `test_reference_series_self_converged_at_band_top`（ka = 4 处 144→192 漂移 × 20 < 5%）
- `test_reference_values_are_finite_and_physical_at_band_top`
- `test_extension_mesh_meets_geometry_precondition`（1024 面元 ∇、A_wp 均 < 2%）
- `test_gate_passes_on_clean_cells` / `test_gate_fails_when_one_quantity_is_off_by_twenty_percent`
  / `test_gate_fails_when_refinement_worsens`（**判据必须能翻 FAIL**，反向注入 20% 误差与
  「加密变差」两种场景）
- `test_committed_extension_artifact_recomputes`（工件存在时逐单元复算 ref 与 rel err；
  本 checkout 尚无工件 → skipped）

命令：`PYTHONPATH=src python -m pytest tests/test_hulme_band_extension.py -q`

## ③ 参数表（研究台，预注册设计）

| 项目 | 取值 |
|---|---|
| 几何 | 浮半球 a = 1.0 m，平面与自由面齐平 |
| ∇ / A_wp | 2.094395 m³ / 3.141593 m² |
| ρ / g / A | 1025.0 kg/m³ / 9.80665 m/s² / 1.0 m |
| ka 扩展频带 | 2.2, 2.58（首个不规则频率）, 3.0, 3.5, 4.0 |
| 主网格 | subdivisions 4 = 1024 常数面元 |
| 对照网格 | subdivisions 3 = 256 常数面元（ka = 3.0, 4.0） |
| CHIEF 对照 | ka = 2.58, 3.0（on / off） |
| 参考截断阶梯 | 96 / 144 / 192 项（ka = 4.0 与 2.58） |
| 容差 | 每量、每 ka < 5% |
| 几何前置 | 体积、水线各 < 2% |
| 派生无量纲量 | λ/a = 2π/ka ∈ [1.57, 2.86]；ω√(a/g) = √ka ∈ [1.48, 2.0]；Froude / Reynolds 不适用（线性势流、无黏、无航速） |

## ④ 对照表

**本轮无扩展段数字。** 已入库段（ka ≤ 2）的逐项 ref vs got 见
`benchmarks/verified/casepool_bem_hulme_hemisphere/report.md` 与 `docs/bem_hulme_hemisphere_case.md`；
独立三方轴（capytaine 3.1.dev0，带内最差 3.36%）见 `docs/case_bem_hulme_independent_axis.md`。
本轮的扩展段判据 E1–E5 **全部 not_run**——按技能纪律，not_run 不等于 pass。

## ⑤ 收敛性与参数敏感性

同样 not_run（数值未产出）。可报告的**先验**证据只有两条已在测试中固定的性质：

1. ka = 4 处参考多极级数 144→192 项漂移的**量级**已由 `test_reference_series_self_converged_at_band_top`
   钉住（要求 ×20 < 5%，即 < 2.5e-3）；参考侧因此不太可能成为扩展带的瓶颈。
2. 1024 面元的几何前置（∇、A_wp < 2%）已由 `test_extension_mesh_meets_geometry_precondition` 钉住。

## ⑥ 失败模式、工件与 honest 记账

### 6.1 本轮失败段位

| 字段 | 值 |
|---|---|
| fail_stage | `study_run_incomplete`（扩展带 9 个单元未产出 `cells/*.json`） |
| failure_reason | ① 首版 `multiprocessing` fork 池在 BLAS 已热的多线程父进程下**死锁**（9 个子进程 8 分钟 CPU time 恒为 0），已改为子进程逐单元执行并保留该教训；② 本地控制面单条 shell 命令有 120 s 上限，1024 面元单元单跑即超时；③ 集群递交 `cj2013`（cores=16, timeout=3600）排队 `queued_ahead=139`，20 分钟未派发，为避免占用队列已 `kill` |
| 已跑出 | 参考级数在 ka = 2/2.58/3/4 的数值（96/144/192 项，漂移 ~1e-5 量级）；`test_hulme_band_extension.py` 8 项通过 |
| 未跑出 | 扩展带 1024/256 面元的 BEM 单元、CHIEF on/off 对照、E1–E5 裁定 |

### 6.2 已知方法风险（写入研究台的文件头，未事后修改）

- **面元 vs 波长**：ka = 4 时 λ/a = 1.57，1024 面元（面元尺度 ~0.1a）每个波长仅 ~15 个面元，
  误差预计显著大于 ka = 2 段；E3 就是它的对照臂，**不允许事后放宽 5%**。
- **不规则频率**：ka = 2.58 的算子条件数恶化只能靠 CHIEF 抑制；capytaine 用的是
  刚盖（lid）路线（`capytaine/bem/problems_checks.py:114 _check_wavelength_and_irregular_frequencies`
  建议 `lid_mesh`），本仓**不引入 lid**：lid 会改变物体拓扑，使 ∇/A_wp 前置校验失去意义。
- **共享内核**：求解器与参考解都调用 `free_surface3d.wave_function_3d`，带内缺口靠 capytaine
  第三方轴补；**ka > 2 段的第三方轴本轮同样缺失**，是本文明确声明的适用边界。

### 6.3 开源横向调研结论（任务步骤①）

`oss_inventory`（postgres `tensorplatform`，33 条）**全部落在 LBM 域**：
`SELECT repo,tier,focus FROM oss_inventory` 无一条 BEM / 势流 / 半球算例实现，
`methods` 列也无 BEM/Green 函数条目。因此「有现成 HEMISPHERE 实现直接学结构」在本环境**不存在**，
这一条按诚实结论记为 NOT_AVAILABLE，而不是「已横向对齐」。
唯一相邻的成熟 OSS 势流实现是**本地安装的 capytaine 3.1.dev0**
（`/usr/lib64/python3.11/site-packages/capytaine`）：Delhommeau Green 函数 + Hess-Smith 面元 +
刚盖抑频，本仓已把它作为独立第三方轴使用（`scripts/capytaine_hulme_excitation_ref.py`）。
节点 `~/oss-cache` 本机只有 3 个 LBM 仓（Maarten-vd-Sande_lbm / hschwane / pylbm），无 BEM 仓。

### 6.4 工件

- `benchmarks/verified/casepool_bem_hulme_hemisphere/band_extension.py`（研究台，预注册判据）
- `benchmarks/verified/casepool_bem_hulme_hemisphere/make_band_doc.py`（程序注入渲染器）
- `tests/test_hulme_band_extension.py`（8 passed / 1 skipped）
- `artifacts/hulme_band_extension/oss_solutions.json`（方法选型 + 迁移方案载荷）
- 缺失（本轮未生成）：`artifacts/hulme_band_extension/results.json`、`report.md`、`figure.png`、
  `cells/*.json`

```bash
# 扩展带数值（重跑即产出全部工件；在 128 核节点上 ~10-20 min）
PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/band_extension.py \
    --out artifacts/hulme_band_extension --parallel 9
# 单元测试
PYTHONPATH=src python -m pytest tests/test_hulme_band_extension.py -q
# 渲染文档与 oss_solutions 载荷（results.json 存在后）
python benchmarks/verified/casepool_bem_hulme_hemisphere/make_band_doc.py \
    --results artifacts/hulme_band_extension/results.json --out-doc docs/hulme_band_extension.md
```

### 6.5 下一轮接棒要点

1. 直接在集群节点上跑 §6.4 的复现命令（子进程模式已可并行，不需要 fork 池），
   或用 `stage_dir` 递交并把 `artifacts/` 取回。
2. 若 E1 在 ka = 4 翻 FAIL，结论应是「本路线在 ka > 2 的带内上限为 XX」——**不许**放宽判据；
   可以改为随 ka 提高面元数（subdivisions 5 = 4096 面元）作第二条收敛臂。
3. 若 E5 在 ka = 2.58 翻 FAIL，则可与 capytaine 的 lid 路线做方法对照，但须先证明 lid 不影响
   ∇/A_wp 前置。