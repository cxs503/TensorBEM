# m14931 · TensorBEM `tensorbem.fmm` 代价/路由独立审计 + 面元几何可容许性缺陷修复

分支 `agent/m14931`（基线 `f669975`）。EDIT_WHITELIST:
`src/tensorbem/fmm.py`, `src/tensorbem/benchmarks/fmm_cost_audit.py`,
`tests/test_fmm.py`, `tests/test_fmm_cost_audit.py`, `docs/fmm_cost_audit/`,
`docs/fmm_acceleration/report.md`, `dev_report_m14931_fmm_cost_audit.md`。

【锚点=NEW:一句话理由】功能点总账内无 BEM 的 FMM 条目（`feature_ledger` 仅
`FEAT-OSS-INVENTORY-BEM`，且为盘点非改进）。建议新条目：
`slug=GAP-BEM-FMM-COST`，title「FMM 加速算子的代价/可容许性验收面」，
所属域 TensorBEM；量化验收：①面元几何 matvec 相对稠密误差 p=4 ≤1e-3（现 1.6e-4）；
②per-interaction 代价比 ≤1（现 ~1e2-1e3，未达）；③复杂度斜率在 N≥1e4 实测而非外推。

## 1. 案例介绍

`tensorbem.fmm` 在能力台账中被判 `REAL`（"Fast multipole acceleration of the
3-D single-layer operator"），其战役报告 `docs/fmm_acceleration/report.md` §3
把 8 条验收判据全标 PASS，其中 (c) 复杂度「N∈[1e3,1e5] 斜率∈[0.9,1.2]」、
(d) 物理不变「球面势 FMM vs 直接 <1e-10、附加质量相对差 <1e-6」、
(e) 内存「N=1e5 峰值 <5% 稠密」。这三条都是**代价/路由**断言，而不是数值精度
断言——审它们不需要新物理，只需要在算子真实使用面（闭合曲面面元质心，即 3-D 中
的二维源分布）上实测。算例：icosphere subdivisions=1/2/3 → 80/320/1280 面板，
单位球，源强度=面板面积；分离构型 = 源填半径 a=0.1 球、靶点在 8a 球面。

## 2. 怎么做的

新增 `src/tensorbem/benchmarks/fmm_cost_audit.py`：一条命令产出 JSON + Markdown +
图（零手抄：文档由同一进程从 JSON 渲染）。它测三件事：(i) 路由计数（近场直接求和
vs 远场多极对，按算子真实判据复算）；(ii) 精度阶梯——分离构型 p=1..8、曲面构型
p=2,4（靶点取网格顶点，源取质心，避开重合奇点）；(iii) 代价——同一 N、同一靶集下
FMM matvec 与 numpy 分块稠密 matvec 的墙钟，分解出 c_fmm（每次靶-叶交互）与
c_dense（每源-靶对），比值 c_fmm/(avg_leaf·c_dense) = 单位工作量代价比。
`tests/test_fmm_cost_audit.py` 把上述事实钉成 8 条门。

## 3. 计算结果（`docs/fmm_cost_audit/audit.json`）

### 3.1 缺陷与修复（曲面几何，p=2 / p=4，靶=顶点，源=质心）

| 版本 | p=2 最大相对误差 | p=4 最大相对误差 |
|---|---:|---:|
| 修复前 | 6.882e-01 | 3.917e+00 |
| 修复后 | 1.130e-03 | 1.601e-04 |

修复：`FMMOperator.matvec` 的远场判据由"叶心距 ≤ 2·max(size)"改为"叶心距 ≤ 2·max(size)
**且** 靶点到源叶中心的距离 > 2×该叶自身源半径"（`_ADMISSIBILITY=2.0`，与单层
`fmm_single_level_matvec` 的 `r_t > 2·max_r_s` 判据一致）。物理含义：多极展开关于
叶心收敛，靶点落在该叶源球内时级数发散；曲面叶是薄板，size 远大于叶心间距，旧判据
把这类对判成远场，于是 p 越高误差越大（0.688 → 3.917）。

### 3.2 代价（同一 N、同一靶集）

| 面板 N | p | 叶数 | t_FMM (s) | t_dense (s) | FMM/dense | 单位工作量代价比 |
|---:|---:|---:|---:|---:|---:|---:|
| 320 | 2 | 36 | 1.680 | 0.0051 | 328x | 328x |
| 1280 | 2 | 60 | 8.138 | 0.0832 | 98x | 98x |
| 320 | 4 | 36 | 4.849 | 0.0042 | 1147x | 1147x |

### 3.3 路由与既有测试的空转

`max_leaf = panel_count`（既有 (d) 判据的配置）→ **1 叶、320 近场对、0 多极对**，
记录的 1e-12 一致度是直接求和与直接求和的比较，且重合质心的 r 在两侧都被夹到
1e-15：该测试钉的是退化分支，不是多极展开。曲面构型下多极对占比 35%（4078/11520）。

### 3.4 分离构型精度阶梯（多极路径确实被使用）

p=1..8 误差 2.453e-2 / 2.893e-3 / 7.274e-4 / 1.932e-4 / 1.665e-5 / 1.036e-6，
单调收敛，斜率与 (ρ/r)^p 上界一致。

## 4. 图

`docs/fmm_cost_audit/fig_fmm_cost.png`（左：FMM vs 稠密 matvec 墙钟 log-log，
p=2；右：分离/曲面两条精度-阶曲线）。

## 5. 注意点

- 纯 Python 实现，p=6/e2 级配置单次 matvec 数十秒，N≥5120 未实测（外推已在 §3.2 标注）。
- 叶心最近邻路由（`_target_leaf`）是近似路由，靶点落在叶边界上时可能选错叶；
  本审计只在 icosphere 质心/顶点上验证，未量化该近似的影响。
- 本单**不改求解器接线**：FMM 仍未接入 `solver3d`/`free_surface_solver3d`。
- `tests/test_fmm.py::test_theoretical_slope` 的判据随修复改为单侧界
  （误差下降不快于 (ρ_max/r)^p 且 p=11 误差 <1e-5），原 ±60% 双侧判据是按
  缺陷态标定的。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---:|---|
| 曲面 matvec（p=4，修复后） | 分块稠密单层求和 | 1.60e-4 | ✅ |
| 曲面 matvec（p=2，修复后） | 同上 | 1.13e-3 | ✅ |
| 分离构型 matvec（p=8） | 同上 | 1.04e-6 | ✅ |
| 曲面 matvec（p=4，修复前） | 同上 | 3.92e+0 | ❌（缺陷，已修） |

**加速比不达标**：所有实测 N 上 FMM 比稠密慢 98–1147 倍，判 NOT_SUPPORTED（如实）。
