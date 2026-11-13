# 算例池 `CASE-BEM-ANNULUS-LAPLACE`（第 1 轮）：环形域 Laplace（多连通）

**仓**：TensorBEM · **任务**：m17178 · **分支**：`agent/m17178` · **证据等级**：REAL
（闭式调和解 vs BEM 数值解，逐项相对误差；所有数字由
`src/tensorbem/benchmarks/case_annulus_laplace.py` 生成，报告与表格零手抄）

**本文件是索引与判断记录**；数字唯一的来源是机器产物
`benchmarks/verified/bem_annulus_laplace/results.json`，由它渲染的完整六段报告是
`benchmarks/verified/bem_annulus_laplace/report.md`（同目录 `case.json` 为算例卡，
`figures/` 为四张图）。

---

## 1. 问题与 ref 出处

环形域 `b < r < a`（内半径 `b`、外半径 `a`）上的二维 Laplace 方程 Dirichlet 问题。
边界由两条互不相连的闭曲线组成，**内环的"域外法向"指向孔洞内部**——正是单环面元
数据结构装不下的拓扑，也是 `docs/verification.md` §1 长期挂"需多边界组件"的那一格。

闭式参考解（两个算例，都是环内调和函数）：

| 算例 | 闭式解 | 性质 |
|---|---|---|
| `radial_log` | `u = A log r + B` | 源点在原点 = 孔洞内 = 流体域外；两个环的边界数据不同 |
| `angular_cos` | `u = (r + c²/r) cos θ` | 平移圆柱势；`du/dr` 沿内环不是常数（比径向算例更能压测面元积分） |

**ref 出处**：这两个解是经典势流的基本解/平移圆柱势，书目为
`Lamb H. Hydrodynamics, 6th ed., CUP, 1932`（与 `docs/classical_benchmarks.md` 参考文献 [1] 同一版本）与
`Milne-Thomson L M. Theoretical Hydrodynamics, 5th ed., Macmillan, 1968`（同文件参考文献 [3]）。
**诚实的缺口**：容器内两本书均不可达，本单**未核到具体页码/条目号**，故
`reference_provenance.page_or_article_number` 登记为 `WITHHELD`——不编页码。
补偿机制是**把引用换成测量**：`harmonicity_check` 在环内随机点用中心差分算
`Laplace(u)`，相对残差当作 ref 合法性的机器门（结果见 report.md §1，判据 `< 1e-6`）。
参考数值本身由 `reference_table` 从公式程序生成（48 个探针/算例），零手抄。

## 2. 方法与实现（复用，不另起炉灶）

| 环节 | 用的东西 | 是否新写 |
|---|---|---|
| 多环网格 + 孔洞法向约定 | `tensorbem.multiloop.MultiLoopMesh` / `annulus_mesh` | 否 |
| Dirichlet BEM（面元解析积分、`Sq=(1/2 I+H)g`） | `tensorbem.multiloop.solve_laplace_multiloop` | 否 |
| 内部场 Green 表示 | `tensorbem.multiloop.evaluate_multiloop_field` | **补了域守卫**（见 §6.2） |
| 闭式场函数 | `tensorbem.benchmarks.annulus_laplace.CASES` | 否（import 复用） |
| 算例卡/阈值/收敛/敏感性/渲染 | `tensorbem.benchmarks.case_annulus_laplace` | 是（本单新增） |

阈值**先声明后计算**（写在模块常量里，再落进 `results.json` 的 `acceptance`，
运行后不再改）：边界 `q` 的相对 L2 / Linf 误差、内部探针 `u` 的相对最大误差、
逐环通量相对误差，四项均 `< 3 %`；拟合收敛阶 `> 1.5`；另加 ref 调和性 `< 1e-6`
与"广告预算"（64/32 面元）必须达标两条。

## 3. 参数表

见 `report.md` §3（`a = 1.0`、`b = 0.5`、`b/a = 0.5`、`radial_log: A = 0.7, B = 1.3`、
`angular_cos: c = 0.4`、核参考长度 `L = 2.0`、探针 `r = b + f(a−b)`，`f ∈ {0.2, 0.5, 0.8}`、
每半径 16 个方位角）。无量纲量：`b/a`、`c/a`、`a/L`、`r/a`——全部在表里列出。

## 4. 对照表

`report.md` §4：每个加密层级的 `cond(S)`、`q` 相对 L2/Linf 误差、内部场相对最大误差、
逐环通量误差、相对残差，逐项对照闭式解；`report.md` 结尾的 verdict 表对每个检查项
给出 `PASS/FAIL` 与阈值。总体结论以 `results.json` 的 `verdict.passed` 为准。

## 5. 收敛性与参数敏感性

* 加密梯队 `16/8 → 256/128`（外/内环面元），两条误差曲线（边界 `q`、内部场）与
  `N^-2` 参考线对照，见 `figures/convergence.png`。
* **敏感性 1**：壁厚比 `b/a ∈ {0.2 … 0.95}`（固定 64/32 面元）——误差与 `cond(S)`
  随环变薄的走势，见 `figures/sensitivity_wall_ratio.png`。
* **敏感性 2**：核参考长度 `L ∈ {1.0, 1.5, 2.0, 4.0, 8.0}`——这是**对数容量退化**
  （自由空间核 `G = -(1/2π) log(r/L)` 在 `a → L` 时退化）的量化，见
  `figures/sensitivity_reference_length.png`。它给出的是"包默认 `L = 2` 相对 `a = 1`
  有多安全"的边界，而不是新算法。

## 6. 失败模式与工件

1. **内环法向未翻转**：线性系统仍解到残差量级不变，但 `q` 与通量错到量级级别——
   残差型验收抓不到，必须由网格类型约束（`MultiLoopMesh.hole_loops`）兜住。
2. **孔洞内探针**：本次发现 `evaluate_multiloop_field` **原先没有域守卫**，
   孔洞内/外环外的目标点会照算 Green 表示并返回"看起来合理"的解析延拓值。
   本单补了 `_point_in_polygon` + `domain_membership` 守卫并加了测试
   （`tensorbem.multicomponent.evaluate_fluid_field` 早就有同类守卫，多环路径漏了）。
3. **`a → L` 退化**：`L = 1.0`（即 `a/L = 1`）时条件数暴涨、误差越过 3 %。

工件：`benchmarks/verified/bem_annulus_laplace/{results.json, case.json, report.md, figures/}`。

复现：

```
PYTHONPATH=src python -m tensorbem.benchmarks.case_annulus_laplace \
    --outdir benchmarks/verified/bem_annulus_laplace --check
PYTHONPATH=src python -m pytest tests/test_case_annulus_laplace.py tests/test_multiloop_annulus.py -q
```

## 7. 生态摩擦（如实记）

* `GET /api/v1/tensor/knowledge/search` → `engine_offline`（`127.0.0.1:8021` 连接被拒）；
  知识库三查的第一查（capability）与第三查（oss_solutions）**无法执行**。
* `GET /api/v1/knowledge/search`（sediment store）对 `TensorBEM` / `annulus` 均返回 0 条。
* `GET /api/v1/tensor/negative-knowledge/search?q=TensorBEM` → 0 命中；
  `?q=annulus` → 0 命中（负知识账本对本域为空）。
* 平台 `/openapi.json` 中**没有** oss_solutions 路由（只有 `/datahub/glossary` 等）。
* 三查的替代：仓内自查（`grep -ril annulus src/ tests/ docs/`）发现既有能力
  `tensorbem.multiloop` + `benchmarks/annulus_laplace`（m3701/m3606），故本单复用而非新建。
---

# 第 2 轮（m17708）：偏心环形域 —— 双极坐标闭式解

**分支** `agent/m17708` · **证据等级** REAL · 机器产物
`benchmarks/verified/bem_annulus_laplace_eccentric/{results.json, report.md, figures/}`，
数字全部由 `python -m tensorbem.benchmarks.eccentric_annulus` 生成（`report.md` 由 JSON 渲染，零手抄）。

## 为什么是这一轮

第 1 轮的同心双环是**特别温和**的配置：两条边界是平行曲线，间隙处处相等，每个面元到对岸的距离都一样。
这一轮把这条对称性拆掉：两条边界取**双极坐标族**的两条圆（同轴、圆心不重合），间隙沿周向从
`a−(e+b)` 变到 `a+e−b`；同时保持「所有奇点都在流体之外」这条前提（`z+` 在孔内、`z-` 在外环外），
所以闭式解仍然精确，而面元积分的近奇异结构比同心情形苛刻得多。

## 新增能力（库级，可复用）

| 位置 | 内容 |
|---|---|
| `tensorbem.multiloop.eccentric_annulus_geometry` | 双极族圆参数 → 半径/圆心/偏距/最小最大间隙/相对偏心；并给出两个焦点的包含性判据 |
| `tensorbem.multiloop.eccentric_annulus_mesh` | 偏心双环网格（内环自动标 hole），**新增内接多边形互不重叠守卫**：圆间隙为正不等于网格可表示，过薄的环会拒绝装配而不是静默返回乱数 |
| `tensorbem.benchmarks.eccentric_annulus` | 三个闭式算例 `point_source` / `bipolar_ratio` / `hole_dipole`、合法性门（常数性 + 调和性）、收敛梯队、间隙扫描、参考长度扫描、两条负控、verdict、四图渲染 |

## 闭式解的合法性怎么保证（引文换成测量）

`bipolar_ratio` 取 `u = log(ρ+/ρ−)`，它在双极族的每条圆上是常数，理论常数 `−τ`。
容器内查不到 Lamb / Milne-Thomson 的条目号（`page_or_article_number = WITHHELD`，不编页码），
所以把引文换成两把机器门：

* 沿**精确圆周**采样 512 点，`u` 的相对极差 `1.665e-15`（外） / `6.168e-16`（内），
  均值与 `−τ` 相对偏差 `<1e-15` —— 「两条环上各是常数、且常数不同」是测出来的；
* 环内随机 64 点中心差分 `Δu`，相对残差 `6.05e-15`（三个算例都 `< 5e-14`）。

顺带一个真实插曲：第一版门把理论常数写成 `−2τ`，被自己的门判 FAIL（相对偏差 0.5）——
`τ = ½log(ρ+/ρ−)`，常数就是 `−τ`。门抓到了作者的因子 2 错误，这正是它的用处。

## 结果（预算 64/32 面元）

| 算例 | q 误差 L2 | q 误差 Linf | 内部场误差 max | 通量误差 | 拟合阶 q | 拟合阶场 |
|---|---:|---:|---:|---:|---:|---:|
| `point_source` | 2.34e-03 | 2.85e-03 | 7.46e-04 | 1.27e-03 | 1.96 | 2.10 |
| `bipolar_ratio` | 2.28e-03 | 2.12e-03 | 6.24e-04 | 1.30e-03 | 1.96 | 2.18 |
| `hole_dipole` | 6.27e-03 | 6.30e-03 | 4.39e-04 | 2.27e-05 | 2.00 | 2.86 |

逐环通量对解析值 `±2π`（偶极子为 0）：`point_source` 外 6.2899 vs 6.2832、内 −6.2876 vs −6.2832；
两环通量和 / ∫|q|ds `3.2e-04`。`verdict.passed = true`，37 条判据全过。

## 敏感性 1：间隙（预算固定，扫描内环族参数 τ_b）

| τ_b | 最小间隙 | 间隙/面元 | q L2 `point_source` | q L2 `bipolar_ratio` |
|---:|---:|---:|---:|---:|
| 0.805 | 0.0021 | — | 拒绝（探针协议：16 条射线带宽全部 < 2%a） | 拒绝 |
| 0.820 | 0.0085 | 0.040 | 2.59e-01 | 4.24e-02 |
| 0.850 | 0.0212 | 0.103 | 8.96e-02 | 2.47e-02 |
| 0.900 | 0.0420 | 0.220 | 2.15e-02 | 1.10e-02 |
| 0.950 | 0.0623 | 0.349 | 9.00e-03 | 6.20e-03 |
| 1.000 | 0.0822 | 0.493 | 4.00e-03 | 4.10e-03 |
| 1.050 | 0.1016 | 0.650 | 2.10e-03 | 3.00e-03 |
| 1.200 | 0.1571 | 1.210 | 1.50e-03 | 1.80e-03 |
| 1.400 | 0.2244 | 2.031 | 1.80e-03 | 1.70e-03 |
| 1.800 | 0.3363 | 3.044 | 2.30e-03 | 2.30e-03 |
| 2.200 | 0.4206 | 3.806 | 2.60e-03 | 2.60e-03 |

**这是本轮最有用的数字**：预算 64/32 在「间隙 ≥ 约 0.22 个面元长」时稳定达标（≤2.2e-2），
间隙掉到 0.10 / 0.04 个面元长时误差升到 2.5e-2 / 4.2e-2，再薄则探针协议直接拒绝。
误差曲线是 U 形而不是单调：薄端由**两环近相切**主导，厚端由**源点离内壁变近**主导
（τ_b 大时内半径小、`z+` 贴壁更近），两个来源在 τ_b≈1.2–1.4 处相交。

## 敏感性 2：核参考长度 `L`

`L ∈ {0.5, 1, 2, 4, 8}` 全部达标（q L2 2.3e-3 ~ 2.4e-3），条件数 55.9 → 247.0 单调上升；
偏心几何下 `a/L` 不再是唯一天敌，但 `L` 变大仍稳定抬高 `cond(S)` 而不动精度。

## 失效模式（实测）

| 算例 | 正确网格通量误差 | 未翻内环法向 | 放大倍数 | 未翻相对残差 | 是否被抓 |
|---|---:|---:|---:|---:|:--:|
| `point_source` | 1.27e-03 | 7.11e-01 | 562× | 4.8e-16 | 是 |
| `bipolar_ratio` | 1.30e-03 | 9.04e-01 | 697× | 3.7e-16 | 是 |
| `hole_dipole` | 2.27e-05 | 2.66e-01 | 11734× | 3.8e-16 | 是 |

与第 1 轮同一结论、更强的量化：**残差型验收对内环法向完全瞎**（残差 1e-16），
必须靠网格类型约束 + 通量判据。孔内 / 外环外探针被 `evaluate_multiloop_field` 双双拒绝。

## 命令

```
PYTHONPATH=src python -m tensorbem.benchmarks.eccentric_annulus \
    --outdir benchmarks/verified/bem_annulus_laplace_eccentric --check
PYTHONPATH=src python -m pytest tests/test_case_annulus_laplace_eccentric.py -q
```
