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