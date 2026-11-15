# CASE-BEM-LAPLACE-CYLINDER 第 7 轮（斜来流方位扫描 + 旋转等变性）— TensorBEM m22326

**锚点** `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** `agent/m22326`
**账本状态**：开工核对 `feature_ledger` 该 slug = `open`（未被他单置 done/verified），故认领做实质增量。

## ① 为什么还要做第 7 轮

前 6 轮已把「BEM 外部速度场 vs 解析复势」打到 $10^{-4}$ 以下（最差判据 2.849e-04，见
`dev_report_m22173_..._round6.md`），并用路线 D/N/Γ/M、近壁窗、方位扫描、刻面归因把
**数值大小**上的欠账清完。但所有数字共享同一个几何前提：

> 来流沿 $+x$，而内接正 $N$ 边形在 $+x$ 轴上**恰有一个顶点**（`geometry.regular_polygon`
> 顶点角 $\theta_k = 2\pi k/N$，$k=0$ 即 $(a,0)$）。物形、边界数据、探针环
> （$\theta = (k+\tfrac12)\,2\pi/16$）三者对 $y=0$ 都严格镜像对称。

在这个前提下，任何**关于 $y$ 为奇**的实现缺陷（单/双层核函数符号翻转、法向定向在部分面板上取反、
面板编号/装配顺序假设、切向差分的单边偏置）都会在上、下两半之间相互抵消，L2 范数看不见。
这是 6 轮都没检验过的一类失效模式，而且它属于「已合并功能对基准复核」的验证欠账，与在飞单不重复
（在飞的是记录重生成/Others 的 capytaine 交叉验证，不涉及方位对称性）。

## ② 本轮做了什么（`src/tensorbem/benchmarks/case_cylinder_stream.py`）

新增**斜来流方位扫描**（additive，不动既有路线），参考值仍是闭式复势，只是换到旋转后的坐标系：

- `rotated_polygon(panels, heading, radius)`：把物体相对固定来流转 `heading` 弧度（等价于转来流，
  但让参考保持在既有坐标系里）。
- `heading_velocity(points, heading)`：$W(z)=Ue^{-ih}z+Ua^2e^{ih}/z$ 的速度，实现为
  $R(h)\,v_0(R(-h)p)$，因此 $h=0$ 时与 `exact_velocity` **逐位相同**（单测钉住）。
- `heading_perturbation_potential` / `heading_neumann_data`：$\phi'=Ua^2\cos(\theta-h)/r$ 及其径向导数。
- `_heading_level` / `heading_study`：每个方位上同时跑路线 D（Dirichlet→外部速度场）与
  路线 N（Neumann→物面速度），度量相对 L2、最大偏差、法向残差、条件数。
- `mirror_partner_indices` / `mirror_error_split`：把 0° 位形的速度误差按 $y=0$ 镜像拆成偶部/奇部。
- `render_heading_section`：从 `heading_study` 的字典单独渲染该节（不依赖整套慢算例，
  便于单测在秒级发现格式错误；这也是本轮踩坑后的加固，见 ⑤）。

判据（`criteria()`）新增 5 条，全部沿用 `<3%` 的算例容差：

| 判据 | 阈值 | 说明 |
| --- | --- | --- |
| `oblique_heading_dirichlet_velocity` | `< 3%` | 三个方位里路线 D 外部速度 L2 的最差值 |
| `oblique_heading_neumann_surface_velocity` | `< 3%` | 路线 N 物面速度 L2 的最差值 |
| `oblique_heading_within_penalty_of_axis` | `≤ 3×` | 斜方位相对 0° 的放大倍数（打掉对称不该掉精度） |
| `lattice_heading_is_covariant` | `≤ 1e-9 U` | 90°（$N/4$ 个面板步长、旋转矩阵精确）**连探针一起转**后与 0° 解逐位一致 |
| `mirror_antisymmetric_error_is_roundoff` | `≤ 1e-9 U` | 0° 位形误差的镜像奇部只剩舍入 |

方位集：$\{0^\circ,\ 180^\circ/N,\ 30^\circ\}$ —— 0° 为镜像对称基线；$180^\circ/N$ 是**半个面板步长**
（上游驻点流线正对弦中点，常值面板迹最不利）；$30^\circ$ 与任何用到的面板步长都不通约。

## ③ 计算结果（本地单进程实测，$a=1,U=1$，64 个外部探针）

路线 D 外部速度相对 L2（vs $dW/dz$）：

| $N$ | 0° | $180^\circ/N$ | 30° | 最大放大 |
| --- | --- | --- | --- | --- |
| 32 | 2.266e-04 | 2.175e-04 | 2.198e-04 | 1.00×（0° 即最差） |
| 64 | 1.229e-06 | 1.082e-06 | 1.120e-06 | 1.00× |
| 128 | 1.366e-07 | 1.366e-07 | 1.366e-07 | 1.00× |

路线 N 物面速度相对 L2（vs $2U|\sin\theta|$）：$N=32/64/128$ 分别为 1.633e-03 / 4.046e-04 / 1.008e-04，
**三个方位逐位相同**（差异 $\le 1.6\times10^{-16}$）；法向残差 $\max|v\cdot n|/U$ 同阶梯 9.65e-03 / 2.41e-03 / 6.03e-04。

结构量：

| 量 | $N=32$ | $N=64$ | $N=128$ |
| --- | --- | --- | --- |
| 格点 90° 旋转等变性 $\max|\Delta v|/U$（探针同转） | 5.98e-16 | 1.17e-15 | 1.04e-15 |
| 镜像奇部 $\max$（0°） | 3.24e-16 $U$ | 4.44e-16 $U$ | 6.00e-16 $U$ |
| 镜像偶部 $\max$（0°） | 6.712e-04 $U$ | 2.420e-06 $U$ | 2.290e-07 $U$ |
| 奇部占比 | 4.8e-13 | 1.8e-10 | 2.6e-09 |

## ④ 结论与解读

1. **打掉镜像对称不掉精度**：三个方位的路线 D 误差差异 $\le 4\%$（且方向是变小），路线 N 物面速度
   在 $10^{-16}$ 量级上相同。斜来流判据（`<3%`，实测 $\le 2.27\times10^{-4}$）现在把
   「不靠对称性吃饭」这层写进了算例门。
2. **全链旋转等变性成立**：90° 旋转（$N/4$ 个面板步长、旋转矩阵精确可表）把物体、边界数据、探针一起转，
   外部速度场与 0° 情形差 $\sim10^{-15}U$ —— 装配、外部 Green 求值、物面切向差分都不依赖面板编号，
   也不依赖来流与格点的相对相位。注意这条**不是恒等性**：物体自转格点角而来流不动是另一个位形，
   其速度场差 $O(U)$（本轮初版就是这样误写的，见 ⑤）。
3. **0° 的 L2 数字没有靠上下抵消**：镜像奇部只剩 $6\times10^{-16}U$，偶部承载全部 $2.29\times10^{-7}U$；
   也就是说在这个本可抵消的位形里，实现没有引入 $y$ 奇的非对称，历史结论未被推翻。
4. 反过来的诚实说明：本轮**没有**发现新缺陷。斜来流是一条「本该失败却没有失败」的负结果
   （falsification attempt），它把该失效模式的排除从「未检验」变成「已检验」。
5. 误差仍由**折面几何**主导（同 6 轮结论）：路线 N 物面速度误差随方位不变，正说明该项由面板局部
   刻面决定、随物体一起转，与来流相位无关。

## ⑤ 注意点 / 踩坑

- **等变性 ≠ 恒等性**：初版把「物体自转 90°、来流不动」后的速度场与 0° 直接相减，得 $\max|\Delta v|/U=1.23$
  —— 那不是 bug，是另一个物理位形。正确做法是把探针集合也转 90°，且速度是矢量
  （比 $v_{\text{lattice}}$ 与 $R\,v_0$）。改对后残差 1e-15。这条已写进判据注释，避免后人重犯。
- **渲染与慢算例解耦**：`render_markdown` 依赖整套算例（近壁/多极矩/环量，单进程数分钟），
  本轮插入新节时一度把 `lines += [` 的列表边界切错、且把 `heading` 局部变量抽成函数后忘加
  `lines: list[str] = [`，两次都是 `SyntaxError`/`UnboundLocalError`。修法是把该节抽成
  `render_heading_section(study)`，用 `heading_study(32)` 直接单测（秒级），不再依赖全量重生成。
- **参考侧符号约定**：`solve_laplace_exterior_neumann` 的 `normal_derivative` 约定与
  $\nabla\phi'\cdot n_{\text{out}}$ 差一个符号（算例历来如此，物面 $\max|v\cdot n|/U\le6\times10^{-4}$
  才是判据）。斜方位数据按同一约定构造，单测 `test_heading_neumann_data_is_the_rotated_axis_formula`
  用有限差分确认它确实是同一个法向导数（比例常数 $=\pm1$）。
- 适用范围同前：无界、无粘、不可压、定常、二维；不含粘性尾流、分离、自由面、3D 效应。

## ⑥ 误差判定（逐项 vs 解析复势）

| 关键量 | 值 | 基准 | 是否 <3% |
| --- | --- | --- | --- |
| 斜来流最差方位路线 D 外部速度 L2 | 2.266e-04（$N=32$）/ 1.366e-07（$N=128$） | $dW/dz$ 闭式 | **是** |
| 斜来流最差方位路线 N 物面速度 L2 | 1.633e-03（$N=32$）/ 1.008e-04（$N=128$） | $2U\|\sin\theta\|$ | **是** |
| 斜方位相对 0° 放大倍数 | 1.00×（两路线） | 0° 位形 | 是（阈值 3×） |
| 格点 90° 旋转等变性 | 1.04e-15 $U$ | 0° 解旋转后 | 是（阈值 1e-9 $U$） |
| 镜像奇部 | 6.00e-16 $U$ | 舍入 | 是（阈值 1e-9 $U$） |

## ⑦ 未完成 / 交下一单

- **提交记录仍未重生成**：`benchmarks/verified/case_bem_laplace_cylinder/results.json` 还是 38 条判据、
  无 `provenance` 的旧记录（第 6 轮遗留欠账），本轮改动使其继续落后 5 条新判据。
  已递交集群作业 **cj4044**（8 核、5400 s、`stage_dir` = 本工作树）执行：
  `PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check`
  至本轮收尾时该作业仍在排队（43 单在飞），工件未取回。取回后应：
  `run_compute fetch cj4044` → 覆盖 `results.json`/`report.md`/`case.json`/`figures/` →
  `pytest tests/test_case_cylinder_stream.py -k "record or document"` 让 provenance 门由 skip 转 pass。
- 下一单建议（slug + 量化验收 + 域）：`CASE-BEM-LAPLACE-CYLINDER-RECORD-REFRESH`
  —— 「把算例池记录重生成到与当前算例模块指纹一致（`provenance.generator.files` 匹配、
  `n_checks=48`、5 张图件落盘、`overall=PASS`）」，域：ship-total-performance/resistance-potential-flow。