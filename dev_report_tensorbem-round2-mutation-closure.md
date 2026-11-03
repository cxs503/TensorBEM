# TensorBEM 第二轮变异盲点回填：水线面一阶矩 + 两条退化分支 + 复活一条死导数路径

**任务**：m2856（自找活·TensorBEM 席，分支 `agent/m2856`）
**上游**：`docs/mutation_audit_tensorbem.md` §5 登记的 4 条存活变异体（m2582 明确写成「下一单待办」）
**定位**：验证欠账清账 + 一处死代码复活（不是新算法、不改数值方案）

---

## 1. 为什么是这四件事

m2582 的抽样变异审计给出 mutation score 0.650 → 0.825，并把 7 条仍存活的变异体逐条分类：3 条判为等价变异体（不为它们写测试），4 条判为真实/疑似覆盖缺口并登记待办。本单把这 4 条逐条关掉，并顺带关掉同类未抽样的一条。

| # | 变异体 | 类型 | 原判定 | 本单处置 |
|---|---|---|---|---|
| 1 | `hydrostatic:236:20:swap_binop` 水线面一阶矩 `/ 6.0 → * 6.0` | 覆盖缺口 | 现有算例水线面近对称，一阶矩≈0 | 新增**偏心**水线面算例（闭式解钉形心） |
| 2 | `hydrostatic:255:15:shift_constant` 净面积退化分支 `0.0 → 1.0` | 覆盖缺口 | 该分支从未被进入 | 新增「净零面积水线环」算例（bowtie） |
| 3 | `hydrostatic:314:29:shift_constant` 零排水量分支 `np.zeros(3) → (4)` | 覆盖缺口 | 同一类退化分支未覆盖 | 钉死返回契约（3 分量），并说明该分支对合法网格**不可达** |
| 4 | `free_surface:243:37:swap_binop` 深度导数 `2.0 * (…) → / (…)` | **死代码** | 只喂给从未被使用的局部量 | 改实现：`dg_d_source_z` 复用 `dg_d_depth`（数值恒等 1e-15），使其变成活路径并被有限差分钉住 |
| 5 | `hydrostatic:161:15:shift_constant` 「无水线面」早返回 `0.0 → 1.0` | 同类补漏（未抽样） | — | 全潜没闭合船体算例 |

---

## 2. 做了什么

### 2.1 实现改动（1 处，数值恒等）

`src/tensorbem/free_surface.py::green_gradient` 里，`df_d_depth`/`dg_d_depth` 算出来后**只被赋值、从未被读取**：竖直源导数把同一表达式在 `dg_d_source_z` 里重新展开了一遍。于是任何注入到深度导数上的缺陷都不可能被任何测试看见（这正是变异体 4 存活的原因）。

按两条路径的解析关系改写成复用：

$$\frac{\partial G}{\partial z_0} = -\frac{\partial G}{\partial A} - \frac{z - z_0}{2\pi\, r^2},\qquad A = -(z+z_0)$$

其中 $-\partial G/\partial A$ 就是 `dg_d_depth`，第二项是 Rankine 半径 $r=|P-S|$ 对 $z_0$ 的显式依赖（$r$ 不随 $A$ 变化，所以必须单独保留）。

- 数值恒等验证：40 组随机点（$\nu\in[0.2,2]$，$|z|,|z_0|\in[0.3,2]$）上新旧表达式最大相对差 **6.12e-15**（32 位浮点舍入量级）。
- 附带效果：m2582 回填的**旧**测试 `tests/test_mutation_gaps.py::test_deep_water_source_gradient_matches_a_finite_difference` 现在能独立杀死这个变异体（脚本里 `round1_alone` 字段）。

### 2.2 新增护栏（`tests/test_mutation_gaps_round2.py`，14 个用例，0.54 s）

| 用例 | 独立基准 | 关掉的变异体 |
|---|---|---|
| `test_eccentric_waterplane_centroid_is_pinned_by_closed_form`（2 参数） | 矩形水线面闭式解：$A=L_xL_y$、形心 $\left(\frac{x_0+x_1}{2},\frac{y_0+y_1}{2}\right)$、$I_{xx}=L_xL_y^3/12$、$I_{yy}=L_x^3L_y/12$、$I_{xy}=0$，相对/绝对容差 1e-12 | #1（形心被放大 36 倍） |
| `test_waterplane_is_empty_for_a_fully_submerged_hull` | 位于 $z=-2$ 的闭合球壳无边界半边 → 无水线面 | #5 |
| `test_cancelling_waterline_loop_reports_zero_area` | 等面积双叶 bowtie（shoelace 恰为 0） | #2 |
| `test_zero_volume_hull_reports_a_three_component_centre` | 返回契约：零排水量 → 零体积 + **3** 分量零矢量 | #3 |
| `test_source_vertical_gradient_matches_finite_differences`（4 点 × 2 辐射分支） | 对公开 `green_function` 的中心差分（步长 1e-6），含水平零偏移点 | #4 |
| `test_submerged_sphere_volume_and_buoyancy_centre` | 球：$V=\frac43\pi r^3$、浮心=球心；并检查二阶收敛（4 倍面元 → 误差 1/3.5 以下） | 反向锚（证明零体积分支是退化而非空白） |

### 2.3 复验工具（`docs/m2856/tools/verify_round2_mutants.py`）

复用 m2582 审计工具自己的 `enumerate_mutants` / `apply_mutant` / `run_tests` / 影子树隔离，**按变异体 key 精确选取**审计报告里的那几行，对每个变异体判两次：

- **before**：审计当时用的单元测试（`tests/test_hydrostatic_analytic_benchmark.py` / `tests/test_free_surface_kernel.py`）；
- **after**：本单新增护栏（kernel 站点额外单独再跑一次 m2582 的有限差分测试）。

只有「before 存活 且 after 被判失败」才算 closed —— 即同时复现了原文的盲点，也证明已关闭。

---

## 3. 计算结果（程序化，零手抄）

`PYTHONPATH=src python3 docs/m2856/tools/verify_round2_mutants.py` → `docs/m2856/outputs/round2_mutants.json`

| 变异体 | before（原套件） | after（新护栏） | 结论 | 触发失败的用例 |
|---|---|---|---|---|
| `hydrostatic:236:20:swap_binop` | survived | **killed** | CLOSED | `test_eccentric_waterplane_centroid_is_pinned_by_closed_form` |
| `hydrostatic:255:15:shift_constant` | survived | **killed** | CLOSED | `test_cancelling_waterline_loop_reports_zero_area` |
| `hydrostatic:314:29:shift_constant` | survived | **killed** | CLOSED | `test_zero_volume_hull_reports_a_three_component_centre` |
| `free_surface:243:37:swap_binop` | survived | **killed**（round-1 有限差分测试单独也 killed） | CLOSED | `test_source_vertical_gradient_matches_finite_differences` |
| `hydrostatic:161:15:shift_constant` | survived | **killed** | CLOSED | `test_waterplane_is_empty_for_a_fully_submerged_hull` |

**5/5 closed。** 其中 4 条是 m2582 登记在册的待办，1 条是同类的未抽样补漏。

### 图

- `docs/m2856/outputs/fig_round2_mutants.png`：逐变异体 before/after 判定（灰=原套件存活，蓝=新护栏杀死）。
- `docs/m2856/outputs/fig_round2_gradients.png`：左=竖直源导数解析值（线）与中心差分（点）沿水平偏移扫描；右=两者相对偏差曲线（$1.9\times10^{-11}$ ~ $3.2\times10^{-10}$，即差分截断误差本底）。

---

## 4. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 实测 | 相对偏差 | <3%？ |
|---|---|---|---|---|
| 偏心矩形水线面面积 / 形心 / 二矩 | 闭式解（$L_xL_y$, 形心中点, $L_xL_y^3/12$ …） | 与闭式解逐项比较 | ≤ 1e-12（facets 精确体） | **是** |
| 竖直源导数 $\partial G/\partial z_0$ | 公开 Green 函数中心差分（步长 1e-6），81 个偏移点 | `fig_round2_gradients` | 最大 3.21e-10 | **是**（受差分截断限制） |
| 重构后 $\partial G/\partial z_0$（改前 vs 改后） | 改前表达式逐点重算，40 随机点 | — | 6.12e-15 | **是**（恒等） |
| 潜没球排水体积 | $\frac43\pi r^3$ | icosphere 1280 / 5120 面元：-8.61e-3 / -2.16e-3 | 观测收敛阶 ≈ 2.0 | **是**（5120 面元 0.22%） |
| 全潜没船体水线面面积 | 0（几何事实） | 0.0 精确 | 0 | **是** |
| mutation 判定本身（5/5） | 无文献基准 | 布尔判定 | — | 无基准可比，不作精度声明 |

诚实结论：本单交付的是**测试判别力**（4 条登记盲点关闭 + 1 处死代码复活），不是求解器精度提升；`src/` 只有 1 处数值恒等重构（6e-15），未改任何离散方案。

---

## 5. 注意点 / 局限

1. **`hydrogen`: 零体积分支对合法网格不可达**：`WettedHullMesh` 强制每块板质心严格位于 $z<0$ 且排水体积 $>0$，并拒绝退化三角形，所以「有面元但净体积≈0」的网格无法通过公开构造器进入。因此 #3 的正确分类是**防御式分支**，本单用最小 duck-typed 船体钉住返回契约（形状必须是 3），而不是假装它是一个可物理到达的算例。
2. **#2 同理**：净零面积水线环意味着自相交的船体（bowtie），物理上不存在；该分支实际是在保护其后的 `x_moment / total_area` 除法。同样用 duck-typed 船体进入并钉住。
3. **$\partial G/\partial z_0$ 在 $P=S$ 处奇异**：$r^2=0$ 时该项 $0/0$。改前的表达式同样如此（不是本单引入），面板求积不会在配置点重合处直接调用它；本单的标杆点因此避开了完全重合点（保留水平零偏移但竖直错开的一点）。这是**未解决**的边界，登记为后续可选工作（若要在重合点定义，需要沿法向的极限分析）。
4. **只针对这几个站点，不做全量变异扫描**：本单是定向复验（按 key 选取），不是新的抽样估计；库级 mutation score 的变化应由下一次全量/抽样扫描给出。
5. **`tests/` 全量套件在本机负载下超过 shell 单次 120 s 上限**：本单以「定向子集 97 passed in 13.76 s」+ 后台全量作为佐证，全量结果以实际日志为准（见平台报告）。
6. **回归风险很低但不为零**：`src/` 的唯一改动是代数恒等重构（已在 40 个随机点上验到 6e-15），但它落在所有 2-D 自由面求解器的核心里；`tests/test_free_surface_kernel.py`、`tests/test_rao_verification.py`、`tests/test_mutation_gaps.py`、`tests/test_hydrostatic_*`、`tests/test_waterplane_multi_loop.py` 已全绿。

---

## 6. 复现

```bash
cd <worktree>
PYTHONPATH=src python3 -m pytest tests/test_mutation_gaps_round2.py -q          # 14 passed
PYTHONPATH=src python3 docs/m2856/tools/verify_round2_mutants.py                # 5/5 closed
PYTHONPATH=src python3 docs/m2856/tools/make_figures.py                         # 两张图 + 数值
```