# `tensorbem.analytic` 变异审计：把「真值来源」本身纳入判别力检验

EDIT_WHITELIST: src/tensorbem/benchmarks/mutation_audit.py, tests/test_mutation_gaps_analytic.py, docs/mutation_analytic/, dev_report_tensorbem_analytic_mutation.md

## 1. 案例介绍：为什么审这根「尺子」

`src/tensorbem/analytic.py`（1442 行）是本仓全部基准的**真值来源**：椭圆/椭球附加质量、内部固体谐波、Ursell 潜没圆柱、Dean 绕射、MacCamy–Fuchs、Struve/Bessel、Hulme 漂浮半球、Wang/Linton 潜没球多极子。其它模块的测试都是「求解器 vs analytic」；一旦 analytic 里的因子写错，**所有基准的参照物会一起平移**，而每一对求解器/参照的比对仍然"自洽地"通过。

2026-11-03 的变异审计（`docs/mutation_audit_tensorbem.md`）审了 5 个数值内核模块（kernels / free_surface / hydrostatic / solver / finite_depth），抽 40 个变异体，前测 score 0.650，回填后 0.825。但它把 `tensorbem.analytic` **明确排除**，理由是其唯一单测文件 `tests/test_analytic_references.py` 单跑 >100 s，"未审计 ≠ 已通过"（原文 §6.2）。

2026-11-04 在 load15≈90 的窗口复测该文件：**29.75 s / 20 passed**，完全落在变异扫描的 300 s 单次超时内。于是本单把它补进审计面。

**被测对象**：`tensorbem.analytic`（1442 行，枚举出 **1347** 个单点变异体）
**映射单测**：`tests/test_analytic_references.py`（20 例，29.75 s）
**抽样**：`seed=20261104`，随机抽 24 个（`mutation_audit.sweep` 同分布同种子可复现）

## 2. 怎么做的

1. **扩审计面**：`TARGETS` 增加 `tensorbem.analytic`，并在代码注释里写明"2026-11-03 因 >100 s 排除，2026-11-04 复测 30 s 后纳入"，让这条历史判断有据可查。
2. **修一个审计引擎缺陷（关键）**：一级判定用的映射单测只覆盖 analytic 的一半函数；对存活者，引擎进入 stage-2「家族检索」——按**函数名**在 `tests/` 里 grep。但测试永远不会写出私有辅助函数名，`submerged_sphere_wave_multipole._solve_azimuthal` 的家族集合因此**为空**，stage 2 变成空操作，任何埋在私有辅助函数里的缺陷都只能被判"存活"。新增 `_family_names()`：私有名继承其外层公开函数名（`x._helper` → `x`）。修完后同一批变异体里 2 个由 `tests/test_submerged_sphere_wave_3d.py` 杀死——它们此前被误报为盲点。
3. **分块可复现扫描**：本机外壳对进程组 120 s 硬收割，故写了 `docs/mutation_analytic/sweep_analytic.py`（复用引擎自己的 `enumerate_mutants / _shadow_package / run_tests / _classify`，逐条落盘、断点续跑），同 seed 重跑得到同一 24 个样本。
4. **判读与回填**：被杀的记录证据；存活的先判"是否等价"（数学上与被变异前不可区分），再对真缺口写**独立判据**的护栏（README 里禁止把实现公式抄回测试）。
5. **再判定**：只重跑未杀死的 5 个（加测试只可能多杀，已杀的行无需重跑），逐条落 `rejudge_detail.json`（含 rc、失败测试名）。

## 3. 计算结果

### 3.1 前测（仅映射单测 + 家族检索）

| 模块 | 样本 | 杀死 | **超时（无断言）** | 存活 | score |
|---|---:|---:|---:|---:|---:|
| `tensorbem.analytic` | 24 | 19 | 2 | 3 | **0.792** |

无效变异体 0（24 个全部是合法可运行程序，判定不靠"跑不起来"充数）。

### 3.2 五个未杀死变异体逐条判读

| 变异体（注入缺陷） | 前测判定 | 独立复核（`rejudge_detail.json`） | 结论 |
|---|---|---|---|
| `analytic:1025` `maccamy_fuchs_diffraction`: `kh < 50.0` → `100.0` | 存活 | `test_helmholtz_maccamy_fuchs.py` 仍全绿 | **等价**：`math.tanh(50)=1.0` 在双精度下精确成立，50→100 不改变任何可观测输出 |
| `analytic:599` `_ursell_multipole_system`: `r * log_x` → `r / log_x` | 存活 | `test_ursell_benchmark.py::test_deep_submergence_recovers_the_displaced_mass` **失败** | **引擎缺陷误报**（模块级私有函数无公开父名，家族集合为空）；已在 §4 记为已知局限 |
| `analytic:351` `ellipsoid_shape_factors`: `nodes < 16` → `< 17` | 超时 | 新护栏 `test_ellipsoid_shape_factors_accept_the_documented_minimum_node_count` **失败** | 真实缺口（输入契约无人钉） |
| `analytic:670` `ursell_cylinder_added_mass`: 默认 `density=1025.0` → `-1025.0` | 超时 | 新护栏 `test_ursell_cylinder_default_density_is_seawater` **失败** | 真实缺口（默认参数无人钉） |
| `analytic:1414` `submerged_sphere_wave_multipole`: `b33_ff_phys / (rho*omega*volume)` → `*` | 存活 | 新护栏 `test_submerged_sphere_far_field_damping_is_consistent_with_haskind` **失败** | 真实缺口（无量纲远场阻尼列无人钉） |

注：两个"超时"不是测试挂死，而是 stage-2 家族（Ursell 全套约 70 s+）超出当时设定的 70 s 单次预算；改为定向运行后两者都在数秒内被新护栏判负。**这条也暴露了一个方法学问题：把"超时"当作"未检出"会让 score 偏低**。

### 3.3 后测（回填 5 条护栏后）

| 判定 | 变异体数 |
|---|---:|
| 断言杀死 | 23 |
| 等价（数学上不可区分） | 1 |
| 存活（真盲点） | **0** |
| score（等价计入） | 0.958 |
| **score（等价剔除，即判别力下界）** | **1.000** |

![analytic mutation score](fig_analytic_mutation_score.png)

图 1 左：24 个注入缺陷按判定分组的回填前后对比（灰=回填前，蓝=回填后）。右：变异分数三种口径——回填前 0.792；回填后把等价变异体计入 0.958；剔除等价变异体后 1.000（该口径是判别力的下界估计，不是"全覆盖"的证明）。

## 4. 回填的 5 条护栏（每条对应一个被注入的缺陷）

| 护栏 | 独立判据（不是抄实现） |
|---|---|
| `test_submerged_sphere_far_field_damping_is_consistent_with_haskind` | **Haskind 恒等式**：同一物理量的辐射解列与 Kochin 远场列必须相等（实测 1.0e-16 级一致，阈值 1e-8）；模块自己只对"物理量"列算 `haskind_error_*`，无量纲列（正是文献表引用的那两列）此前从未比对 |
| `test_submerged_sphere_physical_damping_follows_the_dimensionless_column` | **量纲归一化方向**：`*_damping/(omega*V* *_damping_dimless)` 对垂荡/纵荡必须是同一个正数，且等于海水密度 1025（1e-12） |
| `test_ursell_cylinder_default_density_is_seawater` | **默认参数契约 + 线性**：不传 `density` 与显式 `1025.0` 必须一致（1e-14），且 `density` 翻倍输出翻倍、输出为正 |
| `test_ellipsoid_shape_factors_accept_the_documented_minimum_node_count` | **输入契约**：`nodes=15` 必须报"must be an integer >= 16"；`nodes=16` 允许报"求积未收敛"，但不得被当作非法输入拒绝；默认路径复现 α 和 = 2（1e-9） |
| `test_private_helper_family_includes_the_public_parent` | **审计引擎自身**：`_family_names("x._helper") == ("x",)`、模块级私有名解析为空（已知局限） |

## 5. 注意点与局限（不粉饰）

1. **抽样不是穷举**：analytic.py 单模块可枚举 1347 个单点变异体，本单按种子抽 24 个（约 1.8%）。0.958 / 1.000 是抽样估计，**不能**外推为"analytic 无缺陷"。
2. **模块级私有函数的家族检索仍为空**：`_ursell_multipole_system` 这类"无公开父名"的辅助函数，stage-2 依旧空转（本单靠定向复核才发现它其实被 Ursell 基准杀死）。**已知局限，未修**；修法是补一张 AST 调用图（谁调用了它），留待后续。
3. **超时语义**：引擎把 `rc=124` 记为 `timeout` 且不计入 killed，会让 score 系统性偏低；本单改用定向小测试集复核，但引擎口径未改（属行为变更，需单独立项 + 全仓回归）。
4. **变异算子只覆盖 4 类单点缺陷**（比较符翻转、算术互换、常量位移/取负），不模拟整句删除、参数顺序、算法替换；存活率是判别力**下界**。
5. **未改任何数值实现**：`analytic.py` 一行未动，改动仅为审计引擎的家族解析 + 新测试文件 + 文档。

## 6. 误差判定（本单无物理基准，按方法学判据给）

| 关键量 | 基准 | 实测 / 判据 | 是否达标 |
|---|---|---|---|
| 变异分数（回填前） | 100%（理想） | 0.792（19/24） | 否（记录在案） |
| 变异分数（回填后，等价计入） | 100% | 0.958（23/24） | — |
| 变异分数（回填后，等价剔除） | 100% | **1.000（23/23）** | ✅ |
| 未判读的非杀死变异体 | 0 | 0（5 个逐一判读：4 杀 1 等价） | ✅ |
| 新护栏的判别裕度（Haskind 列） | 1e-8 阈值 | 正常状态 1.0e-16 级一致，变异后偏离数个量级 | ✅ |
| 说明 | 无实验/文献数值参与，故不报"相对解析解误差"；本单的"真值"是**变异体是否被现有护栏捕获**这一可复现判定 | | |

结论：本次审计的**关键量是判别力（mutation score），回填后 23/23 非等价变异体全部被杀死，等价变异体 1 个**；对"<3% 误差"这类物理判据本单不适用，理由如上（无物理基准参与）。

## 7. 复现

```bash
# 前测（分块可续跑；本机外壳 120 s 收割进程组）
BUDGET=58 MUT_TIMEOUT=70 N=24 SEED=20261104 \
  python3 docs/mutation_analytic/sweep_analytic.py docs/mutation_analytic/before/results.json
# 定向复核未杀死变异体
python3 docs/mutation_analytic/sweep_analytic.py ...   # 或见 rejudge 脚本
# 图
python3 docs/mutation_analytic/make_figure.py
# 护栏
python3 -m pytest tests/test_mutation_gaps_analytic.py -q     # 5 passed, 5.2 s
```
