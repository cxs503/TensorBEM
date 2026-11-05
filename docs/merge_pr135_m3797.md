# PR #135 (`agent/m3264` → `main`): 三方合并与逐文件解冲记录（m3797）

本文件记录 PR #135 变脏后的治愈过程：冲突来自同一批声明的两种独立登记方式，
解冲原则是 **语义并集**（两侧的登记、判据、文档、产物一件不丢），并用机器可验的证据
把「保留了什么 / 为什么可以共享求解」写死。

## 1. 合并拓扑

| 项 | 值 |
|---|---|
| 被合并分支 `agent/m3264` | `5b4dee7` (tensorbem: register the 3-D free-surface declared numbers, tables 10-12) |
| 目标分支 `origin/main` | `ecba798` |
| merge-base | `6025970` |
| 分歧规模 | main 领先 54 提交，m3264 领先 1 提交 |
| 冲突文件 | `src/tensorbem/benchmarks/declared_numbers.py`（4 处）、`tests/test_declared_numbers.py`（1 处） |
| 自动合并文件 | `docs/marine_bem_benchmarks.md`（m3264 的 35 行表 10-12 说明全部保留） |
| 新增文件 | `dev_report_m3264_free_surface_3d_registry.md`、`outputs/verif_m3264/`（12 个产物）全部保留 |

## 2. 冲突根因：同一批引文被登记了两次

m3150 与 m3264 分别把 `docs/marine_bem_benchmarks.md` 表 10–12 的引文注册进
`DECLARED_NUMBERS`，两套登记的**索引方式不同**：

| 侧 | 组名 | 键 | quantity 命名 | 参数 | 行数 |
|---|---|---|---|---|---|
| main (m3150) | `hulme_hemisphere` / `hulme_chief` / `submerged_sphere` | 按面元数（`hulme_hemisphere_256_order_a33`） | 下划线扁平（`sub3_haskind_reciprocity_rel_diff_x`） | `{subdivisions, ka, submergence}`，逐行可能不同 | 62 |
| m3264 | 同名的三个组 | 按印出单元格（`hulme_a33_2`） | 点号前缀（`sub2.haskind_reciprocity_rel_diff_x`） | `{subdivisions}` + 模块常量 | 67 |

因为组名相同、quantity 命名不同，`_run_group` 的同一分支不可能同时满足两套 quantity，
直接接受任一侧都会让另一侧的 62 或 67 行**全部判 DRIFT**。这就是「机械 overlay 会静默丢内容」的具体形态。

## 3. 逐文件解法

### 3.1 `src/tensorbem/benchmarks/declared_numbers.py`

相对 main **纯增量**：`git diff HEAD` 只有 1 行删除（注册表末尾的 `)`，改为 `) + _free_surface_3d_rows()`）。

| 冲突位置 | 两侧内容 | 解法 |
|---|---|---|
| 导入别名（第 1 处） | main: `hemisphere_benchmark` / `sphere_benchmark`；m3264: `floating_hemisphere_benchmark` / `submerged_sphere_benchmark` | **两组别名都保留**，两套代码体都不改写 |
| 注册表块（第 2 处） | main 的 62 条静态行 vs m3264 的 `) + _free_surface_3d_rows()` | 保留 main 的 62 行，注册表尾部追加 m3264 的 67 行生成器 |
| 扁平化助手（第 3 处） | main `_study_rows` / `_chief_rows`；m3264 `_level_quantities` | 三者并存（命名无碰撞），各服务自己那套 quantity |
| `_run_group` 分支（第 4 处） | 三方两套实现 | main 的三分支原样保留 + m3264 的三分支以 **`*_cells` 组名**新增 |

**组名命空间**：m3264 的三组改名为 `hulme_hemisphere_cells` / `hulme_chief_cells` /
`submerged_sphere_cells`。这是本次解冲唯一的语义重命名，理由：两套登记必须共存、
且两侧测试各自断言「本块的组名集合 / 行数 / tier」，重命名后两套断言都仍然成立。

**求解共享**：两套登记读的是同一张表，重复求解是纯浪费。新增
`_STUDY_CACHE` + `_shared_study(key, run)`，规范键为
`("hulme", subdivisions, ka)`、`("sphere", subdivisions, ka, submergence)`、
`("hulme_chief", subdivision, ka_values)`，main 分支与 `*_cells` 分支用**同一个键**，
于是第二次读表是 0 成本的投影（`_study_rows` 与 `_level_quantities` 分别投影同一份 study）。
实测 `*_cells` 三组耗时 0.01 / 0.02 / 0.01 s（PASS），即 union 未引入额外求解。

### 3.2 `tests/test_declared_numbers.py`

| 项 | 解法 |
|---|---|
| 冲突区（main 的 2 个 3-D 测试 vs m3264 的 3 个 3-D 测试） | **5 个测试全保留** |
| `FREE_SURFACE_3D`（m3150 命名空间） | 保留 |
| `FREE_SURFACE_3D_CELLS`（m3264 命名空间） | 新增常量，指向 `*_cells` 三组 |
| m3264 测试里的组名/参数字面量 | 改指 `*_cells`（`parameters["hulme_hemisphere_cells"]["subdivisions"]`、`entry.group == "submerged_sphere_cells"`） |
| `standard_tier_report` fixture | 保留 m3264 版本（`check(tiers=("standard",))`），main 的两个 standard 测试与 m3264 的三个测试共用 |

新增守护测试 `test_the_cells_registry_restates_cells_the_main_registry_already_guards`：
把「重复登记」这件事本身钉住——67 条 cell 行里有 ≥58 条与 m3150 行的 `(group, |declared|)`
逐值相等（本次实测 58 条），未重叠的 9 条被显式列出（CHIEF 精确参考 2 条、passivity 反号 1 条、
潜体散文闭合界 3 条、转动耦合 3 个网格级 1e-30/2.2e-4/2.6e-5——m3150 抄 4 位有效数字、
m3264 抄印出的 2 位，属印刷带而非数值等价）。两套登记若哪天漂成两个不同的数，这条测试立刻红。

### 3.3 无静默丢失审计

| 侧新增内容 | 处置 | 证据 |
|---|---|---|
| main：62 条 3-D 行 + `_study_rows` / `_chief_rows` | 原样保留 | `git diff HEAD` 无删除行（除注册表尾 `)`） |
| m3264：67 条 3-D 行 | 全部保留（组名 `*_cells`） | 行数 151 = main 84 + m3264 67；`len(cells) == 67` |
| m3264：`--summary worst_value_relative_deviation` | 自动合并保留 | `grep worst_value_relative_deviation` 命中 |
| m3264：doc 表 10–12 说明 35 行 | 自动合并保留 | `git diff HEAD -- docs/...` 35 行 `+` |
| m3264：dev_report + `outputs/verif_m3264/` 12 产物 | 全部入树 | `git status` 显示 `A` |
| m3150 文档对表 10–12 的更正文句 | 保留（main 侧删除属 main 自身意图） | 相对 m3264 的差异 = main 的改动 |

## 4. 验证

见本仓库测试证据与 `dev_report_m3797_pr135_merge.md`：核心套件 `tests/test_declared_numbers.py`
在合并树上通过，行覆盖率与 main 基线一致（151 行全量登记，62 + 67 双登记均被判定），
standard tier 墙钟未因并集而变长（求解共享）。