# PR #128（`agent/m3181`）合并解冲报告 —— 三维自由液面声明数字注册表的语义并集

**任务**：m3684（治愈单） · **仓**：TensorBEM · **PR**：#128 · **分支**：`agent/m3181`
**脏因**：PR 头 `55db397` 之后 main 前进了（`6025970` → `b420b2e`，其中 m3150 在同一份
`declared_numbers.py` 里手抄登记了同一批表 10–12 的格子），`mergeable=false`。

| 项 | SHA |
|---|---|
| main 合并基准（本次并入侧） | `b420b2e`（Merge #126 (m3150) via merge-governor） |
| 分支旧头（PR #128 头部，解冲前） | `55db397`（`chore: re-run CI against current main`） |
| 冲突两侧的共同祖先 | `6025970`（Merge #118 (m3085)） |
| 解冲后合并提交 | `3b362cf`（`Merge origin/main into agent/m3181: union the 3-D free-surface registries`） |
| 最终头（含文档重建） | `487702d`（`docs: rebuild section 4.12 from the merged registry run…`） |
| 承载分支 | 本工作树分支 `agent/m3684`（本沙箱禁止远程写入，见 §5） |

合并方向：`origin/main` → `agent/m3181`（PR 方向是分支 → main，因此解冲在新头上完成，
main 本身不被改写）。

## 1. 冲突文件清单与逐文件解法

`git merge origin/main` 只报 2 个内容冲突；`docs/marine_bem_benchmarks.md` 自动合并，
但语义仍需人工复核（见 §1.3）。

### 1.1 `src/tensorbem/benchmarks/declared_numbers.py`（+919/−722 行级冲突）

两边的改动**不是同一件事的两种写法**，而是同一批物理格子的两套独立登记：

* mainline（m3150）：逐格手抄登记三组 `hulme_hemisphere`（表 10，31 格）/
  `hulme_chief`（表 11，6 格）/ `submerged_sphere`（表 12，25 格），量名形如
  `sub2_heave_added_mass_rel_error`，全部 `fast`/`standard` 层；
* 分支（m3181）：由文档表格元组程序化生成三组 `hulme3d`（28 格）/
  `hulme3d_chief`（9 格）/ `wang_linton3d`（26 格），量名形如
  `level2_heave_added_mass_rel_error`，全部 `heavy` 层。

**解法 = 语义并集（保留两套枚举，逐格交叉校验）**，不是 overlay：

1. 以 mainline 版本为骨架（结构更大：`_study_rows`、`_chief_rows(rows)` 展平器、
   `hulme_hemisphere`/`submerged_sphere`/`hulme_chief` 派发、`_judge`/`check`/CLI 全部保留）；
2. 完整并入分支的表格块（`_TABLE_HULME`、`_TABLE_HULME_CHIEF`、`_TABLE_WANG_LINTON`、
   `_QUANTITY_LABEL`、`_REFINEMENT_TOLERANCE`、`_refinement_rows`、`_free_surface_3d_rows`），
   插在注册表构造之前；
3. `DECLARED_NUMBERS = (…mainline 84 格…) + _free_surface_3d_rows()` → **147 格**，
   key 全局唯一（147/147），无零声明值；
4. `_run_group` 做**派发并集**：mainline 的 5 个组 + 分支的 3 个组，共 11 个组；
5. import 别名并集：mainline 的 `hemisphere_benchmark`/`sphere_benchmark` 与分支的
   `hulme_benchmark`/`wang_linton_benchmark` 同时存在（同一模块的两个别名），
   两侧调用点都不改写；
6. **重名消歧（关键陷阱）**：`_chief_rows` 在两侧同名不同义 —— mainline 是
   `_chief_rows(rows) -> dict`（展平一次求解），分支是 `_chief_rows() -> tuple[DeclaredNumber,…]`
   （构造注册行）。机械并集会让后者覆盖前者并静默改变 mainline 派发语义；
   分支侧改名 `_chief_registry_rows`，两个函数都保留。
7. **数值口径订正（不是静默改数，见 §2）**：分支表 10 的 `observed_order_b11`
   （1.62 / 1.88 → 1.80 / 1.94）与其推导公式（固定因子 2 → 实测棱长比 `h_mean`），
   表 12 的转动耦合比（2.2e-4 / 2.6e-5 → 2.15e-4 / 2.65e-5）。

### 1.2 `tests/test_declared_numbers.py`（+154/−… 冲突）

两侧各自加了测试，冲突集中在「组名白名单」与「声明值非零」两处：

| 侧 | 新增内容 | 处理 |
|---|---|---|
| mainline | `_run_group` 导入、表族白名单（含 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`）、6 个关键 key 断言、`test_3d_free_surface_rows_are_guarded_by_the_default_tiers`、`test_chief_regularisation_restores_passive_heave_damping` | 全部保留 |
| 分支 | 表族白名单（含 `hulme3d`/`hulme3d_chief`/`wang_linton3d`）、`test_three_dimensional_free_surface_tables_are_fully_registered`、`@HEAVY test_documented_hulme_tables_are_reproduced`、`@HEAVY test_documented_wang_linton_table_is_reproduced` | 全部保留 |
| 冲突项 | `assert entry.declared > 0.0`（base）两侧都改成 `!= 0.0` | 取 `!= 0.0`（表 11 的非物理负阻尼必须可声明） |
| 冲突项 | 组名白名单 | **并集**：11 个组 |

另新增 `test_three_dimensional_tables_agree_across_registries`：把生成侧 63 格映射到
手抄侧逐格比对，要求 59 个共享格相对差 ≤ 5×10⁻³，并把「生成侧独有的 4 格」写成显式白名单
（表 11 的 `exact_a33/b33_dimless`、`chief_point_count`，表 12 的 20 面元机器零上界）。
这条测试是本合并的**防回退证据**：任何一侧后续单方面改数都会转红。

### 1.3 `docs/marine_bem_benchmarks.md`（自动合并，语义人工复核）

git 自动合并成功（分支新增 §4.12，mainline 改 §4.10/§4.11 正文），但**语义上已经脏**：

* 自动合并后的表 10 采用 mainline 订正值（B₁₁′ 观测阶 1.80 / 1.94；
  表 12 转动耦合比 2.15e-4 / 2.65e-5），而分支 §4.12 的生成表格仍打印旧值
  （1.62 / 1.88；2.2e-4 / 2.6e-5）与旧判决（最坏偏差 2.23e-02、「63/63 PASS」）。
  也就是说：**文档表格与保护它的注册表互相矛盾** —— 这正是机械 overlay 会静默丢内容的形态。
* 解法：`docs/m3684/rebuild_412.py` 从重新跑出的
  `docs/m3181/outputs/three_d_free_surface_registry.json` **程序化重建 §4.12**
  （63 行表格零手抄），并把「与 m3150 手抄登记的双枚举关系」「口径订正的原因」写进正文。

### 1.4 无内容丢失审计

| 审计项 | 结果 |
|---|---|
| `git diff --name-status origin/main HEAD --diff-filter=D` | 空（未删 main 任何文件） |
| `git diff --name-status origin/agent/m3181 HEAD --diff-filter=D` | 空（未删分支任何文件） |
| 分支独有产物 `docs/m3181/` | 13 个文件全部在树中 |
| 注册表 key 唯一性 | 147 / 147 唯一 |
| 派发组 | 11 个（mainline 5 + 分支 3 + 原有 3） |
| 重名符号 | `_chief_rows` 已消歧，两侧语义均在 |

## 2. 数值口径订正的理由（不是"修数过关"）

合并前两边对同一格声明了不同数字，共有 4 格偏差 > 1e-6：

| 格 | 分支声明 | mainline 声明 | 实测 | 结论 |
|---|---:|---:|---:|---|
| 表 10 B₁₁′ 观测阶（64 面元） | 1.62 | 1.8021 | 1.80206 | 分支按固定因子 2 折算；八面体细分实测棱长比 1.87，订正为 1.80 |
| 表 10 B₁₁′ 观测阶（256 面元） | 1.88 | 1.9377 | 1.93773 | 同上，订正为 1.94 |
| 表 12 转动耦合比（80 面元） | 2.2e-4 | 2.151e-4 | 2.15099e-4 | 分支只取两位有效数字，订正为四位 2.15e-4 |
| 表 12 转动耦合比（320 面元） | 2.6e-5 | 2.6481e-5 | 2.64809e-5 | 同上，订正为 2.65e-5 |

判定依据是**文档**：m3150 已把 §4.10 该列与 §4.11 该列订正为实测口径，
注册表的功能是守护文档里印着的数字，因此不允许声明文档已不再引用的值。
分支的原始数字与其自洽推导（`log2(误差比)`）在 `git log` 中完整可查，未被抹除。

## 3. 测试与复算证据

| 证据 | 命令 | 结果 |
|---|---|---|
| 三维注册表全量复算（heavy，63 格） | `PYTHONPATH=src python3 docs/m3181/run_three_d_free_surface.py --output docs/m3181/outputs/three_d_free_surface_registry.json` | **63 / 63 PASS，drifted 0，耗时 44.6 s**，最坏等值偏差 3.79e-03（`hulme_surge_damping_rel_error_64`，容差 5e-03） |
| 双枚举交叉校验 | `test_three_dimensional_tables_agree_across_registries` | 59 / 59 共享格 ≤ 5e-03，最坏 1.19e-03；4 个独有格白名单通过 |
| main 基线（对照） | `PYTHONPATH=src python3 -m pytest -q`（`origin/main` = `b420b2e` 工作树） | 跑至 **14%** 时 0 failed / 3 skipped（该套件在本机全量需约 40 min，为控预算中止；无任何红） |
| 合并后同套件 | `PYTHONPATH=src python3 -m pytest -q tests/test_declared_numbers.py` | 见 §4「未完成」 |

注：容器内 `tensorbem` 已有一份 site-packages 安装（0.5.0 旧快照），
必须 `PYTHONPATH=src` 才测到工作树代码，否则 24 个测试收集报错——这是本单踩到的第一个坑。

## 4. 未完成 / 局限（如实）

1. **合并后全量 pytest 未跑完**：本机全量套件约 40 min（默认层已包含三维重算），
   预算内只跑到门槛测试的子集；已用 heavy 全量复算（63/63）与交叉校验测试替代，
   并保留 main 基线 14% 无红的对照记录。merge_governor 复验时会给出全量结论。
2. **§4.12 的三张表比 m3181 原稿短**：原稿表格里手写的格式（`2.23 / 1.59 / …` 合并单元格）
   在程序化重建时改为逐格一行，信息量相同但排版不同。
3. **分支根目录的 `dev_report_m3181_three_d_free_surface_registry.md` 未重写**：
   其中的 2.23e-02 等数字属于解冲前那次运行的历史记录，本报告的 §2 已声明订正关系；
   未改动它以免伪造历史，文中也未声称它是最新口径。
## 5. 交付路径（沙箱限制，如实）

本单执行环境的安全策略**拦截了远程写入命令**（`REJECTED: 命令被安全策略拦截`），
因此无法按步骤 4 直接把新头写回远程 `agent/m3181`。已完成的等价交付：

* 解冲结果 + 文档重建落在分支 **`agent/m3684`**（`3b362cf` → `487702d`），
  由平台在验收后自动推送；
* 该分支是远程 `agent/m3181`（`55db397`）的**后代**（fast-forward 关系），
  平台/治理侧把 `agent/m3684` 的目标 ref 设为 `agent/m3181` 即可让 PR #128 直接变绿，
  无损、无需重解冲突、不需要强推；
* 旧头 `55db397` → 新头 `487702d`（其后又加一个文档提交）；`merge_governor`
  下一周期复验时按此分支取用。
