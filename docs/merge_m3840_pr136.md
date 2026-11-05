# PR #136 治愈记录：agent/m3250 三方合并进 main（m3840, 2026-11-05）

## 1. 案例背景

- 仓库 TensorBEM，PR #136（分支 `agent/m3250`）对 `main` 报 `mergeable=false`。
- 合并基底 `4f5b73e`；`main` 侧 = `ecba798`（含 m3150/m3224/m3336 等 89 个文件、+19045/−149 行）；
  `agent/m3250` 侧 = `b7709c1`（21 个文件、+2566/−3 行：docs/m3250 证据工件 +
  `src/tensorbem/benchmarks/declared_numbers.py` + `tests/test_declared_numbers.py` +
  `docs/marine_bem_benchmarks.md`）。
- 两侧工作**同一主题**：把 `docs/marine_bem_benchmarks.md` 表 10/11/12 的三维自由面数字
  （浮体半球 Hulme、三维内点 CHIEF、潜没球 Wang/Linton）登记为可复算的「声明数字」。

## 2. 做法（机器可复核）

1. `git merge --no-commit --no-ff origin/agent/m3250` on `main` → 冲突 2 个文件 5 处
   （`declared_numbers.py` 3 处、`test_declared_numbers.py` 2 处）；
   `docs/marine_bem_benchmarks.md` 自动合并，需另行验证。
2. 逐处解冲（不做机械 overlay）：每处保留双方内容，命名冲突用**并集命名**而不是二选一。
3. 合并后做**两侧增删行包含性审计**（防「静默丢内容」）：对三个文件，把 `base..main` 与
   `base..m3250` 的每一行新增与删除行逐行取集合，检查合并结果是否包含全部新增行、
   是否复活了任何被删行。结果：`declared_numbers.py` 新增 661/400 行、缺失 0/0；
   `test_declared_numbers.py` 新增 55/60、缺失 0/**1**；doc 新增 9/13、缺失 0/0；
   复活行仅 1 处（见 §3.5，是刻意取舍）。唯一 1 处「缺失/复活」成对出现，说明没有其他静默丢失。

## 3. 冲突文件清单与逐文件解法

### 3.1 `src/tensorbem/benchmarks/declared_numbers.py`（3 处）

| 位置 | main(HEAD) | agent/m3250 | 解法 |
|---|---|---|---|
| import 块 | `floating_hemisphere as hemisphere_benchmark`、`submerged_sphere_wave as sphere_benchmark` | 同两模块改名 `..._benchmark` 长别名 | **并集**：4 个别名全部保留（两套 `_run_group` 分支各自引用自己的别名） |
| `DECLARED_NUMBERS` 尾部 | 62 条字面量 `DeclaredNumber`（group `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`） | 1 行 `*_free_surface_three_dimensional_numbers()`（63 条，group `hemisphere3d`/`hemisphere3d_chief`/`submerged_sphere3d`） | **并集**：字面量整块 + 生成器 splat 都保留（两套命名互不重键，key 唯一性测试通过） |
| `_run_group` 尾部 | `hulme_hemisphere` / `submerged_sphere` / `hulme_chief` 三个分支（`_study_rows`、`_chief_rows`） | `hemisphere3d` / `hemisphere3d_chief` / `submerged_sphere3d` 三个分支（`_flatten_refinement_levels`、`_reference_values`） | **并集**：六个分支全保留；两侧的私有 helper（`_study_rows`/`_chief_rows` 与 `_flatten_refinement_levels`/`_reference_values`）本就在各自自动合并区域完整存在 |

附带修正（合并暴露出的语义冲突，非冲突块内）：`_SPHERE_ROTATION` 中 80/320 面元转动耦合比，
m3250 取两位有效数字 `2.2e-4 / 2.6e-5`，main 在 m3150 复算中已把表 12 订正为 `2.15e-4 / 2.65e-5`
（main 的文档段落保留在合并后的 doc 内）。合并后的文档表只印订正值，故 m3250 的两行同步改为
`2.151e-4 / 2.6481e-5`，并留 merge note 说明来源。否则注册表会引用一个文档已不再印的数字。

### 3.2 `tests/test_declared_numbers.py`（2 处 + 1 处自动合并的语义破坏）

| 位置 | 双方 | 解法 |
|---|---|---|
| `test_registry_covers_every_documented_benchmark_family` 的 group 集合 | main: `hulme_chief/hulme_hemisphere/submerged_sphere`；m3250: `hemisphere3d/hemisphere3d_chief/submerged_sphere3d` | **并集**：7 个 group 全部列入（两套命名都必须在册） |
| 新增三维测试 | main: `test_3d_free_surface_rows_are_guarded_by_the_default_tiers`（62 行不得藏在 heavy tier）+ `test_chief_regularisation_restores_passive_heave_damping`；m3250: `test_free_surface_3d_tables_are_reproduced`（63 行复算 + 三条定性结论） | **两个测试都保留**，各自只针对自己的命名空间 |
| `test_standard_tier_reproduces_the_documented_ursell_numbers`（**自动合并缺陷**） | m3250 把 `check(tiers=("standard",))` 收窄为只跑 `group.startswith("ursell")`；main 在同一函数的断言集合里加了 5 个三维 key | git 行级自动合并产生了「只跑 ursell + 断言必须含三维 key」的组合，**该测试永远不可能通过**。解：恢复 main 的全程 `check(tiers=("standard",))`（唯一能同时满足双方断言的形式），并在 docstring 里写明取舍与原因 |

这是本单最重要的一处发现：**无冲突标记的自动合并也可能在语义上把测试变成永假**。
只看 `git status` 的 `UU` 列表会漏掉它，因此必须逐文件读合并结果。

### 3.3 新增 `test_the_two_three_dimensional_registry_conventions_agree`

并集命名留下的真实风险是「同一格在两套命名下声明值漂移」。新测试把两套命名规范化到
`(group, panels, quantity)` 三元组后逐格比对声明值（观测阶按文档两位小数放宽到 `rel_tol=3e-3`，
CHIEF 的 `b33` 取绝对值口径），并钉住两侧行数 62 / 63。它在本单里就抓到 3 处分歧
（转动耦合比旧舍入 2 处、观测阶两位小数 2 处），并据此驱动了 §3.1 的修正。

### 3.4 `docs/marine_bem_benchmarks.md`（自动合并，已验）

main 的 m3150 订正段（表 10 观测阶 1.80/1.94、表 12 转动耦合比 2.15e-4/2.65e-5）与
m3250 的 `<!-- BEGIN m3250-fs3d-pins -->` 钉桩段**并存**，两侧新增行 9/13 行全部在册，
无复活删除行。

### 3.5 取舍记录（唯一缺失/复活行）

`tests/test_declared_numbers.py` 中 m3250 删除的 `report = check(tiers=("standard",))`
（替换为过滤版）与 main 保留的同一行 —— 合并结果取 main 的行（§3.2）。这是本单唯一
「某侧删除未生效」的地方，是有意为之并已在测试 docstring 里写明。

## 4. 测试证据

环境限制（如实记录）：99 控制面当前 load 200–1500（其他智能体并发），集群 loong64 两节点
**未安装 numpy**（`ModuleNotFoundError`），且从节点拉 GitHub 失败，故三维稠密复算无法在集群上跑；
本单证据分两层：

| 层 | 命令 | 结果 |
|---|---|---|
| 静态 | `python3 -c "ast.parse(...)"` 三个改动文件 | 通过（无残留冲突标记，`grep '^<<<<<<<\|^=======\|^>>>>>>>'` 为空） |
| 注册表结构 + 合并缝合线 | `PYTHONPATH=src pytest tests/test_declared_numbers.py -k "conventions_agree or registry_is_well_formed or covers_every_documented or guarded_by_the_default_tiers"` | **4 passed**（0.44 s；含 new seam test） |
| 全量 `test_declared_numbers.py`（含三维标准层稠密复算） | `PYTHONPATH=src pytest tests/test_declared_numbers.py -q` | main 基线（`git archive origin/main` 到 /tmp 的干净副本）与本合并树各起一次；基线在标准层复算段跑了约 8 min 后继续推进，合并树因多出 m3250 的 63 行（物理同一批表格、两套命名）标准层耗时约 2×，在 99 的高负载下 **本单预算内未跑完，无 PASS/FAIL 结论**（honest-FAIL：此项证据缺失） |

因此「测试不低于 main 基线」在本单只有**结构层与缝合线层**证据齐全；三维稠密复算层的
耗时对比结论欠反馈，已列为待复验项（见 §6）。

## 5. 注意点与适用范围

- **并集命名的代价**：同一批表 10–12 数字现在被两套 group 各算一遍（62 + 63 行），
  `check(tiers=("standard",))` 的三维求解量约为基线 2×。这是「不丢任何一侧」的直接代价，
  去重要靠后续单独立项（把一套映射到另一套的 key 上，或让生成器同时产出两套 group），
  本单**不擅自删除**任何一侧。
- **无冲突标记 ≠ 语义正确**：本次的永假测试就是自动合并产出的。今后此类治愈单必须
  把「两侧增删行包含性审计 + 逐文件读合并结果」列为强制步骤。
- 适用范围：本记录只覆盖三个冲突/半冲突文件；其余 86 个 main 侧文件与 18 个 m3250 侧
  docs 工件为无交叉路径（自动合并干净，且 m3250 侧 `docs/m3250/outputs/verif_m3250/*`
  与 main 侧 `docs/m32xx/*` 路径不相交）。

## 6. 待复验项（交 merge_governor / 下一单）

1. 全量 `pytest tests/`（至少 `test_declared_numbers.py` + `test_doc_claims.py`）在空载环境或
   装好 numpy 的集群节点上跑完，确认标准层 125 行三维复算全 PASS，且总时长可接受。
2. 决定是否收敛两套命名（去重 62/63 行），需要 owner 或后续单拍板。
