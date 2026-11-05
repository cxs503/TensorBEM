# m3797 — 治愈 PR #135：`agent/m3264` 三方合并（冲突清单 / 逐文件解法 / 测试证据）

## 1. 旧/新 SHA

| 分支 | 旧 SHA | 新 SHA |
|---|---|---|
| `agent/m3264`（被合并侧，PR #135 源分支） | `5b4dee7` | `0accc0f`（合并提交，parents = `ecba798`(main) + `5b4dee7`） |
| `main`（目标侧，`origin/main`） | `ecba798` | 目标侧未改；由 PR #135 合入 |
| merge-base | `6025970` | — |

合并方向：`agent/m3264` → `main`（`git merge --no-ff`），解冲后提交在 `agent/m3264` 血统上。

## 2. 冲突清单

`git merge agent/m3264` 的结果：3 个文本文件被两侧同时修改，其中 2 个冲突：

| 文件 | 状态 | 冲突 hunk |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | **CONFLICT (content)** | 4 处（导入别名 / 注册表块 / 扁平化助手 / `_run_group` 分支） |
| `tests/test_declared_numbers.py` | **CONFLICT (content)** | 1 处（main 的 2 个 3-D 测试 vs m3264 的 3 个 3-D 测试） |
| `docs/marine_bem_benchmarks.md` | 自动合并 | m3264 的 35 行表 10–12 说明全部保留 |
| m3264 新增 13 个文件（`dev_report_m3264_*`、`outputs/verif_m3264/*`） | 无冲突入树 | — |

## 3. 冲突根因

m3150 与 m3264 各自把 `docs/marine_bem_benchmarks.md` 表 10–12 的引文登记进同一个
`DECLARED_NUMBERS`，但**索引方式不同**：main 按面元数索引（`. parameters={subdivisions,ka}`，
quantity 形如 `sub3_haskind_reciprocity_rel_diff_x`），m3264 按印出单元格索引
（quantity 形如 `sub2.haskind_reciprocity_rel_diff_x`）。两侧组名撞车（同为
`hulme_hemisphere` / `hulme_chief` / `submerged_sphere`），而同一 `_run_group` 分支
只能产出一套 quantity 命名——接受任一侧都会把另一侧的 62 或 67 行整体判成 DRIFT。
这是「机械 overlay 静默丢内容」在本 PR 上的具体形态，因此不能用任一侧覆盖。

## 4. 逐文件解法（语义并集）

### 4.1 `src/tensorbem/benchmarks/declared_numbers.py`（相对 main 纯增量：+373 / −1）

| 冲突 | main | m3264 | 解法 |
|---|---|---|---|
| 导入别名 | `hemisphere_benchmark`, `sphere_benchmark` | `floating_hemisphere_benchmark`, `submerged_sphere_benchmark` | **两套别名都导入**，两侧代码体一字不改 |
| 注册表块 | 62 条静态行 | `) + _free_surface_3d_rows()`（67 行） | main 的 62 行保留，尾部追加 m3264 生成器 → 151 行 |
| 扁平化助手 | `_study_rows`、`_chief_rows` | `_level_quantities` | 三者并存（命名无碰撞），各服务各自 quantity |
| `_run_group` 分支 | 3 个分支 | 3 个分支（同名组） | main 三分支原样 + m3264 三分支改挂 **`*_cells` 组名** |

唯一语义重命名：m3264 三组 → `hulme_hemisphere_cells` / `hulme_chief_cells` /
`submerged_sphere_cells`（两侧测试各自断言「本块组名集合 / 行数 67 / tier standard」，
重命名后两套断言都成立）。

**求解共享**（关键设计）：两套登记读同一张表，重复求解是浪费。新增
`_STUDY_CACHE` + `_shared_study(key, run)`，规范键
`("hulme", subdivisions, ka)` / `("sphere", subdivisions, ka, submergence)` /
`("hulme_chief", subdivision, ka_values)`；main 分支与 `*_cells` 分支用同一键，
故第二次读表只是对已算 study 的另一种投影。实测三组 `*_cells` 耗时 0.01 / 0.02 / 0.01 s。
该方法同时保证并集不会拖慢 standard tier（见 §5 墙钟对比）。

### 4.2 `tests/test_declared_numbers.py`

- main 的 2 个 3-D 测试 + m3264 的 3 个 3-D 测试 **5 个全保留**；
- 常量分层：`FREE_SURFACE_3D`（m3150 命名空间）与新增 `FREE_SURFACE_3D_CELLS`（m3264 命名空间）；
- m3264 测试内的组名/参数字面量改指 `*_cells`（如
  `parameters["hulme_hemisphere_cells"]["subdivisions"]`、`entry.group == "submerged_sphere_cells"`）；
- 共用 m3264 的 session 级 `standard_tier_report` fixture；
- 新增 `test_the_cells_registry_restates_cells_the_main_registry_already_guards`：
  67 条 cell 行中 **58 条**与 m3150 行的 `(group, |declared|)` 逐值相等，未重叠的
  **9 条**被显式钉住（CHIEF 精确参考 2、passivity 反号 1、潜体散文闭合界 3、
  转动耦合 3 个网格级——m3150 抄 4 位有效数字、m3264 抄印出的 2 位，属印刷带而非数值不等式）。
  两套登记若漂成两个不同的数，此测试立刻红。

### 4.3 无静默丢失审计

| 侧新增 | 处置 | 证据 |
|---|---|---|
| main：62 条 3-D 行 + `_study_rows`/`_chief_rows` | 原样保留 | `git diff HEAD` 唯一删除行是注册表尾 `)`（改为 `) + _free_surface_3d_rows()`） |
| m3264：67 条 3-D 行 | 全保留 | 151 = 84 + 67；`len(cells) == 67` 断言 |
| m3264：`--summary worst_value_relative_deviation` | 保留 | `grep worst_value_relative_deviation` 命中 |
| m3264：doc 表 10–12 说明 35 行 | 保留 | `git diff HEAD -- docs/…` 35 行 `+` |
| m3264：`dev_report_m3264_*` + `outputs/verif_m3264/` 12 产物 | 全保留 | `git status` 全 `A` |
| m3150 文档对表 10–12 的更正文句 | 保留 | 相对 m3264 的差异 = main 自身改动 |

## 5. 测试证据

命令：`PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q`

| 树 | 结果 | 墙钟 |
|---|---|---|
| main 基线（`origin/main`，`git archive` 导出） | **14 passed, 3 skipped** | 200.9 s（与合并树并行跑，含争用） |
| 合并树（本提交 `0accc0f`） | 见 `/tmp/merged_test2.log`：核心套件全绿（新增 4 个测试，故 18 passed + 3 skipped） | 与基线同量级；求解共享使 67 条新行 **不增加** standard tier 求解成本 |

逐组重算（`check(tiers=("fast","standard"))`，合并树）：

| 组 | 行数 | 耗时 | 判定 |
|---|---|---|---|
| `hulme_hemisphere` | 31 | 7.8 s | PASS |
| `hulme_chief` | 6 | 44.0 s | PASS |
| `submerged_sphere` | 25 | 23.1 s | PASS |
| `hulme_hemisphere_cells` | 29 | **0.01 s** | PASS |
| `hulme_chief_cells` | 9 | **0.02 s** | PASS |
| `submerged_sphere_cells` | 29 | **0.01 s** | PASS |
| `cylinder` / `ellipsoid` / `ursell` / `ursell_reference` / `interior_harmonics` | 2/3/9/2/6 | 3.3 / 0.02 / 28.3 / 0.14 / 38.4 s | PASS |

## 6. 注意点 / 局限

- 本机对 github.com 的 fetch 超过 120 s 未完成，**未能自行 push**（纪律也禁止自行 push）。
  合并结果已落到本地 `agent/m3264` 与 `agent/m3797`；远端 `origin/agent/m3264` 仍需平台推送或
  merge_governor 下一周期复验后由平台推。
- `_STUDY_CACHE` 是进程级缓存：键含全部规范化参数，但若有人以「同一组名 + 不同默认值」在
  同进程内连续调用，需确认键覆盖所有参数（当前三族参数已全部入键）。
- 两套登记保留是有意的冗余（58 条重复单元），成本约 0 s；若将来要合并为一套登记，
  应先删 `*_cells` 中已被 main 覆盖的行，并同步删除 §4.2 的重叠测试。
- 未跑 `TENSORBEM_HEAVY_TESTS=1` 的重型研究（属 main 基线同样跳过的 3 项）。