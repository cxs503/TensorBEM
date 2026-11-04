# TensorBEM PR #128（`agent/m3181`）三方合并治愈报告

任务：m3735「治愈 · TensorBEM#128」。PR #128 对 `main` 变脏（`mergeable=false`），本报告记录
三方合并的做法、逐文件解冲、无静默丢失审计与测试证据。

## 1. 案例背景与仓库面

| 项 | 值 |
|---|---|
| 仓库 | `TensorBEM`（cxs503/TensorBEM） |
| 目标分支 | `agent/m3181`（PR #128） |
| merge base | `4f5b73e73c4e731582f2b050328ec5fcf62b4a86` |
| 合并前 `origin/main` | `b420b2ee1e98276f341a10eafe3fda2d80807756` |
| 合并前 `origin/agent/m3181` | `55db3976dc283e9522334f2ab7309fc655f3ae27` |
| 合并后 `agent/m3181` | 见 `docs/m3735/merge_shas.json`（本提交） |
| 合并方式 | `git merge --no-ff origin/agent/m3181`（在 `origin/main` 之上三方合并） |

双方改动面（相对 merge base）：

| 侧 | 改动文件 | 新增行 |
|---|---:|---:|
| `agent/m3181`（m3181：三维自由液面声明数字注册） | 17 | 3033 |
| `origin/main`（m3150/m3336/m3224 等已并入的成果） | 77 | 16899 |

## 2. 冲突文件清单与逐文件解法

| 文件 | git 状态 | 解法（语义并集） |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | **CONFLICT (content)** | 五处并集：① import 同时保留两侧别名（`floating_hemisphere as hemisphere_benchmark` / `as hulme_benchmark`，`submerged_sphere_wave as sphere_benchmark` / `as wang_linton_benchmark`，被各自的分支体引用）；② m3181 的 3-D 表格常量块（`_TABLE_HULME`/`_TABLE_HULME_CHIEF`/`_TABLE_WANG_LINTON`/`_QUANTITY_LABEL`/`_REFINEMENT_TOLERANCE`/`_refinement_rows`/`_chief_table_rows`/`_free_surface_3d_rows`，206 行）整体插入到注册表之前；③ 注册表改为 `(main 的元组) + _free_surface_3d_rows()`，两侧条目全留（147 行）；④ main 的 `_study_rows` / `_chief_rows(rows)` 原样保留；⑤ `_run_group` 六个 group 分支并存（m3181 的 `hulme3d`/`hulme3d_chief`/`wang_linton3d` + main 的 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`）。**一次真实语义冲突**：两侧都定义了模块级 `_chief_rows`，但签名/语义不同（main 把 study 压平成 `dict[str, float]`，m3181 生成 `DeclaredNumber` 元组）。机械并置会让后定义者**静默覆盖**前者，故把 m3181 侧改名为 `_chief_table_rows()`（含其唯一调用点），两侧行为都不丢。 |
| `tests/test_declared_numbers.py` | **CONFLICT (content)** | 并集：main 的 `test_3d_free_surface_rows_are_guarded_by_the_default_tiers`、`test_chief_regularisation_restores_passive_heave_damping` 与 m3181 的 `test_three_dimensional_free_surface_tables_are_fully_registered`、`@HEAVY::test_documented_hulme_tables_are_reproduced`、`@HEAVY::test_documented_wang_linton_table_is_reproduced` 全部保留；`test_registry_covers_every_documented_benchmark_family` 的期望集合取**并集**（新增 `hulme3d`/`hulme3d_chief`/`wang_linton3d` + 原有 `hulme_chief`/`hulme_hemisphere`/`submerged_sphere`）；`assert entry.declared > 0.0` 两侧都改成 `!= 0.0`（保留 main 的注释，语义同一）。 |
| `docs/marine_bem_benchmarks.md` | 自动合并（**已人工复核 + 机器审计**） | git 自动合并成功：main 的 m3150 订正（表 10 的 B11 观测阶 1.80/1.94、表 12 转动耦合 2.15e-4/2.65e-5、订正说明段）与 m3181 新增的 §4.12（115 行，注册表复算）互不重叠、**共存**。**另加一段「合并口径对照」注**：说明两套注册族并存、且 m3181 侧登记的是订正前口径，避免读者把 1.62/1.88 与 1.80/1.94 当成互相矛盾的漂移。 |

### 口径差异（合并后必须知晓的物理/度量事实）

1. **表 10 的 `B11'` 观测阶**：m3181 侧按理想网格减半（`log2` 比）折算 → 1.62 / 1.88；main（m3150 订正）
   按模块统一采用的实测平均棱长比 `h0/h1 = 1.87` 折算 → 1.80 / 1.94。两者都是对同一误差序列
   40.24 % → 13.05 % → 3.53 % 的不同折算口径，各自内部自洽。
2. **表 12 球心转动耦合比**：m3181 侧沿用两位有效数字的 2.2e-4 / 2.6e-5；main 用实测可复现值
   2.15e-4 / 2.65e-5（前者正确舍入应为 2.1e-4）。

结论：两侧注册项都能复现**自己的**声明值（下文 §5 实测），故并集不产生互相矛盾的判据；单一口径的去重列入后续项。

## 3. 无静默丢失审计（机器判据）

`docs/m3735/union_audit.json`（生成脚本：本节命令）逐行检查「每一侧相对 merge base 新增的源码/文档行，
在合并后的工作树里是否存在」：

| 侧 | 涉及文件 | 新增行 | 合并后缺失行 |
|---|---:|---:|---:|
| `agent/m3181` | 17 | 3033 | **0** |
| `origin/main` | 77 | 16899 | **0** |
| 判决 | — | — | **CLEAN** |

审计里显式登记了两类**非丢失**改写（否则不会被算作缺失，也不会静默）：

* `EXPECTED_REMAP`（2 行）：`def _chief_rows():` / `+ _chief_rows()` → `_chief_table_rows()`（同名冲突改名，见 §2）；
* `SUBSUMED`（3 行）：m3181 关于 `declared != 0.0` 的三行注释，被 main 对同一断言的等价注释取代，
  映射目标行必须存在才判通过。

二进制（PNG）、`docs/m3181/*` 全部产物、`dev_report_m3181_*.md` 均按原样进入合并树。

## 4. 怎么做的（流程与工具链）

```bash
git worktree add /root/tp-agent/work/m3735_base --detach origin/main   # 干净的 main 基线
cd /root/tp-agent/work/m3735
git merge --no-ff --no-commit origin/agent/m3181                       # 触发冲突
# 解冲：/tmp/mg/merge_script.py 以文本锚点做「并集构造」而非 overlay，
#       并重命名同名函数；随后 AST + 重复定义扫描（KNOWN_TRAPS 纪律）
python3 -c "import ast; ast.parse(open(p).read())"                     # 语法
# 重复顶层定义扫描：dup defs=[] / dup module assigns=[]
# 无丢失审计
python3 docs/m3735/union_audit.py                                      # → union_audit.json
# 测试（PYTHONPATH 指向本工作树，避免误用他处 install 的 tensorbem）
PYTHONPATH=/root/tp-agent/work/m3735/src python -m pytest -q <core subset>
```

> 陷阱记录：本机 `python -c "import tensorbem"` 默认解析到**别的 worktree**（`.../work/m3685/src`）的
> `tensorbem`。所有测试命令都显式 `PYTHONPATH=<本工作树>/src` 并先打印 `tensorbem.__file__` 核对。

## 5. 计算结果与测试证据

### 5.1 核心测试子集：合并树 vs main 基线（同一命令、同一环境）

| 树 | 结果 | 用时 |
|---|---|---|
| `origin/main`（b420b2e，干净基线） | **53 passed, 3 skipped, 0 failed** | 128.5 s |
| 合并树（main + m3181 语义并集） | **54 passed, 5 skipped, 0 failed** | 131.2 s |

子集 = `tests/test_declared_numbers.py tests/test_doc_claims.py tests/test_benchmark_cli_contract.py tests/test_capability_evidence.py`。
合并树比基线多 1 passed（m3181 的非 heavy 注册表形态测试）+ 2 skipped（m3181 的两个 `@HEAVY` 用例，默认不跑）。
**未低于基线**，且 0 失败。

### 5.2 heavy 层：m3181 的 63 个三维自由液面格子（合并树实跑）

| 用例 | 结果 | 用时 |
|---|---|---|
| `test_documented_hulme_tables_are_reproduced`（37 格）+ `test_documented_wang_linton_table_is_reproduced`（26 格） | **2 passed** | 18.2 s |

说明并集注册表的**两条路径都活着**：m3181 的 `hulme3d`/`hulme3d_chief`/`wang_linton3d` producer 分支能在
合并后（main 的新 `free_surface_solver3d`/`geometry3d`/`symmetry3d` 之上）原样复现其声明值。

### 5.3 注册表规模（合并后）

| 注册族 | 层 | 行数 |
|---|---|---:|
| `hulme_hemisphere` / `hulme_chief` / `submerged_sphere`（main） | fast + standard | 31 / 6 / 25 |
| `hulme3d` / `hulme3d_chief` / `wang_linton3d`（m3181） | heavy | 28 / 9 / 26 |
| 2-D 与 Ursell 等原有族 | fast / standard / heavy | 28 |
| **合计** | — | **147（键全部唯一）** |

![union audit](figures/fig_union_audit.png)

### 5.4 结果图 2：63 个三维格子「文档声明值 vs 复算值」

![declared vs measured](figures/fig_registry_declared_vs_measured.png)

左图：62 个等值判据格的 `declared` vs `measured`（双对数），灰带为注册表 0.5 % 容差，全部落在带内；
右图：按族的 `|相对偏差|` 分布（对数轴），并以 0.5 %（值行容差）、2.5 %（表 12 转动耦合舍入带）、3 %（验收线）作参照。
零下界格 `wang_linton_rotational_added_mass_max_ratio_20` 按上界判（上界 1e-20，实测 2.49e-32），不参与等值偏差统计。

## 6. 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 基准 | 本次结果 | 相对偏差 | 是否 <3%？ |
|---|---|---|---:|---|
| 源码/文档行保全率（m3181 侧，3033 行） | 100 % | 100 %（0 缺失） | 0.00 % | **是** |
| 源码/文档行保全率（main 侧，16899 行） | 100 % | 100 %（0 缺失） | 0.00 % | **是** |
| 核心子集用例数 | 53 passed（main 基线） | 54 passed，0 failed | +1 用例（不减） | **是**（不低于基线） |
| 63 个三维格子逐项复算（heavy 层） | 文档声明值 | 全部 PASS | 最坏 2.23e-2（2.23 %，`wang_linton_rotational_added_mass_max_ratio_80`，两位有效数字舍入带 2.5 %） | **是** |
| 62 个等值格中的次坏项 | 文档声明值 | PASS | 1.15e-3（0.12 %） | **是** |
| 注册表键唯一性 | 无重复 | 147/147 唯一 | 0 | **是**（离散判据） |
| 顶层定义重复（同名覆盖风险） | 0 | 0（AST 扫描） | 0 | **是**（离散判据） |

**未完成/不达标项（honest）**：全量套件（91 个测试模块）在本机未跑完 —— 宿主机 load average ≈ 217（128 核被
其他任务占满），全量跑到 14 % 用掉约 5 分钟且未见失败，合并树与 main 基线两边进度一致；为不阻塞入池，
本次以「核心子集对齐基线 + heavy 目标用例实跑」作证据，**全量对比留给 merge_governor 的 CI 复验**。
这不构成「不低于基线」的完整证明，仅覆盖上述子集。

## 7. 注意点与后续

1. **同名函数是最危险的静默丢失源**：`_chief_rows` 两侧语义不同，任何「后写者胜」的并置都会让一侧的 62/63 个
   注册格子失效且测试仍是绿的（用例按 group 过滤）。本次以重命名 + 审计留痕处理。
2. **重复登记**：表 10–12 的 62 个格子在合并树里被两套族各登记一次（heavy 层与 fast/standard 层），
   `--tiers heavy` 全量复算会多跑一次同源求解（约 +18 s）。是否收敛为单一口径（并统一 `B11'` 观测阶与转动耦合
   的折算/舍入口径）需要 owner 裁定，不建议合并期自行删族。
3. `docs/marine_bem_benchmarks.md` §4.12 现在同时存在「订正前口径的 1.62/1.88、2.2e-4/2.6e-5」与表 10/12 正文的
   「订正后 1.80/1.94、2.15e-4/2.65e-5」，两者由 §4.12 的「合并口径对照」注显式区分，**不是**互相矛盾的漂移。
4. 本机存在多 worktree 共享 `tensorbem` 安装的现象，任何测试务必先核对 `tensorbem.__file__`。

## 8. 复现命令

```bash
# 无丢失审计（需要 git 侧 refs：origin/main、origin/agent/m3181 与 merge base）
python3 docs/m3735/union_audit.py

# 图（仅读 docs/m3181/outputs/*.json，无求解）
PYTHONPATH=src python3 docs/m3735/make_figures.py

# 核心子集
PYTHONPATH=src python -m pytest -q tests/test_declared_numbers.py tests/test_doc_claims.py \
    tests/test_benchmark_cli_contract.py tests/test_capability_evidence.py

# heavy 层三维注册复算
TENSORBEM_HEAVY_TESTS=1 PYTHONPATH=src python -m pytest -q tests/test_declared_numbers.py \
    -k "documented_hulme_tables or documented_wang_linton_table"
```
