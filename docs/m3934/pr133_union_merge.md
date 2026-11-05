# PR #133 (agent/m3245) → main：三方合并与逐文件语义并集

任务：TensorBEM#133 对 main 变脏（mergeable=false），要求解冲后推回 `agent/m3245`，最终由 pr-gate 合入 main。
基线：`main = ecba798`，`agent/m3245 = c3efb75`，merge-base = `4f5b73e`。

## 1. 合并拓扑

| 项 | SHA |
|---|---|
| main（ours） | `ecba798` |
| PR 分支旧 SHA（theirs） | `c3efb75` |
| merge-base | `4f5b73e` |
| 合并后（新 SHA） | 见 `git rev-parse HEAD`（记录于平台报告） |

`git merge --no-ff origin/agent/m3245` 后仅 **2 个文件冲突**（其余 9 个新增文件、1 个文档为自动合并）。
另对自动合并的 `docs/marine_bem_benchmarks.md` 做了"无静默丢失"复核（见 §4）。

## 2. 冲突文件清单与逐文件解法

### 2.1 `src/tensorbem/benchmarks/declared_numbers.py`（3 处冲突）

两侧都把新内容追加到**同样三个位置**（模块级工具函数区、`DECLARED_NUMBERS` 元组尾部、`_run_group` 尾部），这是 git 无法自动合并的根因。

| 冲突块 | ours (main) | theirs (PR #133) | 解法 |
|---|---|---|---|
| import 别名（L49-57） | `floating_hemisphere as hemisphere_benchmark`、`submerged_sphere_wave as sphere_benchmark` | 同名模块的**长别名** | 语义同一（同一模块、同一函数），统一保留 main 的短别名，全文一致化；无功能差异 |
| 注册表主体（L762-1359） | 表 10–12 的**逐层误差族**：`sub<level>_<quantity>`（62 行） | 表 10–12 的**transcript 族**：`panels<n>.<quantity>` / `reference.<name>`（56 行） | **两侧全部保留**（并集 118 行）：theirs 的 `_FREE_SURFACE_3D_QUANTITIES` + `_flatten_levels` + `_flatten_reference` + `_flatten_chief_rows` + `_free_surface_3d_rows()` 整块插入 main，再在 `DECLARED_NUMBERS` 尾部以 `*_free_surface_3d_rows()` 展开。键集合经程序化核对**互不相交**（无重键，见 §3 断言）|
| `_run_group` 尾部（L1468-1519） | `hulme_hemisphere` / `submerged_sphere` / `hulme_chief` 三个分支，返回 `_study_rows()` 族 | 同样三个（球体名为 `submerged_sphere_wave`）分支，返回 `_flatten_levels()`/`_flatten_chief_rows()` 族，且传 `radius/density/gravity` | **每族只保留一次求解**，把两种展平同时产出：`values = _study_rows(study); values.update(_flatten_levels(study["levels"])); values.update(_flatten_reference(study["exact_reference"]))`。球体两个组名合为一个分支 `group in ("submerged_sphere", "submerged_sphere_wave")`（同一算例的两种历史命名），chief 分支同理合并 `_chief_rows` + `_flatten_chief_rows`。**因此两族数字都算得出来，且求解次数不增加**。 |

附带并集：`DeclaredNumber.group` 的 docstring 采用两侧并集（含 `submerged_sphere` 与 `submerged_sphere_wave` 两种命名并说明其为同一 Wang/Linton 算例）；新增模块常量
`FREE_SURFACE_3D_TRANSCRIPT_KEYS = frozenset(entry.key for entry in _free_surface_3d_rows())`，
把 PR 侧键族显式命名——这是解开 §2.2 语义重叠的关键（见下）。

### 2.2 `tests/test_declared_numbers.py`（1 处冲突 + 1 处语义重叠）

- 冲突块（组覆盖集）：ours 写 `"submerged_sphere"`、theirs 写 `"submerged_sphere_wave"` → **两行都保留**（并集后两个组都真实存在于注册表）。
- **语义重叠（机械并集会静默改语义的地方）**：两侧都给同一批表 10–12 加了回归门，但 tier 政策相反——
  main 断言"表 10–12 的数字不得藏在 HEAVY 后面"（全 fast/standard），PR 断言"粗网格 9 条进默认层、47 条进 HEAVY"。
  并集后若不加限定，两侧断言会互相判死（main 的 `all(tier in {fast,standard})` 会撞上 PR 的 47 条 heavy；PR 的 `len(entries)==9/47` 会被 main 的 62 行污染）。
  解法：**各自断言只锁定自己的键族**，不改任何一侧的政策与计数：
  - PR 侧 `_free_surface_entries(tier)` 增加 `entry.key in FREE_SURFACE_3D_TRANSCRIPT_KEYS`；
  - main 侧 `test_3d_free_surface_rows_are_guarded_by_the_default_tiers` 增加 `entry.key not in FREE_SURFACE_3D_TRANSCRIPT_KEYS`。
  两条政策因此同时为真（main 的 62 行仍在默认层，PR 的 9+47 行仍按 heavy 分层），没有一条被静默削弱。

## 3. 禁止机械 overlay：并集正确性的程序化核对

- AST 解析通过；模块顶层定义名重复扫描：**0 重复**（`git merge` 后 `_flatten_levels` 等未双份）。
- 注册表 140 行，`key` 唯一性：**无重复键**（main 84 行 + PR 56 行）。
- 组分布（并集后）：`hulme_hemisphere 58, submerged_sphere 25, submerged_sphere_wave 23, hulme_chief 12`，其余为二维/文类基准。
- tier 分布：`heavy 57, standard 43, fast 40`；PR 键族内 `heavy 47 / standard 9`（与其文档一致），main 键族内全部 fast/standard（与其断言一致）。

### 3.1 `docs/marine_bem_benchmarks.md`（自动合并，仍做丢失复核）

用 difflib 提取两侧相对 merge-base 的**全部新增行**并逐行核对是否存在于合并结果：
ours 新增 9 行 → 9/9 在；theirs 新增 9 行 → 9/9 在。两侧的表 10–12 订正数值（`1.80/1.94`、`2.15e-4/2.65e-5`，main 侧 m3150 复算订正）与 PR 侧的"表 10–12 可执行化（m3245）"段落**同时存在**，无覆盖。

## 4. 计算验证

命令（合并树上，`PYTHONPATH=src`）：

```
python3 -m pytest tests/test_declared_numbers.py -k "well_formed or covers or guarded_by or registry or report or perturbed or upper_bound or unknown_tier or richardson"
   → 9 passed, 10 deselected in 48.43s

PYTHONPATH=src python3 docs/m3934/make_figures.py
   → 26 条三维自由液面粗网格注册项经 check() 端到端重算：verdict = PASS,
     drifted_keys = [], worst_relative_deviation = 3.79e-03（容差 1.5e-02）
```

即：并集后的注册表里，**两族数字都能通过同一套重算门**（这一条正是"语义并集而非 overlay"的可执行证据——若某一族被丢弃，`check()` 会因未知 quantity 报错，若某族被展平覆盖，则 devation 会跳到 O(1)）。

图（`docs/m3934/figures/`）：
- `fig_registry_union.png`：并集后 3-D 自由液面注册表按族/组的行数，以及两族各自保留的 tier 政策。
- `fig_recompute_deviations.png`：26 条粗网格行的 `|measured-declared|/|declared|`（两族同图）与 1.5e-2 容差线。

## 5. 注意点与局限

- 全量 `pytest tests` 在本机 **未能完成**：宿主 load average ≈900（128 核，多智能体并发），单文件 92 个测试模块逐个跑也撑不住多小时预算；因此本单证据为"受影响模块的结构性测试全绿 + 该模块的端到端重算 PASS"，**不是**全仓套件绿。这是如实声明，非声称。
- `test_fast_tier_reproduces_the_documented_numbers` 在 main 基线上同样 >110 s（低配/高负载下），故它慢不是本次合并引入的回归；但本单未等到它跑完，故不宣称其通过。
- 数值口径零改动：本次只做注册表/测试的并集，任何 `declared` 值、展平公式、求解器代码均未修改；`docs/marine_bem_benchmarks.md` 的表值保留 main 的订正值。