# PR #135 —— `agent/m3264` → `main` 冲突解决与并集审计（m3951 复核轮）

- 仓库：TensorBEM；PR：**#135**（分支 `agent/m3264`）
- 合并基：`4f5b73e`（`Merge #120 (m3086)`）
- 两侧顶端：`main = ecba798`，`agent/m3264 = 882201e`（推送前）
- 冲突范围：**2 个文件**（`src/tensorbem/benchmarks/declared_numbers.py`、`tests/test_declared_numbers.py`），
  外加 1 个自动合并文件（`docs/marine_bem_benchmarks.md`，两侧只做追加，git 自动并集）。
  其余 85 条 main-only / 15 条 branch-only 路径均为单侧改动，合并树中逐字节保持不变（见审计 §2）。

## 1. 冲突的实质

两侧**独立登记了同一批三维自由液面基准数字**（`docs/marine_bem_benchmarks.md` §4.10 表 10/11、
§4.11 表 12）：

| | main（m3xxx 系列） | `agent/m3264`（m3264） |
|---|---|---|
| 登记方式 | 62 条**字面量** `DeclaredNumber`，手写键 | 67 条由 `_free_surface_3d_rows()` **生成**，键为 `sub3.b11` 风格 |
| 扁平化助手 | `_study_rows()` / `_chief_rows()` | `_level_quantities()`（点分键） |
| 分组 `_run_group` | 传 `subdivisions/ka/submergence`，用 main 助手 | 传 `radius/density/gravity` 常量，用点分助手 |
| 测试 | 17 个 | 18 个（多一个“全体 3D 行必须落在 standard 层”的守卫） |

机械 overlay 会静默丢内容：直接取任一侧都会让另一侧的 60 余条登记、1 个测试和 1 组助手的语义消失。

## 2. 逐文件解法（语义并集）

### `src/tensorbem/benchmarks/declared_numbers.py`（4 处冲突）

| # | 冲突位置 | 解法 | 依据 |
|---|---|---|---|
| 1 | `tensorbem.benchmarks` 导入别名 | 保留 main 的 `hemisphere_benchmark` / `sphere_benchmark` | 并集后的 `_run_group` 用 main 的调用形态（`subdivisions/ka/submergence`），main 侧未冲突测试直接引用这两个别名；分支只是同一模块的另一种别名，无独立语义 |
| 2 | `DECLARED_NUMBERS` 表 10–12 登记块 | 以 main 的 62 条字面量为骨架；分支**独有的 7 格**按 main 键风格重新声明（表 11 的 `exact_a33/exact_b33`、`unregularized_passivity`、§4.11 三条文字上界、最粗网格旋转耦合）；分支生成器与 main 重复的 60 格不重复登记 | 同一行不能登记两次（`test_registry_is_well_formed` 要求键唯一）；覆盖由审计逐格核对 **67/67** |
| 3 | 扁平化助手 | 保留 main 的 `_study_rows()` / `_chief_rows()`，删除分支的 `_level_quantities()` | 点分键（`sub3.b11`）是分支生成器的私有约定，登记块改为 main 键风格后无引用者；其“一网格层一次求解”的意图由 main 助手的 `parameters.subdivisions` 全层缓存实现（`check()` 按 `(group, parameters)` 缓存） |
| 4 | `_run_group` 的 3D 分发 | 保留 main 版本；分支的 `_HULME_*`/`_SPHERE_*` 常量保留在模块级，main 版本改用这些常量做默认值，物理设置（`ka=1.0 / a=1 / ρ=1025 / g=9.80665`，球 `ka=0.8 / f/a=1.5`）一字不差 | 分支常量即文档引用设置，作为默认值保留避免“魔法数”漂移 |

### `tests/test_declared_numbers.py`（1 处冲突）

两侧测试**全部保留**（main 17 + branch 18 → 合并 20 个测试函数；其中 2 个是各自唯一的新增）。
唯一例外：main 的 `test_3d_free_surface_rows_are_guarded_by_the_default_tiers` 被折叠进分支更严的
`test_free_surface_3d_rows_live_on_the_standard_tier`——折叠后仍逐条断言 main 的四项守卫
（分组集合、`tier` 不落 heavy、`entry.source` 引用文档、键唯一），审计脚本对这四条子句做机器检查。

### 本轮（m3951）修正的上一轮两处缺陷

1. **无依据放宽上界**：最粗网格（subdivision 0）旋转耦合，分支登记 `kind="upper", 1e-30`
   （实测 2.49e-32，40× 裕度，本来就能 PASS），上一轮却放宽到 `1e-20`（弱化 10 个数量级）。
   已改回 **1e-30**。
2. **键名与网格不符**：该行键名 `submerged_sphere_40_rotational_coupling` 的 “40 面板”在表 12
   中不存在（层 0 = 20 面板，兄弟行为 `_20_/_80_/_320_`）。已改为
   `submerged_sphere_20_rotational_coupling`，并同步更新测试、报告生成器与报告正文。

### 一处**被接受**的元数据变化（非内容丢失）

main 的 32 条 3D 行标 `tier="fast"`，并集后统一为 `standard`（分支守卫的要求）。判定为可接受：
① 方向单调（更贵，不会更便宜地跳过）；② 这些行的 `parameters` 都要求整条细化研究
（`subdivisions: [1,2,3]` / `[0,1,2]`），把它标 fast 与成本不符；③ `check()` 默认
`tiers=("fast","standard")`，且默认 pytest 跑 standard 层，登记行仍在默认扫描内被重算。
审计脚本对该模式做机器规则：只允许“不变贵则失败”。

## 3. 证据

### 3.1 并集审计（`docs/m3264_merge/audit_union_independent.py`）

```
merge base  : 4f5b73e73c4e731582f2b050328ec5fcf62b4a86
main / head : origin/main / 535460966f9497155394c36f2227001849581c9a
paths       : PASS  (main-only 86, branch-only 15, shared 3)
main rows   : PASS  (62 main rows, 69 merged rows)
branch rows : PASS  (67/67 mapped)
test funcs  : PASS  (main 17, branch 18, merged 20, folded {test_3d_free_surface_rows_are_guarded_by_the_default_tiers: test_free_surface_3d_rows_live_on_the_standard_tier})
UNION AUDIT: PASS
```

四项都是机械判据：路径集与字节一致、main 62 行键/值/kind/tier/source 逐项相等、分支 67 行按
名字+数值双向匹配、测试函数集合及折叠子句。原始证据：`runs/m3951/union_audit.json`，
图：`docs/m3264_merge/fig_m3951_union_audit.png`。

### 3.2 登记值重算（standard 层，69 条 3-D 行）

```
checked 69, passed 69, drifted 0
worst_value_relative_deviation = 3.79e-3   （等式行，容差 1.5e-2）
```

逐行明细：`runs/m3951/fs3d_check.json`（含 measured / relative_deviation / verdict）。

### 3.3 pytest

- `tests/test_declared_numbers.py` 廉价子集（注册表结构、fast 层重算、JSON 契约、判据机械、
  扰动即 DRIFT）：**10 passed / 10 deselected**（本机负载 >780，含 3-D 求解器的 standard 层用例
  在预算内未跑完，见报告 §5 的如实说明）。
- 全仓 `tests/` 在 main 基线与合并树上各有 2 个**环境性**收集错误
  （`test_cli_neural.py`、`test_neural_pipeline.py` 缺 `safetensors`），两侧一致，与本次合并无关。

## 4. 适用范围与局限

- 本解法只覆盖 `declared_numbers` 注册表与其测试；`agent/m3264` 的其余产物（`docs/m3264/`、
  `dev_report_m3264_*.md`、`docs/marine_bem_benchmarks.md` 的表 10–12 段落）为单侧新增，原样入树。
- 分支生成器被删除：其“程序化生成 + 覆盖映射”的意图以 `audit_union_independent.py` 与
  `registry_coverage.json` 的形式保留为**可执行证据**，而不是留在注册表里与 main 的字面量重复。
- 上界行（`kind="upper"`）的 `relative_deviation` 是“到界的距离”，不是误差；headline 误差统计
  只用等式行（`worst_value_relative_deviation`）。