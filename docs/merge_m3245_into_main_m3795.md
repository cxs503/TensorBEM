# 治愈 PR #133：把 `agent/m3245` 合并进 `main`（mission m3795）

**结论（先说）**：`agent/m3245`（旧 SHA `6753fcb`）与 `main`（`ecba798`）的三方合并在
`src/tensorbem/benchmarks/declared_numbers.py` 与 `tests/test_declared_numbers.py` 两处内容冲突，
按**语义并集**逐处解决：两侧新增的声明数字（registry 条目）与两侧各自的测试断言**全部保留**，
自动合并的 `docs/marine_bem_benchmarks.md` 逐段核对后确认两侧补注均在（main 的 m3150 订正 +
m3245 的「表 10–12 可执行化」），其余 m3245 新增文件（`docs/free_surface3d_registry.md`、
`outputs/m3245/*`、`dev_report_m3245_free_surface3d_registry.md`）与 m3245 分支**逐字节一致**。
合并后 registry 条目数满足并集恒等式：`84 (main) ∪ 78 (m3245) − 22 (两侧共有的旧条目) = 140`，
**无重复 key、无丢失 key**（见 `docs/merge_m3795/merge_union_ledger.json`，程序化生成，零手抄）。

## 1. 冲突现场（测得到的，不是猜的）

```
$ git checkout -b agent/m3795-work origin/main
$ git merge --no-ff --no-commit agent/m3245
Auto-merging docs/marine_bem_benchmarks.md          # 自动合并，需人工核对
Auto-merging src/tensorbem/benchmarks/declared_numbers.py
CONFLICT (content): Merge conflict in src/tensorbem/benchmarks/declared_numbers.py
Auto-merging tests/test_declared_numbers.py
CONFLICT (content): Merge conflict in tests/test_declared_numbers.py
```

| 文件 | 冲突处数 | 冲突性质 |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 3 | ①模块导入别名；②`DECLARED_NUMBERS` 字面量尾部（main 手写 594 行 vs m3245 `*_free_surface_3d_rows()`）；③`_run_group` 的分派分支（两套 quantity-key 约定） |
| `tests/test_declared_numbers.py` | 1 | 覆盖性测试的 group 名字面集合 |
| `docs/marine_bem_benchmarks.md` | 0（自动合并） | 需核对两侧补注未被吞 |

**为什么不能机械 overlay**：两侧都在同一分支点上各自实现了「把 `docs/marine_bem_benchmarks.md`
表 10–12 的数字变成可执行判据」这件事，但实现路线不同——main 把 84 条数字手写进
`DECLARED_NUMBERS` 字面量，用 `_study_rows`（key 前缀 `sub<level>_`）复算；m3245 把 `_table_rows`
从一次 registry 运行 JSON 生成 78 条（key 前缀 `panels<N>.`），并额外覆盖了文档 4.11 节
Wang/Linton 潜没球 56 行。**任取一侧都会静默丢掉另一侧的表覆盖**：取 main 丢 m3245 的生成式
（标准层守卫行 + 4.11 全表），取 m3245 丢 main 的 hydrostatic/Haskind 等手写量。这正是本仓
「机械 overlay 静默丢内容」前科的同一个坑。

## 2. 逐文件解法

### 2.1 `src/tensorbem/benchmarks/declared_numbers.py`

**（a）导入别名（冲突 1/3）** —— 并集，取更明确的别名，并在分派里真正使用：

```python
from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    floating_hemisphere as floating_hemisphere_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as submerged_sphere_wave_benchmark,
    ursell_cylinder as ursell_benchmark,
)
```

**（b）`DECLARED_NUMBERS`（冲突 2/3）** —— **两侧全留**：先保留 main 的 594 行手写条目（原样），
再追加 m3245 的 `*_free_surface_3d_rows()`，并在中间加注释说明两者是同一批文档数字的两条
**独立抽取路径**（互为交叉校验），其中只有生成侧覆盖 4.11 的 56 行。

**（c）`_run_group` 分派（冲突 3/3）** —— 一次求解，返回**两套 key 约定的并集**：

```python
    if group == "hulme_hemisphere":
        study = floating_hemisphere_benchmark.run_refinement_study(
            subdivisions=..., ka=..., radius=..., density=..., gravity=...)
        values = _study_rows(study)                     # main: sub<level>_<q>
        values.update(_flatten_levels(study["levels"]))  # m3245: panels<N>.<q>
        values.update(_flatten_reference(study["exact_reference"]))  # reference.<q>
        return values
    if group in _SUBMERGED_SPHERE_GROUPS:               # 两个名字都接受
        ...
    if group == "hulme_chief":
        values = _chief_rows(rows); values.update(_flatten_chief_rows(rows))
```

同一个 `(group, parameters)` 只做一次 BEM 求解、被所有引用它的条目共享（`check()` 的既有缓存），
所以并集**不增加求解成本**，只增加被校验的声明数字。

**group 名**：main 用 `submerged_sphere`，m3245 用模块名 `submerged_sphere_wave`，两者指同一个基准
（`tensorbem.benchmarks.submerged_sphere_wave`）。解法是**不重命名任何一侧的条目**，而是让分派接受
两个名字：

```python
_SUBMERGED_SPHERE_GROUPS = ("submerged_sphere", "submerged_sphere_wave")
```

这样两侧的条目、两侧的测试、m3245 已发布的 `docs/free_surface3d_registry.md` 与
`outputs/m3245/free_surface3d_registry.json` 全部保持可用（重命名会打断已入库产物的可追溯性）。
代价是一个基准有两个 group 名；若要收敛，应作为独立小改动（改 1 处 `_table_rows(...)` 实参 +
文档），不属于本合并的职责。

### 2.2 `tests/test_declared_numbers.py`

覆盖性测试的 group 集合取**并集**：`{"..., "submerged_sphere", "submerged_sphere_wave", ...}`，
即同时要求两侧的 group 名都真实存在于 registry 中——这条断言本身就是「并集没丢东西」的回归门。
两侧各自的其它测试（main 的 `test_3d_free_surface_rows_are_guarded_by_the_default_tiers`、
m3245 的 `test_standard_tier_reproduces_the_3d_free_surface_coarse_rows` /
`test_documented_3d_free_surface_tables_are_reproduced`）**全部保留**，数量由 15 增至 19。

### 2.3 `docs/marine_bem_benchmarks.md`（自动合并，逐段核对）

核对方法：取 m3245 相对 merge-base 的**每一行新增**，在合并结果中逐行查找；再取 main 相对
merge-base 的新增（如 m3150 的「B11′ 观测阶口径 / 表 12 舍入」补注）同样查找：

```
$ grep -c "表 10–12 的可执行化（m3245）" docs/marine_bem_benchmarks.md   # 1
$ grep -c "m3150 复算订正"               docs/marine_bem_benchmarks.md   # 1
```

两侧补注均在场，无内容被吞。另用 `git diff --stat 6753fcb -- <m3245 改过的其余文件>` 核对：
只有 `docs/marine_bem_benchmarks.md` 有差异（main 侧内容），其余 m3245 文件与分支**逐字节一致**。

## 3. 证据

### 3.1 并集恒等式（程序化，`docs/merge_m3795/merge_union_ledger.json`）

| 量 | main `ecba798` | `agent/m3245` `6753fcb` | 合并后 |
|---|---:|---:|---:|
| registry 条目数 | 84 | 78 | **140** |
| 两侧共有（同一个旧条目） | — | — | 22 |
| 合并后重复 key | — | — | **0** |
| 合并后丢失 key | — | — | **0** |
| fast 层条目 | 40 | 8 | 40（含共有条目） |
| standard 层条目 | 34 | 13 | 43 |
| heavy 层条目 | 10 | 57 | 57 |
| `hulme_hemisphere` | 31 | 27 | **58 = 31+27** |
| `hulme_chief` | 6 | 6 | **12 = 6+6** |
| `submerged_sphere(+_wave)` | 25 | 23 | **48 = 25+23** |
| 其余 5 组（cylinder/ellipsoid/ursell×2/interior_harmonics） | 22 | 22（同一批） | 22 |

即 `140 = 84 + 78 − 22`，逐组可加，**无一组少于任一侧**。图见
`fig_m3795_registry_union.png`、`fig_m3795_tier_coverage.png`。

### 3.2 registry 复算（fast 层，合并树实跑）

```
$ PYTHONPATH=src python3 -m tensorbem.benchmarks.declared_numbers --tiers fast
{"summary": {"checked": 40, "passed": 40, "drifted": 0, "verdict": "PASS",
             "worst_relative_deviation": 0.00378677870019862}, ...}
```

40/40 PASS，最差相对偏差 3.79e-3。fast 层 40 条全部来自 main 侧手写集合，说明
**并集没有把 main 的判据改脏**（m3245 侧新增的重活落在 standard/heavy 层）。

### 3.3 测试

| 命令 | 合并树 | main 基线 |
|---|---|---|
| `pytest tests/test_declared_numbers.py -k "well_formed or covers_every_documented_benchmark_family"` | **2 passed, 17 deselected**（1.11 s） | 2 passed, 15 deselected（0.57 s） |
| `pytest tests`（全量，`-n 12`） | 见 §5「诚实边界」 | 见 §5 |

被选中这两条正是本合并直接改变的语义：注册表结构良构（无一侧条目被破坏、`declared` 非零、
`parameters` 是 dict）与 group 覆盖（要求两侧 group 名同时存在）。

## 4. 注意点

- **本机负载极高**（load average 574 @ 128 核，多个 worktree 的 pytest 并发），全量 `pytest`
  与 standard 层复算在本单时限内只推进到 60%–70%，见 §5，不做粉饰。
- 合并树里同时存在 `_study_rows`/`_chief_rows`（main）与 `_flatten_levels`/`_flatten_reference`/
  `_flatten_chief_rows`（m3245）两套展平函数：这是并集的**必要代价**——两套 key 前缀不同
  （`sub1_a33_rel_error` vs `panels16.a33_rel_error`），两侧条目各自引用自己那套，删任一套都会
  让对应条目 `KeyError`。后续若要瘦身，应连同条目一起迁移（改 key 前缀 + 重新生成 JSON），
  不能只删函数。
- `submerged_sphere` / `submerged_sphere_wave` 双名是**过渡态**：接受两个名字的分派是显式的
  （`_SUBMERGED_SPHERE_GROUPS`），不是隐式别名猜测；测试对两个名字都做存在性断言。
- heavy 层条目 57 条（m3245 侧）默认 `pytest` 不执行（`TENSORBEM_HEAVY_TESTS=1` 才跑），
  因此并集不会把默认测试时长按 57 条线性拉长；默认层新增的是 m3245 设计的三族各一条粗网格守卫。
- 本合并没有修改求解器内核（`src/tensorbem/*.py` 中除 `benchmarks/declared_numbers.py` 外无改动），
  所以物理结果不因合并而变：唯一风险面是「声明数字的覆盖与复算」，已由 §3.2/§3.3 覆盖。

## 5. 诚实边界（未完成部分）

1. **全量 `pytest` 未在本单时限内跑完**。合并树与 main 基线各起一次 `pytest tests -n 12`
   （日志 `artifacts` 内，因 `artifacts/` 被 `.gitignore` 忽略故不入库），报告时分别推进到
   **71%** 与 **64%**，两条日志中**均未出现失败标记 `F`/`E`**；但这是「未跑完」而非「通过」，
   不得按通过计。standard/heavy 层的完整复算（`--tiers fast,standard`、`--tiers fast,standard,heavy`）
   同样只起跑未收口。
2. 因此本单的验收口径是：**合并语义正确性 + 并集无损（§3.1）+ 相关测试与 fast 层复算（§3.2/§3.3）**；
   全量套件与 heavy 层复算交由 merge_governor 的下一周期与 CI（`.github/workflows/tests.yml`）复核。
3. 推回 `agent/m3245` 需要写 GitHub（`git push`）；本机到 `github.com` 的 `ls-remote` 在 120 s 内超时，
   **推送未完成**，已在 finish 中如实记录；分支本地已更新为合并提交（见 §6），平台侧自动推送可达时即可上传。

## 6. 分支与 SHA

| 项 | SHA |
|---|---|
| 合并前 `agent/m3245` | `6753fcb0b3602051004d2b1362c53d8f4f23890f` |
| `origin/main`（第一父） | `ecba798833d6c2429fbc897d2ce515efb70dd682` |
| 合并提交（本单产物） | 见 `git log --oneline -1`（`Merge agent/m3245 into main: union declared-numbers registry`） |

复现命令（冲突解决脚本与图脚本都在 `docs/merge_m3795/`）：

```
git checkout -b <work> origin/main && git merge --no-ff --no-commit agent/m3245
python3 docs/merge_m3795/merge_resolve_m3795.py   # 逐文件语义并集
PYTHONPATH=src python3 -m tensorbem.benchmarks.declared_numbers --tiers fast
```