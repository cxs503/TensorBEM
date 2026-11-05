# PR #128 治愈报告：agent/m3181 → main 三方合并（三维自由液面声明数字注册表）

任务：TensorBEM PR #128（分支 `agent/m3181`）对 main 变脏（`mergeable=false`），
需三方合并、逐文件语义解冲、本地核心测试不低于 main 基线、推回同分支。

| 项 | 值 |
|---|---|
| 仓库 | `cxs503/TensorBEM` |
| PR | #128 |
| 源分支 | `agent/m3181` |
| 目标分支 | `main` |
| merge base | `4f5b73e73c4e731582f2b050328ec5fcf62b4a86` |
| 旧分支 SHA | `55db3976dc283e9522334f2ab7309fc655f3ae27` |
| main SHA（合并时） | `ecba798833d6c2429fbc897d2ce515efb70dd682` |
| 治愈分支（本地） | `heal/m3913` |
| 合并提交 SHA | `f24aa65434a2f72ae2066997c8ebb9fcdc76da30` |
| 新 SHA（承载本治愈分支的平台分支 `agent/m3913`） | 同上 `f24aa65` |

> 推回限制（如实记录）：本工作树的安全策略拒绝对远端执行 `git push`
> （`REJECTED: 命令被安全策略拦截`），因此远端的 `agent/m3181` 分支仍停在旧 SHA
> `55db397`。治愈后的合并提交已落在本地 `heal/m3913`，并已快进（fast-forward）
> 到平台自动推送的分支 `agent/m3913`（`f24aa65`）。由于该合并提交的第一父正是
> `agent/m3181` 的旧 SHA `55db397`，把 `agent/m3913` 合入 main 与把 PR #128 合入
> main 产生**完全相同的树**——PR #128 的全部内容随该分支入库，不丢任何内容。

## 1. 冲突面

`git merge origin/main` 的文本冲突只有 3 个文件、5 个 hunk；其余 89 个文件由 git
自动合并（`docs/marine_bem_benchmarks.md` 属"自动合并但需人工复核"：main 改了表 10/12
的数字与订正说明，本分支在同一文件的 §4.12 追加了注册表复算表）。

| 文件 | 冲突 hunk | 冲突实质 |
|---|---:|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 3 | ①两个父提交为同两个 benchmark 模块各起了不同的本地别名；②主元组尾部（main 追加了表 10–12 的 22 条内联 `DeclaredNumber`，本分支追加了 `_free_surface_3d_rows()` 生成器）；③`_run_group` 两套互不相交的分派分支 |
| `tests/test_declared_numbers.py` | 2 | ①同一处"允许非正声明值"的注释两种措辞；②`test_registry_covers_every_documented_benchmark_family` 的分组集合（各自新增了两个组名） |
| `docs/marine_bem_benchmarks.md` | 0（自动合并） | main 的 m3150 订正段 + 本分支 §4.12 段，落在不同位置；需人工复核两侧表格数字都在 |

**根因**：两个父提交各自独立地把 `docs/marine_bem_benchmarks.md` 表 10–12 的 3-D
自由液面基准数字变成"可执行声明数字"。分歧是**加性**的（同一学术目标、两套注册方案），
因此正确解法是**语义并集**，而非任一侧覆盖。

## 2. 逐文件解法

### 2.1 `src/tensorbem/benchmarks/declared_numbers.py`

| 冲突 | 解法 |
|---|---|
| 模块别名 | 并集：保留 4 个别名（`hemisphere_benchmark`/`hulme_benchmark` 同指 `floating_hemisphere`，`sphere_benchmark`/`wang_linton_benchmark` 同指 `submerged_sphere_wave`）。两侧后续代码各自使用自己那一套，别名并集使两侧代码都无需改写 |
| 注册表主元组 | `main 的 22 条内联行` + `) + _free_surface_3d_rows()`。两个注册家族都保留：内联家族（`hulme_hemisphere` / `submerged_sphere` / `hulme_chief`）与本分支生成器家族（`hulme3d` / `hulme3d_chief` / `wang_linton3d`） |
| `_run_group` 分派 | 并集：先本分支三个 `hulme3d*` / `wang_linton3d` 分支，再 main 的 `hulme_hemisphere` / `submerged_sphere` / `hulme_chief` 分支。六个 `group ==` 分支互不相交 |
| **静默命名冲突**（非文本冲突，自动合并会埋雷） | 两侧各定义一个 `_chief_rows`：本分支是**零参**注册行构造器（被 `_free_surface_3d_rows()` 调用），main 是**单参**研究结果展平器。自动合并后 Python 只保留后出现的那个 → 导入期 `TypeError`。解法：把 main 的展平器改名 `_chief_study_rows` 并同步其唯一调用点（`hulme_chief` 分支）。这是本次合并唯一必须改名的符号 |

### 2.2 `tests/test_declared_numbers.py`

* 注释：把两侧措辞融合成一句（都保留信息：无正则化 3-D BIE 产出非物理负垂荡阻尼、
  表 11 正是把它当作 CHIEF 要消除的缺陷，因此只禁止 0 值）。
* 分组集合：`{hulme3d, hulme3d_chief, hulme_chief, hulme_hemisphere}` 并集（排序去重），
  原有 `cylinder/ellipsoid/interior_harmonics/submerged_sphere/ursell/ursell_reference/wang_linton3d` 保留。

### 2.3 `docs/marine_bem_benchmarks.md`

自动合并结果经双向复核后直接采用：相对 `origin/main` 仅多出本分支 §4.12（94 行），
相对 `agent/m3181` 仅多出 main 的 m3150 订正段与表 10/12 数字订正（8 行），两侧无覆盖。

## 3. 零静默丢失验证

`docs/m3913/verify_union.py` 把"两个父提交相对 merge base 各自新增的每一行"逐行在合并
结果中查找，只在命中**已登记规则**时允许改写：

| 规则 | 语义 |
|---|---|
| R1_imports | 别名并集，单个别名行不再逐字存在 |
| R2_comment_fused | 测试注释融合为一句 |
| R3_registry_tail | 主元组尾 `)` → `) + _free_surface_3d_rows()` |
| R4_group_set | 测试分组字面量改为排序集合 |
| R5_rename | `_chief_rows(rows)` → `_chief_study_rows(rows)`（含调用点） |

验证结果：

```
declared_numbers.py        main 新增 654 行 kept=654 rule-rewritten=2 lost=0
                          branch 新增 249 行 kept=249 rule-rewritten=0 lost=0
test_declared_numbers.py   main 新增  49 行 kept= 47 rule-rewritten=2 lost=0
                          branch 新增  93 行 kept= 91 rule-rewritten=2 lost=0
marine_bem_benchmarks.md   main 新增   8 行 kept=  8 rule-rewritten=0 lost=0
                          branch 新增  94 行 kept= 94 rule-rewritten=0 lost=0
verdict: PASS - zero silent loss
```

合并后注册表：**147 条声明数字、147 个唯一 key、11 个 group**（内联 + 生成器两族）。

## 4. 回归守卫

`tests/test_merge_union_contract_pr128.py`（12 个用例，<1 s，不解任何基准）钉死四种
再合并会静默丢内容的形态：

1. 任一注册家族从 registry 消失；
2. 两套 key 方案并集后撞 key；
3. 同一顶层符号被定义两次（本次 `_chief_rows` 阴影正是此类）；
4. 已注册 group 在 `_run_group` 中没有分派分支；
   另加文档双侧锚点（main 的 m3150 订正段 + 本分支 §4.12）与两套模块别名可导入。

## 5. 注意点 / 局限

* `push` 被安全策略拦截，远端 `agent/m3181` 仍为旧 SHA；治愈树经平台分支 `agent/m3913`
  入库，与 PR #128 合入 main 等价（第一父即旧 SHA，树内容完全一致）。
* 本机负载极高（load average ≈ 700），测试墙钟时间波动巨大，只以**通过/失败计数**作判据，
  不以耗时为判据。
* heavy 层（`TENSORBEM_HEAVY_TESTS=1`）默认不跑；本治愈未重算任何声明数字的物理值，
  也未调任何容差——合并只做内容并集。
* 后端/合并类任务，无场量云图；结果图为"并集账目 + 测试结果"统计图
  （`docs/m3913/figures/fig_pr128_union.png`），数据由脚本程序化注入，零手抄。

## 6. 测试证据（实测）

同一组精选核心测试，在 main 基线树（`/tmp/wt_main`，SHA `ecba798`）与治愈树
（`f24aa65`）各跑一遍，判据为通过/失败计数（非耗时；本机 load≈700）：

| 树 | 用例数 | 结果 | 墙钟 |
|---|---:|---|---:|
| main 基线 `ecba798` | 23 | **23 passed, 0 failed** | 71.2 s |
| 治愈树 `f24aa65` | 36 | **36 passed, 0 failed** | 92.6 s |

治愈树比基线多出 13 个用例，全部来自本次合并新增的守卫与用例：
`tests/test_merge_union_contract_pr128.py` 12 个（并集契约）
+ `test_three_dimensional_free_surface_tables_are_fully_registered` 1 个（本分支原有、
main 上不存在）。因此**治愈树 ⊇ 基线，无回归、无失败**。

同时验证（命令与结果）：

* `python3 docs/m3913/verify_union.py --repo . --merged HEAD` → `PASS - zero silent loss`（六条统计全 0 lost）。
* 注册表体检：`147` 条声明数字 / `147` 个唯一 key / `11` 个 group；`ast` 解析无重复顶层定义、无语法错误。

局限（如实）：`tests/test_declared_numbers.py` 中会真跑基准的慢用例（fast/standard 层复算、
1280 面元行、`test_cli_writes_strict_json_and_passes`）未能在本机本轮完整跑完
——单是 `test_fast_tier_reproduces_the_documented_numbers` 在高负载下就要 92 s，
整文件在 115 s 限额内只推进 4–8 个用例。它们不涉及本次冲突的语义（合并只做注册表并集，
不改任何声明值/容差），但要作为遗留项明示：完整 heavy/慢层复算需在空闲机上补跑。
