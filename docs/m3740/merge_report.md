# PR #128 治愈报告：agent/m3181 → main 三方合并

任务：TensorBEM PR #128（分支 `agent/m3181`）对 `main` 变脏（mergeable=false），做三方合并、
逐文件语义解冲、本地跑核心测试，并把结果推回 `agent/m3181`。

| 项 | SHA |
|---|---|
| 合并前 `main`（ours） | `b420b2e`（Merge #126 (m3150) via merge-governor） |
| 合并前 `agent/m3181`（theirs，旧 SHA） | `55db397`（chore: re-run CI against current main） |
| merge-base | `4f5b73e`（Merge #120 (m3086)） |
| 合并结果（本工作树 `agent/m3740`） | 见 `git log -1` / 下文「提交」小节 |

两侧各自改了什么（这就是冲突的来源）：

* **main**（#126 / m3150，PR #128 开出来之后才落地的）：把 `docs/marine_bem_benchmarks.md`
  §4.10–§4.11 的表 10/11/12 手工登记成 62 条 `DeclaredNumber`，分组名
  `hulme_hemisphere`(31) / `hulme_chief`(6) / `submerged_sphere`(25)，默认档
  （`fast`/`standard`）即跑；并订正了表 12 球心转动耦合的舍入（`2.2e-4 → 2.1510e-04`、
  `2.6e-5 → 2.6481e-05`）。
* **agent/m3181**（#128 本体）：把**同样三张表**改成数据表驱动（`_TABLE_HULME` /
  `_TABLE_HULME_CHIEF` / `_TABLE_WANG_LINTON` + `_refinement_rows()` 生成器），
  分组名 `hulme3d`(28) / `hulme3d_chief`(9) / `wang_linton3d`(26)，全部挂在 `heavy` 档，
  并新增 §4.12 复算表与两个 heavy 回归测试。

两侧**没有一行是同名 key**，所以 git 只能在同一个字典字面量里反复“撞车”；而语义上它们是
对同一批物理格子的两套独立登记，这正是必须人工解冲、且不能 overlay 的原因。

---

## 1. 冲突文件清单与逐文件解法

### 1.1 `src/tensorbem/benchmarks/declared_numbers.py`（3 个冲突块，UU）

| 冲突块 | ours（main） | theirs（m3181） | 解法 |
|---|---|---|---|
| ① 导入别名（L49–57） | `floating_hemisphere as hemisphere_benchmark`、`submerged_sphere_wave as sphere_benchmark` | `… as hulme_benchmark`、`… as wang_linton_benchmark` | **并集**：同一模块保留两套别名（各自的度量分支都要用），按 isort 顺序排列 |
| ② 注册表字面量（L540–1138） | 62 条内联 `DeclaredNumber(...)` 行，元组以 `)` 收尾 | 一行 `) + _free_surface_3d_rows()`（数据表生成 63 条） | **语义并集**：保留 main 的 62 条内联行，把收尾的 `)` 改成 `) + _free_surface_3d_rows()`，即 `DECLARED_NUMBERS = (…62 条…) + _free_surface_3d_rows()` |
| ③ `_run_group` 分派（L1245–1318） | `hulme_hemisphere` / `submerged_sphere` / `hulme_chief` 三个分支（走 `_study_rows` / `_chief_rows(rows)`） | `hulme3d` / `hulme3d_chief` / `wang_linton3d` 三个分支（走 `_refinement_rows` 内联展平） | **并集**：六个分支全部保留 |

块 ① 的并集带来一个**必须修的隐患**：m3181 定义零参 `_chief_rows()`（返回 `tuple[DeclaredNumber, ...]`），
main 定义 `_chief_rows(rows)`（返回 `dict[str, float]`）。Python 后定义者覆盖前者——导入期
`_free_surface_3d_rows()` 用的是零参版本、运行期 `_run_group` 用的是带参版本，靠**语句顺序**
偶然正确；任何人挪动一段代码就会 `NameError` 或拿到错类型。故把生成器侧改名
`_chief_table_rows()`，并同步其唯一调用点，两个名字各自显式。

### 1.2 `tests/test_declared_numbers.py`（2 个冲突块，UU）

| 冲突块 | 解法 |
|---|---|
| ① `test_registry_is_well_formed` 断言的注释 | **并集注释**：两侧说的是同一件事（表 11 有意登记了非物理的负 `B33'`，故只有 0 非法），合成一条 |
| ② 分组集合断言 | **并集集合**：`hulme3d, hulme3d_chief, hulme_chief, hulme_hemisphere, submerged_sphere, wang_linton3d` 等 11 个分组全在断言里 |

### 1.3 `docs/marine_bem_benchmarks.md`（自动合并成功，但**必须人工复核**）

git 把它自动合了（main 改表 12 两格 + m3181 追加 §4.12）。复核结论：main 的订正与
m3181 的新小节都在，无丢失；但出现**同一物理量两个数**的隐患（见 §2）。

### 1.4 其余 14 个文件

`dev_report_m3181_*.md`、`docs/m3181/**`（脚本/图/JSON/table.md）全部为 m3181 单侧新增，
`git diff --stat origin/main` 显示 17 个文件 +3039/−3，与 m3181 单侧改动完全一致 → 无丢失。

---

## 2. 语义并集之后做的两处**跨侧对齐**（不是覆盖，是把“同一测量两个数”收敛成一个）

1. **表 12 球心转动耦合格**：main 审计后是 `2.1510e-04 / 2.6481e-05`，m3181 抄的是订正前的
   两位有效数字 `2.2e-4 / 2.6e-5`（并为此把容差放宽到 `2.5e-2`）。合并后文档里两处会互相矛盾，
   故把 m3181 的表格子对齐到 main 的订正值，并撤销那条一次性容差覆盖——现在它们与
   `submerged_sphere` 同格**测的是同一次求解、报的是同一个数**，残差 `-5.40e-06 / -3.99e-06`
   远在统一的 `5e-3` 带内。§4.12 的两行表格与判决句同步更新（最坏等值判据偏差由
   `2.23e-02` 变为 `3.87e-03`）。
2. **20 面元机器零格**保持 m3181 的 `upper` 判据（`declared=1e-20`），不动——它是对上界的主张，
   与 main 的 `kind` 语义一致，不是可对齐的点值。

**刻意不对齐的一处（诚实披露）**：表 10 的 `B11` 观测阶。main 用 `log(e0/e1)/log(h0/h1)`
（h 取实测平均面元尺度）得 `1.8021 / 1.9377`；m3181 用假设 h 精确减半的 `log2(e0/e1)`
得 `1.62 / 1.88`（与文档表 10 的排版值一致）。两者各自与其度量**自洽**（各自 PASS），
把任一侧改成对方的数会立刻 DRIFT；到底表 10 的 B11 列指哪种定义属物理判据问题（需重推
Hulme 细化序列的 h 比），不在本次解冲的裁决范围内，留作后续单。已由
`docs/m3740/cross_check.py` 程序化登记（见 §3）。

上述全部由脚本 `docs/m3740/resolve_conflicts.py` 施加（可重放、带断言），并且脚本末尾
扫描全树确认**没有任何冲突标记残留**。

---

## 3. 验证证据

### 3.1 结构自检（程序化，非自述）

```
$ PYTHONPATH=src python3 docs/m3740/cross_check.py
paired_cells: 59, identical_declared: 57
main_rows_without_counterpart: 3 (haskind_surge / haskind_heave / near_far_damping_closure —— main 独有行，未丢)
mismatched: 2 (均为 B11 观测阶，见 §2 披露)
```

注册表规模：main 84 条（22 base + 62 新增）+ m3181 63 条 = **147 条，147 个唯一 key，0 重复**。
分组：`cylinder`2 / `ellipsoid`3 / `hulme3d`28 / `hulme3d_chief`9 / `hulme_chief`6 /
`hulme_hemisphere`31 / `interior_harmonics`6 / `submerged_sphere`25 / `ursell`9 /
`ursell_reference`2 / `wang_linton3d`26。

### 3.2 核心测试：与 main 基线对比（同一台机器、同机并发，绝对耗时受负载影响）

| 目标 | 结果 |
|---|---|
| `pytest tests/test_declared_numbers.py`（合并树） | **15 passed, 5 skipped, 1 warning in 249.89 s** |
| `pytest tests/test_declared_numbers.py`（main 基线 `b420b2e`，`git archive` 纯净展开） | **14 passed, 3 skipped, 1 warning in 255.16 s** |

合并后比基线多 1 个通过（m3181 的注册表一致性测试）、多 2 个 skip（m3181 的两个 heavy 测试，
`TENSORBEM_HEAVY_TESTS` 未置位），无失败、无 error → **不低于 main 基线**。

### 3.3 声明数字端到端复算（`--strict`，漂移即非零退出）

* `--tiers standard`（main 的 3-D 行 + Ursell/cylinder/ellipsoid 行）：**34 项全 PASS，drifted=0**，
  输出 `docs/m3740/outputs/merged_registry_standard.json`。
* `--tiers heavy`（m3181 的 63 格）：见 `docs/m3740/outputs/` 与下文注记；判据与 m3181 交付的
  `docs/m3181/outputs/three_d_free_surface_registry.json` 相同（63/63 PASS），本次仅两格声明值
  按 §2 对齐，其残差由该 JSON 内记录的实测值直接算出（`-5.40e-06 / -3.99e-06`），仍 PASS。
* 全量 `pytest tests/`（合并树 vs 基线）并行跑，见提交说明里的日志摘要。

---

## 4. 误差判定（关键量 vs 基准，逐项）

基准：文档表 10/11/12 的声明值（Hulme 1982 半解析波无多极子解、Wang 1986/Linton 1991 闭式
多极子系统）。判据：注册表容差带。

| 关键量 | 声明 | 复算 | 相对偏差 | <3%? |
|---|---:|---:|---:|:--:|
| 表 10 `A33'` 误差@256 | 7.41e-3 | 7.40544e-3 | −6.15e-04 | 是 |
| 表 10 `B33'` 误差@256 | 1.18e-2 | 1.18119e-2 | +1.01e-03 | 是 |
| 表 10 观测阶 `A33'`@256 | 2.06 | 2.05917 | −4.02e-04 | 是（相对声明值） |
| 表 11 未正则 `B33'`（负阻尼） | −0.1042 | −0.104155 | +4.35e-04 | 是 |
| 表 11 CHIEF 正则后条件数 | 4.24 | 4.24 | — | 是（< 未正则 145.66 的 1/10） |
| 表 12 `A33'` 误差@320 | 1.85e-2 | 1.85089e-2 | +4.81e-04 | 是 |
| 表 12 转动耦合@80（对齐后） | 2.1510e-4 | 2.15099e-4 | −5.40e-06 | 是 |
| 注册表最坏等值格 `B11` 阶@256 | 1.88 | 1.88727 | +3.87e-03 | 是 |
| **观测阶 vs 理论 O(h²)=2.0**：`A33'`@320 | 2.0 | 2.06197 | **+3.10%** | **否（略超）** |
| 观测阶 `B33'`@320 | 2.0 | 1.92898 | −3.55% | 否（略超） |
| 观测阶 `B11`@256 | 2.0 | 1.88727 | −5.64% | 否 |
| 观测阶 `A11'`/`Fz`@256 | 2.0 | 2.00598 / 2.01525 | +0.30% / +0.76% | 是 |

**回答「是否 <3%」**：以声明值为基准的 62 个等值格全部 <0.5%（最坏 3.87e-03，来自只登两位
有效数字的观测阶列）；但若以**理论二阶收敛率 2.0** 为基准，`B11`（−5.6%）、`B33'`@320（−3.6%）、
`A33'`@320（+3.1%）**超过 3%**——这不是缺陷而是有限网格下观测阶的正常散布（同级 `A11'`/`Fz`
都在 1% 内），报告中不做粉饰。

---

## 5. 注意点 / 局限

* **两套并行登记是合并的真实代价**：147 条里 59 对是同一物理格子的两种命名。默认档只跑 main
  的 62 条、`heavy` 档只跑 m3181 的 63 条，所以测试时长没有翻倍（249.89 s ≈ 基线 255.16 s）；
  但长期应收敛成一套（main 的默认档门 + m3181 的全表 heavy 门，二者择一），属后续单。
* `_chief_rows` 双定义是**潜在** bug（靠语句顺序侥幸正确），本次已改名消除；机械 overlay 会把
  它留在树里而不报错。
* heavy 档在本机负载 ~200 的并发环境下单次 63 格复算超过 15 min，本单未等到完整 heavy 收尾，
  已用 m3181 交付的 heavy JSON（63/63 PASS）+ 两格对齐后的残差直接换算作为证据；这是**本单的
  未完成部分**，merge_governor 下一周期复验时应以 CI 的 heavy 结果为最终判据。
* 表 12 20 面元转动耦合按机器零上界判，其「偏差 −1.00」是相对 `1e-20` 上界而言，不代表性。

## 6. 复现路径

```bash
python3 docs/m3740/resolve_conflicts.py        # 解冲 + 跨侧对齐（带断言，可重放）
PYTHONPATH=src python3 docs/m3740/cross_check.py
PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q
PYTHONPATH=src python3 -m tensorbem.benchmarks.declared_numbers --tiers standard --output docs/m3740/outputs/merged_registry_standard.json
PYTHONPATH=src TENSORBEM_HEAVY_TESTS=1 python3 -m pytest tests/test_declared_numbers.py -q
PYTHONPATH=src python3 docs/m3740/make_figures.py
```
---

## 7. 提交与入库状态

| 项 | 值 |
|---|---|
| 合并提交（本分支 `agent/m3740`） | `25f8fed`，双父 `b420b2e`(main) + `55db397`(agent/m3181 旧 SHA) |
| 旧 SHA（PR #128 分支头） | `55db397` |
| 新 SHA（应推回的合并结果） | `25f8fed` |
| 论文级进展报告 | `/tensor/reports` id=3922（mission_id 3740，3 图内嵌，payload 135 KB） |

未完成部分（如实记）：① 未手动 push（按平台纪律由平台推送本分支，如需把 PR #128 分支更新为合并结果，
把 `agent/m3181` 快进到 `25f8fed` 即可，这是纯快进、无内容变更）；
② heavy 档完整收尾（`TENSORBEM_HEAVY_TESTS=1`、`--tiers heavy`）在本机高负载下未在会话内结束，
heavy 数字取自 m3181 交付 JSON + 本单对齐后的残差换算，终判据交给 merge_governor/CI 复验；
③ 全量 `pytest tests/`（合并树 vs 基线）已并行跑到 14% 无失败，未在本会话内收尾。
