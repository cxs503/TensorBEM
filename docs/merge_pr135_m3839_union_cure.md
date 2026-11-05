# PR #135（`agent/m3264` → `main`）治愈：语义并集、被丢内容的复原与机器可验的并集审计（m3839）

## 1. 案例介绍（对象与拓扑）

PR #135 把 m3264 的两套「三维自由液面基准表 10–12 的声明数字登记」并入 `main`。
两边独立做了同一件事，命名空间不同，于是三方合并必然同组冲突；而机械 overlay
（任选一侧 / 强行取并集）会让另一侧的 62 或 67 条声明**全部判 DRIFT**——这不是
文本冲突，是语义冲突。合并拓扑（本仓本地快照）：

| 项 | 值 |
|---|---|
| 目标分支 `origin/main` | `ecba798`（PR #103） |
| 被合并分支 `agent/m3264` | `5b4dee7`（tensorbem: register the 3-D free-surface declared numbers, tables 10-12） |
| merge-base | `6025970` |
| 双方共同改动的文件 | 3 个：`src/tensorbem/benchmarks/declared_numbers.py`、`tests/test_declared_numbers.py`、`docs/marine_bem_benchmarks.md` |
| main 单边改动 / m3264 单边改动 | 117 / 15 个文件 |
| 本轮起点（前一次解冲的合并提交） | `cbcac1f`（m3797）→ 本轮新 SHA `33c74e2`（见 §6） |

物理/数值背景：表 10 = Hulme 浮半球（Ka=1，16/64/256 面元），表 11 = 内部 CHIEF
正则化（Ka=2.58，256 面元），表 12 = Wang/Linton 潜没球（Ka=0.8，f/a=1.5，
20/80/320 面元）。登记的行既判「离散误差是否被复现」，也判「观测阶、Haskind
互易、近/远场阻尼闭合」这些自洽恒等式。

参数（模块常量，两侧一致）：ρ=1025 kg/m³，g=9.80665 m/s²，a=1.0 m。

## 2. 怎么做的（方法/流程）

1. **独立复算三方合并**：`git merge-tree --write-tree ecba798 5b4dee7` 给出干净解候选
   与真实冲突（2 个文件、3 处）。逐文件核对 m3797 的既有解与候选解的差异。
2. **并集审计（新工具）** `tools/audit_union.py`：三层机器判据——
   (a) 文件层：任一侧新增的文件必须存在于合并树，任一侧删除的文件不得复活；
   (b) 行层：任一侧相对 merge-base 新增的行必须在合并结果里逐字出现，或由
   `tools/pr135_union_dispositions.json` 声明的**重命名/重写**处置并附「见证串」；
   (c) 注册表层：从各自树里 import `DECLARED_NUMBERS`，逐一核对键是否存在、值是否被改。
   未处置的差异即 DROP，verdict=`DROP_DETECTED`；本分支实测 `PASS`（0 findings，
   107 个文件、173 条声明逐条核对）。
3. **找并集漏掉的东西**：审计第一步就抓到 m3797 的机械 overlay **静默丢了一整块内容**
   ——m3264 给 `check()` 摘要新增的 `worst_value_relative_deviation`（只统计 equality 行的
   头条精度，避免单侧上界行的「到界距离」污染误差统计）在合并结果里彻底消失，
   而当时的测试没有一条会红。本轮已复原并加测试钉死。
4. **处理两侧对同一格引了不同印数**：`main` 在 m3150 复算后把球心转动耦合印数改成
   2.15e-4 / 2.65e-5，m3264 的 `*_cells` 读法仍引旧的两位有效数字 2.2e-4 / 2.6e-5。
   合并后这两套读法**引用同一份文档却声明不同的数**，是新的漂移源。按文档现行印数订正，
   并把「例外清单」从只列名单改成逐条判定（1% 印刷带 + kind 语义）。
5. **测试与证据**：fast 单测（含 2 条新测试）、`check()` 定向复算（Hulme 两组
   60 行/43 判据）、以及并集审计 JSON（`artifacts/pr135_union_audit.json`）。

## 3. 冲突文件清单与逐文件解法

| 文件 | 冲突 | 解法（并集语义） |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 4 处：导入别名、注册表块、扁平化助手、`_run_group` 分支 | 两套注册表**都留**：main 的 62 行（按面元数索引，组名 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`）原地保留；m3264 的 67 行生成器进 `*_cells` 命名空间（唯一的重命名，理由：两侧测试各自断言本块组名/行数/tier，重命名后两套断言同时成立）。`_run_group` 六个分支并存，两侧用**同一个** `_shared_study` 缓存键，因此同一次 refinement study 只解一次。 |
| `tests/test_declared_numbers.py` | 1 处（5 个 3-D 测试对撞） | 两侧 5 个测试**全保留**：main 的 `test_3d_free_surface_rows_are_guarded_by_the_default_tiers`、`test_chief_regularisation_restores_passive_heave_damping`，m3264 的 standard-tier 逐格复现、tier 守卫、漂移负控；新增 `FREE_SURFACE_3D_CELLS` 常量与 §4 的两条新测试。 |
| `docs/marine_bem_benchmarks.md` | 自动合并 | m3264 的 35 行表 10–12 说明 + main 的 m3150 订正段并存，审计确认双方新增行逐条在场。 |

## 4. 本轮修掉的缺陷（都是并集没盖住的地方）

1. **复原** `check()` 摘要字段 `worst_value_relative_deviation`（m3264 的功能，被 m3797 丢掉）。
   新测试 `test_the_headline_accuracy_figure_excludes_one_sided_bounds` 用一个 equality 行
   （误差 0）加一个上界行（测量值远低于界，|rel dev|=1）判别：`worst_relative_deviation=1.0`
   而 `worst_value_relative_deviation=0.0`。
2. **订正** `*_cells` 读法的球心转动耦合：`2.2e-4 → 2.15e-4`、`2.6e-5 → 2.65e-5`，
   与文档现行印数一致；审计在两个键上记录为**主动订正**（`declared_corrections`），不是静默改数。
3. **加固** `test_the_cells_registry_restates_cells_the_main_registry_already_guards`：
   原先只核对 9 个「例外」的名字，值不判。现在例外逐条判定——转动耦合须与四位有效数字行
   相差 <1%（负控：把 2.2e-4 放回去该测试立刻红，实测 FAILED），其余 6 条例外须保持
   上界/标志位的语义（`kind="upper"`、`tolerance=1e-12`、`DEFAULT_TOLERANCE`）。
4. 复原 `DeclaredNumber.group` 文档串中 m3264 的 runner 枚举（并补上 `*_cells` 三个组名）。

## 5. 证据（怎么验证的）

| 检查 | 命令 | 结果 |
|---|---|---|
| 并集审计（文件/行/注册表三层） | `python tools/audit_union.py --merge HEAD --ours ecba798 --theirs 5b4dee7 --dispositions tools/pr135_union_dispositions.json --registry --json artifacts/pr135_union_audit.json` | **PASS**，0 findings；main 84/84、m3264 89/89 条声明全部在场，22 条共有键 0 条被改值 |
| 注册表并集规模 | 同上 `layer_registry` | union 期望 151 = 实际 151；分组：`hulme_hemisphere` 31 + `hulme_hemisphere_cells` 29、`submerged_sphere` 25 + `submerged_sphere_cells` 29、`hulme_chief` 6 + `hulme_chief_cells` 9 |
| 新/改测试（fast，无需解算） | `pytest tests/test_declared_numbers.py -k "headline_accuracy or cells_registry_restates or upper_bound_claims or report_is_strict_json"` | **4 passed**（49 s，宿主 8 任务争用） |
| 负控 | 把 `2.15e-4` 改回 `2.2e-4` 后跑 `-k cells_registry_restates` | **FAILED**（说明加固后的判据真的能抓旧印数），随后复原 |
| 定向物理复算（合并树，本轮改动前） | `check(tiers=("standard",), entries=表10+表10_cells)` | 60 条中 43 条判据 **43 PASS**，最差相对偏差 0.379% |
| main 基线（同口径） | 同上，main 单独 | 31 条中 14 条判据 **14 PASS**，最差相对偏差 0.295% |

诚实说明：`tests/test_declared_numbers.py` 全量套件（含表 11/12 的多级三维解算，单次
~100 s 起，宿主上并行着多个 mission 的同名套件）在本单窗口内**没跑完**，因此「不低于
main 基线」的证据取自同口径的定向 `check()` 与 fast 单测，而非全量 pytest 计时。
遗留观察：全量套件在 151 条注册表上会跑两套读法（共享 study 缓存，故不翻倍解算），
耗时以宿主负载为主。

## 6. 分支 SHA 与产物

| 项 | 值 |
|---|---|
| 旧 `agent/m3264` | `cbcac1fd3e1096a9d76b06de1fff7acc831bc4b7`（m3797 的合并提交，已含 `ecba798`） |
| 本轮新提交（本单分支 `agent/m3839`，快进自 `cbcac1f`） | `33c74e2` |
| 合并提交（仍在祖先链上） | `cbcac1f`，父 1 = `ecba798`(main)、父 2 = `5b4dee7`(m3264) |

产物：`tools/audit_union.py`、`tools/pr135_union_dispositions.json`（42 条处置：
27 条重命名/重写 + 2 条主动订正 + 其余分组重命名）、`artifacts/pr135_union_audit.json`、
`artifacts/fig1_registry_union.png`、`artifacts/fig2_reading_reconciliation.png`。

## 7. 注意点（坑与局限）

- **机械 overlay 的真实形态**：m3797 把两套注册表都留下了（文件层看着完美），却漏了
  同一文件里一个摘要字段；**文件级 diff 与「两套都留」的直觉都抓不到它**，只有「逐键 import
  两侧注册表 + 行级处置清单」能抓到。这就是本单把审计做成工具的原因。
- **重命名不是无损的免费事**：`*_cells` 是本轮唯一的重命名，它改变了 m3264 的组名字符串
  契约；审计要求每条重命名都写进 dispositions 并附见证串，避免「重命名」变成掩盖丢失的口袋。
- **共享 study 缓存**：两侧读同一张表，`_STUDY_CACHE` 让第二次读表 0 成本；副作用是同一进程
  内多次 `check()` 复用解算结果，若要测「真的重算」必须换进程或用自定义 runner。
- 局限：本仓宿主无网络（`git ls-remote` 超时），无法确认 GitHub 上 `agent/m3264` 的实时 SHA；
  本轮以本地快照 `cbcac1f` 为起点，新提交落在本单分支，由平台推送后 governor 复验。

## 8. 误差判定（是否 <3%？）

| 关键量 | 基准 | 偏差 | <3%？ |
|---|---|---|---|
| 并集覆盖：main 声明条数丢失 | 84 | 0 条 | 是（0%） |
| 并集覆盖：m3264 声明条数丢失 | 89 | 0 条 | 是（0%） |
| 共有 22 键的值被静默改动 | 22 | 0 条 | 是（0%） |
| 转动耦合印数 vs 四位有效数字实测（80 面元，2.15e-4 vs 2.151e-4） | 2.151e-4 | 0.047% | 是 |
| 转动耦合印数 vs 实测（320 面元，2.65e-5 vs 2.6481e-5） | 2.6481e-5 | 0.072% | 是 |
| 被替换的旧印数 2.2e-4（负控） | 2.151e-4 | 2.28% | 是（仍在 5% 判据内、超出新的 1% 印刷带，故判红） |
| 三维表复算最差相对偏差（合并树，43 条 equality 判据） | 各表实测/解析基准 | 0.379% | 是 |
| 同上，main 单独（14 条判据） | 同上 | 0.295% | 是 |

结论：所有可量化判据均 <3%（除负控行已标注它按设计只在 1% 印刷带上判红）；
全量 pytest 基线对比未完成，已在 §5 如实标注为未完成项。