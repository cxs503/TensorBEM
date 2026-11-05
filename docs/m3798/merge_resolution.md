# PR #136（agent/m3250）→ main 变脏治愈报告

**任务**：TensorBEM PR #136 分支 `agent/m3250` 对 `main` 变脏（`mergeable=false`），需三方合并、逐文件解冲、语义并集、跑核心测试不低于 main 基线。

## 1. 旧/新 SHA

| 角色 | SHA | 说明 |
|---|---|---|
| main（基线） | `ecba798` | `refs/heads/main`（GitHub 实测） |
| PR #136 头（旧） | `b7709c1` | `refs/heads/agent/m3250`，含 `4538e67 Merge branch 'main' into agent/m3250` |
| 合并基点 | `4f5b73e` | m3250 上次并入 main 的位置 |
| 治愈分支 | `agent/m3798` | 本工作树分支，内容 = `main` ⊕ `agent/m3250` 的语义并集 |

`main` 自基点多出 40 个提交（多为其他链的 merge-governor 合并），冲突全部落在「声明数字注册表」的同一区域。

## 2. 冲突文件清单与逐文件解法

三方合并（`git merge --no-ff origin/agent/m3250`，base=`4f5b73e`）后冲突 2 个文件、5 个冲突块；另 1 个文件（`docs/marine_bem_benchmarks.md`）自动合并但经人工核验为双保（见 §2.3）。

### 2.1 `src/tensorbem/benchmarks/declared_numbers.py`（3 个冲突块）

| # | 冲突位置 | main 侧 | m3250 侧 | 解法（语义并集） |
|---|---|---|---|---|
| 1 | 第 58 行 `from tensorbem.benchmarks import (...)` | `floating_hemisphere as hemisphere_benchmark`, `submerged_sphere_wave as sphere_benchmark` | `floating_hemisphere as floating_hemisphere_benchmark`, `submerged_sphere_wave as submerged_sphere_wave_benchmark` | 两套别名**全部保留**（同一模块对象，绑定不冲突）；重复的 `interior_harmonics` 绑定去重一次 |
| 2 | `DECLARED_NUMBERS` 元组内（约 675–1270 行） | 手写 ~595 行 `hulme_hemisphere` / `hulme_chief` / `submerged_sphere` 注册项（62 条，`_study_rows` 扁平化口径） | 一行 `*_free_surface_three_dimensional_numbers(),`（生成 63 条 `hemisphere3d` / `hemisphere3d_chief` / `submerged_sphere3d`） | 手写块与 splat 行**同时保留** |
| 3 | `_run_group()` 分派（约 1380–1453 行） | `hulme_hemisphere` / `submerged_sphere` / `hulme_chief` 三个分支（`_study_rows`、`_chief_rows`） | `hemisphere3d` / `hemisphere3d_chief` / `submerged_sphere3d` 三个分支（`_flatten_refinement_levels`、`_reference_values`） | 六个分支并存，各自调用本侧引入的辅助函数（辅助函数名不冲突：`_study_rows`/`_chief_rows` 与 `_flatten_refinement_levels`/`_reference_values`） |

结果：注册表 **84（main）+ 85（m3250）→ 147 条**（重叠 22 条老项），键唯一无重复，11 个 group 齐全。
两侧测量的是同一批文档表 10–12，但**口径不同**：main 侧逐格手写声明值，m3250 侧按 `h_mean` 实测阶重算并以生成器登记；二者交叉校验，均保留。

### 2.2 `tests/test_declared_numbers.py`（2 个冲突块）

| # | 冲突位置 | 解法 |
|---|---|---|
| 1 | `test_registry_covers_every_documented_benchmark_family` 的 group 集合 | 两侧 group 名取**并集**（`hulme_chief`,`hulme_hemisphere`,`submerged_sphere` ∪ `hemisphere3d`,`hemisphere3d_chief`,`submerged_sphere3d`），重复的 `interior_harmonics` 去重 |
| 2 | 3-D 自由面回归门 | `test_3d_free_surface_rows_are_guarded_by_the_default_tiers`（main）与 `test_free_surface_3d_tables_are_reproduced`（m3250）**两个测试都保留**，覆盖两条登记链 |

测试函数集合 = 两侧并集（18 个），无丢失（`artifacts/m3798/verify_union.json`）。

### 2.3 `docs/marine_bem_benchmarks.md`（自动合并 + 人工核验）

自动合并未报冲突，但机械合并最易在此静默丢内容，故逐块核验：

* m3250 侧新增 `<!-- BEGIN/END m3250-fs3d-pins -->` 段落（表 10–12 钉桩说明、honest-FAIL 记录、`h_mean` 口径订正）——**在位**；
* main 侧新增「表 10 的 B11′ 观测阶列与表 12 转动耦合比列（m3150 复算订正）」段落——**在位**；
* 两侧对表 10 观测阶（1.62/1.88 → 1.80/1.94）与表 12 转动耦合比（2.2e-4/2.6e-5 → 2.15e-4/2.65e-5）的订正**完全一致**，无矛盾；
* 已知冗余（非矛盾）：两张表各有一段解释同一订正的文字（m3250 在其钉桩块内、main 在 4.10 节正文）。保留双份是并集要求的结果，内容一致，不影响复算。

## 3. 机器化的"无静默丢失"证据

`artifacts/m3798/verify_union.py`（随本分支入库，可复跑）把两个父提交各检出为工作树，导入各自 `declared_numbers` 模块（m3250 侧键由 f-string 生成，静态解析取不到），对合并树做集合比对：

```
parents: main=84 entries, m3250=85 entries
merged=147, expected_union=147
lost_from_main=[]  lost_from_branch=[]  invented_keys=[]  duplicate_keys=[]
tests.lost=[]      docs.m3250_pins_block=true  docs.mainline_correction_note=true
verdict = PASS
```

即：**没有任何一侧的注册项/测试/文档块在合并中丢失**，也没有凭空新增的键。

## 4. 测试证据（honest 记录）

**已完成**：两冲突文件 AST 干净、`declared_numbers` 在合并树可导入（147 条注册项、11 组、键唯一）、三方并集机器判据 `verify_union.py` = **PASS**（两父提交的键/测试/文档块零丢失、零重复、零伪造）。详见 `artifacts/m3798/verify_union.json`。

**未完成（如实记录，honest-FAIL）**：全量 pytest 与 main 基线（`ecba798`）的通过数严格对照**未在本会话跑完**——全量套件单进程 >13 min，超出单次工具调用上限，后台长跑被进程组收割；集群节点缺 pytest/依赖。冲突区专项测试（`tests/test_declared_numbers.py` 的两条 3-D 表回归门）同样需要数分钟三维稠密求解，未在窗口内取得完整 PASS 行。

因此本单**不能**声称"测试不低于 main 基线"；留给 merge_governor / CI 复验的判据是 `pytest tests` 在合并树与 `ecba798` 上的通过数对照。复跑命令与踩坑见 `artifacts/m3798/test_parity.md`。

## 5. 注意点

1. **本机 /tmp 日志名撞车**：`/tmp/base_pytest.log`、`/tmp/merged_pytest.log` 被同机其他智能体进程写走（其内容含本仓不存在的测试文件，如 `test_rigid_contact_graph.py`），据此得出的"合并树多 1 个失败"是假象。教训：并发多智能体机器上日志必须带 mission 前缀。
2. **计时型测试**：`test_*_benchmark` 中若干断言是墙钟阈值（如 `elapsed < 10 s`），在 100+ 并发 pytest 的机器上会假红，判定必须与基线同工况对比或单跑复核。
3. **两侧注册表并存 → CI 成本翻倍**：表 10–12 现在被两条链各复算一遍（约 1 分钟量级的三维稠密求解 ×2）。这是并集要求的直接代价；若后续要收敛，正确做法是让 main 侧改为引用 m3250 的生成器（语义去重），而不是删掉任一链——那需要一次专门的评审，本单不做。
4. 未推送：平台规则禁止智能体自行 push（由平台推送本分支，merge_governor 下一周期复验 PR #136）。

## 6. 复现

```bash
git fetch origin main agent/m3250
git checkout -b heal/m3798 origin/main
git merge --no-ff origin/agent/m3250        # 2 files conflict
python3 artifacts/m3798/verify_union.py --merged . \
    --parent-main <main-worktree> --parent-branch <m3250-worktree>
PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q
```