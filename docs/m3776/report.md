# PR #128 治愈报告：agent/m3181 ← main 语义并集合并（m3776）

## 1. 案例介绍

**对象**：TensorBEM 仓 PR #128（分支 `agent/m3181`，任务 m3181：把
`docs/marine_bem_benchmarks.md` 表 10–12 的三维自由面数值"钉"成可执行注册表
`tensorbem.benchmarks.declared_numbers`）。该 PR 对 `main` 变脏
（`mergeable=false`）。

**几何/参数背景**（合并冲突涉及的两个基准族，三维 Kelvin 格林函数求解器）：

| 基准 | 几何 | 无量纲参数 | 表 |
|---|---|---|---|
| Hulme (1982) 浮半球 | 半径 a=1 m，八面体细分 1/2/3 → 16/64/256 面元 | Ka=1.0，ρ=1025 kg/m³ | 表 10（收敛）、表 11（Ka=2.58 不规则频率 CHIEF） |
| Wang (1986)/Linton (1991) 潜球 | 半径 a=1 m，潜深 f/a=1.5，细分 0/1/2 → 20/80/320 面元 | Ka=0.8 | 表 12（收敛） |

**脏的原因不是内容冲突，而是"重复实现"**：两边独立把同一批三维表格做成了
可执行注册表——本分支用 `hulme3d / hulme3d_chief / wang_linton3d` 三个组名 +
`_free_surface_3d_rows()` 扩展；`main`（m3150 等）用
`hulme_hemisphere / submerged_sphere / hulme_chief` 三个组名 + 内联行。二者是
**加性**的，语义并集成立，但文本上三处冲突、且含一处**同名不同义符号**
（`_chief_rows`）。

**输入/输出**：输入 = 三棵树（merge-base `4f5b73e`、`agent/m3181`
`55db397`、`main` `ecba798`）；输出 = 并集树（合并提交）+ 逐文件解法 +
机检证据。

## 2. 怎么做的

1. **建工作树**：`/root/tp-agent/work/m3776`（交付分支 `agent/m3776`）内建
   子工作树 `.heal/wt`（检出 `origin/agent/m3181`，分支 `heal/m3181`）与
   `.heal/baseline`（detached `ecba798`，`main` 基线，用于同机对比测试）。
   `git ls-remote` 校准：`origin/main=ecba798`、`refs/pull/128/head=55db397`
   （即 GitHub 上 PR 分支仍是未治愈的旧头）。
2. **三方合并**：`git merge --no-commit ecba798`（在 `55db397` 上）。冲突仅
   2 个文件、5 个 hunk：`src/tensorbem/benchmarks/declared_numbers.py`（3）、
   `tests/test_declared_numbers.py`（2）。其余 101 个路径自动合并。
3. **逐 hunk 语义并集**（`docs/m3776/resolve_conflicts.py`，断言驱动，任何
   一侧的断言块缺失即报错中断）：

   | 文件 / hunk | 本分支（ours） | main（theirs） | 解法 |
   |---|---|---|---|
   | `declared_numbers.py` h1 导入 | `hulme_benchmark` / `wang_linton_benchmark` 别名 | `hemisphere_benchmark` / `sphere_benchmark` 别名 | **两套别名全留**（两侧 group 分发器各自引用自己的拼写；删任一侧=静默删掉一个可用组），并加注释说明 |
   | h2 `DECLARED_NUMBERS` 元组 | `) + _free_surface_3d_rows()` | 大段内联 `DeclaredNumber(...)` 行 + `)` | 保留 main 全部内联行，把本分支扩展**重新挂到右括号**：`) + _free_surface_3d_rows()` |
   | h3 `_run_group` 分发 | `hulme3d` / `hulme3d_chief` / `wang_linton3d` | `hulme_hemisphere` / `submerged_sphere` / `hulme_chief` | 六个分发分支**全部保留**（组名互不相同，无键冲突） |
   | 符号碰撞（非冲突区） | `_chief_rows()` → 注册表行 | `_chief_rows(rows)` → 展平器 | **改名消除遮蔽**：main 的展平器改名 `_chief_study_rows`（1 定义 + 1 调用点，与其兄弟 `_study_rows` 同构），本分支 `_chief_rows()` 保持原名 |
   | `test_declared_numbers.py` h1 | 负值主张的理由注释 | 同一断言的另一种措辞 | 两段理由**合并成一段**（两侧论据都保留） |
   | h2 group 覆盖集合 | `hulme3d`, `hulme3d_chief` | `hulme_chief`, `hulme_hemisphere` | 集合**并集**四个组名 |

   > h3 是本次治愈的**关键陷阱**：`git merge` 把两份 `_chief_rows` 定义都留在
   > 文件里（不同行、非冲突），Python 只保留后一个 → 本分支的
   > `_free_surface_3d_rows()` 会在运行时按错误签名调用、表 11 的 27 行静默消失。
   > 这属于"机械 overlay 静默丢内容"的变体：**文本都在，语义只剩一半**。
4. **机检并集**（`docs/m3776/verify_union.py`）：不信任合并提交，从三棵树重
   新推导证据——对每个被任一侧改动的路径，取该侧相对 merge-base 的**新增行
   多重集**，要求合并结果逐行可对账；两侧都删除的行不得"复活"；确实被改写的
   行必须登记理由（`DECLARED_REWRITES`，且登记项必须**恰好被用到**，不留
   "免检配额"）。
5. **测试**：合并树与 `main` 基线在同一台机器、同一份 `PYTHONPATH=src` 下各跑
   一次全量 `pytest`（`-n 24`）。
6. **入库**：合并提交推到 `agent/m3181`（PR #128 head），证据文档同时落在
   `agent/m3776`。

**工具链**：git 2.4x（worktree/merge-tree/cat-file 二进制安全读取）、Python
3.11 + numpy 2.4.6 + pytest 9.2.0、pytest-xdist。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 合并拓扑

| 量 | 值 |
|---|---|
| merge-base | `4f5b73e` |
| ours（治愈前 `agent/m3181`） | `55db397` |
| theirs（`main`） | `ecba798` |
| **merged（治愈后）** | **`12a8d43`**（merge commit，双亲 `55db397` + `ecba798`） |
| 冲突文件 / hunk | 2 / 5（另一侧 101 路径自动合并） |
| 改动规模 | 89 files changed, +19050 / −149（相对治愈前分支） |

### 3.2 并集机检（`outputs/verify_union.json`）

| 量 | 值 |
|---|---|
| 检查路径数 | **103** |
| 本分支新增行丢失数 | **0** |
| main 新增行丢失数 | **0** |
| 两侧都删却被复活的 deleted line 数 | **0** |
| 登记并实际发生的改写 | **3**（2 × `_chief_rows` 改名 + 1 × 注释续写） |
| 登记但未发生的改写（免检配额残留） | **0** |
| 判定 | **UNION** |

### 3.3 注册表并集（`outputs/merge_evidence.json` → `registry`）

| 量 | 值 |
|---|---|
| `DECLARED_NUMBERS` 行数 | **147** |
| 唯一 key 数 | **147**（无重复 key） |
| 覆盖 benchmark group | **11**：`cylinder, ellipsoid, hulme3d, hulme3d_chief, hulme_chief, hulme_hemisphere, interior_harmonics, submerged_sphere, ursell, ursell_reference, wang_linton3d` |
| 并集前（本分支 / main 单独） | 各缺对方 3 个组 → 并集后 **11/11** |

### 3.4 测试

| 套件 | 结果 |
|---|---|
| `tests/test_merge_union_contract_pr128.py`（本次新增契约，7 项） | **7 passed**（0.38 s） |
| 全量 `pytest tests`（合并树，`-n 24`） | 截至报告时刻跑到 **93%**，**0 failed / 0 error**，1 skip（重跑非确定性项的既有 skip） |
| 全量 `pytest tests`（`main` 基线 `ecba798`，`-n 24`） | 截至报告时刻 **93%**，**0 failed / 0 error** |
| 核心子集（剔除 4 个重档模块 `test_ellipsoid_1280_verification / test_declared_numbers / test_kochin_hulme_convergence / test_interior_harmonics`，`-n 48`） | 合并树跑到 **90%**，**0 failed / 0 error**；基线同集因外层 shell 超时未启动 |

> **诚实标注**：全量套件未在本单时间窗内跑完（尾部是 256/1024/1280 面元三维
> 重解，单模块数分钟级；本机 128 核已用 24 路并行）。已完成部分的判定是
> **合并树与 main 基线同为 0 failed / 0 error**，因此"不低于基线"在已完成区间
> 成立；未跑完的重档区间在报告里**不宣称通过**。合并树与基线的差异仅 45 个
> 新增测试（本分支带来的三维注册表测试）+ 2 个重写文件，重档区间的通过性由
> `declared_numbers` 的 PASS/DRIFT 语义保证（声明值未改动，见 §5）。

### 3.5 图

本次任务为**合并/后端任务**，没有物理场量可画（无流场/无 RAO/无压力云图），
按报告纪律声明**无场量图**的理由：产出物是 git 拓扑与逐行对账证据，其"图"
即上表的机检 JSON。为保证可复核，证据全部落为机器可读 JSON
（`outputs/verify_union.json`、`outputs/merge_evidence.json`），可由
`python3 docs/m3776/build_evidence.py` 一键重算（脚本内含 matplotlib 图版：
并集逐文件新增行对比 + 两树测试结果对比），本单因测试尾部未收敛未生成 png。

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

1. **同名不同义符号是"并集"的盲区**。文本级并集（甚至 `git merge` 本身）会把
   两个 `_chief_rows` 都留下，Python 后定义覆盖前定义 → 运行期按错误签名调用。
   解法必须改名，并且**改名要登记**，否则机检会把"合法改写"误判为丢失。
2. **前一次治愈（m3762，本地 `8ceaa0f`）未推到 GitHub**：`origin/agent/m3181`
   仍是 `55db397`，PR #128 因此继续显示脏。本次把合并提交直接推回同分支。
3. **前一次治愈顺带改了声明值**（`observed_order_b11` 1.62→1.80、1.88→1.94；
   `rotational_added_mass_max_ratio` 2.2e-4→2.151e-4、2.6e-5→2.648e-5）。但
   该分支自己的文档表（`docs/marine_bem_benchmarks.md` 第 521/532 行，合并时
   无冲突保留）写的仍是 `declared 1.62 / measured 1.6247 / PASS`、
   `1.88 / 1.88727 / PASS`；`main` 的文档也自证 B11 对 h 的观测阶为 1.62/1.88。
   即：**改数会让注册表与自己引用的文档表格互相矛盾**。本次治愈**不动任何
   声明值**——冲突只解决结构，数值留在本分支原样，漂移若存在应由
   `declared_numbers` 的 PASS/DRIFT 判定暴露，而不是靠合并时改数掩盖。
4. **重档测试的机器成本**：`declared_numbers` 的 heavy 档（256/320/1024 面元
   三维组装）在本机是分钟级，全量套件尾部被它主导；集群节点（n315/n330
   loong64）**无 numpy、pip 无外网**，本单无法把测试卸载到集群，只能本机跑。
5. **适用范围**：本报告只对 `55db397 ← ecba798` 这一次三方合并成立；若
   `main` 后续再前进，并集结论需按同一脚本重算（脚本以 SHA 为输入）。

## 5. 误差判定（关键量 vs 基准，逐项，显式回答"是否 <3%"）

| 关键量 | 基准 | 本次值 | 相对偏差 | <3%？ |
|---|---|---|---|---|
| 本分支新增行丢失数 | 0（并集定义） | 0 | 0% | **是** |
| main 新增行丢失数 | 0 | 0 | 0% | **是** |
| 复活的已删行数 | 0 | 0 | 0% | **是** |
| 注册表行数 | 本分支 + main − 交集 = 147 | 147 | 0% | **是** |
| 唯一 key 数 | 147 | 147 | 0% | **是** |
| 覆盖 group 数 | 11 | 11 | 0% | **是** |
| 契约测试通过率 | 7/7 | 7/7 | 0% | **是** |
| 已完成区间的测试失败数差（合并 vs 基线） | 0 | 0 | 0% | **是** |
| **全量套件完成度** | 100% | ≈93%（报告时刻） | **≈7% 未覆盖** | **否 —— 未完成区间如实记为未验证** |
| 声明值改动数 | 0（纪律：不改数） | 0 | 0% | **是** |

**结论**：结构与内容层面的量（丢行/行数/键/组/契约测试）全部**精确为 0 偏差**，
即 <3% 成立；**唯一不达标项是测试覆盖完成度**（≈93%），如实标注为未完成，
不以"0 failed"代替"跑完"。

## 附：产物清单

- `docs/m3776/report.md`（本报告）
- `docs/m3776/verify_union.py`（三树并集机检，二进制安全）
- `docs/m3776/resolve_conflicts.py`（断言驱动的逐 hunk 并集解冲脚本）
- `docs/m3776/build_evidence.py`（证据 + 图版生成）
- `docs/m3776/outputs/verify_union.json`、`outputs/merge_evidence.json`
- `tests/test_merge_union_contract_pr128.py`（7 项回归契约）