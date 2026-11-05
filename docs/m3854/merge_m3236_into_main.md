# m3236 → main 三方合并治脏报告（任务 m3854，PR #134）

本文件记录一次**手工语义并集合并**：分支 `agent/m3236`（PR #134）相对 `main`
变脏（`mergeable=false`），本任务把它重新合入 `main` 并推回同分支，供
merge_governor 下一周期复验。

## 1. 合并事实（SHA 全部取自 `git rev-parse`）

| 项 | 值 |
|---|---|
| 目标基线 `origin/main` | `ecba798`（Merge pull request #103 from cxs503/agent/m2904） |
| 被治分支 `origin/agent/m3236`（旧 SHA） | `1615c10`（chore: re-run CI against current main） |
| 共同祖先 merge-base | `4f5b73e`（Merge #120 (m3086) via merge-governor） |
| main 领先基线的提交数 | 40 |
| m3236 领先基线的提交数 | 3（`a2a1ed9` 交付 → `f67bbd0` 曾合过一次 main → `1615c10` 空树重跑 CI） |
| 新 SHA（治后合并提交） | 本分支头提交（`git rev-parse agent/m3854`）（合并提交，树 = 治后的 m3236 内容） |
| 本任务分支 | `agent/m3854` @ 本分支头提交（`git rev-parse agent/m3854`）（由平台自动推送） |

脏的根因：m3236 的基线停在 `4f5b73e`，而 main 之后又落了 40 个提交，其中
`src/tensorbem/benchmarks/declared_numbers.py` 被大幅扩写（+661 行：Hulme 半球、
潜没球体、CHIEF 三组声明数字），m3236 同时在同一个文件的**同一位置**追加了
Dean–Ursell–Ogilvie 绕射的 11 条声明数字（+170 行），于是两头都改同一段 →
内容冲突。属于典型的「注册表尾部追加撞车」，不是真的语义对立。

## 2. 冲突文件清单与逐文件解法

| 文件 | 状态 | 解法 |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | **CONFLICT（2 处 hunk）** | 手工语义并集（见下 2.1 / 2.2） |
| `tests/test_declared_numbers.py` | 自动合并成功（`Auto-merging`） | 校验两侧新增测试与分组断言都在（见 §3 审计） |
| `docs/marine_bem_benchmarks.md` | 自动合并成功（`Auto-merging`） | 校验 4.8（Dean，m3236 新增）与 4.10/4.11（Hulme/球体，main 新增）章节共存 |
| `docs/m3236/outputs/m3236/*.json, *.png`（6 个） | 新增（A） | 直接入库，未改动 |

### 2.1 Hunk 1：`DECLARED_NUMBERS` 注册表尾部

- main 侧：以 `submerged_sphere_320_order_fz`（`tier="standard"`）结尾，**缺少**
  该 `DeclaredNumber(...)` 的收尾 `),`；
- m3236 侧：以 `dean_ellipse_haskind_z_64`（`tolerance=5.0e-2`）结尾，同样缺少收尾 `),`；
- git 把两侧的收尾 `),` + 列表闭合 `)` 当作「公共尾部」。

解法：`main 侧全部条目` + 补一个 `),` + `m3236 侧全部 11 条 dean 条目`，
公共尾部 `),` 与 `)` 保留 —— 即两侧条目**各自独立闭合**，共 95 条注册项。
没有采用 `--ours/--theirs`，也没有用机械 overlay（覆盖会静默丢一侧条目）。

### 2.2 Hunk 2：`_run_group()` 的分组分派

- main 侧：`hulme_hemisphere` / `submerged_sphere` / `hulme_chief` 三个 `if` 分派块；
- m3236 侧：`dean` 一个 `if` 分派块（返回 11 个 measured 标量）。

解法：顺序拼接两段分派（main 三块在前、dean 块在后），公共的
`raise ValueError(f"unknown benchmark group: ...")` 保留 —— 9 个分派分支全在。

## 3. 无损校验（可执行判据）

`tools/verify_m3236_merge.py` 对 `origin/main`、`origin/agent/m3236`、合并后工作树
三方做**集合并集契约**检查：注册项 key、分派 group、模块级 def/class、`def test_*`、
文档小节、以及分支新增文件是否全部落地。任何一侧独有的条目消失在合并结果里都会
`exit 1`。

最近一次运行（输出即 `artifacts/m3236_merge_audit.json`）：

| 维度 | main | m3236 | 并集 | 合并后 | 判定 |
|---|---:|---:|---:|---:|---|
| `declared_numbers.py` 注册 key | 84 | 33 | 95 | 95 | OK |
| `declared_numbers.py` 分派 group | 8 | 6 | 9 | 9 | OK |
| `declared_numbers.py` def/class | 10 | 8 | 10 | 10 | OK |
| `test_declared_numbers.py` 测试函数 | 17 | 16 | 18 | 18 | OK |
| `marine_bem_benchmarks.md` 小节 | 22 | 22 | 22 | 22 | OK |
| m3236 新增文件 | — | 6 | 6 | 6 | OK |

`MERGE-CONTENT-UNION: PASS`，`missing_from_merged` 与 `duplicated_in_merged` 均为空。

## 4. 测试与数值证据

1. **注册表全量复算（fast 层）**：合并后树的
   `python -m tensorbem.benchmarks.declared_numbers --tiers fast`
   → `checked=51, passed=51, drifted=0, verdict=PASS`（`artifacts/m3236_fast_tier.json`）。
   即 main 原有的 40 条与 m3236 的 11 条 dean 数字在**同一份合并代码**上同时复现。
2. **合并侧 pytest（冲突文件模块）**：
   `pytest tests/test_declared_numbers.py -k "well_formed or covers_every or dean"`
   → **3 passed**（其中 `test_fast_tier_reproduces_the_documented_dean_numbers` 为 m3236 新测试，13.4 s）。
3. **基线侧对照**（`/tmp/bem_base` = `origin/main` `ecba798`，同机同 load）：
   `pytest tests/test_declared_numbers.py -k "well_formed or covers_every"` → **2 passed**。
   合并侧在同一子集上 2/2 通过 + dean 1/1 通过 ⇒ **不低于 main 基线**。

未完成部分（如实说明）：整包 `pytest`（1002 个用例）在本次会话中**未能跑完**。
宿主 load average 达 ~510（多智能体共用 128 核），两次完整套件运行都在 14~20 分钟
内只推进到 ~14% 且不断被超时收割；因此改为「边界可控的核心子集 + 全量注册表数值复算」
的对照口径，并保留上述可复现命令。

## 5. 注意点（踩坑）

- **editable install 指向别人的 worktree**：`pip show tensorbem` 的 editable 源解析到
  `/root/tp-agent/work/m3742/src`，直接 `pytest` 会**测另一个分支的代码**（实测
  `ModuleNotFoundError: tensorbem.benchmarks.recovery_robustness`）。本任务全部运行都带
  `PYTHONPATH=src`（worktree 内相对路径），这是判据成立的前提。
- 合并脚本必须是**结构化**的（按冲突标记切块再拼接），不能用正则并集或整块 overlay ——
  后者正是历史上静默丢内容的那条路。
- `git show HEAD:file` 读到的是**合并前**的 main 版本；审计脚本对「合并后」一律读工作树。
- 结论只覆盖 `origin/main` 与 `agent/m3236` 的并集语义，不构成对其它在飞分支的保证。
- **推送由平台执行**：把提交推回 `agent/m3236` 的命令被安全策略拦截（智能体不自推），
  因此治后内容落在本任务分支 `agent/m3854` @ 本分支头提交（`git rev-parse agent/m3854`）；本地 `agent/m3236` ref 被
  `/root/tp-agent/work/m3236` 工作树占用且已被他人推进到 `49c7f79`，未强制改写。
  merge_governor 复验以 `agent/m3854` / 本分支头提交（`git rev-parse agent/m3854`） 的树为准，回滚口径对照 `1615c10`。

## 6. 影响面与回滚

- 代码改动仅限冲突文件的解冲（结构性拼接，无逻辑改写），docs 变更来自两侧的自动合并。
- 回滚：`git reset --hard 1615c10`（旧 SHA）即可回到治前状态。