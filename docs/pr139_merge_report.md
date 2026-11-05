# PR #139 冲突治愈报告：`agent/m3306` 语义并集并入 `main`

任务：TensorBEM PR #139（`agent/m3306` → `main`）`mergeable=false`。本文件记录冲突清单、
逐文件解法、判据与证据，全部产物在 `outputs/verif_m3930/`。

- 分支旧 SHA：`36ea84966d3b3568fcfb8cea6f0080e8bcb5a2c4`（origin/agent/m3306）
- main 目标 SHA：`ecba798833d6c2429fbc897d2ce515efb70dd682`
- merge base：`4f5b73e73c4e731582f2b050328ec5fcf62b4a86`
- 新 SHA（本地合并提交）：`95438c6`（第二父 `ecba798`，第一父 `36ea849`）
- 纪律性说明：`git push` 在本容器被安全策略拦截（`REJECTED: 命令被安全策略拦截`），
  因此分支只在本工作树落盘；`agent/m3930` 已指向同一合并提交，交由平台推送/`merge_governor` 复验。

## 1. 案例背景

`agent/m3306` 从 `4f5b73e` 切出，之后 main 前进了 89 个文件（19 045 行新增：3-D 基准族、
`capability_evidence`、大量验证测试）。分支自身交付的是「把文档表 10–12 的 3-D 无界/自由面
表格逐格登记进 `tensorbem.benchmarks.declared_numbers` 注册表」——新增 37 条 `DeclaredNumber`
与三组 handler（`hulme_hemisphere` / `hulme_irregular` / `submerged_sphere`）。
两侧在**同一批锚点**上都做了纯新增（`git diff` 的删除行数两侧均为 0），所以冲突是
add/add，而不是内容打架。

冲突文件（3 个，全部 content 冲突）：

| 文件 | 两侧相对 base 的变化 | git 冲突块 | 冲突性质 |
|---|---|---:|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | ours +407 / theirs +656（均 0 删除） | 28 | add/add，且 hunk 边界切在 entry 内部 |
| `tests/test_declared_numbers.py` | ours +97 / theirs +38 | 2 | add/add |
| `docs/marine_bem_benchmarks.md` | ours +57−1 / theirs +9−4（3 格数字） | 1 | 同一列两格数字不一致 |

## 2. 怎么做的

**为什么不直接解 marker**：`git` 的 hunk 边界正好落在
`DeclaredNumber(key=…, …` 的属性块中间——`declared=`、`)` 落在公共区。实测按 marker
逐块取并集会产出截断的 entry（第一次尝试即 `SyntaxError: '(' was never closed`）。
这正是「机械 overlay 静默丢内容」的复发形态，所以改为**按三方版本重组并集**：

- `outputs/verif_m3930/resolve_merge_v2.py`：以 `ast` 解析三个版本的 `DECLARED_NUMBERS`
  与顶层函数，把「某一侧相对 base 的新增」当作数据重新插回；两处真正打架的点用**复算**裁决。
- `outputs/verif_m3930/audit_union.py`：独立审计（不信任 resolver）：注册表 key 集合
  必须**等于**两侧并集、key 唯一、同名 key 的 `declared` 值必须一致、测试函数两侧全集都在、
  文档两侧新增行全部在场（裁决行除外）、合并后模块可导入。
- `outputs/verif_m3930/registry_contract_check.py`：用合成 study 载荷（不跑 256 面元装配）
  验证**每条注册项**的 `quantity` 都能被其 group handler 提供——这是并集最容易断的胶水。
- `outputs/verif_m3930/adjudicate_table10_cell.py`：真实 16/64/256 面元复算，裁决唯一
  一处数字分歧。

### 逐文件解法

**① `src/tensorbem/benchmarks/declared_numbers.py`（28 块）**

- 骨架取 **main**（`sub<level>_<quantity>` 注册范式、`_study_rows` / `_chief_rows`、
  `hulme_chief` handler）；
- m3306 的 37 条新增 entry **原样**重新插入 `DECLARED_NUMBERS`（值、tier、tolerance 一字不改）；
- 重新插入 m3306 的 `_finest_level`（最细层 + 观测阶展开器）；
- 重新插入 m3306 的 `hulme_irregular` handler（不规则频率 / CHIEF 组）；
- `hulme_hemisphere`、`submerged_sphere` 两个 handler 改为**一次求解喂两套范式**：
  `{**_study_rows(study), **_finest_level(study, …)}`——两套命名不重叠（`sub2_x` vs `x`），
  因此两种钉桩同时可复算，一次 solve 不重复付钱；
- 导入别名折叠为每模块一个（m3306 的 `floating_hemisphere_benchmark` /
  `submerged_sphere_benchmark` 与 main 的 `hemisphere_benchmark` / `sphere_benchmark`
  指向同一模块，属显式折叠，已记入 `merge_resolution_log.json`）。

**② `tests/test_declared_numbers.py`（2 块）**：main 的测试全保留；m3306 的
module 级 `standard_report` fixture 与 3 个 3-D 钉桩测试重新追加；
`test_registry_covers_every_documented_benchmark_family` 的已知 group 元组扩为并集
`{hulme_chief, hulme_hemisphere, hulme_irregular, submerged_sphere}`（显式扩集，非覆盖）。

**③ `docs/marine_bem_benchmarks.md`（1 块）**：骨架取 **m3306**（含 1.94 订正与整节
「declared-number 钉桩」表格）；把 main 的 3 处可复现数字订正重新落地；
main 的「观测阶两种口径（m3150）」段落重新插回。唯一真分歧（表 10 第 2 行末列，
$B_{11}'$ 观测阶 16→64 面元）用复算裁决：

| 口径 | 复算值 | 来源 |
|---|---:|---|
| 实测平均棱长比 $h_0/h_1=1.8681$ | **1.8021** | main 的 1.80 |
| 名义加密因子 $\sqrt{N_1/N_0}=2$ | 1.6247 | m3306 保留的 base 值 1.62 |
| 同列 64→256（m3306 已订正格） | 1.9377（h_mean）/ 1.8873（名义） | m3306 的 1.94 |

该列其余四格与 m3306 自己新增段落主张的口径都是 `h_mean` 比，故取 1.80；m3306 的 1.94
订正在同一张表里被保留。**这是一次口径统一，不是数字翻案**：误差值本身（40.2% / 13.0% /
3.53%）两侧从未分歧。

## 3. 计算结果（程序化注入）

| 指标 | main 基线 | 合并后 | 判据 |
|---|---:|---:|---|
| 注册表条目数 | 84 | **117** | = 84 ∪ 55（22 条为 base 共有，无同名异值） |
| key 唯一 | 是 | 是 | `audit_union.json` |
| 丢失条目 | — | **0** | 集合相等（missing=[] / phantom=[]） |
| 文档新增行保留 | — | 47/47（m3306）+ 8/8（main，除 1 裁决行） | `audit_union.json` |
| 测试函数 | 17 | **21**（两侧并集） | `audit_union.json` |
| 数量契约（117 条 quantity 均可解析） | — | **problems=[]** | `registry_contract_check.json` |
| tier 分布 | — | fast 40 / standard 67 / heavy 10 | 同上 |
| `tests/test_declared_numbers.py`（main 基线，默认 tier） | **14 passed, 3 skipped, 112.05 s** | — | `/tmp/base_declared.log` |

诚实说明：合并后同一测试文件的**完整**运行在本机两次都未跑完（宿主机 load average
682–885，256 面元 3-D 装配被反复拖死，进程无汇总行退出）。因此合并侧的证据是
①两侧测试函数集合完整（审计）②117 条注册项的数量契约全部通过（合成载荷）
③真实 16/64/256 面元复算在裁决脚本里跑通。这不是 full-suite 绿，不能当绿报告。
标准 tier 复算（`python -m tensorbem.benchmarks.declared_numbers --tiers standard`）
必须在空闲机器上补跑，记为本任务唯一未闭合项。

## 4. 结果图

- `outputs/verif_m3930/fig_pr139_registry_union.png`：四个版本的注册表条目数
  （22 → 84 / 55 → 117）与「无条目丢失」的集合相等结论。
- `outputs/verif_m3930/fig_pr139_order_adjudication.png`：$B_{11}'$ 误差对 $h_{mean}$
  的双对数收敛图，附 1.80 与 1.62 两条斜率参考线——裁决的唯一依据。

## 5. 注意点

1. **git hunk 会切在 entry 内部**：add/add 冲突里两侧都在同一锚点追加结构体时，
   marker 级并集必然截断。判据应放在「语义集合相等」（key 集合、函数名集合、新增行集合），
   而不是「冲突块都取双方」。
2. **并集带来冗余而非错误**：合并后同一批 3-D 表格数字被两套范式各钉一次
   （117 = 84 ∪ 55，重叠语义约 33 条）。两套都能复算、都 PASS，但建议后续统一到
   `sub<level>_` 范式（本任务不做，避免再次引入语义改动）。
3. **tier 预算**：并集使默认 CI 扫（fast+standard）条目数从 84 增至 117，
   3-D 组多出 2 个 study 配置，CI 时长上升，需在空闲机上重测时长。
4. **文档双源风险**：m3306 钉桩表把转动耦合比按 2 位有效数字记 2.6e-5（容差放宽到 5%），
   main 表按 3 位记 2.65e-5。两者都 PASS 且相邻出现，属可读性欠账，已在报告中标注。
5. `git push` 被容器安全策略拦截，推送由平台完成；`agent/m3930` 与 `agent/m3306`
   在本工作树指向同一合并提交，任一被推送都可让 `merge_governor` 复验 PR #139。

## 6. 误差判定

| 关键量 | 基准 | 合并后值 | 相对误差 | 是否 <3% |
|---|---|---:|---:|---|
| 表 10 第 2 行 $B_{11}'$ 观测阶（并入值 1.80） | 复算 1.8021 | 1.80 | 1.1e-3 | **是** |
| 表 10 第 3 行 $B_{11}'$ 观测阶（m3306 值 1.94） | 复算 1.9377 | 1.94 | 1.2e-3 | **是** |
| 注册表条目数（并集完整性） | 84 + 55 − 22 = 117 | 117 | 0 | **是** |
| 数量契约违规数 | 0 | 0 | 0 | **是** |

未闭合项（如实登记）：合并后 `tests/test_declared_numbers.py` 的完整运行未在本机跑完，
标准 tier 端到端复算（含 256 面元 3-D 组）**没有**在本次取得 greens 证据。