# 干货工程 v1：任务干货包协议 + 三层记忆压缩策略

**锚点** FEAT-CONTEXT-CRAFT（信息策展）· **仓** TensorBEM · **单** m17171
**工件** `tools/context_craft/`（6 模块 + CLI）、`tests/test_context_craft.py`（25 测试）、
`docs/context_craft_ab/`（A/B 结果 + 3 图）、`docs/context_craft_demo/`（真实干货包样例）

---

## 0. 这份文档解决什么问题

智能体的单次任务失败，很少是"不会算"，多是"没带对东西"：不知道验收阈值是多少、不知道这条路
上次已经被证伪、不知道某个约定会翻符号。反面也有代价：把所有历史都塞进上下文，等于用 2 万 token
买 20 行有用信息（本仓实测语料 24 311 条事实单元，见 §5）。

本协议定义两件事，都用可执行代码 + 测试 + 可复算数据交付，不靠"请记得"：

1. **干货包（task brief package）**：工作包按任务类型带足**当次决策所需**的知识，且**仅带这些**。
   附 token 预算表与密度判据（§1–§4）。
2. **三层记忆压缩**：全局 bootstrap（L1）/ 席位个体（L2）/ 知识库（L3）的留去规则、压缩节奏、
   use_count 衰减与遗忘日志（§6–§9）。

设计对标的是本生态 2026-11-08 skills 试点的三条实证败因（`/root/tp-agent/skills_pilot1_summary.md`）：

| 试点败因 | 本协议的机制性对策 |
|---|---|
| 结构合规替代证据工作（章节齐全但值为 TBD） | 干货包**只能装逐字源行**，每行带 `source:lineno` + `sha256`，`verify` 子命令回读源文件复算哈希，改一个字即 `exit 1` |
| 构造性演示冒充审计发现 | A/B 用**仓内真实历史报告**做留出评估，目标段与查询段不重叠（§5.1） |
| 存档链断裂（复跑覆盖证据） | 遗忘日志 append-only JSONL + 归档文件，`verify_forget_log` 逐条复算 |
| 无敏感性分析 / 验收自选集 | 4 臂 × 4 预算 × 40 任务的网格，报告全部数字与 95% CI |

---

## 1. 干货包输入契约

一个工作包在开工前声明三件事（这就是"按任务类型带足"的判据来源）：

```json
{
  "task_type": "verification | defect_fix | feature | audit",
  "local_sources": ["docs/dev_report_xxx.md", "src/.../module.py 对应的设计文档"],
  "budget_tokens": 800
}
```

* `task_type` 决定 kind 优先级（`tools/context_craft/pack.py: PROFILES`）。
* `local_sources` 是**本工作包自己的工件**，享受预算配比优先（§3）。
* `budget_tokens` 是硬上限，超出即**跳过不截断**（截断会毁掉逐字可验证性）。

---

## 2. 条目模型与 kind 优先级

每条是一个原子事实单元，来自源文件的一行（`FactItem`）：

| kind | 判定（策展语义，非领域语义） | 权重 |
|---|---|---|
| `TOLERANCE` | 含阈值/判据词 **且** 含数字（`误差 3%`、`< 5%`、`tolerance`） | 1.00 |
| `NEGATIVE` | 已证伪/不支持/禁止/FAIL/局限 | 0.95 |
| `TRAP` | 记录在案的坑/反例（`坑`、`trap`、`踩过`） | 0.90 |
| `VERDICT` | 既有裁定/证据等级（`PASS`、`REAL/MOCK/STUB`、`已核`） | 0.75 |
| `PIN` | 锁定值/commit/哈希锚 | 0.70 |
| `CONTEXT` | 无决策含量的背景叙述 | 0.40 |

**保留底线（retention floor）**：`TOLERANCE / NEGATIVE / TRAP` 各保证至少 1 条进入包内，
即使它在词面相关性上排名靠后。这是从失败案例反推的硬约束——一次没带验收阈值的任务，等于没有验收线。

组装顺序（`pack.py: pack()`，契约的一部分）：

1. floor 填充（每 floor kind 取排名最高的一条）；
2. 贪心填充：`BM25(query) × kind 权重 × profile 提升`，近重复（3-gram Jaccard ≥ 0.85）抑制；
3. 硬预算：装不进就跳过，绝不截断。

---

## 3. 两流预算配比（本轮 A/B 的直接产物）

第一轮 A/B（只有全局检索打包器）**打输了**：在 800 token 下，全局打包器留出事实召回 0.141，
而"直接读自己那份文档"基线 0.257。根因不是打分，是**语料对称性**：任务真正需要的阈值/坑，
大量就写在工作包自己的文档里，而全局检索会把预算花在别的任务的"形似"阈值上。

对策（`pack_two_stream`）：预算按 `local_share=0.7 / 0.3` 切两块——

* 流 A：`local_sources`（本工作包工件）拿 70%；
* 流 B：语料其余部分（历史裁定、负面知识、pin）拿 30%，**这块是干货包相对"读自己文档"的增量所在**；
* floor 种类全局只买一次；两流都装不满时按全局排名**回填**，避免配比把包缩小。

第二轮 A/B（§5）显示这一改动把 800 token 下的留出事实召回从 0.137 提到 0.513、验收判据召回
从 0.316 提到 0.544。

---

## 4. token 预算表与密度判据

预估器：CJK 按 1.5 字/token，其余 4 字符/token（`model.estimate_tokens`，确定性、无分词依赖；
A/B 同时报字符量，结论不依赖该估计器）。

| 任务类型 | 预算 | 理由 |
|---|---|---|
| verification | 800–1200 | 需要阈值 + 既有裁定 + 反例，三样都要 |
| defect_fix | 800 | 需要失败模式 + 坑 + 相关阈值 |
| feature | 600 | 主要要 pin 与既有裁定 |
| audit | 1000–1600 | 需要跨文档负面知识 |

**一个干货包算合格，需同时满足**（机器可判）：

| # | 判据 | 阈值 | 实现 |
|---|---|---|---|
| G1 | 至少 1 条 `TOLERANCE` | ≥1 | `pkg.kinds` |
| G2 | 至少 1 条 `NEGATIVE` 或 `TRAP` | ≥1 | `pkg.kinds` |
| G3 | token 用量 ≤ 预算 | 硬 | `pkg.tokens_used <= budget` |
| G4 | 决策行占比（非 CONTEXT） | ≥ 0.50 | `pkg.density` |
| G5 | 逐字可复算 | 全部命中 | `cli verify` 退出码 0 |
| G6 | 确定性 | 同输入同 `package_sha256` | `package_digest()` 复算 |

实测样例（`docs/context_craft_demo/demo_brief.md`）：900 token 预算、32 条、kinds
`{TOLERANCE 21, VERDICT 4, NEGATIVE 3, PIN 2, TRAP 1, CONTEXT 1}`，决策行占比 0.97，
`verify` 32/32 通过。

**反指标（干货包不是这些东西）**：不是摘要（不做改写）、不是文档搬运（不做全文）、
不是知识图谱（不建实体库）。它是一份**逐字、带哈希、装得进预算的开工清单**。

---

## 5. A/B 实证（离线，40 任务）

### 5.1 留出构造

* 查询段 = 报告标题 + §1 案例介绍/§2 方法（`ab.py: QUERY_HEADINGS`）；
* 目标段 = §5 注意点 / §6 误差判定 / 局限（`TARGET_HEADINGS`），**与查询段不重叠**；
* 语料 = 仓内全部 markdown 事实单元（本仓 24 311 条；A/B 语料=报告类文档 5 219 条）；
* 任务数 40（仓内 `docs/dev_report_*.md` + `dev_report_*.md` 中同时具备查询段与目标段者）。

四臂：`head_dump`（读自己文档 top-down）、`bm25_raw`（纯词面检索、无策展）、`brief_global`（策展、无局部性）、
`brief_local`（策展 + 两流配比 = 本协议默认）。

### 5.2 结果（`docs/context_craft_ab/ab_results.md`，95% CI 见原表）

| 预算 | 臂 | 留出召回 | 判据召回 | 闸门命中 | 决策行占比 |
|---|---|---|---|---|---|
| 400 | head_dump | 0.005 | 0.000 | 0.000 | 0.245 |
| 400 | bm25_raw | 0.103 | 0.104 | 0.225 | 0.287 |
| 400 | brief_global | 0.100 | 0.227 | 0.550 | 0.726 |
| 400 | **brief_local** | **0.306** | **0.461** | **0.750** | 0.593 |
| 800 | head_dump | 0.257 | 0.206 | 0.250 | 0.276 |
| 800 | bm25_raw | 0.215 | 0.175 | 0.400 | 0.301 |
| 800 | brief_global | 0.137 | 0.316 | 0.600 | 0.776 |
| 800 | **brief_local** | **0.513** | **0.544** | **0.800** | 0.487 |
| 1200 | head_dump | 0.636 | 0.436 | 0.550 | 0.299 |
| 1200 | brief_local | **0.697** | **0.647** | **0.825** | 0.457 |
| 1600 | head_dump | 0.897 | 0.666 | 0.725 | 0.313 |
| 1600 | brief_local | 0.815 | **0.725** | **0.825** | 0.480 |

（1200/1600 上另两臂：bm25_raw 0.291/0.340、brief_global 0.173/0.208，均被 brief_local 大幅越过。）

三条读数：

1. **全局策展单独用是负收益**（brief_global 全面低于 head_dump 的召回）——这是第一轮的诚实负面结果，
   也是两流配比存在的理由。
2. **紧预算区间（400–1200）干货包明显占优**：召回 2–60×、闸门命中 0.75–0.825 vs 0.00–0.55。
   紧预算才是真实工况（上下文预算从来不够用）。
3. **大预算下退化为平手**：1600 token 时 read-own-doc 的召回 0.897 > 0.815（CI 相接，不算显著），
   但判据召回（0.666 vs 0.725）与闸门（0.725 vs 0.825）干货包仍领先。结论要写成
   "**干货包的价值在大预算下收缩**"，而不是"永远更好"——这是本节的 honest 边界。

---

## 6. 三层记忆与留去规则

| 层 | 内容 | 预算 | 半衰期（单） | 可遗忘 |
|---|---|---|---|---|
| L1 全局 bootstrap | 身份/规则/陷阱/打法头部 | 1400 | ∞ | 否（`stable=True` 保护，append-only） |
| L2 席位个体 | 本席方法偏好、局部结论 | 900 | 12 | 是（冷却 6 单后） |
| L3 knowledge_base | 跨单沉淀的 KO/裁定/pin | 2200 | 30 | 是（冷却 15 单后） |

留存分：`score = kind_weight × 0.5^(age/half_life) + 0.10×ln(1+use_count)`（`memory.py`）。

留去规则（按优先级执行）：

1. **稳定头**：`stable=True` 永不遗忘；超预算只报 `overflow_tokens`，不静默丢。
2. **保留底线**：`TOLERANCE/NEGATIVE/TRAP` 各保 1 条，即使超预算（宁可超也不丢验收线，且超了要报数）。
3. **近重复折叠**：同一层内 3-gram Jaccard ≥ 0.85 折叠为一条，保留 `use_count` 最高者。
4. **按分数填预算**，填不下的条目才进入遗忘候选。
5. **遗忘双条件**：`use_count == 0` **且** `age ≥ min_age_for_forget`。两条缺一，宁可溢出不丢。

---

## 7. 压缩节奏

`should_compress()`：**任一条件成立即压缩**——距上次压缩 ≥ 5 单（时间节奏），或某层用量 ≥ 85% 预算
（容量节奏）。只做容量节奏会在小层上抖，只做时间节奏会在大批新知识到来时撑爆。

## 8. 遗忘协议（遗忘 ≠ 删除）

对每条被遗忘条目依次执行：

1. 归档：`archive/<layer>/<item_id>.md`，front-matter 带 `sha256` 与来源；
2. 追加写 `forget_log.jsonl`：`ts / mission_index / layer / item_id / kind / sha256 / reason /
   retention_score / use_count / age / archive_path`（**只追加，不复写**）；
3. 重写层文件（内容为 keep 集）。

`verify_forget_log()` 逐条回读归档文件、重新计算正文 sha256，给出 `{records, ok, bad, problems}`。
测试 `test_forget_log_is_append_only_and_rehashable` 断言日志只增长且 ok == records。

## 9. use_count 记账

`touch(entries, item_id, mission_index)` 在**消费**时（干货包被组装进某单）调用，写
`use_count += 1`、`last_used_mission = mission`。衰减只作用于"没人再用"的条目，因此记账点必须是
真实消费，不能是压缩时刻（否则每条都会被刷成"刚用过"，衰减失效）。

### 9.1 已知局限：复用密集型工作负载会溢出不丢

`docs/context_craft_ab/simulation.json`（60 单合成负载，每单新增 4–9 条 + 随机复用 5 条）：
54 次压缩、189 条遗忘（全部归档可复算），但终态 L2 2400 / L3 4457 token，**超过各自的 900 / 2200 预算**，
`overflow_tokens` 如实报出。原因是复用密集时几乎所有条目都满足"最近用过"，双条件遗忘规则不放行。
这是**已知的设计取舍**（宁溢出不丢验收线），下一版需要引入"层内硬上限 + 按分数降级到归档索引"的策略；
本版不掩饰这个缺陷。

---

## 10. 用法

```bash
# 组装一个干货包（写 .json 供审计 + .md 供人读）
python -m tools.context_craft.cli pack --root . --pattern 'docs/*.md' 'dev_report_*.md' \
    --query "<本单要打的靶子要点>" --budget 900 --profile verification --out brief

# 零手抄闸门：回读源文件复算每一行哈希（失败 exit 1）
python -m tools.context_craft.cli verify --brief brief.json --root .

# 三层记忆压缩（触发节奏 + 遗忘日志 + 归档复算）
python -m tools.context_craft.cli memory --memory mem.json --mission-index 42 --out-dir mem/

# A/B 复现
python -m tools.context_craft.cli ab --root . --pattern 'docs/dev_report_*.md' 'dev_report_*.md' \
    --budgets 400 800 1200 1600 --limit 40 --out-dir docs/context_craft_ab
```

## 11. 未完成 / 需派单的部分

* 验收条款 ③ 要求"注入干货包 vs 裸单"的**在线**对照 ≥40 单。本单交付的是**离线留出式**代理评估
  （40 任务 × 4 臂 × 4 预算），它测的是"该进窗口的东西有没有进窗口"，不是"任务质量是否更高"。
  在线 A/B 需要成对派单 + 盲评，属跨席协调，建议派 40 单（20 对，模型钉扎，评审异族）——见 §12 建议条目。
* 语料仅本仓；跨仓迁移性未测（工具本身零领域语义，理论上可迁）。
* token 预估器是近似（无真分词器），跨模型的实际计费未校准。