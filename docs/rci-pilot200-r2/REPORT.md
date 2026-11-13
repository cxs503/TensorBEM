# RCI pilot-200 第 2 轮：KO 类型判定 v2（弃权门 + 优先级规则）与前/后判据实测

> 锚点 `KBM-RCI-S2-PILOT200`（open）· 单 m17036 · 基线 = m15462 第 1 轮工件（冻结快照）
> 工件：`frozen/`（v1 objects/documents/sample + 语料快照）· `review/`（round-3 判读 + 本轮盲判）·
> `objects_v2.jsonl` · `metrics_v2.json` · `figures/fig_type_fit_v1_vs_v2.png`
> 工具：`tools/rci_ko_type_v2.py`（规则引擎，含 `classify`/`render-blind`）、`tools/rci_r2_metrics.py`（判分+作图）
> 测试：`tests/test_rci_ko_type_v2.py`（16 passed）

## ① 案例介绍：欠的是什么账

第 1 轮试点（m15462）交付 278 个 KO、36 条逐字段人工核验，交差审计（report 14918）判 L2 confirmed，
但留下两条 honest FAIL：**KO 类型适配 35.2%（<<95% 验收线）**、conditions 未臆造 92.6%（<95%）。
本轮只打第一条：为什么「类型适配」这么低，能不能修，修了以后**在没被用来调规则的样本上**还剩多少。

样本与基线（全部继承，不重采）：

| 参数 | 取值 |
|---|---|
| 语料 | 15 个生态仓 5228 篇 ≥400 字符 `.md/.rst`，分层（仓×四类型×A/B/C）131 层抽 **200** 篇 |
| 冻结快照 | `frozen/corpus/`（2.2 MB，逐字节副本，判定全部在快照上做） |
| KO 总数 | **278**（`frozen/objects.jsonl`，v1 五模板：simulation 144 / technical-plan 55 / innovation 31 / literature 26 / algorithm 22） |
| 基线判读 | round-3 sheet **36 条**逐字段判读（`review/review_manual_round3.json`）：claim 忠实/证据可达/conditions/质量可复算 四项 36/36；类型适配 **8/36 = 0.2222** |
| 本轮盲判 | 从 278 中剔除已判 36 条后按 v1 类型分层各取 8（共 40 抽样），**实际盲判 21 条**（预算耗尽，余 19 条记 `not_run`，不计入通过） |

## ② 怎么做的

1. **失败模式归类（读 round-3 的 36 条 note）**：28 条 unfit 里，绝大多数不是「选错了五个模板之一」，
   而是**根本不该产出 KO**：许可证范围 1、README/AGENTS 自述 4、CHANGELOG/合并/CI 流水 6、
   YAML/命令片段 2、褒扬清单/计划状态/开放问题 4、教科书定义 3、元数据头 2 …… v1 抽取器
   **没有弃权通道**，只能把「不是知识」的内容硬塞进五个正类，这是类型适配低的机制原因。
2. **v2 规则引擎**（`tools/rci_ko_type_v2.py`）：结构只改两处，不含任何领域词条：
   - **A 弃权门**（先于正类判定）：license/自述文档/CHANGELOG/褒扬清单/计划状态/开放问题/
     配置片段/裸命令/状态行/元数据头 → `non_ko:<reason>`（10 个登记理由）；
   - **B 正类优先级**：innovation > simulation（需数字+基准/容差/收敛词）> algorithm（格式/方程/内核）
     > literature（外部锚）> technical-plan（接口/契约/参数取值）> **definition**（教科书定义）；
     全部不中 → `non_ko:unclassifiable_claim`（**宁可弃权，不许猜**）。
   - 铁律不变：只重贴标签，**不改写** statement/quote/evidence/quality（有测试守卫）。
3. **判据（先于盲判写死）**：GT 词表 = 五模板 + `definition` + `non_ko`；判读顺序 = 先问「是不是知识」，
   再按 innovation > simulation > algorithm > literature > technical-plan > definition 取核心内容；
   盲判上下文由 `render-blind` 生成，**抹掉 v1/v2 标签**（`review/review_context_v2_blind.md`）。
4. **两套测量**：(a) legacy36 = 用**既有** round-3 判读作 GT（有选择偏差，规则是看着这些 note 设计的）；
   (b) blind held-out = 未见样本独立盲判（无偏估计，但 n=21）。

## ③ 计算结果（程序化注入自 `metrics_v2.json`）

| 指标 | legacy36 (n=36) | blind held-out (n=21) | pooled (n=57) |
|---|---|---|---|
| v1 类型适配 fit7 | **0.2222** (8/36) | **0.2857** (6/21) | **0.2456** (14/57) |
| v2 类型适配 fit7 | **0.7778** (28/36) | **0.4286** (9/21) | **0.6491** (37/57) |
| GT∈五模板子集 fit5 v1 / v2 | 0.5333 / 0.5333 (n=15) | 0.4615 / **0.1538** (n=13) | 0.5000 / **0.3571** (n=28) |
| GT=non_ko 项上弃权召回 v1 / v2 | 0.00 / **0.95** | 0.00 / **0.88** | 0.00 / **0.93** |
| v2 误弃权数（真知识被判 non_ko） | 2 | **9** | 11 |
| 仅高置信 GT 子集 fit7 v1 / v2 | 0.70 / 0.70 (n=10) | 0.3529 / 0.4706 (n=17) | — |

v2 全语料标签分布（278）：non_ko 151（unclassifiable 97、plan_status 19、self_description 6、
changelog 8、command 7、metadata 4、config 3、status 3、license 2、open_questions 2），
正类 127（simulation 64、algorithm 33、technical-plan 13、innovation 11、definition 3、literature 3）。

## ④ 结果图

`figures/fig_type_fit_v1_vs_v2.png`（三面板）：(1) 同一批判读项上 v1/v2 类型适配对照 + 95% 验收线；
(2) v2 规则在 278 个 KO 上的标签分布；(3) 「GT=非知识」项上 v1/v2 的弃权召回。

## ⑤ 注意点（坑与局限）

- **偏差声明（最重要）**：v2 规则是在读过 round-3 的 36 条失败 note 之后设计的，所以 legacy36 上的
  0.222→0.778 **不可当作无偏改进**；高置信 GT 子集上 v1/v2 完全持平（0.70/0.70）说明这 0.556 的涨幅
  有一半来自「把模糊项判成 non_ko」这件本身就模糊的事。**无偏证据只有 blind 列。**
- **弃权换来的分数不是知识能力**：pooled fit7 0.2456→0.6491 中，0.93 的弃权召回贡献了绝大部分；
  GT 属五模板时 v2 反而**更差**（fit5 0.500→0.357，blind 0.462→0.154），误弃权 9/13。
  一句话：**新管线学会了对垃圾闭嘴，但没学会对知识分类**。
- **单片段判型的信息上限**：一个 KO 的输入只有 statement+quote+(path,行区间)。blind 里被判错的正类，
  多数是「同一句话既可读作方法也可读作结果」（如 DIT 案例设置含参数与判据）。要再上台阶，
  必须引入**章节角色**（该 claim 落在「怎么做的」还是「计算结果」）与跨 KO 上下文，而不是继续堆关键词。
- **本轮盲判只完成 21/40**（预算墙），`not_run` 如实记在 `review/adjudication_v2.json`，未计入通过。
- 冻结快照是活仓库的复制，复算请一律用 `frozen/corpus/` 与其 sha256（`frozen/objects.jsonl` 内嵌）。

## ⑥ 误差判定：是否 <3%？——不适用，且验收线**未达**

本单主量是**判读一致率**（离散计数），没有连续量的解析/文献基准，故不存在「<3%」口径；
验收线是审计给定的 **95% 类型适配**：

| 关键量 | 值 | 基准/验收线 | 误差 | 是否达标 |
|---|---|---|---|---|
| v2 类型适配（pooled fit7, n=57） | 0.6491 | 0.95 | **−0.3009 (−30.1 pp)** | **否** |
| v2 类型适配（blind held-out, n=21） | 0.4286 | 0.95 | **−0.5214 (−52.1 pp)** | **否** |
| 真实知识项上的正类适配（blind fit5） | 0.1538 | 0.95 | **−0.796** | **否** |
| 非知识项弃权召回（pooled） | 0.9286 | 1.00 | −0.0714 | 否（接近） |
| 证据指针不变式（v1→v2 改写 0 处） | 278/278 | 278/278 | 0 | **是** |

**结论（honest FAIL，不粉饰）**：第 1 轮「类型适配 35.2%」的机制原因（无弃权通道）已定位并修复，
弃权能力从 0 建到 0.93；但**验收线 95% 远未达到**，且无偏（blind）列只有 0.429，
正类判型在弃权门之后反而退化。本单**不主张** KBM-RCI-S2-PILOT200 达成验收；
建议拆出后继条目（slug 见下）把「章节角色 + 跨 KO 上下文」作为正类判型的主信号再测。

## 附：建议总账新条目（本单未做，留痕）

- `KBM-RCI-S2B-TYPEFIT-R3`：RCI·正类判型 v3（章节角色+跨 KO 上下文，弃权门保留）——
  验收：blind held-out ≥60 条，正类 fit5 ≥0.75 且 non_ko 弃权召回 ≥0.90，两量同时达标；域=TensorKnowledge。
- `KBM-RCI-S2B-PILOT200-REMAIN40`：完成本轮未判的 40−21=19 条盲判（补样本，纯执行）；域=TensorKnowledge。
