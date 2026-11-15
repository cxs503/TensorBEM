# 第 8 章 冰区操作与法规判据：Polar Code、PWOM 与风险指数

> 独立讲授单元 ch08/09 · 预计 40 min · 现场计算：否（法规判据章）

## 0. 学习目标

1. 说清 Polar Code 的船舶分类 A/B/C 与对应冰况（medium / thin first-year ice 的厚度带）；
2. 说明 PWOM（Polar Water Operational Manual）与 Polar Ship Certificate 在"从设计到营运"
   链条中的位置；
3. 用 POLARIS 的风险指数框架把"船级 + 冰型"转成可追溯的营运限值（RIO）；
4. 判定一份冰区航行方案的法规证据是否完整。

### 0.1 前置知识

* 第 1 章的冰型与冰厚；
* 第 6/7 章的冰级体系（IACS PC、FSICR、RMRS）。

## 1. 物理与工程背景

冰区航行是两个世界的交界：**设计侧**（冰级、结构、机械）与**营运侧**（航速、航线、
护航、排放）。法规的作用是把两侧用可核查的语言连起来：

* IMO **Polar Code**（MSC.385(94)，2017-01-01 生效）给出船舶分类（A/B/C）、证书、
  PWOM 与极地水域防污染条款；
* 冰级（IACS UR I1/I2/I3、FSICR、RMRS）给出**设计**侧的冰况与结构要求；
* 营运侧的量化工具之一是 **POLARIS**：把船舶冰级与冰型按矩阵打分，得到
  Risk Index Outcome（RIO），作为"能不能走、怎么走"的可追溯依据。

## 2. 方法与关键设置（判据怎么用）

| 判据项 | 内容 | 出处 |
|---|---|---|
| 船舶分类 | A 类：至少 medium first-year ice（约 0.70–1.20 m）；B 类：thin first-year ice（约 0.30–0.70 m）；C 类：敞水或更轻冰况 | Polar Code (MSC.385(94)) |
| 手册 | PWOM 必须随船，且其结论须引用可核查的冰情与风险指数 | Polar Code |
| 风险指数 | POLARIS：`RIO = Σ (RIV × 冰型面积占比)`，`RIO >= 0` 视为可继续 | Stoddard et al. (2016) |
| 波罗的海 | FSICR 1A Super / 1A / 1B / 1C / II | TRAFI (2017) |
| 北极航线 | RMRS Arc4–Arc9、Ice1–Ice3 | RMRS Rules |
| 防污染 | 极地水域排放限制（油类、NLS、污水、垃圾） | Polar Code Part II-A（引 MARPOL） |

三条纪律：

1. **A/B/C 不是冰级**：它是 Polar Code 的船舶分类，与 PC1–PC7 是两套语言，混用会让
   合规性论证直接失效；
2. **RIO 是决策辅助不是许可**：`RIO >= 0` 表示"在该冰型/该船级下风险可接受"，
   不表示"一定能走"（还要看冰情预报、护航、船长资质）；
3. **引文必须可核**：法规条款号、版本、生效日期缺一不可，禁止"某规范规定"式表述。

## 3. 判据表

<!-- AUTO-CRITERIA:ch08 -->
判据表（ch08）：可计算 0 行、其中通过 0 行、未通过 0 行；文献/法规行 4 行。

| 判据 ID | 判据（可检验陈述） | 阈值/区间 | 本次实测 | 结论 | 证据 | 出处 |
|---|---|---|---|---|---|---|
| CH08-C1 | Polar Code 船舶分为 A/B/C 三类：A 类对应至少 medium first-year ice（0.70-1.20 m），B 类对应 thin first-year ice（0.30-0.70 m），C 类对应敞水或更轻冰况 | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | IMO Polar Code (International Code for Ships Operating in Polar Waters), MSC.385(94), in force 2017-01-01. |
| CH08-C2 | 冰区营运须随船携带 PWOM（Polar Water Operational Manual），其结论必须引用可核查的冰情与风险指数 | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | IMO Polar Code (International Code for Ships Operating in Polar Waters), MSC.385(94), in force 2017-01-01. |
| CH08-C3 | POLARIS 风险指数法把船舶冰级与冰型按 RIO/RISK 矩阵打分，可作为冰区航速/航线决策的可追溯依据（不是本仓计算能力） | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | Stoddard, M. A., Etienne, L., Fournier, M., Pelot, R. & Beveridge, L. (2016). Making sense of Arctic maritime traffic using POLARIS. IOP Conf. Ser. Earth Environ. Sci. 34, 012034. |
| CH08-C4 | 波罗的海 1A Super 冰级与 Polar Class 不可直接互换：两套体系的等效力须逐项论证 | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | Finnish-Swedish Ice Class Rules (TRAFI ice class regulations, 2017). |
<!-- /AUTO-CRITERIA:ch08 -->

## 4. 怎么核对一份冰区航行方案（替代图）

1. Polar Ship Certificate 与船舶分类（A/B/C）是否齐备？
2. PWOM 是否存在，其冰况假设是否与本次航线一致？
3. 风险指数（POLARIS）输入是否可追溯：冰型、密集度、船级各自出处？
4. 设计侧冰级（PC / FSICR / RMRS）与营运侧假设是否**同一套**冰况？
5. 排放（油类/NLS/污水/垃圾）是否按 Polar Code Part II-A 核对？

## 5. 注意点与局限（证据分级）

1. **WITHHELD**：本章不含任何本仓计算；所有数值是法规/文献条款量级，
   规范原文为准。
2. **法规版本强时效**：Polar Code 及其引用的 MARPOL 条款会修订，引用必须带版本/生效日。
3. **冰级不可换算**：FSICR 1A Super 与 PC6 的等效力必须逐项论证（冰厚、结构冗余、
   机械冗余、操纵性）。
4. **RIO 的输入不确定性**：冰密集度/冰型来自预报与遥感，输入误差会直接传进 RIO；
   报告应给区间。
5. **本教程不提供合规结论**：合规判定属于船级社与主管机关。

## 6. 练习

1. 一条持 PC7 冰级的船被安排进入 1.0 m 厚一年冰区，用 Polar Code 分类语言与 POLARIS
   框架分别写出判断（两套语言都要给出）。
2. 说明 PWOM 与"航线冰情预报"的关系：谁是约束，谁是输入？
3. 给出一条你认为最常见的"冰区方案法规证据缺口"，并写出补齐所需的三份文件。

## 7. 参考文献

1. IMO Polar Code（International Code for Ships Operating in Polar Waters），MSC.385(94)，
   2017-01-01 生效。
2. IACS Unified Requirements I1 / I2 / I3.
3. Finnish–Swedish Ice Class Rules（TRAFI, 2017）.
4. Stoddard, M. A., Etienne, L., Fournier, M., Pelot, R. & Beveridge, L. (2016). Making
   sense of Arctic maritime traffic using POLARIS. *IOP Conf. Ser. Earth Environ. Sci.* **34**,
   012034.
5. RMRS Rules for the Classification and Construction of Sea-Going Ships（冰级章节）.

## 8. 复现命令

本章无可复现计算（法规判据章）。若需查平台侧的冰区资产/知识条目状态：

```bash
runuser -u postgres -- psql -d tensorplatform -c \
 "SELECT slug,status FROM feature_ledger WHERE slug LIKE 'ICE-SHOW-%' ORDER BY slug"
```