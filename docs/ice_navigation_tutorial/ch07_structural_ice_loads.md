# 第 7 章 结构冰载荷与冰级规范：压力-面积、冰级与低温材料

> 独立讲授单元 ch07/09 · 预计 45 min · 现场计算：否（FEM 不在本仓）

## 0. 学习目标

1. 解释冰压力为什么随接触面积**减小**（压力-面积关系），并说明为什么"设计压力取一个
   常数 MPa"是错的；
2. 把冰级（IACS Polar Class PC1–PC7、FSICR、RMRS）与"设计冰厚/冰况"成对讲清楚；
3. 说清低温材料判据（韧性等级、脆性转变）在冰区结构设计里的位置；
4. 判定冰载荷统计分位（p95/p99）报告是否合格：必须给 `n_eff` 修正的置信区间。

### 0.1 前置知识

* 第 1 章的冰强度区间、第 6 章的阻力/载荷区别（阻力是船-冰整体，载荷是局部/整体结构响应）；
* 结构可靠度基础：分位数、有效样本数、置信区间。

## 1. 物理与工程背景

冰与船体/海洋结构的相互作用产生两类载荷：

* **局部载荷**：挤压（crushing）在接触区产生的高压，决定板格、肋骨、局部屈曲；
* **整体载荷**：整船或整个结构的冰力，决定总纵强度与基础设计。

冰的挤压是**脆性-韧性混合**过程：接触面积越大，高压力点越不容易同时出现，因此
**平均压力随面积下降**，这就是压力-面积关系（pressure–area）。工程规范把这条关系写成
幂律包络，设计局部压力不得突破包络。实测压力-面积分析见 Kujala & Arughadhoss (2012)。

## 2. 方法与关键设置（判据怎么用）

| 主题 | 规范/文献 | 用法 |
|---|---|---|
| 局部/整体冰压力 | ISO 19906:2019 | 压力-面积包络、整体冰力公式；具体系数与指数以标准条款为准 |
| 冰级与船体结构 | IACS UR I1（冰级描述）、UR I2（结构设计）、UR I3（机械） | PC1–PC7 与设计冰况一一对应；结构按 UR I2 校核 |
| 波罗的海冰级 | FSICR（1A Super / 1A / 1B / 1C / II） | 波罗的海航线的等效设计工况 |
| 俄罗斯冰级 | RMRS（Arc4–Arc9、Ice1–Ice3） | 北极航线（NSR）常用 |
| 冰载荷统计 | Kujala & Arughadhoss (2012)；平台 `GAP-ICE-TAIL-NEFF` | 设计分位 p95/p99 必须给有效样本数修正的 CI |
| 碎冰堆积/挤压欠载 | 平台 `GAP-ICE-CRUSH-PACKING`（已 done） | ISO 挤压模型在密集碎冰场内的欠载归因 |
| 冰板/冰带 FEM | 平台 `H3-ICE-PLATE-FEM`、`H4-ICE-BELT-FEM`（已 done） | 弯曲破坏与局部结构响应，不在本仓 |

三条纪律：

1. **冰级与冰厚必须成对给**：说"PC6 船"而不说设计冰况等于没说；
2. **压力-面积关系不得内插到标定域外**：小面积高压区的数据最少，外推最危险；
3. **统计分位必须带样本量**：冰载荷是重尾分布，`n_eff` 修正前后区间可能差一个量级。

## 3. 判据表

<!-- AUTO-CRITERIA:ch07 -->
判据表（ch07）：可计算 0 行、其中通过 0 行、未通过 0 行；文献/法规行 4 行。

| 判据 ID | 判据（可检验陈述） | 阈值/区间 | 本次实测 | 结论 | 证据 | 出处 |
|---|---|---|---|---|---|---|
| CH07-C1 | 局部冰压力随接触面积近似幂律衰减，压力-面积指数约 -0.5，设计局部压力不得超过该包络 | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | ISO 19906:2019. Petroleum and natural gas industries - Arctic offshore structures. ISO, Geneva. |
| CH07-C2 | 冰级与设计冰厚须成对给出：Polar Class PC1-PC7 与 IACS UR I1 冰级描述一一对应，PC6 对应 medium first-year ice 量级 | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | IACS Unified Requirements I1 (Polar Class descriptions), I2 (Polar Class hull structural design), I3 (machinery). |
| CH07-C3 | 低温材料判据：冰区结构钢须按 -40 °C 量级韧性等级（E/EH 级）选用，并单独校核脆性转变 | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | IACS Unified Requirements I1 (Polar Class descriptions), I2 (Polar Class hull structural design), I3 (machinery). |
| CH07-C4 | 冰载荷统计分位（p95/p99）必须给 n_eff 修正置信区间，不得只报点值（平台 GAP-ICE-TAIL-NEFF 在飞条目） | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | Kujala, P. & Arughadhoss, S. (2012). Statistical analysis of ice crushing pressures on a ship's hull during hull-ice interaction. Cold Regions Science and Technology 70, 1-11. |
<!-- /AUTO-CRITERIA:ch07 -->

## 4. 怎么核对一份冰载荷报告（替代图）

1. 载荷是**局部**还是**整体**？两者不能用同一个压力值；
2. 压力-面积包络用了哪个规范/指数？接触面积范围是否在包络标定域内？
3. 冰级是否与设计冰厚/冰况成对出现？冰级来源（IACS/FSICR/RMRS）是否写明、能否互认？
4. 材料：低温韧性等级（E/EH 级，-40 °C 量级）与脆性转变是否单独校核？
5. 统计：p95/p99 是否给了 `n_eff` 修正 CI？样本量多少？外推区间是否给出？

## 5. 注意点与局限（证据分级）

1. **WITHHELD**：本仓不做结构 FEM，也不做冰载荷统计；本章全部为规范/文献判据。
2. **规范版本敏感**：ISO 19906 不同版本的压力-面积系数与指数有差异，引用必须写版本年份。
3. **冰级不可直接互换**：FSICR 1A Super 与 IACS PC 的等效力需要逐项论证（冰厚、结构
   冗余、机械），不能做"换算表"。
4. **FEM 的输入不确定性大于求解误差**：冰材料参数（强度、模量、破坏准则）区间宽，
   精致的网格并不会让结论变可靠。
5. **本教程给的数值一律是量级/条款级**，不替代规范原文。

## 6. 练习

1. 说明为什么"设计压力 3 MPa，全船统一取值"在物理上不成立（用压力-面积关系论证）。
2. 一艘船同时持 PC6 与 1A Super 双冰级，写出你认为需要逐项核对的三项内容。
3. 若一份报告给出 p99 冰载荷但样本量 `n = 40`，请估算其 95% 置信区间相对宽度
   （提示：重尾分布下不要用正态近似，至少给出方法与假设）。

## 7. 参考文献

1. ISO 19906:2019. *Petroleum and natural gas industries — Arctic offshore structures*.
2. IACS Unified Requirements I1 / I2 / I3.
3. Finnish–Swedish Ice Class Rules（TRAFI ice class regulations, 2017）.
4. Kujala, P. & Arughadhoss, S. (2012). *Cold Reg. Sci. Technol.* **70**, 1–11.
5. Sanderson, T. J. O. (1988). *Ice Mechanics: Risks to Offshore Structures*. Graham & Trotman.

## 8. 复现命令

本章无可复现计算（`WITHHELD`）。相关实现的归属：

```bash
runuser -u postgres -- psql -d tensorplatform -c \
 "SELECT slug,status FROM feature_ledger WHERE slug LIKE '%ICE-%FEM%' OR slug LIKE 'GAP-ICE-TAIL%' OR slug LIKE 'GAP-ICE-CRUSH%'"
```