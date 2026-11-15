# 第 6 章 冰阻力：三分量分解、经验链与精度判据

> 独立讲授单元 ch06/09 · 预计 45 min · 现场计算：否（实现不在本仓，按 `WITHHELD` 记账）

## 0. 学习目标

学完本章，你应当能够：

1. 写出冰阻力的三分量分解 `R_ice = R_break + R_subm + R_open`，并说明每个分量的
   物理机制与主导参数（冰厚、弯曲强度、首部倾角、摩擦系数、航速）；
2. 说清"阻力公式里的标定常数（`c_break`、`c_submersion`）为什么是必需项"，以及未标定
   公式为什么不得直接用于设计（>15% 红线）；
3. 判定一份冰阻力结果的证据等级：是模型试验/实船对标（≤10%），还是"公式一算"；
4. 说明碎冰航道（brash ice channel）阻力为什么不是"平整冰阻力乘一个系数"。

### 0.1 前置知识

* 第 1 章的冰厚/冰强度区间；
* 敞水阻力与有效功率的基本关系（`P_E = R_T V`）；
* 量纲分析：为什么阻力系数式里会出现 `rho_i g h_i^2`（单位长度力的量纲）。

## 1. 物理与工程背景

冰区航速与功率的瓶颈是冰阻力。工程上可用的模型分三类：

| 类别 | 代表 | 适用 | 局限 |
|---|---|---|---|
| 经验/半经验公式 | Lindqvist (1989) 平整冰；Riska et al. (1997) 碎冰航道 | 概念设计、航速-功率预估 | 必须用模型试验/实船数据标定 |
| 数值（DEM/FEM） | 碎冰-船耦合 | 局部载荷、冰脊、复杂冰况 | 计算贵、接触参数不确定（`TensorDEM` 线） |
| 统计/数据驱动 | 实船航速-冰情回归 | 营运决策 | 外推风险大，需给出置信区间 |

本章只讲**判据**：这三类结果各自在什么条件下可以信、误差红线在哪、该向哪条能力线要
实现。冰阻力的实现**不在本仓**（`TensorBEM` 是边界元水动力库，不含冰-船接触与破碎），
平台侧由下列条目承担：

* `C1-ICE-RESIST`（冰阻力全谱经验链，已 done）；
* `GAP-ICE-LINEAGE-RECOMPUTE`（阻力谱系公式链搬入可跑脚本 + 标定，已 done）；
* `GAP-ICE-UIKKU-DEM`（DEM 平冰阻力对 MT Uikku 实测工况的预测，已 done）；
* `GAP-ICE-POROSITY-SCALING`（碎冰堆积孔隙率-阻力标度律拟合质量，已 done）；
* `KPI-RES-5-10`（静水阻力 ≤5% / 冰区 ≤10%，在飞）。

## 2. 方法与关键设置（判据怎么用）

### 2.1 三分量分解

```
R_ice = R_break + R_subm + R_open
```

* `R_break`：弯曲破冰分量。冰板被首部压弯至断裂，量级 `~ sigma_f h_i^2` 除以某个
  与首部倾角、摩擦有关的几何因子；
* `R_subm`：浸没分量（碎块被压入水下再浮起），与冰厚、航速、摩擦系数、首部形状强相关，
  通常**是最大的分量**；
* `R_open`：敞水阻力分量（同航速），**不是小量**——它随航速平方增长，在高速段可能
  超过破冰分量。

工程含义：把冰阻力当成"冰厚乘常数"是最常见的错误。冰厚只在 `R_break`（`h_i^2`）与
`R_subm`（线性到二次）里出现，而 `R_open` 与冰厚无关。

### 2.2 标定常数

经验公式里的 `c_break`、`c_submersion`（以及摩擦系数）是**必须标定**的自由参数：

* 标定数据：模型试验（冰池）或实船试航；
* 未标定的公式在特定船型/冰况上偏差可以远大于 15% 红线——平台
  `GAP-ICE-LINEAGE-RECOMPUTE` 条目记录了 17.26% 的偏差量级并给出了处理机制，
  具体结论以该条目为准（本教程不转述）；
* 因此判据是：**标定链条（数据来源 → 参数 → 复算误差）必须可追溯**，只报点值不算证据。

### 2.3 碎冰航道

碎冰航道（brash ice channel）里，船面对的是密集碎冰而非连续板。与平整冰的三点差别：

1. 阻力主要来自碎冰的**堆积与推挤**（`R_subm` 类机制），不是弯曲断裂；
2. 航道宽度、碎冰厚度、密实度（孔隙率）是主导参数，**单调性**可先验判断：
   航道越窄/碎冰越厚，阻力越大；
3. 孔隙率-阻力标度律需要拟合，且拟合质量（`R^2`）本身就是证据等级的一部分——平台
   `GAP-ICE-POROSITY-SCALING` 条目的验收正是把 `R^2` 从 0.26 提到 ≥0.5。

## 3. 判据表

<!-- AUTO-CRITERIA:ch06 -->
判据表（ch06）：可计算 0 行、其中通过 0 行、未通过 0 行；文献/法规行 4 行。

| 判据 ID | 判据（可检验陈述） | 阈值/区间 | 本次实测 | 结论 | 证据 | 出处 |
|---|---|---|---|---|---|---|
| CH06-C1 | 冰阻力按 R_ice = R_break + R_subm + R_open 分解（破冰/浸没/敞水三分量），分项定义与 Lindqvist 方法一致；本仓不含该实现，实现与标定在 ship-performance-platform | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | Lindqvist, G. (1989). A straightforward method for calculation of ice resistance of ships. POAC'89, vol. 2, 722-735. |
| CH06-C2 | 冰区阻力复算精度判据：对标模型试验/实船实测误差 ≤ 10%（平台 KPI 静水 ≤5% / 冰区 ≤10%） | 文献/法规条款 | 文献/法规条款 | — | REAL | TensorPlatform 精度标准：对标实验/参考值误差 >15% 不可接受；算例池容差 5%；平台判据 3%。 |
| CH06-C3 | 碎冰航道（brash ice channel）阻力随航道厚度与航道宽度单调增；冰厚 0.9 m 量级的一年冰航道对应 FSICR 1A Super 设计工况 | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | Riska, K., Wilhelmson, M., Englund, K. & Leiviska, T. (1997). Performance of a chemical tanker in level ice - full scale and model scale tests. POAC'97. |
| CH06-C4 | 冰阻力全谱经验链（含 c_break / c_submersion 标定）已由平台 C1-ICE-RESIST 条目落地，本教程只做判据教学，不重复实现 | 文献/法规条款 | 文献/法规条款 | — | REAL | Lindqvist, G. (1989). A straightforward method for calculation of ice resistance of ships. POAC'89, vol. 2, 722-735. |
<!-- /AUTO-CRITERIA:ch06 -->

判据解读：

* **CH06-C1（WITHHELD）**：分解式的定义可以教、可以查，但"本仓算出的冰阻力"不存在，
  所以证据等级只能是 `WITHHELD`。写 `REAL` 就是造假；
* **CH06-C2（REAL，容差判据）**：容差本身是平台标准，可以直接当判据引用；
* **CH06-C3（WITHHELD）**：碎冰航道的单调性与冰级对应关系是文献/规范陈述，
  实现与标定在平台侧；
* **CH06-C4（REAL）**：把"实现归属"写成判据，防止后续有人在本仓重造一遍经验链
  （重复劳动 + 两个版本的系数互不兼容）。

## 4. 怎么核对一份冰阻力报告（替代图）

本章不产图，产出是一张**核对清单**。拿到一份冰阻力结果，按顺序问：

1. **分解了吗？** 三分量是否分开给？只给总值的结果无法定位误差来源；
2. **`R_open` 用的是同一航速/同一阻力模型吗？** 混用不同敞水模型是最常见的偏差源；
3. **标定数据是什么？** 模型试验（哪条船、什么冰况）还是实船试航？误差是多少？
4. **精度对上了吗？** 对标误差是否 ≤10%（`KPI-RES-5-10`）？若 >15% 却以"工程近似"
   措辞放行，按平台纪律判为不可接受；
5. **外推了吗？** 冰厚/航速/航道宽度是否超出标定域？超出就必须给区间而不是点值。

## 5. 注意点与局限（证据分级）

1. **WITHHELD**：本仓无冰阻力实现，本章任何"数字"都只以文献/规范条款形式出现，
   且不给出未经标定的计算值。
2. **经验公式的适用域**：Lindqvist 方法面向平整一年冰、首部有倾角的船；对冰脊、多年冰、
   极宽船体、倒车工况都不适用。
3. **摩擦系数是隐藏参数**：冰-船摩擦系数取值范围宽（0.05–0.15 量级），对 `R_subm`
   影响显著；报告必须写明取值与出处。
4. **DEM 路线不是"更准"，是"另一种不确定"**：接触参数、形状、时间步都会影响结果，
   `GAP-ICE-FLOE-VALID`（在飞）正是标定休止角/力链的条目。
5. **不要跨船型搬运标定常数**：`c_break/c_submersion` 与首部形状强相关。

## 6. 练习

1. 给出两条船（`L = 120 m` 与 `L = 200 m`，同冰厚），解释为什么"冰阻力随船长增加而
   减小（单位排水量）"但仍然可能与冰厚近似平方相关。
2. 某报告的冰区阻力对标误差 12%，写一段判定：能不能用于设计？还需要什么才能用？
3. 设计一个"标定链条可追溯"的最小证据包（提示：原始试验表 + 参数 + 复算脚本 + 误差）。

## 7. 参考文献

1. Lindqvist, G. (1989). A straightforward method for calculation of ice resistance of
   ships. *POAC'89*, vol. 2, 722–735.
2. Riska, K., Wilhelmson, M., Englund, K. & Leiviskä, T. (1997). Performance of a chemical
   tanker in level ice — full scale and model scale tests. *POAC'97*.
3. Kujala, P. & Arughadhoss, S. (2012). Statistical analysis of ice crushing pressures on
   a ship's hull during hull–ice interaction. *Cold Reg. Sci. Technol.* **70**, 1–11.
4. ISO 19906:2019（冰载荷与设计原则，见 ch07）。

## 8. 复现命令

本章无可复现计算（`WITHHELD`）。核对平台条目的状态：

```bash
runuser -u postgres -- psql -d tensorplatform -c \
 "SELECT slug,status FROM feature_ledger WHERE slug LIKE '%ICE-RESIST%' OR slug LIKE 'GAP-ICE-%' ORDER BY slug"
```