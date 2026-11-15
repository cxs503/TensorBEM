# 第 5 章 冰区船体六自由度 RAO 链：mass-loading 冰盖下的辐射与绕射

> 独立讲授单元 ch05/09 · 预计 45 min · 现场计算：否（引用本仓已合并证据）

## 0. 学习目标

学完本章，你应当能够：

1. 把第 2 章的界面 Robin 条件用于**三维船体**的辐射/绕射问题，并说清"冰盖只改变一个
   参数（`nu = k_i`）"为什么是一个可验证的强假设；
2. 说明冰区水动力结果里的两个记账陷阱：阻尼必须按 `omega/sqrt(g k_i)` 重标定、
   激励力对内部 `omega = sqrt(g nu)` 记账**不敏感**（因为入射势带补偿因子 `1/omega`）；
3. 读懂 `mu -> 0` 极限、深水有效性包络（`k_i H >= 3`）与传播分支三项验收门；
4. 判断一份"冰区 RAO"证据是否够用：几何是否先验证、冰区参数是否落域、真实船型是否
   有出处。

### 0.1 前置知识

* 第 2 章（Robin 条件、`k_i`）、第 3 章（冰下波长改变）；
* 频域耐波性：六自由度运动方程 `(-omega^2 (M + A) + i omega B + C) xi = F_wave`；
* 深水 Green 函数与"同一核函数、不同 `nu`"的想法（本仓 `free_surface_solver3d`）。

## 1. 物理与工程背景

船舶在冰区（覆盖稀疏冰或薄一年冰）时的波浪响应，两条链路同时改变：

1. **入射波场**：冰下波长变短（`k_i > k_0`），遭遇频率-波长对应关系变了；
2. **界面条件**：自由面从 `dphi/dz = (omega^2/g) phi` 变成 `dphi/dz = K(omega) phi`，
   `K = omega^2/(g - mu omega^2) = k_i`（深水）。

工程上关心的是：同一海况谱（例如 JONSWAP，`Tp = 8 s`）在有冰/无冰条件下，垂荡与纵摇
RAO 的峰值位置、峰值大小、以及波浪弯矩如何变。这决定了冰区航速与航线的选择裕度。

## 2. 方法与关键设置（本仓已实现的链路）

| 环节 | 实现 | 关键点 |
|---|---|---|
| 冰区波数 | `tensorbem.ice.mass_loading_wavenumber` | 深水闭式 `k_i = omega^2/(g - mu omega^2)`；有限水深由 `k tanh(kH) = K` 求根 |
| 自由面核 | `free_surface_solver3d` 的 `nu` 参数化核 | 冰盖**不**需要新核：只把 `nu` 从 `omega^2/g` 换成 `k_i` |
| 阻尼重标定 | `ice_rao.solve_radiation_3d_ice` | 求解器从 `nu` 反推 `omega_solver = sqrt(g nu)`；物理阻尼 `B = omega * Im(Z)`，须乘 `omega/omega_solver` |
| 激励力 | `ice_rao.solve_diffraction_3d_ice` | `F = -i omega rho int phi n dS`；对 `omega_solver` 记账不敏感（入射势带 `1/omega` 补偿） |
| RAO | `ice_rao.compute_ice_rao_spectrum` | 六自由度；谱级输出 |
| 验收 | `tests/test_ice_rao.py` | 结构判据（见下） |

三项验收门（`tests/test_ice_rao.py` 中已固定）：

* `mu -> 0` 极限必须精确复现敞水的附加质量、阻尼与激励力（抓阻尼重标定的记账错误）；
* 深水冰波数 `k_i` 必须等于 Robin 系数 `K`，且在有限水深求解器的深水极限下一致；
* 有效性包络（`k_i H >= 3`、传播分支）与输入校验必须**拒绝**而不是伪造结果。

算例（已合并证据 `docs/ice_water_ship.md` + `docs/ice_water_ship_numbers.md`）：
理想化 PC6 类冰区船 `L = 120 m`、`B = 24 m`、`T = 9 m`、`Cb = 0.719`（面元 1056），
冰盖 `h_i = 0.9 m`、`rho_i = 917`、`H = 200 m`，顶浪 `T = 4–14 s`，六自由度 RAO
在敞水/冰区两套条件下成对给出。**这些数字不在本章重打**：请直接读
`docs/ice_water_ship_numbers.md`（程序化渲染，无手抄）。

## 3. 判据表

<!-- AUTO-CRITERIA:ch05 -->
判据表（ch05）：可计算 0 行、其中通过 0 行、未通过 0 行；文献/法规行 4 行。

| 判据 ID | 判据（可检验陈述） | 阈值/区间 | 本次实测 | 结论 | 证据 | 出处 |
|---|---|---|---|---|---|---|
| CH05-C1 | 冰区船体六自由度 RAO 链在 mu -> 0 极限必须精确复现敞水附加质量、阻尼与激励力（阻尼须按 omega/sqrt(g k_i) 重标定） | 文献/法规条款 | 文献/法规条款 | — | REAL | TensorBEM tests/test_ice_rao.py and tests/test_ice_mass_loading_dispersion.py (merged verification tests of this repository). |
| CH05-C2 | 任何船体算例必须几何先验证：Cb / L / B / T 与型线参数一致才允许进入水动力计算 | 文献/法规条款 | 文献/法规条款 | — | REAL | TensorPlatform 精度标准：对标实验/参考值误差 >15% 不可接受；算例池容差 5%；平台判据 3%。 |
| CH05-C3 | 真实 PC6 冰区船型值表未取得（私有数据）：本仓 RAO 结果基于参数化船体，对真实船型的适用性证据不足 | 文献/法规条款 | 文献/法规条款 | — | WITHHELD | IACS Unified Requirements I1 (Polar Class descriptions), I2 (Polar Class hull structural design), I3 (machinery). |
| CH05-C4 | 算例池已合并证据：理想化 PC6 船（L=120 m, B=24 m, T=9 m, Cb=0.719）在 h_i=0.9 m 冰盖下 RAO 与敞水对照齐备，数字全部程序化渲染 | 文献/法规条款 | 文献/法规条款 | — | REAL | TensorBEM docs/ice_water_ship_numbers.md and docs/ice_water_ship_evidence (merged evidence bundle, programmatically rendered). |
<!-- /AUTO-CRITERIA:ch05 -->

判据解读：

* **CH05-C1** 是本章的核心可判伪项：`mu -> 0` 时冰区链路必须退化为敞水链路。这条比
  "与某文献某点吻合"更强，因为它排除了整条链路的记账错误（阻尼量纲、激励力相位）；
* **CH05-C2** 是本平台的几何纪律：任何船体算例必须先过 `Cb / L / B / T` 一致性门，
  几何错了后面的水动力数字全部作废；
* **CH05-C3（WITHHELD）** 明确记账：真实 PC6 型值表是私有数据，本仓用的是参数化船体，
  对真实船型的适用性**未证**；
* **CH05-C4** 指向已合并的证据包（不是本章作者的复述）。

## 4. 怎么读这套证据（替代图）

本章不画新图——它不产生新计算，画图会制造"看起来像新证据"的错觉。正确的读法是三步：

1. **先看几何门**：`docs/ice_water_ship.md` §1.2 的参数表，`Cb` 是否由网格与参数积分
   两条路线互证（0.719 vs 0.722）；
2. **再看极限门**：`tests/test_ice_rao.py` 的 `mu -> 0` 与深水一致性测试是否绿；
3. **最后看物理解读**：冰区 RAO 的差异应当出现在"激励"与"共振频率"上；若差异全部
   集中在附加质量项（ch04 已证其量级 < 1%），则结果可疑，应当回到核函数与记账。

## 5. 注意点与局限（证据分级）

1. **REAL**：RAO 链路本身可复算，`mu -> 0` 与深水一致性有测试固化。
2. **WITHHELD**：真实 PC6 型值表、惯性半径（`k_xx = 0.35B`、`k_yy = k_zz = 0.25L`
   为**工程假设**）、重心高度（`z_g = -0.5T` 假设）。
3. **模型局限**：冰盖是"质量加载"，**不含**冰-船接触力、破碎、摩擦与冰的弯曲刚度。
   也就是说，本章的冰区 RAO 回答的是"有冰盖的水动力"，**不是**"船在冰里航行时的
   总载荷"（那需要 ch06/ch07 与 DEM 线）。
4. **有效性包络**：`k_i H >= 3`（深水化）、传播分支（`omega < omega_c`）。越界会抛异常，
   不允许外推。
5. **不要用本章结论替代冰区实船试验**：本仓没有任何冰区耐波性实测数据，
   `ICE-SHOW-ASSETS` 的 EFD 盘点才是对标入口。

## 6. 练习

1. 用 `ice_rao.no_ice_limit_check` 在 `h_i = 0.05 m`（近敞水）与 `h_i = 0.9 m` 两个
   条件下重跑，问：垂荡 RAO 峰值的相对变化是多少？与 ch04 的附加质量结论是否一致？
2. 若把 `h_i` 提到 2.5 m，`T = 4 s` 的波浪是否仍在传播分支内？（用 ch01/ch03 的工具
   判定，再决定能否跑 RAO。）
3. 设计一个只依赖 `mu -> 0` 极限的回归测试，能抓住"忘记重标定阻尼"的 bug。
   （提示：在 `mu != 0` 时构造一个人为的 `omega_solver != omega`。）

## 7. 参考文献

1. Weitz, M. & Keller, J. B. (1950). *Comm. Pure Appl. Math.* **3**, 305–318.
2. Squire, V. A. (2007). Of ocean waves and sea-ice revisited.
   *Cold Regions Science and Technology* **49**(2), 110–133.
3. Newman, J. N. (1977). *Marine Hydrodynamics*. MIT Press.
4. IACS Unified Requirements I1 / I2 / I3（Polar Class 描述、结构、机械）。

## 8. 复现命令

```bash
# 证据包重渲染（重计算，属重活；须递交计算队列）
python -m tensorbem.benchmarks.ice_ship_verification --stage hull \
    --out artifacts/ice_ship_rao.json
python tools/ice_ship_report.py --in artifacts/ice_ship_rao.json
# 链路一致性（轻量测试）
python -m pytest tests/test_ice_rao.py -q
```