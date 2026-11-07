# 波–冰–船辐射绕射：mass-loading 冰盖下的六自由度 RAO 与冰区波浪衰减

**TensorBEM · 战役 IN12-P3（纵贯军 P3）· 波–冰–船辐射绕射腿**

本报告实现并验证了 TensorBEM 的第一条冰区水动力链路：mass-loading 冰盖下的
**频域辐射（六自由度附加质量/阻尼）**、**绕射（波激励力）**、**RAO 谱**与
**冰区波场衰减特性**，并给出了对应的验证证据（REAL / WITHHELD 分级）。

所有数字由 `python -m tensorbem.benchmarks.ice_ship_verification` 产出 JSON 证据、
再经 `tools/ice_ship_report.py` 程序化渲染到
[`docs/ice_water_ship_numbers.md`](ice_water_ship_numbers.md)，正文不手工抄录任何数值。

---

## 1. 案例介绍

### 1.1 物理背景

冰区船舶的波浪载荷与运动与敞水不同，原因有两个层次：

1. **波场改变**。冰盖改变自由面边界条件，同一频率的波在冰下变短、变慢；
   高频（周期短于约 2 s）的波在冰盖下根本不能传播（e^{-kappa x} 衰减）。
   这直接改变了船舶受到的激励频率–波数关系。
2. **附加惯性/阻尼**。浮冰板的质量给自由面增加了惯性（mass loading），
   使辐射问题的自由面条件从 Neumann–Robin（`phi_z = (omega^2/g) phi`）
   变为带冰质量的 Robin 条件；如果进一步计入板的抗弯刚度，则出现
   flexural-gravity 波（本报告**不含**刚度项，见 §5 局限）。

本报告的物理模型是经典的 **mass-loading 薄板近似**（Weitz & Keller 1950；
Squire et al. 1995）：冰盖无抗弯刚度、无耗散，只贡献单位面积质量
`m_s = rho_i h_i`。

### 1.2 几何与参数

| 项目 | 取值 | 说明 |
|---|---|---|
| 船体 | 参数化 PC6 类冰区船，L = 120 m，B = 24 m，T = 9 m | 无真实 PC6 型值表（私有），见 §5 局限 |
| 船型参数 | 水线指数 n = 4，横剖指数 q = 2，平板龙骨比 c_k = 0.70，端部比 0.02 | Cb = 0.719（网格 1056 面元）/ 0.722（参数积分） |
| 排水量 | `m = rho_w V`，设计吃水下 V ≈ 1.86e4 m^3 → m ≈ 1.91e4 t | 程序计算，非输入 |
| 惯性半径（假设） | k_xx = 0.35 B、k_yy = k_zz = 0.25 L；重心 z = −0.5 T | **工程假设**，非实测 |
| 冰盖 | h_i = 0.9 m，rho_i = 917 kg/m^3，H = 200 m | μ = rho_i h_i/rho_w = 0.805 m |
| 水 | rho_w = 1025 kg/m^3，g = 9.80665 m/s^2 | |
| 浪向 | 顶浪（wave angle = 180°） | 波沿 −x 传播 |
| 频率带 | T = 4–14 s（omega = 0.449–1.571 rad/s） | 均在 mass-loading 截止频率以下 |

mass-loading 截止：`omega_cut = sqrt(g/mu) = 3.49 rad/s`（T = 1.80 s），
本报告全部工况都在传播分支内；截止以上的行为在 §4.3 单独验证。

### 1.3 与任务判据的对应

| 任务判据 | 本报告对应证据 | 状态 |
|---|---|---|
| 无冰工况 RAO/附加质量在经典频域解参考带内 | §4.2 潜球 heave 附加质量 vs Lamb 解析解；§4.4 长周期 heave RAO 渐近；§4.4 网格收敛 | 见 §6 逐项误差 |
| 冰盖衰减波数 k(h) 对 mass-loading 解析解误差 < 5 % | §4.1 k(h_i) 表 vs 独立 Robin 根、RK4 打靶、深浅水闭式极限 | **通过**（最大误差 1.1e-14） |
| pytest ≥ 6 含负例（负波高/零吃水拒收） | `tests/test_ice_mass_loading.py`（18 项）、`tests/test_ice_rao.py`（10 项）、`tests/test_pc6_hull.py`（5 项） | **通过**（33 项） |
| 冰区波浪增阻 R_AW | §5.3 明确 **withheld**（无二阶漂移力算子、无参考解，不编造数） | honest WITHHELD |
| 几何先验证 Cb/L/B | §4.4 几何闸门（网格 vs 参数积分，1 % 带）+ `tests/test_pc6_hull.py` | **通过** |

---

## 2. 方法

### 2.1 冰盖自由面条件与色散关系

设速度势 `phi ~ W(z) exp(i(kx − omega t))`，水深 H。冰–水界面的线性化条件由
薄板竖向动量方程 `m_s eta_tt = −rho_w g eta − rho_w phi_t` 给出。对简谐运动消去
eta 得 **Robin 条件**

```
dW/dz = K W,        K = omega^2 / (g − mu omega^2),        mu = rho_i h_i / rho_w      (1)
```

配 `W = cosh(k(z+H))`、底部 `dW/dz = 0`，得 mass-loading 色散关系

```
omega^2 = g k tanh(k H) / (1 + mu k tanh(k H))                                         (2)
```

两个直接推论（都由本模块实现并验证）：

* **波数增移**。同一频率下 `k_ice > k_open`：`omega^2 = g k/(1+mu k)` 在深水给出
  闭式 `k_i = k_0/(1 − mu k_0)`。T = 4 s 时 k 增大 25 %，T = 14 s 时增大 1.7 %。
* **截止**。`K -> inf` 当 `omega^2 -> g/mu`；超过 `omega_cut = sqrt(g/mu)` 无实根，
  自由波变为消逝波 `k = i kappa`，满足

```
kappa tan(kappa H) = omega^2 / (mu omega^2 − g)                                          (3)
```

**一个物理上重要的结论（与常见叙述相反）**：mass loading 不耗散能量。跨冰缘保持
能量通量 `A^2 cg` 不变，则 `A_ice/A_open = sqrt(cg_open/cg_ice) > 1`（本报告 T = 4 s 时为
1.56）。也就是说 mass-loading 冰盖让波场**变短变陡**，而不是变弱；真正的冰区波浪
衰减（观测中的 `alpha ~ 1e-5–1e-3 1/m`）必须来自浮冰散射/耗散，本仓库**未实现**，
故本报告不声称能复现观测衰减率（§5.2）。

### 2.2 冰盖下的三维辐射/绕射（核心实现）

TensorBEM 的三维自由面 Green 函数由单一波数 `nu` 参数化，在 `z = 0` 上强加
`phi_z = nu phi`（深水开放水域 `nu = omega^2/g`）。由 (1)，mass-loading 冰盖下的
条件**同型**，系数为 `K`；而深水中 (2) 的根恰为 `k_i = omega^2/(g − mu omega^2) = K`。
因此 **冰盖 Green 函数 = 现有核在 `nu = k_i` 处取值**，无需新核。实现见
[`src/tensorbem/ice_rao.py`](../src/tensorbem/ice_rao.py)，包含两处必须显式处理的记账：

* **阻尼重标定**。求解器由 `nu` 反推 `omega_int = sqrt(g nu)` 并输出
  `B = omega_int Im(Z)`；物理阻尼为 `B = omega_true Im(Z)`，故返回前乘
  `omega_true/sqrt(g k_i)`。
* **激励力对记账不敏感**。`F = −i omega rho int phi n dS`，而入射势含因子 `1/omega`
  （`phi_inc ∝ g a / omega`），`omega · (1/omega)` 相消，故 `F` 只依赖 `(k, a)`。
  该不变性由 `tests/test_ice_rao.py::test_excitation_force_is_bookkeeping_invariant`
  以「核的重力标定 ×1.5 ⇒ 力精确 ×1.5」验证。

运动方程仍为标准频域形式

```
[−omega^2 (M + A_ice) + i omega B_ice + C] xi = F_exc,ice                              (4)
```

其中 `C` 在冰区与敞水**相同**：mass loading 只增加惯性，不改变水线面静水回复力。

**有效范围是强制的而非假设的**：每次调用都检查 `k_i H >= 3`（`tanh ≥ 0.995`），否则
抛错并说明「有限水深冰盖 Green 函数未实现」。这一闸门在开发中真实拦截过一次
误用工况（§5.4）。

### 2.3 船体几何（参数化 PC6 类）

PC6 真实型值表不可得，故用可审计的参数化船型：

```
b(x,z) = (B/2) · wl(u) · f(v),   u = 2x/L,  v = z/T
wl(u)  = max(1 − |u|^n, f_end),  f(v) = max(1 − (1 − c_k)|v|^q, c_k)
```

得到 Cb ≈ 0.72 的丰满冰区船型（宽平底、丰满水线、小方尾/艏端面）。
**几何先于水动力验证**：网格体积与同一参数定义的独立二重积分比较，`Cb`、`L`、`B`、
`T`、水线面积全部入闸（1 % 带），不通过则抛错而不是继续算水动力
（`validate_hull_geometry`，`tests/test_pc6_hull.py`）。网格是**无水平水线盖板**的
开口湿表面（边界即水线闭环），与仓库既有 `WettedHullMesh` 约定一致。

### 2.4 验证链（为什么这些数字可信）

| 层 | 验证对象 | 参照 | 独立性 |
|---|---|---|---|
| 1 | 冰盖色散 k(omega) | 深水 Weitz–Keller 闭式、浅水闭式、Robin 残差 | 代数等价，属**一致性**检查 |
| 2 | 冰盖色散 k(omega) | 竖直 ODE `W''=k^2W` 的 RK4 打靶（不用 cosh 本征函数） | **数值独立** |
| 3 | k(h_i) | 独立代数路径的 Robin 根 `k tanh(kH) = K` | 代数独立 |
| 4 | 敞水附加质量 | Lamb 无限流体球 `(2/3) pi rho R^3` | **经典闭式** |
| 5 | 冰链路记账正确性 | `mu -> 0` 时必须复现敞水解 | 结构验证（相对差 3e-14） |
| 6 | 激励力记账 | 核重力标定 ×1.5 ⇒ 力 ×1.5 | 解析不变性 |
| 7 | 几何 | 网格 vs 参数积分（Cb/L/B/Awp） | 独立积分 |
| 8 | RAO | 单自由度阻抗平衡回代 | 自洽 |

---

## 3. 怎么做的（流程与工具链）

1. `src/tensorbem/ice.py`：mass-loading 色散（式 2）、Robin 系数、截止与消逝分支
   （式 3）、群速度解析式、能量通量振幅比、RK4 打靶独立核验、厚度研究表；
   所有非物理输入（负冰厚、零/负水深、冰厚 > 水深、负频率、负波高、负距离）一律
   `ValueError`。
2. `src/tensorbem/pc6_hull.py`：参数化冰区船网格（环形放样 + 端面扇形）+ 几何闸门
   + 质量/惯性估计。
3. `src/tensorbem/ice_rao.py`：冰盖辐射/绕射/RAO（§2.2 的映射与重标定）、
   `mu -> 0` 极限检查、withheld 状态声明。
4. `src/tensorbem/benchmarks/ice_ship_verification.py`：三段式证据驱动
   （`dispersion` / `classical` / `hull`），输出 JSON 与 PNG。
5. `tools/ice_ship_report.py`：把 JSON 渲染成
   [`docs/ice_water_ship_numbers.md`](ice_water_ship_numbers.md)。
6. 测试：`pytest tests/test_ice_mass_loading.py tests/test_ice_rao.py tests/test_pc6_hull.py`。

运行方式（全部为仓库内命令）：

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.ice_ship_verification --stage dispersion \
    --out artifacts/ice_dispersion.json
PYTHONPATH=src python -m tensorbem.benchmarks.ice_ship_verification --stage classical \
    --out artifacts/ice_classical_reference.json
PYTHONPATH=src python -m tensorbem.benchmarks.ice_ship_verification --stage hull \
    --out artifacts/ice_ship_rao.json
PYTHONPATH=src python tools/ice_ship_report.py
```

---

## 4. 计算结果

所有表格见 [`docs/ice_water_ship_numbers.md`](ice_water_ship_numbers.md)（程序化渲染）。
以下为可直接判读的结论与关键数字。

### 4.1 冰盖色散与 k(h_i)（判据：< 5 %）

频率扫描（h_i = 0.9 m，H = 200 m）显示波数增移随周期单调减弱：
T = 4 s 时 `k_ice/k_open − 1 = +25.4 %`，T = 8 s 时 `+5.3 %`，T = 14 s 时 `+1.7 %`。
这与物理一致：mass loading 相对效应 ~ `mu k`，短波更强。

冰厚研究（T = 6 s，h_i = 0.3–2.5 m）中 `k(h_i)` 与**独立代数路径**的 Robin 根
最大相对差 **1.1e-14**（判据 < 5 %，余量极大）；且 `k` 随 `h_i` 严格单调增。

### 4.2 敞水附加质量 vs 经典频域解（潜球 / Lamb）

深潜球（r = 5 m，深度 15 m，即 3R）heave 附加质量对 Lamb 无限流体解
`(2/3) pi rho R^3 = 2.68e5 kg`：

| 网格 | 面元 | A33 误差（omega = 0.3/0.5/0.8） |
|---|---|---|
| coarse | 168 | 6.18 % / 5.30 % / 5.81 % |
| fine | 352 | **2.60 % / 1.65 % / 2.21 %** |

`A33/(rho V) = 0.489`（fine），与「低频自由面等效刚壁」的经典极限 0.5 相符。
该误差随面元加密下降（168 → 352 面元误差减半），指示为离散误差而非模型误差。

### 4.3 截止以上：消逝分支

`omega/omega_cut = 1.02 / 1.20 / 2.00` 三档均无实根，返回复波数 `k = i kappa`：
`kappa ≈ 7.85e-3 1/m`（衰减长度约 127 m，100 m 后振幅 0.456），且
`kappa tan(kappa H)` 与式 (3) 的残差 ≤ 7.5e-8。物理解读：`kappa -> pi/(2H)` 当
`omega -> inf`，衰减尺度由水深控制——这是 mass-loading 模型**唯一**的衰减机制。



### 4.4 冰区六自由度 RAO（理想化 PC6 类船体）——结果判读

几何闸门（`numbers` §C）：590 面元网格 `Cb = 0.71631` vs 参数积分 `0.72178`（差 0.76 % < 1 %），
`L/B/T` 与参数值机器精度一致，水线面积差 0.30 %；314 面元为 1.53 %，即离散误差随加密
收敛（`Cb` 误差 1.53 % → 0.76 %）。因此船体几何**先通过闸门**才进入水动力。

冰盖对一阶量的影响（顶浪，h_i = 0.9 m）：

* **波场**：`k_ice > k_open` 全场成立（T = 4 s 时 +25.4 %，T = 14 s 时 +1.7 %），
  全部工况 `k_i H = 4.2–63` 落在深水包线内（闸门 `k_i H >= 3`）。
* **激励力**：heave 激励力在冰区系统性降低（T = 6 s：6.28e5 → 4.89e5 N/m，−22 %；
  T = 10 s：5.42e6 → 5.02e6 N/m，−7.4 %），原因是同一频率下冰下波长更短，
  船体尺度的绕射/FK 积分在更短波长上互相抵消更多。
* **辐射**：heave 附加质量几乎不变（T = 6 s：14.66e6 → 14.86e6 kg，+1.4 %；
  T = 10 s：18.63e6 → 18.23e6 kg，−2.1 %）——mass loading 改变的是**波场**与
  自由面条件，而附加质量的近场部分由船体几何主导；heave 阻尼则明显下降
  （T = 4 s：1.33e6 → 7.07e5 kg/s，−47 %；T = 10 s：10.15e6 → 9.95e6 kg/s，−2.0 %），
  因为同一频率下更短的波需要更小的辐射能量。
* **RAO**：heave RAO 在冰区略降（T = 10 s：0.521 → 0.480，−7.8 %；T = 8 s 基本不变
  0.1595 → 0.1580，−0.9 %），因为激励力下降与阻尼下降部分抵消；pitch RAO 同步略降。
  短周期（T = 4 s）RAO ≈ 1e-3，即船体不跟随短波，符合物理。

网格收敛（`numbers` §C）：314 → 590 面元，heave `A33` 变化 0.66 %、heave RAO 变化 0.03 %，
说明 590 面元结果对本题已收敛；绝对精度仍受几何保真度限制（§5.1）。

长周期渐近：T = 14 s 时 heave RAO（敞水）为 0.870，与经典低频极限 1.0 差 12.97 %。
该频率下 `omega^2 L/g = 0.21`，尚未进入真正的长波极限（λ = 301 m ≈ 2.5 L），
故 13 % 的偏差是**未达渐近区**而非求解错误；因此这一步不构成 < 3 % 的精度声明（见 §6）。

冰区附加质量/阻尼矩阵的对称性、正阻尼与 `kH` 包线均由
`tests/test_ice_rao.py` 断言（`max|A − A^T|/max|A| = 1.1e-3`，属离散精度）。


---

## 5. 注意点、局限与适用范围

### 5.1 模型边界（明确不使用的地方）

* **深水限制**：冰盖三维核只在 `k_i H >= 3` 使用；浅水需要
  `k tanh(kH) = K` 的有限水深 Green 函数（未实现，调用即抛错）。
* **无抗弯刚度**：真实冰盖的 flexural-gravity 波在短波端（`lambda < 2 pi l_c`，
  `l_c = (D/(rho_w g))^{1/4}` 为特征长度）受刚度支配，mass loading 会高估
  短波波数并低估截止频率。h_i = 0.9 m 的冰板 `l_c ≈ 12–17 m`，对应
  T ≲ 4 s 的波段已接近该边界，故 T = 3–4 s 的结果应视为 mass-loading 上限。
* **无冰耗散**：见 §2.1，mass loading 不衰减；只有截止以上才指数衰减。
* **无浮冰场离散性**：真实冰区是浮冰（floe）场，散射衰减与冰缘反射未建模。

### 5.2 观测衰减率不可复现（诚实声明）

现场观测的冰区波衰减 `alpha ~ 1e-5–1e-3 1/m`（周期 5–15 s）在本模型下为 **0**
（mass loading 保守），模型的消逝衰减只出现在 T < 1.8 s。因此本报告**不**用该
链路声称可以预测观测衰减率；这是模型性质，不是实现缺陷。

### 5.3 冰区波浪增阻 R_AW：withheld

`R_AW` 需要二阶漂移力算子（压力二阶项/远场 Kochin 型动量通量）或浮冰散射模型，
本仓库均无，且无可用于验证的独立参考解。为避免产生看似合理的伪造数值，
`artifacts/ice_ship_rao.json` 中 `added_resistance.status = "withheld"` 并给出原因；
报告只提供任何 `R_AW` 模型都需要的一阶驱动量（波数增移、能量通量振幅比、
激励力比、RAO 比、消逝衰减长度）。

### 5.4 开发中踩过的坑

* **深水闸门拦截**：潜球验证在 `omega = 0.3 rad/s`、`H = 200 m` 时 `kH = 1.84 < 3`，
  程序直接抛错——这正是设计意图（拒绝越界工况而不是给出偏差结果）。修正做法是
  该验证用 `H = 1e4 m`（真正的深水）。
* **附加质量矩阵对称性只到离散精度**：CHIEF 最小二乘求解使 `A` 非严格对称，
  实测 `max|A − A^T| / max|A| = 1.1e-3`（164 面元网格），故测试按 5e-3 相对带断言，
  而不是要求机器精度对称。
* **表头竖线**：程序化生成的 markdown 表头里若含 `|` 会破坏表格，已在生成器中移除。

### 5.5 适用范围总结

本链路可用于：冰区（mass-loading、深水、h_i ≲ 1.5 m、T ≳ 4 s）船舶一阶波浪
激励、附加质量/阻尼、RAO 与截止以上波场的快速评估。不可用于：浅水冰区、
短波刚度主导区、需要 R_AW/二阶力、需要复现观测衰减率的场合。

---

## 6. 误差判定（逐项，含「是否 < 3 %」的显式回答）

| 关键量 | 参照 | 误差 | 是否 < 3 % | 备注 |
|---|---|---|---|---|
| 冰盖波数 k(omega)，T = 3–14 s | RK4 打靶（数值独立） | ≤ 1.0e-13（相对） | **是** | 仅截断误差级别 |
| 冰盖波数 k(omega) | Robin 残差 `k tanh(kH) − K` | ≤ 4.5e-15 1/m | **是** | |
| k(h_i)，h_i = 0.3–2.5 m | 独立 Robin 代数根 | ≤ 1.1e-14 | **是** | 判据为 < 5 %，余量 5e11 倍 |
| 深水极限 | Weitz–Keller 闭式 | 2.2e-14 | **是** | 一致性检查（同式极限） |
| 浅水极限 | `omega^2 = g k^2 H/(1+mu k^2 H)` | 3.1e-14 | **是** | 一致性检查（kH = 0.43） |
| 消逝分支 | `kappa tan(kappa H)` 方程残差 | ≤ 7.5e-8 | **是** | |
| heave 附加质量 A33（潜球，fine） | Lamb `(2/3) pi rho R^3` | **1.65 % / 2.21 % / 2.60 %**（omega = 0.5/0.8/0.3） | **是**（三点均 < 3 %） | coarse 网格 5.3–6.2 %，随加密下降 |
| 冰链路 `mu -> 0` | 敞水求解（同网格） | 3.0e-14（A）、2.2e-11（B） | **是** | 记账正确性 |
| 激励力记账不变性 | 核重力 ×1.5 ⇒ 力 ×1.5 | ≤ 1e-9 | **是** | |
| 几何 Cb（1056 面元） | 参数二重积分 | **0.50 %** | **是** | 闸门 1 % |
| 几何 Cb/L/B/Awp（590 面元） | 参数积分 | Cb 0.76 %、Awp 0.30 %、L/B/T 机器精度 | **是** | |
| heave RAO 长周期渐近 | 经典低频极限 RAO -> 1 | 见 numbers §C | 视网格 | 无独立船体参考解 |
| **船体 6-DOF RAO（敞水）** | **无经典参考解** | **unknown** | **无法回答** | 理想化船型、无公开型值与试验对照 |
| **船体 RAO（冰区）** | **无参考解** | **unknown** | **无法回答** | 同上，且冰区无公开基准 |
| **冰区波浪增阻 R_AW** | 无 | **withheld** | **无法回答** | §5.3：不编造 |

结论：**能在有基准的项上给出 < 3 % 的显式结论（冰盖色散、k(h)、潜球附加质量、
几何闸门）；船体 RAO 与 R_AW 因缺基准而明确标 unknown / withheld，不用自洽性
冒充精度。**

---

## 7. 复现清单

| 文件 | 内容 |
|---|---|
| `src/tensorbem/ice.py` | mass-loading 色散、截止/消逝分支、能量通量振幅比、RK4 打靶核验 |
| `src/tensorbem/pc6_hull.py` | 参数化 PC6 类船网格 + Cb/L/B 闸门 + 质量惯性 |
| `src/tensorbem/ice_rao.py` | 冰盖辐射/绕射/RAO（Green 核映射、阻尼重标定、withheld 声明） |
| `src/tensorbem/benchmarks/ice_ship_verification.py` | 三段式证据驱动（JSON + PNG） |
| `tools/ice_ship_report.py` | JSON → markdown 表格渲染 |
| `tests/test_ice_mass_loading.py` | 18 项（含负冰厚/零水深/负频率/负厚度数组等负例） |
| `tests/test_ice_rao.py` | 10 项（含负波高、截止、浅水闸门、记账不变性、`mu -> 0` 极限） |
| `tests/test_pc6_hull.py` | 5 项（含负/零吃水、欠解析网格拒收） |
| `artifacts/ice_dispersion.json` | 色散证据 |
| `artifacts/ice_classical_reference.json` | Lamb 球 + `mu -> 0` 证据 |
| `artifacts/ice_ship_rao.json` | 船体几何闸门 + RAO 谱 + withheld 声明 |
| `artifacts/fig_ice_dispersion.png`、`fig_ice_thickness_study.png`、`fig_ice_rao.png`、`fig_ice_added_mass.png` | 图 |

同一份证据同时归档在版本库内的 `docs/ice_water_ship_evidence/`（JSON + PNG 副本），
以免 `artifacts/` 被 `.gitignore` 忽略后证据随分支丢失。