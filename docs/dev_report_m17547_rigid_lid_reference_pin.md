# m17547 — CASE-BEM-RIGID-LID-HEMISPHERE：参考值钉扎到 1e-11 + 独立解法复核

**mission m17547 · TensorBEM · 算例池锚点 CASE-BEM-RIGID-LID-HEMISPHERE** · 判据 <5 % ·
**结论：锚点算例 PASS 不变（headline +0.2831 %）；新增两条验证：参考值钉扎 ≈1e-11、独立解法 MFS 复核 3.6e-5（0.0036 %）**

> 锚点状态核对：`feature_ledger` 中 `CASE-BEM-RIGID-LID-HEMISPHERE` 为 `open`，但仓内
> 已有 m17243 的交付（`benchmarks/verified/bem_rigid_lid_hemisphere_ka0/`，headline
> +0.283 %，6 条判据全 PASS）——总账状态滞后于交付。按纪律**不重做算例**，改做该交付上
> 尚未有人付的**验证欠账**（下②③），并给出总账建议（见⑥）。

## ① 案例介绍（物理背景 / 几何 / 参数）

浮半球（a = 1 m）升沉辐射问题的 **ka→0 端点**：自由面条件 ∂φ/∂z = Kφ 在 ω→0 时退化为
**刚性盖** ∂φ/∂z = 0。镜像论证的陷阱（m16802 旧锚 1.0 的假 FAIL、m16850 裁决）：升沉速度
垂直于对称面，反射后镜像体速度反号，双体问题是**偶宇称** `∂φ/∂r = -U|cosθ|`（两半滑动），
**不是**整球平动；故端点值不是 ρ∇ 而是 ≈0.831 ρ∇。

| 参数 | 值 |
|---|---|
| 几何 | 半球 a = 1.0 m，平面盖在 z = 0（刚性盖） |
| 排水体积 ∇ | (2/3)πa³ = 2.094395 m³ |
| ρ / g | 1025.0 kg/m³ / 9.80665 m/s² |
| 无量纲化 | A33/(ρ∇)，单位速度幅值，exp(−iωt) |
| 参考端点 | A33(0)/(ρ∇) = **0.830951320470**（本单钉扎，尾项 1.9e-11） |
| 压力释放端点（对照，精确） | A33(∞)/(ρ∇) = 0.5 |

## ② 怎么做的（方法 / 流程 / 工具链）

本单两件事，都是**独立于既有实现**的验证：

**(A) 参考值钉扎（闭式系数 + Richardson，rational 算术）**
既有参考值 `even_mode_added_mass_dimless(degree_max=60, quad_order=400)` 用 Gauss-Legendre
对 **|μ|（在 μ=0 处有折点）** 求积得 Legendre 系数。折点使 Gauss 求积**不再精确**，故该值
带一个**固定的 +3.4 ppm 偏置**；原来的"机器精度/尾项 4.6e-09"只测了内一致性，看不见偏置。

本单改用**闭式系数**（无求积）：

    |μ| = Σ_{n even} a_n P_n(μ),  a_n = (2n+1)∫₀¹ μP_n dμ
    ∫₀¹ P_m dμ = [P_{m−1}(0) − P_{m+1}(0)]/(2m+1),  P_{2j}(0) = (−1)^j C(2j,j)/4^j
    A33/(ρ∇) = −3 Σ c_n I_n = +3 Σ_{n even} a_n²/((n+1)(2n+1))

用 `fractions.Fraction` **精确有理数**求部分和（与 float64 路径对照，舍入差 2.2e-16）；
尾项实测指数 p = **3.965**（≈4，与 a_n ~ n^{−3/2} 一致），Richardson 消去一次 →
`0.830951320470`，尾项修正 1.85e-11。

**(B) 独立解法：轴对称 MFS 环源（方法族完全不同的数值路线）**
仓内既有三方对照（偶模 Legendre 级数、Hulme 1982 多极级数、压力释放端点）**同属 Legendre
型展开**，唯一数值路线就是被测的 3-D Kelvin 面元法本身。本单新增 method of fundamental
solutions：

    φ(P) = Σ_k q_k [G_ring(P; r_k, z_k) + G_ring(P; r_k, −z_k)]
    G_ring = K(κ)/(π√(a+b)), a = r²+r_s²+(z−z_s)², b = 2rr_s, κ² = 2b/(a+b)

镜像环使刚性盖 ∂φ/∂z=0 **精确**成立；源在体内 → **全程不出现奇异积分**；环核为精确
椭圆积分（`scipy.special.ellipk`）；未知强度由 Neumann 条件 ∂φ/∂n = V n_z 的最小二乘拟合
（ncol = 2·nsrc）确定；动能式 A33 = −ρ∫ₛ φ n_z dS 用**与配置网格无关的**独立均匀求积
（4001 点）计算，避免"拟合网格决定积分误差"。

路线自检：同一套环核/法向导数模板/符号约定用在**无限流体中平动整球**上，精确答案 0.5，
MFS 给 0.49999992（−1.5e-7，Neumann 残差 5.4e-10）。

工具链：纯 numpy/scipy（无新增依赖，未用 mpmath/sympy）；单次全量复现 **1.3 s**（轻量，未上
集群，符合 load15≈123 时的容量纪律）。

## ③ 计算结果（数字表，全部由 results.json 单一 payload 渲染，零手抄）

**a) 钉扎参考值与流通中数值的偏差**

| 来源 | 值 | 偏差 |
|---|---|---|
| **本单钉扎值** | **0.830951320470** | — |
| Gauss 求积级数（m16802/m17243 已提交工件） | 0.8309541476 | **+3.402 ppm** |
| m16850 ERRATUM 散文中引用的值 | 0.8309570000 | +6.835 ppm |
| `rigid_lid_hemisphere_attribution` 旧 docstring | 0.8260750000 | −5868 ppm |
| report #15809 旧锚（ρ∇，已被 m16850 否证） | 1.0 | +203440 ppm |

**b) 独立路线（MFS 环源）阶梯**

| 环源数 | 配置点数 | 收缩比 c | A33/(ρ∇) | 相对钉扎值误差 | 条件数 | Neumann 残差 |
|---|---|---|---|---|---|---|
| 16 | 32 | 0.60 | 0.8306314 | −0.0385 % | 9.6e+06 | 5.4e-03 |
| 32 | 64 | 0.60 | 0.8310191 | +0.0082 % | 1.3e+11 | 2.9e-03 |
| 64 | 128 | 0.60 | 0.8316228 | +0.0808 % | 3.2e+11 | 2.1e-03 |
| 96 | 192 | 0.60 | 0.8315573 | +0.0729 % | 3.8e+11 | 2.3e-03 |
| 16 | 32 | 0.80 | 0.8305872 | −0.0438 % | 3.3e+03 | 5.9e-03 |
| 32 | 64 | 0.80 | 0.8308725 | −0.0095 % | 4.2e+06 | 2.2e-03 |
| 64 | 128 | 0.80 | 0.8309214 | −0.0036 % | 2.4e+11 | 1.1e-03 |
| 96 | 192 | 0.80 | 0.8307629 | −0.0227 % | 4.1e+11 | 9.2e-04 |
| **128** | **256** | **0.80** | **0.8309190** | **−0.0039 %** | 4.9e+11 | 9.8e-04 |
| 最佳（64/128, c=0.8） | | | 0.8309214 | **−0.0036 %** | 2.4e+11 | 1.1e-03 |

9 组设置的取值区间 [0.83059, 0.83162]（±0.05 % 内），最佳 3.6e-5。

**c) 对已交付算例（m17243 工件）的影响重算**

| 面板 | A33/(ρ∇) | 对新钉扎值误差 | 对旧 Gauss 参考误差 |
|---|---|---|---|
| 64 | 0.807388 | −2.8357 % | −2.8361 % |
| 256 | 0.828152 | −0.3369 % | −0.3373 % |
| 1024 | 0.833304 | **+0.2831 %** | +0.2828 % |

**d) 判据**

| id | 实测 | 判据 | 结论 |
|---|---|---|---|
| P1 钉扎值无尾项 | 1.85e-11 | Richardson 尾项修正 < 1e-9 | PASS |
| P2 截断稳定性 | 1.22e-10 | \|S(240)−S(400)\| < 1e-8 | PASS |
| P3 有理 vs 浮点 | 2.22e-16 | 一致 < 1e-13 | PASS |
| P4 Gauss 偏置被量化 | 3.40e-6 | 1e-7 < 偏差 < 1e-5（ppm 级偏置，非舍入） | PASS |
| P5 路线自检（平动整球） | −1.5e-7 | 与精确 0.5 差 < 1e-4 | PASS |
| P6 独立路线一致性 | −3.60e-5 | \|MFS−钉扎\|/钉扎 < 1e-3 | PASS |
| P7 已交付算例判据不变 | 2.836 % | 全部 m17243 行 < 5 % | PASS |
| P8 与被否证旧锚可分 | 0.831 | 钉扎值 < 0.9 | PASS |

## ④ 结果图

`benchmarks/verified/bem_rigid_lid_hemisphere_axisym/figure.png`（三面板，全部由
results.json 渲染）：

- **A**：MFS 环源阶梯（c=0.6/0.8 两条）对钉扎值 + ±0.1 % 带，并叠加 3-D Kelvin 面元
  256 面板值 0.828152（−0.34 %）——两条完全不同方法族的收敛线夹住同一参考。
- **B**：精确有理部分和 − 钉扎值 vs 截断阶数（log-log），叠 O(n^{−4}) 参考斜率与
  **Gauss 求积值（+3.4 ppm 水平虚线）**：偏置线明显高于 N=400 时的截断误差（1.9e-11），
  直观说明"尾项小 ≠ 值对"。
- **C**：流通中各数值对钉扎值的偏差（ppm，log 横轴）条形图。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **"尾项收敛"不等于"值正确"**：旧参考值的部分和尾项 4.6e-9 看似机器精度，但系数本身
   由折点函数的 Gauss 求积得到，带 +3.4 ppm 系统偏置；对应测试
   `test_even_mode_series_is_machine_converged` 只比较不同截断阶之间的差，**结构上无法
   发现该偏置**（属"测试跑了但没断言对不变量"的典型）。本单新增的
   `test_pin_is_not_the_gauss_quadrature_value` 显式断言该 3.4 ppm 间隙必须被复现。
2. **MFS 的精度上限是条件数**，不是离散化：c=0.8 / 128 源时条件数 4.9e+11，Neumann 残差
   ~1e-3（近接触线配置点主导），故不同设置间有 ±0.05 % 的非单调抖动；如实按"最佳值 +
   区间"报告，不宣称收敛阶。
3. **接触线（s→π/2）的角点奇性**是所有方法共同的上限（m17243 的 256→1024 表观阶 0.13 即
   源于此）；本单不改这一点，只是在同一物理问题上加一条独立证据。
4. **未改动已提交工件**：`benchmarks/verified/bem_rigid_lid_hemisphere{,_ka0}/` 是 m16802/
   m17243 的记录（含一个 honest FAIL 记录），按纪律不覆盖；3.4 ppm 的偏差写在
   `docs/rigid_lid_zero_frequency_anchor/ERRATUM.md` 的 m17547 addendum 与本报告里。若要把
   `even_mode_added_mass_dimless` 默认切到精确系数，需同步重生成两个工件目录并更新
   `test_rigid_lid_hemisphere_attribution` 里 0.8309541 的 pin（≤1e-6）与
   `test_m17243_rigid_lid_hemisphere` 的 1e-9 断言——留作后续单（3.4 ppm 对 5 % 门无影响，
   不值得在本单制造工件 churn）。
5. **既有测试红灯（与本单无关，先于本单存在）**：
   `tests/test_rigid_lid_hemisphere_attribution.py::test_all_criteria_pass_and_summary_is_rendered`
   断言 `benchmarks/verified/bem_rigid_lid_hemisphere/figure.png` 存在，但该 PNG 不在仓内
   （m16802 工件目录只有 results.json/summary.md）。本单未触碰该目录，也未伪补图。

## ⑥ 误差判定（显式回答）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| 3-D BEM 1024 面板，ka=1e-3（m17243 headline） | 钉扎参考 0.830951320470 | **+0.2831 %** | **是** |
| 3-D BEM 256 面板，ka=1e-3 | 同上 | −0.3369 % | 是 |
| 独立 MFS（最佳设置） | 同上 | **−0.0036 %** | 是 |
| 独立 MFS 路线自检（平动整球） | 精确 0.5 | −1.5e-7 | 是 |
| 钉扎参考值本身 | 精确有理和 + Richardson | 尾项 1.9e-11（相对 2.3e-11） | 是 |
| 旧参考值（Gauss 求积） | 钉扎值 | +3.4 ppm | 是（但属**系统性偏置**，非随机误差；已量化入账） |

**结论：锚点判据（<5 %）与 3 % 线均满足，且新增的独立方法族复核把结论从"单一数值路线
自证"升级为"两种方法族（面元 / 环源 MFS）+ 闭式系数三条互不共享误差机理的路线一致"。**

## 工件与复现

```
src/tensorbem/benchmarks/rigid_lid_reference_pin.py            # 新模块（钉扎 + MFS）
tests/test_rigid_lid_reference_pin.py                          # 12 条不变量测试
benchmarks/verified/bem_rigid_lid_hemisphere_axisym/
    run_study.py  make_figure.py  results.json  summary.md  figure.png
docs/dev_report_m17547_rigid_lid_reference_pin.md              # 本报告
docs/rigid_lid_zero_frequency_anchor/ERRATUM.md                # 追加 m17547 addendum
```

复现（本机 1.3 s）：
`python3 benchmarks/verified/bem_rigid_lid_hemisphere_axisym/run_study.py && python3 benchmarks/verified/bem_rigid_lid_hemisphere_axisym/make_figure.py`

**总账建议（供派发器）**：`CASE-BEM-RIGID-LID-HEMISPHERE` 实际已由 m17243 交付（6 判据
PASS，headline +0.283 %），本单又补了参考钉扎与独立路线复核 → 建议把该 slug 置为
`done/verified`，并新增一条后续条目：

- slug：`BEM-RIGID-LID-REF-EXACT-COEFF`
- 标题：算例池·刚性盖参考值精确系数化（替换折点 Gauss 求积）
- 量化验收：`even_mode_added_mass_dimless` 默认改用闭式系数；与钉扎值
  0.830951320470 相对偏差 <1e-9；`bem_rigid_lid_hemisphere{,_ka0}` 工件重生成后
  m17243 判据仍全 PASS；相关 pin 由 0.8309541 更新并留 delta 记录
- 所属域：TensorBEM（水动力/频域辐射）