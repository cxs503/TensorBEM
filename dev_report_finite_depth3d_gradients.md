# 3-D 有限水深 Green 函数的解析梯度：补齐 BEM 装配所缺的最后一块核函数（m3288）

**任务**：m3288（platform 席自找活） · **分支**：`agent/m3288` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3288`）
**工件**：`src/tensorbem/finite_depth3d.py`（+261 行）、`src/tensorbem/benchmarks/finite_depth3d_gradients.py`（新，可复跑研究）、
`tests/test_finite_depth3d_gradients.py`（新，23 gate）、`outputs/fd3d_grad/study.json`、
`docs/finite_depth3d_gradients.md`（程序化注入的数字表）、`docs/figures/fig_finite_depth3d_gradient_{accuracy,field,farfield}.png`

EDIT_WHITELIST: src/tensorbem/finite_depth3d.py, src/tensorbem/__init__.py, src/tensorbem/benchmarks/finite_depth3d_gradients.py, tests/test_finite_depth3d_gradients.py, docs/finite_depth3d.md, docs/finite_depth3d_gradients.md, docs/figures/, outputs/fd3d_grad/, dev_report_finite_depth3d_gradients.md

---

## 1. 选题依据：缺口账里唯一"物理前沿"级的欠账

`docs/finite_depth3d.md`（m3015，已合并 PR #118）把 3-D 有限水深 John 本征级数核函数做完并验到机器精度，但结尾两条**明文留口**：

> * **No BEM solver yet.** Panel integration, the single/double-layer jump relations and the CHIEF/lid treatment for surface-piercing hulls in finite depth are the next increment; Garrett's truncated dock is the natural first diffraction benchmark.
> * Only $\partial G/\partial z$ is analytic; a horizontal derivative needs $K_1$/$H_1$ and belongs to the panel layer.

即：**核函数只对 z 方向解析可微**，面板装配（单层法向导数 $\partial G/\partial n$、双层法向导数）需要的水平导数 $\partial G/\partial x,\partial G/\partial y$ 缺失。这不是文档缺陷，而是硬能力缺口——没有它，三维有限水深 BEM 一行都装不出来。README roadmap item 6 的 "needs new physics: finite-depth 3-D John eigenseries Green function" 也只完成了核这一半。

近 24h 的 TensorBEM 单据（m3059 heavy 层复算、m3085 三维文档数字钉桩、m2136/m2173 Ursell 复算、m2587 Kutta 闭合）全部集中在**文档数字复核**；本单换轨到**功能补全 + 验证面**，与其无重叠，且是本席首次做核函数开发。

## 2. 做了什么

### 2.1 缺失数学：$K_1$、$H_1^{(1)}$

有限水深 Green 函数

$$G(x,\xi)=\frac{1}{2\pi}\sum_{n\ge1}\frac{\phi_n(z)\phi_n(\zeta)}{N_n}K_0(\mu_nR)+\frac{i}{4}\frac{\phi_0(z)\phi_0(\zeta)}{N_0}H_0^{(1)}(k_0R)$$

对源—场水平间距 $R$ 求导立刻需要 $K_0'=-\mu K_1$、$H_0^{(1)'}=-k_0H_1^{(1)}$，两者模块里都没有：

| 新增 | 实现 |
|---|---|
| `_bessel_k1_small` / `_bessel_k1_large` / `bessel_k1` | A&S 9.6.11 升幂级数（$x<9$）/ Hankel 渐近级数（$x\ge9$，$\mu=4$），纯 NumPy |
| `hankel_h1` | 复用仓库已有且已验的 `analytic.bessel_j01_y01`（$J_1+ iY_1$，Wronskian 1e-15 级） |

### 2.2 解析梯度

$$\frac{\partial G}{\partial R}=-\frac{1}{2\pi}\sum_n\frac{\phi_n(z)\phi_n(\zeta)}{N_n}\mu_nK_1(\mu_nR)-\frac{i}{4}\frac{\phi_0(z)\phi_0(\zeta)}{N_0}k_0H_1^{(1)}(k_0R)$$

$$\frac{\partial G}{\partial x}=\frac{\partial G}{\partial R}\frac{x-\xi}{R},\qquad
\frac{\partial G}{\partial z}=-\frac{1}{2\pi}\sum_n\frac{\mu_n\sin\mu_n(z+h)\,\phi_n(\zeta)}{N_n}K_0+\frac{i}{4}k_0\sinh k_0(z+h)\frac{\phi_0(\zeta)}{N_0}H_0^{(1)}$$

新 API：`green_gradient_3d`（$\nabla_xG$）、`green_gradient_source_3d`（$\nabla_\xi G$）、`green_gradient_field_3d`（别名，对齐深水模块命名）。**源点梯度不是符号取反**：有限水深核的垂直本征函数分别依赖 $z$ 与 $\zeta$，故 $\nabla_\xi G(x,\xi)=\nabla_xG(\xi,x)$（参数转置），这一点与深水核（依赖 $z+\zeta$）不同，已写进 docstring 并有 gate 守住。

## 3. 验证（11 项门，全部独立于被测代码路径）

| 门 | 参考物 | 实测 |
|---|---|---:|
| $K_1$ vs SciPy（$x\in[10^{-8},10^{2.5}]$） | `scipy.special.k1` | 8.9e-9 |
| $H_1^{(1)}$ vs SciPy | `scipy.special.hankel1` | 2.0e-14 |
| $\nabla_xG$ vs $G$ 的中心差分（$h$=2/5/20 m，18 组合） | 核函数本身 | 1.3e-6（差商截断主导） |
| $\nabla_\xi G$ vs 对源坐标差分 | 核函数本身 | 1.3e-6 |
| 自由面条件 $\partial_zG-\nu G$（$z=0$ 环） | 精确边条件 | 4.0e-15 |
| 底部不可穿透 $\partial_zG$（$z=-h$ 环） | 精确边条件 | 0.0（逐位） |
| 调和性 $\nabla^2G$（离轴内点） | 精确解 | ≤3.0e-7 |
| 深水极限 vs 已验 Kelvin 梯度（$k_0h$=12→240） | 深水核 | 8.1e-5 → 7.6e-9（单调指数收敛） |
| 远场辐射律 $\partial_RG=i k_0G$（$k_0R$=36） | 出射波渐近解 | 1.47e-2 |
| 发散波幅值 $\langle\vert G\vert\rangle\sqrt{k_0R}$（$R$=10…160 m） | $R^{-1/2}$ 律 | 波动 <1% |
| 面板级 $\int_S\partial_nG\,dS$（400 面元球，2 目标） | 同一积分的中心差分 | 1.2e-5 / 3.5e-5 |
| $z$ 分量与旧 `green_gradient_z_3d` | 既有实现 | 0.0（逐位，故新函数是旧函数的严格超集） |

测试：`tests/test_finite_depth3d_gradients.py` 23 gate 全绿；与既有 `tests/test_finite_depth3d.py` 合跑 **49 passed**（52 s）。

## 4. 踩坑（都留了守门测试）

1. **分支切换点不能照抄 $K_0$ 的经验值以外的直觉**：$K_1$ 渐近级数在 $x=6$ 的相对误差是 **1.3e-4**（$\mu=4$ 的渐近系数衰减比 $\mu=0$ 慢得多），必须在 $x=9$ 切换（此时渐近 4.7e-9、级数 1.8e-8），全区间最坏 8.9e-9。
2. **源轴线上 $\nabla G$ 真的发散**：$x=\xi,y=\eta$ 时 $\partial G/\partial R\sim1/R$，我最初把 (0,0,-1.9) 这种"在源正上方"的点放进调和性检验，有限差分给出 1e6 的假残差——不是代码错，是判据错。现已离轴取点，并在代码里把水平分量对轴线显式置零（否则 $K_1$ 的 $1/R$ 极点会进到 $\inf\times0$）。
3. **残差归一化 ≠ 差值归一化**：研究脚本里一度用 `max|a-b|/max|b|` 当"残差相对量"，导致自由面残差报 1.0（实为 1e-15）。现 `_scaled` 与 `_relative` 分开，报告纪律要求的中文表数字因此才可信。
4. **画场量云图的网格要避开源点**：$x=0,z=\zeta$ 恰好落在网格结点上会触发"coincident target and source"，已取半步偏移。

## 5. 结论与误差判定

- 关键量 vs 基准：$K_1$ 8.9e-9、$H_1^{(1)}$ 2.0e-14、梯度 1.3e-6、边条件 ≤4e-15、深水极限 7.6e-9 —— **全部远优于 3%**（本单的基准是精确解析关系与既有已验核函数，不是实验/文献，故"是否<3%"以 1e-5 量级回答）。
- 远场辐射律 1.47e-2（1.5%，<3%），面板级法向导数积分 3.5e-5（<3%）。
- **honest 边界**：本单只交付核函数与验证面，**没有**三维有限水深 BEM 求解器，因此**没有任何**与文献/实验的 RAO、附加质量、波浪力对比数字——这项误差声明不适用而非"通过"。装配仍未解的硬点：本征级数把 $1/(4\pi r)$ 奇性含在模态和里，自面板与近奇异求积（相当于 `near_singular` 的有限水深版本）必须先建，否则近场精度会被模态截断吃掉。
- 成本：$K_1$ 让梯度比核函数本身贵约 2 倍（同截断下 $K_0$+$K_1$ 两次求和）。

## 6. 复现

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.finite_depth3d_gradients \
    --output outputs/fd3d_grad/study.json --figures docs/figures \
    --report docs/finite_depth3d_gradients.md      # 11 门 PASS，约 60 s
PYTHONPATH=src python -m pytest tests/test_finite_depth3d_gradients.py -q
```

## 7. 给下一次自找活的线索

1. **下一步（自然续作）**：有限水深自面板求积——把核的奇性从模态和里剥出来（Rankine+底部镜像级数 + 光滑余项），复用 `kernels3d.integrate_single_layer_triangle_self`；判据用"面元加密下自面板积分 vs 解析 polar 积分"。
2. Garrett 截断圆柱的绕射：CHIEF 行在有限水深下的位置判据（`_auto_chief_points_3d` 直接可复用）——这是 roadmap 明写的第一个基准算例。
3. 深水侧仍欠：README roadmap 5 的 "higher-order/curved panels"（近奇异求积已结）。