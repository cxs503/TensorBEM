# m15990 · TensorBEM: 冰区辐射的有限水深闭合（`tensorbem.ice_finite_depth`）

分支 `agent/m15990`（基线 `4d63824`）。仓库 TensorBEM。EDIT_WHITELIST:
`src/tensorbem/ice_finite_depth.py`, `src/tensorbem/benchmarks/ice_finite_depth_ladder.py`,
`tests/test_ice_finite_depth.py`, `docs/ice_finite_depth_radiation.md`,
`docs/ice_finite_depth_radiation/`, `src/tensorbem/capability_evidence.py`,
`docs/audit_baseline.md`, `docs/ice_mass_loading_verification.md`,
`dev_report_m15990_ice_finite_depth.md`。

【锚点=NEW:功能点总账内 BEM 仅有 FEAT-OSS-INVENTORY-BEM（盘点），无冰区有限水深条目】建议新条目：
`slug=GAP-BEM-ICE-FINITE-DEPTH`，title「冰区辐射/绕射的有限水深能力（闭合 ice_rao 的 k_iH≥3 护栏）」，
所属域 TensorBEM；量化验收：①`μ→0` 极限复现开水域有限水深求解器（现 1.4e-12 / 2.4e-10）；
②`k_iH≥3` 重合带与深水冰求解器阻尼相对差 ≤1%（现 8.0e-3 @k_iH=3.3，2.7e-4 @5.6）；
③有限水深核在 `ν=K(ω)` 上满足冰面 Robin 条件（现 <1e-10）；④（未完成项）320 面板以上的网格收敛。

## 1. 案例介绍

`docs/ice_mass_loading_verification.md` 登记了一条诚实缺口：「`ice_rao.py` 要求 `k_i H ≥ 3`；
有限水深冰区 Green 函数未实现」。该护栏把整个北极陆架（H=20–200 m）排除在冰区 RAO 通路之外。
物理：质量加载冰盖（Weitz & Keller 1950）在有限水深上的线性自由面条件

    dφ/dz = K(ω)·φ,   K = ω²/(g − μω²),   μ = ρ_i h_i / ρ_w,

与开水域有限水深核实现的 Robin 条件形式相同（只是 `ν = ω²/g` 换成 `K`），因此冰区有限水深核
= `G_finite_depth(x, ξ; ν = K(ω), H)`；色散关系随之自动满足 `k_i tanh(k_i H) = K`，
即 `tensorbem.ice.mass_loading_wavenumber` 早已实现的那一支。

参数表（全部程序化生成）：ω=1.0 rad/s；h_i=0.9 m（厚度阶梯 0.2–2.5 m）；ρ_i/ρ_w=917/1025；
μ=0.805171 m；截止 `sqrt(g/μ)`=3.4893 rad/s；物体=闭合 icosphere R=1 m、中心 z=−1.5 m；
水深阶梯 H=10/15/20/30/50/100 m；面板 20（单频点另跑 80 面板）。

## 2. 怎么做的

1. 新模块 `tensorbem.ice_finite_depth`：`solve_radiation_finite_depth_ice(mesh, ω, ice)`
   取 `K = ice.robin_coefficient(ω, ice)`，调用既有 `solve_radiation_finite_depth_3d(mesh, ν=K, H)`。
2. **记账修正**（本单最容易出错的地方）：有限水深求解器由 `ν` 反推 `ω_implied = sqrt(g·ν)`，
   冰盖下 `K ≠ ω²/g` 因而 `ω_implied ≠ ω`；阻抗关系 `A + (i/ω)B = −ρ∫φ n dS` 中实部不受影响，
   阻尼需乘 `ω/ω_implied`。结果对象暴露 `implied_omega`，测试钉死该恒等式。
3. 验证驱动 `tensorbem.benchmarks.ice_finite_depth_ladder`：水深/厚度阶梯 + 两条解析极限 +
   一条网格敏感性对照，输出 `ladder.json`/`ladder.md`/PNG；文档零手抄（§3 整块是 `ladder.md` 原文）。
4. 能力台账：`ice-region-hydrodynamics` 行纳入新模块与三条新测试节点，重新生成
   `docs/audit_baseline.md` §4 发布块（原块与新行不一致，`--write` 后该测试转绿）。

## 3. 计算结果（程序化生成，见 `docs/ice_finite_depth_radiation/ladder.md`）

| H (m) | k_i H | A33 (kg) | B33 (kg/s) | 与深水冰区阻尼相对差 |
|---|---|---|---|---|
| 10 | 1.29 | 1742.67 | 8.132 | 深水模块护栏（k_iH<3） |
| 15 | 1.77 | 1744.04 | 8.508 | 护栏 |
| 20 | 2.27 | 1744.55 | 8.953 | 护栏 |
| 30 | 3.34 | 1744.65 | 9.434 | 8.03e-3 |
| 50 | 5.55 | 1744.54 | 9.558 | 2.67e-4 |
| 100 | 11.11 | 1744.53 | 9.561 | 7.50e-5 |

厚度阶梯（H=30 m）：h_i=0.2→2.5 m 时 B33 由 7.822 升到 15.108 kg/s（冰盖越重、辐射阻尼越大），
k_iH 由 3.13 升到 3.97。

## 4. 结果图

`docs/ice_finite_depth_radiation/ice_finite_depth_ladder.png`：左图有限水深冰区 heave 阻尼随
k_iH 阶梯 + 深水冰区参考线 + k_iH=3 护栏竖线；右图两核阻尼相对差（对数轴）与 1% 带。

## 5. 注意点与局限（对抗性自查结论）

1. **网格是主误差，且是实测而非假设**：20→80 面板使 A33 +30.4%、B33 +85.2%。所以本阶梯是
   *同一网格下的双路线一致性*研究，不声称绝对系数量值收敛；320+ 面板的网格收敛研究**未跑**
   （有限水深装配代价约 N²×模态数，超出本单预算）——这一条如实登记为未做。
2. **模态截断不是上述漂移的原因**：H=20 m 时自动模态数 872 与 2000 模态的 A33 差 5e-5。
3. **深水对照在 k_iH≳20 受截断限制**（H=200 m 时自动模态数饱和，一致性退化到 5e-3），
   故有效交叉验证带是 3 ≤ k_iH ≤ 11。
4. 离散互易性只到 1.7e-8 相对（3.7e3 量级上 6.2e-5 绝对），测试门设 1e-6 相对；
   非对角阻尼可以变号（sway-pitch 为负），因此只约束对角非负。
5. 有限水深冰区**绕射/激励力未实现**（有限水深求解器无入射波机制），文档与台账均保持不声称。
6. 绝对冰区 RAO 幅值无实验/第三方数值基准，维持 WITHHELD（与 `ice_rao` 一致）。

## 6. 误差判定（逐项，是否 <3%）

| 量 | 基准 | 实测 | <3% |
|---|---|---|---|
| 色散 `k tanh(kH)=K` | 解析恒等式 | 9.6e-16 1/m | 是（精确） |
| 核在 ν=K 的冰面 Robin 条件 | 解析恒等式 | <1e-10 相对 | 是（精确） |
| μ→0 附加质量 | 开水域有限水深求解器 | 1.41e-12 相对 | 是 |
| μ→0 阻尼 | 开水域有限水深求解器 | 2.39e-10 相对 | 是 |
| k_iH=3.34 阻尼 | 深水冰区求解器 | 8.03e-3 相对 | 是 |
| k_iH=5.55–11.1 阻尼 | 深水冰区求解器 | 2.7e-4–7.5e-5 相对 | 是 |
| 绝对 A33/B33 vs 收敛解 | 无收敛解（仅测到 80 面板） | 未建立 | **否/未完成（honest）** |

## 7. 测试与复现

```
PYTHONPATH=src python -m pytest tests/test_ice_finite_depth.py -q   # 16 passed in 20.5 s
PYTHONPATH=src python -m pytest tests/test_ice_rao.py -q            # 绿
PYTHONPATH=src python -m tensorbem.benchmarks.ice_finite_depth_ladder --output docs/ice_finite_depth_radiation
PYTHONPATH=src python -m tensorbem.capability_evidence --write
```

`tests/test_capability_evidence.py` 在本单改动前就有 4 条红（`tensorbem.sk_fit` 无行/无理由、
maruo 锚点、auxiliary 理由过期、发布块漂移）；本单修掉发布块那一条，剩 3 条为**既有欠账**，
已用 `git stash` 对照确认（stash 后 4 红 → 现在 3 红）。
