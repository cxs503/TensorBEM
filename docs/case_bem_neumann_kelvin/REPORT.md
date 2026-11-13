# CASE-BEM-NEUMANN-KELVIN — 稳态 Neumann-Kelvin 兴波阻力对标 Michell 薄船理论 C_w(Fr)

**仓库** TensorBEM · **分支** `agent/m17246` · **mission** 17246 · **锚点** CASE-BEM-NEUMANN-KELVIN（算例池，tol <5%）
**结论一句话**：主路线（稳态 Kelvin 远场算子 + 薄船源汇）在 Fr=0.20/0.25/0.30/0.35/0.40 五个点与 Michell 薄船理论
参考值的相对误差 **+0.197 % … +2.309 %（全部 < 5 %，判据 PASS）**，而近场压力积分路线（0.51×）与
**真实船体面元 NK 单层法路线失败（+1399 % … +4957 %，honest FAIL）**——两条子路线如实登记为未收敛。

---

## ① 案例介绍（问题 + ref 出处）

均匀来流 U 绕 Wigley III 抛物型数学船型定常兴波，深水、无黏、小扰动。总势 `Φ = -Ux + φ`，
定常自由面线性化条件 `φ_xx + κ φ_z = 0 (z=0)`，`κ = g/U²`；物面条件 `∂φ/∂n = U n_x`（Neumann-Kelvin 线性化）。

**ref（零手抄，可溯源）**：
- 理论原文：W. Michell, *The wave-resistance of a ship*, **Philosophical Magazine 45 (1898) 106–123**，
  形式 `R_w = 4ρg²/(πU²) ∫₀^{π/2} sec³θ |A(θ)|² dθ`，`A` 为薄船 Kochin 幅值。
- 数值取用：仓内经仲裁的规范实现 `tensorbem.wigley.michell_wave_resistance_canonical`，
  仲裁判据（定常色散可行性 / Fr 相似律指数 4 / 幅值一致性 / 求积收敛）见
  **`docs/michell_exit_arbitration.md`（mission m16345，本仓已验收）**；交叉复核用独立实现
  `tensorbem.thin_ship_michell` 的 λ 形式求积（`lam_max=40, n_lam=480`）。
- 两条 ref 路线互差 **≤ 0.036 %**（`numbers.json:route_a.rows[*].ref_route_spread`）。
- 容差：**< 5 %**（算例池声明）；`S_ref = L·T = 6.25 m²`（与
  `benchmarks/verified/bem_wigley_wave_resistance/results.json` 同口径）。

**几何先验证**（`numbers.json:geometry`）：`y = (B/2)(1-(2x/L)²)(1-(z/T)²)` ⇒
`Cb = 4/9 = 0.44444`、`Cm = 2/3`、`Cp = Cb/Cm = 1`、`V = 4LBT/9 = 2.77778 m³`、`A_wp = 2LB/3 = 6.66667 m²`；
`B/L = 0.1`、`T/L = 0.0625`、`L/B = 10`。湿表面积面元阶梯 8×4 → 64×32 收敛，末对差 **3.4e-4**。
（注：战役 metadata 里的 Cb=0.5625 不属于此船型，见 `docs/m10003/REPORT.md`，本单以几何为准。）

## ② 怎么做的（方法与实现）

**核函数**（复用 `tensorbem.nk_panel3d`，非新造）：稳态深水 Green 函数
`G = 1/(4πr) + 1/(4πr₁) + G_W`，`G_W` 为 Havelock 型积分（复指数积分闭式 + 方向积分），
波族满足定常色散 `k = κ sec²θ`。**独立自检**（不含任何 Michell 输入）：
自由面条件残差 `max|G_xx + κG_z|/scale = 4.53e-7`；辐射方向 后/前幅值比 **3.02**（尾迹在船后）。

**主路线（Route A，本算例的 got）**：薄船 NK 物面条件在中心面上给出源强 `σ = -2U ∂b/∂x`
（`b` 为半宽），用稳态 Kelvin **远场（Havelock/Kochin）算子**

```
Rw = ρκ²/(2π) ∫_{-π/2}^{π/2} |H(θ)|² sec³θ dθ ,   H(θ) = ∫_S σ e^{κ sec²θ (z + i(x cosθ + y sinθ))} dS
```

**常数是本次修好的缺陷**：`wave_resistance_far` 旧默认 `prefactor=0.5` 是无量纲数，
而精确值是 `κ/(2πg) = 1/(2πU²)`（带 1/U²）。这正好解释 `docs/m10003/REPORT.md` 报的
"隐含前因子在 Fr=0.2→0.4 漂 4.8×"：缺的 1/U² 在 0.2→0.4 之间正好变 4 倍。本单新增
`nk_panel3d.kochin_prefactor()` 并把默认值改为精确式（含推导 docstring），
`thin_ship_reference()` 同步返回 `rw_michell / rel_error_far_vs_michell`。

**子路线 B（真实船体面元 NK 单层法）**：`solve_nk_panel` 单层积分方程在湿表面求 σ，
同一远场算子出 Rw；辐射模式取本模块自己裁判过的 `halfplane, sign=-1`（比值 3.02；模块默认
`downstream, sign=+1` 的比值 0.99，非物理解）。

**工具链**：numpy/scipy(`expi`,`leggauss`) · matplotlib · `pytest`。
`repro_cmd`：`PYTHONPATH=src python3 scripts/case_bem_neumann_kelvin_michell.py --stage cheap --only geometry,kernel,route_a,route_b,wave_field`
（全量阶梯：`--stage full`，本机 120 s 单命令上限下分块执行，结果合并进同一 `numbers.json`）。

## ③ 参数表（无量纲数完整）

| 量 | 值 | 无量纲 |
|---|---|---|
| L / B / T | 10.0 / 1.0 / 0.625 m | B/L = 0.100，T/L = 0.0625，L/B = 10，B/T = 1.6 |
| ρ / g | 1025 kg/m³ / 9.80665 m/s² | — |
| Cb / Cm / Cp | 0.44444 / 0.66667 / 1.0 | 闭式 |
| V / A_wp | 2.77778 m³ / 6.66667 m² | S_ref = L·T = 6.25 m²，V^(1/3)/L = 0.1405 |
| Fr | 0.20 / 0.25 / 0.30 / 0.35 / 0.40 | U = 1.9806 / 2.4757 / 2.9709 / 3.4660 / 3.9611 m/s |
| κ⁻¹ = U²/g | 0.400 / 0.625 / 0.900 / 1.225 / 1.600 m | κL = 25.0 / 16.0 / 11.11 / 8.16 / 6.25 |

## ④ 对照表（ref vs got + 误差判据；数字由 `numbers.json` 程序化注入）

| Fr | U [m/s] | ref Rw [N] | ref Cw | got Rw [N] | got Cw | 相对误差 | <5% |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 0.20 | 1.9806 | 26.5393 | 2.112197e-03 | 27.1520 | 2.160960e-03 | **+2.309 %** | ✅ |
| 0.25 | 2.4757 | 49.7192 | 2.532503e-03 | 50.2540 | 2.559742e-03 | **+1.076 %** | ✅ |
| 0.30 | 2.9709 | 144.1357 | 5.098406e-03 | 144.8185 | 5.122560e-03 | **+0.474 %** | ✅ |
| 0.35 | 3.4660 | 114.3146 | 2.970783e-03 | 114.7902 | 2.983145e-03 | **+0.416 %** | ✅ |
| 0.40 | 3.9611 | 327.1101 | 6.508478e-03 | 327.7550 | 6.521309e-03 | **+0.197 %** | ✅ |

（`Cw` 以 `S_ref = L·T` 归一；取精确船体积分幅值 `A` 代入同一算子时误差为
**+0.0366 % … +0.0007 %**，见 `route_a.rows[*].err_closed_form`，即误差全部来自中心面面板求积而非算子。）

**子路线对照（honest FAIL，不作证据）**

| 子路线 | Fr=0.30 相对误差 | 判定 |
|---|---:|---|
| Route A 远场算子（1800 面板） | +0.474 % | ✅ |
| 近场压力积分（仅波核，中心面） | −48.6 %（20×10→40×20：−46.6/−48.1/−48.6 %，稳定在 0.51× 远场值） | ❌ |
| Route B 面元 NK 单层法（16×8） | +1399 %（Fr 全扫 +1399 %…+4957 %） | ❌ |

## ⑤ 收敛性 / 参数敏感性（≥2 项）

1. **中心面面板阶梯**（450 → 1800 面板，Route A）：Fr=0.20 误差 11.42 % → 2.31 %，
   Fr=0.30 1.94 % → 0.47 %，Fr=0.40 0.81 % → 0.20 %（单调下降，观测阶 ~1.15，
   2 层阶梯不足 3 层故 Richardson 记 `NaN`）。
2. **方向积分节点数** `n_theta = 501/1001/2001/4001/8001`：误差在 0.04 % 上下摆动（≤4e-4 波动），
   算子求积已收敛。
3. **ref 的 λ 截断** `lam_max = 20/40/60`：Rw 变化 ≤ 1.6e-3 N（相对 6e-5），ref 已收敛。
4. **ref 路线交叉**：仲裁规范式 vs 独立 λ 形式：相对差 ≤ 3.6e-4。
5. **Route B 的网格/方向节点（负结果）**：`n_theta` 24→48 时 Rw 从 1015 N 跳到 2161 N（×2.1）；
   网格 8×4 → 24×12（模块默认辐射模式）1173 → 745 → 534 → 378 N，**不收敛**；
   同一网格换 `halfplane,sign=-1` 得 2161 N（16×8）→ 2847 N（24×12），**随加密发散**。
   B/L = 0.2/0.1/0.05 的阶梯误差 +8206 %/+1399 %/+1529 %（对应面板到自身镜像的最小距离
   0.0275/0.0137/0.0069 m，条件数 71.6/45.0/34.7）——薄船极限下镜像核近奇异。

## ⑥ 失败模式、工件与生态摩擦

**工件**（worktree 内、非 gitignore 路径）：
`docs/case_bem_neumann_kelvin/numbers.json`（本表全部数字的唯一来源）、`table.md`、
`fig_cw_vs_froude.png`、`fig_route_a_convergence.png`、`fig_kelvin_wake.png`、`wave_field.npz`；
脚本 `scripts/case_bem_neumann_kelvin_michell.py`；测试 `tests/test_case_neumann_kelvin_michell.py`（5 项，
含"前因子无 Froude 漂移"回归钉，`pytest` 5 passed；`tests/test_nk_panel3d.py` 38 passed 与
`tests/test_neumann_kelvin.py` 26 passed 未受影响）。

**失败模式**
- **Route B（面元 NK）honest FAIL**：物面条件残差 1e-15（线性方程解得干净），
  但远场阻力随网格/方向节点不收敛且随加密变大 ⇒ 缺陷不在线性代数而在
  **核函数在物面上的离散**：`halfplane` 辐射项是 `D=0` 上的 Heaviside 型不连续，
  逐面元中点求积对该不连续不收敛，σ 带网格相关偏置。建议修法（后续单）：
  按 `D=0` 拆分面板后解析积留数项、并把自由面积分（m-terms）纳入 NK 积分方程。
  这与 `docs/m10003/REPORT.md` 的近场失败同源，故该子能力维持 WITHHELD。
- **近场压力积分路线**：稳定收敛到远场值的 **0.51×**（三次加密不变），说明该路线的
  二阶功/半片处理缺项，不是离散误差——登记为未解决缺陷，不用于验收。
- **波形张角**：本单用最外波峰百分位估计器得 74.6°，**未复现 19.47°**（m10003 独立测得 55°）。
  估计器粗糙 + `halfplane` 辐射模式在近场把非驻相方向也留了残数；波形图仅作定性。
- **远场算子与 Michell 的结构同源**：Route A 与 ref 共享同一个一维 θ 积分结构，故它验证的是
  **稳态核色散 + 留数常数 + 无量纲归一**（这三处正是 m10003 出错的地方），
  不含真实船体积分信息；真实船体积分那一步（Route B）**未通过**。此边界必须如实声明。

**生态摩擦**
1. `m10003` 的"隐含前因子漂移 4.8×"根因是缺 1/U²（本次已修 + 回归钉），旧代码把物理常数
   写成"锚定在薄船极限的 0.5"——**用 ref 反标定算子**是反模式，会把 Fr 依赖藏进常数里。
2. 仓内 Michell 出口曾有三套互不相同的数（`docs/michell_exit_arbitration.md`）：
   `wigley.michell_wave_resistance` 仍是被判 DEFECTIVE 的旧式（本单另一侧：它给 Fr=0.30 的
   Rw=12.29 N，与仲裁正确值 144.14 N 差 11.7×），而 `benchmarks/verified/bem_wigley_wave_resistance`
   正是按这条旧式出的"verified"基准 —— **该 verified 基准的 ref 数值应重出**（新增欠账）。
3. `tensor/knowledge/search` 引擎离线（8021 拒连），开工三查只能走 `negative-knowledge`
   （命中 `negk:gate:126`：nk_panel3d 交付声明被证伪，诚实失败记录即本单 Route B 的前车之鉴）。
4. 本机单命令 120 s 上限 + 集群队列排队，全量阶梯需分块（`--only`）跑。