# CASE-BEM-RIGID-LID-HEMISPHERE — 刚性盖半球升沉附加质量（ka→0）BEM 验证

**mission m17243 · TensorBEM · 算例池第 1 轮** · 判据 <5 % · **结论：PASS（headline 误差 +0.283 %）**

## ① 案例介绍（物理背景 / 几何 / 参数）

浮半球（半径 a=1 m）升沉辐射问题的 **ka→0 端点**：自由面条件 ∂φ/∂z=0 在 ω→0 时
退化为刚性盖（rigid lid），即本算例。镜像论证的关键陷阱（m16802/m16850 已裁决）：
升沉速度垂直于对称面，反射后镜像体速度反号 ⇒ 双体问题是**偶宇称**
`∂φ/∂r = -U|cosθ|`（两半"滑动"），**不是**整球平动。因此 ka→0 端点
**不是** 2/3·ρπa³ = ρ∇（旧错锚 1.0），而是偶模级数精确值。

| 参数 | 值 |
|---|---|
| 几何 | 半球 a = 1.0 m，平面盖在 z=0（刚性盖极限） |
| 排水体积 ∇ | (2/3)πa³ = 2.094395 m³ |
| 流体密度 ρ | 1025.0 kg/m³ |
| g | 9.80665 m/s² |
| ρ∇ | 2146.755 kg |
| 无量纲化 | A33/(ρ∇)；ka = ωa/√(ga)；约定 exp(-iωt)、单位速度幅值 |
| 模式 | heave（升沉） |

**ref 出处（零手抄）**：偶宇称双体 Legendre 级数（仓内
`tensorbem.benchmarks.rigid_lid_hemisphere_attribution.even_mode_added_mass_dimless`，
Gauss-Legendre 机器精度求积，degree_max=60，31 个偶模，末两项部分和差
4.63e-09）：

    A33(ka→0)/(ρ∇) = 0.830954148

独立旁证：Hulme (1982) 多极级数（`tensorbem.analytic.hulme_hemisphere_radiation`，
另一条代码路径）在 ka=2e-3 处为 0.83507，自上方单调逼近同一极限。
（ka→∞ 压力释放端点 = 0.5ρ∇ 为奇宇称单模，精确，作对照线。）

## ② 怎么做的（方法 / 流程 / 工具链）

1. **ref**：偶模级数（镜像刚性盖解析，ka→0 端点），机器精度。
2. **got**：3-D Kelvin-Green 函数辐射求解器 `solve_radiation_3d`（ka→0 时核退化为
   镜像 Rankine 核 = 刚性盖），八面体递归细分网格 64/256/1024 面板。
3. **交叉验证**：Hulme (1982) 半解析解全程对照。
4. 一键复现：`python3 benchmarks/verified/bem_rigid_lid_hemisphere_ka0/run_study.py`
   （本机 ~60 s，未上集群）；图：`make_figure.py`。
5. 全部数字由 `results.json` 单一 payload 渲染，零手抄。

## ③ 参数表（无量纲数完整）

| 量 | 符号 | 值 |
|---|---|---|
| 频率参数（网格阶梯） | ka | 1e-3 |
| 频率参数（ka 阶梯） | ka | 1e-3, 1e-2, 3e-2, 0.1 |
| 网格 | panels | 64 / 256 / 1024 |
| 网格面积相对误差 | — | -4.87e-2 / -1.26e-2 / -3.17e-3 |
| 系统条件数 | cond | 2.18 / 2.10 / 2.06 |
| 参考值 | A33(0)/(ρ∇) | 0.830954148 |
| 压力释放端点 | A33(∞)/(ρ∇) | 0.500000（精确 0.5） |

## ④ 对照表（ref vs got + 误差）

**网格阶梯（ka=1e-3）**

| sub | panels | A33/(rho·nabla) | ref | rel err | Hulme | rel err vs Hulme | cond |
|---|---|---|---|---|---|---|---|
| 2 | 64 | 0.807388 | 0.830954 | -2.836 % | 0.835066 | -3.314 % | 2.18 |
| 3 | 256 | 0.828152 | 0.830954 | -0.337 % | 0.835066 | -0.828 % | 2.1 |
| 4 | 1024 | 0.833304 | 0.830954 | +0.283 % | 0.835066 | -0.211 % | 2.06 |

**ka 阶梯（256 面板）**

| ka | A33/(rho·nabla) | rel err vs rigid-lid ref | within 5 % |
|---|---|---|---|
| 0.001 | 0.828152 | -0.337 % | yes |
| 0.01 | 0.846565 | +1.879 % | yes |
| 0.03 | 0.864358 | +4.020 % | yes |
| 0.1 | 0.856361 | +3.058 % | yes |

**判据**

| id | measured | criterion | verdict |
|---|---|---|---|
| C1_reference_below_rho_nabla | 0.830954 | ref < 0.9 (refutes the 1.0 rho*nabla anchor) | PASS |
| C2_headline_within_5pct | 0.282785 | |rel err| < 5 % at ka=1e-3, finest mesh (sub=4) | PASS |
| C3_mesh_convergence_monotone | [0.028360520670282607, 0.003372581600669222, 0.002827846418132385] | error strictly decreases 64 -> 256 -> 1024 panels | PASS |
| C2b_256panel_within_5pct | -0.337258 | |rel err| < 5 % at ka=1e-3, 256 panels | PASS |
| C4_ka_ladder_bounded_5pct_at_ka_le_0.01 | 0.0187872 | max |rel err| < 5 % for ka <= 0.01 (256 panels) | PASS |
| C5_series_converged | 4.63474e-09 | last two even partial sums differ < 1e-6 | PASS |

**误差判定（显式回答）**：headline（1024 面板, ka=1e-3）误差
**+0.283 %**；256 面板 -0.337 %；对独立 Hulme 参考全程最大
0.829 %。**是否 <3 %：是**（+0.283 % < 3 %）；
**是否 <5 %（本单判据）：是**。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

- **网格**：|err| 2.836 % → 0.337 % → 0.283 %（64→256→1024），单调收敛且
  256 与 1024 两点**夹逼**参考值（0.828152 < 0.830954 < 0.833304）。
  256→1024 表观阶仅 0.13：误差已 <0.34 % 且变号，阶数估计在此区间不可靠
  （如实记录，不粉饰）。
- **ka 敏感性**：对刚性盖参考的偏差随 ka 增大（0.34 % → 1.88 % → 4.02 % @3e-2，
  0.1 处回落到 3.06 %）——这是**物理**（有限 ka 的波辐射修正），不是求解器误差：
  对 Hulme 独立参考的偏差全程 ≤ 0.83 % 且几乎平坦。
  结论：ka ≤ 0.01 时 BEM 对刚性盖端点 <2 %；ka=0.03 时 4.02 % 仍在 5 % 内。

## ⑥ 失败模式与工件

- **无失败**：全部判据 PASS。历史失败模式（旧锚 1.0 造成 14.4 % 假 FAIL）已由
  m16850 裁决为参考错误，本单复现并确认（C1：ref=0.830954 < 0.9）。
- **局限**：(i) 本算例是自由面浮半球的 ka→0 极限（刚性盖=镜像 Rankine 核），
  非"实体刚性盖板+无限流体"的封闭体问题——两者在偶宇称流场意义下同解，
  但几何表述不同，引用时须注明；(ii) 256→1024 的收敛阶估计不可靠（见⑤）；
  (iii) ka>0.1 的端点行为不在本单范围。
- **工件**：`benchmarks/verified/bem_rigid_lid_hemisphere_ka0/{results.json, summary.md,
  figure.png, run_study.py, make_figure.py}`；repro_cmd：
  `python3 benchmarks/verified/bem_rigid_lid_hemisphere_ka0/run_study.py && python3
  benchmarks/verified/bem_rigid_lid_hemisphere_ka0/make_figure.py`
- **生态摩擦（如实记）**：开工三查的 knowledge_base / negative_knowledge_ledger /
  oss_solutions 端点在 :8010 API 上 404（未暴露），三查未能经 API 完成，
  改以仓内 grep + 既有 dev_report（m16802/m16850）替代；建议平台补齐这三个端点。
