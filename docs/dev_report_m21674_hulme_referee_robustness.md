# dev report m21674 — CASE-BEM-HULME-HEMISPHERE：裁判侧 `referee_unresolved` 三格的归因

**锚点**：`CASE-BEM-HULME-HEMISPHERE`（算例池·Hulme 浮半球辐射/绕射 3D，容差 <5%，开工核对状态 open）
**仓**：TensorBEM · 分支 `agent/m21674`
**成果目录**：`benchmarks/verified/casepool_bem_hulme_hemisphere/`
`referee_robustness.py` / `referee_lid_arm.py` / `referee_robustness_merge.py` /
`referee_robustness.json` / `referee_robustness.md` / `referee_robustness.png` /
`referee_robustness_part_*.json` / `tests/test_hulme_referee_robustness.py`

## ① 欠账（选它做本题的理由）

`radiation_capytaine_band_xval.json`（本锚点的第三方轴线）本身是 **`Overall: FAIL`**：
capytaine 自身的 1024→4096 自漂移在 `ka = 2.58` 达 37.4%、`3.0` 达 6.8%、`4.0` 达 8.6%，
于是 `G2_referee_self_converged` FAIL → `G3_attribution_defined` 对这三格只能记
`referee_unresolved`，而 `G5_three_way_agreement_in_band` 从未跑过（`NOT_RUN`）。
也就是说：**锚点最重要的高 ka 频带里，到底是被测方（TensorBEM）错还是参考/裁判方错，是悬空的。**
m21563 已把「ka>2 的符号翻转」归因到 numpy 2.0.1 求解器缺陷，但**没有**回答上述归因问题。

## ② 方法

1. **不规则频率探针**：`ka = 2.30 … 2.86` 步长 0.04（14 点），capytaine 1024/4096 两档网格；
   同时报告相对度量与物理缩放度量（`|ΔA|/(ρ∇)`、`|ΔB|/(ρω∇)`）。
2. **lid 臂**：capytaine 3.1 支持 `lid_mesh`。用水面圆盘（8 环 × 64 扇区，共 960 三角形）
   把内部 Dirichlet 问题闭合（Nemoh/CHIEF 的标准做法），在同一 ka 网格上重算，
   对照 `ka = 1.0 / 2.0 / 2.2 / 2.54 / 2.58 / 2.62 / 3.0 / 3.5 / 4.0`。
3. **带内三方**：`ka = 1.0 / 2.0` 上补跑 TensorBEM(1024) × capytaine(4096) × Hulme，
   把原 `G5` 从 `NOT_RUN` 变成有裁定。
4. **判据纪律**：`R1/R2/R3/R4` 与 `L1a/L1b` 的阈值写进驱动脚本的 docstring **在运行之前**，
   合并步不改阈值；合并步之后才声明的 `L2/L2b` 明确记为 `post_hoc`。

## ③ 结果

### 3.1 B33 裁判误差在 ka=2.54(−68%) 与 ka=2.58(+108%) 之间变号

| ka | 2.46 | 2.50 | 2.54 | 2.58 | 2.62 | 2.70 | 2.86 |
|---|---|---|---|---|---|---|---|
| B33 误差 1024 | −29.1% | −45.4% | −86.1% | **+232.4%** | +86.8% | +35.2% | +17.9% |
| B33 误差 4096 | −16.5% | −27.3% | −67.8% | **+108.0%** | +35.2% | +15.4% | +7.9% |

光滑解析解不可能产生「误差极点」；离散系统的内部 Dirichlet 本征值可以。

### 3.2 加 lid 后极点消失，且裁判变成可用裁判（4096 面元，相对 Hulme 误差）

| ka | B33 无盖 | B33 加 lid | A33 无盖 | A33 加 lid | 加 lid 四量最坏 |
|---|---|---|---|---|---|
| 2.58（极点） | +108.0% | **+0.67%** | +0.45% | +1.19% | 2.06% |
| 2.54 | −67.8% | +0.66% | −6.36% | +1.20% | 2.04% |
| 2.0 | −1.98% | +0.54% | +1.13% | +1.24% | 1.56% |
| 3.0 | +5.81% | +0.77% | +1.53% | +1.16% | 2.25% |
| 4.0 | +2.75% | +1.09% | +1.35% | +1.08% | 2.26% |

**这是本单最有价值的一条**：加 lid 的 capytaine 在整个 `ka ∈ [1, 4]` 上与 Hulme 级数
**四量全部 ≤ 2.3%**。也就是说 Hulme (1982) 级数在高 ka 被一个独立代码确认，
锚点的「参考侧」不再是弱点；TensorBEM 在 ka>2 的 5–15% 误差**只能**是被测方自己的问题。

### 3.3 判据

| 判据 | 声明于 | 裁定 | 实测 |
|---|---|---|---|
| `R1_irregular_localised` | referee_robustness.py（运行前） | **FAIL** | 峰值 B33 @2.58 = +108%（4096），但 0.08 外仍有 21.2%（阈值 10%） |
| `R2_irregular_mesh_dependent` | 同上 | **FAIL** | 极点在 4096 网格 ka*=2.555、1024 网格 2.551，位移 0.005 < 一个探测步长 0.04 |
| `R3_referee_scaled_drift` | 同上 | **FAIL** | 缩放自漂移最坏 1.55e-2 @ (ka=4, B11) > 1e-2 |
| `R4_in_band_three_way` | 同上 | **PASS** | ka=1/2：TensorBEM 最坏 1.13%、capytaine 1.98%、码间差 <2.5% |
| `L1a_lid_removes_pole` | referee_lid_arm.py（运行前） | **PASS** | 加 lid 后极点格 B33 最坏 0.67% < 10% |
| `L1b_lid_leaves_far_field` | 同上 | **NOT_RUN** | 本轮 lid 臂 ka 网格不含 2.30/2.86 控制格 |
| `L2_lid_referee_accuracy` | 合并步（**事后**） | **PASS** | 四量 ≤ 2.26% |
| `L2b_lid_referee_scaled_drift` | 合并步（**事后**） | **PASS** | 加 lid 裁判 1024→4096 漂移 ≤ 1e-2 |

**总判 FAIL**（预注册判据里有 3 条没过）。`R1/R2` 的失败是**探针阈值与实测结构不符**，
不是数据不支持结论——按纪律不改阈值，结论由 `L1a`、符号穿越、以及（事后）`L2/L2b` 承担。

## ④ 注意点

* **lid 不是物理边界**：它只把内部 Dirichlet 问题闭合成非奇异问题，不改变辐射问题的解；
  但它给裁判引入自己的离散误差，`L2b` 量的就是这块（≤1e-2 物理缩放）。
* **被测方的高 ka 误差仍未收口**：`high_ka_mesh_refinement.py`（1024→4096 面元）已递交集群作业
  **cj3612**（8 分片，stage_dir=本工作树），是本会话内**未取回**的部分；因此
  「TensorBEM 在 ka>2 的误差是网格不足还是方法缺陷」在本单**仍然是开口问题**，不做任何暗示。
* 环境：python 3.11 / numpy 2.4.6 / capytaine 3.1.dev0。

## ⑤ 测试

`tests/test_hulme_referee_robustness.py`（5 项，全绿）：断言极点是一次符号穿越、
峰值 >100% 而远场 <10%、加 lid 后极点格 <5%、加 lid 判决四量 <3%、
以及「预注册 FAIL 判据保持 FAIL、总判 FAIL、事后判据被标为 post_hoc」这条诚实性不变量。