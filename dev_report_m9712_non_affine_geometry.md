# TensorBEM 二次曲面单元算子验证：非仿射几何轴（PR#588 重开·管到底第1轮·m9712）

**Task**: m9712（重开 TensorBEM#588）· **Branch**: `agent/m9712` · **Repo**: TensorBEM
**Date**: 2026-11-10 · **Base**: `origin/main` @ b8480c7（从当前 main 切枝）
**Status**: ✓ 功能+测试齐备；增量 = 验证面新增**非仿射几何轴**（首个离开仿射群的算例）+ 固体角残差的归因实验（quadrature-order 不变 ⇒ 几何 C0 折痕，非求积缺陷）。既有球/椭球/剪切算例数值**零漂移**。

---

## ① 案例介绍

### 1.1 物理背景

三维 Laplace 外问题，直接第二类边界积分方程

```
(1/2 I - D) u = -S q,   S_ik = ∫_T N_k G dS,   D_ik = ∫_T N_k ∂G/∂n_y dS,   G = 1/(4π|x-y|)
```

闭曲面 Γ 内含点源 `x0` 时，`u(x)=1/(4π|x−x0|)` 在 Γ 外调和，故对**任意**形状的闭 Γ 该场都是对应 Dirichlet 问题的精确解，`q = −(x−x0)·n/(4π|x−x0|³)` 是精确通量。本单用它把「算子实现是否正确」与「几何形状」解耦。

### 1.2 本轮几何：径向调制球（非仿射）

```
r(t) = 1 + ε·P₃(cos t),   ε = 0.15,  t = 与 z 轴夹角,  P₃(x)=(5x³−3x)/2
```

| 参数 | 值 |
|---|---|
| 单元 | 6 节点二次（曲边）三角形，顶点 + 边中点配点 |
| 网格 | icosphere 二次细分 sub=0/1/2 → 42/162/642 节点，20/80/320 单元 |
| 半径范围 | 0.85 … 1.15（节点全部严格落在调制面上，含边中点） |
| 求积 | 非奇异 Gauss 8 阶；奇异自面板 Duffy 16 阶 |
| 源 | `x0 = (0,0,0)`（体内，几何中心；`r_min=0.85>0` ⇒ 源在闭曲面内） |
| 判据（**动手前预设**） | finest 通量相对 L2 误差 ≤ 3%（硬限 15%）；观测阶 ≥ 1.5；非真空性：`affine_image_residual > 1e-2`（该几何不是单位球的仿射像） |

### 1.3 为什么需要这一轴（既有算例的共同盲区）

既有 verification 面的几何族：球、椭球 `1.0/1.6/0.7`、剪切椭球（m8799）、平移/旋转/缩放/镜像（m8723/8745/8769/8788）——**全部是单位球的仿射像**。二次 Bézier 三角形在仿射变换下协变（像的控制网 = 像的 Bézier 网），所以「线性一致但非线性错误」的 Jacobian/法向实现缺陷对它们**完全不可见**。径向调制是非线性映射，与六控制点表示不相容，因而首次真正踩到 `det J`、`n` 与奇异自面板求积在非均匀曲率下的行为。

---

## ② 怎么做的

### 2.1 改动（单一功能点，3 个文件 + 工件）

1. `src/tensorbem/benchmarks/quadratic_panel_verification.py`
   - `RADIAL_MODULATION`（算例表）+ `legendre_polynomial()`（三项递推）+ `radially_modulated_mesh()`（顶点与边中点都投到调制面；`amplitude=0` 即复现 `icosphere_quadratic`）；
   - `affine_image_residual()`：对应节点集间最小二乘仿射拟合 `y=Ax+t` 的归一化 RMS 残差 —— **几何轴的机器可判标签**（仿射算例 ~1e-16，非仿射 ~5.6e-2）；
   - `min_normal_radial_cos()`：`min n·x/|x|` —— 记录面法向偏离径向的程度（球 = 1）；
   - `GEOMETRY_CASES` 增第四例 `wavy_r3_e0p15`；`run_geometry_study()` 对调制算例建非仿射网格，逐级记录 `affine_image_residual` / `min_normal_radial_cos`，算例记录新增 `radial_modulation` 字段。
   - **未触碰核心模块 `src/tensorbem/panel_quadratic.py`（0 行改动）**：本轮只扩验证面，不改算子。
2. `tests/test_panel_quadratic_operator.py`：+4 钉
   - `test_non_affine_geometry_flux_within_hard_limit`（sub1，162 节点，15% 硬限 + 固体角残差 < 5e-2）；
   - `test_affine_image_residual_classifies_the_geometry_axis`（**非真空性守卫**：球/椭球/剪切 ≤1e-12，调制球 >1e-2；且 S、D 相对差 >5%，不是换了标签的球）；
   - `test_solid_angle_residual_is_not_a_quadrature_error`（**归因实验**：Gauss (8,16) 与 (16,32) 的 `max|D·1+1/2|` 相对差 ≤1e-6；残差若随求积阶变化才是求积缺陷）；
   - `test_committed_study_records_the_non_affine_geometry_case`（入库工件必须带调制算例、`degree/amplitude`、3% 与阶门通过、非仿射残差 >1e-2，且其余算例 `affine_image_residual < 1e-12`）。
3. `docs/quadratic_panel_verification/README.md`：工件表 + 几何 study 说明 + 门列表补第四例。
4. 工件：重跑基准重新生成 `study.json`（provenance sha256 自动更新）与 `fig_geometry_generic.png`；`fig_quadratic_operator.png` 重跑后**字节不变**（证明既有数值零漂移）。

### 2.2 实验设计（预注册）

- **决策**：算子能否在「非仿射等参畸变」几何上维持 3% 通量验收与 O(h^1.5) 阶？能 ⇒ 适用域可从「仿射像」扩到「非线性参数化的光滑闭曲面」；不能 ⇒ 记为实测适用域上限（不调门）。
- **控制变量**：细分级别、求积阶、奇异阶、源（几何中心）、误差度量（质心相对 L2）与既有 geometry study 逐项一致；**唯一改变**的是几何映射（仿射 → 径向非线性）。
- **混杂因素**：调制改变面面积与局部曲率 ⇒ 用无量纲通量误差 + 同层节点数对比；源相对深度 ⇒ 源在所有算例都取几何中心。
- **重复性**：确定性，无随机；provenance sha256 守护（pytest 重算）。

### 2.3 预实验中被否证的两个假设（诚实记录）

1. **「固体角残差大 = 近奇异求积不足」——否证。** 先怀疑 wavy 面上 `max|D·1+1/2|=6.3e-3`（球 9.6e-4）来自近奇异配对，把最差行（极点节点）的邻面板按 m8718 的分级 Duffy 重算：`d/H≈0.75` 处两种规则一致到 1e-16，最差行总残差**完全不变**；再把 Gauss 阶从 8 提到 16、24 与自面板 16→32→48，残差**十位数字不动**（§③.3）。结论：残差不是求积误差，而是配点节点处相邻面片**切平面不连续（C0 折痕）**导致的局部立体角偏离，`O(h²)` 收敛（球 3.87e-2→6.32e-3→9.61e-4）。该结论已固化为回归钉。
2. **「径向调制可以做出真非凸（凹面）算例」——否证。** 对径向图 `r(t)` 恒有 `n·ê_r = 1/√(1+((∇r)/r)²) > 0`，故 `n·x > 0` 恒成立；把 ε 提到 0.4、k 提到 5 时 `n·x<0` 的单元数仍为 **0/320**。**真非凸性（亏格≥1 环面、凹颈）本轮未覆盖**，登记为待补缺口（§⑤.3）。

---

## ③ 计算结果（程序化注入，零手抄，源 `docs/quadratic_panel_verification/study.json`）

### 3.1 四几何并列（同级对比，源均在几何中心）

| 几何 | affine_image_residual @sub1 | finest 通量误差 | 观测阶 | 3% 门 | 阶门 ≥1.5 | min n·x/\|x\| @sub2 |
|---|---|---|---|---|---|---|
| sphere | 5.376e-16 | 6.7660e-03 | 2.44 | True | True | 1.0000 |
| ellipsoid 1.0/1.6/0.7 | 5.821e-16 | 6.4257e-03 | 2.63 | True | True | 0.7381 |
| sheared ellipsoid (det=1) | 5.503e-16 | 7.6679e-03 | 2.72 | True | True | 0.6011 |
| **wavy r=P₃ ε=0.15（非仿射）** | **5.694e-02** | **1.0474e-02** | **2.21** | **True** | **True** | 0.9523 |

### 3.2 非仿射算例逐级记录

| sub | 节点数 | h | 通量相对 L2 误差 | 固体角残差 max\|D·1+1/2\| | 单元度量各向异性 | affine_image_residual | cond(S) |
|---|---|---|---|---|---|---|---|
| 0 | 42 | 0.6927 | 1.6730e-01 | 9.021e-02 | 1.298 | 5.901e-02 | 20.9 |
| 1 | 162 | 0.3846 | 4.1616e-02 | 3.408e-02 | 1.196 | 5.694e-02 | 40.5 |
| 2 | 642 | 0.1982 | **1.0474e-02** | 6.277e-03 | 1.114 | 5.636e-02 | 85.9 |

收敛阶（log-log 最小二乘）= **2.21**（误差比 4.02×、3.97× 每层减半），球同口径 = 2.44。

### 3.3 归因实验：固体角残差 vs 求积阶（非仿射面，162 节点）

| 几何 | Gauss/自面板 (8,16) | (12,24) | (16,32) | 相对离散度 |
|---|---|---|---|---|
| unit sphere | 6.3213951e-03 | 6.3213932e-03 | 6.3213932e-03 | 3.0e-07 |
| ellipsoid 1.0/1.6/0.7 | 1.1732769e-02 | 1.1732674e-02 | 1.1732674e-02 | 8.2e-06 |
| **wavy r=P₃ ε=0.15** | **3.4084417e-02** | **3.4084417e-02** | **3.4084417e-02** | **2.6e-10** |

三个几何的残差都**不随求积阶变化**（非仿射面相对离散度 2.6e-10），互相之间却差 5 倍 ⇒ 残差是几何分辨率（折痕角）的函数，不是求积吞吐的函数。

### 3.4 预实验否证记录（非交付值，仅作局限证据）

| 调制 | sub=2 通量误差 | 是否 ≤3% | 说明 |
|---|---|---|---|
| k=3, ε=0.15（**本轮算例**） | 1.047e-02 | ✓ | 阶 2.21 |
| k=3, ε=0.25 | 2.44e-02 | ✓（贴线） | — |
| k=4, ε=0.30 | 6.33e-02 | ✗ | 分辨率受限 |
| k=4, ε=0.40 | 1.09e-01 | ✗ | 同上 |
| k=5, ε=0.30 | 1.34e-01 | ✗ | 固体角残差亦升至 1.4e-1 |

---

## ④ 结果图

`docs/quadratic_panel_verification/fig_non_affine_geometry_m9712.png`（`make_report_figure_m9712.py` 生成，读 study.json，零手抄）：
- 左：四几何通量收敛 vs h（含 3%/15% 门限与 O(h²) 参考线）；
- 右：固体角残差 vs 求积阶（球 / 椭球 / 非仿射），三条曲线各自水平 ⇒ 求积阶无关。

`docs/quadratic_panel_verification/fig_geometry_generic.png`（基准自带，本轮重生成）：几何泛化收敛 + 薄体范围限；新算例作为第四条曲线出现。

---

## ⑤ 注意点

1. **仅 Laplace 算子**：Helmholtz 频域扩展未做（D 矩阵可复用）。
2. **非仿射 ≠ 非凸**：径向调制能测非线性等参畸变，但径向图的法向恒朝外，凹面/凹颈（真非凸）需亏格≥1 或缝合几何 —— 本轮**未覆盖**，后续单应在环面或双球缝合面上补。
3. **残差判读**：`max|D·1+1/2|` 不是求积精度指标，是「网格折痕角」指标（§③.3）；拿它当求解器精度用会误判（曲面越弯越"差"是正常的）。
4. **精度随曲率降级**：非仿射面同层误差约为球的 1.55×（1.05% vs 0.68%），阶 2.21 vs 2.44 —— 曲率非均匀时是分辨率受限，不是算子错误（否则阶会崩、层间比率会乱）。
5. **调制上限**：ε≥0.25（k=4）后 3% 门失守（§3.4），故算例取 ε=0.15 并如实登记上限。
6. **未改核心模块**：`panel_quadratic.py` 0 行改动，`fig_quadratic_operator.png` 重跑字节不变 ⇒ 既有数值零漂移。

---

## ⑥ 误差判定

| 验收项 | 基准 | 实测 | 是否 <3% | 判定 |
|---|---|---|---|---|
| 非仿射面 finest 通量相对 L2 误差（642 节点） | 精确解 `q=−(x−x0)·n/(4π\|x−x0\|³)` | **1.0474e-02 (1.05%)** | **是** | PASS |
| 硬限制 15% | — | 1.05% | 是 | PASS |
| 观测收敛阶 ≥1.5 | 2.21 | 2.21 | 是（门） | PASS |
| 非真空性：非仿射残差 >1e-2 | 1e-2 | 5.636e-02 … 5.901e-02 | 是（55×） | PASS |
| 仿射算例残差 <1e-12 | 1e-12 | 4.2e-17 … 1.18e-15 | 是 | PASS |
| 固体角残差求积阶无关 | 相对差 ≤1e-6 | 2.6e-10 | 是 | PASS |
| 既有算例零漂移 | 球/椭球/剪切 finest | 6.7660e-03 / 6.4257e-03 / 7.6679e-03 | 与 main 一致 | PASS |
| 相似不变性复验 | 1e-10 | 2.43e-13 | 是 | PASS |
| 算子测试文件 | — | 63 passed（含本轮 4 钉） | — | PASS |
| 全量测试 | 见 §⑦ | 分支/基线失败集逐条相同，`cj357` 排队中 | — | 见 §⑦（诚实声明：未跑完） |

**结论**：非仿射几何轴上算子达 3% 验收（1.05%）并保持 O(h²) 阶；`affine_image_residual` 把「是否为新的几何轴」变成机器可判的事实；固体角残差被归因为几何折痕而非求积。`acceptance.passed=true`（`geometry.passed=true`，四算例全绿）。

---

## ⑦ 测试执行与回归证据

```bash
# 1) 本轮触点（算子 + 基准两个文件）
PYTHONPATH=src python -m pytest tests/test_panel_quadratic_operator.py tests/test_panel_quadratic.py -q
# -> 63 passed in 124.85s   （含本轮新增 4 钉）

# 2) 文档面（我改动了 docs/quadratic_panel_verification/README.md 与 study.json）
PYTHONPATH=src python -m pytest tests/test_doc_claim_audit.py tests/test_doc_inventory.py -q
# -> 35 passed, 2 failed  —— 与未改动的基线 main (b8480c7) 逐条一致（见 3）

# 3) 基线对照（证明全量套件里的红不是本轮引入）
git worktree add /tmp/base b8480c7        # 干净 main 检出
pytest <前 150 个用例 id> -q --tb=no -rf  # 两棵树各跑一遍
# 分支: 9 failed, 141 passed in 170.13s
# 基线: 9 failed, 141 passed in 168.70s
# diff FAILED 列表 -> IDENTICAL_FAILURES（9 条全是既存红）

# 4) 全量套件（2379 用例，重计算，按纪律上集群 loong64）
#    run_cluster cj357  —— 14:06 递交，队列积压 ~150，本单收尾时仍排队中（未跑完）
```

**诚实声明**：全量套件本单**未取得完整结果**（本地单次跑约 2 小时超出控制面限制，集群作业 cj357 排队未派发）。已取得的等价证据是：①本轮触及的两个测试文件全绿（63 passed）；②前 150 个用例在分支与干净 main 上**失败集完全相同**（9 条既存红，与本轮无关）；③文档面 2 条红同样在基线上已存在；④`fig_quadratic_operator.png` 重生成后字节不变，说明既有数值零漂移。

## ⑧ 复现命令

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_panel_verification   # 重生成 study.json + 两张基准图
PYTHONPATH=src python make_report_figure_m9712.py                            # 生成本报告证据图 + 归因表 json
PYTHONPATH=src python -m pytest tests/test_panel_quadratic_operator.py -q
```