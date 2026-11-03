# 变异审计待办清偿：水线面退化/偏心分支与深水核深度导数路径

**日期**：2026-11-03 · **分支**：`agent/m2715` · **仓**：TensorBEM
**上游欠账**：`docs/mutation_audit_tensorbem.md` §5「仍然存活的变异体：逐条分类」中判为
**真实覆盖缺口（待办）** 的四条：

| 变异体 | 审计原文判定 |
|---|---|
| `hydrostatic.py:236` `x_moment` 的 `/ 6.0 → * 6.0` | 现有水线面算例都近似对称，一阶矩≈0，需要**偏心**水线面（如 `x∈[0,2]`、`y∈[0,1]`）把形心钉到 `(1, 0.5)` |
| `hydrostatic.py:255` 退化返回 `0.0 → 1.0` | 没有算例构造「完全没有水线面」的船体，该分支从未被进入 |
| `hydrostatic.py:314` 零排水量返回 `np.zeros(3) → np.zeros(4)` | 同一类退化分支未覆盖 |
| `free_surface.py:243` `df_d_depth` 的 `2.0 * → 2.0 /` | `df_d_depth` 只喂给 `dg_d_depth`，而套件钉的是对 `source_x/source_z` 的导数；「要么它是死代码，要么缺一条导数的有限差分护栏」 |

本单把这四条逐条结清：**前三条用闭合解算例补护栏，第四条先定因（确认是死代码）再做最小重构，
使该导数进入返回值并由有限差分直接钉死**。

---

## 1. 案例介绍（物理背景与几何）

### 1.1 偏心水线面（静水力）

浮体静水力里的水线面一阶矩决定漂心/浮心横向位置，二阶矩决定横/纵稳心半径
`BM_T = I_xx/∇`、`BM_L = I_yy/∇`。若一阶矩归一化写错，**对称船型（驳船、半球、椭球）全部看不出**：
它们的形心本来就落在 `(0,0)`，错归一化只会得到 0。因此必须用**偏心**水线面做判据。

| 参数 | 值 |
|---|---|
| 水线面 | 矩形 `L=2.0 m × B=1.0 m`（x、y 方向） |
| 水平偏移 | `(Δx, Δy) = (1.0, 0.5) m`（即 `x∈[0,2]`、`y∈[0,1]`），另测 `(0,0)` 与 `(-0.7, 0.3)` |
| 吃水 | `T = 1.0 m`（直壁驳船，全部面板落在解析面上） |
| 网格 | `box_hull_mesh(2, 1, 1, n_length=3, n_beam=2, n_draft=2)` |
| 解析基准 | `A = L·B = 2`；形心 `(1.0, 0.5)`；`I_xx = L B³/12 = 1/6`；`I_yy = B L³/12 = 2/3`；`I_xy = 0` |

### 1.2 深水自由面核的深度导数（水动力）

`free_surface.green_gradient` 提供对**源点**坐标的导数，用于双层位势核。其中竖直分量由两部分组成：

- 光滑部分对源深 `A = -(z + z₀)` 的导数 `dG/dA`（含 `-log r_image`、主值积分与留数）；
- Rankine 部分 `r = |P − Q|`，它是唯一依赖 `z₀` 而**不经过** `A` 的项。

审计发现的缺陷正是：`dg_d_depth` 算完之后被丢弃，`dg_d_source_z` 是用同一批中间量手写展开的第二份表达式。
两份表达式任何一处漂移都不会被测试发现（`dg_d_depth` 没有被任何断言触及）。

| 参数 | 值 |
|---|---|
| 无量纲频率 | `ν = ω²/g = 1.0 m⁻¹` |
| 场点 / 源点 | `(0.7, -1.3)` / `(0.0, -2.1) m`（另两组见测试参数化） |
| 参考解 | `include_rankine=False` 的 `green_function` 对 `source_z` 的中心差分（步长 1e-6 m） |
| 恒等式 | `d/dsource_z = -d/dA` |

---

## 2. 怎么做的（方法与工具链）

1. **先定因再改码**：把 `free_surface.py` 中 `df_d_depth`/`dg_d_depth` 的数据流追到返回值，确认
   `dg_d_depth` 未被返回、未参与 `dg_d_source_z`——即审计说的「死代码」分支成立，而不是「缺测试」。
2. **最小重构**：把深度导数集中到 `_depth_derivative(offset, depth, r_image2, frequency, cos_integral)`
   ＋公开包一层 `smooth_depth_derivative(...)`（做校验/广播），`green_gradient` 改为
   `dg_d_source_z = -dg_d_depth + (1/2π)·(-(z-z₀)/r²)`。代数上与旧表达式**逐项等价**（同项重排），
   数值上仍逐位一致；同时删掉重构后变成死代码的 `df_d_depth` 与 `dresidue_d_depth`。
3. **补算例**（`tests/test_hydrostatic_waterplane_gaps.py`，8 个用例）：
   偏心驳船形心/二阶矩、偏移后复原矩阵平移不变性、全潜没球体（无水线面）、
   面积低于 `1e-14` 的水线环（`hydrostatic.py:255` 分支）、零厚度退化面（`hydrostatic.py:314` 分支）、
   空面板表（同名哨兵）。
4. **补导数护栏**（`tests/test_free_surface_smooth_depth_derivative.py`，10 个用例）：
   `-smooth_depth_derivative` 对 `source_z` 的中心差分（三组点位、`rel=1e-8`）、
   非平凡性（实部/虚部量级）、`source_z` 与 `source_x` 全梯度的 FD 一致性、`outgoing=False` 的共轭关系。
5. **用变异审计反过来验收**：重跑
   `python3 -m tensorbem.benchmarks.mutation_audit --module tensorbem.hydrostatic --module tensorbem.free_surface --extra-test tests/test_hydrostatic_waterplane_gaps.py --extra-test tests/test_free_surface_smooth_depth_derivative.py`
   （证据落 `outputs/mutation_backfill/mutation_report.{json,md}`）；对未被该次采样抽中的变异体，
   再用影子副本定点植入单点缺陷、跑新测试文件确认被判死。
6. **证据与图**：`PYTHONPATH=src python3 examples/backfill_guardrail_evidence.py` 生成
   `outputs/mutation_backfill/guardrail_evidence.json` 与 `guardrail_evidence.png`（数字全部程序化注入）。

---

## 3. 计算结果

### 3.1 偏心水线面 vs 解析（程序化注入，零手抄）

| 量 | 实测 | 解析 | 绝对/相对误差 |
|---|---|---|---|
| `A_wp` | 2.000000000000000 | 2.0 | 0.0 |
| 形心 `x_c` | 1.000000000000000 | 1.0 | 0.0 |
| 形心 `y_c` | 0.500000000000000 | 0.5 | 0.0 |
| `I_xx` | 0.166666666666667 | 1/6 | 1.67e-16 |
| `I_yy` | 0.666666666666667 | 2/3 | 1.67e-16 |
| `I_xy` | 0.0 | 0.0 | 0.0 |

（`outputs/mutation_backfill/guardrail_evidence.json → eccentric_waterplane`）

### 3.2 `dG/dA` 的解析值 vs 中心差分

| 差分步长 `h` [m] | 相对误差 |
|---|---|
| 1e-3 | 见 JSON（order ≈ 2） |
| 1e-4 | 同上 |
| 1e-5 | 同上 |
| 1e-6 | **2.45e-11**（本次取到的最小误差） |

最小相对误差 `2.45e-11`，双对数斜率 ≈ 2（图右）：差分误差随 `h²` 下降，说明解析导数与数值导数一致到
差分格式的截断阶，而非「碰巧同量级」。

### 3.3 变异体状态（本次 vs 审计）

| 变异体 | 审计时 | 本次 | 判死它的用例 |
|---|---|---|---|
| `hydrostatic.py:236` `x_moment` `/6 → *6` | survived | **killed** | `test_centroid_and_second_moments_of_an_offset_barge` |
| `hydrostatic.py:314` `np.zeros(3) → np.zeros(4)` | survived | **killed** | `test_zero_volume_hull_hits_the_volume_sentinel` |
| `hydrostatic.py:255` 退化返回 `0.0 → 1.0` | survived | **killed**（影子副本定点验证） | `test_sub_threshold_waterline_loop_hits_the_area_sentinel` |
| `hydrostatic.py:161` 空水线早退 `0.0 → 1.0` | survived | **killed** | `test_fully_submerged_body_has_no_waterplane` |
| `free_surface` `_depth_derivative` 的 `2.0 * → 2.0 /` | survived（原为死代码 `df_d_depth`） | **killed**（影子副本定点验证：10 个用例中 6 个失败；未变异时全绿） | `test_smooth_depth_derivative_matches_a_source_depth_finite_difference` 等 |

**关于总分的诚实说明**：两次审计运行的 mutation score 不可横向比较（`hydrostatic` 一次 0.75、一次 0.583），
因为 harness 按 `max-per-module` 从生成的变异体集中**抽样**，两次抽到的子集不同。本单只主张
**逐条变异体的状态**，不主张总分提升。

---

## 4. 结果图

![guardrail evidence](outputs/mutation_backfill/guardrail_evidence.png)

左：偏移 `(1.0, 0.5) m` 的矩形水线面（蓝）与实测形心（红圈）叠在解析形心（黑十字）上，二者重合到机器精度；
若 `x_moment` 归一化被互换（审计登记的那条存活变异），形心会偏离 36 倍，图上会偏出画面。
右：`dG/dA` 的解析值与中心差分之差随步长的双对数曲线，参考线斜率 2。

---

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **退化分支不能用「合法网格」直接构造**：`WettedHullMesh` 强制正排水体积、禁止退化三角形，
   且默认水线容差 `1e-12` 会把 `3e-14 m` 的吃水直接吸附到水面。`hydrostatic.py:314` 的
   `|V| < 1e-14` 分支因此改用**鸭子类型**的最小网格对象触发（同样的 `vertices`/`faces` 接口，
   功能上正是「塌缩网格」这一数值现场）；`hydrostatic.py:255` 分支则用**面积小于阈值的真实水线环**
   （近重合的双水线边）触发，仍是真实几何输入。
2. **`hydrostatic.py:255` 与 `161` 是两个不同的哨兵**：审计只登记了 255，但空水线早退（161）同样返回
   三元组；补测试时两条都要钉，否则「全潜没」算例只覆盖前者。
3. **重构是等价重排，不是精度提升**：本次没有改变任何水动力系数的数值，`dg_d_source_z` 与旧表达式
   逐项相同；本单交付的是**判别力**（护栏）与**可维护性**（单一导数来源），不是精度改进。
4. **仍有存活变异体，未粉饰**：本次采样中 `free_surface.py:98`（阈值 `<`→`<=`）判为等价变异体；
   `free_surface.py:118`、`143`（`_pv_cosine`/`_pv_sine` 内部近似项）与 `287`（`green_gradient` 里
   `residue` 的指数项）本次仍存活——它们落在主值积分的近似核与留数装配上，需要**独立参考积分**
   而不是有限差分才能钉死，登记为下一单线索。
5. 适用范围不变：线性势流、常水深、深水核 `ν·depth ≳ 1`（`_check_scaled_depth` 保护）；静水力为小倾角假设。

---

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 偏心水线面面积 | 解析 `L·B` | 0.0（相对） | **是** |
| 水线面形心 `(x_c, y_c)` | 解析 `(1.0, 0.5)` | 0.0 m（绝对） | **是** |
| `I_xx`, `I_yy` | 解析 `L B³/12`, `B L³/12` | 1.67e-16（相对） | **是** |
| `I_xy` | 0 | 0.0 | **是**（解析上为 0） |
| `dG/dA` 解析 vs 中心差分 | 自身核的中心差分（`h=1e-6 m`） | 2.45e-11（相对） | **是** |
| 复原矩阵平移不变性 | 平移前后自比 | ≤1e-12（相对，`np.allclose`） | **是** |

以上均为**代码验证**（对闭合解/数值微分），不是实验标定；水动力系数精度不在本单主张范围内。

---

## 7. 交付物与复现

```bash
PYTHONPATH=src python3 -m pytest tests/test_hydrostatic_waterplane_gaps.py \
    tests/test_free_surface_smooth_depth_derivative.py -q
PYTHONPATH=src python3 -m tensorbem.benchmarks.mutation_audit \
    --module tensorbem.hydrostatic --module tensorbem.free_surface \
    --extra-test tests/test_hydrostatic_waterplane_gaps.py \
    --extra-test tests/test_free_surface_smooth_depth_derivative.py \
    --out outputs/mutation_backfill
PYTHONPATH=src python3 examples/backfill_guardrail_evidence.py
```

| 文件 | 作用 |
|---|---|
| `src/tensorbem/free_surface.py` | `_depth_derivative` + `smooth_depth_derivative`；`dg_d_depth` 进入返回值，删除死代码 `df_d_depth`/`dresidue_d_depth` |
| `tests/test_hydrostatic_waterplane_gaps.py` | 8 个用例：偏心水线面、平移不变性、三条退化哨兵 |
| `tests/test_free_surface_smooth_depth_derivative.py` | 10 个用例：深度导数 FD 护栏、非平凡性、全梯度一致性、共轭 |
| `examples/backfill_guardrail_evidence.py` | 生成 JSON 证据与图，数字零手抄 |
| `outputs/mutation_backfill/` | 变异审计载荷、证据 JSON、`guardrail_evidence.png` |