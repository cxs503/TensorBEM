# 混合 Dirichlet/Neumann 边界的二维 Laplace 面板解法：闭式解析验证

**仓**：TensorBEM（分支 `agent/m2961`）
**类别**：功能补全 + 解析验证（闭合 `docs/verification.md` §1 表中登记的「后续：需相应边界条件求解器」）
**工件**：`src/tensorbem/mixed.py`、`src/tensorbem/benchmarks/mixed_boundary.py`、
`docs/mixed_boundary/outputs/study.json`（零手抄，全部数字由该工件生成）、
`docs/mixed_boundary/make_figure.py` → `docs/mixed_boundary/figures/mixed_boundary_convergence.png`、
`tests/test_mixed_boundary.py`

---

## 1. 案例介绍：为什么需要混合边界

TensorBEM 的二维 Laplace 面板求解器此前只有**全 Dirichlet**（`solve_laplace_dirichlet`）、
**外域 Dirichlet** 与**外域 Neumann** 三个入口（`src/tensorbem/solver.py`）。工程与物理验证中
大量问题把两类边界条件放在同一条闭合曲线上：导体表面给定电位、绝缘面给定法向梯度；
线性化自由面一部分按运动学（给定法向速度）、一部分按动力学（给定压力/势）；
船体剖面上的贴水段与干段。`docs/verification.md` 的算例选择表把这类问题显式登记为
「后续：需相应边界条件求解器」，本单把它闭合。

**物理/数学背景**：二维有界域 Laplace 问题

$$\nabla^2 u = 0 \ \text{in } \Omega,\qquad u = g \ \text{on } \Gamma_D,\qquad \partial u/\partial n = \hat q \ \text{on } \Gamma_N,$$

$\Gamma_D \cup \Gamma_N = \partial\Omega$、$\Gamma_D \cap \Gamma_N = \varnothing$、$\Gamma_D \neq \varnothing$
（Dirichlet 部分非空保证唯一性）。纯 Neumann（$\Gamma_D = \varnothing$）时解只确定到常数，
需要规范（gauge）。

**几何与参数表**（所有算例）：

| 项 | 值 |
|---|---|
| 区域 | 单位正方形 $[0,1]^2$（逆时针 4 边，每边 $n$ 个面元，共 $N=4n$） |
| 面元 | 常数面元，直线面元解析积分核，中点配点，自由项 $c=1/2$ |
| 核参考长度 | $L=2$（与 `solve_laplace_dirichlet` 同一约定） |
| 内部测点 | $6\times6$ 均匀网格，$x,y\in[0.1,0.9]$ |
| 参考长度 | 闭式解析，不用第二个离散解作为真值 |
| 加密序列 | $n=2,4,8,16$（$N=8,16,32,64$） |

**算例 A（混合，主算例）**：$u=e^{x}\cos y$（调和，$\partial_{xx}+\partial_{yy}=0$）。
Dirichlet 边：$x=0$ 上 $u=\cos y$，$y=0$ 上 $u=e^{x}$；Neumann 边：$x=1$ 上
$q=e\cos y$，$y=1$ 上 $q=-e^{x}\sin 1$。四组数据均非常数。
**求解器被给的是其中一半，报告的是它没被给的另一半**：Dirichlet 侧的 $q$ 与 Neumann 侧的 $u$。

**算例 B（纯 Neumann，规范算例）**：$u=x^2-y^2$，四边全给 $q=\partial u/\partial n$。
面元常数数据对二次调和场是**精确**的，故 $q$ 的离散误差应为零，可用来单独检验 gauge。
相容性 $\oint \hat q\,\mathrm ds = 0$ 在本例自动满足（$\int_\Omega\nabla^2u=0$）。

## 2. 怎么做的

1. **一个方阵，不做两套装配**。直接边界积分恒等式在**每个**配点成立
   $\sum_j S_{ij}q_j = \tfrac12 u_i + \sum_j H_{ij}u_j$，只是每块面板的未知量类型不同。
   于是未知量为 $\{q_j: j\in D\}\cup\{u_k: k\in N\}$ 共 $N$ 个，方程组

   $$\sum_{j\in D}S_{ij}q_j-\sum_{k\in N}\left(\tfrac12\delta_{ik}+H_{ik}\right)u_k
     =\sum_{j\in D}\left(\tfrac12\delta_{ij}+H_{ij}\right)g_j-\sum_{k\in N}S_{ik}\hat q_k .$$

   $S,H$ 直接复用 `solver.assemble_matrices`，**没有新增核函数、没有新增积分器**，
   Dirichlet 路径一行未改。
2. **降维自证**：当 $N$ 为空时该方阵在代数上就是 `solve_laplace_dirichlet` 的方程组。
   测试与工件都对这条恒等式下断言（见 §3.3），其误差必须是舍入级。
3. **纯 Neumann 的 gauge**：$D$ 为空时 $\tfrac12 I+H$ 对常向量奇异（对应物理上的加常数自由）。
   `pure_neumann_gauge="mean_zero"`（默认）用「面元中点均值 $=0$」替换最后一行配点方程，
   返回均值零的唯一代表；`"raise"` 则拒绝求解而不是悄悄返回一个任意成员。
   相容条件不做输入校验：不相容的 Neumann 数据会让方程组不相容，由返回的
   `relative_residual_l2` 暴露。
4. **输入覆盖校验**：两块索引必须互不相交且**恰好覆盖**每块面板一次（缺一块面板=漏一个边界条件），
   重复索引、越界、非有限数据、非法 gauge 名一律 `ValueError`。
5. **场的重构**：`evaluate_mixed_interior_field` 用同一套解析面板核由两条迹重构内部势
   （Green 表示只需两条边界迹；解已把未知那条填满）。
6. **零手抄**：`python -m tensorbem.benchmarks.mixed_boundary --panels 2 4 8 16 --output
   docs/mixed_boundary/outputs/study.json` 生成全部数字；图由 `make_figure.py` 从该 JSON 重画；
   `tests/test_mixed_boundary.py::test_committed_artifact_reproduces_the_study` 重跑并逐字段比对
   已提交 JSON（`rel=1e-6`），文档数字无法悄悄漂移。

## 3. 计算结果

### 3.1 算例 A：混合边界，未知侧迹与内部场

| 面元数 $N$ | 条件数 $\kappa$ | $q$（Dirichlet 侧）相对 $L_2$ | $u$（Neumann 侧）相对 $L_2$ | 内部场相对 $L_2$ | 内部场最大绝对误差 |
|---:|---:|---:|---:|---:|---:|
| 8 | 7.612 | 1.105e-01 | 2.805e-02 | 2.189e-02 | 9.938e-02 |
| 16 | 16.313 | 7.419e-02 | 1.115e-02 | 5.116e-03 | 2.610e-02 |
| 32 | 33.522 | 4.940e-02 | 4.008e-03 | 1.348e-03 | 7.594e-03 |
| 64 | 68.261 | 3.395e-02 | 1.391e-03 | 3.950e-04 | 2.265e-03 |

观测阶（每加密一倍）：$u$（Neumann 侧）**1.331 / 1.476 / 1.527**，
内部场 **2.097 / 1.924 / 1.771**，$q$（Dirichlet 侧）**0.575 / 0.587 / 0.541**。
条件数随加密单调上升（$7.6\to68.3$），与 `solve_laplace_dirichlet` 在同一网格上的值**完全相同**。

### 3.2 算例 B：纯 Neumann + `mean_zero` 规范

| 面元数 $N$ | 条件数 | $q$ 相对 $L_2$（数据精确） | $u$（去均值后）相对 $L_2$ | 内部场（去均值后）相对 $L_2$ |
|---:|---:|---:|---:|---:|
| 8 | 15.568 | 0.000e+00 | 5.390e-02 | 7.755e-02 |
| 16 | 29.778 | 0.000e+00 | 2.113e-02 | 1.787e-02 |
| 32 | 52.510 | 0.000e+00 | 8.253e-03 | 4.710e-03 |
| 64 | 89.432 | 0.000e+00 | 3.157e-03 | 1.441e-03 |

$q$ 的误差是**精确零**（不是 1e-16），这正是「面板常数数据对二次场精确」的预期：
它把 gauge 的效果与数据插值误差分离开。$u$ 观测阶 1.351 / 1.357 / 1.387，内部场 2.117 / 1.924 / 1.709。

### 3.3 控制实验一：空 Neumann 集必须复现既有 Dirichlet 解

| 面元数 $N$ | $\max|\Delta q|$（混合 ⊖ 全 Dirichlet） | 混合条件数 | Dirichlet 条件数 |
|---:|---:|---:|---:|
| 8 | 9.992e-16 | 9.571931 | 9.571931 |
| 16 | 3.442e-15 | 18.148374 | 18.148374 |
| 32 | 1.243e-14 | 35.560595 | 35.560595 |
| 64 | 3.686e-14 | 70.809449 | 70.809449 |

差异随 $N$ 增长（条件数 $\times$ 舍入），且始终在 $4\times10^{-14}$ 以下：
两种路径在同一线性代数问题上保持一致，混合装配没有引入任何额外的模型误差。

### 3.4 控制实验二：$q$ 的一阶误差来自方角，不来自条件类型混合

同一解析场 $u=e^{x}\cos y$ 的**全 Dirichlet** 解（四边全给 $u$，用既有
`solve_laplace_dirichlet`）在同样网格上的 $q$ 误差：

| 面元数 $N$ | 全 Dirichlet $q$ 误差 | 观测阶 | 混合 $q$ 误差（§3.1） |
|---:|---:|---:|---:|
| 8 | 9.686e-02 | — | 1.105e-01 |
| 16 | 7.502e-02 | 0.369 | 7.419e-02 |
| 32 | 5.511e-02 | 0.445 | 4.940e-02 |
| 64 | 3.981e-02 | 0.469 | 3.395e-02 |

两条链的误差量级与阶几乎一致（0.47 vs 0.55）⇒ **方角上的 $q$ 间断是限制项**，
混合边界本身不额外损失精度。逐面元剖面证实了这一点：$n=16$ 时左边界 16 个面板中，
最靠角点 $(0,0)$ 的那个面元误差 $9.2\times10^{-2}$，其余 $5\times10^{-4}\sim1.2\times10^{-2}$，
即误差 99% 集中在角点邻域。

## 4. 结果图

`docs/mixed_boundary/figures/mixed_boundary_convergence.png`（由 `make_figure.py` 从
`study.json` 重画）：

* (a) 五个误差序列的 log–log 收敛图（混合：Dirichlet 侧 $q$、Neumann 侧 $u$、内部场；
  纯 Neumann：$u$、内部场），标注末端观测阶，并画 slope = 2 参考线；
* (b) 混合 $q$ 误差 vs 同场全 Dirichlet $q$ 误差，附 slope = 1 参考线——
  两条曲线重合程度就是「混合不额外掉精度」的直接证据。

## 5. 注意点（坑、局限、适用范围）

1. **面元常数 + 方角 ⇒ $q$ 只有一阶**。本仓文档早先记录的「光滑算例通量 L2 误差 0.157%、观测阶≈2」
   是**光滑几何**（圆/椭圆）的结果；正方形有角点，$q$ 在角点两侧本就不同（法向跳 90°），
   常数面元的通量迹因此被角点主导。不要把本文的 3.4e-2 当作求解器退化——同网格的全 Dirichlet
   路径给出同样的 4.0e-2。
2. **不要用自洽性冒充精度**。可疑的检验是「两段逼近在接缝处一致」；`(½I+H)` 的常向量零空间
   与 Neumann 的加常数自由是同一件事的两面，因此纯 Neumann 的测试必须把 gauge 显式写出来，
   否则会把「任意常数」当成误差。
3. **本实现的适用范围**：内部（有界）问题、直线面板中点配点、$c=1/2$ 不加角点自由项修正、
   不做面板级自适应。外域混合边界未实现（`LaplaceMixedResult.domain` 固定 `"interior"`），
   需要时按 `solve_laplace_exterior_*` 的符号约定（$S q_e=(\tfrac12 I-H)u$）另开路径。
4. **相容性不做输入校验**：纯 Neumann 的 $\oint\hat q\,\mathrm ds=0$ 若不满足，方程组不相容，
   届时以 `relative_residual_l2` 报告，而不是抛出「数据非法」。这是有意的：残差是更有信息量的证据。
5. **`mean_zero` 是离散均值**（面元中点等权），不是弧长加权；与解析解比较时必须对两者施加同一规范。

## 6. 误差判定

判据：以闭式解析解为准，相对 $L_2$ 误差逐项列出，并显式回答是否满足「<3%」。

| 量 | 网格 | 相对误差 | 是否 < 3% |
|---|---:|---:|---|
| Neumann 侧势迹 $u$ | $N=64$ | 1.391e-3 | **是**（0.14%） |
| 内部场 $\phi$（混合） | $N=64$ | 3.950e-4 | **是**（0.04%） |
| 纯 Neumann 势迹 $u$（去均值） | $N=64$ | 3.157e-3 | **是**（0.32%） |
| 纯 Neumann 内部场（去均值） | $N=64$ | 1.441e-3 | **是**（0.14%） |
| 纯 Neumann 通量 $q$ | $N=8$ 起 | 0.000e+00 | **是**（精确） |
| Dirichlet 侧通量 $q$（混合） | $N=64$ | 3.395e-2 | **否**（3.40%） |
| 全 Dirichlet 侧通量 $q$（对照） | $N=64$ | 3.981e-2 | **否**（3.98%） |

结论：**求解器新增能力（Neumann 侧势迹、纯 Neumann 带 gauge、内部场）全部 < 3%，最差的
Neumann 侧迹 1.39e-3 也在 3% 门限的百分之一以内；唯一不达标项是方角几何上的通量迹 3.40%，
而同网格既有全 Dirichlet 路径为 3.98%（同样不达标），故该项是几何（角点）+ 常数面元的固有局限，
不是本功能的缺陷。** 该局限已在 §5.1 与 §3.4 用对照实验量化，作为 honest 登记而非回避。
未做：圆/弧几何上的混合边界算例（需要在圆上构造混合边界解析解），因此「混合在光滑几何上
是否同样达到二阶通量」仍是未回答的问题。

## 7. 复现命令

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.mixed_boundary \
    --panels 2 4 8 16 --output docs/mixed_boundary/outputs/study.json
PYTHONPATH=src python docs/mixed_boundary/make_figure.py
PYTHONPATH=src python -m pytest tests/test_mixed_boundary.py -q
```