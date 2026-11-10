# TensorBEM 二次曲面单元算子验证报告（PR#574 重开·管到底第1轮·m8788）

**Task**: m8788（重开 TensorBEM#574，管到底第1轮）· **Branch**: `agent/m8788` · **Repo**: TensorBEM
**Date**: 2026-11-09
**Status**: ✓ 功能要点已在 main；本 PR 补算子**镜像反射不变性**回归钉 + 修复其暴露的**外法向约定缺陷**（核心模块，已声明）

---

## 1. 案例介绍

**物理背景**：三维 Laplace 外问题 Dirichlet 边界条件，椭球 `axes=(1.0, 1.6, 0.7)`（icosphere 二次 6 节点曲面单元，sub 1，162 节点 / 80 单元），内部点源。

**精确解**：`u(x) = 1/(4π|x-x0|)`，`q(x) = -(x-x0)·n/(4π|x-x0|³)`。

**数值设置**：非奇异 Gauss 8 阶；奇异自影响 Duffy 16 阶；直接第二类 BIE `(1/2 I - D)u = -S q`。
**验收标准**：通量相对 L2 误差 < 3%（硬限制 15%）；算子恒等式（固体角 `D 1 = -1/2`、`S 1 = 1`）。

## 2. 怎么做的

### 2.1 功能要点核对（PR#574 = m8282 matured mission deliverables，owner 2026-11-05 裁定关闭原 PR）

| 文件 | PR#574 声明 | main 现状 | 判定 |
|---|---|---|---|
| `src/tensorbem/panel_quadratic.py` | +432/-83 | 1059 行（m8282 四缺陷修复 + m8616 椭球 + m8711 自由项） | ✓ 已入 main |
| `src/tensorbem/benchmarks/quadratic_panel_verification.py` | +483/-69 | 934 行（v2 schema + 自由项/相似性研究） | ✓ 已入 main |
| `tests/test_panel_quadratic.py` | +10/-2 | 306 行 | ✓ 已入 main |
| `tests/test_panel_quadratic_operator.py` | +374/-0 | 656 行（+本 PR 2 钉） | ✓ 已入 main |
| `docs/quadratic_panel_verification/study.json` | +114/-0 | v2 schema，sha256 入 provenance | ✓ 已入 main |
| `docs/quadratic_panel_verification/fig_quadratic_operator.png` | +0/-0 | 305 KB，sha256 入 provenance | ✓ 已入 main |
| `dev_report_m8282_quadratic_operator.md` | +135/-0 | 在仓根 | ✓ 已入 main |

**承接链**：m7526 原实现（commit 6390ed0，未入 main）→ m8282 修复 4 个结构性缺陷经 PR#550 入 main → m8616 椭球几何 + 相似不变性（PR#558）→ m8711 自由项研究 → m8723 平移不变性钉（PR#572）→ m8745 旋转不变性钉（PR#577）→ m8769 均匀缩放不变性钉。PR#574 的全部功能要点由上述提交完整覆盖并超越。

### 2.2 本 PR 增量（单一功能点：O(3) 全群不变性收尾 + 外法向约定强制）

1. **新钉 `test_operators_are_reflection_invariant`**：把 m8723 平移 + m8745 旋转（刚体群 SE(3)）+ m8769 均匀缩放（相似群）推进到**全正交群 O(3)**——椭球整体镜像 `M = diag(1,-1,1)`（det = -1）后：
   - 单层算子 `S` 必须逐元素不变（核 `G(x,y)=1/(4π|x-y|)` 对任意正交 M 满足 `G(Mx,My)=G(x,y)`，且 `S` 不含法向）；
   - 双层算子 `D` 必须逐元素不变（反射闭曲面的外法向为 `M n`，固体角与定向约定一致时与变换无关）；
   - 容差 1e-10 相对（逐元素）+ 反射面上固体角恒等式 `< 2e-2`。
   - 平移/旋转/缩放钉全部是**定向保持**变换（det = +1），抓不住「法向由叉积按右手系参数化假设构造」的手性错误——镜像（det = -1）把它暴露为 `D -> -D` 而 `S` 不动。
2. **新钉 `test_inward_wound_mesh_is_rewound_outward`**（白盒伴生钉）：把合法网格所有面反向后构造，构造器必须把存面恢复为原外绕序（逐面精确相等）。
3. **缺陷修复（核心模块 `src/tensorbem/panel_quadratic.py`，热文件，见 §5 声明）**：`QuadraticSurfaceMesh.__post_init__` 现在强制外法向约定——等参法向 = `dx/dL1 × dx/dL2`，方向随面绕序；对闭星形面，用 `(单元中心 - 顶点质心)` 近似外方向，凡平面法向指向质心一侧的面按 `(a,b,c,m_ab,m_bc,m_ca) -> (a,c,b,m_ca,m_bc,m_ab)` 重绕（v1↔v2，边中点随边走）。模块 docstring 承诺 "outward unit normals"，但原代码不保证：镜像网格会**静默**翻转 `D` 与恢复通量（`D 1 = +1/2`），而原网格上所有恒等式照常通过——与 m8282 修掉的符号约定缺陷同族。
   - **无操作性证明**：对全部内建网格（icosphere/椭球/立方体/平移/旋转/缩放副本）重绕判定全为「不重绕」（点积最小值 +0.30 > 0）；修复后重跑 benchmark，`study.json` 与提交版 **0 标量差**、两图 sha256 与 provenance 一致、`git status docs/` 干净。

### 2.3 热文件声明

- 本 PR 改 `src/tensorbem/panel_quadratic.py`（核心模块）+ `tests/test_panel_quadratic_operator.py`（测试面）。
- 开 PR 前查 GitHub `cxs503/TensorBEM` open PR = 1（#592, agent/m8745，改 `tests/test_panel_quadratic_operator.py` +43/-0）。**重叠声明**：#592 的增量（旋转不变性钉）已在 main（本 PR 基线即含该测试），#592 为陈旧重复 PR；本 PR 的新测试追加在缩放钉之后，与 #592 的插入点（平移钉之后）不同，文本冲突风险低；若 #592 先合并，本 PR 可 rebase 后无冲突落地。
- 另注：并行工作树 m8648 的 editable install 指向其工作树（`/root/tp-agent/work/m8648/src`），本单全部测试/基准均以 `PYTHONPATH=src` 强制导入本工作树代码（`tensorbem.__file__` 已验证指向 m8788）。

## 3. 计算结果（程序化注入，零手抄，源 `docs/quadratic_panel_verification/study.json`）

### 3.1 通量收敛（单位球，点源，修复前后一致）

| 级别 | 节点数 | h | 通量相对 L2 误差 | cond(S) | 残差 L2 |
|---|---|---|---|---|---|
| 0 | 42 | 0.6919 | 1.404e-01 | 20.8 | 2.99e-16 |
| 1 | 162 | 0.3819 | 2.854e-02 | 39.4 | 4.65e-16 |
| 2 | 642 | 0.1963 | **6.065e-03** | 80.8 | 8.62e-16 |

### 3.2 收敛阶与恒等式

| 量 | 实测 | 判定 |
|---|---|---|
| 通量误差观测阶 | 2.49 | O(h²)~O(h³) ✓ |
| 最细级通量误差 | 6.065e-03 | < 3% ✓（`acceptance.passed=true`） |
| 椭球几何门 | passed=true | ✓ |
| 相似不变性门 | passed=true | ✓ |

### 3.3 本 PR 新增：S/D 镜像反射不变性（椭球 1.0/1.6/0.7，162 节点，M = diag(1,-1,1)）

| 量 | 修复前 | 修复后 | 容差 | 判定 |
|---|---|---|---|---|
| max\|S₁ − S₀\| | 0.0 | 0.0 | 1e-10 相对 | ✓ |
| max\|D₁ − D₀\| | **0.1279**（D₁ = −D₀ 精确） | 0.0 | 1e-10 相对 | ✓ |
| 反射面 \|D₁·1 + 1/2\|∞ | **0.998**（= +1/2，内法向） | 1.173e-02 | 2e-2 | ✓ |
| 全反向网格重绕 | 不重绕（D 反号） | 存面 == 原外绕序（精确） | 精确 | ✓ |

## 4. 全套测试

- `PYTHONPATH=src python -m pytest tests/`（本工作树代码）：见 §5 汇总。
- 已知 main 既有失败（与本 PR 无关，clean main 复现）：`tests/test_bem_finite_depth_anchors.py::test_case_artifacts_and_documentation_exist`（`benchmarks/verified/bem_finite_depth/figures/` 目录从未入库，git ls-files 证实）。

## 5. 注意点与局限

- 重绕启发式假设表面对顶点质心**星形**（全部内建生成器满足）；非星形面（如环面）可能误判——原代码无任何定向处理，本修复为严格改进，局限已写入 docstring。
- 反射钉的 1e-10 相对容差与平移/旋转/缩放钉一致；固体角 2e-2 容差沿用既有椭球钉（sub 1 离散水平）。
- 本 PR 不改任何已提交数字：study.json/两图字节级不变（§2.2 无操作性证明）。
