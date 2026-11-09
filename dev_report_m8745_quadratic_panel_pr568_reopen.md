# TensorBEM 二次曲面单元算子验证报告（PR#568 重开·管到底第1轮·m8745）

**Task**: m8745（重开 TensorBEM#568，管到底第1轮）· **Branch**: `agent/m8745` · **Repo**: TensorBEM
**Date**: 2026-11-09
**Status**: ✓ 功能要点已在 main，本 PR 补算子刚体旋转不变性回归钉 + 全套测试复验

## 1. 案例介绍

**物理背景**：三维 Laplace 外问题 Dirichlet 边界条件，单位球面（icosphere 二次 6 节点曲面单元），内部点源 `x0=(0,0,0.5)`。

**精确解**：
- 势函数 `u(x) = 1/(4π|x-x0|)`
- 通量 `q(x) = -(x-x0)·n/(4π|x-x0|³)`

**几何离散**：icosphere 二次曲面单元（3 顶点 + 3 边中点），3 个加密级别：

| 级别 | 节点数 | 单元数 | 特征尺寸 h |
|---|---|---|---|
| 0 | 42 | 20 | 0.6919 |
| 1 | 162 | 80 | 0.3819 |
| 2 | 642 | 320 | 0.1963 |

**数值设置**：非奇异 Gauss 8 阶；奇异自影响 Duffy 16 阶；直接第二类 BIE `(1/2 I - D)u = -S q`。
**验收标准**：通量相对 L2 误差 < 3%（硬限制 15%）。

## 2. 怎么做的

**功能要点核对**（PR#568 = m7526 成果，owner 2026-11-05 裁定关闭原 PR）：

| 文件 | PR#568 声明 | main 现状 | 判定 |
|---|---|---|---|
| `src/tensorbem/panel_quadratic.py` | +244/-100 | 1059 行（m8282 修复 + m8616 椭球 + m8711 自由项） | ✓ 已入 main |
| `src/tensorbem/benchmarks/quadratic_panel_verification.py` | +300 | 934+ 行（v2 schema + 自由项研究） | ✓ 已入 main |
| `tests/test_panel_quadratic.py` | +32/-2 | 306 行 | ✓ 已入 main |
| `tests/test_panel_quadratic_operator.py` | +204 | 580 行 | ✓ 已入 main |
| `docs/quadratic_panel_verification/study.json` | +169 | v2 schema（含 free_term 节） | ✓ 已入 main |
| `docs/quadratic_panel_verification/fig_quadratic_operator.png` | +0/-0 | 305 KB，sha256 入 provenance | ✓ 已入 main |

**承接链**：m7526 原实现（commit 6390ed0，未入 main）→ m8282 修复 4 个结构性缺陷经 PR#550 入 main → m8616 椭球几何 + 相似不变性（PR#558）→ m8711 自由项研究（commit 77643fa）。PR#568 的全部功能要点由上述提交完整覆盖并超越。

**本 PR 增量**（单一功能点：算子刚体旋转不变性回归钉）：
1. 新增 `test_operators_are_rotation_invariant`：把 m8723 的平移不变性钉推进到**完整刚体群**——椭球 1.0/1.6/0.7（sub 1，162 节点）绕轴 (1,2,3)/√14 旋转 0.7 rad 后，`S` 与 `D` 的每个元素必须不变（容差 1e-10 相对）。Laplace 核 `G(Rx,Ry)=G(x,y)` 且等参法向协变（`n→Rn`），故 S/D 对任意正交旋转逐元素不变。平移钉抓不住「引用固定全局轴」的符号/链式法则错误（如法向或 Duffy 子坐标混入绝对坐标），此钉可以。
2. 全套测试复验（见 §5）。

**热文件声明**：本 PR 仅改 `tests/test_panel_quadratic_operator.py`（测试面，非核心模块）。开 PR 前查 GitHub `cxs503/TensorBEM` open PR = 0，无重叠。

## 3. 计算结果（程序化注入，零手抄，源 `docs/quadratic_panel_verification/study.json`）

### 3.1 通量收敛（单位球，点源）

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

### 3.3 本 PR 新增：S/D 刚体运动不变性（椭球 1.0/1.6/0.7，162 节点）

| 变换 | max\|ΔS\|/max\|S\| | max\|ΔD\|/max\|D\| | 容差 | 判定 |
|---|---|---|---|---|
| 平移 (3,-2,1.5)（m8723 钉） | 2.486e-15 | 5.411e-12 | 1e-10 | ✓ |
| 旋转 0.7 rad（本 PR 钉） | 5.524e-16 | 1.117e-12 | 1e-10 | ✓ |

旋转后固体角恒等式 `\|D·1+1/2\|∞ = 1.173e-02 < 2e-2` ✓。

## 4. 误差判定（逐项）

| 关键量 | 实测 | 基准 | 是否 <3% |
|---|---|---|---|
| 通量相对 L2 误差（最细级） | 6.065e-03 | 3% 验收线 | **是** |
| 椭球几何门 | passed=true | study.json | **是** |
| 旋转不变性残差（S/D） | ≤1.117e-12 | 1e-10 容差 | **是**（低 2 个量级） |
| 平移不变性残差（S/D） | ≤5.411e-12 | 1e-10 容差 | **是** |

## 5. 全套测试复验

- **本地（x86, 99）**：`tests/test_panel_quadratic*.py` 53/53 通过（88.8 s，含本 PR 新钉）。
- **集群（loong64 n315，8 workers，stage_dir=本工作树）**：227 failed / 1994 passed / 30 skipped / 2 xfailed / 79 errors（33:45）。**二次曲面单元相关测试零失败**（唯一含 "quadratic" 的失败是 `test_helmholtz_maccamy_fuchs` 的收敛阶测试，与 panel_quadratic 模块无关）。失败主因：156× `LinAlgError: Eigenvalues did not converge`（loong64 LAPACK 在 3 个文件的 import 期特征值不收敛，架构相关）+ 作业路径 FileNotFoundError + 数值漂移判词；均不在本 PR 触碰的文件内（本 PR diff = 1 个测试文件）。
- **预存失败留痕**：`test_bem_finite_depth_anchors::test_case_artifacts_and_documentation_exist`（figures/ 目录从未入库）在干净 main 上经 `git stash` 对照复现，与本 PR 无关。

## 6. 注意点

- 本机 `tensorbem` 包安装指向 m8648 工作树——所有本地运行必须 `PYTHONPATH=/root/tp-agent/work/m8745/src`，否则测的是别的树的代码。
- 集群 loong64 上 xdist 收集不稳定（模块级特征值求解 flaky），主套件用 `--dist loadfile -n 8` 并对 3 个 flaky 文件单独串行。
- 本 PR 不触碰 `study.json`/已提交图（其 sha256 被 `test_committed_study_json_matches_its_artifacts` 钉住）。
