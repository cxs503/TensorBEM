# TensorBEM 二次曲面单元算子验证报告（PR#546 重开·管到底第3轮·m8723）

**Task**: m8723（重开 TensorBEM#546，管到底第3轮）· **Branch**: `agent/m8723` · **Repo**: TensorBEM
**Date**: 2026-11-09
**Status**: ✓ 功能要点已在 main，本 PR 补算子平移不变性回归钉 + 全套测试复验

---

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

**功能要点核对**（PR#546 = m7526 成果，merge_master 补开 PR，owner 2026-11-05 裁定关闭）：

| 文件 | PR#546 声明 | main 现状 | 判定 |
|---|---|---|---|
| `src/tensorbem/panel_quadratic.py` | +244/-100 | 1059 行（含 m8282 修复 + m8616 椭球扩展 + `__all__`） | ✓ 已入 main |
| `src/tensorbem/benchmarks/quadratic_panel_verification.py` | +300 | 934 行（v2 schema，含几何/相似/薄体研究） | ✓ 已入 main |
| `tests/test_panel_quadratic.py` | +32/-2 | 306 行 | ✓ 已入 main |
| `tests/test_panel_quadratic_operator.py` | +204 | 554 行 | ✓ 已入 main |
| `docs/quadratic_panel_verification/study.json` | +169 | v2 schema，314 行 | ✓ 已入 main |
| `docs/quadratic_panel_verification/fig_quadratic_operator.png` | +0/-0 | 305 KB，sha256 入 provenance | ✓ 已入 main |

**承接链**：m7526 原实现（commit 6390ed0，未入 main）→ m8282 修复 4 个结构性缺陷（边中点梯度符号 / D 装配为零 / 边中点奇异求积 / 等效源密度混淆）经 PR#550 入 main（commit 553b28a）→ m8616 加椭球几何 + 相似不变性 + 薄体范围限（PR#558，commit 6d60ba3）。PR#546 的全部功能要点由 PR#550 + PR#558 完整覆盖并超越。

**本 PR 增量**（单一功能点：算子刚体平移不变性回归钉）：
1. 新增 `test_operators_are_translation_invariant`：把 S/D 矩阵从「行和恒等式」（球面/椭球已钉）推进到**逐元素**层面——椭球 1.0/1.6/0.7（sub 1，162 节点）整体平移 `(3,-2,1.5)` 后，`S` 与 `D` 的每个元素必须不变（容差 1e-10 相对）。Laplace 核 `G(x,y)=1/(4π|x-y|)` 与等参法向只依赖相对位置，任何把绝对坐标混进位移或法向的符号/链式法则错误都能通过行和检验却会被此钉抓住。
2. 全套测试复验（2379+ collected）。

**热文件声明**：本 PR 仅改 `tests/test_panel_quadratic_operator.py`（测试面，非核心模块）；开 PR 前查 GitHub `cxs503/TensorBEM` open PR 确认无重叠（见 §7）。

## 3. 计算结果（程序化注入，零手抄，源 `docs/quadratic_panel_verification/study.json`）

### 3.1 通量收敛（单位球，点源）

| 级别 | 节点数 | h | 通量相对 L2 误差 | cond(S) | 残差 L2 |
|---|---|---|---|---|---|
| 0 | 42 | 0.6919 | 1.404e-01 | 20.8 | 2.99e-16 |
| 1 | 162 | 0.3819 | 2.854e-02 | 39.4 | 4.65e-16 |
| 2 | 642 | 0.1963 | **6.065e-03** | 80.8 | 8.62e-16 |

### 3.2 收敛阶（log-log 最小二乘）

| 物理量 | 实测阶 | 理论预期 |
|---|---|---|
| 通量相对误差 | 2.49 | O(h²)~O(h³) |
| \|S·1-1\|∞ | 4.13 | 超收敛 |
| \|D·1+1/2\|∞ | 2.93 | O(h³) 几何 |

### 3.3 算子恒等式（最细网格 642 节点）

| 恒等式 | 残差 | 判定 |
|---|---|---|
| \|S·1-1\|∞ | 7.22e-05 | ✓ |
| \|D·1+1/2\|∞ | 9.61e-04 | ✓ |
| \|(1/2 I-D)·1-1\|∞ | 9.61e-04 | ✓ |

### 3.4 本 PR 新增：算子平移不变性（椭球 1.0/1.6/0.7，sub 1）

| 量 | 实测 | 阈值 | 判定 |
|---|---|---|---|
| max\|S_shifted - S_base\| / max\|S_base\| | ≤ 1e-10（测试通过） | 1e-10 | ✓ |
| max\|D_shifted - D_base\| / max\|D_base\| | ≤ 1e-10（测试通过） | 1e-10 | ✓ |
| 平移后 \|D·1+1/2\|∞ | < 2e-2（测试通过） | 2e-2 | ✓ |

### 3.5 几何泛化 / 相似不变性 / 薄体范围限（m8616 已入 main，本 PR 复验）

- 椭球 1.0/1.6/0.7 最细级通量误差 6.43e-03 ≤ 3% 阈值，观测阶 2.63 ≥ 1.5 下限 → `geometry_passed=true`
- 平移+均匀缩放（scale 2.7, shift (3,-2,1.5)）：误差比 0.9999999999997574，相对偏差 2.4e-13 ≤ 1e-10 → `similarity_passed=true`
- 薄椭球 1.0/2.5/0.5：最粗级通量误差 0.33（记录不门控，`gated=false`）

## 4. 结果图

`docs/quadratic_panel_verification/fig_quadratic_operator.png`（sha256 入 study.json provenance）：
左=通量收敛 vs h（标注 O(h²) 参考线、3%/15% 门限）；中=算子恒等式残差 vs h；右=最细网格节点通量偏差场（Mollweide 投影）。
`fig_geometry_generic.png`：球 vs 椭球收敛 + 薄体范围限。
本 PR 未改动任何图（+0/-0），图与 study.json 的 sha256 一致性由 `test_committed_study_json_matches_its_artifacts` 守护。

## 5. 注意点

1. **仅 Laplace 算子**：Helmholtz 频域扩展待后续（D 矩阵可复用）。
2. **固定求积阶数**：非奇异 8 阶 Gauss、自面板 16 阶 Duffy；近奇异自适应未实现（当前精度足够）。
3. **条件数增长**：cond(S) ~ 1/h（20.8→80.8），迭代求解需预处理。
4. **薄体范围限**：局部厚度 < 面板尺寸时点源场近奇异，通量误差受分辨率限制而非算子错误（记录不门控）。
5. **适用范围**：光滑封闭 C² 曲面、Laplace 外问题、h 足够小；不适用于尖锐边缘/薄结构。
6. **平移不变性钉的边界**：只覆盖刚体平移（核的相对性）；非均匀仿射（各向异性缩放）不保持 S/D 逐元素相等，不在本钉范围（相似不变性由 flux-error 层面钉住）。

## 6. 误差判定

| 验收项 | 基准 | 实测 | 是否<3% | 判定 |
|---|---|---|---|---|
| 通量相对 L2 误差（642 节点） | 精确解 | 6.065e-03 (0.61%) | **是** | PASS |
| 硬限制 15% | — | 0.61% | 是 | PASS |
| 椭球几何门（最细级） | 3% 阈值 | 6.43e-03 | 是 | PASS |
| 相似不变性 | 1e-10 容差 | 2.4e-13 | 是 | PASS |
| 算子平移不变性（本 PR 新增） | 0（相对 1e-10） | 测试通过 | 是 | PASS |
| 全套测试 | 见 §7 | 见 §7 | — | 见 §7 |

**结论**：PR#546 功能要点全部在 main 且验收达标；本 PR 补算子平移不变性回归钉，`acceptance.passed=true`。

## 7. 测试执行

```bash
PYTHONPATH=src python -m pytest tests/test_panel_quadratic.py tests/test_panel_quadratic_operator.py -q
# 52 passed（原 51 + 本 PR 新增 1）
```

全套测试（`pytest tests/ -q`）结果见 mission 报告 §7。

## 8. 复现命令

```bash
PYTHONPATH=src python -m pytest tests/test_panel_quadratic.py tests/test_panel_quadratic_operator.py -q
PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_panel_verification   # 复现 study.json + 图
```
