# TensorBEM 二次曲面单元算子验证报告（PR#533 重开·管到底第1轮·m8641）

**Task**: m8641（重开 TensorBEM#533，管到底第1轮）· **Branch**: `agent/m8641` · **Repo**: TensorBEM
**Date**: 2026-11-09
**Status**: ✓ 功能要点已在 main，本 PR 补验证 + 非球面算子回归钉

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

**功能要点核对**（PR#533 = m7526 成果，merge_master 补开 PR，owner 2026-11-05 裁定关闭）：

| 文件 | PR#533 声明 | main 现状 | 判定 |
|---|---|---|---|
| `src/tensorbem/panel_quadratic.py` | +244/-100 | 1028 行（含 m8282 修复 + m8616 椭球扩展） | ✓ 已入 main |
| `src/tensorbem/benchmarks/quadratic_panel_verification.py` | +300 | 934 行（v2 schema，含几何/相似/薄体研究） | ✓ 已入 main |
| `tests/test_panel_quadratic.py` | +32/-2 | 306 行 | ✓ 已入 main |
| `tests/test_panel_quadratic_operator.py` | +204 | 536 行 | ✓ 已入 main |
| `docs/quadratic_panel_verification/study.json` | +169 | v2 schema，202 行 | ✓ 已入 main |
| `docs/quadratic_panel_verification/fig_quadratic_operator.png` | +0/-0 | 305 KB，sha256 入 provenance | ✓ 已入 main |

**承接链**：m7526 原实现（commit 6390ed0，未入 main）→ m8282 修复 4 个结构性缺陷（边中点梯度符号 / D 装配为零 / 边中点奇异求积 / 等效源密度混淆）经 PR#550 入 main（commit 553b28a）→ m8616 加椭球几何 + 相似不变性 + 薄体范围限（PR#558，commit 6d60ba3）。PR#533 的全部功能要点由 PR#550 + PR#558 完整覆盖并超越。

**本 PR 增量**（单一功能点：二次曲面单元算子验证面）：
1. 新增回归钉 `test_ellipsoid_solid_angle_identity`：把 `D·1 = -1/2` 固体角恒等式从球面/立方体扩展到**非球面**（椭球 1.0/1.6/0.7，icosphere 仿射像），钉住等参 Jacobian 与奇异自面板求积在离开球对称时的正确性。
2. 本报告（验证留痕）。

**热文件声明**：`tests/test_panel_quadratic_operator.py` 属 TensorBEM 核心验证面。开 PR 前查 GitHub：`cxs503/TensorBEM` 无 open PR，无重叠冲突。

## 3. 计算结果（程序化注入，零手抄，源 `docs/quadratic_panel_verification/study.json`）

### 3.1 通量收敛（单位球，点源）

| 级别 | 节点数 | h | 通量相对 L2 误差 | cond(S) | 残差 L2 |
|---|---|---|---|---|---|
| 0 | 42 | 0.6919 | 1.404e-01 | 20.8 | 2.99e-16 |
| 1 | 162 | 0.3819 | 2.854e-02 | 39.4 | 4.65e-16 |
| 2 | 642 | 0.1963 | **6.065e-03** | 80.8 | 4.64e-16 |

### 3.2 收敛阶（log-log 最小二乘）

| 物理量 | 实测阶 | 理论预期 |
|---|---|---|
| 通量相对误差 | 2.49 | O(h²)~O(h³) |
| \|S·1-1\|∞ | 4.13 | 超收敛 |
| \|D·1+1/2\|∞ | 2.93 | O(h³) 几何 |

### 3.3 算子恒等式（最细网格 642 节点）

| 恒等式 | 残差 | 判定 |
|---|---|---|
| \|S·1-1\|∞ | 1.06e-03 | ✓ |
| \|D·1+1/2\|∞ | 6.32e-03 | ✓ |
| \|(1/2 I-D)·1-1\|∞ | 6.32e-03 | ✓ |

### 3.4 本 PR 新增：椭球固体角恒等式（非球面）

| 级别 | 节点数 | \|D·1+1/2\|∞ | 阈值 | 判定 |
|---|---|---|---|---|
| 0 | 42 | 8.66e-02 | 1e-1 | ✓ |
| 1 | 162 | 1.17e-02 | 2e-2 | ✓ |

### 3.5 几何泛化 / 相似不变性 / 薄体范围限（m8616 已入 main，本 PR 复验）

- 椭球 1.0/1.6/0.7 最细级通量误差 3.00e-02 ≤ 3% 阈值，观测阶 ≥ 1.5 下限 → `geometry_passed=true`
- 平移+均匀缩放（scale 2.7, shift (3,-2,1.5)）：误差比 0.9999999999997574，相对偏差 2.4e-13 ≤ 1e-10 → `similarity_passed=true`
- 薄椭球 1.0/2.5/0.5：最粗级通量误差 0.33（记录不门控，`gated=false`）

## 4. 结果图

`docs/quadratic_panel_verification/fig_quadratic_operator.png`（305 KB，sha256 入 study.json provenance）：
左=通量收敛 vs h（标注 O(h²) 参考线、3%/15% 门限）；中=算子恒等式残差 vs h；右=最细网格节点通量偏差场（Mollweide 投影）。
本 PR 未改动该图（+0/-0），图与 study.json 的 sha256 一致性由 `test_committed_study_json_matches_its_artifacts` 守护。

## 5. 注意点

1. **仅 Laplace 算子**：Helmholtz 频域扩展待后续（D 矩阵可复用）。
2. **固定求积阶数**：非奇异 8 阶 Gauss、自面板 16 阶 Duffy；近奇异自适应未实现（当前精度足够）。
3. **条件数增长**：cond(S) ~ 1/h（20.8→80.8），迭代求解需预处理。
4. **薄体范围限**：局部厚度 < 面板尺寸时点源场近奇异，通量误差受分辨率限制而非算子错误（记录不门控）。
5. **适用范围**：光滑封闭 C² 曲面、Laplace 外问题、h 足够小；不适用于尖锐边缘/薄结构。

## 6. 误差判定

| 验收项 | 基准 | 实测 | 是否<3% | 判定 |
|---|---|---|---|---|
| 通量相对 L2 误差（642 节点） | 精确解 | 6.065e-03 (0.61%) | **是** | PASS |
| 硬限制 15% | — | 0.61% | 是 | PASS |
| 椭球几何门（最细级） | 3% 阈值 | 3.00e-02 | 是（=阈值内） | PASS |
| 相似不变性 | 1e-10 容差 | 2.4e-13 | 是 | PASS |
| 椭球固体角恒等式（本 PR 新增） | 0 | 1.17e-02 (sub1) | 是 | PASS |
| 全套测试 | 2379 collected | 见 §7 | — | 见 §7 |

**结论**：PR#533 功能要点全部在 main 且验收达标；本 PR 补非球面算子回归钉，`acceptance.passed=true`。

## 7. 测试执行

```bash
PYTHONPATH=src python -m pytest tests/test_panel_quadratic.py tests/test_panel_quadratic_operator.py -q
# 51 passed（原 50 + 本 PR 新增 1）
```

全套测试（2379 collected）结果见 mission 报告。

## 8. 复现命令

```bash
PYTHONPATH=src python -m pytest tests/test_panel_quadratic.py tests/test_panel_quadratic_operator.py -q
PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_panel_verification   # 复现 study.json + 图
```
