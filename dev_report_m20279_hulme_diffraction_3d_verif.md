# Dev Report m20279: CASE-BEM-HULME-DIFF-VERIF — Hulme 3D Diffraction Verification (补全缺失图件)

**日期**: 2026-11-15
**锚点**: CASE-BEM-HULME-DIFF-VERIF
**仓**: TensorBEM
**类型**: capability_extension (验证欠账补全 — 缺失图件修复)

---

## 1. 问题发现

`CASE-BEM-HULME-DIFF-VERIF`（算例池·Hulme 绕射验证（3D））在 feature_ledger 中状态为 `open`。
前序 mission m19969 已完成数值求解和 results.json 生成，但 3 张验证对比图 PNG 未提交到仓库，
导致 `test_figure_artifacts_exist` 测试 FAIL（15 项测试中 14 通过、1 失败）。

## 2. 修复/增量内容

### 2.1 生成 3 张验证对比图

从已有的 `results.json` 数据生成：
- `fig_hulme_diffraction_magnitude.png` (125 KB) — Surge/Heave 力幅值 vs ka（BEM vs Hulme 参考）
- `fig_hulme_diffraction_error.png` (41 KB) — 相对误差 vs ka（含 ±5% 容差带）
- `fig_hulme_diffraction_convergence.png` (122 KB) — 网格收敛对数图（sub=2 vs sub=3）

### 2.2 新增图件生成脚本

`benchmarks/verified/hulme_diffraction_3d/generate_figures.py` — 可复用的图件生成脚本，
从 results.json 读取数据并输出 3 张 PNG。

### 2.3 发布平台报告

报告 ID: 18453，状态: published，含 3 张内嵌图。

### 2.4 更新 feature_ledger

将 `CASE-BEM-HULME-DIFF-VERIF` 状态从 `open` 更新为 `verified`。

## 3. 验证结果

| 测试文件 | 用例数 | 通过 | 失败 |
|---|---|---|---|
| `test_hulme_diffraction_3d_verif.py` | 15 | 15 | 0 |

### 误差判定

| 关键量 | 最坏误差 | <3%? | <5%? |
|---|---|---|---|
| Surge excitation | 2.12% (ka=0.5) | ✅ | ✅ |
| Heave excitation | 1.82% (ka=2.0) | ✅ | ✅ |

**所有 8 个 ka 值均通过 <5% 容差标准。**

## 4. 改动文件

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `benchmarks/verified/hulme_diffraction_3d/fig_hulme_diffraction_magnitude.png` | 新增 | 绕射力幅值对比图（125 KB） |
| `benchmarks/verified/hulme_diffraction_3d/fig_hulme_diffraction_error.png` | 新增 | 相对误差对比图（41 KB） |
| `benchmarks/verified/hulme_diffraction_3d/fig_hulme_diffraction_convergence.png` | 新增 | 网格收敛图（122 KB） |
| `benchmarks/verified/hulme_diffraction_3d/generate_figures.py` | 新增 | 图件生成脚本 |
| `post_report_m20279.py` | 新增 | 报告发布脚本 |

## 5. 对 feature_ledger 的更新

`CASE-BEM-HULME-DIFF-VERIF` status: `open` → `verified`

- 算例已完整运行（8 ka × 2 mesh levels），verdict = ALL PASS
- 所有误差 < 5% 容差，最坏 2.12%
- 网格收敛已验证（64→256 面板单调收敛）
- 独立测试 15 项全通过
- 平台报告已发布（id 18453）
- 3 张验证图件已入库

## 6. 注意点

- 本单不重新运行求解器，仅从已有 results.json 生成图件
- 所有参考值从 `tensorbem.analytic.hulme_hemisphere_radiation` 库函数重算，无手抄数字
- sub=4（1024 面板）因计算成本未在本单运行，results.json 中已注明
