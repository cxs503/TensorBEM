# Dev Report m19969: CASE-BEM-HULME-DIFF-VERIF — Hulme 3D Diffraction Verification

**日期**: 2026-11-14
**锚点**: CASE-BEM-HULME-DIFF-VERIF
**仓**: TensorBEM
**类型**: capability_extension (验证欠账补全 + 测试 + 报告)

---

## 1. 问题发现

`CASE-BEM-HULME-DIFF-VERIF`（算例池·Hulme 绕射验证（3D））在 feature_ledger 中状态为 `open`，
验收标准：Hulme 1982 球谐绕射参考，容差 <5%。

已有工作（mission 17278）完成了初步验证，但 `benchmarks/verified/hulme_diffraction_3d/` 目录缺少：
- `results.json`（仅有 `results_reduced.json`）
- 对比图 PNG 文件
- 独立的验证测试文件

## 2. 修复/增量内容

### 2.1 生成 results.json 与对比图

创建 `make_results_and_figures.py`，运行 8 个 ka 值 × 2 级网格（64/256 面板）的完整绕射求解：

| ka | panels | |Fx|_ref | |Fx|_BEM | err_surge | |Fz|_ref | |Fz|_BEM | err_heave | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.100 | 256 | 3044.03 | 2982.51 | -2.02% ✓ | 27727.50 | 27581.26 | -0.53% ✓ | ✓ |
| 0.200 | 256 | 5874.11 | 5753.97 | -2.05% ✓ | 24314.14 | 24215.65 | -0.41% ✓ | ✓ |
| 0.300 | 256 | 8489.78 | 8313.84 | -2.07% ✓ | 21426.90 | 21365.89 | -0.28% ✓ | ✓ |
| 0.500 | 256 | 12924.43 | 12650.91 | -2.12% ✓ | 16943.22 | 16935.63 | -0.04% ✓ | ✓ |
| 0.800 | 256 | 16754.60 | 16404.84 | -2.09% ✓ | 12386.11 | 12424.08 | +0.31% ✓ | ✓ |
| 1.000 | 256 | 17297.83 | 16951.94 | -2.00% ✓ | 10255.31 | 10308.41 | +0.52% ✓ | ✓ |
| 1.500 | 256 | 15049.47 | 14789.82 | -1.73% ✓ | 6733.83 | 6798.81 | +0.96% ✓ | ✓ |
| 2.000 | 256 | 12038.25 | 11851.04 | -1.56% ✓ | 4671.97 | 4756.78 | +1.82% ✓ | ✓ |

**Verdict: ALL PASS** — 最坏误差 2.12% << 5% 容差 ✅

生成 3 张对比图：
- `fig_hulme_diffraction_magnitude.png` — 绕射力幅值 vs ka
- `fig_hulme_diffraction_error.png` — 相对误差 vs ka（含 ±5% 容差带）
- `fig_hulme_diffraction_convergence.png` — 网格收敛（64 vs 256 面板）

### 2.2 新增验证测试 `tests/test_hulme_diffraction_3d_verif.py`

15 个测试用例，覆盖：
- **工件完整性**（4 项）：results.json 存在/结构正确/全部门通过/图文件存在
- **数值验证**（6 项）：surge/heave 误差 <5%/最坏误差裕度/参考值从库重算/网格收敛
- **物理一致性**（4 项）：surge 相位象限/对称 DOF 近零/低频 heave 主导/heave 单调递减
- **端到端重算**（1 项）：ka=1.0 重新求解验证

### 2.3 发布平台报告

报告 ID: 18191，状态: published，含 3 张内嵌图。

## 3. 验证结果

| 测试文件 | 用例数 | 通过 | 失败 |
|---|---|---|---|
| `test_hulme_diffraction_3d_verif.py` | 15 | 15 | 0 |
| `test_casepool_hulme_diff_verif.py` | 10 | 10 | 0 |
| **合计** | **25** | **25** | **0** |

### 误差判定

| 关键量 | 最坏误差 | <3%? | <5%? |
|---|---|---|---|
| Surge excitation | 2.12% (ka=0.5) | ✅ | ✅ |
| Heave excitation | 1.82% (ka=2.0) | ✅ | ✅ |

**所有 8 个 ka 值均通过 <5% 容差标准。**

## 4. 改动文件

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `benchmarks/verified/hulme_diffraction_3d/results.json` | 新增 | 完整验证结果（35 KB） |
| `benchmarks/verified/hulme_diffraction_3d/fig_hulme_diffraction_magnitude.png` | 新增 | 绕射力幅值对比图（166 KB） |
| `benchmarks/verified/hulme_diffraction_3d/fig_hulme_diffraction_error.png` | 新增 | 相对误差对比图（71 KB） |
| `benchmarks/verified/hulme_diffraction_3d/fig_hulme_diffraction_convergence.png` | 新增 | 网格收敛图（117 KB） |
| `benchmarks/verified/hulme_diffraction_3d/make_results_and_figures.py` | 新增 | 结果+图生成脚本 |
| `tests/test_hulme_diffraction_3d_verif.py` | 新增 | 15 项验证测试 |

## 5. 对 feature_ledger 的建议

`CASE-BEM-HULME-DIFF-VERIF` 当前 status = `open`，但：
- 算例已完整运行（8 ka × 2 mesh levels），verdict = ALL PASS
- 所有误差 < 5% 容差，最坏 2.12%
- 网格收敛已验证（64→256 面板单调收敛）
- 独立测试 15 项全通过
- 平台报告已发布（id 18191）

**建议**: 将 status 更新为 `verified`。

## 6. 注意点

- sub=4（1024 面板）因计算成本未在本机运行，已在 results.json 中注明
- 所有参考值从 `tensorbem.analytic.hulme_hemisphere_radiation` 库函数重算，无手抄数字
- 对称 DOF（sway/roll/pitch/yaw）在正迎浪下正确返回近零值（<2% heave）
