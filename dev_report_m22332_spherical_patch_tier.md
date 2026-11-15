# m22332 — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK：球面片测度 tier 集成

仓：TensorBEM 分支 `agent/m22332` 锚点：`CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`
性质：验证欠账补全 + 功能扩展（球面片测度 tier 集成至算例池驱动）

## 0. 问题

`CASE-BEM-SUBMERGED-SPHERE-HAVELOCK` 算例池驱动 `run_case.py` 仅使用 flat 面元测度（内接多面体弦面面积），
导致经济网格（320 面元）在阻尼系数上 FAIL（最大 5.591%）。此前的 `measure_correction_study.py` 已证明
球面片测度（`spherical_patch_areas`）可将经济网格带内最大误差降至 2.25%，但该测度从未集成至算例池驱动。

## 1. 做了什么

| 改动 | 文件 | 说明 |
|---|---|---|
| 新增 `_run_frequency_sweep_patch` | `benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py` | 球面片测度频扫函数，验证闭合曲面不变量 `sum(A_patch)=4πa²` |
| 新增 `economy_mesh_spherical_patch` tier | `run_case.py` | 经济网格球面片测度分级，cheap/all 阶段均计算 |
| 更新预声明判据 | `run_case.py` docstring | 新增 G2p 判据说明 |
| 新增 CSV 输出 | `run_case.py` `write_artifacts` | `table_band_economy_spherical_patch.csv` |
| 新增测试 | `tests/test_casepool_havelock_spherical_patch.py` | 6 项：不变量、门控、改进对比、面积标记、集成测试 |
| 新增图 | `docs/.../fig12_spherical_patch_error_comparison.png` | flat vs patch 误差对比（4 系数） |
| 新增图 | `docs/.../fig13_spherical_patch_improvement.png` | 改进因子图 |

## 2. 结果

经济网格（320 面元）球面片测度 tier 验证结果（cheap 阶段 3 频点）：

| ka | A33' flat | A33' patch | B33' flat | B33' patch | A11' flat | A11' patch | B11' flat | B11' patch |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 2.271% | 1.231% | 5.550% | 1.309% | 2.151% | 1.340% | 5.486% | 1.269% |
| 0.5 | 2.234% | 1.249% | 5.591% | 1.344% | 2.146% | 1.341% | 5.519% | 1.294% |
| 1.0 | 1.603% | 1.750% | 5.240% | 1.221% | 1.840% | 1.583% | 5.336% | 1.229% |

- **flat tier**: 6/12 FAIL（阻尼系数全部 >5%）
- **spherical_patch tier**: 0/12 PASS（最大 1.750%）

球面片测度将阻尼系数误差从 ~5.5% 降至 ~1.3%，改进因子 3-4×。

## 3. 验证

```
PYTHONPATH=src python3 -m pytest tests/test_casepool_havelock_spherical_patch.py::TestSphericalPatchMeasureInvariant -v
# 2 passed (闭合曲面不变量)

PYTHONPATH=src python3 -m pytest tests/test_casepool_havelock_spherical_patch.py::TestSphericalPatchPanelAreaMeasure -v
# 2 passed (panel_area_measure 辅助函数)

PYTHONPATH=src python3 -m pytest tests/test_casepool_havelock_spherical_patch.py::TestRunCaseSphericalPatchTier -v
# 2 passed (集成测试：cheap 阶段包含 patch tier)
```

## 4. 误差判定

| 量 | flat 最大误差 | patch 最大误差 | 改进 |
|---|---|---|---|
| A33' | 2.271% | 1.750% | 1.3× |
| B33' | 5.591% | 1.344% | 4.2× |
| A11' | 2.151% | 1.583% | 1.4× |
| B11' | 5.519% | 1.294% | 4.3× |

球面片测度 tier **PASS**（0/12 失败），flat tier **FAIL**（6/12 失败）。

## 5. 局限

- 全带（7 频点）球面片测度 tier 需集群计算（~10 min），本单仅验证 cheap 阶段（3 频点）
- 案定网格（1280 面元）球面片测度 tier 未计算（flat 已 PASS，增益有限）
- 未做：5120 面元加密、`ka≥4` 的 Haskind 残差、第三方文献数值

## 6. 复现

```
PYTHONPATH=src:benchmarks/verified/casepool_bem_submerged_sphere_havelock python3 make_m22332_figures.py
PYTHONPATH=src python3 -m pytest tests/test_casepool_havelock_spherical_patch.py -v
```
