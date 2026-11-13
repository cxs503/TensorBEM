# Dev Report m18391: CASE-BEM-HULME-HEMISPHERE 算例池 CI 修复

**日期**: 2026-11-13
**锚点**: CASE-BEM-HULME-HEMISPHERE
**仓**: TensorBEM
**类型**: defect_fix (测试夹具修复 + 缺失工件补全)

---

## 1. 问题发现

在扫描 `CASE-BEM-HULME-HEMISPHERE` 算例池锚点的 CI 状态时，发现两个测试失败：

### 1.1 `test_casepool_hulme_hemisphere.py` — 4 个测试全部 FAIL

**根因**: 测试文件加载了错误的模块。它通过 `importlib.util` 动态加载 `benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py`（分片编排器），但测试断言期望的属性（`reference`, `_panel_cell`, `build_tasks`, `evaluate`, `BAND_KA`, `COARSE_SUBDIVISIONS`, `CASE_SUBDIVISIONS`, `QUANTITIES`, `GATE`, `RADIUS`）全部定义在库模块 `src/tensorbem/benchmarks/hulme_hemisphere_case.py` 中。

`run_case.py` 仅导入 `hulme_hemisphere_case as case` 并提供 `run_sharded()` 和 `main()` 两个编排函数，不暴露物理/参考/门限逻辑。

**失败表现**:
```
AttributeError: module 'hulme_case_driver' has no attribute 'reference'
AttributeError: module 'hulme_case_driver' has no attribute '_panel_cell'
AttributeError: module 'hulme_case_driver' has no attribute 'build_tasks'
AttributeError: module 'hulme_case_driver' has no attribute 'evaluate'
```

### 1.2 `test_casepool_hulme_hemisphere_results.py` — `test_figure_artifact_exists` FAIL

**根因**: `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png` 文件缺失。该图应由 `hulme_hemisphere_case.write_figure()` 在完整 case run 时生成，但之前的运行（mission 17292）可能未生成或工件未保留。

---

## 2. 修复内容

### 2.1 修复测试模块路径 (`tests/test_casepool_humme_hemisphere.py`)

将 `DRIVER` 路径从：
```python
DRIVER = (Path(__file__).resolve().parents[1]
    / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere" / "run_case.py")
```
改为：
```python
DRIVER = (Path(__file__).resolve().parents[1]
    / "src" / "tensorbem" / "benchmarks" / "hulme_hemisphere_case.py")
```

这是最小改动——只改路径，不改任何测试逻辑。库模块 `hulme_hemisphere_case.py` 完整暴露了测试所需的全部公共/私有 API。

### 2.2 补全缺失的 figure.png

通过调用 `hulme_hemisphere_case.write_figure()` 从已提交的 `results.json` 数据重新生成对比图：
- 左图: Heave added mass (A33) — Hulme 参考 vs TensorBEM 1024/256 panels
- 中图: Heave radiation damping (B33) — 同上
- 右图: Wave excitation force magnitude (|Fx|, |Fz|) — 同上

输出: `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png` (159 KB, DPI 150)

---

## 3. 验证结果

### 3.1 修复后测试状态

| 测试文件 | 用例数 | 通过 | 失败 |
|---|---|---|---|
| `test_casepool_hulme_hemisphere.py` | 4 | 4 | 0 |
| `test_casepool_hulme_hemisphere_results.py` | 14 | 14 | 0 |
| **合计** | **18** | **18** | **0** |

### 3.2 关联测试回归检查

| 测试文件 | 用例数 | 通过 | 失败 |
|---|---|---|---|
| `test_hulme_hemisphere_case.py` | 6 | 6 | 0 |
| `test_hulme_benchmark_integration.py` | 6 | 6 | 0 |
| `test_kochin_hulme_convergence.py` | 5 | 5 | 0 |

### 3.3 算例池数值结果（已验证，来自 results.json）

**Case mesh (1024 panels, subdivisions=4)** — 所有量误差 < 5% gate:

| ka | A11 err | B11 err | A33 err | B33 err | |Fx| err | |Fz| err |
|---|---|---|---|---|---|---|
| 0.2 | -0.39% | -0.96% | -0.16% | -0.20% | -0.52% | -0.10% |
| 0.5 | -0.46% | -0.99% | -0.11% | -0.03% | -0.54% | -0.01% |
| 0.8 | -0.42% | -0.96% | -0.13% | +0.16% | -0.53% | +0.08% |
| 1.0 | -0.33% | -0.90% | -0.18% | +0.30% | -0.51% | +0.13% |
| 1.5 | -0.01% | -0.73% | -0.34% | +0.69% | -0.44% | +0.25% |
| 2.0 | +0.23% | -0.58% | -0.53% | **+1.15%** | -0.40% | +0.49% |

**最坏误差**: B33 at ka=2.0 = 1.15% << 5% gate ✅

**参考截断收敛**: 96→144 term drift 最大 6.46e-05 (ka=2.0)，比 5% gate 小 774 倍 ✅

**几何预条件**: 网格体积误差 < 0.5%，水线面误差 < 0.1% ✅

---

## 4. 改动文件

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `tests/test_casepool_hulme_hemisphere.py` | 修复 | DRIVER 路径从 run_case.py 改为 hulme_hemisphere_case.py |
| `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png` | 新增 | 对比图（159 KB），由 write_figure() 生成 |

---

## 5. 对 feature_ledger 的建议

`CASE-BEM-HULME-HEMISPHERE` 当前 status = `open`，但：
- 算例已完整运行（mission 17292），verdict = PASS
- 所有 6 个 ka 值、6 个物理量的误差均 < 5%
- 参考截断收敛已验证（几何尾比 ~0.189）
- 独立交叉验证已就位（capytaine crosscheck + rigid-lid ka→0 端点）

**建议**: 将 status 更新为 `verified` 或 `done`。

---

## 6. 注意点

- 修复仅涉及测试夹具路径，不改变任何物理/数值逻辑
- figure.png 是从已验证的 results.json 数据重新生成的，不引入新计算
- 高负载环境（load15 ≈ 290）下未执行重计算，仅做轻量修复
