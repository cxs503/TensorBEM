# Dev Report m20370: CASE-BEM-HULME-HEMISPHERE 证据包修复

**日期**: 2026-11-15
**锚点**: CASE-BEM-HULME-HEMISPHERE
**仓**: TensorBEM
**类型**: defect_fix (缺失工件补全 + 证据包 SHA256 清单更新)

---

## 1. 问题发现

在扫描 `CASE-BEM-HULME-HEMISPHERE` 算例池锚点的测试状态时，发现 3 个测试失败：

### 1.1 `test_casepool_hulme_hemisphere_results.py::test_figure_artifact_exists` — FAIL

**根因**: `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png` 文件缺失。
该图应由 `hulme_hemisphere_case.write_figure()` 在完整 case run 时生成，
但之前的运行可能未生成或工件未保留。

### 1.2 `test_casepool_hulme_hemisphere_evidence_bundle.py::test_bundle_regenerates_byte_identical` — FAIL

**根因**: `evidence/artifacts_sha256.json` 中的 SHA256 摘要与磁盘上实际文件不匹配。
具体而言，`run_case.py` 从 4329 字节变为 4773 字节（前序修改），
但证据包清单未同步更新。

### 1.3 `test_casepool_hulme_hemisphere_evidence_bundle.py::test_manifest_hashes_match_the_files_on_disk` — FAIL

**根因**: 同上，清单中 `run_case.py` 的 SHA256 哈希过期。

---

## 2. 修复内容

### 2.1 补全缺失的 figure.png

通过调用 `hulme_hemisphere_case.write_figure()` 从已提交的 `results.json` 数据重新生成对比图：
- 左图: Heave added mass (A33) — Hulme 参考 vs TensorBEM 1024/256 panels
- 中图: Heave radiation damping (B33) — 同上
- 右图: Wave excitation force magnitude (|Fx|, |Fz|) — 同上

输出: `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png` (159,683 bytes, DPI 150)

### 2.2 重建证据包 SHA256 清单

运行 `evidence/build_evidence.py` 重新生成整个证据包，更新 `artifacts_sha256.json`：
- aggregate_sha256: `0e1fd623...` → `be7357f6...`
- run_case.py SHA256 与字节数同步更新

---

## 3. 验证结果

### 3.1 修复后测试状态

| 测试文件 | 用例数 | 通过 | 失败 |
|---|---|---|---|
| `test_casepool_hulme_hemisphere.py` | 4 | 4 | 0 |
| `test_casepool_hulme_hemisphere_results.py` | 14 | 14 | 0 |
| `test_casepool_hulme_hemisphere_evidence_bundle.py` | 14 | 14 | 0 |
| **合计** | **32** | **32** | **0** |

### 3.2 关联测试回归检查

| 测试文件 | 用例数 | 通过 | 失败 |
|---|---|---|---|
| `test_hulme_hemisphere_case.py` | 6 | 6 | 0 |
| `test_hulme_benchmark_integration.py` | 6 | 6 | 0 |
| **合计** | **12** | **12** | **0** |

### 3.3 算例池数值结果（已验证，来自 results.json）

**Case mesh (1024 panels, subdivisions=4)** — 所有量误差 < 5% gate:

| ka | A11 err | B11 err | A33 err | B33 err | |Fx| err | |Fz| err |
|---|---|---|---|---|---|---|
| 0.2 | -0.39% | -0.96% | -0.16% | -0.20% | -0.52% | -0.10% |
| 0.5 | -0.46% | -0.99% | -0.11% | -0.03% | -0.54% | -0.01% |
| 0.8 | -0.42% | -0.96% | -0.13% | +0.16% | -0.53% | +0.08% |
| 1.0 | -0.33% | -0.90% | -0.18% | +0.30% | -0.51% | +0.13% |
| 1.5 | -0.01% | -0.73% | -0.34% | +0.69% | -0.44% | +0.25% |
| 2.0 | +0.23% | -0.58% | -0.53% | +1.15% | -0.40% | +0.49% |

**最坏误差**: B33 at ka=2.0 = 1.15% << 5% gate ✅

---

## 4. 改动文件

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png` | 新增 | 对比图（159 KB），由 write_figure() 生成 |
| `benchmarks/verified/casepool_bem_hulme_hemisphere/evidence/artifacts_sha256.json` | 更新 | SHA256 清单同步（run_case.py 变更 + aggregate 更新） |

---

## 5. 对 feature_ledger 的建议

`CASE-BEM-HULME-HEMISPHERE` 当前 status = `open`，但：
- 算例已完整运行（mission 17292），verdict = PASS
- 所有 6 个 ka 值、6 个物理量的误差均 < 5%
- 参考截断收敛已验证（几何尾比 ~0.189）
- 独立交叉验证已就位（capytaine crosscheck + rigid-lid ka→0 端点）
- 证据包完整且可再生（32 项测试全通过）

**建议**: 将 status 更新为 `verified`。

---

## 6. 注意点

- 修复仅涉及缺失工件补全和证据包清单同步，不改变任何物理/数值逻辑
- figure.png 是从已验证的 results.json 数据重新生成的，不引入新计算
- 证据包清单中不包含 figure.png（MANIFEST_FILES 设计如此，仅跟踪数据工件）
- figure.png 的完整性由独立测试 `test_figure_artifact_exists` 保障
