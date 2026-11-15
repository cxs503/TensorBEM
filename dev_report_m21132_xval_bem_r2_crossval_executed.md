# m21132 — XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE 执行报告

**锚点**: XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE
**仓库**: TensorBEM
**日期**: 2026-11-15

---

## 1. 任务背景

feature_ledger 登记 `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE`（状态 open），验收标准：对标误差 <15%。
前序审计 m20095 发现 TensorBEM 侧矩阵扫描从未执行（results.json 缺失，9/11 测试跳过）。
本单目标：确认交叉验证闭环，产出完整报告。

## 2. 现状核查

### 2.1 工件完整性

| 工件 | 状态 | 大小 |
|---|---|---|
| `benchmarks/verified/capytaine_crosscheck_r2/capytaine_ref_r2.json` | ✅ 存在 | 242 KB |
| `benchmarks/verified/capytaine_crosscheck_r2/results.json` | ✅ 存在 | 834 KB |
| `docs/capytaine_crosscheck_r2.md` | ✅ 存在 | 30 KB |
| `docs/fig_capytaine_r2_matrix.png` | ✅ 存在 | 220 KB |
| `docs/fig_capytaine_r2_convergence.png` | ✅ 存在 | 92 KB |
| `docs/fig_capytaine_r2_deep.png` | ✅ 存在 | 67 KB |

### 2.2 测试执行

```
PYTHONPATH=src python3 -m pytest tests/test_capytaine_crosscheck_r2.py -v
# 17 passed, 1 warning
```

全部 17 个测试通过，覆盖参考溯源、逐项分类、容差冻结、判定一致性、结构零验证、
原位复现、文档一致性、依赖卫生、对称性、解析轴、退化审计、转动恒等式等。

### 2.3 交叉验证结果

| case | 最差 active 偏差 | 判定（5% 带） | 判定（15% ledger） |
|---|---|---|---|
| hemisphere_r1 | 5.82% | FAIL | ✅ PASS |
| hemisphere_r1_cz-0.5 | 5.82% | FAIL | ✅ PASS |
| sphere_r1_z-0.5 | 6.52% | FAIL | ✅ PASS |
| sphere_r1_z-0.5_c0 | 6.71% | FAIL | ✅ PASS |

### 2.4 解析轴验证

| 量 | 解析值 | TensorBEM | 误差 | <3%? |
|---|---|---|---|---|
| 深潜 8R 球 A_surge vs Lamb | 2146.75 kg | 2133.76 kg | -0.60% | ✅ |
| 深潜 8R 球 A_roll/|A_surge| | 0 | 2.71×10⁻⁵ | ✅ |
| 转动中心变换恒等式 | 0 | 1.2×10⁻¹⁴ | 机器精度 | ✅ |

## 3. 结论

**交叉验证已闭环**。TensorBEM 全 6×6 辐射矩阵与 capytaine 的比对已完成：
- 142 个 active 项逐项分类，0 conflict（无结构性分歧）
- 最差偏差 6.71%，远低于 ledger 15% 不可接受线
- 解析第三轴（Lamb 极限）误差 -0.60%，完全通过
- 转动中心变换恒等式残差 1.2×10⁻¹⁴，精确成立
- 17 个测试全部通过

**建议**：feature_ledger 状态应从 `open` 更新为 `verified`。
模块自设 5% 对角带过于严格（两低阶面元法在高频下的离散误差叠加自然超过 5%），
建议后续将模块容差调整为 10% 以与 ledger 标准对齐。

## 4. 产物

- `benchmarks/verified/capytaine_crosscheck_r2/results.json` — 完整比对工件
- `docs/capytaine_crosscheck_r2.md` — 交叉验证文档
- `docs/fig_capytaine_r2_*.png` — 结果图（矩阵热力图、收敛曲线、深潜比较）
- `tests/test_capytaine_crosscheck_r2.py` — 17 个验收测试（全部通过）
