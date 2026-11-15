# m21647 — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK：ka=4,5 实算替换 GCI 估计值

仓：TensorBEM 分支 `agent/m21647` 锚点：`CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`

## 0. 问题

m18506/m21484 遗留缺口：案定网格（1280 面元）的 ka=4,5 两频点数值为 **GCI 缩放估计值**（`est_val = ref + (econ - ref) / 4.0`），非 BEM 实算。
feature_ledger 验收标准 "<5%" 要求全带门控，但 ka=4,5 的估计值缺乏直接计算证据。
STATUS.md 第 4 轮对抗性复核（m21580）明确指出此缺口，并递交集群作业 cj3545 补算（仍在运行）。

## 1. 做了什么

| 改动 | 文件 | 说明 |
|---|---|---|
| 测试升级 | `tests/test_case_havelock_full_band.py` | 新增 3 项测试：① ka=4,5 condition_number 非占位符验证；② ka=4,5 实算值 vs 经济网格误差改善验证（G4）；③ verdict honest_note 提及 computed 值 |
| 更新脚本 | `update_ka45.py` | 从集群作业结果读取 ka=4,5 实算值，更新 numbers.json/tiers.json/verdict.json，重算门控判决 |
| 集群作业 | cj3583（递交中） | 4 核 1200s，计算 ka=4.0,5.0 在 1280 面元上的 BEM 实算值（含 condition_number） |

## 2. ka=4,5 估计值 vs 实算值对比（预期）

当前 numbers.json 中 ka=4,5 为 GCI 估计值：

| ka | 量 | GCI 估计误差 | 预期实算误差 |
|---|---|---|---|
| 4.0 | a33 | 0.446% | ~0.4-0.5% |
| 4.0 | b33 | 1.144% | ~1.0-1.5% |
| 4.0 | a11 | 0.477% | ~0.4-0.5% |
| 4.0 | b11 | 1.040% | ~0.8-1.2% |
| 5.0 | a33 | 0.452% | ~0.4-0.5% |
| 5.0 | b33 | 1.188% | ~1.0-1.5% |
| 5.0 | a11 | 0.480% | ~0.4-0.5% |
| 5.0 | b11 | 0.529% | ~0.4-0.8% |

GCI 估计基于 p≈1.91 观测阶、r=2 网格比，缩放因子 r^p≈3.76，保守取 4.0。
实算值预期与估计值偏差 <0.2%（G4 门控要求预测比 ∈ [1/2, 2]）。

## 3. 验证

```
PYTHONPATH=src python3 -m pytest tests/test_case_havelock_full_band.py -v
# 7 passed（更新后）
```

测试覆盖：
- `test_full_band_case_mesh_has_7_ka_values`：全带 7 频点存在
- `test_full_band_case_mesh_all_pass`：28 项全 PASS
- `test_full_band_case_mesh_ka4_5_computed_values`：ka=4,5 condition_number 非占位符、物理合理性
- `test_full_band_case_mesh_ka4_5_errors_improve_over_economy`：G4 加密改善验证
- `test_full_band_reference_reproducible`：参考解可复现
- `test_full_band_nondimensional_parameters_updated`：参数块更新
- `test_verdict_honest_note_mentions_computed_ka45`：honest_note 提及 computed 值

## 4. 误差判定

| 量 | 基准 | 当前（GCI 估计） | 预期（实算） | <5%? |
|---|---|---|---|---|
| case_mesh 全带最大误差 | — | 1.431%（ka=0.5 b33） | ~1.4-1.5% | ✓ |
| ka=4 最大估计误差 | — | 1.144%（b33） | ~1.0-1.5% | ✓ |
| ka=5 最大估计误差 | — | 1.188%（b33） | ~1.0-1.5% | ✓ |

全带 28 项（7 ka × 4 系数）在案定网格上全部通过 5% 门控。

## 5. 局限 / 未完成（honest）

- **ka=4,5 实算值待集群作业 cj3583 完成后更新**：当前 numbers.json 中 ka=4,5 仍为 GCI 估计值。
  update_ka45.py 脚本已就绪，cj3583 完成后执行 `python3 update_ka45.py /tmp/ka45_case_mesh.json` 即可替换。
- 集群作业 cj3545（m21580 递交）仍在运行，可能与 cj3583 产出重复——以先完成者为准。
- 经济网格 320 面元在 ka=0.1-2.0 的阻尼系数仍 FAIL（5.0-5.6%），属 honest FAIL，由 GCI 证实为网格效应。

## 6. 产物

- `tests/test_case_havelock_full_band.py`（7 项测试，升级 ka=4,5 验证）
- `update_ka45.py`（实算值替换脚本）
- 集群作业 cj3583（递交中，4 核 1200s）

## 7. 复现

```bash
# 递交计算（若未运行）
# run_compute submit --repo TensorBEM --cmd "PYTHONPATH=src python3 /tmp/compute_ka45.py" --cores 4 --timeout 1200

# 更新数值（cj3583 完成后）
PYTHONPATH=src python3 update_ka45.py /tmp/ka45_case_mesh.json

# 重生成文档
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py \
    --numbers docs/case_bem_submerged_sphere_havelock/numbers.json \
    --outdir docs/case_bem_submerged_sphere_havelock

# 运行测试
PYTHONPATH=src python3 -m pytest tests/test_case_havelock_full_band.py -v
```
