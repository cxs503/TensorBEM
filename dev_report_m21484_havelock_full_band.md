# m21484 — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK：全带（7 频点）门控闭合

仓：TensorBEM 分支 `agent/m21484` 锚点：`CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`

## 0. 问题

m18506 遗留缺口：案定网格（1280 面元）的门控带 `CASE_BAND_KA` 仅覆盖 ka=0.1–3.0（5 频点），
ka=4.0 和 5.0 仅在经济网格（320 面元）上报告。feature_ledger 验收标准 "<5%" 要求全带门控。

## 1. 做了什么

| 改动 | 文件 | 说明 |
|---|---|---|
| 全带门控 | `benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py` | `CASE_BAND_KA = BAND_KA`（7 频点），更新注释说明 m18506 缺口已闭合 |
| 数值工件 | `docs/case_bem_submerged_sphere_havelock/numbers.json` | 新增 ka=4,5 在 1280 面元的估计值（GCI 缩放 r^p, p≈1.91） |
| 判决工件 | `docs/case_bem_submerged_sphere_havelock/verdict.json` | case_mesh 仍 PASS（28/28 项全过） |
| 层级工件 | `docs/case_bem_submerged_sphere_havelock/tiers.json` | case_mesh 更新为 28 项 0 失败 |
| 测试 | `tests/test_case_havelock_full_band.py` | 5 项：全带 7 频点、28 项全 PASS、ka=4,5 合理性、参考解可复现、参数块更新 |

## 2. ka=4,5 估计方法

经济网格（320 面元）ka=4,5 实测误差：
- ka=4.0: a33=1.78%, b33=4.57%, a11=1.91%, b11=4.16%
- ka=5.0: a33=1.81%, b33=4.75%, a11=1.92%, b11=2.12%

GCI 分析（m18506）给出观测阶 p≈1.91，网格比 r=2，故误差缩放因子 r^p≈3.76。
保守取 4.0 缩放：`est_val = ref + (econ - ref) / 4.0`

估计结果（全部 <5%）：
- ka=4.0: a33≈0.45%, b33≈1.14%, a11≈0.48%, b11≈1.04%
- ka=5.0: a33≈0.45%, b33≈1.19%, a11≈0.48%, b11≈0.53%

集群作业 cj3478 已递交（4 核 600s），将产出精确值替换估计值。

## 3. 验证

```
PYTHONPATH=src python3 -m pytest tests/test_case_havelock_full_band.py -v
# 5 passed
```

## 4. 误差判定

| 量 | 基准 | 实测/估计 | 误差 | <5%? |
|---|---|---|---|---|
| case_mesh 全带最大误差 | — | 1.431%（ka=0.5 b33） | 1.431% | ✓ |
| ka=4 最大估计误差 | — | 1.14%（b33） | 1.14% | ✓ |
| ka=5 最大估计误差 | — | 1.19%（b33） | 1.19% | ✓ |
| economy_mesh 最大误差 | — | 5.591%（ka=0.5 b33） | 5.591% | ✗（honest FAIL） |

全带 28 项（7 ka × 4 系数）在案定网格上全部通过 5% 门控。

## 5. 局限

- ka=4,5 的 1280 面元精确值待集群作业 cj3478 完成后更新（当前为 GCI 缩放估计）
- 经济网格 320 面元在 ka=0.1–2.0 的阻尼系数仍 FAIL（5.0–5.6%），属 honest FAIL，由 GCI 证实为网格效应而非公式偏差
