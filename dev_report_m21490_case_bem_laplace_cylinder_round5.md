# CASE-BEM-LAPLACE-CYLINDER 第 5 轮（测试修复+记录刷新递交）— TensorBEM m21490

**锚点** CASE-BEM-LAPLACE-CYLINDER（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** agent/m21490

## ① 问题陈述

锚点 `CASE-BEM-LAPLACE-CYLINDER` 在 feature_ledger 中状态为 `open`，验收标准为：
「算例池|BEM-LAPLACE-CYLINDER｜ref: 解析复势｜容差: <3%」。

前序工作（m21425 第 4 轮）已实现完整的算例模块（`case_cylinder_stream.py`，~3400 行），
覆盖 44 条预置判据（路线 D/N/Γ/M + 近壁线性密度 + 方位角扫描 + 几何刻面残差归因）。

但存在两个欠账：
1. **记录陈旧**：`results.json` 仅含 38 条判据，`provenance` 为 `"NONE"`，无图件
2. **测试缺陷**：`test_near_wall_half_panel_rows_carry_the_panel_size` 因 `factors` 参数
   缺少 `NEAR_WALL_WINDOW_FACTOR` (2.0) 和 `NEAR_WALL_EXCLUSION_FACTOR` (0.1) 而报 `KeyError`

## ② 本轮工作

### 2.1 测试缺陷修复

**根因**：`near_wall_study()` 函数内部使用 `error_at(level, NEAR_WALL_WINDOW_FACTOR)` 和
`error_at(level, NEAR_WALL_EXCLUSION_FACTOR)` 查询特定 clearance 水平的误差，但测试传入的
`factors=(1.0, 0.5)` 不包含这两个值，导致 `KeyError: 2.0`。

**修复**：将测试的 `factors` 参数改为包含必需的 clearance 水平：
```python
study = case.near_wall_study(
    ladder=(16, 32),
    factors=(case.NEAR_WALL_WINDOW_FACTOR, 1.0, 0.5, case.NEAR_WALL_EXCLUSION_FACTOR),
)
```

**验证**：`pytest tests/test_case_cylinder_stream.py::test_near_wall_half_panel_rows_carry_the_panel_size -v` → PASSED

### 2.2 记录重生成（递交调度）

已递交集群作业 `cj3470`（8 核，3600s 超时）：
```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir benchmarks/verified/case_bem_laplace_cylinder \
  --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check
```

该命令将生成：
- `results.json`（含 44 条判据 + provenance）
- `report.md`
- `case.json`
- `figures/` 目录（5 张图：field.png, convergence.png, near_wall.png, multipole.png, circulation.png）

### 2.3 现有记录数值复核（基于已提交 results.json, N=128）

| 关键量 | got | ref（解析复势）| 误差 | <3% |
| --- | --- | --- | --- | --- |
| 路线 D 外部速度 $L_2$ | — | $dW/dz$ | 1.366e-07 | 是 |
| 路线 D 扰动势 $L_2$ | — | $Ua^2\cos\theta/r$ | 3.578e-07 | 是 |
| 路线 N 物面速度 $L_2$ | — | $2U|\sin\theta|$ | 1.008e-04 | 是 |
| 路线 N 外部速度 $L_2$ | — | $dW/dz$ | 1.532e-04 | 是 |
| 路线 N $C_p$ $L_2$ | — | $1-4\sin^2\theta$ | 2.849e-04 | 是 |
| 零法向流动 $\max|v\cdot n|/U$ | 6.025e-04 | 0 | 同上 | 是 |
| $C_p$ 极小值 | -2.999194 | -3.000000 | 8.060e-04 | 是 |
| 远场 $r=8a$ 速度 | — | $dW/dz$ | 5.591e-09 | 是 |
| 环量路线升力相对误差 | — | Kutta-Joukowski | 4.014e-04 | 是 |
| 驻点角误差 | — | $\sin^{-1}(\Gamma/4\pi Ua)$ | 0.006° | 是 |
| 路线 M 偶极矩 $A_1$ | — | 1.0 | 3.578e-07 | 是 |
| 路线 M 四极矩 $A_2$ | — | 1.0 | 1.048e-04 | 是 |

38 条既有判据全部 PASS，最差误差 2.849e-04（<< 3%）。

### 2.4 收敛性（已提交记录）

| $N$ | $h_{mean}/a$ | D 速度 | D 势 | N 物面速度 | N 外部速度 | N $C_p$ |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.960e-01 | 2.266e-04 | 2.334e-05 | 1.633e-03 | 2.565e-03 | 4.601e-03 |
| 64 | 9.814e-02 | 1.229e-06 | 2.998e-06 | 4.046e-04 | 6.131e-04 | 1.143e-03 |
| 128 | 4.908e-02 | 1.366e-07 | 3.578e-07 | 1.008e-04 | 1.532e-04 | 2.849e-04 |

路线 N 物面收敛阶 [2.017, 2.007]（二阶，符合常值面板理论预期）。

## ③ 待完成项（集群作业 cj3470）

作业完成后需验证：
- `overall=PASS`
- 44 条判据（含 linear/azimuth/faceting 节）
- `provenance.generator` 存在且与当前源码 sha256 一致
- `benchmarks/verified/case_bem_laplace_cylinder/figures/` 含 5 张图件
- `test_committed_record_was_rendered_from_the_current_case_module` 由 skip 转 pass
- `test_committed_case_artifact_matches_the_committed_document` 通过

## ④ 结论

- 测试缺陷已修复（`test_near_wall_half_panel_rows_carry_the_panel_size` 的 `factors` 参数）
- 记录重生成已递交调度（cj3470），待集群资源可用后自动完成
- 本锚点功能实质已就绪，数值结果与解析复势基准误差远优于 3% 容差（最差 2.849e-04）
- 所有关键量误差均 <3%，收敛阶二阶符合理论预期
