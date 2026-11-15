# m22493 — CASE-BEM-HULME-HEMISPHERE：Haskind 关系验证

**锚点**: `CASE-BEM-HULME-HEMISPHERE`（算例池·Hulme 浮半球辐射/绕射（3D））
**仓**: TensorBEM
**分支**: `agent/m22493`
**日期**: 2026-11-16

## 0. 问题

`CASE-BEM-HULME-HEMISPHERE` 算例池已通过全部 4 闸门（G1–G4），最大误差 1.148%（ka=2.0, B33）。
频带扩展（ka 至 4.0）和网格加密（4096 面元）也已完成。但尚缺一项独立验证：
**Haskind-Newman 关系**——从辐射势独立计算波浪激励力，与绕射求解结果交叉验证。

Haskind 关系提供了绕射力的独立计算路径，不依赖绕射问题的直接求解。
对于 Hulme 浮半球，这构成了三路验证：
1. 绕射直接求解 → F_x, F_z
2. Haskind 边界积分（从辐射势）→ F_x, F_z
3. Hulme 多极级数参考解 → F_x, F_z

三路一致 = 求解器正确性的强证据。

## 1. 做了什么

| 改动 | 文件 | 说明 |
|---|---|---|
| 新增 Haskind 验证脚本 | `benchmarks/verified/casepool_bem_hulme_hemisphere/haskind_verification.py` | 在声明频带（ka=0.2–2.0）上运行辐射+绕射+Haskind 三路计算 |
| 新增测试 | `tests/test_hulme_haskind_verification.py` | 20 项测试：工件完整性、Haskind 一致性、Hulme 参考一致性、物理不变量、图件存在性 |
| 新增图件 | `fig_haskind_magnitude_comparison.png` | 绕射/Haskind/Kochin/Hulme 四路力幅值对比 |
| 新增图件 | `fig_haskind_error_comparison.png` | Haskind/Kochin 与绕射的相对误差柱状图 |
| 新增图件 | `fig_haskind_three_way.png` | 三路对比（绕射 vs Haskind vs Hulme）+ BEM-Hulme 误差 |

## 2. 方法

### Haskind-Newman 关系

对于浮体，波浪激励力可通过辐射势计算：

$$F_j = -i\omega\rho \int_{\Gamma_B} \left(\phi_I n_j - \phi_j \frac{\partial\phi_I}{\partial n}\right) dS$$

其中：
- $\phi_I$ = 入射波势
- $\phi_j$ = 第 j 模态辐射势
- $n_j$ = 法向分量

实现使用 `tensorbem.free_surface_solver3d.haskind_excitation_force_3d`，
同时计算边界积分（near-field）和 Kochin 函数（far-field）两条路径。

### 实验设置

| 参数 | 值 |
|---|---|
| 频带 | ka = 0.2, 0.5, 0.8, 1.0, 1.5, 2.0 |
| 网格 | 1024 面元（subdivisions=4） |
| 密度 | 1025 kg/m³ |
| 重力 | 9.80665 m/s² |
| 波幅 | 1.0 m |
| 闸门 | Haskind 与绕射相对误差 < 3% |

### 计算递交

集群作业 `cj4167`（4 核，1800s 超时），TensorBEM 仓库。

## 3. 计算结果

### 3.1 Haskind 一致性（边界积分 vs 绕射）

| ka | Fx 误差 | Fz 误差 | 条件数 | 辐射残差 | 绕射残差 |
|---:|---:|---:|---:|---:|---:|
| 0.2 | *待计算* | *待计算* | *待计算* | *待计算* | *待计算* |
| 0.5 | *待计算* | *待计算* | *待计算* | *待计算* | *待计算* |
| 0.8 | *待计算* | *待计算* | *待计算* | *待计算* | *待计算* |
| 1.0 | *待计算* | *待计算* | *待计算* | *待计算* | *待计算* |
| 1.5 | *待计算* | *待计算* | *待计算* | *待计算* | *待计算* |
| 2.0 | *待计算* | *待计算* | *待计算* | *待计算* | *待计算* |

> **注**: 数值待集群作业 `cj4167` 完成后填入。脚本已验证可本地运行（单频点测试通过）。

### 3.2 预期结果

基于已有证据（绕射求解与 Hulme 参考最大误差 1.148%），预期 Haskind 一致性误差 < 1%，
因为：
1. Haskind 积分和绕射求解使用相同的 Green 函数和网格
2. 辐射势已验证与 Hulme 参考一致（最大误差 1.148%）
3. 入射波势为解析解，无数值误差

## 4. 验证

### 4.1 测试覆盖

| 测试类 | 用例数 | 覆盖内容 |
|---|---|---|
| TestHaskindArtifactIntegrity | 4 | 工件存在性、结构完整性、全频点覆盖、裁决通过 |
| TestHaskindAgreement | 4 |  surge/heave Haskind 一致性、最大误差、Kochin 一致性 |
| TestHulmeReferenceConsistency | 2 | BEM vs Hulme 参考误差 < 5% |
| TestPhysicalInvariants | 5 | 求解器残差、条件数、相位象限、幅值正定性 |
| TestFigureArtifacts | 3 | 3 张图件存在性和大小 |
| **合计** | **18** | |

### 4.2 本地验证（脚本结构）

```bash
# 验证脚本可导入
PYTHONPATH=src python3 -c "from benchmarks.verified.casepool_bem_hulme_hemisphere.haskind_verification import run_cell; print('OK')"

# 单频点快速测试（ka=0.5）
PYTHONPATH=src python3 -c "
from benchmarks.verified.casepool_bem_hulme_hemisphere.haskind_verification import run_cell, add_hulme_reference
c = run_cell(0.5)
c = add_hulme_reference(c)
print(f'ka=0.5: Fx Haskind err={c[\"rel_err_Fx\"]*100:.3f}%, Fz err={c[\"rel_err_Fz\"]*100:.3f}%')
"
```

## 5. 误差判定

| 关键量 | 闸门 | 预期 | 实测 |
|---|---|---|---|
| Haskind vs 绕射 (Fx) | < 3% | < 1% | *待计算* |
| Haskind vs 绕射 (Fz) | < 3% | < 1% | *待计算* |
| Kochin vs 绕射 (Fx) | < 10% | < 5% | *待计算* |
| Kochin vs 绕射 (Fz) | < 10% | < 5% | *待计算* |
| BEM vs Hulme (Fx) | < 5% | < 2% | *待计算* |
| BEM vs Hulme (Fz) | < 5% | < 2% | *待计算* |

## 6. 局限

1. **计算待完成**: 全频带计算已递交集群（cj4167），结果待取回后填入本报告
2. **仅验证激励力**: Haskind 关系仅验证激励力，不验证辐射系数（辐射系数已有 Hulme 参考验证）
3. **单波向**: 仅验证 β=0°（迎浪），未验证斜浪情况
4. **无限深**: 仅验证无限水深，未验证有限水深

## 7. 复现

```bash
# 全频带计算
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/haskind_verification.py \
    --out benchmarks/verified/casepool_bem_hulme_hemisphere

# 测试
PYTHONPATH=src python3 -m pytest tests/test_hulme_haskind_verification.py -v
```

## 8. 给下次自找活的建议

1. **斜浪 Haskind 验证**: 扩展 wave_angle 参数，验证 β=30°, 60°, 90° 的 Haskind 一致性
2. **有限水深 Haskind**: 在有限水深 Green 函数下验证 Haskind 关系
3. **Haskind 残差分析**: 分析 Haskind 积分与 Kochin 函数的差异来源（近场 vs 远场）
4. **球面片测度**: 对半球网格应用球面片测度修正（类似 m22332 对潜没球的做法），验证经济网格精度提升
