# TensorBEM m16830 — Neumann-Kelvin 模块验证测试扩展

**仓库** TensorBEM · **分支** `agent/m16830` · **基线** `main @ 2026-11-12`
**锚点** NEW: `neumann_kelvin` 模块缺 `verify_michell_limit()` 函数测试、缺 Green 函数矢量化/标量版本一致性检查、缺 `_wave_integral_steady` 直接测试、缺已知数值问题文档化

## 1. 案例介绍

`tensorbem.neumann_kelvin` 模块实现了 Neumann-Kelvin 稳态前进速度波阻验证面。
模块已有 26 项测试覆盖常数、Green 函数定性行为、Kelvin 波型、Michell 极限、
求解器基本返回等，但存在以下验证欠账：

| 缺口 | 说明 |
|---|---|
| `verify_michell_limit()` | 函数被导入但从未在测试中调用 |
| `green_function_nk_vectorized` vs `_wave_integral_steady` | 矢量化版本使用 K_0 渐近近似，与标量数值积分版本差异未量化 |
| `_wave_integral_steady` | 波积分函数无直接测试 |
| 已知数值问题 | 求解器产生非物理的大波阻值（Cw ~ 1e8 而非 ~1e-3），且随速度振荡 |

## 2. 怎么做的

### 2.1 验证设计

新增测试文件 `tests/test_neumann_kelvin_verification.py`（27 项测试，6 个测试类）：

| 测试类 | 验证内容 | 判据 |
|---|---|---|
| `TestVerifyMichellLimit` | `verify_michell_limit()` 结构属性 | 返回 dict 含所有键、Michell 参考值为正、Fn 正确 |
| `TestGreenFunctionConsistency` | 矢量化/标量 Green 函数 | 两者有限、矢量化形状正确、标量可为负（文档化） |
| `TestWaveIntegralSteady` | 波积分直接测试 | 有限、幅值随 R/H 衰减、零点处理、振荡性 |
| `TestSolverStructuralProperties` | 求解器结构属性 | Rw≥0、Cw≥0、条件数有限、残差小、Fn 正确 |
| `TestKelvinAngleEdgeCases` | Kelvin 角验证边界 | 无波峰时返回 failed、理论角始终报告 |
| `TestKnownNumericalIssues` | 已知数值问题文档化 | Cw 非物理地大、Rw 随速度振荡 |

### 2.2 关键发现

**发现 1：标量 Green 函数可为负值**
- 配置 `target=[0.5, 0.3, -0.2], source=[0.0, 0.0, -0.5], kappa=5.0` 时
- `green_function_nk()` 返回 `-2.05e-2`（负值）
- 原因：波积分 G_W 为负且幅值超过 Rankine 部分 G_R
- 影响：BEM 矩阵可能出现非物理的负系数

**发现 2：矢量化 Green 函数与标量版本差异巨大**
- 相同配置下，矢量化版本返回 `2.10e-1`（正值）
- 相对误差 > 1000%
- 原因：矢量化版本使用 K_0 渐近近似，完全不同于标量的数值积分

**发现 3：波阻值非物理地大**
- Wigley  hull (L=2, B=0.2, T=0.125) 在 Fn=0.113 时 Cw ≈ 4.9e8
- 物理预期：Cw ~ O(1e-3)
- 偏差 > 11 个数量级

**发现 4：波阻随速度振荡**
- U=0.3: Rw=1.98e9, U=0.5: Rw=1.58e10, U=0.8: Rw=0, U=1.0: Rw=0, U=1.2: Rw=1.15e10
- 在巨大值和零之间交替，表明数值不稳定

## 3. 计算结果

### 3.1 测试统计

| 指标 | 值 |
|---|---|
| 新增测试数 | 27 |
| 通过率 | 100% (27/27) |
| 原有测试通过率 | 100% (26/26) |
| 总测试数 | 53 |

### 3.2 已知数值问题量化

| U | Fn | Rw | Cw | 条件数 |
|---|---|---|---|---|
| 0.3 | 0.068 | 1.98e9 | 1.72e8 | 2.00e4 |
| 0.5 | 0.113 | 1.58e10 | 4.94e8 | 3.93e4 |
| 0.8 | 0.181 | 0.0 | 0.0 | 9.98e4 |
| 1.0 | 0.226 | 0.0 | 0.0 | 4.64e4 |
| 1.2 | 0.271 | 1.15e10 | 6.24e7 | 3.91e4 |
| 1.5 | 0.339 | 0.0 | 0.0 | 3.23e4 |

## 4. 结果图

无图。本任务为测试扩展和数值审计，不涉及场量可视化。

## 5. 注意点

- 本测试文件**不修复**已发现的数值问题，而是**文档化**它们
- `TestKnownNumericalIssues` 类中的测试验证问题存在（而非正确行为）
- 矢量化 Green 函数的 K_0 近似需要重新审视——它可能不适合 Neumann-Kelvin 问题
- 建议后续工作：(1) 修复矢量化 Green 函数使其与标量版本一致；(2) 调查波阻非物理大的根因

## 6. 误差判定

本任务为测试扩展，不涉及新计算结果的误差判定。
已文档化的数值问题：
- Cw 偏差 > 11 个数量级（已知问题，未修复）
- 标量/矢量化 Green 函数差异 > 1000%（已知问题，未修复）
