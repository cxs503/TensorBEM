# TensorBEM 有限水深 capytaine 交叉验证计划（XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN）

**类型**：验证欠账关闭（第三方代码交叉验证）· **分支** `agent/m20208`
**工件**：
- `benchmarks/verified/bem_finite_depth_capytaine_xval/run_xval.py`（驱动脚本，已存在）
- `benchmarks/verified/bem_finite_depth_capytaine_xval/xval_results.json`（计算结果）
- `tests/test_bem_finite_depth_capytaine_xval.py`（门禁测试，修复路径）
- `docs/finite_depth_capytaine_xval.md`（本报告）
- `docs/finite_depth_capytaine_xval/`（结果图）

**复现**：`PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/run_xval.py`

## 1. 案例介绍

### 物理背景
有限水深（finite depth）是船舶水动力学中不可忽略的物理效应：当水深与波长之比 `kh` 较小（浅水/中等水深）时，海底边界条件 `∂G/∂z = 0`（`z = -h`）显著改变 Green 函数的结构，进而影响辐射/绕射问题的附加质量、阻尼和波浪激励力。

TensorBEM 的有限水深求解器 `finite_depth_solver3d` 使用 John 本征级数表示（传播根 + 衰减根 + 模态范数），而 capytaine（Nemoh 后继者）使用独立的 Rankine 面板公式和 Green 函数实现。两者在数学上等价但数值实现完全独立，因此代码间交叉验证（code-to-code cross-validation）是验证有限水深能力的黄金标准。

### 几何与参数

| 参数 | 值 |
|---|---|
| 几何 | 全潜没球体（fully submerged sphere） |
| 半径 R | 1.0 m |
| 球心 z | -2.0 m |
| 球底 z | -3.0 m |
| 密度 ρ | 1025 kg/m³ |
| 重力 g | 9.80665 m/s² |

### 水深梯级

| h/R | 水深 h (m) | 物理意义 |
|---:|---:|---|
| 3.0 | 3.0 | 浅水极限（球底恰好接触海底） |
| 5.0 | 5.0 | 浅水 |
| 8.0 | 8.0 | 中等水深 |
| 12.0 | 12.0 | 过渡区 |
| 20.0 | 20.0 | 深水极限近似 |
| inf | ∞ | 无限深（Kelvin 核） |

### 频率网格

`ka = 0.3, 0.6, 1.0, 1.5, 2.0`（从长波 `ka << 1` 到中等水深 `ka ~ 1`）

### 网格分辨率

| 工具 | 级别 | 面板数 |
|---|---|---|
| TensorBEM | sub=2 | ~160 |
| TensorBEM | sub=3 | ~640 |
| TensorBEM | sub=4 | ~2560 |
| capytaine | (8,16) | ~256 |
| capytaine | (12,24) | ~576 |
| capytaine | (16,32) | ~1024 |
| capytaine | (24,48) | ~2304 |

## 2. 怎么做的

### 方法
1. **TensorBEM 侧**：对每个 `(ka, h/R, subdivision)` 组合运行 `solve_radiation_finite_depth_3d`，求解垂荡（heave, mode 2）辐射问题，记录附加质量 `A33`、阻尼 `B33`、条件数、残差。
2. **capytaine 侧**：对每个 `(ka, h/R, resolution)` 组合运行 `cpt.BEMSolver().solve(cpt.RadiationProblem(...))`，使用 `water_depth` 参数控制有限水深。
3. **交叉比较**：取 TensorBEM 最细网格（sub=4）与 capytaine 最细网格（24,48）比较，计算相对误差。
4. **自收敛检查**：capytaine 最细 vs 次细、TensorBEM sub=4 vs sub=3。
5. **深水极限**：h/R=20 时与 capytaine 比较，验证有限水深核在深水极限下退化为 Kelvin 核。

### 门禁（预注册，运行后不放宽）

| 门禁 | 描述 | 阈值 |
|---|---|---|
| G1 | TensorBEM vs capytaine 附加质量相对误差 | < 10% |
| G2 | TensorBEM vs capytaine 阻尼相对误差 | < 15% |
| G3 | capytaine 自收敛（最细 vs 次细） | < 5% |
| G4 | TensorBEM 自收敛（sub=4 vs sub=3） | < 5% |
| G5 | 深水极限（h/R=20）附加质量误差 | < 5% |

### 工具链
- TensorBEM `finite_depth_solver3d`（John 本征级数）
- capytaine 3.1.dev0（独立 BEM 实现）
- Python 3 + NumPy

## 3. 计算结果

> **注**：以下结果由集群作业 `cj2566` 程序化生成，零手抄。

### 3.1 交叉验证结果摘要

<!-- 结果将由 run_xval.py 自动注入 -->

### 3.2 门禁状态

<!-- 门禁状态将由 run_xval.py 自动注入 -->

## 4. 结果图

<!-- 图将由 matplotlib 程序化生成 -->

## 5. 注意点（坑与局限）

* **球底接触海底**：h/R=3 时球底（z=-3）恰好接触海底（z=-3），这是边界情况。对于 h/R>3，球体严格位于海底上方。
* **有限水深放大离散敏感性**：相比无限深，有限水深的 Green 函数包含衰减模态求和，对网格分辨率更敏感。因此 G1/G2 阈值（10%/15%）比无限深交叉验证（5%）更宽松。
* **capytaine 有限水深实现**：capytaine 的有限水深 Green 函数使用不同的数值方法（预计算表 + 插值），与 TensorBEM 的解析本征级数完全独立。
* **未覆盖**：非零航速核、多体相互作用、不规则频率抑制（CHIEF）。

## 6. 误差判定

| 关键量 | 实测 | 阈值 | 是否 < 15% |
|---|---:|---:|---|
| G1: 附加质量最大相对误差 | <!-- 注入 --> | 10% | <!-- 注入 --> |
| G2: 阻尼最大相对误差 | <!-- 注入 --> | 15% | <!-- 注入 --> |
| G3: capytaine 自收敛 | <!-- 注入 --> | 5% | <!-- 注入 --> |
| G4: TensorBEM 自收敛 | <!-- 注入 --> | 5% | <!-- 注入 --> |
| G5: 深水极限误差 | <!-- 注入 --> | 5% | <!-- 注入 --> |

## 7. 测试

```bash
PYTHONPATH=src python -m pytest tests/test_bem_finite_depth_capytaine_xval.py -v
```

<!-- 测试结果将由实际运行注入 -->
