# Ellipsoid Added Mass Benchmark: TensorBEM vs Lamb (1932)

## Case M17177 · TensorBEM Ecosystem · Benchmark Pool

---

## 1. 案例介绍（Problem Statement & Reference）

### 物理背景

附加质量（added mass）是流体惯性效应的核心表征量。当物体在无粘、不可压缩流体中加速运动时，必须同时加速周围流体，等效于增加了物体质量。对于椭球体，Lamb (1932) §114 给出了基于形状因子 α_i 的闭式解。

### 几何参数

**三轴椭球体（Tri-axial ellipsoid）**：
- 半轴长度：a = 1.5 m (x-axis), b = 1.0 m (y-axis), c = 0.75 m (z-axis)
- 长宽比：a/b = 1.5, b/c = 1.333
- 体积：V = (4/3)πabc = 4.7124 m³
- 流体密度：ρ = 1025 kg/m³（海水）

### 理论参考（Reference）

**Lamb, H. (1932). Hydrodynamics, 5th edition. Cambridge University Press.**

- **章节**：§114 "Motion of an ellipsoid through an infinite fluid"
- **关键公式**：
  - 形状因子（shape factors）α_i 满足：α₁ + α₂ + α₃ = 2（精确恒等式）
  - 附加质量：A_ii = ρ·V·α_i / (2 - α_i)
  - 形状因子通过椭圆积分计算：
    ```
    α_i = a₁a₂a₃ ∫₀^∞ dλ / [(a_i² + λ) √∏ⱼ(a_j² + λ)]
    ```

**零手抄声明**：所有参考数值均由 `tensorbem.analytic.ellipsoid_shape_factors()` 从上述积分公式数值计算得出，使用 Gauss-Legendre 求积（64 节点），并验证 α₁+α₂+α₃=2 残差 < 10⁻⁸。

---

## 2. 方法与实现（Methodology）

### 数值方法

**Boundary Element Method (BEM)** — 基于 Laplace 方程 exterior Dirichlet 问题：

1. **控制方程**：∇²φ = 0（流体域，无旋不可压）
2. **边界条件**：φ = -U·k·x_i（椭球表面，k 为精确线性迹）
3. **积分方程**：间接 BEM，单层势 + 双层势组合
4. **附加质量计算**：边界能量积分
   ```
   A_ii = (ρ/U²) ∮_S φ · (∂φ/∂n) dS
   ```

### 工具链

- **求解器**：`tensorbem.solver.solve_laplace_surface(domain="exterior")`
- **网格生成**：`tensorbem.geometry3d.ellipsoid_mesh()` — 基于二十面体细分（icosphere topology）
- **解析参考**：`tensorbem.analytic.ellipsoid_shape_factors()` + `ellipsoid_added_mass()`
- **后处理**：NumPy + Matplotlib

### 关键设置

- **网格拓扑**：Icosphere 细分，flat triangular panels
- **细分层级**：subdivisions = 0, 1, 2（对应 20, 80, 320 panels）
- **积分阶数**：Gauss-Legendre 64 节点（形状因子）
- **线性求解器**：NumPy dense LU（直接法）
- **CHIEF 点**：自动 interior points（抑制虚假特征值）

---

## 3. 参数表（Parameter Table）

### 无量纲参数

| 参数 | 符号 | 数值 | 说明 |
|------|------|------|------|
| 长宽比 | a/b | 1.500 | x-y 平面 |
| 长宽比 | b/c | 1.333 | y-z 平面 |
| 雷诺数 | Re | ∞ | 无粘假设 |
| Froude 数 | Fr | 0 | 无自由面 |
| 无量纲频率 | ω·a/U | 0 | 无限频率极限（纯惯性） |

### 网格参数

| 细分层级 | 面板数 | 顶点数 | h_max (m) | h_mean (m) | 体积比 V_mesh/V_exact |
|----------|--------|--------|-----------|------------|----------------------|
| 0 | 20 | 12 | 0.9747 | 0.7123 | 0.6055 |
| 1 | 80 | 42 | 0.4874 | 0.3562 | 0.8735 |
| 2 | 320 | 162 | 0.2437 | 0.1781 | 0.9662 |

### Lamb 形状因子（解析值）

| 轴 | α_i（数值积分） | 理论约束 |
|----|----------------|----------|
| x | 0.4009275746 | α₁+α₂+α₃ = 2.0000000000 |
| y | 0.6707129178 | ✓ 验证通过 |
| z | 0.9283595076 | 残差 < 10⁻¹⁰ |

---

## 4. 对照表（Results: BEM vs Lamb）

### 附加质量对比（Subdivision 2, 320 panels）

| 运动轴 | BEM 计算值 (kg) | Lamb 解析值 (kg) | 相对误差 | 判定 (<5%) |
|--------|----------------|-----------------|----------|-----------|
| X (surge) | 1191.30 | 1211.05 | **1.631%** | ✓ PASS |
| Y (sway) | 2394.82 | 2437.15 | **1.737%** | ✓ PASS |
| Z (heave) | 4104.55 | 4184.39 | **1.908%** | ✓ PASS |

### 通量精度（Normal derivative accuracy）

| 运动轴 | 通量 L2 误差 | 通量 RMSE | 通量 max abs error |
|--------|-------------|-----------|-------------------|
| X | 2.729% | 0.01168 | 0.03390 |
| Y | 2.842% | 0.01692 | 0.03496 |
| Z | 3.348% | 0.02545 | 0.07029 |

### 完整收敛历程

| 细分 | 面板数 | X-axis 误差 | Y-axis 误差 | Z-axis 误差 |
|------|--------|------------|------------|------------|
| 0 | 20 | 20.855% | 21.126% | 22.541% |
| 1 | 80 | 6.162% | 6.473% | 7.020% |
| 2 | 320 | **1.631%** | **1.737%** | **1.908%** |

**结论**：所有三个轴向的附加质量误差均 < 5%，满足容差判据。

---

## 5. 收敛性与参数敏感性（Convergence & Sensitivity）

### 网格收敛性

**观察到的收敛阶**：

从 sub=0 到 sub=2，误差从 ~21% 降至 ~1.7%，收敛速率约 O(h²)，符合 flat triangular panel 的理论预期。

**体积逼近**：
- sub=0: V_mesh/V_exact = 0.605（粗网格，内接多面体）
- sub=1: V_mesh/V_exact = 0.873
- sub=2: V_mesh/V_exact = 0.966（接近真值）

**关键发现**：附加质量误差与体积误差强相关——几何逼近精度主导了整体精度。

### 参数敏感性

**形状因子敏感性**：
- α_i 的数值积分使用 64 节点 Gauss-Legendre，残差 < 10⁻¹⁰
- 降低积分节点至 32 时，残差增至 ~10⁻⁶，但仍满足工程精度
- 对于极端长细比（a/c > 10），需增加积分节点或改用自适应求积

**网格拓扑敏感性**：
- Icosphere 细分保证各向同性面板分布
- 若使用参数化网格（如 latitude-longitude），极点附近面板畸变会导致精度下降
- 当前实现避免了参数化奇点

### 条件数诊断

- 单层势条件数：κ(S) ≈ 44.7（sub=2）
- 双层势行和误差：< 10⁻¹⁵（机器精度）
- 线性系统残差：< 10⁻¹⁵（直接求解器）

**结论**：数值稳定性良好，无条件数恶化问题。

---

## 6. 失败模式与工件（Failure Modes & Artifacts）

### 已知局限

1. **几何逼近误差**：Flat panels 内接于真椭球，体积偏小 → 附加质量偏小
   - 缓解：增加细分层级，或使用 curved panels（二次单元）
   
2. **无自由面效应**：本算例为无限域无自由面问题
   - 实际船舶附加质量需考虑自由面、有限水深、粘性效应
   - 扩展：使用 `solve_radiation_3d()` 计算有限频率下的附加质量

3. **无粘性假设**：BEM 无法捕捉边界层分离、涡脱落
   - 对于钝体或大振幅运动，需耦合 RANS/LES

4. **计算成本**：Dense BEM，O(N³) 复杂度
   - sub=2 (320 panels) 需 ~10s
   - sub=3 (1280 panels) 预计 ~3min
   - 大规模问题需 FMM 加速

### 生态摩擦（Ecosystem Friction）

- **无**：TensorBEM 已具备完整的椭球附加质量能力
- **复用**：直接调用 `tensorbem.benchmarks.ellipsoid_added_mass.run()`
- **文档**：解析公式在 `tensorbem.analytic` 模块文档字符串中完整引用 Lamb (1932)

### 工件清单

| 工件 | 路径 | 说明 |
|------|------|------|
| BEM 结果 (sub=2) | `artifacts/ellipsoid_added_mass_sub2.json` | 完整数值结果 |
| BEM 结果 (sub=0) | `artifacts/ellipsoid_sub0.json` | 粗网格 |
| BEM 结果 (sub=1) | `artifacts/ellipsoid_sub1.json` | 中等网格 |
| 几何图 | `artifacts/figures/ellipsoid_geometry.png` | 三轴椭球体可视化 |
| 对比图 | `artifacts/figures/added_mass_comparison.png` | BEM vs Lamb 柱状图 |
| 收敛图 | `artifacts/figures/convergence_study.png` | 网格收敛曲线 |
| 形状因子图 | `artifacts/figures/shape_factors.png` | α_i 验证 |
| 复现脚本 | `run_ellipsoid_benchmark.py` | 一键复现 |
| 出图脚本 | `create_report_figures.py` | 生成所有图表 |

### 复现命令

```bash
cd /root/tp-agent/work/m17177
export PYTHONPATH=/root/tp-agent/work/m17177/src:$PYTHONPATH

# 运行基准算例（subdivision=2, 320 panels）
python -m tensorbem.benchmarks.ellipsoid_added_mass \
  --subdivisions 2 \
  --semi-axes 1.5 1.0 0.75 \
  --density 1025.0 \
  --output artifacts/ellipsoid_added_mass_sub2.json

# 生成报告图表
python create_report_figures.py
```

---

## 误差判定总结

| 关键量 | 基准 | 误差 | 是否 < 3% | 是否 < 5% |
|--------|------|------|----------|----------|
| A₁₁ (surge) | Lamb (1932) §114 | 1.631% | ✓ | ✓ |
| A₂₂ (sway) | Lamb (1932) §114 | 1.737% | ✓ | ✓ |
| A₃₃ (heave) | Lamb (1932) §114 | 1.908% | ✓ | ✓ |
| α₁+α₂+α₃ | 2.0 (exact) | < 10⁻¹⁰ | ✓ | ✓ |

**最终判定**：✓ **ALL PASS** — 所有轴向附加质量误差 < 5%，满足算例池容差判据。

---

## 参考文献

1. Lamb, H. (1932). *Hydrodynamics*, 5th edition. Cambridge University Press. Section 114, pp. 152-156.
2. Newman, J.N. (1977). *Marine Hydrodynamics*. MIT Press. Section 4.10.
3. TensorBEM analytic module documentation: `tensorbem.analytic.ellipsoid_shape_factors()`

---

**报告生成时间**：2026-11-13  
**计算平台**：TensorBEM v0.5.0, Python 3.12, NumPy 1.26  
**任务编号**：M17177 · CASE-BEM-ELLIPSOID-ADDED-MASS
