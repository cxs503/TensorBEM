# Finite-Depth BEM Cross-Validation Plan (TensorBEM vs capytaine)

**Anchor**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`
**仓**: TensorBEM · **分支**: `agent/m20134`
**日期**: 2026-11-14

---

## 1. 物理背景与动机

三维有限水深自由面 Green 函数 $G(x,\xi;\nu,h)$ 满足：

- 自由面条件：$\partial_z G = \nu G$ at $z=0$，其中 $\nu = \omega^2/g$
- 海底条件：$\partial_z G = 0$ at $z=-h$
- 远场辐射条件：出射柱面波 $O(R^{-1/2})$

TensorBEM 的有限水深求解器（`finite_depth_solver3d.py`）基于 John (1950) 本征级数表示，
将 Green 函数分解为 Rankine 奇性部分 $1/(4\pi r)$ 加光滑余项，采用直接配点格式。

capytaine（v3.1.dev0）是国际公认的水动力 BEM 基准代码，其有限水深求解基于不同的数值路线
（波数积分 + 色散关系求根）。**两条独立路线在相同几何/参数下的一致性**是有限水深能力
最硬的验证证据——此前本仓的有限水深验证（`xval_bem_finite_depth/`）只有 3 个水深站、
2 个频率点，且 3 条门禁未通过（G4 模态截断、G8 频率网格、G11 对称性）。

本计划采用 capytaine 的交叉验证方法论：**多分辨率网格阶梯 + 水深阶梯 + 频率阶梯 + 预注册门禁**，
对完全淹没球体做辐射问题（heave/surge/sway）的系统性比对。

---

## 2. 几何与参数

### 2.1 几何

| 参数 | 值 | 说明 |
|---|---|---|
| 形体 | 球体 | 完全淹没，无自由面/海底穿透 |
| 半径 $a$ | 1.0 m | 特征长度 |
| 球心 $z_c$ | −2.0 m | 淹没深度 |
| 球底 $z_b$ | −3.0 m | 最浅水深站 h/R=3 时恰好触底 |

选择完全淹没球体的理由：
- 无需 CHIEF 点或水面板（无不规则频率问题）
- 有 Hulme (1982) 解析解可做三方分解（TensorBEM vs capytaine vs analytic）
- 轴对称性提供额外的对称性门禁（A11≈A22, A13≈0）

### 2.2 水深阶梯

| h/R | h (m) | 物理意义 |
|---:|---:|---|
| 3 | 3 | 浅水极限，球底距海底 0 m（边界情况） |
| 5 | 5 | 中等浅水，球底距海底 2 m |
| 8 | 8 | 过渡区 |
| 12 | 12 | 中等水深 |
| 20 | 20 | 深水近似（$k_0 h \approx 20$，$\tanh(k_0 h) \approx 1$） |
| ∞ | ∞ | 深水极限（解析基准） |

### 2.3 频率网格

| $ka$ | $k$ (1/m) | $\omega$ (rad/s) | 波长 $\lambda$ (m) | 区域 |
|---:|---:|---:|---:|---|
| 0.3 | 0.3 | 1.715 | 20.94 | 长波 |
| 0.6 | 0.6 | 2.426 | 10.47 | 中长波 |
| 1.0 | 1.0 | 3.132 | 6.28 | 中等 |
| 1.5 | 1.5 | 3.836 | 4.19 | 中短波 |
| 2.0 | 2.0 | 4.430 | 3.14 | 短波 |

### 2.4 网格阶梯

**TensorBEM**：icosphere 细分 2/3/4（160/640/2560 面板）

**capytaine**：(meridian, parallel) = (8,16)/(12,24)/(16,32)/(24,48)
（≈256/576/1024/2304 面）

四阶梯设计使得**裁判代码自身的离散化不确定性可测**，而非断言。

---

## 3. 方法

### 3.1 求解路线

| 路线 | 模块 | 核心方法 |
|---|---|---|
| TensorBEM | `finite_depth_solver3d` | John 本征级数 + 直接配点 + Rankine 奇性分裂 |
| capytaine | `BEMSolver` | 波数积分 + 色散关系求根 + 间接单层 |
| 解析 | Hulme (1982) | 球谐展开闭式解（仅深水） |

### 3.2 计算流程

```
for depth_ratio in [3, 5, 8, 12, 20, inf]:
    for ka in [0.3, 0.6, 1.0, 1.5, 2.0]:
        for mesh_level in TensorBEM_levels:
            run TensorBEM radiation (surge+sway+heave)
        for mesh_level in capytaine_levels:
            run capytaine radiation (surge+sway+heave)
        compare finest-vs-finest
        check self-convergence within each code
```

### 3.3 预注册门禁（run 前锁定，run 后不放宽）

| 门禁 | 判据 | 阈值 | 物理意义 |
|---|---|---:|---|
| G1 | TensorBEM-fine vs capytaine-fine A33 相对误差 | 10% | 附加质量跨代码一致性 |
| G2 | TensorBEM-fine vs capytaine-fine B33 相对误差 | 15% | 阻尼跨代码一致性（相位更敏感） |
| G3 | capytaine 自收敛：最细 vs 次细 A33 漂移 | 5% | 裁判代码离散化不确定性 |
| G4 | TensorBEM 自收敛：最细 vs 次细 A33 漂移 | 5% | 本代码离散化不确定性 |
| G5 | 深水极限 (h/R=20)：TensorBEM vs capytaine A33 | 5% | 有限水深→深水极限一致性 |
| G6 | capytaine 绕射能量守恒：$|R|^2+|T|^2$ vs 1 | 2% | 绕射问题能量守恒 |

---

## 4. 预期困难与缓解

### 4.1 浅水 h/R=3 的边界情况

球底恰好触底（$z_b = -3 = -h$），此时：
- TensorBEM 的网格生成可能产生穿透海底的面板 → 需在网格生成时做 z 裁剪
- capytaine 的 `water_depth=3.0` 对触底球体的处理需验证

**缓解**：在结果中单独标注 h/R=3 站，若误差显著增大则归因于几何边界而非核函数。

### 4.2 模态截断

有限水深 Green 函数的衰减根 $\mu_n$ 满足 $\mu_n \tan(\mu_n h) = -\nu$，
模态数随 $\nu h$ 增大而增多。截断准则：省略模态的衰减因子 $e^{-\mu_n d} < e^{-10}$，
其中 $d$ 是最近目标/求积点间距。

**风险**：浅水高频时模态数可能爆炸 → 设硬上限 `_MODES_CAP = 4096`。

### 4.3 计算成本

总工况数：6 水深 × 5 频率 × (4+4) 网格 = 240 次求解（不含三方分解）。
单次 TensorBEM 求解（sub=4）约 30-60 s → 总计算时间约 2-4 小时（并行）。

---

## 5. 结果发布格式

### 5.1 JSON 工件

`benchmarks/verified/bem_finite_depth_capytaine_xval/xval_results.json`：
- `tbem_study.rows[]`：TensorBEM 全网格结果
- `cpt_study.levels[]`：capytaine 四阶梯结果
- `comparison.comparison[]`：逐点对比
- `comparison.gates{}`：门禁状态

### 5.2 文档

`docs/xval_bem_finite_depth_plan.md`（本文件）：计划与方法
`docs/xval_bem_finite_depth_results.md`：结果与误差分析（程序化生成）

### 5.3 图

- `docs/figures/xval_fd_am_heave_vs_ka.png`：附加质量 vs $ka$（多水深曲线）
- `docs/figures/xval_fd_b_heave_vs_ka.png`：阻尼 vs $ka$（多水深曲线）
- `docs/figures/xval_fd_error_heatmap.png`：相对误差热力图（水深×频率）
- `docs/figures/xval_fd_convergence.png`：自收敛曲线

---

## 6. 与既有工作的关系

| 工件 | 状态 | 与本计划的关系 |
|---|---|---|
| `xval_bem_finite_depth/` | 3 门禁未通过 | 前身，网格/频率覆盖不足 |
| `bem_finite_depth/` | 已验证（解析恒等式） | 无第三方代码 |
| `capytaine_crosscheck/` | 已验证（深水） | 仅深水，无有限水深 |
| `casepool_bem_hulme_hemisphere/` | 已验证 | 半球（非完全淹没），深水 |

本计划是**首次**在有限水深条件下对 TensorBEM 与 capytaine 做系统性跨代码比对。

---

## 7. 复现

```bash
# 完整交叉验证（需 capytaine 已安装）
cd /root/tp-agent/work/m20134
PYTHONPATH=src python3 benchmarks/verified/bem_finite_depth_capytaine_xval/run_xval.py

# 仅读冻结结果跑测试
PYTHONPATH=src python3 -m pytest tests/test_bem_finite_depth_capytaine_xval.py -v

# 生成报告图
PYTHONPATH=src python3 benchmarks/verified/bem_finite_depth_capytaine_xval/make_figures.py
```

---

## 8. 局限

1. **仅辐射问题**：绕射问题的跨代码比对需额外工作（G6 能量守恒是 capytaine 单侧检查）
2. **仅球体**：复杂船体的有限水深跨代码验证是下一步
3. **零航速**：有航速有限水深 Green 函数（Neumann-Kelvin 有限水深版本）尚未实现
4. **h/R=3 边界**：球底触底的物理/数值处理需单独讨论
