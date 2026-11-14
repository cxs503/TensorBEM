# 有限水深 BEM 交叉验证：TensorBEM vs capytaine（m20134）

**锚点**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`
**仓**: TensorBEM · **分支**: `agent/m20134`
**日期**: 2026-11-14
**计算作业**: cj2511（集群递交中）

---

## 1. 案例介绍

### 1.1 物理背景

三维有限水深自由面 Green 函数 $G(x,\xi;\nu,h)$ 是水动力 BEM 的核心核函数，满足：
- 自由面条件：$\partial_z G = \nu G$ at $z=0$（$\nu = \omega^2/g$）
- 海底条件：$\partial_z G = 0$ at $z=-h$
- 远场辐射条件：出射柱面波 $O(R^{-1/2})$

TensorBEM 采用 John (1950) 本征级数表示 + 直接配点格式；capytaine（v3.1.dev0）
采用波数积分 + 色散关系求根路线。**两条独立数值路线在相同几何/参数下的一致性**
是有限水深能力最硬的验证证据。

### 1.2 几何

| 参数 | 值 |
|---|---|
| 形体 | 完全淹没球体 |
| 半径 $a$ | 1.0 m |
| 球心 $z_c$ | −2.0 m |
| 球底 $z_b$ | −3.0 m |

### 1.3 参数表

**水深阶梯**：h/R = 3, 5, 8, 12, 20, ∞
**频率网格**：ka = 0.3, 0.6, 1.0, 1.5, 2.0
**TensorBEM 网格**：icosphere sub 2/3/4（160/640/2560 面板）
**capytaine 网格**：(8,16)/(12,24)/(16,32)/(24,48)（≈256/576/1024/2304 面）

---

## 2. 方法

### 2.1 求解路线对比

| 路线 | 模块 | 核心方法 |
|---|---|---|
| TensorBEM | `finite_depth_solver3d` | John 本征级数 + 直接配点 + Rankine 奇性分裂 |
| capytaine | `BEMSolver` | 波数积分 + 色散关系求根 + 间接单层 |

### 2.2 预注册门禁

| 门禁 | 判据 | 阈值 |
|---|---|---:|
| G1 | TensorBEM-fine vs capytaine-fine A33 相对误差 | 10% |
| G2 | TensorBEM-fine vs capytaine-fine B33 相对误差 | 15% |
| G3 | capytaine 自收敛（最细 vs 次细 A33） | 5% |
| G4 | TensorBEM 自收敛（最细 vs 次细 A33） | 5% |
| G5 | 深水极限 (h/R=20) A33 跨代码误差 | 5% |

### 2.3 工具链

- 生产者：`benchmarks/verified/bem_finite_depth_capytaine_xval/run_xval.py`
- 图生成：`benchmarks/verified/bem_finite_depth_capytaine_xval/make_figures.py`
- 测试：`tests/test_bem_finite_depth_capytaine_xval.py`
- 结果工件：`benchmarks/verified/bem_finite_depth_capytaine_xval/xval_results.json`

---

## 3. 计算结果

**状态**：计算作业 cj2511 已递交集群，结果待回填。

### 3.1 预期结果结构

```json
{
  "tbem_study": {"rows": [...], "seconds": ...},
  "cpt_study": {"levels": [...], "seconds": ...},
  "comparison": {
    "comparison": [...],
    "gates": {
      "G1_tbem_cpt_am_heave": {"passed": bool, "max_error": float, "threshold": 0.10},
      "G2_tbem_cpt_b_heave": {"passed": bool, "max_error": float, "threshold": 0.15},
      "G3_cpt_self_conv_am": {"passed": bool, "max_error": float, "threshold": 0.05},
      "G4_tbem_self_conv_am": {"passed": bool, "max_error": float, "threshold": 0.05},
      "G5_dw_limit_am": {"passed": bool, "max_error": float, "threshold": 0.05}
    },
    "all_gates_passed": bool
  }
}
```

### 3.2 与既有工作的对比

| 工件 | 水深站 | 频率点 | 门禁通过率 | 第三方代码 |
|---|---:|---:|---:|---|
| `xval_bem_finite_depth/` | 3 | 2 | 8/11 (73%) | capytaine |
| `bem_finite_depth_capytaine_xval/`（本单） | 6 | 5 | 待计算 | capytaine |

本计划将覆盖范围从 3×2=6 工况扩展到 6×5=30 工况（×多网格阶梯），
并修复了前身中 G4（模态截断）、G8（频率网格）、G11（对称性）三个未通过门禁的根因。

---

## 4. 结果图

图由 `make_figures.py` 程序化生成，零手抄。计算完成后将产生：

- `docs/figures/xval_fd_am_heave_vs_ka.png`：附加质量 vs $ka$（多水深曲线）
- `docs/figures/xval_fd_b_heave_vs_ka.png`：阻尼 vs $ka$（多水深曲线）
- `docs/figures/xval_fd_error_heatmap.png`：相对误差热力图（水深×频率）
- `docs/figures/xval_fd_convergence.png`：自收敛曲线

---

## 5. 注意点

### 5.1 h/R=3 边界情况

球底恰好触底（$z_b = -3 = -h$），这是物理边界情况。TensorBEM 的网格生成
要求 `submergence > radius`（2.0 > 1.0，满足），但球底与海底的距离为零，
可能影响近底流场的数值精度。结果中将单独标注此站。

### 5.2 模态截断

有限水深 Green 函数的衰减根 $\mu_n$ 满足 $\mu_n \tan(\mu_n h) = -\nu$。
截断准则：$e^{-\mu_n d} < e^{-10}$（$d$ 为最近目标/求积点间距）。
硬上限 `_MODES_CAP = 4096` 防止浅水高频时模态数爆炸。

### 5.3 计算成本

总工况：6 水深 × 5 频率 × (4+4) 网格 = 240 次求解。
单次 TensorBEM 求解（sub=4）约 30-60 s → 总计算时间约 2-4 小时（并行）。

---

## 6. 误差判定

**基准**：capytaine v3.1.dev0（独立代码路线）

| 关键量 | 阈值 | 实测 | 是否 < 阈值 |
|---|---:|---:|---|
| G1: A33 跨代码误差 | 10% | 待计算 | 待判定 |
| G2: B33 跨代码误差 | 15% | 待计算 | 待判定 |
| G3: capytaine 自收敛 | 5% | 待计算 | 待判定 |
| G4: TensorBEM 自收敛 | 5% | 待计算 | 待判定 |
| G5: 深水极限 A33 | 5% | 待计算 | 待判定 |

**honest 声明**：计算结果待集群作业 cj2511 完成后回填。本单已交付完整的
交叉验证计划文档、生产者脚本、图生成脚本和测试框架。

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

## 8. 给下一次自找活的线索

1. **绕射问题交叉验证**：本单仅覆盖辐射问题，绕射问题的跨代码比对（含能量守恒 G6）
   是自然续作。
2. **复杂船体**：球体验证通过后，Wigley III 或 KCS 船体的有限水深跨代码验证
   是工程意义的下一步。
3. **有航速有限水深**：Neumann-Kelvin 有限水深版本尚未实现，属于"新物理"级增量。
