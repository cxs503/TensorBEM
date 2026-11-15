# CASE-BEM-CUMMINS-WIGLEY：$A_\infty$ 的独立锚点与**零自由参数**的 Kramers-Kronig 闭合

m21471 · platform 席自驱工时 · 仓 TensorBEM · 锚点 `CASE-BEM-CUMMINS-WIGLEY`（算例池 id=145，容差 <10%）

## 1 案例介绍

Wigley 数学船型（$L=3.0$ m, $B=0.3$ m, $T=0.1875$ m, $C_b=4/9$，排水体积 $0.075$ m³，$\rho=1025$ kg/m³, $g=9.80665$ m/s²），6 自由度刚性辐射。算例判据来自 Cummins (1962) 时域卷积与 Kramers-Kronig 因果对：由 $B(\omega)$ 反演 $A(\omega)-A_\infty$，容差 10%。

**本单解决的验证欠账**：闸门 `C3` 里 $A_\infty$ 是**在同一段被判的频带上最小二乘拟合出来的自由参数**。一个自由参数足以吸收常数偏置，于是「KK 闭合 <10%」这条验收判据可以被一个非物理的 $A_\infty$ 满足。本单用**另一条边界值问题**（自由面高频极限 $\phi=0$，反对称镜像精确闭合）给出 $A_\infty$，把判据改成**零自由参数**版本（新闸门 `C3a`），并新增 `C7` 反过来量化被拟合的那个参数偏离物理值多远。

## 2 怎么做的

1. `tensorbem.added_mass_limit`（镜像 BEM，m21326）给出 6×6 的 $A_\infty$；本单把它接进算例：`added_mass_inf_anchor(n_sections, n_levels)` 用**与扫频相同的旋转中心** `CENTER=(0,0,0)` 求解（旋转中心不一致会让两列 $A_\infty$ 相差一个刚体变换，比较作废——这是本轮踩到的第一个坑）。
2. 新增 `C7`：$|A_\infty^{\text{fit}}-A_\infty^{\text{anchor}}|/\max|A_{ii}| \le 10\%$，并以半浸球解析解 $A_\infty(\text{heave})=m_{\text{full}}/4$ 对锚点自身做前置校验（误差须 <3%）。
3. 新增 `C3a`：同一 KK 闭合机器（`kk_gate`，含截断预算判可判性窗口）但 $A_\infty$ 由锚点**代入**，自由参数个数 0；原 `C3` 一字未动（判据不放宽、不替换）。
4. 全部结果由**已提交的扫频工件**重放得到（`gates_from_artifact`），不重解 BEM：`scripts/case_cummins_wigley_anchor_audit.py`。

## 3 计算结果

### 3.1 锚点自校验（闭式真值）

| 校验量 | 值 | 基准 | 误差 | 门限 |
| --- | --- | --- | --- | --- |
| 半浸球 $A_\infty$(heave)，360 面元 | 1051.67 kg | 闭式 1073.38 kg | **2.02%** | 3% |

### 3.2 锚点网格阶梯（$A_\infty$ 对角项）

| DOF | 12×4 (192 面元) | 20×6 (480 面元) | 28×8 (896 面元) | coarse→medium | medium→fine |
| --- | --- | --- | --- | --- | --- |
| surge | 0.488384 | 0.484905 | 0.483492 | 0.72% | 0.29% |
| sway | 68.6414 | 66.6249 | 65.6683 | 3.03% | 1.46% |
| heave | 49.6575 | 50.0661 | 50.1958 | 0.82% | 0.26% |
| roll | 0.395467 | 0.374906 | 0.363907 | 5.48% | 3.02% |
| pitch | 14.4336 | 14.5733 | 14.6222 | 0.96% | 0.33% |
| yaw | 50.7397 | 49.2552 | 48.3912 | 3.01% | 1.79% |

### 3.3 拟合 $A_\infty$ vs 物理锚点，以及两种闭合残差（medium 网格）

| DOF | $A_\infty$ 拟合 | $A_\infty$ 锚点 | $\Delta/\max|A_{ii}|$ | $\Delta/|A_\infty^{anchor}|$ | KK 残差（拟合，C3） | KK 残差（锚定，C3a） | 可判样本 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| surge | 0.585909 | 0.484905 | 4.45% | 20.83% | 2.62% | **7.07%** | 0/12 |
| sway | 75.733 | 66.6249 | 4.81% | 13.67% | 2.81% | **7.62%** | 0/12 |
| heave | 50.9302 | 50.0661 | 0.61% | 1.73% | 0.30% | **0.91%** | 12/12 |
| roll | 0.384917 | 0.374906 | 1.73% | 2.67% | 0.72% | **2.45%** | 12/12 |
| pitch | 14.8116 | 14.5733 | 0.70% | 1.64% | 0.48% | **1.17%** | 12/12 |
| yaw | 56.7785 | 49.2552 | 5.25% | 15.27% | 3.60% | **8.85%** | 0/12 |

### 3.4 判据结果

| 闸门 | 网格 | 自由参数 | 结论 | worst DOF | 值 | 门限 |
| --- | --- | --- | --- | --- | --- | --- |
| C3（原判据） | medium | 1 | PASS | yaw | 3.60% | 10% |
| **C3a（零自由参数，新）** | medium | **0** | PASS | yaw | **8.85%** | 10% |
| C7（$A_\infty$ 锚点一致性，新） | medium | — | PASS | yaw | 5.25% | 10% |
| C3（原判据） | coarse | 1 | PASS | yaw | 3.40% | 10% |
| **C3a（零自由参数，新）** | coarse | **0** | PASS | yaw | **8.40%** | 10% |
| C7（$A_\infty$ 锚点一致性，新） | coarse | — | PASS | yaw | 5.00% | 10% |

medium 网格整案裁决：**10/10 通过**，failed = []。

![fig](fig_m21471_cummins_anchor.png)

*左：拟合 $A_\infty$ 与镜像锚点（symlog）；中：拟合 vs 锚定的 KK 残差与 10% 判据线；右：锚定闭合残差随频率的分布（medium 网格）。*

## 4 注意点

1. **旋转中心必须一致**：扫频以原点（水线面内）为旋转中心，首轮探针用 $(0,0,-T/2)$ 得到 roll 的 $A_\infty$ 相差 105%——那不是物理差异而是刚体变换。修正后 roll 只差 2.7%。
2. **截断预算仍是空窗**：surge/sway/yaw 的实测阻尼尾部指数 0.28–1.09（未进入代数衰减），上端未解频段的贡献无法界定，故这三条 DOF 的 `n_decidable=0`；它们的锚定残差 7.1%/7.6%/8.9% 是**经验一致**而非带限数据蕴含。heave/roll/pitch 三条 12/12 可判。
3. **余量薄**：锚定后 worst 由 3.6%（yaw）涨到 8.9%，离 10% 只剩 1.1 pp；最坏样本固定在 $\omega=9.29$ rad/s（带顶附近），正是被截断尾部支配的样本。要把余量做厚必须解更高频段，而 $k\sqrt{A}\le1$ 在 medium 网格上到 $\omega\approx15.5$ rad/s 就失守，$\omega=20$ 需 $\sqrt{A}\le0.0245$ m （约 2000+ 面元）。
4. **拟合 $A_\infty$ 是坏估计量**：surge/sway/yaw 上比物理值高 21%/14%/15%（占带内幅值 4.4%/4.8%/5.3%）；而两参数尾拟合（$A_\infty+c/\omega^2$）在 surge/sway/yaw 上仍给出**负** $A_\infty$（非物理），与 m21326 结论一致——带内截断使外推不可用，锚点法才是正解。
5. 适用范围：本单只判**因果性/闭合**，不判 $A(\omega)$ 的绝对精度；该算例的 L5 外部锚点（实验/解析辐射系数表）仍为 WITHHELD。

## 5 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | 是否 <3% |
| --- | --- | --- | --- |
| 半浸球 $A_\infty$(heave)，锚点机器自校验 | 闭式 $m_\text{full}/4$ | 2.02% | **是**（门限 3%） |
| KK 闭合（零自由参数，C3a），worst DOF = yaw | 直接 BEM $A(\omega)$（同路线） | 8.85% | **否**（>3%，但 <10% 算例判据） |
| 同上，heave | 直接 BEM | 0.91% | **是** |
| 同上，roll | 直接 BEM | 2.45% | **是** |
| 同上，pitch | 直接 BEM | 1.17% | **是** |
| 拟合 $A_\infty$ vs 锚点，worst = yaw | 镜像 BEM 锚点 | 20.83%（相对锚点值） | **否**（估计量缺陷，已如实记入 C7） |

结论：算例判据（<10%）在**零自由参数**下成立（worst 8.9%），但只有 heave/roll/pitch 三条 DOF 同时满足 <3% 与截断可判；surge/sway/yaw 的 <10% 属经验一致。

## 6 复现

```bash
# 全部由已提交的扫频工件重放（不重解 BEM），约 1 min
PYTHONPATH=src python3 scripts/case_cummins_wigley_anchor_audit.py
# 单元测试（锚点自校验、零自由参数不变量、C7 咬合、工件重放）
PYTHONPATH=src python3 -m pytest tests/test_case_bem_cummins_wigley.py tests/test_added_mass_limit.py -q
```

改动文件：`src/tensorbem/benchmarks/case_bem_cummins_wigley.py`（锚点 + C3a/C7 + 旋转中心常量化 + 文档渲染）、`src/tensorbem/added_mass_limit.py`（半浸球解析体，供锚点自校验复用）、`tests/test_case_bem_cummins_wigley.py`（+5 测试）、`scripts/case_cummins_wigley_anchor_audit.py`、`benchmarks/verified/casepool_bem_cummins_wigley/anchor_audit.json`。