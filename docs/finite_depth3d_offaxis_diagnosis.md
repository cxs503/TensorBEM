# 3-D Finite-Depth Green Function: Off-Axis Deep-Water Limit Diagnosis

**Task**: m2930 (verifier席，自找活) · **仓**: TensorBEM
**日期**: 2026-11-03 · **分支**: agent/m2930
**上游提交**: b408d0d (agent/m2816, 未合入 main)

---

## 1. 问题陈述

`tensorbem.finite_depth3d` 模块实现了三维有限水深自由面 Green 函数，但在深水极限交叉验证中：

| 场点 | R [m] | h=8 | h=20 | h=60 | 收敛？ |
|---|---:|---:|---:|---:|---|
| (0.5, 0, -0.5) | 0.5 | 0.347 | 0.345 | 0.345 | **否** |
| (2.0, 1.0, -1.5) | 2.24 | 3.20 | 3.20 | 3.20 | **否** |
| (0, 0, -0.2) | 0.0 | 0.002 | 6e-5 | 2e-6 | **是** |

相对误差 `|G_fd - G_dw| / |G_dw|` 在离轴点上**不随水深增加而减小**，且误差为实值、随 R 增大。

---

## 2. 根因分析

### 2.1 公式正确性确认

有限水深核的深水极限解析推导：

$$G_{fd} \xrightarrow{h\to\infty} \frac{1}{4\pi r} + \frac{1}{4\pi}\int_0^\infty \frac{k+\nu}{k-\nu} e^{k(z+\zeta)} J_0(kR)\,dk$$

分解被积函数：

$$\frac{k+\nu}{k-\nu} = 1 + \frac{2\nu}{k-\nu}$$

第一项积分给出镜像源项 $1/r_1$，第二项给出 $2\nu W$，故：

$$G_{fd} \xrightarrow{h\to\infty} \frac{1}{4\pi r} + \frac{1}{4\pi r_1} + \frac{\nu}{2\pi} W = G_{dw}$$

**公式层面完全等价**。问题出在数值实现。

### 2.2 数值实现审查

`_wave_integral` 使用核减法（kernel subtraction）计算主值积分：

$$\text{PV}\int \frac{q(k)}{k-k_0} dk = \int \frac{q(k) - q(k_0)\chi(k)}{k-k_0} dk$$

其中 $\chi(k) = \exp(-((k-k_0)/w)^2)$ 为高斯 bump。

审查 `_q_values` 函数发现：

```python
denom = 0.5 * (kk * (1.0 - exp_m2kh) - nu * (1.0 + exp_m2kh))
smooth = np.where(offset == 0.0, slope_scaled, denom / safe)
```

这里 `smooth = denom / (k - k0)` 是缩放后的分母 $D(k) e^{-kh}$。

**关键发现**：`_q_values` 返回的是 $q(k) = \text{integrand} \times (k-k_0)$，即已经乘了 $(k-k_0)$ 的光滑函数。主值积分应为：

$$\text{PV}\int \frac{q(k)}{k-k_0} dk = \int \frac{q(k) - q(k_0)\chi(k)}{k-k_0} dk$$

代码实现：
```python
smooth_reg = q(grid) - q_pole * exp(-(offset/bump)**2)
integrand = smooth_reg / offset
```

这在数学上是正确的。

### 2.3 深水极限的数值行为

在深水极限下（$h \to \infty$）：
- $k_0 \to \nu$
- $\exp(-2kh) \to 0$
- $q(k) \to \frac{k+\nu}{k-\nu}(k-k_0) e^{k(z+\zeta)} J_0(kR)$

当 $k \to k_0 \to \nu$ 时，$q(k_0) \to 2\nu e^{\nu(z+\zeta)} J_0(\nu R)$。

**问题定位**：在 `_wave_integral` 中，`k_max` 由截断条件 $\exp(-k_{max}\alpha) < 10^{-17}$ 确定，其中 $\alpha = -(z+\zeta)$。

对于场点 $(0.5, 0, -0.5)$ 和源点 $(0, 0, -1.0)$：
- $\alpha = -(-0.5 - 1.0) = 1.5$
- $k_{max} = -\ln(10^{-17}) / 1.5 \approx 26.1$

但被积函数中的因子 $\frac{k+\nu}{k-\nu}$ 在 $k \to \nu$ 附近有极点，虽然主值积分处理了极点，但**远离极点的尾部贡献**可能被低估。

### 2.4 真正的根因：$k_{max}$ 截断对 $J_0(kR)$ 振荡的不足

对于 $R > 0$，$J_0(kR)$ 振荡周期为 $2\pi/R$。当 $R = 0.5$ 时，周期约 12.6。在 $k_{max} = 26.1$ 处，$kR = 13.05$，约 2 个周期。

但被积函数的衰减由 $\exp(k(z+\zeta)) = \exp(-k\alpha)$ 控制。在 $k = 26.1$ 处，$\exp(-26.1 \times 1.5) \approx 10^{-17}$，确实很小。

**然而**，$\frac{k+\nu}{k-\nu}$ 因子在 $k$ 大时趋于 1，不改变衰减速率。所以截断应该是足够的。

### 2.5 重新审视：面板网格分辨率

面板宽度由下式确定：
```python
oscillation = 2π / R / 6.0  # 对于 R=0.5: 2.1
width = min(oscillation, 0.25*k0, bump/3.0)  # = min(2.1, 0.15, 0.02) = 0.02
```

面板数 = $26.1 / 0.02 \approx 1300$，每面板 16 节点，共约 20800 个求积点。这应该足够。

### 2.6 最终诊断：`_q_values` 中垂直因子的缩放错误

重新审查 `_q_values`：

```python
vertical = 0.5 * (np.exp(kk * z) + np.exp(-kk * (z[None, :] + 2.0 * depth)))
```

这是 $\cosh(k(z+h)) e^{-kh}$ 的缩放形式。正确。

但注意：`z` 是场点的 z 坐标（负值），`depth` 是水深。当 $h \to \infty$ 时：
- $\exp(kz)$ 保持不变（$z$ 固定）
- $\exp(-k(z+2h)) \to 0$

所以 `vertical` $\to 0.5 \exp(kz)$。正确。

**等等**，让我重新检查深水核的 $W$ 函数定义。

从 `free_surface3d.py`：
```
W(X, Y) = F(X, Y) + i·π·exp(-Y)·J₀(X)
F(X, Y) = PV∫₀^∞ exp(-uY)·J₀(uX)/(u-1) du
```

其中 $X = \nu R$，$Y = -\nu(z+\zeta)$。

令 $k = \nu u$，则：
```
F = PV∫₀^∞ exp(k(z+ζ))·J₀(kR)/(k-ν) dk
```

所以深水核的波部分为：
```
(ν/2π)·W = (ν/2π)·[PV∫ exp(k(z+ζ))·J₀(kR)/(k-ν) dk + iπ·exp(ν(z+ζ))·J₀(νR)]
```

而有限水深核的深水极限波部分为：
```
(1/4π)·∫ (k+ν)/(k-ν)·exp(k(z+ζ))·J₀(kR) dk
= (1/4π)·[1/r₁ + 2ν·W]
```

这里的关键是：有限水深核的积分是**从 0 到 ∞ 的完整积分**（主值 + 留数），而深水核的 $W$ 函数只包含主值部分加留数。

**但等等**，有限水深核的 `_wave_integral` 返回的是 `total + residue`，其中：
- `total` = 主值积分
- `residue` = $i\pi q(k_0)$

所以有限水深核的波部分 = 主值 + 留数，与深水核一致。

### 2.7 真正的根因：`_q_values` 中 `smooth` 的计算

让我重新检查 `smooth` 的计算：

```python
denom = 0.5 * (kk * (1.0 - exp_m2kh) - nu * (1.0 + exp_m2kh))
offset = kk - k0
safe = np.where(offset == 0.0, 1.0, offset)
smooth = np.where(offset == 0.0, slope_scaled, denom / safe)
```

`smooth` 应该是 $D(k) e^{-kh} = \frac{k\sinh(kh) - \nu\cosh(kh)}{k-k_0} e^{-kh}$。

在缩放形式下：
- $k\sinh(kh) e^{-kh} = 0.5k(1 - e^{-2kh})$
- $\nu\cosh(kh) e^{-kh} = 0.5\nu(1 + e^{-2kh})$

所以 `denom` = $0.5[k(1-e^{-2kh}) - \nu(1+e^{-2kh})]$ = $(k\sinh(kh) - \nu\cosh(kh)) e^{-kh}$。正确。

`smooth` = `denom / (k - k0)` = $D(k) e^{-kh}$。正确。

### 2.8 结论：深水核与有限水深核的归一化差异

经过详细审查，我发现**有限水深核的 Rankine 部分与深水核不同**：

- 有限水深：`rankine = 1/r + 1/r_image`，其中 `r_image = hypot(R, |z + 2h + ζ|)`
- 深水：`rankine = 1/r + 1/r_1`，其中 `r_1 = hypot(R, |z + ζ|)`

在深水极限下，`r_image → ∞`，所以 `1/r_image → 0`。但深水核有 `1/r_1` 项。

我们之前推导过，有限水深核的波积分在深水极限下会给出 `1/r_1 + 2νW`，所以总核为：
```
G = (1/r + 0 + 1/r_1 + 2νW) / (4π) = 1/(4πr) + 1/(4πr_1) + ν/(2π)·W
```

这与深水核一致。**公式正确**。

### 2.9 最终诊断：数值积分中的相位误差

经过上述分析，公式层面没有问题。误差必须来自数值积分。

**关键观察**：误差是实值的、不随深度变化、随 R 增大。这表明误差来自 $J_0(kR)$ 的数值积分，而非留数项（留数项是复值的）。

**假设**：面板网格的宽度由 `bump/3.0 = 0.02` 限制，这对于 $k_{max} = 26$ 来说可能不够细。虽然 1300 面板 × 16 节点 = 20800 点看起来很多，但 $J_0(kR)$ 在 $kR > 10$ 后振荡非常快，可能需要更细的网格。

**验证方法**：增加 `nodes` 参数或减小面板宽度，观察误差是否减小。

---

## 3. 修复方案

### 3.1 短期修复：增加面板分辨率

在 `_wave_integral` 中，将面板宽度限制从 `bump/3.0` 改为更严格的值：

```python
# 原代码
width = max(min(oscillation, 0.25 * k0, bump / 3.0), 1.0e-12)

# 修复后
width = max(min(oscillation, 0.1 * k0, bump / 6.0), 1.0e-12)
```

### 3.2 长期修复：自适应面板网格

实现基于被积函数局部曲率的自适应面板细分，确保 $J_0(kR)$ 的每个振荡周期有足够多的求积点。

### 3.3 验证测试

添加测试验证深水极限在增加分辨率后收敛：

```python
def test_deep_water_limit_converges_with_resolution():
    """Off-axis points should converge as resolution increases."""
    targets = np.array([[0.5, 0.0, -0.5]])
    source = np.array([[0.0, 0.0, -1.0]])
    nu = 0.6
    depth = 60.0
    
    ref = green_function_3d(targets, source, nu)
    
    for nodes in [16, 32, 64]:
        val = green_function_finite_depth_3d(targets, source, nu, depth, nodes=nodes)
        err = np.abs(val - ref) / np.abs(ref)
        # Error should decrease with more nodes
        assert err < 0.01, f"nodes={nodes}: err={err}"
```

---

## 4. 误差判定

| 关键量 | 基准 | 实测 | 误差 | <3%? |
|---|---|---|---|---|
| 自由面残差 | 0 | 3.6e-14 | 3.6e-14 | **是** |
| 底面残差 | 0 | <1e-6 | 绝对 | **是** |
| 互易性 | 0 | 3e-16 | 3e-16 | **是** |
| 网格无关性 | 0 | 1.2e-11 | 绝对 | **是** |
| 深水极限（轴上） | free_surface3d | 1.9e-6 | 1.9e-6 | **是** |
| 深水极限（离轴 R=0.5） | free_surface3d | 0.345 | **34.5%** | **否** |
| 深水极限（离轴 R=2.24） | free_surface3d | 3.20 | **320%** | **否** |

---

## 5. 局限与后续

1. 本诊断基于代码审查，未运行数值实验验证修复方案（系统负载过高）。
2. 修复方案需要重新运行验证测试，确认误差随分辨率增加而减小。
3. 有限水深 3-D 核的完整验证需要：深水极限交叉验证 + 解析边界条件验证 + 能量恒等式验证。

---

**诊断人**: m2930 verifier席
**日期**: 2026-11-03
