"""Post the paper-quality report for m11546 to the TensorPlatform reports API."""

import json
import os
import base64
import requests

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts')
API_BASE = os.environ.get('TENSOR_API_BASE', 'http://127.0.0.1:8010/api/v1')
USER = os.environ.get('TENSOR_AGENT_USER', 'eco_admin')
PASS = os.environ.get('TENSOR_AGENT_PASS', '')

# ── Login ────────────────────────────────────────────────────────────────
resp = requests.post(f'{API_BASE}/auth/login', json={'username': USER, 'password': PASS})
resp.raise_for_status()
token = resp.json()['access_token']
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json',
}

# ── Load figures as base64 ──────────────────────────────────────────────
figures = []
fig_files = [
    ('fig1_radiation_coefficients.png', '辐射系数 A(ω) 与 B(ω)：附加质量与辐射阻尼的频率依赖关系'),
    ('fig2_memory_kernel.png', '辐射记忆核 K(τ)：从频域阻尼 B(ω) 经余弦变换得到的时域记忆函数'),
    ('fig3_round_trip.png', '频域→时域→频域往返闭合：B(ω)→K(τ)→B(ω) 的重构与误差'),
    ('fig4_kk_causality.png', 'Kramers-Kronig 因果性校验：附加质量的 KK 预测与 multipole 参考值对比'),
    ('fig5_free_decay.png', 'Cummins 方程自由衰减：时域卷积结果与指数包络 exp(-ζωₙt) 对比'),
    ('fig6_error_envelope.png', '常数 A(∞) 近似误差包络：峰值/相位/阻尼比误差随 ω_p/ω_n 的变化'),
    ('fig7_causality.png', '双侧记忆核因果性可视化：K(τ<0) = 0 的严格因果约束'),
]

for fname, caption in fig_files:
    fpath = os.path.join(ARTIFACTS, fname)
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        figures.append({
            'name': fname,
            'caption': caption,
            'png_base64': b64,
        })

# ── Markdown report ──────────────────────────────────────────────────────
markdown = r"""
# Cummins 时域记忆核从零到一：辐射 impulse response function K(t) + Kramers-Kronig 因果性 + 时域卷积与频域 RAO 往返闭合

**Mission**: m11546 | **Campaign**: P106W3A2-N05 | **Status**: ✅ PASS  
**Date**: 2026-11-10 | **Repository**: TensorBEM | **Branch**: agent/m11546

---

## ① 案例介绍

### 物理背景

Cummins (1962) 方程是频域辐射问题与时域运动模拟之间的桥梁。频域辐射解给出"该频率下的稳态力"，而时域模拟需要"运动停止后仍然作用的瞬态力"。桥梁就是辐射记忆核 K(τ)：

$$K(\tau) = \frac{2}{\pi} \int_0^\infty B(\omega) \cos(\omega \tau) \, d\omega$$

Cummins 方程：

$$(M + A_\infty) \ddot{x}(t) + \int_0^t K(t-s) \dot{x}(s) \, ds + C x(t) = F(t)$$

### 算例几何与参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 浮体形状 | 深潜球体 | 深潜比 f/a = 3 |
| 半径 a | 1.0 m | |
| 密度 ρ | 1025 kg/m³ | 海水 |
| 重力 g | 9.80665 m/s² | |
| 排水体积 ∇ | 4π/3 ≈ 4.189 m³ | |
| 排水质量 | 4293.5 kg | |
| 方块系数 Cb | π/6 ≈ 0.5236 | 球体精确值 |
| 湿表面积 S | 4π ≈ 12.566 m² | |
| 自由度 | 垂荡 (heave) | 单自由度 |
| 参考解 | Hulme (1982) 球谐函数 | 精确到机器精度 |

### 频率扫描范围

- ω ∈ [0.294, 49.03] rad/s (ka ∈ [0.03, 6.0])
- 150 个频率点
- 阻尼峰值：B_max = 286.7 kg/s at ω = 5.52 rad/s

---

## ② 方法

### (a) 频域→时域：K(t) 的数值计算

由附加质量 A(ω) 与辐射阻尼 B(ω) 经逆 Fourier 余弦变换得 K(t)：

$$K(\tau) = \frac{2}{\pi} \int_0^\infty B(\omega) \cos(\omega \tau) \, d\omega$$

**数值实现**：
- 分段线性精确积分（非简单梯形法则）：在每段 [ω_i, ω_{i+1}] 上 B(ω) 线性插值，余弦矩有解析表达式
- 低频闭合：B(ω) ~ c·ω³ 长波模型（低频端）
- 高频截断：报告截断误差界
- ω→∞ 渐近处理：A(ω) - A(∞) ~ c/ω² 最小二乘拟合上尾

### (b) Kramers-Kronig 因果性校验

KK 关系（Kotik & Mangulis 1962）：

$$A(\omega) - A_\infty = \frac{2}{\pi} \text{PV} \int_0^\infty \frac{B(s)}{s^2 - \omega^2} \, ds$$

**数值实现**：
- 主值积分精确求解（分段线性插值 + 对数奇点精确消去）
- 逐点校验，要求 ≤ 2%

### (c) 时域卷积自由衰减

Cummins 方程无外力自由衰减积分：
- 梯形法则离散卷积
- Richardson 外推消除时间步长误差
- 阻尼比 ζ_decay 与频域能量恒等式 ζ_energy = B(ω_n)/(2ω_n(M+A(ω_n))) 对比
- 要求 |ζ_decay - ζ_energy|/ζ_energy ≤ 5%

### (d) 误差包络

常数 A(∞) 近似（不用卷积，只用 A_∞ 和 B(ω_n)）的误差随 ω_p/ω_n 的变化：
- 峰值振幅误差
- 相位误差
- 阻尼比误差

### (e) 几何前置门

浮体 Cb/L/B/∇/S 先过平台几何门：
- Cb ∈ (0.30, 1.00]
- |∇ - L·B·T·Cb|/∇ ≤ 5%
- 负控制：膨胀排水量/不可能 Cb 必须被拒绝

---

## ③ 计算结果

### 验收判据逐项

| 判据 | 要求 | 实测值 | 结果 |
|------|------|--------|------|
| **C5 几何门** | Cb=π/6, 一致性≤5% | Cb=0.5236, 一致性=0.0% | ✅ PASS |
| **C1 阻尼往返误差** | ≤ 2% | 0.0285% | ✅ PASS |
| **C1 附加质量往返误差** | ≤ 2% | 1.94% | ✅ PASS |
| **C2 K(τ<0)** | = 0 精确 | 0.0 精确 | ✅ PASS |
| **C2 K(τ) 偶对称** | 残差 = 0 | 0.0 精确 | ✅ PASS |
| **C2 K(0) 一致性** | ≤ 0.1% | 0.0014% | ✅ PASS |
| **C2 KK 因果性误差** | ≤ 2% | 1.90% | ✅ PASS |
| **C3 ζ_decay vs ζ_energy** | ≤ 5% | 1.46% | ✅ PASS |
| **C3 ζ_decay vs ζ_half_power** | ≤ 5% | 2.13% | ✅ PASS |
| **C3 收敛比** | ≈ 2.0 | 1.98 | ✅ PASS |
| **C4 零阻尼衰减比** | ≤ 0.5% | 0.0012% | ✅ PASS |

### 关键数字

- **K(0)** = 1428.8 kg/s²（阻尼积分预测 1428.8，相对误差 0.0014%）
- **A(∞)** = 2113.4 kg（锚值 ρV/2 = 2146.8 kg，误差 1.56%——深潜球体高频渐近与无限流体极限的已知物理差异）
- **ζ_decay** = 0.01037（能量恒等式 ζ = 0.01022，半功率带宽 ζ = 0.01059）
- **衰减率** = 0.0572 s⁻¹（Richardson 外推值，R² > 0.9999997）
- **核衰减时间**（至 1%）= 1.13 s

### 误差包络（常数 A(∞) 近似）

| ω_p/ω_n | 峰值振幅误差 | 相位误差 | 阻尼比误差 |
|---------|-------------|---------|-----------|
| 0.50 | -2.22% | 79.6° | 2.10% |
| 0.75 | -0.10% | 51.4° | 1.68% |
| **1.00** | **0.01%** | **0.9°** | **0.03%** |
| 1.50 | -2.99% | -48.3° | 2.62% |
| 2.00 | -5.12% | -61.7° | 3.74% |

→ 共振时（ω_p = ω_n）常数近似几乎精确；偏离共振时相位误差急剧增大。

---

## ④ 结果图

见 figures 数组（7 张图，base64 内嵌）。

---

## ⑤ 注意点

### 已验证的
1. **往返闭合**：B→K→B 的误差 0.028%，远低于 2% 阈值——变换的数值实现正确
2. **因果性**：K(τ<0) = 0 精确成立，偶对称残差 = 0，KK 关系误差 1.90% < 2%
3. **时域/频域一致性**：自由衰减阻尼比与频域预测差 1.46%，收敛比 1.98 ≈ 2（一阶梯形法则的 Richardson 极限）
4. **负控制**：零阻尼给出零核、零衰减——排除积分器自身数值阻尼的伪信号

### 局限与适用范围
1. **线性、深水、零航速**：不含非线性横摇阻尼、参数横摇、航速效应
2. **单自由度**：多自由度耦合核矩阵 K_ij(t) 已在 `memory_kernel_matrix.py` 实现并通过测试
3. **深潜球体的 A(∞) 锚误差 1.56%**：这是物理的——深潜球体的高频渐近 A(ω→∞) 不完全等于无限流体极限 ρV/2，因为自由面项以 exp(-2kf) 衰减但不严格为零
4. **频率截断**：扫描到 ka=6，更高频的阻尼已衰减至机器零——截断误差可忽略
5. **时间截断**：τ_max = 200s，核在 τ ≈ 1.1s 已衰减至 1%——截断误差远低于离散化误差

### 陷阱
1. numpy 2.0+ 移除了 `np.trapz`，需用 `np.trapezoid`
2. `radiation_memory_kernel` 返回 `MemoryKernel` 对象（callable），不是数组
3. `kkr_added_mass` 需要显式传 `added_mass_inf` 关键字参数
4. KK 主值积分要求采样频率严格落在网格节点之间（不能在节点上）

---

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|--------|------|------|-----------|
| 往返阻尼误差 | 2% 阈值 | 0.0285% | ✅ 是 |
| 往返附加质量误差 | 2% 阈值 | 1.94% | ✅ 是 |
| KK 因果性误差 | 2% 阈值 | 1.90% | ✅ 是 |
| ζ_decay vs ζ_energy | 5% 阈值 | 1.46% | ✅ 是 |
| ζ_decay vs ζ_half_power | 5% 阈值 | 2.13% | ✅ 是 |
| A(∞) vs ρV/2 锚 | 物理差异 | 1.56% | ✅ 是（已知物理差异） |

**全部判据满足，无 honest-FAIL。**

---

## 审计链

- **代码模块**：`src/tensorbem/time_domain.py`, `src/tensorbem/causality.py`, `src/tensorbem/memory_kernel_matrix.py`, `src/tensorbem/benchmarks/cummins_memory_closure.py`
- **测试文件**：`tests/test_cummins_memory_closure.py` (13 tests), `tests/test_time_domain.py` (12 tests), `tests/test_causality.py` (17 tests), `tests/test_kkr_causality.py` (4 tests), `tests/test_memory_kernel_matrix.py` (13 tests)
- **总计**：59 个测试全部通过
- **工件路径**：`artifacts/verification_report_m11546.json`, `artifacts/fig*.png`
- **基线**：main 分支（commit 25203d3）

## 参考文献

1. Cummins, W.E. (1962), *The impulse response function and ship motions*, Schiffstechnik 9:101-109.
2. Ogilvie, T.F. (1964), *Recent progress toward the understanding and application of the impulse response function in ship motions*, Int. Shipbuilding Progress 11.
3. Kotik, J. & Mangulis, F. (1962), *On the Kramers-Kronig relations for ship motions*, Int. Shipbuilding Progress 9.
4. Hulme, A. (1982), *The wave forces acting on a floating hemisphere undergoing forced periodic oscillations*, J. Fluid Mech. 121:443-463.
5. Newman, J.N. (1977), *Marine Hydrodynamics*, MIT Press, ch. 6.
"""

# ── Build report payload ─────────────────────────────────────────────────
report_payload = {
    'title': 'Cummins时域记忆核从零到一：辐射impulse response function K(t)+Kramers-Kronig因果性+时域卷积与频域RAO往返闭合',
    'summary': 'TensorBEM Cummins 方程辐射记忆核 K(t) 完整实现与验证：频域→时域往返闭合（阻尼误差 0.028%，附加质量误差 1.94%），Kramers-Kronig 因果性校验（误差 1.90%），自由衰减阻尼比与频域预测一致（1.46%），零阻尼负控制通过，常数 A(∞) 近似误差包络量化。59 个 pytest 全部通过。',
    'scenario_key': 'cummins_memory_kernel_closure',
    'mission_id': 11546,
    'tags': ['cummins', 'memory-kernel', 'kramers-kronig', 'causality', 'time-domain', 'frequency-domain', 'radiation', 'added-mass', 'free-decay', 'verification'],
    'markdown': markdown,
    'figures': figures,
}

# ── Post ─────────────────────────────────────────────────────────────────
resp = requests.post(
    f'{API_BASE}/tensor/reports',
    headers=headers,
    json=report_payload,
    timeout=120,
)
print(f"Status: {resp.status_code}")
if resp.status_code >= 400:
    print(f"Error: {resp.text[:500]}")
else:
    print(f"Report posted successfully: {resp.json()}")
