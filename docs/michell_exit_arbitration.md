# Michell 兴波阻力出口口径仲裁（mission m16345）

本文件的所有数字由 `scripts/m16345_michell_exit_arbitration.py` 生成
（`docs/michell_exit_arbitration/numbers.json` + `table.md` + `figure.png`），
下面的表就是 `table.md` 的原文，无一手抄。

## 1 问题

仓内有多条"Wigley 船型 Michell 兴波阻力"公共出口，彼此给出不同的数：
`wigley.michell_wave_resistance`（m16274 已判 DEFECTIVE：量纲非力、低 10–25 倍）、
`michell.michell_resistance`（`4 rho/U^2` 单波数口径，挂在面元路径的"参考值"上）、
`rankine_wave.michell_wave_resistance`（`4 rho g^2/(pi U^4)` 单波数口径）。
m16274 新增的 `wigley.michell_wave_resistance_canonical` 是唯一满足定常自由面条件的实现，
但它把"其余出口重钉"留给了后续单。本单做两件事：**给出可机器裁的仲裁判据**，
并**修好其中一条仍被用户直接调用的出口**（`rankine_wave`）。

## 2 仲裁判据（先于计算固定）

1. **色散可行性**：定常自由面条件 `U^2 phi_xx + g phi_z = 0` 对
   `phi ~ exp(k z + i alpha x)` 只允许 `(k, alpha) = (alpha^2/k0, alpha)`，
   `k0 = g/U^2`。任何在整个 `lambda` 区间用单波数 `k = alpha` 采样 Kochin 幅值的积分，
   逐点违反该条件，相对残差 `|1 - 1/lambda|`（`lambda=2` 时 50%，`lambda=9` 时 89%）。
   这是数学恒等式，与实现无关。
2. **量纲/相似律**：`Rw = rho U^2 L^2 f(Fr, B/L, T/L)`。在 Fr 不变下把
   `(L,B,T,g,U)` 同乘 `s`，`Rw` 必须按 `s^4` 缩放。实测
   `rankine_wave` 新旧出口均给 **4.000000 / 4.000000**（合律）；
   `wigley.michell_wave_resistance` 给 2.0（不合律，m16274 已判）。
3. **幅值一致性**：两条闭式 Kochin 幅值必须描述同一船型。修正后
   `H_rankine = -(2 U T / 3) H_wigley` 逐点成立（比值恒 `1.2379`），
   修正前该比值随 `k` 从 1.71 漂到 1.51 —— 说明深度积分丢了 `(z/T)^2` 项。
4. **求积收敛**：被积函数在 `k = k0` 有可积的 `1/sqrt(k-k0)` 端点，
   普通 Gauss-Legendre 网格不下潜，值随 `n_quad` 单调漂移。

## 3 结果

| Fr | U [m/s] | Rw canonical [N] | Rw legacy (256) [N] | legacy/canonical | legacy quad drift 256->1024 |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 1.9806 | 26.5393 | 239.5979 | 9.0281 | +1.171 % |
| 0.25 | 2.4757 | 49.7192 | 224.7930 | 4.5212 | +0.090 % |
| 0.30 | 2.9709 | 144.1357 | 302.4771 | 2.0986 | +0.624 % |
| 0.35 | 3.4660 | 114.3146 | 172.3650 | 1.5078 | +0.195 % |
| 0.40 | 3.9611 | 327.1101 | 242.6623 | 0.7418 | +0.777 % |

| n_quad | Rw legacy [N] |
|---:|---:|
| 64 | 295.0069 |
| 128 | 299.9705 |
| 256 | 302.4771 |
| 512 | 303.7354 |
| 1024 | 304.3656 |

| k/k0 | |H_rankine| | |H_wigley| | ratio |
|---:|---:|---:|---:|
| 0.50 | 0.695344 | 0.561732 | 1.2379 |
| 1.00 | 0.251491 | 0.203166 | 1.2379 |
| 1.50 | 0.0980943 | 0.0792453 | 1.2379 |
| 3.00 | 0.0335378 | 0.0270934 | 1.2379 |
| 8.00 | 0.0103922 | 0.00839531 | 1.2379 |

similarity exponent (Fr-preserving, expected 4 for a force): public entry = 4.000000, legacy = 4.000000

energy_flux relative_error = 0.5666667442762443 (tautology)

（`legacy/canonical` 在 Fr=0.20 达 9.03×，在 Fr=0.40 反而
0.74× —— 缺陷是 **Froude 相关**的，不是常数因子；
`legacy` 求积从 64→1024 节点单调爬升 295.0→304.4 N，无收敛。）

## 4 代码改动（`src/tensorbem/rankine_wave.py`）

1. `michell_kochin_function`：深度积分补全为 `int_{-T}^0 (1-(z/T)^2) exp(kz) dz`，
   `k -> 0` 极限回到精确的 `2T/3`（旧代码只保留 `(1-exp(-kT))/k`，极限为 `T`，幅值 +50%）。
2. `michell_wave_resistance`：改为委托 `wigley.michell_wave_resistance_canonical`
   （全包唯一实现，规范波数对 + `lambda^2` 权 + `lam = 1+s^2` 去奇点）。
3. `michell_wave_resistance_legacy`：旧公式逐字保留，docstring 标 DEFECTIVE 并列出三条缺陷，
   回归测试钉死旧值（Wigley III Fr=0.30 → 302.4771343584 N），防止无声回潮。
4. `michell_wave_resistance_converged`：同口径高分辨率求值（反射性对照）。
5. `wave_energy_flux`：加注说明其 `energy_flux_scaled` 是**恒等式**（先乘
   `Rw_V/energy_flux` 再比对），不是能量守恒证据。

## 5 注意点与未完成

- **`wigley.michell_wave_resistance` 与 `michell.michell_resistance` 仍是旧口径**
  （m16274 明确留给"重钉单"，它们有大量 pin 与面元路径参考值挂在上面，本单不动）。
- 两舷因子（`H` 含两侧）沿用 m16274 的裁决；对实验的绝对量级仍 **WITHHELD**
  （仓内无 Wigley 实验 Cw）。
- `rankine_panel_wave_resistance` 自述 `method = rankine_panel_michell_hybrid`：
  它本质仍是薄船积分 + 网格诊断，不是 Rankine 面元求解；本单没改这个标签语义。
- 本单只做轻量一维求积，未起重算力进程（load15 ≈ 100+）。

## 6 误差判定

| 量 | 基准 | 误差 | <3% |
|---|---|---:|---|
| `rankine_wave.michell_wave_resistance` | 规范谱（唯一合规实现） | 0（同实现委托） | 是 |
| 旧出口 Fr=0.30 | 规范谱 | 109.9% 偏高 | 否（缺陷，已判） |
| 旧出口 Fr=0.20 / 0.40 | 规范谱 | -803% 偏高 / 26% 偏低 | 否 |
| Kochin 幅值（修正后） | `H_wigley` 解析关系 | 0（恒等，1e-12 内） | 是 |
| 深度积分解析式 | 直接数值积分 | <1e-9 | 是 |
| 相似律指数 | 4（力） | 0 | 是 |
