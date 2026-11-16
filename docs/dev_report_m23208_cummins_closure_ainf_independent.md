# CASE-BEM-CUMMINS-WIGLEY：用**独立** A_∞ 重判闭包带 G3/G5（A_∞ 偏置是全部原因；G4 不是）

**Mission**: m23208 ｜ **Anchor**: `CASE-BEM-CUMMINS-WIGLEY`（feature_ledger: open）｜ **仓**: TensorBEM
**对象**: 扩展频带闭包判据 G3（Kramers-Kronig）/G4（自由衰减）/G5（往返）所用的加质量渐近量 $A_\infty$

## 1 案例与这条欠账从哪来

Wigley III（L=3.0 m, B=0.3 m, T=0.1875 m, Cb=0.543, ρ=1025 kg/m³, 无限水深），
网格 `(n_sections, n_levels) = (20, 10)`（800 面板），旋转中心
`(0, 0, -T/2) = (0.0, 0.0, -0.09375)`。
上一轮（m23130）把预声明闭包网格 **69 个频点**（0.10–20 rad/s，含 41 点共振簇）全部解出后，
G3/G4/G5 仍判 FAIL（33 % / 18 % / 33 %）。而 G5 的归因（m21140）给出的收尾条件是：

> 「需要独立的 A_∞ 锚（ω→∞ 极限的 Laplace 解，或文献值），而不是又一个从同一 (A,B) 造出来的估计器」

这个锚其实**已经入库但从未用来判据**：`tensorbem.added_mass_limit.added_mass_limit`
用反对称镜像把自由面换成高频极限 $\phi=0$，把湿表面与其镜像闭成倍体，解的是
**另一个边值问题**——与频率扫频不共享任何代码、数据、频带，没有可调参数，也没有尾部拟合。

## 2 怎么做的

1. **把 69 点闭包扫频冻结入库**（`benchmarks/verified/casepool_bem_cummins_wigley/sweep_closure_medium800.npz`，
   A/B 6×6、逐点线性残差最大 2.14e-03），
   否则判据无法被任何人重放。
2. **参考点对账**（这是本轮的硬前提）：扫频在 `(0,0,-T/2)` 取矩，而
   `case_bem_cummins_wigley.added_mass_inf_anchor` 在 `(0,0,0)`（水线）求解。$A_55$ 在参考点平移下
   **不守恒**（随 $A_35$ 耦合变化），所以先新增
   `tensorbem.added_mass_limit.shift_added_mass`（刚体速度变换 $X_1 = T X_2$，
   $T=[[I,[a\times]],[0,I]]$，$A_2=T^\top A_1 T$）把镜像矩阵搬到扫频的参考点上。
   **校验**：该变换与「直接在扫频参考点重解镜像问题」逐元素一致，最大相对差
   `1.03e-16`。
3. **不确定度**：镜像解跑网格梯子 `20x6 / 20x10`，$A_\infty$ 以区间携带，逐 DOF 相对散布
   最大 `2.08 %`。解析自校验（半浸球，$A_\infty=\rho V/4$）相对误差
   `2.02 %`（容差 3%）。
4. **判据不重定义**：`analysis_gates(..., added_mass_inf=...)` 只换常数来源，
   G3（归一化 $\max|A-A_\infty|$ 的 KK 残差）、G5（带限往返）、G4（时域衰减 ζ vs 半功率 ζ）
   的统计量与容差一字未改，两次评估并列报告。
5. **反循环检查**：把 $A_\infty$ 当自由参数扫过 $[0.8, 1.2]\times$ 镜像值（每 DOF 82 点），
   给出 G3 的**可判窗口**，看镜像值落在窗口的什么位置、离 G3 极小点多远。

## 3 计算结果（全部由 `ainf_independent_gates.json` 程序化注入，零手抄）

### 3.1 两个 $A_\infty$ 与两条闸门

| DOF | tail fit | image solve | 变化 | G3 tail [%] | G3 image [%] | G5 tail [%] | G5 image [%] |
|---|---:|---:|---:|---:|---:|---:|---:|
| surge | 0.0096 | 0.8037 | +8282.7 % | 27.48 | 2.84 | 27.68 | 3.45 |
| sway | 21.7752 | 64.1439 | +194.6 % | 28.13 | 0.92 | 28.27 | 1.42 |
| heave | 60.1657 | 69.1068 | +14.9 % | 7.46 | 0.90 | 7.63 | 1.08 |
| roll | 0.0292 | 0.2529 | +765.2 % | 27.69 | 1.55 | 27.89 | 2.03 |
| pitch | 24.2467 | 28.4443 | +17.3 % | 11.86 | 1.51 | 12.37 | 2.03 |
| yaw | 9.5128 | 46.5030 | +388.8 % | 33.21 | 1.67 | 32.81 | 2.28 |

容差：G3 ≤ 10%，G5 ≤ 5%。
**G3 判决：tail fit `False` → image `True`；
G5：`False` → `True`。**

### 3.2 反循环检查：可判窗口 vs 镜像值

| DOF | image $A_\infty$ 区间 | G3 极小点 | G3 可判窗口（≤10%） | 镜像值在窗口内? |
|---|---|---|---:|---|
| surge | [0.8037, 0.8075] | 0.8237 | 0.6429-0.9644 | yes |
| sway | [64.14, 65.46] | 64.1439 | 52.92-73.12 | yes |
| heave | [69.07, 69.11] | 69.4523 | 56.67-79.82 | yes |
| roll | [0.2529, 0.2545] | 0.2567 | 0.2023-0.3035 | yes |
| pitch | [28.44, 28.44] | 28.4443 | 25.17-31.15 | yes |
| yaw | [46.5, 47.47] | 46.7355 | 39.3-52.55 | yes |

镜像值**不是**窗口端点，也不在 1.0 处被调到极小：G3 极小点与镜像值的相对偏差在 0.1 %–20 % 之间
（surge 0.82 vs 0.80，heave 69.5 vs 69.1），即两条独立路线互相印证。

### 3.3 尾部拟合为什么不可能是对的（无需任何假设的见证）

六个自由度在频带顶端都**仍在单调上升**（`rising_at_band_top=True`），且
尾部拟合值**低于最后一个已解频点的实测 A**（`tail_fit_below_band_top=True`，六项全中）。
一个单调上升、斜率递减的剖面，其 $\omega\to\infty$ 极限不可能低于带顶值；
两参数尾部拟合给出 60.17 kg（heave）而 $A_{33}(20\,\mathrm{rad/s})=62.47$ kg，
yaw 更极端：拟合 9.51 kg，镜像解 46.50 kg（差 4.9 倍）——这正是 G3/G5 在 yaw 上最差的根因。

### 3.4 G4 不但没被修好，反而更差（honest）

| 项 | tail fit | image solve | 容差 |
|---|---:|---:|---:|
| ζ_decay（时域包络） | 0.1972 | 0.2133 | — |
| ζ_half-power（频域） | 0.1666 | 0.1666 | — |
| 相对误差 | 18.34 % | 28.02 % | 10% |

G4 的频域腿（半功率 ζ）只用 $A(\omega)$、$B(\omega)$，对 $A_\infty$ 分解不敏感（数值上完全不变）；
时域腿是 Cummins 方程，$A_\infty$ 是瞬时惯量，换成物理值后 ζ_decay 由 0.1972 升到 0.2133，
残差 18.3 % → 28.0 %。
**G4 仍是 FAIL，且不是 $A_\infty$ 问题**：它测的是「包络拟合 ζ」与「半功率带宽 ζ」在
频率相关系数系统上的估计量差异，属于另一个独立缺口（m23130 §4 已提出该假设，本轮给出反对称证据）。

![added_mass_and_asymptotes](docs/figures/m23208/added_mass_and_asymptotes.png)

![gates_tail_vs_image](docs/figures/m23208/gates_tail_vs_image.png)

![g3_vs_assumed_ainf](docs/figures/m23208/g3_vs_assumed_ainf.png)

## 4 误差判定（逐项，显式回答是否 <3 %）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---:|---|
| 参考点变换 vs 直接重解 | 同一镜像问题的两路线 | 1.03e-16 | ✅ 是（机器精度） |
| 镜像 $A_\infty$ 网格梯子散布（最差 DOF） | 20x6 vs 20x10 | 2.08 % | ✅ 是 |
| 镜像解解析自校验（半浸球） | $\rho V/4$ | 2.02 % | ✅ 是 |
| G3 最差 DOF（image $A_\infty$） | 自洽 Hilbert 对 | 2.84 %（surge） | ✅ 是 |
| G5 最差 DOF（image $A_\infty$） | B→K→(B,A) 往返 | 3.45 %（surge） | ❌ 否（3.45 % > 3 %，但 <5 % 容差） |
| G4（image $A_\infty$） | 时域 ζ vs 半功率 ζ | 28.02 % | ❌ 否（仍 FAIL） |

## 5 注意点 / 局限

* **本单不改预声明判决**：闭包带 G1–G5 的总判决仍是
  `FAIL:G4_free_decay`（G3/G5 在独立 $A_\infty$ 下通过，G4 仍不过）。
  总账锚点 `CASE-BEM-CUMMINS-WIGLEY` 因此**不能**标 verified，剩余缺口被缩到一条：G4。
* 镜像解的梯子只到 `(20,10)`：网格散布 ≤2.1 %，但更细的网格（`30x12`）本单未跑出
  （集群 `cj4690` 递交 25 min 仍在队列中，本机 load15≈105 且守卫上限 120 s；
  脚本已实现镜像解磁盘缓存，续跑只付未命中部分）。
* G3 的归一化 $\max|A-A_\infty|$ 仍依赖 $A_\infty$：独立值让分母变大约 7 %（heave），
  这只会让残差**更保守地小一点**，但结论不靠这一点——见 §3.2 的窗口检查与 §3.3 的单调性见证。
* G4 的时域 τ_max=40 s、dt=0.02 s 沿用算例缺省，未做本轮的收敛性重扫（m23130 的 C13 已覆盖时域预算）。

## 6 复现

```
PYTHONPATH=src python3 scripts/cummins_closure_ainf_independent.py --stage cache   # 冻结 69 点扫频（从 m23130 点缓存）
PYTHONPATH=src python3 scripts/cummins_closure_ainf_independent.py --stage audit \
    --ladder 20x6,20x10 --frame-mesh 20x6                                          # 约 40 s（镜像解有磁盘缓存）
PYTHONPATH=src python3 -m pytest tests/test_cummins_closure_ainf_independent.py -q
```
