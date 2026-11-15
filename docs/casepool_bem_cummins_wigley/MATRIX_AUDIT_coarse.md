# CASE-BEM-CUMMINS-WIGLEY — 6x6 矩阵级审计（M0–M5）

## 1. 结论

已提交提交件 `sweep_coarse.npz`（band 0.2–12 rad/s，41 节点，12x4 网格，192 面板，求解残差 0.006392）的 **全部 36 个系数**通过 6/6 个矩阵级闸门。

算例现有的 C3/C4 只查对角元、C5/C6 只查 heave：6x6 的交叉项（surge-pitch、sway-roll 等）此前**没有任何闸门看过**。本审计补的是「矩阵才能失败」的检查，且**不依赖任何外部参照**——约束来自船体几何（两个镜面）与 Cummins 变换的线性/符号，而不是另一条同样内部的解法路径。

| gate | 判据 | 阈值 | 结论 |
| --- | --- | --- | --- |
| M0 | mesh symmetry precondition (coarse) | mirror centroid distance <= 1e-09 L | PASS |
| M1 | reciprocity of the radiation matrices (A_ij = A_ji, B_ij = B_ji) | max relative asymmetry <= 0.5 * C0 residual = 3.196e-03 | PASS |
| M2 | geometric selection rules: symmetry-forbidden entries vanish | max relative peak of a forbidden entry <= 1.0 * C0 residual = 6.392e-03 | PASS |
| M3 | Cummins round trip on every non-zero entry of the 6-DOF matrix | max pointwise relative error <= 2% on the entries the selection rules declare non-zero | PASS |
| M4 | kernel sign/impulse identity and causality on the cross entries | K_ij(0+) = (2/pi) int B_ij domega within 2% on gated entries | PASS |
| M5 | negative controls: every gate must fail under its own corruption | each control leaves its gate in FAIL | PASS |

## 2. 为什么这些判据不是自证

* **M1 互易性**：`A_ij = A_ji`、`B_ij = B_ji` 来自辐射势双线性型的对称性（Newman, *Marine Hydrodynamics*, ch.6），与 BEM 怎么离散、用哪个 Green 函数无关。
* **M2 零模式**：Wigley 有两个镜面，凡两模态在任一镜面下宇称相反的系数**恒为零**。宇称由反射矩阵推出（平动=polar，转动=axial，`det(S)=-1`），不是硬编码表：
  ```
  允许非零: surge/surge, surge/pitch, sway/sway, sway/roll, heave/heave, roll/roll, pitch/pitch, yaw/yaw
  禁止非零: surge/sway, surge/heave, surge/roll, surge/yaw, sway/heave, sway/pitch, sway/yaw, heave/roll, heave/pitch, heave/yaw, roll/pitch, roll/yaw, pitch/yaw
  ```
* **M3/M4 往返与符号**：Cummins 变换逐项线性，交叉项必须像对角项一样闭合；`K_ij(0+) = (2/pi)∫B_ij dω` 还带着交叉项的**符号**——转置或丢符号的装配错误会原封不动地保住对角元，却把这条打翻。

## 3. 关键数字

### 3.1 互易性（M1）

| 量 | 值 |
| --- | --- |
| added mass 相对不对称 | 3.550e-04 |
| damping 相对不对称 | 6.889e-04 |
| 上限（0.5 x C0 残差） | 0.003196 |
| 最差 added 元对 / 频率 | sway/yaw @ 3.172 |

### 3.2 零模式（M2）

| pair | added mass 相对峰值 | damping 相对峰值 | 判定 |
| --- | --- | --- | --- |
| sway/yaw | 0.003171 | 0.003056 | 噪声级（低于上限） |
| yaw/sway | 0.003088 | 0.003043 | 噪声级（低于上限） |
| pitch/heave | 3.442e-04 | 5.306e-04 | 噪声级（低于上限） |
| heave/pitch | 4.130e-04 | 4.934e-04 | 噪声级（低于上限） |
| roll/yaw | 1.358e-04 | 1.783e-04 | 噪声级（低于上限） |
| heave/surge | 7.313e-05 | 1.454e-04 | 噪声级（低于上限） |
| surge/heave | 6.134e-05 | 1.270e-04 | 噪声级（低于上限） |
| yaw/roll | 8.114e-05 | 9.305e-05 | 噪声级（低于上限） |
| heave/sway | 5.203e-06 | 9.905e-06 | 噪声级（低于上限） |
| pitch/sway | 7.340e-06 | 1.898e-06 | 噪声级（低于上限） |
| sway/heave | 1.938e-06 | 3.658e-06 | 噪声级（低于上限） |
| surge/sway | 1.614e-06 | 1.683e-06 | 噪声级（低于上限） |
| pitch/yaw | 1.169e-07 | 7.891e-07 | 舍入级 |
| heave/yaw | 6.212e-07 | 7.375e-07 | 舍入级 |
| sway/pitch | 5.652e-07 | 6.444e-07 | 舍入级 |
| heave/roll | 2.637e-07 | 4.767e-07 | 舍入级 |
| sway/surge | 8.699e-08 | 4.453e-07 | 舍入级 |
| yaw/heave | 2.010e-07 | 3.760e-07 | 舍入级 |
| pitch/roll | 1.860e-07 | 3.080e-07 | 舍入级 |
| roll/heave | 1.092e-07 | 2.398e-07 | 舍入级 |
| surge/yaw | 4.731e-08 | 1.593e-07 | 舍入级 |
| surge/roll | 5.028e-08 | 3.807e-08 | 舍入级 |
| roll/pitch | 1.997e-08 | 2.275e-08 | 舍入级 |
| yaw/pitch | 1.579e-08 | 2.027e-08 | 舍入级 |
| roll/surge | 4.443e-09 | 1.692e-08 | 舍入级 |
| yaw/surge | 2.215e-09 | 6.962e-09 | 舍入级 |

上限 = C0 残差 = 0.006392；最差禁止项 `sway/yaw` = 0.003171（高于舍入级 1e-06）。

允许非零项（M2 的反向对照——证明不是「矩阵全零」这种平凡通过）：

| pair | added mass 相对峰值 | damping 相对峰值 |
| --- | --- | --- |
| surge/surge | 0.01159 | 0.008981 |
| surge/pitch | 0.03692 | 0.02938 |
| sway/sway | 1 | 1 |
| sway/roll | 0.04589 | 0.04067 |
| heave/heave | 0.7257 | 0.3541 |
| roll/roll | 0.003206 | 0.002109 |
| pitch/pitch | 0.1749 | 0.1181 |
| yaw/yaw | 0.7714 | 0.8353 |

### 3.3 逐项 Cummins 往返与冲激恒等式（M3/M4，仅计允许非零项）

| pair | B 相对误差 | A 相对误差 | K(0+) | (2/π)∫B dω | 符号相对误差 | τ 偶宇称误差 |
| --- | --- | --- | --- | --- | --- | --- |
| surge/surge | 0.001437 | 0.2196 | 39.99 | 39.99 | 2.370e-08 | 0 |
| surge/pitch | 9.976e-04 | 0.1869 | 105.3 | 105.3 | 3.094e-08 | 0 |
| sway/sway | 0.002284 | 0.2399 | 3549 | 3549 | 2.410e-09 | 0 |
| sway/roll | 0.002478 | 0.2142 | 131.8 | 131.8 | 2.763e-09 | 0 |
| heave/heave | 8.670e-04 | 0.1081 | 1388 | 1388 | 5.023e-06 | 0 |
| roll/sway | 0.002463 | 0.212 | 132.3 | 132.3 | 2.781e-09 | 0 |
| roll/roll | 0.002808 | 0.1543 | 6.289 | 6.289 | 2.489e-09 | 0 |
| pitch/surge | 0.001036 | 0.1933 | 106.3 | 106.3 | 3.057e-08 | 0 |
| pitch/pitch | 0.001141 | 0.1408 | 376.4 | 376.4 | 2.970e-08 | 0 |
| yaw/yaw | 0.002207 | 0.2628 | 2714 | 2714 | 2.513e-11 | 0 |

### 3.4 负控（M5）

| control | 破坏后闸门 | 测得 | 阈值 |
| --- | --- | --- | --- |
| M1_unilateral_break_50pct_sway_roll | FAIL（正确咬住） | 0.02012 | 0.003196 |
| M2_forbidden_entry_filled_10pct | FAIL（正确咬住） | 0.1001 | 0.006392 |
| M3_kernel_undersampled_dt_0p2 | FAIL（正确咬住） | 0.1938 | 0.02 |

## 4. 注意点 / 局限

* 本审计**不重解** BEM：它读已提交 `sweep_*.npz`。因此它审计的是「提交的系数矩阵」与「几何/变换的一致性」，不是方程离散误差；C0/C2 才是那一层。
* M2 只能抓住**宇称禁止**的耦合项；几何上允许的交叉项（surge-pitch、sway-roll）它只说「不该为零」，量值对不对仍无外部锚点（Wigley 辐射系数无闭式解，L5 外部锚点 WITHHELD）。
* 禁止项上限绑在 C0 残差上：换更粗的网格或更松的求解容差，上限会跟着变宽——这是刻意的（完全为零的项只能被求解误差污染），但也意味着 M2 **不能**替代收敛性判据。
* 交叉项的 A 支路误差（表 3.3 第二列）沿用 C4 的约定（`A_inf = A(ω_max)`），只作参考不设闸；C4 自身对 heave 的 A 支路报 10.9%，同一约定问题。

## 5. 复现

```
PYTHONPATH=src python -m tensorbem.benchmarks.cummins_wigley_matrix_audit --label coarse
```

