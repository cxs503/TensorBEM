# CASE-BEM-MARUO-DRIFT — Maruo 远场漂移力 vs 散射幅值闭式（竖直圆柱）

**Case**: `CASE-BEM-MARUO-DRIFT` · **Repo**: TensorBEM · **Branch**: agent/m17276 · **Mission**: 17276
**Server-side report**: id 16290（`/tensor/reports`）

## 1. 判据（先于运行声明）

| 判据 | 阈值 | 含义 |
|---|---|---|
| G3 | BEM 远场漂移力 vs 闭式 ref 相对误差 **< 5%** | 本算例容差（任务给定） |
| G1 | 闭式（模态和）vs 仓库求积路线 < 1e-10 | ref 自身可复算 |
| G2 | 闭式 vs 符号常量 5π²/16（ka=0.01/0.02）< 1e-3 | 长波锚点 |
| G4 | Maruo 核方位求积散布（n_theta=128…4096）< 1e-2 | 求积不主导误差 |

## 2. ref（闭式，零手抄）

Maruo (1957) 远场公式 `F_x = (rho g A^2 / 2 pi k) ∫|S(θ)|²(1-cosθ)dθ`。
圆柱散射谱为有限三角多项式，故积分有精确闭式
`∫|S|²(1-cosθ)dθ = 2π[Σ|c_m|² - Re Σ c_m conj(c_{m+1})]`，`c_m = A_m exp(-imπ/2)`。

闭式 vs 仓库独立求积路线（`cylinder_drift.far_field_cylinder_drift`）：

| ka | closed form [N] | 求积路线 [N] | rel diff |
|---|---|---|---|
| 0.2 | 240.5907 | 240.5907 | 2.3627e-16 |
| 0.5 | 2875.6570 | 2875.6570 | 0.0000 |
| 1 | 6685.9935 | 6685.9935 | 0.0000 |
| 2 | 6306.3981 | 6306.3981 | 0.0000 |

长波锚点（5π²/16 由 `math.pi` 计算）：

| ka | (5π²/16) | measured | rel err |
|---|---|---|---|
| 0.01 | 3.084251375 | 3.084203600 | 1.5490e-05 |
| 0.02 | 3.084251375 | 3.083891060 | 1.1684e-04 |
| 0.05 | 3.084251375 | 3.080622456 | 0.0012 |

## 3. 参数

ρ=1025 kg/m³, g=9.81 m/s², a=1 m, A=1 m,
H/a=16, m_max=14, n_azimuth=96, n_theta=2048,
ka ∈ [0.2, 0.5, 1.0, 2.0]，生产分辨率 `fine`。

## 4. 对照表（全部 16 个网格单元，不挑）

| ka | resolution | panels | got F_BEM [N] | ref F_closed [N] | rel err | <5% |
|---|---|---|---|---|---|---|
| 0.2 | coarse | 312 | 222.7147 | 240.5907 | 7.43% | FAIL |
| 0.2 | medium | 544 | 229.4834 | 240.5907 | 4.62% | PASS |
| 0.2 | fine | 800 | 233.2949 | 240.5907 | 3.03% | PASS |
| 0.2 | finer | 1200 | 233.9049 | 240.5907 | 2.78% | PASS |
| 0.5 | coarse | 312 | 2580.7903 | 2875.6570 | 10.25% | FAIL |
| 0.5 | medium | 544 | 2701.6146 | 2875.6570 | 6.05% | FAIL |
| 0.5 | fine | 800 | 2779.9799 | 2875.6570 | 3.33% | PASS |
| 0.5 | finer | 1200 | 2788.3032 | 2875.6570 | 3.04% | PASS |
| 1 | coarse | 312 | 5919.1530 | 6685.9935 | 11.47% | FAIL |
| 1 | medium | 544 | 6239.4142 | 6685.9935 | 6.68% | FAIL |
| 1 | fine | 800 | 6480.7722 | 6685.9935 | 3.07% | PASS |
| 1 | finer | 1200 | 6485.0329 | 6685.9935 | 3.01% | PASS |
| 2 | coarse | 312 | 4884.7166 | 6306.3981 | 22.54% | FAIL |
| 2 | medium | 544 | 5509.1819 | 6306.3981 | 12.64% | FAIL |
| 2 | fine | 800 | 5970.2151 | 6306.3981 | 5.33% | FAIL |
| 2 | finer | 1200 | 5988.7699 | 6306.3981 | 5.04% | FAIL |

## 5. 判定（逐 ka，生产分辨率 `fine`）

| ka | 相对误差 | <5% | <3% |
|---|---|---|---|
| 0.2 | 3.03% | 是 | 否 |
| 0.5 | 3.33% | 是 | 否 |
| 1 | 3.07% | 是 | 否 |
| 2 | 5.33% | **否（honest FAIL）** | 否 |

闸门：

| gate | threshold | measured | verdict |
|---|---|---|---|
| `G1_closed_form_vs_quadrature` | 1e-10 | 2.362661e-16 | pass |
| `G2_long_wave_coefficient` | 0.001 | 1.168380e-04 | pass |
| `G3_bem_vs_ref_production` | 0.05 | 0.053308 | **FAIL** |
| `G4_quadrature_spread` | 0.01 | 1.713690e-07 | pass |

**结论（honest）**：<5% 在 ka ≤ 1.0 成立（最大 3.33%）；
**ka=2.0 未达标（5.33% @ fine，finer 网格 5.04%）**，
`G3_bem_vs_ref_production` 记 FAIL，未粉饰。误差随网格收敛在 ka=2 处趋于平台（12.64% → 5.33% → 5.04%），
指向短波段的模型/离散极限而非随机噪声。

## 6. 敏感性

1. 网格阶梯（表 4）：误差单调下降；关键变量是自由面附近垂向 grading，而非单纯环数。
2. Maruo 核方位求积（精确场，n_theta 128…4096）：

| ka | max n_theta | 全域散布 |
|---|---|---|
| 0.2 | 4096 | 1.2127e-07 |
| 0.5 | 4096 | 1.1962e-07 |
| 1.0 | 4096 | 7.9752e-08 |
| 2.0 | 4096 | 1.7137e-07 |

3. Kochin 方位审计与求解残差（生产网格）：

| ka | panels | levels | grading | Kochin azimuth rel. change | solver residual |
|---|---|---|---|---|---|
| 0.2 | 32 | 12 | 3 | 3.7093e-04 | 2.1745e-15 |
| 0.5 | 32 | 12 | 3 | 4.3081e-04 | 2.2594e-15 |
| 1 | 32 | 12 | 3 | 8.0723e-04 | 2.1929e-15 |
| 2 | 32 | 12 | 3 | 0.0021 | 2.5185e-15 |

4. 长波律的偏离按 (ka)² 衰减（表 2：ka=0.01 → 1.5e-5，0.02 → 1.2e-4，0.05 → 1.2e-3）。

## 7. 失败模式与生态摩擦

- 网格不足系统性**低估**漂移力（coarse 11–23%）。
- ka=2 平台：短波段 5% 边界，需更大方位/垂向分辨率或高阶面板（未在本单解决）。
- 幅值约定双轨：`cylinder_drift.kochin_3d`（单位势幅值）喂 `drift_far_field3d.maruo_far_field_drift`
  会低估 (g/ω)²=g/k 倍（`quadrature_sensitivity` 实测比值对 k/g 误差 <1e-5）。
- `maruo_drift.maruo_mean_drift_force`（ρΩ²/4k）与 `drift_far_field3d.maruo_far_field_drift`（ρk²/8π）
  前因子不同，对同一 K 表差 2π/(ka)——两模块各自自洽、混用即错，本单只选一条并记录。
- 控制面 99 负载过高（load≈370/128 核），BEM 网格阶梯最终以 4 个分离进程在本机检出内完成；
  集群作业 cj1477 排队未在窗口内派发（队列积压 >90 min），已如实记录。

## 8. 复现

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.maruo_drift_far_field_case --ka 1.0 \
    --json artifacts/maruo_drift_far_field_case/cell.json
PYTHONPATH=src python3 -m tensorbem.benchmarks.maruo_drift_far_field_case --merge \
    artifacts/maruo_drift_far_field_case/cell_ka_*.json \
    --json artifacts/maruo_drift_far_field_case/study.json
PYTHONPATH=src python3 -m pytest tests/test_maruo_drift_far_field_case.py -q
```
