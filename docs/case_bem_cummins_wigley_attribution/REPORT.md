# CASE-BEM-CUMMINS-WIGLEY · G3 残差归因：渐近值偏移 vs 形状

**一句话**：锚点声明的 G3（Kramers-Kronig 因果）残差 coarse 53.23% / medium 81.62% 中，**形状（真因果）部分只有 5.63% / 31.88%**，其余是 `A_inf` 尾段外推的常数偏移（最坏 49.23% / 55.62%）。精确因果对照（Ursell 圆柱闭式解，`B(0)=0`、`A_inf=rho*pi*a^2` 精确已知）给出的估计器地板 = **1.12%（coarse 22 点）/8.04%（medium 16 点频带）**。故「求解器违反因果」不成立；锚点仍 open（声明判据 <10% 未破）。

## 1 案例（物理背景/几何/参数）
Wigley III：L=3.0 m、B=0.3 m、T=0.1875 m、Cb=0.543、ρ=1025 kg/m³、g=9.80665、无限水深、旋转中心 (0,0,-T/2)；TensorBEM 3D 面元辐射扫频（coarse 400 三角形、medium 800）。判据 `A(w)-A_inf=(2/pi)PV∫B(s)/(s^2-w^2)ds`，`A_inf` 不在扫频内 → 只能尾段外推，本报告数值证明这正是残差主项。

## 2 方法（流程/关键设置/工具链）
新增 `src/tensorbem/causality_attribution.py`：①`kkr_decomposition` 对逐点残差做 Chebyshev(minimax) 常数分解 → 「形状残差」（任何常数都压不下去）+「隐含 A_inf」；②Ursell 闭式精确因果对标定地板，振荡子替代对做 below-band 闭合敏感性对照；③`sum_rule_asymptote`：`A-A_inf=-(2/pi)w^-2∫B ds+O(w^-4)` 的斜率只含 B，单参数解 A_inf 作第二路线；④反号负对照。数据源 = 审计方冻结件 `docs/board_audit/m17340/subject_results.json`（coarse 22/22、medium 16 点）。

## 3 计算结果（程序化注入，零手抄）
| level | DOF | 声明 G3 % | 形状 % | 偏移 % | A_inf(尾段) | A_inf(KK隐含) | A_inf(和规则) | 和规则下 G3 % | 反号负对照形状 % |
|---|---|---|---|---|---|---|---|---|---|
| coarse | surge | 44.26 | 5.63 | 38.62 | -0.1941 | 1.092 | 1.456 | 32.86 | 90.62 |
| coarse | sway | 48.81 | 4.23 | 44.58 | -2.394 | 79.19 | 97.11 | 30.73 | 84.67 |
| coarse | heave | 14.13 | 0.56 | 13.57 | 50.06 | 69.3 | 66.79 | 2.64 | 106.86 |
| coarse | roll | 49.44 | 5.60 | 43.85 | -0.05482 | 0.3377 | 0.4501 | 41.62 | 86.53 |
| coarse | pitch | 22.78 | 1.87 | 20.91 | 19.36 | 28.12 | 27.38 | 4.51 | 104.95 |
| coarse | yaw | 53.23 | 4.00 | 49.23 | -14.25 | 57.8 | 75.22 | 40.91 | 79.22 |
| medium | surge | 64.35 | 15.23 | 49.11 | 0.4034 | 1.748 | 3.804 | 88.30 | 92.62 |
| medium | sway | 75.56 | 19.94 | 55.62 | 46.22 | 117 | 219.3 | 90.96 | 92.65 |
| medium | heave | 39.84 | 4.46 | 35.37 | 22.63 | 83.66 | 157.2 | 70.16 | 93.57 |
| medium | roll | 76.55 | 21.22 | 55.33 | 0.1937 | 0.5576 | 1.093 | 92.64 | 93.52 |
| medium | pitch | 58.89 | 5.69 | 53.19 | 2.734 | 35.22 | 68.44 | 74.13 | 78.85 |
| medium | yaw | 81.62 | 31.88 | 49.74 | 55.49 | 87.58 | 149.7 | 102.66 | 110.01 |

估计器地板（Ursell，A_inf 精确）：

| level | A_inf 真值 | 形状残差 % | 总残差 % | 尾段拟合 A_inf | 尾段拟合总残差 % |
|---|---|---|---|---|---|
| coarse | 28.3019 | 1.12 | 40.92 | 26.9945 | 10.85 |
| medium | 28.3019 | 8.04 | 34.26 | 24.3920 | 59.73 |

地板 vs 频带采样密度（Ursell，oracle A_inf）：

| level | 节点数 | 形状残差 % | 尾段拟合总残差 % | 尾段拟合 A_inf |
|---|---|---|---|---|
| coarse | 22 | 1.95 | 35.47 | 25.8025 |
| coarse | 44 | 0.44 | 33.36 | 25.8556 |
| coarse | 88 | 0.10 | 34.54 | 25.7510 |
| coarse | 176 | 0.09 | 35.26 | 25.7005 |
| medium | 22 | 9.06 | 56.31 | 24.7995 |
| medium | 44 | 11.61 | 60.13 | 24.6212 |
| medium | 88 | 15.91 | 63.60 | 24.6857 |
| medium | 176 | 21.00 | 68.05 | 24.7173 |

below-band 闭合敏感性（Ursell，`B(0)=0`）：

| level | 闭合模型 | 形状残差 % |
|---|---|---|
| coarse | constant | 1.12 |
| coarse | zero | 1.12 |
| coarse | linear | 1.12 |
| medium | constant | 8.04 |
| medium | zero | 8.04 |
| medium | linear | 8.04 |

## 4 图
`fig_attribution_bars.png`（残差三段分解 + 10% 容差线）、`fig_heave_prediction.png`（heave 实测 A33 vs Hilbert(B)，两种 A_inf）、`fig_calibration.png`（精确因果对地板随采样加密）。

## 5 注意点（坑/局限/适用范围）
- 审计 F4 把 G3 归因于低频截断；分层分解与闭合敏感性实验都指向 **A_inf 尾段外推**（误差 4.62% of A，折合动态范围几十个百分点），不是低频截断。
- medium 仅 16 点、频带止 7.8 rad/s（缺 4.7/6.2/8.6/9.6/11.0/12.5），其形状残差 31.88% 含频带缺口的分段线性求积误差，不可与 coarse 直接比较；补算已递交集群（cj2005/cj2006），本单未采用未收敛结果。
- 振荡子替代对核均值非零（`B(0)>0`）与浮体不同，只能作闭合敏感性对照，不可当地板。
- `sum_rule_asymptote` 用截断频带求 ∫B dω，缺高端尾，sway/yaw 仍偏高；是第二意见，不是新判据。
- 结论限于本几何/本频带；病态程度随 `max|A-A_inf|/A` 变小而加剧（sway/roll/yaw 最病态）。

## 6 误差判定（vs 基准，显式回答 <3%）
- 跨求解器 G1（capytaine 3.1.dev0 冻结参照）：coarse 6.29%、medium 3.26% → **否（>3%）**，但满足锚点 10%。
- 形状残差 vs 精确因果对地板：coarse 最坏 5.63% vs 地板 1.12%（5.0×）→ 相对地板 **否（>3%）**；相对判据 10% → 是。
- `A_inf` 尾段拟合 vs Ursell 精确值：4.62%（coarse）/ 13.82%（medium）of A → **是（<3%）? 否（medium >3%）**，如实列出；绝对值小但折合动态范围即上面几十个百分点，这是放大机制。
- 反号负对照形状残差最坏 110.01% → 判据能识别非因果，非空门。

## 7 处置与返修
1. G3 要过必须独立定 `A_inf`：自由面 `φ=0` 刚壁的高频/脉冲极限面元解，或把扫频上端推到 A 真正进入 `1/w²` 段；**不要**再用两参数尾段拟合判 G3。
2. medium 补满 22 点后重生成 `results.json`（审计 F6），再给 coarse→medium 形状残差收敛证据。
3. 五门登记进 `tensor_benchmark_cases`（审计 F2）——本单未写库（控仓令）。
4. 锚点保持 open：声明判据未通过，本单只更正**归因**。

## 8 复现
```
PYTHONPATH=src python3 scripts/run_cummins_wigley_kk_attribution.py --from-results docs/board_audit/m17340/subject_results.json --levels coarse,medium --outdir docs/case_bem_cummins_wigley_attribution --figures
PYTHONPATH=src python3 -m pytest tests/test_causality_attribution.py -q
```
