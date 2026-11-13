## ① 案例介绍（问题 + ref 出处）

**问题**：固定、坐底、竖直圆柱（半径 a，深水）在规则头浪下的二阶平均漂移力，
由 Maruo (1957) 远场公式给出：

    F_x = (rho g A^2 / (2 pi k)) ∫_0^{2pi} |S(theta)|^2 (1 - cos theta) dtheta,

其中 S(θ) 是散射势远场散射幅值（`psi_s ~ sqrt(2/(pi k r)) exp(i(kr - pi/4)) S(theta)`）。
本算例的决定是：TensorBEM 的三维 BEM 链路（`solve_diffraction_3d` → `kochin_function_3d`
→ 约化 Maruo 远场核）能否把该闭式复现到 **<5%**。

**ref 出处（闭式，非手抄表）**：
1. Maruo, H. (1957) "On the theory of ship drift in waves", J. Soc. Naval Arch. Japan **102**
   —— 远场平均漂移力公式（本仓约定见 `src/tensorbem/maruo_drift.py` 模块 docstring 与
   `docs/maruo_drift.md`）。
2. Maruo, H. (1960) "The drift of a body floating on waves", J. Ship Research **4**(3), 1–10。
3. Newman, J.N. (1977) *Marine Hydrodynamics*, MIT Press, **Sec. 7.3** —— 远场漂移力的教科书形式。
4. MacCamy, R.C. & Fuchs, R.A. (1954) "Wave forces on piles: a diffraction theory",
   Tech. Memo. 69, Beach Erosion Board —— 圆柱精确散射幅值（闭式级数）。
5. 长波极限 `F_x = (5 pi^2/16) rho g A^2 a (ka)^3`（本报告符号常量由 `math.pi` 计算，
   推导见 `docs/cylinder_drift_asymptotic.md` §2）。

**零手抄说明（诚实申报）**：沙箱无外网，无法逐页核对原文表格；因此 ref 数值不是抄表，而是
**对上述闭式做机器求值**：圆柱的散射谱是有限项三角多项式 S(θ)=Σ_m c_m e^{imθ}
（c_m = A_m exp(-imπ/2)），于是 Maruo 方位积分有精确闭式

    ∫|S|^2(1-cosθ)dθ = 2π [ Σ_m |c_m|^2 - Re Σ_m c_m conj(c_{m+1}) ],

与任何方位求积无关（本报告用仓库独立求积路线交叉验证到 2.3627e-16）。
`(ka)→0` 时该闭式退化为 5π²/16 长波律，这是第二个独立锚点（见 ④）。

**无量纲参数表**：

| 参数 | 值 | 含义 |
|---|---|---|
| ρ | 1025 kg/m³ | 海水密度 |
| g | 9.81 m/s² | 重力加速度 |
| a | 1 m | 圆柱半径（几何尺度） |
| A | 1 m | 入射波幅 |
| ka | 0.2, 0.5, 1, 2 | 无量纲波数（覆盖长波/峰值/短波段） |
| H/a | 16 | 吃水/半径（深水 Green 函数看不到底：exp(-2kH)≤1e-14） |
| m_max | 14 | MacCamy–Fuchs 模态截断（m_max=8 与 16 差 <1e-9） |
| n_azimuth | 96 | BEM Kochin 方位采样点数 |
| n_theta | 2048 | Maruo 核方位求积点数 |

## ② 方法与实现

**got（被测路线）**：
1. `tensorbem.diffraction_drift_maturity.vertical_cylinder_hull_mesh` 建水密壁面圆柱网格
   （三角形，自由面附近按 `z=-H(j/n_l)^grading` 加密）。
2. `tensorbem.free_surface_solver3d.solve_diffraction_3d` 解一阶绕射（Rankine 面板 + 自由面条件）。
3. `tensorbem.drift_far_field3d.bem_kochin` 由散射势与 `-∂φ_I/∂n` 求物理 Kochin 函数
   （幅值约定：入射势带 gA/ω，见 `AMPLITUDE_NORMALISATION`）。
4. `tensorbem.drift_far_field3d.maruo_far_field_drift` 用无半径依赖的约化 Maruo 核算
   `F_x = (ρk²/8π)∫|K|²(1-cosθ)dθ`（周期梯形求积 n_theta=2048）。

**ref（对照）**：`maruo_integral_closed_form` / `closed_form_case`（本单新增，
`src/tensorbem/benchmarks/maruo_drift_far_field_case.py`）——散射幅值模态闭式，零求积。

**为什么复用而不另起炉灶**：BEM 解、Kochin、约化 Maruo 核、圆柱精确解四条路线仓内都已存在并有测试；
本单只补「闭式 ref + 算例驱动 + 分辨率阶梯」这一层。

**一键复现**：

```bash
cd <worktree>
PYTHONPATH=src python3 -m tensorbem.benchmarks.maruo_drift_far_field_case \
    --json artifacts/maruo_drift_far_field_case/study.json \
    --figures artifacts/maruo_drift_far_field_case
PYTHONPATH=src python3 -m pytest tests/test_maruo_drift_far_field_case.py -q
```

## ③ 参数表

几何唯一（a=1 m 壁面圆柱，H/a=16），工况由 ka 与网格分辨率二维网格给出：

| 分辨率 | n_theta 面板 | 环数 | grading | 三角形总数 |
|---|---|---|---|---|
| coarse | 24 | 6 | 2.5 | 312 |
| medium | 32 | 8 | 2.5 | 544 |
| fine | 32 | 12 | 3 | 800 |
| finer | 48 | 12 | 3 | 1200 |

生产分辨率 = **fine**（阶梯全部结果都列出，不挑好看的）。

## ④ 对照表（ref vs got，误差逐项）

| ka | resolution | panels | got F_BEM [N] | ref F_closed [N] | rel err | <5% ? |
|---|---|---|---|---|---|---|
| 0.2 | coarse | 312 | 222.7147 | 240.5907 | 7.43% | ❌ |
| 0.2 | medium | 544 | 229.4834 | 240.5907 | 4.62% | ✅ |
| 0.2 | fine **(prod)** | 800 | 233.2949 | 240.5907 | 3.03% | ✅ |
| 0.2 | finer | 1200 | 233.9049 | 240.5907 | 2.78% | ✅ |
| 0.5 | coarse | 312 | 2580.7903 | 2875.6570 | 10.25% | ❌ |
| 0.5 | medium | 544 | 2701.6146 | 2875.6570 | 6.05% | ❌ |
| 0.5 | fine **(prod)** | 800 | 2779.9799 | 2875.6570 | 3.33% | ✅ |
| 0.5 | finer | 1200 | 2788.3032 | 2875.6570 | 3.04% | ✅ |
| 1 | coarse | 312 | 5919.1530 | 6685.9935 | 11.47% | ❌ |
| 1 | medium | 544 | 6239.4142 | 6685.9935 | 6.68% | ❌ |
| 1 | fine **(prod)** | 800 | 6480.7722 | 6685.9935 | 3.07% | ✅ |
| 1 | finer | 1200 | 6485.0329 | 6685.9935 | 3.01% | ✅ |
| 2 | coarse | 312 | 4884.7166 | 6306.3981 | 22.54% | ❌ |
| 2 | medium | 544 | 5509.1819 | 6306.3981 | 12.64% | ❌ |
| 2 | fine **(prod)** | 800 | 5970.2151 | 6306.3981 | 5.33% | ❌ |
| 2 | finer | 1200 | 5988.7699 | 6306.3981 | 5.04% | ❌ |

无量纲形式（F/(ρ g A² a)，与 a 无关）：

| ka | F_ref/(rho g A^2 a) | BEM/(rho g A^2 a) (prod) | rel err |
|---|---|---|---|
| 0.2 | 0.0239 | 0.0232 | 3.03% |
| 0.5 | 0.2860 | 0.2765 | 3.33% |
| 1 | 0.6649 | 0.6445 | 3.07% |
| 2 | 0.6272 | 0.5937 | 5.33% |

长波锚点（`(ka)^3` 律 5π²/16 的独立核对）：

| ka | (5π²/16) | measured coefficient | rel err of the (ka)^3 law |
|---|---|---|---|
| 0.01 | 3.084251375 | 3.084203600 | 1.5490e-05 |
| 0.02 | 3.084251375 | 3.083891060 | 1.1684e-04 |
| 0.05 | 3.084251375 | 3.080622456 | 0.0012 |

闭式 ref 与仓库独立求积路线（`cylinder_drift.far_field_cylinder_drift`）的对照：

| ka | closed form (modal) [N] | quadrature route [N] | rel diff |
|---|---|---|---|
| 0.2 | 240.5907 | 240.5907 | 2.3627e-16 |
| 0.5 | 2875.6570 | 2875.6570 | 0 |
| 1 | 6685.9935 | 6685.9935 | 0 |
| 2 | 6306.3981 | 6306.3981 | 0 |

判据汇总：

| gate | threshold | measured | verdict |
|---|---|---|---|
| `G1_closed_form_vs_quadrature` | 1e-10 | 2.3627e-16 | ✅ pass |
| `G2_long_wave_coefficient` | 0.001 | 1.1684e-04 | ✅ pass |
| `G3_bem_vs_ref_production` | 0.05 | 0.0533 | ❌ FAIL |
| `G4_quadrature_spread` | 0.01 | 1.7137e-07 | ✅ pass |

**显式回答「是否 <5% / <3%」**（生产分辨率 fine，逐 ka 列出）：

ka=0.2 误差 3.03%（<5% ✅，<3% ❌）；ka=0.5 误差 3.33%（<5% ✅，<3% ❌）；ka=1 误差 3.07%（<5% ✅，<3% ❌）；ka=2 误差 5.33%（<5% ❌，<3% ❌）

结论：<5% 在 ka ∈ {0.2, 0.5, 1} 成立；ka ∈ {2} **未达标（honest FAIL）**。最坏 5.33% → 「是否 <3%」= 否。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **网格分辨率阶梯**（4 档，见 ④）：误差随面板数单调下降，
   ka=1.0 由 coarse 的 11.47% 降到 fine 的 3.01%。
   加密的**关键变量是自由面附近的垂向分级（grading）**，不是单纯环数：
   (32,6,2.5)=416 三角形 → 11.4%，(32,12,3.0)=832 三角形 → 3.1%。
2. **Maruo 核方位求积**（n_theta=128…4096，在精确场上）：

| ka | n_theta | ∫\|K\|²(1-cosθ)dθ | kernel spread over the sweep |
|---|---|---|---|
| 0.2 | 128 | 3.006734 | 1.2127e-07 |
| 0.2 | 256 | 3.006734 | 1.2127e-07 |
| 0.2 | 512 | 3.006734 | 1.2127e-07 |
| 0.2 | 1024 | 3.006734 | 1.2127e-07 |
| 0.2 | 2048 | 3.006734 | 1.2127e-07 |
| 0.2 | 4096 | 3.006734 | 1.2127e-07 |
| 0.5 | 128 | 14.375180 | 1.1962e-07 |
| 0.5 | 256 | 14.375178 | 1.1962e-07 |
| 0.5 | 512 | 14.375178 | 1.1962e-07 |
| 0.5 | 1024 | 14.375178 | 1.1962e-07 |
| 0.5 | 2048 | 14.375178 | 1.1962e-07 |
| 0.5 | 4096 | 14.375178 | 1.1962e-07 |
| 1 | 128 | 16.711384 | 7.9752e-08 |
| 1 | 256 | 16.711383 | 7.9752e-08 |
| 1 | 512 | 16.711383 | 7.9752e-08 |
| 1 | 1024 | 16.711383 | 7.9752e-08 |
| 1 | 2048 | 16.711383 | 7.9752e-08 |
| 1 | 4096 | 16.711383 | 7.9752e-08 |
| 2 | 128 | 7.881289 | 1.7137e-07 |
| 2 | 256 | 7.881288 | 1.7137e-07 |
| 2 | 512 | 7.881288 | 1.7137e-07 |
| 2 | 1024 | 7.881288 | 1.7137e-07 |
| 2 | 2048 | 7.881288 | 1.7137e-07 |
| 2 | 4096 | 7.881288 | 1.7137e-07 |

   全域散布 ≤ 1.7137e-07，说明 far-field 误差来自 BEM 离散，不来自方位求积。
3. **BEM Kochin 方位采样审计**（`kochin_azimuth_rel_change`，加倍网格后的相对变化）：

| ka | panels | levels | grading | n_azimuth | Kochin azimuth rel. change | solver residual |
|---|---|---|---|---|---|---|
| 0.2 | 32 | 12 | 3 | 96 | 3.7093e-04 | 2.1745e-15 |
| 0.5 | 32 | 12 | 3 | 96 | 4.3081e-04 | 2.2594e-15 |
| 1 | 32 | 12 | 3 | 96 | 8.0723e-04 | 2.1929e-15 |
| 2 | 32 | 12 | 3 | 96 | 0.0021 | 2.5185e-15 |

4. **模态截断**：m_max=8 vs 16 的积分差 <1e-9（测试 `test_closed_form_is_converged_in_the_modal_truncation`）。
5. **长波律的适用带**：闭式 vs 5π²/16 的相对偏差按 (ka)² 衰减（ka=0.01 → 1.5e-5，0.05 → 1.2e-3，0.1 → 6.1e-3）。

## ⑥ 失败模式、工件与生态摩擦

**失败模式（实测，不是推测）**
- 网格不足时误差可达 11%（coarse），且**低估**漂移力：自由面附近垂向分级不足会系统性少算散射能量。
- 有限半径控制圆通量路线（`radius_scan`）对圆柱不可外推（实测散布 O(1)），本算例刻意使用无半径的约化核；
  这是仓内已登记的 NOT EVALUABLE（`drift_far_field3d.RADIUS_SCAN_NOTE`）。
- 深水假设：Green 函数深度因子按 tanh(kh)→1 处理；有限水深未在本算例实现。

**生态摩擦（如实记）**
1. **Kochin 幅值约定双轨**：`cylinder_drift.kochin_3d` 返回单位势幅值 K，直接喂
   `drift_far_field3d.maruo_far_field_drift` 会低估 (g/ω)²=g/k 倍。本单把该因子**实测钉住**：
   ka=0.2 处实测比值与 k/g 相差 <1e-5
   （`quadrature_sensitivity` 的 `unit_amplitude_force_ratio`）。
2. **`maruo_drift.maruo_mean_drift_force` 与 `drift_far_field3d.maruo_far_field_drift` 前因子不同**
   （rho ω²/4k vs rho k²/8π），对同一 K 表相差 2π/(ka) 倍——两模块各自自洽、混用即错。
   本单不改两模块（其测试各自钉住约定），只在算例侧选一条并记录。
3. **控制面 99 本地重计算不可用**：本机 load≈370/128 核，中等网格 case 被 OOM/kill；
   因此本算例的网格阶梯全部走集群作业（loong64 节点），CLI 与 `run_study` 均可在检出内直接跑。

**工件**
- `artifacts/maruo_drift_far_field_case/study.json`（全部数字，程序生成）
- `artifacts/maruo_drift_far_field_case/fig_bem_vs_closed_form.png`
- `artifacts/maruo_drift_far_field_case/fig_coefficient_vs_ka.png`
- `artifacts/maruo_drift_far_field_case/fig_convergence_and_quadrature.png`
- `src/tensorbem/benchmarks/maruo_drift_far_field_case.py`（算例驱动 + 闭式 ref）
- `tests/test_maruo_drift_far_field_case.py`
