# Case `CASE-BEM-KP505-GCI` — KP505 三密度 GCI 收敛（TensorBEM）

> 本文件由 `tools/build_case_bem_kp505_gci_report.py` 从 `benchmarks/verified/bem_kp505_gci/case.json` 渲染，正文数字**全部程序化注入**，无手抄。工件：`case.json` / `levels.csv` / `gci.csv` / `figures/*.png`。

> 复现命令：`PYTHONPATH=src python -m tensorbem.benchmarks.kp505_three_density_gci`（分片执行：`--phase ladder --advance <J>` / `--phase span` / `--phase sweep`，缓存于 `bem_kp505_gci/cache/`）。

## 1. 问题与参考出处

* 算例（`tensor_case_pool`）：`BEM-KP505-GCI` — KP505 三密度 GCI 收敛；仓 `TensorBEM`；`ref_source = Roache GCI (r = 2, three-density ladder)`；`tol_policy = <3%`，即**离散化不确定度 GCI ≤ 3%** 为验收线（本单容差判据）。
* 参考（GCI 程序本身，非测值）出处：
  * P. J. Roache (1994), Perspective: a method for uniform reporting of grid refinement studies, J. Fluids Eng. 116(3):405-413 (GCI, Fs = 1.25 for three grids).
  * P. J. Roache (1998), Verification and Validation in Computational Science and Engineering, Hermosa (asymptotic-range ratio GCI_medium/(r^p GCI_fine) -> 1).
  * ASME V&V 20-2009 / I. B. Celik et al. (2008), J. Fluids Eng. 130(7):078001 (procedure adopted by tensorbem.uq).
* 参考锚值（敞水系数）出处：Iftikhar, H. & Takinaci, A.C. (2023). Performance Prediction of KP505 Propeller Using Lifting Line Theory and RANS. X International Conference on Computational Methods in Marine Engineering (MARINE 2023).（DOI 10.23967/marine.2023.120），**Table 2, page 10，Table 2. Comparison of CFD and Experimental Results**；J = 0.70 行逐字转录：`0.7 0.173 0.356 0.193 0.342 10.31% 4.01%`；列头逐字：`CFD Experimental Error / J KT 10KQ KT 10KQ KT 10KQ`；provenance 等级 = `secondary tabulation: the paper prints the experimental columns of the KP505 open-water test, but page 10 does not cite the primary experimental report and the paper's reference list (p. 15) contains no KRISO/NMRI experimental entry. The numbers therefore enter this study through one tabulation step, which is declared rather than hidden.`（该表为二手转录，非 KRISO 原表，误差判定只在 GCI 上做验收，锚值只作报告项）。
* 物理问题：KP505 五叶定距桨敞水，在 J = 0.50, 0.70, 0.90 三点上做三密度（r = 2）面板收敛研究，回答「这套升力面/涡格 BEM 的敞水系数离散化不确定度是否 < 3%」。

## 2. 方法与实现

* 求解器（仓内既有能力，未另起炉灶）：`tensorbem.propeller.open_water_point`；几何模型：tensorbem.propeller.BladeGeometry (KP505 particulars from GEOMETRY_SOURCE; chord/camber/skew modelled)。
* 三级等比加密（r = 2，三条面板轴同时加密）：

| 级 | n_radial | n_chord | wake_steps | 每叶片束缚面元 |
|---|---|---|---|---|
| D1 | 8 | 2 | 24 | 16 |
| D2 | 16 | 4 | 48 | 64 |
| D3 | 32 | 8 | 96 | 256 |

* 持常量：`{"wake_model": "geometric", "wake_length": 1.0, "rev_per_second": 10.0, "thickness_ratio": 0.06, "radial_spacing": "cosine", "root_chord_offset": 0.12, "viscous": true, "core_ratio": 0.05}` — 同一物理模型三个网格，GCI 程序的前提。
* GCI（Roache）：`p = ln|(f_c-f_m)/(f_m-f_f)|/ln r`，`f_ext = f_f + (f_f-f_m)/(r^p-1)`，`GCI_f = Fs|(f_m-f_f)/f_f|/|r^p-1|`，Fs = 1.25（三级研究），Roache 渐近判据 `GCI_m/(r^p GCI_f) ≈ 1`。实现复用 `tensorbem.uq.grid_convergence_index`，并在本算例驱动内**独立写出闭式**对拍。
* 参考程序先验证再用（G7）：(a) 制造解 `f(h)=A+C h^pp`（pp = 1.7）反演阶数与极限；(b) `GCI/Fs` 必须等于制造解真实细网相对误差；(c) 仓内库函数 vs 驱动内闭式；(d) 振荡三元组必须被标记。
* 失败模式定位（预声明）：设计点做**平分方向纯加密扫描** n_radial ∈ [8, 12, 16, 20, 24, 32]（n_chord = 4、wake_steps = 48 冻结），判断非单调是单步偶然还是整个展向。

## 3. 参数表（无量纲数 + 模型输入）

| 量 | 值 | 说明 |
|---|---|---|
| J（进速系数） | 0.50, 0.70, 0.90 | 设计点 J = 0.70（验收点） |
| 叶数 Z | 5 | 几何来源 `GEOMETRY_SOURCE`（Table 1, 二手转录） |
| 直径 D | 0.250 m | 模型尺度，由转录 Table 1 `Diameter (m)` 行解析；求解器内部以 D = 1 无量纲化 |
| 毂径比 d_h/D | 0.180 | 模型输入（Table 1 Hub Ratio 行） |
| 剖面厚度比 t/c | 0.060 | 模型输入 |
| 弦长分布 | 余弦展向节点 + 模型弦长律 | 0.7R 处 c/D = 0.3825 → c = 95.6 mm（模型尺度） |
| 转速 n | 10 rev/s | 仅用于剖面 Reynolds 数 |
| 代表 Reynolds 数 Re(0.7R) | 4.840e+05 | V_res = 5.770 m/s, c = 95.6 mm, ν = 1.140e-06 m²/s |
| 涡核松弛 core_ratio | 0.050 | 正则化，三级恒定 |
| 尾涡模型 / 长度 | geometric / 1.00 转 | 三级恒定 |
| 加密比 r | 2.0 | 三密度阶梯 |
| 安全因子 Fs | 1.25 | Roache 三级默认 |
| 理论阶数 p_th | 2.0 | 面板中点积分（尖部 √ 载荷退化 → 1.5 为次阶） |

## 4. 对照表（ref vs got，逐项打勾/叉）

### 4.1 主判据：GCI vs 算例容差 3%（J = 0.70）

| 量 | ref（验收线） | got GCI_fine | 估计细网误差 GCI/Fs | 观测阶 p | 渐近比 | 单调 | <3% |
|---|---|---|---|---|---|---|---|
| KT | ≤ 3% | 10.074% | 8.059% | -3.967 | 0.9299 | ✓ | ✗ |
| 10KQ | ≤ 3% | 8.474% | 6.779% | -1.125 | 0.9646 | ✗ | ✗ |
| eta0 | ≤ 3% | 11.249% | 9.000% | -0.774 | 0.9640 | ✓ | ✗ |

### 4.2 三个进速系数全覆盖（GCI_fine）

| J | K_T GCI | 10K_Q GCI | eta_0 GCI | 全部 <3% |
|---|---|---|---|---|
| 0.50 | 9.908% | 12.905% | 14.272% | ✗ |
| 0.70 | 10.074% | 8.474% | 11.249% | ✗ |
| 0.90 | 14.539% | 0.101% | 27.002% | ✗ |

### 4.3 锚值对照（参考项，非验收项；J = 0.70，Table 2 page 10 逐字转录）

| 量 | ref 锚值 | got D3（细网） | 误差 | got 外推 | 误差 |
|---|---|---|---|---|---|
| KT | 0.1930 | 0.1582 | -18.01% | 0.1710 | -11.40% |
| 10KQ | 0.3420 | 0.2746 | -19.69% | 0.2933 | -14.25% |
| eta0 | 0.6287 | 0.6419 | 2.10% | 0.6997 | 11.29% |

### 4.4 参考程序自验（G7）与既有工件互校

| 检查 | 目标 | got | 相对误差 | 判 |
|---|---|---|---|---|
| G7a-observed-order | 1.700000 | 1.700000 | 1.18e-15 | ✓ |
| G7b-extrapolated-limit | 0.500000 | 0.500000 | 2.22e-16 | ✓ |
| G7c-gci-equals-true-error-over-fs | 0.002383 | 0.002383 | 5.37e-14 | ✓ |
| G7d-library-vs-closed-form:observed_order | 1.700000 | 1.700000 | 0.00e+00 | ✓ |
| G7d-library-vs-closed-form:extrapolated | 0.500000 | 0.500000 | 0.00e+00 | ✓ |
| G7d-library-vs-closed-form:gci_fine | 0.002979 | 0.002979 | 0.00e+00 | ✓ |
| G7d-library-vs-closed-form:gci_medium | 0.009626 | 0.009626 | 0.00e+00 | ✓ |
| G7e-oscillatory-flagged | True | True | n/a | ✓ |
| 互校 vs `benchmarks/verified/bem_kp505_openwater_w20/gci.json` (KT GCI) | 10.0739% | 10.0739% | 0.00e+00 pp | ✓ |
| 互校 vs `benchmarks/verified/bem_kp505_openwater_w20/gci.json` (10KQ GCI) | 8.4737% | 8.4737% | 0.00e+00 pp | ✓ |

## 5. 收敛性与参数敏感性

* **三级阶梯本身**（K_T，三进速）：
  * J = 0.50: D1=0.302720 → D2=0.301497 → D3=0.280558（D2→D3 变化 -6.95%）
  * J = 0.70: D1=0.170944 → D2=0.170180 → D3=0.158243（D2→D3 变化 -7.01%）
  * J = 0.90: D1=0.031264 → D2=0.030808 → D3=0.028089（D2→D3 变化 -8.83%）
* **观测阶 vs 理论阶**：理论 2.0（中点积分；尖部载荷退化给 1.5），实测 KT@J0.5: p=-4.10, 10KQ@J0.5: p=-0.51, eta0@J0.5: p=-0.67, KT@J0.7: p=-3.97, 10KQ@J0.7: p=-1.13, eta0@J0.7: p=-0.77, KT@J0.9: p=-2.58, 10KQ@J0.9: p=2.77, eta0@J0.9: p=-0.80。负阶意味着**增量随加密反而增大**（反收敛），不是幂律收敛。
* **渐近区间判据（含方法论注记）**：实测渐近比 0.912–0.995，全部落在 ±10% 带内，但观测阶几乎全为负 — 因为 `GCI_m/(r^p GCI_f)` 在单个三元组上**代数恒等于 `f_f/f_m`**（同一 Fs、同一三元组），三级研究里它几乎恒 ≈1，**并不检验 p**。真正检验渐近区需四级以上比较相邻三元组（见 `benchmarks/kp505_gci_ladder.py` 的 triplet 分歧 ~30×）。本算例因此把 G6 门控在「单调且 p>0」上；否则 p<0 的反收敛序列会被这条恒等式错判为渐近。
* **展向纯加密扫描**（J = 0.70，n_chord = 4、wake_steps = 48 冻结）：8→0.169749, 12→0.165844, 16→0.170180, 20→0.155237, 24→0.139722, 32→0.145370；跨扫描 K_T 相对极差 **19.32%**，单调性判 ✗。
* **展向三密度阶梯（隔离诊断）**：K_T 观测阶 -5.845、GCI 21.71%；10K_Q 观测阶 -2.353、GCI 20.61% — 非单调性在**展向**已单独复现。
* **进速系数敏感性**：同一阶梯在 J = 0.5 / 0.7 / 0.9 上 K_T 的 D3→(D2,D1) 落差分别为 -6.95%, -7.01%, -8.83%，量级一致 → 不是某一点的偶然，而是展向离散化的系统性行为。

## 6. 失败模式、判定与工件

* **判定：honest-FAIL**。`fail_stage = G3-monotone`；`failure_reason = predeclared criterion(s) not met: G3-monotone, G4-observed-order>=1, G5-case-tolerance-3pct, G6-asymptotic-admission; the GCI value itself is reported regardless (never smeared, never re-thresholded)`。
* 逐条判据：
  * ✓ G1-coverage
  * ✓ G2-physical
  * ✗ G3-monotone
  * ✗ G4-observed-order>=1
  * ✗ G5-case-tolerance-3pct
  * ✗ G6-asymptotic-admission
  * ✓ G7-reference-procedure-verified
  * ✓ G8-anchor-reported
* **失败机理（可复核）**：D1→D2 几乎不动（K_T -0.45%），而 D2→D3 跳 -7.01%；展向隔离扫描 8→32 之间 K_T 极差 19.32% 且非单调。即：**展向面元数不是本 BEM 收敛的有效旋钮**（尖部载荷 √(1-r) 退化 + 中点积分），GCI 因此既给不出幂律阶数，也给不出可信误差界。3% 判据在这套格式上**当前不可达**。
* **无法进一步验证的边界（如实声明）**：加第四级（r = 2 再加密）需 1024 束缚面元 × 192 尾涡步，直接稠密求解的内存/时间在本容器不可承载（D3 已 6.4 GB 峰值、约 50 s），故只能用 Roache 渐近比**判准入**而不能用再加密**验收敛**。
* 生态摩擦（friction）：
  * The platform knowledge search engine (:8021) was offline during the kick-off sweep (knowledge_base / oss_solutions checks returned 'engine_offline'), so the three-question pre-check could only be done against the in-repo code and the negative-knowledge API.
  * The lattice solve is direct-dense in (bound + wake) segments: D3 (256 bound panels, 96 wake steps) already needs ~6.4 GB peak RSS and 52 s, so a fourth density level (r = 2 again) is out of reach in this container - the asymptotic range therefore cannot be *tested* by refinement, only *admitted* by Roache's ratio.
* 工件：`benchmarks/verified/bem_kp505_gci/{case.json, levels.csv, gci.csv, figures/kp505_gci_ladder.png, figures/kp505_gci_indices.png}`；缓存 `cache/`；测试 `tests/test_case_bem_kp505_gci.py`。
* 环境：python 3.11.6 / numpy 2.4.6 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 / commit `05167765cb6b`；总墙钟 167 s。

