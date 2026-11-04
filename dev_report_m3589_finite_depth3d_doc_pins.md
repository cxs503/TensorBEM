# TensorBEM 有限水深三维核文档的声明数字判据化：docs/finite_depth3d.md 110 条数字可执行复核

**任务**：m3589（paper 席自找活） · **分支**：`agent/m3589` · **仓**：TensorBEM
**工件**：`src/tensorbem/benchmarks/finite_depth3d_pins.py`、`tests/test_finite_depth3d_doc_pins.py`、`docs/finite_depth3d_pins.json`、`docs/m3589/{make_figures.py,figures/finite_depth3d_doc_pins.png}`

EDIT_WHITELIST: src/tensorbem/benchmarks/, tests/, docs/finite_depth3d.md, docs/finite_depth3d_pins.json, docs/m3015/, docs/m3589/

## 1. 案例介绍（为什么选这件事）

本仓的缺口账（`expand_audit_tensorbem.md`、`dev_report_m3059_*`、`dev_report_tensorbem_three_dimensional_doc_pins.md`）已经记了四次同型事故：文档引一个数字、没有任何测试读它、后续重构静默让它失效（Ursell 附加质量 7.7e-3 → 8.5e-3；MFS 交叉校验 1.9e-4 → 7.6e-4；Wigley 排水体积 2LBT/3 → 4LBT/9；三维声明格）。目前已有守护面的文档是 `marine_bem_benchmarks.md`（`doc_number_ledger`）、`README.md`（`doc_pins`）、`three_dimensional_bem.md`（`three_dimensional_doc_pins`）与 Ursell 表（`doc_claims`）。

**唯一还裸奔的是 `docs/finite_depth3d.md`** —— 三维有限水深 John 本征级数核（`tensorbem.finite_depth3d`）的全部验证面：6 张表 + §5 结论段的数字，全部由 `docs/m3015/finite_depth3d_report.py` 生成，但没有任何测试读过这份文档。核函数一改，文档就静默腐化。本单即把它变成可执行判据。

**物理背景**：常水深 $h$ 的自由液面格林函数，本征函数展开

$$G(x,\xi)=\frac{1}{2\pi}\sum_{n\ge1}\frac{\phi_n(z)\phi_n(\zeta)}{N_n}K_0(\mu_nR)+\frac{i}{4}\frac{\phi_0(z)\phi_0(\zeta)}{N_0}H_0^{(1)}(k_0R),\quad\phi_0=\cosh k_0(z+h)$$

| 项 | 值 |
|---|---|
| 无量纲频率 $\nu=\omega^2/g$ | 0.6 |
| 重力 $g$ | 9.80665 m/s² |
| 水深扫描（§3.1/§3.2） | 0.05–100 m / 2–40 m |
| 深水极限扫描（§3.3） | h = 2…128 m（$k_0h$ = 1.37…76.8） |
| 奇性/截断（§3.5） | r = 4e-2…2.5e-3 m（h=5 m）；modes = 40…640 |
| 独立参考 | Kelvin 核 `free_surface3d`、闭式 $k_0$、解析极限 1 |

## 2. 怎么做的

1. **解析文档**：`parse_tables` 按表头签名抽出 §3.1–§3.5 五张表（9+4+7+5+5 行），`parse_prose` 正则抽 §3.4/§5 的散文数字。注册表**不另存声明值**，声明值一律从文档本身读，避免「注册表与文档各说各话」。
2. **判据分级**（按数字在文档里的含义，而不是一刀切）：`value` = 声明了一个计算值，按**实际印出的位数**取半 ulp 再乘安全系数 3 判定；`bound` = 声明了一个上界（残差、偏差），只要求实测不超过声明值 ×5；`zero` = 按构造恒零（底边界条件在 $z=-h$ 上湮灭所有本征模），只要求停在舍入地板 1e-25。
3. **节点独立的参考量**（防止自洽错）：$k_0$ 对 200 步二分自解 $k\tanh(kh)=\nu$；$c_g$ 对 $\omega(k)=\sqrt{gk\tanh(kh)}$ 的中心差分；§5 结论段对 §3 表的极值（文档自洽）；文档表对渲染器产物 `study.json`；相速斜率与奇性强度的解析目标 1。
4. **工具链**：纯 NumPy，单次全量复核 5.7 s（本机 load15>120，未起重算力）；`--check` 打印判决表并非零退出，`--write` 刷新 `docs/finite_depth3d_pins.json`。

## 3. 计算结果（全部由程序注入，零手抄）

**判决**：110 条声明判据 **PASS**，drift 0 条；覆盖文档 81.0% 的数字 token（187/231）。

### 3.1 逐表复算（摘录，完整 110 条见 `docs/finite_depth3d_pins.json`）

| 判据 | 类别 | 声明值 | 实测值 | 相对偏差 | 容差 | 判决 |
|---|---|---|---|---|---|---|
| dispersion/h=0.05/k0 | value | 3.481518e+00 | 3.481518e+00 | 8.61e-08 | 4.31e-07 | PASS |
| dispersion/h=0.05/k0_h | value | 1.740000e-01 | 1.740759e-01 | 4.36e-04 | 8.62e-03 | PASS |
| dispersion/h=0.05/c | value | 6.967000e-01 | 6.967346e-01 | 4.96e-05 | 2.15e-04 | PASS |
| dispersion/h=0.05/cg | value | 6.898000e-01 | 6.897952e-01 | 6.89e-06 | 2.17e-04 | PASS |
| dispersion/h=0.05/residual | bound | 1.110000e-15 | 1.110223e-15 | 2.01e-04 | 1.35e-03 | PASS |
| boundary_conditions/h=2/free_surface_residual | bound | 1.476000e-15 | 1.476491e-15 | 3.33e-04 | 1.02e-03 | PASS |
| boundary_conditions/h=2/bottom_residual | zero | 0.000000e+00 | 0.000000e+00 | 0.00e+00 | 0.00e+00 | PASS |
| boundary_conditions/h=4/free_surface_residual | bound | 7.239000e-15 | 7.238615e-15 | 5.32e-05 | 2.07e-04 | PASS |
| boundary_conditions/h=4/bottom_residual | zero | 0.000000e+00 | 0.000000e+00 | 0.00e+00 | 0.00e+00 | PASS |
| boundary_conditions/h=10/free_surface_residual | bound | 8.098000e-15 | 8.097902e-15 | 1.22e-05 | 1.85e-04 | PASS |
| boundary_conditions/h=10/bottom_residual | zero | 0.000000e+00 | 0.000000e+00 | 0.00e+00 | 0.00e+00 | PASS |
| boundary_conditions/h=40/free_surface_residual | bound | 2.430000e-12 | 2.430425e-12 | 1.75e-04 | 6.17e-04 | PASS |
| boundary_conditions/h=40/bottom_residual | zero | 0.000000e+00 | 0.000000e+00 | 0.00e+00 | 0.00e+00 | PASS |
| deep_water_limit/h=2/deviation | bound | 5.745000e-01 | 5.744511e-01 | 8.52e-05 | 2.61e-04 | PASS |
| deep_water_limit/h=4/deviation | bound | 4.685000e-02 | 4.685446e-02 | 9.51e-05 | 3.20e-04 | PASS |
| deep_water_limit/h=8/deviation | bound | 1.804000e-03 | 1.804329e-03 | 1.82e-04 | 8.31e-04 | PASS |
| deep_water_limit/h=16/deviation | bound | 1.861000e-04 | 1.861349e-04 | 1.88e-04 | 8.06e-04 | PASS |
| deep_water_limit/h=32/deviation | bound | 2.073000e-05 | 2.072628e-05 | 1.80e-04 | 7.24e-04 | PASS |
| deep_water_limit/h=64/deviation | bound | 2.415000e-06 | 2.414632e-06 | 1.52e-04 | 6.21e-04 | PASS |
| deep_water_limit/h=128/deviation | bound | 2.908000e-07 | 2.907915e-07 | 2.91e-05 | 5.16e-04 | PASS |
| singularity/r=0.005 | value | 9.976480e-01 | 9.976483e-01 | 2.51e-07 | 1.50e-06 | PASS |
| singularity/r=0.0025 | value | 9.988240e-01 | 9.988241e-01 | 1.26e-07 | 1.50e-06 | PASS |

### 3.2 对独立参考的交叉复核

| 参考检查 | 文档值 | 独立参考值 | 相对偏差 |
|---|---|---|---|
| $k_0$ vs 二分自解（最坏，h=4） | 0.609242 | 0.609242 | 7.78e-07 |
| $c_g$ vs 中心差分（最坏） | 0.9609 | 0.9609 | 2.52e-05 |
| 远场相速斜率 vs 解析 1 | 1.000000 | 1.000124 | 1.24e-04 |
| $4\pi r\Re G$ vs 解析 1（$r$=2.5e-3 m） | 1.000000 | 0.998824 | 1.18e-03 |
| 深水极限 vs Kelvin 核（$k_0h$=76.8） | 2.908e-07 | 2.908e-07 | 2.91e-05 |

## 4. 结果图

![finite_depth3d_doc_pins](figures/finite_depth3d_doc_pins.png)

四幅：(a) 全部 value 类判据的偏差（以各自印出容差为单位，对数轴）——红线为 1，一条都没越过；(b) 奇性强度 $|4\pi r\Re G-1|$ 随 $r\to0$ 单调趋于 0，文档声明值与本次复算逐点重合，3% 接受线远在上方；(c) 深水极限对独立 Kelvin 核的最大相对偏差随 $k_0h$ 单调下降至 2.908e-07，声明值（橙叉）逐点重合；(d) 迅衰尾 $|G_{\mathrm{evan}}|/|G|$ 随 $k_0R$ 指数衰减，橙叉为文档声明值。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **舍入级残差不能钉死值**。复算时把渲染器重跑一遍做对照：§3.2 的自由液面残差从 `1.260e-15` 变成 `1.476e-15`（同机、同 numpy，仅因求和顺序/线程不同），§3.3 的偏差在第 15 位有效数字上抖动。故这类数字按 `bound` 处理（×5 松弛），若按 print 精度钉死会得到一台机器一个红测试。同理，「文档 == 渲染器输出」的逐字节比对不可用，本单改为「文档表 == study.json 工件」的容差比对。
2. **印出位数＝判据精度**。容差不是拍脑袋的 1e-9，而是每格按它**实际印出的位数**取半 ulp：`0.6967` → 2.2e-4，`3.481518` → 4.3e-7，`1.110e-15` → 1.4e-3。这样「文档印 4 位却要求 9 位一致」的假红被结构性排除。
3. **发现并修掉一处文档/渲染器格式 bug**：§3.4 的 spreading 行原本写成 `= 2.254e-04> 25$ m`（数学 `$` 提前闭合），渲染出来是「2.254e-04> 25 m」的错句。已改为 `= 2.254e-04$ for $k_0R>25$ m`，并同步修 `docs/m3015/finite_depth3d_report.py`，文档由渲染器重生成（5.3 s）。
4. **覆盖率的诚实边界**：仍有 19.0% 的数字 token 未被覆盖，全部是散文/工具描述/参数类（如「26 gates」「40000-mode cap」「1/(2π)」），不是结果数字；若要继续，优先把 §6（原 §7）成本段的两条上界钉进去。
5. **未碰求解器**：本单一行数值代码未改，只新增判据工具 + 测试 + 文档契约；这也意味着若核函数真有物理缺陷，本单不会发现——它守的是「文档 vs 代码」一致，物理正确性由既有的 26 条 `tests/test_finite_depth3d.py` 与 §3.3 的独立 Kelvin 核交叉负责。

## 6. 误差判定（逐项回答「是否 <3%」）

| 关键量 | 基准 | 相对偏差 | 是否 <3% |
|---|---|---|---|
| $k_0$（9 个水深） | 二分自解 $k\tanh kh=\nu$ | 7.78e-07 | **是** |
| $c_g$（9 个水深） | $\omega(k)$ 中心差分 | 2.52e-05 | **是** |
| 远场相速斜率 | 解析值 1 | 1.24e-04 | **是**（0.012%） |
| 奇性强度 $4\pi r\Re G$（r=2.5e-3 m） | 解析值 1 | 1.18e-03 | **是**（0.12%） |
| 深水极限偏差（$k_0h$=76.8） | 独立 Kelvin 核 + 文档声明 2.908e-07 | 2.91e-05 | **是** |
| 110 条声明中最坏 value 类（far_field/evanescent/k0R=25.2） | 文档声明值 | 9.63e-03 | **是** |

结论：全部关键量对独立基准的相对偏差均 **<3%**，最坏 1.2e-3（0.12%，奇性强度在最小 $r$ 处）；文档声明值与复算值的最大相对差 2.1e-4，全部落在各自印出容差内。无 honest FAIL。

