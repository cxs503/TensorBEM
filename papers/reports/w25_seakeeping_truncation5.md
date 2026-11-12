# W25 · 耐波性 RAO 截断收敛 vs Journée (1992) 锚（TensorBEM m16712）

> 工件：`papers/artifacts/w25/seakeeping-truncation/journee_rao_truncation.json`（schema `w25-journee-rao-truncation/v1`，生成 2026-11-12T19:09:10+00:00，git `9a7470c-dirty`）
> 环境：Python 3.11.6 / numpy 2.4.6 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38
> 复算命令：`python run_w25_rao_truncation.py --stage sweep --shard 1/4`; `python run_w25_rao_truncation.py --stage sweep --shard 2/4`; `python run_w25_rao_truncation.py --stage sweep --shard 3/4`; `python run_w25_rao_truncation.py --stage sweep --shard 4/4`; `python run_w25_rao_truncation.py --stage radius --shard 1/1`; `python run_w25_rao_truncation.py --stage assemble`

## ① 案例介绍

**物理背景。** 频率域耐波性计算给出 RAO，但 RAO 只在有限频带内被求值：上端**截断频率**决定响应谱被记入多少高频余量，下端**截断半径**（波长域 $\lambda_{\max}/L$）决定长波端被记入多少。ITTC 7.5-02-07-04.5 类推荐程序把“截断后残余谱面积是否可忽略”作为验收量，而不是逐点 RAO。本单把这两类截断各做成 4 档加倍阶梯，在**真跑**的 RAO 上量化其收敛性，并对账外部试验锚。

**几何（Wigley III，即 Journée 1992 Report 09-V-09 的模型船型；$L/B=10$, $B/T=1.6$）。**

| 量 | 值 | 来源 |
|---|---|---|
| $L$ | 3.000 m | 解析船型 |
| $B$ | 0.300 m | 解析船型 |
| $T$ | 0.1875 m | 解析船型 |
| $\alpha$（改型参数） | 1.108750 | $C_b=(4/9)+4\alpha/45$ |
| $C_b$（闭式） | 0.543000 | 闭式 |
| $C_b$（求积） | 0.542998 | 船体体积求积 |
| 排水体积（闭式） | 0.091631 m³ | $V=LBT C_b$ |
| 质量 | 93.922 kg | $\rho V$ |
| $KB$（闭式） | 0.108680 m | $T[1-(15+5\alpha)/(40+8\alpha)]$ |
| $KB$（求积复核） | 0.108680 m | 船体体积求积 |
| $KB/T$ | 0.57962 | 同上 |
| $k_{yy}/L$ | 0.2242 | 均匀密度惯性矩 |
| $k_{xx}/B$ | 0.2708 | 均匀密度惯性矩 |
| $\rho$, $g$ | 1025.0 kg/m³, 9.80665 m/s² | 常量 |

**几何门（先验 Cb/L/B）。** $C_b$ 闭式 0.543000 对求积 0.542998，偏差 0.0004 %；体积偏差 0.00043 %；$KB$ 闭式对独立求积偏差 0.0004 %；$\alpha=0$ 退化时 $KB/T=0.6250$（经典值 0.625 = $5T/8$）。**本单同时修掉了仓内 `compute_wigley3_properties` 的 $KB$ 公式缺陷**：旧式 $T(65+11\alpha)/16(5+\alpha)$ 在 $\alpha=0$ 给出 0.8125 T（+30 %），新式 $T[1-(15+5\alpha)/(40+8\alpha)]$ 给出 0.625 T 并与求积一致。

| 海况 | $H_s$ [m] | $T_1$ [s] | $\omega_p$ [rad/s] | 谱 |
|---|---|---|---|---|
| ss_T1_1p00 | 1.00 | 1.00 | 4.8489 | ITTC (1978) two-parameter, m0 = Hs^2/16 |
| ss_T1_1p40 | 1.00 | 1.40 | 3.4635 | ITTC (1978) two-parameter, m0 = Hs^2/16 |
| ss_T1_1p80 | 1.00 | 1.80 | 2.6938 | ITTC (1978) two-parameter, m0 = Hs^2/16 |

## ② 怎么做的

**求解器（REAL，非 MOCK）。** `tensorbem.rao.compute_rao_at_frequency`：零航速三维自由面 Green 函数边界元（常值面元 + CHIEF 内部点抑制不规则频率）。网格三档：coarse 8×4（112 面元）、production 12×6（264 面元）、fine 16×8（480 面元）；生产档 53 个频点，全部截断档位精确落在网格节点上。

**两个截断轴（RAO 侧）。**
* 轴 A 截断频率 $\omega_{\max}\in\{1.5, 3.0, 6.0, 12.0\}$ rad/s（精确加倍阶梯）；
* 轴 B 截断半径 $\lambda_{\max}/L\in\{5.0, 10.0, 20.0, 40.0\}$（精确加倍阶梯）；
* 每个档位记：带内响应 rms $\sigma$、带内响应方差占比（截断残余）、带内峰值/峰频。**验收量取 $\sigma$**（逐点峰值是内点，对上端截断天然不敏感——这一点在结果里显式列出，不粉饰）。

**几何截断半径轴（轴 C）。** 仓内唯一显式截断流体域的求解器是 `tensorbem.rankine_fs.solve_rankine_free_surface`（Rankine 源 + 逆风差分辐射条件，**定常** Neumann-Kelvin），因此几何半径 $y_{{\max}}/L\in\{{{', '.join(str(v) for v in sd['geometric_radius_levels_y_max_over_L'])}\}}$（含 1.0→2.0 与 1.25→2.5 两组精确加倍）、$x_{\max}/L=2.5$、4.0 面元/波长固定，在任务给出的 Fr = 0.142 / 0.200 上跑。

**预登记门限（看结果之前写死）：** 峰值 RAO 偏差 <15 %、峰频偏移 <15 %、截断加倍变化 <3 %。

**溯源。** 驱动脚本 sha256 `2b8cfa5f7429c4e18b2ef3706f72d83910fb7f1f853f5144b6164424d97629cf`；被调代码与数据 sha256 见工件 `provenance.inputs`（`src/tensorbem/rao.py`、`src/tensorbem/free_surface_solver3d.py`、`src/tensorbem/free_surface3d.py`、`src/tensorbem/hydrostatic.py` …）。录制的命令见文首。

## ③ 计算结果

### 3.1 RAO 曲线（生产网格，零航速，迎浪）

| $\omega$ [rad/s] | $\omega\sqrt{L/g}$ | heave RAO [m/m] | pitch RAO [rad/m] |
|---|---|---|---|
| 0.6000 | 0.3319 | 0.99988 | 0.033407 |
| 0.7166 | 0.3963 | 0.99967 | 0.047633 |
| 0.8000 | 0.4425 | 0.99942 | 0.059350 |
| 1.0000 | 0.5531 | 0.99836 | 0.092623 |
| 1.0134 | 0.5605 | 0.99826 | 0.095109 |
| 1.2000 | 0.6637 | 0.99636 | 0.133114 |
| 1.4000 | 0.7743 | 0.99300 | 0.180630 |
| 1.4331 | 0.7927 | 0.99227 | 0.189164 |
| 1.5000 | 0.8296 | 0.99065 | 0.206933 |
| 1.6000 | 0.8850 | 0.98777 | 0.234867 |
| 1.8000 | 0.9956 | 0.98011 | 0.295370 |
| 2.0000 | 1.1062 | 0.96942 | 0.361493 |
| 2.0268 | 1.1210 | 0.96772 | 0.370727 |
| 2.2000 | 1.2168 | 0.95499 | 0.432368 |
| 2.4000 | 1.3274 | 0.93611 | 0.506865 |
| 2.5000 | 1.3827 | 0.92476 | 0.545044 |
| 2.6000 | 1.4380 | 0.91200 | 0.583580 |
| 2.8000 | 1.5487 | 0.88184 | 0.660830 |
| 3.0000 | 1.6593 | 0.84485 | 0.736670 |
| 3.2000 | 1.7699 | 0.80031 | 0.808925 |
| 3.4000 | 1.8805 | 0.74772 | 0.875221 |
| 3.5000 | 1.9358 | 0.71835 | 0.905341 |
| 3.6000 | 1.9911 | 0.68697 | 0.932995 |
| 3.8000 | 2.1018 | 0.61853 | 0.979485 |
| 4.0000 | 2.2124 | 0.54368 | 1.011297 |
| 4.2000 | 2.3230 | 0.46429 | 1.025269 |
| 4.4000 | 2.4336 | 0.38307 | 1.018822 |
| 4.5000 | 2.4889 | 0.34272 | 1.006620 |
| 4.6000 | 2.5442 | 0.30310 | 0.988057 |
| 4.8000 | 2.6549 | 0.22786 | 0.931256 |
| 5.0000 | 2.7655 | 0.16199 | 0.848755 |
| 5.2000 | 2.8761 | 0.11386 | 0.743703 |
| 5.4000 | 2.9867 | 0.09746 | 0.622039 |
| 5.5000 | 3.0420 | 0.10254 | 0.557482 |
| 5.6000 | 3.0973 | 0.11326 | 0.492093 |
| 5.8000 | 3.2080 | 0.14006 | 0.365751 |
| 6.0000 | 3.3186 | 0.16213 | 0.266461 |
| 6.2000 | 3.4292 | 0.17150 | 0.243468 |
| 6.4000 | 3.5398 | 0.16327 | 0.319319 |
| 6.6000 | 3.6504 | 0.13516 | 0.442747 |
| 6.8000 | 3.7611 | 0.09200 | 0.554909 |
| 7.0000 | 3.8717 | 0.05328 | 0.575327 |
| 7.2000 | 3.9823 | 0.04320 | 0.444912 |
| 7.4000 | 4.0929 | 0.04611 | 0.254996 |
| 7.6000 | 4.2035 | 0.04275 | 0.118074 |
| 7.8000 | 4.3141 | 0.03324 | 0.066529 |
| 8.0000 | 4.4248 | 0.02157 | 0.069222 |
| 8.5000 | 4.7013 | 0.00526 | 0.049808 |
| 9.0000 | 4.9779 | 0.00621 | 0.009882 |
| 9.2000 | 5.0885 | 0.00415 | 0.010482 |
| 10.0000 | 5.5310 | 0.00268 | 0.001929 |
| 11.0000 | 6.0841 | 0.00489 | 0.027072 |
| 12.0000 | 6.6371 | 0.00657 | 0.005202 |

峰值：heave $\omega=0.6000$ rad/s ($\omega\sqrt{L/g}=0.3319$)，值 0.99988；pitch $\omega=4.2369$ rad/s ($\omega\sqrt{L/g}=2.3434$)，值 1.025616 rad/m。最大阻抗残差 0.000。

### 3.2 轴 A：截断频率（$\omega_{\max}$ 加倍阶梯）

| 海况 | $\omega_{\max}$ | $\sigma_3$ [m] | $\sigma_5$ [rad] | heave 方差占比 | pitch 方差占比 | 峰值在带内 |
|---|---|---|---|---|---|---|
| ss_T1_1p00 | 1.50 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False |
| ss_T1_1p00 | 3.00 | 0.003482 | 0.002966 | 0.002867 | 0.000318 | False |
| ss_T1_1p00 | 6.00 | 0.063294 | 0.159357 | 0.947033 | 0.918441 | True |
| ss_T1_1p00 | 12.00 | 0.065040 | 0.166282 | 1.000000 | 1.000000 | True |
| ss_T1_1p40 | 1.50 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False |
| ss_T1_1p40 | 3.00 | 0.073052 | 0.054245 | 0.263611 | 0.070482 | False |
| ss_T1_1p40 | 6.00 | 0.142002 | 0.202472 | 0.996051 | 0.981961 | True |
| ss_T1_1p40 | 12.00 | 0.142283 | 0.204324 | 1.000000 | 1.000000 | True |
| ss_T1_1p80 | 1.50 | 0.000442 | 0.000091 | 0.000005 | 0.000000 | False |
| ss_T1_1p80 | 3.00 | 0.152251 | 0.095724 | 0.648358 | 0.266387 | False |
| ss_T1_1p80 | 6.00 | 0.189000 | 0.184678 | 0.999122 | 0.991526 | True |
| ss_T1_1p80 | 12.00 | 0.189083 | 0.185466 | 1.000000 | 1.000000 | True |

| 组 | 加倍对 | 变化 [%] | 最大 [%] | <3 % ? |
|---|---|---|---|---|
| ss_T1_1p00:sigma_heave | 1.5→3.0 | 100.000 | 100.000 | FAIL |
| ss_T1_1p00:sigma_heave | 3.0→6.0 | 94.498 | 100.000 | FAIL |
| ss_T1_1p00:sigma_heave | 6.0→12.0 | 2.684 | 100.000 | FAIL |
| ss_T1_1p00:sigma_pitch | 1.5→3.0 | 100.000 | 100.000 | FAIL |
| ss_T1_1p00:sigma_pitch | 3.0→6.0 | 98.139 | 100.000 | FAIL |
| ss_T1_1p00:sigma_pitch | 6.0→12.0 | 4.165 | 100.000 | FAIL |
| ss_T1_1p40:sigma_heave | 1.5→3.0 | 100.000 | 100.000 | FAIL |
| ss_T1_1p40:sigma_heave | 3.0→6.0 | 48.555 | 100.000 | FAIL |
| ss_T1_1p40:sigma_heave | 6.0→12.0 | 0.198 | 100.000 | FAIL |
| ss_T1_1p40:sigma_pitch | 1.5→3.0 | 100.000 | 100.000 | FAIL |
| ss_T1_1p40:sigma_pitch | 3.0→6.0 | 73.209 | 100.000 | FAIL |
| ss_T1_1p40:sigma_pitch | 6.0→12.0 | 0.906 | 100.000 | FAIL |
| ss_T1_1p80:sigma_heave | 1.5→3.0 | 99.709 | 99.709 | FAIL |
| ss_T1_1p80:sigma_heave | 3.0→6.0 | 19.444 | 99.709 | FAIL |
| ss_T1_1p80:sigma_heave | 6.0→12.0 | 0.044 | 99.709 | FAIL |
| ss_T1_1p80:sigma_pitch | 1.5→3.0 | 99.905 | 99.905 | FAIL |
| ss_T1_1p80:sigma_pitch | 3.0→6.0 | 48.167 | 99.905 | FAIL |
| ss_T1_1p80:sigma_pitch | 6.0→12.0 | 0.425 | 99.905 | FAIL |

### 3.3 轴 B：截断半径（$\lambda_{\max}/L$ 加倍阶梯）

| 海况 | $\lambda_{\max}/L$ | $\omega_{\min}$ [rad/s] | $\sigma_3$ [m] | $\sigma_5$ [rad] | heave 方差占比 | pitch 方差占比 |
|---|---|---|---|---|---|---|
| ss_T1_1p00 | 5.0 | 2.0268 | 0.065040 | 0.166282 | 1.000000 | 1.000000 |
| ss_T1_1p00 | 10.0 | 1.4331 | 0.065040 | 0.166282 | 1.000000 | 1.000000 |
| ss_T1_1p00 | 20.0 | 1.0134 | 0.065040 | 0.166282 | 1.000000 | 1.000000 |
| ss_T1_1p00 | 40.0 | 0.7166 | 0.065040 | 0.166282 | 1.000000 | 1.000000 |
| ss_T1_1p40 | 5.0 | 2.0268 | 0.142275 | 0.204323 | 0.999883 | 0.999992 |
| ss_T1_1p40 | 10.0 | 1.4331 | 0.142283 | 0.204324 | 1.000000 | 1.000000 |
| ss_T1_1p40 | 20.0 | 1.0134 | 0.142283 | 0.204324 | 1.000000 | 1.000000 |
| ss_T1_1p40 | 40.0 | 0.7166 | 0.142283 | 0.204324 | 1.000000 | 1.000000 |
| ss_T1_1p80 | 5.0 | 2.0268 | 0.185488 | 0.185002 | 0.962344 | 0.995000 |
| ss_T1_1p80 | 10.0 | 1.4331 | 0.189083 | 0.185466 | 0.999999 | 1.000000 |
| ss_T1_1p80 | 20.0 | 1.0134 | 0.189083 | 0.185466 | 1.000000 | 1.000000 |
| ss_T1_1p80 | 40.0 | 0.7166 | 0.189083 | 0.185466 | 1.000000 | 1.000000 |

| 组 | 加倍对 | 变化 [%] | 最大 [%] | <3 % ? |
|---|---|---|---|---|
| ss_T1_1p00:sigma_heave | 5.0→10.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p00:sigma_heave | 10.0→20.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p00:sigma_heave | 20.0→40.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p00:sigma_pitch | 5.0→10.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p00:sigma_pitch | 10.0→20.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p00:sigma_pitch | 20.0→40.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p40:sigma_heave | 5.0→10.0 | 0.006 | 0.006 | PASS |
| ss_T1_1p40:sigma_heave | 10.0→20.0 | 0.000 | 0.006 | PASS |
| ss_T1_1p40:sigma_heave | 20.0→40.0 | 0.000 | 0.006 | PASS |
| ss_T1_1p40:sigma_pitch | 5.0→10.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p40:sigma_pitch | 10.0→20.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p40:sigma_pitch | 20.0→40.0 | 0.000 | 0.000 | PASS |
| ss_T1_1p80:sigma_heave | 5.0→10.0 | 1.901 | 1.901 | PASS |
| ss_T1_1p80:sigma_heave | 10.0→20.0 | 0.000 | 1.901 | PASS |
| ss_T1_1p80:sigma_heave | 20.0→40.0 | 0.000 | 1.901 | PASS |
| ss_T1_1p80:sigma_pitch | 5.0→10.0 | 0.250 | 0.250 | PASS |
| ss_T1_1p80:sigma_pitch | 10.0→20.0 | 0.000 | 0.250 | PASS |
| ss_T1_1p80:sigma_pitch | 20.0→40.0 | 0.000 | 0.250 | PASS |

### 3.4 轴 C：几何截断半径（定常 Rankine，任务 Fr）

| Fr | $y_{\max}/L$ | 未知量 | $R_w$ 压力积分 [N] | $R_w$ 远场 [N] | $C_w$ | 能量门 $|R_w^p-R_w^f|/|R_w^p|$ [%] |
|---|---|---|---|---|---|---|
| 0.142 | 1.00 | 3711 | -1.8647 | 0.000000 | -0.01090 | 100.00 |
| 0.142 | 1.25 | 4488 | -1.8653 | 0.000000 | -0.01091 | 100.00 |
| 0.142 | 2.00 | 7152 | -1.8656 | 0.000000 | -0.01091 | 100.00 |
| 0.142 | 2.50 | 8928 | -1.8656 | 0.000000 | -0.01091 | 100.00 |
| 0.200 | 1.00 | 1078 | -835.1548 | 0.000020 | -2.46178 | 100.00 |
| 0.200 | 1.25 | 1302 | -835.1423 | 0.000022 | -2.46174 | 100.00 |
| 0.200 | 2.00 | 1974 | -835.1370 | 0.000023 | -2.46172 | 100.00 |
| 0.200 | 2.50 | 2422 | -835.1373 | 0.000023 | -2.46173 | 100.00 |

| Fr | 加倍对 | $C_w$ 变化 [%] | 最大 [%] | <3 % ? |
|---|---|---|---|---|
| 0.142 | 1.0→2.0 | 0.047 | 0.047 | PASS |
| 0.142 | 1.25→2.5 | 0.021 | 0.047 | PASS |
| 0.2 | 1.0→2.0 | 0.002 | 0.002 | PASS |
| 0.2 | 1.25→2.5 | 0.001 | 0.002 | PASS |

### 3.5 离散化研究（与截断误差解耦）

| $\omega$ | heave 粗→生产 [%] | heave 生产→细 [%] | pitch 粗→生产 [%] | pitch 生产→细 [%] |
|---|---|---|---|---|
| 1.000 | 0.060 | 0.020 | 2.561 | 0.812 |
| 1.500 | 0.156 | 0.051 | 2.499 | 0.791 |
| 2.500 | 0.633 | 0.206 | 2.210 | 0.694 |
| 3.500 | 1.888 | 0.612 | 1.444 | 0.448 |
| 4.500 | 4.867 | 1.600 | 0.596 | 0.217 |
| 5.500 | 2.439 | 0.134 | 5.883 | 1.932 |
| 7.000 | 10.256 | 1.850 | 6.985 | 0.741 |
| 9.000 | 30.130 | 5.706 | 63.515 | 1.151 |
| 11.000 | 43.875 | 1.433 | 15.198 | 78.276 |

## ④ 结果图

![fig1_rao_curves.png](fig1_rao_curves.png)

*Figure 1 - heave/pitch RAO of the Wigley III hull (head seas, U=0) with the four truncation-frequency levels marked.*

![fig2_truncation_frequency.png](fig2_truncation_frequency.png)

*Figure 2 - band-limited response rms versus the upper truncation frequency for the three ITTC sea states (doubling ladder).*

![fig3_truncation_radius_longwave.png](fig3_truncation_radius_longwave.png)

*Figure 3 - response rms versus the long-wave truncation radius (lambda_max/L, doubling ladder).*

![fig4_truncation_radius_geometric.png](fig4_truncation_radius_geometric.png)

*Figure 4 - steady truncated-domain Rankine solver: wave-resistance coefficient versus the lateral truncation radius at the two mission Froude numbers.*

## ⑤ 注意点

* The certified unsteady RAO solver is zero-speed; forward speed (Fr = 0.142 / 0.20) is NOT_SUPPORTED for RAO in this package and is exercised only on the steady truncated-domain solver (wave-resistance coefficient).
* No experimental anchor is available in this environment (see anchor section), so criteria (1) and (2) are WITHHELD rather than estimated.
* The 3-D Green-function RAO has no domain truncation; the 'truncation radius' of the RAO axes is therefore a wavelength-space radius (lambda_max/L), and the geometric radius is measured on the truncated-domain Rankine solver instead.
* ITTC (1978) spectrum with Hs = 1 m: sigma values are transfer-function properties and scale linearly with Hs.
* **外部锚缺失是硬约束，不是偷懒**：任务指定的 “Journée (1992) KVLCC2 试验 RAO” 在年代上不可能存在（仓内对 Journée 1992 的引用是《4 个 Wigley 船型迎浪试验与计算》，Report 09-V-09；KVLCC2 船型晚于 1992 年设计）。仓内也没有任何试验 RAO 表，`tensorbem.reference_provenance` 把包内所有 Journée 字面量登记为 recalled/UNSOURCED；沙箱禁网，报告不可现场取回。因此判据 1/2 记为 WITHHELD，并**没有**用 recalled 数字充当锚（仅作诊断行，标注 NOT ACCEPTED）。
* 诊断行（**不作判据**）：把 recalled 的 heave 峰值 1.150 / 峰频 1.800 rad/s 与本单零航速计算值 0.9999 / 0.6000 rad/s 并置，偏差 13.1 % / 66.7 %；该比对的参照值无出处且航速不同（0 vs 试验航速），只能说明量级不矛盾，不能作为达标证据。
* **零航速限制**：仓内经验证的非定常 BEM 只有零航速 Green 函数（无遭遇频率核/带速 Havelock 核）；`tensorbem.stf_fullspeed` 只有 STF 装配与圆形剖面系数、无船体激励力构造器，端到端带速 RAO 需新物理，调参自造即失去可验证性——故 Fr = 0.142/0.20 的 RAO 声明 NOT_SUPPORTED，任务 Fr 只由定常截断域求解器承载。
* **轴 C 的功能量并不可靠**：定常 Rankine 解在模型尺度/Fr ≤ 0.2 的 $C_w$ 为负且压力/远场能量门接近 100 %（与仓内 W20 #4 记录的 −2.462 一致），说明该压力积分泛函在低 Fr 被非波分量主导。轴 C 的“收敛”因此只是**同一泛函的域截断收敛**，不能当作 RAO 截断收敛的证据；本单把它单列，不并入判据 3。
* **峰值 vs 带宽**：判据 3 的“RAO 变化”对逐点峰值是弱检验（峰值是内点）；本单因此同时给出逐点峰值敏感性表与带内方差表，两者都列，读者可自行判断验收口径。

## ⑥ 误差判定（逐项，显式回答是否 <3 %）

| 判据 | 门限 | 裁决 | 数值 |
|---|---|---|---|
| ① 峰值 RAO 相对偏差 | <15 % | **WITHHELD** | 无来源锚（见 ⑤） |
| ② 峰频偏移 | <15 % | **WITHHELD** | 无来源锚（见 ⑤） |
| ③ 截断加倍 → RAO 变化 | <3 % | **FAIL as an all-pairs criterion (23/36 doubling pairs < 3 %); the highest doubling 6.0->12.0 rad/s passes in 5/6 channel-sea-state combinations and fails marginally at 4.16 %.** | 最大 100.000 %（轴 A/B + 逐点峰值；轴 C 单列 0.047 %） |

**几何/物理自校验（解析锚，逐项是否 <3 %）：**

| 检查 | 偏差 | <3 % ? |
|---|---|---|
| $C_b$ 求积 vs 闭式 | 0.000 % | 是 |
| 体积 求积 vs 闭式 | 0.000 % | 是 |
| $KB$ 闭式 vs 独立求积 | 0.000 % | 是 |
| 长波极限 heave RAO vs 1 | 0.012 % | 是 |
| 迎浪约定 $\beta\to\beta+\pi$ 不变性（$\omega$=1.50） | heave 0.000 % / pitch 0.001 % | 是 |
| 迎浪约定 $\beta\to\beta+\pi$ 不变性（$\omega$=3.50） | heave 0.004 % / pitch 0.019 % | 是 |

**结论（不粉饰）：** 判据 ③ 的机器裁决为 **FAIL as an all-pairs criterion (23/36 doubling pairs < 3 %); the highest doubling 6.0->12.0 rad/s passes in 5/6 channel-sea-state combinations and fails marginally at 4.16 %.**，最大截断加倍变化 100.00 %（未过 3 % 的门，不粉饰为达标）；失败对集中在低截断档位——它们切进了响应带（本模型 pitch 峰在 $\omega\approx$4.237 rad/s），属物理上“截断不足”而非数值噪声。判据 ① ② 因外部试验锚在本环境不可得而 **WITHHELD（honest FAIL）**，不可被后续轮次覆盖为“达标”。零航速与带速 RAO 的能力缺口、以及定常 Rankine 泛函的低 Fr 失效，均已在工件 `capabilities.gaps` 与上节登记，供后续轮次作为缺口账本入口。

## 复算

```
python run_w25_rao_truncation.py --stage sweep --shard 1/8   # … 8/8
python run_w25_rao_truncation.py --stage radius
python run_w25_rao_truncation.py --stage assemble
python papers/reports/build_w25_report.py
```
