# TensorBEM：Wigley III 船型定义缺陷裁定与修复（α 轴：垂向 → 纵向，m16935）

## ① 案例介绍
TensorBEM 有**两个** Wigley III 生成器：`tensorbem.wigley3_hull`（多体/耐波路径）与
`tensorbem.wigley3_freq`（频域验证路径，仓库全部公开 Wigley III 数字的来源）。
两者把丰满度参数 α 分别加在 $\zeta=z/T$ 与 $\xi=2x/L$ 上，**块系数都等于 0.543** ——
所以大家会去核的那个数看不出差别；但它们是两条**不同的船**。

判定依据（先几何后数值）：仓库自己的船型定义（`docs/wigley3_freq_verification_report.md` §1
"fuller ends than the standard parabolic Wigley (Wigley I)"，
$y=(B/2)(1-\xi^2)(1-\zeta^2)(1+c\xi^2)$）+ Journée (1992) 尺寸表
（L=3.0 m, B=0.30 m, T=0.1875 m, Cb=0.543）。若剖面保持标准抛物线（Cm=2/3），
则 $C_{wp}=C_b/C_m=0.8145$；把 α 加在垂向只在**改变剖面**的情况下保持 Cwp=2/3，
与"更丰满的**端部**"自相矛盾。故 `wigley3_hull` 是缺陷方。

## ② 怎么做的
1. 修 `wigley3_half_breadth`：α 作用于 $\xi^2$（纵向），与文档定义一致。
2. 同步修 `compute_wigley3_properties` 的闭式解（该函数原来的 A_wp、I_xx、I_yy、KB
   是按旧几何推的）：$A_{wp}=BL(2/3+2\alpha/15)$、
   $I_{xx}=(B^3L/24)(32/35+32\alpha/105+32\alpha^2/385+32\alpha^3/3003)$、
   $I_{yy}=(BL^3/8)(4/15+4\alpha/35)$、$KB=5T/8$（纵向丰满度下剖面形状不变，故 KB 与 α 无关）。
3. 新增几何门 `tests/test_wigley3_hull_alpha_axis.py`（6 项）：两生成器逐点恒等、
   Cwp=2/3+2α/15 且距退役配方 >15%（回退哨兵）、Cb/Cm/Cwp 三角一致、
   五个闭式解 vs 同一曲面的 Gauss–Legendre 积分、网格体积、以及两生成器的**辐射收敛交叉验证**。
4. 相关钉桩更新：`test_multibody_frequency_dependence_3d` 的 A_wp 由 2BL/3 改为 BL(2/3+2α/15)；
   `tests/test_wigley3_recalled_ref.py` 的几何门改为"冻结记录保留退役缺陷 + 现行生成器必须已修复"，
   并把冻结件里 `wigley3_hull` 的单元标注为**修复前历史**（capytaine 冻结网格签名同理）。

## ③ 计算结果（全部由 `docs/wigley3_hull_alpha_axis/study.json` 注入）
| 量 | 值 |
|---|---|
| 水面线系数 Cwp（修复后，积分） | 0.8144996 |
| Cwp（解析 $2/3+2\alpha/15$，α=1.1088） | 0.8145000 |
| Cwp（退役配方，α 加垂向） | 0.6666665 |
| Cb / Cm | 0.5430 / 0.6667 |
| 水线面面积 $A_{wp}$ [m²] | 0.73305 |
| KB/T | 0.6250 |
| 两生成器半宽最大差 $\max|\Delta y|$ | 5.551e-17 |

ω=1.0 rad/s 升沉（$\rho\nabla$=93.9220 kg）：

| 生成器 | 面元 | $A_{33}/(\rho\nabla)$ | $B_{33}/(\rho\nabla\omega)$ | 网格体积误差 | 墙钟 |
|---|---|---|---|---|---|
| wigley3_hull | 180 | 2.009101 | 0.264762 | -3.58% | 2.0 s |
| wigley3_freq | 200 | 2.037964 | 0.265991 | -2.08% | 3.09 s |

**跨生成器分裂**：A33 +1.44%，B33 +0.46%
（修复前冻结件 level-20 单元：1.6113 vs 1.4557
→ -9.66%；m16892 叙述里的 1.51 vs 2.07 = 36% 是它另一组单元）。

## ④ 结果图
左：修复后的静水线 $y(x,0)/(B/2)$ vs 退役配方（Cwp 0.8145 vs 0.6667）。
右：同一 ω 下两生成器的 $A_{33}/(\rho\nabla)$，分裂已塌缩到 +1.44%。
图 sha256 = `0d1cbf373c20766514e0819af571c285598caadbea0873568360667efa75017e`。

## ⑤ 注意点（局限）
- **未重跑 capytaine 冻结件**：`benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json`
  是在退役几何上取的，现按"修复前历史记录"保留并由测试显式豁免；重取需重跑
  `scripts/wigley3_recalled_ref_capytaine.py`（本单未做）。
- 760 面元 `wigley3_freq` 求解超出单次 118 s 墙钟（hull 760 已单独测得
  $A_{33}/(\rho\nabla)$=2.0654），故本单的交叉验证取 ~200 面元；结论方向不受影响，
  但**未宣称** 760 面元级别的跨生成器收敛。
- 只跑了 `wigley or multibody or seakeeping or capability or doc_claim` 子集（exit 0）+ 新增门，
  未跑全量套件。
- 线性势流、无粘、深水；α 的物理解释按仓库文档与 Journée 尺寸表，**未**取得 Journée (1992)
  原始图表做逐点核对（离线不可得）——这是本单裁定的最大外部依赖。

## ⑥ 误差判定（是否 <3%）
| 关键量 | 基准 | 误差 | <3% |
|---|---|---|---|
| 半宽场 $y(x,z)$，hull vs freq | 同一曲面（文档定义） | 5.55e-17 | 是（机器精度） |
| Cwp | 解析 2/3+2α/15 | 4.32e-07 | 是 |
| Cb·(Cm=2/3) vs Cwp | Cb/Cm 恒等 | <1e-12 | 是 |
| A33 跨生成器 | 彼此 | 1.44% | 是 |
| B33 跨生成器 | 彼此 | 0.46% | 是 |
| A33 vs 独立码 capytaine | m16892 冻结（同一曲面 = freq 生成器） | −0.9%（继承自 m16892） | 是 |

复算：`PYTHONPATH=src python docs/wigley3_hull_alpha_axis/study.py`；
`PYTHONPATH=src pytest tests/test_wigley3_hull_alpha_axis.py -q` → 6 passed。
