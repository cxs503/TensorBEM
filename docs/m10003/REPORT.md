# p315w1-n01 — 三维稳态 Neumann–Kelvin（Rankine 源 + 自由面 Green 函数）面板法首验

**任务**：BEM 三维势流面板法：Rankine source + Green function 兴波阻力与波形验证
**仓库**：TensorBEM · 分支 `agent/m10003` · 模块 `src/tensorbem/nk_panel3d.py`
**结论一句话**：核函数层面取得**可验证的真实进展**（自由面条件残差 7.5e-6、调和性 3e-7、辐射方向正确 3:1），但**兴波阻力与波形的定量验收全部 honest FAIL**——近场压力积分被离散噪声主导（Cw 相对 Michell 误差 −60×…+47× 且随网格变号），远场 Kochin 算子归一化标定失败（隐含前因子随 Fr 漂移 4.8×），Kelvin 尾迹张角未复现（55° vs 19.47°）。判据 (a)(b)(c)(d)(e) 均未通过，不许美化。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

**物理问题**：均匀来流 U 绕 Wigley III 数学船型的定常兴波。总势 $\Phi=-Ux+\varphi$，自由面线性化条件
$\varphi_{xx}+\kappa\varphi_z=0\ (z=0)$，$\kappa=g/U^2$；物面 $\partial\varphi/\partial n=U n_x$。

**几何**（几何先验证，见 ③）：半宽 $y=\tfrac{B}{2}\left(1-(2x/L)^2\right)\left(1-(z/T)^2\right)$，$x\in[-L/2,L/2]$，$z\in[-T,0]$。

| 参数 | 值 |
|---|---|
| L | 10.0 m（另含 L=100 m 等价缩比关系：一切量以 $\kappa^{-1}$ 归一） |
| B | 1.0 m（L/B = 10） |
| T | 0.625 m（B/T = 1.6） |
| Fr | 0.2 / 0.3 / 0.4 |
| ρ, g | 1025 kg/m³, 9.80665 m/s² |

**几何审计（先做）**：该 Wigley 型式的方形系数闭式为 $C_b=4LBT/9 \div LBT=4/9=0.4444$（面元湿表面积收敛到 7.4395 m²，误差 0.47%→0.03% 随网格 50→800 面元），**与战役 metadata 声称的 Cb=0.5625 不符**——0.5625 不属于此船型（见 `artifacts/nk_panel3d_study.json:geometry.Cb_note`）。按纪律以几何为准并登记差异，未按错误 Cb 反推任何量。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**核函数（本次核心新增）**。稳态深水 Green 函数 $G=G_R+G_W$，$G_R=\frac{1}{4\pi r}+\frac{1}{4\pi r_1}$（镜像），$G_W$ 由 $z=0$ 上条件 $W_{xx}+\kappa W_z=-R_{xx}$ 的双重 Fourier 变换导出，极坐标化后留数为 $k=\kappa\sec^2\theta$ 单极点：

$$G_W=\frac{1}{2\pi^2}\int_{-\pi/2}^{\pi/2}\!\!\Big[2\,\mathrm{Re}\big(e^{-\beta}\mathrm{Ei}(\beta)-\tfrac{1}{\beta}\big)+(2q-1)\pi\,\mathrm{Im}\,e^{-\beta}\Big]d\theta,\qquad
\beta=\frac{\kappa(a-iD)}{\cos^2\theta},\ a=-(z+\zeta),\ D=(x-\xi)\cos\theta+(y-\eta)\sin\theta .$$

推导中三处**必须踩对**的地方（都曾把残差推到 O(1)）：
1. 极坐标 Jacobian 多一个 $k$ → 被积函数是 $u/(1-u)$ 而不是 $1/(1-u)$，留下 $\kappa\sec^2\theta$ 权重与 $-1/\beta$ 项；
2. $\mathrm{PV}\int_0^\infty u e^{-\beta u}/(1-u)\,du=e^{-\beta}\mathrm{Ei}(\beta)-1/\beta$——已用 `scipy.integrate.quad` 对 $\mathrm{Im}\beta\gtrless0$ 五个复数 β 逐点核对（差值 0，脚本见报告正文说明）；`expi` 的分支恰为实轴主值，辐射项必须**显式**加；
3. $e^{-\beta}\mathrm{Ei}(\beta)$ 在 $|\beta|>60$ 用渐近级数（16 项，误差 ~1e-17），否则 `expi` 溢出成 NaN。

**求解**：单层（indirect）源汇法，未知量 $\sigma$ 只布在湿表面（半体 + 镜像 ⇒ 精确对称，未知数=面元数）；跳变项 $-\tfrac12\sigma$，核梯度全部解析（$\partial\beta/\partial(x,y,z)$ 每 θ 节点为常数），近场用 3×3 子胞细化，自面元位势用 8×8 Gauss。

**工具链**：numpy/scipy（`expi`/`leggauss`）、matplotlib 出图；`scripts/nk_panel3d_study.py` 程序化写 JSON（零手抄），`scripts/nk_panel3d_figures.py` 只从 JSON 取数出图；`pytest` 38 项。

## ③ 计算结果（全部由脚本写入 `artifacts/nk_panel3d_study.json`）

**核函数验证（通过）**

| 检查 | 值 | 判定 |
|---|---|---|
| 自由面条件 $\lvert G_{xx}+\kappa G_z\rvert$ 相对残差（halfplane，64 θ 点） | **7.53e-06** | ✅ |
| 同上（PV only） | 2.68e-06 | ✅ |
| 调和性 $\nabla^2 G_W$（h=1e-5） | 2.9e-07（量级 $G_W\sim10^{-2}$） | ✅ |
| 辐射方向：源后/源前振幅比（halfplane, sign=−1） | **3.0**（后 0.114 / 前 0.038） | ✅ 尾迹在船后 |
| 镜像选择（sign=+1） | 0.32（波在船前） | ✅ 作为反证 |
| θ 方向积分收敛 | $n_\theta$=64 时相对误差 1e-6，96 起 1e-9 | ✅ |
| Kelvin 尾迹张角 | **55.3°**（vs 19.47°） | ❌ **未复现** |
| 远场 Kochin 算子（薄船分布理应回到 Michell） | 隐含前因子 0.00390 / 0.00153 / 0.00082（Fr=0.2/0.3/0.4，漂移 4.8×） | ❌ **归一化不成立** |

**兴波阻力（诚实结果）**：$C_w=R_w/(\tfrac12\rho U^2 LT)$

| Fr | Michell $C_w$ | 50 面元 | 128 面元 | 288 面元 | 与 Michell 误差 |
|---|---|---|---|---|---|
| 0.2 | 2.05e-04 | +9.81e-03 | −4.76e-03 | −0.0e+00 | +4688% / −2424% / −100% |
| 0.3 | 4.35e-04 | −2.53e-02 | −4.31e-03 | −5.56e-03 | −5921% / −1091% / −1380% |
| 0.4 | 5.24e-04 | −5.05e-04 | −8.40e-03 | −2.45e-03 | −196% / −1704% / −567% |

符号随机、随网格不收敛 ⇒ 近场一阶压力积分 $\;-\rho U\oint\varphi_x n_x\,dS\;$ 对前后对称船体是"大数相消"，被离散噪声淹没（与本仓早前 m3100 审计在中心面上的结论同源）。**判据 (a) ≤5% 失败，(d) 收敛阶 ≥1.5 无法给出（无单调收敛序列）。**

**波形**：Fr=0.3、128 面元，三条纵剖线（y=0, B/2, B）与同核薄船参考的最大偏差 = **603%**（参考峰值归一）。**判据 (b) ≤8% 失败。**

**能量守恒**：需要一条可信的远场算子才能谈"辐射能量=兴波阻力×船速"；远场归一化已判不成立，故**判据 (e) 不可评估**（不是"通过了"，是不成立）。

**SUBOFF (c)**：本仓无 SUBOFF 几何模块（`grep -ri suboff src/` 为空），本次未实现；且裸 SUBOFF 在深潜状态兴波阻力相对总阻力为小量，"Cw 与实验 ≤10%"需先从总阻力扣摩擦（ITTC-57），判据本身病态——**NOT_SUPPORTED**，如实登记而非硬凑。

**测试 (f)**：`pytest tests/test_nk_panel3d.py` **38 passed**（≥20 ✅）。覆盖率工具在本环境不可用（`coverage` 缺 C tracer，`--cov` 直接收集错误），覆盖度**未验证**，不宣称达标。

## ④ 结果图

| 图 | 文件 | 说明 |
|---|---|---|
| 核函数验证 | `docs/m10003/fig_nk_kernel_checks.png` | (a) θ 积分收敛（64 点后 1e-6）(b) 辐射方向比值，标注自由面残差 7.5e-6 |
| 点源尾迹场 | `docs/m10003/fig_kelvin_wake.png` | 自由面波高云图 + 经典 caustic ±19.47° 虚线；可见**本核尾迹明显宽于 caustic**（已知缺陷） |
| Wigley 波形 | `docs/m10003/fig_wigley_wave_cuts.png` | 三条纵剖线：NK 面元解 vs 同核薄船参考（偏差 603%） |
| 阻力 vs 网格 | `docs/m10003/fig_wigley_resistance.png` | symlog 纵轴：面元解在 0 附近乱跳，虚线为 Michell——honest FAIL 的可视证据 |

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **极坐标 Jacobian 与 $1/\beta$ 项**：漏掉 $k$ 导致 $W_{xx}+\kappa W_z=-2R_{xx}$，自由面残差 O(1)；补正后残差降到 1e-6。这是本次最大的技术坑，已写入模块 docstring。
2. **`expi` 分支即实轴主值**：辐射项不能靠分支差异"免费"获得，必须显式加；「符号」不能靠猜——用"波只在船后"这一物理判据数值定（sign=−1 时比值 3.0）。
3. **整体因子 2**：解析推导留下一个来自源/镜像在 $z=0$ 双重化的常数因子，本实现以 `_WAVE_SCALE=0.5` 显式登记，**由自由面条件验证确定**（不是对阻力数据拟合）。若后续重推变换，应把这个因子吸收回推导。
4. **近场一阶压力积分不可用于本船型**：与 m3100 审计一致；正确路线应是"压力积分 + 水线 $\tfrac{\rho g}{2}\zeta^2$ 项"或可靠远场能量法。本模块目前两者都没有。
5. **远场算子未标定**：$\lvert H\rvert^2\sec^3\theta$ 形式在本实现里随 Fr 漂移 4.8×，说明 $\sec^3\theta$ 权重与 $\kappa$ 幂次的归一化还不够；在它被"薄船极限回到 Michell（<1%）"验证之前，任何由它给出的 Cw 都不能入库。
6. **尾迹张角未复现（55° vs 19.47°）**：说明 $\theta$ 积分的驻相结构尚未正确——这与远场归一化失败是同一个根因（残差项/方向权重），修复它应能同时解决 (b)(e)。
7. **环境**：本仓安装态 `tensorbem` 指向另一工作树，运行测试需 `PYTHONPATH=$PWD/src`；单次面元解 128 面元约 10 s、288 面元约 40 s（Python 逐对核），加密到工程可用量级需上集群或做核缓存。
8. 适用范围：本模块目前只适合做**核函数级**的验证与教学/原型引用，**不可**用于给出实船/型号的兴波阻力数值。

## ⑥ 误差判定（逐项 vs 基准，显式回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 自由面条件残差（核） | 0（解析恒等式） | 7.5e-6 | ✅ 是 |
| 调和性 | 0 | 2.9e-7 | ✅ 是 |
| 辐射方向 | 尾迹在船后（定性） | 比值 3.0 vs 0.32（反证） | ✅ 方向正确 |
| θ 积分 64 点 | 128 点值 | 1e-6 | ✅ 是 |
| 面元湿表面积（800 面元） | 解析积分 7.4395 m² | 0.029% | ✅ 是 |
| $C_w$ (Fr=0.2/0.3/0.4) | Michell | −60×…+47×，变号 | ❌ **否** |
| 波形（3 剖线） | 同核薄船参考 | 603% of peak | ❌ **否** |
| Kelvin 张角 | 19.47° | 55.3°（+184%） | ❌ **否** |
| 远场算子自洽 | 薄船⇒Michell（<1%） | 隐含前因子漂移 4.8× | ❌ **否** |
| SUBOFF $C_w$ | DARPA 实验 | 未实现 | ⛔ NOT_SUPPORTED |

**总判定：honest FAIL。** 交付物中可入库的是**核函数与面板框架本身**（有强验证证据），阻力/波形验收判据一律不通过，不允许用核函数验证的"绿"掩护 (a)(b)(d)(e) 的"红"。

## 附：产物与复现

```
src/tensorbem/nk_panel3d.py           # 核函数 + 面元求解器 + 验证/后处理 API
tests/test_nk_panel3d.py              # 38 项 pytest（含 1 项"已知缺陷回归门"）
scripts/nk_panel3d_study.py           # 全部数字 → artifacts/nk_panel3d_study.json
scripts/nk_panel3d_figures.py         # 4 张图（只从 JSON 取数）
artifacts/nk_panel3d_study.json       # 审计链主工件
artifacts/kelvin_wake_field.npz       # 尾迹场数据
docs/m10003/fig_*.png                 # 报告图
```
复现：`PYTHONPATH=src python scripts/nk_panel3d_study.py --quick`（约 6 min）、
`PYTHONPATH=src python -m pytest tests/test_nk_panel3d.py -q`（6.6 s，38 passed）。