# 船舶与海洋工程边界元基准问题综述与实现

## 摘要

边界元法（BEM）以边界积分方程代替流场区域离散，常用于船舶耐波性、海洋结构物水动力、波浪辐射与绕射及水声问题。可信的数值验证应从有解析解的问题开始，再逐步扩展到自由液面解析基准、三维几何和试验数据。本文梳理了适用于船舶与海洋工程的主要基准问题，并在 TensorBEM 中实现了不含自由液面的一层基准：二维圆柱与椭圆柱横向平移附加质量、三维球体与椭球沿主轴平移附加质量，以及度 1–3 的内部固体谐波。二维问题的解析附加质量为每单位跨长 \(\rho\pi a^2\)；以半径 \(a=1\)、水密度 \(\rho=1025\) 和 16–128 个直线面元计算，基于求解所得法向通量的附加质量相对误差由 \(2.62\times10^{-2}\) 降至 \(4.02\times10^{-4}\)。三维问题使用常数三角形面元和球体外域 Dirichlet 公式，解析附加质量为 \(\tfrac12\rho V\)；20–1280 个面元时相对误差由 \(2.13\times10^{-1}\) 降至 \(4.47\times10^{-3}\)，观测阶接近 2。椭圆柱基准首次使用真正的二维外域积分方程（128 面元时相对误差 \(7.5\times10^{-4}\)，观测阶 2.00）；椭球基准在 1280 面元时三轴相对误差为 \(4.2\text{–}4.9\times10^{-3}\)，观测阶 1.99–2.01。在此基础上，本文进一步完成四层船海关键水动力基准的验证与实现：
1. **外域刚体 Neumann 附加质量张量与转动附加惯性矩（Lamb 1932）**：不预设解析表面势，直接由刚体物面无穿透条件 \(\partial\phi_j/\partial n_{\text{body}}=n_j\) 建立第二类边界积分方程 \((\tfrac12\mathbf I-\mathbf H)\boldsymbol\phi=\mathbf S(-\mathbf n)\)，一次求解二维椭圆截面的 \(3\times3\) 张量（含横摇附加转动惯量 \(A_{66}=\tfrac18\rho\pi(a^2-b^2)^2\)）与三维三轴椭球的 \(6\times6\) 张量（含横摇、纵摇、首摇三轴转动附加惯量），条件数保持 \(O(1)\)，全部模态均呈 \(O(h^2)\) 二阶收敛，并精确满足平行轴定理；
2. **Ursell 潜没圆柱辐射（自由液面，深水）**：以 Ursell 多极子闭式解为参考（复现文献阻尼零点至机器精度），\(ka=0.6\)、\(a/f=0.5\) 时 128 面元附加质量与阻尼误差分别为 \(7.7\times10^{-3}\) 与 \(1.3\times10^{-2}\)，并以近场/远场能量恒等式验证辐射条件；
3. **Dean–Ursell–Ogilvie 潜没柱体规则波绕射与 Haskind 互易关系（自由液面，深水）**：验证潜没圆柱在任意频率下的零反射定理（数值 \(|R|\approx 2.2\times10^{-15}\)）、纯相移全透射（\(|T|=1\)）、波浪激励力圆偏振特性（\(F_z=-\mathrm i F_x\)，误差 \(1.1\times10^{-14}\)），以及无需绕射势的 Haskind–Newman 辐射–绕射互易关系（相对差 \(2.2\times10^{-5}\)），并以潜没椭圆柱对比验证非对称截面的有限反射与能量守恒 \(|R|^2+|T|^2=1\)；
4. **MacCamy–Fuchs 贯底直立柱体波浪绕射与 CHIEF 不规则频率消除**：利用等截面直立柱在有限水深 \(h\) 下的精确垂向分离变量，实现向量化任意阶 Bessel/Hankel 函数与二维外域 Helmholtz 求解器 `tensorbem.helmholtz`，验证波浪水平合力 \(F_x\)、海底倾覆力矩 \(M_y\) 与柱面总势的 \(O(h^2)\) 收敛（128 面元相对误差 \(1.32\times10^{-4}\)）、\(ka\to 0\) 时等效惯性力系数 \(C_M\to 2.0\) 的 Morison 极限过渡，以及 Schenck (1968) 内部零场点（CHIEF）对内域 Dirichlet 特征值处不规则频率病态尖峰的消除（在 \(J_1(ka)=0\) 偶极共振点将条件数从 665 压回 3.34，波浪力误差从 10.35% 降至 0.28%）；
5. **三维深水自由液面 Kelvin/Havelock Green 函数与穿水/潜没浮体 6-DOF 辐射–绕射求解器（Hulme 1982；Wang 1986 / Linton 1991）**：纯 NumPy 实现全向量化 Struve 函数 \(\mathbf H_0,\mathbf H_1\) 与四分区高精度三维兴波核 `tensorbem.free_surface3d`（各分区接缝连续性达 \(10^{-11}\text{–}10^{-13}\)），通过直接源 + \(z=0\) 正镜像三角面元的解析立体角与极坐标/Duffy 奇异分离消除水线近奇异性，建立通用三维 6-DOF 辐射与绕射面板求解器 `tensorbem.free_surface_solver3d` 及开口穿水船体网格 `WettedHullMesh`。在 **Hulme (1982) 漂浮半球**基准（\(Ka=1.0\)，256 面元）中，纵荡/垂荡附加质量、兴波阻尼与复波浪激励力观测阶均为 **1.98–2.06**（相对误差 \(0.57\%\text{–}2.03\%\)），并在内域水线共振点 \(Ka=2.58\) 处证明无需离散水线盖板（lid panels）、仅用 5 个三维内部 CHIEF 零场约束即可将条件数从 145.66 压回 4.24，消除未正则化方程出现的非物理负阻尼（\(B_{33}'=-0.104\to +0.0687\)）；在 **Wang/Linton 三维潜没球体**基准（\(f/a=1.5,\ Ka=0.8\)，20–320 面元）中，与零数值求积的闭式阶乘/指数积分多极子参考解对比观测阶达 **1.93–2.06**，三维 Kochin 远场阻尼守恒与 Haskind 互易误差均低于 \(4\times 10^{-3}\)。

有限水深三维 John 特征级数 Green 函数（如 Garrett 截断圆柱）、有航速兴波阻力（Wigley 船型）以及升力体 Kutta 条件仍不在当前代码范围内，详见 §2 与 §5。

**关键词：** 边界元法；船舶水动力；附加质量；三维自由液面 Green 函数；不规则频率；数值验证

## 1. 引言

船舶与海洋工程中的经典势流模型通常将流体速度表示为速度势的梯度，并在无旋、不可压条件下求解 Laplace 方程。BEM 将该区域方程转换为物体表面、自由液面或远场边界上的积分方程；因此，边界条件、法向方向、自由液面处理以及几何离散误差均会直接影响结果。附加质量、阻尼、波浪激励力和运动响应等工程量又通常由边界势函数积分获得。

基准测试应区分**验证**与**确认**：解析解或独立高精度解用于检验离散方程、积分和求解器；与水池试验的对比则检验物理模型与工程预测能力，受黏性、尺度效应和试验不确定度影响。Newman 对海洋势流水动力问题及辐射、绕射理论作了系统论述；Ursell 和 John 分别给出自由液面圆柱运动问题及浮体简谐运动的经典解析理论 [1–3]。

TensorBEM 目前提供二维直线常数面元求解器和三维三角形常数面元求解器。二维求解器采用中点配点与解析面元积分；三维求解器采用三角形质心配点、解析双层立体角积分和 Duffy 单层求积。本文按"参考量必为闭式解"的次序实现：先用圆柱附加质量边界迹基准固定二维内核，再用椭圆柱引入真正的二维外域算子；三维侧从球体推广到椭球（引入 Lamb 形状因子），并用内部固体谐波补上高阶角向误差。随后依次完成二维深水辐射层（§4.6）、外域刚体 Neumann 转动/平动附加质量张量（§4.7）、二维深水潜没体规则波绕射与 Haskind 互易基准（§4.8）、MacCamy–Fuchs 贯底直立柱绕射与 CHIEF 不规则频率消除（§4.9），并在 v0.5.0 中补齐**三维深水自由液面 Kelvin Green 函数、穿水浮体 6-DOF 辐射/绕射面板法及 Hulme 漂浮半球与潜没球体三维基准（§4.10–§4.11）**。有限水深三维核与有航速 Rankine 源留作后续里程碑。

## 2. 船海 BEM 基准问题全景与覆盖矩阵

综合国际主流船海水动力边界元程序（WAMIT、Capytaine、NEMOH、HAMS、OrcaWave、AQWA）的标准验证集与经典水波解析理论，可将船海领域 BEM 基准划分为以下十二类。下表给出各类基准的物理检验目标及与当前 TensorBEM（v0.5.0）的对应关系：

| 类别 | 典型基准及参考量 | 主要检验内容 | 当前 TensorBEM 覆盖状态 |
|---|---|---|---|
| 1. 制造解 / 固体谐波 | 圆域调和场、度 1–3 三维内部固体谐波 [8] | 核函数符号、立体角与 Duffy 奇异积分、高阶角向误差 | **已覆盖**（`unit_disk`、`interior_harmonics`） |
| 2. 无界流刚体平动（Dirichlet 迹） | 二维圆柱/椭圆柱、三维球体/椭球沿主轴平动附加质量 [1, 4] | 第一类算子 \(\mathbf S\)、外域跳项、Lamb 椭球形状因子、切向速度恢复 | **已覆盖**（`cylinder_added_mass`、`ellipse_added_mass`、`sphere_added_mass`、`ellipsoid_added_mass`、`sphere_velocity`） |
| 3. 无界流刚体 3-DOF / 6-DOF Neumann 张量 | 椭圆截面横摇转动惯量 \(A_{66}\)、三轴椭球 \(6\times6\) 附加质量与转动惯量张量 [1, 4] | **真实物面无穿透条件** \(\partial_n\phi=n_j\)、第二类算子 \((\tfrac12\mathbf I-\mathbf H)\) 的 \(O(1)\) 条件数、张量对称性与平行轴定理 | **本次新增覆盖**（`rigid_body_added_mass`，§4.7） |
| 4. 二维自由液面强迫辐射 | Ursell 潜没圆柱横荡/垂荡附加质量 \(A(\omega)\)、兴波阻尼 \(B(\omega)\) 及阻尼零点 [2] | 线性自由液面条件、远场出射波条件、近/远场阻尼能量恒等式 | **已覆盖**（`ursell_cylinder`，§4.6；本次修复奇数 Gauss 点自面元裂项与顶点深度校验） |
| 5. 二维自由液面波浪绕射与互易性 | Dean (1948) / Ursell (1950) / Ogilvie (1963) 潜没圆柱零反射、透射相移与圆偏振激励力；潜没椭圆柱反射 [9–11] | 入射 Airy 波耦合、Kochin 反射/透射积分、波能守恒 \(|R|^2+|T|^2=1\)、**Haskind–Newman 辐射–绕射互易关系** | **本次新增覆盖**（`dean_diffraction`，§4.8） |
| 6. 贯底直立柱体三维波浪绕射 | MacCamy & Fuchs (1954) 有限水深/深水贯底圆柱波浪力 \(F_x\)、倾覆力矩 \(M_y\) 与 \(C_M(ka)\) 曲线 [5] | 垂向 \(\cosh k(z+h)\) 分离变量、二维外域 Helmholtz 算子、长波 Morison 极限 \(C_M\to 2\) | **本次新增覆盖**（`maccamy_fuchs`，§4.9） |
| 7. 水线穿出体不规则频率消除 | 直立柱/穿水半球在内域水线特征值处的条件数爆炸与负阻尼失真 [6, 12] | 边界积分算子秩亏机理、**二维/三维 Schenck CHIEF 内部零场增广**（免水线盖板面元） | **已覆盖**（`maccamy_fuchs` + `floating_hemisphere` CHIEF 验证，§4.9–§4.10） |
| 8. 三维自由液面轴对称/球体解析基准 | **Hulme (1982) 漂浮半球** [6]、**Wang / Linton (1991) 深水潜没球** [13]；Garrett (1971) 有限水深截断圆柱 [14] | 三维深水 Kelvin/Delhommeau Green 函数、水线穿出体开口网格 `WettedHullMesh`、三维 Kochin/Haskind 恒等式 | **深水已覆盖**（`floating_hemisphere`、`submerged_sphere_wave`，§4.10–§4.11；有限水深 John 核待建） |
| 9. 多体水动力耦合与月池/窄缝共振 | 双圆柱/双浮体窄缝波面共振、带月池结构活塞/晃荡模态（WAMIT L00 算例） | 多连通流体域耦合矩阵、高频强驻波高梯度离散 | **基础算子已就绪**（二维/三维深水多连通网格可直接组装，专门窄缝共振算例待建） |
| 10. 有航速兴波阻力（Neumann–Kelvin / Rankine 源） | Wigley 抛物线船型、Series 60 (\(C_B=0.60\))、KCS 集装箱船兴波阻力系数 \(C_w(Fr)\) 与波形图 | 自由液面网格离散、上游辐射偏置差分、跨声速/亚临界 Froude 数波系 | **未覆盖**（需自由液面 Rankine 面元与航速对流项） |
| 11. 船用螺旋桨与水翼升力势流 | 二维 Joukowski / NACA 水翼环量与升力、三维椭圆机翼、DTMB P4119 螺旋桨敞水性能 | 偶极子尾涡片（wake sheet）离散、后缘 **Kutta 压力等值/速度有限条件** | **未覆盖**（当前仅无环量非升力势流） |
| 12. 二阶波浪漂移力与浮式风机/WEC 试验确认 | Maruo / Newman 远场平均漂移力、差频/和频 QTF、DeepCwind 半潜风机 (OC4/OC5)、WaveBot WEC | 二阶水线积分、流体速度平方项（需高精度切向速度恢复）、系泊/PTO 运动方程耦合 | **未覆盖**（切向速度恢复层 `recovery.py` 已就绪，自由液面二阶项待建） |

本文已按“制造解 → 圆柱/球体解析解”的顺序完成前两步；后续应验证自由液面圆柱辐射与绕射，再开展有限水深、复杂船型和试验对比。对试验基准，应报告实验条件及不确定度；不可将与某一组试验吻合等同于通用正确性。

对自由液面问题，可将独立势流程序（例如 WAMIT、NEMOH 或 Capytaine）作为数值互证来源；它们不是解析真值。比较前必须统一时间因子、坐标、法向、系数定义和正负号，并报告参考网格与收敛信息。Capytaine 的公开方法论文可作为软件和数值方法的可追溯来源 [7]。

## 3. 已实现基准：圆柱与球体横向平移

### 3.1 物理模型与解析解

考虑半径 \(a\) 的无限长圆柱在无限、无黏、不可压流体中沿 \(x\) 方向以速度 \(U\) 平移。令极坐标角 \(\theta\) 从 \(x\) 轴起算，则满足无穿透条件且远场速度趋于零的外域势为

\[
\phi(r,\theta)=-\frac{Ua^2}{r}\cos\theta,\qquad r\ge a.
\]

圆柱表面上的势为 \(\phi|_{r=a}=-Ua\cos\theta\)。流体外域的边界法向指向物体内部，故表面法向导数为 \(q_f=-U\cos\theta\)。以物体外法向 \(n_{b,x}=\cos\theta\) 计算每单位跨长附加质量：

\[
m_A'=\frac{\rho}{U^2}\int_{S}\phi\,q_f\,ds
=-\frac{\rho}{U}\int_{S}\phi\,n_{b,x}\,ds
=\rho\pi a^2.
\]

这一解析结果来自经典势流理论 [1, 4]。需特别指出，当前程序的多边形法向是有界多边形的外法向，而物理外域法向指向圆柱内部。对本例的一阶角向模态，程序所得有界域法向导数标量与外域物理法向导数相同；这一特定边界迹关系不能推广为一般外域 Dirichlet-to-Neumann 算子的等价性。

### 3.2 数值实现

`tensorbem.benchmarks.cylinder_added_mass` 在圆周上生成 \(N\) 边形，取面元中点角度 \(\theta_i\)，给定常数面元势值
\[
\phi_i=-Ua\cos\theta_i.
\]
随后调用现有二维 Laplace Dirichlet 求解器得到法向导数 \(q_{N,i}\)，并与 \(q_i=-U\cos\theta_i\) 比较。附加质量使用求解器所得通量，通过边界能量恒等式作离散积分
\[
m_{A,N}'=\frac{\rho}{U^2}\sum_{i=1}^{N}\phi_i q_{N,i}\,\Delta s_i
\]
计算，并与 \(\rho\pi a^2\) 比较。因此，附加质量结果依赖于数值求解得到的通量，而不只是对预设解析势作几何积分。直线面元几何逼近圆形，故应考察网格加密趋势而非要求任一有限面元数下完全相等。

可复现实验命令：

```bash
python -m tensorbem.benchmarks.cylinder_added_mass --panels 64
```

命令将 JSON 结果写入标准输出，支持 `--radius`、`--speed` 和 `--density` 参数。返回值包含计算及解析附加质量、相对误差、法向导数均方根误差、线性系统残差和条件数。

### 3.3 三维球体横向平移

在无界、无黏、不可压流体中，半径 \(a\) 的球体沿 \(x\) 方向以速度 \(U\) 平移。取球坐标极角 \(\theta\) 自 \(x\) 轴起算，满足远场衰减和无穿透条件的外域势为

\[
\phi(r,\theta)=-\frac12\frac{Ua^3\cos\theta}{r^2},\qquad r\ge a,
\]

球面边界势为 \(\phi|_S=-\frac12 Ua\cos\theta\)。流体域外法向指向球内，即 \(n_e=-e_r\)，故表面外域法向导数为

\[
q_e=\frac{\partial\phi}{\partial n_e}=-U\cos\theta.
\]

球体附加质量为

\[
m_A=-\rho\int_S\phi\,n_{b,x}\,dS=\frac12\rho V,\qquad V=\frac{4}{3}\pi a^3,
\]

其中 \(n_{b,x}=\cos\theta\) 为物体外法向的 \(x\) 分量。

`tensorbem.benchmarks.sphere_added_mass` 使用二十面体递归细分得到 \(N=20\cdot4^k\) 个平面三角形，质心配点。由于面元是平面三角形，光滑边界上的自由项为 \(c=1/2\)，外域 Dirichlet 积分方程为

\[
\mathbf S\mathbf q_e=\left(\tfrac12\mathbf I-\mathbf H\right)\boldsymbol\phi.
\]

双层矩阵由三角形立体角解析计算，

\[
H_{ij}=-\frac{\Omega_{ij}}{4\pi},
\]

\[
\Omega_{ij}=2\operatorname{atan2}\left(
\mathbf a\cdot(\mathbf b\times\mathbf c),\ 
|\mathbf a||\mathbf b||\mathbf c|
+(\mathbf a\cdot\mathbf b)|\mathbf c|
+(\mathbf b\cdot\mathbf c)|\mathbf a|
+(\mathbf c\cdot\mathbf a)|\mathbf b|
\right),
\]

其中 \(\mathbf a,\mathbf b,\mathbf c\) 为质点到三角形三个顶点的向量；自面元主值为零。单层矩阵的非自面元用 Duffy 变换后的 \(8\times8\) Gauss 求积，自面元在质心处按极坐标解析积分。附加质量由计算所得通量的边界能量积分给出：

\[
m_{A,N}=\frac{\rho}{U^2}\sum_{i=1}^{N}\phi_i q_{e,i}A_i.
\]

边界势取质心角坐标处的球面解，\(\phi_i=-\tfrac12 Ua\cos\theta_i\)，而不是直接用质心的 \(x\) 坐标，以避免平面面元径向深度造成的 \(O(h^2)\) 幅值偏差。解析通量同样在质心角坐标处取值。

可复现命令：

```bash
python -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2
python -m tensorbem.benchmarks.three_dimensional_study --levels 0 1 2 3
```

基准 JSON 包含附加质量、解析值、相对误差、通量误差、面积/体积比、网格质量、条件数和残差；研究命令额外输出内域线性调和场与外域常值 Dirichlet 两个交叉验证算例。


## 4. 结果

### 4.1 二维圆柱

基准参数为 \(a=1\)、\(U=1\)、\(\rho=1025\)，解析附加质量为 \(3220.13247\)（质量/长度单位）。表 1 报告直接运行基准所得数值；法向导数误差与速度同量纲，此处 \(U=1\)。

**表 1. 圆柱附加质量基准的网格收敛**

| 面元数 \(N\) | 数值附加质量 \(m_{A,N}'\) | 相对误差 | 法向导数 RMSE |
|---:|---:|---:|---:|
| 16 | 3304.60551 | \(2.6233\times10^{-2}\) | \(2.3233\times10^{-2}\) |
| 32 | 3241.00053 | \(6.4805\times10^{-3}\) | \(5.7269\times10^{-3}\) |
| 64 | 3225.32509 | \(1.6125\times10^{-3}\) | \(1.4248\times10^{-3}\) |
| 128 | 3221.42800 | \(4.0232\times10^{-4}\) | \(3.5551\times10^{-4}\) |

该系列中附加质量相对误差及法向导数误差均随细化降低；附加质量误差约按 \(N^{-2}\) 下降，这是解析直线面元积分取代中点核积分后的结果。64 面元时附加质量相对误差约 \(1.61\times10^{-3}\)，线性系统残差约 \(6.58\times10^{-16}\)。小代数残差仅说明离散线性系统已求解，并不能单独证明连续问题或特定外域算子正确。

### 4.2 三维球体

基准参数为 \(a=1\)、\(U=1\)、\(\rho=1025\)，解析附加质量为 \(2146.75498\)。网格由二十面体递归细分得到；面积与体积比说明平面三角形始终位于球内。表 2 使用质心角坐标处的球面边界迹与解析通量，附加质量由计算通量积分得到。

**表 2. 球体附加质量基准的网格收敛**

| 面元数 \(N\) | \(h_{\max}\) | 数值附加质量 \(m_{A,N}\) | 相对误差 | 通量相对 L2 误差 | 附加质量观测阶 |
|---:|---:|---:|---:|---:|---:|
| 20 | 1.05146 | 1690.18039 | \(2.1268\times10^{-1}\) | \(3.3338\times10^{-2}\) | — |
| 80 | 0.61803 | 2007.73279 | \(6.4759\times10^{-2}\) | \(2.7884\times10^{-2}\) | 2.238 |
| 320 | 0.32492 | 2109.44290 | \(1.7381\times10^{-2}\) | \(1.4542\times10^{-2}\) | 2.046 |
| 1280 | 0.16465 | 2137.16456 | \(4.4674\times10^{-3}\) | \(6.8458\times10^{-3}\) | 1.999 |

附加质量作为边界积分量呈近二阶收敛；但逐面元通量误差只呈约一阶下降（观测阶 0.34、1.01、1.11）。原因是平面多面体的棱边使外域解出现奇异性，质心离棱边距离随 \(h\) 缩小，点值通量误差受其支配；积分量中的奇异性贡献相互抵消，故附加质量仍保持二阶。报告同时给出两者，避免用积分量的良好收敛掩盖局部通量误差。三维验证的完整数据、网格面积/体积比、条件数和残差见[三维验证报告](three_dimensional_bem.md)。

### 4.3 二维椭圆柱

椭圆柱把"特定 Fourier 模态的巧合"去掉：对圆而言，边界势 \(-Ua\cos\theta\) 在圆内外的调和延拓是同一个线性函数，因此有界域求解器的通量在连续意义下等于外域流体的法向导数。椭圆不再具有这一性质，因此该算例是二维**外域**算子（\(\mathbf S\mathbf q_e=(\tfrac12\mathbf I-\mathbf H)\mathbf u\)）的第一个真正测试。

参数 \(a=1.5\)、\(b=0.75\)、\(U=1\)、\(\rho=1025\)，解析值 \(m_{A,x}'=\rho\pi b^2=1811.324514\)、\(m_{A,y}'=\rho\pi a^2=7245.298057\)（每方向使用与之垂直的半轴）。

**表 3. 椭圆柱附加质量（每单位跨长）的网格收敛**

| 面元数 \(N\) | \(h_{\max}\) | \(m_{A,x}'\) | 相对误差 | 观测阶 | \(m_{A,y}'\) | 相对误差 | 观测阶 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.57686 | 1724.874626 | \(4.7727\times10^{-2}\) | — | 6846.350244 | \(5.5063\times10^{-2}\) | — |
| 32 | 0.29299 | 1789.507302 | \(1.2045\times10^{-2}\) | 2.032 | 7142.640682 | \(1.4169\times10^{-2}\) | 2.004 |
| 64 | 0.14707 | 1805.863574 | \(3.0149\times10^{-3}\) | 2.010 | 7219.345656 | \(3.5820\times10^{-3}\) | 1.995 |
| 128 | 0.07361 | 1809.959574 | \(7.5356\times10^{-4}\) | 2.003 | 7238.778944 | \(8.9977\times10^{-4}\) | 1.996 |

逐面元通量误差同样接近二阶（相对 L2 由 \(3.4\times10^{-2}\) 降至 \(5.7\times10^{-4}\)），与三维情形不同：二维多边形的角点奇异性弱，且平面弦上的常数密度积分本来就二阶精确。单层矩阵条件数约 \(0.6N\)。取 \(a=b=1\) 时两个方向都退化到 \(\rho\pi\)，与圆柱基准一致。

### 4.4 三维椭球

解析附加质量为 \(m_{A,i}=\rho V\alpha_i/(2-\alpha_i)\)，其中 \(\alpha_i\) 为 Lamb 形状因子，满足 \(\alpha_1+\alpha_2+\alpha_3=2\)（代码每次求值都校验该恒等式，数值积分未收敛时报错）。因子在球极限给出 \(2/3\)，在薄圆盘向极限给出经典 \(\tfrac83\rho b^3\)，在细长体横向极限给出 \(\rho V\)，并与长球闭式解在 \(10^{-8}\) 内一致。边界数据在质心沿原点射线的椭球投影点上取值，故仍是精确的线性迹 \(u=-Ukx_i\)，\(k=\alpha_i/(2-\alpha_i)\)。

半轴 \((1.5,1.0,0.75)\)，\(\alpha=(0.400928,\ 0.670713,\ 0.928360)\)，解析值 \((1211.052,\ 2437.1535,\ 4184.3892)\)。

**表 4. 椭球三轴附加质量的网格收敛**

| 面元数 \(N\) | \(h_{\max}\) | \(V/V_{\text{exact}}\) | \(x\) 相对误差 | 观测阶 | \(y\) 相对误差 | 观测阶 | \(z\) 相对误差 | 观测阶 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1.57719 | 0.605461 | \(2.0855\times10^{-1}\) | — | \(2.1126\times10^{-1}\) | — | \(2.2541\times10^{-1}\) | — |
| 80 | 0.92705 | 0.873453 | \(6.1623\times10^{-2}\) | 2.294 | \(6.4728\times10^{-2}\) | 2.226 | \(7.0201\times10^{-2}\) | 2.195 |
| 320 | 0.48738 | 0.966161 | \(1.6313\times10^{-2}\) | 2.067 | \(1.7371\times10^{-2}\) | 2.046 | \(1.9080\times10^{-2}\) | 2.026 |
| 1280 | 0.24697 | 0.991394 | \(4.1679\times10^{-3}\) | 2.007 | \(4.4621\times10^{-3}\) | 1.999 | \(4.9359\times10^{-3}\) | 1.989 |

三轴误差同时下降且量级相近，说明误差由面元尺寸而非某一方向的网格偏斜主导；缩放 icosphere 在三个主方向的角分布相同，正好避免"某一轴网格更细"造成的假收敛。逐面元通量误差仍只呈约一阶（\(9.8\times10^{-2}\)、\(7.3\times10^{-2}\)、\(8.0\times10^{-2}\) 降至 \(1.3\)、\(1.4\)、\(1.7\times10^{-2}\)），机制与球体相同。

### 4.5 内部固体谐波

度 1–3 的实调和多项式在真实多面体域上仍是精确解，因此该组算例检验的是纯离散误差。取 \(x,\ y,\ z\)；\(x^2-y^2,\ 2xy,\ 2z^2-x^2-y^2\)；\(x^3-3xy^2,\ 3x^2y-y^3,\ x(4z^2-x^2-y^2),\ z(2z^2-3x^2-3y^2)\) 共十个多项式，解析通量为 \(\nabla u\cdot\boldsymbol n\)。

同一 1280 面元网格上，通量相对 L2 误差随度数上升：度 1 为 \(3.76\times10^{-3}\)、度 2 为 \(7.34\times10^{-3}\)、度 3 为 \(1.19\times10^{-2}\)；内部探针 RMSE 则基本不变（\(2.29\times10^{-4}\)、\(2.31\times10^{-4}\)、\(1.82\times10^{-4}\)）。也就是说，更强的角向变化主要伤害边界通量，而不明显伤害内部场——此前只用线性场的三维验证会低估这一效应。度 1 的数值与既有线性场验证逐位一致，作为回归保护。

### 4.6 Ursell 潜没圆柱辐射（自由液面，深水）

前五节全部不含自由液面。本节补上第一个真正的频域水动力基准：半径 \(a\) 的圆柱，轴心潜深 \(f\)，在无限水深中做单位速度的垂荡（或横荡）简谐振动，频率 \(\omega\)，波数 \(k=\omega^2/g\)。

**参考解独立于 BEM。** 采用 Ursell (1949) 的多极子解：把速度势展开为满足自由液面条件与辐射条件的无波谐函数，物面条件化为无限代数方程组

\[
-\frac{a_m}{m}+\sum_{n\ge 1}a_n A_{mn}=-\delta_{m1},
\]

深水时系数有闭式表达（Evans 等 1979）：

\[
A_{mn}=\frac{(-ka)^{m+n}}{m!\,n!}\left[\frac{(m+n-1)!}{(2kf)^{m+n}}+2\sum_{r=1}^{m+n-1}\frac{(r-1)!}{(2kf)^{r}}+2e^{-2kf}\bigl(\mathrm i\pi-\mathrm{Ei}(2kf)\bigr)\right],
\]

水动力系数由第一个未知量给出：\(\mu+\mathrm i\nu=-1+2a_1\)，其中 \(\mu=A/(\rho\pi a^2)\)、\(\nu=B/(\omega\rho\pi a^2)\)。这里没有积分方程、没有面元、没有用本项目的核函数：唯一的数值成分是截断的有限方程组与指数积分 \(\mathrm{Ei}\)（本仓库自行实现，见 `tensorbem.analytic.exponential_integral`）。

**参考解自身的验证**（不依赖任何 BEM）：

| 检验 | 结果 |
|---|---|
| 深潜极限 \(f/a\to\infty\) | \(a/f=0.04\) 时 \(\mu=0.99908\)，\(a/f=0.02\) 时 \(A=3219.44\)，极限 \(\rho\pi a^2=3220.13\) |
| 文献给出的阻尼零点（McIver & McIver 附录） | \(a/f=0.95\)、\(ka=1.50394\) 处 \(\nu=3\times10^{-13}\)；\(a/f=0.9\)、\(ka=2.16016\) 处 \(\nu=6\times10^{-16}\) |
| 截断收敛 | 20 项与 40 项的结果相对差 \(<10^{-12}\) |

第二条尤其有力：这两组频率是文献独立算出的阻尼零点，本实现把它们复现到机器精度。

**BEM 与参考解的对比**（\(a=1\)、\(f=2\)、\(ka=0.6\)、\(\rho=1025\)、\(g=9.80665\)，附加质量与阻尼均为单位长度值）：

| 面元数 | \(A\) | 相对误差 | \(B\) | 相对误差 | 近场/远场阻尼比 |
|---|---|---|---|---|---|
| 64 | 2583.69 | \(1.51\times10^{-2}\) | 2785.28 | \(2.47\times10^{-2}\) | 1.0113 |
| 128 | 2564.81 | \(7.72\times10^{-3}\) | 2753.51 | \(1.30\times10^{-2}\) | 1.0057 |
| 256 | 2555.09 | \(3.90\times10^{-3}\) | 2736.27 | \(6.67\times10^{-3}\) | 1.0029 |
| 参考解（Ursell） | 2545.17 | — | 2718.15 | — | — |
| 基本解方法（MFS，64 面元） | 2544.69 | \(1.9\times10^{-4}\) | 2715.76 | \(8.8\times10^{-4}\) | — |

**近场/远场一致性才是辐射条件的检验。** \(B\) 有两种算法：一是物面压力积分 \(\omega\rho\,\mathrm{Im}\!\int\varphi n\,\mathrm ds\)；二是 Haskind 关系 \(B=\omega\rho|a_w|^2\)，其中 \(a_w\) 由远场波浪振幅给出。两者在上表中相差 1.1%、0.6%、0.3%，随网格加密单调收敛到零。任何只报告附加质量、不检查该恒等式的自由液面实现都可能「看起来正确」——本项目此前正是如此：早期采用的直接（Green 恒等式）公式给出的附加质量可信，而阻尼比正确值小 3–4 个数量级，原因是在二维情形波浪不衰减，恒等式在有限物体上不闭合。改用 Ursell 所用的间接（源分布）公式后该恒等式才成立。这条经验已写入代码文档，避免后人重新引入。

顺带澄清一个此前困扰的因子 2：Green 函数的远场振幅确为朴素估计的两倍，因为主值积分项与留数项在远场辐射强度相同；因此远场系数取完整留数强度而非其一半。把两处都按一半处理会得到 4 倍的阻尼误差。

**无限水深下垂荡与横荡系数相同**，这不是丢了一个自由度：多极子矩阵与运动模态无关。数值上，垂荡与横荡的解是两个不同的势场，但其合力在 128 面元下相差 \(1.7\times10^{-14}\)（测试中作为断言）。有限水深时两者才会分离。

**离散精度说明。** 该常数面元间接公式在本问题上只有约一阶收敛（面元数每翻一倍误差减半），128 面元附加质量误差 0.8%。同样的核函数配合基本解方法（把点源放在物体内部，无奇异积分）则达到 \(10^{-4}\)。因此本算例的正确用法是看收敛趋势并以 Ursell 解析解为准，而不是把单个面元数的结果当作最终精度。

运行方式：

```
python -m tensorbem.benchmarks.ursell_cylinder --panels 128 --ka 0.6 --output out.json
```

输出为严格 JSON（`allow_nan=False`），同时给出参考解、面元解、MFS 解、远近场阻尼与残差。

### 4.7 外域刚体 Neumann 附加质量张量与转动附加惯性矩（Lamb 1932）

§4.1–§4.4 的无自由液面基准均采用预设解析表面势 \(\phi|_\Gamma\)、求解第一类积分方程 \(\mathbf S\mathbf q_e=(\tfrac12\mathbf I-\mathbf H)\boldsymbol\phi\) 的策略。然而对一般船体截面以及**刚体转动模态（横摇/纵摇/首摇）**，表面势在求解前是未知量；工程水动力程序（如 WAMIT 零频/高频极限）实际施加的是刚体无穿透 **Neumann 边界条件**

\[
\frac{\partial\phi_j}{\partial n_{\text{body}}}=n_j,\qquad q_{e,j}=-\frac{\partial\phi_j}{\partial n_{\text{body}}}=-n_j,
\]

并直接求解第二类 Fredholm 边界积分方程

\[
\left(\tfrac12\mathbf I-\mathbf H\right)\boldsymbol\phi_j=\mathbf S(-\mathbf n_j),\qquad A_{ij}=-\rho\int_\Gamma \phi_j n_i\,\mathrm dS.
\]

与第一类算子 \(\mathbf S\)（条件数随网格加密按 \(O(h^{-1})\) 增长，且在二维存在对数容量退化）相比，外域第二类算子 \(\tfrac12\mathbf I-\mathbf H\) 的谱远离零点，条件数在任意加密下保持 \(O(1)\)，同时刚体运动天然满足零净通量 \(\int_\Gamma n_j\,\mathrm dS=0\)。

**解析闭式参考解（Lamb 1932, §§71, 114–115）：**
* 二维椭圆截面（半轴 \(a,b\)）绕中心的单位跨长横摇附加转动惯量：
  \[
  A_{66}'=\frac{1}{8}\rho\pi(a^2-b^2)^2
  \]
  （圆截面 \(a=b\) 时精确为零，薄平板极限 \(b\to 0\) 时退化为 \(\tfrac18\rho\pi a^4\)）；
* 三维三轴椭球（半轴 \(a_1,a_2,a_3\)，体积 \(V=\tfrac43\pi a_1a_2a_3\)，Lamb 形状因子 \(\alpha_1,\alpha_2,\alpha_3\)）绕第 \(i\) 主轴的转动附加惯性矩（对循环指标 \((i,j,k)\)）：
  \[
  A_{i+3,i+3}=\frac{1}{5}\rho V\frac{(a_j^2-a_k^2)^2(\alpha_k-\alpha_j)}{2(a_j^2-a_k^2)+(a_j^2+a_k^2)(\alpha_j-\alpha_k)}.
  \]

**表 5. 二维椭圆（\(a=1.5,\ b=0.75,\ \rho=1025\)）刚体 Neumann \(3\times3\) 张量收敛**

| 面元数 \(N\) | 条件数 | \(A_{11}'\) 相对误差 | \(A_{22}'\) 相对误差 | 转动 \(A_{66}'\) 相对误差 | \(A_{66}'\) 观测阶 | 张量对称误差 \(\|\mathbf A-\mathbf A^\top\|_{\max}\) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3.078 | \(2.64\times10^{-3}\) | \(5.15\times10^{-3}\) | \(3.23\times10^{-2}\) | — | \(6.4\times10^{-13}\) |
| 32 | 3.019 | \(7.42\times10^{-4}\) | \(1.41\times10^{-3}\) | \(8.71\times10^{-3}\) | 1.936 | \(9.3\times10^{-13}\) |
| 64 | 3.005 | \(1.94\times10^{-4}\) | \(3.75\times10^{-4}\) | \(2.25\times10^{-3}\) | 1.966 | \(4.5\times10^{-13}\) |
| 128 | 3.001 | \(4.94\times10^{-5}\) | \(9.70\times10^{-5}\) | \(5.70\times10^{-4}\) | 1.982 | \(8.0\times10^{-13}\) |

**表 6. 三维三轴椭球（\(a_1=1.5,\ a_2=1.0,\ a_3=0.75,\ \rho=1025\)）刚体 Neumann \(6\times6\) 张量收敛**

| 细分层 | 面元数 \(N\) | 条件数 | 平动最大相对误差（纵荡/横荡/垂荡） | 转动相对误差（横摇 / 纵摇 / 首摇） | 转动观测阶 |
|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 2.447 | \(2.63\times10^{-1}\) | \(4.90\times10^{-1}\) / \(5.19\times10^{-1}\) / \(5.11\times10^{-1}\) | — |
| 1 | 80 | 2.246 | \(8.11\times10^{-2}\) | \(1.63\times10^{-1}\) / \(1.78\times10^{-1}\) / \(1.74\times10^{-1}\) | 2.068 / 2.012 / 2.022 |
| 2 | 320 | 2.130 | \(2.14\times10^{-2}\) | \(4.60\times10^{-2}\) / \(4.84\times10^{-2}\) / \(4.85\times10^{-2}\) | 1.972 / 2.028 / 1.992 |

测试套件还验证了刚体网格平移时的 Newman 平行轴变换恒等式至 \(10^{-12}\) 精度。运行命令：

```bash
python -m tensorbem.benchmarks.rigid_body_added_mass --panels 16 32 64 128 --subdivisions 0 1 2
```

### 4.8 Dean–Ursell–Ogilvie 潜没柱体规则波绕射与 Haskind 互易关系

在二维深水自由液面层（时间因子 \(e^{-\mathrm i\omega t}\)）中引入沿 \(+x\) 传播的规则 Airy 入射波

\[
\phi_I(x,z)=-\frac{\mathrm i g A}{\omega}e^{\nu z+\mathrm i\nu x},\qquad \frac{\partial\phi_I}{\partial n}=\omega A(n_x-\mathrm i n_z)e^{\nu z+\mathrm i\nu x},
\]

通过散射源分布积分方程 \((\tfrac12\mathbf I+\mathbf K')\boldsymbol\sigma_S=-\partial_n\phi_I\) 求解散射势 \(\phi_S\)，并利用 Gauss–Legendre 求积由远场留数渐近式提取反射系数 \(R\) 与透射系数 \(T\)：

\[
R=\frac{\omega}{gA}\int_\Gamma e^{\nu z_0+\mathrm i\nu x_0}\sigma_S\,\mathrm ds,\qquad T=1+\frac{\omega}{gA}\int_\Gamma e^{\nu z_0-\mathrm i\nu x_0}\sigma_S\,\mathrm ds.
\]

同时实现**无需绕射势**的 Haskind–Newman 互易公式（`haskind_excitation_force`）：仅由静水单位速度辐射解 \(\phi_j\) 与入射波在物面的 Green 积分计算波浪激励力 \(F_j=-\mathrm i\omega\rho\int_\Gamma(\phi_I n_j-\phi_j\partial_n\phi_I)\,\mathrm ds\)。

对潜没圆柱（半径 \(a=1\)、潜深 \(f=2\)、\(ka=0.6\)），独立实现的 Ursell (1950) / Ogilvie (1963) 多极子闭式解给出三大精确物理定律：**全频段零反射** \(R\equiv 0\)、**纯相移全透射** \(|T|\equiv 1\)（\(\delta=26.655^\circ\)），以及**圆偏振波浪力** \(F_z=-\mathrm i F_x\)。

**表 7. 潜没圆柱与潜没椭圆柱规则波绕射验证（\(ka=0.6,\ N=64\)）**

| 物理量 / 守恒律 | 潜没圆柱（\(a=1,\ f=2\)） | 潜没椭圆柱（\(a=1.4,\ b=0.7,\ f=2\)） |
|---|---:|---:|
| 反射系数模 \(|R|\) | \(2.20\times10^{-15}\)（解析真值 \(0\)） | \(9.751\times10^{-2}\)（对称破缺产生有限反射） |
| 透射系数模 \(|T|\) | \(1.000000\)（透射误差 \(8.39\times10^{-3}\)） | \(0.995234\) |
| 波浪能量守恒误差 \(\||R|^2+|T|^2-1\|\) | \(0.0\)（机器精度） | \(0.0\)（机器精度） |
| 圆偏振误差 \(\|F_z+\mathrm i F_x\|/\|F_x\|\) | \(1.10\times10^{-14}\) | —（非圆截面不具圆偏振） |
| 绕射积分与 Haskind 互易相对差（水平 / 垂向） | \(2.16\times10^{-5}\) / \(2.16\times10^{-5}\) | \(7.97\times10^{-4}\) / \(1.99\times10^{-3}\) |
| MFS 独立参考解水平波浪力相对误差 | \(1.51\times10^{-3}\) | — |

运行命令：

```bash
python -m tensorbem.benchmarks.dean_diffraction --panels 128 --ka 0.6
```

### 4.9 MacCamy–Fuchs 贯底直立圆柱波浪绕射与 CHIEF 不规则频率消除

对于水深 \(h\) 中自海底 \(z=-h\) 贯通至自由液面 \(z=0\) 的任意等截面直立柱体，三维线性绕射势可精确分离变量为

\[
\Phi(x,y,z)=-\frac{\mathrm i g A}{\omega}\frac{\cosh k(z+h)}{\cosh kh}\,\varphi(x,y),\qquad \omega^2=gk\tanh kh,
\]

其中水平面总势 \(\varphi=\varphi_I+\varphi_S\) 满足二维外域 Helmholtz 刚体散射问题 \((\nabla_{xy}^2+k^2)\varphi_S=0\)、\(\partial_n\varphi|_\Gamma=0\) 及 Sommerfeld 辐射条件。`tensorbem.helmholtz` 采用出射波核 \(G_k(r)=\frac{\mathrm i}{4}H_0^{(1)}(kr)\)，将双层核拆分为直线面元解析立体角项与光滑 Hankel 余项求积，建立直接边界积分方程

\[
\left(\tfrac12\mathbf I-\mathbf H_k\right)\boldsymbol\varphi=\boldsymbol\varphi_I.
\]

本仓库纯 NumPy 实现了全向量化的任意阶第一类/第二类 Bessel 函数（带防溢出缩放的 Miller 逆向递推 + Hankel 大宗量渐近展开，Wronskian 恒等式精度达 \(10^{-14}\)），并以此构建 MacCamy & Fuchs (1954) 的精确级数参考解：

\[
F_x=\frac{4\rho g A}{k^2}\tanh(kh)\,\frac{1}{H_1^{(1)\prime}(ka)},\qquad M_y=\frac{4\rho g A}{k^3}\bigl(kh\tanh kh+\operatorname{sech}kh-1\bigr)\frac{1}{H_1^{(1)\prime}(ka)}.
\]

**表 8. MacCamy–Fuchs 直立圆柱绕射收敛性（\(ka=1.0,\ h/a=3.0,\ \rho=1025\)）**

| 面元数 \(N\) | 条件数 | 波浪力 \(F_x\) 相对误差 | 倾覆力矩 \(M_y\) 相对误差 | 柱面总势 L2 相对误差 | \(F_x\) 观测阶 |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.015 | \(8.71\times10^{-3}\) | \(8.71\times10^{-3}\) | \(1.19\times10^{-2}\) | — |
| 32 | 2.665 | \(2.14\times10^{-3}\) | \(2.14\times10^{-3}\) | \(2.94\times10^{-3}\) | 2.040 |
| 64 | 2.509 | \(5.31\times10^{-4}\) | \(5.31\times10^{-4}\) | \(7.29\times10^{-4}\) | 2.016 |
| 128 | 2.434 | \(1.32\times10^{-4}\) | \(1.32\times10^{-4}\) | \(1.82\times10^{-4}\) | 2.007 |

在频率扫掠（\(N=64\)）中，数值等效惯性力系数 \(C_M(ka)\) 从长波 Morison 区（\(ka=0.1\) 时计算值 \(2.0184\)、精确值 \(2.0192\)）平滑下降至短波绕射区（\(ka=3.0\) 时计算值 \(0.3078\)、精确值 \(0.3077\)）。

**不规则频率（Irregular Frequencies）机理与 Schenck CHIEF 消除：**
在闭合水线边界上，外域算子 \(\tfrac12\mathbf I-\mathbf H_k\) 在内域 Dirichlet 特征值 \(J_m(ka)=0\) 处严格秩亏——这正是 WAMIT、Capytaine 等穿水浮体面板法最著名的数值陷阱。在 \(N=64\) 正多边形的离散共振波数处对比未增广方程与追加 5 个非对称内部零场点（`chief_points="auto"`）的最小二乘解：

**表 9. 内域共振波数处未增广 BIE 与 Schenck CHIEF 增广解的对比（\(N=64\)）**

| 内域 Dirichlet 共振模态 | 离散共振 \(ka\) | 未增广条件数 | CHIEF 条件数 | 未增广 \(F_x\) 相对误差 | CHIEF \(F_x\) 相对误差 |
|---|---:|---:|---:|---:|---:|
| \(J_0(j_{0,1})=0\) 轴对称模态 | 2.4068 | **1002.5** | **2.31** | \(2.03\times10^{-3}\)（与 \(m=1\) 正交） | \(2.02\times10^{-3}\) |
| \(J_1(j_{1,1})=0\) 偶极模态 | 3.8348 | **665.0** | **3.34** | **\(1.035\times10^{-1}\)（10.35% 失真）** | **\(2.81\times10^{-3}\)（0.28%）** |

表 9 清晰展示了模态选择性：\(m=0\) 轴对称共振虽令矩阵条件数暴涨至 \(10^3\)，但因与 \(\cos\theta\) 水平力积分正交而容易在粗略测试中被漏检；\(m=1\) 偶极共振则直接污染 \(\cos\theta\) 分量，导致波浪力产生 **10.35%** 的伪共振尖峰。5 个内部 CHIEF 零场约束同时将两处条件数恢复至 \(O(1)\) 并将波浪力误差压回 \(0.28\%\)。运行命令：

```bash
python -m tensorbem.benchmarks.maccamy_fuchs --panels 16 32 64 128
```

### 4.10 Hulme (1982) 三维穿水漂浮半球辐射、绕射与三维 CHIEF 不规则频率消除

对于一般非柱状三维穿水浮体（\(z\le 0\)，水线回路位于静水面 \(z=0\)），垂向无法分离变量，必须在三维流体半空间中直接构造满足线性自由液面条件与远场柱面波辐射条件的 Kelvin/Havelock Green 函数 [1, 15, 16]：

\[
G(\mathbf x,\boldsymbol\xi;\nu)=\frac{1}{4\pi r}+\frac{1}{4\pi r_1}+\frac{\nu}{2\pi}\Bigl[F(\nu R,-\nu(z+\zeta))+\mathrm i\pi e^{\nu(z+\zeta)}J_0(\nu R)\Bigr],
\]

其中 \(R=\sqrt{(x-\xi)^2+(y-\eta)^2}\)，\(r=|\mathbf x-\boldsymbol\xi|\)，\(r_1=\sqrt{R^2+(z+\zeta)^2}\)，\(-\nabla_{\mathbf x}^2 G=\delta(\mathbf x-\boldsymbol\xi)\)（与三维 Rankine 层 `tensorbem.kernels3d` 保持同一正号约定，因此在 \(\operatorname{Re}\{\phi e^{-\mathrm i\omega t}\}\) 下出射波残数符号为 \(+\mathrm i\)）。`tensorbem.free_surface3d` 将无量纲四分之一平面 \((X\ge 0,Y\ge 0)\) 划分为近轴精确阶乘递推锥区（\(0<Y\le 4.5,\ X\le 0.35Y\)）、浅水/水线 Havelock 微分方程表示区（\(Y\le 4.5\)，结合纯 NumPy 实现的 Struve 函数 \(\mathbf H_0,\mathbf H_1\)）、中水深去奇异 Gauss–Legendre + Gauss–Laguerre 求积区与远场 17 项 Newman 渐近 Legendre 展开区（\(D\ge 26\)），无需外部特殊函数库或插值查表即可在全部接缝处达到 \(10^{-11}\text{–}10^{-13}\) 连续精度，静水面边界条件残差 `verify_free_surface_bc_3d` 低于 \(10^{-14}\)。

与二维不同，三维下 \(1/(4\pi r)=O(r^{-1})\) 且出射表面波按 \(O(R^{-1/2})\) 衰减并严格满足 Sommerfeld 辐射条件，因此 Green 第三恒等式在无穷远处精确闭合，`tensorbem.free_surface_solver3d` 直接在湿表面 \(\Gamma_B\)（支持闭合潜体 `TriangleMesh` 与水线开口船体 `WettedHullMesh`）上组装直接势积分方程

\[
\left(\tfrac12\mathbf I-\mathbf H_\nu\right)\boldsymbol\phi=\mathbf S_\nu\mathbf q_e,\qquad \mathbf q_e=-\frac{\partial\boldsymbol\phi}{\partial n_{\text{body}}}.
\]

针对水线面元处 \(r\to 0\) 与镜像距离 \(r_1\to 0\) 同时退化的数值难点，求解器将直接源与 \(z=0\) 镜像反射面元的双层项全部采用解析立体角公式积分，单层项采用极坐标扇区 / Duffy 求积，仅对光滑兴波余项 \(G_W\) 采用二阶 3 点对称三角面元求积。独立参考真值由 `hulme_hemisphere_radiation` 按 Hulme (1982) [6] 构造：以原点 Kelvin 波源/水平偶极子叠加 48 项球面无波势（wave-free potentials），精确复现文献表 1（垂荡）与表 2（纵荡）数据，并由原点 Kochin 振幅给出精确复波浪激励力。

**表 10. Hulme 漂浮半球三维辐射与绕射网格收敛（\(Ka=1.0,\ a=1,\ \rho=1025\)）**

| 细分层级 | 面元数 \(N\) | 条件数 | \(A_{33}'\) 相对误差 | \(B_{33}'\) 相对误差 | \(A_{11}'\) 相对误差 | 垂向波浪力 \(F_z\) 相对误差 | \(A_{33}' / B_{33}' / A_{11}' / F_z\) 观测阶 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 3.893 | \(1.20\times10^{-1}\) | \(1.22\times10^{-1}\) | \(1.80\times10^{-1}\) | \(6.67\times10^{-2}\) | — |
| 2 | 64 | 3.052 | \(2.97\times10^{-2}\) | \(4.50\times10^{-2}\) | \(5.05\times10^{-2}\) | \(2.21\times10^{-2}\) | 2.23 / 1.59 / 2.04 / 1.77 |
| 3 | 256 | 2.783 | **\(7.41\times10^{-3}\)** | **\(1.18\times10^{-2}\)** | **\(1.30\times10^{-2}\)** | **\(5.67\times10^{-3}\)** | **2.06 / 1.98 / 2.01 / 2.02** |

256 面元（单频组装求解耗时约 1.1 s）下，近场压力积分阻尼与远场三维 Kochin 方位角积分阻尼相对差仅 \(5.45\times10^{-4}\)，三维直接绕射解与边界 Haskind 互易积分相对差为 \(2.09\times10^{-4}\)（纵荡）与 \(8.09\times10^{-4}\)（垂荡）。

**三维穿水浮体不规则频率与免盖板三维 CHIEF 消除：**
当波数接近半球内域水线问题的第一轴对称特征值 \(Ka\approx 2.58\) 时，在流体外域 \(\Omega_f\) 对船体内部水线以下点 \(\mathbf x_{\text{int}}\in\Omega_B\ (z_{\text{int}}<0)\) 应用 Green 第二恒等式，由于外自由液面 \(\Gamma_F\) 与无穷远柱面上的边界积分精确抵消，**无需像传统盖板法（extended-boundary lid method）那样额外剖分内部水线面网格**，即可直接追加严格零场方程 \(-\mathbf H_\nu^{\text{int}}\boldsymbol\phi=\mathbf S_\nu^{\text{int}}\mathbf q_e\)：

**表 11. 漂浮半球内域水线共振区（\(Ka=2.58,\ N=256\)）三维 CHIEF 正则化对比**

| 方案 | 条件数 | 计算 \(A_{33}'\)（精确 \(0.4007\)） | 计算 \(B_{33}'\)（精确 \(0.0631\)） | 物理耗散性（\(B_{33}>0\)） |
|---|---:|---:|---:|---|
| 未增广外域 BIE | **145.66** | 0.3481 | **\(-0.1042\)（严重违背无源耗散性）** | **违背（出现非物理负阻尼）** |
| 5 点三维内部 CHIEF（`chief_points="auto"`） | **4.24** | **0.3919** | **\(+0.0687\)** | **严格满足** |

运行命令：

```bash
python -m tensorbem.benchmarks.floating_hemisphere --subdivisions 1 2 3
```

### 4.11 Wang / Linton 三维深水潜没球体辐射与绕射基准

为提供完全不依赖 `tensorbem.free_surface3d` 兴波核求值器的**绝对独立三维真值锚点**，`submerged_sphere_wave_multipole` 基于 Thorne 多极子展开与球谐加法定理（Wang 1986; Linton 1991 [13]）推导了球心位于 \((0,0,-f)\) 的潜没球体在 \(m=0\)（垂荡/轴对称绕射）与 \(m=1\)（纵荡/反对称绕射）下的线性代数耦合系统——全部耦合矩阵元均化简为精确阶乘与指数积分 \(\operatorname{Ei}(2\nu f)\) 的闭式代数表达式（**零数值求积**），在任意频率与潜深下满足 Newman 三维轴对称 Haskind 恒等式 \(|F_z/A|^2=\frac{2\rho g\omega}{\nu^2}B_{33}\)、\(|F_x/A|^2=\frac{4\rho g\omega}{\nu^2}B_{11}\) 至机器精度（\(3.1\text{–}6.5\times10^{-16}\)），并在深潜极限 \(f/a\to\infty\) 下精确退化至无界流 Lamb 附加质量 \(A'=0.5\) 与 Morison 惯性力系数 \(C_M=1.5\)。

**表 12. 三维潜没球体辐射与绕射网格收敛（\(f/a=1.5,\ Ka=0.8,\ \rho=1025\)）**

| 细分层级 | 面元数 \(N\) | 条件数 | \(A_{33}'\) 相对误差 | \(A_{11}'\) 相对误差 | 水平/垂向波浪力相对误差 | 球心转动耦合比 | \(A_{33}' / B_{33}' / A_{11}' / F_z\) 观测阶 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 2.407 | \(2.55\times10^{-1}\) | \(2.68\times10^{-1}\) | \(3.60\times10^{-1}\) / \(3.70\times10^{-1}\) | \(2.5\times10^{-32}\) | — |
| 1 | 80 | 2.259 | \(7.30\times10^{-2}\) | \(8.02\times10^{-2}\) | \(1.16\times10^{-1}\) / \(1.19\times10^{-1}\) | \(2.2\times10^{-4}\) | 2.12 / 1.72 / 2.04 / 1.91 |
| 2 | 320 | 2.188 | **\(1.85\times10^{-2}\)** | **\(2.08\times10^{-2}\)** | **\(3.10\times10^{-2}\)** / **\(3.20\times10^{-2}\)** | **\(2.6\times10^{-5}\)** | **2.06 / 1.93 / 2.03 / 1.98** |

运行命令：

```bash
python -m tensorbem.benchmarks.submerged_sphere_wave --subdivisions 0 1 2
```

## 5. 讨论与适用限制

1. **无界流层已同时覆盖 Dirichlet 迹与真实刚体 Neumann 6-DOF 张量。** 二维椭圆柱（\(3\times3\)）与三维三轴椭球（\(6\times6\)）均由真实无穿透条件第二类积分方程给出二阶收敛的平动与转动附加惯量，但三维逐面元局部通量仍受多面体棱边奇异性限制为约一阶精度，多连通升力体（Kutta 尾涡条件）尚未引入。
2. **自由液面能力已完整打通二维与三维深水 6-DOF 辐射、绕射、Kochin 远场能量守恒、Haskind 互易及 CHIEF 不规则频率消除。** 无论是闭合潜体（`TriangleMesh`）、穿水一般三维船体（`WettedHullMesh`）还是有限水深等截面贯底柱体（`helmholtz`），均具备完整的解析/半解析对照基准。当前尚缺的是**有限水深三维一般浮体 Green 函数**（John 垂向特征级数核，对应 Garrett 截断圆柱基准）。
3. **三维稠密直接求解的规模限制。** 当前二维与三维求解器均组装稠密矩阵并使用直接法/最小二乘求解（\(O(N^2)\) 内存、\(O(N^3)\) 计算量），适合百至千余面元的算法验证与教学研究，尚不包含 \(\mathcal H\)-matrix / FMM 加速或对称平面（\(xOz\)/\(yOz\) 镜像）减半技术。
4. **势流假设不包含黏性效应与非线性碎波。** 小直径管件或大波陡工况下的涡脱落阻力（Morison 曳力项）、横摇黏性阻尼及甲板上浪均超出线性势流理论范畴。
5. **后续研发路线建议：**
   * **第一步（有限水深三维核与多浮体窄缝共振）：** 扩展 `free_surface3d.py` 支持有限水深 John 特征级数与镜像级数，覆盖 **Garrett (1971) 截断直立圆柱**及双船并靠窄缝波面共振算例；
   * **第二步（对称面加速与完整 6-DOF RAO 静水恢复力耦合）：** 利用 `WettedHullMesh` 水线面几何自动积分静水恢复力矩阵 \(\mathbf C_{\text{hydrostatic}}\)，形成包含惯性矩阵、附加质量 \(\mathbf A(\omega)\)、辐射阻尼 \(\mathbf B(\omega)\) 与波浪激励力 \(\mathbf F_{\text{exc}}(\omega,\beta)\) 的频域 6-DOF 运动响应算子（RAO）基准；
   * **第三步（有航速 Rankine 源与二阶漂移力）：** 引入自由液面 Rankine 面元验证 **Wigley 标准船型**兴波阻力，并结合 `recovery.py` 表面切向速度恢复层构建平均二阶波浪漂移力（Maruo 远场动量法与近场压力积分对比）。

## 6. 结论

船海 BEM 的验证体系应严格遵循「数学制造解 → 无界流刚体平动/转动解析解 → 二维自由液面辐射/绕射与互易律 → 贯底柱三维分离变量绕射与不规则频率消除 → 三维自由液面穿水/潜没浮体 → 公开船模试验确认」的递进链条。本文在 TensorBEM（v0.5.0）中系统梳理了十二类船海 BEM 基准，完整实现了从二维/三维无界流 6-DOF Neumann 附加质量张量、二维 Ursell/Dean 潜没柱辐射与绕射、MacCamy–Fuchs 贯底柱绕射，直至**三维深水 Kelvin 自由液面 Green 函数、穿水浮体 6-DOF 辐射–绕射面板法、免盖板三维 CHIEF 不规则频率消除，以及 Hulme (1982) 漂浮半球与 Wang/Linton 潜没球体基准**的全套验证体系。全部基准均以不共享离散面元算子的独立闭式解/多极子级数为真值参照，通过严格 JSON CLI 与自动化回归测试验证了 \(O(h^2)\) 二阶收敛阶、良态条件数及 Kochin/Haskind 物理守恒律。

## 参考文献

[1] Newman J N. *Marine Hydrodynamics*. Cambridge, MA: MIT Press, 1977.

[2] Ursell F. On the heaving motion of a circular cylinder on the surface of a fluid. *Quarterly Journal of Mechanics and Applied Mathematics*, 1949, 2(2): 218–231. https://doi.org/10.1093/qjmam/2.2.218.

[3] John F. On the motion of floating bodies. II. Simple harmonic motions. *Communications on Pure and Applied Mathematics*, 1950, 3(1): 45–101. https://doi.org/10.1002/cpa.3160030105.

[4] Lamb H. *Hydrodynamics*. 6th ed. Cambridge: Cambridge University Press, 1932.

[5] MacCamy R C, Fuchs R A. *Wave Forces on Piles: A Diffraction Theory*. Beach Erosion Board Technical Memorandum No. 69. U.S. Army Corps of Engineers, 1954.

[6] Hulme A. The wave forces acting on a floating hemisphere undergoing forced periodic oscillations. *Journal of Fluid Mechanics*, 1982, 121: 443–463. https://doi.org/10.1017/S0022112082001930.

[7] Ancellin M, Dias F. Capytaine: a Python-based linear potential flow solver. *Journal of Open Source Software*, 2019, 4(36): 1345. https://doi.org/10.21105/joss.01345.

[8] Hobson E W. *The Theory of Spherical and Ellipsoidal Harmonics*. Cambridge: Cambridge University Press, 1931.

[9] Dean W R. On the reflexion of surface waves by a submerged circular cylinder. *Mathematical Proceedings of the Cambridge Philosophical Society*, 1948, 44(4): 483–491. https://doi.org/10.1017/S0305004100024506.

[10] Ursell F. Surface waves on deep water in the presence of a submerged circular cylinder. I. *Mathematical Proceedings of the Cambridge Philosophical Society*, 1950, 46(1): 141–152. https://doi.org/10.1017/S0305004100025561.

[11] Ogilvie T F. First- and second-order forces on a cylinder submerged under a free surface. *Journal of Fluid Mechanics*, 1963, 16(3): 451–472. https://doi.org/10.1017/S0022112063000896.

[12] Schenck H A. Improved integral formulation for acoustic radiation problems. *The Journal of the Acoustical Society of America*, 1968, 44(1): 41–58. https://doi.org/10.1121/1.1911085.

[13] Linton C M. Radiation and diffraction of water waves by a submerged sphere in finite depth. *Ocean Engineering*, 1991, 18(1–2): 61–74. https://doi.org/10.1016/0029-8018(91)90034-N.

[14] Garrett C J R. Wave forces on a circular dock. *Journal of Fluid Mechanics*, 1971, 46(1): 129–139. https://doi.org/10.1017/S0022112071000430.

[15] Newman J N. Algorithms for the free-surface Green function. *Journal of Engineering Mathematics*, 1985, 19(1): 57–67. https://doi.org/10.1007/BF00055041.

[16] Telste J G, Noblesse F. Numerical evaluation of the Green function of water-wave radiation and diffraction. *Journal of Ship Research*, 1986, 30(2): 69–84. https://doi.org/10.5957/jsr.1986.30.2.69.
