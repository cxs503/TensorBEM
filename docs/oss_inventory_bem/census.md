# OSS BEM 技术盘点（wave 1）— FEAT-OSS-INVENTORY-BEM

生成器：`docs/oss_inventory_bem/build_registry.py`（本文件禁止手改，registry sha256[:16]=`161faa039f81b818`，时间 2026-11-12T14:19:40Z）

- 候选扫描：**82** 仓（关键词搜索 10 条 + 种子直查 12 个，其中 5 个 404 已排除）
- 登记并定级：**22**（A=5 / B=17），筛查剔除 C 档 5 个
- README 指纹：14 个仓（原文留存 raw/readme/）

## 定级规则

- **A**：船舶/海洋工程水动力 BEM 求解器：频域线性势流或时域 Cummins，维护活跃（pushed_at <= 24 个月），可直接作为 TensorBEM 的对表基准或算法移植源。
- **B**：可复用组件或窄域高质量实现：面元积分核/快速多极/网格与格式/时域模态/专用物理（波浪阻力、复 Green 函数）。是能力拼图的一块，不是完整对标体。
- **C**：边缘相关：声学/电磁/弹性断裂/教学级 2D Laplace/名称误命中。登记但不进入对表清单，理由逐条写明。

## A/B 档登记表

| Tier | 仓库 | ★ | 许可证 | 最近推送 | 域 | 可迁移到 TensorBEM 的东西 |
|---|---|---|---|---|---|---|
| A | [capytaine/capytaine](https://github.com/capytaine/capytaine) | 235 | Apache-2.0 | 2026-11-10 | offshore / WEC / ship seakeeping，频域线性势流 | Nemoh 的现代 Python 重写：同一物理模型下的独立对标实现 |
| A | [BEMRosetta/BEMRosetta](https://github.com/BEMRosetta/BEMRosetta) | 123 | GPL-3.0 | 2026-10-28 | BEM 结果格式枢纽 / 系数查看与转换 | 权威格式全集：WAMIT(.out/.1/.3/.4/.hst/.5p/.7/.8/.9/.12s/.12d/.3sc/3fk)、AQWA(.lis/.ah1/.qtf)、OrcaFlex/.owr、Nemoh/Capytaine(tec+cal)、BEMIO(.h5)、Hydrostar(.out)、OpenFAST HydroDyn.dat |
| A | [LHEEA/Nemoh](https://github.com/LHEEA/Nemoh) | 65 | Apache-2.0 | 2023-01-26 | offshore / WEC 一阶波浪载荷（频域） | 该谱系原点实现：本仓自由面 Green 函数与不规则频率处理的历史源头 |
| A | [JuliaOceanWaves/MarineHydro.jl](https://github.com/JuliaOceanWaves/MarineHydro.jl) | 33 | MIT | 2026-10-29 | marine hydrodynamics 灵敏度分析（可微 BEM） | 公开的「全可微 BEM」路线：形状灵敏度端到端反传（Khanal et al., AOR 163, 2025） |
| A | [PengYunqiang/CRESTU](https://github.com/PengYunqiang/CRESTU) | 1 | GPL-3.0 | 2026-09-08 | 船舶水动力（3D Rankine 源 BEM） | 与本仓 rankine 路线同族但独立编写的 3D Rankine 求解器 —— 差异化对标体 |
| B | [xclaeys/BemTool](https://github.com/xclaeys/BemTool) | 21 | LGPL-3.0 | 2021-10-28 | C++ 面元库（capytaine 的 C++ 后端） | capytaine 近奇异积分精度的实际承载者 —— 复现其精度需读这里 |
| B | [WEC-Sim/WEC-Sim](https://github.com/WEC-Sim/WEC-Sim) | 196 | Apache-2.0 | 2026-10-12 | WEC 时域多体仿真（BEMIO 已并入） | 时域多体 + 系泊耦合公开参考实现（本仓 mooring 侧无耦合求解面） |
| B | [WEC-Sim/bemio](https://github.com/WEC-Sim/bemio) | 28 | None | 2023-07-12 | BEM 系数 I/O 与归一化（Python 版已停止支持） | 系数归一化/无量纲约定公开实现（与本仓手抄对标常量的痛点直接相关） |
| B | [JuliaOceanWaves/Hydrodynamics.jl](https://github.com/JuliaOceanWaves/Hydrodynamics.jl) | 1 | MIT | 2026-09-10 | 时域线性势流水动力（Cummins） | 与本仓 cummins 路线独立实现的时域对照（IRF 因果性、状态空间拟合） |
| B | [bempp/bempp-cl](https://github.com/bempp/bempp-cl) | 229 | MIT | 2026-09-21 | 通用 BEM 库（Laplace/Helmholtz/修正 Helmholtz/Maxwell） | 奇异/近奇异积分的高阶处理参考（对照本仓 near_singular 侧） |
| B | [exafmm/exafmm-t](https://github.com/exafmm/exafmm-t) | 83 | BSD-3-Clause | 2021-09-09 | kernel-independent FMM 库（C++/Python） | 本仓 fmm.py 仅 Laplace、Helmholtz 超 ka_max 即 NOT_SUPPORTED —— 此处是补齐路线 |
| B | [mathLab/pi-BEM](https://github.com/mathLab/pi-BEM) | 72 | LGPL-2.1 | 2026-10-20 | 并行 BEM 求解器（低秩压缩方向） | O(N^2) 装配的分块/低秩工程实践（本仓装配无分块） |
| B | [ZeppSav/BEMUse](https://github.com/ZeppSav/BEMUse) | 17 | GPL-3.0 | 2025-07-15 | 轻量 BEM 库（水动力 + 声学） | 小体量高可读库结构（适合做教学/回归小例） |
| B | [Savin-Viswanathan/OMHyD-PA](https://github.com/Savin-Viswanathan/OMHyD-PA) | 17 | BSD-3-Clause | 2022-09-23 | offshore hydrodynamics 面元分析 | 论文随附输入算例 —— 可作第三方几何/频率扫描基准 |
| B | [shash-sharma/OpenBEM](https://github.com/shash-sharma/OpenBEM) | 20 | NOASSERTION | 2026-10-06 | 模块化 BEM 框架（通用） | 单元/核/装配解耦的框架抽象（对本仓面元族可扩展性有借鉴） |
| B | [krcools/BEAST.jl](https://github.com/krcools/BEAST.jl) | 96 | NOASSERTION | 2026-11-04 | BEM 工具链（声学/电磁为主，含势流核） | 高阶基函数与奇异积分处理的活跃实现（2026-11 有提交） |
| B | [matthieuaussal/gypsilab](https://github.com/matthieuaussal/gypsilab) | 69 | None | 2021-09-17 | MATLAB 快速 BEM 工具箱 | 快速算法与 BEM 组合的完整教学式实现 |
| B | [Applied-Scientific-Research/Omega2D](https://github.com/Applied-Scientific-Research/Omega2D) | 36 | GPL-3.0 | 2025-02-15 | 2D 势流/涡粒子流动求解（含面元） | 2D 势流与涡动力学耦合视角（对照本仓 2D 面元） |
| B | [AzadSaeed/Parallel-Nemoh](https://github.com/AzadSaeed/Parallel-Nemoh) | 2 | None | 2023-08-24 | Nemoh 批量作业并行（作业级，非求解器内核并行） | 参数扫描的作业级并行范式（对「>1000 工况无人值守」有参考） |
| B | [krcools/CompScienceMeshes.jl](https://github.com/krcools/CompScienceMeshes.jl) | 34 | NOASSERTION | 2026-11-03 | BEM/FEM 的几何与网格基础（Julia） | 面元几何量与网格一致性检查的独立实现（对照本仓 panel geometry validation） |
| B | [bempp/bempp-rs](https://github.com/bempp/bempp-rs) | 39 | BSD-3-Clause | 2025-08-04 | 通用 BEM 库（Rust 重写，同 BEM++ 家族） | 同家族第二语言实现：装配性能工程（内存布局/并行）的可比参照 |
| B | [mipals/BoundaryIntegralEquations.jl](https://github.com/mipals/BoundaryIntegralEquations.jl) | 27 | MIT | 2026-04-09 | 边界积分方程教学/实用实现（Julia） | 可读性极高的 BIE 全流程（含奇异积分处理），适合做对照阅读 |

## C 档筛查记录（登记但不对标）

| 仓库 | ★ | 排除理由 |
|---|---|---|
| [HomerReid/scuff-em](https://github.com/HomerReid/scuff-em) | 172 | Maxwell 核而非势流核；分层 Green 函数思想可借但物理不同 |
| [fjargsto/abem](https://github.com/fjargsto/abem) | 19 | 声学域；本仓已有 acoustic3d/helmholtz 模块，无新增价值 |
| [Timmmdavis/CutAndDisplace](https://github.com/Timmmdavis/CutAndDisplace) | 63 | 固体弹性静力学，与势流水动力无共享核 |
| [wcs211/BEM-3D-Python](https://github.com/wcs211/BEM-3D-Python) | 34 | 教学代码、无自由面、无验证基准，2015 后无提交 |
| [jerryjappinen/scroll-scope](https://github.com/jerryjappinen/scroll-scope) | 105 | 描述含 'element' 的名称误命中，与 BEM 无关 |

## GAP-BEM-* 锚点（缺口由执行探针验证，非断言）

| 锚点 | 优先级 | 探针命中 | 状态 | 量化验收 | 工作量 |
|---|---|---|---|---|---|
| `GAP-BEM-COEFF-IO` | P0 | 0 | verified_absent | I/O <1e-12；外部解析 <1e-6；现有验证劣化 <1% | M（约 1.5 单） |
| `GAP-BEM-ADJOINT-GEOM-GRAD` | P1 | 3 | partially_present_review_required | 梯度误差 <1e-3；方向余弦 >0.999；成本 <3x 前向 | L（约 3 单） |
| `GAP-BEM-FMM-HELMHOLTZ-KERNEL` | P1 | 12 | partially_present_review_required | ka>=25 仍 <1e-6 且加速 >=2x；5 档扫描表无静默回退 | L（约 3 单） |
| `GAP-BEM-ASSEMBLY-SCALING` | P2 | 0 | verified_absent | 墙钟 >=3x；系数偏差 <1e-10；内存 <=+10% | M（约 1.5 单） |
| `GAP-BEM-TD-MOORING-COUPLING` | P2 | 2 | partially_present_review_required | RAO 偏差 <5%；自由体极限一致性 <1e-6；OC4 复现劣化 <1% | L（约 3 单） |
| `GAP-BEM-GREEN-FAMILY-CROSSCHECK` | P2 | 4 | partially_present_review_required | 20 点最大偏差 <1e-4；近场偏差单列 | M（约 1.5 单） |
| `GAP-BEM-CROSS-SOLVER-REGRESSION` | P1 | 2 | partially_present_review_required | max 相对误差 <3%；CI <90 s；三体 x 8 频点 | M（约 1 单） |
| `GAP-BEM-REGFILE-ROUNDTRIP` | P3 | 0 | verified_absent | 几何自检 <1e-8；tec 列比对 <1e-6 | S（约 0.5 单，依赖 COEFF-IO） |

> 探针命中数 = 在 `src/` + `tests/` 上执行 terms 正则命中的文件数；0 = 该能力在代码库中确实缺席（可在 registry.json 的 `gap_probe` 里复核）。


## 复现方式

```bash
python docs/oss_inventory_bem/collect_oss_bem.py   # 采集（需网络，限速 10/min）
python docs/oss_inventory_bem/fetch_readmes.py     # README 取证
python docs/oss_inventory_bem/build_registry.py    # 定级+探针+校验（离线可跑）
```

