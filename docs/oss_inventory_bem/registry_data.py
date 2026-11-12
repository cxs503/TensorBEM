"""Curated content for the OSS BEM census (FEAT-OSS-INVENTORY-BEM).

Kept separate from build_registry.py so that the *judgement* (tier, transferable
value, gap mapping) is auditable independently of the *mechanics* (metadata
merge, gap probes, validation). Nothing here is a numeric claim about a solver's
accuracy; those live in registry.json only where a README quote backs them.
"""

TIER_RULES = {
    "A": "船舶/海洋工程水动力 BEM 求解器：频域线性势流或时域 Cummins，维护活跃"
         "（pushed_at <= 24 个月），可直接作为 TensorBEM 的对表基准或算法移植源。",
    "B": "可复用组件或窄域高质量实现：面元积分核/快速多极/网格与格式/时域模态/"
         "专用物理（波浪阻力、复 Green 函数）。是能力拼图的一块，不是完整对标体。",
    "C": "边缘相关：声学/电磁/弹性断裂/教学级 2D Laplace/名称误命中。"
         "登记但不进入对表清单，理由逐条写明。",
}

# ---------------------------------------------------------------------------
# Curated annotations: `offers` = what is actually transferable to TensorBEM.
# `readme_keywords` drives automatic extraction of supporting README lines.
# ---------------------------------------------------------------------------
CURATED = {
    # ------------------------------- tier A --------------------------------
    "capytaine/capytaine": {
        "tier": "A",
        "domain": "offshore / WEC / ship seakeeping，频域线性势流",
        "method_family": "Rankine 面元 BEM + Delhommeau 自由面 Green 函数，直接法求解，"
                         "线性求解器可插拔",
        "offers": [
            "Nemoh 的现代 Python 重写：同一物理模型下的独立对标实现",
            "Delhommeau 积分的高阶可调实现 + 并行装配，公开 CI 测试面可直接借为回归基准",
            "多体/多自由度公开算例与教程（对表几何现成）",
        ],
        "gaps_addressed": ["GAP-BEM-COEFF-IO", "GAP-BEM-CROSS-SOLVER-REGRESSION"],
        "license_note": "Apache-2.0 —— 可读可重实现，衍生分发保留声明",
        "readme_keywords": ["Nemoh", "frequency domain", "BEM"],
    },
    "BEMRosetta/BEMRosetta": {
        "tier": "A",
        "domain": "BEM 结果格式枢纽 / 系数查看与转换",
        "method_family": "文件格式 I/O + 网格与水静力工具 + 跨程序同几何对比",
        "offers": [
            "权威格式全集：WAMIT(.out/.1/.3/.4/.hst/.5p/.7/.8/.9/.12s/.12d/.3sc/3fk)、"
            "AQWA(.lis/.ah1/.qtf)、OrcaFlex/.owr、Nemoh/Capytaine(tec+cal)、BEMIO(.h5)、"
            "Hydrostar(.out)、OpenFAST HydroDyn.dat",
            "「同几何跨求解器对比」工作流 = 本仓 CROSS-SOLVER-REGRESSION 缺口的现成模板",
        ],
        "gaps_addressed": ["GAP-BEM-COEFF-IO", "GAP-BEM-CROSS-SOLVER-REGRESSION",
                           "GAP-BEM-REGFILE-ROUNDTRIP"],
        "license_note": "GPL-3.0 —— 只可读规范与设计，代码不得并入本仓",
        "readme_keywords": ["Wamit", "Nemoh", "formats"],
    },
    "LHEEA/Nemoh": {
        "tier": "A",
        "domain": "offshore / WEC 一阶波浪载荷（频域）",
        "method_family": "Fortran BEM，自由面 Green 函数（Delhommeau），常数面元",
        "offers": [
            "该谱系原点实现：本仓自由面 Green 函数与不规则频率处理的历史源头",
            "Nemoh.cal / Mesh / RadiationCoefficients.tec 等事实标准输入输出约定",
        ],
        "gaps_addressed": ["GAP-BEM-COEFF-IO", "GAP-BEM-REGFILE-ROUNDTRIP"],
        "license_note": "Apache-2.0；README 自述已迁移 GitLab（github 镜像自 2023-01 冻结），"
                        "引用应指向 gitlab.com/lheea/Nemoh",
        "readme_keywords": ["Boundary Element Methods", "migrated", "first order"],
    },
    "JuliaOceanWaves/MarineHydro.jl": {
        "tier": "A",
        "domain": "marine hydrodynamics 灵敏度分析（可微 BEM）",
        "method_family": "BEM + 自动微分（DifferentiationInterface）；Wu 与 Delhommeau 两种 "
                         "Green 函数；直接/间接格式",
        "offers": [
            "公开的「全可微 BEM」路线：形状灵敏度端到端反传（Khanal et al., AOR 163, 2025）",
            "同一框架内 Wu 与 Delhommeau 双 Green 函数并列 —— 本仓缺的 Green 族交叉校验范式",
        ],
        "gaps_addressed": ["GAP-BEM-ADJOINT-GEOM-GRAD", "GAP-BEM-GREEN-FAMILY-CROSSCHECK"],
        "license_note": "MIT",
        "readme_keywords": ["Differentiable", "Green functions", "Delhommeau"],
    },
    "PengYunqiang/CRESTU": {
        "tier": "A",
        "domain": "船舶水动力（3D Rankine 源 BEM）",
        "method_family": "Rankine source panel method，自由面条件分阶",
        "offers": [
            "与本仓 rankine 路线同族但独立编写的 3D Rankine 求解器 —— 差异化对标体",
            "面向船舶（非海洋平台）的算例组织方式",
        ],
        "gaps_addressed": ["GAP-BEM-CROSS-SOLVER-REGRESSION"],
        "license_note": "GPL-3.0（2026-09 建仓，活跃）",
        "readme_keywords": ["Rankine", "BEM"],
    },    # ------------------------------- tier B --------------------------------
    "xclaeys/BemTool": {
        "tier": "B",
        "domain": "C++ 面元库（capytaine 的 C++ 后端）",
        "method_family": "Delhommeau 积分 + 面元几何 + 线性求解接口",
        "offers": ["capytaine 近奇异积分精度的实际承载者 —— 复现其精度需读这里"],
        "gaps_addressed": ["GAP-BEM-ASSEMBLY-SCALING"],
        "license_note": "LGPL-3.0 —— 只可读算法",
    },
    "WEC-Sim/WEC-Sim": {
        "tier": "B",
        "domain": "WEC 时域多体仿真（BEMIO 已并入）",
        "method_family": "Cummins 时域（辐射 IRF）+ 多体约束 + 系泊耦合",
        "offers": [
            "时域多体 + 系泊耦合公开参考实现（本仓 mooring 侧无耦合求解面）",
            "BEM 系数 -> 时域模型的标准化流程",
        ],
        "gaps_addressed": ["GAP-BEM-TD-MOORING-COUPLING"],
        "license_note": "Apache-2.0",
        "readme_keywords": ["WEC-Sim", "wave energy"],
    },
    "WEC-Sim/bemio": {
        "tier": "B",
        "domain": "BEM 系数 I/O 与归一化（Python 版已停止支持）",
        "method_family": "WAMIT/Nemoh/AQWA -> 标准 .h5 归一化",
        "offers": ["系数归一化/无量纲约定公开实现（与本仓手抄对标常量的痛点直接相关）"],
        "gaps_addressed": ["GAP-BEM-COEFF-IO"],
        "license_note": "未声明（None）-> 只可读约定，不可复制代码",
        "readme_keywords": ["BEMIO"],
    },
    "JuliaOceanWaves/Hydrodynamics.jl": {
        "tier": "B",
        "domain": "时域线性势流水动力（Cummins）",
        "method_family": "辐射 IRF / 状态空间时域积分",
        "offers": ["与本仓 cummins 路线独立实现的时域对照（IRF 因果性、状态空间拟合）"],
        "gaps_addressed": ["GAP-BEM-TD-MOORING-COUPLING"],
        "license_note": "MIT",
        "readme_keywords": ["Time domain", "linear potential flow"],
    },
    "bempp/bempp-cl": {
        "tier": "B",
        "domain": "通用 BEM 库（Laplace/Helmholtz/修正 Helmholtz/Maxwell）",
        "method_family": "高阶面元 + 奇异积分精确化 + JIT 核函数",
        "offers": [
            "奇异/近奇异积分的高阶处理参考（对照本仓 near_singular 侧）",
            "Helmholtz 核装配约定 —— 支撑本仓 FMM 的 Helmholtz 缺口",
        ],
        "gaps_addressed": ["GAP-BEM-FMM-HELMHOLTZ-KERNEL"],
        "license_note": "MIT",
        "readme_keywords": ["Helmholtz", "integral kernels"],
    },
    "exafmm/exafmm-t": {
        "tier": "B",
        "domain": "kernel-independent FMM 库（C++/Python）",
        "method_family": "FMM：Laplace/Helmholtz/Elastic 核，blocked + diagonal translation",
        "offers": [
            "本仓 fmm.py 仅 Laplace、Helmholtz 超 ka_max 即 NOT_SUPPORTED —— 此处是补齐路线",
            "JOSS 6(61):3145 (2021) 公开误差/性能数据可做加速比判据",
        ],
        "gaps_addressed": ["GAP-BEM-FMM-HELMHOLTZ-KERNEL", "GAP-BEM-ASSEMBLY-SCALING"],
        "license_note": "BSD-3-Clause",
        "readme_keywords": ["kernel-independent", "fast multipole"],
    },
    "mathLab/pi-BEM": {
        "tier": "B",
        "domain": "并行 BEM 求解器（低秩压缩方向）",
        "method_family": "parallel BEM assembly + H-matrix/ACA 压缩",
        "offers": ["O(N^2) 装配的分块/低秩工程实践（本仓装配无分块）"],
        "gaps_addressed": ["GAP-BEM-ASSEMBLY-SCALING"],
        "license_note": "LGPL-2.1",
        "readme_keywords": ["Parallel", "Boundary Element"],
    },
    "ZeppSav/BEMUse": {
        "tier": "B",
        "domain": "轻量 BEM 库（水动力 + 声学）",
        "method_family": "常数面元 BEM，DSL 式问题定义",
        "offers": ["小体量高可读库结构（适合做教学/回归小例）"],
        "gaps_addressed": ["GAP-BEM-CROSS-SOLVER-REGRESSION"],
        "license_note": "GPL-3.0",
        "readme_keywords": ["boundary element", "hydrodynamics"],
    },
    "Savin-Viswanathan/OMHyD-PA": {
        "tier": "B",
        "domain": "offshore hydrodynamics 面元分析",
        "method_family": "panel method（论文随附算例）",
        "offers": ["论文随附输入算例 —— 可作第三方几何/频率扫描基准"],
        "gaps_addressed": ["GAP-BEM-CROSS-SOLVER-REGRESSION"],
        "license_note": "BSD-3-Clause",
        "readme_keywords": ["OMHyD"],
    },
    "shash-sharma/OpenBEM": {
        "tier": "B",
        "domain": "模块化 BEM 框架（通用）",
        "method_family": "可插拔单元类型/核函数的 BEM 框架",
        "offers": ["单元/核/装配解耦的框架抽象（对本仓面元族可扩展性有借鉴）"],
        "gaps_addressed": ["GAP-BEM-ASSEMBLY-SCALING"],
        "license_note": "NOASSERTION -> 需人工判定后方可使用",
        "readme_keywords": ["boundary element"],
    },
    "krcools/BEAST.jl": {
        "tier": "B",
        "domain": "BEM 工具链（声学/电磁为主，含势流核）",
        "method_family": "BEM + 高阶基函数 + 快速算法",
        "offers": ["高阶基函数与奇异积分处理的活跃实现（2026-11 有提交）"],
        "gaps_addressed": ["GAP-BEM-FMM-HELMHOLTZ-KERNEL"],
        "license_note": "NOASSERTION -> 需人工判定",
    },
    "matthieuaussal/gypsilab": {
        "tier": "B",
        "domain": "MATLAB 快速 BEM 工具箱",
        "method_family": "FMM / H-matrix 驱动的通用 BEM",
        "offers": ["快速算法与 BEM 组合的完整教学式实现"],
        "gaps_addressed": ["GAP-BEM-ASSEMBLY-SCALING"],
        "license_note": "未声明",
    },
    "Applied-Scientific-Research/Omega2D": {
        "tier": "B",
        "domain": "2D 势流/涡粒子流动求解（含面元）",
        "method_family": "vortex particle + panel method，2D",
        "offers": ["2D 势流与涡动力学耦合视角（对照本仓 2D 面元）"],
        "gaps_addressed": [],
        "license_note": "GPL-3.0",
    },
    "AzadSaeed/Parallel-Nemoh": {
        "tier": "B",
        "domain": "Nemoh 批量作业并行（作业级，非求解器内核并行）",
        "method_family": "job-level parallel wrapper over Nemoh + MATLAB wrappers",
        "offers": ["参数扫描的作业级并行范式（对「>1000 工况无人值守」有参考）"],
        "gaps_addressed": [],
        "license_note": "未声明",
        "readme_keywords": ["parallel implementation", "Matlab"],
        "annotation_caveat": "README 明言不修改 Nemoh 内核 -> 不得据此声称求解器并行度。",
    },    "krcools/CompScienceMeshes.jl": {
        "tier": "B",
        "domain": "BEM/FEM 的几何与网格基础（Julia）",
        "method_family": "simplicial mesh + 面元几何量/奇异积分几何支撑",
        "offers": ["面元几何量与网格一致性检查的独立实现（对照本仓 panel geometry validation）"],
        "gaps_addressed": [],
        "license_note": "NOASSERTION -> 需人工判定",
    },
    "bempp/bempp-rs": {
        "tier": "B",
        "domain": "通用 BEM 库（Rust 重写，同 BEM++ 家族）",
        "method_family": "Rust 高性能 BEM 装配 + 核函数求值",
        "offers": ["同家族第二语言实现：装配性能工程（内存布局/并行）的可比参照"],
        "gaps_addressed": ["GAP-BEM-ASSEMBLY-SCALING"],
        "license_note": "BSD-3-Clause",
    },
    "mipals/BoundaryIntegralEquations.jl": {
        "tier": "B",
        "domain": "边界积分方程教学/实用实现（Julia）",
        "method_family": "BIE 离散 + 奇异积分 + 快速算法入门",
        "offers": ["可读性极高的 BIE 全流程（含奇异积分处理），适合做对照阅读"],
        "gaps_addressed": ["GAP-BEM-CROSS-SOLVER-REGRESSION"],
        "license_note": "MIT",
    },
    # ------------------------------- tier C --------------------------------
    "HomerReid/scuff-em": {
        "tier": "C", "domain": "计算电磁 BEM（含分层介质 Green 函数）",
        "method_family": "surface integral equation, EM",
        "exclude_reason": "Maxwell 核而非势流核；分层 Green 函数思想可借但物理不同",
    },
    "fjargsto/abem": {
        "tier": "C", "domain": "声学 BEM", "method_family": "Helmholtz BEM",
        "exclude_reason": "声学域；本仓已有 acoustic3d/helmholtz 模块，无新增价值",
    },
    "Timmmdavis/CutAndDisplace": {
        "tier": "C", "domain": "断层/形变（弹性静力学 BEM）",
        "method_family": "displacement discontinuity BEM",
        "exclude_reason": "固体弹性静力学，与势流水动力无共享核",
    },
    "wcs211/BEM-3D-Python": {
        "tier": "C", "domain": "教学级 3D Laplace BEM", "method_family": "constant panel Laplace",
        "exclude_reason": "教学代码、无自由面、无验证基准，2015 后无提交",
    },
    "jerryjappinen/scroll-scope": {
        "tier": "C", "domain": "前端 JS 插件", "method_family": "n/a",
        "exclude_reason": "描述含 'element' 的名称误命中，与 BEM 无关",
    },}

# ---------------------------------------------------------------------------
# GAP-BEM-* anchors. Each carries an executed probe so absence is measured.
# ---------------------------------------------------------------------------
ANCHORS = [
    {
        "slug": "GAP-BEM-COEFF-IO",
        "title": "BEM 系数文件互通层（WAMIT .1/.3 + Nemoh tec 读写，终结手抄对标常量）",
        "priority": "P0",
        "tensor_bem_gap":
            "无任何第三方 BEM 系数格式读写；对标参考值以硬编码常量存在"
            "（benchmarks/wigley3_complete_verification.py 的 _WAMIT_A33_MASS_RATIO 等），"
            "新增基准只能手抄，违反零手抄纪律。",
        "probe_terms": ["nemoh", "bemrosetta", "aqwa", "12d", "hydrostar"],
        "source_repos": ["BEMRosetta/BEMRosetta", "LHEEA/Nemoh", "WEC-Sim/bemio",
                         "capytaine/capytaine"],
        "transferable_method":
            "BEMRosetta README 给出权威格式全集与跨求解器对比流程；Nemoh 的 "
            "Nemoh.cal/Mesh/RadiationCoefficients.tec 是该谱系事实标准。按公开描述实现"
            "列式与无量纲约定，不复制 GPL 代码。",
        "acceptance": [
            "读写 round-trip：写出->读回 A/B/F 相对误差 < 1e-12（I/O 无损性）",
            "外部文件解析：capytaine 标准 .1/.3 与 Nemoh tec 文件，A33/B33 相对误差 < 1e-6",
            "导出文件可被 BEMRosetta 解析（一次性人工核验，记录版本号于报告）",
            "原硬编码基准改为文件注入，Wigley 验证误差不劣化（变化 < 1%）",
        ],
        "quantified_target": "I/O <1e-12；外部解析 <1e-6；现有验证劣化 <1%",
        "effort": "M（约 1.5 单）",
    },
    {
        "slug": "GAP-BEM-ADJOINT-GEOM-GRAD",
        "title": "面元几何灵敏度（伴随/自动微分）：d(A,B)/d(节点坐标)",
        "priority": "P1",
        "tensor_bem_gap":
            "无几何形状梯度能力（src 中 adjoint/differentiable 命中均非几何灵敏度语境），"
            "故 BEM 只能分析、无法进入船型/布置优化回路。",
        "probe_terms": ["adjoint", "differentiab", "shape.optim", "design.sensitiv"],
        "probe_reading":
            "adjoint 3 命中均为【算子伴随】语境（finite_depth3d_quadrature.py:473 入射/伴随核；pekeris_acoustics.py:15 自伴随性），differentiab/shape.optim/design.sensitiv 0 命中 -> 设计梯度能力确证缺席，状态 verified_absent。",
        "source_repos": ["JuliaOceanWaves/MarineHydro.jl",
                         ],
        "transferable_method":
            "MarineHydro.jl 证明「problem+formulation->solve->实系数」的窄接口可整体交给 "
            "DifferentiationInterface 反传；伴随路线见 BEMRosetta 生态的伴随型兴波阻力实现（见报告「未验证项」）。"
            "本仓做法：求解器纯函数化，配有限差分校验的伴随或 autograd 包装。",
        "acceptance": [
            "椭球与 Wigley 两例，dA33/dx_k 与中心差分（步长 1e-3/1e-4/1e-5 m）相对误差 < 1e-3",
            "梯度方向余弦 > 0.999（对中心差分参照）",
            "单次梯度墙钟 < 3x 前向求解（AD/伴随盈亏线）",
        ],
        "quantified_target": "梯度误差 <1e-3；方向余弦 >0.999；成本 <3x 前向",
        "effort": "L（约 3 单）",
    },
    {
        "slug": "GAP-BEM-FMM-HELMHOLTZ-KERNEL",
        "title": "Helmholtz FMM 的高 ka 区间（当前 ka>10 退回直接求和 = O(N^2)）",
        "priority": "P1",
        "tensor_bem_gap":
            "src/tensorbem/fmm.py 的 fmm_helmholtz_matvec 已实现 Helmholtz 核的 FMM，但 "
            "helmholtz_supported(ka, ka_max=10.0) 为 fail-closed：ka>10 时直接跳回 "
            "_helmholtz_direct 求和（NOT_SUPPORTED 语义）-> 中高频散射/短波问题实际无加速。"
            "缺口是【高 ka 传递算子】，不是整个 Helmholtz 能力。",
        "probe_terms": ["ka_max", "helmholtz_supported"],
        "probe_reading":
            "ka_max 命中 5 文件：fmm.py 是能力本体（含 ka_max=10 阈值），其余 4 个 "
            "benchmarks/kkr_* 与 cummins_memory_closure 为同名无关参数 -> 读作"
            "「能力存在但有明确 ka 上限」，故状态为 partially_present。",
        "source_repos": ["exafmm/exafmm-t", "bempp/bempp-cl", "krcools/BEAST.jl"],
        "transferable_method":
            "exafmm-t 的 kernel-independent FMM（blocked + diagonal translation，JOSS 2021）"
            "在高频段用对角传递算子替代稠密传递矩阵，避免 ka 增大时 p 指数膨胀；"
            "bempp-cl 给出 Helmholtz 核面元装配约定。本仓只需替换级数展开与传递算子的高频分支。",
        "acceptance": [
            "ka ∈ {25, 50, 100} 三档仍需加速：N=4096 单层 matvec 相对误差 < 1e-6（vs 直接求和）",
            "同规模加速比 >= 2x（当前 ka>10 时加速比 = 1.0x，作为基线数字必须成表）",
            "ka ∈ {1,5,10,25,50} 五档扫描（误差/墙钟/是否回退）入库，回退档必须显式标注",
            "原有 Laplace FMM 与 ka<=10 Helmholtz 测试全绿（不回归）",
        ],
        "quantified_target": "ka>=25 仍 <1e-6 且加速 >=2x；5 档扫描表无静默回退",
        "effort": "L（约 3 单）",
    },
    {
        "slug": "GAP-BEM-ASSEMBLY-SCALING",
        "title": "面元装配可扩展性（分块 BLAS3 + 低秩），保持结果严格等价",
        "priority": "P2",
        "tensor_bem_gap":
            "装配为朴素 O(N^2) 逐面元循环，无分块/无 H-matrix/ACA（fmm.py 的八叉树是 FMM 内部结构，不是面元装配分块）；N>=8192 时墙钟与内存"
            "成瓶颈（条件数诊断已有，规模化没有）。",
        "probe_terms": ["\\bACA\\b", "h-matrix", "\\bTLD\\b", "low-rank"],
        "source_repos": ["mathLab/pi-BEM", "exafmm/exafmm-t", "matthieuaussal/gypsilab",
                         "shash-sharma/OpenBEM"],
        "transferable_method":
            "pi-BEM/gypsilab 的分块与低秩压缩组织（TLD/ACA）；先做纯分块 BLAS3（改动小、"
            "等价性强），再按需上低秩。",
        "acceptance": [
            "N=8192 单核装配墙钟较基线下降 >= 3x",
            "A/B/F 与基线相对偏差 < 1e-10（严格等价，不许用精度换速度）",
            "峰值内存增长 <= 10%",
            "基线与改造后数字同机同参数成表（禁跨机比较）",
        ],
        "quantified_target": "墙钟 >=3x；系数偏差 <1e-10；内存 <=+10%",
        "effort": "M（约 1.5 单）",
    },
    {
        "slug": "GAP-BEM-TD-MOORING-COUPLING",
        "title": "时域 Cummins 与系泊/多体约束耦合（WEC-Sim 路线）",
        "priority": "P2",
        "tensor_bem_gap":
            "时域 Cummins 已有，但系泊仅以【一个硬编码刚度常量】出现在 OC4 慢漂算例"
            "（benchmarks/qtf_drift_study.py: stiffness=1.0e5 N/m, catenary surge stiffness），"
            "没有系泊动力学模型、没有多体约束耦合求解面 -> 无法做 WEC/浮式整机响应。",
        "probe_terms": ["moordyn", "fairlead", "\\bcatenary\\b", "mooring.stiffness"],
        "probe_reading":
            "仅 catenary 与 mooring.stiffness 各 1 命中，且同在 qtf_drift_study.py 的参考"
            "常量里；moordyn/fairlead 0 命中 -> 读作「只有常数项、无耦合求解」，状态 partially_present。",
        "source_repos": ["WEC-Sim/WEC-Sim", "JuliaOceanWaves/Hydrodynamics.jl", "WEC-Sim/bemio"],
        "transferable_method":
            "WEC-Sim 的 Cummins 时域 + 约束/系泊分块耦合（辐射 IRF 卷积用状态空间替代提速）；"
            "Hydrodynamics.jl 给出 IRF 与状态空间拟合的独立实现。系泊侧先上线性/拟静态刚度即可闭环。",
        "acceptance": [
            "单/双体 heave-pitch 时域 RAO 与 WEC-Sim 同输入参考偏差 < 5%",
            "零系泊刚度极限退化为已有自由体解，与频域 IRF 一致性 < 1e-6",
            "系泊刚度扫 3 档，固有周期漂移方向与解析估计一致（单调性检查）",
            "原硬编码 1.0e5 N/m 改为输入参数，OC4 慢漂算例复现误差不劣化（变化 <1%）",
        ],
        "quantified_target": "RAO 偏差 <5%；自由体极限一致性 <1e-6；OC4 复现劣化 <1%",
        "effort": "L（约 3 单）",
    },
    {
        "slug": "GAP-BEM-GREEN-FAMILY-CROSSCHECK",
        "title": "Green 函数族交叉校验（Delhommeau vs Wu vs 复 Green 函数）",
        "priority": "P2",
        "tensor_bem_gap":
            "自由面 Green 函数仅单一实现路线（finite_depth_green/free_surface_green），"
            "缺同物理的第二条独立路线，误差无法归因（离散 vs 核）。",
        "probe_terms": ["delhommeau", "noblesse"],
        "probe_reading":
            "free_surface3d.py:41 一行文献引用同时点名 Newman(1985)/Telste & Noblesse(1986)/Delhommeau(1989)，即多族 Green 函数只在注释里并列，代码只有一条实现路线 -> 状态 partially_present（需人工确认是否 == 单一 Delhommeau 实现）。",
        "source_repos": ["JuliaOceanWaves/MarineHydro.jl",
                         ],
        "transferable_method":
            "MarineHydro.jl 在同一 BEM 框架内并列 Wu 与 Delhommeau 两种 Green 函数，"
            "证明「同物理双实现」是可行校验范式；复 Green 函数用于有限水深/定常兴波。",
        "acceptance": [
            "两路线在 kh ∈ {0.5,1,2,3,5} x 4 频率共 20 点上 A/B 相对偏差 < 1e-4",
            "近场（面元间距 < 0.1 特征长度）逐点偏差表入库，最大点误差单列",
            "超判据时须有归因结论（级数截断 vs 奇异积分处理）",
        ],
        "quantified_target": "20 点最大偏差 <1e-4；近场偏差单列",
        "effort": "M（约 1.5 单）",
    },
    {
        "slug": "GAP-BEM-CROSS-SOLVER-REGRESSION",
        "title": "跨求解器自动回归（>=3 体 x >=8 频率，误差表入库）",
        "priority": "P1",
        "tensor_bem_gap":
            "对标以单点硬编码常量为主，无自动跨体/跨频率回归与误差表；新能力无法被机器"
            "判定是否劣化。",
        "probe_terms": ["cross_solver", "error_table"],
        "source_repos": ["capytaine/capytaine", "BEMRosetta/BEMRosetta", "PengYunqiang/CRESTU",
                         "Savin-Viswanathan/OMHyD-PA", "ZeppSav/BEMUse"],
        "transferable_method":
            "capytaine CI 的解析解回归 + BEMRosetta 的同几何跨程序对比工作流；本仓只需"
            "「读参考文件->逐点比较->出表->卡门」的薄壳。",
        "acceptance": [
            "椭球/圆柱/Wigley 三体 x 8 频率点的 A/B 误差表（CSV + JSON）自动生成",
            "最大相对误差 < 3%（A33、B33 各一列），逐点值单列不得只报均值",
            "回归壳在 CI 轻量档 < 90 s 跑完（不做全量扫频）",
            "失败时给出首个越界点（体/频率/自由度）与数值",
        ],
        "quantified_target": "max 相对误差 <3%；CI <90 s；三体 x 8 频点",
        "effort": "M（约 1 单）",
    },
    {
        "slug": "GAP-BEM-REGFILE-ROUNDTRIP",
        "title": "Nemoh 输入/输出约定兼容层（.cal + Mesh + tec）",
        "priority": "P3",
        "tensor_bem_gap":
            "Nemoh 约定（Nemoh.cal、Mesh/*.dat、tec 列式）完全缺失，既有 Nemoh 算例无法"
            "直接迁入做同题对照。",
        "probe_terms": ["nemoh.cal", "radiationcoefficients"],
        "source_repos": ["LHEEA/Nemoh", "BEMRosetta/BEMRosetta", "AzadSaeed/Parallel-Nemoh"],
        "transferable_method":
            "Nemoh README + BEMRosetta 的 Nemoh 读写描述；按公开约定实现解析器，不复制 GPL 代码。",
        "acceptance": [
            "解析一个 Nemoh 算例目录（.cal + Mesh）后几何自检通过（体积/湿表面误差 < 1e-8）",
            "tec 列数与无量纲约定与 Nemoh 一致（逐列比对 1 个公开算例，< 1e-6）",
            "与 GAP-BEM-COEFF-IO 共享 I/O 骨架（不得重复实现两套）",
        ],
        "quantified_target": "几何自检 <1e-8；tec 列比对 <1e-6",
        "effort": "S（约 0.5 单，依赖 COEFF-IO）",
    },
]