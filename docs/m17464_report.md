# CASE-BEM-RIGID-LID-HEMISPHERE（TensorBEM）返修轮：刚性盖半球附加质量 ka→0，ref vs got 逐项对照

## ① 案例介绍（物理背景 / 几何 / 参数表）

半潜刚性盖半球（平底贴 z=0 刚性自由面/镜像面）在 heave 方向的**零频率附加质量**。
物理本质：ka→0 时自由面条件退化为刚性壁 ∂φ/∂z=0，用镜像法把半球映成整球——
但**竖直平动体在对称面内的镜像速度反向**，双体携带的是滑移 Neumann 数据 ∂φ/∂r=−U|cosθ|（仅偶 Legendre 模），不是整球平动的 −U cosθ。
因此 ka→0 端点 A33/(ρ∇) = 0.8309513，而**不是**「映像整球平动」读法给出的 1.0 ρ∇；后者曾让报告 #15809 误报 14.4% honest-FAIL（ERRATUM 已订正）。

| 参数 | 值 |
|---|---|
| 半径 a | 1.000 m |
| 密度 ρ | 1025.0 kg/m³ |
| 重力 g | 9.80665 m/s² |
| 排水体积 ∇=2/3 πa³ | 2.094395 m³ |
| ρ∇ | 1783.8488 kg |
| 无量纲频率 ka 带 | 1e-03, 1e-02, 3e-02, 1e-01 |
| 网格 | 64 / 256 常数面元（icosphere 细分 2/3） |
| 物理模型 | 无粘、线性势流、零航速，exp(−iωt)，单位速度幅值 |

**ref 出处**：不是手抄常数，而是运行时求值的偶宇称双体 Legendre 级数，三条独立路线：
(i) `tensorbem.rigid_lid` 用 `fractions.Fraction` 在单项式基上**精确有理数**积分 J_n=∫₀¹ μP_n dμ 再求和（无 |μ| 断裂点的求积误差、无浮点截断误差）；(ii)(iii) 算例驱动的动能恒等式路线与压强积分路线；外加 Hulme (1982) J. Fluid Mech. 121:443–463 多极解向 ka→0 外推作旁证。

| 路线 | A33/(ρ∇) | A33 [kg] |
|---|---|---|
| 精确有理数（裁判） | 0.830951320 | 1783.8489 |
| 动能恒等式 | 0.830951287 | 1783.8488 |
| 压强积分 | 0.830954148 | 1783.8550 |
| 压力释放 ka→∞（奇宇称单模，控制） | 0.5000000 | — |
| 映像整球（被否定的负控制） | 1.0000000 | — |

三路线一致性：exact vs energy `3.91e-08`（门 1e-06），force vs energy `3.44e-06`（门 1e-04，差值定位为 |μ| 拐点的高斯求积误差）；Hulme 外推 `0.8309772`（gap `3.11e-05`）。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **前置几何门**：面元网格的湿表面积/体积/水线面积对闭式解打门（体积误差 256 面元 `-2.320%`，另设细分收缩门），且附加质量**始终用精确 ∇ 归一**，不吃网格体积亏欠。
2. **求解**：`tensorbem.free_surface_solver3d.solve_radiation_3d`（深水 Kelvin 格林函数，ka→0 退化为镜像 Rankine 核；常数面元 + 精确 Van Oosterom–Strackee 立体角 + CHIEF）。
3. **裁判**：本次新增 `src/tensorbem/rigid_lid.py`（精确有理数偶宇称级数，并把压力释放 0.5 作为同一机器上的控制量），算例驱动把它作为第三条参考路线与门 G3b。
4. **集成调用面**：主求解器模块导出 `free_surface_solver3d.rigid_lid_heave_added_mass_dimless()` 与 `free_surface_solver3d.verify_rigid_lid_heave_limit(level, ka)`（现场求解并打分），`tensorbem.benchmarks.__init__` 导入算例模块，`capability_evidence` 登记 `case-bem-rigid-lid-hemisphere`（REAL）。
5. **预注册判据**（先定后跑）：G0 几何、G1 带内 |err|<5%%（逐 ka）、G2 细分收敛、G3/G3b 参考自一致、G4 Hulme 旁证 <1%%、G5 负控制 <0.95、G6 带外扫描**必须**破 5%%。
6. **一键复现**：`bash scripts/repro_case_bem_rigid_lid_hemisphere.sh`（驱动 --check → 从 payload 独立重算判决 → 跑算例测试）。本次实跑 exit=0，见 `runs/m17464_rigid_lid_hemisphere/repro.log`。

## ③ 计算结果（数字表，程序化注入）

`err = A33/(ρ∇)/ref − 1`，ref = 0.830951287，池容差 5%。

| 面元 | ka | ref A33/(ρ∇) | got A33/(ρ∇) | err | <3% | <5% | err vs Hulme | err vs 映像负控制 | cond |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 1e-03 | 0.8309513 | 0.8073879 | -2.836% | ✓ | ✓ | -3.314% | -19.26% | 2.18 |
| 64 | 1e-02 | 0.8309513 | 0.8253131 | -0.679% | ✓ | ✓ | -3.319% | -17.47% | 2.19 |
| 64 | 3e-02 | 0.8309513 | 0.8429342 | +1.442% | ✓ | ✓ | -3.277% | -15.71% | 2.21 |
| 64 | 1e-01 | 0.8309513 | 0.8369115 | +0.717% | ✓ | ✓ | -3.000% | -16.31% | 2.31 |
| 256 | 1e-03 | 0.8309513 | 0.8281517 | -0.337% | ✓ | ✓ | -0.828% | -17.18% | 2.1 |
| 256 | 1e-02 | 0.8309513 | 0.8465655 | +1.879% | ✓ | ✓ | -0.829% | -15.34% | 2.11 |
| 256 | 3e-02 | 0.8309513 | 0.8643584 | +4.020% | ✗ | ✓ | -0.818% | -13.56% | 2.14 |
| 256 | 1e-01 | 0.8309513 | 0.8563606 | +3.058% | ✗ | ✓ | -0.745% | -14.36% | 2.23 |

头行（256 面元、ka=1e−3）：got 0.8281517 vs ref 0.8309513，误差 **-0.337%**（绝对 -6.01 kg）。

## ④ 结果图

图 1 误差–频率对照曲线（含 3%/5% 带与带外扫描）；图 2 参考值三路线与偶模截断收敛；图 3 heave 解在 256 面元湿表面上的势场云图（附加质量正是该场量的面积分）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **坑 1（最贵）**：池行一句「刚性盖=镜像=映像整球」把端点误导到 1.0 ρ∇，造成 14% 量级的假 FAIL；本次把「映像整球」当**负控制**保留，实测每个 ka/网格都比它低 13–19%%，从测量上否定它。
- **坑 2**：驱动原两条参考路线差 3.4e−6，非物理而是 |μ| 在 0 处不可导导致的 Gauss 求积误差；新增精确有理数路线把该差值压到 3.9e−8（对 energy）/1e−9 级，裁判不再受求积污染。
- **局限 1（诚实披露）**：端点参考只在 ka≤0.1 适用；带外扫描 ka=0.2/0.3/0.6/1.0 误差 -5.1%、-14.3%、-35.5%、-48.8%——端点不是曲线，算例用 G6 强制它破带，防止越权宣称。
- **局限 2**：网格证据只到 256 面元（1024 超出本容器单命令预算），故收敛阶非渐近；几何本身（内接多面体）面积 −1.26%%、体积 −2.32%%。
- **局限 3**：无实验数据进入本算例，绝对量对的是解析解（双体级数 + Hulme 多极），不是 EFD。
- **修复面的证据分级**：实现 `src/tensorbem/rigid_lid.py` + 算例驱动 = REAL；集成调用 = 主求解器模块内真调用（非仅有 import）；工件 = runs/ 下真实运行日志 + sha256。

## ⑥ 误差判定：是否 <3%？

**关键量（头行，256 面元 ka=1e−3）：是，-0.337% < 3%（且 < 5%）。**
池判据（<5%）逐项：带内 8 行全部 ✓（6 行同时 <3%，2 行落在 3%–5% 之间：ka=3e-02 +4.02%、ka=1e-01 +3.06%）。
最坏带内行 ka=3e-02：+4.020%（网格 256 面元）——满足池的 5%，但**不满足 3%**，如实列出不粉饰。

预注册判据 **14/14 全 PASS**（G0 几何 / G1 带内 <5% / G2 收敛 / G3+G3b 参考自一致 / G4 Hulme / G5 负控制 / G6 带外必破带）。测试：`tests/test_case_bem_rigid_lid_hemisphere.py` 11 passed、`tests/test_rigid_lid_exact.py` 7 passed；复现脚本 exit=0。

## 工件

- `benchmarks/verified/case_bem_rigid_lid_hemisphere/{results.json,report.md,case.json,figures/}`
- `src/tensorbem/rigid_lid.py（精确裁判 + `python -m tensorbem.rigid_lid` 入口）`
- `src/tensorbem/benchmarks/case_bem_rigid_lid_hemisphere.py（算例实现 + `python -m` 入口）`
- `src/tensorbem/free_surface_solver3d.py（集成调用：verify_rigid_lid_heave_limit）`
- `runs/m17464_rigid_lid_hemisphere/{run.log,repro.log,results.json,benchmark_ref_vs_got.json,meta.json,figures/}`
- `scripts/repro_case_bem_rigid_lid_hemisphere.sh、scripts/make_report_m17464.py`
- `docs/case_bem_rigid_lid_hemisphere.md、docs/m17464/（三张图 + manifest）`

环境：python 3.11.6、numpy 2.4.6、loongarch64、git `9ebee0f77bb8`。
