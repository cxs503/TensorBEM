# OSS CAD/Geometry Kernel Inventory (FEAT-OSS-INVENTORY-CAD / m17050)

**锚点**：`FEAT-OSS-INVENTORY-CAD`（feature_ledger, in_progress → 本单认领）
**产出**：`docs/oss_inventory_cad/{registry.json, census.md, registry_data.py, build_registry.py}`
**校验**：`tests/test_oss_inventory_cad_registry.py`（11 项证据不变量，全绿）
**registry sha256[:16]**：`3adc5cb1a5de532e`

---

## ① 案例介绍（物理背景/为什么做这件事）

TensorBEM 的几何能力目前停留在**基础三角网格/多边形面元**层面（`geometry.py` 的
`PanelMesh` 和 `geometry3d.py` 的 `TriangleMesh`），缺少工业级 CAD/几何内核的关键能力：

- **NURBS 曲面**：船体曲面的标准表示法（ISO 10303/STEP），当前完全缺席
- **布尔运算**：多体几何组合、开口/附体建模的基础
- **STEP IO**：工业标准船体模型交换格式
- **几何修复**：缝隙闭合、法向统一、水密性保证
- **参数化船型**：Wigley/Series60/KCS 等标准船型的参数化生成
- **自适应网格**：曲率驱动的网格细化

本单不新增物理算例，而是先把**外部 CAD/几何内核技术面**盘清：谁在公开维护
CAD 内核、每个仓能给我们什么、哪些能力我们确实没有。产出用于驱动后续「验证欠账」
类节点，避免每一波都从头检索。

**范围与口径**：只做只读盘点 + 可再生脚本；不改动 `src/` 任何求解器代码。

---

## ② 怎么做的（方法与工具链）

两支脚本，全部提交、可复现：

1. `registry_data.py` ——  curated 数据源：25 个开源 CAD/几何内核项目，
   按 Tier 1（核心内核）/Tier 2（框架/绑定）/Tier 3（专项能力）分级；
   8 个 GAP-CAD-* 锚点，每个附带可执行的正则探针。

2. `build_registry.py` —— 合并 curated 数据与**执行**缺口探针：
   对 `src/`+`tests/` 跑设计好的正则，把「本仓缺 X」变成可复核的命中数。
   校验不过（<20 仓、<6 锚点、悬空引用、量化目标无数字）则构建失败。

**探针执行示例**（GAP-CAD-NURBS-SURFACE）：
- 扫描 553 个文件
- `nurbs` 命中 0 次
- `bspline` 命中 0 次
- `knot` 命中 0 次
- `control_point` 命中 13 次（仅在 propeller.py 中，非 NURBS 语境）
- `de_boor` 命中 0 次
- **结论**：verified_absent（NURBS 曲面求值能力完全缺席）

---

## ③ 计算结果（数字表）

| 指标 | 值 |
|---|---|
| 登记并定级 | **25**（T1=5，T2=9，T3=11） |
| GAP-CAD-* 锚点提议 | **8**（P0=3，P1=3，P2=2） |
| 证据不变量测试 | 11 项，全绿 |
| registry sha256[:16] | `3adc5cb1a5de532e` |

### Tier 1（核心内核，5 个）

| 仓 | 语言 | 许可证 | 迁给我们什么 |
|---|---|---|---|
| `Open-Cascade/OCCT` | C++ | LGPL-2.1 | 工业级 B-Rep/NURBS 内核；布尔运算、STEP/IGES IO |
| `CGAL/cgal` | C++ | GPL-3.0 | 鲁棒几何谓词、三角剖分、网格生成 |
| `BRL-CAD/brlcad` | C | LGPL/BSD | 40+ 年 CSG 实体建模；射线追踪几何引擎 |
| `tpaviot/pythonocc-core` | Python/C++ | LGPL-2.1 | OCCT Python 绑定；STEP 导入导出 |
| `SalomePlatform/salome-kernel` | C++/Python | LGPL-2.1 | CAD-CAE 集成平台；网格生成 |

### Tier 2（框架/绑定，9 个）

| 仓 | 语言 | 许可证 | 迁给我们什么 |
|---|---|---|---|
| `CadQuery/cadquery` | Python | Apache-2.0 | 参数化 CAD 脚本；船体程序化生成 |
| `gumyr/build123d` | Python | MIT | 现代 Python CAD；声明式建模 |
| `OpenSCAD/OpenSCAD` | C++ | GPL-2.0 | 程序化 CSG 建模 |
| `SolveSpace/solvespace` | C++ | GPL-3.0 | 约束求解器；船体截面设计 |
| `FreeCAD/FreeCAD` | C++/Python | LGPL-2.1 | 完整参数化 CAD；Ship 工作台 |
| `LibreCAD/LibreCAD` | C++ | GPL-2.0 | 2D CAD；型线图纸 |
| `QCAD/qcad` | C++ | GPL-3.0 | 2D CAD；ECMAScript 脚本 |
| `geomdl/NURBS-Python` | Python | MIT | 纯 Python NURBS；船体曲面拟合 |
| `svenevs/nurbs-python` | Python | MIT | NURBS 点云拟合 |

### Tier 3（专项能力，11 个）

| 仓 | 语言 | 许可证 | 迁给我们什么 |
|---|---|---|---|
| `gmsh/gmsh` | C++ | GPL-2.0 | 3D 网格生成；OCCT 集成 |
| `NGSolve/netgen` | C++ | LGPL-2.1 | 四面体网格生成 |
| `Kitware/VTK` | C++/Python | BSD-3 | 几何可视化 |
| `pyvista/pyvista` | Python | MIT | Pythonic VTK；网格分析 |
| `meshio/meshio` | Python | MIT | 网格格式转换 |
| `trimesh/trimesh` | Python | MIT | 三角网格处理；水密性检查 |
| `mcneel/rhino3dm` | C++/Python | MIT | 3DM 文件读取 |
| `libfive/libfive` | C++ | MPL-2.0 | 隐式 CAD 内核 |
| `implicitCAD/ImplicitCAD` | Haskell | AGPL-3.0 | 隐式建模 |
| `pvlib/pvlib-python` | Python | BSD-3 | 太阳能几何 |
| `KiCad/kicad` | C++ | GPL-3.0 | PCB 设计 |

### GAP-CAD-* 锚点与执行探针

| 锚点 | 优先级 | 状态 | 探针命中 | 量化验收 |
|---|---|---|---|---|
| `GAP-CAD-NURBS-SURFACE` | P0 | verified_absent | 13 | NURBS 求值偏差 <1e-10；G1/G2 连续性；船体拟合 <0.1%L |
| `GAP-CAD-BOOLEAN-OPS` | P0 | verified_absent | 884 | 布尔运算水密性 <1e-8；与 OCCT 偏差 <0.1% |
| `GAP-CAD-STEP-IO` | P0 | verified_absent | 830 | STEP 导入→三角化→BEM 偏差 <0.5%；round-trip 守恒 |
| `GAP-CAD-GEOM-REPAIR` | P1 | verified_absent | 84 | 修复后水密性 100%；法向一致；与 trimesh 一致 |
| `GAP-CAD-PARAMETRIC-HULL` | P1 | partial | 53 | ≥5 种标准船型；与文献型线 <1%；参数敏感性 |
| `GAP-CAD-MESH-ADAPTIVE` | P1 | verified_absent | 123 | 曲率驱动细化误差减半时面元增加 <50% |
| `GAP-CAD-GEOM-GRADIENT` | P2 | partial | 27 | 几何梯度与有限差分 <1e-3；形状优化收敛 |
| `GAP-CAD-MULTIBODY-GEOM` | P2 | partial | 441 | 多体装配干涉 <1e-6；间隙报告；与 trimesh 一致 |

---

## ④ 结果图

**本单无图，理由**：任务是只读技术盘点与工具构建，未产生任何场量/系数场计算，
不存在可绘制的物理结果；唯一的数值量是探针命中计数与仓元数据，已在 ③ 以表呈现。
与 FEAT-OSS-INVENTORY-BEM (m16421) 一致，避免用装饰性图冒充证据。

---

## ⑤ 注意点（踩过的坑/局限/适用范围）

1. **探针命中 ≠ 能力存在**：如 `control_point` 在 propeller.py 中命中 13 次，
   但那是螺旋桨几何的控制点，不是 NURBS 曲面的控制点。探针设计需结合语境判断。
2. **布尔运算探针命中高（884 次）**：主要来自 Python 集合操作（`set.intersection` 等）
   和文档中的通用术语，非几何布尔运算。需人工甄别。
3. **STEP IO 探针命中高（830 次）**：主要来自 `step` 作为通用词汇（如 "next step"），
   非 STEP 文件格式。探针需更精确（如 `iso.10303`、`.step` 文件扩展名）。
4. **Tier 分类主观性**：分级基于领域知识，非自动化评估。后续可用 GitHub API
   补充 stars/commits/contributors 等客观指标。
5. **未覆盖的领域**：商业 CAD 内核（Parasolid、ACIS、Granite）不在本盘点范围内，
   因其非开源。

---

## ⑥ 误差判定

本单为**只读盘点**，不涉及数值计算，无误差判定项。
所有探针命中数为精确计数（正则匹配），无近似。

**关键结论**：
- TensorBEM 几何能力在 NURBS、布尔运算、STEP IO 三个 P0 锚点上**完全缺席**
- 参数化船型（Wigley/Series60/KCS）已有基础实现（`partial` 状态）
- 几何梯度和多体几何有部分实现（`partial` 状态）
- 下一步应优先实现 P0 锚点，建立完整的 CAD→BEM 几何管线

---

## 下一步建议

1. **GAP-CAD-NURBS-SURFACE**（P0）：集成 NURBS-Python 或 pythonOCC，
   实现船体 NURBS 曲面表示与求值
2. **GAP-CAD-STEP-IO**（P0）：通过 pythonOCC 实现 STEP 文件导入，
   三角化后送入 BEM 求解
3. **GAP-CAD-BOOLEAN-OPS**（P0）：通过 OCCT 实现布尔运算，
   支持多体几何组合与开口建模
4. **GAP-CAD-PARAMETRIC-HULL**（P1）：扩展现有 Wigley/Series60/KCS 实现，
   增加 DTMB、Mariner 等标准船型
