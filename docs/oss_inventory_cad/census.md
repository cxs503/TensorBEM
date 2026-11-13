# OSS CAD/Geometry Kernel Census (FEAT-OSS-INVENTORY-CAD)

**registry sha256[:16]**: `3adc5cb1a5de532e`
**repositories**: 25  |  **GAP anchors**: 8
**generated**: 2026-11-13  |  **mission**: m17050

---

## Repository Registry

| # | Repository | Tier | Language | License | Focus |
|---|---|---|---|---|---|
| 1 | `Open-Cascade/OCCT` | T1 | C++ | LGPL-2.1 | B-Rep, NURBS, CSG (+3) |
| 2 | `CGAL/cgal` | T1 | C++ | GPL-3.0 / commercial | computational-geometry, arrangements, mesh-generation (+1) |
| 3 | `BRL-CAD/brlcad` | T1 | C | LGPL/BSD | CSG, ray-tracing, military (+1) |
| 4 | `tpaviot/pythonocc-core` | T1 | Python/C++ | LGPL-2.1 | OCCT-bindings, Python, B-Rep (+1) |
| 5 | `SalomePlatform/salome-kernel` | T1 | C++/Python | LGPL-2.1 | CAD-CAE, mesh, geometry (+2) |
| 6 | `CadQuery/cadquery` | T2 | Python | Apache-2.0 | parametric, Python, OCCT (+1) |
| 7 | `gumyr/build123d` | T2 | Python | MIT | parametric, Python, OCCT (+1) |
| 8 | `OpenSCAD/OpenSCAD` | T2 | C++ | GPL-2.0 | CSG, scripting, 3D-printing (+1) |
| 9 | `SolveSpace/solvespace` | T2 | C++ | GPL-3.0 | parametric, 2D-3D, constraint-solver (+1) |
| 10 | `FreeCAD/FreeCAD` | T2 | C++/Python | LGPL-2.1 | parametric, OCCT, workbenches (+1) |
| 11 | `LibreCAD/LibreCAD` | T2 | C++ | GPL-2.0 | 2D-CAD, DXF, Qt |
| 12 | `QCAD/qcad` | T2 | C++ | GPL-3.0 | 2D-CAD, DXF, ECMAScript |
| 13 | `geomdl/NURBS-Python` | T2 | Python | MIT | NURBS, B-spline, Python (+1) |
| 14 | `svenevs/nurbs-python` | T2 | Python | MIT | NURBS, Python, fitting |
| 15 | `gmsh/gmsh` | T3 | C++ | GPL-2.0 | mesh-generation, CAD, geo (+1) |
| 16 | `NGSolve/netgen` | T3 | C++ | LGPL-2.1 | mesh-generation, OCCT, FEM (+1) |
| 17 | `Kitware/VTK` | T3 | C++/Python | BSD-3-Clause | visualization, geometry, C++ (+1) |
| 18 | `pvlib/pvlib-python` | T3 | Python | BSD-3-Clause | geometry, solar, Python |
| 19 | `pyvista/pyvista` | T3 | Python | MIT | visualization, VTK, Python (+1) |
| 20 | `meshio/meshio` | T3 | Python | MIT | mesh-IO, format-conversion, Python |
| 21 | `trimesh/trimesh` | T3 | Python | MIT | mesh-processing, collision, Python |
| 22 | `mcneel/rhino3dm` | T3 | C++/Python | MIT | openNURBS, 3DM, read-only (+1) |
| 23 | `libfive/libfive` | T3 | C++ | MPL-2.0 | implicit, functional-CAD, C++ |
| 24 | `implicitCAD/ImplicitCAD` | T3 | Haskell | AGPL-3.0 | implicit, Haskell, CSG |
| 25 | `KiCad/kicad` | T3 | C++ | GPL-3.0 | PCB, 2D-3D, ECAD |

### Tier Distribution

| Tier | Count | Description |
|---|---|---|
| 1 (核心内核) | 5 | 核心几何内核/求解器（NURBS/B-Rep/CSG/隐式曲面，可独立做几何运算） |
| 2 (框架/绑定) | 9 | 框架/绑定/可微几何（Python-first、参数化建模框架、CAD 应用） |
| 3 (专项能力) | 11 | 专项能力（网格生成、可视化、格式转换、教育/niche 物理） |

---

## GAP-CAD-* Anchors (TensorBEM Geometry Gaps)

| Slug | Priority | Status | Probe Hits | Description |
|---|---|---|---|---|
| `GAP-CAD-NURBS-SURFACE` | P0 | verified_absent | 13 | NURBS/B-spline 曲面表示与求值（船体曲面标准表示法） |
| `GAP-CAD-BOOLEAN-OPS` | P0 | verified_absent | 884 | 布尔运算（并/交/差）用于多体几何组合与开口/附体建模 |
| `GAP-CAD-STEP-IO` | P0 | verified_absent | 830 | STEP (ISO 10303) 几何文件读写，工业标准船体模型交换 |
| `GAP-CAD-GEOM-REPAIR` | P1 | verified_absent | 84 | 几何修复（缝隙闭合、法向统一、自交检测、水密性保证） |
| `GAP-CAD-PARAMETRIC-HULL` | P1 | partial | 53 | 参数化船型生成（Wigley/Series60/KCS 等标准船型 + 自定义参数） |
| `GAP-CAD-MESH-ADAPTIVE` | P1 | verified_absent | 123 | 自适应网格细化（曲率驱动/误差驱动网格加密） |
| `GAP-CAD-GEOM-GRADIENT` | P2 | partial | 27 | 几何梯度/灵敏度分析（形状优化基础） |
| `GAP-CAD-MULTIBODY-GEOM` | P2 | partial | 441 | 多体几何装配（多船体、附体、推进器几何组合与干涉检查） |

### Probe Details

#### `GAP-CAD-NURBS-SURFACE`
- **Priority**: P0
- **Status**: verified_absent
- **Files scanned**: 553
- **Total hits**: 13
- **Acceptance**: NURBS 曲面求值与 OCCT/NURBS-Python 参考偏差 <1e-10；G1/G2 连续性检查；船体曲面拟合误差 <0.1%L
- **Sample hits**:
  - `control_point` → `src/tensorbem/propeller.py:161`
  - `control_point` → `src/tensorbem/propeller.py:193`
  - `control_point` → `src/tensorbem/propeller.py:211`

#### `GAP-CAD-BOOLEAN-OPS`
- **Priority**: P0
- **Status**: verified_absent
- **Files scanned**: 553
- **Total hits**: 884
- **Acceptance**: 布尔运算结果水密性检查 <1e-8；与 OCCT 参考几何偏差 <0.1% 特征尺寸
- **Sample hits**:
  - `boolean` → `src/tensorbem/cathodic_protection.py:113`
  - `boolean` → `src/tensorbem/cathodic_protection.py:281`
  - `boolean` → `src/tensorbem/cathodic_protection.py:283`
  - `union` → `src/tensorbem/acoustic3d.py:554`
  - `union` → `src/tensorbem/cathodic_protection.py:11`
  - `union` → `src/tensorbem/coefficients.py:430`
  - `intersection` → `src/tensorbem/hydrostatic.py:139`
  - `intersection` → `src/tensorbem/hydrostatic.py:164`
  - `intersection` → `src/tensorbem/hydrostatic.py:420`
  - `difference` → `src/tensorbem/acoustic3d.py:574`
  - `difference` → `src/tensorbem/acoustic3d_pulse.py:428`
  - `difference` → `src/tensorbem/capability_evidence.py:195`
  - `fuse` → `src/tensorbem/capability_evidence.py:1922`
  - `fuse` → `src/tensorbem/free_surface3d.py:535`
  - `fuse` → `src/tensorbem/kernels.py:40`
  - `cut` → `src/tensorbem/acoustic3d.py:927`
  - `cut` → `src/tensorbem/finite_depth3d.py:86`
  - `cut` → `src/tensorbem/finite_depth3d.py:330`

#### `GAP-CAD-STEP-IO`
- **Priority**: P0
- **Status**: verified_absent
- **Files scanned**: 553
- **Total hits**: 830
- **Acceptance**: STEP 导入→三角化→BEM 求解→与原始几何偏差 <0.5% 特征尺寸；round-trip 几何守恒
- **Sample hits**:
  - `step` → `src/tensorbem/acoustic3d_pulse.py:109`
  - `step` → `src/tensorbem/acoustic3d_pulse.py:120`
  - `step` → `src/tensorbem/acoustic3d_pulse.py:136`

#### `GAP-CAD-GEOM-REPAIR`
- **Priority**: P1
- **Status**: verified_absent
- **Files scanned**: 553
- **Total hits**: 84
- **Acceptance**: 修复后网格水密性 100%；法向一致性检查；与 trimesh/Netgen 修复结果一致
- **Sample hits**:
  - `heal` → `src/tensorbem/benchmarks/doc_claim_audit.py:86`
  - `heal` → `src/tensorbem/benchmarks/panel_operator_order.py:618`
  - `heal` → `tests/test_two_cylinder_added_mass.py:260`
  - `repair` → `src/tensorbem/finite_depth3d_quadrature.py:73`
  - `repair` → `src/tensorbem/near_wall_envelope.py:184`
  - `repair` → `src/tensorbem/rankine_wave.py:18`
  - `sew` → `src/tensorbem/acoustic3d.py:304`
  - `sew` → `src/tensorbem/acoustic3d.py:312`
  - `sew` → `src/tensorbem/solver3d.py:103`
  - `watertight` → `src/tensorbem/diffraction_drift_maturity.py:382`
  - `watertight` → `src/tensorbem/geometry3d.py:332`
  - `watertight` → `src/tensorbem/hydrostatic_pressure_route.py:193`
  - `self.intersect` → `tests/test_mutation_gaps_hydrostatic.py:117`
  - `manifold` → `src/tensorbem/acoustic3d.py:537`
  - `manifold` → `src/tensorbem/acoustic3d.py:540`
  - `manifold` → `src/tensorbem/qtf_drift.py:887`

#### `GAP-CAD-PARAMETRIC-HULL`
- **Priority**: P1
- **Status**: partial
- **Files scanned**: 553
- **Total hits**: 53
- **Acceptance**: ≥5 种标准船型参数化生成；与文献型线偏差 <1%；参数扰动敏感性分析
- **Sample hits**:
  - `parametric.hull` → `src/tensorbem/kcs_hull.py:6`
  - `parametric.hull` → `src/tensorbem/kcs_hull.py:13`
  - `parametric.hull` → `src/tensorbem/pc6_hull.py:188`
  - `series.60` → `src/tensorbem/capability_evidence.py:1477`
  - `series.60` → `src/tensorbem/capability_evidence.py:1478`
  - `series.60` → `src/tensorbem/series60.py:1`
  - `kcs.hull` → `src/tensorbem/capability_evidence.py:1016`
  - `kcs.hull` → `src/tensorbem/resistance.py:190`
  - `kcs.hull` → `src/tensorbem/benchmarks/added_resistance_cloud.py:81`

#### `GAP-CAD-MESH-ADAPTIVE`
- **Priority**: P1
- **Status**: verified_absent
- **Files scanned**: 553
- **Total hits**: 123
- **Acceptance**: 曲率驱动细化使 BEM 系数误差减半时面元数增加 <50%；与 Gmsh 自适应结果对比 <5%
- **Sample hits**:
  - `adaptive.mesh` → `src/tensorbem/benchmarks/multibody_frequency_dependence_3d.py:113`
  - `adaptive.mesh` → `src/tensorbem/benchmarks/multibody_frequency_dependence_3d.py:196`
  - `adaptive.mesh` → `src/tensorbem/benchmarks/multibody_frequency_dependence_3d.py:440`
  - `error.estimate` → `src/tensorbem/quadrature_diagnostics.py:165`
  - `error.estimate` → `src/tensorbem/quadrature_diagnostics.py:166`
  - `error.estimate` → `src/tensorbem/quadrature_diagnostics.py:167`
  - `h.refinement` → `src/tensorbem/capability_evidence.py:228`
  - `h.refinement` → `src/tensorbem/cathodic_protection.py:948`
  - `h.refinement` → `src/tensorbem/conditioning.py:240`

#### `GAP-CAD-GEOM-GRADIENT`
- **Priority**: P2
- **Status**: partial
- **Files scanned**: 553
- **Total hits**: 27
- **Acceptance**: 几何梯度与有限差分偏差 <1e-3；形状优化收敛；与 OpenCASCADE 灵敏度一致
- **Sample hits**:
  - `shape.gradient` → `src/tensorbem/panel_quadratic.py:74`
  - `shape.gradient` → `src/tensorbem/panel_quadratic.py:76`
  - `shape.gradient` → `src/tensorbem/panel_quadratic.py:132`
  - `geometry.sensitivity` → `src/tensorbem/benchmarks/b4_70_error_surface.py:1219`
  - `shape.derivative` → `tests/test_shape_gradient.py:1`
  - `adjoint.geom` → `src/tensorbem/shape_gradient.py:8`
  - `adjoint.geom` → `tests/test_shape_gradient.py:3`

#### `GAP-CAD-MULTIBODY-GEOM`
- **Priority**: P2
- **Status**: partial
- **Files scanned**: 553
- **Total hits**: 441
- **Acceptance**: 多体装配干涉检测 <1e-6 精度；间隙报告；与 trimesh 碰撞检测结果一致
- **Sample hits**:
  - `clearance` → `src/tensorbem/__init__.py:35`
  - `clearance` → `src/tensorbem/__init__.py:35`
  - `clearance` → `src/tensorbem/__init__.py:35`

---

## Methodology

1. **Curated survey**: 25 open-source CAD/geometry kernel projects identified
   through domain knowledge, GitHub search, and cross-referencing with
   FreeCAD/OCCT ecosystem.
2. **Tier classification**: Based on capability depth (kernel vs framework vs tool).
3. **GAP probes**: Executed regex probes over TensorBEM `src/` + `tests/` to
   quantify absence of each capability — not asserted, measured.
4. **Integrity**: `registry.json` sha256 fingerprinted; census.md generated
   from data, no hand-copied numbers.

## Next Steps

- **P0 anchors** (NURBS, Boolean, STEP IO): highest priority for hull geometry pipeline.
- **P1 anchors** (Repair, Parametric Hull, Adaptive Mesh): enable production workflows.
- **P2 anchors** (Geometry Gradient, Multibody): shape optimization and complex assemblies.
- Deep-dive wave: follow LBM protocol to implement selected GAP anchors.
