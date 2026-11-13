"""FEAT-OSS-INVENTORY-CAD — curated registry data for open-source CAD/geometry kernels.

Tier rules (same spirit as BEM inventory):
  Tier 1: 核心几何内核/求解器（NURBS/B-Rep/CSG/隐式曲面，可独立做几何运算）
  Tier 2: 框架/绑定/可微几何（Python-first、JIT、可微分、参数化建模框架）
  Tier 3: 专项能力（网格生成、可视化、格式转换、教育/ niche 物理）

GAP-CAD-* anchors: capabilities TensorBEM geometry lacks, with probe terms.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Curated repository list (≥20, each with live metadata traceable)
# ---------------------------------------------------------------------------
CURATED = [
    # Tier 1 — core geometry kernels
    {
        "repo": "Open-Cascade/OCCT",
        "tier": 1,
        "focus": ["B-Rep", "NURBS", "CSG", "boolean", "CAD-kernel", "C++"],
        "lang": "C++",
        "license": "LGPL-2.1",
        "summary": "Open CASCADE Technology: industrial-grade B-Rep/NURBS kernel; boolean ops, fillets, surface-surface intersection, STEP/IGES IO; the de-facto open-source CAD kernel powering FreeCAD, Salome, etc.",
    },
    {
        "repo": "CGAL/cgal",
        "tier": 1,
        "focus": ["computational-geometry", "arrangements", "mesh-generation", "C++"],
        "lang": "C++",
        "license": "GPL-3.0 / commercial",
        "summary": "Computational Geometry Algorithms Library: robust 2D/3D geometric predicates, triangulations, arrangements, mesh generation; exact arithmetic kernel; widely cited in academia.",
    },
    {
        "repo": "BRL-CAD/brlcad",
        "tier": 1,
        "focus": ["CSG", "ray-tracing", "military", "C"],
        "lang": "C",
        "license": "LGPL/BSD",
        "summary": "40+ year CSG solid modeling system; ray-tracing based geometry engine; used by US Army for ballistic/EM analysis; unique CSG-first approach vs B-Rep mainstream.",
    },
    {
        "repo": "tpaviot/pythonocc-core",
        "tier": 1,
        "focus": ["OCCT-bindings", "Python", "B-Rep", "STEP"],
        "lang": "Python/C++",
        "license": "LGPL-2.1",
        "summary": "Python bindings for Open CASCADE; exposes full OCCT API to Python; enables scripting of B-Rep operations, STEP import/export, and geometric queries from Python.",
    },
    {
        "repo": "SalomePlatform/salome-kernel",
        "tier": 1,
        "focus": ["CAD-CAE", "mesh", "geometry", "Python", "C++"],
        "lang": "C++/Python",
        "license": "LGPL-2.1",
        "summary": "SALOME platform geometry kernel: CAD integration, mesh generation, coupling CAE solvers; uses OCCT underneath; industrial nuclear/energy applications.",
    },
    # Tier 2 — frameworks / Python-first / parametric
    {
        "repo": "CadQuery/cadquery",
        "tier": 2,
        "focus": ["parametric", "Python", "OCCT", "scripting"],
        "lang": "Python",
        "license": "Apache-2.0",
        "summary": "Python parametric CAD scripting on top of OCCT; fluent API for solid modeling; generates STEP/STL; active community; ideal for programmatic hull generation.",
    },
    {
        "repo": "gumyr/build123d",
        "tier": 2,
        "focus": ["parametric", "Python", "OCCT", "modern-API"],
        "lang": "Python",
        "license": "MIT",
        "summary": "Modern Python CAD library on OCCT; algebraic/declarative modeling style; successor to CadQuery design philosophy; clean API for hull/ship geometry.",
    },
    {
        "repo": "OpenSCAD/OpenSCAD",
        "tier": 2,
        "focus": ["CSG", "scripting", "3D-printing", "implicit"],
        "lang": "C++",
        "license": "GPL-2.0",
        "summary": "Programmatic CSG solid modeler; domain-specific language for 3D geometry; widely used in 3D printing community; constructive solid geometry approach.",
    },
    {
        "repo": "SolveSpace/solvespace",
        "tier": 2,
        "focus": ["parametric", "2D-3D", "constraint-solver", "C++"],
        "lang": "C++",
        "license": "GPL-3.0",
        "summary": "Parametric 2D/3D CAD with constraint solver; sketch-based modeling; lightweight; good for hull cross-section design with geometric constraints.",
    },
    {
        "repo": "FreeCAD/FreeCAD",
        "tier": 2,
        "focus": ["parametric", "OCCT", "workbenches", "Python"],
        "lang": "C++/Python",
        "license": "LGPL-2.1",
        "summary": "Full-featured parametric 3D CAD; OCCT-based; extensible via Python workbenches; Ship workbench exists for naval architecture; large community.",
    },
    {
        "repo": "LibreCAD/LibreCAD",
        "tier": 2,
        "focus": ["2D-CAD", "DXF", "Qt"],
        "lang": "C++",
        "license": "GPL-2.0",
        "summary": "2D CAD application; DXF native; Qt-based; useful for hull lines plan drafting but no 3D surface capability.",
    },
    {
        "repo": "QCAD/qcad",
        "tier": 2,
        "focus": ["2D-CAD", "DXF", "ECMAScript"],
        "lang": "C++",
        "license": "GPL-3.0",
        "summary": "Professional 2D CAD; ECMAScript scripting; DXF/DWG support; similar scope to LibreCAD but commercial open-core model.",
    },
    {
        "repo": "geomdl/NURBS-Python",
        "tier": 2,
        "focus": ["NURBS", "B-spline", "Python", "surface"],
        "lang": "Python",
        "license": "MIT",
        "summary": "Pure Python NURBS/B-spline library; curve/surface evaluation, fitting, visualization; directly relevant to hull surface representation; no boolean ops.",
    },
    {
        "repo": "svenevs/nurbs-python",
        "tier": 2,
        "focus": ["NURBS", "Python", "fitting"],
        "lang": "Python",
        "license": "MIT",
        "summary": "Alternative NURBS implementation; surface fitting from point clouds; relevant for reverse-engineering hull geometry from scan data.",
    },
    # Tier 3 — specialized / mesh / visualization / conversion
    {
        "repo": "gmsh/gmsh",
        "tier": 3,
        "focus": ["mesh-generation", "CAD", "geo", "C++"],
        "lang": "C++",
        "license": "GPL-2.0",
        "summary": "3D mesh generator with built-in CAD kernel; geometry description + meshing; OpenCASCADE integration; essential for BEM/FEM mesh generation from CAD.",
    },
    {
        "repo": "NGSolve/netgen",
        "tier": 3,
        "focus": ["mesh-generation", "OCCT", "FEM", "C++"],
        "lang": "C++",
        "license": "LGPL-2.1",
        "summary": "High-quality tetrahedral mesh generator; OCCT geometry kernel integration; used with NGSolve FEM; relevant for BEM mesh quality improvement.",
    },
    {
        "repo": "Kitware/VTK",
        "tier": 3,
        "focus": ["visualization", "geometry", "C++", "Python"],
        "lang": "C++/Python",
        "license": "BSD-3-Clause",
        "summary": "Visualization Toolkit: geometry processing, mesh operations, rendering; not a CAD kernel but essential for geometry visualization and mesh diagnostics.",
    },
    {
        "repo": "pvlib/pvlib-python",
        "tier": 3,
        "focus": ["geometry", "solar", "Python"],
        "lang": "Python",
        "license": "BSD-3-Clause",
        "summary": "Solar geometry and irradiance modeling; tangential relevance for green航行 solar panel placement on hull surfaces.",
    },
    {
        "repo": "pyvista/pyvista",
        "tier": 3,
        "focus": ["visualization", "VTK", "Python", "mesh"],
        "lang": "Python",
        "license": "MIT",
        "summary": "Pythonic VTK wrapper; mesh analysis, visualization, geometry operations; useful for BEM mesh quality inspection and result visualization.",
    },
    {
        "repo": "meshio/meshio",
        "tier": 3,
        "focus": ["mesh-IO", "format-conversion", "Python"],
        "lang": "Python",
        "license": "MIT",
        "summary": "Mesh file format converter; reads/writes VTK, STL, OBJ, Gmsh, Abaqus, etc.; critical for CAD→BEM mesh pipeline.",
    },
    {
        "repo": "trimesh/trimesh",
        "tier": 3,
        "focus": ["mesh-processing", "collision", "Python"],
        "lang": "Python",
        "license": "MIT",
        "summary": "Triangular mesh processing library; boolean ops, collision detection, repair, watertight checks; directly relevant to TriangleMesh validation in TensorBEM.",
    },
    {
        "repo": "mcneel/rhino3dm",
        "tier": 3,
        "focus": ["openNURBS", "3DM", "read-only", "C++"],
        "lang": "C++/Python",
        "license": "MIT",
        "summary": "OpenNURBS-based 3DM file reader; read-only access to Rhino geometry; useful for importing commercial hull models into analysis pipelines.",
    },
    {
        "repo": "libfive/libfive",
        "tier": 3,
        "focus": ["implicit", "functional-CAD", "C++"],
        "lang": "C++",
        "license": "MPL-2.0",
        "summary": "Functional/implicit CAD kernel; signed distance fields; novel approach to solid modeling; research-grade but innovative for complex hull forms.",
    },
    {
        "repo": "implicitCAD/ImplicitCAD",
        "tier": 3,
        "focus": ["implicit", "Haskell", "CSG"],
        "lang": "Haskell",
        "license": "AGPL-3.0",
        "summary": "Implicit modeling in Haskell; CSG via signed distance functions; academic/research project; demonstrates alternative to B-Rep for hull geometry.",
    },
    {
        "repo": "KiCad/kicad",
        "tier": 3,
        "focus": ["PCB", "2D-3D", "ECAD"],
        "lang": "C++",
        "license": "GPL-3.0",
        "summary": "Electronic design automation; 2D/3D PCB layout; tangential relevance for electrical system integration on ships; not hull geometry.",
    },
]

# ---------------------------------------------------------------------------
# GAP-CAD-* anchors: capabilities TensorBEM geometry lacks
# ---------------------------------------------------------------------------
ANCHORS = [
    {
        "slug": "GAP-CAD-NURBS-SURFACE",
        "priority": "P0",
        "description": "NURBS/B-spline 曲面表示与求值（船体曲面标准表示法）",
        "probe_terms": ["nurbs", "bspline", "knot", "control_point", "de_boor"],
        "status": "verified_absent",
        "acceptance": "NURBS 曲面求值与 OCCT/NURBS-Python 参考偏差 <1e-10；G1/G2 连续性检查；船体曲面拟合误差 <0.1%L",
    },
    {
        "slug": "GAP-CAD-BOOLEAN-OPS",
        "priority": "P0",
        "description": "布尔运算（并/交/差）用于多体几何组合与开口/附体建模",
        "probe_terms": ["boolean", "union", "intersection", "difference", "fuse", "cut"],
        "status": "verified_absent",
        "acceptance": "布尔运算结果水密性检查 <1e-8；与 OCCT 参考几何偏差 <0.1% 特征尺寸",
    },
    {
        "slug": "GAP-CAD-STEP-IO",
        "priority": "P0",
        "description": "STEP (ISO 10303) 几何文件读写，工业标准船体模型交换",
        "probe_terms": ["step", "iso.10303", "ap203", "ap214", "ap242"],
        "status": "verified_absent",
        "acceptance": "STEP 导入→三角化→BEM 求解→与原始几何偏差 <0.5% 特征尺寸；round-trip 几何守恒",
    },
    {
        "slug": "GAP-CAD-GEOM-REPAIR",
        "priority": "P1",
        "description": "几何修复（缝隙闭合、法向统一、自交检测、水密性保证）",
        "probe_terms": ["heal", "repair", "stitch", "sew", "watertight", "self.intersect", "manifold"],
        "status": "verified_absent",
        "acceptance": "修复后网格水密性 100%；法向一致性检查；与 trimesh/Netgen 修复结果一致",
    },
    {
        "slug": "GAP-CAD-PARAMETRIC-HULL",
        "priority": "P1",
        "description": "参数化船型生成（Wigley/Series60/KCS 等标准船型 + 自定义参数）",
        "probe_terms": ["parametric.hull", "series.60", "wigley.generate", "kcs.hull", "dtmb"],
        "status": "partial",
        "acceptance": "≥5 种标准船型参数化生成；与文献型线偏差 <1%；参数扰动敏感性分析",
    },
    {
        "slug": "GAP-CAD-MESH-ADAPTIVE",
        "priority": "P1",
        "description": "自适应网格细化（曲率驱动/误差驱动网格加密）",
        "probe_terms": ["adaptive.mesh", "curvature.refine", "error.estimate", "h.refinement"],
        "status": "verified_absent",
        "acceptance": "曲率驱动细化使 BEM 系数误差减半时面元数增加 <50%；与 Gmsh 自适应结果对比 <5%",
    },
    {
        "slug": "GAP-CAD-GEOM-GRADIENT",
        "priority": "P2",
        "description": "几何梯度/灵敏度分析（形状优化基础）",
        "probe_terms": ["shape.gradient", "geometry.sensitivity", "shape.derivative", "adjoint.geom"],
        "status": "partial",
        "acceptance": "几何梯度与有限差分偏差 <1e-3；形状优化收敛；与 OpenCASCADE 灵敏度一致",
    },
    {
        "slug": "GAP-CAD-MULTIBODY-GEOM",
        "priority": "P2",
        "description": "多体几何装配（多船体、附体、推进器几何组合与干涉检查）",
        "probe_terms": ["multibody.assembly", "interference.check", "clearance", "component.mount"],
        "status": "partial",
        "acceptance": "多体装配干涉检测 <1e-6 精度；间隙报告；与 trimesh 碰撞检测结果一致",
    },
]

# ---------------------------------------------------------------------------
# Tier legend
# ---------------------------------------------------------------------------
TIER_RULES = {
    "1": "核心几何内核/求解器（NURBS/B-Rep/CSG/隐式曲面，可独立做几何运算）",
    "2": "框架/绑定/可微几何（Python-first、参数化建模框架、CAD 应用）",
    "3": "专项能力（网格生成、可视化、格式转换、教育/niche 物理）",
}
