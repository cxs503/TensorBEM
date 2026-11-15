"""Fix the lattice heading to an exact quadrant rotation (equivariance, not identity)."""

from pathlib import Path

PATH = Path("src/tensorbem/benchmarks/case_cylinder_stream.py")
text = PATH.read_text(encoding="utf-8")

old = '''# One heading is structural: 45 deg at N = 128 is exactly 16 panel steps, so the
# rotated body is the *same* vertex set and every number must be reproduced to
# roundoff.  That pins rotational covariance of the assembly and independence of
# the panel ordering, which no axis-aligned case can see.
OBLIQUE_HEADING_DEG = 30.0
LATTICE_HEADING_DEG = 45.0
# The oblique configuration must not be materially worse than the axis-aligned
# reference; the symmetry argument above is only worth something if breaking the
# symmetry costs nothing.
OBLIQUE_HEADING_PENALTY_MAX = 3.0
# Same point set, rotated vertex ordering: only roundoff may remain.
HEADING_LATTICE_COVARIANCE_TOL = 1.0e-10
'''
new = '''# One heading is structural: a *quadrant* is a whole number of panel steps for
# every even panel count, and its rotation matrix is exact, so rotating the body
# by 90 deg **and the probe set by the same 90 deg** must reproduce the
# axis-aligned solve to roundoff.  That is rotational equivariance of the whole
# chain (assembly, boundary data, exterior evaluation): it fails if any stage
# leans on the panel ordering or on the phase between the inflow and the lattice.
# It is not an identity test -- the body rotated by 90 deg with the inflow left
# along +x is a *different* configuration (its field differs by O(U)), which is
# why the probes have to rotate with it.
OBLIQUE_HEADING_DEG = 30.0
LATTICE_HEADING_DEG = 90.0
# The oblique configuration must not be materially worse than the axis-aligned
# reference; the symmetry argument above is only worth something if breaking the
# symmetry costs nothing.
OBLIQUE_HEADING_PENALTY_MAX = 3.0
# Equivariance is a roundoff statement; the exact rotation makes the two solves
# bit-comparable up to the 1e-16 error in cos(90 deg).
HEADING_LATTICE_COVARIANCE_TOL = 1.0e-9
QUADRANT_ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]])
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''    lattice_row, lattice_field = _heading_level(
        panels, LATTICE_HEADING_DEG, radius, speed, points
    )
'''
new = '''    lattice_row, lattice_field = _heading_level(
        panels,
        LATTICE_HEADING_DEG,
        radius,
        speed,
        points @ QUADRANT_ROTATION.T,
    )
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''        "note": (
            "rotating the body by a whole number of panel steps maps the mesh onto "
            "itself, so the recovered field must be reproduced to roundoff; the "
            "half-panel and 30 deg headings are the ones that break the mirror "
            "symmetry of the axis-aligned configuration"
        ),'''
new = '''        "note": (
            "the lattice level rotates body and probes together by 90 deg (a whole "
            "number of panel steps with an exact rotation matrix), so it must "
            "reproduce the axis-aligned solve to roundoff; the half-panel and 30 "
            "deg headings break the mirror symmetry of the axis-aligned "
            "configuration while leaving the body in place"
        ),
        "lattice_probe_rotation_deg": LATTICE_HEADING_DEG,'''
assert text.count(old) == 1
text = text.replace(old, new)

# render: the lattice sentence and the table's "（格点）" note
old = '''        f"$N={heading['panels']}$ 时面板步长 {heading['panel_step_deg']:.4f}$^\\\\circ$，"
        f"$\\\\gamma={heading['lattice']['heading_deg']:g}^\\\\circ$ 恰为 "
        f"{heading['lattice']['panel_steps']:g} 个面板步长 —— 旋转后仍是**同一顶点集**，"
        "故所有数字必须逐位重现（装配的旋转协方差 + 面板编号无关性）。",'''
new = '''        f"$N={heading['panels']}$ 时面板步长 {heading['panel_step_deg']:.4f}$^\\\\circ$；"
        f"$\\\\gamma={heading['lattice']['heading_deg']:g}^\\\\circ$ 恰为 "
        f"{heading['lattice']['panel_steps']:g} 个面板步长且旋转矩阵精确可表，"
        "把**探针集合同样转该角度**后，数值解必须与 0$^\\\\circ$ 情形逐位一致"
        "（装配、边界数据、外部求值全链的旋转等变性：任何环节依赖面板编号或来流-格点相位都会破）。",'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''        f"格点旋转（{heading['lattice']['panel_steps']:g} 个步长）后外部速度场最大差 "
        f"{heading['lattice']['velocity_field_max_difference_over_speed']:.2e} $U$，"
        "落在舍入量级 —— 装配与求值都不依赖面板编号或来流与格点的相对相位。",'''
new = '''        f"格点角 {heading['lattice']['heading_deg']:g}$^\\\\circ$（{heading['lattice']['panel_steps']:g} 个步长，"
        "探针同转）后外部速度场与 0$^\\\\circ$ 情形最大差 "
        f"{heading['lattice']['velocity_field_max_difference_over_speed']:.2e} $U$，"
        "落在舍入量级；注意「物体自转格点角、来流不动」是**另一个**位形（其速度场差 $O(U)$），"
        "脚本比对的是旋转等变性，不是恒等性。",'''
assert text.count(old) == 1
text = text.replace(old, new)

PATH.write_text(text, encoding="utf-8")
print("ok")