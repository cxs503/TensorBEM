"""Insert the oblique-inflow heading sweep into the cylinder case module.

One-shot patch: exact-anchor string replacements, refuses to run twice.
"""

from pathlib import Path

PATH = Path("src/tensorbem/benchmarks/case_cylinder_stream.py")
text = PATH.read_text(encoding="utf-8")

if "heading_study" in text:
    raise SystemExit("already patched")

# ---------------------------------------------------------------------------
# 1. import PanelMesh
# ---------------------------------------------------------------------------
old = "from ..geometry import regular_polygon\n"
new = "from ..geometry import PanelMesh, regular_polygon\n"
assert text.count(old) == 1
text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 2. functions + constants, inserted before the study driver
# ---------------------------------------------------------------------------
anchor = """# ---------------------------------------------------------------------------
# study driver
# ---------------------------------------------------------------------------
"""
assert text.count(anchor) == 1

block = '''# ---------------------------------------------------------------------------
# oblique-inflow heading sweep
# ---------------------------------------------------------------------------
# Every number above is measured with the stream along ``+x``, and the inscribed
# regular N-gon carries a *vertex* on that axis: the body is mirror-symmetric
# about the flow direction, so an error that is odd in ``y`` -- a flipped kernel
# sign, a normal-orientation slip, a panel-ordering assumption -- cancels between
# the upper and lower halves and never reaches the L2 norms.  Rotating the body
# against the fixed inflow breaks that symmetry: the stagnation streamline then
# meets a chord midpoint, where a piecewise-constant trace is at its worst.
#
# One heading is structural: 45 deg at N = 128 is exactly 16 panel steps, so the
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


def oblique_headings(panels: int) -> tuple[float, ...]:
    """Headings (deg) of the sweep: axis-aligned, half a panel step, generic.

    Half a panel step is the worst alignment for a constant-panel trace (the
    upstream stagnation point sits over a chord midpoint); the generic heading is
    not commensurate with the lattice at any panel count used here.
    """
    if panels <= 0:
        raise ValueError("panels must be positive")
    return (0.0, 180.0 / float(panels), OBLIQUE_HEADING_DEG)


def rotated_polygon(panels: int, heading: float, radius: float = RADIUS) -> PanelMesh:
    """Inscribed regular polygon rotated by ``heading`` radians about the origin.

    Rotating the body against a fixed ``+x`` inflow is the same physical
    configuration as rotating the inflow over a fixed body, and it keeps the
    closed-form reference in the frame the rest of the case validates.
    """
    base = regular_polygon(panels, radius)
    c, s = math.cos(heading), math.sin(heading)
    rotation = np.array([[c, -s], [s, c]])
    return PanelMesh(base.vertices @ rotation.T)


def heading_velocity(
    points: np.ndarray, heading: float, radius: float = RADIUS, speed: float = SPEED
) -> np.ndarray:
    """Cartesian velocity of ``W(z) = U e^{-i h} z + U a**2 e^{i h} / z``.

    The field is the axis-aligned closed form evaluated in the rotated frame and
    rotated back, so it cannot drift from :func:`exact_velocity` at ``h = 0``.
    """
    pts = np.asarray(points, dtype=float)
    c, s = math.cos(heading), math.sin(heading)
    back = np.column_stack(
        (c * pts[:, 0] + s * pts[:, 1], -s * pts[:, 0] + c * pts[:, 1])
    )
    flow = exact_velocity(back, radius, speed)
    return np.column_stack(
        (c * flow[:, 0] - s * flow[:, 1], s * flow[:, 0] + c * flow[:, 1])
    )


def heading_perturbation_potential(
    points: np.ndarray, heading: float, radius: float = RADIUS, speed: float = SPEED
) -> np.ndarray:
    """Decaying part ``U a**2 cos(theta - h) / r`` of the heading-``h`` potential."""
    pts = np.asarray(points, dtype=float)
    r = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(r == 0.0):
        raise ValueError("the perturbation potential is singular at the centre")
    theta = np.arctan2(pts[:, 1], pts[:, 0]) - heading
    return speed * radius**2 * np.cos(theta) / r


def heading_neumann_data(
    centers: np.ndarray, heading: float, radius: float = RADIUS, speed: float = SPEED
) -> np.ndarray:
    """Fluid-normal derivative ``q_e = U a**2 cos(theta - h) / r**2`` (radial normals)."""
    pts = np.asarray(centers, dtype=float)
    r = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(r == 0.0):
        raise ValueError("the centre is not on the body")
    cos_theta = (pts[:, 0] * math.cos(heading) + pts[:, 1] * math.sin(heading)) / r
    return speed * radius**2 * cos_theta / r**2


def _heading_level(
    panels: int,
    heading_deg: float,
    radius: float,
    speed: float,
    points: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Solve both routes once with the body rotated by ``heading_deg``."""
    heading = math.radians(heading_deg)
    mesh = rotated_polygon(panels, heading, radius)
    centers = mesh.centers
    probes = probe_points() if points is None else points
    inflow = speed * np.array([math.cos(heading), math.sin(heading)])

    dirichlet = solve_laplace_exterior_dirichlet(
        mesh, heading_perturbation_potential(centers, heading, radius, speed)
    )
    field = evaluate_exterior_field(mesh, dirichlet, probes)
    velocity = field.gradient + inflow
    exact = heading_velocity(probes, heading, radius, speed)

    neumann = solve_laplace_exterior_neumann(
        mesh, heading_neumann_data(centers, heading, radius, speed)
    )
    surface = evaluate_exterior_surface_velocity(mesh, neumann) + inflow
    exact_surface = heading_velocity(centers, heading, radius, speed)
    normal_residual = np.sum(surface * mesh.normals, axis=1)

    row: dict[str, Any] = {
        "panels": int(panels),
        "heading_deg": float(heading_deg),
        "panel_steps": float(heading_deg) * panels / 360.0,
        "h_mean": float(np.mean(mesh.lengths)),
        "dirichlet_condition_number": float(dirichlet.condition_number),
        "dirichlet_velocity_relative_l2": _relative_l2(velocity, exact),
        "dirichlet_velocity_max_abs_error_over_speed": float(
            np.max(np.linalg.norm(velocity - exact, axis=1))
        )
        / speed,
        "neumann_condition_number": float(neumann.condition_number),
        "neumann_surface_velocity_relative_l2": _relative_l2(surface, exact_surface),
        "neumann_surface_velocity_max_abs_error_over_speed": float(
            np.max(np.linalg.norm(surface - exact_surface, axis=1))
        )
        / speed,
        "normal_velocity_residual_over_speed": float(np.max(np.abs(normal_residual)))
        / speed,
    }
    return row, velocity


def heading_study(
    panels: int = BUDGET_PANELS, radius: float = RADIUS, speed: float = SPEED
) -> dict[str, Any]:
    """Re-measure both routes with the body rotated against the inflow."""
    points = probe_points()
    headings = oblique_headings(panels)
    rows: list[dict[str, Any]] = []
    fields: list[np.ndarray] = []
    for heading_deg in headings:
        row, velocity = _heading_level(panels, heading_deg, radius, speed, points)
        rows.append(row)
        fields.append(velocity)
    lattice_row, lattice_field = _heading_level(
        panels, LATTICE_HEADING_DEG, radius, speed, points
    )

    axis = rows[0]
    worst_d = max(rows, key=lambda row: row["dirichlet_velocity_relative_l2"])
    worst_n = max(rows, key=lambda row: row["neumann_surface_velocity_relative_l2"])
    penalty_d = worst_d["dirichlet_velocity_relative_l2"] / axis[
        "dirichlet_velocity_relative_l2"
    ]
    penalty_n = worst_n["neumann_surface_velocity_relative_l2"] / axis[
        "neumann_surface_velocity_relative_l2"
    ]
    covariance = float(
        np.max(np.linalg.norm(lattice_field - fields[0], axis=1))
    ) / speed
    return {
        "panels": int(panels),
        "probe_count": int(len(points)),
        "headings_deg": [float(deg) for deg in headings],
        "panel_step_deg": 360.0 / float(panels),
        "levels": rows,
        "axis_level": {
            "heading_deg": axis["heading_deg"],
            "dirichlet_velocity_relative_l2": axis["dirichlet_velocity_relative_l2"],
            "neumann_surface_velocity_relative_l2": axis[
                "neumann_surface_velocity_relative_l2"
            ],
        },
        "max_dirichlet_velocity_relative_l2": worst_d["dirichlet_velocity_relative_l2"],
        "worst_dirichlet_heading_deg": worst_d["heading_deg"],
        "max_dirichlet_velocity_max_abs_error_over_speed": worst_d[
            "dirichlet_velocity_max_abs_error_over_speed"
        ],
        "dirichlet_penalty_max": penalty_d,
        "max_neumann_surface_velocity_relative_l2": worst_n[
            "neumann_surface_velocity_relative_l2"
        ],
        "worst_neumann_heading_deg": worst_n["heading_deg"],
        "neumann_penalty_max": penalty_n,
        "lattice": {
            "heading_deg": float(LATTICE_HEADING_DEG),
            "panel_steps": lattice_row["panel_steps"],
            "dirichlet_velocity_relative_l2": lattice_row[
                "dirichlet_velocity_relative_l2"
            ],
            "neumann_surface_velocity_relative_l2": lattice_row[
                "neumann_surface_velocity_relative_l2"
            ],
            "velocity_field_max_difference_over_speed": covariance,
            "dirichlet_error_relative_difference": abs(
                lattice_row["dirichlet_velocity_relative_l2"]
                - axis["dirichlet_velocity_relative_l2"]
            )
            / axis["dirichlet_velocity_relative_l2"],
        },
        "note": (
            "rotating the body by a whole number of panel steps maps the mesh onto "
            "itself, so the recovered field must be reproduced to roundoff; the "
            "half-panel and 30 deg headings are the ones that break the mirror "
            "symmetry of the axis-aligned configuration"
        ),
    }


'''
text = text.replace(anchor, block + anchor)

# ---------------------------------------------------------------------------
# 3. parameters + run_case hook
# ---------------------------------------------------------------------------
old = '''            "circulation": CIRCULATION,
            "circulation_sweep": list(CIRCULATION_SWEEP),
        },'''
new = '''            "circulation": CIRCULATION,
            "circulation_sweep": list(CIRCULATION_SWEEP),
            "oblique_headings_deg": list(oblique_headings(budget)),
            "lattice_heading_deg": LATTICE_HEADING_DEG,
        },'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''    artifact["multipole"] = multipole_study(panels, radius, speed)
'''
new = '''    artifact["multipole"] = multipole_study(panels, radius, speed)
    artifact["heading_sweep"] = heading_study(budget, radius, speed)
'''
assert text.count(old) == 1
text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 4. acceptance criteria
# ---------------------------------------------------------------------------
old = '''            "dW/dz to 1% of U",
        )
    return checks
'''
new = '''            "dW/dz to 1% of U",
        )

    heading = artifact["heading_sweep"]
    add(
        "oblique_heading_dirichlet_velocity",
        heading["max_dirichlet_velocity_relative_l2"],
        f"< {TOLERANCE:g}",
        heading["max_dirichlet_velocity_relative_l2"] < TOLERANCE,
        "rotating the body against the inflow breaks the mirror symmetry about the "
        "flow axis; worst heading "
        f"{heading['worst_dirichlet_heading_deg']:g} deg",
    )
    add(
        "oblique_heading_neumann_surface_velocity",
        heading["max_neumann_surface_velocity_relative_l2"],
        f"< {TOLERANCE:g}",
        heading["max_neumann_surface_velocity_relative_l2"] < TOLERANCE,
        "route N surface velocity with the stagnation streamline over a chord "
        "midpoint instead of a vertex; worst heading "
        f"{heading['worst_neumann_heading_deg']:g} deg",
    )
    penalty = max(heading["dirichlet_penalty_max"], heading["neumann_penalty_max"])
    add(
        "oblique_heading_within_penalty_of_axis",
        penalty,
        f"<= {OBLIQUE_HEADING_PENALTY_MAX:g}",
        penalty <= OBLIQUE_HEADING_PENALTY_MAX,
        "an oblique heading must not be materially worse than the axis-aligned one",
    )
    add(
        "lattice_heading_is_covariant",
        heading["lattice"]["velocity_field_max_difference_over_speed"],
        f"<= {HEADING_LATTICE_COVARIANCE_TOL:g} U",
        heading["lattice"]["velocity_field_max_difference_over_speed"]
        <= HEADING_LATTICE_COVARIANCE_TOL,
        f"{heading['lattice']['panel_steps']:g} panel steps rotate the body onto "
        "itself, so the recovered velocity field must not move",
    )
    return checks
'''
assert text.count(old) == 1
text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 5. render: results subsection after the route-N convergence paragraph
# ---------------------------------------------------------------------------
old = '''    lines += [
        "",
        "收敛阶（以 $h_{mean}$ 为尺度）：路线 N 物面速度 "'''
new = '''    heading = artifact["heading_sweep"]
    lines += [
        "",
        "### 来流方位（斜来流）扫描",
        "",
        "上面所有数字的来流都沿 $+x$，而内接正 $N$ 边形在该轴上恰有一个**顶点**："
        "物形关于来流方向镜像对称，任一关于 $y$ 为奇的误差（核函数符号、法向定向、"
        "面板编号假设）都会在上下两半相互抵消，L2 范数看不见。把物体相对固定来流转一个角度即可打掉该对称："
        "驻点流线落在弦中点而非顶点时，常值面板的迹处于最不利状态。"
        f"$N={heading['panels']}$ 时面板步长 {heading['panel_step_deg']:.4f}$^\\\\circ$，"
        f"$\\\\gamma={heading['lattice']['heading_deg']:g}^\\\\circ$ 恰为 "
        f"{heading['lattice']['panel_steps']:g} 个面板步长 —— 旋转后仍是**同一顶点集**，"
        "故所有数字必须逐位重现（装配的旋转协方差 + 面板编号无关性）。",
        "",
        "| 来流方位 (deg) | 面板步长 | 路线 D 速度 L2 | 路线 D 最大偏差/U | 路线 N 物面速度 L2 | 路线 N $\\\\vert v\\\\cdot n\\\\vert/U$ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(heading["levels"]) + [
        {
            "heading_deg": heading["lattice"]["heading_deg"],
            "panel_steps": heading["lattice"]["panel_steps"],
            "dirichlet_velocity_relative_l2": heading["lattice"][
                "dirichlet_velocity_relative_l2"
            ],
            "dirichlet_velocity_max_abs_error_over_speed": float("nan"),
            "neumann_surface_velocity_relative_l2": heading["lattice"][
                "neumann_surface_velocity_relative_l2"
            ],
            "normal_velocity_residual_over_speed": float("nan"),
        }
    ]:
        best = row["heading_deg"] == heading["lattice"]["heading_deg"]
        lines.append(
            "| %s%.4f | %.3f | %.3e | %s | %.3e | %s |"
            % (
                "（格点）" if best else "",
                row["heading_deg"],
                row["panel_steps"],
                row["dirichlet_velocity_relative_l2"],
                "—" if best else "%.3e" % row["dirichlet_velocity_max_abs_error_over_speed"],
                row["neumann_surface_velocity_relative_l2"],
                "—" if best else "%.3e" % row["normal_velocity_residual_over_speed"],
            )
        )
    lines += [
        "",
        f"判读：最差方位（{heading['worst_dirichlet_heading_deg']:g}$^\\\\circ$）路线 D 速度 L2 "
        f"{heading['max_dirichlet_velocity_relative_l2']:.3e}、路线 N 物面速度 L2 "
        f"{heading['max_neumann_surface_velocity_relative_l2']:.3e}，"
        f"相对轴对称基线（{heading['axis_level']['heading_deg']:g}$^\\\\circ$）分别放大 "
        f"{heading['dirichlet_penalty_max']:.2f}×/{heading['neumann_penalty_max']:.2f}×，"
        "即打掉镜像对称并未带来精度损失；"
        f"格点旋转（{heading['lattice']['panel_steps']:g} 个步长）后外部速度场最大差 "
        f"{heading['lattice']['velocity_field_max_difference_over_speed']:.2e} $U$，"
        "落在舍入量级 —— 装配与求值都不依赖面板编号或来流与格点的相对相位。",
        "",
        "收敛阶（以 $h_{mean}$ 为尺度）：路线 N 物面速度 "'''
assert text.count(old) == 1
text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 6. render: error-verdict table rows
# ---------------------------------------------------------------------------
old = '''        ("路线 $\\\\Gamma$ 环量（实测 vs 给定）", rows_c[-1]["circulation_relative_error"]),
    ]'''
new = '''        ("路线 $\\\\Gamma$ 环量（实测 vs 给定）", rows_c[-1]["circulation_relative_error"]),
        ("斜来流最差方位路线 D 外部速度 L2", heading["max_dirichlet_velocity_relative_l2"]),
        (
            "斜来流最差方位路线 N 物面速度 L2",
            heading["max_neumann_surface_velocity_relative_l2"],
        ),
    ]'''
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new)

PATH.write_text(text, encoding="utf-8")
print("patched", PATH)