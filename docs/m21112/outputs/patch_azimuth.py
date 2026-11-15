from pathlib import Path

path = Path("src/tensorbem/benchmarks/case_cylinder_stream.py")
text = path.read_text()
edits = []

# ---------------------------------------------------------------- constants
anchor = "NEAR_WALL_ORDER_MIN = 0.5\n"
assert text.count(anchor) == 1
constants = anchor + '''
# Azimuthal (intra-panel) sweep.  The window above is measured on radial probes
# above the panel *midpoints*, which is where a constant-density panel is at its
# best; callers evaluate wherever the geometry puts them, so the same ladder is
# re-probed across one panel up to the vertex, where the two adjacent densities
# meet and the trace is not smooth.
NEAR_WALL_AZIMUTH_FACTORS: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25, 0.1)
NEAR_WALL_AZIMUTH_OFFSETS: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5)
# Six panels per level, strided: they reach the stagnation, shoulder and
# broadside azimuths without evaluating the closed-form kernels at every panel.
# ``tests/test_case_cylinder_stream.py`` checks the sampled maximum against the
# all-panel maximum.
NEAR_WALL_AZIMUTH_PANELS = 6
# The documented window must not depend on where across the panel the target
# sits, so the same 1% tolerance is demanded of the worst azimuth.
NEAR_WALL_AZIMUTH_WINDOW_TOL = 0.01
# Inside the exclusion zone the vertex direction is measurably worse than the
# midpoint direction: the midpoint-only column understates the worst case by at
# least this factor.
NEAR_WALL_AZIMUTH_VERTEX_PENALTY = 2.0
# The two studies probe from different reference surfaces (the circle for the
# radial ladder, the chord for the azimuthal sweep), so their midpoint errors
# agree to a few percent, not to roundoff.
NEAR_WALL_AZIMUTH_FRAME_TOL = 0.10
# First-moment (linear) density is not a uniform improvement: at the documented
# window it is measurably worse than constant panels, because the arc-length
# slope extrapolates a density that is discontinuous across the vertex.
NEAR_WALL_LINEAR_WINDOW_GAIN_MAX = 1.1
'''
text = text.replace(anchor, constants, 1)
edits.append("constants")

# ------------------------------------------------------- new study function
anchor = "\ndef gradient_kernel_crosscheck(\n"
assert text.count(anchor) == 1
function = '''

def _distance_to_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Distance from each point to the closed polygon boundary."""
    ends = np.roll(vertices, -1, axis=0)
    nearest = np.full(len(points), np.inf)
    for start, end in zip(vertices, ends):
        segment = end - start
        projection = np.clip(
            ((points - start) @ segment) / float(segment @ segment), 0.0, 1.0
        )
        closest = start + projection[:, None] * segment
        nearest = np.minimum(nearest, np.linalg.norm(points - closest, axis=1))
    return nearest


def near_wall_azimuth_study(
    radius: float = RADIUS,
    speed: float = SPEED,
    ladder: Sequence[int] = NEAR_WALL_LADDER,
    factors: Sequence[float] = NEAR_WALL_AZIMUTH_FACTORS,
    offsets: Sequence[float] = NEAR_WALL_AZIMUTH_OFFSETS,
) -> dict[str, Any]:
    """Re-probe the near-wall ladder across a panel, not only above its midpoint.

    The classic near-wall table (:func:`near_wall_study`) puts its probes on the
    radial rays through the panel midpoints.  That is the most favourable
    azimuth a constant-density panel offers, so the window it quotes is only a
    statement about midpoints.  This study walks the probe along one panel,
    ``zeta in [0, 0.5] h_max`` from the midpoint to the vertex, and pushes it out
    along the local outward normal (the bisector of the two adjacent panel
    normals at the vertex).  The offset distance actually realised is measured
    against the polygon, not assumed, because the bisector probe is slightly
    closer to the boundary than the nominal offset.

    Reported per level and clearance: the midpoint error (both panel bases), the
    worst error over the azimuth sweep, where that worst azimuth sits, and the
    ratio between them.  Aggregates answer three questions the midpoint table
    cannot:

    * does the documented window survive at every azimuth?
    * how much worse than the quoted exclusion-zone number is the worst azimuth?
    * does ``linear_density=True`` keep its advantage once the azimuth is free?
    """
    levels: list[dict[str, Any]] = []
    for panels in ladder:
        mesh = regular_polygon(int(panels), radius)
        centers, vectors, lengths = mesh.centers, mesh.vectors, mesh.lengths
        h_max = float(np.max(lengths))
        stride = max(1, int(panels) // NEAR_WALL_AZIMUTH_PANELS)
        select = np.arange(0, int(panels), stride)
        tangents = vectors / lengths[:, None]
        normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        following = np.roll(normals, -1, axis=0)
        bisectors = normals + following
        bisectors /= np.linalg.norm(bisectors, axis=1)[:, None]
        r = np.hypot(centers[:, 0], centers[:, 1])
        normal_derivative = speed * radius**2 * (centers[:, 0] / r) / r**2
        result = solve_laplace_exterior_neumann(mesh, normal_derivative)
        result_linear = solve_laplace_exterior_neumann(
            mesh, normal_derivative, linear_density=True
        )
        blocks: list[np.ndarray] = []
        tags: list[tuple[float, float]] = []
        for factor in factors:
            delta = float(factor) * h_max
            for offset in offsets:
                if float(offset) >= 0.5:
                    base = centers[select] + 0.5 * h_max * tangents[select]
                    direction = bisectors[select]
                else:
                    base = centers[select] + float(offset) * h_max * tangents[select]
                    direction = normals[select]
                blocks.append(base + delta * direction)
                tags.append((float(factor), float(offset)))
        points = np.vstack(blocks)
        exact = exact_velocity(points, radius, speed)
        exact_speed = np.linalg.norm(exact, axis=1)

        def _relative(field: Any) -> np.ndarray:
            total = field.gradient + np.array([speed, 0.0])
            return np.linalg.norm(total - exact, axis=1) / exact_speed

        constant = _relative(evaluate_exterior_field(mesh, result, points))
        linear = _relative(
            evaluate_exterior_field(mesh, result_linear, points, linear_density=True)
        )
        clearance = _distance_to_polygon(points, mesh.vertices) / h_max
        shape = (len(factors), len(offsets), len(select))
        constant = constant.reshape(shape)
        linear = linear.reshape(shape)
        clearance = clearance.reshape(shape)
        sweep: list[dict[str, Any]] = []
        for i, factor in enumerate(factors):
            rows = [
                {
                    "offset_over_h_max": float(offset),
                    "max_relative_velocity_error": float(constant[i, j].max()),
                    "max_relative_velocity_error_linear": float(linear[i, j].max()),
                    "min_clearance_over_h_max": float(clearance[i, j].min()),
                }
                for j, offset in enumerate(offsets)
            ]
            midpoint = next(
                row for row in rows if row["offset_over_h_max"] == 0.0
            )
            worst = max(rows, key=lambda row: row["max_relative_velocity_error"])
            worst_linear = max(
                rows, key=lambda row: row["max_relative_velocity_error_linear"]
            )
            midpoint_error = midpoint["max_relative_velocity_error"]
            midpoint_error_linear = midpoint["max_relative_velocity_error_linear"]
            sweep.append(
                {
                    "clearance_over_h_max": float(factor),
                    "offsets": rows,
                    "midpoint_error": midpoint_error,
                    "midpoint_error_linear": midpoint_error_linear,
                    "worst_error": worst["max_relative_velocity_error"],
                    "worst_error_linear": worst_linear[
                        "max_relative_velocity_error_linear"
                    ],
                    "worst_offset_over_h_max": worst["offset_over_h_max"],
                    "worst_offset_over_h_max_linear": worst_linear[
                        "offset_over_h_max"
                    ],
                    "azimuth_penalty": (
                        worst["max_relative_velocity_error"] / midpoint_error
                        if midpoint_error > 0.0
                        else float("inf")
                    ),
                    "azimuth_penalty_linear": (
                        worst_linear["max_relative_velocity_error_linear"]
                        / midpoint_error_linear
                        if midpoint_error_linear > 0.0
                        else float("inf")
                    ),
                    "min_clearance_over_h_max": min(
                        row["min_clearance_over_h_max"] for row in rows
                    ),
                }
            )
        levels.append(
            {
                "panels": int(panels),
                "h_max_over_radius": h_max / radius,
                "sampled_panels": int(len(select)),
                "panel_stride": int(stride),
                "sweep": sweep,
            }
        )

    def _at(level: dict[str, Any], factor: float) -> dict[str, Any]:
        return next(
            row for row in level["sweep"] if row["clearance_over_h_max"] == float(factor)
        )

    def _window_factor(key: str) -> float | None:
        factor: float | None = None
        for candidate in sorted(factors, key=float, reverse=True):
            worst = max(
                float(_at(level, candidate)[key]) for level in levels
            )
            if worst <= NEAR_WALL_AZIMUTH_WINDOW_TOL:
                factor = float(candidate)
            else:
                break
        return factor

    window = {
        "clearance_over_h_max": NEAR_WALL_WINDOW_FACTOR,
        "relative_error_tolerance": NEAR_WALL_AZIMUTH_WINDOW_TOL,
        "levels": [
            {
                "panels": level["panels"],
                "h_max_over_radius": level["h_max_over_radius"],
                "midpoint_error": _at(level, NEAR_WALL_WINDOW_FACTOR)["midpoint_error"],
                "worst_error": _at(level, NEAR_WALL_WINDOW_FACTOR)["worst_error"],
                "worst_error_linear": _at(level, NEAR_WALL_WINDOW_FACTOR)[
                    "worst_error_linear"
                ],
                "azimuth_penalty": _at(level, NEAR_WALL_WINDOW_FACTOR)[
                    "azimuth_penalty"
                ],
            }
            for level in levels
        ],
    }
    window["worst_over_levels"] = max(
        row["worst_error"] for row in window["levels"]
    )
    window["worst_over_levels_linear"] = max(
        row["worst_error_linear"] for row in window["levels"]
    )
    exclusion = {
        "clearance_over_h_max": NEAR_WALL_EXCLUSION_FACTOR,
        "levels": [
            {
                "panels": level["panels"],
                "midpoint_error": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "midpoint_error"
                ],
                "worst_error": _at(level, NEAR_WALL_EXCLUSION_FACTOR)["worst_error"],
                "worst_error_linear": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "worst_error_linear"
                ],
                "worst_offset_over_h_max": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "worst_offset_over_h_max"
                ],
                "azimuth_penalty": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "azimuth_penalty"
                ],
                "azimuth_penalty_linear": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "azimuth_penalty_linear"
                ],
            }
            for level in levels
        ],
    }
    exclusion["min_azimuth_penalty"] = min(
        row["azimuth_penalty"] for row in exclusion["levels"]
    )
    exclusion["worst_over_levels"] = max(
        row["worst_error"] for row in exclusion["levels"]
    )
    linear_gain = {
        "clearance_over_h_max": NEAR_WALL_WINDOW_FACTOR,
        "levels": [
            {
                "panels": level["panels"],
                "constant_worst": _at(level, NEAR_WALL_WINDOW_FACTOR)["worst_error"],
                "linear_worst": _at(level, NEAR_WALL_WINDOW_FACTOR)[
                    "worst_error_linear"
                ],
                "ratio_linear_over_constant": (
                    _at(level, NEAR_WALL_WINDOW_FACTOR)["worst_error_linear"]
                    / _at(level, NEAR_WALL_WINDOW_FACTOR)["worst_error"]
                ),
            }
            for level in levels
        ],
    }
    linear_gain["min_ratio"] = min(
        row["ratio_linear_over_constant"] for row in linear_gain["levels"]
    )
    linear_gain["max_ratio"] = max(
        row["ratio_linear_over_constant"] for row in linear_gain["levels"]
    )
    return {
        "ladder": [int(value) for value in ladder],
        "factors": [float(value) for value in factors],
        "offsets_over_h_max": [float(value) for value in offsets],
        "levels": levels,
        "window": window,
        "exclusion": exclusion,
        "linear_at_window": linear_gain,
        "windows_including_azimuth": {
            "relative_error_tolerance": NEAR_WALL_AZIMUTH_WINDOW_TOL,
            "constant_density": _window_factor("worst_error"),
            "linear_density": _window_factor("worst_error_linear"),
        },
        "probe_directions": (
            "along one panel from the midpoint to the vertex (bisector normal at "
            "the vertex), pushed out along the local outward normal"
        ),
        "note": (
            "zones are reported per azimuth: the midpoint column reproduces the "
            "radial ladder (from the circle, so it agrees to a few percent) while "
            "the worst column is the honest bound for an arbitrary target"
        ),
    }

'''
text = text.replace(anchor, function + "\ndef gradient_kernel_crosscheck(\n", 1)
edits.append("function")

# ------------------------------------------- wire it into near_wall_study
anchor = '''    return {
        "faceting": {
            "clearance_over_h_max": NEAR_WALL_EXCLUSION_FACTOR,'''
assert text.count(anchor) == 1
wiring = '''    azimuth = near_wall_azimuth_study(radius=radius, speed=speed, ladder=ladder)
    frame = []
    for index, level in enumerate(levels):
        azimuth_level = azimuth["levels"][index]
        row = next(
            row
            for row in azimuth_level["sweep"]
            if row["clearance_over_h_max"] == NEAR_WALL_WINDOW_FACTOR
        )
        midpoint = float(row["midpoint_error"])
        circle = float(window[index]["max_relative_velocity_error"])
        frame.append(
            {
                "panels": level["panels"],
                "midpoint_error_from_chord": midpoint,
                "midpoint_error_from_circle": circle,
                "relative_difference": (
                    abs(midpoint - circle) / circle if circle > 0.0 else float("inf")
                ),
            }
        )
    azimuth["midpoint_cross_check"] = {
        "levels": frame,
        "max_relative_difference": max(
            item["relative_difference"] for item in frame
        ),
        "tolerance": NEAR_WALL_AZIMUTH_FRAME_TOL,
        "note": (
            "the radial ladder probes at radius a + delta (measured from the "
            "circle) and the azimuthal sweep at chord midpoint + delta (measured "
            "from the polygon), so the same midpoint direction differs by the "
            "faceting offset; agreement within NEAR_WALL_AZIMUTH_FRAME_TOL "
            "confirms both studies measure the same regime"
        ),
    }
''' + anchor.replace('    return {', '    return {\n        "azimuth": azimuth,', 1)
text = text.replace(anchor, wiring, 1)
edits.append("wiring")
path.write_text(text)
print("applied:", edits)
