from pathlib import Path

path = Path("src/tensorbem/benchmarks/case_cylinder_stream.py")
text = path.read_text()

# ------------------------------------------------------------------ criteria
anchor = '''    multipole = artifact["multipole"]
    for family, block in multipole["families"].items():'''
assert text.count(anchor) == 1, text.count(anchor)
checks = '''    azimuth = near["azimuth"]
    add(
        "near_wall_window_is_azimuth_independent",
        {
            "worst_over_levels": azimuth["window"]["worst_over_levels"],
            "levels": azimuth["window"]["levels"],
        },
        f"<= {NEAR_WALL_AZIMUTH_WINDOW_TOL:g} on every level",
        azimuth["window"]["worst_over_levels"] <= NEAR_WALL_AZIMUTH_WINDOW_TOL,
        "the documented window was measured above the panel midpoints only; "
        "walking the probe across the panel (midpoint -> vertex) must not break "
        "the same 1% tolerance, otherwise the window is an azimuth-specific "
        "statement rather than a validity claim",
    )
    add(
        "near_wall_exclusion_is_worse_at_vertices",
        {
            "clearance_over_h_max": azimuth["exclusion"]["clearance_over_h_max"],
            "min_azimuth_penalty": azimuth["exclusion"]["min_azimuth_penalty"],
            "worst_over_levels": azimuth["exclusion"]["worst_over_levels"],
            "levels": azimuth["exclusion"]["levels"],
        },
        f">= {NEAR_WALL_AZIMUTH_VERTEX_PENALTY:g} on every level",
        azimuth["exclusion"]["min_azimuth_penalty"] >= NEAR_WALL_AZIMUTH_VERTEX_PENALTY,
        "inside the exclusion zone the midpoint column is the *best* azimuth: at "
        "the vertex the piecewise-constant density is discontinuous across the "
        "kink, so the quoted midpoint number understates the worst case",
    )
    add(
        "near_wall_azimuth_window_is_not_extended_by_linear_density",
        azimuth["windows_including_azimuth"],
        f"constant and linear both {NEAR_WALL_WINDOW_FACTOR:g} h_max",
        azimuth["windows_including_azimuth"]["constant_density"]
        == NEAR_WALL_WINDOW_FACTOR
        and azimuth["windows_including_azimuth"]["linear_density"]
        == NEAR_WALL_WINDOW_FACTOR,
        "the midpoint-only study moves the 1% window of the first-moment basis in "
        "to 0.5 h_max; once the azimuth is free that gain disappears, so the "
        "linear basis must not be advertised as widening the usable window",
    )
    add(
        "near_wall_linear_density_is_worse_in_the_window",
        {
            "min_ratio": azimuth["linear_at_window"]["min_ratio"],
            "levels": azimuth["linear_at_window"]["levels"],
        },
        f">= {NEAR_WALL_LINEAR_WINDOW_GAIN_MAX:g} on every level",
        azimuth["linear_at_window"]["min_ratio"] >= NEAR_WALL_LINEAR_WINDOW_GAIN_MAX,
        "inside the documented window the first-moment basis is measurably worse "
        "than constant panels (its arc-length slope extrapolates the density "
        "across the vertex), which is the qualification the library docstring "
        "needs",
    )
    add(
        "near_wall_azimuth_midpoint_cross_check",
        azimuth["midpoint_cross_check"]["max_relative_difference"],
        f"<= {NEAR_WALL_AZIMUTH_FRAME_TOL:g}",
        azimuth["midpoint_cross_check"]["max_relative_difference"]
        <= NEAR_WALL_AZIMUTH_FRAME_TOL,
        "the azimuthal sweep and the radial ladder probe the same midpoint "
        "direction from different reference surfaces (chord vs circle); their "
        "agreement is what makes the two windows comparable",
    )

'''
text = text.replace(anchor, checks + anchor, 1)

# ----------------------------------------------------------------- document
doc_anchor = '''要再收窄窗口需曲边面板/更高阶求解。",
        "",
    ]
    rows_c = artifact["routes"]["circulation"]["levels"]
'''
assert text.count(doc_anchor) == 1, text.count(doc_anchor)
block = Path("outputs/azimuth_doc_block.txt").read_text().rstrip("\n")
block = "    " + block  # first line of the block is unindented in the source file
replacement = '''要再收窄窗口需曲边面板/更高阶求解。",
        "",
    ]
''' + block + '''
    rows_c = artifact["routes"]["circulation"]["levels"]
'''
text = text.replace(doc_anchor, replacement, 1)
path.write_text(text)
print("case module patched")

# --------------------------------------------------------- solver docstring
solver = Path("src/tensorbem/solver.py")
body = solver.read_text()
old = '''    ``linear_density=True`` removes that first-order limitation: the densities
    are expanded as ``sigma(s) = sigma_i + sigma'_i (s - L_i/2)`` on each panel,
    where ``sigma'_i`` is the arc-length slope of the panel trace estimated by
    :func:`tangential_slope`, and the first-moment panel integrals of
    :mod:`tensorbem.kernels` supply the extra term in closed form (value and
    target gradient, no extra quadrature). The near-wall error then falls with
    the square of the panel size instead of linearly; the far field is
    unchanged to the same order. ``CASE-BEM-LAPLACE-CYLINDER`` reports both
    modes side by side (``near_wall`` in
    ``benchmarks/verified/case_bem_laplace_cylinder/results.json``).
    """'''
assert body.count(old) == 1
new = '''    ``linear_density=True`` expands the densities as
    ``sigma(s) = sigma_i + sigma'_i (s - L_i/2)`` on each panel, where
    ``sigma'_i`` is the arc-length slope of the panel trace estimated by
    :func:`tangential_slope`, and the first-moment panel integrals of
    :mod:`tensorbem.kernels` supply the extra term in closed form (value and
    target gradient, no extra quadrature). That basis is *not* a uniform
    improvement; the extent of its gain is measured, not assumed
    (``near_wall`` in
    ``benchmarks/verified/case_bem_laplace_cylinder/results.json``):

    * above a panel *midpoint* it buys an order of magnitude at
      ``delta <= 0.5 h_max`` (cylinder: 2.8e-2 -> 3.2e-3 at ``delta = 0.5 h_max``)
      and moves the 1%-accurate window in from ``2 h_max`` to ``0.5 h_max``;
    * inside the documented window it is ~1.2x *worse* than constant panels, and
      its 1% window does not move in at all once the target is allowed to sit
      anywhere across the panel, not just above the midpoint;
    * towards a vertex it degrades sharply, because the arc-length slope
      extrapolates a density that is discontinuous across the kink: on the
      cylinder at ``delta = 0.1 h_max`` the worst-azimuth error is 1.40, 0.129
      and 1.26 on the 32/64/128-panel levels, i.e. not even monotone in ``h``.

    Use it for smooth (curved) boundaries and targets away from vertices; keep
    the constant basis plus a clearance guard as the default elsewhere.
    """'''
solver.write_text(body.replace(old, new, 1))
print("solver docstring corrected")
