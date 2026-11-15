from pathlib import Path

path = Path("src/tensorbem/benchmarks/case_cylinder_stream.py")
text = path.read_text()

# ------------------------------------------------------------------ criteria
anchor = '''    multipole = artifact["multipole"]
    for family, block in multipole["families"].items():'''
assert text.count(anchor) == 1
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
        "kink, so the quoted number understates the worst case",
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

# --------------------------------------------------------------- document
anchor = "要再收窄窗口需曲边面板/更高阶求解。\",\n        \"\",\n    ]\n    rows_c = artifact[\"routes\"][\"circulation\"][\"levels\"]\n"
assert text.count(anchor) == 1
render = '''    azimuth = near["azimuth"]
    azimuth_window = azimuth["window"]
    azimuth_exclusion = azimuth["exclusion"]
    lines += [
        "",
        "**方位（面内）敏感性（第 4 轮补充）**：前面两张表的探针都落在**面板中点法向**上，"
        "那是常值面板最有利的方向；用户可以在任意位置求值。本轮把探针在同一面板内从中点"
        "（$\\\\zeta=0$）走到顶点（$\\\\zeta=0.5h_{max}$，顶点处取相邻面板法向的角平分线），"
        "逐层给出面内最差误差（线性基一列同测）。实测间隙由程序对折面逐段量距，非名义值：",
        "",
        "| N | $\\\\delta/h_{max}$ | 中点误差 | 面内最差 | 最差位置 $\\\\zeta/h_{max}$ | 顶点/中点 | 线性基面内最差 | 实测最小间隙/h |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for level in azimuth["levels"]:
        for row in level["sweep"]:
            lines.append(
                "| %d | %.2f | %.3e | %.3e | %.3f | %.2f | %.3e | %.3f |"
                % (
                    level["panels"],
                    row["clearance_over_h_max"],
                    row["midpoint_error"],
                    row["worst_error"],
                    row["worst_offset_over_h_max"],
                    row["azimuth_penalty"],
                    row["worst_error_linear"],
                    row["min_clearance_over_h_max"],
                )
            )
    lines += [
        "",
        f"判读一（窗内方位无关）：$\\\\delta={azimuth_window['clearance_over_h_max']:g}h_{{max}}$ 处逐层最差/中点比为 "
        + "、".join(f"{row['azimuth_penalty']:.2f}" for row in azimuth_window["levels"])
        + f"，最差 {azimuth_window['worst_over_levels']:.3e} 仍在 1% 容差内 —— 所以"
        "「$\\\\delta\\\\ge2h_{max}$」这条窗对任意方位成立，不限于中点法向。",
        "",
        f"判读二（排除区在中点方向被低估）：$\\\\delta={azimuth_exclusion['clearance_over_h_max']:g}h_{{max}}$ 处逐层中点误差 "
        + "、".join(f"{row['midpoint_error']:.3f}" for row in azimuth_exclusion["levels"])
        + "，而面内最差高达 "
        + "、".join(f"{row['worst_error']:.3f}" for row in azimuth_exclusion["levels"])
        + f"（顶点/中点比 {azimuth_exclusion['min_azimuth_penalty']:.2f}–"
        f"{max(row['azimuth_penalty'] for row in azimuth_exclusion['levels']):.2f}）。"
        "理由：常值密度在顶点处**不连续**，折面顶点相对真实圆的偏差又最直接地投影到该方向。"
        "结论：排除区里「18%」只是最有利方位的数字，贴壁任意方向应按 ~60% 或更大预留误差；"
        "守门参数照旧建议 $\\\\ge1$–$2h_{max}$。",
        "",
        f"判读三（一阶矩基的窗没有前移）：窗内 $\\\\delta={azimuth['linear_at_window']['clearance_over_h_max']:g}h_{{max}}$ 处线性基面内最差比常值 "
        f"大 {azimuth['linear_at_window']['min_ratio']:.2f}–{azimuth['linear_at_window']['max_ratio']:.2f}×；"
        f"含方位后 1% 窗对两种基都是 $\\\\delta\\\\ge"
        f"{azimuth['windows_including_azimuth']['constant_density']:g}h_{{max}}$"
        f"（线性 {azimuth['windows_including_azimuth']['linear_density']:g}）。"
        "线性基的优势只在**中点方向**、$\\\\delta\\\\le0.5h_{max}$ 处（上一表 3–9×），"
        "换到顶点方向在 $\\\\delta=0.1h_{max}$ 反而更差且逐层不单调"
        "（"
        + "、".join(f"{row['worst_error_linear']:.3f}" for row in azimuth_exclusion["levels"])
        + "，随加密 1.40→0.129→1.26）：弧长中心差分估计的斜率把密度外推过了顶点折角。"
        "故 `linear_density=True` 应作为「光滑曲边 + 离壁」的选项，不能当作通用改进（见 `evaluate_exterior_field` docstring）。",
        "",
        f"两条研究的中点在 $\\\\delta=2h_{{max}}$ 互相校核：同一方向、不同参考面（折面弦 vs 圆周）"
        f"的中误差最大相差 {azimuth['midpoint_cross_check']['max_relative_difference']:.2%}，"
        "与折面偏移 $a(1-\\\\cos\\\\pi/N)\\\\sim h^2/8$ 量级一致，说明两表量的是同一 regime。",
        "",
    ]
'''
text = text.replace(anchor, anchor.replace('    rows_c = artifact', '    rows_c = artifact', 1).replace('\n    rows_c = artifact', '', 1) + render + '    rows_c = artifact["routes"]["circulation"]["levels"]\n', 1)
path.write_text(text)
print("criteria + document rendered")

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
    target gradient, no extra quadrature).

    That basis is **not** a uniform improvement, and the measured evidence is
    narrower than "removes the first-order limitation"
    (``near_wall`` in ``benchmarks/verified/case_bem_laplace_cylinder/results.json``):

    * above a panel *midpoint* it buys one order of magnitude at
      ``delta <= 0.5 h_max`` (cylinder: 2.8e-2 -> 3.2e-3 at ``delta = 0.5 h_max``)
      and moves the 1%-accurate window in from ``2 h_max`` to ``0.5 h_max``;
    * inside the documented window it is 1.2x *worse* than constant panels, and
      its 1% window does not move in at all once the target is allowed to sit
      anywhere across the panel;
    * towards a vertex it degrades sharply, because the arc-length slope
      extrapolates a density that is discontinuous across the kink: on the
      cylinder at ``delta = 0.1 h_max`` the worst-azimuth error is 1.40, 0.129
      and 1.26 on the 32/64/128-panel levels, i.e. not even monotone in ``h``.

    Use it for smooth (curved) boundaries and targets away from vertices; keep
    the constant basis (and a clearance guard) as the default elsewhere.
    """'''
solver.write_text(body.replace(old, new, 1))
print("solver docstring corrected")
