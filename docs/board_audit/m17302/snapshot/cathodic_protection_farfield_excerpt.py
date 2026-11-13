# auditor excerpt (verbatim) of the m17265 delta in src/tensorbem/cathodic_protection.py
# subject branch agent/m17265 commit 912bec2; full file: /root/tp-agent/work/m17265/src/tensorbem/cathodic_protection.py
def fibonacci_sphere_directions(count: int) -> FloatArray:
    """Deterministic quasi-uniform directions on the unit sphere.

    Returns ``count`` unit vectors from the Fibonacci (golden-angle) spiral.
    The set is deterministic (no RNG state) and its directional average kills
    low-order spherical harmonics to a geometric accuracy that improves with
    ``count``, which is what :func:`far_field_current` needs.
    """
    count = int(count)
    if count < 4:
        raise ValueError("count must be at least 4")
    index = np.arange(count, dtype=float) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)        # cos(theta) uniform
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * index        # golden angle
    return np.column_stack(
        (np.sin(polar) * np.cos(azimuth),
         np.sin(polar) * np.sin(azimuth),
         np.cos(polar))
    )


def far_field_current(
    mesh: TriangleMesh,
    result: CathodicProtectionResult,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    radii: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0, 20.0),
    n_directions: int = 24,
    quadrature_order: int = 8,
) -> dict[str, Any]:
    """Total current recovered from the far field (Gauss-law cross-check).

    The surface route integrates the boundary current density over the panels
    (:attr:`CathodicProtectionResult.anode_current_total`).  This function
    recovers the *same* total from the exterior potential alone: outside the
    body Laplace's equation gives the multipole expansion

        phi(r, theta, phi) = I / (4 pi sigma r) + sum_{l>=1} c_l r^{-(l+1)} P_l ,

    so the directional mean of ``phi`` on a sphere of radius ``R`` isolates the
    monopole and the total current follows from

        I_R = 4 pi sigma R * <phi>_{|x|=R} .

    Averaging over a quasi-uniform direction set removes ``l >= 1`` up to the
    quadrature error, leaving a decay ``I_R = I_inf + c / R``; the reported
    ``extrapolated_current_A`` is the least-squares intercept in ``1/R``.
    Agreement between the surface and far-field routes is the discrete form of
    charge conservation (no current is generated or lost in the electrolyte).

    Parameters
    ----------
    mesh, result :
        Mesh and BEM solution from :func:`solve_cathodic_protection`.
    conductivity :
        Electrolyte conductivity (S/m).
    radii :
        Radii (m) of the enclosing spheres; all must exceed the body extent.
    n_directions :
        Number of directions per sphere (:func:`fibonacci_sphere_directions`).
    quadrature_order :
        Quadrature order for the exterior representation formula.

    Returns
    -------
    Dictionary with the per-radius currents, the monopole trace error and the
    ``1/R`` extrapolated current.
    """
    radii_arr = np.asarray(radii, dtype=float)
    if radii_arr.ndim != 1 or radii_arr.size < 2:
        raise ValueError("radii must be a 1-D sequence with at least two entries")
    if np.any(radii_arr <= 0.0):
        raise ValueError("radii must be positive")
    if np.any(np.diff(radii_arr) <= 0.0):
        raise ValueError("radii must be strictly increasing")
    extent = float(np.max(np.linalg.norm(mesh.vertices - mesh.vertices.mean(0), axis=1)))
    if float(radii_arr[0]) <= extent:
        raise ValueError(
            f"innermost radius {radii_arr[0]:g} m must exceed the body extent "
            f"({extent:.3f} m); the exterior representation is invalid on or inside"
        )
    directions = fibonacci_sphere_directions(n_directions)
    conductivity = float(conductivity)

    currents: list[float] = []
    monopole_rel_error: list[float] = []
    for radius in radii_arr:
        points = float(radius) * directions
        phi = evaluate_exterior_potential(
            mesh, result, points, quadrature_order=quadrature_order
        )
        mean_phi = float(np.mean(phi))
        current = _FOUR_PI * conductivity * float(radius) * mean_phi
        currents.append(current)
        # Deviations of the individual directions from the pure-monopole law
        # measure how well the enclosing sphere is "far enough".
        monopole = current / (_FOUR_PI * conductivity * float(radius))
        monopole_rel_error.append(
            float(np.max(np.abs(phi - monopole)) / abs(monopole))
            if monopole != 0.0 else float("nan")
        )

    # Least-squares intercept of I_R = I_inf + c / R.
    inv_r = 1.0 / radii_arr
    design = np.column_stack((np.ones_like(inv_r), inv_r))
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(currents), rcond=None)
    extrapolated = float(coefficients[0])
    fit = design @ coefficients
    residual = float(np.max(np.abs(fit - currents)) / abs(extrapolated)) if extrapolated else float("nan")
    surface_current = float(result.anode_current_total)
    return {
        "radii_m": [float(v) for v in radii_arr],
        "n_directions": int(n_directions),
        "current_by_radius_A": [float(v) for v in currents],
        "monopole_max_rel_error_by_radius": [float(v) for v in monopole_rel_error],
        "extrapolated_current_A": extrapolated,
        "extrapolation_intercept_shape": [float(v) for v in coefficients],
        "extrapolation_max_rel_residual": residual,
        "surface_current_A": surface_current,
        "far_field_vs_surface_rel_error": float(
            abs(extrapolated - surface_current) / abs(surface_current)
        ) if surface_current else float("nan"),
        "body_extent_m": extent,
    }


def verify_conducting_sphere_routes(
    subdivisions: int = 2,
    radius: float = 1.0,
    potential: float = -1.0,
    conductivity: float = SEAWATER_CONDUCTIVITY,
    radii: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0, 20.0),
    n_directions: int = 24,
    quadrature_order: int = 8,
) -> dict[str, Any]:
    """Cross-check the two current routes of the conducting-sphere anchor.

    Solves the uniformly-polarised sphere once and reports

    * the surface (panel-integral) current and its error against
      ``4 pi sigma a phi_a``,
    * the far-field (Gauss-law) current, its ``1/R`` extrapolated value and its
      error against the same closed form,
    * the surface-vs-far-field conservation error.

    This is the *second* independent route for the DNV-RP-B401 spreading
    resistance ``R = 1/(4 pi sigma a)``: the surface route trusts the boundary
    quadrature, the far-field route trusts only the decay of the exterior
    field, so a defect in either one shows up as a disagreement between them.
    """
    from .geometry3d import icosphere

    mesh = icosphere(int(subdivisions), radius=float(radius))
    surface_panels = np.arange(mesh.panel_count, dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    result = solve_cathodic_protection(
        mesh,
        empty,
        empty,
        surface_panels,
        protection_potential=float(potential),
        anode_potential=float(potential),
        conductivity=float(conductivity),
        quadrature_order=quadrature_order,
    )
    exact = _FOUR_PI * float(conductivity) * float(radius) * float(potential)
    far = far_field_current(
        mesh,
        result,
        conductivity=conductivity,
        radii=radii,
        n_directions=n_directions,
        quadrature_order=quadrature_order,
    )
    surface_current = float(result.anode_current_total)
    return {
        "case": "conducting_sphere_current_routes",
        "subdivisions": int(subdivisions),
        "n_panels": int(mesh.panel_count),
        "radius_m": float(radius),
        "potential_V": float(potential),
        "conductivity_S_m": float(conductivity),
        "exact_current_A": float(exact),
        "exact_spreading_resistance_ohm": float(1.0 / (_FOUR_PI * float(conductivity) * float(radius))),
        "surface_current_A": surface_current,
        "surface_rel_error": float(abs(surface_current - exact) / abs(exact)),
        "spreading_resistance_numeric_ohm": float(-float(potential) / surface_current),
        "far_field": far,
        "far_field_rel_error": float(abs(far["extrapolated_current_A"] - exact) / abs(exact)),
        "tolerance": 0.05,
        "pass": bool(
            abs(surface_current - exact) / abs(exact) < 0.05
            and abs(far["extrapolated_current_A"] - exact) / abs(exact) < 0.05
            and far["far_field_vs_surface_rel_error"] < 0.05
        ),
    }
