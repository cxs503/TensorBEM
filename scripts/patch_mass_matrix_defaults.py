"""Patch hydrostatic/rao/ice_rao: geometry-anchored inertia instead of the origin-scaled guess."""
from pathlib import Path

# ---------------------------------------------------------------- hydrostatic
p = Path("src/tensorbem/hydrostatic.py")
t = p.read_text()

t = t.replace(
    "from .geometry3d import WettedHullMesh",
    "from .geometry3d import WettedHullMesh\nfrom .rigid_body_mass import (\n"
    "    equivalent_inertia_from_hydrostatics,\n    waterplane_central_moments,\n)",
    1,
)
t = t.replace("import math\n", "import math\nimport warnings\n", 1)

# 1. signature + docstring of compute_mass_matrix
t = t.replace(
    '''def compute_mass_matrix(
    mass: float,
    center_of_gravity: FloatArray,
    inertia: FloatArray | None = None,
    *,
    inertia_about: str = "cg",
) -> FloatArray:''',
    '''def compute_mass_matrix(
    mass: float,
    center_of_gravity: FloatArray,
    inertia: FloatArray | None = None,
    *,
    inertia_about: str = "cg",
    radius_of_gyration: float | None = None,
) -> FloatArray:''',
    1,
)

old_doc = '''    For simplicity, if inertia is not provided, we use a diagonal approximation
    with the moments of inertia computed from the mass and CG.'''
new_doc = '''    If ``inertia`` is not provided, a diagonal approximation
    ``I_cg = m*k^2*I3`` is used.  ``k`` must then come from the caller
    (``radius_of_gyration``), because a rigid body's inertia is a property of
    the *body* and no length scale of the body is available here:

    * ``radius_of_gyration`` given -- invariant by construction: translating
      the body (and the origin) leaves ``I_cg`` unchanged, so the assembled
      matrix obeys the change-of-origin law.
    * neither given -- the historical fallback ``k = 0.4*|r_g|`` is used and a
      ``RuntimeWarning`` is emitted.  That fallback scales the inertia with the
      distance from the coordinate origin to the CG, so it is *not* a physical
      inertia: moving the origin from the bow to midships changes it, and it
      was measured to vary by a factor of 30000 for one body
      (``benchmarks/rigid_body_mass_inertia_default.py``).  It is kept only so
      that old call sites keep running; every in-repo caller that has geometry
      now derives ``I_cg`` from the hull instead, through
      :func:`mass_matrix_from_hull_geometry` (used by
      :func:`tensorbem.rao.compute_rao_at_frequency` and
      :func:`tensorbem.ice_rao.compute_ice_rao`).'''
assert old_doc in t
t = t.replace(old_doc, new_doc, 1)

old_params = '''    inertia_about :
        ``"origin"`` or ``"cg"`` -- reference point of ``inertia``.'''
new_params = '''    inertia_about :
        ``"origin"`` or ``"cg"`` -- reference point of ``inertia``.
    radius_of_gyration :
        Radius of gyration ``k`` (m) for the ``inertia=None`` diagonal
        approximation, ``I_cg = m*k^2*I3``.  Invariant: it does not depend on
        where the origin or the CG sits.  Ignored when ``inertia`` is given.'''
assert old_params in t
t = t.replace(old_params, new_params, 1)

# 2. the fallback branch
old_branch = '''    if inertia is not None:
        j = np.asarray(inertia, dtype=float)
        if convention == "cg":
            # parallel-axis transfer to the body-frame origin
            j = j + mass * (float(cg @ cg) * np.eye(3) - np.outer(cg, cg))
    else:'''
new_branch = '''    if inertia is not None:
        j = np.asarray(inertia, dtype=float)
        if convention == "cg":
            # parallel-axis transfer to the body-frame origin
            j = j + mass * (float(cg @ cg) * np.eye(3) - np.outer(cg, cg))
    elif radius_of_gyration is not None:
        k = float(radius_of_gyration)
        if not math.isfinite(k) or k <= 0.0:
            raise ValueError("radius_of_gyration must be finite and positive")
        j = mass * k * k * np.eye(3) + mass * (
            float(cg @ cg) * np.eye(3) - np.outer(cg, cg)
        )
    else:
        warnings.warn(
            "compute_mass_matrix: inertia=None and radius_of_gyration=None -- "
            "falling back to the historical k = 0.4*|r_g| approximation, which "
            "makes the rotational block a function of the distance from the "
            "coordinate origin to the centre of gravity rather than a property "
            "of the body (measured variation up to 3e4 for one body). Pass an "
            "inertia tensor, a radius_of_gyration, or use "
            "mass_matrix_from_hull_geometry() with a wetted hull mesh.",
            RuntimeWarning,
            stacklevel=2,
        )'''
assert old_branch in t
t = t.replace(old_branch, new_branch, 1)

# 3. new public helper, inserted right after compute_mass_matrix's return
anchor = '''    m[3:6, 3:6] = m22

    return m
'''
new_helper = anchor + '''

def mass_matrix_from_hull_geometry(
    hydrostatic_props: HydrostaticProperties,
    mass: float,
    center_of_gravity: FloatArray,
    *,
    density: float = _DEFAULT_DENSITY,
    reference_point: FloatArray = (0.0, 0.0, 0.0),
    inertia: FloatArray | None = None,
) -> FloatArray:
    """6x6 rigid-body mass matrix with the inertia taken from the hull geometry.

    When a caller does not know the inertia tensor, the hull mesh still does:
    its displaced volume and waterplane second moments define an equivalent
    half-ellipsoid (:func:`tensorbem.rigid_body_mass.equivalent_inertia_from_hydrostatics`)
    whose inertia is a property of the body and therefore invariant under a
    change of coordinate origin.

    The reconstruction is exact -- down to the panelisation error of the
    waterplane integral -- for hulls whose waterplane is an ellipse (spheroid /
    hemisphere verification hulls: measured 0.32 % / 0.21 % / 0.16 % on roll /
    pitch / yaw at ``subdivisions=4``), and it is a *declared* approximation for
    box-like hulls, where the second moments do not determine the inertia: a
    rectangular barge comes out ~16-20 % low on all three rotational entries
    (``benchmarks/rigid_body_mass_inertia_default.py``).  A ``RuntimeWarning``
    is emitted whenever ``pi*a*b/A`` departs from 1, i.e. whenever the hull is
    measurably far from the equivalent body, so the caller is never silently
    given an approximation whose accuracy class is unknown.

    Parameters
    ----------
    hydrostatic_props :
        Result of :func:`compute_hydrostatic_restoring_matrix` for the same
        hull, mass and CG.
    mass :
        Body mass (kg).
    center_of_gravity :
        CG position (m) the returned matrix is referred to (through the
        parallel-axis term of :func:`compute_mass_matrix`).
    density :
        Fluid density used only if ``mass`` is None -- unused here, kept for
        symmetry with :func:`compute_mass_matrix` call sites.
    reference_point :
        Point the generalized coordinates are taken about (m), default the
        body-frame origin.
    inertia :
        Optional explicit tensor; when given it is passed through unchanged, so
        this function can be used as the single assembly entry point.

    Returns
    -------
    M :
        6x6 rigid-body mass matrix (kg, kg*m).
    """
    if inertia is not None:
        return compute_mass_matrix(
            mass, center_of_gravity, inertia, inertia_about="cg"
        )
    derived = equivalent_inertia_from_hydrostatics(
        hydrostatic_props.displaced_volume,
        hydrostatic_props.waterplane_area,
        hydrostatic_props.waterplane_moments,
        hydrostatic_props.waterplane_center,
        mass=mass,
        density=density,
        center_of_gravity=center_of_gravity,
        reference_point=reference_point,
    )
    ratio = float(derived["waterplane_area_ratio"])
    if abs(ratio - 1.0) > 2.0e-2:
        warnings.warn(
            "mass_matrix_from_hull_geometry: the waterplane is not elliptical "
            f"(pi*a*b/A = {ratio:.4f}), so the equivalent-half-ellipsoid inertia "
            "is an approximation -- for box-like hulls the rotational entries "
            "are ~16-20 % low.  Pass an explicit inertia tensor for such hulls.",
            RuntimeWarning,
            stacklevel=2,
        )
    return compute_mass_matrix(
        mass, center_of_gravity, np.asarray(derived["inertia_cg"], dtype=float),
        inertia_about="cg",
    )
'''
assert anchor in t
t = t.replace(anchor, new_helper, 1)
p.write_text(t)
print("patched", p)

# ---------------------------------------------------------------------- rao.py
p = Path("src/tensorbem/rao.py")
t = p.read_text()
t = t.replace(
    "    compute_mass_matrix,",
    "    compute_mass_matrix,\n    mass_matrix_from_hull_geometry,",
    1,
)
old = "    # Compute mass matrix\n    m = compute_mass_matrix(mass, center_of_gravity, inertia)"
new = '''    # Compute mass matrix.  With no inertia tensor the rotational block is
    # derived from the hull hydrostatics (invariant under a change of origin)
    # rather than from the distance between the origin and the CG; see
    # ``tensorbem.hydrostatic.mass_matrix_from_hull_geometry``.
    if inertia is None:
        m = mass_matrix_from_hull_geometry(
            hydrostatic_props, mass, center_of_gravity, density=density
        )
    else:
        m = compute_mass_matrix(mass, center_of_gravity, inertia)'''
assert old in t
t = t.replace(old, new, 1)
p.write_text(t)
print("patched", p)

# ------------------------------------------------------------------- ice_rao.py
p = Path("src/tensorbem/ice_rao.py")
t = p.read_text()
assert "    compute_mass_matrix,\n" in t
t = t.replace(
    "    compute_mass_matrix,\n",
    "    compute_mass_matrix,\n    mass_matrix_from_hull_geometry,\n",
    1,
)
old = "    mass_matrix = compute_mass_matrix(mass_value, cg, inertia)"
new = '''    if inertia is None:
        # Same geometry-anchored default as the open-water RAO path.
        mass_matrix = mass_matrix_from_hull_geometry(
            hydrostatic_props, mass_value, cg, density=density
        )
    else:
        mass_matrix = compute_mass_matrix(mass_value, cg, inertia)'''
assert old in t
t = t.replace(old, new, 1)
p.write_text(t)
print("patched", p)