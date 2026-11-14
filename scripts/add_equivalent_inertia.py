"""Insert the hydrostatics-anchored equivalent-body inertia into rigid_body_mass."""
from pathlib import Path

SRC = Path("src/tensorbem/rigid_body_mass.py")
text = SRC.read_text()

NEW = '''
def waterplane_central_moments(
    waterplane_area: float,
    waterplane_moments: ArrayLike,
) -> FloatArray:
    """Second moments of the waterplane about its own centroid, ``[I_xx, I_yy]``.

    Convention pin (measured, not assumed): the ``waterplane_moments`` returned
    by :func:`tensorbem.hydrostatic.compute_waterplane_properties` are already
    **central** -- translating the hull by ``(7, -4) m`` leaves them unchanged
    at ``[0.7829, 0.7829] m^4`` while the origin-referred values would differ by
    ``A*y_c^2 = 50.3`` and ``A*x_c^2 = 153.9`` m^4.  The note in
    :func:`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` that calls
    the same integrals "taken about the coordinate origin" describes the
    textbook notation, not the returned numbers; the pin is
    ``tests/test_rigid_body_mass_equivalent_inertia.py::test_waterplane_moments_are_central``.

    A body property cannot depend on where the coordinate origin sits, so only
    central moments are accepted here.
    """
    area = float(waterplane_area)
    moments = np.asarray(waterplane_moments, dtype=float).reshape(-1)
    if moments.shape[0] != 3:
        raise ValueError("waterplane_moments must have three entries [I_xx, I_yy, I_xy]")
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("waterplane_area must be finite and positive")
    if not np.all(np.isfinite(moments)):
        raise ValueError("waterplane moments must be finite")
    i_xx, i_yy = float(moments[0]), float(moments[1])
    if i_xx <= 0.0 or i_yy <= 0.0:
        raise ValueError(
            "waterplane central second moments are not positive "
            f"(I_xx={i_xx:.6g} m^4, I_yy={i_yy:.6g} m^4)"
        )
    return np.array([i_xx, i_yy], dtype=float)


def equivalent_half_ellipsoid(
    displaced_volume: float,
    waterplane_area: float,
    waterplane_moments: ArrayLike,
) -> dict[str, object]:
    """Equivalent half-ellipsoid of a floating hull, from its hydrostatics only.

    When no inertia tensor is supplied a rigid-body mass matrix still has to be
    assembled, and the scale of its rotational block has to come from the
    *body*: it cannot come from the distance between the centre of gravity and
    whatever point the coordinate origin happens to be at (that fallback is
    origin-dependent by up to four orders of magnitude -- see
    ``benchmarks/rigid_body_mass_inertia_default.py``).  This function rebuilds
    the simplest solid whose hydrostatics match the computed ones: the lower
    half of an ellipsoid cut by an elliptical waterplane.

    For a waterplane ellipse with semi-axes ``(a, b)``, ``A = pi*a*b`` and the
    central transverse moment is ``I_xx = pi*a*b^3/4 = A*b^2/4``, so
    ``b = sqrt(4*I_xx/A)`` and ``a = sqrt(4*I_yy/A)``.  The half body has
    ``nabla = (2/3)*pi*a*b*c``, hence the draft ``c = 3*nabla/(2*pi*a*b)``.

    The reconstruction is *exact* for the spheroid (half-ellipsoid) hull family
    used by the verification cases, whose waterplane is an ellipse with
    ``pi*a*b = A``; ``waterplane_area_ratio`` reports ``pi*a*b/A``, which is
    ``1`` there and ``pi/3 = 1.047`` for a rectangular waterplane -- the
    measure of how far the hull is from the equivalent body.  For a rectangular
    barge the roll inertia then comes out ~25 % high and pitch/yaw ~12-20 % low
    (measured in the benchmark above); that is the price of a geometry-only
    fallback and it is reported, not hidden.

    Returns
    -------
    dict with ``semi_axes`` ``(a, b, c)`` (m), ``waterplane_area_ratio``
    (dimensionless) and ``waterplane_ellipse_area`` (m^2).
    """
    volume = float(displaced_volume)
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("displaced_volume must be finite and positive")
    area = float(waterplane_area)
    i_xx, i_yy = waterplane_central_moments(area, waterplane_moments)
    b_axis = math.sqrt(4.0 * i_xx / area)
    a_axis = math.sqrt(4.0 * i_yy / area)
    c_axis = 3.0 * volume / (2.0 * math.pi * a_axis * b_axis)
    ellipse_area = math.pi * a_axis * b_axis
    return {
        "semi_axes": np.array([a_axis, b_axis, c_axis], dtype=float),
        "waterplane_area_ratio": ellipse_area / area,
        "waterplane_ellipse_area": ellipse_area,
    }


def equivalent_inertia_from_hydrostatics(
    displaced_volume: float,
    waterplane_area: float,
    waterplane_moments: ArrayLike,
    waterplane_center: ArrayLike = (0.0, 0.0),
    *,
    mass: float | None = None,
    density: float = DEFAULT_DENSITY,
    center_of_gravity: ArrayLike | None = None,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    """Inertia tensor of the equivalent half-ellipsoid, referred to a CG.

    The solid returned by :func:`equivalent_half_ellipsoid` is uniform, so its
    second moments about the **waterplane-section centre** ``O`` are the
    classical solid-ellipsoid values (Lamb 1932 Ch. VI; Fossen 2011 App. B)

        ``I_x(O) = m*(b^2 + c^2)/5``, ``I_y(O) = m*(a^2 + c^2)/5``,
        ``I_z(O) = m*(a^2 + b^2)/5``,

    and its centroid sits ``3*c/8`` below ``O`` (a half body's centroid lies
    between the section and the apex, at ``3c/8`` from the section).  Hence the
    tensor about the centroid is ``I_x(O) - m*(3c/8)^2`` for roll and pitch and
    ``I_z(O)`` for yaw.  It is then carried to the reference point with one
    parallel-axis step and *back* to the supplied ``center_of_gravity`` with
    another, so that the returned ``inertia_cg`` is referred to exactly the
    point :func:`tensorbem.hydrostatic.compute_mass_matrix` expects.

    Parameters
    ----------
    mass :
        Body mass (kg).  Default: ``density * displaced_volume`` (flotation
        equilibrium).
    center_of_gravity :
        CG the tensor must be referred to (m).  Default: the centroid of the
        equivalent solid.
    reference_point :
        Point the tensor is also referred to in the returned
        ``inertia_reference`` (the origin of the generalized coordinates, m).

    Returns
    -------
    dict with ``semi_axes``, ``centroid``, ``inertia_cg`` (3x3, referred to
    ``center_of_gravity``), ``inertia_reference`` (3x3, referred to
    ``reference_point``), ``waterplane_area_ratio`` and ``mass``.

    Raises
    ------
    ValueError
        If the hydrostatics are degenerate (no waterplane), or if transferring
        the tensor to the supplied ``center_of_gravity`` destroys its positive
        definiteness -- the declared CG is then incompatible with the
        geometry-derived mass distribution, and refusing beats returning a
        non-physical tensor.
    """
    volume = float(displaced_volume)
    body_mass = float(density) * volume if mass is None else float(mass)
    if not math.isfinite(body_mass) or body_mass <= 0.0:
        raise ValueError("mass must be finite and positive")

    equivalent = equivalent_half_ellipsoid(
        volume, waterplane_area, waterplane_moments
    )
    a_axis, b_axis, c_axis = (float(v) for v in equivalent["semi_axes"])
    centre_xy = np.asarray(waterplane_center, dtype=float).reshape(-1)
    if centre_xy.shape[0] != 2 or not np.all(np.isfinite(centre_xy)):
        raise ValueError("waterplane_center must be a finite (x_c, y_c) pair")
    centroid = np.array(
        [centre_xy[0], centre_xy[1], -3.0 * c_axis / 8.0], dtype=float
    )

    lever = 3.0 * c_axis / 8.0
    about_centroid = np.diag(
        [
            body_mass * (b_axis ** 2 + c_axis ** 2) / 5.0 - body_mass * lever ** 2,
            body_mass * (a_axis ** 2 + c_axis ** 2) / 5.0 - body_mass * lever ** 2,
            body_mass * (a_axis ** 2 + b_axis ** 2) / 5.0,
        ]
    )

    def about(point: FloatArray) -> FloatArray:
        """Parallel-axis step from the centroid to ``point``.

        ``I(P)_xx = I(C)_xx + m*[(C_y - P_y)^2 + (C_z - P_z)^2]``: the transfer
        adds when the point is away from the mass centre, i.e. the theorem is
        ``I(P) = I(C) + m*(|C - P|^2 I3 - (C - P)(C - P)^T)`` and *not* the same
        expression with ``P - C`` (which would flip the sign of the transfer on
        every diagonal entry that is perpendicular to the offset).  Pinned by
        ``tests/test_rigid_body_mass_equivalent_inertia.py``.
        """
        d = centroid - point
        return about_centroid + body_mass * (
            float(d @ d) * np.eye(3) - np.outer(d, d)
        )

    ref = np.asarray(reference_point, dtype=float).reshape(3)
    cg = (
        centroid
        if center_of_gravity is None
        else np.asarray(center_of_gravity, dtype=float).reshape(3)
    )
    if not np.all(np.isfinite(cg)) or not np.all(np.isfinite(ref)):
        raise ValueError("center_of_gravity and reference_point must be finite")
    about_reference = about(ref)
    about_cg = about(cg)
    if float(np.min(np.linalg.eigvalsh(about_cg))) <= 0.0:
        raise ValueError(
            "the inertia of the equivalent half-ellipsoid is not positive "
            f"definite about the declared centre of gravity {_as_list(cg)} m; "
            "the declared CG is incompatible with the geometry-derived mass "
            "distribution"
        )

    return {
        "semi_axes": np.array([a_axis, b_axis, c_axis], dtype=float),
        "centroid": centroid,
        "inertia_cg": about_cg,
        "inertia_reference": about_reference,
        "waterplane_area_ratio": float(equivalent["waterplane_area_ratio"]),
        "mass": body_mass,
    }


'''

anchor = "\ndef _as_list(values: Sequence[float]) -> list[float]:"
assert anchor in text
assert "def equivalent_half_ellipsoid(" not in text
text = text.replace(anchor, NEW + anchor[1:], 1)

old_all = '    "mass_at_equilibrium",\n'
assert old_all in text
text = text.replace(
    old_all,
    old_all
    + '    "waterplane_central_moments",\n'
    + '    "equivalent_half_ellipsoid",\n'
    + '    "equivalent_inertia_from_hydrostatics",\n',
    1,
)
SRC.write_text(text)
print("patched", SRC)