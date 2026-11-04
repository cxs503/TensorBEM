"""Splice the complete (coupled) hydrostatic restoring matrix into hydrostatic.py.

Run once from the worktree root; it is idempotent by assertion (fails loudly if the
anchor text is not found exactly once).
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "tensorbem" / "hydrostatic.py"

OLD_MODULE_CLAIM = """For small-angle linearization, all off-diagonal terms are zero (no cross-coupling
between heave/roll/pitch, and surge/sway/yaw have no hydrostatic restoring).
"""

NEW_MODULE_CLAIM = """For small-angle linearization the matrix is symmetric and surge/sway/yaw carry no
hydrostatic restoring diagonal, but the off-diagonal terms are **not** zero in
general: they are the waterplane first moments (``C[2,3]``, ``C[2,4]``) and the
waterplane product moment (``C[3,4]``), so a hull whose waterplane centroid is
displaced from the origin (``LCF``/``TCF`` != 0) couples heave into roll and
pitch. See :func:`compute_hydrostatic_restoring_matrix`.
"""

OLD_FUNC_HEAD = '''def compute_hydrostatic_restoring_matrix(
    hull: WettedHullMesh,'''

NEW_BLOCK = '''def _waterplane_integrals_about_origin(
    wp_area: float,
    wp_center: FloatArray,
    wp_moments: FloatArray,
) -> tuple[float, float, float, float, float]:
    """Return ``(S_x, S_y, I_xx, I_yy, I_xy)``: waterplane integrals about the origin.

    :func:`compute_waterplane_properties` reports the area, the centroid and the
    second moments **about the centroid**. The restoring matrix needs the same
    integrals about the coordinate origin, obtained with the parallel-axis
    theorem::

        S_x  = A x_c,          S_y  = A y_c
        I_xx = I_xx^c + A y_c^2
        I_yy = I_yy^c + A x_c^2
        I_xy = I_xy^c + A x_c y_c

    Maintaining one source of truth (the centroidal properties) keeps the
    waterline-integration code single-purpose; the shift is exact algebra, so no
    accuracy is lost.
    """
    area = float(wp_area)
    x_c = float(wp_center[0])
    y_c = float(wp_center[1])
    i_xx_c = float(wp_moments[0])
    i_yy_c = float(wp_moments[1])
    i_xy_c = float(wp_moments[2])

    s_x = area * x_c
    s_y = area * y_c
    i_xx = i_xx_c + area * y_c**2
    i_yy = i_yy_c + area * x_c**2
    i_xy = i_xy_c + area * x_c * y_c
    return s_x, s_y, i_xx, i_yy, i_xy


def compute_hydrostatic_restoring_matrix(
    hull: WettedHullMesh,
    mass: float,
    center_of_gravity: FloatArray,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> HydrostaticProperties:
    """Compute the 6×6 hydrostatic restoring matrix ``C``.

    The restoring matrix is the Hessian of the hydrostatic potential energy in
    the six rigid-body modes ``(surge, sway, heave, roll, pitch, yaw)`` at
    equilibrium flotation. With the waterplane integrals taken *about the
    coordinate origin*

        ``A = ∫∫ dA``, ``S_x = ∫∫ x dA``, ``S_y = ∫∫ y dA``,
        ``I_xx = ∫∫ y² dA``, ``I_yy = ∫∫ x² dA``, ``I_xy = ∫∫ x y dA``,

    and ``∇`` the displaced volume, ``(x_B, y_B, z_B)`` the centre of buoyancy and
    ``(x_G, y_G, z_G)`` the centre of gravity, the non-zero entries are
    (Journée & Massie, *Offshore Hydromechanics*, §3.2.2)::

        C[2,2]          = ρ g A                      heave
        C[2,3] = C[3,2] = -ρ g S_y                   heave-roll
        C[2,4] = C[4,2] = -ρ g S_x                   heave-pitch
        C[3,3]          = ρ g I_xx + ρ g ∇ z_B - m g z_G   roll
        C[3,4] = C[4,3] = -ρ g I_xy                  roll-pitch
        C[3,5] = C[5,3] = -ρ g ∇ x_B + m g x_G       roll-yaw
        C[4,4]          = ρ g I_yy + ρ g ∇ z_B - m g z_G   pitch
        C[4,5] = C[5,4] = -ρ g ∇ y_B + m g y_G       pitch-yaw

    Surge, sway and yaw have no hydrostatic restoring diagonal, and the matrix is
    symmetric because it is a Hessian. The diagonal roll/pitch entries are the
    familiar metacentric stiffness: ``C[3,3] = ρ g ∇ GM_T`` with
    ``GM_T = I_T^centroid / ∇ + z_B - z_G`` and ``I_T = I_xx``.

    The off-diagonal entries are not zero in general. ``-C[2,3] / (ρ g)`` and
    ``-C[2,4] / (ρ g)`` are the beamwise and lengthwise first moments of the
    waterplane, so a hull whose waterplane centroid is displaced from the origin
    (``TCF``/``LCF`` != 0 — i.e. almost every real ship) couples heave into roll
    and pitch; ``C[3,4]`` couples roll and pitch through the waterplane product
    moment. A body whose waterplane centroid sits on the origin (a symmetric
    barge, a floating hemisphere) and whose waterplane has a symmetry axis gets a
    strictly diagonal heave/roll/pitch block, which is the case every earlier
    validation case in this repository exercised.

    Parameters
    ----------
    hull :
        Wetted hull mesh (intersection of floating body with free surface).
    mass :
        Mass of the floating body (kg). For equilibrium, ``mass = ρ ∇``; see the
        notes below.
    center_of_gravity :
        Coordinates of center of gravity (m).
    density :
        Fluid density (kg/m³). Default: 1025.0 (seawater).
    gravity :
        Gravitational acceleration (m/s²). Default: 9.80665.

    Returns
    -------
    properties :
        Hydrostatic properties including the restoring matrix.

    Notes
    -----
    The gravity terms are evaluated at the equilibrium mass ``m = ρ ∇``. This is
    the same assumption the metacentric form ``GM = KB + BM - KG`` makes, and it
    keeps the result independent of a mass supplied out of equilibrium (a
    floating body out of equilibrium is not a restoring problem — it
    accelerates). ``mass`` is therefore accepted for API compatibility and is
    not consumed; the equilibrium condition is what fixes `m`.

    Fully submerged bodies (no waterplane) have no heave, roll or pitch restoring
    from the free surface: their ``C`` block is exactly zero.
    """
    center_of_gravity = np.asarray(center_of_gravity, dtype=float)

    # Waterplane geometry and submerged volume
    wp_area, wp_center, wp_moments = compute_waterplane_properties(hull)
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)

    s_x, s_y, i_xx, i_yy, i_xy = _waterplane_integrals_about_origin(
        wp_area, wp_center, wp_moments
    )

    x_g, y_g, z_g = (float(value) for value in center_of_gravity)
    x_b, y_b, z_b = (float(value) for value in center_of_buoyancy)

    rho_g = density * gravity
    # Equilibrium mass (m = ρ ∇): potential energy of the body weight.
    weight = rho_g * volume

    c = np.zeros((6, 6), dtype=float)

    # Heave-roll and heave-pitch: the waterplane first moments.
    c[2, 2] = rho_g * wp_area
    c[2, 3] = c[3, 2] = -rho_g * s_y
    c[2, 4] = c[4, 2] = -rho_g * s_x

    # Roll: waterplane transverse inertia (about the origin) plus the
    # buoyancy/weight couple through the vertical separation of B and G.
    vertical_separation = weight * (z_b - z_g)
    c[3, 3] = rho_g * i_xx + vertical_separation
    c[4, 4] = rho_g * i_yy + vertical_separation

    # Roll-pitch: waterplane product of inertia about the origin.
    c[3, 4] = c[4, 3] = -rho_g * i_xy

    # Roll-yaw / pitch-yaw: horizontal separation of B and G (zero at equilibrium).
    c[3, 5] = c[5, 3] = weight * (x_g - x_b)
    c[4, 5] = c[5, 4] = weight * (y_g - y_b)

    return HydrostaticProperties(
        displaced_volume=volume,
        center_of_buoyancy=center_of_buoyancy,
        waterplane_area=wp_area,
        waterplane_center=wp_center,
        waterplane_moments=wp_moments,
        restoring_matrix=c,
    )
'''


def main() -> None:
    text = SRC.read_text()
    assert text.count(OLD_MODULE_CLAIM) == 1, "module claim anchor not unique"
    text = text.replace(OLD_MODULE_CLAIM, NEW_MODULE_CLAIM)

    marker = "def compute_mass_matrix("
    head, sep, tail = text.partition(marker)
    assert sep, "compute_mass_matrix anchor missing"
    start = head.rindex(OLD_FUNC_HEAD)
    assert start > 0, "function head anchor missing"
    head = head[:start] + NEW_BLOCK + "\n\n"
    SRC.write_text(head + sep + tail)
    print("patched", SRC)


if __name__ == "__main__":
    main()