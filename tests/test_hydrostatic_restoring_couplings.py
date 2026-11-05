"""Guards for the coupled hydrostatic restoring matrix.

Before this mission ``compute_hydrostatic_restoring_matrix`` filled in only the
heave, roll and pitch diagonals and its docstring asserted that *"all
off-diagonal terms are zero under the small-angle assumption"*. That is true only
for a hull whose waterplane centroid sits on the coordinate origin and whose
waterplane has a symmetry axis — i.e. for the symmetric barges and hemispheres
that every earlier validation case in this repository used. It is false for a
real ship, whose ``LCF``/``TCF`` are almost never zero: ``C[2,4]/ρg`` *is* the
lengthwise first moment of the waterplane, so heave and pitch are statically
coupled, and ``rao.py`` inverts the full 6×6 system.

Each reference below is independent of the implementation it polices:

* an **exact closed form** for a rectangular waterplane (area, first moments and
  second moments of a rectangle are rational numbers), and
* an **independent numerical Hessian** of the hydrostatic potential energy
  (``tensorbem.benchmarks.hydrostatic_energy_reference``), which knows only
  Archimedes plus the weight of the body and evaluates the submerged volume by
  polygon clipping — it never touches a waterplane formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_analytic import box_hull_mesh
from tensorbem.benchmarks.hydrostatic_energy_reference import (
    restoring_matrix_from_energy,
)
from tensorbem.geometry3d import WettedHullMesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
)

RHO = 1025.0
G = 9.80665

#: Mode indices (0-based): 2 = heave, 3 = roll, 4 = pitch.
HEAVE, ROLL, PITCH = 2, 3, 4


def _shifted_box(
    length: float, beam: float, draft: float, offset: tuple[float, float]
) -> WettedHullMesh:
    """A ``length x beam x draft`` barge whose waterplane is shifted in ``x``/``y``.

    The wetted surface is an exact faceted box at every refinement level, so all
    waterplane integrals of the shifted rectangle are exact rational numbers and
    the *only* source of error in the comparison is the implementation itself.
    Shifting the mesh (rather than building a different body) puts the waterplane
    centroid at ``(x_1 + L/2, y_1 + B/2)`` while keeping the displaced volume and
    the waterplane moments known exactly.
    """
    hull = box_hull_mesh(length, beam, draft)
    offset_vector = np.array([offset[0], offset[1], 0.0])
    return WettedHullMesh(hull.vertices + offset_vector, hull.faces)


def _rectangle_reference(
    length: float, beam: float, draft: float, offset: tuple[float, float], cg_z: float
) -> dict:
    """Exact restoring-matrix entries of a shifted rectangular barge.

    ``box_hull_mesh`` is centred on the origin, so after the shift the
    waterplane is the rectangle ``x in [x0-L/2, x0+L/2]``,
    ``y in [y0-B/2, y0+B/2]``, ``z in [-T, 0]``, i.e. its centroid is ``(x0, y0)``:

        A    = L B,      S_x = A (x0 + L/2),      S_y = A (y0 + B/2)
        I_xx = A [(y0+B/2)^2 + B^2/12],   I_yy = A [(x0+L/2)^2 + L^2/12]
        I_xy = A (x0+L/2)(y0+B/2),        ∇ = A T,   z_B = -T/2

    so the four non-zero coupling mechanisms are each isolated: ``S_y`` drives
    heave-roll, ``S_x`` drives heave-pitch, and ``I_xy`` drives roll-pitch.
    """
    x0, y0 = offset
    area = length * beam
    volume = area * draft
    x_c = x0
    y_c = y0
    s_x = area * x_c
    s_y = area * y_c
    i_xx = area * (y_c**2 + beam**2 / 12.0)
    i_yy = area * (x_c**2 + length**2 / 12.0)
    i_xy = area * x_c * y_c
    z_b = -0.5 * draft
    rho_g = RHO * G
    return {
        "volume": volume,
        "c_heave": rho_g * area,
        "c_heave_roll": rho_g * s_y,
        "c_heave_pitch": -rho_g * s_x,
        "c_roll": rho_g * i_xx + rho_g * volume * (z_b - cg_z),
        "c_pitch": rho_g * i_yy + rho_g * volume * (z_b - cg_z),
        "c_roll_pitch": -rho_g * i_xy,
    }


def test_eccentric_barge_matches_the_exact_rectangle_closed_form() -> None:
    """A barge with ``LCF``/``TCF`` both non-zero: every entry is exact.

    The barge is 2 m x 1 m x 1 m with its waterplane at ``x in [1.3, 3.3]``,
    ``y in [0.45, 1.45]``. With the old diagonal-only matrix all six quantities
    below other than the three diagonals came out exactly zero, and the roll and
    pitch diagonals were short by ``ρ g A y_c^2`` and ``ρ g A x_c^2`` (the
    parallel-axis shift of the waterplane inertia) — 71 % and 84 % here.
    """
    length, beam, draft = 2.0, 1.0, 1.0
    offset = (1.3, 0.45)
    cg_z = -0.5
    hull = _shifted_box(length, beam, draft, offset)
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)

    reference = _rectangle_reference(length, beam, draft, offset, cg_z)
    assert volume == pytest.approx(reference["volume"], rel=1.0e-14)
    # Equilibrium: the centre of gravity sits directly above the centre of buoyancy.
    assert center_of_buoyancy[0] == pytest.approx(offset[0], abs=1.0e-12)
    assert center_of_buoyancy[1] == pytest.approx(offset[1], abs=1.0e-12)

    cg = np.array([offset[0], offset[1], cg_z])
    props = compute_hydrostatic_restoring_matrix(hull, RHO * volume, cg, RHO, G)
    c = props.restoring_matrix

    assert c[HEAVE, HEAVE] == pytest.approx(reference["c_heave"], rel=1.0e-12)
    assert c[HEAVE, ROLL] == pytest.approx(reference["c_heave_roll"], rel=1.0e-12)
    assert c[HEAVE, PITCH] == pytest.approx(reference["c_heave_pitch"], rel=1.0e-12)
    assert c[ROLL, ROLL] == pytest.approx(reference["c_roll"], rel=1.0e-12)
    assert c[PITCH, PITCH] == pytest.approx(reference["c_pitch"], rel=1.0e-12)
    assert c[ROLL, PITCH] == pytest.approx(reference["c_roll_pitch"], rel=1.0e-12)

    # The couplings are large enough to matter: the heave-pitch entry is 1.3x the
    # heave stiffness itself for this (deliberately extreme) LCF.
    assert abs(c[HEAVE, PITCH]) > c[HEAVE, HEAVE]


def test_restoring_matrix_is_symmetric() -> None:
    """``C`` is a Hessian of a potential, so it must be symmetric to machine zero."""
    hull = _shifted_box(2.0, 1.0, 1.0, (1.3, 0.45))
    volume, _ = compute_displaced_volume_and_buoyancy(hull)
    cg = np.array([1.3, 0.45, -0.5])
    c = compute_hydrostatic_restoring_matrix(hull, RHO * volume, cg, RHO, G).restoring_matrix
    assert np.max(np.abs(c - c.T)) == 0.0


@pytest.mark.parametrize("offset", [(1.3, 0.45), (0.0, 0.0), (-0.8, 0.2)])
def test_matrix_matches_the_independent_energy_hessian(offset: tuple[float, float]) -> None:
    """Cross-check every entry against the potential-energy Hessian.

    This is the guard that matters: the closed-form test above restates the same
    waterplane-integral formula the implementation uses, so it cannot catch a
    wrong formula. The energy reference solves the physics directly (Archimedes
    plus body weight on a clipped mesh) and reproduces the whole matrix to 1e-8
    relative — including the opposite signs of the heave-roll and heave-pitch
    entries, which is exactly the kind of detail a formula clone would share with
    a bug.
    """
    hull = _shifted_box(2.0, 1.0, 1.0, offset)
    volume, _ = compute_displaced_volume_and_buoyancy(hull)
    cg = np.array([offset[0], offset[1], -0.5])
    props = compute_hydrostatic_restoring_matrix(hull, RHO * volume, cg, RHO, G)
    reference = restoring_matrix_from_energy(
        hull.vertices, hull.faces, RHO * volume, cg, RHO, G
    )
    scale = float(np.max(np.abs(reference)))
    assert np.max(np.abs(props.restoring_matrix - reference)) / scale < 1.0e-6


def test_yaw_rows_and_columns_are_identically_zero() -> None:
    """Vertical forces exert no moment about a vertical axis: yaw never restores.

    Checked on the eccentric barge, where every other coupling is non-zero, so
    the zero cannot come from a symmetry of the body.
    """
    hull = _shifted_box(2.0, 1.0, 1.0, (1.3, 0.45))
    volume, _ = compute_displaced_volume_and_buoyancy(hull)
    cg = np.array([1.3, 0.45, -0.5])
    c = compute_hydrostatic_restoring_matrix(hull, RHO * volume, cg, RHO, G).restoring_matrix
    for index in (0, 1, 5):  # surge, sway, yaw
        assert np.max(np.abs(c[index, :])) == 0.0
        assert np.max(np.abs(c[:, index])) == 0.0


def test_symmetric_hull_keeps_the_couplings_at_exactly_zero() -> None:
    """Backward compatibility: the historical symmetric cases are unchanged.

    A barge centred on the origin has ``S_x = S_y = I_xy = 0``, so the new
    entries vanish identically and the earlier validation numbers (and the
    ``test_restoring_matrix_is_diagonal_in_heave_roll_pitch`` guard) still hold.
    """
    hull = box_hull_mesh(10.0, 4.0, 1.0)
    volume, _ = compute_displaced_volume_and_buoyancy(hull)
    cg = np.array([0.0, 0.0, -0.5])
    c = compute_hydrostatic_restoring_matrix(hull, RHO * volume, cg, RHO, G).restoring_matrix
    assert c[HEAVE, ROLL] == 0.0
    assert c[HEAVE, PITCH] == 0.0
    assert c[ROLL, PITCH] == 0.0
    assert c[HEAVE, HEAVE] == pytest.approx(RHO * G * 40.0, rel=1.0e-12)


def test_heave_and_pitch_are_statically_coupled_for_an_lcf_offset_hull() -> None:
    """Why the missing entry mattered: a pitch moment now heaves the hull.

    Solving the 2x2 static system ``C[heave,pitch] xi = Q`` for a unit pitch
    moment, the heave response is ``-C[2,4] / (C[2,2] C[4,4] - C[2,4]^2)``: zero
    for the old diagonal-only matrix, non-zero for any hull whose waterplane
    centroid is off the origin. A ship responds to a longitudinal weight shift by
    trimming *and* by changing draft, which the diagonal-only matrix missed
    entirely.
    """
    hull = _shifted_box(2.0, 1.0, 1.0, (1.3, 0.45))
    volume, _ = compute_displaced_volume_and_buoyancy(hull)
    cg = np.array([1.3, 0.45, -0.5])
    c = compute_hydrostatic_restoring_matrix(hull, RHO * volume, cg, RHO, G).restoring_matrix

    block = np.array([[c[HEAVE, HEAVE], c[HEAVE, PITCH]], [c[PITCH, HEAVE], c[PITCH, PITCH]]])
    response = np.linalg.solve(block, np.array([0.0, 1.0]))
    heave, pitch = response
    assert abs(heave) > 0.0
    # Sign: a positive pitch moment trims the bow down. With the waterplane
    # centroid forward of the origin the lost immersion at the stern wins nowhere
    # near the bow, so the hull also rises (``heave > 0``).
    assert heave > 0.0
    # The diagonal-only matrix gives exactly zero heave for the same load.
    diagonal_only = np.diag(np.diag(block))
    assert np.linalg.solve(diagonal_only, np.array([0.0, 1.0]))[0] == 0.0
    # And it under-predicts the pitch response: the classical coupled-response
    # factor is 1/(1 - mu^2) with mu = C[2,4]/sqrt(C[2,2] C[4,4]), which here is
    # 6x because the LCF offset was chosen to be extreme (1.3 m on a 2 m box).
    mu_squared = block[0, 1] ** 2 / (block[0, 0] * block[1, 1])
    assert pitch == pytest.approx(
        1.0 / block[1, 1] / (1.0 - mu_squared), rel=1.0e-12
    )
    assert pitch > 3.0 * abs(1.0 / block[1, 1])


def test_fully_submerged_body_has_no_restoring_block() -> None:
    """No waterplane -> no heave/roll/pitch restoring from the free surface."""
    cx, cz, radius = 0.4, -5.0, 1.0
    vertices = np.array(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ]
    ) + np.array([cx, 0.0, cz])
    faces = np.array(
        [
            [4, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ]
    )
    hull = WettedHullMesh(vertices, faces)
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)
    props = compute_hydrostatic_restoring_matrix(
        hull, RHO * volume, np.array([cx, center_of_buoyancy[1], cz]), RHO, G
    )
    assert props.waterplane_area == 0.0
    # z_B and z_G agree only to floating point, so the roll/pitch entries are
    # rho g volume (z_B - z_G) ~ 1e-11 N m rather than a hard zero.
    scale = RHO * G * volume
    assert np.max(np.abs(props.restoring_matrix)) < 1.0e-12 * scale