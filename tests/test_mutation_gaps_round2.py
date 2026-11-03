"""Round-2 back-fill for the mutation survivors recorded in ``docs/mutation_audit_tensorbem.md`` §5.

The first audit left four *real* coverage gaps open (the other three survivors
were classified as equivalent mutants and correctly left alone):

===========================================  ====================================================
survivor                                     answer in this file
===========================================  ====================================================
``hydrostatic.py`` waterplane ``x_moment``    eccentric waterplane whose centroid is *not* at the
``/ 6.0`` -> ``* 6.0``                        origin, pinned to a closed-form value
``hydrostatic.py`` ``total_area`` guard       degenerate (collinear) waterline loop -> zero
``0.0`` -> ``1.0``                            area/centroid/moments
``hydrostatic.py`` zero-volume guard          zero displaced volume -> ``(0.0, zeros(3))``,
``np.zeros(3)`` -> ``np.zeros(4)``            *shape* asserted
``free_surface.py`` ``df_d_depth``            ``df_d_depth`` no longer has an unread copy: the
``2.0 * (...)`` -> ``2.0 / (...)``            source/field vertical derivatives are pinned by
                                              central finite differences
===========================================  ====================================================

Every assertion is against an independent reference (Green's-theorem centroid of a
rectangle, the analytic waterplane second moments ``LB^3/12`` and ``BL^3/12``, a
central difference of the Green function, and a documented return-type contract),
not against the formula under test.

Two of the three hydrostatic branches are *unreachable through a validated
``WettedHullMesh``*: the constructor refuses a non-positive displaced volume
(``geometry3d.WettedHullMesh``) and it only accepts waterline loops with at least
three vertices, each of non-zero area, so ``total_area`` can never fall below
``1e-14`` for a constructed hull (strictly nested loops make the signed sum
strictly positive).  Those two tests therefore exercise the guards through a
raw instance built without validation -- that is the only contract they have, and
pinning it is what makes the guard non-silent.  The accompanying assertions in
``test_raw_hull_states_are_rejected_by_the_validated_constructor`` record *why*
the raw path is needed.
"""

import math

import numpy as np
import pytest

from tensorbem.free_surface import green_function, green_gradient, green_gradient_field
from tensorbem.geometry3d import WettedHullMesh, box_hull_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)


def _raw_wetted_hull(vertices, faces, waterline_edges) -> WettedHullMesh:
    """Build a ``WettedHullMesh`` without the validating constructor.

    Used only to reach defensive guards for states the constructor excludes; the
    returned object is a plain container for ``vertices``/``faces``/
    ``waterline_edges`` and carries no claim of geometric validity.
    """
    hull = object.__new__(WettedHullMesh)
    object.__setattr__(hull, "vertices", np.asarray(vertices, dtype=float))
    object.__setattr__(hull, "faces", np.asarray(faces, dtype=np.int64))
    object.__setattr__(
        hull, "waterline_edges", np.asarray(waterline_edges, dtype=np.int64)
    )
    return hull


# --------------------------------------------------------------------------- #
# 1. eccentric waterplane: the centroid must move, not stay at the origin
# --------------------------------------------------------------------------- #
def test_eccentric_barge_waterplane_matches_closed_form() -> None:
    """Waterplane of the box ``x in [0, 2]``, ``y in [0, 1]``, draft 0.5 m.

    Closed form (Green's theorem on a rectangle, or direct integration):

    * area ``A = L B = 2``
    * centroid ``(L/2, B/2) = (1, 0.5)`` -- deliberately *not* the origin
    * ``I_xx = L B^3 / 12 = 1/6``, ``I_yy = B L^3 / 12 = 2/3``, ``I_xy = 0``

    The longitudinal first moment is ``A * x_c = 2``, so a mutant that turns the
    ``/ 6.0`` of ``sum((x + x_next) * cross) / 6.0`` into a multiplication moves
    the centroid to ``x_c = 36`` and is caught by three orders of magnitude.
    """
    length, beam, draft = 2.0, 1.0, 0.5
    centred = box_hull_mesh(length, beam, draft)
    # The generator builds the barge about the origin; shift it so the
    # waterplane centroid sits at (L/2, B/2) instead of at (0, 0).
    vertices = centred.vertices + np.array([length / 2.0, beam / 2.0, 0.0])
    hull = WettedHullMesh(vertices, centred.faces)

    area, centroid, moments = compute_waterplane_properties(hull)

    assert area == pytest.approx(length * beam, rel=1e-14)
    assert centroid[0] == pytest.approx(length / 2.0, rel=1e-12)
    assert centroid[1] == pytest.approx(beam / 2.0, rel=1e-12)
    assert moments[0] == pytest.approx(length * beam**3 / 12.0, rel=1e-12)
    assert moments[1] == pytest.approx(beam * length**3 / 12.0, rel=1e-12)
    assert moments[2] == pytest.approx(0.0, abs=1e-12)

    # The same barge without the offset has a centroid at the origin, so the
    # eccentric case above is genuinely testing the first moment and not a
    # translation of the whole mesh.
    area0, centroid0, _ = compute_waterplane_properties(centred)
    assert area0 == pytest.approx(area, rel=1e-14)
    assert np.max(np.abs(centroid0)) < 1e-12


# --------------------------------------------------------------------------- #
# 2. degenerate (zero-area) waterline loop guard
# --------------------------------------------------------------------------- #
def test_degenerate_waterline_loop_returns_zero_waterplane() -> None:
    """A waterline loop with zero enclosed area must yield a zero waterplane.

    The three vertices are collinear, so every cross product of the shoelace sum
    is zero: the loop *exists* (every vertex has two incident waterline edges)
    but encloses nothing.  The guard must return ``(0.0, zeros(2), zeros(3))``
    rather than divide by zero -- an area of ``1.0`` would poison every GM
    computed downstream.
    """
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    faces = [[0, 1, 2], [2, 1, 0], [0, 2, 1]]
    waterline_edges = [[0, 1], [1, 2], [2, 0]]
    hull = _raw_wetted_hull(vertices, faces, waterline_edges)

    area, centroid, moments = compute_waterplane_properties(hull)

    assert area == 0.0
    assert np.shape(centroid) == (2,)
    assert np.shape(moments) == (3,)
    assert not np.any(centroid)
    assert not np.any(moments)


# --------------------------------------------------------------------------- #
# 3. zero displaced volume guard
# --------------------------------------------------------------------------- #
def test_zero_volume_guard_returns_a_three_vector() -> None:
    """A zero-volume panel set must return ``(0.0, zeros(3))`` -- exactly three.

    The two coincident, oppositely oriented triangles have equal and opposite
    tetrahedron volumes about the origin, so ``sum(signed_volumes)`` cancels to
    zero.  The centre of buoyancy is then undefined; the contract is a
    three-vector of zeros, and the *shape* is what the mutation
    ``np.zeros(3) -> np.zeros(4)`` breaks.
    """
    vertices = [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]
    faces = [[0, 1, 2], [0, 2, 1]]
    hull = _raw_wetted_hull(vertices, faces, [])

    volume, centre = compute_displaced_volume_and_buoyancy(hull)

    assert volume == 0.0
    assert np.shape(centre) == (3,)
    assert not np.any(centre)

    # Same contract for a hull with no panels at all.
    empty = _raw_wetted_hull(np.zeros((0, 3)), np.zeros((0, 3), dtype=int), [])
    volume, centre = compute_displaced_volume_and_buoyancy(empty)
    assert volume == 0.0
    assert np.shape(centre) == (3,)


def test_raw_hull_states_are_rejected_by_the_validated_constructor() -> None:
    """Why the two guards above need a raw instance: geometry3d refuses them.

    ``WettedHullMesh`` requires a positive, panel-consistent displaced volume and
    a waterline whose boundary edges form closed loops.  Recording the rejection
    here documents that the guards are defensive, not dead: they are the only
    thing standing between an unvalidated mesh and a division by zero.
    """
    collapsing = np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], dtype=float
    )
    with pytest.raises(ValueError):
        WettedHullMesh(collapsing, np.array([[0, 1, 2], [0, 2, 1]]))


# --------------------------------------------------------------------------- #
# 4. source / field vertical derivatives of the deep-water Green function
# --------------------------------------------------------------------------- #
def _fd_gradient(function, point, step=1e-6):
    """Central difference of ``function`` at ``point`` w.r.t. both coordinates."""
    x, z = point
    dx = (function(x + step, z) - function(x - step, z)) / (2.0 * step)
    dz = (function(x, z + step) - function(x, z - step)) / (2.0 * step)
    return dx, dz


@pytest.mark.parametrize(
    "source, field, nu",
    [
        ((0.25, -1.10), (0.40, -0.12), 0.6),
        ((0.10, -0.80), (0.55, -0.20), 1.3),
        ((-0.35, -1.70), (0.20, -0.05), 0.9),
    ],
)
def test_deep_water_source_derivatives_match_finite_differences(source, field, nu) -> None:
    """``green_gradient`` must equal the central difference of ``green_function``.

    Both source components are checked, so the depth derivative ``df_d_depth``
    (which enters the vertical source derivative) is policed even though it is
    no longer returned as a separate quantity.
    """
    expect_x, expect_z = _fd_gradient(
        lambda sx, sz: green_function(field[0], field[1], sx, sz, nu),
        source,
    )
    got_x, got_z = green_gradient(field[0], field[1], source[0], source[1], nu)

    # The reference is a second-order central difference, so agreement is
    # limited by the difference stencil (~1e-8 here), not by the analytic form.
    assert abs(got_x - expect_x) / max(abs(expect_x), 1e-12) < 1e-5
    assert abs(got_z - expect_z) / max(abs(expect_z), 1e-12) < 1e-5


@pytest.mark.parametrize("nu", [0.6, 1.3])
def test_deep_water_field_vertical_derivative_matches_finite_differences(nu) -> None:
    """``green_gradient_field`` must equal the central difference in the field point.

    A free surface breaks vertical translation invariance, so
    ``dG/dz != -dG/dz0``; the field derivative shares ``df_d_depth`` with the
    source derivative and this test keeps that shared expression honest.
    """
    source = (0.25, -1.10)
    field = (0.40, -0.12)
    expect_x, expect_z = _fd_gradient(
        lambda fx, fz: green_function(fx, fz, source[0], source[1], nu),
        field,
    )
    got_x, got_z = green_gradient_field(field[0], field[1], source[0], source[1], nu)

    assert abs(got_x - expect_x) / max(abs(expect_x), 1e-12) < 1e-5
    assert abs(got_z - expect_z) / max(abs(expect_z), 1e-12) < 1e-5

    # Translation invariance in x must still hold, and the vertical pair must
    # *not* be antisymmetric -- that is the whole point of the free surface.
    source_x_derivative = green_gradient(
        field[0], field[1], source[0], source[1], nu
    )[0]
    assert abs(got_x + source_x_derivative) / max(abs(got_x), 1e-12) < 1e-12
    assert abs(got_z + green_gradient(field[0], field[1], source[0], source[1], nu)[1]) > 1e-3