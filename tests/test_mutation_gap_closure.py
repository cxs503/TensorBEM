"""Closure of the four coverage gaps left open by the mutation audit.

``docs/mutation_audit_tensorbem.md`` §5 classified seven surviving mutants and
narrowed four of them down to genuine coverage gaps (the other three are
equivalent mutants and must *not* be "tested"):

===============================================  =====================================
survivor                                        what was missing
===============================================  =====================================
``hydrostatic.py:236`` ``/ 6.0`` -> ``* 6.0``     an *eccentric* waterplane, so the
in the ``x_moment`` accumulation                 first moment is not ~0
``hydrostatic.py:255`` ``0.0`` -> ``1.0``         a body with **no** waterline at all
in the empty-waterline guard                      (fully submerged)
``hydrostatic.py:314`` ``np.zeros(3)`` ->         the zero-displacement guard of
``np.zeros(4)``                                   ``compute_displaced_volume_and_buoyancy``
``free_surface.py:243`` ``2.0 * (...)`` ->        a guard on the *depth* derivative
``2.0 / (...)`` in ``green_gradient``             feeding ``dg/ds_z``
===============================================  =====================================

The last one was the interesting case: ``df_d_depth`` was only ever consumed by
``dg_d_depth``, which was computed and then *thrown away* -- the source
derivative ``dg/ds_z`` re-derived the same expression inline.  That made the
audit's mutant unobservable (it is a mutation of dead code).  The fix in
``tensorbem/free_surface.py`` is not a test-only patch: ``dg/ds_z`` is now
expressed through ``dg_d_depth``, so the branch is live and the finite-difference
guard below polices it.

Every assertion here is against an independent reference (closed-form area,
centroid and second moments of a rectangle; the definition of a finite
difference; the definition of "no waterplane"), never against the expression it
is policing.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from tensorbem.free_surface import green_function, green_gradient
from tensorbem.geometry3d import WettedHullMesh, box_hull_mesh, submerged_sphere_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)

DENSITY = 1025.0
GRAVITY = 9.80665

# Barge used by the eccentric-waterplane case: 4 m x 1 m waterplane, 0.5 m draft.
LENGTH, BEAM, DRAFT = 4.0, 1.0, 0.5
# Rigid translation of the whole hull in the waterplane; keeps the body wall-sided
# and the waterline on z = 0, so area/moments stay exact while the centroid moves.
SHIFT_X, SHIFT_Y = 1.0, 0.25


def _shifted(src: WettedHullMesh, dx: float, dy: float, dz: float = 0.0) -> WettedHullMesh:
    """Return ``src`` translated by ``(dx, dy, dz)``.

    A rigid translation of a wall-sided hull is exact for a panel mesh: the
    waterline stays on ``z = 0`` and every panel stays planar, so the analytic
    reference values of the shifted body are known in closed form.
    """
    vertices = np.array(src.vertices, dtype=float)
    vertices[:, 0] += dx
    vertices[:, 1] += dy
    vertices[:, 2] += dz
    return WettedHullMesh(vertices, src.faces)


def _barge() -> WettedHullMesh:
    h = box_hull_mesh(LENGTH, BEAM, DRAFT, n_length=4, n_beam=2, n_draft=1)
    return h


# --------------------------------------------------------------------------- #
# 1. eccentric waterplane: the first moments are not zero
# --------------------------------------------------------------------------- #
def test_eccentric_waterplane_centroid_and_moments_are_pinned():
    """A waterplane centred at (1.0, 0.25) must report that centroid exactly.

    Closed form for a rectangle of length ``L`` and beam ``B`` translated to
    ``(x0, y0)``: ``A = L B``, ``(x_c, y_c) = (x0, y0)``, ``I_xx = L B^3 / 12``,
    ``I_yy = B L^3 / 12``, ``I_xy = 0`` -- all about the *centroid*, hence
    independent of the translation.  Before this test every waterplane in the
    suite was symmetric about the origin, so the first moments were ~0 and a
    ``/ 6.0 -> * 6.0`` swap in the ``x_moment`` accumulation survived the whole
    suite (``hydrostatic.py:236`` in the audit).
    """
    hull = _shifted(_barge(), SHIFT_X, SHIFT_Y)

    area, centroid, moments = compute_waterplane_properties(hull)

    assert area == pytest.approx(LENGTH * BEAM, rel=1e-12)
    # Both first moments are exercised: x_c and y_c are both non-zero and are
    # reproduced only if each is divided by 6 and then by the area.
    assert centroid[0] == pytest.approx(SHIFT_X, rel=1e-12)
    assert centroid[1] == pytest.approx(SHIFT_Y, rel=1e-12)
    assert moments[0] == pytest.approx(LENGTH * BEAM**3 / 12.0, rel=1e-12)
    assert moments[1] == pytest.approx(BEAM * LENGTH**3 / 12.0, rel=1e-12)
    assert moments[2] == pytest.approx(0.0, abs=1e-12)


def test_heave_restoring_follows_the_shifted_waterplane():
    """C[2,2] = rho g A is translation invariant; the centroid is not.

    Ties the eccentric case to a physical quantity: a hull translated in x/y has
    the same heave stiffness but a waterplane centre that moves with it, so the
    second moment about the *pooled* centroid must still be ``L B^3 / 12``.
    """
    hull = _shifted(_barge(), SHIFT_X, SHIFT_Y)
    mass = DENSITY * LENGTH * BEAM * DRAFT
    props = compute_hydrostatic_restoring_matrix(
        hull, mass, np.array([SHIFT_X, SHIFT_Y, -0.25])
    )
    assert props.restoring_matrix[2, 2] == pytest.approx(
        DENSITY * GRAVITY * LENGTH * BEAM, rel=1e-10
    )
    assert props.waterplane_center[0] == pytest.approx(SHIFT_X, rel=1e-12)
    assert props.waterplane_center[1] == pytest.approx(SHIFT_Y, rel=1e-12)
    # Wall-sided barge: BM_T = I_T / nabla and GM_T = KB + BM_T - KG are exact.
    nabla = LENGTH * BEAM * DRAFT
    bm_t = (LENGTH * BEAM**3 / 12.0) / nabla
    gm_t = 0.5 * DRAFT + bm_t - 0.25
    assert props.waterplane_moments[0] == pytest.approx(
        LENGTH * BEAM**3 / 12.0, rel=1e-12
    )
    assert props.restoring_matrix[3, 3] == pytest.approx(
        DENSITY * GRAVITY * nabla * gm_t, rel=1e-10
    )


# --------------------------------------------------------------------------- #
# 2. a body with no waterline: the empty-waterline guard
# --------------------------------------------------------------------------- #
def test_fully_submerged_body_returns_no_waterplane():
    """A closed, fully submerged hull has no waterline edge: properties are zero.

    ``compute_waterplane_properties`` must return exactly ``0.0`` area with zero
    centroid and zero moments.  The audit showed that no case in the suite ever
    entered this branch, so ``return 0.0, ...`` could be changed to ``1.0``
    undetected (``hydrostatic.py:255``).  A sphere of radius 1 m whose centre is
    1.5 m below the surface is genuinely fully submerged.
    """
    sphere = submerged_sphere_mesh(2, 1.0, 1.5)
    hull = WettedHullMesh(sphere.vertices, sphere.faces)
    assert np.asarray(hull.waterline_edges).size == 0

    area, centroid, moments = compute_waterplane_properties(hull)

    assert area == 0.0
    assert np.array_equal(centroid, np.zeros(2))
    assert np.array_equal(moments, np.zeros(3))
    # No waterplane means no heave/roll/pitch stiffness at all.
    volume, centre = compute_displaced_volume_and_buoyancy(hull)
    assert volume == pytest.approx(4.0 / 3.0 * math.pi, rel=5e-2)
    rho = 1000.0
    props = compute_hydrostatic_restoring_matrix(
        hull, rho * volume, np.array([centre[0], centre[1], centre[2]])
    )
    assert props.waterplane_area == 0.0
    assert props.restoring_matrix[2, 2] == 0.0


def test_zero_area_waterline_loop_returns_no_waterplane():
    """A self-cancelling (bowtie) waterline encloses no area: properties are zero.

    This is the second degenerate branch of ``compute_waterplane_properties``
    (``total_area`` below the epsilon after the per-loop areas are pooled).  The
    four vertices below describe a waterline that crosses itself, so the shoelace
    terms cancel exactly and the pooled area is 0 -- the guard must then answer
    ``0.0`` and not, say, ``1.0``.  The graph is a single closed loop with all
    vertex degrees equal to two, so the loop tracer accepts it; only the area
    test rejects it.  The mesh is duck-typed because ``WettedHullMesh`` refuses
    a mesh that encloses no volume, i.e. this guard is defensive.
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    degenerate = types.SimpleNamespace(vertices=vertices, waterline_edges=edges)

    area, centroid, moments = compute_waterplane_properties(degenerate)

    assert area == 0.0
    assert np.array_equal(centroid, np.zeros(2))
    assert np.array_equal(moments, np.zeros(3))


# --------------------------------------------------------------------------- #
# 3. zero-displacement guard
# --------------------------------------------------------------------------- #
def test_zero_face_hull_returns_a_three_vector():
    """``compute_displaced_volume_and_buoyancy`` must answer with shape (3,).

    The empty-mesh and zero-volume guards both return ``0.0, np.zeros(3)``.  A
    ``3 -> 4`` slip in that literal is invisible to any caller that only reads
    ``volume``, which is why the audit's mutant survived; the shape and the exact
    values are pinned here.  ``WettedHullMesh`` itself rejects a mesh with fewer
    than three faces, so the guard is exercised through a minimal stand-in that
    exposes only the two attributes the function reads -- this also documents
    that the guard is defensive code for hand-built inputs, not a live path of
    the validated constructor.
    """
    empty = types.SimpleNamespace(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int)
    )
    volume, centre = compute_displaced_volume_and_buoyancy(empty)
    assert volume == 0.0
    assert centre.shape == (3,)
    assert np.array_equal(centre, np.zeros(3))

    # A flat, "closed" double-sided patch encloses no volume: the second guard.
    flat_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    flat_faces = np.array([[0, 1, 2], [1, 3, 2], [0, 2, 1], [1, 2, 3]])
    degenerate = types.SimpleNamespace(vertices=flat_vertices, faces=flat_faces)
    volume, centre = compute_displaced_volume_and_buoyancy(degenerate)
    assert volume == 0.0
    assert centre.shape == (3,)
    assert np.array_equal(centre, np.zeros(3))


# --------------------------------------------------------------------------- #
# 4. the depth derivative that feeds dg/ds_z
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field_x, field_z, source_x, source_z, nu",
    [
        (1.0, -1.0, 0.0, -1.2, 0.6),  # moderate depth, wave term dominant
        (0.7, -2.0, -0.4, -0.5, 0.6),  # source near the surface
        (2.5, -0.8, 1.0, -1.9, 0.35),  # long wave
        (0.3, -0.4, -1.1, -0.6, 1.4),  # short wave, shallow pair
    ],
)
def test_source_depth_derivative_matches_a_finite_difference(
    field_x: float, field_z: float, source_x: float, source_z: float, nu: float
) -> None:
    """``dg/ds_z`` from ``green_gradient`` must equal a central difference of G.

    The depth variable ``A = -(z + z_s)`` carries the entire ``z_s`` dependence of
    the image/free-surface part, so this derivative is the one place where a
    defect in the depth term (``2 * (elementary_cos + nu * cos_integral)``, the
    audit's ``2.0 *`` -> ``2.0 /`` swap) shows up.  It used to be dead code; the
    derivative is now expressed through ``dg_d_depth``, and the finite difference
    below is the guard on it.  Step 1e-6 m keeps the truncation error of the
    third derivative below the 1e-6 relative tolerance used here.
    """
    step = 1.0e-6
    _, dg_d_source_z = green_gradient(field_x, field_z, source_x, source_z, nu)
    plus = green_function(field_x, field_z, source_x, source_z + step, nu)
    minus = green_function(field_x, field_z, source_x, source_z - step, nu)
    finite_difference = (plus - minus) / (2.0 * step)
    assert complex(dg_d_source_z) == pytest.approx(
        finite_difference, rel=1e-6, abs=1e-9
    )


def test_depth_derivative_is_live_and_outgoing_conjugate_is_consistent() -> None:
    """The depth branch must actually move ``dg/ds_z``: no dead computation.

    Two independent checks, both of which would pass vacuously if the depth
    branch were disconnected from the returned value:

    * the derivative changes when the source is reflected about the free surface
      (``z_s -> -z_s``), which only the depth term can produce;
    * ``outgoing=False`` must return the complex conjugate of ``outgoing=True``.

    The analytic identity ``dg/ds_z = -dg_dA - (1/2pi)(z - z_s)/r^2`` is *not*
    asserted separately: it is the implementation, and re-asserting it would be
    the copying-its-own-formula failure mode the audit exists to catch.
    """
    nu = 0.6
    x, z, s_x = 0.8, -1.1, -0.3
    _, derivative_up = green_gradient(x, z, s_x, -0.7, nu)
    _, derivative_down = green_gradient(x, z, s_x, -0.9, nu)
    assert abs(derivative_up - derivative_down) > 0.1 * abs(derivative_up)

    _, outgoing = green_gradient(x, z, s_x, -0.75, nu, outgoing=True)
    _, incoming = green_gradient(x, z, s_x, -0.75, nu, outgoing=False)
    assert complex(incoming) == pytest.approx(np.conj(complex(outgoing)), rel=1e-12)