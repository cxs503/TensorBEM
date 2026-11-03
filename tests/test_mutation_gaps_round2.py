"""Second round of back-filled guardrails for mutation-audit survivors.

``docs/mutation_audit_tensorbem.md`` §5 listed four survivors that were *not*
closed by the first round and registered them as follow-up work:

1. ``hydrostatic.py:236`` -- the waterplane first moment ``x_moment`` had its
   ``/ 6.0`` swapped to ``* 6.0`` and the suite stayed green, because every
   existing case had a (near) symmetric waterplane whose first moment vanishes.
   Answer: an **eccentric** waterline rectangle, whose centroid is known in
   closed form.
2. ``hydrostatic.py:255`` -- the "no waterplane" branch returned ``0.0`` and the
   suite stayed green when it was changed to ``1.0``: no case ever entered it.
   Answer: a **fully submerged** hull, which turns out to be constructible and
   to make the branch reachable.
3. ``hydrostatic.py:314`` -- the zero-displacement branch of
   ``compute_displaced_volume_and_buoyancy`` returned ``np.zeros(3)`` and
   ``np.zeros(4)`` survived. See
   :func:`test_zero_volume_hull_reports_a_three_component_centre` for why this
   one is *defensive* rather than reachable through the public mesh type, and
   for the contract test that still pins it.
4. ``free_surface.py:243`` -- the depth derivative ``df_d_depth`` was only ever
   consumed by ``dg_d_depth``, which was a **dead local**: the vertical source
   derivative re-expanded the same expression inline instead of reusing it, so
   no test could see a defect in it. This module closes it from both ends: the
   implementation now reuses ``dg_d_depth`` (see the ``green_gradient`` diff,
   numerically identical to 1e-15) and
   :func:`test_source_vertical_gradient_matches_finite_differences` pins the
   resulting path against central differences at several points and for both
   radiation branches.

Every reference below is independent of the code it polices: closed-form
rectangle/sphere properties, or a finite difference of the public Green
function. None of them restates an implementation formula.
"""

import math

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_analytic import box_hull_mesh
from tensorbem.free_surface import green_function, green_gradient
from tensorbem.geometry3d import WettedHullMesh, icosphere
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)


def _translated_box(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    draft: float,
) -> WettedHullMesh:
    """Wall-sided box wetted surface with an off-centre waterline rectangle."""
    x0, x1 = x_range
    y0, y1 = y_range
    if not (x0 < x1 and y0 < y1 and draft > 0.0):
        raise ValueError("x_range and y_range must be increasing and draft positive")
    centred = box_hull_mesh(x1 - x0, y1 - y0, draft)
    shift = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0])
    return WettedHullMesh(centred.vertices + shift, centred.faces)


@pytest.mark.parametrize(
    "x_range, y_range",
    [((0.0, 2.0), (0.0, 1.0)), ((0.5, 2.5), (-1.5, -0.5))],
)
def test_eccentric_waterplane_centroid_is_pinned_by_closed_form(
    x_range: tuple[float, float], y_range: tuple[float, float]
) -> None:
    """A wall-sided box has an exactly rectangular waterplane.

    Closed form for a rectangle ``[x0, x1] x [y0, y1]``:

    * area ``L_x * L_y``,
    * centroid ``((x0+x1)/2, (y0+y1)/2)``,
    * second moments about that centroid ``I_xx = L_x L_y^3 / 12``,
      ``I_yy = L_x^3 L_y / 12``, ``I_xy = 0``.

    Both cases are off-centre in *x* and *y*, so the first moments are O(1)
    rather than zero. The ``/ 6.0`` -> ``* 6.0`` mutation of ``x_moment`` scales
    the reported centroid by 36; the ``y_moment`` sibling is covered by the same
    assertion because both cases are eccentric in ``y`` too.
    """
    hull = _translated_box(x_range, y_range, draft=1.0)
    area, centroid, moments = compute_waterplane_properties(hull)

    length = x_range[1] - x_range[0]
    beam = y_range[1] - y_range[0]
    expected_area = length * beam
    expected_centroid = np.array([0.5 * (x_range[0] + x_range[1]), 0.5 * (y_range[0] + y_range[1])])
    expected_moments = np.array(
        [length * beam**3 / 12.0, length**3 * beam / 12.0, 0.0]
    )

    assert area == pytest.approx(expected_area, rel=1e-12)
    # Absolute tolerance as well: this is an exact (faceted) body, but the
    # centroid is the quantity the survivor corrupts, so do not let a relative
    # tolerance on a large offset hide a small additive error.
    assert centroid == pytest.approx(expected_centroid, rel=1e-12, abs=1e-12)
    assert moments == pytest.approx(expected_moments, rel=1e-12, abs=1e-12)


def test_waterplane_is_empty_for_a_fully_submerged_hull() -> None:
    """No waterline loop at all -> empty waterplane, not a unit-valued one.

    A closed hull strictly below ``z = 0`` has no boundary half-edges, so the
    "no waterplane" early return of ``compute_waterplane_properties`` is taken.
    That branch must report zero area; the audit's mutation of its literal
    ``0.0`` to ``1.0`` would silently inject a phantom waterplane area, and with
    it a phantom heave stiffness ``rho*g*A_wp``.
    """
    sphere = icosphere(subdivisions=2, radius=0.5)
    vertices = sphere.vertices - np.array([0.0, 0.0, 2.0])
    assert np.all(vertices[:, 2] <= -1.5)
    hull = WettedHullMesh(vertices, sphere.faces)
    assert hull.waterline_edges.size == 0

    area, centroid, moments = compute_waterplane_properties(hull)

    assert area == 0.0
    assert centroid.shape == (2,)
    assert moments.shape == (3,)
    assert not np.any(centroid)
    assert not np.any(moments)


class _CancellingHull:
    """Waterline whose single loop encloses zero *signed* area (a bowtie).

    The audit's survivor ``hydrostatic:255:15`` lives in the second degenerate
    branch of ``compute_waterplane_properties`` -- the one taken when loops
    exist but their net area vanishes -- and it survived because no case ever
    reached it.  A physical floating body cannot get there (a cancelling
    waterline would mean a self-intersecting hull), so the branch is defensive:
    it protects the ``x_moment / total_area`` division that follows.  This shim
    enters it directly instead of pretending a valid hull can.
    """

    def __init__(self) -> None:
        # Bowtie: A(-1,-1) -> B(1,1) -> C(1,-1) -> D(-1,1), two equal lobes of
        # opposite orientation, so the shoelace sum is exactly zero.
        self.vertices = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        )
        self.waterline_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


def test_cancelling_waterline_loop_reports_zero_area() -> None:
    """Net-zero waterplane area -> zero area, not a divide-by-zero or a unit value."""
    area, centroid, moments = compute_waterplane_properties(_CancellingHull())

    assert area == 0.0
    assert centroid.shape == (2,)
    assert moments.shape == (3,)
    assert not np.any(centroid)
    assert not np.any(moments)


def test_submerged_sphere_volume_and_buoyancy_centre() -> None:
    """Sanity anchor for the branch's *other* side: a sphere is analytic.

    ``V = 4/3 pi r^3`` and the centre of buoyancy is the sphere centre. This is
    what makes the zero-volume case below a *degenerate* case rather than an
    untested corner: the same code path is verified against closed form here.
    """
    radius = 0.5
    centre = np.array([0.3, -0.2, -2.0])
    exact = 4.0 / 3.0 * math.pi * radius**3

    errors = {}
    for subdivisions in (3, 4):
        sphere = icosphere(subdivisions=subdivisions, radius=radius)
        hull = WettedHullMesh(sphere.vertices + centre, sphere.faces)
        volume, buoyancy_centre = compute_displaced_volume_and_buoyancy(hull)

        # An inscribed polyhedron always *under*-estimates the sphere: the sign
        # of the discretisation error is a structural check, not a tolerance.
        assert volume < exact
        errors[subdivisions] = (exact - volume) / exact
        assert buoyancy_centre == pytest.approx(centre, abs=1e-9)

    # Second-order convergence: four times the panels, a quarter of the error.
    assert errors[4] < errors[3] / 3.5
    assert errors[4] < 3.0e-3


class _FlatHull:
    """Minimal duck-typed hull whose panels all lie in the plane ``z = 0``.

    ``WettedHullMesh`` refuses such a mesh on purpose (its panels must have
    centroids strictly below ``z = 0`` and enclose a positive displaced volume),
    so the zero-displacement branch of
    :func:`compute_displaced_volume_and_buoyancy` is **not reachable through the
    public mesh type** -- that is why the ``np.zeros(3)`` -> ``np.zeros(4)``
    survivor survived, and the honest classification is "defensive branch", not
    "missing verification". This shim reaches it anyway so the documented
    return contract (a 3-component centre) is still pinned.
    """

    def __init__(self) -> None:
        self.vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])


def test_zero_volume_hull_reports_a_three_component_centre() -> None:
    """Zero displaced volume -> zero volume *and* a 3-component zero centre."""
    volume, centre = compute_displaced_volume_and_buoyancy(_FlatHull())

    assert volume == 0.0
    assert centre.shape == (3,)
    assert not np.any(centre)


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize(
    "x, z, s_x, s_z, nu",
    [
        (0.8, -0.9, -0.4, -1.1, 0.6),
        (2.5, -0.4, 0.1, -0.4, 0.35),
        (-1.3, -1.6, 0.9, -0.7, 1.4),
        (0.0, -0.6, 0.0, -1.2, 0.25),
    ],
)
def test_source_vertical_gradient_matches_finite_differences(
    x: float, z: float, s_x: float, s_z: float, nu: float, outgoing: bool
) -> None:
    """``dG/ds_z`` must equal a central difference of the public ``G``.

    ``green_gradient`` computes this derivative along two paths: the depth sum
    ``A = -(z + z0)`` and the Rankine radius ``r = |P - S|``. The audit showed
    that the depth path was a dead local, so a sign/power defect on it was
    invisible. The points below span different scaled depths and offsets, and
    include the ``r = 0``-in-horizontal but separated-vertically point
    ``(0, -0.6; 0, -1.2)``, where the horizontal offset vanishes and only the
    depth path can be right.
    """
    step = 1.0e-6
    _, dg_d_source_z = green_gradient(x, z, s_x, s_z, nu, outgoing=outgoing)
    plus = green_function(x, z, s_x, s_z + step, nu, outgoing=outgoing)
    minus = green_function(x, z, s_x, s_z - step, nu, outgoing=outgoing)
    finite_difference = (plus - minus) / (2.0 * step)

    assert complex(dg_d_source_z) == pytest.approx(
        finite_difference, rel=1e-6, abs=1e-9
    )