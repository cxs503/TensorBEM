"""Guards for the last four mutation survivors of ``mutation_audit_tensorbem.md``.

The mutation audit (``docs/mutation_audit_tensorbem.md`` §5, mission m2582) left
four mutants alive and classified them as real/ suspected coverage gaps:

=======================================================  ===================
survivor                                                 closed here by
=======================================================  ===================
``hydrostatic:236`` ``/ 6.0`` -> ``* 6.0`` (x_moment)     eccentric waterplane
``hydrostatic:255`` ``0.0`` -> ``1.0`` (no waterline)     submerged hull
``hydrostatic:314`` ``np.zeros(3)`` -> ``np.zeros(4)``    zero-displacement branch
``free_surface:243`` ``2.0 * (...)`` -> ``2.0 / (...)``   depth-derivative path
=======================================================  ===================

Two lessons are encoded here deliberately:

* every existing waterplane case in the suite is *centred on the origin*, so its
  first moments are ~0 and a divide/multiply swap of ``x_moment`` is invisible.
  The guard therefore needs an **eccentric** waterplane whose centroid is an
  exact, non-zero rational number.
* every reference below is independent of the implementation it polices: a
  closed-form rectangle integral, an exactly degenerate input, or a finite
  difference of ``green_function``. None of them restates a formula from
  ``hydrostatic.py`` / ``free_surface.py``.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_analytic import box_hull_mesh
from tensorbem.free_surface import green_function, green_gradient
from tensorbem.geometry3d import WettedHullMesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)


def _closed_octahedron(centre_z: float, radius: float = 1.0) -> WettedHullMesh:
    """A *closed*, fully submerged octahedron: no waterline edges at all."""
    c = np.array([0.0, 0.0, centre_z])
    vertices = np.array(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ]
    ) + c
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
    return WettedHullMesh(vertices, faces)


def test_eccentric_waterplane_pins_the_centroid_away_from_the_origin() -> None:
    """A 2 m x 1 m box shifted to ``x in [0, 2]``, ``y in [0, 1]``.

    Closed-form reference for a rectangle of length ``L`` (x) and beam ``B`` (y)
    about its own centroid: area ``L B``, centroid ``(L/2, B/2)``,
    ``I_xx = L B^3 / 12``, ``I_yy = B L^3 / 12``, ``I_xy = 0``.  Both first
    moments are non-zero, so a divide/multiply swap in either moment accumulator
    (the surviving ``hydrostatic:236`` mutant) changes the answer by a factor of
    36 and cannot hide behind a symmetric hull.
    """
    hull = box_hull_mesh(2.0, 1.0, 1.0)
    shifted = WettedHullMesh(hull.vertices + np.array([1.0, 0.5, 0.0]), hull.faces)

    area, centroid, moments = compute_waterplane_properties(shifted)

    length, beam = 2.0, 1.0
    assert area == pytest.approx(length * beam, abs=1.0e-12)
    assert centroid == pytest.approx([length / 2.0, beam / 2.0], abs=1.0e-12)
    assert moments == pytest.approx(
        [length * beam**3 / 12.0, beam * length**3 / 12.0, 0.0], abs=1.0e-12
    )
    # The eccentricity must survive the pooled-centroid shift: the transverse
    # moment about the origin is L*B^3/12 + L*B*(B/2)^2, which is 4x larger.
    assert moments[0] > 0.0 and centroid[0] > 0.0 and centroid[1] > 0.0


def test_submerged_hull_without_waterline_reports_exactly_zero_area() -> None:
    """No waterline edges -> area 0, centroid ``(0, 0)``, moments ``(0, 0, 0)``.

    A closed body fully below ``z = 0`` is accepted by ``WettedHullMesh`` (no
    boundary half-edges), and its waterplane is empty.  The degenerate return of
    ``compute_waterplane_properties`` must then be exactly zero: reporting a
    fictitious area (the surviving ``0.0 -> 1.0`` mutant) would put a spurious
    ``rho g A_wp`` into the heave restoring entry ``C[2, 2]``.
    """
    hull = _closed_octahedron(centre_z=-3.0)
    assert hull.waterline_edges.shape[0] == 0  # precondition: branch is entered

    area, centroid, moments = compute_waterplane_properties(hull)

    assert area == 0.0
    assert np.array_equal(centroid, np.zeros(2))
    assert np.array_equal(moments, np.zeros(3))


def test_waterline_with_cancelling_signed_area_reports_no_waterplane() -> None:
    """A self-intersecting ("bowtie") waterline encloses zero net area.

    The loop tracer accepts the edge set (every vertex has two incident edges),
    but the signed area of the bowtie below is exactly zero, so the integrator
    must fall into its degenerate branch and report an *empty* waterplane.  No
    validated ``WettedHullMesh`` can produce such a loop, so a duck-typed mesh is
    used; the point is that the returned area is exactly zero rather than a
    fictitious 1 m^2 (the surviving ``0.0 -> 1.0`` mutant at this branch).
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    stub = types.SimpleNamespace(
        vertices=vertices,
        waterline_edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64),
    )

    area, centroid, moments = compute_waterplane_properties(stub)

    assert area == 0.0
    assert np.array_equal(centroid, np.zeros(2))
    assert np.array_equal(moments, np.zeros(3))


def test_zero_displacement_branch_returns_a_three_component_centre() -> None:
    """Volume that cancels exactly -> ``(0.0, zeros(3))``, never a longer vector.

    No validated hull can reach this branch (the constructor rejects a
    non-positive displaced volume), so the branch is exercised with a minimal
    duck-typed mesh: two triangles with identical vertices and opposite winding,
    whose signed tetrahedron volumes cancel to exactly zero.  The assertion is on
    the *shape* and value of the returned centre, which is what the surviving
    ``np.zeros(3) -> np.zeros(4)`` mutant corrupts -- a silent 4-vector here
    would propagate into ``center_of_buoyancy`` of the public dataclass.
    """
    vertices = np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, -2.0], [0.0, 1.0, -3.0]], dtype=float
    )
    stub = types.SimpleNamespace(
        vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    )

    volume, centre = compute_displaced_volume_and_buoyancy(stub)

    assert volume == 0.0
    assert centre.shape == (3,)
    assert np.array_equal(centre, np.zeros(3))

    # Same contract for the empty-mesh branch (a distinct AST node, same shape).
    empty = types.SimpleNamespace(
        vertices=vertices, faces=np.zeros((0, 3), dtype=np.int64)
    )
    volume_empty, centre_empty = compute_displaced_volume_and_buoyancy(empty)
    assert volume_empty == 0.0 and centre_empty.shape == (3,)


@pytest.mark.parametrize(
    "field_x, field_z, source_x, source_z, nu",
    [
        (0.9, -0.3, 0.3, -2.0, 0.6),
        (0.0, -1.0, 0.0, -1.5, 1.2),
        (-2.5, -0.2, 1.1, -0.9, 0.35),
    ],
)
def test_source_depth_derivative_matches_finite_difference_of_green_function(
    field_x: float, field_z: float, source_x: float, source_z: float, nu: float
) -> None:
    """``dG/ds_z`` must equal a central difference of ``G`` itself.

    This is the finite-difference guard the audit found missing for the
    ``df_d_depth`` path.  After the composition fix in ``green_gradient`` the
    depth derivative feeds the source-normal derivative, so a
    ``2.0 * (...) -> 2.0 / (...)`` swap now shows up as a 10^-2 relative error
    instead of being absorbed by an unused intermediate variable.
    """
    _, analytic = green_gradient(field_x, field_z, source_x, source_z, nu)
    step = 1.0e-6
    numeric = (
        green_function(field_x, field_z, source_x, source_z + step, nu)
        - green_function(field_x, field_z, source_x, source_z - step, nu)
    ) / (2.0 * step)
    assert analytic == pytest.approx(numeric, rel=1.0e-7, abs=1.0e-12)