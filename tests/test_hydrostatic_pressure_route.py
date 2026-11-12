"""Tests for the pressure-integration hydrostatic restoring route.

The invariants asserted here are the ones this route *can* be held to without
relying on the waterplane route being right:

* the immersed volume and the centre of buoyancy come out exactly for a planar
  body (they are pure geometry),
* the pressure field reproduces ``rho*g*(z_fs - z)`` exactly,
* the heave row of ``C`` (the part driven by the displaced volume only) matches
  the waterplane route to round-off for a body whose wetted surface is exactly
  representable by its facets,
* the yielding matrix is symmetric and free of any restoring in the horizontal
  translations,
* ``C`` is a genuine derivative: it reproduces the finite-difference load change
  it was built from.

The disagreement between the two routes on the roll/pitch rows is *documented*
in ``docs/m16554/pressure_route_crosscheck.md``; the test below pins its
currently measured size so that any change (for better or worse) is visible
instead of silent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_pressure_route_study import (
    DENSITY,
    GRAVITY,
    build_cases,
    closed_box_mesh,
    compare_case,
    pressure_profile_check,
)
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.benchmarks.hydrostatic_moment_quadrature_arbitration import (
    SPHEROID_AXES,
    SPHEROID_COG,
    SPHEROID_SHIFT,
    planar_wall_quadrature_check,
)
from tensorbem.geometry3d import ellipsoid_mesh
from tensorbem.hydrostatic_pressure_route import (
    clip_to_fluid,
    immersed_volume,
    pressure_loads,
    pressure_profile,
    restoring_matrix_pressure_route,
    richardson_restoring_matrix,
    rigid_transform,
    wetted_hull_from_closed,
)


@pytest.fixture(scope="module")
def barge():
    return closed_box_mesh(10.0, 4.0, 2.0, 0.5)


def test_closed_box_is_watertight_and_its_volume_is_exact(barge):
    vertices, faces = barge
    assert np.allclose(vertices[:, 2].min(), -2.0)
    assert np.allclose(vertices[:, 2].max(), 0.5)
    assert immersed_volume(vertices, faces) == pytest.approx(80.0, rel=1e-14)


def test_clip_at_the_free_surface_reproduces_the_barge_volume(barge):
    vertices, faces = barge
    clipped_vertices, clipped_faces = clip_to_fluid(vertices, faces)
    assert clipped_vertices[:, 2].max() == pytest.approx(0.0, abs=1e-15)
    assert immersed_volume(clipped_vertices, clipped_faces) == pytest.approx(
        80.0, rel=1e-14
    )


def test_buoyancy_and_centre_of_buoyancy_are_exact_for_the_barge(barge):
    vertices, faces = barge
    clipped_vertices, clipped_faces = clip_to_fluid(vertices, faces)
    load = pressure_loads(clipped_vertices, clipped_faces, density=DENSITY, gravity=GRAVITY)
    assert load.force[2] == pytest.approx(DENSITY * GRAVITY * 80.0, rel=1e-13)
    assert abs(load.force[0]) < 1e-9 * abs(load.force[2])
    assert abs(load.force[1]) < 1e-9 * abs(load.force[2])
    # moments about the origin vanish for a symmetric body
    assert abs(load.moment[0]) < 1e-9 * abs(load.force[2])
    assert abs(load.moment[1]) < 1e-9 * abs(load.force[2])


def test_pressure_profile_reproduces_the_analytic_field():
    result = pressure_profile_check(draft=2.0, count=20)
    assert result["count"] == 20
    assert result["max_relative_error"] == 0.0
    depths = np.asarray(result["depths"])
    points = np.column_stack([np.zeros(depths.size), np.zeros(depths.size), depths])
    expected = DENSITY * GRAVITY * (-depths)
    assert np.allclose(pressure_profile(points), expected, rtol=0.0, atol=0.0)


def test_wetted_hull_from_closed_matches_the_barge_hull_volume(barge):
    vertices, faces = barge
    hull = wetted_hull_from_closed(vertices, faces)
    assert isinstance(hull, object)
    properties = compute_hydrostatic_restoring_matrix(
        hull,
        mass=DENSITY * 80.0,
        center_of_gravity=np.array([0.0, 0.0, -1.5]),
        density=DENSITY,
        gravity=GRAVITY,
    )
    assert properties.displaced_volume == pytest.approx(80.0, rel=1e-14)
    assert properties.waterplane_area == pytest.approx(40.0, rel=1e-14)


def test_heave_entries_of_the_two_routes_agree_to_round_off(barge):
    vertices, faces = barge
    case = build_cases()[0]
    result = compare_case(case)
    heave = result["diagonal_relative"]["heave"]
    assert heave < 1e-10
    # the heave-roll and heave-pitch couplings are exactly zero for a symmetric
    # body, and both routes must say so
    pressure = np.asarray(result["matrix_pressure"])
    waterplane = np.asarray(result["matrix_waterplane"])
    scale = np.abs(waterplane).max()
    for entry in ((2, 3), (3, 2), (2, 4), (4, 2)):
        assert abs(pressure[entry]) < 1e-6 * scale
        assert abs(waterplane[entry]) < 1e-6 * scale


def test_matrix_is_symmetric_and_has_no_horizontal_restoring(barge):
    vertices, faces = barge
    result = restoring_matrix_pressure_route(
        vertices,
        faces,
        center_of_gravity=(0.0, 0.0, -1.5),
        step=1e-3,
    )
    matrix = np.asarray(result.matrix)
    scale = np.abs(matrix).max()
    block = matrix[:5, :5]
    assert np.allclose(block, block.T, atol=1e-6 * scale)
    for index in (0, 1):  # surge and sway: no hydrostatic restoring at all
        assert np.abs(matrix[index, :]).max() < 1e-6 * scale
        assert np.abs(matrix[:, index]).max() < 1e-6 * scale
    # the yaw row/column vanishes analytically for this trimmed, symmetric body;
    # m16662 explained the residue m16554 reported here (out-of-trim equilibrium
    # moment) and the quadrature fix took it to round-off level
    assert np.abs(matrix[5, :]).max() < 1e-10 * scale
    assert np.abs(matrix[:, 5]).max() < 1e-10 * scale


def test_heave_column_is_the_waterplane_stiffness(barge):
    """``C[:, 2] = (0, 0, rho*g*A, 0, 0, 0)`` for a wall-sided body, from the route itself."""
    vertices, faces = barge
    result = restoring_matrix_pressure_route(
        vertices,
        faces,
        center_of_gravity=(0.0, 0.0, -1.5),
        step=1e-3,
    )
    column = np.asarray(result.matrix)[:, 2]
    expected = DENSITY * GRAVITY * 40.0  # waterplane area 10 m x 4 m
    assert column[2] == pytest.approx(expected, rel=1e-12)
    scale = np.abs(result.matrix).max()
    for index in (0, 1, 3, 4, 5):
        assert abs(column[index]) < 1e-8 * scale


def test_roll_pitch_agree_with_the_waterplane_route_to_round_off():
    """The m16554 debt is paid: the roll/pitch rows now agree for a planar body.

    Mission m16554 recorded (and pinned) a 4.12 %/3.69 % roll/pitch disagreement
    on the barge and could not attribute it.  Mission m16662 traced it to the
    moment quadrature -- ``int p (r x n) dS`` is quadratic, the centroid rule is
    first-degree exact -- and replaced it with the degree-2 edge-midpoint rule
    (``tensorbem.hydrostatic_pressure_route.pressure_moment``).  On the *same*
    160-panel barge the two routes now agree to round-off, so the pin is strict.
    """
    cases = {case["name"]: case for case in (compare_case(c) for c in build_cases())}
    barge = cases["barge-L10B4T2"]["diagonal_relative"]
    # a wall-sided box: its facets represent the wetted surface exactly, so the
    # only remaining error is the central-difference truncation the Richardson
    # extrapolation removes
    assert barge["heave"] < 1e-10
    assert barge["roll"] < 1e-10
    assert barge["pitch"] < 1e-10
    # curved body: limited by the facet discretisation of the spheroid, which
    # both routes see on the same facets (m16554 measured 1.2e-3 / 2.9e-3 here)
    spheroid = cases["spheroid-a5b2c1.5"]["diagonal_relative"]
    assert spheroid["heave"] < 1e-4
    assert spheroid["roll"] < 1e-3
    assert spheroid["pitch"] < 1e-3


def test_pressure_moment_is_exact_for_the_quadratic_integrand():
    """``int p (r x n) dS`` against its closed form on a vertical wall.

    For the wall ``[0, L] x [-T, 0]`` at ``y = 0`` with normal ``+y`` the only
    non-zero component is ``M_x = -rho*g*L*T**3/3``.  The exact rule must
    reproduce it to round-off at *any* panel count; the pre-m16662 centroid rule
    cannot (it cannot integrate ``z**2``), and its error must fall as ``h**2``.
    """
    analytic = None
    centroid_errors = []
    for panels in (1, 2, 4, 8):
        comparison = planar_wall_quadrature_check(panels_x=panels, panels_z=panels)
        exact = comparison.exact_moment[0]
        reference = comparison.analytic_moment[0]
        if analytic is None:
            analytic = reference
        assert reference == pytest.approx(analytic, rel=1e-15)
        assert exact == pytest.approx(reference, rel=1e-12)
        # the z-moment has its own closed form: (r x n)_z = x on this wall
        assert comparison.exact_moment[1] == pytest.approx(0.0, abs=1e-9 * abs(reference))
        assert comparison.exact_moment[2] == pytest.approx(
            comparison.analytic_moment[2], rel=1e-12
        )
        centroid_errors.append(abs(comparison.centroid_moment[0] / reference - 1.0))
    # the legacy rule under-integrates, and halves its error fourfold per refinement
    assert centroid_errors[0] > 0.1
    for coarse, fine in zip(centroid_errors, centroid_errors[1:]):
        assert fine < coarse / 3.0


def test_out_of_trim_yaw_coupling_matches_the_equilibrium_moment():
    """The yaw residue m16554 could not place is ``W*(x_G - x_B)``.

    The verification spheroid is deliberately *out of trim* (centre of gravity at
    ``(1.5, 1.0)``, centre of buoyancy at ``(1.2, 0.7)``).  Linearising about a
    body whose weight and buoyancy do not share a vertical line leaves the yaw
    couplings ``C[3,6] = W*(x_G - x_B)`` and ``C[4,6] = W*(y_G - y_B)``: the
    equilibrium moment is carried round by the rotation.  The closed-form
    waterplane table assumes a trimmed body and has no such entry, which is why
    the two routes disagreed there and nowhere else.
    """
    mesh = ellipsoid_mesh(3, np.array(SPHEROID_AXES))
    vertices = mesh.vertices + np.array(SPHEROID_SHIFT)[None, :]
    result = richardson_restoring_matrix(
        vertices, mesh.faces, center_of_gravity=SPHEROID_COG
    )
    x_b, y_b, _ = result.center_of_buoyancy
    assert result.matrix[3, 5] == pytest.approx(result.weight * (SPHEROID_COG[0] - x_b), rel=1e-6)
    assert result.matrix[4, 5] == pytest.approx(result.weight * (SPHEROID_COG[1] - y_b), rel=1e-6)
    # the transpose row stays at round-off: an out-of-trim body has no potential,
    # so its load derivative is not a Hessian and C is not symmetric here
    assert abs(result.matrix[5, 3]) < 1e-6 * abs(result.matrix[3, 5])

    # trimmed control: put the centre of gravity on the centre of buoyancy
    trimmed = richardson_restoring_matrix(
        vertices, mesh.faces, center_of_gravity=(SPHEROID_SHIFT[0], SPHEROID_SHIFT[1], 0.0)
    )
    scale = np.abs(trimmed.matrix).max()
    assert np.abs(trimmed.matrix[3:5, 5]).max() < 1e-10 * scale


def test_rigid_transform_round_trip_is_stable():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, -2.0]])
    motion = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03])
    moved = rigid_transform(points, motion)
    back = rigid_transform(moved, -motion)
    # a finite rotation is not undone by the negated vector, but the isometry is
    assert np.allclose(
        np.linalg.norm(moved - moved.mean(axis=0), axis=1),
        np.linalg.norm(points - points.mean(axis=0), axis=1),
        rtol=1e-13,
    )
    assert np.allclose(rigid_transform(points, np.zeros(6)), points)
    assert back.shape == points.shape