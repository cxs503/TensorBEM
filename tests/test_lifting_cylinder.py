"""Tests for the prescribed-circulation (rotating / Magnus) cylinder solve.

The reference is closed form (``W(z) = U(z + a^2/z) - i Gamma/(2 pi) ln(z/a)``),
so every assertion below is against an analytic value evaluated at the same
points, not against another discrete solution.  The failure modes that matter
are (a) a wrong circulation sign, (b) a vortex term that is not made tangent to
the *faceted* wall, and (c) a pressure integral that does not reproduce
Kutta-Joukowski -- each is covered explicitly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import lifting_cylinder as lc
from tensorbem.benchmarks import case_cylinder_stream as case
from tensorbem.geometry import regular_polygon
from tensorbem.lifting import vortex_velocity_field

RADIUS = 1.0
SPEED = 1.0
CIRCULATION = 3.0
PROBES = np.array(
    [[1.5, 0.4], [2.0, -0.7], [3.0, 1.0], [-1.3, 2.2], [0.0, 1.4]]
)


def test_stagnation_angles_zero_the_surface_speed() -> None:
    angles = lc.stagnation_angles(RADIUS, SPEED, CIRCULATION)
    assert len(angles) == 2
    speed = lc.exact_surface_speed(np.array(angles), RADIUS, SPEED, CIRCULATION)
    assert np.max(np.abs(speed)) < 1.0e-15
    # sin(theta_s) = Gamma/(4 pi U a) is the closed-form statement
    assert math.isclose(
        4.0 * math.pi * SPEED * RADIUS / CIRCULATION * math.sin(angles[0]), 1.0
    )


def test_closed_form_field_is_harmonic_and_irrotational() -> None:
    step = 1.0e-5
    div = np.zeros(len(PROBES))
    curl = np.zeros(len(PROBES))
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        plus = lc.exact_velocity(PROBES + offset, RADIUS, SPEED, CIRCULATION)
        minus = lc.exact_velocity(PROBES - offset, RADIUS, SPEED, CIRCULATION)
        derivative = (plus - minus) / (2.0 * step)
        div += derivative[:, axis]
        other = 1 - axis
        curl += (-1.0) ** axis * derivative[:, other]
    assert np.max(np.abs(div)) < 1.0e-9
    assert np.max(np.abs(curl)) < 1.0e-9


def test_joukowski_lift_sign_and_magnitude() -> None:
    # rho = 1: a counter-clockwise (positive) circulation lifts downward.
    assert lc.joukowski_lift(RADIUS, SPEED, CIRCULATION) == -SPEED * CIRCULATION


def test_vortex_flux_is_zero_on_inscribed_polygon_midpoints() -> None:
    """Pin why the response solve is a no-op on this case's mesh.

    The midpoint normal of an inscribed regular polygon is exactly radial, and
    the vortex velocity is tangential there, so its normal flux vanishes.  This
    is a *measured* property (1e-16), not an assumption: the response solve is
    kept for meshes where it is non-zero, which the next test demonstrates.
    """
    circle_points = np.column_stack(
        (
            np.cos(np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)),
            np.sin(np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)),
        )
    )
    radial = np.sum(
        vortex_velocity_field(circle_points, (0.0, 0.0), 1.0) * circle_points, axis=1
    )
    assert np.max(np.abs(radial)) < 1.0e-15
    for panels in (16, 32, 64, 128):
        mesh = regular_polygon(panels, RADIUS)
        assert np.max(np.abs(lc.vortex_flux(mesh))) < 1.0e-14


def test_vortex_flux_is_nonzero_off_the_circle() -> None:
    """A panel normal that is not radial makes the correction term real."""
    from tensorbem.geometry import ellipse_mesh

    mesh = ellipse_mesh(32, 1.5, 0.75)
    flux = lc.vortex_flux(mesh)
    assert np.max(np.abs(flux)) > 1.0e-2


@pytest.mark.parametrize("panels", [32, 64])
def test_solve_reproduces_surface_speed_and_circulation(panels: int) -> None:
    mesh = regular_polygon(panels, RADIUS)
    solution = lc.solve_rotating_cylinder(mesh, RADIUS, SPEED, CIRCULATION)
    theta = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
    exact = lc.exact_surface_speed(theta, RADIUS, SPEED, CIRCULATION)
    computed = np.linalg.norm(solution.surface_velocity, axis=1)
    relative = np.linalg.norm(computed - exact) / np.linalg.norm(exact)
    assert relative < 5.0e-3
    # The contour integral must reproduce the prescribed value, not merely a
    # flow that resembles the closed form.
    assert abs(solution.circulation_ccw - CIRCULATION) / CIRCULATION < 5.0e-3


@pytest.mark.parametrize("panels", [32, 64])
def test_lift_matches_kutta_joukowski(panels: int) -> None:
    mesh = regular_polygon(panels, RADIUS)
    solution = lc.solve_rotating_cylinder(mesh, RADIUS, SPEED, CIRCULATION)
    assert solution.lift_relative_error_vs_joukowski < 0.01
    # d'Alembert: the pressure integral has no drag component
    assert abs(solution.drag) < 1.0e-9


def test_exterior_field_matches_complex_potential() -> None:
    mesh = regular_polygon(64, RADIUS)
    solution = lc.solve_rotating_cylinder(mesh, RADIUS, SPEED, CIRCULATION)
    computed = lc.exterior_velocity_field(solution, PROBES)
    exact = lc.exact_velocity(PROBES, RADIUS, SPEED, CIRCULATION)
    assert np.linalg.norm(computed - exact) / np.linalg.norm(exact) < 1.0e-3


def test_stagnation_angles_from_surface_match_closed_form() -> None:
    mesh = regular_polygon(64, RADIUS)
    solution = lc.solve_rotating_cylinder(mesh, RADIUS, SPEED, CIRCULATION)
    found = sorted(lc.stagnation_angles_from_surface(mesh, solution.surface_velocity))
    reference = sorted(lc.stagnation_angles(RADIUS, SPEED, CIRCULATION))
    assert len(found) == len(reference) == 2
    assert max(
        lc.angle_difference(a, b) for a, b in zip(found, reference)
    ) < math.radians(0.2)


def test_error_decreases_with_refinement() -> None:
    """A converged route: the faceting error must shrink, not plateau."""
    errors = []
    for panels in (32, 64, 128):
        mesh = regular_polygon(panels, RADIUS)
        solution = lc.solve_rotating_cylinder(mesh, RADIUS, SPEED, CIRCULATION)
        errors.append(solution.lift_relative_error_vs_joukowski)
    assert errors[0] > errors[1] > errors[2]
    # second order in h: refinement by two must cut the error by ~4
    # two-second-order halvings: 4x refinement must cut the error by ~16
    assert 8.0 < errors[0] / errors[2] < 32.0


def test_supercritical_circulation_is_refused() -> None:
    critical = 4.0 * math.pi * SPEED * RADIUS
    with pytest.raises(ValueError):
        lc.stagnation_angles(RADIUS, SPEED, 1.01 * critical)


def test_reference_refuses_points_inside_the_body() -> None:
    with pytest.raises(ValueError):
        lc.exact_velocity(np.array([[0.5, 0.0]]), RADIUS, SPEED, CIRCULATION)


def test_case_route_meets_its_own_gates() -> None:
    """The case-pool route and the artifact criteria must agree."""
    row = case.route_circulation(64)
    assert row["lift_relative_error"] < case.TOLERANCE
    assert abs(row["drag_coefficient"]) < case.DRAG_TOL
    assert row["circulation_relative_error"] < case.CIRCULATION_FLUX_TOL
    sweep = case.circulation_sweep(64, circulations=(1.0, 3.0))
    assert sweep["max_lift_deviation_from_joukowski"] < case.TOLERANCE
    assert len(sweep["rows"]) == 2