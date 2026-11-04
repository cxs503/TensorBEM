"""Domain and clearance guards for the 2-D interior field evaluator (P0-2).

Background: ``docs/project_review_2026-10-22.md`` §6.1 P0-2 recorded that
``evaluate_interior_field`` validated only the *shape* and finiteness of its
targets, never whether a target belongs to the domain. Outside the polygon the
interior Green representation converges to a finite number that is not the
solution, so the call reported a plausible-looking wrong value with no warning:

===========================  ==============  ==================
target (u = x, 32 panels)    returned        exact
===========================  ==============  ==================
``(2.0, 0.0)`` (outside)     ``0.000806``    ``2.0``
``(1.0, 0.0)`` (boundary)    ``0.4690``      ``1.0``
``(0.5, 0.0)`` (inside)      ``0.4992``      ``0.5``
===========================  ==============  ==================

The guards below are independent of the implementation they police: the
references are a closed-form square (distance and point-in-polygon are exact
for an axis-aligned box), an exact analytic harmonic field ``u = x``, and an
exactly degenerate input (a target on a vertex).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.geometry import PanelMesh, regular_polygon
from tensorbem.kernels import validate_clearance_factor
from tensorbem.solver import evaluate_interior_field, solve_laplace_dirichlet


def _unit_box() -> PanelMesh:
    """The 2 m x 1 m box ``x in [0, 2]``, ``y in [0, 1]`` (CCW)."""
    return PanelMesh(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]))


def test_point_in_polygon_matches_the_closed_form_box_indicator() -> None:
    box = _unit_box()
    points = np.array(
        [
            [1.0, 0.5],  # centre
            [0.0, 0.5],  # on the left edge
            [2.0, 0.5],  # on the right edge
            [-0.1, 0.5],  # just outside, left
            [2.1, 0.5],  # just outside, right
            [1.0, -0.1],  # just outside, below
            [1.0, 1.1],  # just outside, above
            [3.0, 0.5],  # far outside
            [0.5, 0.5],  # interior, off centre
        ]
    )
    # Exact indicator for the open box x in (0, 2), y in (0, 1); the two
    # on-edge targets are unclassified by a ray cast and are asserted only
    # through the clearance function below.
    expected = np.array([True, True, True, False, False, False, False, False, True])
    inside = box.contains_point(points)
    assert inside.dtype == bool
    assert np.array_equal(inside[[0, 3, 4, 5, 6, 7, 8]], expected[[0, 3, 4, 5, 6, 7, 8]])


def test_boundary_clearance_matches_the_closed_form_box_distance() -> None:
    box = _unit_box()
    points = np.array(
        [[1.0, 0.5], [0.1, 0.1], [1.9, 0.9], [1.0, 0.0], [1.4, 1.0], [2.5, 0.5], [0.0, 0.0]]
    )
    # min distance to the four sides of x in [0, 2], y in [0, 1]
    expected = np.array(
        [
            min(1.0, 1.0, 0.5, 0.5),
            min(0.1, 1.9, 0.1, 0.9),
            min(1.9, 0.1, 0.9, 0.1),
            0.0,
            0.0,
            0.5,
            0.0,
        ]
    )
    assert np.allclose(box.boundary_clearance(points), expected, atol=1.0e-15)


def test_circle_clearance_is_bounded_by_inscribed_and_circumscribed_radii() -> None:
    """A 64-gon's boundary lies between the inscribed and circumscribed circles."""
    mesh = regular_polygon(64)
    radius = 1.0
    apothem = np.cos(np.pi / 64)
    origin_clearance = float(mesh.boundary_clearance(np.array([[0.0, 0.0]]))[0])
    assert apothem - 1.0e-15 <= origin_clearance <= radius + 1.0e-15
    assert np.isclose(origin_clearance, apothem, atol=1.0e-14)
    # A point on a vertex has exactly zero clearance.
    vertex = mesh.vertices[7]
    assert abs(float(mesh.boundary_clearance(vertex[None, :])[0])) < 1.0e-15


def test_outside_targets_are_rejected_instead_of_returning_a_wrong_number() -> None:
    """The P0-2 regression: an exterior target now raises instead of lying."""
    mesh = regular_polygon(32)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0].copy())
    with pytest.raises(ValueError, match="strictly inside the domain"):
        evaluate_interior_field(mesh, result, np.array([[2.0, 0.0]]))
    # A batch is refused as a whole, and the message counts the offenders.
    with pytest.raises(ValueError, match=r"1 of 2 point"):
        evaluate_interior_field(mesh, result, np.array([[0.5, 0.0], [3.0, 0.0]]))


def test_boundary_targets_are_rejected() -> None:
    mesh = regular_polygon(32)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0].copy())
    # A polygon vertex is on the boundary to machine precision.
    with pytest.raises(ValueError, match="on or too close to the boundary"):
        evaluate_interior_field(mesh, result, mesh.vertices[:1])
    # So is the midpoint of a facet.
    with pytest.raises(ValueError, match="on or too close to the boundary"):
        evaluate_interior_field(mesh, result, mesh.centers[:1])


def test_clearance_factor_rejects_only_points_inside_its_band() -> None:
    mesh = regular_polygon(32)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0].copy())
    h_max = float(np.max(mesh.lengths))
    # distance from (r, 0) to the 32-gon facet; the apothem is cos(pi/32)
    apothem = np.cos(np.pi / 32)
    near = np.array([[apothem + 0.4 * h_max * 0.05, 0.0]])
    safe = np.array([[apothem - 100.0 * h_max * 0.05, 0.0]])
    assert float(mesh.boundary_clearance(near)[0]) < 0.05 * h_max
    assert float(mesh.boundary_clearance(safe)[0]) > 0.05 * h_max

    # The default factor 0.0 accepts the near-boundary point (the 2-D analytic
    # panel integrals are stable there); the explicit band rejects it.
    assert np.isfinite(evaluate_interior_field(mesh, result, near)[0])
    with pytest.raises(ValueError, match="on or too close to the boundary"):
        evaluate_interior_field(mesh, result, near, minimum_clearance_factor=0.05)
    assert np.isfinite(
        evaluate_interior_field(mesh, result, safe, minimum_clearance_factor=0.05)[0]
    )


@pytest.mark.parametrize("factor", [-1.0, -1.0e-12, float("nan"), float("inf"), True])
def test_clearance_factor_must_be_finite_and_non_negative(factor: float) -> None:
    """A factor <= 0 must not be able to disable the guard silently."""
    with pytest.raises(ValueError, match="finite non-negative"):
        validate_clearance_factor(factor)


def test_accepted_interior_points_still_match_the_analytic_field() -> None:
    """The guard must not damage the accuracy of legitimate interior targets."""
    mesh = regular_polygon(64)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0].copy())
    points = np.array([[0.1, 0.2], [-0.3, 0.1], [0.6, -0.2], [0.0, 0.0], [-0.5, -0.5]])
    values = evaluate_interior_field(mesh, result, points, minimum_clearance_factor=0.05)
    error = np.abs(values - points[:, 0])
    assert float(np.max(error)) < 2.0e-2
    assert float(np.sqrt(np.mean(error**2))) < 1.0e-2