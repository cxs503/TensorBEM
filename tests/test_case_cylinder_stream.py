"""Tests for the exterior field evaluator and the cylinder-stream case pool entry.

The kernel gradients are checked against central differences of the panel
integrals they differentiate (no BEM involved); the case is then checked against
the closed-form cylinder complex potential, plus the guards that stop the
exterior representation from being evaluated where it is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks import case_cylinder_stream as case
from tensorbem.geometry import regular_polygon
from tensorbem.kernels import (
    integrate_double_layer_panel,
    integrate_double_layer_panel_gradient,
    integrate_single_layer_panel,
    integrate_single_layer_panel_gradient,
)
from tensorbem.solver import (
    evaluate_exterior_field,
    evaluate_exterior_surface_velocity,
    solve_laplace_dirichlet,
    solve_laplace_exterior_dirichlet,
    solve_laplace_exterior_neumann,
)


def _central_difference(function, point: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    gradient = np.empty(2, dtype=float)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        gradient[axis] = (function(point + offset) - function(point - offset)) / (
            2.0 * step
        )
    return gradient


@pytest.mark.parametrize("eta", [0.35, -0.6, 0.0])
@pytest.mark.parametrize("xi", [0.4, -0.9, 2.5])
def test_single_layer_gradient_matches_central_difference(xi: float, eta: float) -> None:
    start = np.array([0.3, -0.2])
    tangent = np.array([np.cos(0.7), np.sin(0.7)])
    normal = np.array([-tangent[1], tangent[0]])
    length = 1.7
    target = start + xi * tangent + eta * normal
    if eta == 0.0 and 0.0 < xi < length:
        pytest.skip("target lies on the log-singular panel")
    numeric = _central_difference(
        lambda point: integrate_single_layer_panel(point, start, tangent, length),
        target,
    )
    analytic = integrate_single_layer_panel_gradient(target, start, tangent, length)
    assert np.max(np.abs(numeric - analytic)) < 1.0e-8


@pytest.mark.parametrize("eta", [0.35, -0.6, 0.0, 1.0e-3])
@pytest.mark.parametrize("xi", [0.4, -0.9, 2.5])
def test_double_layer_gradient_matches_central_difference(xi: float, eta: float) -> None:
    start = np.array([0.3, -0.2])
    tangent = np.array([np.cos(0.7), np.sin(0.7)])
    normal = np.array([-tangent[1], tangent[0]])
    length = 1.7
    target = start + xi * tangent + eta * normal
    if eta == 0.0 and 0.0 < xi < length:
        pytest.skip("target lies on the panel, where the double-layer potential is "
                    "defined as its principal value and the gradient is singular")
    numeric = _central_difference(
        lambda point: integrate_double_layer_panel(
            point, start, tangent, normal, length
        ),
        target,
    )
    analytic = integrate_double_layer_panel_gradient(
        target, start, tangent, normal, length
    )
    assert np.max(np.abs(numeric - analytic)) < 1.0e-6


def _cylinder_field(panels: int = 32):
    mesh = regular_polygon(panels, case.RADIUS)
    data = case.exact_perturbation_potential(mesh.centers)
    return mesh, solve_laplace_exterior_dirichlet(mesh, data)


def test_field_gradient_matches_finite_difference_of_the_same_evaluator() -> None:
    mesh, result = _cylinder_field()
    points = case.probe_points(rings=(1.3, 2.0), angles=6)
    field = evaluate_exterior_field(mesh, result, points)
    step = 1.0e-6
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        numeric = (
            evaluate_exterior_field(mesh, result, points + offset).potential
            - evaluate_exterior_field(mesh, result, points - offset).potential
        ) / (2.0 * step)
        assert np.max(np.abs(numeric - field.gradient[:, axis])) < 1.0e-6


def test_exterior_field_agrees_with_complex_potential() -> None:
    mesh, result = _cylinder_field(panels=64)
    points = case.probe_points()
    field = evaluate_exterior_field(mesh, result, points)
    velocity = field.gradient + np.array([case.SPEED, 0.0])
    exact = case.exact_velocity(points)
    assert np.linalg.norm(velocity - exact) / np.linalg.norm(exact) < 0.03
    exact_potential = case.exact_perturbation_potential(points)
    assert np.linalg.norm(field.potential - exact_potential) / np.linalg.norm(
        exact_potential
    ) < 0.03


def test_exterior_field_refuses_interior_result() -> None:
    mesh = regular_polygon(32, case.RADIUS)
    interior = solve_laplace_dirichlet(mesh, np.zeros(mesh.panel_count))
    with pytest.raises(ValueError, match="exterior solve"):
        evaluate_exterior_field(mesh, interior, case.probe_points(rings=(2.0,)))


def test_exterior_field_refuses_points_inside_the_body_or_on_the_boundary() -> None:
    mesh, result = _cylinder_field()
    with pytest.raises(ValueError, match="strictly outside"):
        evaluate_exterior_field(mesh, result, np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="too close"):
        evaluate_exterior_field(mesh, result, mesh.centers)
    with pytest.raises(ValueError, match="too close"):
        evaluate_exterior_field(
            mesh, result, 1.02 * mesh.centers, minimum_clearance_factor=1.0
        )


def test_surface_velocity_is_normal_to_the_body_and_matches_the_reference() -> None:
    mesh = regular_polygon(64, case.RADIUS)
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    flux = case.SPEED * case.RADIUS**2 * (centers[:, 0] / r) / r**2
    result = solve_laplace_exterior_neumann(mesh, flux)
    velocity = evaluate_exterior_surface_velocity(mesh, result) + np.array(
        [case.SPEED, 0.0]
    )
    exact = case.exact_velocity(centers)
    assert np.linalg.norm(velocity - exact) / np.linalg.norm(exact) < 0.03
    normal_component = np.sum(velocity * mesh.normals, axis=1)
    assert np.max(np.abs(normal_component)) / case.SPEED <= 0.01


def test_case_acceptance_passes_on_the_quick_ladder() -> None:
    artifact = case.run_case(ladder=case.QUICK_LADDER, with_controls=False)
    assert artifact["overall"] == "PASS", artifact["acceptance"]["failed_checks"]
    assert artifact["geometry_gate"]["area_relative_deviation"] <= 1.0e-12
    assert (
        artifact["routes"]["neumann"]["finest"]["surface_velocity_relative_l2"]
        < case.TOLERANCE
    )
    assert (
        artifact["routes"]["dirichlet"]["finest"]["velocity_relative_l2"]
        < case.TOLERANCE
    )
    assert artifact["harmonicity"]["laplacian_scaled_max"] <= case.HARMONICITY_TOL


def test_committed_case_artifact_matches_the_committed_document() -> None:
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    results = root / "benchmarks/verified/case_bem_laplace_cylinder/results.json"
    payload = json.loads(results.read_text())
    assert payload["case_id"] == case.CASE_ID
    assert payload["overall"] == "PASS"
    assert payload["acceptance"]["failed_checks"] == []
    for path in (
        root / "benchmarks/verified/case_bem_laplace_cylinder/case.json",
        root / "docs/CASE-BEM-LAPLACE-CYLINDER.md",
        root / "benchmarks/verified/case_bem_laplace_cylinder/figures/field.png",
        root / "benchmarks/verified/case_bem_laplace_cylinder/figures/convergence.png",
        root / "benchmarks/verified/case_bem_laplace_cylinder/figures/near_wall.png",
    ):
        assert path.is_file(), path

def test_near_wall_window_and_exclusion_are_measured_not_assumed() -> None:
    """The documented validity window must be a measured property, not a note."""
    near = case.near_wall_study(ladder=(32, 64))
    window = near["window"]
    assert window["clearance_over_h_max"] == case.NEAR_WALL_WINDOW_FACTOR
    errors = [row["max_relative_velocity_error"] for row in window["levels"]]
    assert max(errors) <= case.NEAR_WALL_WINDOW_TOL
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    exclusion = [row["max_relative_velocity_error"] for row in near["exclusion"]["levels"]]
    assert min(exclusion) >= case.TOLERANCE
    low, high = case.NEAR_WALL_SATURATION_BAND
    assert low <= near["exclusion"]["saturation_ratio_coarsest_over_finest"] <= high
    assert near["guard"]["all_levels_rejected"]


def test_exterior_field_refuses_targets_inside_the_window() -> None:
    """A too-close target must raise instead of returning a plausible velocity."""
    import numpy as np
    import pytest

    mesh = case.regular_polygon(64, case.RADIUS)
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    result = case.solve_laplace_exterior_neumann(mesh, centers[:, 0] / r**2)
    h_max = float(np.max(mesh.lengths))
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    near_points = np.column_stack(
        (
            (case.RADIUS + h_max) * np.cos(theta),
            (case.RADIUS + h_max) * np.sin(theta),
        )
    )
    with pytest.raises(ValueError):
        case.evaluate_exterior_field(
            mesh, result, near_points, minimum_clearance_factor=4.0
        )
    # outside the window the same points are accepted
    field = case.evaluate_exterior_field(mesh, result, near_points)
    assert np.all(np.isfinite(field.gradient))
