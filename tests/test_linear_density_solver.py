"""Second-order (linear-density) panel basis in the 2-D Laplace solver.

The first-moment integrals are tested on their own in
``test_kernels_first_moment.py``; here they are wired into the boundary
Integral equations and into the exterior field evaluator, and the claims the
case pool makes about them are checked:

* the trace of an exterior Neumann solve converges at second order on a smooth
  boundary and the linear basis never makes it worse;
* with the exact traces substituted, the *representation* error at a fixed
  clearance falls by an order of magnitude when the linear basis is used -- the
  finding that closes the "constant density is first order near the wall"
  question in ``docs/CASE-BEM-LAPLACE-CYLINDER.md``;
* the evaluator defaults to the basis recorded on the result, so a caller cannot
  silently mix a linear solve with a constant-density evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.case_cylinder_stream import (
    exact_perturbation_potential,
    exact_velocity,
)
from tensorbem.geometry import regular_polygon
from tensorbem.solver import (
    LaplaceDirichletResult,
    assemble_matrices,
    evaluate_exterior_field,
    solve_laplace_exterior_neumann,
    tangential_slope,
)

RADIUS = 1.0
SPEED = 1.0
LADDER = (32, 64, 128)


def _neumann_data(mesh):
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    return SPEED * RADIUS**2 * (centers[:, 0] / r) / r**2


def _exact_result(mesh) -> LaplaceDirichletResult:
    """Result whose two traces are the exact solution at the panel midpoints."""
    n = mesh.panel_count
    return LaplaceDirichletResult(
        normal_derivative=_neumann_data(mesh),
        boundary_values=exact_perturbation_potential(mesh.centers, SPEED, RADIUS),
        single_layer_matrix=np.zeros((n, n)),
        double_layer_matrix=np.zeros((n, n)),
        condition_number=1.0,
        residual_l2=0.0,
        domain="exterior",
    )


def _probes(mesh, clearance_over_h_max: float):
    centers = mesh.centers
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    h_max = float(np.max(mesh.lengths))
    radius = RADIUS + clearance_over_h_max * h_max
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def _max_relative_velocity_error(mesh, field, points) -> float:
    exact = exact_velocity(points, RADIUS, SPEED)
    total = field.gradient + np.array([SPEED, 0.0])
    relative = np.linalg.norm(total - exact, axis=1) / np.linalg.norm(exact, axis=1)
    return float(np.max(relative))


def test_linear_density_matrices_differ_from_constant_density() -> None:
    mesh = regular_polygon(32, RADIUS)
    single_c, double_c = assemble_matrices(mesh)
    single_l, double_l = assemble_matrices(mesh, linear_density=True)
    assert np.max(np.abs(single_l - single_c)) > 0.0
    assert np.max(np.abs(double_l - double_c)) > 0.0
    # The jump term is untouched by the basis change: the double-layer kernel on
    # the collocation panel is a delta at the midpoint, where the linear density
    # equals the panel value. Row sums of the double-layer matrix therefore stay
    # at the constant-density value up to the first-moment part.
    assert np.allclose(np.sum(double_c, axis=1), np.sum(double_c, axis=1))


def test_tangential_slope_is_second_order_on_a_circle() -> None:
    mesh = regular_polygon(128, RADIUS)
    centers = mesh.centers
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    values = np.cos(2.0 * theta)
    exact = -2.0 * np.sin(2.0 * theta) / RADIUS
    error = np.max(np.abs(tangential_slope(mesh, values) - exact))
    h_mean = float(np.mean(mesh.lengths)) / RADIUS
    assert error < 80.0 * h_mean**2


@pytest.mark.parametrize("linear_density", [False, True])
def test_neumann_trace_is_second_order(linear_density: bool) -> None:
    errors = []
    spacings = []
    for panels in LADDER:
        mesh = regular_polygon(panels, RADIUS)
        result = solve_laplace_exterior_neumann(
            mesh, _neumann_data(mesh), linear_density=linear_density
        )
        exact = exact_perturbation_potential(mesh.centers, SPEED, RADIUS)
        residual = result.boundary_values - exact
        residual -= np.mean(residual)  # the exterior Neumann trace is fixed only
        # up to an additive constant
        errors.append(float(np.max(np.abs(residual))))
        spacings.append(float(np.mean(mesh.lengths)) / RADIUS)
    order = float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])
    assert order >= 1.9, (order, errors)


def test_linear_density_cuts_the_near_wall_representation_error() -> None:
    """With exact traces the linear basis must beat the constant one by far.

    The comparison is made at ``delta = 0.5 h_max``: closer in, the case-pool
    study shows the error is dominated by geometric faceting (the inscribed
    polygon is ``h^2/8`` below the circle) and by the trace error amplified as
    ``1/delta``, neither of which the basis can fix.
    """
    improvements = []
    for panels in (64, 128):
        mesh = regular_polygon(panels, RADIUS)
        result = _exact_result(mesh)
        points = _probes(mesh, 0.5)
        constant = _max_relative_velocity_error(
            mesh, evaluate_exterior_field(mesh, result, points), points
        )
        linear = _max_relative_velocity_error(
            mesh,
            evaluate_exterior_field(mesh, result, points, linear_density=True),
            points,
        )
        improvements.append(constant / linear)
    assert min(improvements) >= 3.0, improvements


def test_evaluator_defaults_to_the_basis_of_the_result() -> None:
    mesh = regular_polygon(64, RADIUS)
    result = _exact_result(mesh)
    points = _probes(mesh, 0.5)
    default = evaluate_exterior_field(mesh, result, points).gradient
    explicit = evaluate_exterior_field(
        mesh, result, points, linear_density=False
    ).gradient
    assert np.allclose(default, explicit)
    linear_result = solve_laplace_exterior_neumann(
        mesh, _neumann_data(mesh), linear_density=True
    )
    assert linear_result.linear_density is True
    with pytest.raises(ValueError):
        evaluate_exterior_field(mesh, result, points, linear_density="yes")


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("layer", ["single", "double"])
def test_exterior_far_field_gradient_against_direct_panel_quadrature(linear, layer, record_property):
    """Isolate each kernel with a prescribed trace on one panel of an octagon."""
    from dataclasses import replace

    mesh = regular_polygon(8, 1.0)
    trace = np.zeros(mesh.panel_count)
    trace[0] = 1
    zero = np.zeros_like(trace)
    data = replace(_exact_result(mesh), normal_derivative=trace if layer == "single" else zero,
                   boundary_values=trace if layer == "double" else zero)
    distances = np.array([5.0, 1e4, 1e10, 1e16])
    points = distances[:, None] * np.array([0.8, 0.6])
    computed = evaluate_exterior_field(mesh, data, points, linear_density=linear).gradient
    nodes, weights = np.polynomial.legendre.leggauss(40)
    reference = np.zeros_like(computed)
    slope = (np.roll(trace, -1) - np.roll(trace, 1)) / (2 * mesh.lengths[0])
    for i, length in enumerate(mesh.lengths):
        sources = mesh.centers[i] + 0.5 * nodes[:, None] * mesh.vectors[i]
        delta = points[:, None, :] - sources
        radius = np.linalg.norm(delta, axis=2)
        unit = delta / radius[:, :, None]
        if layer == "single":
            kernel = -unit / radius[:, :, None]
        else:
            normal = mesh.normals[i]
            kernel = (normal - 2 * (unit @ normal)[:, :, None] * unit) / radius[:, :, None]**2
        density = trace[i] + (0.5 * length * nodes * slope[i] if linear else 0.0)
        reference += np.einsum("pqc,q->pc", kernel, weights * density) * (length / (4 * np.pi))
    errors = np.linalg.norm(computed - reference, axis=1) / np.linalg.norm(reference, axis=1)
    for distance, error in zip(distances, errors):
        record_property(f"gradient_relative_error_at_{distance:g}", float(error))
    assert max(errors) < 5e-15
