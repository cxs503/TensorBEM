import numpy as np
import pytest

from tensorbem import regular_polygon, solve_laplace_dirichlet
from tensorbem.solver import evaluate_interior_field


def test_solver_returns_finite_solution_and_small_algebraic_residual():
    mesh = regular_polygon(24)
    values = mesh.centers[:, 0]
    result = solve_laplace_dirichlet(mesh, values)
    assert result.normal_derivative.shape == (24,)
    assert np.isfinite(result.normal_derivative).all()
    assert result.residual_l2 < 1.0e-10
    assert result.condition_number > 0.0


def test_solver_rejects_wrong_boundary_value_shape():
    mesh = regular_polygon(12)
    with pytest.raises(ValueError, match="shape"):
        solve_laplace_dirichlet(mesh, np.zeros(11))


def test_constant_boundary_value_has_near_zero_flux():
    mesh = regular_polygon(48)
    result = solve_laplace_dirichlet(mesh, np.ones(48))
    assert np.sqrt(np.mean(result.normal_derivative**2)) < 3.0e-2


def test_manufactured_linear_harmonic_solution_improves_with_refinement():
    errors = []
    for n in (24, 48):
        mesh = regular_polygon(n)
        result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
        theta = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
        errors.append(np.sqrt(np.mean((result.normal_derivative - np.cos(theta)) ** 2)))
    assert errors[1] < errors[0]


def test_interior_field_matches_linear_harmonic_solution():
    mesh = regular_polygon(64)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
    points = np.array([[0.1, 0.2], [-0.3, 0.1], [0.6, -0.2]])

    values = evaluate_interior_field(mesh, result, points)

    assert values.shape == (len(points),)
    assert np.sqrt(np.mean((values - points[:, 0]) ** 2)) < 2.0e-2


def test_interior_field_rejects_invalid_evaluation_points():
    mesh = regular_polygon(16)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])

    with pytest.raises(ValueError, match=r"shape \(M, 2\)"):
        evaluate_interior_field(mesh, result, np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        evaluate_interior_field(mesh, result, np.array([[np.nan, 0.0]]))
