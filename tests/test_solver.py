import numpy as np
import pytest

from tensorbem import regular_polygon, solve_laplace_dirichlet


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
    # Constant Dirichlet data represent a constant harmonic field.
    assert np.sqrt(np.mean(result.normal_derivative**2)) < 0.15
