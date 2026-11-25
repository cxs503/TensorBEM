import numpy as np
import pytest

from tensorbem import regular_polygon, solve_laplace_dirichlet
from tensorbem.solver import assemble_matrices, evaluate_interior_field


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


def test_assembled_double_layer_matrix_has_zero_self_terms_and_finite_entries():
    single, double = assemble_matrices(regular_polygon(32))
    assert np.isfinite(single).all()
    assert np.isfinite(double).all()
    assert np.allclose(np.diag(double), 0.0)
    assert np.all(np.diag(single) > 0.0)


def test_exact_panel_integration_preserves_refinement_trend():
    errors = []
    for n in (16, 32, 64):
        mesh = regular_polygon(n)
        result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
        theta = np.arctan2(mesh.normals[:, 1], mesh.normals[:, 0])
        errors.append(np.sqrt(np.mean((result.normal_derivative - np.cos(theta)) ** 2)))
    assert all(b < a for a, b in zip(errors, errors[1:]))


def test_dirichlet_solve_survives_an_infinite_condition_estimate(monkeypatch):
    """The condition-number gate must not reject a solvable system.

    The gate exists to reject the logarithmic-capacity degeneracy, not to
    depend on the LAPACK SVD succeeding: on the loongarch64 nodes
    ``numpy.linalg.cond`` returned ``inf`` for the 128-panel single-layer
    matrix whose true condition number is ~76, which made every case that
    touches this solve unrunnable there.  With the SVD unavailable the solve
    must still succeed, keep its algebraic residual, and record that the
    condition number came from the LU-based fallback.
    """
    mesh = regular_polygon(32)
    values = mesh.centers[:, 0]
    reference = solve_laplace_dirichlet(mesh, values)
    assert reference.condition_number_source == "svd"

    monkeypatch.setattr(np.linalg, "cond", lambda *a, **k: float("inf"))
    result = solve_laplace_dirichlet(mesh, values)
    assert result.condition_number_source == "power_iteration"
    assert np.isfinite(result.condition_number)
    assert result.relative_residual_l2 < 1.0e-10
    assert np.allclose(result.normal_derivative, reference.normal_derivative, rtol=0.0, atol=1.0e-12)


def test_dirichlet_gate_still_rejects_a_singular_system(monkeypatch):
    """If the fallback also fails to bound the condition number, raise."""
    import tensorbem.solver as solver_module

    mesh = regular_polygon(16)
    monkeypatch.setattr(np.linalg, "cond", lambda *a, **k: float("inf"))
    monkeypatch.setattr(
        solver_module,
        "condition_number_with_fallback",
        lambda matrix: (float("inf"), "power_iteration"),
    )
    with pytest.raises(ValueError, match="numerically singular"):
        solve_laplace_dirichlet(mesh, mesh.centers[:, 0])


@pytest.mark.parametrize(("size", "origin"), [(1.0, 0.0), (1.0, 2.0**45), (1e-60, 0.0), (1e60, 0.0)])
def test_rectangle_harmonic_field_in_scaled_coordinates(size, origin, record_property):
    from tensorbem.geometry import PanelMesh

    corners = np.array([[0, 0], [4, 0], [4, 2], [0, 2]])
    edges = np.roll(corners, -1, axis=0) - corners
    fractions = np.arange(16) / 16
    vertices = corners[:, None, :] + fractions[None, :, None] * edges[:, None, :]
    mesh = PanelMesh(origin + size * vertices.reshape(-1, 2))
    values = (mesh.centers[:, 0] - origin) / size
    solution = solve_laplace_dirichlet(mesh, values, reference_length=8 * size)
    probes = np.array([[1.0, 0.75], [2.5, 1.25]])
    computed = evaluate_interior_field(mesh, solution, origin + size * probes)
    error = float(np.max(abs(computed - probes[:, 0])))
    record_property("max_absolute_potential_error", error)
    assert error < 6e-6
