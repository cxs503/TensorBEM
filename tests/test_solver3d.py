import math

import numpy as np
import pytest

from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import (
    LaplaceSurfaceResult,
    assemble_surface_matrices,
    evaluate_surface_field,
    solve_laplace_surface,
)


def test_assembled_matrices_have_exact_double_layer_row_sums():
    mesh = icosphere(1)
    single, double = assemble_surface_matrices(mesh)
    assert single.shape == (mesh.panel_count, mesh.panel_count)
    assert double.shape == single.shape
    assert np.isfinite(single).all()
    assert np.isfinite(double).all()
    assert np.allclose(np.diag(double), 0.0)
    assert np.all(np.diag(single) > 0.0)
    assert np.max(np.abs(np.sum(double, axis=1) + 0.5)) < 1.0e-12


def test_interior_linear_harmonic_flux_converges():
    errors = []
    for level in (1, 2):
        mesh = icosphere(level)
        result = solve_laplace_surface(mesh, mesh.centroids[:, 0], domain="interior")
        exact = mesh.normals[:, 0]
        errors.append(
            np.linalg.norm(result.normal_derivative - exact) / np.linalg.norm(exact)
        )
        assert result.domain == "interior"
        assert result.residual_l2 < 1.0e-10
        assert result.relative_residual_l2 < 1.0e-12
        assert result.condition_number > 0.0
        assert result.double_layer_row_sum_error < 1.0e-12
    assert errors[1] < errors[0]
    assert errors[1] < 2.0e-2


def test_exterior_constant_dirichlet_total_flux():
    mesh = icosphere(2)
    result = solve_laplace_surface(mesh, np.ones(mesh.panel_count), domain="exterior")
    total_flux = float(np.sum(result.normal_derivative * mesh.areas))
    exact_total_flux = 4.0 * math.pi
    assert abs(total_flux - exact_total_flux) / exact_total_flux < 2.0e-2
    assert abs(float(np.mean(result.normal_derivative)) - 1.0) < 2.0e-2
    assert result.relative_residual_l2 < 1.0e-12


def test_exterior_translating_sphere_added_mass_converges():
    errors = []
    for level in (1, 2):
        mesh = icosphere(level)
        radial = np.linalg.norm(mesh.centroids, axis=1)
        cos_theta = mesh.centroids[:, 0] / radial
        boundary = -0.5 * cos_theta
        result = solve_laplace_surface(mesh, boundary, domain="exterior")
        added_mass = float(np.sum(boundary * result.normal_derivative * mesh.areas))
        exact = 0.5 * 4.0 / 3.0 * math.pi
        errors.append(abs(added_mass - exact) / exact)
    assert errors[1] < errors[0]
    assert errors[1] < 2.0e-2


def test_interior_field_matches_linear_harmonic_solution():
    mesh = icosphere(2)
    result = solve_laplace_surface(mesh, mesh.centroids[:, 0], domain="interior")
    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [-0.2, 0.1, 0.0],
            [0.3, -0.1, 0.2],
            [0.0, 0.0, 0.0],
            [-0.4, 0.2, 0.1],
            [0.2, 0.3, -0.2],
        ]
    )
    values = evaluate_surface_field(mesh, result, points)
    errors = values - points[:, 0]
    assert values.shape == (len(points),)
    assert np.sqrt(np.mean(errors**2)) < 2.0e-3
    assert np.max(np.abs(errors)) < 4.0e-3


def test_exterior_field_matches_translating_sphere_potential():
    mesh = icosphere(2)
    radial = np.linalg.norm(mesh.centroids, axis=1)
    cos_theta = mesh.centroids[:, 0] / radial
    result = solve_laplace_surface(mesh, -0.5 * cos_theta, domain="exterior")
    points = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [-3.0, 0.0, 0.0]])
    values = evaluate_surface_field(mesh, result, points)
    exact = -0.5 * points[:, 0] / np.linalg.norm(points, axis=1) ** 3
    assert np.max(np.abs(values - exact)) < 4.0e-3


def test_field_evaluation_validates_points_domain_and_clearance():
    mesh = icosphere(1)
    result = solve_laplace_surface(mesh, mesh.centroids[:, 0], domain="interior")
    with pytest.raises(ValueError, match=r"shape \(M, 3\)"):
        evaluate_surface_field(mesh, result, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        evaluate_surface_field(mesh, result, np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="too close"):
        evaluate_surface_field(mesh, result, mesh.centroids[0][None, :])
    with pytest.raises(ValueError, match="match"):
        evaluate_surface_field(mesh, result, np.array([[0.0, 0.0, 0.0]]), domain="exterior")
    with pytest.raises(ValueError, match="domain"):
        evaluate_surface_field(mesh, result, np.array([[0.0, 0.0, 0.0]]), domain="invalid")
    values = evaluate_surface_field(mesh, result, np.array([[0.0, 0.0, 0.0]]), domain=None)
    assert values.shape == (1,)


def test_solver_rejects_invalid_inputs():
    mesh = icosphere(0)
    with pytest.raises(ValueError, match=f"shape \\({mesh.panel_count},\\)"):
        solve_laplace_surface(mesh, np.zeros(mesh.panel_count - 1))
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros(mesh.panel_count)
        bad[0] = np.nan
        solve_laplace_surface(mesh, bad)
    with pytest.raises(ValueError, match="domain"):
        solve_laplace_surface(mesh, np.zeros(mesh.panel_count), domain="outside")
    with pytest.raises(ValueError, match="quadrature_order"):
        solve_laplace_surface(mesh, np.zeros(mesh.panel_count), quadrature_order=1)
    with pytest.raises(TypeError):
        solve_laplace_surface("not a mesh", np.zeros(4))
    with pytest.raises(ValueError, match="quadrature_order"):
        assemble_surface_matrices(mesh, quadrature_order=1)
    with pytest.raises(TypeError):
        assemble_surface_matrices("not a mesh")


def test_result_dataclass_exposes_expected_fields():
    mesh = icosphere(0)
    result = solve_laplace_surface(mesh, mesh.centroids[:, 0])
    assert isinstance(result, LaplaceSurfaceResult)
    assert result.normal_derivative.shape == (mesh.panel_count,)
    assert result.boundary_values.shape == (mesh.panel_count,)
    assert result.single_layer_matrix.shape == (mesh.panel_count, mesh.panel_count)
    assert result.double_layer_matrix.shape == (mesh.panel_count, mesh.panel_count)
    assert result.quadrature_order == 8
