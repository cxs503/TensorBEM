"""Tests for the 2-D exterior Dirichlet formulation.

The exterior operator is new: the cylinder benchmark could be answered with the
bounded-domain solver because its first harmonic mode is the same linear field
on both sides of the boundary. Here it is tested directly against the exact
exterior flux of a translating cylinder, against the bounded-domain solve, and
against the logarithmic-capacity identity of the single-layer kernel.
"""
import math

import numpy as np
import pytest

from tensorbem import regular_polygon
from tensorbem.geometry import PanelMesh, ellipse_mesh
from tensorbem.kernels import REFERENCE_LENGTH, validate_reference_length
from tensorbem.solver import (
    assemble_matrices,
    evaluate_interior_field,
    solve_laplace_dirichlet,
    solve_laplace_exterior_dirichlet,
)


def test_exterior_cylinder_flux_matches_the_analytic_trace():
    # Translating cylinder: u = -U*a*cos(theta) and q_e = -U*cos(theta).
    errors = []
    for panels in (32, 64, 128):
        mesh = regular_polygon(panels)
        boundary = -mesh.centers[:, 0]
        result = solve_laplace_exterior_dirichlet(mesh, boundary)
        theta = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
        exact = -np.cos(theta)
        errors.append(float(np.sqrt(np.mean((result.normal_derivative - exact) ** 2))))
        assert result.domain == "exterior"
        assert result.relative_residual_l2 < 1.0e-12
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 5.0e-4


def test_exterior_cylinder_added_mass_converges_to_pi():
    masses = []
    for panels in (32, 64, 128):
        mesh = regular_polygon(panels)
        boundary = -mesh.centers[:, 0]
        result = solve_laplace_exterior_dirichlet(mesh, boundary)
        masses.append(float(np.sum(boundary * result.normal_derivative * mesh.lengths)))
    assert abs(masses[-1] - math.pi) / math.pi < 2.0e-3
    assert abs(masses[-1] - math.pi) < abs(masses[0] - math.pi)


def test_exterior_and_bounded_domain_agree_for_the_cylinder_mode():
    # For the first harmonic mode of a circle the two formulations coincide in
    # the continuum, so both converge to the same -U*cos(theta) flux. Their
    # *discrete* fluxes differ by the discretisation error, which is the point
    # of the next assertion: agreement to machine precision is not expected.
    mesh = regular_polygon(64)
    boundary = -mesh.centers[:, 0]
    theta = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
    exact = -np.cos(theta)
    interior = solve_laplace_dirichlet(mesh, boundary).normal_derivative
    exterior = solve_laplace_exterior_dirichlet(mesh, boundary).normal_derivative
    assert np.max(np.abs(interior - exact)) < 5.0e-3
    assert np.max(np.abs(exterior - exact)) < 5.0e-3
    assert np.max(np.abs(interior - exterior)) < 5.0e-3


def test_exterior_ellipse_added_mass_matches_pi_b_squared():
    major, minor = 1.5, 0.75
    mesh = ellipse_mesh(128, major, minor)
    trace_scale = minor / major
    boundary = -trace_scale * mesh.centers[:, 0]
    result = solve_laplace_exterior_dirichlet(mesh, boundary)
    added = float(np.sum(boundary * result.normal_derivative * mesh.lengths))
    assert abs(added - math.pi * minor**2) / (math.pi * minor**2) < 2.0e-3


def test_solution_is_independent_of_the_reference_length():
    mesh = regular_polygon(48)
    values = mesh.centers[:, 0]
    interior_two = solve_laplace_dirichlet(mesh, values, reference_length=2.0)
    interior_three = solve_laplace_dirichlet(mesh, values, reference_length=3.7)
    assert np.allclose(
        interior_two.normal_derivative, interior_three.normal_derivative, atol=1.0e-12
    )
    exterior_two = solve_laplace_exterior_dirichlet(mesh, -values, reference_length=2.0)
    exterior_three = solve_laplace_exterior_dirichlet(
        mesh, -values, reference_length=3.7
    )
    assert np.allclose(
        exterior_two.normal_derivative,
        exterior_three.normal_derivative,
        atol=1.0e-12,
    )


def test_single_layer_of_constant_density_obeys_the_capacity_identity():
    # Integrating a constant density over a circle of radius a gives
    # a*log(reference_length/a). This is what makes the exterior first-kind
    # equation degenerate when capacity == reference_length.
    for radius in (0.5, 1.0, 4.0):
        for reference in (2.0, 3.5):
            mesh = regular_polygon(256, radius=radius)
            single, _ = assemble_matrices(mesh, reference_length=reference)
            mean_row = float(np.mean(single @ np.ones(mesh.panel_count)))
            expected = radius * math.log(reference / radius)
            assert math.isclose(mean_row, expected, rel_tol=1.0e-2, abs_tol=1.0e-6)


def test_exterior_evaluation_of_the_interior_field_is_refused():
    mesh = regular_polygon(16)
    result = solve_laplace_exterior_dirichlet(mesh, -mesh.centers[:, 0])
    with pytest.raises(ValueError, match="solve_laplace_dirichlet"):
        evaluate_interior_field(mesh, result, np.array([[0.1, 0.1]]))


def test_exterior_solve_rejects_invalid_inputs():
    mesh = regular_polygon(8)
    with pytest.raises(ValueError, match="shape"):
        solve_laplace_exterior_dirichlet(mesh, np.zeros(7))
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros(8)
        bad[0] = np.inf
        solve_laplace_exterior_dirichlet(mesh, bad)
    with pytest.raises(TypeError):
        evaluate_interior_field(mesh, "not a result", np.zeros((1, 2)))


def test_reference_length_must_be_positive_and_finite():
    assert validate_reference_length(2.0) == 2.0
    assert REFERENCE_LENGTH == 2.0
    for bad in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="reference_length"):
            validate_reference_length(bad)


def test_reference_length_is_rejected_by_the_assembly_and_solve():
    mesh = regular_polygon(8)
    with pytest.raises(ValueError, match="reference_length"):
        assemble_matrices(mesh, reference_length=0.0)
    with pytest.raises(ValueError, match="reference_length"):
        solve_laplace_dirichlet(mesh, np.zeros(8), reference_length=-2.0)


def test_ellipse_mesh_is_a_valid_inscribed_polygon():
    major, minor = 2.0, 0.5
    mesh = ellipse_mesh(64, major, minor)
    assert isinstance(mesh, PanelMesh)
    assert mesh.panel_count == 64
    assert mesh.area < math.pi * major * minor
    assert mesh.area / (math.pi * major * minor) > 0.99
    on_curve = (mesh.vertices[:, 0] / major) ** 2 + (mesh.vertices[:, 1] / minor) ** 2
    assert np.allclose(on_curve, 1.0, atol=1.0e-14)


def test_ellipse_mesh_and_regular_polygon_reject_bad_panel_counts():
    for bad in (2, 0, -3, 3.5, True, "16"):
        with pytest.raises(ValueError, match="panel_count"):
            ellipse_mesh(bad)
        with pytest.raises(ValueError, match="panel_count"):
            regular_polygon(bad)
    with pytest.raises(ValueError, match="semi_major"):
        ellipse_mesh(16, semi_major=0.0)
    with pytest.raises(ValueError, match="semi_minor"):
        ellipse_mesh(16, semi_minor=-1.0)
