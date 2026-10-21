"""Tests for the superconvergent recovery and the recovered surface velocity."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry3d import icosphere
from tensorbem.recovery import (
    panel_patches,
    recover_tangent_gradient,
    tangent_basis,
    vertex_incident_panels,
)
from tensorbem.solver3d import (
    evaluate_representation_surface_velocity_at_panels,
    evaluate_surface_velocity_at_panels,
    solve_laplace_surface,
)
from tensorbem.benchmarks.sphere_velocity import analytic_sphere_velocity


def test_vertex_incident_panels_cover_every_panel():
    mesh = icosphere(1, radius=1.0)
    incident = vertex_incident_panels(mesh)
    assert len(incident) == mesh.vertex_count
    total = sum(len(panels) for panels in incident)
    assert total == 3 * mesh.panel_count
    for panels in incident:
        assert panels == sorted(panels)


def test_panel_patches_contain_the_panel_itself():
    mesh = icosphere(2, radius=1.0)
    patches = panel_patches(mesh)
    assert len(patches) == mesh.panel_count
    for index, patch in enumerate(patches):
        assert index in patch
        assert patch == sorted(patch)
        assert len(patch) == len(set(patch))


def test_tangent_basis_is_orthonormal_and_perpendicular_to_the_normal():
    mesh = icosphere(1, radius=1.0)
    first, second = tangent_basis(mesh.normals)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(second, axis=1), 1.0)
    assert np.allclose(np.sum(first * second, axis=1), 0.0, atol=1.0e-12)
    assert np.allclose(np.sum(first * mesh.normals, axis=1), 0.0, atol=1.0e-12)
    assert np.allclose(np.sum(second * mesh.normals, axis=1), 0.0, atol=1.0e-12)


def test_tangent_basis_rejects_bad_shapes():
    with pytest.raises(ValueError):
        tangent_basis(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        tangent_basis(np.zeros((3, 3)))


def test_recovery_is_exact_for_an_affine_field():
    # On a flat-faceted mesh the samples of an affine function determine the
    # gradient exactly, so the tangential part is recovered to machine
    # precision.
    mesh = icosphere(2, radius=1.0)
    direction = np.array([0.3, -0.5, 0.8])
    direction = direction / np.linalg.norm(direction)
    samples = mesh.centroids @ direction
    recovered = recover_tangent_gradient(mesh, samples)
    expected = direction[None, :] - (
        direction @ mesh.normals.T
    )[:, None] * mesh.normals
    assert np.allclose(recovered, expected, atol=1.0e-12)


def test_recovery_returns_a_tangential_vector():
    mesh = icosphere(1, radius=1.0)
    recovered = recover_tangent_gradient(mesh, mesh.centroids[:, 0].copy())
    assert recovered.shape == (mesh.panel_count, 3)
    assert np.allclose(np.sum(recovered * mesh.normals, axis=1), 0.0, atol=1.0e-12)


def test_recovery_of_a_constant_field_is_zero():
    mesh = icosphere(1, radius=1.0)
    recovered = recover_tangent_gradient(mesh, np.ones(mesh.panel_count))
    assert np.allclose(recovered, 0.0, atol=1.0e-14)


def test_recovery_rejects_mismatched_and_invalid_input():
    mesh = icosphere(1, radius=1.0)
    with pytest.raises(TypeError):
        recover_tangent_gradient("not a mesh", np.ones(mesh.panel_count))
    with pytest.raises(ValueError):
        recover_tangent_gradient(mesh, np.ones(mesh.panel_count + 1))
    with pytest.raises(ValueError):
        recover_tangent_gradient(mesh, np.full(mesh.panel_count, np.nan))
    with pytest.raises(ValueError):
        recover_tangent_gradient(mesh, np.ones(mesh.panel_count), minimum_patch_size=2)


def _interior_reference(mesh) -> np.ndarray:
    # The interior harmonic extension of u = x is the linear function x.
    return np.tile([1.0, 0.0, 0.0], (mesh.panel_count, 1))


def test_recovered_surface_velocity_is_exact_for_a_linear_boundary_value():
    mesh = icosphere(1, radius=1.0)
    boundary = mesh.centroids[:, 0].copy()
    result = solve_laplace_surface(mesh, boundary, domain="interior")
    velocity = evaluate_surface_velocity_at_panels(mesh, result)
    assert np.allclose(velocity, _interior_reference(mesh), rtol=5.0e-2, atol=5.0e-2)


def test_recovered_surface_velocity_is_exact_for_a_constant_boundary_value():
    mesh = icosphere(1, radius=1.0)
    result = solve_laplace_surface(mesh, np.ones(mesh.panel_count), domain="interior")
    velocity = evaluate_surface_velocity_at_panels(mesh, result)
    assert np.allclose(velocity, 0.0, atol=1.0e-12)


def test_recovered_surface_velocity_beats_the_representation_one():
    mesh = icosphere(2, radius=1.0)
    radial = np.linalg.norm(mesh.centroids, axis=1)
    boundary = -0.5 * mesh.centroids[:, 0] / radial
    result = solve_laplace_surface(mesh, boundary, domain="exterior")
    exact = analytic_sphere_velocity(mesh.centroids, radius=1.0, speed=1.0)
    recovered = evaluate_surface_velocity_at_panels(mesh, result)
    representation = evaluate_representation_surface_velocity_at_panels(mesh, result)

    def relative(values: np.ndarray) -> float:
        return float(
            math.sqrt(
                float(np.sum((values - exact) ** 2)) / float(np.sum(exact**2))
            )
        )

    assert relative(recovered) < 0.1
    assert relative(recovered) < 0.2 * relative(representation)


def test_recovered_surface_velocity_converges_for_the_translating_sphere():
    errors = []
    for subdivisions in (1, 2, 3):
        mesh = icosphere(subdivisions, radius=1.0)
        radial = np.linalg.norm(mesh.centroids, axis=1)
        boundary = -0.5 * mesh.centroids[:, 0] / radial
        result = solve_laplace_surface(mesh, boundary, domain="exterior")
        exact = analytic_sphere_velocity(mesh.centroids, radius=1.0, speed=1.0)
        velocity = evaluate_surface_velocity_at_panels(mesh, result)
        errors.append(
            math.sqrt(
                float(np.sum((velocity - exact) ** 2)) / float(np.sum(exact**2))
            )
        )
    assert errors[0] > errors[1] > errors[2]
    # Second order in the panel size, with a margin for the O(h**2) faceting
    # error of the flat-faceted sphere.
    assert errors[0] / errors[1] > 2.5
    assert errors[1] / errors[2] > 2.5


def test_recovered_surface_velocity_converges_for_the_interior_sphere():
    errors = []
    for subdivisions in (1, 2, 3):
        mesh = icosphere(subdivisions, radius=1.0)
        boundary = mesh.centroids[:, 0].copy()
        result = solve_laplace_surface(mesh, boundary, domain="interior")
        velocity = evaluate_surface_velocity_at_panels(mesh, result)
        exact = _interior_reference(mesh)
        errors.append(
            math.sqrt(
                float(np.sum((velocity - exact) ** 2)) / float(np.sum(exact**2))
            )
        )
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] > 2.5
    assert errors[1] / errors[2] > 2.5


def test_surface_velocity_rejects_bad_input():
    mesh = icosphere(1, radius=1.0)
    result = solve_laplace_surface(
        mesh, np.ones(mesh.panel_count), domain="interior"
    )
    with pytest.raises(TypeError):
        evaluate_surface_velocity_at_panels("not a mesh", result)
    with pytest.raises(TypeError):
        evaluate_surface_velocity_at_panels(mesh, "not a result")
    with pytest.raises(ValueError):
        evaluate_representation_surface_velocity_at_panels(
            mesh, result, quadrature_order=1
        )
