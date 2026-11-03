"""Verification of the exact flat-panel single-layer integrals (near-surface fields).

The closed form in :mod:`tensorbem.near_field3d` is checked against an
independent polar/analytic-radial scheme, against the analytic self-panel
integral, against the far-field monopole limit, and -- at the solver level --
against the closed-form potential of a sphere.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.geometry3d import icosphere
from tensorbem.kernels3d import duffy_quadrature_batch, integrate_single_layer_triangle_self
from tensorbem.near_field3d import (
    integrate_single_layer_flat_triangles,
    integrate_single_layer_flat_triangles_batch,
    polar_reference_integrals,
)
from tensorbem.solver3d import evaluate_surface_field, solve_laplace_surface

TRIANGLE = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]])


def _one(vertices):
    return vertices[0:1], vertices[1:2], vertices[2:3]


def _legacy_duffy_row(target, order=8):
    """The fixed-order Duffy rule the exact form replaces."""
    v0, v1, v2 = _one(TRIANGLE)
    points, weights = duffy_quadrature_batch(v0, v1, v2, order=order)
    points = points.reshape(-1, 3)
    weights = weights.reshape(-1)
    distance = np.linalg.norm(target[None, :] - points, axis=1)
    return float(np.sum(weights / (4.0 * np.pi * distance)))


@pytest.mark.parametrize("clearance", [2.0, 0.3, 0.05, 0.01, 1.0e-3])
def test_closed_form_matches_independent_polar_reference(clearance):
    v0, v1, v2 = _one(TRIANGLE)
    for target in (
        np.array([0.1, 0.1, clearance]),
        np.array([0.33, 0.06, clearance]),
        np.array([-0.2, 0.4, clearance]),
    ):
        potential, gradient = integrate_single_layer_flat_triangles(target, v0, v1, v2, order=24)
        ref_potential, ref_gradient = polar_reference_integrals(
            target, v0, v1, v2, angular_order=800
        )
        assert abs(potential[0] - ref_potential[0]) / abs(ref_potential[0]) < 2.0e-7
        assert (
            np.linalg.norm(gradient[0] - ref_gradient[0])
            / np.linalg.norm(ref_gradient[0])
            < 1.0e-5
        )


def test_exact_rule_beats_legacy_duffy_near_a_panel():
    v0, v1, v2 = _one(TRIANGLE)
    for clearance in (0.05, 0.01, 1.0e-3):
        target = np.array([0.1, 0.1, clearance])
        exact, _ = integrate_single_layer_flat_triangles(target, v0, v1, v2)
        reference, _ = polar_reference_integrals(target, v0, v1, v2, angular_order=800)
        legacy = _legacy_duffy_row(target)
        exact_error = abs(exact[0] - reference[0]) / reference[0]
        legacy_error = abs(legacy - reference[0]) / reference[0]
        assert exact_error < 1.0e-6
        assert legacy_error > 20.0 * max(exact_error, 1.0e-12)


def test_coplanar_target_matches_exact_self_panel_integral():
    """At zero clearance the panel integral equals the analytic polar self term."""
    v0, v1, v2 = _one(TRIANGLE)
    centroid = TRIANGLE.mean(axis=0)
    potential, _ = integrate_single_layer_flat_triangles(centroid, v0, v1, v2, order=32)
    exact = integrate_single_layer_triangle_self(TRIANGLE, order=64)
    assert abs(potential[0] - exact) / exact < 1.0e-7


def test_result_is_invariant_under_cyclic_vertex_rotation():
    target = np.array([0.1, 0.1, 2.0e-3])
    reference = None
    for shift in range(3):
        vertices = np.roll(TRIANGLE, shift, axis=0)
        potential, gradient = integrate_single_layer_flat_triangles(
            target, *_one(vertices), order=24
        )
        if reference is None:
            reference = (potential[0], gradient[0])
        assert abs(potential[0] - reference[0]) / abs(reference[0]) < 1.0e-12
        assert np.linalg.norm(gradient[0] - reference[1]) < 1.0e-12 * max(
            np.linalg.norm(reference[1]), 1.0e-12
        )


def test_edge_rule_converges_spectrally_in_order():
    target = np.array([0.1, 0.1, 1.0e-3])
    v0, v1, v2 = _one(TRIANGLE)
    _, coarse = integrate_single_layer_flat_triangles(target, v0, v1, v2, order=24)
    _, fine = integrate_single_layer_flat_triangles(target, v0, v1, v2, order=44)
    assert np.linalg.norm(coarse[0] - fine[0]) / np.linalg.norm(fine[0]) < 1.0e-6


def test_far_field_reduces_to_the_monopole_limit():
    area = 0.5 * np.linalg.norm(np.cross(TRIANGLE[1] - TRIANGLE[0], TRIANGLE[2] - TRIANGLE[0]))
    height = 500.0
    target = np.array([0.1, 0.1, height])
    potential, gradient = integrate_single_layer_flat_triangles(target, *_one(TRIANGLE))
    monopole = area / (4.0 * np.pi * height)
    assert abs(potential[0] - monopole) / monopole < 1.0e-6
    assert abs(gradient[0][2] + monopole / height) / (monopole / height) < 1.0e-5


def test_batch_matches_scalar_calls():
    targets = np.array([[0.1, 0.1, 1.0e-3], [0.2, 0.05, 0.3], [-0.3, 0.2, 2.0]])
    v0, v1, v2 = _one(TRIANGLE)
    potentials, gradients = integrate_single_layer_flat_triangles_batch(targets, v0, v1, v2)
    for index, target in enumerate(targets):
        single_potential, single_gradient = integrate_single_layer_flat_triangles(
            target, v0, v1, v2
        )
        assert potentials[index, 0] == single_potential[0]
        assert np.allclose(gradients[index, 0], single_gradient[0], rtol=0.0, atol=0.0)


def test_near_surface_sphere_field_is_accurate_and_does_not_blow_up():
    """Exterior Dirichlet u=1 on a sphere: phi = 1/r for every clearance.

    The Green representation is assembled with the exact single-layer rows of
    :mod:`tensorbem.near_field3d` (the built-in ``evaluate_surface_field``
    keeps its ``0.05 h_max`` clearance guard and rejects the closest rows).
    """
    from tensorbem.kernels3d import solid_angle_triangles_batch

    mesh = icosphere(2, radius=1.0)
    result = solve_laplace_surface(mesh, np.ones(mesh.panel_count), domain="exterior")
    clearances = np.array([0.3, 0.05, 0.01, 2.0e-3, 5.0e-4])
    targets = np.stack(
        [np.zeros_like(clearances), np.zeros_like(clearances), 1.0 + clearances], axis=1
    )
    single_rows, _ = integrate_single_layer_flat_triangles_batch(
        targets, mesh.v0, mesh.v1, mesh.v2, order=24
    )
    values = np.empty(len(targets))
    for index, target in enumerate(targets):
        double_row = -solid_angle_triangles_batch(target, mesh.v0, mesh.v1, mesh.v2) / (
            4.0 * np.pi
        )
        values[index] = (
            single_rows[index] @ result.normal_derivative
            + double_row @ result.boundary_values
        )
    exact = 1.0 / np.linalg.norm(targets, axis=1)
    error = np.abs(values - exact) / exact
    assert np.all(error < 2.0e-2)
    # the error must stay bounded by the discretisation level, not grow as the
    # field point approaches the surface
    assert error[-1] < 1.3 * error[0]


def test_far_field_values_agree_with_the_legacy_duffy_rule():
    """Away from the surface the two quadratures differ only by Duffy's own error."""
    from tensorbem.kernels3d import duffy_quadrature_batch, solid_angle_triangles_batch

    mesh = icosphere(2, radius=1.0)
    result = solve_laplace_surface(mesh, np.ones(mesh.panel_count), domain="exterior")
    target = np.array([0.0, 0.0, 2.0])
    value = evaluate_surface_field(mesh, result, target[None, :])[0]
    points, weights = duffy_quadrature_batch(mesh.v0, mesh.v1, mesh.v2, order=8)
    points = points.reshape(-1, 3)
    weights = weights.reshape(-1)
    distance = np.linalg.norm(target[None, :] - points, axis=1)
    single_row = (weights / (4.0 * np.pi * distance)).reshape(mesh.panel_count, -1).sum(axis=1)
    double_row = -solid_angle_triangles_batch(target, mesh.v0, mesh.v1, mesh.v2) / (4.0 * np.pi)
    legacy = single_row @ result.normal_derivative + double_row @ result.boundary_values
    # legacy Duffy's own error is ~1e-3 at this distance; the exact rule is the
    # more accurate of the two, so they differ by roughly that much
    assert abs(value - legacy) / abs(legacy) < 1.0e-2


def test_surface_points_are_still_rejected():
    mesh = icosphere(2, radius=1.0)
    result = solve_laplace_surface(mesh, np.ones(mesh.panel_count), domain="exterior")
    with pytest.raises(ValueError, match="too close"):
        evaluate_surface_field(mesh, result, mesh.centroids[:1])