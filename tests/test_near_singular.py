"""Tests for the near-singular field quadrature (``tensorbem.near_singular``).

The assertions encode the verification numbers of
``src/tensorbem/benchmarks/near_surface_field.py``: the adaptive rule is compared
with a converged reference of the same rule (order 14, ``split_ratio`` 2), which
isolates quadrature error from the discretisation error of the traces.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.near_surface_field import (
    _analytic_disturbance_velocity,
    _analytic_field,
    _exact_trace_result,
    run,
)
from tensorbem.geometry3d import icosphere
from tensorbem.kernels3d import duffy_quadrature_batch, solid_angle_triangles_batch
from tensorbem.near_singular import (
    single_layer_gradients,
    single_layer_integrals,
    triangle_distances_and_sizes,
)
from tensorbem.solver3d import (
    evaluate_surface_field,
    evaluate_surface_field_near_surface,
    evaluate_surface_velocity,
    evaluate_surface_velocity_near_surface,
    solve_laplace_surface,
)

_PANEL_V0 = np.array([[0.0, 0.0, 0.0]])
_PANEL_V1 = np.array([[0.1, 0.0, 0.0]])
_PANEL_V2 = np.array([[0.0, 0.1, 0.0]])
_REFERENCE = {"order": 14, "split_ratio": 2.0, "max_level": 32}
_FAST = {"order": 8, "split_ratio": 1.0, "max_level": 24}
_CLEARANCES = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)


def _target(ratio: float, scale: float = 0.1) -> np.ndarray:
    return np.array([0.3 * scale, 0.3 * scale, ratio * scale])


def test_triangle_distance_and_size():
    distance, size = triangle_distances_and_sizes(
        np.array([0.02, 0.02, 0.05]), _PANEL_V0, _PANEL_V1, _PANEL_V2
    )
    assert distance[0] == pytest.approx(0.05)
    assert size[0] == pytest.approx(0.1 * np.sqrt(2.0))

    # A point outside the triangle footprint: the distance is the edge distance,
    # not the (much smaller) distance to the panel plane.
    outside = np.array([0.3, -0.4, 0.0])
    distance, _ = triangle_distances_and_sizes(outside, _PANEL_V0, _PANEL_V1, _PANEL_V2)
    point = np.array([0.0, 0.0, 0.0])
    expected = min(
        np.linalg.norm(outside - point),
        np.linalg.norm(outside - np.array([0.1, 0.0, 0.0])),
    )
    assert distance[0] == pytest.approx(expected)
    assert distance[0] > 0.1


def test_single_layer_integral_converges_to_reference():
    for ratio in _CLEARANCES:
        target = _target(ratio)
        reference = single_layer_integrals(target, _PANEL_V0, _PANEL_V1, _PANEL_V2, **_REFERENCE)[0]
        value = single_layer_integrals(target, _PANEL_V0, _PANEL_V1, _PANEL_V2, **_FAST)[0]
        assert abs(value - reference) / abs(reference) < 1.0e-8


def test_fixed_order_rule_degrades_at_small_clearance():
    """The documented failure mode of the rule the new module replaces."""

    points, weights = duffy_quadrature_batch(_PANEL_V0, _PANEL_V1, _PANEL_V2, order=8)
    errors = {}
    for ratio in (1.0e-1, 1.0e-3, 1.0e-5):
        target = _target(ratio)
        reference = single_layer_integrals(target, _PANEL_V0, _PANEL_V1, _PANEL_V2, **_REFERENCE)[0]
        fixed = np.sum(
            weights / np.linalg.norm(target[None, None, :] - points, axis=2)
        ) / (4.0 * np.pi)
        errors[ratio] = abs(fixed - reference) / abs(reference)
    assert errors[1.0e-1] < 2.0e-2
    assert errors[1.0e-3] > 5.0e-2
    assert errors[1.0e-5] > 5.0e-2


def test_single_layer_gradient_converges_to_reference():
    for ratio in _CLEARANCES:
        target = _target(ratio)
        reference = single_layer_gradients(target, _PANEL_V0, _PANEL_V1, _PANEL_V2, **_REFERENCE)[0]
        value = single_layer_gradients(target, _PANEL_V0, _PANEL_V1, _PANEL_V2, **_FAST)[0]
        assert np.linalg.norm(value - reference) / np.linalg.norm(reference) < 1.0e-6


def test_adaptive_sum_matches_analytic_sphere_single_layer():
    """Sum over a closed sphere: ``sum_j int G dS = a**2 / r`` for ``r > a``."""

    mesh = icosphere(subdivisions=2, radius=1.0)
    target = np.array([0.0, 0.0, 1.032])
    total = single_layer_integrals(target, mesh.v0, mesh.v1, mesh.v2, **_FAST).sum()
    # The faceted mesh differs from the sphere by O(h**2): 1.7 % at 320 panels.
    assert total == pytest.approx(1.0 / target[2], rel=3.0e-2)
    # Refinement settings agree far more tightly than that band.
    fine = single_layer_integrals(target, mesh.v0, mesh.v1, mesh.v2, **_REFERENCE).sum()
    assert total == pytest.approx(fine, rel=1.0e-10)


def test_representation_quadrature_error_is_clearance_independent():
    """With exact traces, the adaptive rule reproduces its own converged limit."""

    mesh = icosphere(subdivisions=2, radius=1.0)
    result = _exact_trace_result(mesh)
    for ratio in (1.0e-2, 1.0e-3, 1.0e-4):
        target = np.array([[0.0, 0.0, 1.0 + ratio * float(mesh.h_max)]])
        value = evaluate_surface_field_near_surface(mesh, result, target)[0]
        reference = evaluate_surface_field_near_surface(
            mesh, result, target, quadrature_order=14, split_ratio=2.0, max_level=32
        )[0]
        assert abs(value - reference) / abs(reference) < 1.0e-8


def test_old_api_still_rejects_and_new_api_accepts():
    mesh = icosphere(subdivisions=1, radius=1.0)
    result = solve_laplace_surface(mesh, _analytic_field(mesh.centroids), domain="exterior")
    near = np.array([[0.0, 0.0, 1.0 + 1.0e-3 * float(mesh.h_max)]])
    with pytest.raises(ValueError):
        evaluate_surface_field(mesh, result, near)
    value = evaluate_surface_field_near_surface(mesh, result, near)[0]
    assert np.isfinite(value)
    with pytest.raises(ValueError):
        # Exactly on the surface is still rejected: the representation jumps there.
        evaluate_surface_field_near_surface(mesh, result, mesh.vertices[:1])


def test_near_surface_matches_existing_path_where_both_are_valid():
    """Regression: the new entry points must reproduce the old ones far away."""

    mesh = icosphere(subdivisions=1, radius=1.0)
    result = solve_laplace_surface(mesh, _analytic_field(mesh.centroids), domain="exterior")
    points = np.array([[0.0, 0.0, 2.0], [1.5, 0.2, 0.1]])
    field_old = evaluate_surface_field(mesh, result, points)
    field_new = evaluate_surface_field_near_surface(mesh, result, points)
    assert np.allclose(field_old, field_new, rtol=1.0e-9, atol=0.0)
    velocity_old = evaluate_surface_velocity(mesh, result, points)
    velocity_new = evaluate_surface_velocity_near_surface(mesh, result, points)
    assert np.allclose(velocity_old, velocity_new, rtol=1.0e-8, atol=1.0e-12)


def test_potential_error_is_bounded_and_clearance_independent():
    """End-to-end: the potential error stays O(h) however close the point is."""

    mesh = icosphere(subdivisions=1, radius=1.0)
    result = solve_laplace_surface(mesh, _analytic_field(mesh.centroids), domain="exterior")
    errors = []
    for ratio in (0.5, 0.1, 0.02, 1.0e-3):
        target = np.array([[0.0, 0.0, 1.0 + ratio * float(mesh.h_max)]])
        value = evaluate_surface_field_near_surface(mesh, result, target)[0]
        analytic = _analytic_field(target[0])
        errors.append(abs(value - analytic) / abs(analytic))
    assert max(errors) < 3.0e-2
    assert max(errors) < 5.0 * min(errors)


def test_benchmark_module_reports_the_documented_quantities():
    results = run(subdivisions=1)
    assert results["panel_quadrature"]["potential"][0]["adaptive_error"] < 1.0e-10
    assert results["panel_quadrature"]["potential"][-1]["adaptive_error"] < 1.0e-8
    assert results["panel_quadrature"]["potential"][-1]["fixed_order_error"] > 1.0e-2
    assert results["representation"]["rows"][-1]["adaptive_quadrature_error"] < 1.0e-8
    assert results["end_to_end"]["rows"][0]["old_api_rejected"] is False
    assert results["end_to_end"]["rows"][-1]["old_api_rejected"] is True
    assert results["end_to_end"]["rows"][-1]["relative_error"] < 3.0e-2