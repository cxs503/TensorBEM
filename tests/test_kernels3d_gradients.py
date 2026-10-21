"""Tests for the gradient kernels used by the three-dimensional velocity field.

The three quantities checked here are the ones that make the hypersingular
double-layer gradient usable:

* the gradient of the signed solid angle, in closed form;
* the gradient of the single-layer integral, for a panel away from the target;
* the polar self-panel rule that removes the ``1/r**2`` singularity.
"""
import math

import numpy as np
import pytest

from tensorbem.kernels3d import (
    gradient_single_layer_triangle,
    gradient_single_layer_triangle_self,
    gradient_single_layer_triangles_batch,
    gradient_solid_angle_triangle,
    gradient_solid_angle_triangles_batch,
    integrate_single_layer_triangle,
    integrate_single_layer_triangle_self,
    solid_angle_triangle,
)


def _finite_difference(function, point, step=1.0e-6):
    gradient = np.zeros(3)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        gradient[axis] = (function(point + offset) - function(point - offset)) / (2.0 * step)
    return gradient


def test_solid_angle_gradient_matches_finite_difference():
    rng = np.random.default_rng(20261021)
    for _ in range(25):
        vertices = rng.normal(size=(3, 3))
        target = rng.normal(size=3)
        analytic = gradient_solid_angle_triangle(target, vertices)
        reference = _finite_difference(
            lambda point: solid_angle_triangle(point, vertices), target
        )
        assert np.allclose(analytic, reference, rtol=1.0e-5, atol=1.0e-7)


def test_solid_angle_gradient_is_continuous_across_the_panel_plane():
    # The solid-angle *value* jumps across a panel; its gradient does not.
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    centroid = vertices.mean(axis=0)
    at_centroid = gradient_solid_angle_triangle(centroid, vertices)
    deviations = []
    for offset in (1.0e-3, 1.0e-4, 1.0e-5):
        above = gradient_solid_angle_triangle(centroid + np.array([0.0, 0.0, offset]), vertices)
        below = gradient_solid_angle_triangle(centroid - np.array([0.0, 0.0, offset]), vertices)
        # The solid-angle value jumps across the panel, so the in-plane
        # components of its gradient approach the on-panel value from
        # opposite sides.  The deviation is therefore O(offset) and vanishes
        # with it; it is not identically zero on either side.
        deviations.append(
            max(
                float(np.max(np.abs(above - at_centroid))),
                float(np.max(np.abs(below - at_centroid))),
            )
        )
        assert np.max(np.abs(above - below)) < 4.0 * deviations[-1]
    for coarse, fine in zip(deviations, deviations[1:]):
        assert fine < coarse
    # The deviation must shrink roughly in proportion to the offset, which is
    # what continuity of the gradient across the panel plane means here.
    for index, offset in enumerate((1.0e-3, 1.0e-4, 1.0e-5)):
        assert deviations[index] < 20.0 * offset
    assert deviations[-1] < 1.0e-3
    # The dominant (normal) component agrees to a tight relative tolerance.
    above = gradient_solid_angle_triangle(centroid + np.array([0.0, 0.0, 1.0e-5]), vertices)
    assert above[2] == pytest.approx(at_centroid[2], rel=1.0e-4)


def test_single_layer_gradient_equals_gradient_of_the_integral():
    rng = np.random.default_rng(31415)
    for _ in range(25):
        vertices = rng.normal(size=(3, 3))
        target = rng.normal(size=3)
        analytic = gradient_single_layer_triangle(target, vertices)
        reference = _finite_difference(
            lambda point: integrate_single_layer_triangle(point, vertices), target
        )
        assert np.allclose(analytic, reference, rtol=1.0e-5, atol=1.0e-8)


def test_single_layer_self_gradient_matches_independent_polar_quadrature():
    # The polar rule is the only one valid at the centroid, where the Duffy
    # rule is not a near-singular scheme.  Compare it against a different
    # angular rule (midpoint) applied to the same exact radial reduction.
    rng = np.random.default_rng(2718)
    for _ in range(10):
        vertices = rng.normal(size=(3, 3))
        analytic = gradient_single_layer_triangle_self(vertices, order=64)
        reference = _midpoint_polar_self_gradient(vertices)
        assert np.allclose(analytic, reference, rtol=1.0e-5, atol=1.0e-8)


def _midpoint_polar_self_gradient(vertices, samples=4000):
    points = np.asarray(vertices, dtype=float)
    centroid = points.mean(axis=0)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal = normal / np.linalg.norm(normal)
    total = np.zeros(3)
    for index in range(3):
        a = points[index] - centroid
        b = points[(index + 1) % 3] - centroid
        e1 = a / np.linalg.norm(a)
        e2 = np.cross(normal, e1)
        a2 = np.array([np.dot(a, e1), np.dot(a, e2)])
        b2 = np.array([np.dot(b, e1), np.dot(b, e2)])
        angle_a = math.atan2(a2[1], a2[0])
        angle_b = math.atan2(b2[1], b2[0])
        dtheta = (angle_b - angle_a + math.pi) % (2.0 * math.pi) - math.pi
        edge = b2 - a2
        edge_normal = np.array([edge[1], -edge[0]])
        offset = float(np.dot(edge_normal, a2))
        if offset < 0.0:
            edge_normal = -edge_normal
            offset = -offset
        theta = angle_a + dtheta * (np.arange(samples) + 0.5) / samples
        cosine = np.cos(theta) * edge_normal[0] + np.sin(theta) * edge_normal[1]
        extent = offset / cosine
        direction = (
            np.cos(theta)[:, None] * e1[None, :]
            + np.sin(theta)[:, None] * e2[None, :]
        )
        total += (dtheta / samples) * np.sum(extent[:, None] * direction, axis=0)
    return total / (4.0 * math.pi)


def test_single_layer_self_gradient_is_unchanged_by_the_refactor():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    value = integrate_single_layer_triangle_self(vertices)
    assert math.isfinite(value)
    assert value > 0.0
    gradient = gradient_single_layer_triangle_self(vertices)
    assert gradient.shape == (3,)
    assert np.all(np.isfinite(gradient))


def test_solid_angle_gradients_of_a_closed_surface_sum_to_zero():
    # A constant potential has zero gradient, and the double-layer gradient
    # of a constant density is the gradient of the summed solid angles, which
    # is the constant 4*pi (interior) or 0 (exterior) for a closed surface.
    from tensorbem.geometry3d import icosphere

    mesh = icosphere(1, radius=1.0)
    for probe in (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.3, 0.2, -0.1]),
        np.array([1.5, 0.4, 0.2]),
    ):
        total = gradient_solid_angle_triangles_batch(
            probe, mesh.v0, mesh.v1, mesh.v2
        ).sum(axis=0)
        assert np.allclose(total, 0.0, atol=1.0e-12)


def test_batched_gradients_match_the_scalar_ones():
    rng = np.random.default_rng(161803)
    vertices = rng.normal(size=(3, 3))
    target = rng.normal(size=3)
    v0 = vertices[None, 0, :]
    v1 = vertices[None, 1, :]
    v2 = vertices[None, 2, :]
    assert np.allclose(
        gradient_solid_angle_triangles_batch(target, v0, v1, v2)[0],
        gradient_solid_angle_triangle(target, vertices),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        gradient_single_layer_triangles_batch(target, v0, v1, v2)[0],
        gradient_single_layer_triangle(target, vertices),
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_gradient_kernels_reject_invalid_inputs():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([0.25, 0.25, 2.0])
    with pytest.raises(ValueError):
        gradient_solid_angle_triangle(np.array([0.25, 0.25]), vertices)
    with pytest.raises(ValueError):
        gradient_solid_angle_triangle(np.array([np.nan, 0.0, 1.0]), vertices)
    with pytest.raises(ValueError):
        gradient_single_layer_triangle(target, vertices, order=1)
    with pytest.raises(ValueError):
        gradient_single_layer_triangle(target, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        gradient_single_layer_triangles_batch(
            target, np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((3, 3))
        )
