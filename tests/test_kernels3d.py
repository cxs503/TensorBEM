import math

import numpy as np
import pytest

from tensorbem.kernels3d import (
    double_layer_kernel_3d,
    duffy_quadrature,
    duffy_quadrature_batch,
    fundamental_solution_3d,
    integrate_double_layer_triangle,
    integrate_single_layer_triangle,
    integrate_single_layer_triangle_self,
    solid_angle_triangle,
    solid_angle_triangles_batch,
    triangle_area,
)

UNIT_TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
VERTICAL_TRIANGLE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])


def test_fundamental_solution_and_double_layer_kernel_values():
    assert np.isclose(float(fundamental_solution_3d(1.0)), 1.0 / (4.0 * math.pi))
    assert np.isclose(float(fundamental_solution_3d(2.0)), 1.0 / (8.0 * math.pi))
    assert np.isclose(
        double_layer_kernel_3d(
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        1.0 / (4.0 * math.pi),
    )
    assert np.isclose(
        double_layer_kernel_3d(
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ),
        -1.0 / (4.0 * math.pi),
    )


def test_kernels_reject_singular_and_invalid_inputs():
    with pytest.raises(ValueError):
        fundamental_solution_3d(0.0)
    with pytest.raises(ValueError):
        double_layer_kernel_3d(np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        triangle_area(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="degenerate"):
        triangle_area(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))


def test_duffy_quadrature_integrates_polynomials_and_matches_batch():
    points, weights = duffy_quadrature(UNIT_TRIANGLE, order=8)
    assert points.shape == (64, 3)
    assert weights.shape == (64,)
    assert np.isclose(np.sum(weights), 0.5)
    assert np.isclose(np.sum(weights * points[:, 0]), 1.0 / 6.0, atol=1.0e-14)
    assert np.isclose(np.sum(weights * points[:, 1]), 1.0 / 6.0, atol=1.0e-14)
    inside = (
        (points[:, 0] >= -1.0e-14)
        & (points[:, 1] >= -1.0e-14)
        & (points[:, 0] + points[:, 1] <= 1.0 + 1.0e-14)
    )
    assert inside.all()

    batch_points, batch_weights = duffy_quadrature_batch(
        UNIT_TRIANGLE[None, 0], UNIT_TRIANGLE[None, 1], UNIT_TRIANGLE[None, 2], order=8
    )
    assert np.allclose(batch_points[0], points)
    assert np.allclose(batch_weights[0], weights)


def test_single_layer_off_panel_matches_high_order_reference():
    target = np.array([0.2, 0.1, 0.7])
    coarse = integrate_single_layer_triangle(target, VERTICAL_TRIANGLE, order=12)
    fine = integrate_single_layer_triangle(target, VERTICAL_TRIANGLE, order=32)
    assert np.isclose(coarse, fine, rtol=1.0e-8, atol=1.0e-12)


def test_single_layer_self_integral_matches_equilateral_analytic_value():
    side = 1.0
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side / 2.0, side * math.sqrt(3.0) / 2.0, 0.0],
        ]
    )
    inradius = side * math.sqrt(3.0) / 6.0
    exact = 3.0 * inradius / (2.0 * math.pi) * math.log(2.0 + math.sqrt(3.0))
    assert np.isclose(
        integrate_single_layer_triangle_self(triangle), exact, rtol=1.0e-12, atol=1.0e-14
    )
    assert np.isclose(
        integrate_single_layer_triangle_self(2.0 * triangle),
        2.0 * exact,
        rtol=1.0e-12,
    )
    for permutation in (
        triangle,
        triangle[[1, 2, 0]],
        triangle[[2, 0, 1]],
        triangle[[1, 0, 2]],
        triangle[[0, 2, 1]],
        triangle[[2, 1, 0]],
    ):
        assert np.isclose(
            integrate_single_layer_triangle_self(permutation),
            exact,
            rtol=1.0e-12,
        )


def test_single_layer_rejects_singular_in_panel_target():
    with pytest.raises(ValueError, match="singular"):
        integrate_single_layer_triangle(np.array([0.2, 0.2, 0.0]), UNIT_TRIANGLE)
    value = integrate_single_layer_triangle(np.array([2.0, 2.0, 0.0]), UNIT_TRIANGLE)
    assert np.isfinite(value) and value > 0.0


def test_double_layer_equals_negative_solid_angle_over_four_pi():
    target = np.zeros(3)
    angle = solid_angle_triangle(target, VERTICAL_TRIANGLE)
    assert np.isclose(angle, 0.3398369094541219, rtol=1.0e-13)
    value = integrate_double_layer_triangle(target, VERTICAL_TRIANGLE)
    assert np.isclose(value, -angle / (4.0 * math.pi), rtol=1.0e-13)
    # Independent high-order quadrature of (x-y).n/(4*pi*r**3).
    normal = np.array([0.0, 0.0, 1.0])
    points, weights = duffy_quadrature(VERTICAL_TRIANGLE, order=32)
    reference = np.sum(
        weights
        * np.einsum("kj,j->k", target[None, :] - points, normal)
        / (4.0 * math.pi * np.linalg.norm(points, axis=1) ** 3)
    )
    assert np.isclose(value, reference, rtol=1.0e-10, atol=1.0e-12)
    above = np.array([0.0, 0.0, 2.0])
    assert np.isclose(
        integrate_double_layer_triangle(above, VERTICAL_TRIANGLE),
        -solid_angle_triangle(above, VERTICAL_TRIANGLE) / (4.0 * math.pi),
    )
    assert integrate_double_layer_triangle(VERTICAL_TRIANGLE[0], VERTICAL_TRIANGLE) == 0.0


def test_solid_angle_batch_matches_scalar_values():
    triangles = np.stack(
        (
            VERTICAL_TRIANGLE,
            VERTICAL_TRIANGLE + np.array([0.5, 0.0, 0.0]),
            UNIT_TRIANGLE,
        )
    )
    target = np.array([0.1, 0.2, 0.3])
    batch = solid_angle_triangles_batch(
        target, triangles[:, 0], triangles[:, 1], triangles[:, 2]
    )
    scalar = np.array(
        [solid_angle_triangle(target, triangle) for triangle in triangles]
    )
    assert np.allclose(batch, scalar)


def test_duffy_quadrature_rejects_invalid_order_and_shapes():
    with pytest.raises(ValueError):
        duffy_quadrature(UNIT_TRIANGLE, order=1)
    with pytest.raises(ValueError):
        duffy_quadrature(UNIT_TRIANGLE, order=2.0)
