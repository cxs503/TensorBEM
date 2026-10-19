import numpy as np
import pytest

from tensorbem.kernels import (
    fundamental_solution,
    double_layer_kernel,
    integrate_single_layer_panel,
    integrate_double_layer_panel,
)


def test_fundamental_solution_reference_length_convention():
    assert np.isclose(float(fundamental_solution(2.0)), 0.0)
    assert np.isclose(float(fundamental_solution(1.0)), np.log(2.0) / (2.0 * np.pi))


def test_fundamental_solution_rejects_singularity():
    with pytest.raises(ValueError):
        fundamental_solution(0.0)


def test_double_layer_kernel():
    value = double_layer_kernel(
        np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    )
    assert np.isclose(value, 1.0 / (2.0 * np.pi))


def test_exact_single_layer_panel_self_integral():
    # Unit-length panel centred at the origin: integral -log(|s|/2)/(2*pi) ds.
    value = integrate_single_layer_panel(
        np.array([0.5, 0.0]), np.array([0.0, 0.0]),
        np.array([1.0, 0.0]), 1.0
    )
    expected = (1.0 + np.log(4.0)) / (2.0 * np.pi)
    assert np.isclose(value, expected, rtol=1e-12, atol=1e-12)


def test_exact_single_layer_integral_matches_high_order_quadrature_off_panel():
    from numpy.polynomial.legendre import leggauss

    target = np.array([0.4, 0.7])
    start = np.array([-0.5, 0.0])
    tangent = np.array([1.0, 0.0])
    length = 1.0
    value = integrate_single_layer_panel(target, start, tangent, length)
    nodes, weights = leggauss(400)
    points = start + (0.5 * (nodes + 1.0) * length)[:, None] * tangent
    distances = np.linalg.norm(target[None, :] - points, axis=1)
    reference = 0.5 * length * np.sum(
        weights * fundamental_solution(distances)
    )
    assert np.isclose(value, reference, rtol=1e-11, atol=1e-11)


def test_exact_double_layer_integral_matches_high_order_quadrature():
    from numpy.polynomial.legendre import leggauss

    target = np.array([0.4, 0.7])
    start = np.array([-0.5, 0.0])
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, -1.0])
    length = 1.0
    value = integrate_double_layer_panel(target, start, tangent, normal, length)
    nodes, weights = leggauss(400)
    points = start + (0.5 * (nodes + 1.0) * length)[:, None] * tangent
    delta = target[None, :] - points
    integrand = (delta @ normal) / (2.0 * np.pi * np.sum(delta**2, axis=1))
    reference = 0.5 * length * np.sum(weights * integrand)
    assert np.isclose(value, reference, rtol=1e-11, atol=1e-11)


def test_double_layer_principal_value_on_straight_panel_is_zero():
    value = integrate_double_layer_panel(
        np.array([0.5, 0.0]), np.array([0.0, 0.0]),
        np.array([1.0, 0.0]), np.array([0.0, -1.0]), 1.0
    )
    assert value == 0.0
