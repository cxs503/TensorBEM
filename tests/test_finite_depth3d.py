"""Verification of the 3-D finite-depth free-surface Green function.

The kernel in :mod:`tensorbem.finite_depth3d` is checked against properties that
are fixed before any number is computed:

* the vertical eigenvalue problem (dispersion, deep-water mode factor),
* the two boundary conditions it is supposed to satisfy
  (``dG/dz - nu*G = 0`` at ``z = 0`` and ``dG/dz = 0`` at ``z = -h``),
* reciprocity and the analytic gradient,
* the ``k0*h -> inf`` limit against :mod:`tensorbem.free_surface3d`, whose
  Kelvin kernel is validated against Hulme (1982) and Wang/Linton (1991),
* the classical dimension-reduction identity that links the evanescent sector to
  the 2-D kernel :mod:`tensorbem.finite_depth`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import finite_depth
from tensorbem.analytic import bessel_j01_y01
from tensorbem.finite_depth3d import (
    convergence_history,
    green_function,
    green_gradient_field,
    modified_bessel_k01,
    propagating_mode_factor,
    suggested_mode_count,
)
from tensorbem.free_surface3d import green_function_3d


def _pair(radius, z, z0):
    return (
        np.array([[radius, 0.0, z]], dtype=float),
        np.array([[0.0, 0.0, z0]], dtype=float),
    )


# --------------------------------------------------------------------------- #
# modified Bessel functions used by the evanescent sector
# --------------------------------------------------------------------------- #


def test_modified_bessel_matches_reference_values():
    scipy_special = pytest.importorskip("scipy.special")
    x = np.array([1e-6, 1e-3, 0.1, 1.0, 2.0, 7.99, 8.0, 8.01, 20.0, 120.0, 700.0])
    k0, k1 = modified_bessel_k01(x)
    assert np.allclose(k0, scipy_special.k0(x), rtol=1e-11)
    assert np.allclose(k1, scipy_special.k1(x), rtol=1e-10)


def test_modified_bessel_satisfies_its_differential_identities():
    """K_0' = -K_1 and the K_1 recurrence x*K_1' + K_1 + x*K_0 = 0."""
    x = np.linspace(0.05, 60.0, 300)
    k0, k1 = modified_bessel_k01(x)
    step = 1e-7
    d_k0 = (modified_bessel_k01(x + step)[0] - modified_bessel_k01(x - step)[0]) / (2.0 * step)
    d_k1 = (modified_bessel_k01(x + step)[1] - modified_bessel_k01(x - step)[1]) / (2.0 * step)
    assert np.allclose(d_k0, -k1, rtol=2e-6, atol=1e-12)
    residual = x * d_k1 + k1 + x * k0
    scale = np.abs(k1) + x * np.abs(k0)
    assert np.max(np.abs(residual) / scale) < 1e-5


# --------------------------------------------------------------------------- #
# vertical eigenvalue problem
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("depth", [1.5, 8.0, 40.0, 300.0])
@pytest.mark.parametrize("nu", [0.05, 0.4, 3.0])
def test_propagating_mode_is_the_dispersive_one(nu, depth):
    k0 = finite_depth.finite_depth_wavenumber(nu, depth)
    assert k0 * math.tanh(k0 * depth) == pytest.approx(nu, rel=1e-12)
    # phi_0(z)/sqrt(N_0) must satisfy the free-surface and bottom conditions
    z = np.linspace(-depth, 0.0, 41)
    factor = propagating_mode_factor(z, nu, depth)
    h = 1e-7 * min(1.0, depth)
    top = (
        propagating_mode_factor(np.array([h]), nu, depth)
        - propagating_mode_factor(np.array([-h]), nu, depth)
    ) / (2.0 * h)
    bottom = (
        propagating_mode_factor(np.array([-depth + h]), nu, depth)
        - propagating_mode_factor(np.array([-depth - h]), nu, depth)
    ) / (2.0 * h)
    assert top[0] == pytest.approx(nu * factor[-1], rel=1e-6)
    assert abs(bottom[0]) <= max(1e-4 * abs(factor[0]), 1e-30)


@pytest.mark.parametrize("nu", [0.1, 1.0])
def test_deep_water_propagating_factor_limit(nu):
    """P_0 = phi_0(z)*phi_0(z0)/N_0 -> 2*k0*exp(k0*(z+z0)) once k0*h is large."""
    depth = 260.0
    k0 = finite_depth.finite_depth_wavenumber(nu, depth)
    z, z0 = -0.5, -0.25
    factor = propagating_mode_factor(np.array([z, z0]), nu, depth)
    value = factor[0] * factor[1]
    expected = 2.0 * k0 * math.exp(k0 * (z + z0))
    assert value == pytest.approx(expected, rel=1e-4)

    # and the same factor must reproduce the exact N_0 integral
    grid = np.linspace(-depth, 0.0, 200001)
    phi = propagating_mode_factor(grid, nu, depth)
    norm = np.trapezoid(phi * phi, grid)
    assert norm == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# boundary conditions, gradient, reciprocity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nu,depth", [(0.35, 12.0), (0.05, 2.0), (2.0, 60.0)])
def test_free_surface_boundary_condition(nu, depth):
    rng = np.random.default_rng(7)
    count = 6
    radius = rng.uniform(0.1, 5.0, count)
    source = np.stack(
        [np.zeros(count), np.zeros(count), rng.uniform(-0.95 * depth, -0.05 * depth, count)],
        axis=1,
    )
    target = np.stack([radius, rng.uniform(-0.5, 0.5, count), np.zeros(count)], axis=1)
    value = green_function(target, source, nu, depth)
    _, _, dg_dz = green_gradient_field(target, source, nu, depth)
    residual = np.abs(dg_dz - nu * value)
    assert np.max(residual) < 1e-11 * max(1.0, np.max(np.abs(value)))


@pytest.mark.parametrize("nu,depth", [(0.35, 12.0), (0.05, 2.0), (2.0, 60.0)])
def test_bottom_boundary_condition(nu, depth):
    rng = np.random.default_rng(11)
    count = 6
    radius = rng.uniform(0.1, 5.0, count)
    source = np.stack(
        [np.zeros(count), np.zeros(count), rng.uniform(-0.95 * depth, -0.02 * depth, count)],
        axis=1,
    )
    target = np.stack(
        [radius, rng.uniform(-0.5, 0.5, count), np.full(count, -depth)], axis=1
    )
    value = green_function(target, source, nu, depth)
    _, _, dg_dz = green_gradient_field(target, source, nu, depth)
    assert np.max(np.abs(dg_dz)) < 1e-12 * max(1.0, np.max(np.abs(value)))


def test_gradient_matches_central_differences():
    nu, depth = 0.35, 12.0
    rng = np.random.default_rng(3)
    count = 5
    target = np.stack(
        [rng.uniform(0.2, 4.0, count), rng.uniform(-1.0, 1.0, count),
         rng.uniform(-0.95 * depth, -0.1 * depth, count)],
        axis=1,
    )
    source = np.stack(
        [np.zeros(count), np.zeros(count), rng.uniform(-0.9 * depth, -0.2 * depth, count)],
        axis=1,
    )
    analytic = green_gradient_field(target, source, nu, depth)
    step = 1e-6
    for axis, component in ((0, analytic[0]), (1, analytic[1]), (2, analytic[2])):
        plus = target.copy()
        minus = target.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        numeric = (
            green_function(plus, source, nu, depth) - green_function(minus, source, nu, depth)
        ) / (2.0 * step)
        assert np.max(np.abs(component - numeric)) < 1e-6 * max(1.0, np.max(np.abs(numeric)))


def test_reciprocity_and_horizontal_translation_invariance():
    nu, depth = 0.6, 9.0
    rng = np.random.default_rng(5)
    count = 5
    first = np.stack(
        [rng.uniform(-3.0, 3.0, count), rng.uniform(-3.0, 3.0, count),
         rng.uniform(-0.9 * depth, -0.1 * depth, count)],
        axis=1,
    )
    second = np.stack(
        [rng.uniform(-3.0, 3.0, count), rng.uniform(-3.0, 3.0, count),
         rng.uniform(-0.9 * depth, -0.1 * depth, count)],
        axis=1,
    )
    direct = green_function(first, second, nu, depth)
    swapped = green_function(second, first, nu, depth)
    assert np.allclose(direct, swapped, rtol=0.0, atol=0.0)

    shift = np.array([13.7, -4.2, 0.0])
    shifted = green_function(first + shift, second + shift, nu, depth)
    assert np.allclose(direct, shifted, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------------------- #
# deep-water limit against the validated Kelvin kernel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("point", [(1.0, -0.5, -0.5), (2.0, 0.0, -1.0), (0.7, -0.2, -2.0)])
def test_deep_water_limit_against_kelvin_kernel(point):
    nu = 1.0
    radius, z, z0 = point
    target, source = _pair(radius, z, z0)
    reference = green_function_3d(target, source, nu)[0]
    value = green_function(target, source, nu, 240.0)[0]
    assert abs(value - reference) / abs(reference) < 1e-6


def test_deep_water_limit_converges_first_order_in_inverse_depth():
    nu = 1.0
    target, source = _pair(1.0, -0.5, -0.5)
    reference = green_function_3d(target, source, nu)[0]
    errors = [
        abs(green_function(target, source, nu, depth)[0] - reference) / abs(reference)
        for depth in (30.0, 60.0, 120.0, 240.0)
    ]
    # the leading finite-depth correction is the bottom image, O(1/depth)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(ratio > 5.0 for ratio in ratios)


# --------------------------------------------------------------------------- #
# dimension reduction onto the 2-D kernel
# --------------------------------------------------------------------------- #


def test_evanescent_sector_reduces_to_the_2d_kernel_term():
    """integral K_0(mu*hypot(X, y)) dy = (pi/mu) * exp(-mu*|X|)."""
    mu, x = 1.3, 0.8
    y = np.linspace(0.0, 60.0, 600001)
    integrand = modified_bessel_k01(mu * np.hypot(x, y))[0]
    integral = 2.0 * np.trapezoid(integrand, y)
    expected = math.pi / mu * math.exp(-mu * x)
    assert integral == pytest.approx(expected, rel=1e-9)


def test_propagating_sector_kernel_normalisation_matches_kelvin_limit():
    """The wave part tends to ``(k0/2)*(J_0 + i*Y_0)``-type Kelvin form, i.e. its
    *radiating* (imaginary) sector is ``(k0/2)*J_0(k0*R)`` and the total agrees
    with the deep-water kernel to a couple of percent at ``k0*h = 200``."""
    nu, depth = 0.5, 400.0
    k0 = finite_depth.finite_depth_wavenumber(nu, depth)
    radius = np.array([2.5])
    target, source = _pair(2.5, 0.0, 0.0)
    value = green_function(target, source, nu, depth, include_rankine=False)
    reference = green_function_3d(target, source, nu, include_rankine=False)[0]
    # the evanescent remainder is exponentially small here, so the wave part must match
    assert abs(value[0] - reference) / abs(reference) < 2e-2
    # the radiating sector carries exactly (k0/2)*J_0(k0*R) in its imaginary part
    j0, _, _, _ = bessel_j01_y01(k0 * radius)
    assert value[0].imag == pytest.approx(0.5 * k0 * j0[0], rel=1e-3)


# --------------------------------------------------------------------------- #
# truncation rule and input guards
# --------------------------------------------------------------------------- #


def test_truncation_history_is_monotone_and_rule_is_sufficient():
    nu, depth = 1.0, 20.0
    target, source = _pair(0.9, -1.0, -0.7)
    history = convergence_history(target, source, nu, depth, (16, 32, 64, 128, 256, 512))
    errors = [record["relative_to_finest"] for record in history]
    assert errors[-1] == 0.0
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))
    # the rule must actually deliver the requested tolerance
    count = suggested_mode_count(nu, depth, 0.9, tolerance=1e-8)
    fine = green_function(target, source, nu, depth, modes=8 * count)[0]
    rule = green_function(target, source, nu, depth, modes=count)[0]
    assert abs(rule - fine) / abs(fine) < 1e-5


def test_input_guards():
    with pytest.raises(ValueError):
        green_function(
            np.array([[0.0, 0.0, -1.0]]), np.array([[0.0, 0.0, -2.0]]), 1.0, 10.0
        )
    with pytest.raises(ValueError):
        green_function(np.array([[1.0, 0.0, 0.5]]), np.array([[0.0, 0.0, -2.0]]), 1.0, 10.0)
    with pytest.raises(ValueError):
        green_function(np.array([[1.0, 0.0, -11.0]]), np.array([[0.0, 0.0, -2.0]]), 1.0, 10.0)
    with pytest.raises(ValueError):
        green_function(np.array([[1.0, 0.0, -1.0]]), np.array([[0.0, 0.0, -2.0]]), 1.0, 0.0)
    with pytest.raises(ValueError):
        modified_bessel_k01(np.array([0.0]))