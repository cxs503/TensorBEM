"""Verification suite for the 3-D finite-depth Green function (John eigenseries).

Gates (each is a *physics* statement, not a self-comparison):

1. dispersion relation ``k0*tanh(k0*h) = nu`` closes to machine precision and
   matches its shallow-water asymptote;
2. evanescent roots bracket and satisfy ``mu*tan(mu*h) = -nu``;
3. the vertical modes are orthogonal with the analytic norms;
4. the kernel satisfies the free-surface condition ``dG/dz - nu*G = 0`` at
   ``z = 0``;
5. the analytic ``dG/dz`` agrees with a central difference of ``G``;
6. the kernel is harmonic at field points;
7. the source singularity is ``1/(4*pi*r)`` (unit-strength convention shared
   with :mod:`tensorbem.free_surface3d`);
8. the deep-water limit ``k0*h >> 1`` reproduces the independent, previously
   verified Kelvin kernel to a tolerance that shrinks with ``k0*h``;
9. the shallow-water limit of the wave term matches the closed form
   ``(i/(4h))*H_0^{(1)}(k0*R)``;
10. the far field is an outgoing cylindrical wave with phase slope ``k0`` and
    the evanescent sum is negligible there;
11. evanescent truncation self-convergence;
12. the pure-NumPy Bessel helpers match SciPy (reference for the quadrature-free
    evaluation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.finite_depth import (
    evanescent_wavenumbers,
    finite_depth_wavenumber,
    mode_norms,
    propagating_mode_norm,
)
from tensorbem.finite_depth3d import (
    bessel_j0_y0,
    bessel_k0,
    depth_modes,
    dispersion_residual,
    green_function_3d,
    green_gradient_z_3d,
    hankel_h0,
    vertical_mode,
    wave_amplitude_coefficient,
)
from tensorbem.free_surface3d import green_function_3d as deep_green_3d

NU = 0.6
DEPTHS = (3.0, 8.0, 40.0)


def _target(coords):
    return np.asarray(coords, dtype=float)


# ---------------------------------------------------------------------------
# 1. dispersion relation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nu", [0.05, 0.6, 3.0])
@pytest.mark.parametrize("depth", [0.5, 3.0, 20.0])
def test_dispersion_relation_closes(nu, depth):
    k0 = finite_depth_wavenumber(nu, depth)
    assert dispersion_residual(k0, nu, depth) < 1e-13
    # the two classical limits bracket the root: k0 -> nu (deep),
    # k0 -> sqrt(nu/h) (shallow), so k0/root sits between the two branches
    assert math.sqrt(nu / depth) * (1.0 - 1e-12) <= k0 <= nu * (1.0 + 1e-12) + math.sqrt(nu / depth)


def test_shallow_water_wavenumber_asymptote():
    """``k0 -> sqrt(nu/h)*(1 + nu*h/6)`` as ``kh -> 0``."""
    nu = 0.6
    ratios = []
    for depth in (0.4, 0.2, 0.1, 0.05):
        k0 = finite_depth_wavenumber(nu, depth)
        asymptotic = math.sqrt(nu / depth) * (1.0 + nu * depth / 6.0)
        ratios.append(abs(k0 / asymptotic - 1.0))
    assert ratios[-1] < 1e-4
    assert all(later < earlier for earlier, later in zip(ratios, ratios[1:]))


# ---------------------------------------------------------------------------
# 2. evanescent roots
# ---------------------------------------------------------------------------
def test_evanescent_roots_bracket_dispersion_relation():
    depth, nu = 5.0, 0.6
    mu = evanescent_wavenumbers(nu, depth, 40)
    residual = np.abs(mu * np.tan(mu * depth) + nu) / nu
    assert np.max(residual) < 1e-10
    lower = np.array([(n - 0.5) * math.pi / depth for n in range(1, 41)])
    upper = np.array([n * math.pi / depth for n in range(1, 41)])
    assert np.all(mu > lower) and np.all(mu < upper)
    assert np.all(np.diff(mu) > 0.0)


# ---------------------------------------------------------------------------
# 3. vertical modes
# ---------------------------------------------------------------------------
def test_vertical_modes_are_orthogonal_with_analytic_norms():
    depth, nu = 4.0, 0.6
    k0, mu, norms, norm0 = depth_modes(nu, depth, modes=6)
    nodes, weights = np.polynomial.legendre.leggauss(80)
    z = 0.5 * depth * (nodes - 1.0)  # map [-1, 1] -> [-h, 0]
    w = 0.5 * depth * weights
    phi0 = np.cosh(k0 * (z + depth))
    assert abs(np.dot(w, phi0**2) / norm0 - 1.0) < 1e-13
    for index in range(mu.size):
        phi = np.cos(mu[index] * (z + depth))
        assert abs(np.dot(w, phi**2) / norms[index] - 1.0) < 1e-12
        assert abs(np.dot(w, phi * phi0)) < 1e-12 * norm0
    # cross orthogonality of the lowest evanescent pair
    phi1 = np.cos(mu[0] * (z + depth))
    phi2 = np.cos(mu[1] * (z + depth))
    assert abs(np.dot(w, phi1 * phi2)) < 1e-12 * math.sqrt(norms[0] * norms[1])


def test_wave_amplitude_coefficient_normalisation():
    depth, nu = 6.0, 0.6
    k0, _, _, norm0 = depth_modes(nu, depth, modes=4)
    coeff = wave_amplitude_coefficient(_target([0.0, -2.0]), nu, depth)
    assert abs(coeff[0] - math.cosh(k0 * depth) / norm0) < 1e-12 * abs(coeff[0])
    assert abs(coeff[1] - math.cosh(k0 * (depth - 2.0)) / norm0) < 1e-12 * abs(coeff[1])


# ---------------------------------------------------------------------------
# 4./5./6. differential gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depth", DEPTHS)
def test_free_surface_condition_is_satisfied(depth):
    targets = np.stack(
        [np.linspace(0.3, 9.0, 12), np.zeros(12), np.zeros(12)], axis=-1
    )
    sources = np.array([[0.0, 0.0, -0.5 * depth], [1.0, 0.0, -0.25 * depth]])
    worst = 0.0
    for source in sources:
        value = green_function_3d(targets, source, NU, depth)
        grad = green_gradient_z_3d(targets, source, NU, depth)
        residual = np.abs(grad - NU * value)
        worst = max(worst, np.max(residual) / np.max(np.abs(NU * value)))
    # the gate is limited by the dispersion residual of the evanescent roots
    # (tan(mu*h) at mu*h ~ 1e2 costs ~3 digits), not by the series itself
    assert worst < 1e-10


def test_bottom_is_impermeable():
    depth = 5.0
    targets = np.stack([np.linspace(0.5, 6.0, 8), np.zeros(8), -depth * np.ones(8)], axis=-1)
    sources = np.array([0.0, 0.0, -2.0])
    grad = green_gradient_z_3d(targets, sources, NU, depth)
    value = green_function_3d(targets, sources, NU, depth)
    assert np.max(np.abs(grad)) < 1e-12 * np.max(np.abs(value)) / depth


def test_analytic_z_gradient_matches_central_difference():
    depth = 6.0
    base = np.array([[0.8, 0.0, -1.5], [2.5, 1.0, -0.5], [4.0, 0.0, -3.0]])
    sources = np.array([0.0, 0.0, -2.0])
    step = 1e-5
    plus = base.copy()
    minus = base.copy()
    plus[:, 2] += step
    minus[:, 2] -= step
    fd = (green_function_3d(plus, sources, NU, depth)
          - green_function_3d(minus, sources, NU, depth)) / (2.0 * step)
    analytic = green_gradient_z_3d(base, sources, NU, depth)
    assert np.max(np.abs(fd - analytic)) / np.max(np.abs(analytic)) < 1e-6


def test_kernel_is_harmonic_at_field_points():
    depth = 5.0
    sources = np.array([0.0, 0.0, -2.0])
    point = np.array([[1.3, 0.7, -1.1]])
    step = 2e-4
    lap = 0.0
    for axis in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[0, axis] += step
        minus[0, axis] -= step
        lap = lap + (
            green_function_3d(plus, sources, NU, depth)
            - 2.0 * green_function_3d(point, sources, NU, depth)
            + green_function_3d(minus, sources, NU, depth)
        ) / step**2
    scale = np.abs(green_function_3d(point, sources, NU, depth)) / step**2
    assert abs(lap[0]) < 1e-4 * scale


# ---------------------------------------------------------------------------
# 7. singularity
# ---------------------------------------------------------------------------
def test_source_singularity_is_unit_strength():
    depth = 5.0
    sources = np.array([0.0, 0.0, -2.0])
    radii = np.array([2e-2, 1e-2, 5e-3])
    targets = np.stack([radii, np.zeros_like(radii), -2.0 * np.ones_like(radii)], axis=-1)
    value = green_function_3d(targets, sources, NU, depth)
    strength = 4.0 * math.pi * radii * np.real(value)
    assert np.all(np.abs(strength - 1.0) < 1e-2)
    assert abs(strength[-1] - 1.0) < abs(strength[1] - 1.0) < abs(strength[0] - 1.0)


# ---------------------------------------------------------------------------
# 8. deep-water limit -- cross-check against the independent Kelvin kernel
# ---------------------------------------------------------------------------
def test_deep_water_limit_recovers_kelvin_kernel():
    targets = np.array(
        [[0.5, 0.0, -1.0], [2.0, 0.0, 0.0], [3.0, 1.0, -2.0], [6.0, 0.0, -0.5]]
    )
    sources = np.array([0.0, 0.0, -2.0])
    errors = []
    for depth in (10.0, 20.0, 40.0):
        finite = green_function_3d(targets, sources, NU, depth)
        deep = deep_green_3d(targets, sources, NU)
        errors.append(np.max(np.abs(finite - deep)) / np.max(np.abs(deep)))
    assert errors[-1] < 1e-3
    assert errors[-1] < errors[1] < errors[0]


# ---------------------------------------------------------------------------
# 9. shallow-water limit of the wave term
# ---------------------------------------------------------------------------
def test_shallow_water_wave_term_matches_closed_form():
    """Long-wave limit: the wave term -> (i/(4h))*H_0^{(1)}(k0*R) as k0*h -> 0."""
    nu = 0.6
    targets = np.array([[1.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    sources = np.array([0.0, 0.0, 0.0])
    deviations = []
    for depth in (0.08, 0.04):
        k0 = finite_depth_wavenumber(nu, depth)
        total = green_function_3d(targets, sources, nu, depth)
        without_wave = green_function_3d(targets, sources, nu, depth, include_wave=False)
        wave = total - without_wave
        long_wave = 0.25j * (1.0 / depth) * hankel_h0(k0 * targets[:, 0])
        deviations.append(
            np.max(np.abs(wave - long_wave)) / np.max(np.abs(long_wave))
        )
    assert deviations[0] < 5e-2
    assert deviations[1] < deviations[0]


# ---------------------------------------------------------------------------
# 10. far field
# ---------------------------------------------------------------------------
def test_far_field_is_outgoing_with_wavenumber_k0():
    nu, depth = 0.6, 4.0
    k0 = finite_depth_wavenumber(nu, depth)
    radii = np.linspace(25.0, 35.0, 11)
    targets = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=-1)
    sources = np.array([0.0, 0.0, -2.0])
    value = green_function_3d(targets, sources, nu, depth)
    evanescent = green_function_3d(targets, sources, nu, depth, include_wave=False)
    assert np.max(np.abs(evanescent)) < 1e-5 * np.max(np.abs(value))
    # the evanescent sum keeps dying exponentially with R
    far = np.array([[45.0, 0.0, 0.0]])
    ratio_far = abs(green_function_3d(far, sources, nu, depth, include_wave=False)[0]) / abs(
        green_function_3d(far, sources, nu, depth)[0]
    )
    assert ratio_far < 1e-9 * 1e0
    # outgoing cylindrical spreading: |G| sqrt(k0 R) is flat
    amplitude = np.abs(value) * np.sqrt(k0 * radii)
    assert np.ptp(amplitude) / np.mean(amplitude) < 2e-3
    # phase slope equals the finite-depth wavenumber
    phase = np.unwrap(np.angle(value))
    slope = np.polyfit(k0 * radii, phase, 1)[0]
    assert abs(slope - 1.0) < 2e-3


# ---------------------------------------------------------------------------
# 11. truncation self-convergence
# ---------------------------------------------------------------------------
def test_evanescent_truncation_self_convergence():
    depth = 4.0
    targets = np.array([[0.7, 0.0, -0.4], [1.5, 1.0, -2.0]])
    sources = np.array([0.0, 0.0, -1.0])
    coarse = green_function_3d(targets, sources, NU, depth, modes=120)
    fine = green_function_3d(targets, sources, NU, depth, modes=400)
    assert np.max(np.abs(fine - coarse)) / np.max(np.abs(fine)) < 1e-9


# ---------------------------------------------------------------------------
# 12. Bessel references
# ---------------------------------------------------------------------------
def test_bessel_helpers_match_scipy():
    special = pytest.importorskip("scipy.special")
    x = np.concatenate([np.logspace(-3, 1.0, 40), np.linspace(10.0, 60.0, 40)])
    k0 = bessel_k0(x)
    assert np.max(np.abs(k0 - special.kv(0, x)) / special.kv(0, x)) < 1e-8
    j0, y0 = bessel_j0_y0(x)
    assert np.max(np.abs(j0 - special.jv(0, x)) / np.maximum(np.abs(special.jv(0, x)), 1e-2)) < 1e-8
    assert np.max(np.abs(y0 - special.yv(0, x)) / np.abs(special.yv(0, x))) < 1e-8
    assert np.max(np.abs(hankel_h0(x) - (special.jv(0, x) + 1j * special.yv(0, x)))) < 1e-8


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        green_function_3d(_target([0.0, 0.0, 0.1]), _target([0.0, 0.0, -1.0]), NU, 5.0)
    with pytest.raises(ValueError):
        green_function_3d(_target([0.0, 0.0, -6.0]), _target([0.0, 0.0, -1.0]), NU, 5.0)
    with pytest.raises(ValueError):
        green_function_3d(_target([1.0, 0.0, -1.0]), _target([0.0, 0.0, -1.0]), -1.0, 5.0)
    with pytest.raises(ValueError):
        green_function_3d(_target([1.0, 0.0, -1.0]), _target([0.0, 0.0, -1.0]), NU, 0.0)

# ---------------------------------------------------------------------------
# 17. on-axis regime (R = 0) -- known limitation, found while merging PR #104
# ---------------------------------------------------------------------------
def test_near_axis_agrees_with_deep_water_kernel():
    """``R >= 1e-3``: the evanescent sum is still resolved (measured 1.0e-4)."""
    nu, depth = 0.6, 20.0
    source = _target([0.0, 0.0, -1.0])
    radii = (1.0e-3, 1.0e-2, 0.05, 0.1, 0.5)
    targets = np.array([[radius, 0.0, -0.2] for radius in radii])
    finite = np.ravel(green_function_3d(targets, source, nu, depth))
    deep = np.ravel(deep_green_3d(targets, source, nu))
    assert np.max(np.abs(finite - deep) / np.abs(deep)) < 1e-3


@pytest.mark.xfail(
    strict=True,
    reason=(
        "R = 0 exactly: the evanescent sum sum_n K_0(mu_n*R) is logarithmically "
        "divergent on the source axis and is truncated, so G_fd gets a spurious "
        "real part (measured at nu*h = 12: -4.434e1 vs 1.136e-1, 2.4e2 relative). "
        "Measured 1.0e-4 at R = 1e-3 and 3.2e-2 at R = 1e-4: the defect is confined "
        "to R <~ 1e-4 and is cured by a closed-form log subtraction, not by more "
        "modes.  Found during the m3878 merge of PR #104; registered so the regime "
        "is covered instead of silently unclaimed."
    ),
)
def test_on_axis_point_matches_deep_water_kernel():
    nu, depth = 0.6, 20.0
    source = _target([0.0, 0.0, -1.0])
    targets = np.array([[0.0, 0.0, -0.2]])
    finite = np.ravel(green_function_3d(targets, source, nu, depth))[0]
    deep = np.ravel(deep_green_3d(targets, source, nu))[0]
    assert abs(finite - deep) / abs(deep) < 1e-6
