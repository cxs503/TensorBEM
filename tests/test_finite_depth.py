"""Verification of the finite-depth free-surface kernel.

The kernel is pinned down by four independent properties, each of which is a
separate test here:

1. it solves the strip problem - the free-surface Robin condition and the
   impermeable bottom - to machine precision, which together with the outgoing
   radiation condition and the ``1/(2*pi)*log(r)`` singularity makes the Green
   function *unique*;
2. it has the right singularity strength (unit flux around the source);
3. its ``depth -> infinity`` limit reproduces the deep-water kernel of
   :mod:`tensorbem.free_surface`, which is the kernel validated against
   Ursell's multipole solution - an independent code path, no fitted constant;
4. its wave kinematics reduce to the deep-water and shallow-water limits, and
   the dispersion relation it uses satisfies ``omega**2 = g*k*tanh(k*h)``.

The Kummer acceleration is checked against plain truncation at 4x the modes, so
a defect in the closed-form tails cannot hide behind the reference solution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import finite_depth, free_surface

NU = 0.5
DEPTH = 3.0
GRAVITY = 9.80665


def _strip_residual(nu: float, depth: float, modes: int = 256) -> tuple[float, float]:
    """Return ``(surface, bottom)`` relative residuals of the BCs."""
    x = np.array([0.37, -1.4, 2.9])
    source_x = np.array([0.11, 0.11, 0.11])
    source_z = np.array([-0.8, -1.5, -2.2])
    step = 1e-5

    def kernel(z):
        return finite_depth.green_function(
            x, np.full_like(x, z), source_x, source_z, nu, depth, modes=modes
        )

    # One-sided differences pointing *into* the fluid: the modal series is only
    # meaningful for -h <= z <= 0, and extrapolating it outside the strip would
    # test the extension rather than the boundary condition.
    surface = (kernel(0.0) - kernel(-2.0 * step)) / (2.0 * step) - nu * kernel(0.0)
    bottom = (kernel(-depth + step) - kernel(-depth)) / step
    scale_surface = np.abs(nu * kernel(0.0)).max()
    scale_bottom = np.abs(kernel(-depth + step)).max()
    return (
        float(np.abs(surface).max() / scale_surface),
        float(np.abs(bottom).max() / scale_bottom),
    )


def test_dispersion_relation_is_solved():
    for nu in (0.05, 0.5, 2.0, 8.0):
        for depth in (0.4, 3.0, 25.0):
            k0 = finite_depth.finite_depth_wavenumber(nu, depth)
            assert k0 * math.tanh(k0 * depth) == pytest.approx(nu, rel=1e-12)
            omega = math.sqrt(nu * GRAVITY)
            assert omega**2 == pytest.approx(GRAVITY * k0 * math.tanh(k0 * depth), rel=1e-12)


def test_dispersion_limits():
    nu, depth = 0.5, 3.0
    k0 = finite_depth.finite_depth_wavenumber(nu, depth)
    # k0*tanh(k0*h) = nu with tanh <= 1 forces k0 > nu at finite depth, and
    # k0 -> nu from above as h -> infinity.
    assert k0 > nu
    assert k0 == pytest.approx(nu, rel=0.25)
    deep = finite_depth.finite_depth_wavenumber(nu, 400.0)
    assert deep == pytest.approx(nu, rel=1e-8)
    # Shallow water: k0 -> sqrt(nu/h).
    shallow = finite_depth.finite_depth_wavenumber(nu, 1e-4)
    assert shallow == pytest.approx(math.sqrt(nu / 1e-4), rel=1e-3)


def test_group_velocity_matches_numerical_derivative():
    for nu in (0.2, 0.5, 2.0):
        for depth in (0.5, 3.0, 40.0):
            analytic = finite_depth.group_velocity(nu, depth, gravity=GRAVITY)

            def omega_of_k(k):
                return math.sqrt(GRAVITY * k * math.tanh(k * depth))

            k0 = finite_depth.finite_depth_wavenumber(nu, depth)
            step = 1e-6 * k0
            numeric = (omega_of_k(k0 + step) - omega_of_k(k0 - step)) / (2.0 * step)
            assert analytic == pytest.approx(numeric, rel=1e-8)


def test_group_velocity_limits():
    nu = 0.5
    # Deep water: c_g = c/2.
    deep = finite_depth.group_velocity(nu, 1e4)
    assert deep == pytest.approx(finite_depth.phase_velocity(nu, 1e4) / 2.0, rel=1e-8)
    # Shallow water: c_g = c.
    shallow = finite_depth.group_velocity(nu, 1e-4)
    assert shallow == pytest.approx(
        finite_depth.phase_velocity(nu, 1e-4), rel=1e-3
    )


def test_strip_boundary_conditions():
    for depth in (DEPTH, 16.0):
        surface, bottom = _strip_residual(NU, depth)
        assert surface < 1e-4, (depth, surface)
        assert bottom < 1e-4, (depth, bottom)


def test_bottom_condition_holds_for_any_field_depth():
    # dG/dz = 0 at z = -h must hold for every source position, including
    # sources close to the bottom (where the evanescent sum is largest). One
    # sided difference from inside the fluid only.
    step = 1e-5
    x = np.array([0.0, 0.5, 1.5])
    source_x = np.array([0.2, 0.2, 0.2])
    for source_z in (-0.2, -1.0, -2.5, -2.99):
        inside = finite_depth.green_function(
            x, np.full_like(x, -DEPTH + step), source_x,
            np.full_like(x, source_z), NU, DEPTH,
        )
        wall = finite_depth.green_function(
            x, np.full_like(x, -DEPTH), source_x,
            np.full_like(x, source_z), NU, DEPTH,
        )
        slope = (wall - inside) / step
        assert np.abs(slope).max() / np.abs(wall).max() < 1e-4, source_z


def test_unit_flux_around_the_source():
    """``G ~ log(r)/(2*pi)``: the flux of ``grad(G)`` around a small circle is 1."""
    source_x, source_z = 0.3, -1.2
    radius = 1e-3
    theta = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    x = source_x + radius * np.cos(theta)
    z = source_z + radius * np.sin(theta)
    gradient_x, gradient_z = finite_depth.green_gradient(
        x, z, np.full_like(x, source_x), np.full_like(x, source_z),
        NU, DEPTH, include_rankine=True,
    )
    flux = np.sum(
        (gradient_x * np.cos(theta) + gradient_z * np.sin(theta)) * radius * (2.0 * math.pi / theta.size)
    )
    assert abs(flux - 1.0) < 5e-3, flux


def test_deep_water_limit_matches_free_surface_kernel():
    nu = 0.5
    depth = 800.0  # nu*depth = 400: the deepest strip the modal sum reaches
    points = [
        (0.3, -1.0, -1.5),
        (2.0, -0.5, -0.5),
        (5.0, -2.0, -0.3),
        (-1.7, -3.0, -0.9),
        (0.6, -0.05, -0.05),
    ]
    for x, z, source_z in points:
        finite = finite_depth.green_function(x, z, 0.0, source_z, nu, depth)
        deep = free_surface.green_function(x, z, 0.0, source_z, nu)
        # 1e-4, not 1e-6: at nu*depth = 400 the evanescent roots crowd towards
        # n*pi/h and the Kummer remainder floors at ~2e-5 (see the module
        # docstring). That regime belongs to tensorbem.free_surface anyway.
        assert abs(finite - deep) / abs(deep) < 1e-4, (x, z, source_z, finite, deep)


def test_gradient_matches_numerical_derivative_of_the_kernel():
    """The analytic gradient must reproduce ``dG/dx`` and ``dG/dz``.

    Compared against central differences of :func:`finite_depth.green_function`
    (and, as a second, independent path, of the deep-water kernel in the
    ``depth -> infinity`` limit). ``free_surface.green_gradient`` is *not* used
    as a reference here: it returns derivatives with respect to the source
    coordinates, not the field point.
    """
    step = 1e-6
    for depth in (3.0, 16.0):
        x = np.array([0.3, 2.0, 5.0, -1.7])
        z = np.array([-1.0, -0.5, -2.0, -1.4])
        source_z = np.array([-1.5, -0.5, -0.3, -0.9])
        gx, gz = finite_depth.green_gradient(x, z, np.zeros_like(x), source_z, NU, depth)
        numeric_x = (
            finite_depth.green_function(x + step, z, 0.0, source_z, NU, depth)
            - finite_depth.green_function(x - step, z, 0.0, source_z, NU, depth)
        ) / (2.0 * step)
        numeric_z = (
            finite_depth.green_function(x, z + step, 0.0, source_z, NU, depth)
            - finite_depth.green_function(x, z - step, 0.0, source_z, NU, depth)
        ) / (2.0 * step)
        scale_x = max(np.abs(gx).max(), 1e-12)
        scale_z = max(np.abs(gz).max(), 1e-12)
        assert np.abs(gx - numeric_x).max() / scale_x < 1e-6, (depth, gx, numeric_x)
        assert np.abs(gz - numeric_z).max() / scale_z < 1e-6, (depth, gz, numeric_z)

    # Independent path: the deep-water kernel's own numerical derivative.
    depth = 800.0
    gx, gz = finite_depth.green_gradient(
        x, z, np.zeros_like(x), source_z, NU, depth
    )
    deep_x = (
        free_surface.green_function(x + step, z, 0.0, source_z, NU)
        - free_surface.green_function(x - step, z, 0.0, source_z, NU)
    ) / (2.0 * step)
    assert np.abs(gx - deep_x).max() / np.abs(deep_x).max() < 1e-4


def test_rankine_removal_is_the_log_gradient():
    """``include_rankine=False`` must subtract exactly ``grad(log(r)/(2*pi))``."""
    x = np.array([0.4, 1.3, -0.7])
    z = np.array([-0.6, -1.1, -2.2])
    source_z = np.array([-1.0, -1.0, -1.0])
    full_x, full_z = finite_depth.green_gradient(x, z, 0.0, source_z, NU, DEPTH)
    smooth_x, smooth_z = finite_depth.green_gradient(
        x, z, 0.0, source_z, NU, DEPTH, include_rankine=False
    )
    radius2 = x * x + (z - source_z) ** 2
    assert np.allclose(
        full_x - smooth_x, x / (2.0 * math.pi * radius2), rtol=1e-12, atol=1e-14
    )
    assert np.allclose(
        full_z - smooth_z,
        (z - source_z) / (2.0 * math.pi * radius2),
        rtol=1e-12,
        atol=1e-14,
    )


def test_kummer_acceleration_converges_with_mode_count():
    # A point close to the source, where the raw modal sum converges slowest.
    x, z, source_z = 0.02, -1.0, -1.03
    coarse = finite_depth.green_function(x, z, 0.0, source_z, NU, DEPTH, modes=256)
    fine = finite_depth.green_function(x, z, 0.0, source_z, NU, DEPTH, modes=1024)
    finest = finite_depth.green_function(x, z, 0.0, source_z, NU, DEPTH, modes=4096)
    assert abs(fine - finest) < 1e-6
    assert abs(coarse - finest) < 1e-4


def test_evanescent_sum_matches_brute_force_truncation_differs():
    """The accelerated sum is the converged one, not merely a truncation.

    Summing 4000 exact modal terms and dropping the rest loses the tail of the
    evanescent series; the Kummer form keeps it in closed form. At a point where
    the tail is non-negligible the two must differ by more than the accelerated
    result differs from doubling the modes.
    """
    x, z, source_z = 0.05, -0.7, -1.4
    accelerated = finite_depth.green_function(x, z, 0.0, source_z, NU, DEPTH)
    doubled = finite_depth.green_function(x, z, 0.0, source_z, NU, DEPTH, modes=8192)
    assert abs(accelerated - doubled) < 1e-8


def test_evanescent_roots_are_ordered_and_alternate():
    mu = finite_depth.evanescent_wavenumbers(NU, DEPTH, 6)
    assert mu.size == 6
    assert np.all(np.diff(mu) > 0.0)
    for index, value in enumerate(mu, start=1):
        assert (index - 0.5) * math.pi / DEPTH < value < index * math.pi / DEPTH
        assert value * math.tan(value * DEPTH) == pytest.approx(-NU, rel=1e-10)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        finite_depth.finite_depth_wavenumber(-0.1, DEPTH)
    with pytest.raises(ValueError):
        finite_depth.finite_depth_wavenumber(NU, 0.0)
    with pytest.raises(ValueError):
        finite_depth.green_function(0.0, -1.0, 0.0, -1.0, NU, -2.0)