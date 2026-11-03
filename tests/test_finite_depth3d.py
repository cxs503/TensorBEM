"""Verification of the 3-D finite-depth free-surface kernel.

The kernel (:mod:`tensorbem.finite_depth3d`) is the modal expansion of the
strip; every test below pins one of its defining properties against an
*independent* reference: SciPy's special functions, the transcendental
dispersion relations, the closed-form source singularity, the deep-water
kernel of :mod:`tensorbem.free_surface3d` (a different code path, built from
the Rankine + image + E1 wave decomposition), and exact symmetries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.finite_depth import (
    evanescent_wavenumbers,
    finite_depth_wavenumber,
    propagating_mode_norm,
)
from tensorbem.finite_depth3d import (
    bessel_k0,
    free_surface_residual,
    green_function_3d_finite_depth,
    hankel_h0,
    wave_part_3d_finite_depth,
)
from tensorbem.free_surface3d import green_function_3d

NU = 1.0
FIELD_POINTS = np.array(
    [
        [0.5, 0.0, 0.0],
        [1.0, 0.7, -0.4],
        [2.5, -1.0, -1.0],
        [-3.0, 2.0, -0.9],
        [6.0, 0.5, -2.0],
    ]
)


def _source(depth: float) -> np.ndarray:
    return np.array([0.0, 0.0, -0.35 * depth])


# ---------------------------------------------------------------------------
# special functions
# ---------------------------------------------------------------------------
def test_bessel_k0_matches_scipy():
    scipy = pytest.importorskip("scipy.special")
    x = np.linspace(1e-3, 25.0, 400)
    got = bessel_k0(x)
    ref = scipy.k0(x)
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 3.0e-7
    # where it matters most (small argument, large values) the polynomial is tighter
    small = x < 2.0
    assert np.max(np.abs(got[small] - ref[small])) / np.max(np.abs(ref[small])) < 1e-8


def test_bessel_j0_y0_matches_scipy():
    scipy = pytest.importorskip("scipy.special")
    x = np.linspace(1e-3, 200.0, 600)
    h = hankel_h0(x)
    assert np.max(np.abs(h.real - scipy.j0(x))) / np.max(np.abs(scipy.j0(x))) < 2e-9
    assert np.max(np.abs(h.imag - scipy.y0(x))) / np.max(np.abs(scipy.y0(x))) < 2e-9


def test_hankel_incoming_is_conjugate():
    x = np.linspace(0.2, 20.0, 50)
    out = hankel_h0(x)
    inc = hankel_h0(x, outgoing=False)
    assert np.max(np.abs(inc - np.conj(out))) == 0.0


# ---------------------------------------------------------------------------
# dispersion relations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [0.25, 1.0, 7.5, 40.0])
def test_dispersion_relations(depth):
    k0 = finite_depth_wavenumber(NU, depth)
    assert abs(k0 * math.tanh(k0 * depth) - NU) < 1e-10 * max(1.0, NU)
    mu = evanescent_wavenumbers(NU, depth, 6)
    assert np.all(np.diff(mu) > 0.0)
    residual = mu * np.tan(mu * depth) + NU
    scale = np.abs(mu * np.tan(mu * depth)) + NU
    assert np.max(np.abs(residual) / scale) < 1e-9
    # evanescent roots are bracketed between consecutive poles of tan(mu*h)
    n = np.arange(1, 7)
    assert np.all(mu > (n - 0.5) * np.pi / depth) and np.all(mu < n * np.pi / depth)


def test_propagating_norm_against_scaled_form():
    from tensorbem.finite_depth3d import _scaled_propagating_factor

    for depth in (0.5, 3.0, 25.0):
        k0 = finite_depth_wavenumber(NU, depth)
        _, w = _scaled_propagating_factor(k0, depth)
        recovered = w * math.exp(2.0 * k0 * depth)
        assert abs(recovered - propagating_mode_norm(k0, depth)) < 1e-12 * recovered


# ---------------------------------------------------------------------------
# defining boundary conditions of the kernel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [1.0, 5.0, 20.0])
def test_free_surface_condition_second_order(depth):
    """``dG/dz = nu*G`` at ``z = 0``, verified with a second-order stencil."""
    xy = np.array([[0.5, 0.0], [1.3, -0.6], [4.0, 1.0]])
    coarse = np.max(free_surface_residual(xy, -0.5 * depth, NU, depth, step=1e-3 * depth))
    fine = np.max(free_surface_residual(xy, -0.5 * depth, NU, depth, step=1e-4 * depth))
    assert fine < 1e-7
    assert fine < 0.05 * coarse  # O(step**2) convergence of the stencil


@pytest.mark.parametrize("depth", [2.0, 9.0])
def test_bottom_is_impermeable(depth):
    source = np.array([0.4, 0.1, -0.5 * depth])
    target_xy = np.array([0.9, 0.3])
    step = 1e-6 * depth
    g_bottom = green_function_3d_finite_depth(
        np.array([[target_xy[0], target_xy[1], -depth]]), source, NU, depth
    )[0]
    g_above = green_function_3d_finite_depth(
        np.array([[target_xy[0], target_xy[1], -depth + step]]), source, NU, depth
    )[0]
    derivative = abs(g_bottom - g_above) / step
    assert derivative * depth < 1e-4 * abs(g_bottom)


def test_reciprocity_is_exact():
    a = np.array([[0.7, 0.2, -0.3], [3.0, -1.0, -4.0]])
    b = np.array([[0.0, 0.0, -2.0], [-0.5, 1.5, -0.1]])
    forward = green_function_3d_finite_depth(a, b, NU, 5.0)
    backward = green_function_3d_finite_depth(b, a, NU, 5.0)
    assert np.max(np.abs(forward - backward)) == 0.0
    assert np.max(np.abs(forward)) > 0.0


def test_incoming_wave_is_conjugate():
    a = np.array([[0.7, 0.2, -0.3], [3.0, -1.0, -4.0]])
    b = np.array([[0.0, 0.0, -2.0], [-0.5, 1.5, -0.1]])
    outgoing = green_function_3d_finite_depth(a, b, NU, 5.0)
    incoming = green_function_3d_finite_depth(a, b, NU, 5.0, outgoing=False)
    assert np.max(np.abs(incoming - np.conj(outgoing))) == 0.0


def test_source_singularity_is_the_free_space_plus_bottom_image():
    """Removing ``1/(4 pi r) + 1/(4 pi r_bottom)`` leaves a bounded field."""
    depth = 4.0
    source = np.array([0.0, 0.0, -1.5])
    remainders = []
    for radius in (0.4, 0.2, 0.1, 0.05):
        target = np.array([[radius, 0.0, -1.5]])
        g = green_function_3d_finite_depth(target, source, NU, depth)[0]
        r_direct = radius
        r_bottom = math.hypot(radius, -1.5 - 1.5 + 2.0 * depth)
        remainder = g - (1.0 / (4.0 * np.pi)) * (1.0 / r_direct + 1.0 / r_bottom)
        remainders.append(remainder)
    assert all(abs(val) < 1.0 for val in remainders)
    # the remainder converges to a finite limit as the singular terms are removed
    assert abs(remainders[-1] - remainders[-2]) < 5e-3
    assert abs(remainders[-1]) < 0.2


# ---------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nu_h", [50.0, 100.0, 200.0])
def test_deep_water_limit_matches_the_kelvin_kernel(nu_h):
    """``h -> inf`` must reproduce :func:`tensorbem.free_surface3d.green_function_3d`.

    The geometry is held *fixed* and only the depth grows, so this is a pure
    representation test: both kernels describe the same physics in that limit.
    The measured agreement is 5.5e-6, 6.7e-7 and 8.3e-8 at ``nu*h = 50, 100,
    200`` (see ``docs/finite_depth3d.md``); the residual decays like
    ``h**-3``, so the tolerances below are the measured values with headroom.
    """
    depth = nu_h / NU
    source = np.array([0.0, 0.0, -1.5])
    finite = green_function_3d_finite_depth(FIELD_POINTS, source, NU, depth)
    deep = green_function_3d(FIELD_POINTS, source, NU)
    relative = np.max(np.abs(finite - deep)) / np.max(np.abs(deep))
    tolerance = {50.0: 2e-5, 100.0: 3e-6, 200.0: 4e-7}[nu_h]
    assert relative < tolerance


def test_deep_water_limit_error_decreases_with_depth():
    source = np.array([0.0, 0.0, -1.5])
    errors = []
    for nu_h in (50.0, 100.0, 200.0):
        depth = nu_h / NU
        finite = green_function_3d_finite_depth(FIELD_POINTS, source, NU, depth)
        deep = green_function_3d(FIELD_POINTS, source, NU)
        errors.append(np.max(np.abs(finite - deep)) / np.max(np.abs(deep)))
    assert errors[0] > errors[1] > errors[2]
    # observed order is close to h**-3 between successive depths
    ratios = [errors[i] / errors[i + 1] for i in range(2)]
    assert all(4.0 < ratio < 16.0 for ratio in ratios)


@pytest.mark.parametrize("nu_h", [60.0, 150.0])
def test_propagating_amplitude_matches_deep_water_wave_term(nu_h):
    """The outgoing cylindrical term must be the deep-water wave block.

    Isolating it (``include_rankine=False, include_image=False``) from the
    Kelvin kernel and taking the deep-water limit of the modal amplitude pins
    the normalisation of ``phi_0(z)*phi_0(z0)/N_0`` independently of the
    Rankine part.
    """
    depth = nu_h / NU
    source = np.array([0.0, 0.0, -1.5])
    points = FIELD_POINTS.copy()
    finite_wave = wave_part_3d_finite_depth(points, source, NU, depth)
    deep_wave = green_function_3d(points, source, NU, include_rankine=False, include_image=False)
    # Only the imaginary (radiating) part is comparable: the real part of the
    # deep-water wave block also carries the analytic free-surface correction,
    # which the cylindrical outgoing term does not.
    assert np.max(np.abs(finite_wave.imag - deep_wave.imag)) / np.max(
        np.abs(deep_wave.imag)
    ) < 1e-8


def test_far_field_phase_advances_at_the_finite_depth_wavenumber():
    """Far from the source the wave part rotates by ``k0 * dR``."""
    depth = 3.0
    k0 = finite_depth_wavenumber(NU, depth)
    source = np.array([0.0, 0.0, -1.0])
    radii = np.array([40.0, 40.5, 41.0])
    points = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])
    wave = wave_part_3d_finite_depth(points, source, NU, depth)
    phase_steps = np.angle(wave[1:] / wave[:-1])
    expected = k0 * np.diff(radii)
    assert np.max(np.abs(phase_steps - expected)) < 0.5 * np.max(expected)


# ---------------------------------------------------------------------------
# truncation control and input validation
# ---------------------------------------------------------------------------
def test_mode_count_insensitivity():
    depth = 2.0
    source = np.array([0.0, 0.0, -0.8])
    points = np.array([[0.4, 0.0, 0.0], [1.5, 0.5, -0.6]])
    automatic = green_function_3d_finite_depth(points, source, NU, depth)
    explicit = green_function_3d_finite_depth(points, source, NU, depth, modes=2000)
    assert np.max(np.abs(automatic - explicit)) < 1e-10 * np.max(np.abs(automatic))


def test_shallow_and_deep_depth_limits_of_k0():
    shallow = finite_depth_wavenumber(NU, 1e-6)
    assert abs(shallow - math.sqrt(NU / 1e-6)) < 1e-4 * shallow
    deep = finite_depth_wavenumber(NU, 500.0)
    assert abs(deep - NU) < 1e-10


def test_invalid_inputs_are_rejected():
    source = np.array([0.0, 0.0, -1.0])
    with pytest.raises(ValueError):
        green_function_3d_finite_depth([[1.0, 0.0, 0.0]], source, NU, -1.0)
    with pytest.raises(ValueError):
        green_function_3d_finite_depth([[1.0, 0.0, 0.0]], source, 0.0, 5.0)
    with pytest.raises(ValueError):
        green_function_3d_finite_depth([[0.0, 0.0, -0.5]], source, NU, 5.0)  # R = 0
    with pytest.raises(ValueError):
        green_function_3d_finite_depth([[1.0, 0.0, 0.5]], source, NU, 5.0)  # above z = 0
    with pytest.raises(ValueError):
        green_function_3d_finite_depth([[1.0, 0.0, -6.0]], source, NU, 5.0)  # below the bed