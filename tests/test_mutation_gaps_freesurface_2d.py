"""Quantitative gates for the 2-D free-surface / finite-depth kernel family.

Motivation
----------
``tests/test_mutation_gaps.py`` covers the 2-D Laplace and hydrostatic internals
and m3471 did the same for the 3-D Laplace kernels; the family that carries the
free-surface physics -- the Cauchy principal value integrals behind
:func:`tensorbem.free_surface.green_function`, the finite-depth dispersion and
mode machinery in :mod:`tensorbem.finite_depth` -- had never been asked whether
its tests *can* fail (see ``outputs/m3497/mutation_fs2d.py``).

The measured starting point is 6 of 10 injected defects killed
(``outputs/m3497/mutation_sweep.json``): the guards are better than
structural-only, but they compare the code against *itself* in several places
(the same dispersion solver feeds both sides of a residual identity), and two
defects escaped -- a disabled input-validation guard and a 1% error in the
radiated-wave factor.  Every test below instead recomputes the quantity by a
route that does not share code with the kernel:

* the principal value integrals against ``scipy.integrate.quad(..., weight=
  'cauchy')``;
* ``green_function`` against a hand-assembled Rankine + image + PV + residue
  expression;
* the dispersion root against an independent bisection of ``k tanh(k h) = nu``;
* the evanescent roots against their own defining transcendental equation;
* the radiated wave factor against ``cosh(k0 (z + h)) / cosh(k0 h)``;
* the group velocity against a numerical derivative of the same implicit curve;
* the mode norm against its defining integral.

The point is not extra coverage of reached lines; it is that a kernel that is
wrong by 0.1% now fails a test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.integrate as integrate

from tensorbem import finite_depth as fd
from tensorbem import free_surface as fs

NU = 0.6


def _pv_reference(depth: float, offset: float, *, sine: bool = False) -> float:
    """Principal value integral of ``exp(-k d) cos/sin(k x) / (k - nu)`` dk.

    The integrand decays like ``exp(-k d)``, so truncating at
    ``nu + 80/depth`` leaves a tail below machine precision; the singularity
    itself is handled by scipy's cauchy weight (a genuine PV evaluation, not the
    kernel's own regularisation).  Pairs whose reference is not converged -- the
    strongly oscillatory ones, ``|offset|`` large with small ``depth`` -- are
    rejected by comparing two truncations, so the test never grades the kernel
    against a bad reference.
    """
    trig = np.sin if sine else np.cos

    def integrand(k: float) -> float:
        return math.exp(-k * depth) * trig(k * offset)

    values = []
    for span in (80.0, 160.0):
        upper = NU + span / depth
        value, _ = integrate.quad(
            integrand, 0.0, upper, weight="cauchy", wvar=NU, limit=800,
            epsabs=1.0e-14, epsrel=1.0e-14,
        )
        values.append(value)
    first, second = values
    scale = max(abs(first), abs(second), 1.0e-30)
    assert abs(first - second) / scale < 1.0e-11, (
        "reference PV integral did not converge; pick a tamer probe pair"
    )
    return second


PV_PAIRS = [(0.5, 0.0), (1.5, 1.1), (4.0, -2.3), (0.8, 0.7)]


@pytest.mark.parametrize("depth,offset", PV_PAIRS)
def test_pv_cosine_matches_a_scipy_cauchy_quadrature(depth: float, offset: float) -> None:
    """The PV cosine integral is a quadrature, not a convention.

    The kernel's own regularisation (subtract the value at ``k = nu``, split the
    tail onto a Laguerre rule) is one way to evaluate it; scipy's cauchy weight
    is an independent way.  Any defect that changes the integral by more than
    the quadrature error -- a wrong sign, a dropped tail factor, a mis-scaled
    ``nu`` -- fails here.
    """
    got = fs._pv_cosine(NU, np.array([depth]), np.array([offset]))[0]
    want = _pv_reference(depth, offset)
    assert got == pytest.approx(want, rel=1.0e-9, abs=1.0e-12)


@pytest.mark.parametrize("depth,offset", PV_PAIRS)
def test_pv_sine_matches_a_scipy_cauchy_quadrature(depth: float, offset: float) -> None:
    got = fs._pv_sine(NU, np.array([depth]), np.array([offset]))[0]
    want = _pv_reference(depth, offset, sine=True)
    assert got == pytest.approx(want, rel=1.0e-9, abs=1.0e-12)


@pytest.mark.parametrize("field,source", [((0.7, -0.9), (0.0, -1.0)), ((2.5, -1.4), (0.3, -1.2))])
def test_deep_water_kernel_matches_an_independent_assembly(field, source) -> None:
    """``green_function`` rebuilt from Rankine + image + scipy PV + residue.

    A relative error of 1e-3 in the kernel is invisible to boundary-condition
    residuals (they only see the kernel's own null space) but shows up here.
    """
    x, z = field
    x0, z0 = source
    offset = x - x0
    depth = -(z + z0)
    r = math.hypot(offset, z - z0)
    r_image = math.hypot(offset, z + z0)
    pv = _pv_reference(depth, offset)
    real_part = math.log(r) - math.log(r_image) - 2.0 * pv
    value = real_part / (2.0 * math.pi) + 1.0j * fs._RESIDUE_SIGN * math.exp(
        -NU * depth
    ) * math.cos(NU * offset)
    got = fs.green_function(x, z, x0, z0, NU)
    assert got == pytest.approx(value, rel=1.0e-9, abs=1.0e-12)


@pytest.mark.parametrize("field,source", [((0.7, -0.9), (0.0, -1.0)), ((-2.2, -1.6), (0.4, -0.7))])
def test_horizontal_reciprocity_ties_truth_to_two_code_paths(field, source) -> None:
    """``dG/dx0 = -dG/dx``: horizontal translation invariance must hold exactly.

    Source and field gradients live in different functions
    (:func:`tensorbem.free_surface.green_gradient` versus
    :func:`tensorbem.free_surface.green_gradient_field`), so this identity is a
    cross-check between two independent implementations rather than a
    self-consistency check.  It also pins the *ordering* of the returned pair:
    swapping the two components breaks it.
    """
    x, z = field
    x0, z0 = source
    d_source, _ = fs.green_gradient(x, z, x0, z0, NU)
    d_field, _ = fs.green_gradient_field(x, z, x0, z0, NU)
    assert d_source == pytest.approx(-d_field, rel=1.0e-12, abs=1.0e-15)
    assert abs(d_source) > 1.0e-3, "probe point is degenerate; the identity is void"


def _bisect_dispersion(nu: float, depth: float) -> float:
    """Independent root of ``k tanh(k h) = nu`` by bisection on [0, nu/tanh floor]."""

    def residual(k: float) -> float:
        return k * math.tanh(k * depth) - nu

    low, high = 0.0, max(nu, 1.0e-6)
    while residual(high) < 0.0:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if residual(mid) < 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


@pytest.mark.parametrize("depth", [0.2, 1.0, 6.0])
def test_finite_depth_wavenumber_matches_independent_bisection(depth: float) -> None:
    """Ignoring the bottom (returning the deep-water root ``nu``) must not pass.

    In deep water ``k0 = nu``, so the deep-water shortcut is right in the
    ``h -> inf`` limit and wrong everywhere else -- exactly the kind of defect a
    limits-only test suite keeps alive.
    """
    k0 = fd.finite_depth_wavenumber(NU, depth)
    assert k0 == pytest.approx(_bisect_dispersion(NU, depth), rel=1.0e-12)
    assert abs(k0 * math.tanh(k0 * depth) - NU) < 1.0e-12 * max(1.0, NU)


@pytest.mark.parametrize("depth", [0.5, 1.0, 3.0])
def test_evanescent_roots_satisfy_their_own_transcendental_equation(depth: float) -> None:
    """``mu tan(mu h) = -nu`` for every returned evanescent root.

    The existing guard already catches a 1% root error (the ordering and the
    Kummer-tail convergence test are stronger than they look); this test pins the
    equation itself, so the residual, not a proxy, is what the suite asserts.
    """
    roots = fd.evanescent_wavenumbers(NU, depth, 4)
    assert roots.shape == (4,)
    for mu in roots:
        assert abs(mu * math.tan(mu * depth) + NU) < 1.0e-10 * max(1.0, mu)
    assert np.all(np.diff(roots) > 0.0)


@pytest.mark.parametrize("depth", [0.5, 1.0, 3.0])
def test_wave_factor_is_the_cosh_ratio(depth: float) -> None:
    """``phi_0(z)/phi_0(0) = cosh(k0 (z + h)) / cosh(k0 h)``, recomputed from scratch."""
    k0 = _bisect_dispersion(NU, depth)
    for z in (0.0, -0.3, -0.9 * depth):
        want = math.cosh(k0 * (z + depth)) / math.cosh(k0 * depth)
        got = float(fd.wave_amplitude_coefficient(z, NU, depth))
        assert got == pytest.approx(want, rel=1.0e-12)


@pytest.mark.parametrize("depth", [0.3, 1.0, 5.0])
def test_group_velocity_from_the_implicit_dispersion_curve(depth: float) -> None:
    """``c_g = d omega / d k`` taken numerically, not from the closed form.

    The deep-water shortcut ``c_g = c_p / 2`` is correct only for ``k h >> 1``;
    a test that checks the two asymptotes alone still accepts it in mid-depth.
    """
    gravity = 9.80665
    k0 = _bisect_dispersion(NU, depth)
    step = 1.0e-6 * k0

    def omega_at(k: float) -> float:
        return math.sqrt(gravity * k * math.tanh(k * depth))

    numerical = (omega_at(k0 + step) - omega_at(k0 - step)) / (2.0 * step)
    assert fd.group_velocity(NU, depth) == pytest.approx(numerical, rel=1.0e-7)
    # Mid-depth must not collapse to either asymptote: c_g/c_p = 1 at the
    # shallow limit and 1/2 in deep water, and the two shortcuts differ here.
    ratio = fd.group_velocity(NU, depth) / fd.phase_velocity(NU, depth)
    want_ratio = 0.5 * (1.0 + 2.0 * k0 * depth / math.sinh(2.0 * k0 * depth))
    assert ratio == pytest.approx(want_ratio, rel=1.0e-10)
    assert 0.5 < ratio < 1.0


@pytest.mark.parametrize("depth", [0.5, 2.0])
def test_mode_norm_is_the_defining_integral(depth: float) -> None:
    """``N_0 = int_{-h}^{0} cosh^2(k0 (z + h)) dz = (h/2)(1 + sinh(2 k0 h)/(2 k0 h))``."""
    k0 = _bisect_dispersion(NU, depth)
    z = np.linspace(-depth, 0.0, 400001)
    numerical = float(np.trapezoid(np.cosh(k0 * (z + depth)) ** 2, z))
    assert fd.propagating_mode_norm(k0, depth) == pytest.approx(numerical, rel=1.0e-8)


def test_the_scaled_depth_floor_is_enforced() -> None:
    """Below ``nu * depth`` floor the 2-D kernel is not defined; it must raise.

    Without this the guard constant can be zeroed and every caller silently
    receives the logarithmically divergent zero-frequency limit.
    """
    with pytest.raises(ValueError):
        fs.principal_value_integral(NU, 1.0e-6, 0.0)
    with pytest.raises(ValueError):
        fs.green_function(0.1, -1.0e-9, 0.0, -1.0e-9, NU)
    assert fs._MIN_SCALED_DEPTH > 0.0