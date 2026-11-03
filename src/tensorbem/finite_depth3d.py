"""Three-dimensional free-surface Green function in water of finite constant depth.

Why a separate kernel
---------------------
:mod:`tensorbem.free_surface3d` builds the *deep-water* kernel from the
closed-form Rankine + free-surface-image + wave decomposition, and the wave
term is a one-dimensional principal-value integral with a pole at ``nu``.
With a flat bottom at ``z = -h`` the vertical structure ``exp(k*z)`` no longer
decays, the vertical spectrum becomes discrete, and the single pole at ``nu``
is replaced by the transcendental finite-depth dispersion relation. The
correct representation is therefore the modal expansion of the strip, exactly
as in the two-dimensional case (:mod:`tensorbem.finite_depth`), with the
horizontal operator now being the *three*-dimensional one.

Eigenvalue problem (shared with :mod:`tensorbem.finite_depth`)
-------------------------------------------------------------
With ``z`` up, mean free surface at ``z = 0``, bottom at ``z = -h``, time
factor ``exp(-i*omega*t)``, the linearised free-surface condition is
``d(phi)/dz - nu*phi = 0`` at ``z = 0`` with ``nu = omega**2/g``, and the
bottom is impermeable. The vertical eigenfunctions are

    propagating:   phi_0 = cosh(k0*(z + h)),   k0*tanh(k0*h) = nu,
    evanescent:    phi_n = cos(mu_n*(z + h)),  mu_n*tan(mu_n*h) = -nu,
                   mu_n in ((n - 1/2)*pi/h, n*pi/h),  n = 1, 2, ...

with norms ``N_n = integral_{-h}^{0} phi_n**2 dz``.

Modal form of the Green function
--------------------------------
Expanding the delta function ``delta(z - z0)`` in this complete orthogonal set
and solving, for each mode, the horizontal problem for the source condition
``-Delta_3D G = delta(x - x0)`` gives, with ``R`` the horizontal distance and
``delta_2`` the two-dimensional delta,

    -(f_n'' + f_n'/R + lambda_n f_n) = delta_2(R),

so the propagating mode carries an outgoing cylindrical wave and each
evanescent mode a modified Bessel function::

    G(x, x0) = sum_{n>=1} phi_n(z)*phi_n(z0)/N_n * K0(mu_n*R) / (2*pi)
             + phi_0(z)*phi_0(z0)/N_0 * (i/4) * H0^{(1)}(k0*R)

where ``(mu_n**2 - Delta_2) K0(mu_n R) = 2*pi*delta_2`` and
``(Delta_2 + k0**2) (i/4) H0^{(1)}(k0 R) = -delta_2``. The overall sign is
*not* assumed here: it is pinned by requiring that the ``k0*h -> inf`` limit
reproduce :func:`tensorbem.free_surface3d.green_function_3d` at every field
point -- Rankine part, image part and wave part together. That limit is a
test (``tests/test_finite_depth3d.py``), not an assumption, mirroring the
two-dimensional module.

Convergence
-----------
Unlike the two-dimensional case, where the evanescent sum decays only like
``1/n``, here every term carries ``K0(mu_n*R) ~ exp(-mu_n*R)``, so the sum
converges *exponentially* in ``n``: the modes are summed until
``mu_n*R_min >= 36`` (``K0 < 1e-16``) or until the largest term is below
``tol`` times the running total. No Kummer acceleration is needed. The price
is that the number of modes grows like ``h/R``: a target at ``R = 0.01*h``
needs O(10^3) modes, and a target on the source axis itself (``R = 0``) is
rejected -- the sum there is only conditionally convergent and needs the
closed-form log-sums that the two-dimensional module implements.

Bessel functions
----------------
The package depends on ``numpy`` only, so ``K0``, ``J0`` and ``Y0`` are
implemented here from the Abramowitz & Stegun rational approximations
(9.8.5-9.8.8 and 9.4.1-9.4.6, |error| ~ 1e-8), and are checked against
``scipy.special`` in the tests when SciPy is available. That accuracy floor is
three orders of magnitude below any bench supply-chain tolerance and is
reported honestly in the verification tables.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .finite_depth import (
    evanescent_wavenumbers,
    finite_depth_wavenumber,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_TWO_PI = 2.0 * np.pi
_FOUR_PI = 4.0 * np.pi
_EULER_GAMMA = 0.5772156649015329

#: Cut-off in the exponent of ``K0``: ``mu_n*R >= _DECAY_CUTOFF`` is dropped.
_DECAY_CUTOFF = 36.0
#: Never sum fewer than this many evanescent modes.
_MIN_MODES = 8
#: Hard cap, so a pathological input fails loudly instead of hanging.
_MAX_MODES = 400_000


# ---------------------------------------------------------------------------
# Bessel functions (numpy only)
# ---------------------------------------------------------------------------
def _i0_small(x: FloatArray) -> FloatArray:
    """Modified Bessel ``I0(x)`` by its power series (accurate for |x| <= 2)."""
    half = 0.5 * x
    half2 = half * half
    term = np.ones_like(half2)
    total = np.ones_like(half2)
    for m in range(1, 14):
        term = term * half2 / (m * m)
        total = total + term
    return total


def bessel_k0(x: FloatArray) -> FloatArray:
    """Modified Bessel function ``K0(x)`` for real ``x > 0`` (|error| ~ 1e-8).

    A&S 9.8.5 for ``0 < x <= 2`` and A&S 9.7.2 for ``x > 2``.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("bessel_k0 is defined for x > 0")
    out = np.empty_like(x)

    small = x <= 2.0
    if np.any(small):
        xs = x[small]
        y2 = (xs * 0.5) ** 2
        poly = (
            0.42278420
            + y2
            * (
                0.23069756
                + y2 * (0.03488590 + y2 * (0.00262698 + y2 * (0.00010750 + y2 * 0.00000740)))
            )
        )
        out[small] = -np.log(xs * 0.5) * _i0_small(xs) - _EULER_GAMMA + y2 * poly

    large = ~small
    if np.any(large):
        xl = x[large]
        z = 2.0 / xl
        poly = (
            1.25331414
            + z
            * (
                -0.07832358
                + z
                * (
                    0.02189568
                    + z * (-0.01062446 + z * (0.00587872 + z * (-0.00251540 + z * 0.00053208)))
                )
            )
        )
        out[large] = np.exp(-xl) / np.sqrt(xl) * poly
    return out


def _asymptotic_pq(x: FloatArray, terms: int = 5) -> tuple[FloatArray, FloatArray]:
    """Asymptotic factors ``(P, Q)`` with ``J0 = sqrt(2/(pi x)) (cos(chi) P - sin(chi) Q)``.

    ``P = sum_k (-1)^k a_k x^{-2k}`` and ``Q = sum_k (-1)^k b_k x^{-2k-1}`` with
    the factorial-ratio coefficients

        a_k = prod_{j=1}^{2k} (2j-1)^2 / ((2k)! 8^{2k}),
        b_k = prod_{j=1}^{2k+1} (2j-1)^2 / ((2k+1)! 8^{2k+1}),

    generated recursively here rather than transcribed as magic decimals. Five
    terms leave a truncation error of ``~1e-10`` at ``x = 12`` and ``O(x^-8)``
    beyond.
    """
    u = 1.0 / x
    p = np.ones_like(u)
    q = -0.125 * u
    prod_even = 1.0
    for k in range(1, terms):
        prod_even *= (4 * k - 3) ** 2 * (4 * k - 1) ** 2
        coef = prod_even / (math.factorial(2 * k) * 8.0 ** (2 * k))
        p = p + (-1) ** k * coef * u ** (2 * k)
    for k in range(1, terms):
        prod_odd = 1.0
        for j in range(1, 2 * k + 2):
            prod_odd *= (2 * j - 1) ** 2
        coef = prod_odd / (math.factorial(2 * k + 1) * 8.0 ** (2 * k + 1))
        q = q + (-1) ** (k + 1) * coef * u ** (2 * k + 1)
    return p, q


def _j0_y0(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Bessel ``J0(x)``, ``Y0(x)`` for ``x >= 0``.

    Convergent power series for ``x <= 12`` (the alternating series is
    well-conditioned there) and the standard asymptotic branch beyond, giving
    ``|error| ~ 1e-10`` or better over the whole range.
    """
    x = np.asarray(x, dtype=float)
    j0 = np.empty_like(x)
    y0 = np.empty_like(x)

    small = x <= 12.0
    if np.any(small):
        xs = x[small]
        half = 0.5 * xs
        half2 = half * half
        term = np.ones_like(xs)
        jsum = np.ones_like(xs)
        ysum = np.zeros_like(xs)
        harmonic = np.zeros_like(xs)
        for k in range(1, 60):
            term = -term * half2 / (k * k)
            jsum = jsum + term
            harmonic = harmonic + 1.0 / k
            ysum = ysum - term * harmonic
        j0[small] = jsum
        y0[small] = (2.0 / np.pi) * (np.log(half) + _EULER_GAMMA) * jsum + (2.0 / np.pi) * ysum
        y0[small & (x <= 0.0)] = -np.inf

    large = ~small
    if np.any(large):
        xl = x[large]
        p, q = _asymptotic_pq(xl)
        amp = np.sqrt(2.0 / (np.pi * xl))
        chi = xl - 0.25 * np.pi
        c, s_ = np.cos(chi), np.sin(chi)
        j0[large] = amp * (c * p - s_ * q)
        y0[large] = amp * (s_ * p + c * q)
    return j0, y0


def hankel_h0(x: FloatArray, *, outgoing: bool = True) -> ComplexArray:
    """Hankel function ``H0^{(1)}(x)`` (or its conjugate for incoming waves)."""
    j0, y0 = _j0_y0(x)
    h = j0 + 1j * y0
    return np.conjugate(h) if not outgoing else h


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _validate_nu(nu: float) -> float:
    nu = float(nu)
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError(
            "nu = omega**2/g must be positive and finite, got %r" % (nu,)
        )
    return nu


def _validate_depth(depth: float) -> float:
    depth = float(depth)
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError(
            "finite depth must be positive and finite, got %r; use "
            "tensorbem.free_surface3d.green_function_3d for h -> inf" % (depth,)
        )
    return depth


def _broadcast_pairs(targets, sources):
    tgt = np.asarray(targets, dtype=float)
    src = np.asarray(sources, dtype=float)
    if tgt.shape[-1] != 3 or src.shape[-1] != 3:
        raise ValueError("targets and sources must have shape (..., 3)")
    tgt, src = np.broadcast_arrays(tgt, src)
    lead = tgt.shape[:-1]
    return tgt.reshape(-1, 3), src.reshape(-1, 3), lead


def _scaled_propagating_factor(k0: float, depth: float):
    """``u(z)`` with ``cosh(k0*(z+h)) = exp(k0*h)*u(z)`` and ``w = N_0/exp(2*k0*h)``.

    Keeps ``depth = 800`` evaluable: both ``phi_0`` and ``N_0`` overflow while
    their ratio does not.
    """
    def u(z):
        return 0.5 * (np.exp(k0 * z) + np.exp(-k0 * (z + 2.0 * depth)))

    exponential = math.exp(-4.0 * k0 * depth)
    w = (1.0 - exponential) / (8.0 * k0) + 0.5 * depth * math.exp(-2.0 * k0 * depth)
    return u, w


def _required_modes(nu: float, depth: float, r_min: float, modes: int | None) -> int:
    if modes is not None:
        modes = int(modes)
        if modes < 0:
            raise ValueError("modes must be non-negative")
        return modes
    # mu_n ~ n*pi/h for large n, so mu_n*r_min >= cutoff needs n ~ cutoff*h/(pi*r_min)
    needed = int(math.ceil(_DECAY_CUTOFF * depth / (math.pi * max(r_min, 1e-12)))) + 8
    needed = max(needed, _MIN_MODES)
    if needed > _MAX_MODES:
        raise ValueError(
            "a target at horizontal distance R = %.3g from its source needs about %d "
            "evanescent modes (the modal sum converges like exp(-mu_n*R)); that "
            "exceeds the cap %d. Offset the target horizontally to R >= %.3g*depth "
            "or pass an explicit `modes`." % (r_min, needed, _MAX_MODES, 0.01)
        )
    return needed


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------
def green_function_3d_finite_depth(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: int | None = None,
    tol: float = 0.0,
) -> ComplexArray:
    """Pulsating-source Green function in water of constant depth ``depth``.

    ``G`` solves ``-nabla^2 G = delta(x - x0)`` in the strip
    ``-depth <= z <= 0`` with ``dG/dz = nu*G`` on ``z = 0`` (the linearised
    free-surface condition, ``nu = omega**2/g``), ``dG/dz = 0`` on the bottom,
    and outgoing waves under ``exp(-i*omega*t)``.

    Parameters
    ----------
    targets, sources:
        Broadcastable coordinate arrays with ``shape[-1] == 3`` and
        ``-depth <= z <= 0``. Horizontal separations must satisfy
        ``R > 0`` for every pair (see the module docstring on convergence).
    nu:
        ``omega**2 / g > 0``.
    depth:
        Constant water depth ``h > 0``.
    outgoing:
        Outgoing ``H_0^{(1)}`` waves (default) or incoming conjugate waves.
    modes:
        Number of evanescent modes to sum. ``None`` (default) chooses it from
        the smallest horizontal separation, so that the truncated tail is
        below ``exp(-36) ~ 1e-16`` relative.
    tol:
        Optional extra stopping rule: stop once the largest single term is
        below ``tol`` times the running magnitude of the sum.

    Returns
    -------
    complex ndarray with the broadcast leading shape of the inputs.
    """
    freq = _validate_nu(nu)
    h = _validate_depth(depth)
    tgt, src, lead = _broadcast_pairs(targets, sources)
    if np.any(tgt[:, 2] > 1e-12) or np.any(src[:, 2] > 1e-12):
        raise ValueError("both targets and sources must lie at or below z = 0")
    if np.any(tgt[:, 2] < -h - 1e-12) or np.any(src[:, 2] < -h - 1e-12):
        raise ValueError("both targets and sources must lie at or above z = -depth")

    dx = tgt[:, 0] - src[:, 0]
    dy = tgt[:, 1] - src[:, 1]
    horiz = np.hypot(dx, dy)
    if np.any(horiz <= 0.0):
        raise ValueError(
            "a target shares its horizontal position with a source (R = 0); the "
            "modal sum is only conditionally convergent there. Offset the target "
            "horizontally, or use tensorbem.finite_depth.green_function for the "
            "2-D log-singular kernel."
        )

    zt = tgt[:, 2]
    zs = src[:, 2]

    # propagating mode ------------------------------------------------------
    k0 = finite_depth_wavenumber(freq, h)
    u, w = _scaled_propagating_factor(k0, h)
    prop_factor = u(zt) * u(zs) / w  # phi_0(z)phi_0(z0)/N_0
    wave = prop_factor * (0.25j) * hankel_h0(k0 * horiz)
    if not outgoing:
        wave = np.conjugate(wave)
    result = np.asarray(wave, dtype=np.complex128)

    # evanescent modes ------------------------------------------------------
    n_modes = _required_modes(freq, h, float(np.min(horiz)), modes)
    mu = evanescent_wavenumbers(freq, h, n_modes)  # (M,)
    norms = 0.5 * h * (1.0 + np.sin(2.0 * mu * h) / (2.0 * mu * h))

    # sum in chunks of modes to bound memory: (chunk, n_pairs)
    chunk = max(1, int(4.0e6 // max(1, horiz.size)))
    total = np.zeros_like(result)
    scale = 1.0
    for start in range(0, n_modes, chunk):
        m = mu[start : start + chunk]
        nrm = norms[start : start + chunk]
        phi_t = np.cos(np.outer(m, zt + h))  # (chunk, n_pairs)
        phi_s = np.cos(np.outer(m, zs + h))
        kvals = bessel_k0(np.outer(m, horiz))
        terms = phi_t * phi_s * kvals / (nrm[:, None] * _TWO_PI)
        total += np.sum(terms, axis=0)
        scale = max(scale, float(np.max(np.abs(np.sum(terms, axis=0)))) if terms.size else scale)
        if tol > 0.0 and start + chunk < n_modes:
            if float(np.max(np.abs(terms))) <= tol * max(scale, 1e-300):
                break

    result = result + total
    return np.asarray(result, dtype=np.complex128).reshape(lead)


def wave_part_3d_finite_depth(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
) -> ComplexArray:
    """Propagating (cylindrical outgoing) part of the finite-depth kernel only.

    This is the piece that carries energy to infinity, so it is the piece a
    far-field damping matrix would be built from. Returned separately for
    diagnostics and for the far-field amplitude check in the tests.
    """
    freq = _validate_nu(nu)
    h = _validate_depth(depth)
    tgt, src, lead = _broadcast_pairs(targets, sources)
    dx = tgt[:, 0] - src[:, 0]
    dy = tgt[:, 1] - src[:, 1]
    horiz = np.hypot(dx, dy)
    if np.any(horiz <= 0.0):
        raise ValueError("R = 0 is not supported by the modal wave part")
    k0 = finite_depth_wavenumber(freq, h)
    u, w = _scaled_propagating_factor(k0, h)
    prop_factor = u(tgt[:, 2]) * u(src[:, 2]) / w
    wave = prop_factor * (0.25j) * hankel_h0(k0 * horiz)
    if not outgoing:
        wave = np.conjugate(wave)
    return np.asarray(wave, dtype=np.complex128).reshape(lead)


def free_surface_residual(
    targets_xy,
    z_source: float,
    nu: float,
    depth: float,
    *,
    step: float | None = None,
) -> FloatArray:
    """Relative residual of ``dG/dz - nu*G`` at ``z = 0``, per target.

    Evaluates ``G`` at ``z = 0, -step, -2*step`` and uses the second-order
    one-sided stencil ``(3G(0) - 4G(-s) + G(-2s)) / (2s)`` for the derivative.
    A correct kernel drives this to the discretisation floor ``O(step**2)``.
    """
    h = _validate_depth(depth)
    xy = np.atleast_2d(np.asarray(targets_xy, dtype=float))
    if xy.shape[-1] != 2:
        raise ValueError("targets_xy must have shape (..., 2)")
    s = 1e-4 * h if step is None else float(step)
    src = np.array([0.0, 0.0, z_source])
    zs = [0.0, -s, -2.0 * s]
    vals = np.array(
        [
            green_function_3d_finite_depth(
                np.column_stack([xy, np.full(len(xy), zz)]), src, nu, h
            )
            for zz in zs
        ]
    )
    dg = (3.0 * vals[0] - 4.0 * vals[1] + vals[2]) / (2.0 * s)
    resid = np.abs(dg - nu * vals[0])
    ref = np.abs(dg) + np.abs(nu * vals[0])
    return resid / ref


__all__ = [
    "bessel_k0",
    "hankel_h0",
    "green_function_3d_finite_depth",
    "wave_part_3d_finite_depth",
    "free_surface_residual",
]