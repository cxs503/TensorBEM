"""Three-dimensional free-surface Green function in water of finite depth.

Why a separate kernel
---------------------
``tensorbem.free_surface3d`` builds the *deep-water* Kelvin/Havelock kernel: the
vertical structure ``exp(k*z)`` decays at ``z -> -inf``, so the free-surface
condition reduces to a pole ``k - nu`` on a *continuous* wavenumber axis.  With a
flat bottom at ``z = -h`` the vertical spectrum becomes *discrete* and the
propagating wavenumber is fixed by the transcendental dispersion relation
``k0*tanh(k0*h) = nu``.  Exactly as in two dimensions
(:mod:`tensorbem.finite_depth`), the correct representation is therefore a modal
expansion in the vertical eigenfunctions of the strip, not a Fourier integral.

Eigenvalue problem
------------------
With ``z`` up, mean free surface ``z = 0``, bottom at ``z = -h`` and the time
factor ``exp(-i*omega*t)``, the linearised free-surface condition is
``d(phi)/dz - nu*phi = 0`` at ``z = 0`` (``nu = omega**2 / g``) and the bottom is
impermeable, ``d(phi)/dz = 0`` at ``z = -h``.  The vertical eigenproblem

    phi'' = lambda*phi,   phi'(0) - nu*phi(0) = 0,   phi'(-h) = 0

has one positive eigenvalue (one *propagating* mode) and countably many negative
ones (evanescent modes)::

    propagating:   phi_0 = cosh(k0*(z + h)),   k0*tanh(k0*h) = nu,
    evanescent:    phi_n = cos(mu_n*(z + h)),  mu_n*tan(mu_n*h) = -nu,
                   mu_n in ((n - 1/2)*pi/h, n*pi/h),   n = 1, 2, ...

with norms ``N_n = integral_{-h}^{0} phi_n**2 dz``, i.e.
``N_0 = (h/2)*(1 + sinh(2*k0*h)/(2*k0*h))`` and
``N_n = (h/2)*(1 + sin(2*mu_n*h)/(2*mu_n*h))``.

The three-dimensional kernel
----------------------------
Resolving the one-dimensional operator in ``z`` spectrally and inverting the
horizontal part with the outgoing cylindrical Green function ``-(i/4)*H_0^{(1)}``
gives the John (1950) eigenseries, for horizontal separation ``R``,

    G(x, xi) = (1/(4*pi)) * [ 2 * sum_{n>=1} (phi_n(z)*phi_n(zeta)/N_n) * K_0(mu_n*R)
                              - (i/(4*pi)) * (phi_0(z)*phi_0(zeta)/N_0)
                                * H_0^{(1)}(k0*R) ] .

The ``1/(4*pi)`` prefactor matches the convention of
:func:`tensorbem.free_surface3d.green_function_3d`, so ``G ~ 1/(4*pi*r)`` at the
source, ``G`` is complex, and the wave term is *outgoing* under
``exp(-i*omega*t)``.  ``K_0`` decays exponentially, so the evanescent sum
converges exponentially in ``n`` for any ``R > 0``; the mode count needed grows
like ``h/(pi*R)``, which is why the singular near field is taken from the series
itself instead of an extra analytic subtraction.

Limitations
-----------
* Only the vertical (``z``) derivative is provided analytically.  A horizontal
  derivative needs ``K_1``/``H_1`` and is left to the boundary element layer,
  which also owns panel integration.
* Constant depth, linearised potential flow: no sloping/rough bottom, no
  current, no nonlinear free-surface effects.
* ``modes`` must be large enough for the requested minimum ``R``; the automatic
  choice targets ``exp(-mu_n*R_min) <= 1e-12`` and is capped at
  :data:`MODES_CAP` (the cap binds only for ``h/(pi*R) > ~1e4``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

from .analytic import bessel_j01_y01
from .finite_depth import (
    evanescent_wavenumbers,
    finite_depth_wavenumber,
    mode_norms,
    propagating_mode_norm,
)

_FOUR_PI = 4.0 * math.pi
_TWO_PI = 2.0 * math.pi
_EULER_GAMMA = 0.5772156649015328606065120900824024
_LOG_DECAY = 30.0  # exp(-30) ~ 1e-13 evanescent-mode cut-off
MODES_CAP = 40000
_K0_SERIES_LIMIT = 4.0  # ascending K_0/K_1 series below this argument
_BESSEL_SERIES_LIMIT = 16.0
#: Above this argument the 25-term asymptotic series of ``K_0``/``K_1`` is at
#: machine precision (``6e-14`` at 14, ``2e-15`` at 16).  Between
#: ``_K0_SERIES_LIMIT`` and this value the ascending series has already lost
#: accuracy to cancellation (``1/K_0`` amplification) while the asymptotic
#: series has not converged yet, so the Laplace-integral quadrature below is
#: used instead -- without it ``K_0`` carried a ``2.1e-8`` relative-accuracy
#: cliff at ``x ~ 8.9`` (see :mod:`tensorbem.benchmarks.finite_depth3d_gradient`).
_K_ASYMPTOTIC_LIMIT = 16.0
_QUAD_NODES = 96  # Gauss-Legendre nodes for the K_0/K_1 Laplace integral
_CHUNK = 2000  # modes evaluated per vectorised block
_LEGENDRE_CACHE: Optional[tuple[FloatArray, FloatArray]] = None


# ---------------------------------------------------------------------------
# Bessel helpers (pure NumPy)
# ---------------------------------------------------------------------------
def _bessel_k0_small(x: FloatArray) -> FloatArray:
    """``K_0`` by the ascending series (accurate for ``x <~ 6``)."""
    x = np.asarray(x, dtype=float)
    x2 = 0.25 * x * x
    i0 = np.ones_like(x)
    series = np.zeros_like(x)
    harmonic = np.zeros_like(x)
    term = np.ones_like(x)
    for k in range(1, 90):
        term = term * x2 / (k * k)
        i0 = i0 + term
        harmonic = harmonic + 1.0 / k
        series = series + term * harmonic
        if np.max(np.abs(term)) < 1e-18:
            break
    return -(np.log(0.5 * x) + _EULER_GAMMA) * i0 + series


def _bessel_k0_large(x: FloatArray) -> FloatArray:
    """``K_0`` by the asymptotic expansion (``x >= 6``)."""
    x = np.asarray(x, dtype=float)
    inv = 1.0 / x
    coeff = 1.0
    poly = np.ones_like(x)
    power = np.ones_like(x)
    for k in range(1, 26):
        coeff = -coeff * (2.0 * k - 1.0) ** 2 / (8.0 * k)
        power = power * inv
        poly = poly + coeff * power
    return np.sqrt(math.pi / (2.0 * x)) * np.exp(-x) * poly


def _bessel_k1_small(x: FloatArray) -> FloatArray:
    """``K_1`` by the ascending series (A&S 9.6.11, accurate for ``x <~ 4``)."""
    x = np.asarray(x, dtype=float)
    x2 = 0.25 * x * x
    # I_1(x) = sum_k (x/2)^(2k+1) / (k! (k+1)!)
    i1 = np.zeros_like(x)
    term = 0.5 * x
    k = 0
    while True:
        i1 = i1 + term
        k += 1
        term = term * x2 / (k * (k + 1))
        if np.max(np.abs(term)) < 1e-20:
            break
    # K_1 = 1/x + ln(x/2) I_1 - (x/4) sum_k (H_k + H_{k+1} - 2 gamma)/(k!(k+1)!) (x^2/4)^k
    series = np.zeros_like(x)
    power = np.ones_like(x)
    harmonic = 0.0
    k = 0
    while True:
        next_harmonic = harmonic + 1.0 / (k + 1)
        series = series + power * (harmonic + next_harmonic - 2.0 * _EULER_GAMMA)
        harmonic = next_harmonic
        power = power * x2 / ((k + 1) * (k + 2))
        k += 1
        if np.max(np.abs(power)) < 1e-22:
            break
    return 1.0 / x + np.log(0.5 * x) * i1 - 0.25 * x * series


def _bessel_k1_large(x: FloatArray) -> FloatArray:
    """``K_1`` by the asymptotic expansion (``x >= 16``)."""
    x = np.asarray(x, dtype=float)
    inv = 1.0 / x
    coeff = 1.0
    poly = np.ones_like(x)
    power = np.ones_like(x)
    for k in range(1, 26):
        # a_k = (mu - 1)(mu - 9)...(mu - (2k-1)^2) / (k! 8^k), mu = 4*nu^2 = 4
        coeff = coeff * (4.0 - (2.0 * k - 1.0) ** 2) / (8.0 * k)
        power = power * inv
        poly = poly + coeff * power
    return np.sqrt(math.pi / (2.0 * x)) * np.exp(-x) * poly


def _legendre_nodes() -> tuple[FloatArray, FloatArray]:
    """Cached Gauss-Legendre nodes/weights on ``[-1, 1]``."""
    global _LEGENDRE_CACHE
    if _LEGENDRE_CACHE is None:
        _LEGENDRE_CACHE = leggauss(_QUAD_NODES)
    return _LEGENDRE_CACHE


def _bessel_k0_k1_quad(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``(K_0, K_1)`` from the Laplace integral, cancellation free.

    With ``K_nu(x) = int_0^inf exp(-x cosh t) cosh(nu t) dt`` and the
    substitution ``cosh t = 1 + u**2``,

        K_0(x) = 2 exp(-x) int_0^inf exp(-x u^2) / sqrt(u^2 + 2) du,
        K_1(x) = 2 exp(-x) int_0^inf exp(-x u^2) (1 + u^2) / sqrt(u^2 + 2) du.

    Both integrands are positive and analytic, so Gauss-Legendre on
    ``u in [0, 8/sqrt(x)]`` (where ``exp(-x u^2) <= exp(-64)``) reaches
    ``~5e-15`` relative accuracy -- the regime where the ascending series
    cancels and the asymptotic series diverges.
    """
    x = np.asarray(x, dtype=float)
    nodes, weights = _legendre_nodes()
    upper = 8.0 / np.sqrt(x)
    u = 0.5 * upper[..., None] * (nodes + 1.0)
    jacobian = 0.5 * upper
    expo = np.exp(-x[..., None] * u * u)
    inv = 1.0 / np.sqrt(u * u + 2.0)
    prefactor = 2.0 * np.exp(-x)
    k0 = prefactor * np.sum(weights * expo * inv, axis=-1) * jacobian
    k1 = prefactor * np.sum(weights * expo * (1.0 + u * u) * inv, axis=-1) * jacobian
    return k0, k1


def _select_branches(x: FloatArray):
    """Boolean masks for the series / quadrature / asymptotic branches."""
    small = x < _K0_SERIES_LIMIT
    large = x >= _K_ASYMPTOTIC_LIMIT
    middle = ~small & ~large
    return small, middle, large


def bessel_k0(x: ArrayLike) -> FloatArray:
    """Modified Bessel function ``K_0`` for positive real argument."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("K_0 argument must be strictly positive")
    out = np.empty_like(x)
    small, middle, large = _select_branches(x)
    if np.any(small):
        out[small] = _bessel_k0_small(x[small])
    if np.any(middle):
        out[middle] = _bessel_k0_k1_quad(x[middle])[0]
    if np.any(large):
        out[large] = _bessel_k0_large(x[large])
    return out


def bessel_k1(x: ArrayLike) -> FloatArray:
    """Modified Bessel function ``K_1`` for positive real argument."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("K_1 argument must be strictly positive")
    out = np.empty_like(x)
    small, middle, large = _select_branches(x)
    if np.any(small):
        out[small] = _bessel_k1_small(x[small])
    if np.any(middle):
        out[middle] = _bessel_k0_k1_quad(x[middle])[1]
    if np.any(large):
        out[large] = _bessel_k1_large(x[large])
    return out


def _j0_y0_series(x: FloatArray):
    """``J_0``, ``Y_0`` by ascending series (accurate for ``x <~ 16``)."""
    x = np.asarray(x, dtype=float)
    x2 = 0.25 * x * x
    j0 = np.ones_like(x)
    series = np.zeros_like(x)
    harmonic = np.zeros_like(x)
    term = np.ones_like(x)
    sign = -1.0
    for k in range(1, 60):
        term = term * x2 / (k * k)
        j0 = j0 + sign * term
        harmonic = harmonic + 1.0 / k
        series = series - sign * term * harmonic
        sign = -sign
        if np.max(np.abs(term)) < 1e-18:
            break
    y0 = (2.0 / math.pi) * (np.log(0.5 * x) + _EULER_GAMMA) * j0 + (
        2.0 / math.pi
    ) * series
    return j0, y0


def _j0_y0_asymptotic(x: FloatArray):
    """``J_0``, ``Y_0`` by the Hankel asymptotic series (``x >= 16``)."""
    x = np.asarray(x, dtype=float)
    inv = 1.0 / x
    abs_a = [1.0]
    for k in range(1, 15):
        abs_a.append(abs_a[-1] * (2.0 * k - 1.0) ** 2 / (8.0 * k))
    even = np.zeros_like(x)
    odd = np.zeros_like(x)
    even = np.zeros_like(x)
    odd = np.zeros_like(x)
    p2 = np.ones_like(x)  # inv^(2k)
    for k in range(7):
        even = even + (-1.0) ** k * abs_a[2 * k] * p2
        odd = odd + (-1.0) ** k * abs_a[2 * k + 1] * p2 * inv
        p2 = p2 * inv * inv
    chi = x - 0.25 * math.pi
    amp = np.sqrt(2.0 / (math.pi * x))
    j0 = amp * (np.cos(chi) * even + np.sin(chi) * odd)
    y0 = amp * (np.sin(chi) * even - np.cos(chi) * odd)
    return j0, y0


def bessel_j0_y0(x: ArrayLike):
    """Return ``(J_0(x), Y_0(x))`` for positive real ``x``."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("Bessel argument must be strictly positive")
    j0 = np.empty_like(x)
    y0 = np.empty_like(x)
    small = x < _BESSEL_SERIES_LIMIT
    if np.any(small):
        js, ys = _j0_y0_series(x[small])
        j0[small] = js
        y0[small] = ys
    if np.any(~small):
        ja, ya = _j0_y0_asymptotic(x[~small])
        j0[~small] = ja
        y0[~small] = ya
    return j0, y0


def hankel_h0(x: ArrayLike, *, outgoing: bool = True) -> ComplexArray:
    """``H_0^{(1)}(x)`` (outgoing) or its conjugate ``H_0^{(2)}(x)``."""
    j0, y0 = bessel_j0_y0(x)
    wave = j0 + 1j * y0
    return wave if outgoing else np.conjugate(wave)


def hankel_h1(x: ArrayLike, *, outgoing: bool = True) -> ComplexArray:
    """``H_1^{(1)}(x)`` (outgoing) or its conjugate ``H_1^{(2)}(x)``."""
    _, j1, _, y1 = bessel_j01_y01(x)
    wave = j1 + 1j * y1
    return wave if outgoing else np.conjugate(wave)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def polish_evanescent_roots(mu: FloatArray, nu: float, depth: float, *, steps: int = 3) -> FloatArray:
    """Newton-polish the evanescent roots of ``mu*tan(mu*h) = -nu``.

    :func:`tensorbem.finite_depth.evanescent_wavenumbers` brackets the roots by
    bisection, which leaves a dispersion residual of ``~1e-10`` relative.  A few
    Newton steps with the analytic derivative
    ``f'(mu) = tan(mu*h) + mu*h*sec(mu*h)**2`` drive that to machine precision,
    which matters because every mode's free-surface residual is proportional to
    the root error.
    """
    mu = np.array(mu, dtype=float, copy=True)
    for _ in range(steps):
        tan = np.tan(mu * depth)
        f = mu * tan + nu
        derivative = tan + mu * depth * (1.0 + tan * tan)
        step = f / derivative
        mu = mu - step
    return mu


def depth_modes(nu: float, depth: float, *, modes: Optional[int] = None):
    """Return ``(k0, mu, norms_evanescent, norm_propagating)`` for the strip."""
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu = omega**2/g must be finite and positive")
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth h must be finite and positive")
    k0 = finite_depth_wavenumber(nu, depth)
    count = _resolve_mode_count(depth, modes)
    mu = polish_evanescent_roots(evanescent_wavenumbers(nu, depth, count), nu, depth)
    return k0, mu, mode_norms(nu, depth, mu), propagating_mode_norm(k0, depth)


def _resolve_mode_count(depth: float, modes: Optional[int], *, r_min: float = 0.0) -> int:
    if modes is not None:
        wanted = int(modes)
        if wanted < 1:
            raise ValueError("modes must be >= 1")
        return min(wanted, MODES_CAP)
    # mu_n ~ n*pi/h, so n_cut such that exp(-mu_n*R) < 1e-12
    automatic = int(math.ceil(_LOG_DECAY * depth / (math.pi * max(r_min, 1e-9)))) + 4
    return int(min(max(automatic, 32), MODES_CAP))


def vertical_mode(z: ArrayLike, k0: float, depth: float) -> FloatArray:
    """Propagating vertical eigenfunction ``phi_0(z) = cosh(k0*(z + h))``."""
    z = np.asarray(z, dtype=float)
    if np.any(z > 1e-12) or np.any(z < -depth - 1e-12):
        raise ValueError("elevation z outside the fluid domain [-h, 0]")
    return np.cosh(k0 * (np.clip(z, -depth, 0.0) + depth))


def wave_amplitude_coefficient(z: ArrayLike, nu: float, depth: float) -> FloatArray:
    """Normalised propagating-mode shape ``phi_0(z) / N_0`` of the strip.

    Evaluated through the log-scaled factors, so it stays finite (and
    positive) for ``k0*h`` beyond the floating-point range of ``cosh`` where
    ``phi_0`` and ``N_0`` individually overflow.
    """
    z = np.asarray(z, dtype=float)
    if np.any(z > 1e-12) or np.any(z < -depth - 1e-12):
        raise ValueError("elevation z outside the fluid domain [-h, 0]")
    k0 = finite_depth_wavenumber(nu, depth)
    u, _, weight = _propagating_factors_3d(k0, depth)
    return u(np.clip(z, -depth, 0.0)) * math.exp(-k0 * depth) / weight



def _propagating_factors_3d(k0: float, depth: float):
    """Return ``(u, v, w)`` with ``cosh(k0*(z + h)) = exp(k0*h)*u(z)``.

    Copy of :func:`tensorbem.finite_depth._scaled_mode_factors` for the 3-D
    kernel.  In deep water (``k0*h >> 1``) the propagating mode is
    ``O(exp(k0*h))`` and so is its norm ``N_0``, so the *ratio*
    ``phi_0(z)*phi_0(zeta)/N_0`` is ``O(1)`` while every individual factor
    overflows (``cosh`` at ``k0*h >~ 710``, ``N_0`` earlier).  Factoring out
    ``exp(2*k0*h)`` (``N_0 = exp(2*k0*h)*w``) keeps the wave term finite for the
    whole deep-water regime instead of returning ``nan``::

        u(z) = (exp(k0*z) + exp(-k0*(z + 2*h)))/2,
        v(z) = (exp(k0*z) - exp(-k0*(z + 2*h)))/2,      # u' = k0*v
        w    = -expm1(-4*k0*h)/(8*k0) + (h/2)*exp(-2*k0*h).

    ``u``, ``v`` and ``w`` are all bounded on ``-h <= z <= 0``; the products
    below use ``u/sqrt(w)`` so no intermediate is large.
    """
    decay4 = math.exp(-4.0 * k0 * depth)
    weight = -math.expm1(-4.0 * k0 * depth) / (8.0 * k0) + 0.5 * depth * math.exp(
        -2.0 * k0 * depth
    )
    del decay4

    def u(z):
        return 0.5 * (np.exp(k0 * z) + np.exp(-k0 * (z + 2.0 * depth)))

    def v(z):
        return 0.5 * (np.exp(k0 * z) - np.exp(-k0 * (z + 2.0 * depth)))

    return u, v, weight

# ---------------------------------------------------------------------------
# Green function
# ---------------------------------------------------------------------------
def _broadcast_points_3d(targets, sources):
    tgt = np.asarray(targets, dtype=float)
    src = np.asarray(sources, dtype=float)
    if tgt.shape[-1] != 3 or src.shape[-1] != 3:
        raise ValueError("targets and sources must have trailing dimension 3")
    tgt, src = np.broadcast_arrays(tgt, src)
    return tgt, src


def green_function_3d(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: Optional[int] = None,
    include_wave: bool = True,
) -> ComplexArray:
    """3-D finite-depth free-surface Green function ``G(x, xi; nu, h)``.

    Parameters
    ----------
    targets, sources:
        Broadcastable arrays with trailing dimension 3 and ``z <= 0``; both
        points must lie inside the fluid strip ``-h <= z <= 0``.
    nu:
        ``omega**2 / g > 0``.
    depth:
        Water depth ``h > 0``; ``h -> inf`` recovers the deep-water kernel of
        :func:`tensorbem.free_surface3d.green_function_3d`.
    outgoing:
        Outgoing ``H_0^{(1)}`` wave term (default), or its conjugate.
    modes:
        Override the number of evanescent modes (default: automatic, targeting
        ``exp(-mu_n*R_min) <= 1e-12``).
    include_wave:
        Set ``False`` to drop the propagating term (evanescent sum only), which
        is handy for isolating the wave contribution in tests.

    Returns
    -------
    Complex array of shape ``np.broadcast(targets, sources).shape[:-1]``.
    """
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu = omega**2/g must be finite and positive")
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth h must be finite and positive")

    tgt, src = _broadcast_points_3d(targets, sources)
    z = tgt[..., 2]
    zeta = src[..., 2]
    if np.any(z > 1e-12) or np.any(zeta > 1e-12):
        raise ValueError("targets/sources must satisfy z <= 0 (z = 0 is the free surface)")
    if np.any(z < -depth - 1e-9) or np.any(zeta < -depth - 1e-9):
        raise ValueError("targets/sources must lie inside the fluid, z >= -depth")

    horiz = np.hypot(tgt[..., 0] - src[..., 0], tgt[..., 1] - src[..., 1])
    if np.any(horiz <= 0.0) and np.any(np.abs(z - zeta) <= 0.0):
        raise ValueError("coincident target and source is singular")

    r_min = float(np.min(horiz[horiz > 0.0])) if np.any(horiz > 0.0) else 1.0
    count = _resolve_mode_count(depth, modes, r_min=r_min)
    k0, mu, norms, norm0 = depth_modes(nu, depth, modes=count)

    mode_u, _, mode_weight = _propagating_factors_3d(k0, depth)
    # phi_0(z)*phi_0(zeta)/N_0 == u(z)*u(zeta)/w, finite for every k0*h
    wave_factor = (mode_u(z) / math.sqrt(mode_weight)) * (
        mode_u(zeta) / math.sqrt(mode_weight)
    )

    result = np.zeros(horiz.shape, dtype=complex)
    safe_horiz = np.maximum(horiz, 1e-300)
    for start in range(0, count, _CHUNK):
        block = mu[start : start + _CHUNK]
        nb = norms[start : start + _CHUNK]
        # phi_n(z) = cos(mu_n*(z + h)); outer product in mode index
        cz = np.cos(np.multiply.outer(z, block) + block * depth)
        cs = np.cos(np.multiply.outer(zeta, block) + block * depth)
        arg = np.multiply.outer(safe_horiz, block)
        kern = bessel_k0(arg)
        result += np.sum(cz * cs * kern * (1.0 / nb), axis=-1)
    result = result / _TWO_PI

    if include_wave:
        wave = hankel_h0(k0 * safe_horiz, outgoing=outgoing)
        result = result + 0.25j * wave_factor * wave
    return result.reshape(horiz.shape)


def green_gradient_z_3d(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: Optional[int] = None,
) -> ComplexArray:
    """Analytic ``dG/dz`` with respect to the *target* elevation."""
    tgt, src = _broadcast_points_3d(targets, sources)
    z = tgt[..., 2]
    zeta = src[..., 2]
    horiz = np.hypot(tgt[..., 0] - src[..., 0], tgt[..., 1] - src[..., 1])
    r_min = float(np.min(horiz[horiz > 0.0])) if np.any(horiz > 0.0) else 1.0
    count = _resolve_mode_count(depth, modes, r_min=r_min)
    k0, mu, norms, norm0 = depth_modes(nu, depth, modes=count)
    safe_horiz = np.maximum(horiz, 1e-300)

    mode_u, mode_v, mode_weight = _propagating_factors_3d(k0, depth)
    # dphi_0(z)/dz * phi_0(zeta)/N_0 == k0*v(z)*u(zeta)/w
    gradient_factor = k0 * (mode_v(z) / math.sqrt(mode_weight)) * (
        mode_u(zeta) / math.sqrt(mode_weight)
    )

    deriv = np.zeros(horiz.shape, dtype=complex)
    for start in range(0, count, _CHUNK):
        block = mu[start : start + _CHUNK]
        nb = norms[start : start + _CHUNK]
        dz_t = -block * np.sin(np.multiply.outer(z, block) + block * depth)
        cs = np.cos(np.multiply.outer(zeta, block) + block * depth)
        kern = bessel_k0(np.multiply.outer(safe_horiz, block))
        deriv += np.sum(dz_t * cs * kern * (1.0 / nb), axis=-1)
    deriv = deriv / _TWO_PI
    wave = hankel_h0(k0 * safe_horiz, outgoing=outgoing)
    deriv = deriv + 0.25j * gradient_factor * wave
    return deriv.reshape(horiz.shape)


def green_gradient_horizontal_3d(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: Optional[int] = None,
) -> ComplexArray:
    """Horizontal target gradient ``(dG/dx, dG/dy)`` of the finite-depth kernel.

    ``G`` depends on the horizontal coordinates only through the radial
    separation ``R``, so with ``dK_0/dR = -K_1`` and ``dH_0^{(1)}/dR = -H_1^{(1)}``

        dG/dR = -(1/2pi) sum_n mu_n phi_n(z) phi_n(zeta)/N_n K_1(mu_n R)
                - (i/4) k0 phi_0(z) phi_0(zeta)/N_0 H_1^{(1)}(k0 R),

    and ``dG/dx = (x - xi)/R dG/dR``, ``dG/dy = (y - eta)/R dG/dR``.  Together
    with :func:`green_gradient_z_3d` this completes ``nabla_x G``, which is what
    a boundary-element panel layer needs (the Rankine part of
    :mod:`tensorbem.free_surface_solver3d` splits off its ``1/(4 pi r)``
    singularity analytically; in finite depth the singular part is carried by
    the modal sum itself, so ``K_1`` must be accurate at small and moderate
    ``mu_n R``).

    Returns
    -------
    Complex array of shape ``(*broadcast_shape, 2)`` holding ``(dG/dx, dG/dy)``.
    """
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu = omega**2/g must be finite and positive")
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth h must be finite and positive")

    tgt, src = _broadcast_points_3d(targets, sources)
    z = tgt[..., 2]
    zeta = src[..., 2]
    if np.any(z > 1e-12) or np.any(zeta > 1e-12):
        raise ValueError("targets/sources must satisfy z <= 0 (z = 0 is the free surface)")
    if np.any(z < -depth - 1e-9) or np.any(zeta < -depth - 1e-9):
        raise ValueError("targets/sources must lie inside the fluid, z >= -depth")

    dx = tgt[..., 0] - src[..., 0]
    dy = tgt[..., 1] - src[..., 1]
    horiz = np.hypot(dx, dy)
    if np.any(horiz <= 0.0) and np.any(np.abs(z - zeta) <= 0.0):
        raise ValueError("coincident target and source is singular")

    r_min = float(np.min(horiz[horiz > 0.0])) if np.any(horiz > 0.0) else 1.0
    count = _resolve_mode_count(depth, modes, r_min=r_min)
    k0, mu, norms, norm0 = depth_modes(nu, depth, modes=count)

    mode_u, _, mode_weight = _propagating_factors_3d(k0, depth)
    wave_factor = (mode_u(z) / math.sqrt(mode_weight)) * (
        mode_u(zeta) / math.sqrt(mode_weight)
    )
    safe_horiz = np.maximum(horiz, 1e-300)

    radial = np.zeros(horiz.shape, dtype=complex)
    for start in range(0, count, _CHUNK):
        block = mu[start : start + _CHUNK]
        nb = norms[start : start + _CHUNK]
        cz = np.cos(np.multiply.outer(z, block) + block * depth)
        cs = np.cos(np.multiply.outer(zeta, block) + block * depth)
        arg = np.multiply.outer(safe_horiz, block)
        # d/dR K_0(mu*R) = -mu*K_1(mu*R)
        kern = -block * bessel_k1(arg)
        radial += np.sum(cz * cs * kern * (1.0 / nb), axis=-1)
    radial = radial / _TWO_PI
    # d/dR H_0^{(sigma)}(k0*R) = -k0*H_1^{(sigma)}(k0*R)
    radial = radial - 0.25j * k0 * wave_factor * hankel_h1(
        k0 * safe_horiz, outgoing=outgoing
    )

    nonzero = horiz > 0.0
    safe_r = np.where(nonzero, horiz, 1.0)
    scale = np.where(nonzero, radial / safe_r, 0.0j)
    gx = scale * dx
    gy = scale * dy
    return np.stack((gx, gy), axis=-1)


def green_gradient_source_3d(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: Optional[int] = None,
) -> ComplexArray:
    """Gradient ``nabla_xi G(x, xi)`` with respect to the *source* coordinates.

    The kernel is symmetric, ``G(x, xi) = G(xi, x)``, so the horizontal part is
    minus the field gradient while the vertical part is a genuinely different
    expression (the free surface breaks vertical translation invariance):

        dG/dzeta = (1/2pi) sum_n phi_n(z) phi_n'(zeta)/N_n K_0(mu_n R)
                   + (i/4) k0 phi_0(z) phi_0'(zeta)/N_0 H_0^{(1)}(k0 R),

    with ``phi_n'(zeta) = -mu_n sin(mu_n(zeta + h))`` and
    ``phi_0'(zeta) = k0 sinh(k0(zeta + h))``.  The identity
    ``nabla_x G(x, xi) == nabla_xi G(xi, x)`` therefore cross-validates *both*
    analytic derivatives (:mod:`tensorbem.benchmarks.finite_depth3d_gradient`).

    Returns
    -------
    Complex array of shape ``(*broadcast_shape, 3)`` holding
    ``(dG/dxi, dG/deta, dG/dzeta)``.
    """
    horizontal = green_gradient_horizontal_3d(
        targets, sources, nu, depth, outgoing=outgoing, modes=modes
    )
    tgt, src = _broadcast_points_3d(targets, sources)
    z = tgt[..., 2]
    zeta = src[..., 2]
    horiz = np.hypot(tgt[..., 0] - src[..., 0], tgt[..., 1] - src[..., 1])
    r_min = float(np.min(horiz[horiz > 0.0])) if np.any(horiz > 0.0) else 1.0
    count = _resolve_mode_count(depth, modes, r_min=r_min)
    k0, mu, norms, norm0 = depth_modes(nu, depth, modes=count)
    safe_horiz = np.maximum(horiz, 1e-300)

    mode_u, mode_v, mode_weight = _propagating_factors_3d(k0, depth)
    # phi_0(z) * dphi_0(zeta)/dzeta / N_0 == k0 * u(z) * v(zeta) / w
    wave_factor = k0 * (mode_u(z) / math.sqrt(mode_weight)) * (
        mode_v(zeta) / math.sqrt(mode_weight)
    )

    vertical = np.zeros(horiz.shape, dtype=complex)
    for start in range(0, count, _CHUNK):
        block = mu[start : start + _CHUNK]
        nb = norms[start : start + _CHUNK]
        cz = np.cos(np.multiply.outer(z, block) + block * depth)
        dzeta = -block * np.sin(np.multiply.outer(zeta, block) + block * depth)
        kern = bessel_k0(np.multiply.outer(safe_horiz, block))
        vertical += np.sum(cz * dzeta * kern * (1.0 / nb), axis=-1)
    vertical = vertical / _TWO_PI
    wave = hankel_h0(k0 * safe_horiz, outgoing=outgoing)
    vertical = vertical + 0.25j * wave_factor * wave

    return np.concatenate((-horizontal, vertical[..., None]), axis=-1)


def green_gradient_3d(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: Optional[int] = None,
) -> ComplexArray:
    """Full target gradient ``nabla_x G(x, xi)`` of the finite-depth kernel.

    Stacks :func:`green_gradient_horizontal_3d` and
    :func:`green_gradient_z_3d` into one ``(..., 3)`` complex array, matching
    the layout of :func:`tensorbem.free_surface3d.green_gradient_field_3d`.
    """
    horizontal = green_gradient_horizontal_3d(
        targets, sources, nu, depth, outgoing=outgoing, modes=modes
    )
    vertical = green_gradient_z_3d(
        targets, sources, nu, depth, outgoing=outgoing, modes=modes
    )
    return np.concatenate((horizontal, vertical[..., None]), axis=-1)


def dispersion_residual(k0: float, nu: float, depth: float) -> float:
    """Scaled residual ``|k0*tanh(k0*h) - nu| / nu`` of the dispersion relation."""
    return abs(k0 * math.tanh(k0 * depth) - nu) / nu


def deep_water_limit_error(
    points: ArrayLike,
    source: ArrayLike,
    nu: float,
    depth: float,
    *,
    modes: Optional[int] = None,
) -> float:
    """Max relative deviation from the deep-water kernel for ``k0*h >> 1``."""
    from .free_surface3d import green_function_3d as deep_green

    finite = green_function_3d(points, source, nu, depth, modes=modes)
    deep = deep_green(points, source, nu)
    scale = np.max(np.abs(deep))
    return float(np.max(np.abs(finite - deep)) / scale)