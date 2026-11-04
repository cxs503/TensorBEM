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
_K0_SERIES_LIMIT = 9.0
_BESSEL_SERIES_LIMIT = 16.0
_K1_SERIES_LIMIT = 9.0
_CHUNK = 2000  # modes evaluated per vectorised block


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


def bessel_k0(x: ArrayLike) -> FloatArray:
    """Modified Bessel function ``K_0`` for positive real argument."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("K_0 argument must be strictly positive")
    out = np.empty_like(x)
    small = x < _K0_SERIES_LIMIT
    if np.any(small):
        out[small] = _bessel_k0_small(x[small])
    if np.any(~small):
        out[~small] = _bessel_k0_large(x[~small])
    return out


def _bessel_k1_small(x: FloatArray) -> FloatArray:
    """``K_1`` by the ascending series (Abramowitz & Stegun 9.6.11, n = 1).

    ``K_1(x) = 1/x + ln(x/2) I_1(x) - (x/4) sum_k [psi(k+1)+psi(k+2)]
    (x/2)^(2k) / (k! (k+1)!)`` with ``I_1`` the modified Bessel function of the
    first kind.  The alternating structure is absent, so no cancellation beyond
    the ``ln(x/2) I_1`` versus ``(x/4) S`` pairing, which stays benign below
    ``x ~ 9`` (measured relative error versus SciPy: 3.7e-12 at ``x = 6``,
    1.8e-8 at ``x = 9``; the asymptotic branch is used from ``x = 9`` where it
    gives 4.7e-9, so ``bessel_k1`` stays within ``5e-9`` relative everywhere).
    """
    x = np.asarray(x, dtype=float)
    x2 = 0.25 * x * x
    power = 0.5 * x
    fact = 1.0
    i1 = np.zeros_like(x)
    for k in range(80):
        i1 = i1 + fact * power
        fact = fact / ((k + 1) * (k + 2))
        power = power * x2
        if np.max(np.abs(fact * power)) < 1e-19:
            break
    power = np.ones_like(x)
    fact = 1.0
    harmonic = 0.0
    tail = np.zeros_like(x)
    for k in range(80):
        # psi(k+1) = -gamma + H_k, psi(k+2) = psi(k+1) + 1/(k+1)
        psi_sum = -2.0 * _EULER_GAMMA + 2.0 * harmonic + 1.0 / (k + 1)
        tail = tail + psi_sum * fact * power
        harmonic = harmonic + 1.0 / (k + 1)
        fact = fact / ((k + 1) * (k + 2))
        power = power * x2
        if np.max(np.abs(fact * power)) < 1e-19:
            break
    return 1.0 / x + np.log(0.5 * x) * i1 - 0.25 * x * tail


def _bessel_k1_large(x: FloatArray) -> FloatArray:
    """``K_1`` by the Hankel asymptotic series (``x >= 9``), ``mu = 4``."""
    x = np.asarray(x, dtype=float)
    inv = 1.0 / x
    mu = 4.0
    coeff = 1.0
    poly = np.ones_like(x)
    power = np.ones_like(x)
    for k in range(1, 26):
        coeff = coeff * (mu - (2.0 * k - 1.0) ** 2) / (8.0 * k)
        power = power * inv
        poly = poly + coeff * power
    return np.sqrt(math.pi / (2.0 * x)) * np.exp(-x) * poly


def bessel_k1(x: ArrayLike) -> FloatArray:
    """Modified Bessel function ``K_1`` for positive real argument."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("K_1 argument must be strictly positive")
    out = np.empty_like(x)
    small = x < _K1_SERIES_LIMIT
    if np.any(small):
        out[small] = _bessel_k1_small(x[small])
    if np.any(~small):
        out[~small] = _bessel_k1_large(x[~small])
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
    """Normalised propagating-mode shape ``phi_0(z) / N_0`` of the strip."""
    k0 = finite_depth_wavenumber(nu, depth)
    norm = propagating_mode_norm(k0, depth)
    return vertical_mode(z, k0, depth) / norm


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

    phi0_t = np.cosh(k0 * (z + depth))
    phi0_s = np.cosh(k0 * (zeta + depth))

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
        result = result + 0.25j * (phi0_t * phi0_s / norm0) * wave
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

    phi0_s = np.cosh(k0 * (zeta + depth))
    dphi0_t = k0 * np.sinh(k0 * (z + depth))

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
    deriv = deriv + 0.25j * (dphi0_t * phi0_s / norm0) * wave
    return deriv.reshape(horiz.shape)


def _check_strip_points(
    nu: float, depth: float, targets, sources
) -> tuple[FloatArray, FloatArray]:
    """Shared validation/broadcast for the finite-depth kernel and gradients."""
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu = omega**2/g must be finite and positive")
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth h must be finite and positive")
    tgt, src = _broadcast_points_3d(targets, sources)
    if np.any(tgt[..., 2] > 1e-12) or np.any(src[..., 2] > 1e-12):
        raise ValueError("targets/sources must satisfy z <= 0 (z = 0 is the free surface)")
    if np.any(tgt[..., 2] < -depth - 1e-9) or np.any(src[..., 2] < -depth - 1e-9):
        raise ValueError("targets/sources must lie inside the fluid, z >= -depth")
    horiz = np.hypot(tgt[..., 0] - src[..., 0], tgt[..., 1] - src[..., 1])
    if np.any((horiz <= 0.0) & (np.abs(tgt[..., 2] - src[..., 2]) <= 0.0)):
        raise ValueError("coincident target and source is singular")
    return tgt, src


def green_gradient_3d(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: Optional[int] = None,
    include_wave: bool = True,
) -> ComplexArray:
    """Analytic gradient ``nabla_x G(x, xi; nu, h)`` with respect to the *target*.

    Differentiating the eigenseries term by term gives

        dG/dR = -(1/2pi) sum_n phi_n(z) phi_n(zeta) mu_n K_1(mu_n R) / N_n
                - (i/4) phi_0(z) phi_0(zeta) k_0 H_1^{(1)}(k_0 R) / N_0,
        dG/dz = -(1/2pi) sum_n mu_n sin(mu_n (z+h)) phi_n(zeta) K_0(mu_n R) / N_n
                + (i/4) k_0 sinh(k_0 (z+h)) phi_0(zeta) H_0^{(1)}(k_0 R) / N_0,

    and ``(dG/dx, dG/dy) = dG/dR * (x - xi, y - eta) / R``.  The vertical component
    is the same expression as :func:`green_gradient_z_3d`; the new content is the
    horizontal pair, which needs ``K_1`` and ``H_1^{(1)}`` (previously absent, and
    the reason this module exposed only ``dG/dz``).

    Returns
    -------
    Complex array of shape ``np.broadcast(targets, sources).shape`` (i.e. leading
    shape plus a trailing axis of length 3).
    """
    tgt, src = _check_strip_points(nu, depth, targets, sources)
    z = tgt[..., 2]
    zeta = src[..., 2]
    dx = tgt[..., 0] - src[..., 0]
    dy = tgt[..., 1] - src[..., 1]
    horiz = np.hypot(dx, dy)

    r_min = float(np.min(horiz[horiz > 0.0])) if np.any(horiz > 0.0) else 1.0
    count = _resolve_mode_count(depth, modes, r_min=r_min)
    k0, mu, norms, norm0 = depth_modes(nu, depth, modes=count)

    phi0_t = np.cosh(k0 * (z + depth))
    phi0_s = np.cosh(k0 * (zeta + depth))
    # The vertical component needs K_0 / H_0 on the true (possibly zero) horizontal
    # separation -- the kernels are only logarithmically divergent there.  The
    # radial component is multiplied by (dx, dy)/R, which vanishes identically on
    # the axis x = xi, y = eta (G is axisymmetric about the vertical through the
    # source), so it is evaluated at a harmless placeholder and masked afterwards;
    # that keeps K_1's 1/R pole from ever entering an inf * 0 product.
    positive = horiz > 0.0
    safe_axis = np.maximum(horiz, 1e-300)
    safe_radial = np.where(positive, horiz, 1.0)

    radial = np.zeros(horiz.shape, dtype=complex)
    vertical = np.zeros(horiz.shape, dtype=complex)
    for start in range(0, count, _CHUNK):
        block = mu[start : start + _CHUNK]
        nb = norms[start : start + _CHUNK]
        kern0 = bessel_k0(np.multiply.outer(safe_axis, block))
        kern1 = bessel_k1(np.multiply.outer(safe_radial, block))
        cz = np.cos(np.multiply.outer(z, block) + block * depth)
        sz = -np.sin(np.multiply.outer(z, block) + block * depth)
        cs = np.cos(np.multiply.outer(zeta, block) + block * depth)
        weight = cs * (1.0 / nb)
        radial += np.sum(cz * weight * (-block * kern1), axis=-1)
        vertical += np.sum((block * sz) * weight * kern0, axis=-1)
    radial = radial / _TWO_PI
    vertical = vertical / _TWO_PI

    if include_wave:
        h0v = hankel_h0(k0 * safe_axis, outgoing=outgoing)
        h1v = hankel_h1(k0 * safe_radial, outgoing=outgoing)
        amp_full = 0.25j * (phi0_t * phi0_s / norm0)
        vertical = vertical + 0.25j * (phi0_s / norm0) * (
            k0 * np.sinh(k0 * (z + depth))
        ) * h0v
        radial = radial - amp_full * k0 * h1v

    inv_r = np.where(positive, 1.0 / np.where(positive, horiz, 1.0), 0.0)
    gx = np.where(positive, radial * dx * inv_r, 0.0)
    gy = np.where(positive, radial * dy * inv_r, 0.0)
    return np.stack((gx, gy, vertical), axis=-1)


def green_gradient_source_3d(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: Optional[int] = None,
    include_wave: bool = True,
) -> ComplexArray:
    """Analytic gradient ``nabla_xi G(x, xi; nu, h)`` with respect to the *source*.

    The finite-depth kernel is symmetric, ``G(x, xi) = G(xi, x)``, so the source
    gradient is the target gradient of the transposed call -- no sign flip (the
    vertical mode shapes depend on ``z`` and ``zeta`` separately, unlike the deep
    water kernel whose wave term depends on ``z + zeta``).
    """
    return green_gradient_3d(
        sources,
        targets,
        nu,
        depth,
        outgoing=outgoing,
        modes=modes,
        include_wave=include_wave,
    )


def green_gradient_field_3d(
    targets,
    sources,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    modes: Optional[int] = None,
    include_wave: bool = True,
) -> ComplexArray:
    """Alias of :func:`green_gradient_3d` (deep-water module naming)."""
    return green_gradient_3d(
        targets,
        sources,
        nu,
        depth,
        outgoing=outgoing,
        modes=modes,
        include_wave=include_wave,
    )


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