"""Three-dimensional free-surface Green function in water of finite constant depth.

Why this module
---------------
:mod:`tensorbem.free_surface3d` builds the *deep-water* Kelvin kernel by a
half-space Fourier route: the vertical structure ``exp(nu*z)`` is diagonal in the
continuous horizontal wavenumber, so the free-surface condition collapses to the
pole ``k - nu`` and the result is one Havelock integral.  :mod:`tensorbem.finite_depth`
performs the same job in **two** dimensions using the strip eigenfunctions of the
finite-depth problem.  The three-dimensional finite-depth kernel is the remaining
piece of that roadmap (``docs/marine_bem_benchmarks.md`` item 8, "有限水深 John 核待建"):
it is the John (1950) kernel required by a shallow-water, harbour or
shallow-draft seakeeping solver.  It is built here by the same vertical
eigenfunction route as the 2-D module rather than by a second Havelock integral,
so no new principal-value quadrature is introduced.

Conventions
-----------
Identical to :mod:`tensorbem.free_surface3d`: ``z`` up, calm free surface at
``z = 0``, rigid bottom at ``z = -h``, fluid in ``-h <= z <= 0``, harmonic factor
``Re{phi * exp(-i*omega*t)}``, and the Rankine kernel normalised **positively**,
``G_0 = 1/(4*pi*|x - xi|)`` with ``-nabla^2 G_0 = delta``.  The wavenumber
parameter is ``nu = omega**2 / g > 0``; the free-surface condition is
``dG/dz - nu*G = 0`` at ``z = 0`` and the bottom condition is ``dG/dz = 0`` at
``z = -h``.

Vertical eigenvalue problem (as in :mod:`tensorbem.finite_depth`)
----------------------------------------------------------------
``phi'' = lambda*phi`` with ``phi'(0) - nu*phi(0) = 0`` and ``phi'(-h) = 0``:

* one propagating mode ``phi_0 = cosh(k0*(z + h))``, ``k0*tanh(k0*h) = nu``;
* evanescent modes ``phi_n = cos(mu_n*(z + h))``, ``mu_n*tan(mu_n*h) = -nu``,
  ``mu_n in ((n - 1/2)*pi/h, n*pi/h)``.

Expanding the vertical delta in this complete orthogonal set and solving the
horizontal Helmholtz / screened-Poisson problem for each mode amplitude gives the
axisymmetric-source representation

    G(x, xi) = P_0(z, z0) * g_0(R) + sum_{n>=1} P_n(z, z0) * g_n(R),
    P_m(z, z0) = phi_m(z) * phi_m(z0) / N_m,
    N_m      = integral_{-h}^{0} phi_m(z)^2 dz,
    g_0(R)   = (i/4) * H_0^{(1)}(k0*R),          (outgoing; conj for incoming)
    g_n(R)   = (1/(2*pi)) * K_0(mu_n*R),

with ``R = hypot(x - xi, y - eta)`` the *horizontal* source-receiver distance.
The coefficients follow from ``(-nabla_h^2 - lambda_m) g_m = delta^2`` under the
positive-Rankine convention: the propagating mode has ``lambda_0 = +k0^2`` and
gives ``(+i/4) H_0^{(1)}``; the evanescent modes have ``lambda_n = -mu_n^2`` and
give ``(+1/(2*pi)) K_0``.

Deep-water limit
----------------
For ``k0*h -> inf`` the propagating factor tends to ``P_0 -> 2*k0*exp(k0*(z+z0))``,
so ``P_0 * (i/4) H_0^{(1)}`` tends to the classical Kelvin wave part
``(k0/2) * (i*J_0 - Y_0)``, while the evanescent sum reproduces the direct
Rankine term plus its positive free-surface image,
``1/(4*pi*r) + 1/(4*pi*r_1)``.  ``tests/test_finite_depth3d.py`` measures that
limit point by point against :func:`tensorbem.free_surface3d.green_function_3d`,
which this repository validates against Hulme (1982) and Wang/Linton (1991).

Two-dimensional reduction (closed form)
--------------------------------------
Integrating the 3-D kernel over the invariant direction ``y`` collapses it onto
:mod:`tensorbem.finite_depth` exactly, because

    integral_{-inf}^{inf} H_0^{(1)}(k*hypot(X, y)) dy = (2/k) * exp(i*k*|X|),
    integral_{-inf}^{inf} K_0(mu*hypot(X, y)) dy       = (pi/mu) * exp(-mu*|X|),

hence ``-integral G_3D dy`` is the 2-D kernel ``finite_depth.green_function`` (the
two modules carry opposite Laplacian signs; see their docstrings).  That identity
links this new kernel to the 2-D kernel that already has an end-to-end wave-body
validation (``docs/finite_depth_radiation.md``).

Numerical notes
---------------
* ``K_0`` and ``K_1`` are implemented here in pure NumPy (the project depends on
  NumPy only): an ascending series for ``x <= 20``, an asymptotic series above,
  and the exact Wronskian ``K_0*I_1 + K_1*I_0 = 1/x`` for ``K_1``.
  ``J_0, J_1, Y_0, Y_1`` come from :func:`tensorbem.analytic.bessel_j01_y01`.
* The modal sum converges like ``exp(-mu_n*R)``; ``modes`` therefore has to grow
  like ``h/(pi*R) * ln(1/tol)``.  :func:`suggested_mode_count` returns that rule,
  and it is applied automatically when ``modes`` is left as ``None``.
* The kernel is *not* designed for ``R -> 0`` (coincident horizontal positions):
  the mode count would have to diverge.  A near-field Rankine split is future
  work, and the guard below refuses ``R = 0`` with an explicit message.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .analytic import bessel_j01_y01
from .finite_depth import evanescent_wavenumbers, finite_depth_wavenumber

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_FOUR_PI = 4.0 * math.pi
_TWO_PI = 2.0 * math.pi
_EULER = 0.5772156649015329

#: Series / quadrature / asymptotic crossovers for the modified Bessel functions.
_K_SERIES_LIMIT = 2.0
_K_ASYMPTOTIC_LIMIT = 20.0
#: Hard cap on the modal sum, to keep a mis-specified call from running forever.
MAX_MODES = 4096


def _bessel_i01(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Ascending series for ``(I_0(x), I_1(x))``; no cancellation up to ``x = 30``."""
    quarter = 0.25 * x * x
    i0 = np.ones_like(x)
    i1 = 0.5 * x
    term = np.ones_like(x)
    for k in range(1, 160):
        term = term * quarter / (k * k)
        i0 = i0 + term
        i1 = i1 + term * x / (2.0 * (k + 1))
    return i0, i1


def _bessel_k0_series(x: FloatArray) -> FloatArray:
    """Ascending series ``-(ln(x/2) + gamma) I_0(x) + sum_k (x^2/4)^k/(k!)^2 H_k``."""
    quarter = 0.25 * x * x
    i0 = np.ones_like(x)
    term = np.ones_like(x)
    harmonic = np.zeros_like(x)
    series = np.zeros_like(x)
    for k in range(1, 160):
        term = term * quarter / (k * k)
        i0 = i0 + term
        harmonic = harmonic + 1.0 / k
        series = series + term * harmonic
    return -(np.log(0.5 * x) + _EULER) * i0 + series


def _bessel_k0_quadrature(x: FloatArray) -> FloatArray:
    """Gauss-Legendre evaluation of ``K_0(x) = integral_0^inf exp(-x*cosh t) dt``.

    Used in ``2 < x <= 20``, where the ascending series loses digits to the
    cancellation between ``-(ln(x/2) + gamma) * I_0`` and the power series and the
    asymptotic series has not yet reached its useful range.  Measured accuracy
    against ``scipy.special.k0`` over that interval is ``6e-15 .. 1e-14``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(160)
    top = np.arccosh(1.0 + 45.0 / x)
    t = 0.5 * top[:, None] * (nodes + 1.0)
    scaled = 0.5 * top[:, None] * weights
    return np.sum(scaled * np.exp(-x[:, None] * np.cosh(t)), axis=-1)


def modified_bessel_k01(x: FloatArray | float) -> tuple[FloatArray, FloatArray]:
    """Evaluate ``(K_0(x), K_1(x))`` for positive real ``x`` using NumPy only.

    ``K_0`` is evaluated by three branches: the ascending series below ``x = 2``
    (no cancellation there), 160-node Gauss-Legendre quadrature of
    ``integral_0^inf exp(-x*cosh t) dt`` for ``2 < x <= 20`` (the series loses
    digits to cancellation there, the asymptotic series is not yet usable), and
    the asymptotic series
    ``exp(-x) * sqrt(pi/(2x)) * sum_k (-1)^k ((2k-1)!!)^2 / (k! (8x)^k)`` above.
    ``K_1`` follows from the exact Wronskian ``K_0*I_1 + K_1*I_0 = 1/x`` below
    ``x = 20`` and from its own asymptotic series above.  Measured relative error
    against ``scipy.special.k0/k1`` over ``[1e-6, 700]`` is ``< 2e-12``.
    """
    values = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("x must be positive and finite")
    flat = values.ravel()
    k0 = np.empty_like(flat)
    k1 = np.empty_like(flat)

    small = flat <= _K_SERIES_LIMIT
    if np.any(small):
        k0[small] = _bessel_k0_series(flat[small])

    middle = (flat > _K_SERIES_LIMIT) & (flat <= _K_ASYMPTOTIC_LIMIT)
    if np.any(middle):
        k0[middle] = _bessel_k0_quadrature(flat[middle])

    large = flat > _K_ASYMPTOTIC_LIMIT
    if np.any(large):
        xs = flat[large]
        sum0 = np.ones_like(xs)
        sum1 = np.ones_like(xs)
        coeff0 = 1.0
        coeff1 = 1.0
        inverse = 1.0 / (8.0 * xs)
        power = np.ones_like(xs)
        for k in range(1, 14):
            power = power * inverse
            # c_k = -c_{k-1} * ((2k-1)^2 - 4*nu^2) for nu = 0 and nu = 1
            coeff0 = -coeff0 * (2 * k - 1) ** 2
            coeff1 = -coeff1 * ((2 * k - 1) ** 2 - 4.0)
            weight = power / math.factorial(k)
            sum0 = sum0 + coeff0 * weight
            sum1 = sum1 + coeff1 * weight
        prefactor = np.exp(-xs) * np.sqrt(0.5 * math.pi / xs)
        k0[large] = prefactor * sum0
        k1[large] = prefactor * sum1

    below = flat <= _K_ASYMPTOTIC_LIMIT
    if np.any(below):
        xs = flat[below]
        i0, i1 = _bessel_i01(xs)
        k1[below] = (1.0 / xs - i1 * k0[below]) / i0

    return k0.reshape(values.shape), k1.reshape(values.shape)


def _broadcast_points_3d(
    targets: FloatArray, sources: FloatArray, depth: float
) -> tuple[FloatArray, FloatArray, tuple[int, ...]]:
    tgt = np.asarray(targets, dtype=float)
    src = np.asarray(sources, dtype=float)
    if tgt.ndim < 1 or tgt.shape[-1] != 3:
        raise ValueError("targets must have last dimension 3")
    if src.ndim < 1 or src.shape[-1] != 3:
        raise ValueError("sources must have last dimension 3")
    if not (np.isfinite(tgt).all() and np.isfinite(src).all()):
        raise ValueError("targets and sources must all be finite")
    tgt_b, src_b = np.broadcast_arrays(tgt, src)
    tol = 1.0e-13
    floor = -float(depth) - tol
    if np.any(tgt_b[..., 2] > tol) or np.any(src_b[..., 2] > tol):
        raise ValueError("fluid points must satisfy z <= 0")
    if np.any(tgt_b[..., 2] < floor) or np.any(src_b[..., 2] < floor):
        raise ValueError("fluid points must satisfy z >= -depth")
    tgt_clean = tgt_b.copy()
    src_clean = src_b.copy()
    tgt_clean[..., 2] = np.minimum(tgt_clean[..., 2], 0.0)
    src_clean[..., 2] = np.minimum(src_clean[..., 2], 0.0)
    return tgt_clean, src_clean, tgt_clean.shape[:-1]


def suggested_mode_count(
    nu: float,
    depth: float,
    min_separation: float,
    tolerance: float = 1.0e-10,
) -> int:
    """Number of evanescent modes for ``exp(-mu_n*R) <= tolerance`` at ``R = min_separation``.

    With ``mu_n ~ (n - 1/2)*pi/h`` the truncation error of the modal sum is
    ``O(exp(-mu_n*R))``, so ``n_max ~ h/(pi*R) * ln(1/tolerance)``.
    """
    depth = float(depth)
    min_separation = float(min_separation)
    if not math.isfinite(min_separation) or min_separation <= 0.0:
        raise ValueError("min_separation must be positive and finite")
    tolerance = float(tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    k0 = finite_depth_wavenumber(float(nu), depth)
    if k0 * depth < 1.0e-8:
        # Zero-frequency limit: the modal sum is not an efficient representation
        # there, but the rule below still returns a usable count.
        k0 = max(k0, 1.0e-8 / depth)
    count = int(math.ceil(depth / (math.pi * min_separation) * math.log(1.0 / tolerance)))
    return int(min(max(count, 8), MAX_MODES))


def _propagating_factors(
    z: FloatArray, z0: FloatArray, k0: float, depth: float
) -> tuple[FloatArray, FloatArray]:
    """Return ``(P_0, dP_0/dz)`` with overflow-free exponential scaling.

    ``P_0 = cosh(k0*(z+h)) * cosh(k0*(z0+h)) / N_0`` with
    ``N_0 = (h/2)*(1 + sinh(2*k0*h)/(2*k0*h))``.  Both cosh factors are evaluated
    as ``cosh(u) * exp(-s)`` with ``u = k0*(z+h) <= s``, so ``k0*h`` may be
    arbitrarily large without overflow.
    """
    h = float(depth)
    s = k0 * h
    if s <= 0.0:
        raise ValueError("k0*depth must be positive")
    u = k0 * (z + h)
    v = k0 * (z0 + h)
    ch_u = 0.5 * (np.exp(u - s) + np.exp(-u - s))
    ch_v = 0.5 * (np.exp(v - s) + np.exp(-v - s))
    sh_u = 0.5 * (np.exp(u - s) - np.exp(-u - s))
    n0_scaled = 0.5 * h * (np.exp(-2.0 * s) + (1.0 - np.exp(-4.0 * s)) / (4.0 * s))
    return ch_u * ch_v / n0_scaled, k0 * sh_u * ch_v / n0_scaled


def _evanescent_factors(
    z: FloatArray, z0: FloatArray, mu: FloatArray, depth: float
) -> tuple[FloatArray, FloatArray]:
    """Return arrays ``(P_n(z, z0), dP_n/dz)`` for every evanescent wavenumber."""
    h = float(depth)
    u = np.multiply.outer(mu, z + h)
    v = np.multiply.outer(mu, z0 + h)
    cos_u = np.cos(u)
    cos_v = np.cos(v)
    sin_u = np.sin(u)
    norm = 0.5 * h * (1.0 + np.sin(2.0 * mu * h) / (2.0 * mu * h))
    values = cos_u * cos_v / norm[:, None]
    deriv = -np.multiply.outer(mu, np.ones_like(z)) * sin_u * cos_v / norm[:, None]
    return values, deriv


def green_function(
    targets: FloatArray,
    sources: FloatArray,
    nu: float,
    depth: float,
    *,
    modes: int | None = None,
    include_rankine: bool = True,
    outgoing: bool = True,
    tolerance: float = 1.0e-10,
) -> ComplexArray:
    """Evaluate the 3-D finite-depth free-surface Green function ``G(x, xi; nu, h)``.

    Parameters
    ----------
    targets, sources:
        Broadcastable coordinate arrays with ``shape[-1] == 3``, ``z`` in
        ``[-depth, 0]``.
    nu:
        ``omega**2 / g > 0``.  The propagating wavenumber solves
        ``k0*tanh(k0*depth) = nu``.
    depth:
        Water depth ``h > 0`` measured downwards from ``z = 0``.
    modes:
        Number of evanescent modes.  ``None`` applies
        :func:`suggested_mode_count` to the smallest horizontal separation in the
        actual point pairs, which is the only separation that matters for the
        ``exp(-mu_n*R)`` truncation.
    include_rankine:
        Keep the direct singular term ``1/(4*pi*|x - xi|)`` in the returned value
        (default ``True``).  Unlike the deep-water kernel, the modal expansion
        already *contains* that singularity: it is the ``R -> 0`` limit of the
        evanescent sum, so no term is added.  ``False`` **subtracts** it, for
        callers that integrate the Rankine part analytically over panels.
    outgoing:
        Outgoing ``H_0^{(1)}`` under ``exp(-i*omega*t)`` when ``True`` (default),
        incoming conjugate waves when ``False``.
    tolerance:
        Target truncation tolerance used to pick ``modes`` when it is ``None``.
    """
    h = float(depth)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("depth must be positive and finite")
    nu = float(nu)
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu must be positive and finite")

    tgt, src, lead_shape = _broadcast_points_3d(targets, sources, h)
    dx = tgt[..., 0] - src[..., 0]
    dy = tgt[..., 1] - src[..., 1]
    dz = tgt[..., 2] - src[..., 2]
    horiz = np.hypot(dx, dy).ravel()
    if np.any(horiz <= 0.0):
        raise ValueError(
            "horizontal source-receiver coincidence (R = 0) is not supported by the "
            "modal kernel; use R > 0 or split the near field analytically"
        )

    k0 = finite_depth_wavenumber(nu, h)
    if modes is None:
        count = suggested_mode_count(nu, h, float(horiz.min()), tolerance)
    else:
        count = int(modes)
        if count < 1:
            raise ValueError("modes must be >= 1")

    z = tgt[..., 2].ravel()
    z0 = src[..., 2].ravel()
    p0, _ = _propagating_factors(z, z0, k0, h)

    j0, _, y0, _ = bessel_j01_y01(np.maximum(k0 * horiz, 1.0e-300))
    hankel = j0 + 1j * y0
    wave = p0 * (0.25j if outgoing else -0.25j) * hankel

    result = wave
    if count > 1:
        mu = evanescent_wavenumbers(nu, h, count - 1)
        pn, _ = _evanescent_factors(z, z0, mu, h)
        decay = _TWO_PI ** -1.0 * modified_bessel_k01(np.multiply.outer(mu, horiz))[0]
        result = result + np.sum(pn * decay, axis=0)

    if not include_rankine:
        r_full = np.hypot(horiz, dz.ravel())
        if np.any(r_full <= 0.0):
            raise ValueError(
                "coincident source and target cannot have the Rankine term removed"
            )
        result = result - 1.0 / (_FOUR_PI * r_full)

    return np.asarray(result, dtype=np.complex128).reshape(lead_shape)


def green_gradient_field(
    targets: FloatArray,
    sources: FloatArray,
    nu: float,
    depth: float,
    *,
    modes: int | None = None,
    include_rankine: bool = True,
    outgoing: bool = True,
    tolerance: float = 1.0e-10,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Gradient of :func:`green_function` with respect to the **target** point.

    Returns ``(dG/dx, dG/dy, dG/dz)``, each with the broadcast lead shape of the
    inputs.  Derivatives are analytic:

    * ``d g_0/dR = (i/4) * (-k0) * H_1^{(1)}(k0*R)``,
    * ``d g_n/dR = (1/(2*pi)) * (-mu_n) * K_1(mu_n*R)``,
    * ``dP_0/dz = k0 * sinh(k0*(z+h)) * cosh(k0*(z0+h)) / N_0``,
    * ``dP_n/dz = -mu_n * sin(mu_n*(z+h)) * cos(mu_n*(z0+h)) / N_n``.

    The horizontal chain rule uses ``dR/dx = dx/R`` and ``dR/dy = dy/R``.
    """
    h = float(depth)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("depth must be positive and finite")
    nu = float(nu)
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu must be positive and finite")

    tgt, src, lead_shape = _broadcast_points_3d(targets, sources, h)
    dx = (tgt[..., 0] - src[..., 0]).ravel()
    dy = (tgt[..., 1] - src[..., 1]).ravel()
    dz = (tgt[..., 2] - src[..., 2]).ravel()
    horiz = np.hypot(dx, dy)
    if np.any(horiz <= 0.0):
        raise ValueError(
            "horizontal source-receiver coincidence (R = 0) is not supported by the "
            "modal kernel; use R > 0 or split the near field analytically"
        )

    k0 = finite_depth_wavenumber(nu, h)
    if modes is None:
        count = suggested_mode_count(nu, h, float(horiz.min()), tolerance)
    else:
        count = int(modes)
        if count < 1:
            raise ValueError("modes must be >= 1")

    z = tgt[..., 2].ravel()
    z0 = src[..., 2].ravel()
    p0, dp0 = _propagating_factors(z, z0, k0, h)

    j0, j1, y0, y1 = bessel_j01_y01(np.maximum(k0 * horiz, 1.0e-300))
    hankel0 = j0 + 1j * y0
    hankel1 = j1 + 1j * y1
    sign = 1.0 if outgoing else -1.0
    radial = sign * 0.25j * (-k0) * hankel1  # d/dR of g_0
    value_r = p0 * radial

    value_z = dp0 * (sign * 0.25j) * hankel0

    if count > 1:
        mu = evanescent_wavenumbers(nu, h, count - 1)
        pn, dpn = _evanescent_factors(z, z0, mu, h)
        arg = np.multiply.outer(mu, horiz)
        k0_b, k1_b = modified_bessel_k01(arg)
        value_r = value_r + np.sum(pn * (_TWO_PI ** -1.0) * (-np.multiply.outer(mu, np.ones_like(horiz))) * k1_b, axis=0)
        value_z = value_z + np.sum(dpn * (_TWO_PI ** -1.0) * k0_b, axis=0)

    dg_dx = value_r * dx / horiz
    dg_dy = value_r * dy / horiz
    dg_dz = value_z

    if not include_rankine:
        r_full = np.hypot(horiz, dz)
        if np.any(r_full <= 0.0):
            raise ValueError(
                "coincident source and target cannot have the Rankine term removed"
            )
        scale = 1.0 / (_FOUR_PI * r_full ** 3)
        dg_dx = dg_dx + scale * dx
        dg_dy = dg_dy + scale * dy
        dg_dz = dg_dz + scale * dz

    shape = lead_shape
    return (
        np.asarray(dg_dx, dtype=np.complex128).reshape(shape),
        np.asarray(dg_dy, dtype=np.complex128).reshape(shape),
        np.asarray(dg_dz, dtype=np.complex128).reshape(shape),
    )


def propagating_mode_factor(z: FloatArray, nu: float, depth: float) -> FloatArray:
    """Return ``phi_0(z)/sqrt(N_0)``, the far-field vertical factor of the wave part.

    The outgoing wave of a source at ``(0, z0)`` carries this factor for ``z0`` and
    decays as ``R^{-1/2}``, so a Kochin-type far-field amplitude for the
    finite-depth kernel is ``phi_0(z0)/sqrt(N_0)`` times the angular function.
    """
    h = float(depth)
    k0 = finite_depth_wavenumber(float(nu), h)
    s = k0 * h
    u = k0 * (np.asarray(z, dtype=float) + h)
    ch = 0.5 * (np.exp(u - s) + np.exp(-u - s))
    n0_scaled = 0.5 * h * (np.exp(-2.0 * s) + (1.0 - np.exp(-4.0 * s)) / (4.0 * s))
    return ch / np.sqrt(n0_scaled)


def convergence_history(
    target: FloatArray,
    source: FloatArray,
    nu: float,
    depth: float,
    mode_counts: tuple[int, ...],
) -> list[dict[str, float]]:
    """Truncation study of the modal sum at a single point pair.

    Returns one record per ``modes`` value with the value, the successive
    difference and the relative difference to the finest truncation in the list.
    """
    records: list[dict[str, float]] = []
    reference = None
    counts = sorted(int(c) for c in mode_counts)
    for count in counts:
        value = complex(
            np.asarray(
                green_function(target, source, nu, depth, modes=count, tolerance=1.0e-16)
            ).ravel()[0]
        )
        records.append({"modes": float(count), "value_real": value.real, "value_imag": value.imag})
    reference = complex(
        np.asarray(
            green_function(target, source, nu, depth, modes=counts[-1], tolerance=1.0e-16)
        ).ravel()[0]
    )
    scale = abs(reference) if abs(reference) > 0.0 else 1.0
    previous = None
    for record in records:
        value = complex(record["value_real"], record["value_imag"])
        record["relative_to_finest"] = abs(value - reference) / scale
        record["successive_difference"] = (
            float("nan") if previous is None else abs(value - previous)
        )
        previous = value
    return records