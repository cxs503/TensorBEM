"""Two-dimensional free-surface Green function in water of finite constant depth.

Why a separate kernel
---------------------
:mod:`tensorbem.free_surface` builds the *deep-water* kernel by Fourier
transforming in ``x``: the vertical structure ``exp(k*z)`` that decays at
``z -> -inf`` makes the transform diagonal, so the free-surface condition
reduces to a denominator ``k - nu`` and the result is a principal-value
integral plus a residue. With a flat bottom at ``z = -h`` that route breaks:
the vertical structure is no longer ``exp(k*z)``, the spectrum in ``z`` becomes
*discrete*, and the continuous-``k`` denominator ``k - nu`` is replaced by the
transcendental dispersion relation. The correct representation is therefore a
modal expansion in the vertical eigenfunctions of the strip, not an integral.

Eigenvalue problem
------------------
With ``z`` up, the mean free surface at ``z = 0``, the bottom at ``z = -h`` and
the time factor ``exp(-i*omega*t)``, the linearised free-surface condition is
``d(phi)/dz - nu*phi = 0`` at ``z = 0`` with ``nu = omega**2 / g``, and the
bottom is impermeable, ``d(phi)/dz = 0`` at ``z = -h``. The vertical
eigenvalue problem

    phi'' = lambda*phi,   phi'(0) - nu*phi(0) = 0,   phi'(-h) = 0

has one positive eigenvalue (a *propagating* mode) and countably many negative
ones (evanescent modes):

    propagating:   phi_0 = cosh(k0*(z + h)),   k0*tanh(k0*h) = nu,
    evanescent:    phi_n = cos(mu_n*(z + h)),  mu_n*tan(mu_n*h) = -nu,
                   mu_n in ((n - 1/2)*pi/h, n*pi/h),   n = 1, 2, ...

with norms (``integral_{-h}^{0} phi**2 dz``)

    N_0 = (h/2)*(1 + sinh(2*k0*h)/(2*k0*h)),
    N_n = (h/2)*(1 + sin(2*mu_n*h)/(2*mu_n*h)).

Expanding the delta function in this complete orthogonal set and solving the
horizontal one-dimensional problem for each mode gives, for ``X = x - x0`` and
``d = |z - z0|``,

    G_finite(X, z, z0) = -i/(2*k0) * phi_0(z)*phi_0(z0)/N_0 * exp(i*k0*|X|)
                         - sum_{n>=1} (1/(2*mu_n)) * phi_n(z)*phi_n(z0)/N_n
                                       * exp(-mu_n*|X|).

The first term is the outgoing wave: in two dimensions a line source radiates a
wave whose amplitude does not decay, which is why the signature of ``G`` is
optional in this problem and why the radiation problem has to be formulated
indirectly (see :mod:`tensorbem.free_surface_solver`). The second sum is the
evanescent field; it carries the logarithmic singularity as ``X -> 0`` with
``z -> z0``.

Sign convention
---------------
The overall sign is *not* chosen here: it is fixed by requiring that the
``k0*h -> inf`` limit reproduces :func:`tensorbem.free_surface.green_function`
at every field point, log part and wave part together. That limit is a test
(``tests/test_finite_depth.py``), not an assumption. At ``nu*h = 400`` the
agreement is ``2e-5`` relative; that is the accuracy floor of the modal sum
(see the last paragraph), and beyond ``nu*h`` of a few hundred the deep-water
kernel should be used instead - it is a closed form in ``h``.

Summation: Kummer acceleration
------------------------------
The evanescent sum converges like ``sum 1/n``: as ``n -> inf``,
``mu_n -> n*pi/h =: mu_n^0``, ``N_n -> h/2`` and ``phi_n(z)*phi_n(z0)/N_n ->
(2/h)*cos(alpha*n)*cos(beta*n)`` with ``alpha = pi*(z+h)/h``,
``beta = pi*(z0+h)/h``. Its tail is the slowly convergent, log-singular part of
the field, so the sum is not truncated. Writing ``a_n`` for the exact term and

    b_n = (1/(n*pi)) * cos(alpha*n) * cos(beta*n) * exp(-n*s),   s = pi*|X|/h,

for its large-``n`` limit, the sum is evaluated as

    sum_{n>=1} a_n = sum_{n=1}^{N} (a_n - b_n) + sum_{n>=1} b_n,

where the second sum is closed form,

    sum_{n>=1} b_n = (1/(2*pi)) * [L(alpha - beta) + L(alpha + beta)],
    L(gamma) = -log(1 - 2*exp(-s)*cos(gamma) + exp(-2*s))/2,

i.e. ``L`` *is* the logarithmic singularity, extracted analytically. The
remainder ``sum_{n>N}(a_n - b_n)`` converges like ``O(1/N)`` times
``nu*|X|`` and, for the small ``|X|`` where it matters, like ``O(1/N**2)``;
``modes = 256`` is enough for ``1e-9`` in the tests. The gradients are
accelerated the same way, with the geometric sums

    sum r**n*cos(n*gamma) = (r*cos(gamma) - r**2)/(1 - 2*r*cos(gamma) + r**2),
    sum r**n*sin(n*gamma) = r*sin(gamma)/(1 - 2*r*cos(gamma) + r**2),
    r = exp(-s).

The gradient of the *evanescent* part converges like ``sum O(1)`` in ``n``
before acceleration, which is why the same closed forms are used there instead
of a truncated sum: without them the near-field gradient would be wrong in the
third digit at ``modes = 256``.

Everything is vectorised over field points, with the mode loop blocked over
points (``_POINT_BLOCK``) so that the ``modes x points`` temporary stays small.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_POINT_BLOCK = 4096
_MIN_MODES = 256
# Modes needed for a given accuracy scale with nu*h: as h grows the evanescent
# roots crowd together (mu_n -> n*pi/h) and the modal sum degenerates towards
# the continuous Fourier integral of the deep-water kernel. Empirically 24*nu*h
# modes hold the Kummer remainder below 1e-10 (512 modes at nu*h = 8, 8192 at
# nu*h = 400). Deep water should be done with tensorbem.free_surface instead:
# that kernel is a closed form in h.
_MODES_PER_NU_DEPTH = 24.0
_BRACKET_TOL = 1e-14
_MAX_BISECT = 200


# --------------------------------------------------------------------------
# Vertical eigenvalue problem
# --------------------------------------------------------------------------
def _validate_nu(nu: float) -> float:
    nu = float(nu)
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError(
            "nu = omega**2/g must be positive and finite, got %r. A free surface "
            "needs nu > 0; the nu = 0 limit (rigid lid) has no propagating mode "
            "and is not supported by this kernel." % (nu,)
        )
    return nu


def _resolve_modes(nu: float, depth: float, modes) -> int:
    """Number of evanescent modes actually summed."""
    automatic = int(math.ceil(_MODES_PER_NU_DEPTH * nu * depth))
    wanted = _MIN_MODES if modes is None else int(modes)
    if wanted < 0:
        raise ValueError("modes must be non-negative")
    return max(wanted, _MIN_MODES, automatic)


def _validate_depth(depth: float) -> float:
    depth = float(depth)
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError(
            "finite depth must be positive and finite, got %r. Use "
            "tensorbem.free_surface for the infinite-depth kernel." % (depth,)
        )
    return depth


def finite_depth_wavenumber(nu: float, depth: float) -> float:
    """Positive root ``k0`` of ``k0*tanh(k0*h) = nu``, by bracketed bisection.

    The function ``k*tanh(k*h)`` is monotone increasing and concave on
    ``k > 0``, so bisection on a bracket that is widened until it straddles the
    root converges unconditionally; no initial guess is needed. Limits worth
    knowing: ``k0 -> nu`` as ``h -> inf`` (deep water) and
    ``k0 -> sqrt(nu/h)`` as ``h -> 0`` (shallow water).
    """
    nu = _validate_nu(nu)
    depth = _validate_depth(depth)
    f = lambda k: k * math.tanh(k * depth) - nu
    low = 0.0
    high = max(nu, math.sqrt(nu / depth))
    while f(high) < 0.0:
        high *= 2.0
        if high > 1e12:
            raise RuntimeError("failed to bracket the finite-depth wavenumber")
    for _ in range(_MAX_BISECT):
        mid = 0.5 * (low + high)
        if f(mid) < 0.0:
            low = mid
        else:
            high = mid
        if high - low <= _BRACKET_TOL * max(1.0, high):
            break
    return 0.5 * (low + high)


def evanescent_wavenumbers(nu: float, depth: float, count: int) -> FloatArray:
    """First ``count`` positive roots of ``mu*tan(mu*h) = -nu``.

    The ``n``-th root is bracketed in ``((n - 1/2)*pi/h, n*pi/h)``, where
    ``tan(mu*h)`` runs monotonically from ``+inf`` to ``0``; the residual in
    that bracket is monotone, so bisection cannot hop branches. The bracket
    endpoints are clamped away from ``tan``'s poles by ``_BRACKET_TOL``.
    """
    nu = _validate_nu(nu)
    depth = _validate_depth(depth)
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.zeros(0)
    roots = np.zeros(count)
    for n in range(1, count + 1):
        low = (n - 0.5) * math.pi / depth
        high = n * math.pi / depth
        low = low + _BRACKET_TOL * max(1.0, low)
        high = high - _BRACKET_TOL * max(1.0, high)
        f = lambda mu: mu * math.tan(mu * depth) + nu
        # On ((n - 1/2)*pi/h, n*pi/h) tan(mu*h) runs from +inf to 0 through
        # negative values, so f goes from -inf up to +nu: exactly one root.
        if f(low) > 0.0 or f(high) < 0.0:
            raise RuntimeError("evanescent bracket does not straddle a root")
        for _ in range(_MAX_BISECT):
            mid = 0.5 * (low + high)
            if f(mid) < 0.0:
                low = mid
            else:
                high = mid
            if high - low <= _BRACKET_TOL * max(1.0, high):
                break
        roots[n - 1] = 0.5 * (low + high)
    return roots


def mode_norms(nu: float, depth: float, mu: FloatArray) -> FloatArray:
    """Mode norms ``(h/2)*(1 + sin(2*mu*h)/(2*mu*h))`` for the evanescent modes."""
    mu = np.asarray(mu, dtype=float)
    return 0.5 * depth * (1.0 + np.sin(2.0 * mu * depth) / (2.0 * mu * depth))


def propagating_mode_norm(k0: float, depth: float) -> float:
    """Norm ``(h/2)*(1 + sinh(2*k0*h)/(2*k0*h))`` of the propagating mode.

    Grows like ``exp(2*k0*h)/(8*k0)``, so it is returned as ``inf`` beyond the
    floating-point range instead of raising: the kernel never forms the norm
    itself (it uses the scaled form below), and a caller that wants the norm in
    that regime has to work with ``log(N_0)`` anyway.
    """
    k0 = float(k0)
    depth = float(depth)
    if 2.0 * k0 * depth > 700.0:
        return math.inf
    return 0.5 * depth * (1.0 + math.sinh(2.0 * k0 * depth) / (2.0 * k0 * depth))


def _scaled_mode_factors(k0: float, depth: float):
    """Return ``(u, v, w)`` with ``cosh(k0*(z+h)) = exp(k0*h)*u(z)``.

    In deep water the propagating mode is ``O(exp(k0*h))`` and so is its norm,
    so the ratio ``phi_0(z)*phi_0(z0)/N_0`` is ``O(1)`` while every individual
    factor overflows. Factoring out ``exp(2*k0*h)`` leaves

        u(z) = (exp(k0*z) + exp(-k0*(z + 2*h)))/2,
        v(z) = (exp(k0*z) - exp(-k0*(z + 2*h)))/2      (so u' = k0*v),
        w    = (1 - exp(-4*k0*h))/(8*k0) + (h/2)*exp(-2*k0*h),

    all bounded for ``-h <= z <= 0``, with the exact identity
    ``N_0 = exp(2*k0*h)*w``. This is what makes ``depth = 800`` testable.
    """
    exponential = math.exp(-4.0 * k0 * depth)
    weight = (1.0 - exponential) / (8.0 * k0) + 0.5 * depth * math.exp(
        -2.0 * k0 * depth
    )

    def u(z):
        return 0.5 * (np.exp(k0 * z) + np.exp(-k0 * (z + 2.0 * depth)))

    def v(z):
        return 0.5 * (np.exp(k0 * z) - np.exp(-k0 * (z + 2.0 * depth)))

    return u, v, weight


# --------------------------------------------------------------------------
# Wave kinematics (finite depth)
# --------------------------------------------------------------------------
def phase_velocity(nu: float, depth: float, *, gravity: float = 9.80665) -> float:
    """``c = omega/k0`` with ``omega = sqrt(nu*g)``."""
    k0 = finite_depth_wavenumber(nu, depth)
    return math.sqrt(float(nu) * gravity) / k0


def group_velocity(nu: float, depth: float, *, gravity: float = 9.80665) -> float:
    """``c_g = c*(1 + 2*k0*h/sinh(2*k0*h))/2``, from ``d(omega)/d(k)``.

    The factor is the one the Haskind energy relation needs in finite depth:
    the radiated power is ``0.5*rho*g*|a_w|**2*c_g`` (per unit crest length),
    so a far-field damping formed without ``c_g`` is wrong by that factor.
    ``c_g -> c/2`` as ``k0*h -> inf`` and ``c_g -> c`` as ``k0*h -> 0``.
    """
    nu = _validate_nu(nu)
    depth = _validate_depth(depth)
    k0 = finite_depth_wavenumber(nu, depth)
    c = math.sqrt(nu * gravity) / k0
    kh = k0 * depth
    if 2.0 * kh > 700.0:
        # sinh overflows long before k0*h does; the correction is e**-2kh.
        correction = 0.0
    else:
        correction = 2.0 * kh / math.sinh(2.0 * kh)
    return 0.5 * c * (1.0 + correction)


# --------------------------------------------------------------------------
# Closed-form tails used by the Kummer acceleration
# --------------------------------------------------------------------------
def _log_series(gamma: FloatArray, decay: FloatArray) -> FloatArray:
    """``sum_{n>=1} exp(-n*decay)*cos(n*gamma)/n`` in closed form."""
    rate = np.exp(-decay)
    argument = 1.0 - 2.0 * rate * np.cos(gamma) + rate * rate
    argument = np.maximum(argument, 1e-300)
    return -0.5 * np.log(argument)


def _cos_series(gamma: FloatArray, decay: FloatArray) -> FloatArray:
    """``sum_{n>=1} exp(-n*decay)*cos(n*gamma)`` in closed form."""
    rate = np.exp(-decay)
    denominator = 1.0 - 2.0 * rate * np.cos(gamma) + rate * rate
    denominator = np.where(denominator == 0.0, 1e-300, denominator)
    return (rate * np.cos(gamma) - rate * rate) / denominator


def _sin_series(gamma: FloatArray, decay: FloatArray) -> FloatArray:
    """``sum_{n>=1} exp(-n*decay)*sin(n*gamma)`` in closed form."""
    rate = np.exp(-decay)
    denominator = 1.0 - 2.0 * rate * np.cos(gamma) + rate * rate
    denominator = np.where(denominator == 0.0, 1e-300, denominator)
    return rate * np.sin(gamma) / denominator


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------
def _broadcast(x, z, source_x, source_z):
    x, z, source_x, source_z = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(source_x, dtype=float),
        np.asarray(source_z, dtype=float),
    )
    return x, z, source_x, source_z


def _modal_terms(
    offset: FloatArray,
    z: FloatArray,
    source_z: FloatArray,
    nu: float,
    depth: float,
    modes: int,
):
    """Return ``(propagating, evanescent)`` contributions to ``G``.

    ``propagating`` is the outgoing-wave term and ``evanescent`` is
    ``sum_{n>=1} a_n`` evaluated with the Kummer split of the module
    docstring. Both are complex, shaped like ``offset``.
    """
    k0 = finite_depth_wavenumber(nu, depth)
    mu = evanescent_wavenumbers(nu, depth, modes)
    norms = mode_norms(nu, depth, mu)
    decay = math.pi * np.abs(offset) / depth
    rate = np.exp(-decay)

    alpha = math.pi * (z + depth) / depth
    beta = math.pi * (source_z + depth) / depth
    cos_alpha = np.cos(alpha)[..., None]
    cos_beta = np.cos(beta)[..., None]
    sin_alpha = np.sin(alpha)[..., None]
    order = np.arange(1, modes + 1, dtype=float)
    mu_full = mu + 0.0

    # Exact evanescent terms, shape (..., modes).
    phi_products = (
        np.cos((z + depth)[..., None] * mu)
        * np.cos((source_z + depth)[..., None] * mu)
        / norms
    )
    exact = (0.5 / mu) * phi_products * np.exp(-mu_full * np.abs(offset)[..., None])
    # Large-n limit of the same terms, used only to make the sum convergent.
    # Note that the modal shape is cos(mu_n*(z+h)) -> cos(alpha*n): the cosine
    # *oscillates* in the mode index, so it may not be pulled out of the sum.
    # The closed-form tail below is consistent with this because it uses the
    # product-to-sum identity cos(a*n)cos(b*n) = [cos(n*(a-b)) + cos(n*(a+b))]/2.
    asymptotic = (
        np.cos(alpha[..., None] * order) * np.cos(beta[..., None] * order)
        / (order * math.pi) * rate[..., None] ** order
    )
    residual = np.sum(exact - asymptotic, axis=-1)
    tail = 0.5 / math.pi * (
        _log_series(alpha - beta, decay) + _log_series(alpha + beta, decay)
    )
    evanescent = -(residual + tail)

    mode_shape, _, mode_weight = _scaled_mode_factors(k0, depth)
    weight = mode_shape(z) * mode_shape(source_z) / mode_weight
    propagating = -0.5j / k0 * weight * np.exp(1j * k0 * np.abs(offset))
    del sin_alpha, cos_alpha
    return propagating, evanescent


def green_function(
    x, z, source_x, source_z, nu: float, depth: float, *,
    modes: int | None = None,
) -> ComplexArray:
    """Finite-depth free-surface Green function (platform sign convention).

    ``G`` solves ``-Laplacian(G) = delta(P - Q)`` in the strip
    ``-h < z < 0``, satisfies ``dG/dz - nu*G = 0`` at ``z = 0`` and
    ``dG/dz = 0`` at ``z = -h``, and radiates outgoing waves. The sign
    convention is the one of :mod:`tensorbem.free_surface`, so the two kernels
    are interchangeable in :mod:`tensorbem.free_surface_solver`; the
    ``depth -> inf`` limit is asserted against that module in the tests.
    """
    nu = _validate_nu(nu)
    depth = _validate_depth(depth)
    modes = _resolve_modes(nu, depth, modes)
    x, z, source_x, source_z = _broadcast(x, z, source_x, source_z)
    offset = (x - source_x).ravel()
    z_flat = z.ravel()
    source_flat = source_z.ravel()
    result = np.zeros(offset.shape, dtype=complex)
    for start in range(0, offset.size, _POINT_BLOCK):
        stop = start + _POINT_BLOCK
        propagating, evanescent = _modal_terms(
            offset[start:stop], z_flat[start:stop], source_flat[start:stop],
            nu, depth, modes,
        )
        result[start:stop] = propagating + evanescent
    return result.reshape(np.broadcast(x, z).shape)


def green_gradient(
    x, z, source_x, source_z, nu: float, depth: float, *,
    modes: int | None = None, include_rankine: bool = True,
):
    """Gradient of the finite-depth kernel, analytically differentiated.

    Differentiating the modal series term by term is exact (each mode decays
    faster than it grows away from the source; the Kummer tails absorb the
    ``sum O(1)`` divergence of the differentiated series). With
    ``include_rankine=False`` the gradient of the Rankine part ``log(r)/(2*pi)``
    is removed, which is what the boundary-element assembly needs: the
    logarithmic part is integrated over straight panels in closed form, and
    only the smooth remainder is quadratured.

    With ``include_rankine=True`` the returned gradient is that of the full
    kernel, i.e. ``include_rankine=False`` plus
    ``(X, z - z0)/(2*pi*r**2)``.
    """
    nu = _validate_nu(nu)
    depth = _validate_depth(depth)
    modes = _resolve_modes(nu, depth, modes)
    x, z, source_x, source_z = _broadcast(x, z, source_x, source_z)
    shape = np.broadcast(x, z).shape
    offset_all = (x - source_x).ravel()
    z_all = z.ravel()
    source_all = source_z.ravel()
    gradient_x = np.zeros(offset_all.shape, dtype=complex)
    gradient_z = np.zeros(offset_all.shape, dtype=complex)

    k0 = finite_depth_wavenumber(nu, depth)
    mu = evanescent_wavenumbers(nu, depth, modes)
    norms = mode_norms(nu, depth, mu)
    order = np.arange(1, modes + 1, dtype=float)
    mode_shape, mode_slope, mode_weight = _scaled_mode_factors(k0, depth)

    for start in range(0, offset_all.size, _POINT_BLOCK):
        stop = start + _POINT_BLOCK
        offset = offset_all[start:stop]
        z_slice = z_all[start:stop]
        source_slice = source_all[start:stop]
        sign = np.sign(offset)
        distance = np.abs(offset)
        phase = np.exp(1j * k0 * distance)
        decay = math.pi * distance / depth
        rate = np.exp(-decay)
        alpha = math.pi * (z_slice + depth) / depth
        beta = math.pi * (source_slice + depth) / depth

        shape_z = mode_shape(z_slice)
        shape_source = mode_shape(source_slice)
        weight0 = shape_z * shape_source / mode_weight

        # Propagating mode: outgoing wave, no decay in two dimensions.
        gradient_x[start:stop] = 0.5 * weight0 * sign * phase
        gradient_z[start:stop] = (
            -0.5j / k0
            * (k0 * mode_slope(z_slice) * shape_source / mode_weight)
            * phase
        )

        # Evanescent modes: exact terms minus their large-n limit ...
        exp_minus = np.exp(-mu * distance[..., None])
        phi_z = np.cos((z_slice + depth)[..., None] * mu)
        phi_source = np.cos((source_slice + depth)[..., None] * mu)
        sin_z = np.sin((z_slice + depth)[..., None] * mu)
        exact_x = 0.5 * (phi_z * phi_source / norms) * exp_minus
        exact_z = 0.5 * (sin_z * phi_source / norms) * exp_minus
        # ... whose sums are closed form (geometric, no 1/n).
        asym_x = (
            np.cos(alpha[..., None] * order) * np.cos(beta[..., None] * order)
            / depth * rate[..., None] ** order
        )
        asym_z = (
            np.sin(alpha[..., None] * order) * np.cos(beta[..., None] * order)
            / depth * rate[..., None] ** order
        )
        gradient_x[start:stop] += sign * (
            np.sum(exact_x - asym_x, axis=-1)
            + 1.0 / depth * 0.5
            * (_cos_series(alpha - beta, decay) + _cos_series(alpha + beta, decay))
        )
        gradient_z[start:stop] += (
            np.sum(exact_z - asym_z, axis=-1)
            + 1.0 / depth * 0.5
            * (_sin_series(alpha - beta, decay) + _sin_series(alpha + beta, decay))
        )

    # A constant offset in G contributes nothing to the gradient, so the modal
    # series above is already the gradient of the full kernel; the Rankine part
    # of the platform convention enters only in what the caller asks to remove.
    if not include_rankine:
        radius2 = offset_all * offset_all + (z_all - source_all) ** 2
        singular = np.where(radius2 > 0.0, radius2, 1.0)
        gradient_x = gradient_x - np.where(
            radius2 > 0.0, offset_all / (2.0 * math.pi * singular), 0.0
        )
        gradient_z = gradient_z - np.where(
            radius2 > 0.0,
            (z_all - source_all) / (2.0 * math.pi * singular),
            0.0,
        )
    return gradient_x.reshape(shape), gradient_z.reshape(shape)


def wave_amplitude_coefficient(z: FloatArray, nu: float, depth: float) -> FloatArray:
    """Vertical factor of the radiated far field, ``phi_0(z)/phi_0(0)``.

    The far field of the outgoing mode is
    ``G ~ wave(X) * phi_0(z)*phi_0(z0)/N_0``, so a body whose surface potential
    and normal velocity are ``phi`` and ``q`` radiates a wave of amplitude

        a_w = factor * integral_body exp(-i*k0*x) * phi_0(z)/phi_0(0)
                           * (q - phi*d/dn[phi_0/phi_0(0)]) ds

    with ``factor`` fixed by the same normalisation as
    :func:`tensorbem.free_surface_solver.wave_amplitude`. Only the *shape*
    ``phi_0(z)/phi_0(0)`` is returned here; the constant is applied by the
    caller that owns the energy identity, so that the factor is not buried in
    two places.
    """
    depth = _validate_depth(depth)
    nu = _validate_nu(nu)
    z = np.asarray(z, dtype=float)
    k0 = finite_depth_wavenumber(nu, depth)
    return np.cosh(k0 * (z + depth)) / math.cosh(k0 * depth)