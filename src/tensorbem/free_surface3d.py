"""Three-dimensional frequency-domain deep-water Kelvin/Havelock Green function.

Conventions
-----------
Coordinates have horizontal plane ``(x, y)`` and vertical axis ``z`` pointing
upward with the calm free surface at ``z = 0`` and fluid in ``z <= 0``.

To match the three-dimensional Laplace layer (:mod:`tensorbem.kernels3d`), the
singular Rankine kernel is normalized with a *positive* sign,

    G_0(x, xi) = 1 / (4 * pi * |x - xi|),      -nabla_x^2 G = delta(x - xi),

and physical fields carry the harmonic time factor ``Re{phi(x) * exp(-i*omega*t)}``
with deep-water wavenumber ``nu = omega**2 / g > 0``. Note that this Laplacian
sign is opposite to the 2-D convention ``+log(r)/(2*pi)`` in
:mod:`tensorbem.free_surface`, so the outgoing radiation residue carries ``+i``
here (giving ``H_0^{(1)}(nu*R) = J_0(nu*R) + i*Y_0(nu*R)`` at infinity) rather
than ``-i``.

Decomposition
-------------
Writing ``R = hypot(x - xi, y - eta)``, ``r = |x - xi|``, and
``r_1 = hypot(R, z + zeta)`` for the mirror image ``xi' = (xi, eta, -zeta)``,
the Kelvin Green function splits into direct Rankine, positive image Rankine,
and a pure wave part:

    G(x, xi; nu) = 1/(4*pi*r) + 1/(4*pi*r_1) + (nu / (2*pi)) * W(X, Y),

where ``X = nu * R >= 0``, ``Y = -nu * (z + zeta) >= 0``, and

    W(X, Y) = F(X, Y) + i * pi * exp(-Y) * J_0(X)       (outgoing=True),
    F(X, Y) = PV integral_0^inf exp(-u*Y) * J_0(u*X) / (u - 1) du.

``F`` satisfies the exact vertical differential identity
``dF/dY + F = -1 / hypot(X, Y)``, which guarantees the Kelvin free-surface
boundary condition ``(dG/dz - nu * G)|_{z=0} = 0`` algebraically when combined
with ``1/(4*pi*r) + 1/(4*pi*r_1)``.

Numerical evaluation of ``F(X, Y)`` and ``dF/dX``
-------------------------------------------------
Following Newman (1985), Telste & Noblesse (1986), and Delhommeau (1989), the
quarter-plane ``(X >= 0, Y >= 0)`` is partitioned into four non-overlapping
regions evaluated in memory-bounded blocks of ``4096`` points:

1. **Near-axis moderate-depth cone** (``0 < Y <= 4.5`` and ``X <= 0.35 * Y``):
   power series in ``-(X/2)**2`` with exact upward factorial recurrence from
   ``I_0(Y) = -exp(-Y) * Ei(Y)`` (:func:`tensorbem.analytic.exponential_integral`).
2. **Shallow / surface-piercing strip** (``Y <= 4.5`` outside the cone):
   Havelock's finite-interval ODE integral on ``[0, arsinh(Y / X)]`` combined
   with ``H_0(X) + Y_0(X)`` and ``H_1(X) + Y_1(X)`` via
   :func:`tensorbem.analytic.struve_h01` and
   :func:`tensorbem.analytic.bessel_j01_y01`.
3. **Intermediate depth** (``Y > 4.5`` and ``hypot(X, Y) < 26`` or ``X <= 0.25``):
   pole-subtracted 40-point Gauss-Legendre quadrature on ``[0, 2]`` plus a
   rescaled 40-point Gauss-Laguerre tail on ``[2, inf)``.
4. **Far field** (``Y > 4.5``, ``hypot(X, Y) >= 26``, and ``X > 0.25``):
   17-term Newman asymptotic Legendre expansion around ``-pi * exp(-Y) * Y_0(X)``.

Adjacent regions agree to ``1e-11 .. 1e-13`` across every seam without lookup
tables or external special-function libraries.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from .analytic import bessel_j01_y01, exponential_integral, struve_h01

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_FOUR_PI = 4.0 * math.pi
_TWO_PI = 2.0 * math.pi
_BLOCK_SIZE = 8192

_LEG20_X, _LEG20_W = leggauss(20)
_LEG40_X, _LEG40_W = leggauss(40)
_LAG40_U, _LAG40_W = laggauss(40)


def _validate_nu(nu: float) -> float:
    if isinstance(nu, (bool, np.bool_)):
        raise ValueError("wavenumber nu must be a positive finite real number")
    value = float(nu)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("wavenumber nu must be positive and finite")
    return value


def _eval_wave_block(
    x_flat: FloatArray, y_flat: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    count = x_flat.size
    f_out = np.empty(count, dtype=float)
    fx_out = np.empty(count, dtype=float)
    d_flat = np.hypot(x_flat, y_flat)

    cone = (y_flat > 0.0) & (y_flat <= 4.5) & (x_flat <= 0.35 * y_flat)
    if np.any(cone):
        xc = x_flat[cone]
        yc = y_flat[cone]
        inv_y = 1.0 / yc
        ik = -np.exp(-yc) * np.array(
            [exponential_integral(float(val)) for val in yc], dtype=float
        )
        sum_f = ik.copy()
        sum_fx = np.zeros_like(xc)
        x_half2 = 0.25 * xc * xc
        pow_x = np.ones_like(xc)
        inv_y_pow = np.ones_like(yc)
        fact = 1.0
        nz_x = xc > 0.0
        safe_x = np.where(nz_x, xc, 1.0)
        for n in range(1, 24):
            if n > 1:
                fact *= (2 * n - 3) * (2 * n - 2)
            inv_y_pow = inv_y_pow * inv_y
            ik = ik + fact * inv_y_pow
            fact_odd = fact * (2 * n - 1)
            inv_y_pow = inv_y_pow * inv_y
            ik = ik + fact_odd * inv_y_pow
            pow_x = pow_x * (-x_half2 / (n * n))
            sum_f = sum_f + pow_x * ik
            sum_fx = sum_fx + np.where(nz_x, (2.0 * n) * (pow_x / safe_x) * ik, 0.0)
        f_out[cone] = sum_f
        fx_out[cone] = sum_fx

    reg_a = (~cone) & (y_flat <= 4.5)
    if np.any(reg_a):
        xa = x_flat[reg_a]
        ya = y_flat[reg_a]
        da = d_flat[reg_a]
        _, _, y0_a, y1_a = bessel_j01_y01(xa)
        h0_a, h1_a = struve_h01(xa)
        exp_my = np.exp(-ya)
        upper = np.arcsinh(ya / xa)
        u_grid = 0.5 * upper[:, None] * (_LEG20_X[None, :] + 1.0)
        w_scale = 0.5 * upper
        sh = np.sinh(u_grid)
        integrand = np.exp(xa[:, None] * sh - ya[:, None])
        int0 = w_scale * (integrand @ _LEG20_W)
        int1 = w_scale * ((sh * integrand) @ _LEG20_W)
        f_out[reg_a] = -0.5 * math.pi * exp_my * (h0_a + y0_a) - int0
        fx_out[reg_a] = (
            exp_my * (-1.0 + 0.5 * math.pi * (h1_a + y1_a))
            + ya / (xa * da)
            - int1
        )

    reg_b = (y_flat > 4.5) & ((d_flat < 26.0) | (x_flat <= 0.25))
    if np.any(reg_b):
        xb = x_flat[reg_b]
        yb = y_flat[reg_b]
        db = d_flat[reg_b]

        u1 = 1.0 + _LEG40_X[None, :]
        exp_u1 = np.exp(-yb[:, None] * u1)
        exp_y = np.exp(-yb)
        ux1 = xb[:, None] * u1
        j0_u1 = np.ones_like(ux1)
        j1_u1 = np.zeros_like(ux1)
        nz1 = ux1 > 1.0e-15
        if np.any(nz1):
            j0_u1[nz1], j1_u1[nz1], _, _ = bessel_j01_y01(ux1[nz1])

        j0_x = np.ones_like(xb)
        j1_x = np.zeros_like(xb)
        nzx = xb > 1.0e-15
        if np.any(nzx):
            j0_x[nzx], j1_x[nzx], _, _ = bessel_j01_y01(xb[nzx])

        head0 = ((exp_u1 * j0_u1 - (exp_y * j0_x)[:, None]) / _LEG40_X[None, :]) @ _LEG40_W
        head1 = ((exp_u1 * j1_u1 - (exp_y * j1_x)[:, None]) / _LEG40_X[None, :]) @ _LEG40_W

        u2 = 2.0 + _LAG40_U[None, :] / yb[:, None]
        ux2 = xb[:, None] * u2
        j0_u2 = np.ones_like(ux2)
        j1_u2 = np.zeros_like(ux2)
        nz2 = ux2 > 1.0e-15
        if np.any(nz2):
            j0_u2[nz2], j1_u2[nz2], _, _ = bessel_j01_y01(ux2[nz2])
        denom2 = u2 - 1.0
        scale2 = np.exp(-2.0 * yb) / yb
        tail0 = scale2 * ((j0_u2 / denom2) @ _LAG40_W)
        tail1 = scale2 * ((j1_u2 / denom2) @ _LAG40_W)

        f_out[reg_b] = head0 + tail0
        fx_out[reg_b] = -xb / (db * (db + yb)) - (head1 + tail1)

    reg_c = (y_flat > 4.5) & (d_flat >= 26.0) & (x_flat > 0.25)
    if np.any(reg_c):
        xc = x_flat[reg_c]
        yc = y_flat[reg_c]
        dc = d_flat[reg_c]
        mu = yc / dc
        inv_d = 1.0 / dc
        inv_d3 = inv_d**3
        p_prev = np.zeros_like(mu)
        p_curr = np.ones_like(mu)
        dp_prev = np.zeros_like(mu)
        dp_curr = np.zeros_like(mu)
        fact = 1.0
        d_pow = inv_d.copy()
        sum_f = p_curr * d_pow
        sum_fx = -(xc * inv_d3) * p_curr
        for m in range(1, 17):
            fact *= m
            p_next = ((2 * m - 1) * mu * p_curr - (m - 1) * p_prev) / m
            dp_next = (2 * m - 1) * p_curr + dp_prev
            p_prev, p_curr = p_curr, p_next
            dp_prev, dp_curr = dp_curr, dp_next
            d_pow = d_pow * inv_d
            sum_f = sum_f + fact * p_curr * d_pow
            sum_fx = sum_fx - fact * xc * (d_pow * inv_d3) * (
                yc * dp_curr + (m + 1) * dc * p_curr
            )
        _, _, y0_c, y1_c = bessel_j01_y01(xc)
        exp_yc = np.exp(-yc)
        f_out[reg_c] = -math.pi * exp_yc * y0_c - sum_f
        fx_out[reg_c] = math.pi * exp_yc * y1_c - sum_fx

    fy_out = -f_out - 1.0 / d_flat
    return f_out, fx_out, fy_out


def wave_function_3d(
    x_dimless: ArrayLike, y_dimless: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate real dimensionless wave integral ``(F, dF/dX, dF/dY)``.

    Parameters
    ----------
    x_dimless, y_dimless:
        Broadcastable non-negative real arrays ``X = nu * R >= 0`` and
        ``Y = -nu * (z + zeta) >= 0`` with ``hypot(X, Y) > 0``.
    """
    x_arr, y_arr = np.broadcast_arrays(
        np.asarray(x_dimless, dtype=float),
        np.asarray(y_dimless, dtype=float),
    )
    if not (np.isfinite(x_arr).all() and np.isfinite(y_arr).all()):
        raise ValueError("x_dimless and y_dimless must be finite")
    if np.any(x_arr < 0.0) or np.any(y_arr < 0.0):
        raise ValueError("x_dimless and y_dimless must be non-negative")
    if np.any((x_arr == 0.0) & (y_arr == 0.0)):
        raise ValueError("wave_function_3d is singular at X = Y = 0")

    shape = x_arr.shape
    x_flat = x_arr.ravel()
    y_flat = y_arr.ravel()
    f_flat = np.empty_like(x_flat)
    fx_flat = np.empty_like(x_flat)
    fy_flat = np.empty_like(x_flat)
    for start in range(0, x_flat.size, _BLOCK_SIZE):
        stop = min(start + _BLOCK_SIZE, x_flat.size)
        f_flat[start:stop], fx_flat[start:stop], fy_flat[start:stop] = _eval_wave_block(
            x_flat[start:stop], y_flat[start:stop]
        )
    return (
        f_flat.reshape(shape),
        fx_flat.reshape(shape),
        fy_flat.reshape(shape),
    )


def _broadcast_points_3d(
    targets: ArrayLike, sources: ArrayLike
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
    if np.any(tgt_b[..., 2] > tol) or np.any(src_b[..., 2] > tol):
        raise ValueError("3-D free-surface points must lie in the fluid half-space z <= 0")
    tgt_clean = tgt_b.copy()
    src_clean = src_b.copy()
    tgt_clean[..., 2] = np.minimum(tgt_clean[..., 2], 0.0)
    src_clean[..., 2] = np.minimum(src_clean[..., 2], 0.0)
    return tgt_clean, src_clean, tgt_clean.shape[:-1]


def _wave_complex_and_partials(
    x_dim: FloatArray, y_dim: FloatArray, outgoing: bool
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    f_val, fx_val, fy_val = wave_function_3d(x_dim, y_dim)
    flat_x = x_dim.ravel()
    j0_flat = np.ones_like(flat_x)
    j1_flat = np.zeros_like(flat_x)
    nz = flat_x > 1.0e-15
    if np.any(nz):
        for start in range(0, flat_x.size, _BLOCK_SIZE):
            stop = min(start + _BLOCK_SIZE, flat_x.size)
            sub_nz = nz[start:stop]
            if np.any(sub_nz):
                j0_sub, j1_sub, _, _ = bessel_j01_y01(flat_x[start:stop][sub_nz])
                j0_flat[start:stop][sub_nz] = j0_sub
                j1_flat[start:stop][sub_nz] = j1_sub
    j0 = j0_flat.reshape(x_dim.shape)
    j1 = j1_flat.reshape(x_dim.shape)
    sgn = 1.0 if outgoing else -1.0
    exp_my = np.exp(-y_dim)
    res0 = (1.0j * sgn * math.pi) * exp_my * j0
    res1 = (1.0j * sgn * math.pi) * exp_my * j1
    w_val = f_val + res0
    wx_val = fx_val - res1
    wy_val = fy_val - res0
    return w_val, wx_val, wy_val


def green_function_3d(
    targets: ArrayLike,
    sources: ArrayLike,
    nu: float,
    *,
    outgoing: bool = True,
    include_rankine: bool = True,
    include_image: bool = True,
) -> ComplexArray:
    """Evaluate the 3-D deep-water Kelvin Green function ``G(x, xi; nu)``.

    Parameters
    ----------
    targets, sources:
        Broadcastable coordinate arrays with ``shape[-1] == 3`` and ``z <= 0``.
    nu:
        Deep-water wavenumber ``omega**2 / g > 0``.
    outgoing:
        Select outgoing ``H_0^{(1)}`` cylindrical waves under ``exp(-i*omega*t)``
        when ``True`` (default), or incoming conjugate waves when ``False``.
    include_rankine:
        Include the direct singular term ``1 / (4*pi*|x - xi|)``. Set ``False``
        when integrating the direct Rankine part analytically over panels.
    include_image:
        Include the mirror-image Rankine term ``+1 / (4*pi*r_1)``. Set ``False``
        when integrating the ``z = 0`` mirror panel analytically as well.
    """
    freq = _validate_nu(nu)
    tgt, src, lead_shape = _broadcast_points_3d(targets, sources)
    dx = tgt[..., 0] - src[..., 0]
    dy = tgt[..., 1] - src[..., 1]
    dz = tgt[..., 2] - src[..., 2]
    z_sum = tgt[..., 2] + src[..., 2]

    horiz = np.hypot(dx, dy)
    r_img = np.hypot(horiz, z_sum)
    if np.any(r_img <= 0.0):
        raise ValueError(
            "image distance r_1 = 0 (coincident source and target on z = 0 is singular)"
        )

    w_val, _, _ = _wave_complex_and_partials(freq * horiz, -freq * z_sum, outgoing)
    result = (freq / _TWO_PI) * w_val

    if include_image:
        result = result + 1.0 / (_FOUR_PI * r_img)
    if include_rankine:
        r_dir = np.hypot(horiz, dz)
        if np.any(r_dir <= 0.0):
            raise ValueError(
                "coincident source and target with include_rankine=True is singular"
            )
        result = result + 1.0 / (_FOUR_PI * r_dir)

    return np.asarray(result, dtype=np.complex128).reshape(lead_shape)


def green_gradient_source_3d(
    targets: ArrayLike,
    sources: ArrayLike,
    nu: float,
    *,
    outgoing: bool = True,
    include_rankine: bool = True,
    include_image: bool = True,
) -> ComplexArray:
    """Gradient ``nabla_xi G(x, xi; nu)`` with respect to *source* coordinates."""
    freq = _validate_nu(nu)
    tgt, src, lead_shape = _broadcast_points_3d(targets, sources)
    dx = tgt[..., 0] - src[..., 0]
    dy = tgt[..., 1] - src[..., 1]
    dz = tgt[..., 2] - src[..., 2]
    z_sum = tgt[..., 2] + src[..., 2]

    horiz = np.hypot(dx, dy)
    r_img = np.hypot(horiz, z_sum)
    if np.any(r_img <= 0.0):
        raise ValueError(
            "image distance r_1 = 0 (coincident source and target on z = 0 is singular)"
        )

    _, wx_val, wy_val = _wave_complex_and_partials(
        freq * horiz, -freq * z_sum, outgoing
    )
    nz_h = horiz > 1.0e-14
    safe_h = np.where(nz_h, horiz, 1.0)
    wx_over_r = np.where(nz_h, wx_val / safe_h, 0.0j)
    wave_scale = -(freq * freq / _TWO_PI)

    gx = wave_scale * dx * wx_over_r
    gy = wave_scale * dy * wx_over_r
    gz = wave_scale * wy_val

    if include_image:
        inv_r1_3 = 1.0 / (_FOUR_PI * (r_img**3))
        gx = gx + dx * inv_r1_3
        gy = gy + dy * inv_r1_3
        gz = gz - z_sum * inv_r1_3

    if include_rankine:
        r_dir = np.hypot(horiz, dz)
        if np.any(r_dir <= 0.0):
            raise ValueError(
                "coincident source and target with include_rankine=True is singular"
            )
        inv_r_3 = 1.0 / (_FOUR_PI * (r_dir**3))
        gx = gx + dx * inv_r_3
        gy = gy + dy * inv_r_3
        gz = gz + dz * inv_r_3

    return np.stack((gx, gy, gz), axis=-1).reshape((*lead_shape, 3))


def pure_wave_value_and_source_gradient_3d(
    targets: ArrayLike,
    sources: ArrayLike,
    nu: float,
    *,
    outgoing: bool = True,
) -> tuple[ComplexArray, ComplexArray]:
    """Fused evaluation of ``(G_W, nabla_xi G_W)`` without Rankine or image terms."""
    freq = _validate_nu(nu)
    tgt, src, lead_shape = _broadcast_points_3d(targets, sources)
    dx = tgt[..., 0] - src[..., 0]
    dy = tgt[..., 1] - src[..., 1]
    z_sum = tgt[..., 2] + src[..., 2]
    horiz = np.hypot(dx, dy)
    r_img = np.hypot(horiz, z_sum)
    if np.any(r_img <= 0.0):
        raise ValueError("image distance r_1 = 0 is singular")

    w_val, wx_val, wy_val = _wave_complex_and_partials(
        freq * horiz, -freq * z_sum, outgoing
    )
    gw = (freq / _TWO_PI) * w_val
    nz_h = horiz > 1.0e-14
    safe_h = np.where(nz_h, horiz, 1.0)
    wx_over_r = np.where(nz_h, wx_val / safe_h, 0.0j)
    wave_scale = -(freq * freq / _TWO_PI)
    gx = wave_scale * dx * wx_over_r
    gy = wave_scale * dy * wx_over_r
    gz = wave_scale * wy_val
    return (
        np.asarray(gw, dtype=np.complex128).reshape(lead_shape),
        np.stack((gx, gy, gz), axis=-1).reshape((*lead_shape, 3)),
    )


def green_gradient_field_3d(
    targets: ArrayLike,
    sources: ArrayLike,
    nu: float,
    *,
    outgoing: bool = True,
    include_rankine: bool = True,
    include_image: bool = True,
) -> ComplexArray:
    """Gradient ``nabla_x G(x, xi; nu)`` with respect to *field* coordinates.

    Because the free surface breaks vertical translation invariance while
    preserving horizontal invariance, ``dG/dx = -dG/dxi`` and ``dG/dy = -dG/deta``,
    whereas ``dG/dz != -dG/dzeta``: the ``z + zeta`` dependence of both ``r_1``
    and ``Y = -nu*(z + zeta)`` enters with the *same* sign in ``d/dz`` and
    ``d/dzeta``.
    """
    freq = _validate_nu(nu)
    tgt, src, lead_shape = _broadcast_points_3d(targets, sources)
    dx = tgt[..., 0] - src[..., 0]
    dy = tgt[..., 1] - src[..., 1]
    dz = tgt[..., 2] - src[..., 2]
    z_sum = tgt[..., 2] + src[..., 2]

    horiz = np.hypot(dx, dy)
    r_img = np.hypot(horiz, z_sum)
    if np.any(r_img <= 0.0):
        raise ValueError(
            "image distance r_1 = 0 (coincident source and target on z = 0 is singular)"
        )

    _, wx_val, wy_val = _wave_complex_and_partials(
        freq * horiz, -freq * z_sum, outgoing
    )
    nz_h = horiz > 1.0e-14
    safe_h = np.where(nz_h, horiz, 1.0)
    wx_over_r = np.where(nz_h, wx_val / safe_h, 0.0j)
    wave_scale = freq * freq / _TWO_PI

    gx = wave_scale * dx * wx_over_r
    gy = wave_scale * dy * wx_over_r
    gz = -wave_scale * wy_val

    if include_image:
        inv_r1_3 = 1.0 / (_FOUR_PI * (r_img**3))
        gx = gx - dx * inv_r1_3
        gy = gy - dy * inv_r1_3
        gz = gz - z_sum * inv_r1_3

    if include_rankine:
        r_dir = np.hypot(horiz, dz)
        if np.any(r_dir <= 0.0):
            raise ValueError(
                "coincident source and target with include_rankine=True is singular"
            )
        inv_r_3 = 1.0 / (_FOUR_PI * (r_dir**3))
        gx = gx - dx * inv_r_3
        gy = gy - dy * inv_r_3
        gz = gz - dz * inv_r_3

    return np.stack((gx, gy, gz), axis=-1).reshape((*lead_shape, 3))


def verify_free_surface_bc_3d(
    surface_xy: ArrayLike, source: ArrayLike, nu: float
) -> ComplexArray:
    """Return residual ``dG/dz - nu * G`` at points ``(x, y, 0)`` on ``z = 0``."""
    xy = np.asarray(surface_xy, dtype=float)
    if xy.ndim < 1 or xy.shape[-1] != 2:
        raise ValueError("surface_xy must have last dimension 2")
    zeros = np.zeros((*xy.shape[:-1], 1), dtype=float)
    targets = np.concatenate((xy, zeros), axis=-1)
    src = np.asarray(source, dtype=float)
    if src.shape != (3,) or src[2] >= 0.0:
        raise ValueError("source must have shape (3,) with source[2] < 0")
    val = green_function_3d(targets, src, nu)
    grad = green_gradient_field_3d(targets, src, nu)
    return grad[..., 2] - float(nu) * val


def verify_laplace_3d(
    target: ArrayLike, source: ArrayLike, nu: float, *, step: float = 1.0e-4
) -> complex:
    """Second-order central-difference residual of ``nabla_x^2 G`` away from ``source``."""
    tgt = np.asarray(target, dtype=float)
    src = np.asarray(source, dtype=float)
    if tgt.shape != (3,) or src.shape != (3,):
        raise ValueError("target and source must have shape (3,)")
    if tgt[2] + step > 0.0:
        raise ValueError("target[2] + step must stay <= 0")
    center = complex(green_function_3d(tgt, src, nu))
    laplacian = 0.0j
    for axis in range(3):
        shift = np.zeros(3, dtype=float)
        shift[axis] = step
        plus = complex(green_function_3d(tgt + shift, src, nu))
        minus = complex(green_function_3d(tgt - shift, src, nu))
        laplacian += (plus - 2.0 * center + minus) / (step * step)
    return laplacian


def verify_radiation_condition_3d(
    radii: ArrayLike,
    azimuth: float,
    depth_z: float,
    source: ArrayLike,
    nu: float,
    *,
    outgoing: bool = True,
) -> FloatArray:
    """Return scaled Sommerfeld residuals ``sqrt(R) * |dG/dR -/+ i*nu*G|``."""
    r_arr = np.asarray(radii, dtype=float)
    if r_arr.ndim != 1 or np.any(r_arr <= 0.0):
        raise ValueError("radii must be a 1-D positive array")
    if depth_z > 0.0:
        raise ValueError("depth_z must be <= 0")
    cos_a = math.cos(float(azimuth))
    sin_a = math.sin(float(azimuth))
    targets = np.column_stack(
        (r_arr * cos_a, r_arr * sin_a, np.full_like(r_arr, float(depth_z)))
    )
    val = green_function_3d(targets, source, nu, outgoing=outgoing)
    grad = green_gradient_field_3d(targets, source, nu, outgoing=outgoing)
    dg_dr = grad[:, 0] * cos_a + grad[:, 1] * sin_a
    sgn = 1.0 if outgoing else -1.0
    return np.sqrt(r_arr) * np.abs(dg_dr - 1.0j * sgn * float(nu) * val)
