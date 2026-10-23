"""Two-dimensional deep-water free-surface Green function.

This kernel is what makes a frequency-domain free-surface problem tractable
with panels on the body alone: the free-surface condition, the decay at
infinity and the radiation condition are all built into it, so there is no
truncated fluid domain, no sponge layer and no free-surface mesh.

Geometry and conventions
------------------------
``z`` points upwards, the mean free surface is ``z = 0`` and the fluid occupies
``z < 0``. The time factor is ``exp(-i*omega*t)``, so the linearised
free-surface condition on the potential is

    d(phi)/dz - nu*phi = 0   at z = 0,      nu = omega**2 / g.

The Green function satisfies

    -Laplacian(G) = delta(x - x0, z - z0)      in the fluid,
    dG/dz - nu*G = 0                          at z = 0,
    G -> outgoing waves                       as |x| -> infinity,

i.e. the same sign convention as ``kernels.fundamental_solution``.

Closed form
-----------
Fourier transforming in ``x`` and imposing the free-surface condition gives,
with ``A = -(z + z0) > 0`` (the sum of the two depths) and ``X = x - x0``,

    G = (1/2pi) * [ log(r) - log(r1)
                    - 2 * PV integral_0^inf exp(-k A) cos(k X)/(k - nu) dk ]
        - i * exp(-nu A) * cos(nu X)

with ``r`` the distance between the points and ``r1`` the distance to the image
of the source in the free surface. The last term is the residue at ``k = nu``
and carries the radiation condition; its sign is not a convention but a
physical statement (the wave carries energy away from the body). The sign was
fixed by requiring positive radiation damping and is re-derived from the
radiation problem in the tests, not trusted as a constant.

The two-dimensional kernel has *no* decay in its wave term: a line source
spreads over a line, so its far field has constant amplitude. That is a
property of the two-dimensional problem, not a numerical artifact, and it is
why the two-dimensional added mass is only defined together with a damping
coefficient.

Principal value integral
------------------------
The integral is evaluated without oscillatory quadrature. Splitting at
``k = nu`` and using ``PV integral_0^{2 nu} dk/(k - nu) = 0``,

    I = integral_0^{2 nu} [f(k) - f(nu)]/(k - nu) dk
        + integral_{2 nu}^inf f(k)/(k - nu) dk,      f(k) = exp(-k A) cos(k X).

The first piece is regular; with ``k = nu(1 + t)`` it is a smooth integral over
``[-1, 1]``. The second is regular and decaying; with ``k = nu(2 + t)`` and
``u = nu A t`` it is a Gauss-Laguerre rule. Node counts are checked in
``tests/test_free_surface.py`` against 1.5x and 2x the nodes.

The kernel is evaluated for every panel pair in a boundary-element assembly, so
the quadrature is vectorised and processed in blocks; ``_PAIR_BLOCK`` bounds the
temporary storage rather than the problem size.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_LAGUERRE_NODES = 160
_LEGENDRE_NODES = 80
_PAIR_BLOCK = 8192
# The residue sign that makes the radiation damping positive; see the module
# docstring.
_RESIDUE_SIGN = -1.0
_MIN_SCALED_DEPTH = 1.0e-6

_LEGENDRE_X, _LEGENDRE_W = leggauss(_LEGENDRE_NODES)
_LAGUERRE_X, _LAGUERRE_W = laggauss(_LAGUERRE_NODES)


def _validate_frequency(nu: float) -> float:
    if isinstance(nu, (bool, np.bool_)):
        raise ValueError("nu must be a positive finite number")
    value = float(nu)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("nu must be a positive finite number")
    return value


def _check_scaled_depth(nu: float, depth) -> None:
    if np.any(np.asarray(depth, dtype=float) * nu < _MIN_SCALED_DEPTH):
        raise ValueError(
            "nu * depth is below the supported range of the 2-D free-surface "
            "Green function; its zero-frequency limit diverges logarithmically"
        )


def _pv_cosine(nu: float, depth: FloatArray, offset: FloatArray) -> FloatArray:
    """``PV integral_0^inf exp(-k*depth) cos(k*offset)/(k - nu) dk``, vectorised."""
    flat_depth = np.ravel(np.asarray(depth, dtype=float))
    flat_offset = np.ravel(np.asarray(offset, dtype=float))
    flat_depth, flat_offset = np.broadcast_arrays(flat_depth, flat_offset)
    result = np.empty(flat_depth.size, dtype=float)
    for start in range(0, flat_depth.size, _PAIR_BLOCK):
        block = slice(start, start + _PAIR_BLOCK)
        d = flat_depth[block][:, None]
        x = flat_offset[block][:, None]
        reference = np.exp(-nu * d) * np.cos(nu * x)
        # [0, 2nu], with k = nu (1 + t) and a removable point at t = 0.
        k = nu * (1.0 + _LEGENDRE_X)[None, :]
        difference = np.exp(-k * d) * np.cos(k * x) - reference
        head = np.sum(_LEGENDRE_W[None, :] * difference / _LEGENDRE_X[None, :], axis=1)
        # [2nu, inf), with k = nu (2 + t) and u = nu*depth*t.
        sigma = nu * d
        t = _LAGUERRE_X[None, :] / sigma
        integrate = np.cos(nu * (2.0 + t) * x) / (1.0 + t)
        tail = (
            np.exp(-2.0 * sigma[:, 0])
            / sigma[:, 0]
            * np.sum(_LAGUERRE_W[None, :] * integrate, axis=1)
        )
        result[block] = head + tail
    return result.reshape(np.broadcast(np.asarray(depth), np.asarray(offset)).shape)


def _pv_sine(nu: float, depth: FloatArray, offset: FloatArray) -> FloatArray:
    """``PV integral_0^inf exp(-k*depth) sin(k*offset)/(k - nu) dk``, vectorised."""
    flat_depth = np.ravel(np.asarray(depth, dtype=float))
    flat_offset = np.ravel(np.asarray(offset, dtype=float))
    flat_depth, flat_offset = np.broadcast_arrays(flat_depth, flat_offset)
    result = np.empty(flat_depth.size, dtype=float)
    for start in range(0, flat_depth.size, _PAIR_BLOCK):
        block = slice(start, start + _PAIR_BLOCK)
        d = flat_depth[block][:, None]
        x = flat_offset[block][:, None]
        reference = np.exp(-nu * d) * np.sin(nu * x)
        k = nu * (1.0 + _LEGENDRE_X)[None, :]
        difference = np.exp(-k * d) * np.sin(k * x) - reference
        head = np.sum(_LEGENDRE_W[None, :] * difference / _LEGENDRE_X[None, :], axis=1)
        sigma = nu * d
        t = _LAGUERRE_X[None, :] / sigma
        integrate = np.sin(nu * (2.0 + t) * x) / (1.0 + t)
        tail = (
            np.exp(-2.0 * sigma[:, 0])
            / sigma[:, 0]
            * np.sum(_LAGUERRE_W[None, :] * integrate, axis=1)
        )
        result[block] = head + tail
    return result.reshape(np.broadcast(np.asarray(depth), np.asarray(offset)).shape)


def principal_value_integral(nu: float, depth: float, offset: float) -> float:
    """Scalar convenience wrapper around the vectorised cosine integral."""
    _check_scaled_depth(nu, depth)
    return float(_pv_cosine(nu, np.array([depth]), np.array([offset]))[0])


def _broadcast_inputs(x, z, source_x, source_z):
    field_x, field_z, s_x, s_z = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(source_x, dtype=float),
        np.asarray(source_z, dtype=float),
    )
    if not (np.isfinite(field_x).all() and np.isfinite(field_z).all()):
        raise ValueError("field coordinates must be finite")
    if np.any(field_z >= 0.0):
        raise ValueError("field points must lie strictly below the free surface")
    if np.any(s_z >= 0.0):
        raise ValueError("sources must lie strictly below the free surface")
    return field_x, field_z, s_x, s_z


def green_function(
    x,
    z,
    source_x,
    source_z,
    nu: float,
    *,
    outgoing: bool = True,
    include_rankine: bool = True,
) -> ComplexArray:
    """Free-surface Green function ``G(x, z; source_x, source_z)``.

    When ``include_rankine=False`` the direct ``log(r)/(2*pi)`` singularity is
    omitted so the smooth free-surface/image remainder can be integrated with
    standard Gauss rules even across a collocation point ``r = 0``.
    """
    frequency = _validate_frequency(nu)
    field_x, field_z, s_x, s_z = _broadcast_inputs(x, z, source_x, source_z)
    offset = field_x - s_x
    depth = -(field_z + s_z)
    _check_scaled_depth(frequency, depth)
    r_image = np.hypot(offset, field_z + s_z)
    integral = _pv_cosine(frequency, depth, offset)
    if include_rankine:
        r = np.hypot(offset, field_z - s_z)
        real_part = np.log(r) - np.log(r_image) - 2.0 * integral
    else:
        real_part = -np.log(r_image) - 2.0 * integral
    value = (1.0 / (2.0 * math.pi)) * real_part + (
        1.0j * _RESIDUE_SIGN * np.exp(-frequency * depth) * np.cos(frequency * offset)
    )
    return value if outgoing else np.conj(value)


def green_gradient(x, z, source_x, source_z, nu: float, *,
                   outgoing: bool = True):
    """Derivatives of ``G`` with respect to the *source* coordinates.

    The direct boundary integral equation needs source derivatives, because its
    double-layer kernel is taken along the source normal. They are analytic; no
    finite differences are involved. With ``P0`` and ``Q`` the cosine and sine
    principal value integrals, ``k/(k - nu) = 1 + nu/(k - nu)`` reduces the
    k-weighted derivatives to ``integral exp(-kA) cos(kX) = A/(A^2 + X^2)``,
    ``integral exp(-kA) sin(kX) = X/(A^2 + X^2)`` and ``P0``, ``Q``.
    """
    frequency = _validate_frequency(nu)
    field_x, field_z, s_x, s_z = _broadcast_inputs(x, z, source_x, source_z)
    offset = field_x - s_x
    depth = -(field_z + s_z)
    _check_scaled_depth(frequency, depth)
    r2 = offset * offset + (field_z - s_z) ** 2
    r_image2 = offset * offset + (field_z + s_z) ** 2

    cos_integral = _pv_cosine(frequency, depth, offset)
    sin_integral = _pv_sine(frequency, depth, offset)
    elementary_cos = depth / (depth * depth + offset * offset)
    elementary_sin = offset / (depth * depth + offset * offset)

    # Derivatives with respect to X and A.
    df_d_offset = offset * (1.0 / r2 - 1.0 / r_image2) + 2.0 * (
        elementary_sin + frequency * sin_integral
    )
    df_d_depth = -depth / r_image2 + 2.0 * (elementary_cos + frequency * cos_integral)
    residue = np.exp(-frequency * depth) * np.cos(frequency * offset)
    dresidue_d_offset = -frequency * np.exp(-frequency * depth) * np.sin(
        frequency * offset
    )
    dresidue_d_depth = -frequency * residue

    dg_d_offset = (1.0 / (2.0 * math.pi)) * df_d_offset + (
        1.0j * _RESIDUE_SIGN * dresidue_d_offset
    )
    dg_d_depth = (1.0 / (2.0 * math.pi)) * df_d_depth + (
        1.0j * _RESIDUE_SIGN * dresidue_d_depth
    )

    # Vertical source derivative: both r and r1 depend on source_z, as well as A.
    dg_d_source_x = -dg_d_offset
    dg_d_source_z = (1.0 / (2.0 * math.pi)) * (
        -(field_z - s_z) / r2 + depth / r_image2 - 2.0 * (
            elementary_cos + frequency * cos_integral
        )
    ) + 1.0j * _RESIDUE_SIGN * (frequency * residue)
    if not outgoing:
        dg_d_source_x = np.conj(dg_d_source_x)
        dg_d_source_z = np.conj(dg_d_source_z)
    return dg_d_source_x, dg_d_source_z


def green_gradient_field(
    x,
    z,
    source_x,
    source_z,
    nu: float,
    *,
    outgoing: bool = True,
    include_rankine: bool = True,
):
    """Derivatives of ``G`` with respect to the *field* coordinates.

    These are **not** minus the source derivatives. A free surface breaks
    vertical translation invariance: ``G`` depends on ``z`` and ``z0`` through
    both the difference ``z - z0`` and the depth sum ``-(z + z0)``, so only the
    horizontal translation invariance ``d/dx = -d/dx0`` survives. Using
    ``-grad_source`` where the field gradient is required produces a
    plausible-looking but wrong solution -- the method of fundamental solutions
    in this package was built on that mistake and disagreed with the panel
    method by 4% in added mass before the finite-difference check caught it.

    Setting ``include_rankine=False`` drops the singular ``grad_P(log r / 2pi)``
    part analytically instead of forming ``1/r**2`` and subtracting it later,
    which avoids ``0/0`` when a Gauss node coincides with a collocation centre.
    """
    frequency = _validate_frequency(nu)
    field_x, field_z, s_x, s_z = _broadcast_inputs(x, z, source_x, source_z)
    offset = field_x - s_x
    depth = -(field_z + s_z)
    _check_scaled_depth(frequency, depth)
    image2 = offset * offset + (field_z + s_z) ** 2

    cos_integral = _pv_cosine(frequency, depth, offset)
    sin_integral = _pv_sine(frequency, depth, offset)
    elementary_cos = depth / (depth * depth + offset * offset)
    elementary_sin = offset / (depth * depth + offset * offset)
    residue = np.exp(-frequency * depth) * np.cos(frequency * offset)

    if include_rankine:
        radial2 = offset * offset + (field_z - s_z) ** 2
        rankine_x = offset / radial2
        rankine_z = (field_z - s_z) / radial2
    else:
        rankine_x = 0.0
        rankine_z = 0.0

    dg_d_x = (1.0 / (2.0 * math.pi)) * (
        rankine_x
        - offset / image2
        + 2.0 * (elementary_sin + frequency * sin_integral)
    ) - 1.0j * _RESIDUE_SIGN * frequency * np.exp(-frequency * depth) * np.sin(
        frequency * offset
    )
    dg_d_z = (1.0 / (2.0 * math.pi)) * (
        rankine_z
        + depth / image2
        - 2.0 * (elementary_cos + frequency * cos_integral)
    ) + 1.0j * _RESIDUE_SIGN * frequency * residue
    if not outgoing:
        return np.conj(dg_d_x), np.conj(dg_d_z)
    return dg_d_x, dg_d_z
