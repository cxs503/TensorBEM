"""Three-dimensional frequency-domain finite-depth free-surface Green function.

Provenance (merge of ``agent/m2930`` into ``main``)
---------------------------------------------------
This module is the *quadrature* (kernel-subtraction wave-integral) formulation of
the 3-D finite-depth kernel, authored on ``agent/m2930`` from upstream ``b408d0d``
plus the m2930 panel-width fix in ``_wave_integral`` (``bump/6.0`` and
``0.1*k0`` instead of ``bump/3.0`` and ``0.25*k0``).  Independently, ``main``
evolved the *eigenseries* (John 1950) formulation of the same kernel, which is
the package kernel and lives in :mod:`tensorbem.finite_depth3d`.  The two lines
therefore collided as an add/add conflict on the module path; instead of
dropping either side, the eigenseries module keeps the canonical path and this
file is retained verbatim (only this header added) so that the m2930 diagnosis
in ``docs/finite_depth3d_offaxis_diagnosis.md`` stays reproducible against the
exact code it was written for.  It is **not** imported by :mod:`tensorbem` and it
keeps its original UNVERIFIED banner below.

STATUS: UNVERIFIED -- do not use in a solver.  The free-surface condition,
bottom condition, reciprocity and grid independence hold at roundoff
(``3.6e-14``, ``<1e-6``, ``3e-16``, ``1.2e-11``), but the independent
deep-water cross-check against :func:`tensorbem.free_surface3d.green_function_3d`
fails for off-axis field points (relative differences 0.34 / 0.56 / 3.2, flat
in depth) while the on-axis point converges to ``1.9e-6`` at ``nu*h = 36``.
See ``docs/finite_depth3d_green_function.md`` §3.3 and §6 for the measured
numbers and the next step.

This is the finite-depth counterpart of :mod:`tensorbem.free_surface3d` (deep
water) and the three-dimensional counterpart of :mod:`tensorbem.finite_depth`
(2-D strip).  It supplies the kernel that roadmap item 6 of ``README.md`` lists
as missing ("finite-depth 3-D John eigenseries Green function") without
assembling a hull solver around it.

Conventions
-----------
Identical to :mod:`tensorbem.free_surface3d`: ``(x, y)`` horizontal, ``z``
upward, calm free surface at ``z = 0``, impermeable bottom at ``z = -h``, fluid
in ``-h <= z <= 0``, time factor ``exp(-i*omega*t)``, and the finite-depth
wavenumber ``k0`` the positive root of ``k0*tanh(k0*h) = nu`` with
``nu = omega**2 / g``.  The singular Rankine kernel is normalised with a
positive sign,

    G_0(x, xi) = 1 / (4*pi*|x - xi|),      -nabla_x^2 G = delta(x - xi),

so the outgoing-wave residue carries ``+i`` (giving ``H_0^{(1)}`` at infinity),
exactly as in the deep-water module.

Derivation of the representation
--------------------------------
Write ``R = hypot(x - xi, y - eta)``, ``r = |x - xi|`` and, with the source at
depth ``zeta`` (``-h < zeta < 0``), the image of the source in the bottom at
``zeta_b = -2*h - zeta``.  The pair

    s(x, xi) = 1 / r + 1 / |x - (xi, eta, zeta_b)|

satisfies the bottom condition ``d(s)/dz = 0`` at ``z = -h`` identically and
carries the source singularity, so only the free-surface condition has to be
repaired.  Using ``1/r = integral_0^inf exp(-k|z-zeta|) J_0(k R) dk`` on both
members of the pair, its free-surface residual is

    (d/dz - nu) s |_{z=0}
        = -integral_0^inf N(k) J_0(k*R) dk,
    N(k) = (k+nu) * [ exp(k*zeta) + exp(-k*(2h+zeta)) ],

because the source and its bottom image both sit below ``z = 0``, so both
exponentials carry the same ``(k + nu)`` factor.  Any function built from the
vertical eigenfunctions ``cosh(k*(z+h))`` satisfies the bottom condition, so
adding

    (1/(4*pi)) * integral_0^inf N(k) cosh(k*(z+h)) J_0(k*R)
                          / (k*sinh(k*h) - nu*cosh(k*h)) dk

cancels that residual analytically, since
``(d/dz - nu) cosh(k*(z+h)) |_{z=0} = k*sinh(k*h) - nu*cosh(k*h)``.  Hence

    G = (1/(4*pi)) * [ 1/r + 1/r_b ]  +  (1/(4*pi)) * integral_0^inf ...

The dispersion relation ``k*tanh(k*h) = nu`` gives the denominator one positive
real zero ``k0`` (the propagating radial wave) plus poles on the imaginary axis
(evanescent modes, never on the integration path).  With
``D(k) = (k*sinh(k*h) - nu*cosh(k*h)) / (k - k0)`` smooth and non-zero at
``k0``, the *outgoing* solution replaces the principal value by

    PV integral + i*pi*q(k0),    q(k) = N(k) cosh(k*(z+h)) J_0(k*R) / D(k),

(``-i*pi`` for the incoming solution), i.e. the pole is displaced below the
real axis.  Note ``(d/dz - nu)`` annihilates the residue term at ``z = 0``
(``k0*sinh(k0*h) - nu*cosh(k0*h) = 0``), so the construction satisfies the
free-surface condition exactly and not merely approximately.

In the deep-water limit ``h -> inf`` the representation collapses to
``1/(4*pi*r)`` plus the classical
``(1/(4*pi)) * integral (k+nu)/(k-nu) * exp(k*(z+zeta)) J_0(k*R) dk``; that
limit is checked against the independently implemented deep-water kernel
:func:`tensorbem.free_surface3d.green_function_3d` in
``tests/test_finite_depth3d.py``, and it is what fixes the overall sign and the
``i*pi`` convention used here.

Numerical evaluation
--------------------
* The integrand decays like ``exp(k*(z+zeta))`` for large ``k`` (the bottom
  image term decays like ``exp(k*(z-2h-zeta))``), so the integral is truncated
  where ``exp(-kmax * alpha) < 1e-17``, with ``alpha = -(z + zeta)`` the
  combined depth of field point and source below the free surface.  A source
  sitting just below ``z = 0`` therefore needs a large ``kmax``; that is the
  one knob the caller may override.
* Vertical factors are stored in the scaled form
  ``cosh(k*(z+h)) * exp(-k*h) = (exp(k*z) + exp(-k*(z+2h)))/2`` and the
  denominator in the same scaling, so no ``cosh(k*h)`` overflow occurs for
  deep water (``nu*h`` up to a few hundred is safe).
* ``[0, kmax]`` is cut into panels narrow enough for the ``J_0(k*R)``
  oscillation and for the smooth factor, each integrated with a
  ``nodes``-point Gauss-Legendre rule.  The single panel that contains ``k0``
  is replaced by its exact principal value
  ``q(k0) * log|(b-k0)/(k0-a)| + integral (q(k)-q(k0))/(k-k0) dk``, which
  removes the cancellation and leaves no large-``kmax`` logarithm behind.
* ``derivative=1`` returns ``dG/dz`` by the same machinery with the vertical
  factor replaced by ``k*sinh(k*(z+h))``.

Scope
-----
Value and vertical derivative of the Green function only.  Horizontal
derivatives, the double-layer kernel for waterline-piercing bodies, the
3-D finite-depth diffraction/radiation solver, multi-body gap resonance and the
John eigenseries form are *not* part of this module (see
``docs/finite_depth3d_green_function.md`` §6).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from .analytic import bessel_j01_y01
from .finite_depth import _validate_depth, _validate_nu, finite_depth_wavenumber

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_FOUR_PI = 4.0 * math.pi
#: Truncation: ``exp(-kmax * alpha)`` below this is treated as zero.
_TRUNCATION = 1.0e-17
_DEFAULT_NODES = 16
_MAX_PANELS = 40000
#: Nodes closer than this (relative) to the pole are unused (measure zero).
_POLE_GUARD = 1.0e-12


@lru_cache(maxsize=32)
def _gauss(nodes: int) -> tuple[FloatArray, FloatArray]:
    return leggauss(int(nodes))


def dispersion_derivative(nu: float, depth: float, k0: float) -> float:
    """``d/dk [k*sinh(k*h) - nu*cosh(k*h)]`` at the finite-depth root ``k0``."""
    kh = k0 * depth
    return math.sinh(kh) + kh * math.cosh(kh) - nu * depth * math.sinh(kh)


def pole_data(nu: float, depth: float) -> tuple[float, float]:
    """Return ``(k0, D)`` -- propagating wavenumber and its denominator slope."""
    k0 = finite_depth_wavenumber(nu, depth)
    return k0, dispersion_derivative(nu, depth, k0)


def _scaled_pole(nu: float, depth: float) -> tuple[float, float]:
    """``(k0, D*exp(-k0*h))``: the slope in the exponentially scaled form."""
    k0, slope = pole_data(nu, depth)
    return k0, slope * math.exp(-k0 * depth)


def _panel_grid(k0: float, k_max: float, alpha: float, r_max: float):
    """Uniform grid on ``[0, k_max]`` plus the index of the pole panel."""
    oscillation = 2.0 * math.pi / max(r_max, 1.0e-12) / 6.0
    smoothness = min(0.25 * max(k0, 1.0e-12), 0.5 * alpha)
    width = max(min(oscillation, smoothness), 1.0e-12)
    count = max(1, min(int(math.ceil(k_max / width)), _MAX_PANELS))
    edges = np.linspace(0.0, k_max, count + 1)
    pole_panel = int(np.searchsorted(edges, k0, side="right") - 1)
    pole_panel = min(max(pole_panel, 0), count - 1)
    return edges, pole_panel


def _bessel_j0(values: FloatArray) -> FloatArray:
    out = np.ones_like(values)
    positive = values > 0.0
    if np.any(positive):
        j0, _, _, _ = bessel_j01_y01(values[positive])
        out[positive] = j0
    return out


def _q_values(
    k: FloatArray,
    *,
    nu: float,
    depth: float,
    zeta: float,
    z: FloatArray,
    radius: FloatArray,
    k0: float,
    slope_scaled: float,
    order: int,
) -> FloatArray:
    """Scaled ``q(k) = integrand * (k - k0)`` on ``k`` for every target.

    Shape ``(k.size, z.size)``.  The scaling is by ``exp(-k*h)`` in both the
    numerator's vertical factor and the denominator, which is an identity but
    keeps ``nu*h`` of a few hundred free of overflow.
    """
    kk = k[:, None]
    kh = kk * depth
    exp_m2kh = np.exp(-2.0 * kh)
    denom = 0.5 * (kk * (1.0 - exp_m2kh) - nu * (1.0 + exp_m2kh))
    offset = kk - k0
    safe = np.where(offset == 0.0, 1.0, offset)
    smooth = np.where(offset == 0.0, slope_scaled, denom / safe)
    numerator = (kk + nu) * (
        np.exp(kk * zeta) + np.exp(-kk * (2.0 * depth + zeta))
    )
    z_plus_h = z[None, :] + depth
    if order == 0:
        vertical = 0.5 * (np.exp(kk * z) + np.exp(-kk * (z[None, :] + 2.0 * depth)))
    elif order == 1:
        vertical = 0.5 * kk * (
            np.exp(kk * z) - np.exp(-kk * (z[None, :] + 2.0 * depth))
        )
    else:  # pragma: no cover - guarded by the wrappers
        raise ValueError("order must be 0 (value) or 1 (d/dz)")
    del z_plus_h
    x = kk * radius[None, :]
    j0 = _bessel_j0(x.ravel()).reshape(x.shape)
    return numerator * vertical * j0 / smooth


def _wave_integral(
    *,
    nu: float,
    depth: float,
    zeta: float,
    z: FloatArray,
    radius: FloatArray,
    order: int,
    outgoing: bool,
    nodes: int,
    k_max: float | None,
) -> ComplexArray:
    """Outgoing radial-wave part of ``G`` and its ``z`` derivative.

    The principal value is computed by *kernel subtraction*: with a Gaussian
    bump ``chi(k) = exp(-((k-k0)/w)**2)``,

        PV integral q/(k-k0) dk = integral (q - q(k0)*chi)/(k-k0) dk
                                  + q(k0) * PV integral chi/(k-k0) dk,

    and the second principal value vanishes to ``exp(-(k0/w)**2)`` (odd
    integrand, effective limits symmetric about ``k0``), i.e. for ``w <= 0.1*k0``
    it is below roundoff.  The first integrand is analytic at ``k0`` because its
    numerator vanishes there, so one uniform panel grid with a Gaussian rule is
    enough -- no pole panel, no logarithm, and the result is independent of the
    panel grid (``tests/test_finite_depth3d.py`` sweeps it).
    """
    k0, slope_scaled = _scaled_pole(nu, depth)
    alpha = float(np.min(-(z + zeta)))
    if alpha <= 0.0:
        raise ValueError(
            "finite-depth Green function requires field point and source strictly "
            "below the free surface (z + zeta < 0)"
        )
    if k_max is None:
        k_max = min(-math.log(_TRUNCATION) / alpha, 1.0e8)
    k_max = max(float(k_max), 4.0 * k0)

    leg_x, leg_w = _gauss(nodes)
    bump = 0.1 * k0

    def q(k_values):
        return _q_values(
            k_values,
            nu=nu,
            depth=depth,
            zeta=zeta,
            z=z,
            radius=radius,
            k0=k0,
            slope_scaled=slope_scaled,
            order=order,
        )

    oscillation = 2.0 * math.pi / max(float(np.max(radius)), 1.0e-12) / 6.0
    width = max(min(oscillation, 0.1 * k0, bump / 6.0), 1.0e-12)
    count = max(1, min(int(math.ceil(k_max / width)), _MAX_PANELS))
    edges = np.linspace(0.0, k_max, count + 1)
    left, right = edges[:-1], edges[1:]
    half = 0.5 * (right - left)
    mid = 0.5 * (right + left)
    grid = mid[:, None] + half[:, None] * leg_x[None, :]
    weights = (half[:, None] * leg_w[None, :]).reshape(*grid.shape)

    q_pole = q(np.array([k0]))[0]
    offset = grid[:, :, None] - k0
    smooth = (q(grid.ravel()).reshape(*grid.shape, z.size)
              - q_pole[None, None, :] * np.exp(-(offset[..., 0] / bump) ** 2)[..., None])
    near = np.abs(offset[..., 0]) <= 1.0e-12 * max(1.0, k0)
    integrand = np.where(
        near[..., None],
        np.zeros_like(smooth),
        smooth / np.where(near, 1.0, offset[..., 0])[..., None],
    )
    total = np.einsum("pn,pnt->t", weights, integrand).astype(complex)

    residue = (1.0 if outgoing else -1.0) * 1.0j * math.pi * q_pole
    return total + residue


def _broadcast_points(
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
    return tgt_b, src_b, tgt_b.shape[:-1]


def _green(
    targets: ArrayLike,
    sources: ArrayLike,
    nu: float,
    depth: float,
    *,
    order: int,
    outgoing: bool,
    nodes: int,
    k_max: float | None,
) -> ComplexArray:
    nu = _validate_nu(nu)
    depth = _validate_depth(depth)
    nodes = int(nodes)
    if nodes < 4:
        raise ValueError("nodes must be at least 4")
    tgt, src, shape = _broadcast_points(targets, sources)
    tol = 1.0e-13
    if np.any(tgt[..., 2] > tol) or np.any(src[..., 2] > tol):
        raise ValueError("finite-depth points must lie in the fluid layer z <= 0")
    if np.any(tgt[..., 2] < -depth - tol) or np.any(src[..., 2] < -depth - tol):
        raise ValueError("finite-depth points must lie above the bottom z >= -depth")
    tgt = np.minimum(tgt.copy(), 0.0)
    src = np.minimum(src.copy(), 0.0)

    dx = tgt[..., 0] - src[..., 0]
    dy = tgt[..., 1] - src[..., 1]
    dz = tgt[..., 2] - src[..., 2]
    dz_image = tgt[..., 2] + 2.0 * depth + src[..., 2]
    radius = np.hypot(dx, dy)
    r = np.hypot(radius, np.abs(dz))
    r_image = np.hypot(radius, np.abs(dz_image))
    if np.any(r <= 1.0e-14):
        raise ValueError("finite-depth Green function is singular at the source")
    if order == 0:
        rankine = 1.0 / r + 1.0 / r_image
    else:
        rankine = -dz / r**3 - dz_image / r_image**3

    z_flat = tgt[..., 2].ravel()
    src_flat = src.reshape(-1, 3)
    out = np.empty(z_flat.size, dtype=complex)
    # The wave integral depends on the source only through its depth zeta, so one
    # quadrature grid serves every elementwise pair that shares that depth.
    for zeta in np.unique(np.minimum(src_flat[:, 2], 0.0)):
        index = np.flatnonzero(
            np.isclose(src_flat[:, 2], zeta, rtol=0.0, atol=1.0e-15)
        )
        out[index] = _wave_integral(
            nu=nu,
            depth=depth,
            zeta=float(zeta),
            z=z_flat[index],
            radius=radius.ravel()[index],
            order=order,
            outgoing=outgoing,
            nodes=nodes,
            k_max=k_max,
        )
    return ((rankine.ravel() + out) / _FOUR_PI).reshape(shape)


def green_function_finite_depth_3d(
    targets: ArrayLike,
    sources: ArrayLike,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    nodes: int = _DEFAULT_NODES,
    k_max: float | None = None,
) -> ComplexArray:
    """Finite-depth free-surface Green function ``G(x, xi)`` (broadcasting).

    Parameters
    ----------
    targets, sources:
        Broadcastable ``(..., 3)`` arrays of field points and source points,
        both inside ``-depth <= z <= 0``.  The result has the broadcast shape.
    nu:
        ``omega**2 / g > 0`` [1/m].
    depth:
        Water depth ``h > 0`` [m].
    outgoing:
        ``True`` selects outgoing waves (``+i*pi`` residue, ``exp(-i*omega*t)``
        convention); ``False`` gives the incoming (adjoint) kernel.
    nodes:
        Gauss-Legendre nodes per panel (default 16).
    k_max:
        Optional truncation of the wavenumber integral; by default it is set
        where ``exp(-k_max * alpha) = 1e-17`` with ``alpha = -(z + zeta)``.

    Notes
    -----
    Sources and field points must be strictly below the free surface
    (``z + zeta < 0``); the wavenumber integral is not truncated for a source
    sitting on ``z = 0``.
    """
    return _green(
        targets,
        sources,
        nu,
        depth,
        order=0,
        outgoing=outgoing,
        nodes=nodes,
        k_max=k_max,
    )


def green_function_finite_depth_z_derivative_3d(
    targets: ArrayLike,
    sources: ArrayLike,
    nu: float,
    depth: float,
    *,
    outgoing: bool = True,
    nodes: int = _DEFAULT_NODES,
    k_max: float | None = None,
) -> ComplexArray:
    """``dG/dz`` of :func:`green_function_finite_depth_3d` at the field point."""
    return _green(
        targets,
        sources,
        nu,
        depth,
        order=1,
        outgoing=outgoing,
        nodes=nodes,
        k_max=k_max,
    )


def wave_amplitude_coefficient(
    z: ArrayLike, zeta: float, nu: float, depth: float
) -> ComplexArray:
    """Complex amplitude ``A`` of the outgoing radial wave at depth ``z``.

    The propagating part of the Green function is ``A * H_0^{(1)}(k0*R)`` with

        A(z, zeta) = i*pi * N(k0) * cosh(k0*(z+h))
                        / (4*pi * (k*sinh(k*h) - nu*cosh(k*h))'(k0)),

    so ``|A|`` is the coefficient multiplying the radiated cylindrical wave.
    Its ``h -> inf`` limit is ``(i*k0/2) * exp(k0*(z+zeta))``; the height/depth
    study in ``docs/finite_depth3d_green_function.md`` compares against it.
    """
    nu = _validate_nu(nu)
    depth = _validate_depth(depth)
    zeta = float(zeta)
    if not (-depth < zeta < 0.0):
        raise ValueError("source depth must satisfy -depth < zeta < 0")
    z_arr = np.asarray(z, dtype=float)
    if np.any(z_arr > 0.0) or np.any(z_arr < -depth):
        raise ValueError("field depth must lie in [-depth, 0]")
    k0, slope_scaled = _scaled_pole(nu, depth)
    numerator = (k0 + nu) * (
        math.exp(k0 * zeta) + math.exp(-k0 * (2.0 * depth + zeta))
    )
    vertical = 0.5 * (
        np.exp(k0 * z_arr) + np.exp(-k0 * (z_arr + 2.0 * depth))
    )
    return 1.0j * math.pi * numerator * vertical / (_FOUR_PI * slope_scaled)


def deep_water_amplitude_coefficient(
    z: ArrayLike, zeta: float, nu: float
) -> ComplexArray:
    """Deep-water limit ``(i*k0/2) * exp(k0*(z+zeta))`` with ``k0 = nu``."""
    nu = _validate_nu(nu)
    return 0.5j * nu * np.exp(nu * (np.asarray(z, dtype=float) + float(zeta)))