"""Standard-form Michell thin-ship wave resistance with a panel discretisation.

Why this module exists
----------------------
:mod:`tensorbem.wigley` ships a Kochin/Michell path for the mathematical Wigley
hull, but it is a *single-hull, closed-form* path: the offsets and their
streamwise derivatives are analytic, nothing is discretised, and no second
geometry (spheroid, tandem pair) can be evaluated.  There was therefore no way
to ask the question a verification anchor asks -- *does a panel discretisation
of the Rankine centreplane source sheet reproduce the theory it approximates,
and at what rate?*

This module supplies three independent pieces:

``EllipsoidHull`` / ``WigleyHull``
    Centreplane descriptors: half breadth ``y(x, z)`` and ``y_x`` for a
    surface-piercing spheroid and for the parabolic Wigley form.

``ellipsoid_amplitude_reference`` / ``wigley_amplitude_reference``
    The Michell/Kochin wave amplitude ``A`` by *smooth* quadrature: closed
    forms for the Wigley, and a singularity-free ``(psi, phi)``
    parameterisation of the spheroid in which the integrable
    ``1/sqrt(1 - (x/a)^2 - (z/c)^2)`` factor of ``y_x`` cancels the Jacobian
    exactly.

``CenterplanePanels``
    The same amplitude from a *panel method*: flat panels on the centreplane,
    source strength ``y_x`` sampled at 2x2 Gauss points inside every panel, a
    bilinear panel map.  This is the Rankine-source discretisation a thin-ship
    panel code assembles; its error is a discretisation error and it vanishes
    as the panels are refined.

Theory and conventions
----------------------
Steady, deep water, Neumann-Kelvin linearisation, ship at speed ``U`` in
``+x``, calm water at ``z = 0``, no sinkage or trim (thin-ship).  With
``k_0 = g/U^2`` the free-surface condition fixes, for a wave travelling at
angle ``theta`` to the ``x`` axis, the wavenumber ``k = k_0 sec^2 theta`` and
the streamwise phase wavenumber ``alpha = k cos(theta) = k_0 sec(theta)``.
The Kochin function of the centreplane source sheet is

    A(theta) = int int y_x(x, z) exp(k z + i alpha x) dx dz

and Michell's far-field result is

    R_w = (4 rho g^2 / (pi U^2)) int_0^{pi/2} sec^3(theta) |A|^2 dtheta
        = (4 rho g^2 / (pi U^2)) int_1^inf |A(lambda)|^2 lambda^2 dlambda
          / sqrt(lambda^2 - 1),

the second form following from ``lambda = sec(theta)``, with ``k = k_0
lambda^2`` and ``alpha = k_0 lambda``.  Both forms are implemented
(:func:`wave_resistance_from_amplitude`, :func:`wave_resistance_theta_form`)
and a test pins them to each other; they share no quadrature rule, only the
physics.

Two numerical details are worth recording because both are easy to get wrong
and both are quantified in ``benchmarks/verified/bem_ellipsoid_wave``:

* the substitution ``lambda = 1 + s^2`` removes the integrable cusp
  ``(lambda^2 - 1)^(-1/2)`` at the transverse-wave limit ``lambda = 1``;
* for a hull whose ``y_x`` does not vanish on the waterline (the spheroid is
  one) the ``lambda`` integrand decays like ``lambda^-3``, so the truncation
  ``lambda_max`` leaves a ``C/lambda_max^2`` residual.  Pushing ``lambda_max``
  out and eliminating ``C`` from two truncations
  (:func:`michell_wave_resistance`) reproduces the untruncated value to
  ``~1e-4`` relative.

Where the absolute value stands
-------------------------------
Every number here is *Michell theory against its own discretisation*.  This
module makes no claim about agreement with experiment: Michell theory is a
documented slender-body approximation (no viscosity, no wave breaking, no
sinkage or trim, no finite thickness) and no experimental data set is shipped
with this repository.  The evidence level for the absolute-value question is
therefore ``WITHHELD``; what the accompanying benchmarks verify is the
discretisation and the internal consistency of the two paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DEFAULT_GRAVITY = 9.80665
DEFAULT_DENSITY = 1025.0

__all__ = [
    "EllipsoidHull",
    "WigleyHull",
    "CenterplanePanels",
    "ellipsoid_amplitude_reference",
    "ellipsoid_amplitude_function",
    "ellipsoid_amplitude_quadrature_2d",
    "wigley_amplitude_reference",
    "wigley_amplitude_function",
    "tandem_amplitude",
    "wave_resistance_from_amplitude",
    "wave_resistance_theta_form",
    "michell_wave_resistance",
    "grid_convergence_index",
    "ellipsoid_panels",
    "wigley_panels",
]


# ============================================================================
# Hull descriptors (centreplane half breadth y(x, z) and its x derivative)
# ============================================================================


@dataclass(frozen=True)
class EllipsoidHull:
    """Surface-piercing spheroid with half axes ``a = L/2``, ``b = B/2``, ``c = T``.

    The submerged half of ``(x/a)^2 + (y/b)^2 + (z/c)^2 = 1`` has half breadth

        ``y(x, z) = b sqrt(1 - (x/a)^2 - (z/c)^2)``, ``z in [-c, 0]``,

    so ``B`` is the waterline beam (the maximum beam of the body) and ``T`` the
    draft from the calm waterline.  ``y`` and ``y_x`` are singular on the
    waterline ellipse ``(x/a)^2 + (z/c)^2 = 1``, where the surface is tangent
    to the vertical; the singularity is integrable and is removed exactly by
    the ``(psi, phi)`` parameterisation used below.
    """

    L: float
    B: float
    T: float

    def __post_init__(self) -> None:
        for name in ("L", "B", "T"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)

    @property
    def a(self) -> float:
        return 0.5 * self.L

    @property
    def b(self) -> float:
        return 0.5 * self.B

    @property
    def c(self) -> float:
        return self.T

    def half_breadth(self, x: ArrayLike, z: ArrayLike) -> FloatArray:
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        radicand = 1.0 - (x / self.a) ** 2 - (z / self.c) ** 2
        return self.b * np.sqrt(np.maximum(radicand, 0.0))

    def slope(self, x: ArrayLike, z: ArrayLike) -> FloatArray:
        """``y_x = -b x / (a^2 sqrt(1 - (x/a)^2 - (z/c)^2))``."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        radicand = np.maximum(1.0 - (x / self.a) ** 2 - (z / self.c) ** 2, 0.0)
        return -self.b * x / (self.a**2 * np.sqrt(radicand + 1.0e-300))

    def hydrostatic_reference(self) -> dict[str, float]:
        """Closed-form hydrostatics of the submerged half ellipsoid.

        ``volume = 2 pi a b c / 3``, ``Cb = volume/(L B T) = pi/6``,
        ``A_wp = pi a b``, ``I_xx = pi a b^3/4`` (about x), ``I_yy = pi b a^3/4``
        (about y) and ``z_B = -3c/8`` (centroid of a semi-ellipsoid).
        """
        a, b, c = self.a, self.b, self.c
        volume = 2.0 * math.pi * a * b * c / 3.0
        return {
            "L": self.L,
            "B": self.B,
            "T": self.T,
            "volume": volume,
            "Cb": volume / (self.L * self.B * self.T),
            "Cm": math.pi / 4.0,
            "Cp": 2.0 / 3.0,
            "waterplane_area": math.pi * a * b,
            "I_xx": math.pi * a * b**3 / 4.0,
            "I_yy": math.pi * b * a**3 / 4.0,
            "center_of_buoyancy_z": -3.0 * c / 8.0,
            "KB": 5.0 * c / 8.0,
        }

    def wetted_area_reference(
        self, n_psi: int = 120, n_phi: int = 240, order: int = 6
    ) -> float:
        """Wetted area of the submerged half spheroid by 2-D Gauss quadrature.

        ``S_w = 2 int int sqrt(1 + y_x^2 + y_z^2) dx dz`` over the submerged
        half of the centreplane projection.  The ``(psi, phi)`` mapping
        (``x = a sin psi cos phi``, ``z = -c sin psi sin phi``) turns the
        area element into ``a c sin psi cos psi dpsi dphi``, which cancels the
        ``1/cos psi`` growth of ``sqrt(1 + y_x^2 + y_z^2)`` on the keel line, so
        a uniform Gauss rule converges quickly (a ``2x`` refinement moves the
        value by ``~1e-4`` relative, see the benchmark artifact).
        """
        gx, gw = _legendre_gauss(order)
        edges_psi = np.linspace(0.0, 0.5 * math.pi, n_psi + 1)
        edges_phi = np.linspace(0.0, math.pi, n_phi + 1)
        psi = (0.5 * (edges_psi[:-1] + edges_psi[1:])[:, None]
               + 0.5 * (edges_psi[1:] - edges_psi[:-1])[:, None] * gx[None, :]).ravel()
        w_psi = (0.5 * (edges_psi[1:] - edges_psi[:-1])[:, None] * gw[None, :]).ravel()
        phi = (0.5 * (edges_phi[:-1] + edges_phi[1:])[:, None]
               + 0.5 * (edges_phi[1:] - edges_phi[:-1])[:, None] * gx[None, :]).ravel()
        w_phi = (0.5 * (edges_phi[1:] - edges_phi[:-1])[:, None] * gw[None, :]).ravel()
        PS, PH = np.meshgrid(psi, phi, indexing="ij")
        WW = np.outer(w_psi, w_phi)
        x = self.a * np.sin(PS) * np.cos(PH)
        z = -self.c * np.sin(PS) * np.sin(PH)
        radicand = np.maximum(1.0 - (x / self.a) ** 2 - (z / self.c) ** 2, 1.0e-300)
        yx = -self.b * x / (self.a**2 * np.sqrt(radicand))
        yz = -self.b * z / (self.c**2 * np.sqrt(radicand))
        jac = self.a * self.c * np.sin(PS) * np.cos(PS)
        return float(2.0 * np.sum(WW * np.sqrt(1.0 + yx**2 + yz**2) * jac))


@dataclass(frozen=True)
class WigleyHull:
    """Parabolic Wigley hullform ``y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2)``.

    ``y_x`` is finite everywhere, so no singular quadrature is needed; the
    closed-form amplitude is the one used by
    :func:`tensorbem.wigley.michell_kochin_function` when ``alpha = k`` (up to
    the factor two that older path folded into the Kochin function to account
    for the two sides of the hull).
    """

    L: float
    B: float
    T: float

    def __post_init__(self) -> None:
        for name in ("L", "B", "T"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)

    def half_breadth(self, x: ArrayLike, z: ArrayLike) -> FloatArray:
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        return (
            0.5
            * self.B
            * (1.0 - (2.0 * x / self.L) ** 2)
            * (1.0 - (z / self.T) ** 2)
        )

    def slope(self, x: ArrayLike, z: ArrayLike) -> FloatArray:
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        return -4.0 * self.B * x / self.L**2 * (1.0 - (z / self.T) ** 2)

    def hydrostatic_reference(self) -> dict[str, float]:
        """Closed forms of the Wigley form (see :mod:`tensorbem.wigley_hydrostatics`)."""
        L, B, T = self.L, self.B, self.T
        volume = 4.0 * L * B * T / 9.0
        return {
            "L": L,
            "B": B,
            "T": T,
            "volume": volume,
            "Cb": 4.0 / 9.0,
            "Cm": 2.0 / 3.0,
            "Cp": 2.0 / 3.0,
            "waterplane_area": 2.0 * L * B / 3.0,
            "I_xx": 4.0 * B**3 * L / 105.0,
            "I_yy": B * L**3 / 30.0,
            "center_of_buoyancy_z": -3.0 * T / 8.0,
            "KB": 5.0 * T / 8.0,
        }

# ============================================================================
# Reference amplitudes
# ============================================================================


_LEGENDRE_CACHE: dict[int, tuple[FloatArray, FloatArray]] = {}


def _legendre_gauss(n: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre rule, cached by order.

    ``numpy.polynomial.legendre.leggauss`` builds the rule through a dense
    symmetric eigenvalue solve, which costs ~1.4 s at ``n = 300`` on the
    loongarch64 build used here -- and the ``lambda`` integration calls it once
    per evaluation, so an anchor sweep spent most of its time rebuilding the
    same rule.  Caching it makes the whole study I/O bound again.
    """
    rule = _LEGENDRE_CACHE.get(n)
    if rule is None:
        rule = np.polynomial.legendre.leggauss(n)
        _LEGENDRE_CACHE[n] = rule
    return rule


_DEPTH_CHEBYSHEV: Any = None

#: ``c_n (2n)! = (2n)!^2 / (4^n (n!)^2)``, the ten-term optimal truncation of the
#: Watson series for the depth kernel.
_DEPTH_SERIES = np.array(
    [math.factorial(2 * n) ** 2 / (4**n * math.factorial(n) ** 2) for n in range(10)],
    dtype=float,
)


def _depth_kernel_table():
    """Chebyshev interpolant of ``G`` on ``[0, 12]``, built once from Bessel functions.

    Calling ``scipy.special.iv``/``modstruve`` for every quadrature point cost an
    order of magnitude more than the rest of the ``lambda`` integral (they are
    ~1 us each and the integral has millions of points).  ``G`` is analytic on
    the positive real axis, so a degree-64 Chebyshev interpolant reproduces it to
    ``~1e-12`` relative -- checked in ``tests/test_thin_ship_michell.py`` -- and
    evaluation is a vectorised polynomial sum.  The domain stops at ``v = 18``
    because ``I_0 - L_0`` loses digits to cancellation beyond that (both terms
    are ``~e^v`` while their difference decays).
    """
    global _DEPTH_CHEBYSHEV
    if _DEPTH_CHEBYSHEV is None:
        from scipy.special import iv, modstruve

        def exact(x: float) -> float:
            return 0.5 * math.pi * (iv(0, x) - modstruve(0, x))

        _DEPTH_CHEBYSHEV = np.polynomial.chebyshev.Chebyshev.interpolate(
            exact, 96, domain=[0.0, 17.0]
        )
    return _DEPTH_CHEBYSHEV


def _depth_integral(v: FloatArray) -> FloatArray:
    """``G(v) = int_0^1 exp(-v u) du / sqrt(1 - u^2) = int_0^{pi/2} exp(-v sin t) dt``.

    This is the depth integral left after the vertical coordinate is eliminated
    analytically (see :func:`ellipsoid_amplitude_reference`).  For ``v >= 12``
    Watson's lemma for the boundary layer at ``t -> 0`` gives

        ``G ~ (1/v)(1 + 1/(2v^2) + 3/(8v^4) + 5/(16v^6))``,

    whose next term is below ``1e-9`` relative at the switch-over; below that the
    Chebyshev interpolant of the exact ``(pi/2)(I_0 - L_0)`` is used.
    """
    v_arr = np.asarray(v, dtype=float)
    out = np.empty(v_arr.shape, dtype=float)
    small = v_arr < 17.0
    if np.any(small):
        out[small] = _depth_kernel_table()(v_arr[small])
    if np.any(~small):
        inv = 1.0 / v_arr[~small]
        inv2 = inv * inv
        series = np.zeros_like(inv)
        for coefficient in _DEPTH_SERIES[::-1]:
            series = series * inv2 + coefficient
        out[~small] = inv * series
    return out


def _theta_midpoint_rule(n_cell: int, order: int) -> tuple[FloatArray, FloatArray]:
    """Uniform composite Gauss rule on ``[-pi/2, pi/2]``."""
    gx, gw = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_cell + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1:] - edges[:-1])
    nodes = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    weights = (half[:, None] * gw[None, :]).ravel()
    return nodes, weights


def _theta_quadrature(n_cell: int, order: int):
    return _theta_midpoint_rule(n_cell, order)


def ellipsoid_amplitude_reference(
    hull: EllipsoidHull,
    k: float,
    alpha: float,
    *,
    n_cell: int | None = None,
    order: int = 6,
) -> complex:
    """Michell amplitude of the spheroid with the depth integral done analytically.

    Integrating ``y_x exp(k z)`` over ``z`` first leaves a one-dimensional
    integral.  With ``x = a sin(theta)``:

    .. math::
        A = -b c \\int_{-\\pi/2}^{\\pi/2} \\sin\\theta\\cos\\theta\\,
            G(k c \\cos\\theta)\\, e^{i \\alpha a \\sin\\theta}\\,d\\theta,

    where ``G`` is :func:`_depth_integral`.  Derivation: for fixed ``x`` the
    vertical extent is ``z in [-c sqrt(1-(x/a)^2), 0]`` and
    ``y_x = -b x /(a^2 sqrt(1-(x/a)^2-(z/c)^2))``; the substitution
    ``z = -c sqrt(1-(x/a)^2) u`` turns the ``z`` integral into ``G`` exactly.

    This form has no boundary layer at the tip (``|theta| -> pi/2``) -- the
    ``sin(theta) cos(theta)`` factor cancels the ``1/(k c cos theta)`` tail of
    ``G``, leaving a flat ``1/(k c)`` plateau -- so a *uniform* composite Gauss
    rule converges at machine precision once its cells resolve the streamwise
    oscillation ``exp(i alpha a sin theta)``.  ``n_cell`` defaults to six cells
    per oscillation period, ``max(64, ceil(6 |alpha| a / pi))``.
    """
    if n_cell is None:
        n_cell = max(64, int(math.ceil(6.0 * abs(alpha) * hull.a / math.pi)))
    theta, weights = _theta_quadrature(n_cell, order)
    v = k * hull.c * np.cos(theta)
    kernel = _depth_integral(v)
    integrand = np.sin(theta) * np.cos(theta) * kernel * np.exp(1j * alpha * hull.a * np.sin(theta))
    return complex(-hull.b * hull.c * np.sum(weights * integrand))


def ellipsoid_amplitude_function(
    hull: EllipsoidHull,
    *,
    order: int = 6,
    cells_per_period: float = 6.0,
    min_cells: int = 64,
) -> Callable[[float | FloatArray, float | FloatArray], ComplexArray]:
    """Vectorised :func:`ellipsoid_amplitude_reference` for a whole ``lambda`` array.

    The ``theta`` grid is rebuilt for the largest ``|alpha|`` in the request (its
    resolution requirement is the streamwise oscillation), so one call is a
    single matrix evaluation rather than a Python loop over ``lambda``.
    """
    payload_cache: dict[int, tuple[FloatArray, FloatArray, FloatArray]] = {}

    def grid_for(alpha_max: float):
        n_cell = max(min_cells, int(math.ceil(cells_per_period * abs(alpha_max) * hull.a / math.pi)))
        cached = payload_cache.get(n_cell)
        if cached is None:
            theta, weights = _theta_quadrature(n_cell, order)
            cached = (theta, weights, np.sin(theta) * np.cos(theta))
            payload_cache.clear()
            payload_cache[n_cell] = cached
        return cached

    def amplitude(k: float | FloatArray, alpha: float | FloatArray) -> ComplexArray:
        k_arr, alpha_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(k, dtype=float)),
            np.atleast_1d(np.asarray(alpha, dtype=float)),
        )
        theta, weights, shape = grid_for(float(np.max(np.abs(alpha_arr))))
        kk = k_arr.ravel()[:, None]
        aa = alpha_arr.ravel()[:, None]
        kernel = _depth_integral(kk * hull.c * np.cos(theta)[None, :])
        integrand = (
            shape[None, :]
            * kernel
            * np.exp(1j * aa * hull.a * np.sin(theta)[None, :])
        )
        out = -hull.b * hull.c * np.sum(weights[None, :] * integrand, axis=1)
        return out.reshape(k_arr.shape)

    return amplitude


def ellipsoid_amplitude_quadrature_2d(
    hull: EllipsoidHull,
    k: float,
    alpha: float,
    *,
    n_psi: int = 200,
    n_phi: int = 400,
    order: int = 6,
) -> complex:
    """Independent 2-D quadrature of ``A`` on the ``(psi, phi)`` rectangle.

    Only useful as a cross-check at *small* wavenumber: the parameterisation
    ``x = a sin(psi) cos(phi)``, ``z = -c sin(psi) sin(phi)`` makes the
    integrand analytic (``y_x dx dz = -b c sin^2(psi) cos(phi) dpsi dphi``),
    but a uniform grid has to resolve both the waterline boundary layer of
    width ``1/(k c)`` and the oscillation ``exp(i alpha a sin psi cos phi)``, so
    the cost grows quickly with ``k``.  :func:`ellipsoid_amplitude_reference`
    is the production reference; this one exists so that the analytic depth
    reduction is checked against a quadrature that never uses ``G``.
    """
    gx, gw = np.polynomial.legendre.leggauss(order)
    edges_psi = np.linspace(0.0, 0.5 * math.pi, n_psi + 1)
    edges_phi = np.linspace(0.0, math.pi, n_phi + 1)
    psi_mid = 0.5 * (edges_psi[:-1] + edges_psi[1:])
    psi_half = 0.5 * (edges_psi[1:] - edges_psi[:-1])
    phi_mid = 0.5 * (edges_phi[:-1] + edges_phi[1:])
    phi_half = 0.5 * (edges_phi[1:] - edges_phi[:-1])
    PS = (psi_mid[:, None] + psi_half[:, None] * gx[None, :]).ravel()
    WPS = (psi_half[:, None] * gw[None, :]).ravel()
    PH = (phi_mid[:, None] + phi_half[:, None] * gx[None, :]).ravel()
    WPH = (phi_half[:, None] * gw[None, :]).ravel()
    PSg, PHg = np.meshgrid(PS, PH, indexing="ij")
    W = np.outer(WPS, WPH)
    x = hull.a * np.sin(PSg) * np.cos(PHg)
    z = -hull.c * np.sin(PSg) * np.sin(PHg)
    density = -hull.b * hull.c * np.sin(PSg) ** 2 * np.cos(PHg)
    return complex(np.sum(W * density * np.exp(k * z + 1j * alpha * x)))


def _wigley_amplitude_axis(k: float, L: float) -> complex:
    """``int_{-L/2}^{L/2} x exp(i k x) dx`` in closed form."""
    if abs(k) < 1.0e-14:
        return 0.0j
    ik = 1j * k
    half = 0.5 * L
    return complex(
        half * np.exp(ik * half) / ik
        - np.exp(ik * half) / (ik * ik)
        + half * np.exp(-ik * half) / ik
        + np.exp(-ik * half) / (ik * ik)
    )


def _wigley_amplitude_depth(k: float, T: float) -> float:
    """``int_{-T}^{0} (1 - (z/T)^2) exp(k z) dz`` in closed form."""
    if abs(k) < 1.0e-14:
        return 2.0 * T / 3.0
    exp_kT = math.exp(-k * T)
    first = (1.0 - exp_kT) / k
    second = 2.0 / k**3 - exp_kT * (T**2 / k + 2.0 * T / k**2 + 2.0 / k**3)
    return float(first - second / T**2)


def wigley_amplitude_reference(hull: WigleyHull, k: float, alpha: float) -> complex:
    """Michell amplitude of the parabolic Wigley form, closed form.

    ``A(k, alpha) = (-4B/L^2) I_x(alpha) I_z(k)`` with
    ``I_x(alpha) = int_{-L/2}^{L/2} x exp(i alpha x) dx`` and
    ``I_z(k) = int_{-T}^{0} (1 - (z/T)^2) exp(k z) dz``.  The two wavenumbers
    are different on purpose: ``alpha`` carries the streamwise phase, ``k`` the
    depth decay.
    """
    return (
        (-4.0 * hull.B / hull.L**2)
        * _wigley_amplitude_axis(alpha, hull.L)
        * _wigley_amplitude_depth(k, hull.T)
    )


def wigley_amplitude_function(
    hull: WigleyHull,
) -> Callable[[float | FloatArray, float | FloatArray], ComplexArray]:
    """Vectorised :func:`wigley_amplitude_reference` (closed forms, no quadrature)."""

    def amplitude(k: float | FloatArray, alpha: float | FloatArray) -> ComplexArray:
        k_arr, alpha_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(k, dtype=float)),
            np.atleast_1d(np.asarray(alpha, dtype=float)),
        )
        al = alpha_arr
        ik = 1j * al
        half = 0.5 * hull.L
        safe_ik = np.where(np.abs(al) < 1.0e-12, 1.0, ik)
        axis = (
            half * np.exp(ik * half) / safe_ik
            - np.exp(ik * half) / (safe_ik * safe_ik)
            + half * np.exp(-ik * half) / safe_ik
            + np.exp(-ik * half) / (safe_ik * safe_ik)
        )
        axis = np.where(np.abs(al) < 1.0e-12, 1j * al * hull.L**3 / 12.0, axis)
        kk = k_arr
        exp_kT = np.exp(-kk * hull.T)
        safe_k = np.where(np.abs(kk) < 1.0e-12, 1.0, kk)
        first = (1.0 - exp_kT) / safe_k
        second = 2.0 / safe_k**3 - exp_kT * (
            hull.T**2 / safe_k + 2.0 * hull.T / safe_k**2 + 2.0 / safe_k**3
        )
        depth = np.where(np.abs(kk) < 1.0e-12, 2.0 * hull.T / 3.0, first - second / hull.T**2)
        return (-4.0 * hull.B / hull.L**2) * axis * depth

    return amplitude


def tandem_amplitude(
    single: Callable[[float, float], complex],
    separation: float,
) -> Callable[[float, float], complex]:
    """Amplitude of two identical hulls in tandem, centres ``separation`` apart.

    In the far field the two Kochin amplitudes superpose with the streamwise
    phase lag ``alpha * separation`` (``alpha = k_0 lambda``):

        ``A_pair = A_single (1 + exp(i alpha d))``.

    ``d`` is centre to centre, so a clear gap ``s`` between the fore hull's AP
    and the aft hull's FP is ``d = L + s``.  ``separation = 0`` therefore gives
    ``|A_pair| = 2|A|``, i.e. a doubled hull, ``R = 4 R_single`` and an
    interference factor ``sigma = R_pair/(2 R_single) = 2``; for ``d -> inf``
    the oscillatory cross term averages out and ``sigma -> 1``.  Both limits
    are asserted in the tests.
    """
    if separation < 0.0:
        raise ValueError("separation must be non-negative")

    def amplitude(k: float, alpha: float) -> complex:
        return single(k, alpha) * (1.0 + np.exp(1j * alpha * separation))

    return amplitude


# ============================================================================
# Panel discretisation of the centreplane source sheet
# ============================================================================


@dataclass(frozen=True)
class CenterplanePanels:
    """Flat centreplane panels with a source density sampled inside each panel.

    Each panel is a quadrilateral in the ``(x, z)`` centreplane, stored as the
    points and weights of a 2x2 Gauss rule mapped through the panel's bilinear
    map.  The source density ``y_x`` is sampled at those same Gauss points
    rather than once per panel, which is how a thin-ship panel code evaluates a
    density that is finite but steep (or integrable-singular) at a panel edge
    without smearing it over the whole panel.

    Attributes
    ----------
    nodes :
        ``(n_panel, 4, 2)`` physical coordinates of the Gauss points.
    weights :
        ``(n_panel, 4)`` quadrature weights times the local Jacobian
        determinant (an area measure).
    density :
        ``(n_panel, 4)`` source density ``y_x`` at the Gauss points.
    """

    nodes: FloatArray
    weights: FloatArray
    density: FloatArray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        if nodes.ndim != 3 or nodes.shape[1:] != (4, 2):
            raise ValueError("nodes must have shape (n_panel, 4, 2)")
        weights = np.asarray(self.weights, dtype=float)
        density = np.asarray(self.density, dtype=float)
        if weights.shape != nodes.shape[:2] or density.shape != nodes.shape[:2]:
            raise ValueError("weights and density must have shape (n_panel, 4)")

    @property
    def panel_count(self) -> int:
        return int(self.nodes.shape[0])

    def amplitude(
        self,
        k: float | FloatArray,
        alpha: float | FloatArray,
        x_offset: float = 0.0,
    ) -> ComplexArray:
        """Kochin amplitude of the sheet, vectorised over ``(k, alpha)``.

        ``k`` and ``alpha`` may be scalars or broadcastable 1-D arrays; the
        result has their broadcast shape.  ``x_offset`` shifts the whole sheet
        along ``x`` (used for the aft hull of a tandem pair).
        """
        k_arr, alpha_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(k, dtype=float)),
            np.atleast_1d(np.asarray(alpha, dtype=float)),
        )
        x = self.nodes[:, :, 0] + x_offset
        z = self.nodes[:, :, 1]
        contribution = self.weights * self.density
        flat_k = k_arr.ravel()
        flat_alpha = alpha_arr.ravel()
        out = np.empty(flat_k.size, dtype=complex)
        # chunk over lambda so the (n_lam, n_panel, 4) working array stays small
        chunk = max(1, int(2.0e6 // max(1, contribution.size)))
        for start in range(0, flat_k.size, chunk):
            kk = flat_k[start : start + chunk][:, None, None]
            aa = flat_alpha[start : start + chunk][:, None, None]
            out[start : start + chunk] = np.sum(
                contribution[None, :, :] * np.exp(kk * z[None, :, :] + 1j * aa * x[None, :, :]),
                axis=(1, 2),
            )
        return out.reshape(k_arr.shape)


def _panels_from_corner_grid(
    x_corners: FloatArray,
    z_corners: FloatArray,
    slope: Callable[[FloatArray, FloatArray], FloatArray],
    order: int = 2,
) -> CenterplanePanels:
    """2x2 Gauss rule inside every cell of a structured corner grid."""
    gx, gw = _legendre_gauss(order)
    u = 0.5 * (gx + 1.0)
    wu = 0.5 * gw
    U, V = np.meshgrid(u, u, indexing="ij")
    W = np.outer(wu, wu).ravel()

    nodes: list[FloatArray] = []
    weights: list[FloatArray] = []
    densities: list[FloatArray] = []
    n_x = x_corners.shape[0] - 1
    n_z = z_corners.shape[1] - 1
    for i in range(n_x):
        for j in range(n_z):
            x00, x10 = x_corners[i, j], x_corners[i + 1, j]
            x01, x11 = x_corners[i, j + 1], x_corners[i + 1, j + 1]
            z00, z10 = z_corners[i, j], z_corners[i + 1, j]
            z01, z11 = z_corners[i, j + 1], z_corners[i + 1, j + 1]
            px = (
                (1 - U) * (1 - V) * x00
                + U * (1 - V) * x10
                + (1 - U) * V * x01
                + U * V * x11
            )
            pz = (
                (1 - U) * (1 - V) * z00
                + U * (1 - V) * z10
                + (1 - U) * V * z01
                + U * V * z11
            )
            dpx_du = (1 - V) * (x10 - x00) + V * (x11 - x01)
            dpz_du = (1 - V) * (z10 - z00) + V * (z11 - z01)
            dpx_dv = (1 - U) * (x01 - x00) + U * (x11 - x10)
            dpz_dv = (1 - U) * (z01 - z00) + U * (z11 - z10)
            det = np.abs(dpx_du * dpz_dv - dpx_dv * dpz_du).ravel()
            nodes.append(np.column_stack([px.ravel(), pz.ravel()]))
            weights.append(W * det)
            densities.append(np.asarray(slope(px, pz), dtype=float).ravel())
    return CenterplanePanels(
        nodes=np.asarray(nodes, dtype=float),
        weights=np.asarray(weights, dtype=float),
        density=np.asarray(densities, dtype=float),
    )


def ellipsoid_panels(
    hull: EllipsoidHull,
    n_psi: int,
    n_phi: int,
    order: int = 2,
    *,
    psi_spacing: str = "cosine",
) -> CenterplanePanels:
    """Centreplane panels of the spheroid on a ``(psi, phi)`` grid.

    ``psi`` runs over ``[0, pi/2]`` (transverse parameter) and ``phi`` over
    ``[0, pi]`` (streamwise), so corners follow ``x = a sin(psi) cos(phi)``,
    ``z = -c sin(psi) sin(phi)``; the waterline (``psi = 0``) and the keel line
    (``psi = pi/2``) are panel edges and the density is sampled strictly inside
    the panels.

    ``psi_spacing`` selects the panel distribution along ``psi``:

    * ``"cosine"`` (default): ``psi_j = (pi/2) sin(pi j / (2 n))``, which
      clusters cells at *both* ends of the ``psi`` range.  Those are exactly
      the two boundary layers of this hull: at ``psi -> 0`` the kernel
      ``exp(k z)`` decays over a depth ``1/k``, and at ``psi -> pi/2`` the
      source density ``y_x ~ 1/cos(psi)`` diverges (logarithmically).
    * ``"uniform"``: equidistant in ``psi``, the plain generator, kept so the
      cost of the clustering can be measured (it halves the error constant at
      the same panel count, see ``benchmarks/verified/bem_ellipsoid_wave``).
    """
    if n_psi < 1 or n_phi < 1:
        raise ValueError("n_psi and n_phi must be >= 1")
    xi = np.linspace(0.0, 1.0, n_psi + 1)
    if psi_spacing == "cosine":
        psi = 0.5 * math.pi * np.sin(0.5 * math.pi * xi)
    elif psi_spacing == "uniform":
        psi = 0.5 * math.pi * xi
    else:
        raise ValueError("psi_spacing must be 'cosine' or 'uniform'")
    phi = np.linspace(0.0, math.pi, n_phi + 1)
    PS, PH = np.meshgrid(psi, phi, indexing="ij")
    x_corners = hull.a * np.sin(PS) * np.cos(PH)
    z_corners = -hull.c * np.sin(PS) * np.sin(PH)
    return _panels_from_corner_grid(x_corners, z_corners, hull.slope, order)


def wigley_panels(
    hull: WigleyHull,
    n_x: int,
    n_z: int,
    order: int = 2,
    cluster: bool = True,
) -> CenterplanePanels:
    """Centreplane panels of the Wigley form on an ``(x, z)`` grid.

    ``cluster=True`` uses cosine-spaced stations ``x = -(L/2) cos(pi xi)``,
    which resolve the finite but steep ``y_x ~ x`` ramp at the posts;
    ``cluster=False`` is uniform.  ``z`` is uniform over ``[-T, 0]``.
    """
    if n_x < 1 or n_z < 1:
        raise ValueError("n_x and n_z must be >= 1")
    xi = np.linspace(0.0, 1.0, n_x + 1)
    x = -0.5 * hull.L * np.cos(math.pi * xi) if cluster else -0.5 * hull.L + hull.L * xi
    z = np.linspace(-hull.T, 0.0, n_z + 1)
    X, Z = np.meshgrid(x, z, indexing="ij")
    return _panels_from_corner_grid(X, Z, hull.slope, order)


# ============================================================================
# Wave resistance from an amplitude
# ============================================================================


def wave_resistance_from_amplitude(
    amplitude: Callable[[float | FloatArray, float | FloatArray], ComplexArray],
    U: float,
    *,
    lam_max: float = 30.0,
    n_lam: int = 160,
    rho: float = DEFAULT_DENSITY,
    g: float = DEFAULT_GRAVITY,
) -> float:
    """``R_w`` from the ``lambda`` form of Michell's integral.

    ``R_w = (4 rho g^2/(pi U^2)) int_1^{lam_max} |A|^2 lam^2 dlam / sqrt(lam^2-1)``
    with the ``lam = 1 + s^2`` substitution, which turns the weight into
    ``2 lam^2 ds / sqrt(2 + s^2)`` and removes the cusp at ``lam = 1``.
    """
    if U <= 0:
        raise ValueError("speed U must be positive")
    if lam_max <= 1.0:
        raise ValueError("lam_max must exceed 1")
    k0 = g / U**2
    s_max = math.sqrt(lam_max - 1.0)
    x, w = _legendre_gauss(n_lam)
    s = 0.5 * s_max * (x + 1.0)
    weights = 0.5 * s_max * w
    lam = 1.0 + s * s
    ampl = np.asarray(amplitude(k0 * lam**2, k0 * lam), dtype=complex)
    integrand = 2.0 * lam**2 * np.abs(ampl) ** 2 / np.sqrt(2.0 + s * s)
    return float((4.0 * rho * g * g / (math.pi * U**2)) * np.sum(weights * integrand))


def wave_resistance_theta_form(
    amplitude_theta: Callable[[FloatArray], ComplexArray],
    U: float,
    *,
    n_theta: int = 400,
    order: int = 8,
    rho: float = DEFAULT_DENSITY,
    g: float = DEFAULT_GRAVITY,
    theta_max: float | None = None,
) -> float:
    """``R_w = (4 rho g^2/(pi U^2)) int_0^{pi/2} sec^3(theta)|A(theta)|^2 dtheta``.

    ``amplitude_theta`` receives the quadrature angles and must return the
    amplitude at ``k = k_0 sec^2 theta``, ``alpha = k_0 sec theta``.  The rule is
    a graded composite Gauss (cells clustered toward ``theta = pi/2``), the
    discrete counterpart of the ``lambda`` substitution, and exists only as a
    quadrature-independent cross-check of
    :func:`wave_resistance_from_amplitude`.

    ``theta_max`` (default ``pi/2``) truncates the angular integral; setting it
    to ``arccos(1/lambda_max)`` matches the truncation of the ``lambda`` form, so
    the two can be compared without an ``alpha`` range that no quadrature
    resolves (``sec(theta)`` diverges there even though the ``exp(k z)`` decay
    makes the contribution negligible).
    """
    if U <= 0:
        raise ValueError("speed U must be positive")
    k0 = g / U**2
    span = 0.5 * math.pi if theta_max is None else float(theta_max)
    gx, gw = _legendre_gauss(order)
    nodes: list[FloatArray] = []
    weights: list[FloatArray] = []
    power = 3.0
    for j in range(n_theta):
        lo = span * (j / n_theta) ** power
        hi = span * ((j + 1) / n_theta) ** power
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        nodes.append(mid + half * gx)
        weights.append(half * gw)
    theta = np.concatenate(nodes)
    dtheta = np.concatenate(weights)
    lam = 1.0 / np.cos(theta)
    ampl = np.asarray(amplitude_theta(theta), dtype=complex)
    return float(
        (4.0 * rho * g * g / (math.pi * U**2)) * np.sum(dtheta * lam**3 * np.abs(ampl) ** 2)
    )


def michell_wave_resistance(
    amplitude: Callable[[float, float], complex],
    U: float,
    *,
    lam_max: float = 20.0,
    n_lam: int = 160,
    rho: float = DEFAULT_DENSITY,
    g: float = DEFAULT_GRAVITY,
    extrapolate_tail: bool = True,
) -> dict[str, float]:
    """``R_w`` with the ``lambda``-truncation residual removed by extrapolation.

    For a hull whose ``y_x`` does not vanish on the waterline (the spheroid is
    one) the ``lambda`` integrand decays like ``lambda^-3``, so truncating at
    ``lam_max`` leaves ``C/lam_max^2``.  Two truncations, ``lam_max`` and
    ``2 lam_max``, determine ``C`` and remove it::

        R_inf = (4 R(2 lam_max) - R(lam_max)) / 3

    Both raw values are returned so the size of the correction stays visible.
    """
    raw = float(
        wave_resistance_from_amplitude(
            amplitude, U, lam_max=lam_max, n_lam=n_lam, rho=rho, g=g
        )
    )
    if not extrapolate_tail:
        return {
            "Rw": raw,
            "Rw_truncated": raw,
            "tail_correction": 0.0,
            "lam_max": float(lam_max),
        }
    wide = float(
        wave_resistance_from_amplitude(
            amplitude, U, lam_max=2.0 * lam_max, n_lam=n_lam, rho=rho, g=g
        )
    )
    value = (4.0 * wide - raw) / 3.0
    return {
        "Rw": value,
        "Rw_truncated": raw,
        "tail_correction": value - raw,
        "lam_max": float(lam_max),
    }


# ============================================================================
# Grid convergence index (Roache)
# ============================================================================


def grid_convergence_index(
    coarse: float,
    medium: float,
    fine: float,
    refinement: float = 2.0,
    safety: float = 1.25,
) -> dict[str, float]:
    """Roache's grid convergence index from three refinement levels.

    ``p`` is the observed order from the three values (``ln|(f3-f2)/(f2-f1)| /
    ln r``); ``GCI = safety * |(f2 - f1)/f1| / (r^p - 1)`` with ``f1`` the fine
    value.  When the three values are effectively equal (``|f2 - f1|`` below
    ``1e-14`` relative) both ``p`` and the GCI are reported as ``0``.
    """
    if refinement <= 1.0:
        raise ValueError("refinement ratio must exceed 1")
    f1, f2, f3 = float(fine), float(medium), float(coarse)
    if not all(math.isfinite(v) and v != 0.0 for v in (f1, f2, f3)):
        raise ValueError("all three values must be finite and non-zero")
    d_medium = abs(f2 - f1) / abs(f1)
    if d_medium < 1.0e-14:
        return {"order": 0.0, "gci_fine": 0.0, "rel_change": d_medium, "refinement": refinement}
    order = 0.0
    delta_coarse = abs(f3 - f2)
    delta_medium = abs(f2 - f1)
    if delta_coarse > 1.0e-14 and delta_medium > 1.0e-14:
        ratio = delta_coarse / delta_medium
        if ratio > 1.0e-14:
            order = abs(math.log(ratio) / math.log(refinement))
    gci = float("inf")
    if order > 1.0e-8:
        gci = safety * d_medium / (refinement**order - 1.0)
    return {
        "order": order,
        "gci_fine": gci,
        "rel_change": d_medium,
        "refinement": refinement,
    }
