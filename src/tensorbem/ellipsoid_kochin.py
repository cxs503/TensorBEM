"""Michell/Havelock thin-ship wave resistance of a half-submerged ellipsoid.

Why this module
---------------
``tensorbem.wigley`` carries the Michell/Kochin anchor for the *separable*
Wigley hullform, whose Kochin function factorises into closed-form ``x`` and
``z`` integrals.  A separable hullform cannot answer the natural question
"does the thin-ship spectral machinery reproduce a *non-separable* reference
body?", and the ellipsoid is the textbook non-separable case (Havelock 1932,
*The wave resistance of a spheroid*, Proc. R. Soc. A 131; Michell 1898).

The ellipsoid ``x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1`` trimmed at the calm
waterplane ``z = 0`` has the centreplane ordinate

    y(x, z) = b * sqrt(1 - x^2/a^2 - z^2/c^2),   z in [-c, 0],

so it is *not* separable.  Its Kochin function nevertheless reduces to a
single quadrature, because the longitudinal integral is Bessel-analytic:

    int_{-A}^{A} sqrt(1 - x^2/A^2) e^{ikx} dx = pi * J1(kA) / k ,

and, integrating the defining integral by parts in ``x`` (``y`` vanishes at
the elliptical edge ``x = +-A(z)``),

    H(k) = 2 * int int_{one side} dy/dx * e^{kz} e^{ikx} dx dz
         = -2 * i * k * int int y e^{kz} e^{ikx} dx dz
         = -2 * pi * i * b * int_{-c}^{0} sqrt(1 - z^2/c^2) J1(k A(z)) e^{kz} dz

with ``A(z) = a sqrt(1 - z^2/c^2)``.  Substituting ``z = -c u^2`` removes the
square-root endpoints and leaves a smooth integrand on ``u in [0, 1]``, so 128
Gauss-Legendre nodes are at the machine-precision floor (verified in
``benchmarks.hull_anchor_maturation`` against a *definition* quadrature and
against the ``-ik`` identity).

Three independent evaluations of ``H(k)`` are therefore available here:

``kochin_analytic``
    Bessel closed form above (one spectral integral in ``u``).
``kochin_quadrature``
    The definition, ``2 int int y_x e^{kz} e^{ikx} dx dz``, evaluated on the
    elliptical projection with the ``r = sin(psi)`` substitution that cancels
    the ``1/sqrt(1 - r^2)`` edge singularity of ``y_x``.
``kochin_panels``
    A genuinely discrete route: constant-strength panels on the centreplane
    projection (the classical thin-ship Rankine-source discretisation, see
    ``docs/wigley_wave_resistance.md``), with either the source strength
    ``-U y_x`` (``scheme="source"``) or the ordinate ``y`` (``scheme="ordinate"``,
    exact by the ``-ik`` identity) held constant on each panel.

``kelvin_wave_resistance`` turns any of the three into a resistance through the
Kelvin-wake integral ``R_W = (4 rho / U^2) int_1^inf |H(k0 lambda)|^2 /
(lambda^2 sqrt(lambda^2 - 1)) dlambda``; the ``lambda = 1 + s^2`` substitution
(already used by ``tensorbem.wigley``) cancels the transverse-wave cusp, and the
same routine reproduces ``tensorbem.wigley.michell_wave_resistance`` when fed
the Wigley Kochin function -- that identity is the convention check.

Limitations (honest scope)
--------------------------
* Linearised free surface, deep water, steady forward speed, no sinkage/trim
  (Neumann-Kelvin, i.e. the body is frozen in its calm-water position).
* Thin-ship: ``b/a`` is assumed small; at ``b/a = 0.2`` (the study case) the
  neglected terms are ``O((b/a)^2)``, i.e. percent level on ``R_W`` -- the
  anchor therefore verifies the *numerics*, not the thin-ship modelling error.
* ``kochin_panels`` is a discretisation of the thin-ship source distribution;
  it does not solve a free-surface boundary integral equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.special import j1

__all__ = [
    "EllipsoidHull",
    "kochin_analytic",
    "kochin_quadrature",
    "kochin_panels",
    "kelvin_wave_resistance",
    "kelvin_wave_resistance_theta",
    "wave_resistance_analytic",
    "wave_resistance_panels",
    "wetted_surface_area",
    "wetted_volume_quadrature",
]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


@lru_cache(maxsize=None)
def _gauss(n: int) -> tuple[FloatArray, FloatArray]:
    """Cached Gauss-Legendre rule (node generation dominates otherwise)."""
    nodes, weights = leggauss(n)
    return nodes, weights


@dataclass(frozen=True)
class EllipsoidHull:
    """Half-submerged ellipsoid trimmed at its centre plane ``z = 0``.

    Parameters
    ----------
    a, b, c :
        Semi-length (m), semi-beam (m) and semi-draft (m).  The wetted body is
        ``x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1`` with ``z <= 0``, so the principal
        dimensions are ``L = 2a``, ``B = 2b``, ``T = c``.

    Notes
    -----
    The geometry is validated before any wave-resistance number is produced:
    ``V = 2 pi a b c / 3`` (half ellipsoid), ``Cb = V/(L B T) = pi/6``,
    ``Cm = pi/4``, ``Cp = 2/3``.  Those closed forms are the geometry gate of
    the benchmarks and tests.
    """

    a: float
    b: float
    c: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite length (got {value!r})")
        if self.b >= self.a:
            raise ValueError(
                "thin-ship theory requires the semi-beam b to be smaller than "
                f"the semi-length a (got b/a = {self.b / self.a:.3f})"
            )

    # --- principal dimensions -------------------------------------------------
    @property
    def L(self) -> float:
        return 2.0 * self.a

    @property
    def B(self) -> float:
        return 2.0 * self.b

    @property
    def T(self) -> float:
        return float(self.c)

    @property
    def thinness(self) -> float:
        """``b / a`` -- the small parameter of the thin-ship expansion."""
        return self.b / self.a

    # --- closed-form geometry (the geometry gate) ----------------------------
    @property
    def volume(self) -> float:
        """Wetted (submerged) volume ``2 pi a b c / 3``."""
        return 2.0 * math.pi * self.a * self.b * self.c / 3.0

    @property
    def waterplane_area(self) -> float:
        """``pi a b`` (ellipse of semi-axes ``a`` and ``b``)."""
        return math.pi * self.a * self.b

    @property
    def midship_section_area(self) -> float:
        """``pi b c / 2`` (half ellipse of semi-axes ``b`` and ``c``)."""
        return 0.5 * math.pi * self.b * self.c

    @property
    def Cb(self) -> float:
        return self.volume / (self.L * self.B * self.T)

    @property
    def Cm(self) -> float:
        return self.midship_section_area / (self.B * self.T)

    @property
    def Cp(self) -> float:
        return self.volume / (self.midship_section_area * self.L)

    # --- centreplane ordinate -------------------------------------------------
    def half_breadth(self, x: ArrayLike, z: ArrayLike) -> FloatArray:
        """``y = b sqrt(1 - x^2/a^2 - z^2/c^2)``; zero outside the wetted body."""
        xx, zz = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(z, dtype=float)
        )
        disc = 1.0 - (xx / self.a) ** 2 - (zz / self.c) ** 2
        return self.b * np.sqrt(np.clip(disc, 0.0, None))

    def half_breadth_x_gradient(self, x: ArrayLike, z: ArrayLike) -> FloatArray:
        """``dy/dx = -b x / (a^2 sqrt(1 - x^2/a^2 - z^2/c^2))``, zero outside.

        The gradient is unbounded at the elliptical centreplane edge; points
        outside the body (or exactly on it) are returned as zero, which is the
        convention used by the panel discretisation (a panel whose centroid
        leaves the wetted region contributes a zero source strength).
        """
        xx, zz = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(z, dtype=float)
        )
        disc = 1.0 - (xx / self.a) ** 2 - (zz / self.c) ** 2
        out = np.zeros(np.broadcast(disc, xx, zz).shape, dtype=float)
        inside = disc > 0.0
        out[inside] = -self.b * xx[inside] / (self.a**2 * np.sqrt(disc[inside]))
        return out


# ============================================================================
# Kochin function: three independent evaluations
# ============================================================================


def kochin_analytic(k: float, hull: EllipsoidHull, *, n_quad: int = 128) -> complex:
    """Closed-form/Bessel Kochin function ``H(k)`` of the ellipsoid.

    ``H(k) = -2 pi i b int_{-c}^{0} sqrt(1 - z^2/c^2) J1(k A(z)) e^{kz} dz``
    with ``A(z) = a sqrt(1 - z^2/c^2)``; the substitution ``z = -c u^2`` makes
    the integrand smooth on ``[0, 1]`` (the square-root edge factor is absorbed
    by the Jacobian and ``J1 ~ kA/2`` as ``A -> 0``), so Gauss-Legendre
    converges spectrally.
    """
    k = float(k)
    if k < 0.0:
        raise ValueError("k must be non-negative")
    if k == 0.0:
        return 0.0j

    nodes, weights = _gauss(n_quad)
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights

    sqrt_term = np.sqrt(np.clip(1.0 - u**4, 0.0, None))
    ka = k * hull.a * sqrt_term
    integrand = u * sqrt_term * j1(ka) * np.exp(-k * hull.c * u * u)
    total = float(np.sum(w * integrand))

    return -4.0j * math.pi * hull.b * hull.c * total


def kochin_quadrature(
    k: float, hull: EllipsoidHull, *, n_psi: int = 96, n_phi: int = 192
) -> complex:
    """The *definition* of ``H``: ``2 int int y_x e^{kz} e^{ikx} dx dz``.

    Parameters along the elliptical projection as ``x = a r cos(phi)``,
    ``z = -c r sin(phi)``, ``r = sin(psi)``; the ``1/sqrt(1 - r^2)`` edge
    singularity of ``y_x`` is cancelled exactly by ``dr = cos(psi) d psi``, so
    the tensor-product Gauss-Legendre rule converges spectrally.  This route
    shares no algebra with :func:`kochin_analytic` beyond the ordinate itself.
    """
    k = float(k)
    if k < 0.0:
        raise ValueError("k must be non-negative")

    g_psi, w_psi = _gauss(n_psi)
    g_phi, w_phi = _gauss(n_phi)
    psi = 0.25 * math.pi * (g_psi + 1.0)  # [0, pi/2]
    wp = 0.25 * math.pi * w_psi
    phi = 0.5 * math.pi * (g_phi + 1.0)  # [0, pi]
    wf = 0.5 * math.pi * w_phi

    sin_psi = np.sin(psi)[:, None]
    cos_phi = np.cos(phi)[None, :]
    sin_phi = np.sin(phi)[None, :]

    # x = a sin(psi) cos(phi), z = -c sin(psi) sin(phi)
    phase = np.exp(-k * hull.c * sin_psi * sin_phi + 1j * k * hull.a * sin_psi * cos_phi)
    # 2 * y_x * (a c r dr dphi) with 1/sqrt(1-r^2) * r dr -> sin^2(psi) dpsi
    integrand = -2.0 * hull.b * hull.c * sin_psi**2 * cos_phi * phase
    return complex(np.sum(wp[:, None] * wf[None, :] * integrand))


def panel_grid(
    hull: EllipsoidHull, n_r: int, n_phi: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Quadrilateral panels on the wetted centreplane projection.

    Returns ``(x_centroid, z_centroid, area, source_strength)`` flattened over
    the ``n_r x n_phi`` grid, where ``source_strength`` is the *panel-averaged*
    ``dy/dx``,

        (1/A) int int_panel dy/dx dx dz
            = -(b c / A) [sin(phi)] (psi/2 - sin(2 psi)/4)  between the corners,

    evaluated in closed form.  Radial edges are cosine-clustered towards the
    elliptical rim (``r = sin(pi/2 * xi)``), which is where ``dy/dx`` grows.
    """
    if n_r < 1 or n_phi < 1:
        raise ValueError("n_r and n_phi must be positive")
    return _panel_grid_cached(hull, int(n_r), int(n_phi))


@lru_cache(maxsize=64)
def _panel_grid_cached(
    hull: EllipsoidHull, n_r: int, n_phi: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Cached body of :func:`panel_grid` (the grid is reused for every ``k``)."""

    xi = np.linspace(0.0, 1.0, n_r + 1)
    r = np.sin(0.5 * math.pi * xi)
    psi = 0.5 * math.pi * xi
    phi = np.linspace(0.0, math.pi, n_phi + 1)
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    xc = hull.a * rr * np.cos(pp)
    zc = -hull.c * rr * np.sin(pp)

    x00, z00 = xc[:-1, :-1], zc[:-1, :-1]
    x10, z10 = xc[1:, :-1], zc[1:, :-1]
    x11, z11 = xc[1:, 1:], zc[1:, 1:]
    x01, z01 = xc[:-1, 1:], zc[:-1, 1:]

    diag1 = np.stack([x11 - x00, z11 - z00], axis=-1)
    diag2 = np.stack([x10 - x01, z10 - z01], axis=-1)
    area = 0.5 * np.abs(diag1[..., 0] * diag2[..., 1] - diag1[..., 1] * diag2[..., 0])
    x_c = 0.25 * (x00 + x10 + x11 + x01)
    z_c = 0.25 * (z00 + z10 + z11 + z01)

    psi_0, psi_1 = psi[:-1][:, None], psi[1:][:, None]
    phi_0, phi_1 = phi[:-1][None, :], phi[1:][None, :]
    antideriv = lambda p: 0.5 * p - 0.25 * np.sin(2.0 * p)  # int sin^2 dp
    int_yx = (
        -hull.b
        * hull.c
        * (np.sin(phi_1) - np.sin(phi_0))
        * (antideriv(psi_1) - antideriv(psi_0))
    )
    strength = np.where(area > 1.0e-15, int_yx / np.maximum(area, 1.0e-300), 0.0)

    return x_c.ravel(), z_c.ravel(), area.ravel(), strength.ravel()


def kochin_panels(
    k: float,
    hull: EllipsoidHull,
    *,
    n_r: int,
    n_phi: int,
    scheme: Literal["source_integrated", "source", "ordinate"] = "source_integrated",
) -> complex:
    """Kochin function from a constant-per-panel centreplane discretisation.

    ``scheme="source_integrated"`` (default)
        Constant source strength per panel equal to the *panel-averaged*
        ``dy/dx`` (closed form, see :func:`panel_grid`).  This is the consistent
        thin-ship Rankine-source panel method: the panel singularity strength is
        the surface integral of the source density over the panel, so the rim
        singularity of ``dy/dx`` is integrated, not sampled.
    ``scheme="source"``
        Panel strength proportional to ``dy/dx`` *sampled at the panel
        centroid* -- the naive first-order panel code.  It stalls on the
        elliptical rim (``dy/dx ~ 1/sqrt(1 - r^2)``): measured ``R_W`` error
        28.7 % / 15.1 % / 7.8 % at 200 / 800 / 3200 panels, against
        0.53 % / 0.12 % / 0.03 % for ``source_integrated``.
    ``scheme="ordinate"``
        Panel-constant ordinate ``y`` used through the exact identity
        ``H = -2 i k int int y e^{kz} e^{ikx}``; same panels, but the panel
        unknown is the ordinate, which also removes the rim singularity.

    The panel's plane-wave integral is evaluated with the centroid phase
    (first-order panel quadrature), which is the accuracy of a standard
    constant-strength panel code.
    """
    k = float(k)
    if k < 0.0:
        raise ValueError("k must be non-negative")

    x_c, z_c, area, strength = panel_grid(hull, n_r, n_phi)
    phase = area * np.exp(k * z_c + 1j * k * x_c)

    if scheme == "source_integrated":
        return 2.0 * complex(np.sum(strength * phase))
    if scheme == "source":
        y_x = hull.half_breadth_x_gradient(x_c, z_c)
        return 2.0 * complex(np.sum(y_x * phase))
    if scheme == "ordinate":
        y = hull.half_breadth(x_c, z_c)
        return 2.0 * (-1j * k) * complex(np.sum(y * phase))
    raise ValueError(f"unknown panel scheme {scheme!r}")


# ============================================================================
# Kelvin-wake integral -> wave resistance
# ============================================================================


def kelvin_wave_resistance(
    kochin: Callable[[float], complex],
    *,
    k0: float,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
    lambda_max: float | None = None,
) -> float:
    """``R_W = (4 rho / U^2) int_1^inf |H(k0 lambda)|^2 / (lambda^2 sqrt(lambda^2-1)) dlambda``.

    The ``lambda = 1 + s^2`` substitution (``dlambda = 2 s ds``,
    ``sqrt(lambda^2 - 1) = s sqrt(2 + s^2)``) cancels the transverse-wave cusp
    ``lambda -> 1`` exactly, so plain Gauss-Legendre on ``s`` converges fast.
    This is the same convention as
    :func:`tensorbem.wigley.michell_wave_resistance`; feeding
    :func:`tensorbem.wigley.michell_kochin_function` here reproduces that
    routine to round-off (asserted in the tests).
    """
    if k0 <= 0.0:
        raise ValueError("k0 = g/U^2 must be positive")

    if lambda_max is None:
        lam_max = 60.0
    else:
        lam_max = float(lambda_max)
    s_max = math.sqrt(max(lam_max - 1.0, 1.0e-12))
    nodes, weights = _gauss(n_quad)
    s = 0.5 * s_max * (nodes + 1.0)
    w = 0.5 * s_max * weights
    lam = 1.0 + s * s

    acc = 0.0
    for si, wi, li in zip(s, w, lam):
        hk = kochin(li * k0)
        acc += wi * abs(hk) ** 2 * 2.0 / (li * li * math.sqrt(2.0 + si * si))

    return max(4.0 * rho * k0 / g * acc, 0.0)


def wave_resistance_analytic(
    hull: EllipsoidHull,
    froude: float,
    *,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 128,
    n_kelvin: int = 256,
    lambda_max: float | None = None,
) -> float:
    """Michell/Havelock wave resistance of the ellipsoid at Froude number ``Fr``.

    ``Fr`` is based on the waterline length ``L = 2a``.
    """
    if froude <= 0.0:
        raise ValueError("froude must be positive")
    u = froude * math.sqrt(g * hull.L)
    k0 = g / (u * u)
    lam_max = lambda_max
    if lam_max is None:
        lam_max = max(5.0, 10.0 / (k0 * hull.T))
    return kelvin_wave_resistance(
        lambda kk: kochin_analytic(kk, hull, n_quad=n_quad),
        k0=k0,
        rho=rho,
        g=g,
        n_quad=n_kelvin,
        lambda_max=lam_max,
    )


def wave_resistance_panels(
    hull: EllipsoidHull,
    froude: float,
    *,
    n_r: int,
    n_phi: int,
    scheme: Literal["source_integrated", "source", "ordinate"] = "source_integrated",
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_kelvin: int = 256,
    lambda_max: float | None = None,
) -> float:
    """Wave resistance of the ellipsoid from the panel Kochin function."""
    u = froude * math.sqrt(g * hull.L)
    k0 = g / (u * u)
    lam_max = lambda_max
    if lam_max is None:
        lam_max = max(5.0, 10.0 / (k0 * hull.T))
    return kelvin_wave_resistance(
        lambda kk: kochin_panels(kk, hull, n_r=n_r, n_phi=n_phi, scheme=scheme),
        k0=k0,
        rho=rho,
        g=g,
        n_quad=n_kelvin,
        lambda_max=lam_max,
    )


def wetted_surface_area(
    hull: EllipsoidHull, *, n_psi: int = 96, n_phi: int = 192
) -> float:
    """Wetted surface area of the half ellipsoid (quadrature reference).

    ``S = 2 int int sqrt(1 + y_x^2 + y_z^2) dx dz`` over the wetted
    centreplane projection, evaluated on the ``r = sin(psi)`` grid so that the
    integrable rim singularity of ``y_x``, ``y_z`` is resolved spectrally.  No
    elementary closed form exists for the general ellipsoid.
    """
    g_psi, w_psi = leggauss(n_psi)
    g_phi, w_phi = leggauss(n_phi)
    psi = 0.25 * math.pi * (g_psi + 1.0)
    wp = 0.25 * math.pi * w_psi
    phi = 0.5 * math.pi * (g_phi + 1.0)
    wf = 0.5 * math.pi * w_phi

    sin_psi = np.sin(psi)[:, None]
    cos_psi = np.cos(psi)[:, None]
    x = hull.a * sin_psi * np.cos(phi)[None, :]
    z = -hull.c * sin_psi * np.sin(phi)[None, :]

    # gradients, computed on the open interior (rim handled by the Jacobian)
    disc = np.clip(1.0 - (x / hull.a) ** 2 - (z / hull.c) ** 2, 1.0e-300, None)
    yx = -hull.b * x / (hull.a**2 * np.sqrt(disc))
    yz = -hull.b * z / (hull.c**2 * np.sqrt(disc))
    jac = np.sqrt(1.0 + yx * yx + yz * yz)
    # 2 * jac * (a c r dr dphi), r dr = sin(psi) cos(psi) dpsi
    integrand = 2.0 * hull.a * hull.c * jac * sin_psi * cos_psi
    return float(np.sum(wp[:, None] * wf[None, :] * integrand))

def kelvin_wave_resistance_theta(
    kochin: Callable[[float], complex],
    *,
    k0: float,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
) -> float:
    """The same Kelvin-wake integral in wave-angle form.

    Substituting ``lambda = sec(theta)`` maps the cusped
    ``int_1^inf |H|^2 / (lambda^2 sqrt(lambda^2 - 1)) dlambda`` onto

        R_W = (4 rho / U^2) int_0^{pi/2} |H(k0 sec theta)|^2 cos(theta) dtheta ,

    which has a different truncation behaviour from
    :func:`kelvin_wave_resistance` (the integrand decays through
    ``cos(theta)`` instead of an explicit ``lambda_max``) and therefore
    provides an independent check of the spectral quadrature.
    """
    nodes, weights = _gauss(n_quad)
    theta = 0.25 * math.pi * (nodes + 1.0)
    w = 0.25 * math.pi * weights
    acc = 0.0
    for th, wt in zip(theta, w):
        sec = 1.0 / math.cos(th)
        hk = kochin(k0 * sec)
        acc += wt * abs(hk) ** 2 * math.cos(th)
    return max(4.0 * rho * k0 / g * acc, 0.0)


def wetted_volume_quadrature(
    hull: EllipsoidHull, *, n_psi: int = 64, n_phi: int = 128
) -> float:
    """Wetted volume from ``2 int int y dx dz`` on the spectral ``(psi, phi)`` grid.

    Used as the geometry gate of the wave-resistance study: the value must
    reproduce ``2 pi a b c / 3`` (and hence ``Cb = pi/6``) to round-off before
    any resistance number is quoted.
    """
    g_psi, w_psi = _gauss(n_psi)
    g_phi, w_phi = _gauss(n_phi)
    psi = 0.25 * math.pi * (g_psi + 1.0)
    wp = 0.25 * math.pi * w_psi
    phi = 0.5 * math.pi * (g_phi + 1.0)
    wf = 0.5 * math.pi * w_phi
    sin_psi = np.sin(psi)[:, None]
    cos_psi = np.cos(psi)[:, None]
    # 2 * y * (a c r dr dphi) with r = sin(psi), y = b cos(psi)
    integrand = 2.0 * hull.a * hull.c * hull.b * sin_psi * cos_psi * cos_psi
    return float(np.sum(wp[:, None] * wf[None, :] * integrand))
