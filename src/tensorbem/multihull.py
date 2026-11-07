"""Tandem (in-line) interference of identical thin hulls -- Rankine-panel route.

Why this module exists
----------------------
``tensorbem.wigley`` gives the Michell/Kochin wave resistance of a *single*
hull, and ``tensorbem.ellipsoid_kochin`` the same for a half-submerged
ellipsoid.  Neither can answer the multihull question that a real design study
asks immediately after the single-hull number: **how does the resistance of two
identical hulls in line depend on their spacing?**

That question has a clean answer in the same linearised (Michell) framework, and
the answer is *not* "twice the single-hull resistance".  Because the free-surface
problem is linear, the Kochin function of the pair is the superposition

    H_pair(k) = H(k) * (1 + exp(i k d)),
    |H_pair(k)|^2 = 2 |H(k)|^2 (1 + cos(k d)),

for two identical hulls with the second one displaced by ``d`` in ``+x``, and
therefore the Kelvin-wake integral gives the *interference factor*

    sigma(d) = R_W,pair(d) / (2 R_W,single)
             = int w(lam) |H(k0 lam)|^2 (1 + cos(k0 lam d)) dlam
               / int w(lam) |H(k0 lam)|^2 dlam,
    w(lam)   = 1 / (lam^2 sqrt(lam^2 - 1)),   k0 = g/U^2.

``sigma`` is the quantity a multihull design trades against (``sigma < 1`` is
beneficial, ``sigma > 1`` is detrimental), it reduces to ``sigma -> 2`` as
``d -> 0`` (two coincident hulls: four times the single resistance, halved by the
normalisation) and to ``sigma -> 1`` with an algebraically decaying oscillation
as ``d -> inf``.

What is computed here
---------------------
``CentreplanePanels`` / ``wigley_centreplane_panels``
    The classical thin-ship Rankine-source discretisation of the Wigley hull:
    constant-strength panels on the *centreplane projection* ``(x, z)``, each
    carrying the panel-averaged half-breadth slope ``y_x`` (which, for the
    parabolic Wigley hull, separates exactly in ``x`` and ``z``).  This is the
    same discretisation family as ``tensorbem.ellipsoid_kochin.kochin_panels``
    and it converges to the closed-form Michell Kochin function.

``CentreplanePanels.kochin``
    ``H(k) = 2 int int y_x exp(k z + i k x) dx dz`` evaluated as an exact
    per-panel exponential moment (the ``(x, z)`` grid is rectilinear, so the
    exponent separates and each panel moment is a product of two 1-D integrals).
    ``scheme="source_integrated"`` uses the panel-averaged slope,
    ``scheme="centroid"`` the value at the panel centre -- the second is the
    cruder of the two and is kept so that the panel error is measurable rather
    than hidden in an exact moment.

``tandem_kochin`` / ``interference_factor`` / ``interference_curve``
    The ``(1 + exp(i k d))`` superposition and the ratio above, evaluated with
    the Kelvin-wake quadrature of
    :func:`tensorbem.ellipsoid_kochin.kelvin_wave_resistance` (same convention,
    so ``sigma`` is independent of the resistance normalisation).

``explicit_pair_kochin``
    A deliberately *un-factorised* route: the panel list of both hulls is
    concatenated (the second hull's panels are physically shifted by ``d``) and
    summed with the generic panel kernel.  It must reproduce
    ``H(k)(1 + exp(i k d))`` to round-off; that identity is what licenses using
    the factorised form anywhere else.

Honest scope
------------
* Linearised free surface, deep water, steady forward speed, no sinkage/trim:
  the two hulls are frozen side by side in their *calm-water* positions, exactly
  as in the single-hull Michell anchor.  Viscous resistance is excluded by
  construction, so ``sigma`` is a *wave*-resistance interference factor.
* "In line" means a pure longitudinal offset ``d`` (no lateral offset, no
  stagger in draft).  Side-by-side (catamaran) interference needs the lateral
  wave-number phase ``exp(i k0 lam? ...)`` of the transverse system and is a
  different, harder integral; it is **not** implemented here.
* Identical hulls only.  Different hulls are a trivial extension of
  :func:`tandem_kochin` (sum two different Kochin functions) but are untested
  and therefore not offered.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .ellipsoid_kochin import kelvin_wave_resistance
from .wigley import michell_kochin_function

FloatArray = NDArray[np.float64]
KochinFunction = Callable[[float], complex]

__all__ = [
    "CentreplanePanels",
    "wigley_centreplane_panels",
    "wigley_panel_kochin",
    "explicit_pair_kochin",
    "tandem_kochin",
    "interference_factor",
    "interference_curve",
    "wave_number_k0",
    "tandem_wave_resistance",
]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


def wave_number_k0(froude: float, length: float, g: float = _DEFAULT_GRAVITY) -> float:
    """``k0 = g/U^2`` with ``U = Fr sqrt(g L)``, i.e. ``k0 = 1/(Fr^2 L)``."""
    if froude <= 0.0:
        raise ValueError("froude must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")
    u = froude * math.sqrt(g * length)
    return g / (u * u)


def _exp_moments(a: FloatArray, b: FloatArray, rate: complex) -> NDArray[np.complex128]:
    """``int_a^b exp(rate*s) ds`` elementwise, stable for ``rate -> 0``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    t = rate * d
    out = np.empty(d.shape, dtype=np.complex128)
    big = np.abs(t) >= 1.0e-8
    if np.any(big):
        out[big] = np.exp(rate * a[big]) * np.expm1(t[big]) / rate
    if np.any(~big):
        tt = t[~big]
        out[~big] = d[~big] * np.exp(rate * a[~big]) * (1.0 + 0.5 * tt + tt * tt / 6.0)
    return out


@dataclass(frozen=True)
class CentreplanePanels:
    """Constant-strength Rankine panels on a hull's centreplane projection.

    Attributes
    ----------
    x_edges, z_edges :
        Panel grid lines (``n_x + 1`` and ``n_z + 1`` stations).
    y_x :
        Panel-averaged ``d(half-breadth)/dx`` on the ``(n_x, n_z)`` grid.
    length :
        Waterline length of the hull the panels belong to (used only for
        reporting / non-dimensionalisation).
    """

    x_edges: FloatArray
    z_edges: FloatArray
    y_x: FloatArray
    length: float
    n_x: int
    n_z: int

    def __post_init__(self) -> None:
        if self.n_x < 1 or self.n_z < 1:
            raise ValueError("n_x and n_z must be positive")
        if self.x_edges.shape != (self.n_x + 1,):
            raise ValueError("x_edges must have n_x + 1 entries")
        if self.z_edges.shape != (self.n_z + 1,):
            raise ValueError("z_edges must have n_z + 1 entries")
        if self.y_x.shape != (self.n_x, self.n_z):
            raise ValueError("y_x must have shape (n_x, n_z)")

    @property
    def panel_count(self) -> int:
        return int(self.n_x * self.n_z)

    @property
    def x_centroid(self) -> FloatArray:
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    @property
    def z_centroid(self) -> FloatArray:
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def area(self) -> FloatArray:
        dx = np.diff(self.x_edges)[:, None]
        dz = np.diff(self.z_edges)[None, :]
        return np.broadcast_to(dx * dz, self.y_x.shape).copy()

    def shifted(self, offset: float) -> "CentreplanePanels":
        """The same discretisation translated by ``offset`` in ``+x``."""
        return CentreplanePanels(
            x_edges=self.x_edges + float(offset),
            z_edges=self.z_edges.copy(),
            y_x=self.y_x.copy(),
            length=self.length,
            n_x=self.n_x,
            n_z=self.n_z,
        )

    def kochin(self, k: float) -> complex:
        """``H(k) = 2 int int y_x exp(k z + i k x) dx dz`` (exact panel moments)."""
        if k <= 0.0:
            raise ValueError("k must be positive")
        ix = _exp_moments(self.x_edges[:-1], self.x_edges[1:], 1j * k)
        iz = _exp_moments(self.z_edges[:-1], self.z_edges[1:], k)
        return 2.0 * complex(np.einsum("ij,i,j->", self.y_x, ix, iz))

    def panel_sum(self, k: float) -> complex:
        """``sum_p A_p s_p exp(k z_p + i k x_p)`` -- the *point* (centroid) sum.

        Kept as a separate, deliberately non-integrated route: it is the plain
        "source at the panel centre" evaluation whose ``O(h)`` error is visible
        in the convergence study.
        """
        if k <= 0.0:
            raise ValueError("k must be positive")
        xx = self.x_centroid[:, None]
        zz = self.z_centroid[None, :]
        kernel = np.exp(k * zz + 1j * k * xx)
        return 2.0 * complex(np.sum(self.area * self.y_x * kernel))


@lru_cache(maxsize=32)
def _wigley_panels(
    L: float, B: float, T: float, n_x: int, n_z: int, scheme: str
) -> CentreplanePanels:
    x_edges = np.linspace(-0.5 * L, 0.5 * L, int(n_x) + 1)
    z_edges = np.linspace(-T, 0.0, int(n_z) + 1)

    # dy/dx = -(4 B / L^2) x (1 - (z/T)^2) is separable: the panel average of the
    # product is the product of the two exact 1-D panel averages.
    x_mean = 0.5 * (x_edges[:-1] + x_edges[1:])
    z0, z1 = z_edges[:-1], z_edges[1:]
    dz = z1 - z0
    z_parabola_mean = 1.0 - (z1**3 - z0**3) / (3.0 * T * T * dz)
    y_x = np.outer(-(4.0 * B / L**2) * x_mean, z_parabola_mean)

    if scheme == "centroid":
        z_c = 0.5 * (z0 + z1)
        y_x = -(4.0 * B / L**2) * np.outer(x_mean, 1.0 - (z_c / T) ** 2)
    elif scheme != "source_integrated":
        raise ValueError(f"unknown scheme {scheme!r}")

    return CentreplanePanels(
        x_edges=x_edges,
        z_edges=z_edges,
        y_x=y_x,
        length=L,
        n_x=int(n_x),
        n_z=int(n_z),
    )


def wigley_centreplane_panels(
    L: float,
    B: float,
    T: float,
    *,
    n_x: int,
    n_z: int,
    scheme: str = "source_integrated",
) -> CentreplanePanels:
    """Rankine-source panel discretisation of the parabolic Wigley hull.

    The hull surface is ``y(x, z) = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2)`` for
    ``x in [-L/2, L/2]``, ``z in [-T, 0]``; the source strength of the thin-ship
    formulation is proportional to ``-y_x`` and the panels are the rectangles of
    a uniform ``(x, z)`` grid.  The common factor between the source strength and
    the Kochin normalisation is the one already used by
    :func:`tensorbem.wigley.michell_kochin_function`, so the panel Kochin
    function converges to that closed form (checked in
    ``tests/test_multihull.py``).
    """
    for name, value in (("L", L), ("B", B), ("T", T)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if n_x < 1 or n_z < 1:
        raise ValueError("n_x and n_z must be >= 1")
    return _wigley_panels(float(L), float(B), float(T), int(n_x), int(n_z), str(scheme))


def wigley_panel_kochin(
    k: float,
    L: float,
    B: float,
    T: float,
    *,
    n_x: int,
    n_z: int,
    scheme: str = "source_integrated",
) -> complex:
    """Panel-discretised Wigley Kochin function at wave number ``k``."""
    return wigley_centreplane_panels(
        L, B, T, n_x=n_x, n_z=n_z, scheme=scheme
    ).kochin(float(k))


def tandem_kochin(kochin: KochinFunction, spacing: float) -> KochinFunction:
    """``H_pair(k) = H(k) (1 + exp(i k d))`` for two identical hulls in line."""
    d = float(spacing)
    if d < 0.0:
        raise ValueError("spacing must be non-negative")

    def pair(k: float) -> complex:
        return kochin(k) * (1.0 + cmath.exp(1j * k * d))

    return pair


def explicit_pair_kochin(
    k: float,
    L: float,
    B: float,
    T: float,
    *,
    spacing: float,
    n_x: int,
    n_z: int,
    scheme: str = "source_integrated",
) -> complex:
    """Pair Kochin function from the *concatenated* panels of both hulls.

    No superposition identity is used: the second hull's panel grid is shifted by
    ``spacing`` and both grids are summed with the generic per-panel kernel.  The
    result must equal :func:`tandem_kochin` applied to the single-hull Kochin
    function (to round-off), which is the identity the rest of the module relies
    on.
    """
    if spacing < 0.0:
        raise ValueError("spacing must be non-negative")
    head = wigley_centreplane_panels(L, B, T, n_x=n_x, n_z=n_z, scheme=scheme)
    tail = head.shifted(spacing)
    return head.kochin(float(k)) + tail.kochin(float(k))


def tandem_wave_resistance(
    kochin: KochinFunction,
    *,
    k0: float,
    spacing: float,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
    lambda_max: float | None = None,
) -> tuple[float, float]:
    """``(R_W,single, R_W,pair)`` at wave number ``k0`` for spacing ``spacing``.

    Both numbers come from the *same* Kelvin-wake quadrature
    (:func:`tensorbem.ellipsoid_kochin.kelvin_wave_resistance`), so the ratio is
    free of the usual ``Cw`` normalisation ambiguity.
    """
    common = dict(k0=k0, rho=rho, g=g, n_quad=n_quad, lambda_max=lambda_max)
    single = kelvin_wave_resistance(kochin, **common)
    pair = kelvin_wave_resistance(tandem_kochin(kochin, spacing), **common)
    return single, pair


def interference_factor(
    kochin: KochinFunction,
    *,
    k0: float,
    spacing: float,
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
    lambda_max: float | None = None,
) -> float:
    """``sigma(d) = R_W,pair(d) / (2 R_W,single)``.

    ``sigma(0) = 2`` exactly (two coincident hulls), ``sigma < 1`` means the
    in-line arrangement pays less than two isolated hulls.
    """
    single, pair = tandem_wave_resistance(
        kochin,
        k0=k0,
        spacing=spacing,
        rho=rho,
        g=g,
        n_quad=n_quad,
        lambda_max=lambda_max,
    )
    if single <= 0.0:
        return float("nan")
    return pair / (2.0 * single)


def interference_curve(
    kochin: KochinFunction,
    *,
    k0: float,
    spacings: Sequence[float],
    rho: float = _DEFAULT_DENSITY,
    g: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
    lambda_max: float | None = None,
) -> FloatArray:
    """``sigma`` on a spacing sweep (one Kelvin integral per spacing)."""
    return np.asarray(
        [
            interference_factor(
                kochin,
                k0=k0,
                spacing=float(d),
                rho=rho,
                g=g,
                n_quad=n_quad,
                lambda_max=lambda_max,
            )
            for d in spacings
        ],
        dtype=float,
    )


def analytic_wigley_kochin(L: float, B: float, T: float) -> KochinFunction:
    """Convenience wrapper: the closed-form Michell Kochin function of Wigley."""

    def kochin(k: float) -> complex:
        return michell_kochin_function(L, B, T, k)

    return kochin