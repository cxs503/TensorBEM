"""Michell thin-ship wave resistance for general hull forms and tandem multi-hulls.

Why this module exists
----------------------
:mod:`tensorbem.wigley` implements Michell's thin-ship theory **for the
parabolic Wigley hull only**: :func:`~tensorbem.wigley.michell_kochin_function`
contains the closed-form ``x``- and ``z``-integrals of that one hullform, and
:func:`~tensorbem.wigley.michell_wave_resistance` hard-wires them into the
Kochin resistance integral.  Consequently the platform could answer "what is
R_w for *this* Wigley hull?" but had no way to ask the same question of any
other hull, and no way at all to ask about **two hulls in each other's wave
field** -- the tandem interference problem that dominates multi-hull wave
resistance work.  This module supplies both, without touching the verified
Wigley path: it reduces the theory to the two ingredients that are actually
hullform-independent,

    H(k)  = 2 * int int_S  dy/dx * exp(k z + i k x) dx dz        (Kochin)
    R_w   = (4 rho / U^2) * int_1^inf |H(lambda k_0)|^2 /
                                (lambda^2 sqrt(lambda^2 - 1)) dlambda,

and parameterises the hull by its **centreplane source distribution**
``2 dy/dx dA`` sampled on a panel mesh.  Any hullform for which
``y(x, z)`` (the half-breadth) is available becomes a panel list, the Kochin
function is one matrix-vector product, and the resistance is a fixed
one-dimensional quadrature that only depends on the draft ``T`` through its
exponential decay length.

Conventions (identical to :mod:`tensorbem.wigley`, checked in the tests)
-----------------------------------------------------------------------
* Length ``L`` along ``x``, beam ``B`` across ``y``, draft ``T`` down ``z``,
  calm water at ``z = 0``, hull in ``z in [-T, 0]``, both sides mirrored about
  ``y = 0`` (hence the factor 2 in ``H``).
* Steady forward speed ``U``, Neumann-Kelvin linearisation, deep water,
  ``k_0 = g/U^2``, Froude number ``Fn = U/sqrt(g L)``.
* ``R_w`` is normalised by ``0.5 rho U^2 L T`` (the same reference used by
  :class:`~tensorbem.wigley.WigleyWaveResistanceResult`), i.e.
  ``Cw = R_w / (0.5 rho U^2 L T)``.

Tandem multi-hull interference
------------------------------
For two *identical* hulls with the second displaced a distance ``s``
downstream, linear theory superposes the two centreplane source
distributions, so the pair Kochin function is

    H_pair(k) = H(k) * (1 + exp(i k s)),

and the pair resistance and the per-hull interference factor are

    R_pair = (4 rho / U^2) int |H_pair|^2 w(lambda) dlambda,
    sigma  = R_pair / (2 R_single)
           = 1 + [int |H|^2 cos(k s) w dlambda] / [int |H|^2 w dlambda],

with ``k = lambda k_0``.  ``sigma`` therefore lies in ``[0, 2]``: ``sigma -> 2``
as ``s -> 0`` (the hulls coincide, each sees a doubled wave system) and
``sigma -> 1`` as ``s -> inf`` (decoupled, the cosine integral averages out).
The oscillation of ``sigma(s)`` is the classical thin-ship tandem interference
result; :func:`tandem_interference_factor` evaluates it from the closed-form
superposition, and :func:`tandem_pair_resistance` evaluates the *same* quantity
by merging the two panel sets -- the two routes are independent in the sense
that the second never uses the factor ``(1 + exp(i k s))``.

What is *not* claimed
---------------------
Everything here is inviscid thin-ship theory.  It contains no viscous
resistance, no wave breaking, no hull-to-hull blockage beyond linear
superposition, and it is only asymptotically valid for small
``B/L``, ``T/L`` and small wave steepness.  Agreement between the panel route
and the quadrature route validates an *implementation*; it says nothing about
how well Michell's theory matches experiment, which for full hulls is known to
be poor (typically tens of percent on absolute resistance).  See
``docs/michell_validation.md`` for the verification surface and its limits.

References
----------
- Michell, J.H. (1898). "The wave resistance of a ship."
  Philosophical Magazine, 45, 106-120.
- Havelock, T.H. (1932). "The theory of wave resistance."
  Proc. R. Soc. A, 138, 339-348.
- Tuck, E.O. (1989). "The wave resistance formula of J.H. Michell (1898) and
  its significance." J. Eng. Math., 23, 1-15.
- Tuck, E.O. & Newman, J.N. (1975). "Interference of wave resistance of two
  slender hulls in tandem." (thin-ship superposition of Kochin functions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss

from .wigley import michell_kochin_function

DEFAULT_GRAVITY = 9.80665
DEFAULT_DENSITY = 1025.0

#: Number of e-folding lengths ``1/(k0 T)`` kept in the resistance integral.
#: The integrand carries ``|H|^2 ~ exp(-2 k0 lambda T)``, so the truncation
#: error at ``lambda_max = DEPTH_DECADES / (k0 T)`` is below ``exp(-2 *
#: DEPTH_DECADES) ~ 2e-9`` relative for any hull whose Kochin spectrum decays
#: no slower than the draft scale (all hulls here do).
DEPTH_DECADES = 10.0

#: Cap on the number of panel-by-node products held in memory at once in
#: :meth:`CenterplanePanels.kochin` (16 bytes per complex element).
_CHUNK_ELEMENTS = 2_000_000

__all__ = [
    "CenterplanePanels",
    "DEFAULT_DENSITY",
    "DEFAULT_GRAVITY",
    "DEPTH_DECADES",
    "default_lambda_max",
    "ellipsoid_hydrostatic_reference",
    "ellipsoid_kochin_reference",
    "ellipsoid_panels",
    "ellipsoid_wave_resistance",
    "froude_number",
    "canonical_resistance_nodes",
    "ellipsoid_wave_resistance_canonical",
    "michell_resistance",
    "michell_resistance_canonical",
    "resistance_nodes",
    "speed_for_froude",
    "tandem_interference_factor",
    "tandem_interference_factor_canonical",
    "tandem_pair_resistance",
    "tandem_pair_resistance_canonical",
    "wave_resistance_from_panels",
    "wave_resistance_from_panels_canonical",
    "wigley_kochin_closed_form",
    "wigley_panels",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def froude_number(U: float, L: float, gravity: float = DEFAULT_GRAVITY) -> float:
    """``Fn = U / sqrt(g L)``."""
    return U / math.sqrt(gravity * L)


def speed_for_froude(
    froude: float, L: float, gravity: float = DEFAULT_GRAVITY
) -> float:
    """``U = Fn sqrt(g L)``."""
    return froude * math.sqrt(gravity * L)


def _tensor_nodes(a: float, b: float, n: int, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """One-dimensional node/weight pair on ``[a, b]``.

    ``rule='midpoint'`` gives ``n`` equal cells sampled at their centres with
    weight = cell size (a genuine panel discretisation, second order);
    ``rule='gauss'`` gives an ``n``-point Gauss-Legendre rule (spectral for
    smooth integrands, used for the reference and for the discretisation-error
    series).
    """
    if n < 1:
        raise ValueError("need n >= 1")
    if rule == "midpoint":
        edges = np.linspace(a, b, n + 1)
        return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
    if rule == "gauss":
        x, w = leggauss(n)
        return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w
    raise ValueError(f"unknown rule {rule!r}")


def default_lambda_max(k0: float, T: float, decades: float = DEPTH_DECADES) -> float:
    """Truncation point of the resistance integral, ``max(5, decades/(k0 T))``."""
    if k0 <= 0.0 or T <= 0.0:
        return 5.0
    return max(5.0, decades / (k0 * T))


# --------------------------------------------------------------------------- #
# Centreplane source distribution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CenterplanePanels:
    """Discrete centreplane source distribution of a thin hull.

    Attributes
    ----------
    x, z :
        Panel-centre coordinates on the hull centreplane, ``x in [-L/2, L/2]``,
        ``z in [-T, 0]``.
    weight :
        ``2 * (dy/dx) * area`` per panel, i.e. the source strength of *both*
        sides times the projected area element.  The Kochin function is then
        the single sum

            ``H(k) = sum_p weight_p * exp(k z_p + i k x_p)``,

        which is exactly the closed form used by :mod:`tensorbem.wigley`
        evaluated by quadrature instead of analytically.
    area :
        Projected (centreplane) area of each panel, kept separately so that the
        discretised footprint can be validated against ``L*T`` (Wigley) or
        ``pi*L*T/4`` (ellipsoid) before any physics is trusted.
    label :
        Free-form provenance tag.
    """

    x: np.ndarray
    z: np.ndarray
    weight: np.ndarray
    area: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        x = np.ascontiguousarray(np.asarray(self.x, dtype=float).ravel())
        z = np.ascontiguousarray(np.asarray(self.z, dtype=float).ravel())
        w = np.ascontiguousarray(np.asarray(self.weight, dtype=float).ravel())
        a = np.ascontiguousarray(np.asarray(self.area, dtype=float).ravel())
        if not (x.size == z.size == w.size == a.size):
            raise ValueError("x, z, weight and area must have the same length")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "weight", w)
        object.__setattr__(self, "area", a)

    @property
    def panel_count(self) -> int:
        return int(self.x.size)

    @property
    def projected_area(self) -> float:
        """Sum of the projected panel areas (footprint of the wetted hull)."""
        return float(np.sum(self.area))

    def kochin(self, k):
        """Kochin function ``H(k)`` for a scalar or an array of wavenumbers."""
        kk = np.ascontiguousarray(np.asarray(k, dtype=float).ravel())
        out = np.zeros(kk.size, dtype=complex)
        chunk = max(1, int(_CHUNK_ELEMENTS // max(self.panel_count, 1)))
        for start in range(0, kk.size, chunk):
            kc = kk[start : start + chunk]
            phase = np.exp(np.outer(kc, self.z) + 1j * np.outer(kc, self.x))
            out[start : start + chunk] = phase @ self.weight
        return out if np.ndim(k) else out[0]

    def kochin_pair(self, k_depth, alpha):
        """Canonical Kochin amplitude ``H(k_depth, alpha)``.

        The steady free-surface condition ``U^2 phi_xx + g phi_z = 0`` admits a
        wave of the form ``exp(k_depth z + i alpha x)`` only on the pair
        ``(k_depth, alpha) = (alpha^2/k0, alpha)``, ``k0 = g/U^2``.  The
        single-argument :meth:`kochin` forces ``k_depth = alpha``, which is why
        it is only valid as a *diagnostic* amplitude, not inside the resistance
        integral (see :func:`michell_resistance_canonical`).  This method
        evaluates the two wavenumbers independently::

            H(k_depth, alpha) = sum_p weight_p exp(k_depth z_p + i alpha x_p)

        ``k_depth`` and ``alpha`` broadcast against each other; equal-length
        arrays are evaluated pair-by-pair (that is what the resistance
        quadrature needs), scalars give a complex scalar back.
        """
        kd = np.asarray(k_depth, dtype=float)
        al = np.asarray(alpha, dtype=float)
        scalar = kd.ndim == 0 and al.ndim == 0
        kd_b, al_b = np.broadcast_arrays(kd, al)
        shape = kd_b.shape
        kd_f = np.ascontiguousarray(kd_b, dtype=float).ravel()
        al_f = np.ascontiguousarray(al_b, dtype=float).ravel()
        out = np.zeros(kd_f.size, dtype=complex)
        chunk = max(1, int(_CHUNK_ELEMENTS // max(self.panel_count, 1)))
        for start in range(0, kd_f.size, chunk):
            kc = kd_f[start : start + chunk]
            ac = al_f[start : start + chunk]
            phase = np.exp(np.outer(kc, self.z) + 1j * np.outer(ac, self.x))
            out[start : start + chunk] = phase @ self.weight
        if scalar:
            return complex(out[0])
        return out.reshape(shape)

    def shifted(self, dx: float) -> "CenterplanePanels":
        """The same hull displaced a distance ``dx`` downstream (``+x``)."""
        return CenterplanePanels(
            self.x + float(dx), self.z, self.weight, self.area,
            label=f"{self.label}+{dx:g}",
        )

    def merged(self, other: "CenterplanePanels") -> "CenterplanePanels":
        """Superpose two hulls' source distributions (linear superposition)."""
        return CenterplanePanels(
            np.concatenate([self.x, other.x]),
            np.concatenate([self.z, other.z]),
            np.concatenate([self.weight, other.weight]),
            np.concatenate([self.area, other.area]),
            label=f"{self.label}|{other.label}",
        )


# --------------------------------------------------------------------------- #
# Hullform-specific panel builders
# --------------------------------------------------------------------------- #
def wigley_panels(
    L: float, B: float, T: float, n_x: int = 32, n_z: int = 16, rule: str = "midpoint"
) -> CenterplanePanels:
    """Centreplane panels of the parabolic Wigley hull.

    ``y(x, z) = (B/2) (1 - (2x/L)^2) (1 - (z/T)^2)`` so that
    ``dy/dx = -(4B/L^2) x (1 - (z/T)^2)``; the ``(x, z)`` integrand is a
    polynomial times an exponential, hence smooth and spectrally convergent
    with ``rule='gauss'``.
    """
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")
    xs, wx = _tensor_nodes(-0.5 * L, 0.5 * L, n_x, rule)
    zs, wz = _tensor_nodes(-T, 0.0, n_z, rule)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    W = np.outer(wx, wz)
    slope = -(4.0 * B / L**2) * X * (1.0 - (Z / T) ** 2)
    return CenterplanePanels(
        X.ravel(), Z.ravel(), (2.0 * slope * W).ravel(), W.ravel(),
        label=f"wigley/{rule}/{n_x}x{n_z}",
    )


def ellipsoid_panels(
    L: float, B: float, T: float, n_psi: int = 48, n_t: int = 96,
    rule: str = "gauss",
) -> CenterplanePanels:
    """Centreplane panels of a three-axis ellipsoid hull.

    The wetted hull of the ellipsoid ``(x/a)^2 + (y/b)^2 + (z/c)^2 = 1`` with
    ``a = L/2``, ``b = B/2``, ``c = T`` and the calm waterplane at ``z = 0`` is
    the lower half; its half-breadth is

        ``y(x, z) = b * sqrt(1 - (x/a)^2 - (z/c)^2)``.

    The projected footprint is a half ellipse, so a rectangular ``(x, z)`` mesh
    would have to resolve a ``1/sqrt(1 - r^2)`` slope singularity at the
    waterline/keel boundary.  Instead the panels are laid out in the elliptic
    coordinates ``(psi, t)`` with ``r = sin(psi)``, ``psi in [0, pi/2]`` and

        ``x = -a r cos(pi t),  z = -c r sin(pi t),  t in [0, 1]``,

    for which the jacobian ``pi a c sin(psi) cos(psi)`` cancels the
    ``1/cos(psi)`` in ``dy/dx`` exactly: the panel weight
    ``2 dy/dx dA = -2 pi b c x sin(psi) dpsi dt / a`` is analytic, so both the
    ``midpoint`` (second-order panel) and the ``gauss`` (spectral) rule
    converge cleanly.  This is the "geometry first" panel mesh of a real
    diffraction program: the body surface is meshed, not a bounding box.
    """
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")
    a, b, c = 0.5 * L, 0.5 * B, float(T)
    psis, wpsi = _tensor_nodes(0.0, 0.5 * math.pi, n_psi, rule)
    ts, wt = _tensor_nodes(0.0, 1.0, n_t, rule)
    P, Tt = np.meshgrid(psis, ts, indexing="ij")
    W = np.outer(wpsi, wt)
    r = np.sin(P)
    x = -a * r * np.cos(math.pi * Tt)
    z = -c * r * np.sin(math.pi * Tt)
    area = math.pi * a * c * np.sin(P) * np.cos(P)
    slope = -b * x / (a * a * np.cos(P))
    return CenterplanePanels(
        x.ravel(), z.ravel(), (2.0 * slope * area * W).ravel(),
        (area * W).ravel(), label=f"ellipsoid/{rule}/{n_psi}x{n_t}",
    )


def ellipsoid_hydrostatic_reference(L: float, B: float, T: float) -> dict[str, float]:
    """Closed-form hydrostatics of the (half-)ellipsoid hull.

    ``nabla = (2/3) pi a b c = pi L B T / 12`` and ``Cb = pi/12``
    (``= 0.2618``, against ``4/9 = 0.4444`` for the Wigley hull), waterplane
    area ``pi L B / 4``.  Used to validate the panel footprint before any wave
    resistance is computed.
    """
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")
    volume = math.pi * L * B * T / 12.0
    return {
        "L": float(L),
        "B": float(B),
        "T": float(T),
        "volume": volume,
        "Cb": volume / (L * B * T),
        "waterplane_area": math.pi * L * B / 4.0,
        "projected_area": math.pi * L * T / 4.0,
    }


def _wigley_kochin_terms(kk: np.ndarray, L: float, T: float):
    """Vectorised, cancellation-free ``x`` and ``z`` integrals of the Wigley hull.

    With ``a = L/2`` and ``x = k a``,

        ``Ix = int_{-a}^{a} x exp(i k x) dx = 2 i a^2 (sin x - x cos x) / x^2``,

    and with ``eps = k T``, ``u = -z/T``,

        ``Iz = int_{-T}^{0} (1 - (z/T)^2) exp(k z) dz = T int_0^1 (1-u^2)
        exp(-eps u) du``.

    Both radicands are written in a form that loses no digits at small argument:
    ``(sin x - x cos x)/x^2`` is replaced by its Maclaurin series ``x/3 - x^3/30
    + x^5/840`` below ``|x| = 0.5``, and the ``z`` integral by the series
    ``2 T sum_n (-eps)^n / n! / ((n+1)(n+3))`` below ``|eps| = 0.5``.  The naive
    closed forms subtract ``O(1/eps^3)`` numbers to produce an ``O(1)`` result
    and lose about four digits at ``eps = 1e-3``; the resistance integral of the
    platform never goes below ``eps ~ 0.4`` (``k >= k0``), but the Kochin
    function is a public entry point.
    """
    a = 0.5 * L
    x = kk * a
    small_x = np.abs(x) < 0.5
    x_safe = np.where(small_x, 1.0, x)
    f = (np.sin(x_safe) - x_safe * np.cos(x_safe)) / x_safe**2
    series = x / 3.0 - x**3 / 30.0 + x**5 / 840.0 - x**7 / 45360.0
    f = np.where(small_x, series, f)
    ix = 2j * a * a * f

    eps = kk * T
    small_e = np.abs(eps) < 0.5
    eps_safe = np.where(small_e, 1.0, eps)
    iz = T * (
        (1.0 - np.exp(-eps_safe)) / eps_safe
        - (2.0 / eps_safe**3
           - np.exp(-eps_safe) * (1.0 / eps_safe + 2.0 / eps_safe**2
                                  + 2.0 / eps_safe**3))
    )
    n = np.arange(0, 24, dtype=float)
    coeff = 2.0 / ((n + 1.0) * (n + 3.0))
    terms = coeff[None, :] * np.power(-eps[:, None], n[None, :]) / _FACTORIAL[None, :]
    iz_series = T * np.sum(terms, axis=1)
    iz = np.where(small_e, iz_series, iz)
    return ix, iz


_FACTORIAL = np.array([math.factorial(i) for i in range(24)], dtype=float)


def wigley_kochin_closed_form(L: float, B: float, T: float, k):
    """Reference ``H(k)`` of the Wigley hull: the verified closed form.

    Vectorised evaluation of ``H(k) = 2 (-4B/L^2) Ix(k) Iz(k)``, i.e. the closed
    form of :func:`tensorbem.wigley.michell_kochin_function` (which builds the
    ``x`` and ``z`` integrals term by term for one ``k`` at a time).  The two are
    pinned to round-off against each other in
    ``tests/test_michell_thin_ship.py``; the vectorised version exists because
    the tandem interference sweep evaluates ``H`` on ~10^4 wavenumbers per
    spacing.
    """
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B and T must be positive")
    kk = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
    ix, iz = _wigley_kochin_terms(kk, L, T)
    H = 2.0 * (-4.0 * B / L**2) * ix * iz
    return H if np.ndim(k) else complex(H[0])


def ellipsoid_kochin_reference(
    L: float, B: float, T: float, k, n_psi: int = 96, n_t: int = 384
) -> complex:
    """High-order reference ``H(k)`` of the ellipsoid hull.

    Spectral Gauss-Legendre rule in the elliptic coordinates; ``n_t`` must
    exceed ``~ pi k a`` to resolve the ``exp(i k a sin(psi) cos(pi t))``
    oscillation (checked by doubling in the tests).
    """
    panels = ellipsoid_panels(L, B, T, n_psi=n_psi, n_t=n_t, rule="gauss")
    return panels.kochin(k)


# --------------------------------------------------------------------------- #
# Michell resistance integral
# --------------------------------------------------------------------------- #
def resistance_nodes(
    U: float, T: float, gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None, n_seg: int = 8, n_quad: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes/weights for ``int_1^inf f(lambda)/(lambda^2 sqrt(lambda^2-1)) dlambda``.

    The integrand is singular at ``lambda = 1`` (the transverse-wave cut-off),
    so the substitution ``lambda = 1 + s^2`` is applied exactly as in
    :func:`tensorbem.wigley.michell_wave_resistance`:

        ``dlambda = 2 s ds``, ``sqrt(lambda^2 - 1) = s sqrt(2 + s^2)``,

    which cancels the singularity and leaves the smooth weight
    ``w = 2 / (lambda^2 sqrt(2 + s^2))``.  The truncated ``s`` interval is split
    into ``n_seg`` Gauss-Legendre panels of ``n_quad`` nodes because the
    *tandem* integrand carries ``cos(lambda k0 s)`` and is oscillatory over the
    same range; ``n_seg = 8`` is already ample for a single hull.

    Returns
    -------
    (lambda, weight) : ndarray
        Wavenumber ratios ``k/k0`` and the corresponding quadrature weights of
        the resistance integral.
    """
    if U <= 0:
        raise ValueError("speed U must be positive")
    if T <= 0:
        raise ValueError("draft T must be positive")
    if n_seg < 1 or n_quad < 1:
        raise ValueError("need n_seg >= 1 and n_quad >= 1")
    k0 = gravity / U**2
    lam_max = default_lambda_max(k0, T) if lambda_max is None else float(lambda_max)
    if lam_max <= 1.0:
        raise ValueError("lambda_max must exceed 1")
    s_max = math.sqrt(lam_max - 1.0)

    x, w = leggauss(n_quad)
    edges = np.linspace(0.0, s_max, n_seg + 1)
    half = 0.5 * np.diff(edges)                       # (n_seg,)
    centres = 0.5 * (edges[:-1] + edges[1:])
    s = (centres[:, None] + half[:, None] * x[None, :]).ravel()
    ws = (half[:, None] * w[None, :]).ravel()

    lam = 1.0 + s * s
    weight = ws * 2.0 / (lam**2 * np.sqrt(2.0 + s * s))
    return lam, weight


def michell_resistance(
    kochin: Callable[[np.ndarray], np.ndarray],
    U: float,
    T: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> float:
    """Michell wave resistance from a Kochin-function callable.

    ``R_w = (4 rho / U^2) * int_1^inf |H(lambda k0)|^2 / (lambda^2
    sqrt(lambda^2 - 1)) dlambda``, evaluated with
    :func:`resistance_nodes`.  ``kochin`` must accept an array of wavenumbers
    (both the panel method and the Wigley closed form do).
    """
    lam, weight = resistance_nodes(
        U, T, gravity=gravity, lambda_max=lambda_max, n_seg=n_seg, n_quad=n_quad
    )
    k0 = gravity / U**2
    H = np.asarray(kochin(lam * k0), dtype=complex)
    return max((4.0 * rho / U**2) * float(np.sum(weight * np.abs(H) ** 2)), 0.0)


def wave_resistance_from_panels(
    panels: CenterplanePanels,
    U: float,
    T: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> float:
    """Michell wave resistance of a panel source distribution."""
    return michell_resistance(
        lambda k: panels.kochin(k), U, T, rho=rho, gravity=gravity,
        lambda_max=lambda_max, n_seg=n_seg, n_quad=n_quad,
    )


def canonical_resistance_nodes(
    U: float, T: float, gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None, n_seg: int = 8, n_quad: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes/weights of the **canonical** Michell integral.

    Canonical (steady free-surface) form, the same one used by
    :func:`tensorbem.wigley.michell_wave_resistance_canonical`::

        Rw = (rho g^2 / (pi U^2)) * int_1^inf lam^2 |H(k0 lam^2, k0 lam)|^2
                                            / sqrt(lam^2 - 1) dlam.

    With ``lam = 1 + s^2`` the ``1/sqrt(lam^2 - 1)`` endpoint cancels exactly
    (``dlam = 2 s ds``, ``sqrt(lam^2-1) = s sqrt(2+s^2)``) and the returned
    weight is ``2 lam^2 / sqrt(2 + s^2) * (Gauss weight)``, so the caller only
    has to form ``sum(weight * |H|^2)``.

    Returns
    -------
    (lam, weight) : ndarray
        Wavenumber ratios ``k/k0`` and the weights of the canonical integral
        (already including the endpoint-cancelling Jacobian).
    """
    if U <= 0:
        raise ValueError("speed U must be positive")
    if T <= 0:
        raise ValueError("draft T must be positive")
    if n_seg < 1 or n_quad < 1:
        raise ValueError("need n_seg >= 1 and n_quad >= 1")
    k0 = gravity / U**2
    lam_max = default_lambda_max(k0, T) if lambda_max is None else float(lambda_max)
    if lam_max <= 1.0:
        raise ValueError("lambda_max must exceed 1")
    s_max = math.sqrt(lam_max - 1.0)

    x, w = leggauss(n_quad)
    edges = np.linspace(0.0, s_max, n_seg + 1)
    half = 0.5 * np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    s = (centres[:, None] + half[:, None] * x[None, :]).ravel()
    ws = (half[:, None] * w[None, :]).ravel()

    lam = 1.0 + s * s
    weight = ws * 2.0 * lam**2 / np.sqrt(2.0 + s * s)
    return lam, weight


def michell_resistance_canonical(
    kochin_pair: Callable[[np.ndarray, np.ndarray], np.ndarray],
    U: float,
    T: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> float:
    """Michell wave resistance on the canonical spectrum, for a general hull.

    This is the hullform-independent counterpart of
    :func:`tensorbem.wigley.michell_wave_resistance_canonical`: the same
    physics, driven by a **two-argument** Kochin callable
    ``H(k_depth, alpha)`` (see :meth:`CenterplanePanels.kochin_pair`) instead of
    the Wigley closed form, so centreplane panel hulls, ellipsoids and merged
    tandem pairs all get the correct spectrum.

    Why the legacy :func:`michell_resistance` is not enough
    -----------------------------------------------------
    :func:`michell_resistance` samples ``H`` at a single wavenumber
    ``k = alpha = k0 lam`` and weighs it with ``lam^-2``.  That integrand
    violates the steady free-surface condition pointwise by ``lam - 1``
    (50 % at ``lam = 2``, 89 % at ``lam = 9``) and the prefactor ``4 rho/U^2``
    is dimensionally not a force; measured against this routine the legacy
    panel route is low by a factor ~12-24 on the Wigley hull.  The legacy
    routine is kept (pinned numbers depend on it) but must not be used for new
    work -- see ``docs/michell_canonical_panel_route/README.md``.

    Parameters
    ----------
    kochin_pair :
        ``f(k_depth, alpha)`` for equal-length float arrays, returning the
        complex amplitude ``sum_p weight_p exp(k_depth z_p + i alpha x_p)``.
    """
    lam, weight = canonical_resistance_nodes(
        U, T, gravity=gravity, lambda_max=lambda_max, n_seg=n_seg, n_quad=n_quad
    )
    k0 = gravity / U**2
    alpha = lam * k0
    k_depth = alpha * alpha / k0                     # = k0 lam^2, canonical pair
    H = np.asarray(kochin_pair(k_depth, alpha), dtype=complex)
    if H.shape != lam.shape:
        raise ValueError(
            f"kochin_pair returned shape {H.shape}, expected {lam.shape}"
        )
    value = (rho * gravity**2 / (math.pi * U**2)) * float(np.sum(weight * np.abs(H) ** 2))
    return max(value, 0.0)


def wave_resistance_from_panels_canonical(
    panels: CenterplanePanels,
    U: float,
    T: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> float:
    """Canonical-spectrum wave resistance of a centreplane panel hull."""
    return michell_resistance_canonical(
        panels.kochin_pair, U, T, rho=rho, gravity=gravity,
        lambda_max=lambda_max, n_seg=n_seg, n_quad=n_quad,
    )


def ellipsoid_wave_resistance_canonical(
    L: float,
    B: float,
    T: float,
    U: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_psi: int = 96,
    n_t: int = 384,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> float:
    """Canonical-spectrum wave resistance of the ellipsoid hull (spectral panels)."""
    panels = ellipsoid_panels(L, B, T, n_psi=n_psi, n_t=n_t, rule="gauss")
    return wave_resistance_from_panels_canonical(
        panels, U, T, rho=rho, gravity=gravity, lambda_max=lambda_max,
        n_seg=n_seg, n_quad=n_quad,
    )


def ellipsoid_wave_resistance(
    L: float,
    B: float,
    T: float,
    U: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_psi: int = 96,
    n_t: int = 384,
    lambda_max: float | None = None,
) -> float:
    """Ellipsoid wave resistance from the high-order reference Kochin rule."""
    return michell_resistance(
        lambda k: ellipsoid_kochin_reference(L, B, T, k, n_psi=n_psi, n_t=n_t),
        U, T, rho=rho, gravity=gravity, lambda_max=lambda_max,
    )


# --------------------------------------------------------------------------- #
# Tandem multi-hull interference
# --------------------------------------------------------------------------- #
def _auto_segments(k0: float, lambda_max: float, separation: float, n_seg: int) -> int:
    """Segments needed to resolve ``cos(k0 lambda separation)`` oscillations."""
    if separation <= 0.0:
        return n_seg
    phase_span = (lambda_max - 1.0) * k0 * separation
    return max(n_seg, int(math.ceil(phase_span / (2.0 * math.pi))) * 4)


def tandem_interference_factor(
    kochin: Callable[[np.ndarray], np.ndarray],
    U: float,
    T: float,
    separation: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> dict[str, float]:
    """Interference factor ``sigma = R_pair / (2 R_single)`` of a tandem pair.

    Uses the superposition ``H_pair = H (1 + exp(i k s))``, which gives

        ``sigma = 1 + C/N``,
        ``C = int |H|^2 cos(k s) w dlambda``, ``N = int |H|^2 w dlambda``,

    with the *same* quadrature nodes for both integrals and the segment count
    raised automatically so that the ``cos(k s)`` oscillation
    (``(lambda_max - 1) k0 s`` radians of phase) is resolved.

    Returns
    -------
    dict
        ``sigma``, ``r_single``, ``r_pair``, ``interference`` (``R_pair -
        2 R_single``, signed) and the quadrature description.
    """
    if separation < 0:
        raise ValueError("separation must be non-negative")
    k0 = gravity / U**2
    lam_max = default_lambda_max(k0, T) if lambda_max is None else float(lambda_max)
    n_seg_used = _auto_segments(k0, lam_max, separation, n_seg)
    lam, weight = resistance_nodes(
        U, T, gravity=gravity, lambda_max=lam_max, n_seg=n_seg_used, n_quad=n_quad
    )
    H = np.asarray(kochin(lam * k0), dtype=complex)
    spectrum = np.abs(H) ** 2 * weight
    norm = float(np.sum(spectrum))
    scale = 4.0 * rho / U**2
    r_single = max(scale * norm, 0.0)
    cosine = float(np.sum(spectrum * np.cos(lam * k0 * separation)))
    r_pair = max(scale * (2.0 * norm + 2.0 * cosine), 0.0)
    sigma = r_pair / (2.0 * r_single) if r_single > 0.0 else float("nan")
    return {
        "sigma": sigma,
        "r_single": r_single,
        "r_pair": r_pair,
        "interference": r_pair - 2.0 * r_single,
        "separation": float(separation),
        "lambda_max": lam_max,
        "n_seg": n_seg_used,
        "n_quad": n_quad,
        "nodes": int(lam.size),
    }


def tandem_pair_resistance(
    panels: CenterplanePanels,
    U: float,
    T: float,
    separation: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> float:
    """Pair resistance of a tandem pair by merging the two panel sets.

    Independent of :func:`tandem_interference_factor`: the two source
    distributions are concatenated (:meth:`CenterplanePanels.merged`) and the
    resistance integral is applied to the merged list, so the ``(1 + exp(i k
    s))`` factor is never formed explicitly.
    """
    if separation < 0:
        raise ValueError("separation must be non-negative")
    merged = panels.merged(panels.shifted(separation))
    k0 = gravity / U**2
    lam_max = default_lambda_max(k0, T) if lambda_max is None else float(lambda_max)
    n_seg_used = _auto_segments(k0, lam_max, separation, n_seg)
    return wave_resistance_from_panels(
        merged, U, T, rho=rho, gravity=gravity, lambda_max=lam_max,
        n_seg=n_seg_used, n_quad=n_quad,
    )


def tandem_interference_factor_canonical(
    kochin_pair: Callable[[np.ndarray, np.ndarray], np.ndarray],
    U: float,
    T: float,
    separation: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> dict[str, float]:
    """Tandem interference factor ``sigma`` on the canonical spectrum.

    Same definition as :func:`tandem_interference_factor`
    (``sigma = R_pair / (2 R_single)``, so ``sigma in [0, 2]``) but with both
    resistances evaluated through
    :func:`wave_resistance_from_panels_canonical`, i.e. with the canonical
    ``lam^2 / sqrt(lam^2-1)`` weight.  The legacy factor weights the same
    Kochin product with ``lam^-2``, so the two factors are **not** the same
    number for the same gap: the canonically weighted spectrum emphasises the
    long transverse waves (``lam -> 1``) that carry the interference, and the
    shift is reported in ``docs/michell_canonical_panel_route``.

    ``kochin_pair`` is a two-argument Kochin callable (e.g.
    :meth:`CenterplanePanels.kochin_pair`); the single-hull reference is the
    same callable, so ``sigma`` carries no discretisation bias of its own.
    """
    if separation < 0:
        raise ValueError("separation must be non-negative")
    k0 = gravity / U**2
    lam_max = default_lambda_max(k0, T) if lambda_max is None else float(lambda_max)
    n_seg_used = _auto_segments(k0, lam_max, separation, n_seg)
    single = michell_resistance_canonical(
        kochin_pair, U, T, rho=rho, gravity=gravity, lambda_max=lam_max,
        n_seg=n_seg_used, n_quad=n_quad,
    )
    lam, weight = canonical_resistance_nodes(
        U, T, gravity=gravity, lambda_max=lam_max,
        n_seg=n_seg_used, n_quad=n_quad,
    )
    k0_ = gravity / U**2
    alpha = lam * k0_
    k_depth = alpha * alpha / k0_
    shifted = lambda kd, al: kochin_pair(kd, al) * (1.0 + np.exp(1j * al * float(separation)))
    H_pair = np.asarray(shifted(k_depth, alpha), dtype=complex)
    pair = (rho * gravity**2 / (math.pi * U**2)) * float(
        np.sum(weight * np.abs(H_pair) ** 2)
    )
    if single <= 0.0:
        raise ValueError("single-hull resistance must be positive")
    return {
        "sigma": pair / (2.0 * single),
        "r_single": single,
        "r_pair": pair,
        "interference": pair - 2.0 * single,
        "separation": float(separation),
        "lambda_max": lam_max,
        "n_seg": n_seg_used,
        "n_quad": n_quad,
    }


def tandem_pair_resistance_canonical(
    panels: CenterplanePanels,
    U: float,
    T: float,
    separation: float,
    rho: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    lambda_max: float | None = None,
    n_seg: int = 8,
    n_quad: int = 32,
) -> float:
    """Pair resistance of a tandem pair on the canonical spectrum.

    The two source distributions are concatenated
    (:meth:`CenterplanePanels.merged`) and the *canonical* integral is applied
    to the merged list, so the ``(1 + exp(i alpha s))`` factor is never formed
    explicitly -- same independence argument as
    :func:`tandem_pair_resistance`.
    """
    if separation < 0:
        raise ValueError("separation must be non-negative")
    merged = panels.merged(panels.shifted(separation))
    return wave_resistance_from_panels_canonical(
        merged, U, T, rho=rho, gravity=gravity, lambda_max=lambda_max,
        n_seg=n_seg, n_quad=n_quad,
    )
