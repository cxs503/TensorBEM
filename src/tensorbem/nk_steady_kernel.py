"""Steady Neumann-Kelvin Green function: exact spectral form and a candidate kernel.

Status (honest): **WITHHELD / CANDIDATE**.  This module does *not* close the debt
recorded in :mod:`tensorbem.neumann_kelvin`; it narrows it.  What it establishes,
machine-checked in ``tests/test_nk_steady_kernel_candidate.py``:

1. The **exact spectral form** of the steady deep-water Neumann-Kelvin kernel
   (equations below) and its reduction to a single contour integral over the
   Kelvin curve, with Lighthill's indentation sign ``s = sign(cos theta)``.
2. A **candidate implementation** of that reduction whose quadrature is fully
   converged (2000 and 32000 nodes agree to ``1e-13`` at the test points) and
   which yields a **real-valued** kernel to ``1e-15``, as any real Green
   function must.
3. The candidate nevertheless **fails the free-surface referee**:
   ``|K G_z + G_xx| / (|K G_z| + |G_xx|)`` = 1.82e-1 (x=1, y=0), 1.00e-1
   (x=0.7, y=0.7) and 2.02e-2 (x=-2, y=0.5), with a source at ``(0, 0, -1)``,
   ``K = 1`` and ``h_fd = 1e-3``, against 2.17e-1, 1.00e0 and 4.61e-2 for the
   shipped frequency-domain kernel.  It beats the shipped kernel at two of the
   three points and still leaves a residual two to three orders of magnitude
   above what the identity allows.  The failure is unchanged
   under quadrature refinement and under the finite-difference step, so it is
   **systematic in the local-term bookkeeping** (the
   ``1 + c - c^2 exp(-c) Ei(c)`` part that has to cancel ``-d_xx G_R``), not a
   discretisation error.  The capability therefore stays ``WITHHELD``; the next
   increment is to isolate the local term with the ``R -> 0, h -> 0`` limit of
   the spectral integral instead of by assembling it term by term.

Spectral form
-------------
Deep water, fluid in ``z <= 0``, source at the origin of the horizontal plane at
depth ``zeta < 0``, field point ``(x, y, z)``, ``h = -(z + zeta) >= 0``,
``K = g / U**2``.  The disturbance satisfies Laplace's equation, the steady
Neumann-Kelvin condition ``K phi_z + phi_xx = 0`` on ``z = 0`` and the
radiation condition (no waves ahead of the source).  With
``G_R = 1/(4 pi r) + 1/(4 pi r_1)`` (Rankine plus mirror; ``d_z G_R = 0`` on
``z = 0``), the wave part is

    G_W = (1/(2 pi)^2) int int_R2  A(alpha, beta) e^{i(alpha x + beta y)} e^{-k h} d alpha d beta,
    A(alpha, beta) = alpha**2 / (k (K k - alpha**2)),          k = |(alpha, beta)|,

because the free-surface condition turns into
``(K k - alpha**2) A = -d_xx G_R`` and ``F[d_xx G_R]_{(z=0)} = -alpha**2 / k``.
The pole locus ``alpha**2 = K k`` is the Kelvin curve
``k0(theta) = K sec**2 theta``.

Polar reduction
---------------
``alpha = k cos theta``, ``d alpha d beta = k dk d theta``, and the ``k``
integral is elementary once ``a = h - i R cos(theta - theta_R)``, ``c = k0 a``:

    G_W = -(1/(2 pi)^2) int_0^{2 pi} Q(c) / a**2 d theta,
    Q(c) = 1 + c + c**2 J(c),      J(c) = int_0^inf e^{-t} / (t - c) dt,
    Q(c) = 1 + c - c**2 exp(-c) (Ei(c) + i pi s),        (|c| small)
    Q(c) ~ -2/c - 6/c**2 - 24/c**3 - ...                 (|c| -> inf),

where ``J`` is the principal value plus the half residue left by the
indentation: ``s = +1`` on the downstream half of the wavevector circle and
``-1`` on the upstream half.  That sign is what makes the ``theta`` integral
real (the pair ``theta, theta + pi`` then contributes ``i * (purely imaginary)
/ 2`` to a real total) and it is the only choice we found that does; the
uniform indentation ``s = const`` leaves an imaginary part of order the wave
amplitude, which is recorded in the mission report.

Usage::

    from tensorbem.nk_steady_kernel import steady_green, free_surface_residual
    G = steady_green([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], kappa=1.0)
    free_surface_residual([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], kappa=1.0)
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import exp1

__all__ = [
    "KERNEL_STATUS",
    "KERNEL_MEASUREMENTS",
    "kelvin_pole_wavenumber",
    "q_kernel",
    "steady_wave_part",
    "steady_green",
    "free_surface_residual",
]

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi

KERNEL_STATUS = (
    "CANDIDATE / WITHHELD: exact spectral form implemented; the polar reduction "
    "is real-valued and quadrature-converged (4e-10) but violates the "
    "free-surface referee K G_z + G_xx = 0 by 2.0e-2..1.8e-1 relative, "
    "independent of quadrature and finite-difference refinement -> local-term "
    "bookkeeping, not discretisation. Do not use as evidence."
)

#: Measured referee values of this candidate (source (0,0,-1), K = 1, h_fd = 1e-3),
#: stored so that a future fix has a baseline it must beat rather than a memory.
KERNEL_MEASUREMENTS = {
    "referee": "|K G_z + G_xx| / (|K G_z| + |G_xx|)",
    "source": [0.0, 0.0, -1.0],
    "kappa": 1.0,
    "fd_step": 1e-3,
    "candidate_relative_residual": {
        "x=1.0,y=0.0": 1.82e-1,
        "x=0.7,y=0.7": 1.00e-1,
        "x=-2.0,y=0.5": 2.02e-2,
    },
    "shipped_frequency_domain_kernel_relative_residual": {
        "x=1.0,y=0.0": 2.17e-1,
        "x=0.7,y=0.7": 1.00e0,
        "x=-2.0,y=0.5": 4.61e-2,
    },
    "candidate_quadrature_convergence": "2000 vs 32000 theta nodes agree to 4e-10",
    "candidate_reality": "|Im G_W| / |G_W| < 1e-15",
}

_ASYMPTOTIC_SWITCH = 25.0


def kelvin_pole_wavenumber(theta: ArrayLike, kappa: float) -> np.ndarray:
    """Pole of the steady spectral integral, ``k0(theta) = K sec**2 theta``."""
    return kappa / np.cos(np.asarray(theta, dtype=float)) ** 2


def _q_small(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``1 + c - c**2 exp(-c) (Ei(c) + i pi s)`` via ``Ei(c) = -E1(-c)``."""
    return 1.0 + c - c**2 * np.exp(-c) * (-exp1(-c) + 1j * np.pi * s)


def _q_large(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Asymptotic branch: ``-sum_{n>=1} (n+1)! / c**(n+1)`` minus the residue term."""
    acc = np.zeros_like(c)
    term = 1.0 / c
    factorial = 2.0
    for n in range(1, 9):
        acc = acc - factorial * term
        term = term / c
        factorial = factorial * (n + 2)
    return acc - 1j * np.pi * s * c**2 * np.exp(-c)


def q_kernel(c: ArrayLike, s: ArrayLike) -> np.ndarray:
    """``Q(c)`` with the indentation sign ``s`` (array broadcasting).

    Both branches are evaluated on the same ``c`` in their overlap so the test
    suite can assert they agree; ``|c| = 25`` is the switch, and the relative
    error of the asymptotic branch there is ``~2e-3`` at worst.
    """
    cc = np.asarray(c, dtype=complex)
    ss = np.asarray(s, dtype=float)
    out = np.empty_like(cc)
    large = np.abs(cc) > _ASYMPTOTIC_SWITCH
    if np.any(large):
        out[large] = _q_large(cc[large], ss[large])
    if np.any(~large):
        out[~large] = _q_small(cc[~large], ss[~large])
    return out


def steady_wave_part(
    x: float, y: float, depth_sum: float, kappa: float, n_theta: int = 2000,
) -> float:
    """Candidate wave part ``G_W`` at horizontal offset ``(x, y)``, ``h = depth_sum``.

    ``depth_sum`` is ``-(z + zeta) >= 0``: the wave part depends on the sum of
    field and source depth below the free surface, so a deep source and a deep
    field point both attenuate it.
    """
    radius = math.hypot(x, y)
    if radius < 1e-15:
        return 0.0
    theta_r = math.atan2(y, x)
    theta = (np.arange(n_theta) + 0.5) * (_TWO_PI / n_theta)
    cos_theta = np.cos(theta)
    a = depth_sum - 1j * radius * np.cos(theta - theta_r)
    c = kelvin_pole_wavenumber(theta, kappa) * a
    s = np.sign(cos_theta)
    return float(np.real(-(1.0 / _TWO_PI**2) * np.sum(q_kernel(c, s) / a**2) * (_TWO_PI / n_theta)))


def steady_green(
    target: ArrayLike, source: ArrayLike, kappa: float, n_theta: int = 2000,
) -> float:
    """Candidate steady Neumann-Kelvin Green function ``G = G_R + G_W``."""
    tgt = np.asarray(target, dtype=float)
    src = np.asarray(source, dtype=float)
    dx, dy, dz = tgt - src
    radius = math.hypot(dx, dy)
    r = math.hypot(radius, dz)
    if r < 1e-15:
        raise ValueError("coincident source and target")
    r1 = math.hypot(radius, tgt[2] + src[2])
    g_rankine = 1.0 / (_FOUR_PI * r) + 1.0 / (_FOUR_PI * r1)
    depth_sum = -(tgt[2] + src[2])
    if depth_sum <= 0.0:
        depth_sum = 1e-15
    return g_rankine + steady_wave_part(dx, dy, depth_sum, kappa, n_theta=n_theta)


def free_surface_residual(
    target: ArrayLike, source: ArrayLike, kappa: float,
    h: float = 1.0e-3, n_theta: int = 2000,
) -> dict[str, float]:
    """Referee: ``K G_z + G_xx`` on ``z = 0`` with the term scale and the ratio.

    Same instrument and same finite-difference steps as
    :func:`tensorbem.neumann_kelvin.steady_free_surface_residual`, so candidate
    and shipped kernel are compared on identical numbers.
    """
    tgt = np.asarray(target, dtype=float)
    src = np.asarray(source, dtype=float)
    if abs(tgt[2]) > 1.0e-12:
        raise ValueError("target must lie on the free surface z = 0")
    x, y = float(tgt[0]), float(tgt[1])

    def g(xx: float, yy: float, zz: float) -> float:
        return steady_green(np.array([xx, yy, zz]), src, kappa, n_theta=n_theta)

    g_z = (g(x, y, -h) - g(x, y, -2.0 * h)) / h
    g_xx = (g(x + h, y, 0.0) - 2.0 * g(x, y, 0.0) + g(x - h, y, 0.0)) / (h * h)
    residual = kappa * g_z + g_xx
    scale = abs(kappa * g_z) + abs(g_xx)
    return {
        "residual": residual,
        "k_gz": kappa * g_z,
        "g_xx": g_xx,
        "scale": scale,
        "relative": abs(residual) / scale if scale > 0.0 else math.inf,
    }


def residual_table(
    points: Iterable[tuple[float, float]], source: ArrayLike, kappa: float,
    steps: Iterable[float] = (1e-2, 1e-3), n_theta: int = 2000,
) -> list[dict[str, float]]:
    """Referee table used by the report: candidate vs shipped kernel, per FD step."""
    from .neumann_kelvin import steady_free_surface_residual as shipped

    rows: list[dict[str, float]] = []
    for x, y in points:
        for h in steps:
            cand = free_surface_residual([x, y, 0.0], source, kappa, h=h, n_theta=n_theta)
            old = shipped([x, y, 0.0], source, kappa, h=h)
            rows.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "fd_step": float(h),
                    "candidate_relative": cand["relative"],
                    "shipped_relative": old["relative"],
                }
            )
    return rows