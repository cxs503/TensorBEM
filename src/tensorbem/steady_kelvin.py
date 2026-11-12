"""Steady (Kelvin) Neumann-Kelvin kernel: derivation, validated closed form, candidate.

Why this module exists
----------------------
:mod:`tensorbem.neumann_kelvin` carries ``KERNEL_STATUS`` = WITHHELD because the
wave part it implements,

    G_W = (K/2pi) PV int_0^inf exp(-K u |z+zeta|) J_0(K u R) / (u - 1) du,

is the *frequency-domain* Havelock kernel (radial ``J_0(k0 R)``, single pole at
the radial wavenumber ``k0 = K``).  The steady free-surface condition
``K phi_z + phi_xx = 0`` is not an isotropic radial condition, so that kernel
cannot satisfy it (measured: 2 %-60 % of term scale, and the residual does not
shrink under quadrature refinement).

The correct steady kernel
-------------------------
With ``K = g/U^2``, a unit source at ``(0, 0, -zeta)``, field point ``(x, y, z)``
(``z <= 0``), ``rho = hypot(x, y)``, ``a = rho cos(phi - psi)``, ``s = mu - i a``
and ``mu = |z| + zeta >= 0``, the free-surface correction of the steady Green
function is the one-parameter (Kelvin family) spectral integral

    W(P) = (1/(2 pi)^2) int_0^{2 pi} dphi  int_0^inf dk  k e^{-s k} / (k_p - k),
    k_p(phi) = K sec^2(phi)                                    (Kelvin curve),

obtained from the 2-D Fourier transform of the Rankine + mirror pair on ``z = 0``
(``Rhat = exp(-k zeta)/k``) and the steady free-surface condition
``(-alpha^2 + K k) What = alpha^2 Rhat``.  The ``k``-integration is exactly
elementary, so one closed form replaces the singular double integral:

    PV int_0^inf k^m e^{-s k}/(k - k_p) dk
        = sum_{j=0}^{m-1} k_p^j (m-1-j)! / s^{m-j}  -  k_p^m e^{-s k_p} Ei(s k_p),

checked against an independent singularity-subtracted, spectrally accurate
quadrature (``pv_moment_int_quadrature``) to ``1e-15`` relative on the smooth
cases and ``1.4e-7`` on the stiff ``mu = 0.05`` case (``tests/test_steady_kelvin_identity.py``).  The dispersion
constraint is satisfied *mode by mode* on the Kelvin curve, so every ``phi``
term obeys ``K W_z + W_xx = 0`` identically; the inhomogeneous part (the
cancellation of the Rankine residual) is carried by the same integral.

What is solved and what is open
------------------------------
*Solved*: the closed form above (reusable), and the fact that the shipped
radial kernel is structurally wrong for the steady condition.

*Open*: the eigenvalue ``k = k_p(phi)`` sits **on** the integration path, so the
physical solution needs the radiation-condition (Rayleigh / Lighthill)
prescription, i.e. the ``+/- i pi k_p^m e^{-s k_p}`` residue term with the sign
that makes the wave field one-sided (waves only downstream of the source).  The
candidate implemented here (``radial_mode``) is still **symmetric** in
``x -> -x`` and its free-surface residual is 0.4 %-24 % of term scale, i.e. it
does **not** pass the referee ``neumann_kelvin.steady_free_surface_residual``.
The measured status, and therefore the WITHHELD verdict of
``solve_nk_wave_resistance``, is unchanged; see
``docs/m16955_steady_kernel_evidence.md`` for the numbers and the figure.

This module is therefore a *formulation and identity* asset plus a measured
honest-FAIL, not a replacement kernel.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import expi

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

KELVIN_ANGLE_DEG = math.degrees(math.asin(1.0 / 3.0))

STATUS = (
    "CANDIDATE: correct steady (Kelvin-curve) formulation and exact k-integral; "
    "radiation-condition sign not yet resolved -- wave part still symmetric in "
    "x -> -x and the free-surface residual is 0.4-24 % of term scale, so "
    "neumann_kelvin.KERNEL_STATUS stays WITHHELD.  See docs/m16955_steady_kelvin_evidence.md"
)


# ---------------------------------------------------------------------------
# Exponential integral with the exponential factor folded in (stable)
# ---------------------------------------------------------------------------

def exp_ei(z: ComplexArray | complex) -> ComplexArray:
    """``exp(-z) * Ei(z)`` without overflow.

    ``|z| <= 25``: direct ``exp(-z) Ei(z)``; ``|z| > 25``: the asymptotic
    ``1/z (1 + 1/z + 2!/z^2 + ...)`` (20 terms, error ``O(20!/|z|^20)``).
    """
    z = np.asarray(z, dtype=complex)
    out = np.empty(z.shape, dtype=complex)
    big = np.abs(z) > 25.0
    small = ~big
    if np.any(small):
        zs = z[small]
        out[small] = np.exp(-zs) * expi(zs)
    if np.any(big):
        inv = 1.0 / z[big]
        acc = np.ones_like(inv)
        term = np.ones_like(inv)
        for k in range(1, 21):
            term = term * (k * inv)
            acc = acc + term
        out[big] = acc * inv
    return out


def pv_moment_int(m: int, pole: ComplexArray | float, s: ComplexArray | complex) -> ComplexArray:
    """Closed form of ``PV int_0^inf k^m e^{-s k}/(k - pole) dk``.

    Exact for ``Re s > 0`` and ``pole`` real (the pole of the steady problem);
    ``pole`` may be complex to model a Rayleigh-shifted (damped) pole.
    """
    m = int(m)
    if m < 0:
        raise ValueError("m must be >= 0")
    kp = np.asarray(pole, dtype=complex)
    ss = np.asarray(s, dtype=complex)
    kp, ss = np.broadcast_arrays(kp, ss)
    tot = np.zeros(kp.shape, dtype=complex)
    for j in range(m):
        tot = tot + kp ** j * math.factorial(m - 1 - j) / ss ** (m - j)
    return tot - kp ** m * exp_ei(ss * kp)


def pv_moment_int_quadrature(
    m: int, pole: float, s: complex, upper: float = 400.0,
    panels: int = 40, nodes: int = 64,
) -> complex:
    """Independent reference: PV by singularity subtraction + composite Gauss.

    The subtracted integrand ``g(k) = (k^m e^{-sk} - a^m e^{-sa})/(k - a)`` is
    analytic at ``k = a``, so composite Gauss-Legendre is spectrally accurate
    and the logarithmic remainder is added in closed form.  No exponential
    integral is used anywhere -- this is the external check of
    :func:`pv_moment_int`.
    """
    m = int(m)
    a = float(pole)
    a_m = a ** m * np.exp(-s * a)
    edges = np.linspace(0.0, upper, panels + 1)
    leg_x, leg_w = np.polynomial.legendre.leggauss(nodes)
    total = 0.0 + 0.0j
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = 0.5 * (hi - lo) * leg_x + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * leg_w
        f_k = k ** m * np.exp(-s * k)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (f_k - a_m) / (k - a)
        hit = np.abs(k - a) < 1e-13
        if np.any(hit):
            g[hit] = m * a ** (m - 1) - s * a ** m
        total += complex(np.sum(w * g))
    return total + a_m * (math.log(abs(upper - a)) - math.log(abs(a)))


# ---------------------------------------------------------------------------
# Candidate steady kernel
# ---------------------------------------------------------------------------

def kelvin_wavenumber(phi: FloatArray | float, kappa: float) -> FloatArray:
    """``k_p(phi) = kappa sec^2(phi)`` -- the steady dispersion curve."""
    return kappa / np.cos(phi) ** 2


def wave_part_complex(
    x: float, y: float, z: float,
    xs: float = 0.0, ys: float = 0.0, zs: float = 0.0,
    kappa: float = 1.0, nodes: int = 512, deriv: str | None = None,
    radial_mode: str = "pv",
) -> complex:
    """Candidate steady wave part ``W`` of the Green function (complex device value).

    ``deriv`` selects the analytic derivative: ``None`` (value), ``"z"``,
    ``"x"``, ``"xx"``.  ``radial_mode`` seeds the radiation-condition sign:
    ``"pv"`` (no residue term), ``"all+"``/``"all-"`` (uniform), ``"sign"``
    (``+`` for downstream directions).  The physical combination is not yet
    settled -- see :data:`STATUS`.
    """
    rho = math.hypot(x - xs, y - ys)
    psi = math.atan2(y - ys, x - xs)
    mu = abs(z) + abs(zs)
    phi = (np.arange(nodes) + 0.5) * (2.0 * math.pi / nodes)
    weight = 2.0 * math.pi / nodes
    cp = np.cos(phi)
    kp = kelvin_wavenumber(phi, kappa)
    a = rho * np.cos(phi - psi)
    s = mu - 1j * a
    if radial_mode == "pv":
        sig = np.zeros_like(a)
    elif radial_mode == "all+":
        sig = np.ones_like(a)
    elif radial_mode == "all-":
        sig = -np.ones_like(a)
    elif radial_mode == "sign":
        sig = np.where(a > 0.0, 1.0, -1.0)
    else:  # pragma: no cover - guarded by tests
        raise ValueError(f"unknown radial_mode {radial_mode!r}")

    order = {None: 0, "z": 1, "x": 1, "xx": 2}[deriv]
    val = pv_moment_int(order, kp, s) + sig * 1j * math.pi * kp ** order * np.exp(-s * kp)
    if deriv == "x":
        val = val * (-1j * cp)
    elif deriv == "xx":
        val = val * cp ** 2
    val = -val
    return complex(np.sum(val)) * weight / (2.0 * math.pi) ** 2


def rankine_part(
    x: float, y: float, z: float,
    xs: float = 0.0, ys: float = 0.0, zs: float = 0.0,
) -> float:
    """``1/(4 pi r) + 1/(4 pi r1)`` (source plus free-surface mirror)."""
    r = math.sqrt((x - xs) ** 2 + (y - ys) ** 2 + (z - zs) ** 2)
    r1 = math.sqrt((x - xs) ** 2 + (y - ys) ** 2 + (z + zs) ** 2)
    return 1.0 / (4.0 * math.pi * r) + 1.0 / (4.0 * math.pi * r1)


def rankine_xx(x: float, y: float, z: float, xs: float = 0.0, ys: float = 0.0, zs: float = 0.0) -> float:
    """``d^2/dx^2`` of :func:`rankine_part` on ``z = 0`` (where ``r = r1``)."""
    rho2 = (x - xs) ** 2 + (y - ys) ** 2 + zs ** 2
    return 2.0 * (2.0 * (x - xs) ** 2 - (y - ys) ** 2 - zs ** 2) / (4.0 * math.pi * rho2 ** 2.5)


def candidate_green(x: float, y: float, z: float, zs: float, kappa: float,
                    nodes: int = 512, radial_mode: str = "pv") -> float:
    """Total candidate Green function ``G = Rankine + Re(W)``."""
    w = wave_part_complex(x, y, z, 0.0, 0.0, zs, kappa, nodes, None, radial_mode)
    return rankine_part(x, y, z, 0.0, 0.0, zs) + w.real


def free_surface_residual(kernel: Callable[[float, float, float], float],
                          point: Sequence[float], kappa: float, h: float = 1e-5) -> dict:
    """Referee ``K G_z + G_xx`` at the free-surface point ``point = (x, y, 0)``.

    Mirrors ``neumann_kelvin.steady_free_surface_residual`` (one-sided vertical
    difference into the fluid, central streamwise second difference) so that the
    shipped kernel and a candidate are scored by exactly the same rule.
    """
    x0, y0 = float(point[0]), float(point[1])
    g_z = (kernel(x0, y0, -h) - kernel(x0, y0, -2.0 * h)) / h
    g_xx = (kernel(x0 + h, y0, 0.0) - 2.0 * kernel(x0, y0, 0.0)
            + kernel(x0 - h, y0, 0.0)) / (h * h)
    residual = kappa * g_z + g_xx
    scale = abs(kappa * g_z) + abs(g_xx)
    return {
        "residual": residual,
        "k_gz": kappa * g_z,
        "g_xx": g_xx,
        "scale": scale,
        "relative": abs(residual) / scale if scale > 0.0 else math.inf,
    }


def candidate_free_surface_residual(x: float, zs: float, kappa: float,
                                   nodes: int = 512, radial_mode: str = "pv",
                                   h: float = 1e-5) -> dict:
    """Referee applied to :func:`candidate_green` at the free-surface point ``(x, 0, 0)``."""
    def kernel(xx: float, yy: float, zz: float) -> float:
        return candidate_green(xx, yy, zz, zs, kappa, nodes, radial_mode)
    return free_surface_residual(kernel, (x, 0.0, 0.0), kappa, h)


def radiation_asymmetry(xs: Sequence[float], zs: float, kappa: float,
                        nodes: int = 512, radial_mode: str = "pv") -> list[dict]:
    """Upstream/downstream sample of the candidate wave part (radiation check).

    A physical steady kernel is one-sided: ``|W|`` must be negligible for
    ``x < 0`` (upstream of the source) at fixed distance.  The candidate is
    measured, not assumed, to be symmetric -- that is the open defect.
    """
    rows = []
    for x in xs:
        w = wave_part_complex(float(x), 0.0, 0.0, 0.0, 0.0, zs, kappa, nodes, None, radial_mode)
        rows.append({"x": float(x), "abs_w": abs(w), "re": w.real, "im": w.imag})
    return rows
