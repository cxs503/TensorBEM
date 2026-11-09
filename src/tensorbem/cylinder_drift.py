"""Exact MacCamy-Fuchs reference for the mean second-order drift force on a
bottom-mounted vertical circular cylinder (deep water).

Why this module exists
----------------------
``tensorbem.drift_force`` budgets the drift force of a *vertical wall* exactly
(``rho g a^2 / 2``) but its 3-D Maruo far-field route is **not convergent in the
radiation radius** for a cylinder: the un-reduced interference flux oscillates
like ``exp(i k R cos theta)`` and the ``(ka)^3`` long-wave gate was reported as an
*honest FAIL* (``dev_report_m5242_second_order_drift.md``).  This module closes
that gap with a route that does not depend on a radiation radius at all: the
exact MacCamy-Fuchs (MF) diffraction field of a circular cylinder gives both a
near-field momentum integral and a far-field Kochin function analytically, so the
two independent drift-force formulas can be cross-checked against each other (and
against the wall anchor) and the long-wave ``(ka)^3`` law can be written in
closed form.

Physics and conventions
-----------------------
Deep water, fixed body, bottom-mounted vertical cylinder of radius ``a``.  Linear
potential ``Phi = Re{Phi_hat e^{-i omega t}}`` with

    Phi_hat = (g A / omega) * exp(k z) * psi(x, y),      omega^2 = g k,

``A`` the incident wave amplitude and ``psi`` dimensionless::

    psi = psi_I + psi_s,   psi_I = exp(i k (x cos beta + y sin beta)),
    psi(r, theta) = sum_m [ i^m J_m(kr) + A_m H_m^(1)(kr) ] e^{i m theta},
    A_m = - i^m J_m'(ka) / H_m'^(1)(ka)          (MacCamy & Fuchs, 1954).

Near field.  The classical second-order mean pressure integral for a fixed body,
with ``<|grad phi|^2> = |grad Phi_hat|^2 / 2`` and ``<zeta^2> = |zeta_hat|^2 / 2``,

    F_i = (1/4) rho Int |grad Phi_hat|^2 n_i dS - (1/4) rho g Int |zeta_hat|^2 n_i dl,

``n`` the outward normal of the body.  On the cylinder wall
(``n = (cos theta, sin theta, 0)``) the depth integral is analytic,
``Int_{-inf}^{0} exp(2 k z) dz = 1/(2k)``, and ``zeta_hat = A psi(a, theta)``, so

    F_x = (rho/4)(g A/omega)^2 (1/(2k)) a Int [|dpsi/dr|^2 + |dpsi/dtheta|^2/a^2
          + k^2|psi|^2] cos(theta) dtheta - (rho g A^2/4) a Int |psi|^2 cos(theta) dtheta.

The same combination applied to the standing wave in front of a fully reflecting
vertical wall returns exactly ``rho g A^2 / 2`` per unit length
(``tests/test_cylinder_drift_asymptotic.py``), which fixes sign and the 1/4.

Far field (reduced Maruo kernel).  With
``psi_s ~ sqrt(2/(pi k r)) exp(i(k r - pi/4)) S(theta)`` at large ``r`` the
scattered flux plus the stationary-phase momentum balance give the
``k R``-independent form

    F_x = (rho g A^2 / (2 pi k)) Int_0^{2pi} |S(theta)|^2 (1 - cos theta) dtheta,

i.e. ``F_x = P_scattered / c_g`` with the interference already reduced -- this is
what removes the radius oscillation of the un-reduced flux route.  The Kochin
function of :mod:`tensorbem.free_surface_solver3d` is documented as
``phi ~ (i k/2) H_0^(1)(kR) exp(k z) K(theta)``, hence ``K = -2 i S / k`` and

    F_x = (rho g A^2 k / (8 pi)) Int |K(theta)|^2 (1 - cos theta) dtheta.

Long-wave asymptotics.  For ``ka -> 0`` only ``m = 0, +-1`` survive and
``A_0 ~ -i kappa``, ``A_1 ~ -kappa``, ``A_{-1} ~ +kappa`` with
``kappa = pi (ka)^2 / 4``, so ``S ~ i kappa (2 cos theta - 1)`` and

    F_x = (5 pi^2 / 16) rho g A^2 a (ka)^3  ~  3.084 rho g A^2 a (ka)^3,

the ``(ka)^3`` law of the drifting cylinder, here as a closed form to be compared
with the exact evaluation over ``ka``.

Author: TensorBEM node p41w3-n01 (m7374).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "DEFAULT_DENSITY",
    "DEFAULT_GRAVITY",
    "bessel_j",
    "bessel_y",
    "maccamy_fuchs_modes",
    "wall_field_values",
    "cylinder_wall_field",
    "near_field_cylinder_drift",
    "kochin_2d",
    "kochin_3d",
    "far_field_cylinder_drift",
    "long_wave_drift_coefficient",
    "energy_balance",
    "richardson_extrapolate",
]

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.81
_EULER_GAMMA = 0.5772156649015328606


# ---------------------------------------------------------------------------
# pure-NumPy integer-order Bessel functions (the package has no SciPy dependency)
# ---------------------------------------------------------------------------
def bessel_j(m: int, x: float) -> float:
    """Integer-order Bessel function ``J_m(x)`` (ascending series)."""
    m = int(m)
    x = float(x)
    if x == 0.0:
        return 1.0 if m == 0 else 0.0
    if m < 0:
        return (-1.0) ** (m % 2) * bessel_j(-m, x)
    half = 0.5 * x
    term = math.exp(m * math.log(half) - math.lgamma(m + 1.0))
    total = term
    k = 0
    while True:
        k += 1
        term *= -half * half / (k * (k + m))
        total += term
        if abs(term) <= 1e-18 * max(abs(total), 1e-300):
            break
        if k > 200:  # pragma: no cover - series always terminates far earlier
            break
    return total


def bessel_y(m: int, x: float) -> float:
    """Integer-order Bessel function ``Y_m(x)`` for ``x > 0``.

    ``Y_0`` and ``Y_1`` come from the A&S 9.1.13 ascending series; higher orders
    follow the upward recurrence ``Y_{m+1} = (2m/x) Y_m - Y_{m-1}`` (stable for
    ``Y`` when ``m > x``, the regime used here).
    """
    m = int(m)
    x = float(x)
    if x <= 0.0:
        raise ValueError("x must be positive")
    if m < 0:
        return (-1.0) ** (m % 2) * bessel_y(-m, x)
    if m == 0:
        return _bessel_y0(x)
    y_prev = _bessel_y0(x)
    y_cur = _bessel_y1(x)
    for n in range(1, m):
        y_next = (2.0 * n / x) * y_cur - y_prev
        y_prev, y_cur = y_cur, y_next
    return y_cur


def _harmonic(n: int) -> float:
    """Harmonic number ``H_n``."""
    return float(sum(1.0 / i for i in range(1, int(n) + 1)))


def _bessel_y0(x: float) -> float:
    half = 0.5 * x
    total = 0.0
    term_k = 1.0  # (x/2)^{2k} / (k!)^2 at k = 0
    for k in range(1, 120):
        term_k *= half * half / (k * k)
        contrib = (-1.0) ** (k + 1) * _harmonic(k) * term_k
        total += contrib
        if abs(contrib) <= 1e-18 * max(abs(total), 1e-300):
            break
    return (2.0 / math.pi) * ((math.log(half) + _EULER_GAMMA) * bessel_j(0, x) + total)


def _bessel_y1(x: float) -> float:
    """``Y_1(x) = -dY_0/dx``, differentiated term by term from the ``Y_0`` series."""
    half = 0.5 * x
    total = 0.0
    term_k = 1.0
    for k in range(1, 120):
        term_k *= half * half / (k * k)
        contrib = (-1.0) ** (k + 1) * _harmonic(k) * (2.0 * k / x) * term_k
        total += contrib
        if abs(contrib) <= 1e-18 * max(abs(total), 1e-300):
            break
    derivative = (
        (1.0 / x) * bessel_j(0, x)
        + (math.log(half) + _EULER_GAMMA) * (-bessel_j(1, x))
        + total
    )
    return -(2.0 / math.pi) * derivative


def _bessel_jp(m: int, x: float) -> float:
    return 0.5 * (bessel_j(m - 1, x) - bessel_j(m + 1, x))


def _bessel_yp(m: int, x: float) -> float:
    return 0.5 * (bessel_y(m - 1, x) - bessel_y(m + 1, x))


def _hankel1(m: int, x: float) -> complex:
    return complex(bessel_j(m, x), bessel_y(m, x))


def _hankel1_prime(m: int, x: float) -> complex:
    return complex(_bessel_jp(m, x), _bessel_yp(m, x))


# ---------------------------------------------------------------------------
# MacCamy-Fuchs modes
# ---------------------------------------------------------------------------
def maccamy_fuchs_modes(ka: float, m_max: int = 14) -> np.ndarray:
    """Scattering coefficients ``A_m`` for ``m = -m_max .. m_max``."""
    x = float(ka)
    if not (x > 0.0):
        raise ValueError("ka must be positive")
    m_max = int(m_max)
    if m_max < 2:
        raise ValueError("m_max must be >= 2")
    out = np.zeros(2 * m_max + 1, dtype=np.complex128)
    for m in range(-m_max, m_max + 1):
        am = -complex(0.0, 1.0) ** m * _bessel_jp(m, x) / _hankel1_prime(m, x)
        out[m + m_max] = am
    return out


def _modal_sum(ka: float, m_max: int, radius: float, theta: np.ndarray, kind: str) -> np.ndarray:
    """Evaluate ``psi``, ``dpsi/dr`` or ``dpsi/dtheta`` on ``r = radius``."""
    x = float(ka)
    a_m = maccamy_fuchs_modes(x, m_max)
    radii = radius / 1.0  # radius given in units of the cylinder radius a
    theta = np.asarray(theta, dtype=float)
    total = np.zeros(theta.shape, dtype=np.complex128)
    for m in range(-m_max, m_max + 1):
        am = a_m[m + m_max]
        arg = x * radii
        if kind == "psi":
            inc = complex(0.0, 1.0) ** m * bessel_j(m, arg)
            sca = am * _hankel1(m, arg)
            total += (inc + sca) * np.exp(1j * m * theta)
        elif kind == "dr":
            inc = complex(0.0, 1.0) ** m * _bessel_jp(m, arg) * x
            sca = am * _hankel1_prime(m, arg) * x
            total += (inc + sca) * np.exp(1j * m * theta)
        elif kind == "dtheta":
            total += 1j * m * (
                complex(0.0, 1.0) ** m * bessel_j(m, arg) + am * _hankel1(m, arg)
            ) * np.exp(1j * m * theta)
        else:  # pragma: no cover - guarded by call sites
            raise ValueError(f"unknown kind {kind!r}")
    return total


def cylinder_wall_field(
    ka: float, theta: np.ndarray, *, m_max: int = 14
) -> dict[str, np.ndarray]:
    """``psi``, ``dpsi/dr``, ``dpsi/dtheta`` on the cylinder wall ``r = a``."""
    x = float(ka)
    psi = _modal_sum(x, m_max, 1.0, theta, "psi")
    dr = _modal_sum(x, m_max, 1.0, theta, "dr")
    dt = _modal_sum(x, m_max, 1.0, theta, "dtheta")
    return {"psi": psi, "dpsi_dr": dr, "dpsi_dtheta": dt}


def wall_field_values(nu: float, x_wall: float = 0.0) -> dict[str, float]:
    """Exact fully-reflecting-wall field used as the drift-force anchor.

    Fluid on ``x < x_wall``; incident + reflected standing wave
    ``psi = 2 cos(k (x - x_wall))``.  Returns the pieces the near-field formula
    needs: ``|psi|^2``, ``|dpsi/dx|^2``, ``k^2|psi|^2`` at the wall and the wall
    elevation amplitude factor ``|zeta_hat|/A``.
    """
    nu = float(nu)
    if not (nu > 0.0):
        raise ValueError("nu must be positive")
    return {
        "psi_sq": 4.0,
        "dpsi_dx_sq": 0.0,
        "surface_amp_factor": 2.0,
    }


def near_field_cylinder_drift(
    ka: float,
    *,
    amplitude: float = 1.0,
    radius: float = 1.0,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_theta: int = 720,
    m_max: int = 14,
) -> dict[str, float]:
    """Exact near-field mean drift force on the fixed cylinder."""
    x = float(ka)
    if not (x > 0.0):
        raise ValueError("ka must be positive")
    amp = float(amplitude)
    a = float(radius)
    rho = float(density)
    g = float(gravity)
    n_th = int(n_theta)
    if n_th < 64 or n_th % 2:
        raise ValueError("n_theta must be an even integer >= 64")
    theta = (2.0 * math.pi / n_th) * np.arange(n_th, dtype=float)
    dtheta = 2.0 * math.pi / n_th
    nu = x / a
    field = cylinder_wall_field(x, theta, m_max=m_max)
    psi = field["psi"]
    dr = field["dpsi_dr"]
    dt = field["dpsi_dtheta"]
    grad_sq = np.abs(dr) ** 2 + np.abs(dt / a) ** 2 + nu**2 * np.abs(psi) ** 2
    i1 = float(np.sum(grad_sq * np.cos(theta)) * dtheta)
    i2 = float(np.sum(np.abs(psi) ** 2 * np.cos(theta)) * dtheta)
    g_over_omega_sq = g * amp**2 / nu  # (g A / omega)^2 = g A^2 / k in deep water
    force = (0.25 * rho * g_over_omega_sq / (2.0 * nu) * a * i1
             - 0.25 * rho * g * amp**2 * a * i2)
    return {
        "mean_drift_force_x": force,
        "surface_integral": i1,
        "waterline_integral": i2,
        "wavenumber": nu,
        "ka": x,
        "amplitude": amp,
        "radius": a,
    }


def kochin_2d(ka: float, theta: np.ndarray, *, m_max: int = 14) -> np.ndarray:
    """Far-field scattering amplitude ``S(theta)`` of the exact MF field."""
    x = float(ka)
    theta = np.asarray(theta, dtype=float)
    a_m = maccamy_fuchs_modes(x, m_max)
    total = np.zeros(theta.shape, dtype=np.complex128)
    for m in range(-m_max, m_max + 1):
        am = a_m[m + m_max]
        total += am * np.exp(-0.5j * m * math.pi) * np.exp(1j * m * theta)
    return total


def kochin_3d(ka: float, theta: np.ndarray, *, m_max: int = 14) -> np.ndarray:
    """Kochin function in the ``(i k/2) H_0^(1) exp(k z) K`` convention."""
    x = float(ka)
    return -2.0j * kochin_2d(x, theta, m_max=m_max) / x


def far_field_cylinder_drift(
    ka: float,
    *,
    amplitude: float = 1.0,
    radius: float = 1.0,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_theta: int = 2880,
    m_max: int = 14,
) -> dict[str, float]:
    """Reduced Maruo far-field mean drift force (radius independent)."""
    x = float(ka)
    amp = float(amplitude)
    a = float(radius)
    rho = float(density)
    g = float(gravity)
    n_th = int(n_theta)
    if n_th < 64 or n_th % 2:
        raise ValueError("n_theta must be an even integer >= 64")
    nu = x / a
    theta = (2.0 * math.pi / n_th) * np.arange(n_th, dtype=float)
    dtheta = 2.0 * math.pi / n_th
    s = kochin_2d(x, theta, m_max=m_max)
    pattern = np.abs(s) ** 2
    integral = float(np.sum(pattern * (1.0 - np.cos(theta))) * dtheta)
    force = rho * g * amp**2 * integral / (2.0 * math.pi * nu)
    integral_iso = float(np.sum(pattern) * dtheta)
    return {
        "mean_drift_force_x": force,
        "flux_integral_weighted": integral,
        "flux_integral_total": integral_iso,
        "wavenumber": nu,
        "ka": x,
        "amplitude": amp,
        "radius": a,
    }


def long_wave_drift_coefficient() -> float:
    """``C`` in ``F_x = C rho g A^2 a (ka)^3`` from the ``(ka)^3`` asymptote."""
    return 5.0 * math.pi**2 / 16.0


def energy_balance(
    ka: float,
    *,
    radius: float = 1.0,
    radius_probe: float = 60.0,
    n_theta: int = 4096,
    m_max: int = 14,
) -> dict[str, float]:
    """Kochin energy-balance residual for the non-absorbing cylinder.

    The exact series field is evaluated on the circle ``r = radius_probe``; the
    outward energy flux of the *scattered* wave, of the *interference* term and
    of the total field are computed from ``Im{psi* dpsi/dr}`` with no far-field
    approximation.  A fixed rigid body absorbs nothing and does no work, so the
    total flux must vanish: ``flux_scattered + flux_interference = 0`` and the
    residual measures how well the radiation condition is satisfied.
    """
    x = float(ka)
    r_probe = float(radius_probe)
    n_th = int(n_theta)
    if r_probe <= 1.0:
        raise ValueError("radius_probe must exceed the cylinder radius")
    if x * (r_probe / float(radius)) > 35.0:
        raise ValueError(
            "k*R must not exceed 35: the ascending Bessel series used here loses "
            "accuracy for larger arguments (use the reduced far-field route instead)"
        )
    theta = (2.0 * math.pi / n_th) * np.arange(n_th, dtype=float)
    dtheta = 2.0 * math.pi / n_th
    scale = r_probe / float(radius)
    # incident
    a_m = maccamy_fuchs_modes(x, m_max)
    inc = np.zeros(theta.shape, dtype=np.complex128)
    sca = np.zeros(theta.shape, dtype=np.complex128)
    inc_r = np.zeros(theta.shape, dtype=np.complex128)
    sca_r = np.zeros(theta.shape, dtype=np.complex128)
    arg = x * scale
    for m in range(-m_max, m_max + 1):
        am = a_m[m + m_max]
        phase = np.exp(1j * m * theta)
        inc += complex(0.0, 1.0) ** m * bessel_j(m, arg) * phase
        inc_r += complex(0.0, 1.0) ** m * _bessel_jp(m, arg) * x * phase
        sca += am * _hankel1(m, arg) * phase
        sca_r += am * _hankel1_prime(m, arg) * x * phase
    flux_s = float(np.sum(np.imag(np.conj(sca) * sca_r)) * scale * dtheta)
    flux_cross = float(
        np.sum(np.imag(np.conj(inc) * sca_r + np.conj(sca) * inc_r)) * scale * dtheta
    )
    flux_total = float(
        np.sum(np.imag(np.conj(inc + sca) * (inc_r + sca_r))) * scale * dtheta
    )
    return {
        "ka": x,
        "probe_radius": r_probe,
        "flux_scattered": flux_s,
        "flux_interference": flux_cross,
        "flux_total": flux_total,
        "residual": abs(flux_total) / abs(flux_s),
        "pair_residual": abs(flux_s + flux_cross) / abs(flux_s),
    }


def richardson_extrapolate(
    coarse: float, fine: float, ratio: float, order: float
) -> float:
    """Richardson extrapolation ``f_inf = f_f + (f_f - f_c)/(r^p - 1)``."""
    ratio = float(ratio)
    order = float(order)
    if ratio <= 1.0:
        raise ValueError("ratio must exceed 1")
    return float(fine + (fine - coarse) / (ratio**order - 1.0))