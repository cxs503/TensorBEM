"""Finite-depth Green function with Newton-Raphson dispersion solver and
Havelock wave-making resistance for shallow-water seakeeping.

This module extends :mod:`tensorbem.finite_depth` with:

1. **Newton-Raphson dispersion solver** for ``omega^2 = g*k*tanh(k*h)``
   with initial guess ``k_0 = omega^2/g`` (deep-water approximation) and
   convergence tolerance ``1e-12``.

2. **Deep-water degeneration verification**: as ``h -> inf``, ``tanh(kh) -> 1``
   and the finite-depth Green function must recover the deep-water form
   with error < 1e-10.

3. **Havelock finite-depth wave-making resistance**:
   ``R_w = rho*g/(4*pi) * int |A(theta)|^2 * cos(theta) dtheta``
   with finite-depth corrections to the Kochin function and dispersion.

4. **Shallow-water long-wave limit**: ``c = sqrt(g*h)`` with error < 1%.

References
----------
- Wehausen, J.V. & Laitone, E.V. (1960). "Surface Waves."
  Handbuch der Physik, Vol. 9, Springer.
- Havelock, T.H. (1932). "The theory of wave resistance."
  Proc. R. Soc. A, 138, 339-348.
- Newman, J.N. (1977). "Marine Hydrodynamics." MIT Press.
- Faltinsen, O.M. (1990). "Sea Loads on Ships and Offshore Structures."
  Cambridge University Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from . import finite_depth as _fd
from . import michell

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DEFAULT_GRAVITY = 9.80665
DEFAULT_DENSITY = 1025.0

# Newton-Raphson convergence settings
_NR_MAX_ITER = 100
_NR_TOL = 1e-12


def _safe_cosh(x: float) -> float:
    """Compute cosh(x) with overflow protection."""
    if abs(x) > 350:
        return math.inf
    return math.cosh(x)


def _safe_sinh(x: float) -> float:
    """Compute sinh(x) with overflow protection."""
    if x > 700:
        return math.inf
    if x < -700:
        return -math.inf
    return math.sinh(x)


def _safe_exp(x: float) -> float:
    """Compute exp(x) with overflow protection."""
    if x > 700:
        return math.inf
    if x < -700:
        return 0.0
    return math.exp(x)


def _safe_tanh(x: float) -> float:
    """Compute tanh(x) with overflow protection."""
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    return math.tanh(x)


# ---------------------------------------------------------------------------
# 1. Newton-Raphson dispersion relation solver (oscillatory)
# ---------------------------------------------------------------------------
def dispersion_newton_raphson(
    omega: float, depth: float, *,
    gravity: float = DEFAULT_GRAVITY,
    tol: float = _NR_TOL,
    max_iter: int = _NR_MAX_ITER,
) -> float:
    """Solve ``omega^2 = g*k*tanh(k*h)`` for ``k`` by Newton-Raphson.

    Initial guess: ``k_0 = omega^2/g`` (deep-water approximation).
    The iteration is:

        k_{n+1} = k_n - (g*k_n*tanh(k_n*h) - omega^2) / (g*(tanh(k_n*h) + k_n*h/cosh^2(k_n*h)))

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    depth : float
        Water depth ``h`` [m].
    gravity : float
        Gravitational acceleration [m/s^2].
    tol : float
        Relative convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    float
        Wavenumber ``k`` [1/m].
    """
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if depth <= 0.0:
        raise ValueError("depth must be positive")

    nu = omega ** 2 / gravity
    k = nu  # Initial guess: deep-water approximation

    for _ in range(max_iter):
        kh = k * depth
        tanh_kh = _safe_tanh(kh)
        cosh_kh = _safe_cosh(kh)
        if cosh_kh == math.inf or cosh_kh > 1e150:
            cosh2_kh = math.inf
        else:
            cosh2_kh = cosh_kh * cosh_kh
        f = gravity * k * tanh_kh - omega ** 2
        df = gravity * (tanh_kh + kh / cosh2_kh)
        if abs(df) < 1e-300:
            break
        dk = f / df
        k_new = k - dk
        if k_new <= 0.0:
            return _fd.finite_depth_wavenumber(nu, depth)
        if abs(dk) / max(abs(k), 1.0) < tol:
            return k_new
        k = k_new

    return k


def dispersion_residual(
    k: float, omega: float, depth: float, *,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Return the residual ``g*k*tanh(k*h) - omega^2``."""
    kh = k * depth
    tanh_kh = _safe_tanh(kh)
    return gravity * k * tanh_kh - omega ** 2


# ---------------------------------------------------------------------------
# Steady-flow dispersion solver (for wave resistance)
# ---------------------------------------------------------------------------
def steady_flow_wavenumber(
    U: float, depth: float, cos_theta: float = 1.0, *,
    gravity: float = DEFAULT_GRAVITY,
    tol: float = _NR_TOL,
    max_iter: int = _NR_MAX_ITER,
) -> float:
    """Solve the steady-flow dispersion relation for wavenumber ``k``.

    For a ship moving at speed ``U``, the encounter frequency is
    ``omega_e = omega - k*U*cos(theta)``. For steady waves (``omega_e = 0``):

        ``omega = k*U*cos(theta)``

    Substituting into the dispersion relation ``omega^2 = g*k*tanh(k*h)``:

        ``k^2 * U^2 * cos^2(theta) = g * k * tanh(k*h)``
        ``k * U^2 * cos^2(theta) / g = tanh(k*h)``

    In deep water (``tanh -> 1``): ``k = g / (U^2 * cos^2(theta))``.

    Parameters
    ----------
    U : float
        Ship speed [m/s].
    depth : float
        Water depth [m].
    cos_theta : float
        Cosine of wave propagation angle relative to ship heading.
    gravity : float
        Gravitational acceleration [m/s^2].
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    float
        Wavenumber ``k`` [1/m].
    """
    if U <= 0.0:
        raise ValueError("U must be positive")
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    if cos_theta <= 0.0:
        return 0.0  # No wave propagation at theta = pi/2

    # Deep-water initial guess
    k0_dw = gravity / (U ** 2 * cos_theta ** 2)
    k = k0_dw

    # Solve: tanh(k*h) = k * U^2 * cos^2(theta) / g
    nu_steady = U ** 2 * cos_theta ** 2 / gravity

    for _ in range(max_iter):
        kh = k * depth
        tanh_kh = _safe_tanh(kh)
        cosh2_kh = _safe_cosh(kh) ** 2
        f = tanh_kh - k * nu_steady
        df = depth / cosh2_kh - nu_steady
        if abs(df) < 1e-300:
            break
        dk = f / df
        k_new = k - dk
        if k_new <= 0.0:
            # Fall back to bisection on the steady-flow equation
            return _steady_flow_bisection(U, depth, cos_theta, gravity)
        if abs(dk) / max(abs(k), 1e-300) < tol:
            return k_new
        k = k_new

    return k


def _steady_flow_bisection(
    U: float, depth: float, cos_theta: float, gravity: float,
) -> float:
    """Bisection solver for steady-flow dispersion as fallback."""
    nu_steady = U ** 2 * cos_theta ** 2 / gravity
    f = lambda k: _safe_tanh(k * depth) - k * nu_steady

    low = 0.0
    high = max(gravity / (U ** 2 * cos_theta ** 2), 1.0)
    while f(high) > 0.0:
        high *= 2.0
        if high > 1e12:
            return gravity / (U ** 2 * cos_theta ** 2)

    for _ in range(200):
        mid = 0.5 * (low + high)
        if f(mid) > 0.0:
            low = mid
        else:
            high = mid
        if high - low < 1e-14 * max(1.0, high):
            break
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# 2. Deep-water degeneration
# ---------------------------------------------------------------------------
def deep_water_degeneration_error(
    nu: float, depth: float, *,
    x: float = 1.0, z: float = -0.5, source_z: float = -1.0,
) -> float:
    """Relative error of finite-depth Green function vs deep-water limit."""
    from . import free_surface

    g_fd = _fd.green_function(x, z, 0.0, source_z, nu, depth)
    g_dw = free_surface.green_function(x, z, 0.0, source_z, nu)

    if abs(g_dw) < 1e-300:
        return abs(g_fd - g_dw)
    return abs(g_fd - g_dw) / abs(g_dw)


# ---------------------------------------------------------------------------
# 3. Shallow-water long-wave limit
# ---------------------------------------------------------------------------
def shallow_water_phase_speed(depth: float, *, gravity: float = DEFAULT_GRAVITY) -> float:
    """Long-wave phase speed ``c = sqrt(g*h)`` in shallow water."""
    return math.sqrt(gravity * depth)


def shallow_water_limit_error(
    nu: float, depth: float, *, gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Relative error of finite-depth phase speed vs shallow-water limit."""
    c_fd = _fd.phase_velocity(nu, depth, gravity=gravity)
    c_sw = shallow_water_phase_speed(depth, gravity=gravity)
    if abs(c_sw) < 1e-300:
        return abs(c_fd - c_sw)
    return abs(c_fd - c_sw) / abs(c_sw)


# ---------------------------------------------------------------------------
# 4. Havelock finite-depth wave-making resistance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HavelockFiniteDepthResult:
    """Result of Havelock finite-depth wave resistance calculation."""
    R_w: float
    C_w: float
    k0: float
    depth: float
    Fn: float
    h_L_ratio: float
    A_theta: ComplexArray
    theta_nodes: FloatArray


def _finite_depth_vertical_factor(z_arr, k, depth):
    """Vertical mode factor phi_0(z)/phi_0(0) = cosh(k*(z+h))/cosh(k*h).

    Numerically stable form.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    kh = k * depth
    if kh > 350:
        return np.exp(k * z_arr)
    exp_kz = np.exp(k * z_arr)
    exp_neg = np.exp(-k * (z_arr + 2.0 * depth))
    denom = 1.0 + math.exp(-2.0 * kh)
    return (exp_kz + exp_neg) / denom


def havelock_finite_depth_resistance(
    panels: "michell.CenterplanePanels",
    U: float,
    depth: float,
    L: float,
    T: float,
    *,
    gravity: float = DEFAULT_GRAVITY,
    density: float = DEFAULT_DENSITY,
    n_theta: int = 128,
) -> HavelockFiniteDepthResult:
    """Havelock wave-making resistance in finite depth.

    The finite-depth Kochin function is:

        A(theta) = int int y_x(x,z) * phi_0(z)/phi_0(0) *
                   exp(i * alpha * x) dx dz

    where ``alpha = k * cos(theta)`` and ``k`` satisfies the steady-flow
    dispersion relation ``tanh(k*h) = k * U^2 * cos^2(theta) / g``.

    The wave resistance is:

        R_w = rho*g/(4*pi) * int_0^{pi/2} |A(theta)|^2 * cos(theta) dtheta

    Parameters
    ----------
    panels : CenterplanePanels
        Centrepanel source distribution.
    U : float
        Forward speed [m/s].
    depth : float
        Water depth [m].
    L : float
        Ship length [m].
    T : float
        Ship draft [m].
    gravity : float
        Gravitational acceleration [m/s^2].
    density : float
        Water density [kg/m^3].
    n_theta : int
        Number of quadrature points in theta.

    Returns
    -------
    HavelockFiniteDepthResult
    """
    # Wavenumber at theta=0 (transverse waves)
    k0 = steady_flow_wavenumber(U, depth, cos_theta=1.0, gravity=gravity)

    # Froude number and depth ratio
    Fn = U / math.sqrt(gravity * L)
    h_L = depth / L

    theta_nodes = np.linspace(0.0, math.pi / 2.0 - 1e-10, n_theta)
    A_theta = np.zeros(n_theta, dtype=complex)

    for i, theta in enumerate(theta_nodes):
        cos_t = math.cos(theta)
        if cos_t < 1e-10:
            A_theta[i] = 0.0
            continue

        # Finite-depth wavenumber at angle theta
        k_theta = steady_flow_wavenumber(U, depth, cos_theta=cos_t, gravity=gravity)

        # Horizontal wavenumber component
        alpha = k_theta * cos_t

        # Kochin function with finite-depth vertical factor
        vf = _finite_depth_vertical_factor(panels.z, k_theta, depth)
        phase = np.exp(1j * alpha * panels.x)
        A_theta[i] = np.sum(panels.weight * vf * phase)

    # Wave resistance integral
    # Using the same formula as deep-water; finite-depth effects are captured
    # in the Kochin function through the vertical mode factor and wavenumber.
    integrand = np.zeros(n_theta)
    for i, theta in enumerate(theta_nodes):
        cos_t = math.cos(theta)
        if cos_t < 1e-10:
            continue
        integrand[i] = abs(A_theta[i]) ** 2 * cos_t

    # Trapezoidal integration
    integral = np.trapezoid(integrand, theta_nodes)

    R_w = density * gravity / (4.0 * math.pi) * integral
    C_w = R_w / (0.5 * density * U ** 2 * L * T)

    return HavelockFiniteDepthResult(
        R_w=R_w,
        C_w=C_w,
        k0=k0,
        depth=depth,
        Fn=Fn,
        h_L_ratio=h_L,
        A_theta=A_theta,
        theta_nodes=theta_nodes,
    )


def havelock_deep_water_resistance(
    panels: "michell.CenterplanePanels",
    U: float,
    L: float,
    T: float,
    *,
    gravity: float = DEFAULT_GRAVITY,
    density: float = DEFAULT_DENSITY,
    n_theta: int = 128,
) -> float:
    """Deep-water Havelock wave resistance (reference for degeneration test)."""
    k0 = gravity / U ** 2  # deep-water wavenumber

    theta_nodes = np.linspace(0.0, math.pi / 2.0 - 1e-10, n_theta)
    A_theta = np.zeros(n_theta, dtype=complex)

    for i, theta in enumerate(theta_nodes):
        cos_t = math.cos(theta)
        if cos_t < 1e-10:
            A_theta[i] = 0.0
            continue

        k_theta = k0 / (cos_t ** 2)
        alpha = k_theta * cos_t

        phase = np.exp(k_theta * panels.z + 1j * alpha * panels.x)
        A_theta[i] = np.sum(panels.weight * phase)

    integrand = np.zeros(n_theta)
    for i, theta in enumerate(theta_nodes):
        cos_t = math.cos(theta)
        if cos_t < 1e-10:
            continue
        integrand[i] = abs(A_theta[i]) ** 2 * cos_t

    integral = np.trapezoid(integrand, theta_nodes)
    R_w = density * gravity / (4.0 * math.pi) * integral
    return R_w


# ---------------------------------------------------------------------------
# 5. Parameter study utilities
# ---------------------------------------------------------------------------
def parameter_study(
    L: float = 100.0,
    B: float = 15.0,
    T: float = 5.0,
    h_L_ratios: Optional[list[float]] = None,
    Fn_values: Optional[list[float]] = None,
    *,
    gravity: float = DEFAULT_GRAVITY,
    density: float = DEFAULT_DENSITY,
    n_theta: int = 128,
) -> list[dict]:
    """Run a parameter study over depth ratios and Froude numbers."""
    if h_L_ratios is None:
        h_L_ratios = [0.1, 0.2, 0.5, 1.0, 2.0]
    if Fn_values is None:
        Fn_values = [0.1, 0.2, 0.3, 0.4]

    panels = michell.wigley_panels(L, B, T, n_x=32, n_z=16)
    results = []

    for h_L in h_L_ratios:
        depth = h_L * L
        for Fn in Fn_values:
            U = Fn * math.sqrt(gravity * L)
            nu = U ** 2 / gravity

            # Dispersion relation (oscillatory)
            k_nr = dispersion_newton_raphson(U, depth, gravity=gravity)
            k_bisect = _fd.finite_depth_wavenumber(nu, depth)
            dispersion_error = abs(k_nr - k_bisect) / max(abs(k_bisect), 1e-300)

            # Phase and group velocities
            c_phase = _fd.phase_velocity(nu, depth, gravity=gravity)
            c_group = _fd.group_velocity(nu, depth, gravity=gravity)

            # Shallow water limit
            c_shallow = shallow_water_phase_speed(depth, gravity=gravity)
            shallow_error = abs(c_phase - c_shallow) / max(abs(c_shallow), 1e-300)

            # Deep water limit
            c_deep = math.sqrt(gravity / k_bisect)
            deep_error = abs(c_phase - c_deep) / max(abs(c_deep), 1e-300)

            # Wave resistance
            try:
                hfd = havelock_finite_depth_resistance(
                    panels, U, depth, L, T,
                    gravity=gravity, density=density, n_theta=n_theta,
                )
                R_w = hfd.R_w
                C_w = hfd.C_w
                k0 = hfd.k0
            except Exception:
                R_w = float("nan")
                C_w = float("nan")
                k0 = float("nan")

            results.append({
                "h_L": h_L,
                "Fn": Fn,
                "depth": depth,
                "U": U,
                "R_w": R_w,
                "C_w": C_w,
                "k0": k0,
                "c_phase": c_phase,
                "c_group": c_group,
                "c_shallow": c_shallow,
                "c_deep": c_deep,
                "dispersion_error": dispersion_error,
                "shallow_error": shallow_error,
                "deep_error": deep_error,
            })

    return results
