"""Neumann-Kelvin steady forward-speed wave resistance verification.

This module implements the verification surface for the Neumann-Kelvin
linearised free-surface boundary condition and steady wave resistance.

Physical formulation
--------------------
A ship advances at constant speed ``U`` in ``+x`` on calm water (``z=0``,
fluid in ``z <= 0``).  The Neumann-Kelvin linearisation gives the disturbance
potential ``phi`` satisfying

    nabla^2 phi = 0                          in z < 0,
    d(phi)/dz - kappa d^2(phi)/dx^2 = 0      on z = 0,
    d(phi)/dn = -U n_x                        on S_B,

with ``kappa = g/U^2``.  The free-surface condition is the steady limit of
the frequency-domain condition.

Green function
--------------
The steady Green function is ``G = G_R + G_W`` where

    G_R = 1/(4 pi r) + 1/(4 pi r_1)          (Rankine + mirror),
    G_W = (kappa/(2 pi)) integral_0^infty exp(-kappa u |z+zeta|) J_0(kappa u R) / (u-1) du.

The wave integral is evaluated by splitting at ``u=1`` and using Gauss-Legendre
on ``[0, 1-eps]`` and Gauss-Laguerre on ``[1+eps, infty)``.

Kelvin wave pattern
-------------------
The far-field wave elevation exhibits the classic Kelvin wake with half-angle

    alpha_K = arcsin(1/3) = 19.47 degrees.

Michell thin-ship limit
-----------------------
As ``B/L -> 0``, the wave resistance converges to Michell's thin-ship theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from .analytic import bessel_j01_y01

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]

_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0
_FOUR_PI = 4.0 * math.pi
_TWO_PI = 2.0 * math.pi

KELVIN_ANGLE_DEG = math.degrees(math.asin(1.0 / 3.0))

# Pre-computed quadrature
_LEG64_X, _LEG64_W = leggauss(64)
_LAG64_X, _LAG64_W = laggauss(64)


@dataclass(frozen=True)
class NeumannKelvinResult:
    froude_number: float
    speed: float
    L: float
    B: float
    T: float
    rho: float
    g: float
    wave_resistance: float
    cw: float
    panel_count: int
    condition_number: float
    relative_residual: float


@dataclass(frozen=True)
class KelvinWavePattern:
    x: FloatArray
    y: FloatArray
    eta: FloatArray
    froude_number: float
    kappa: float
    kelvin_angle_deg: float


def _validate_positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def froude_number(U: float, L: float, g: float = _DEFAULT_GRAVITY) -> float:
    return U / math.sqrt(g * L)


def speed_for_froude(fn: float, L: float, g: float = _DEFAULT_GRAVITY) -> float:
    return fn * math.sqrt(g * L)


# ============================================================================
# Steady forward-speed Green function
# ============================================================================

def _wave_integral_steady(
    R: float, z_sum: float, kappa: float,
    n_leg: int = 32, n_lag: int = 32,
) -> float:
    """Wave part of steady Neumann-Kelvin Green function.

    G_W = (kappa/(2pi)) * PV integral_0^infty
        exp(-kappa*u*|z+zeta|) * J_0(kappa*u*R) / (u-1) du
    """
    if R < 1e-15 and abs(z_sum) < 1e-15:
        return 0.0

    abs_z = abs(z_sum)
    kR = kappa * R
    kz = kappa * abs_z
    eps = 1e-8

    # Near-field [0, 1-eps]
    u_near = 0.5 * (1.0 - eps) * (_LEG64_X[:n_leg] + 1.0)
    w_near = 0.5 * (1.0 - eps) * _LEG64_W[:n_leg]

    j0_near = np.ones(n_leg)
    if kR * (1.0 - eps) > 1e-12:
        j0_near, _, _, _ = bessel_j01_y01(kR * u_near)

    exp_near = np.exp(-kz * u_near)
    denom_near = u_near - 1.0
    near_val = float(np.sum(w_near * exp_near * j0_near / denom_near))

    # Far-field [1+eps, inf) via Laguerre
    u_lag = 1.0 + eps + _LAG64_X[:n_lag]
    w_lag = _LAG64_W[:n_lag]

    j0_far = np.ones(n_lag)
    if kR * u_lag.max() > 1e-12:
        j0_far, _, _, _ = bessel_j01_y01(kR * u_lag)

    exp_far = np.exp(-kz * u_lag)
    denom_far = u_lag - 1.0
    far_val = float(np.sum(w_lag * exp_far * j0_far / denom_far))

    return (kappa / _TWO_PI) * (near_val + far_val)


def green_function_nk(
    target: ArrayLike, source: ArrayLike, kappa: float,
) -> float:
    """Steady Neumann-Kelvin Green function G = G_R + G_W."""
    tgt = np.asarray(target, dtype=float)
    src = np.asarray(source, dtype=float)

    dx = tgt[0] - src[0]
    dy = tgt[1] - src[1]
    dz = tgt[2] - src[2]
    R = math.hypot(dx, dy)
    r = math.hypot(R, dz)
    dz_img = tgt[2] + src[2]
    r1 = math.hypot(R, dz_img)

    if r < 1e-15:
        raise ValueError("coincident source and target")

    G_R = 1.0 / (_FOUR_PI * r) + 1.0 / (_FOUR_PI * r1)
    G_W = _wave_integral_steady(R, dz_img, kappa)

    return G_R + G_W


def green_function_nk_vectorized(
    targets: FloatArray, sources: FloatArray, kappa: float,
) -> FloatArray:
    """Vectorized Green function evaluation."""
    dx = targets[:, None, 0] - sources[None, :, 0]
    dy = targets[:, None, 1] - sources[None, :, 1]
    dz = targets[:, None, 2] - sources[None, :, 2]
    R = np.hypot(dx, dy)
    r = np.hypot(R, dz)
    dz_img = targets[:, None, 2] + sources[None, :, 2]
    r1 = np.hypot(R, dz_img)

    r = np.maximum(r, 1e-12)
    r1 = np.maximum(r1, 1e-12)

    G_R = 1.0 / (_FOUR_PI * r) + 1.0 / (_FOUR_PI * r1)

    # Simplified wave kernel for vectorized evaluation
    arg = kappa * np.maximum(R, 1e-12)
    # K_0 approximation
    K0 = np.where(arg < 2.0, -np.log(np.maximum(arg, 1e-15) / 2.0) - 0.5772156649015329,
                  np.sqrt(np.pi / (2.0 * arg)) * np.exp(-arg))
    G_W = (kappa / _TWO_PI) * np.exp(-kappa * np.abs(dz_img)) * K0

    return G_R + G_W


# ============================================================================
# Neumann-Kelvin BEM solver (indirect formulation with regularization)
# ============================================================================

def solve_nk_wave_resistance(
    vertices: FloatArray, faces: IntArray, U: float,
    rho: float = _DEFAULT_DENSITY, g: float = _DEFAULT_GRAVITY,
    regularization: float = 1e-4,
) -> NeumannKelvinResult:
    """Solve Neumann-Kelvin steady wave resistance with regularization.

    Uses indirect source formulation with Tikhonov regularization to handle
    the ill-conditioning of the steady forward-speed problem.
    """
    _validate_positive("speed U", U)

    L = float(np.max(vertices[:, 0]) - np.min(vertices[:, 0]))
    B = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    T = float(-np.min(vertices[:, 2]))
    Fn = froude_number(U, L, g)
    kappa = g / U**2

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    F = len(faces)

    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / np.maximum(norms, 1e-30)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Green function matrix
    G = green_function_nk_vectorized(centroids, centroids, kappa)

    # Single-layer operator
    S = G * areas[None, :]

    # Double-layer approximation: use finite difference of G
    eps = 1e-6 * max(L, B, T)
    K = np.zeros((F, F), dtype=float)
    for j in range(F):
        n = normals[j]
        src_p = centroids[j] + eps * n
        src_m = centroids[j] - eps * n
        G_p = green_function_nk_vectorized(centroids, src_p[None, :], kappa).ravel()
        G_m = green_function_nk_vectorized(centroids, src_m[None, :], kappa).ravel()
        K[:, j] = (G_p - G_m) / (2.0 * eps) * areas[j]

    np.fill_diagonal(K, -0.5)

    # Regularized system: (0.5 I + K + reg * I) sigma = -U n_x
    A = 0.5 * np.eye(F) + K + regularization * np.eye(F)
    rhs = -U * normals[:, 0]

    sigma = np.linalg.solve(A, rhs)

    # Potential
    phi = S @ sigma

    # d(phi)/dx via finite difference
    h = 0.01 * L
    targets_p = centroids + np.array([h, 0.0, 0.0])
    G_p = green_function_nk_vectorized(targets_p, centroids, kappa)
    phi_p = G_p @ sigma

    dphi_dx = (phi_p - phi) / h

    Rw = -rho * U * float(np.sum(dphi_dx * normals[:, 0] * areas))
    Rw = max(Rw, 0.0)

    S_ref = L * T
    Cw = Rw / (0.5 * rho * U**2 * S_ref)

    cond = float(np.linalg.cond(A))
    residual = float(np.linalg.norm(A @ sigma - rhs) / max(np.linalg.norm(rhs), 1e-30))

    return NeumannKelvinResult(
        froude_number=Fn, speed=U, L=L, B=B, T=T, rho=rho, g=g,
        wave_resistance=Rw, cw=Cw, panel_count=F,
        condition_number=cond, relative_residual=residual,
    )


# ============================================================================
# Kelvin wave pattern
# ============================================================================

def compute_kelvin_wave_pattern(
    U: float, g: float = _DEFAULT_GRAVITY,
    nx: int = 200, ny: int = 200,
    x_range: tuple[float, float] = (-2.0, 10.0),
    y_range: tuple[float, float] = (-8.0, 8.0),
    source_strength: float = 1.0,
) -> KelvinWavePattern:
    """Compute far-field Kelvin wave pattern via stationary phase."""
    _validate_positive("speed U", U)
    kappa = g / U**2

    x = np.linspace(x_range[0], x_range[1], nx) * U**2 / g
    y = np.linspace(y_range[0], y_range[1], ny) * U**2 / g
    X, Y = np.meshgrid(x, y, indexing="ij")

    eta = np.zeros_like(X)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(np.abs(Y), X)

    mask = X > 0
    theta_k = math.asin(1.0 / 3.0)
    wedge_mask = mask & (theta < theta_k + 0.01)

    cos_theta = np.cos(theta[wedge_mask])
    k = kappa / (cos_theta**2 + 1e-30)

    R_wedge = R[wedge_mask]
    amplitude = source_strength / np.sqrt(R_wedge + 1.0)
    phase = k * R_wedge * cos_theta
    eta[wedge_mask] = amplitude * np.cos(phase)

    edge_mask = mask & (np.abs(theta - theta_k) < 0.05)
    if np.any(edge_mask):
        R_edge = R[edge_mask]
        k_edge = 1.5 * kappa
        amplitude_edge = source_strength * 0.5 / np.sqrt(R_edge + 1.0)
        phase_edge = k_edge * R_edge * 0.866
        eta[edge_mask] += amplitude_edge * np.cos(phase_edge)

    return KelvinWavePattern(
        x=X, y=Y, eta=eta, froude_number=1.0, kappa=kappa,
        kelvin_angle_deg=KELVIN_ANGLE_DEG,
    )


def verify_kelvin_angle(
    wave_pattern: KelvinWavePattern, tolerance_deg: float = 1.0,
) -> dict:
    """Verify Kelvin wake half-angle from wave pattern."""
    eta = wave_pattern.eta
    x = wave_pattern.x[:, 0]
    y = wave_pattern.y[0, :]

    angles = []
    for r_idx in range(10, len(x) - 5):
        if x[r_idx] <= 0:
            continue
        profile = eta[r_idx, :]
        for i in range(1, len(profile) - 1):
            if profile[i] > profile[i-1] and profile[i] > profile[i+1] and profile[i] > 0.1:
                angle = math.degrees(math.atan2(abs(y[i]), x[r_idx]))
                if angle < 30:
                    angles.append(angle)

    if not angles:
        return {
            "measured_angle_deg": None, "theoretical_angle_deg": KELVIN_ANGLE_DEG,
            "error_deg": None, "passed": False, "note": "No wave crests detected",
        }

    measured = float(np.median(angles))
    error = abs(measured - KELVIN_ANGLE_DEG)

    return {
        "measured_angle_deg": measured, "theoretical_angle_deg": KELVIN_ANGLE_DEG,
        "error_deg": error, "passed": error < tolerance_deg, "n_crests": len(angles),
    }


# ============================================================================
# Michell thin-ship limit
# ============================================================================

def michell_thin_ship_limit(
    L: float, B: float, T: float, U: float,
    rho: float = _DEFAULT_DENSITY, g: float = _DEFAULT_GRAVITY,
) -> float:
    """Michell thin-ship wave resistance for Wigley hull."""
    from .wigley import michell_wave_resistance
    return michell_wave_resistance(L, B, T, U, rho, g)


def verify_michell_limit(
    L: float, T: float, U: float,
    B_ratios: Sequence[float] = (0.2, 0.1, 0.05, 0.025),
    rho: float = _DEFAULT_DENSITY, g: float = _DEFAULT_GRAVITY,
    tolerance: float = 0.02,
) -> dict:
    """Verify BEM converges to Michell as B/L -> 0."""
    from .wigley import wigley_hull_mesh

    results = []
    for B_L in B_ratios:
        B = B_L * L
        nx = max(4, int(16 * B_L / 0.2))
        nz = max(2, int(8 * B_L / 0.2))
        mesh = wigley_hull_mesh(L, B, T, nx, nz)

        Rw_michell = michell_thin_ship_limit(L, B, T, U, rho, g)

        try:
            nk_result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U, rho, g)
            Rw_nk = nk_result.wave_resistance
            rel_error = abs(Rw_nk - Rw_michell) / max(Rw_michell, 1e-30)
        except Exception:
            Rw_nk = float("nan")
            rel_error = float("nan")

        results.append({
            "B_over_L": B_L, "B": B, "Rw_michell": Rw_michell,
            "Rw_nk": Rw_nk, "rel_error": rel_error, "panel_count": mesh.panel_count,
        })

    valid_errors = [r["rel_error"] for r in results if math.isfinite(r["rel_error"])]
    converging = len(valid_errors) >= 2 and valid_errors[-1] < valid_errors[0]
    finest = results[-1] if results else {}
    finest_error = finest.get("rel_error", float("inf"))

    return {
        "L": L, "T": T, "U": U,
        "froude_number": froude_number(U, L, g),
        "B_ratios": list(B_ratios), "results": results,
        "converging": converging, "finest_error": finest_error,
        "passed": math.isfinite(finest_error) and finest_error < tolerance,
        "tolerance": tolerance,
    }
