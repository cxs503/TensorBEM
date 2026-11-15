"""BGK collision operator for D2Q9 Lattice Boltzmann Method.

This module implements the Bhatnagar-Gross-Krook (BGK) collision operator
for the D2Q9 lattice, which is the foundational collision model in LBM.

Reference: Bhatnagar, P.L., Gross, E.P. & Krook, M. (1954),
"A Model for Collision Processes in Gases. I. Small Amplitude Processes
in Charged and Neutral One-Component Systems", Phys. Rev. 94, 511.

The BGK operator relaxes the distribution function towards equilibrium:
    f_i^post = f_i - (1/tau) * (f_i - f_i^eq)

where tau is the relaxation time, related to kinematic viscosity by:
    nu = c_s^2 * (tau - 0.5) * dt

For D2Q9: c_s^2 = 1/3, dt = 1 (lattice units).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# D2Q9 lattice vectors
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CS_SQ = 1.0 / 3.0  # Speed of sound squared (lattice units)


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Compute equilibrium distribution function for D2Q9.

    f_i^eq = w_i * rho * [1 + 3(e_i·u) + 9/2(e_i·u)^2 - 3/2(u·u)]

    Parameters
    ----------
    rho : ndarray
        Density field, shape (ny, nx).
    ux : ndarray
        x-velocity field, shape (ny, nx).
    uy : ndarray
        y-velocity field, shape (ny, nx).

    Returns
    -------
    ndarray
        Equilibrium distribution, shape (ny, nx, 9).
    """
    feq = np.zeros((*rho.shape, 9))
    u_sq = ux**2 + uy**2
    for i in range(9):
        eu = CX[i] * ux + CY[i] * uy
        feq[:, :, i] = W[i] * rho * (1 + 3*eu + 4.5*eu**2 - 1.5*u_sq)
    return feq


def bgk_collision(f: np.ndarray, tau: float) -> np.ndarray:
    """Apply BGK collision operator.

    f_i^post = f_i - (1/tau) * (f_i - f_i^eq)

    Parameters
    ----------
    f : ndarray
        Distribution function, shape (ny, nx, 9).
    tau : float
        Relaxation time (must be > 0.5 for stability).

    Returns
    -------
    ndarray
        Post-collision distribution, shape (ny, nx, 9).

    Raises
    ------
    ValueError
        If tau <= 0.5 (unstable regime).
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5 for stability, got {tau}")

    rho = f.sum(axis=-1)
    ux = (f * CX).sum(axis=-1) / rho
    uy = (f * CY).sum(axis=-1) / rho

    feq = equilibrium(rho, ux, uy)
    omega = 1.0 / tau
    return f - omega * (f - feq)


def tau_from_viscosity(nu: float) -> float:
    """Compute relaxation time from kinematic viscosity.

    nu = c_s^2 * (tau - 0.5)  =>  tau = nu / c_s^2 + 0.5

    Parameters
    ----------
    nu : float
        Kinematic viscosity (lattice units).

    Returns
    -------
    float
        Relaxation time tau.
    """
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    return nu / CS_SQ + 0.5


def viscosity_from_tau(tau: float) -> float:
    """Compute kinematic viscosity from relaxation time.

    Parameters
    ----------
    tau : float
        Relaxation time.

    Returns
    -------
    float
        Kinematic viscosity (lattice units).
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    return CS_SQ * (tau - 0.5)


def macroscopic(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute macroscopic quantities from distribution function.

    rho = sum(f_i)
    rho*u = sum(f_i * e_i)

    Parameters
    ----------
    f : ndarray
        Distribution function, shape (ny, nx, 9).

    Returns
    -------
    tuple
        (rho, ux, uy) each with shape (ny, nx).
    """
    rho = f.sum(axis=-1)
    ux = (f * CX).sum(axis=-1) / rho
    uy = (f * CY).sum(axis=-1) / rho
    return rho, ux, uy


def poissonuille_analytical(y: np.ndarray, H: int, u_max: float) -> np.ndarray:
    """Analytical Poiseuille profile for channel flow.

    u(y) = u_max * (1 - (2y/H - 1)^2)

    Parameters
    ----------
    y : ndarray
        y-coordinates.
    H : int
        Channel height.
    u_max : float
        Maximum velocity (centerline).

    Returns
    -------
    ndarray
        Velocity profile.
    """
    return u_max * (1 - (2*y/H - 1)**2)


def benchmark_bgk_relaxation() -> dict:
    """Benchmark: verify BGK relaxation towards equilibrium.

    Starting from a non-equilibrium state, the BGK operator should
    relax the distribution towards equilibrium with rate 1/tau.

    Returns
    -------
    dict
        Benchmark results with relaxation rate verification.
    """
    ny, nx = 16, 16
    rho0 = 1.0
    u0 = 0.05

    # Initial state: equilibrium + perturbation
    rng = np.random.default_rng(42)
    feq = equilibrium(np.full((ny, nx), rho0), np.full((ny, nx), u0), np.zeros((ny, nx)))
    f = feq + rng.normal(scale=0.01, size=feq.shape)

    tau = 1.0
    omega = 1.0 / tau

    # Track relaxation
    initial_diff = ((f - feq)**2).sum()
    results = {"tau": tau, "omega": omega, "initial_diff": initial_diff, "steps": []}

    for step in range(10):
        f = bgk_collision(f, tau)
        feq_new = equilibrium(*macroscopic(f))
        diff = ((f - feq_new)**2).sum()
        expected_diff = initial_diff * (1 - omega)**(2*(step+1))
        results["steps"].append({
            "step": step + 1,
            "actual_diff": float(diff),
            "expected_diff": float(expected_diff),
            "ratio": float(diff / expected_diff) if expected_diff > 0 else float("inf"),
        })

    return results
