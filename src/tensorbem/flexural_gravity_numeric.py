"""Independent finite-difference check of the flexural-gravity dispersion.

The closed-form relation implemented in :mod:`tensorbem.flexural_gravity`
assumes a plane wave ``phi = psi(z) exp(i(kx - omega t))`` in a fluid of depth
``H``.  Laplace's equation then gives ``psi'' = k^2 psi`` with ``psi'(-H) = 0``,
so the *depth factor*

    g0(k, H) = psi'(0) / psi(0) = k tanh(kH)

is the only place where the geometry enters.  This module computes ``g0`` by a
**second-order finite-difference discretisation** of that two-point boundary
value problem (tridiagonal solve, ghost-node Neumann condition at the bottom)
instead of evaluating ``tanh`` analytically, and then solves the linearised
ice/water free-surface condition

    (rho_w g + D k^4 - rho_i h omega^2) g0 = rho_w omega^2

for ``omega``.  Rearranging gives the same eq. (3) as
:func:`tensorbem.flexural_gravity.plate_fluid_omega2`, but the depth factor now
carries the discretisation error, so agreement is a numerical check rather than
an algebraic identity.  Convergence is second order in ``dz`` (verified in
``tests/test_flexural_gravity_evidence.py`` and in the benchmark artifact).

All quantities are SI.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_GRAVITY = 9.80665
DEFAULT_WATER_DENSITY = 1025.0
DEFAULT_ICE_DENSITY = 917.0


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def _non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return value


def fd_depth_gradient_ratio(
    k: float,
    depth: float,
    *,
    dz: float | None = None,
    max_nodes: int = 400_000,
) -> tuple[float, float, int]:
    """``psi'(0)/psi(0)`` from a finite-difference solve of ``psi'' = k^2 psi``.

    Returns ``(g0, dz, nodes)``.  ``psi(-H)`` Neumann (rigid bottom) and
    ``psi(0) = 1`` (normalisation); ``psi'(0)`` is a second-order one-sided
    difference.
    """
    from scipy.linalg import solve_banded

    k = _positive("wavenumber", k)
    H = _positive("depth", depth)
    if dz is None:
        dz = min(0.05, 0.02 / k)
    dz = _positive("dz", dz)
    n = int(math.ceil(H / dz))
    n = max(4, min(n, max_nodes))
    dz = H / n
    a = (k * dz) ** 2

    # Unknowns x_0..x_n; row 0 uses the ghost node x_-1 = x_1 (Neumann).
    #   row 0    : 2 x_1 - (2 + a) x_0            = 0
    #   row i    : x_{i-1} - (2 + a) x_i + x_{i+1} = 0
    #   row n    : x_n = 1
    ab = np.zeros((3, n + 1))
    ab[1, :] = -(2.0 + a)
    ab[0, 1:] = 1.0
    ab[2, :-1] = 1.0
    ab[1, 0] = -(2.0 + a)
    ab[0, 1] = 2.0
    ab[2, -2] = 0.0
    ab[1, -1] = 1.0
    rhs = np.zeros(n + 1)
    rhs[-1] = 1.0
    x = solve_banded((1, 1), ab, rhs)

    gradient = (3.0 * x[n] - 4.0 * x[n - 1] + x[n - 2]) / (2.0 * dz)
    return float(gradient / x[n]), dz, n


def fd_omega2(
    k: float,
    *,
    depth: float,
    rigidity: float = 0.0,
    mass_per_area: float = 0.0,
    fluid_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    dz: float | None = None,
) -> float:
    """``omega^2`` from the FD depth factor and the free-surface condition."""
    g0, _, _ = fd_depth_gradient_ratio(k, depth, dz=dz)
    return omega2_from_depth_factor(
        g0,
        k=k,
        rigidity=rigidity,
        mass_per_area=mass_per_area,
        fluid_density=fluid_density,
        gravity=gravity,
    )


def omega2_from_depth_factor(
    g0: float,
    *,
    k: float,
    rigidity: float = 0.0,
    mass_per_area: float = 0.0,
    fluid_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Solve the free-surface condition for ``omega^2`` given ``g0``.

    ``(rho g + D k^4 - m'' omega^2) g0 = rho omega^2``.
    """
    k = _positive("wavenumber", k)
    g0 = _positive("depth factor", g0)
    rho = _positive("fluid_density", fluid_density)
    g = _positive("gravity", gravity)
    D = _non_negative("rigidity", rigidity)
    m_pp = _non_negative("mass_per_area", mass_per_area)
    return g0 * (rho * g + D * k ** 4) / (rho + m_pp * g0)


def fd_omega(
    k: float,
    *,
    depth: float,
    rigidity: float = 0.0,
    mass_per_area: float = 0.0,
    fluid_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    dz: float | None = None,
) -> float:
    """``omega`` from the finite-difference depth factor."""
    return math.sqrt(
        fd_omega2(
            k,
            depth=depth,
            rigidity=rigidity,
            mass_per_area=mass_per_area,
            fluid_density=fluid_density,
            gravity=gravity,
            dz=dz,
        )
    )