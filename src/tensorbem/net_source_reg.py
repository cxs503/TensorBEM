"""Low-frequency net-source regularization for 2-D BEM.

Problem
-------
At low wavenumbers (ka ≪ 1), the indirect BEM system for radiation and
diffraction becomes ill-conditioned.  The root cause is the *net source mode*:
a constant source density σ = const on a closed body contour produces a
potential whose normal derivative on the body is nearly zero (exactly zero
for a circle in free space).  The system matrix has a near-null space in the
constant-source direction.

Solution
--------
We regularize by adding a small Tikhonov term that pins the net source
component:

    (K' + ½I + ε · P_net) σ = rhs

where P_net is the orthogonal projector onto the net source mode:

    P_net σ = (1/L) · (∮ σ dl) · 1

and ε = α · ‖K' + ½I‖_∞ with α a small dimensionless number (default 10⁻⁸).

This approach is equivalent to the Lewis/Ursell multipole truncation at low
order: the net source mode corresponds to the monopole term.

References
----------
- Ursell, F. (1949).  *Q.J. Mech. Appl. Math.*, 2(2), 218–231.
- Yeung, R.W. (1973).  *PhD thesis, UC Berkeley*.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from .free_surface_solver import (
    DiffractionResult,
    RadiationResult,
    assemble_indirect_matrix,
    _evaluate_source_potential,
    _validate_mesh_depth,
    _validate_panel_nodes,
    _validate_positive_scalar,
    incident_airy_wave,
)
from .geometry import PanelMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0
_DEFAULT_REG_ALPHA = 1e-8


@dataclass(frozen=True)
class RegularizationInfo:
    """Information about the regularization applied."""
    alpha: float
    epsilon: float
    operator_norm: float
    condition_before: float
    condition_after: float
    net_source_component: float


def _net_source_projector(mesh: PanelMesh) -> FloatArray:
    """Build the net source projector P_net = (1/L) · 1 · 1^T · diag(dl)."""
    lengths = mesh.lengths
    total_length = np.sum(lengths)
    n = mesh.panel_count
    return np.outer(np.ones(n), lengths) / total_length


def _regularized_operator(
    mesh: PanelMesh,
    nu: float,
    panel_nodes: int,
    alpha: float,
) -> tuple[ComplexArray, RegularizationInfo]:
    """Build the regularized BEM operator."""
    K = assemble_indirect_matrix(mesh, nu, panel_nodes=panel_nodes)
    operator = K + 0.5 * np.eye(mesh.panel_count)

    op_norm = float(np.linalg.norm(operator, ord=np.inf))
    cond_before = float(np.linalg.cond(operator))

    epsilon = alpha * op_norm
    P_net = _net_source_projector(mesh)
    reg_operator = operator + epsilon * P_net

    cond_after = float(np.linalg.cond(reg_operator))

    info = RegularizationInfo(
        alpha=alpha,
        epsilon=epsilon,
        operator_norm=op_norm,
        condition_before=cond_before,
        condition_after=cond_after,
        net_source_component=0.0,
    )

    return reg_operator, info


def solve_radiation_regularized(
    mesh: PanelMesh,
    nu: float,
    direction: str,
    *,
    density: float = _DEFAULT_DENSITY,
    panel_nodes: int = 24,
    alpha: float = _DEFAULT_REG_ALPHA,
) -> tuple[RadiationResult, RegularizationInfo]:
    """Solve the radiation problem with net-source regularization.

    Returns
    -------
    result : RadiationResult
    info : RegularizationInfo
    """
    if direction == "heave":
        direction_vector = np.array([0.0, 1.0])
    elif direction == "sway":
        direction_vector = np.array([1.0, 0.0])
    else:
        raise ValueError("direction must be 'heave' or 'sway'")

    nodes_count = _validate_panel_nodes(panel_nodes)
    wavenumber = _validate_mesh_depth(mesh, nu)

    reg_operator, info = _regularized_operator(mesh, wavenumber, nodes_count, alpha)

    data = mesh.normals @ direction_vector
    sources = np.linalg.solve(reg_operator, data)
    potential = _evaluate_source_potential(mesh, sources, wavenumber, nodes_count)

    force = -(potential * mesh.lengths) @ mesh.normals
    omega = math.sqrt(wavenumber * _GRAVITY)

    # Net source component
    lengths = mesh.lengths
    total_length = np.sum(lengths)
    net_source = float(abs(np.sum(sources * lengths)) / total_length)

    residual = reg_operator @ sources - data

    info = RegularizationInfo(
        alpha=info.alpha,
        epsilon=info.epsilon,
        operator_norm=info.operator_norm,
        condition_before=info.condition_before,
        condition_after=info.condition_after,
        net_source_component=net_source,
    )

    result = RadiationResult(
        potential=potential,
        normal_velocity=data,
        added_mass=density * np.real(force),
        damping=omega * density * np.imag(force),
        condition_number=float(np.linalg.cond(reg_operator)),
        relative_residual=float(
            np.linalg.norm(residual) / max(np.linalg.norm(data), 1e-300)
        ),
    )

    return result, info


def solve_diffraction_regularized(
    mesh: PanelMesh,
    nu: float,
    amplitude: float = 1.0,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _GRAVITY,
    direction: int = 1,
    panel_nodes: int = 24,
    alpha: float = _DEFAULT_REG_ALPHA,
) -> tuple[DiffractionResult, RegularizationInfo]:
    """Solve the diffraction problem with net-source regularization."""
    nodes_count = _validate_panel_nodes(panel_nodes)
    wavenumber = _validate_mesh_depth(mesh, nu)
    amp = _validate_positive_scalar("amplitude", amplitude)
    rho = _validate_positive_scalar("density", density)
    g = _validate_positive_scalar("gravity", gravity)

    reg_operator, info = _regularized_operator(mesh, wavenumber, nodes_count, alpha)

    phi_inc, dphi_inc_dn = incident_airy_wave(
        mesh.centers, mesh.normals, wavenumber, amp,
        gravity=g, direction=direction,
    )

    rhs = -dphi_inc_dn
    sources = np.linalg.solve(reg_operator, rhs)
    phi_scat = _evaluate_source_potential(mesh, sources, wavenumber, nodes_count)
    phi_total = phi_inc + phi_scat

    omega = math.sqrt(wavenumber * g)
    excitation_force = -1.0j * omega * rho * ((phi_total * mesh.lengths) @ mesh.normals)

    # Reflection/transmission coefficients
    nodes, weights = leggauss(nodes_count)
    fraction = 0.5 * (nodes + 1.0)
    qx = mesh.vertices[:, 0][:, None] + fraction[None, :] * mesh.vectors[:, 0][:, None]
    qz = mesh.vertices[:, 1][:, None] + fraction[None, :] * mesh.vectors[:, 1][:, None]
    panel_weights = 0.5 * mesh.lengths[:, None] * weights[None, :]
    c_plus = (omega / (g * amp)) * complex(
        np.sum(
            np.sum(np.exp(wavenumber * qz - 1.0j * wavenumber * qx) * panel_weights, axis=1)
            * sources
        )
    )
    c_minus = (omega / (g * amp)) * complex(
        np.sum(
            np.sum(np.exp(wavenumber * qz + 1.0j * wavenumber * qx) * panel_weights, axis=1)
            * sources
        )
    )
    if direction == 1:
        reflection = c_minus
        transmission = 1.0 + c_plus
    else:
        reflection = c_plus
        transmission = 1.0 + c_minus

    energy_error = float(abs(reflection)**2 + abs(transmission)**2 - 1.0)

    lengths = mesh.lengths
    total_length = np.sum(lengths)
    net_source = float(abs(np.sum(sources * lengths)) / total_length)

    residual = reg_operator @ sources - rhs

    info = RegularizationInfo(
        alpha=info.alpha,
        epsilon=info.epsilon,
        operator_norm=info.operator_norm,
        condition_before=info.condition_before,
        condition_after=info.condition_after,
        net_source_component=net_source,
    )

    result = DiffractionResult(
        total_potential=phi_total,
        scattered_potential=phi_scat,
        incident_potential=phi_inc,
        incident_normal_velocity=dphi_inc_dn,
        source_density=sources,
        excitation_force=excitation_force,
        reflection_coefficient=complex(reflection),
        transmission_coefficient=complex(transmission),
        energy_balance_error=energy_error,
        condition_number=float(np.linalg.cond(reg_operator)),
        relative_residual=float(
            np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300)
        ),
    )

    return result, info


def gauge_band_regularization(
    mesh: PanelMesh,
    ka: float,
    radius: float,
    mode: str = "heave",
    *,
    alpha_values: tuple[float, ...] = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
    density: float = _DEFAULT_DENSITY,
    panel_nodes: int = 24,
) -> dict:
    """Measure the gauge band (sensitivity to regularization) at given ka.

    The gauge band is the ratio of the maximum to minimum added mass over
    the range of regularization parameters.  A well-posed problem has gauge
    band close to 1.0.
    """
    nu = ka / radius
    added_masses = []
    conditions = []

    # Mode index: heave=1 (z), sway=0 (x)
    mode_idx = 1 if mode == "heave" else 0

    for alpha in alpha_values:
        result, info = solve_radiation_regularized(
            mesh, nu, mode,
            density=density,
            panel_nodes=panel_nodes,
            alpha=alpha,
        )
        added_masses.append(float(result.added_mass[mode_idx]))
        conditions.append(info.condition_after)

    added_masses = np.array(added_masses)
    am_max = np.max(np.abs(added_masses))
    am_min = np.min(np.abs(added_masses))
    gauge_band = am_max / am_min if am_min > 0 else float('inf')

    median_am = np.median(np.abs(added_masses))
    optimal_idx = np.argmin(np.abs(np.abs(added_masses) - median_am))
    optimal_alpha = alpha_values[optimal_idx]

    return {
        "ka": ka,
        "added_masses": added_masses.tolist(),
        "gauge_band": float(gauge_band),
        "optimal_alpha": optimal_alpha,
        "condition_numbers": conditions,
        "alpha_values": list(alpha_values),
    }
