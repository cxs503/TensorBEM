"""Mesh convergence study for Wigley III RAO.

Validates that the RAO converges with mesh refinement. The criterion is
that the heave RAO at a representative frequency (ω = 1.0 rad/s) changes
by less than 2% between the finest two mesh levels.

Mesh levels: N×M from 10×5 to 40×20 (longitudinal × vertical).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.hydrostatic import (
    HydrostaticProperties,
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)
from tensorbem.wigley3_hull import (
    WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_ALPHA,
    wigley3_displaced_volume,
    wigley3_wetted_hull_mesh,
)

FloatArray = NDArray[np.float64]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


@dataclass(frozen=True)
class MeshConvergenceStudy:
    """Result of mesh convergence study."""

    # Hull parameters
    length: float
    beam: float
    draft: float
    # Test frequency
    omega: float
    # Mesh levels
    nx_values: list[int]
    nz_values: list[int]
    panel_counts: list[int]
    h_mean_values: list[float]
    # RAO results
    heave_rao_amplitudes: list[float]
    heave_added_mass: list[float]
    heave_damping: list[float]
    heave_excitation: list[float]
    # Convergence metrics
    relative_changes: list[float | None]  # Between consecutive levels
    max_relative_change: float | None
    converged: bool
    convergence_threshold: float


def run_wigley_mesh_convergence(
    omega: float = 1.0,
    nx_values: Sequence[int] | None = None,
    nz_values: Sequence[int] | None = None,
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    amplitude: float = 1.0,
    convergence_threshold: float = 0.02,
) -> MeshConvergenceStudy:
    """Run mesh convergence study for Wigley III RAO.

    Parameters
    ----------
    omega : float
        Test frequency (rad/s).
    nx_values : sequence
        Longitudinal panel counts. Default: [10, 15, 20, 30, 40].
    nz_values : sequence
        Vertical panel counts. Default: [5, 8, 10, 15, 20].
    L, B, T : float
        Hull dimensions (m).
    alpha : float
        Wigley III modification parameter.
    density : float
        Fluid density (kg/m³).
    gravity : float
        Gravitational acceleration (m/s²).
    amplitude : float
        Wave amplitude (m).
    convergence_threshold : float
        Maximum allowed relative change between finest two levels.

    Returns
    -------
    MeshConvergenceStudy
    """
    if nx_values is None:
        nx_values = [10, 15, 20, 30, 40]
    if nz_values is None:
        nz_values = [5, 8, 10, 15, 20]

    nx_list = list(nx_values)
    nz_list = list(nz_values)
    n_levels = min(len(nx_list), len(nz_list))
    if n_levels < 2:
        raise ValueError("need at least 2 mesh levels for convergence study")

    nu = omega ** 2 / gravity
    volume = wigley3_displaced_volume(L=L, B=B, T=T, alpha=alpha)
    mass = density * volume
    cg = np.array([0.0, 0.0, -0.5 * T])

    rao_amps: list[float] = []
    a33_vals: list[float] = []
    b33_vals: list[float] = []
    f_exc_vals: list[float] = []
    panel_counts: list[int] = []
    h_means: list[float] = []

    for idx in range(n_levels):
        nx = nx_list[idx]
        nz = nz_list[idx]

        mesh = wigley3_wetted_hull_mesh(nx=nx, nz=nz, L=L, B=B, T=T, alpha=alpha)
        panel_counts.append(mesh.panel_count)
        h_means.append(mesh.h_mean)

        # Hydrostatics
        hydro_props: HydrostaticProperties = compute_hydrostatic_restoring_matrix(
            mesh, mass=mass, center_of_gravity=cg,
            density=density, gravity=gravity,
        )
        C = hydro_props.restoring_matrix
        # Mass matrix - note: compute_mass_matrix takes mass as first arg
        M_rigid = compute_mass_matrix(
            mass=mass, center_of_gravity=cg,
        )

        # Radiation
        ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
        rad = solve_radiation_3d(
            mesh, nu, modes=(2,), density=density, gravity=gravity,
            operators=ops,
        )

        # Diffraction
        diff = solve_diffraction_3d(
            mesh, nu, wave_angle=0.0, amplitude=amplitude,
            density=density, gravity=gravity, operators=ops,
        )

        a33 = float(rad.added_mass[0, 0])
        b33 = float(rad.damping[0, 0])
        f_exc = complex(diff.excitation_force[2])

        a33_vals.append(a33)
        b33_vals.append(b33)
        f_exc_vals.append(abs(f_exc))

        # RAO
        m33 = M_rigid[2, 2]
        c33 = C[2, 2]
        impedance = -omega ** 2 * (m33 + a33) + 1j * omega * b33 + c33
        rao = f_exc / impedance
        rao_amps.append(abs(rao))

    # Compute relative changes
    rel_changes: list[float | None] = [None]
    for idx in range(1, n_levels):
        if rao_amps[idx - 1] > 0:
            change = abs(rao_amps[idx] - rao_amps[idx - 1]) / rao_amps[idx - 1]
        else:
            change = 0.0
        rel_changes.append(change)

    max_change = max(c for c in rel_changes if c is not None) if any(c is not None for c in rel_changes) else None
    finest_change = rel_changes[-1] if rel_changes[-1] is not None else max_change
    converged = finest_change is not None and finest_change < convergence_threshold

    return MeshConvergenceStudy(
        length=L, beam=B, draft=T,
        omega=omega,
        nx_values=nx_list[:n_levels],
        nz_values=nz_list[:n_levels],
        panel_counts=panel_counts,
        h_mean_values=h_means,
        heave_rao_amplitudes=rao_amps,
        heave_added_mass=a33_vals,
        heave_damping=b33_vals,
        heave_excitation=f_exc_vals,
        relative_changes=rel_changes,
        max_relative_change=max_change,
        converged=converged,
        convergence_threshold=convergence_threshold,
    )
