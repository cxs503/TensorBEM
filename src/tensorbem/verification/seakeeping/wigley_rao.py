"""Wigley III RAO frequency response with limit behavior validation.

The Wigley III hull (Journee 1992) is a standard mathematical hullform for
seakeeping verification. This module computes the RAO (Response Amplitude
Operator) spectrum and validates:

1. **Long-wave limit** (kL < 0.3): heave RAO → 1.0 (body follows wave surface)
2. **High-frequency limit** (ω = 3.0 rad/s): heave RAO < 0.2 (body cannot respond)
3. **Full spectrum**: ω ∈ [0.05, 3.0] rad/s

The RAO equation of motion in frequency domain is:

    [-ω²(M + A(ω)) + iω B(ω) + C] ξ(ω) = F_exc(ω)

where M = mass matrix, A = added mass, B = damping, C = hydrostatic restoring,
F_exc = excitation force, ξ = RAO vector.

Note: The default Wigley III parameters (L=3m) are model-scale. For limit
validation, we use a full-scale hull (L=100m) to properly exercise the
long-wave and high-frequency regimes.
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
from tensorbem.geometry3d import WettedHullMesh
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
ComplexArray = NDArray[np.complex128]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665

# Full-scale Wigley III for limit validation (L=100m ship-scale)
WIGLEY3_FULL_L = 100.0
WIGLEY3_FULL_B = 15.0
WIGLEY3_FULL_T = 6.0


@dataclass(frozen=True)
class WigleyRAOStudy:
    """Result of Wigley III RAO frequency sweep."""

    # Hull parameters
    length: float
    beam: float
    draft: float
    alpha: float
    displaced_volume: float
    mass: float
    # Frequency sweep
    omegas: FloatArray
    nus: FloatArray
    kL_values: FloatArray
    # RAO results (heave, index 2)
    heave_rao_amplitude: FloatArray
    heave_rao_phase: FloatArray
    heave_added_mass: FloatArray
    heave_damping: FloatArray
    heave_excitation_magnitude: FloatArray
    # Limit validation
    long_wave_rao_mean: float  # Mean heave RAO for kL < 0.3
    long_wave_rao_std: float
    high_freq_rao: float  # Heave RAO at omega = 3.0
    # Mesh info
    nx: int
    nz: int
    panel_count: int


def compute_wigley_rao_spectrum(
    omega_values: Sequence[float] | None = None,
    nx: int = 20,
    nz: int = 10,
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    alpha: float = WIGLEY3_ALPHA,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    amplitude: float = 1.0,
) -> WigleyRAOStudy:
    """Compute Wigley III RAO spectrum across frequency range.

    Parameters
    ----------
    omega_values : sequence or None
        Wave frequencies (rad/s). Default: np.linspace(0.05, 3.0, 30).
    nx : int
        Longitudinal panel count.
    nz : int
        Vertical panel count.
    L, B, T : float
        Hull length, beam, draft (m).
    alpha : float
        Wigley III modification parameter.
    density : float
        Fluid density (kg/m³).
    gravity : float
        Gravitational acceleration (m/s²).
    amplitude : float
        Wave amplitude (m).

    Returns
    -------
    WigleyRAOStudy
    """
    if omega_values is None:
        omega_values = np.linspace(0.05, 3.0, 30)

    omegas = np.asarray(omega_values, dtype=float)
    if np.any(omegas <= 0):
        raise ValueError("omega values must be positive")

    # Generate hull mesh
    mesh = wigley3_wetted_hull_mesh(nx=nx, nz=nz, L=L, B=B, T=T, alpha=alpha)

    # Hull properties
    volume = wigley3_displaced_volume(L=L, B=B, T=T, alpha=alpha)
    mass = density * volume  # Archimedes equilibrium

    # Center of gravity (at draft midpoint for simplicity)
    cg = np.array([0.0, 0.0, -0.5 * T])

    # Hydrostatic properties (includes restoring matrix)
    hydro_props: HydrostaticProperties = compute_hydrostatic_restoring_matrix(
        mesh, mass=mass, center_of_gravity=cg,
        density=density, gravity=gravity,
    )
    C = hydro_props.restoring_matrix

    # Mass matrix (rigid body) - note: compute_mass_matrix takes mass as first arg
    M_rigid = compute_mass_matrix(
        mass=mass, center_of_gravity=cg,
    )

    n_freq = len(omegas)
    heave_rao_amp = np.zeros(n_freq)
    heave_rao_phase = np.zeros(n_freq)
    heave_a33 = np.zeros(n_freq)
    heave_b33 = np.zeros(n_freq)
    heave_f_exc = np.zeros(n_freq)
    nus = np.zeros(n_freq)
    kL_vals = np.zeros(n_freq)

    for idx, omega in enumerate(omegas):
        nu = omega ** 2 / gravity
        nus[idx] = nu
        kL_vals[idx] = nu * L

        # Radiation solve (heave mode = 2)
        ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
        rad = solve_radiation_3d(
            mesh, nu, modes=(2,), density=density, gravity=gravity,
            operators=ops,
        )

        # Diffraction solve
        diff = solve_diffraction_3d(
            mesh, nu, wave_angle=0.0, amplitude=amplitude,
            density=density, gravity=gravity, operators=ops,
        )

        # Extract heave coefficients (mode index 0 since we only solved mode 2)
        a33 = float(rad.added_mass[0, 0])
        b33 = float(rad.damping[0, 0])
        f_exc = complex(diff.excitation_force[2])

        heave_a33[idx] = a33
        heave_b33[idx] = b33
        heave_f_exc[idx] = abs(f_exc)

        # RAO: ξ = F_exc / [-ω²(M + A) + iωB + C]
        # For heave: M33 = mass, C33 = ρ·g·A_wp (waterplane area restoring)
        m33 = M_rigid[2, 2]
        c33 = C[2, 2]

        impedance = -omega ** 2 * (m33 + a33) + 1j * omega * b33 + c33
        rao = f_exc / impedance

        heave_rao_amp[idx] = abs(rao)
        heave_rao_phase[idx] = math.degrees(math.atan2(rao.imag, rao.real))

    # Long-wave limit validation (kL < 0.3)
    long_wave_mask = kL_vals < 0.3
    if np.any(long_wave_mask):
        long_wave_rao_mean = float(np.mean(heave_rao_amp[long_wave_mask]))
        long_wave_rao_std = float(np.std(heave_rao_amp[long_wave_mask]))
    else:
        long_wave_rao_mean = float(heave_rao_amp[0])
        long_wave_rao_std = 0.0

    # High-frequency limit (omega = 3.0 or closest)
    hf_idx = np.argmin(np.abs(omegas - 3.0))
    high_freq_rao = float(heave_rao_amp[hf_idx])

    return WigleyRAOStudy(
        length=L, beam=B, draft=T, alpha=alpha,
        displaced_volume=volume, mass=mass,
        omegas=omegas, nus=nus, kL_values=kL_vals,
        heave_rao_amplitude=heave_rao_amp,
        heave_rao_phase=heave_rao_phase,
        heave_added_mass=heave_a33,
        heave_damping=heave_b33,
        heave_excitation_magnitude=heave_f_exc,
        long_wave_rao_mean=long_wave_rao_mean,
        long_wave_rao_std=long_wave_rao_std,
        high_freq_rao=high_freq_rao,
        nx=nx, nz=nz,
        panel_count=mesh.panel_count,
    )


def compute_wigley_rao_with_limits(
    nx: int = 15,
    nz: int = 8,
    L: float = WIGLEY3_FULL_L,
    B: float = WIGLEY3_FULL_B,
    T: float = WIGLEY3_FULL_T,
    alpha: float = WIGLEY3_ALPHA,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    amplitude: float = 1.0,
) -> dict[str, Any]:
    """Compute Wigley III RAO and validate limit behaviors.

    Uses full-scale hull (L=100m) for proper limit validation.

    Returns a dict with the full spectrum and limit validation results.
    """
    # Use fewer frequencies for faster computation with large hull
    omega_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    study = compute_wigley_rao_spectrum(
        omega_values=omega_values,
        nx=nx, nz=nz, L=L, B=B, T=T, alpha=alpha,
        density=density, gravity=gravity, amplitude=amplitude,
    )

    # Long-wave limit: heave RAO should be in [0.95, 1.05] for kL < 0.3
    long_wave_pass = 0.95 <= study.long_wave_rao_mean <= 1.05

    # High-frequency limit: heave RAO should be < 0.2 at omega = 3.0
    high_freq_pass = study.high_freq_rao < 0.2

    return {
        "hull": {
            "L": study.length, "B": study.beam, "T": study.draft,
            "alpha": study.alpha, "Cb": study.displaced_volume / (L * B * T),
            "volume": study.displaced_volume, "mass": study.mass,
        },
        "mesh": {
            "nx": study.nx, "nz": study.nz, "panel_count": study.panel_count,
        },
        "spectrum": {
            "omegas": study.omegas.tolist(),
            "kL": study.kL_values.tolist(),
            "heave_rao_amplitude": study.heave_rao_amplitude.tolist(),
            "heave_rao_phase_deg": study.heave_rao_phase.tolist(),
            "heave_added_mass": study.heave_added_mass.tolist(),
            "heave_damping": study.heave_damping.tolist(),
            "heave_excitation_magnitude": study.heave_excitation_magnitude.tolist(),
        },
        "limit_validation": {
            "long_wave_rao_mean": study.long_wave_rao_mean,
            "long_wave_rao_std": study.long_wave_rao_std,
            "long_wave_pass": long_wave_pass,
            "long_wave_criterion": "heave RAO ∈ [0.95, 1.05] for kL < 0.3",
            "high_freq_rao": study.high_freq_rao,
            "high_freq_pass": high_freq_pass,
            "high_freq_criterion": "heave RAO < 0.2 at ω = 3.0 rad/s",
        },
    }
