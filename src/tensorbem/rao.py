"""Frequency-domain RAO (Response Amplitude Operator) solver.

The RAO describes the linear response of a floating body to regular waves.
The equation of motion in frequency domain is:

    [-ω²(M + A(ω)) + iω B(ω) + C] ξ(ω) = F_exc(ω)

where:
    M     = rigid-body mass matrix (6×6)
    A(ω)  = frequency-dependent added mass matrix (6×6)
    B(ω)  = frequency-dependent damping matrix (6×6)
    C     = hydrostatic restoring matrix (6×6)
    F_exc = wave excitation force vector (6×1)
    ξ(ω)  = complex RAO vector (6×1)

The transfer function H(ω) is:
    H(ω) = [-ω²(M + A(ω)) + iω B(ω) + C]^(-1)

And the RAO is:
    ξ(ω) = H(ω) F_exc(ω)

For each degree of freedom, |ξ_i(ω)| gives the amplitude of motion per unit
wave amplitude at frequency ω.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .free_surface_solver3d import (
    Diffraction3DResult,
    Radiation3DResult,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from .geometry3d import WettedHullMesh
from .hydrostatic import (
    HydrostaticProperties,
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0


@dataclass(frozen=True)
class RAOResult:
    """Result of RAO computation at a single frequency.

    Attributes
    ----------
    omega :
        Wave circular frequency (rad/s).
    nu :
        Wave number (rad/m).
    rao :
        Complex RAO vector (6,) for surge, sway, heave, roll, pitch, yaw.
        Units: m/m for translation, rad/m for rotation.
    rao_amplitude :
        Absolute value of RAO (|ξ|).
    excitation_force :
        Complex excitation force vector (6,).
    added_mass :
        Added mass matrix A(ω) at this frequency (6×6).
    damping :
        Damping matrix B(ω) at this frequency (6×6).
    impedance_residual :
        Residual of the impedance equation (should be ~0 if converged).
    """

    omega: float
    nu: float
    rao: ComplexArray
    rao_amplitude: FloatArray
    excitation_force: ComplexArray
    added_mass: FloatArray
    damping: FloatArray
    impedance_residual: float


@dataclass(frozen=True)
class RAOSpectrum:
    """RAO spectrum over a range of frequencies.

    Attributes
    ----------
    omegas :
        Array of wave circular frequencies (rad/s).
    nus :
        Array of wave numbers (rad/m).
    rao_amplitudes :
        Array of RAO amplitudes, shape (n_freq, 6).
    excitation_forces :
        Array of excitation forces, shape (n_freq, 6).
    added_masses :
        Array of added mass matrices, shape (n_freq, 6, 6).
    dampings :
        Array of damping matrices, shape (n_freq, 6, 6).
    """

    omegas: FloatArray
    nus: FloatArray
    rao_amplitudes: FloatArray
    excitation_forces: ComplexArray
    added_masses: FloatArray
    dampings: FloatArray


def _omega_to_nu(omega: float, gravity: float) -> float:
    """Convert angular frequency to wavenumber (deep water dispersion)."""
    return omega * omega / gravity


def compute_rao_at_frequency(
    hull: WettedHullMesh,
    omega: float,
    wave_angle: float,
    mass: float,
    center_of_gravity: FloatArray,
    hydrostatic_props: HydrostaticProperties | None = None,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    inertia: FloatArray | None = None,
    extra_damping: FloatArray | None = None,
    chief_points: int | str | None = "auto",
) -> RAOResult:
    """Compute RAO at a single frequency.

    Parameters
    ----------
    hull :
        Wetted hull mesh.
    omega :
        Wave circular frequency (rad/s).
    wave_angle :
        Wave direction (radians). 0 = head seas (propagating in +x direction).
    mass :
        Body mass (kg).
    center_of_gravity :
        Center of gravity coordinates (m).
    hydrostatic_props :
        Pre-computed hydrostatic properties. If None, computed from hull.
    amplitude :
        Wave amplitude (m). Default: 1.0.
    density :
        Fluid density (kg/m³).
    gravity :
        Gravitational acceleration (m/s²).
    inertia :
        Inertia tensor about the origin (3×3). If None, approximated.
    extra_damping :
        Additional damping matrix (6×6) to add (e.g., viscous damping).
    chief_points :
        Number of CHIEF points for irregular frequency suppression.

    Returns
    -------
    result :
        RAO result at this frequency.
    """
    omega = float(omega)
    nu = _omega_to_nu(omega, gravity)

    # Compute hydrostatic properties if not provided
    if hydrostatic_props is None:
        hydrostatic_props = compute_hydrostatic_restoring_matrix(
            hull, mass, center_of_gravity, density, gravity
        )

    # Get restoring matrix
    c = hydrostatic_props.restoring_matrix

    # Compute mass matrix
    m = compute_mass_matrix(mass, center_of_gravity, inertia)

    # Solve radiation problem to get added mass and damping
    rad_result: Radiation3DResult = solve_radiation_3d(
        hull, nu, density=density, gravity=gravity, chief_points=chief_points
    )

    # Solve diffraction problem to get excitation force
    diff_result: Diffraction3DResult = solve_diffraction_3d(
        hull, nu, amplitude=amplitude, wave_angle=wave_angle,
        density=density, gravity=gravity, chief_points=chief_points
    )

    # Extract added mass and damping matrices
    added_mass = rad_result.added_mass
    damping = rad_result.damping

    # Add extra damping if provided
    if extra_damping is not None:
        damping = damping + np.asarray(extra_damping)

    # Build impedance matrix
    # Z(ω) = -ω²(M + A(ω)) + iω B(ω) + C
    omega_sq = omega * omega
    z = -omega_sq * (m + added_mass) + 1j * omega * damping + c

    # Excitation force
    f_exc = diff_result.excitation_force

    # Solve for RAO
    # Z ξ = F_exc  =>  ξ = Z^(-1) F_exc
    rao = np.linalg.solve(z, f_exc)

    # Compute residual
    residual_vec = z @ rao - f_exc
    impedance_residual = float(np.linalg.norm(residual_vec))

    return RAOResult(
        omega=omega,
        nu=nu,
        rao=rao,
        rao_amplitude=np.abs(rao),
        excitation_force=f_exc,
        added_mass=added_mass,
        damping=damping,
        impedance_residual=impedance_residual,
    )


def compute_rao_spectrum(
    hull: WettedHullMesh,
    omegas: Sequence[float],
    wave_angle: float,
    mass: float,
    center_of_gravity: FloatArray,
    hydrostatic_props: HydrostaticProperties | None = None,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    inertia: FloatArray | None = None,
    extra_damping: FloatArray | None = None,
    chief_points: int | str | None = "auto",
) -> RAOSpectrum:
    """Compute RAO spectrum over a range of frequencies.

    Parameters
    ----------
    hull :
        Wetted hull mesh.
    omegas :
        Sequence of wave circular frequencies (rad/s).
    wave_angle :
        Wave direction (radians).
    mass :
        Body mass (kg).
    center_of_gravity :
        Center of gravity coordinates (m).
    hydrostatic_props :
        Pre-computed hydrostatic properties. If None, computed from hull.
    amplitude :
        Wave amplitude (m).
    density :
        Fluid density (kg/m³).
    gravity :
        Gravitational acceleration (m/s²).
    inertia :
        Inertia tensor about the origin (3×3).
    extra_damping :
        Additional damping matrix (6×6).
    chief_points :
        Number of CHIEF points.

    Returns
    -------
    spectrum :
        RAO spectrum over the frequency range.
    """
    omegas = np.asarray(omegas, dtype=float)
    n_freq = len(omegas)
    nus = np.array([_omega_to_nu(w, gravity) for w in omegas])

    rao_amplitudes = np.zeros((n_freq, 6), dtype=float)
    excitation_forces = np.zeros((n_freq, 6), dtype=complex)
    added_masses = np.zeros((n_freq, 6, 6), dtype=float)
    dampings = np.zeros((n_freq, 6, 6), dtype=float)

    # Compute hydrostatic properties once
    if hydrostatic_props is None:
        hydrostatic_props = compute_hydrostatic_restoring_matrix(
            hull, mass, center_of_gravity, density, gravity
        )

    for i, omega in enumerate(omegas):
        rao_result = compute_rao_at_frequency(
            hull=hull,
            omega=omega,
            wave_angle=wave_angle,
            mass=mass,
            center_of_gravity=center_of_gravity,
            hydrostatic_props=hydrostatic_props,
            amplitude=amplitude,
            density=density,
            gravity=gravity,
            inertia=inertia,
            extra_damping=extra_damping,
            chief_points=chief_points,
        )

        rao_amplitudes[i] = rao_result.rao_amplitude
        excitation_forces[i] = rao_result.excitation_force
        added_masses[i] = rao_result.added_mass
        dampings[i] = rao_result.damping

    return RAOSpectrum(
        omegas=omegas,
        nus=nus,
        rao_amplitudes=rao_amplitudes,
        excitation_forces=excitation_forces,
        added_masses=added_masses,
        dampings=dampings,
    )


def estimate_natural_frequency(
    mass: float,
    added_mass_inf: FloatArray,
    restoring_matrix: FloatArray,
    dof: int,
) -> float:
    """Estimate natural frequency for a single DOF.

    The natural frequency is:
        ω_n = sqrt(C[i,i] / (M[i,i] + A_∞[i,i]))

    Parameters
    ----------
    mass :
        Body mass (kg).
    added_mass_inf :
        Added mass matrix at infinite frequency (6×6).
    restoring_matrix :
        Hydrostatic restoring matrix (6×6).
    dof :
        Degree of freedom index (0-5).

    Returns
    -------
    omega_n :
        Natural circular frequency (rad/s).
    """
    m_eff = mass + added_mass_inf[dof, dof]
    c = restoring_matrix[dof, dof]

    if m_eff <= 0 or c <= 0:
        return 0.0

    return math.sqrt(c / m_eff)
