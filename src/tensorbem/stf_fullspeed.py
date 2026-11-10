"""Salvesen-Tuck-Faltinsen (1970) strip theory with forward speed.

This module implements the classical STF slender-body strip theory for ship
motions in regular waves, extended to non-zero forward speed.  The key
ingredients are:

1. **Encounter frequency.**  For a ship advancing at speed *U* in head seas
   (wave direction β = π), the encounter frequency is

       ω_e = ω − k U cos β = ω + k U   (head seas, cos π = −1)

   where k = ω²/g is the deep-water wavenumber.  All 2-D section coefficients
   are evaluated at ω_e rather than ω.

2. **Speed-corrected section coefficients.**  The STF body condition for the
   radiation potential φ_j at station *x* includes the so-called *m*-terms
   (Ogilvie 1964; Timman & Janssen 1953):

       ∂φ_j / ∂n = i ω_e n_j + m_j

   For heave (j = 3) and pitch (j = 5) in head seas the relevant m-terms are:

       m_3 = 0
       m_5 = −U ∂n_5/∂x

   In the 2-D section problem the x-derivative is replaced by the sectional
   geometry derivative, and the resulting correction to the added mass and
   damping is proportional to U/ω_e.

3. **Cross-coupling and end terms.**  The forward speed introduces:
   - Heave–pitch cross-coupling A_35, A_53 (zero at U = 0 for symmetric sections)
   - End terms from integration by parts of the (U/ω_e) d/dx operators along
     the ship length.  These are evaluated at the bow and stern stations.

4. **Zero-speed degeneration.**  When U → 0, all speed corrections vanish and
   the formulation reduces to the standard zero-speed strip theory (Gerritsma &
   Beukelman 1967).

Conventions
-----------
- Time factor: exp(−i ω t)
- Coordinate system: x forward, z upward, origin at midship waterline
- Wave heading: β = π for head seas (waves propagate in −x direction)
- Section coefficients a_ij, b_ij are per unit length
- The 2-D BEM returns force as a 2-vector [Fx, Fz]; for heave mode the
  diagonal added mass is Fz (index 1), for sway mode it is Fx (index 0).

References
----------
- Salvesen, N., Tuck, E.O. & Faltinsen, O. (1970).  "Ship motions and
  sea loads."  *Trans. SNAME*, 78, 250–287.
- Ogilvie, T.F. (1964).  *1st Symp. on Naval Hydrodynamics*, NAS.
- Timman, R. & Janssen, J. (1953).  *Int. Shipbuilding Progress*.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .free_surface_solver import (
    RadiationResult,
    solve_radiation_indirect,
)
from .geometry import PanelMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionCoefficients:
    """2-D hydrodynamic coefficients at one station.

    Attributes
    ----------
    a33, a35, a53, a55 : float
        Added mass coefficients per unit length [kg/m].
    b33, b35, b53, b55 : float
        Damping coefficients per unit length [kg/(m·s)].
    station : float
        Longitudinal position x of this section [m].
    encounter_omega : float
        Encounter frequency ω_e at which these were evaluated [rad/s].
    """
    station: float
    a33: float
    a35: float
    a53: float
    a55: float
    b33: float
    b35: float
    b53: float
    b55: float
    encounter_omega: float


@dataclass(frozen=True)
class STFCoefficients:
    """Integrated STF strip-theory coefficients for heave-pitch.

    The 6×6 matrix is reduced to the 2×2 heave-pitch block for head seas.
    Indices: 0 = heave (mode 3), 1 = pitch (mode 5).
    """
    A: FloatArray
    B: FloatArray
    C: FloatArray
    F_exc: ComplexArray
    speed_correction_A: FloatArray
    speed_correction_B: FloatArray
    end_terms_A: FloatArray
    end_terms_B: FloatArray


@dataclass(frozen=True)
class STFRaoResult:
    """Result of an STF RAO computation."""
    omega: FloatArray
    heave_rao: ComplexArray
    pitch_rao: ComplexArray
    coefficients: list


# ---------------------------------------------------------------------------
# Encounter frequency
# ---------------------------------------------------------------------------

def encounter_frequency(omega: float, forward_speed: float,
                        heading: float = math.pi,
                        gravity: float = _GRAVITY) -> float:
    """Compute encounter frequency ω_e = ω − k·U·cos(β).

    For head seas (β = π): ω_e = ω + k·U.
    For following seas (β = 0): ω_e = ω − k·U.
    """
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if forward_speed < 0.0:
        raise ValueError("forward_speed must be non-negative")
    k = omega**2 / gravity
    return omega - k * forward_speed * math.cos(heading)


# ---------------------------------------------------------------------------
# Section coefficient computation
# ---------------------------------------------------------------------------

def _circle_mesh(radius: float, depth: float, n_panels: int = 64) -> PanelMesh:
    """Create a circular section mesh at given submergence depth."""
    theta = np.linspace(0, 2 * np.pi, n_panels, endpoint=False)
    x = radius * np.cos(theta)
    z = -depth + radius * np.sin(theta)
    return PanelMesh(vertices=np.column_stack([x, z]))


def compute_section_coefficients_circle(
    radius: float,
    depth: float,
    omega: float,
    forward_speed: float = 0.0,
    heading: float = math.pi,
    n_panels: int = 64,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _GRAVITY,
) -> SectionCoefficients:
    """Compute 2-D section coefficients for a submerged circular section.

    Uses the existing 2-D BEM radiation solver (indirect formulation).
    For a circular section, the heave added mass is the z-component of
    the force due to heave motion (index 1 in the force vector).

    Parameters
    ----------
    radius : float
        Section radius [m].
    depth : float
        Submergence of section centre below still water [m].
    omega : float
        Wave frequency [rad/s].
    forward_speed : float
        Ship forward speed [m/s].
    heading : float
        Wave heading [rad].
    n_panels : int
        Number of panels for the 2-D BEM.
    density : float
        Water density [kg/m³].
    gravity : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    SectionCoefficients
    """
    omega_e = encounter_frequency(omega, forward_speed, heading, gravity)
    if omega_e <= 0.0:
        raise ValueError(
            f"encounter frequency ω_e = {omega_e:.4f} rad/s is non-positive; "
            f"reduce forward speed or increase wave frequency"
        )

    mesh = _circle_mesh(radius, depth, n_panels)
    nu = omega_e**2 / gravity  # deep-water wavenumber at encounter freq

    # Heave radiation: force[1] = z-component = heave added mass
    heave_result = solve_radiation_indirect(mesh, nu, "heave",
                                            density=density)
    # Sway radiation: force[0] = x-component = sway added mass
    sway_result = solve_radiation_indirect(mesh, nu, "sway",
                                           density=density)

    # For a circular section:
    # - a33 = heave added mass = force_z from heave radiation
    # - a55 = pitch added moment about centre ≈ a33 * r² (deep limit)
    # - cross-coupling a35 = 0 by symmetry at zero speed
    a33 = float(heave_result.added_mass[1])  # z-component
    b33 = float(heave_result.damping[1])

    # For a circle rotating about its own centre, the added moment of inertia
    # is related to the translational added mass by the polar moment.
    # In the deep-submergence limit: a55 = a33 * r²
    a55 = a33 * radius**2
    b55 = b33 * radius**2

    # Cross-coupling is zero for a symmetric section at zero speed
    a35 = 0.0
    a53 = 0.0
    b35 = 0.0
    b53 = 0.0

    return SectionCoefficients(
        station=0.0,
        a33=a33, a35=a35, a53=a53, a55=a55,
        b33=b33, b35=b35, b53=b53, b55=b55,
        encounter_omega=omega_e,
    )


# ---------------------------------------------------------------------------
# STF speed corrections
# ---------------------------------------------------------------------------

def stf_speed_correction_terms(
    section_coeffs: Sequence[SectionCoefficients],
    stations: FloatArray,
    omega: float,
    forward_speed: float,
    gravity: float = _GRAVITY,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute STF forward speed correction terms.

    The speed corrections to the integrated added mass and damping arise from
    the m-terms in the body boundary condition.  For heave-pitch in head seas:

    Correction to A_33:  ΔA_33 = (U/ω_e) · [a_33(bow) − a_33(stern)]
    Correction to A_35:  ΔA_35 = (U/ω_e) · ∫ a_33(x) dx
    Correction to A_55:  ΔA_55 = (U/ω_e)² · ∫ a_33(x) dx

    The end terms come from integration by parts of the d/dx operators.

    Returns
    -------
    delta_A, delta_B, end_A, end_B : ndarray (2, 2) each
    """
    n = len(stations)
    if n < 2:
        return (np.zeros((2, 2)), np.zeros((2, 2)),
                np.zeros((2, 2)), np.zeros((2, 2)))

    omega_e = encounter_frequency(omega, forward_speed, math.pi, gravity)
    speed_ratio = forward_speed / omega_e if omega_e > 0 else 0.0

    a33 = np.array([c.a33 for c in section_coeffs])
    b33 = np.array([c.b33 for c in section_coeffs])

    delta_A = np.zeros((2, 2))
    delta_B = np.zeros((2, 2))
    end_A = np.zeros((2, 2))
    end_B = np.zeros((2, 2))

    # --- End terms for A33: (U/ω_e) · [a33(bow) − a33(stern)] ---
    end_A[0, 0] = speed_ratio * (a33[-1] - a33[0])
    end_B[0, 0] = speed_ratio * (b33[-1] - b33[0])

    # --- A35 cross-coupling from speed: (U/ω_e) · ∫a33 dx ---
    dx = np.diff(stations)
    a33_mid = 0.5 * (a33[:-1] + a33[1:])
    b33_mid = 0.5 * (b33[:-1] + b33[1:])
    integral_a33 = float(np.sum(a33_mid * dx))
    integral_b33 = float(np.sum(b33_mid * dx))

    delta_A[0, 1] = speed_ratio * integral_a33
    delta_A[1, 0] = delta_A[0, 1]
    delta_B[0, 1] = speed_ratio * integral_b33
    delta_B[1, 0] = delta_B[0, 1]

    # --- A55 correction: (U/ω_e)² · ∫ a33(x) dx ---
    delta_A[1, 1] = speed_ratio**2 * integral_a33
    delta_B[1, 1] = speed_ratio**2 * integral_b33

    return delta_A, delta_B, end_A, end_B


# ---------------------------------------------------------------------------
# Strip theory integration
# ---------------------------------------------------------------------------

def integrate_strip_coefficients(
    section_coeffs: Sequence[SectionCoefficients],
    stations: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Integrate section coefficients along the ship length (trapezoidal rule)."""
    n = len(stations)
    A_int = np.zeros((2, 2))
    B_int = np.zeros((2, 2))

    for i in range(n - 1):
        dx = stations[i + 1] - stations[i]
        c0 = section_coeffs[i]
        c1 = section_coeffs[i + 1]
        A_int[0, 0] += 0.5 * (c0.a33 + c1.a33) * dx
        A_int[0, 1] += 0.5 * (c0.a35 + c1.a35) * dx
        A_int[1, 0] += 0.5 * (c0.a53 + c1.a53) * dx
        A_int[1, 1] += 0.5 * (c0.a55 + c1.a55) * dx
        B_int[0, 0] += 0.5 * (c0.b33 + c1.b33) * dx
        B_int[0, 1] += 0.5 * (c0.b35 + c1.b35) * dx
        B_int[1, 0] += 0.5 * (c0.b53 + c1.b53) * dx
        B_int[1, 1] += 0.5 * (c0.b55 + c1.b55) * dx

    return A_int, B_int


# ---------------------------------------------------------------------------
# Full STF assembly
# ---------------------------------------------------------------------------

def assemble_stf_coefficients(
    section_coeffs: Sequence[SectionCoefficients],
    stations: FloatArray,
    omega: float,
    forward_speed: float,
    hydrostatic_C: FloatArray | None = None,
    excitation_forces: ComplexArray | None = None,
    gravity: float = _GRAVITY,
) -> STFCoefficients:
    """Assemble the full STF heave-pitch equation of motion.

    The equation is: [−ω²(M + A) + iω B + C] ξ = F_exc
    """
    A_bare, B_bare = integrate_strip_coefficients(section_coeffs, stations)

    delta_A, delta_B, end_A, end_B = stf_speed_correction_terms(
        section_coeffs, stations, omega, forward_speed, gravity
    )

    A_total = A_bare + delta_A + end_A
    B_total = B_bare + delta_B + end_B

    C = np.zeros((2, 2)) if hydrostatic_C is None else np.asarray(hydrostatic_C)
    F_exc = np.zeros(2, dtype=complex)
    if excitation_forces is not None:
        F_exc = np.asarray(excitation_forces, dtype=complex)

    return STFCoefficients(
        A=A_total, B=B_total, C=C, F_exc=F_exc,
        speed_correction_A=delta_A, speed_correction_B=delta_B,
        end_terms_A=end_A, end_terms_B=end_B,
    )


def compute_stf_rao(
    stf_coeffs: STFCoefficients,
    omega: float,
    mass_matrix: FloatArray | None = None,
) -> tuple[complex, complex]:
    """Compute heave and pitch RAO from assembled STF coefficients.

    Solves: [−ω²(M + A) + iω B + C] ξ = F_exc
    """
    M = np.zeros((2, 2)) if mass_matrix is None else np.asarray(mass_matrix)
    A = stf_coeffs.A
    B = stf_coeffs.B
    C = stf_coeffs.C
    F = stf_coeffs.F_exc

    Z = -(omega**2) * (M + A) + 1j * omega * B + C

    try:
        xi = np.linalg.solve(Z, F)
    except np.linalg.LinAlgError:
        return complex(0.0), complex(0.0)

    return complex(xi[0]), complex(xi[1])


# ---------------------------------------------------------------------------
# Zero-speed degeneration check
# ---------------------------------------------------------------------------

def check_zero_speed_degeneration(
    section_coeffs: Sequence[SectionCoefficients],
    stations: FloatArray,
    omega: float,
    tolerance: float = 1e-10,
) -> dict:
    """Verify that STF speed corrections vanish when U → 0.

    This is a fail-closed check: if the end terms or speed corrections are
    non-negligible at U = 0, the implementation has a bug.
    """
    delta_A, delta_B, end_A, end_B = stf_speed_correction_terms(
        section_coeffs, stations, omega, forward_speed=0.0
    )

    max_correction = max(np.max(np.abs(delta_A)), np.max(np.abs(delta_B)))
    end_norm = max(np.max(np.abs(end_A)), np.max(np.abs(end_B)))
    passed = max_correction < tolerance and end_norm < tolerance

    return {
        "passed": bool(passed),
        "max_correction": float(max_correction),
        "end_terms_norm": float(end_norm),
        "speed_correction_norm": float(max_correction),
        "tolerance": tolerance,
    }
