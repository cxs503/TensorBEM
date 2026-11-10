"""Froude-Krylov excitation force closed-form benchmark.

The Froude-Krylov (FK) force is the wave excitation force computed by
integrating the undisturbed incident wave pressure over the wetted hull
surface, ignoring diffraction (scattering) effects.

For a box barge of length L, beam B, draft T in head seas (wave angle = 0),
the heave FK force has a closed-form solution. The incident wave pressure is:

    p = ρ·g·A·exp(kz)·cos(kx - ωt)

For heave, only the bottom surface contributes (sides have horizontal normals):

    F_FK = ρ·g·A·B·∫_{-L/2}^{L/2} exp(-kT)·cos(kx) dx
         = ρ·g·A·B·L·exp(-kT)·sinc(kL/2)

where sinc(x) = sin(x)/x.

This module computes the FK closed-form and compares it with the full
BEM diffraction solution at long wavelengths (where diffraction is small
and FK dominates).
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
)
from tensorbem.geometry3d import box_hull_mesh

FloatArray = NDArray[np.float64]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


@dataclass(frozen=True)
class FKExcitationResult:
    """Result of Froude-Krylov vs BEM diffraction comparison."""

    # Box barge geometry
    length: float
    beam: float
    draft: float
    # Wave parameters
    omega: float
    wavenumber: float
    amplitude: float
    # FK closed-form
    fk_heave_force: complex
    fk_heave_magnitude: float
    # BEM diffraction
    bem_heave_force: complex
    bem_heave_magnitude: float
    # Comparison
    magnitude_rel_error: float
    # FK validity indicator
    kl: float  # kL product


def fk_heave_force_box_barge(
    length: float,
    beam: float,
    draft: float,
    omega: float,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> complex:
    """Compute the Froude-Krylov heave excitation force for a box barge.

    For a rectangular barge in head seas, the FK heave force is:

        F_FK = ρ·g·A·B·L · exp(-kT) · sinc(kL/2)

    where sinc(x) = sin(x)/x.

    Parameters
    ----------
    length : float
        Barge length (m).
    beam : float
        Barge beam (m).
    draft : float
        Barge draft (m).
    omega : float
        Wave circular frequency (rad/s).
    amplitude : float
        Wave amplitude (m).
    density : float
        Fluid density (kg/m³).
    gravity : float
        Gravitational acceleration (m/s²).

    Returns
    -------
    complex
        FK heave force (complex, with phase relative to wave elevation at origin).
    """
    if length <= 0 or beam <= 0 or draft <= 0:
        raise ValueError("dimensions must be positive")
    if omega <= 0:
        raise ValueError("omega must be positive")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")

    k = omega ** 2 / gravity  # Deep water dispersion
    kl = k * length

    # sinc(kL/2) = sin(kL/2) / (kL/2)
    sinc_arg = kl / 2.0
    if abs(sinc_arg) < 1e-12:
        sinc_val = 1.0
    else:
        sinc_val = math.sin(sinc_arg) / sinc_arg

    # Vertical decay at bottom: exp(-kT)
    vertical_factor = math.exp(-k * draft)

    # FK heave force magnitude
    fk_magnitude = density * gravity * amplitude * beam * length * sinc_val * vertical_factor

    # Phase: at long waves (kL → 0), FK force is in phase with wave elevation
    phase = 0.0  # Bottom pressure is in phase with wave elevation at x=0
    return complex(fk_magnitude * math.cos(phase), fk_magnitude * math.sin(phase))


def compute_fk_heave_force_box_barge(
    length: float = 100.0,
    beam: float = 20.0,
    draft: float = 8.0,
    omega: float = 0.5,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> FKExcitationResult:
    """Compute FK closed-form heave force for a box barge."""
    k = omega ** 2 / gravity
    fk_force = fk_heave_force_box_barge(length, beam, draft, omega, amplitude, density, gravity)

    return FKExcitationResult(
        length=length, beam=beam, draft=draft,
        omega=omega, wavenumber=k, amplitude=amplitude,
        fk_heave_force=fk_force, fk_heave_magnitude=abs(fk_force),
        bem_heave_force=complex(0, 0), bem_heave_magnitude=0.0,
        magnitude_rel_error=0.0, kl=k * length,
    )


def compute_fk_vs_bem_comparison(
    length: float = 50.0,
    beam: float = 10.0,
    draft: float = 4.0,
    omega_values: Sequence[float] = (0.2, 0.3, 0.5, 0.8, 1.0),
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_length: int = 10,
    n_beam: int = 5,
    n_draft: int = 3,
) -> list[dict[str, Any]]:
    """Compare FK closed-form with BEM diffraction across frequencies."""
    mesh = box_hull_mesh(
        length=length, beam=beam, draft=draft,
        n_length=n_length, n_beam=n_beam, n_draft=n_draft,
    )
    results: list[dict[str, Any]] = []

    for omega in omega_values:
        omega = float(omega)
        k = omega ** 2 / gravity
        nu = k

        # FK closed-form
        fk_force = fk_heave_force_box_barge(length, beam, draft, omega, amplitude, density, gravity)

        # BEM diffraction
        ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
        diff = solve_diffraction_3d(
            mesh, nu, wave_angle=0.0, amplitude=amplitude,
            density=density, gravity=gravity, operators=ops,
        )
        bem_force = complex(diff.excitation_force[2])  # Heave (index 2)

        fk_mag = abs(fk_force)
        bem_mag = abs(bem_force)
        rel_error = abs(bem_mag - fk_mag) / max(fk_mag, 1e-12)

        results.append({
            "omega": omega, "k": k, "kL": k * length,
            "fk_magnitude": fk_mag,
            "fk_phase_deg": math.degrees(math.atan2(fk_force.imag, fk_force.real)),
            "bem_magnitude": bem_mag,
            "bem_phase_deg": math.degrees(math.atan2(bem_force.imag, bem_force.real)),
            "magnitude_rel_error": rel_error,
            "long_wave_limit": k * length < 0.3,
        })

    return results
