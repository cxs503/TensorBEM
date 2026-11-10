"""2-D Haskind excitation force for strip theory.

The Haskind-Newman relation (Haskind 1953; Newman 1977) expresses the
first-order wave excitation force on a body in terms of the radiation
potential alone, without solving a separate diffraction problem:

    F_j = −i ω ρ ∮_Γ (φ_I n_j − φ_j ∂φ_I/∂n) dl

This module provides:

1. **haskind_strip_excitation**: Compute the 2-D excitation force per unit
   length at a strip-theory station using the Haskind relation.  Reuses the
   existing :func:`tensorbem.free_surface_solver.haskind_excitation_force`
   implementation and its phase/sign convention.

2. **direct_diffraction_excitation**: Compute the excitation force by directly
   solving the diffraction problem (independent path).

3. **haskind_cross_validation**: Compare the two paths and report the
   relative discrepancy.  Agreement < 2% is the acceptance criterion (C3).

Conventions
-----------
- Time factor: exp(−i ω t)
- The existing ``haskind_excitation_force`` returns a scalar complex force
  for a given radiation mode (heave or sway).
- The diffraction solver returns excitation_force as a 2-vector [Fx, Fz].
- For cross-validation, we compare the component corresponding to the
  radiation mode: Fz for heave, Fx for sway.

References
----------
- Haskind, M.D. (1953).  *Izvestiya Akad. Nauk SSSR*.
- Newman, J.N. (1977).  *Marine Hydrodynamics*.  MIT Press, §4.4.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .free_surface_solver import (
    DiffractionResult,
    RadiationResult,
    haskind_excitation_force,
    solve_diffraction_indirect,
    solve_radiation_indirect,
)
from .geometry import PanelMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0


@dataclass(frozen=True)
class HaskindValidation:
    """Result of Haskind vs direct-diffraction cross-validation.

    Attributes
    ----------
    haskind_force : complex
        Excitation force from Haskind relation [N/m per unit wave amplitude].
    direct_force : complex
        Excitation force from direct diffraction solve [N/m].
    relative_magnitude_error : float
        |F_H − F_D| / |F_D| — the relative magnitude discrepancy.
    phase_error_degrees : float
        Phase difference in degrees.
    passed : bool
        True if relative_magnitude_error < threshold.
    """
    haskind_force: complex
    direct_force: complex
    relative_magnitude_error: float
    phase_error_degrees: float
    passed: bool


def haskind_strip_excitation(
    mesh: PanelMesh,
    radiation: RadiationResult,
    nu: float,
    amplitude: float = 1.0,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _GRAVITY,
    direction: int = 1,
) -> complex:
    """Compute 2-D excitation force via Haskind relation.

    This is a thin wrapper around
    :func:`tensorbem.free_surface_solver.haskind_excitation_force` that
    preserves the same phase/sign convention and returns the complex
    excitation force per unit length.
    """
    return haskind_excitation_force(
        mesh, radiation, nu,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
        direction=direction,
    )


def direct_diffraction_excitation(
    mesh: PanelMesh,
    nu: float,
    amplitude: float = 1.0,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _GRAVITY,
    direction: int = 1,
    panel_nodes: int = 24,
    mode: str = "heave",
) -> complex:
    """Compute 2-D excitation force by direct diffraction solve.

    Solves the full diffraction problem and extracts the force component
    corresponding to the specified mode.

    Parameters
    ----------
    mesh : PanelMesh
        Section body contour.
    nu : float
        Wavenumber.
    amplitude : float
        Wave amplitude [m].
    mode : str
        'heave' (z-component, index 1) or 'sway' (x-component, index 0).

    Returns
    -------
    complex
        Complex excitation force per unit length [N/m].
    """
    result = solve_diffraction_indirect(
        mesh, nu,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
        direction=direction,
        panel_nodes=panel_nodes,
    )
    # excitation_force is a 2-vector [Fx, Fz]
    idx = 1 if mode == "heave" else 0
    return complex(result.excitation_force[idx])


def haskind_cross_validation(
    mesh: PanelMesh,
    nu: float,
    mode: str = "heave",
    amplitude: float = 1.0,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _GRAVITY,
    direction: int = 1,
    panel_nodes: int = 24,
    threshold: float = 0.02,
) -> HaskindValidation:
    """Cross-validate Haskind relation against direct diffraction.

    Computes the excitation force by two independent paths:
    1. Haskind-Newman relation (radiation potential only)
    2. Direct diffraction solve (incident + scattered potential)

    The two must agree to within ``threshold`` (default 2%).
    """
    # Path 1: Haskind (radiation → excitation)
    radiation = solve_radiation_indirect(
        mesh, nu, mode, density=density, panel_nodes=panel_nodes
    )
    f_haskind = haskind_strip_excitation(
        mesh, radiation, nu,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
        direction=direction,
    )

    # Path 2: Direct diffraction
    f_direct = direct_diffraction_excitation(
        mesh, nu,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
        direction=direction,
        panel_nodes=panel_nodes,
        mode=mode,
    )

    # Compare
    mag_direct = abs(f_direct)
    if mag_direct < 1e-30:
        rel_err = abs(f_haskind - f_direct)
    else:
        rel_err = abs(f_haskind - f_direct) / mag_direct

    phase_haskind = math.atan2(f_haskind.imag, f_haskind.real)
    phase_direct = math.atan2(f_direct.imag, f_direct.real)
    phase_err_deg = math.degrees(phase_haskind - phase_direct)
    phase_err_deg = ((phase_err_deg + 180) % 360) - 180

    return HaskindValidation(
        haskind_force=complex(f_haskind),
        direct_force=complex(f_direct),
        relative_magnitude_error=float(rel_err),
        phase_error_degrees=float(phase_err_deg),
        passed=rel_err < threshold,
    )


def haskind_sweep(
    mesh: PanelMesh,
    ka_values: FloatArray,
    radius: float,
    mode: str = "heave",
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _GRAVITY,
    direction: int = 1,
    panel_nodes: int = 24,
    threshold: float = 0.02,
) -> list[HaskindValidation]:
    """Run Haskind cross-validation over a range of ka values."""
    results = []
    for ka in ka_values:
        nu = ka / radius
        val = haskind_cross_validation(
            mesh, nu, mode=mode,
            density=density, gravity=gravity,
            direction=direction, panel_nodes=panel_nodes,
            threshold=threshold,
        )
        results.append(val)
    return results
