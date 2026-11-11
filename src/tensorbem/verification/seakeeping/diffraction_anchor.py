"""Hulme (1982) floating-hemisphere diffraction excitation anchor.

The existing ``hemisphere_anchor`` module validates *radiation* (added mass
and damping) against Hulme's semi-analytical spherical wave-free multipole
solution.  This module closes the loop by validating the *diffraction*
excitation force — the other half of the seakeeping boundary-value problem.

For a floating hemisphere of radius ``a`` in head seas, Hulme (1982) provides
the exact complex excitation forces in surge (``F_x``) and heave (``F_z``)
via the Kochin function evaluated at the origin.  The BEM diffraction solve
must reproduce these forces within the discretisation error of the panel mesh.

This module provides:

1. **Single-point diffraction anchor** — compute BEM vs Hulme at one ``Ka``.
2. **Frequency sweep** — compare across a range of ``Ka`` values.
3. **Mesh convergence study** — verify that diffraction error decreases with
   panel refinement, and report the observed order of convergence.

References
----------
- Hulme, A. (1982). "The wave forces acting on a floating hemisphere
  in periodic motion." J. Fluid Mech. 121, 443–463.
- Newman, J.N. (1977). "Marine Hydrodynamics." MIT Press, §6.5.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

FloatArray = NDArray[np.float64]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


@dataclass(frozen=True)
class DiffractionAnchorResult:
    """Result of one frequency-point diffraction comparison."""

    ka: float
    subdivisions: int
    panel_count: int
    # BEM direct diffraction
    bem_fx: complex
    bem_fz: complex
    # Hulme reference
    hulme_fx: complex
    hulme_fz: complex
    # Relative errors (magnitude)
    surge_rel_error: float
    heave_rel_error: float
    # Haskind reciprocity check
    haskind_fx: complex
    haskind_fz: complex
    haskind_surge_rel_error: float
    haskind_heave_rel_error: float
    # Condition number
    condition_number: float


@dataclass(frozen=True)
class DiffractionSweepResult:
    """Result of a frequency-sweep diffraction study."""

    ka_values: tuple[float, ...]
    subdivisions: int
    panel_count: int
    # Per-frequency results
    levels: tuple[DiffractionAnchorResult, ...]
    # Summary statistics
    max_surge_error: float
    max_heave_error: float
    mean_surge_error: float
    mean_heave_error: float


@dataclass(frozen=True)
class DiffractionConvergenceResult:
    """Result of a mesh-convergence diffraction study."""

    ka: float
    # Per-refinement-level results
    levels: tuple[DiffractionAnchorResult, ...]
    panel_counts: tuple[int, ...]
    surge_errors: tuple[float, ...]
    heave_errors: tuple[float, ...]
    # Observed order of convergence (None if < 2 levels)
    observed_order_surge: float | None
    observed_order_heave: float | None
    # Convergence criterion: error at finest level
    finest_surge_error: float
    finest_heave_error: float


def compute_diffraction_anchor(
    ka: float = 1.0,
    subdivisions: int = 2,
    radius: float = 1.0,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> DiffractionAnchorResult:
    """Compute the diffraction excitation anchor at one frequency.

    Parameters
    ----------
    ka : float
        Non-dimensional wavenumber (k·a).
    subdivisions : int
        Mesh refinement level (panel count = 16 × 4^subdivisions for the
        full sphere; the wetted hemisphere has half that).
    radius : float
        Hemisphere radius (m).
    amplitude : float
        Wave amplitude (m).
    density : float
        Fluid density (kg/m³).
    gravity : float
        Gravitational acceleration (m/s²).

    Returns
    -------
    DiffractionAnchorResult
    """
    if ka <= 0:
        raise ValueError("ka must be positive")
    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")
    if radius <= 0:
        raise ValueError("radius must be positive")

    mesh = hemisphere_hull_mesh(subdivisions, radius=radius)
    ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")

    # Direct BEM diffraction
    diff = solve_diffraction_3d(
        mesh, ka, wave_angle=0.0, amplitude=amplitude,
        density=density, gravity=gravity, operators=ops,
    )
    bem_fx = complex(diff.excitation_force[0])
    bem_fz = complex(diff.excitation_force[2])

    # Hulme reference
    hulme = hulme_hemisphere_radiation(
        ka, radius=radius, amplitude=amplitude, density=density, gravity=gravity,
    )
    hulme_fx = complex(hulme.excitation_force_x)
    hulme_fz = complex(hulme.excitation_force_z)

    # Relative errors (magnitude)
    surge_rel_error = abs(bem_fx - hulme_fx) / max(abs(hulme_fx), 1e-12)
    heave_rel_error = abs(bem_fz - hulme_fz) / max(abs(hulme_fz), 1e-12)

    # Haskind reciprocity check
    rad = solve_radiation_3d(mesh, ka, operators=ops)
    hask = haskind_excitation_force_3d(mesh, rad, wave_angle=0.0, amplitude=amplitude)
    kochin_f = hask["kochin_haskind"]
    haskind_fx = complex(kochin_f[0])
    haskind_fz = complex(kochin_f[2])

    haskind_surge_rel_error = abs(haskind_fx - hulme_fx) / max(abs(hulme_fx), 1e-12)
    haskind_heave_rel_error = abs(haskind_fz - hulme_fz) / max(abs(hulme_fz), 1e-12)

    return DiffractionAnchorResult(
        ka=ka,
        subdivisions=subdivisions,
        panel_count=mesh.panel_count,
        bem_fx=bem_fx,
        bem_fz=bem_fz,
        hulme_fx=hulme_fx,
        hulme_fz=hulme_fz,
        surge_rel_error=surge_rel_error,
        heave_rel_error=heave_rel_error,
        haskind_fx=haskind_fx,
        haskind_fz=haskind_fz,
        haskind_surge_rel_error=haskind_surge_rel_error,
        haskind_heave_rel_error=haskind_heave_rel_error,
        condition_number=diff.condition_number,
    )


def compute_diffraction_sweep(
    ka_values: Sequence[float] = (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
    subdivisions: int = 2,
    radius: float = 1.0,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> DiffractionSweepResult:
    """Compute diffraction comparison across a frequency sweep.

    Parameters
    ----------
    ka_values : sequence
        Non-dimensional wavenumbers to evaluate.
    subdivisions : int
        Mesh refinement level.
    radius : float
        Hemisphere radius (m).
    amplitude : float
        Wave amplitude (m).
    density : float
        Fluid density (kg/m³).
    gravity : float
        Gravitational acceleration (m/s²).

    Returns
    -------
    DiffractionSweepResult
    """
    levels: list[DiffractionAnchorResult] = []
    for ka in ka_values:
        result = compute_diffraction_anchor(
            ka=float(ka), subdivisions=subdivisions,
            radius=radius, amplitude=amplitude,
            density=density, gravity=gravity,
        )
        levels.append(result)

    surge_errors = [r.surge_rel_error for r in levels]
    heave_errors = [r.heave_rel_error for r in levels]

    return DiffractionSweepResult(
        ka_values=tuple(float(k) for k in ka_values),
        subdivisions=subdivisions,
        panel_count=levels[0].panel_count if levels else 0,
        levels=tuple(levels),
        max_surge_error=max(surge_errors) if surge_errors else 0.0,
        max_heave_error=max(heave_errors) if heave_errors else 0.0,
        mean_surge_error=float(np.mean(surge_errors)) if surge_errors else 0.0,
        mean_heave_error=float(np.mean(heave_errors)) if heave_errors else 0.0,
    )


def _observed_order(
    h_values: Sequence[float], errors: Sequence[float],
) -> float | None:
    """Estimate observed order of convergence from two refinement levels."""
    if len(h_values) < 2 or len(errors) < 2:
        return None
    h0, h1 = h_values[-2], h_values[-1]
    e0, e1 = errors[-2], errors[-1]
    if e0 <= 0 or e1 <= 0 or h0 <= h1:
        return None
    return float(math.log(e0 / e1) / math.log(h0 / h1))


def compute_diffraction_convergence(
    ka: float = 1.0,
    subdivisions: Sequence[int] = (1, 2, 3),
    radius: float = 1.0,
    amplitude: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> DiffractionConvergenceResult:
    """Compute diffraction error vs mesh refinement.

    Parameters
    ----------
    ka : float
        Non-dimensional wavenumber.
    subdivisions : sequence
        Mesh refinement levels to evaluate.
    radius : float
        Hemisphere radius (m).
    amplitude : float
        Wave amplitude (m).
    density : float
        Fluid density (kg/m³).
    gravity : float
        Gravitational acceleration (m/s²).

    Returns
    -------
    DiffractionConvergenceResult
    """
    if not subdivisions:
        raise ValueError("at least one subdivision level is required")

    levels: list[DiffractionAnchorResult] = []
    panel_counts: list[int] = []
    surge_errors: list[float] = []
    heave_errors: list[float] = []

    for sub in subdivisions:
        result = compute_diffraction_anchor(
            ka=ka, subdivisions=int(sub),
            radius=radius, amplitude=amplitude,
            density=density, gravity=gravity,
        )
        levels.append(result)
        panel_counts.append(result.panel_count)
        surge_errors.append(result.surge_rel_error)
        heave_errors.append(result.heave_rel_error)

    # Characteristic panel size h ~ 1/sqrt(N)
    h_values = [1.0 / math.sqrt(n) for n in panel_counts]

    observed_order_surge = _observed_order(h_values, surge_errors)
    observed_order_heave = _observed_order(h_values, heave_errors)

    return DiffractionConvergenceResult(
        ka=ka,
        levels=tuple(levels),
        panel_counts=tuple(panel_counts),
        surge_errors=tuple(surge_errors),
        heave_errors=tuple(heave_errors),
        observed_order_surge=observed_order_surge,
        observed_order_heave=observed_order_heave,
        finest_surge_error=surge_errors[-1] if surge_errors else 0.0,
        finest_heave_error=heave_errors[-1] if heave_errors else 0.0,
    )
