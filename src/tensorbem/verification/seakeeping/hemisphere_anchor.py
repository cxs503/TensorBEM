"""Hemisphere radiation added-mass analytical anchor.

For a floating hemisphere of radius ``a`` in heave, the zero-frequency
added mass has the closed-form limit:

    A₃₃(ω→0) / (ρ·V) → 1.0

where V = 2/3·π·a³ is the displaced volume. At zero frequency, the added
mass equals the displaced mass for a body with a vertical symmetry axis
(Newman 1977, §4.14).

At finite frequency, Hulme (1982) provides a semi-analytical reference via
spherical wave-free multipole expansion. At Ka = 0.01, Hulme gives
A₃₃' ≈ 0.854, approaching 1.0 as Ka → 0.
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
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

FloatArray = NDArray[np.float64]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


@dataclass(frozen=True)
class HemisphereAnchorResult:
    """Result of the hemisphere zero-frequency added-mass anchor."""

    radius: float
    density: float
    # Analytical zero-frequency limit
    a33_zero_freq_dimless: float  # = 1.0 (normalized by displaced mass)
    displaced_volume: float
    displaced_mass: float
    # Hulme reference at low Ka
    ka_low: float
    hulme_a33_dimless: float
    # BEM results at low frequency
    a33_bem_dimless: float
    a33_bem_dim: float
    a33_vs_hulme_rel_error: float  # Error vs Hulme reference
    a33_vs_zero_freq_rel_error: float  # Error vs theoretical zero-freq limit
    # Mesh info
    subdivisions: int
    panel_count: int
    condition_number: float


def compute_hemisphere_zero_frequency_anchor(
    radius: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    subdivisions: int = 3,
    ka_low: float = 0.01,
) -> HemisphereAnchorResult:
    """Compute the zero-frequency added-mass anchor for a floating hemisphere.

    The analytical limit is A₃₃/(ρ·V) → 1.0 as Ka → 0.
    We validate against Hulme (1982) at low Ka and the BEM solve.

    Parameters
    ----------
    radius : float
        Hemisphere radius (m).
    density : float
        Fluid density (kg/m³).
    subdivisions : int
        Mesh refinement level (panel count = 16 × 4^subdivisions for the
        full sphere; the wetted hemisphere has half that).
    ka_low : float
        Non-dimensional wavenumber for the low-frequency BEM solve.

    Returns
    -------
    HemisphereAnchorResult
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if density <= 0:
        raise ValueError("density must be positive")
    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")
    if ka_low <= 0:
        raise ValueError("ka_low must be positive")

    # Analytical: zero-frequency limit A33/(rho*V) = 1.0
    displaced_volume = (2.0 / 3.0) * math.pi * radius ** 3
    displaced_mass = density * displaced_volume
    a33_zero_freq_dimless = 1.0

    # Hulme reference at low Ka
    ref = hulme_hemisphere_radiation(ka_low, radius=radius, density=density)
    hulme_a33_dimless = ref.heave_added_mass_dimless

    # BEM solve at low frequency
    mesh = hemisphere_hull_mesh(subdivisions, radius=radius)
    nu = ka_low / radius
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
    rad = solve_radiation_3d(
        mesh, nu, modes=(2,), density=density, gravity=_DEFAULT_GRAVITY,
        operators=ops,
    )

    a33_bem_dim = float(rad.added_mass[0, 0])
    a33_bem_dimless = a33_bem_dim / (density * displaced_volume)

    # Errors
    a33_vs_hulme_rel_error = abs(a33_bem_dimless - hulme_a33_dimless) / hulme_a33_dimless
    a33_vs_zero_freq_rel_error = abs(a33_bem_dimless - a33_zero_freq_dimless)

    return HemisphereAnchorResult(
        radius=radius,
        density=density,
        a33_zero_freq_dimless=a33_zero_freq_dimless,
        displaced_volume=displaced_volume,
        displaced_mass=displaced_mass,
        ka_low=ka_low,
        hulme_a33_dimless=hulme_a33_dimless,
        a33_bem_dimless=a33_bem_dimless,
        a33_bem_dim=a33_bem_dim,
        a33_vs_hulme_rel_error=a33_vs_hulme_rel_error,
        a33_vs_zero_freq_rel_error=a33_vs_zero_freq_rel_error,
        subdivisions=subdivisions,
        panel_count=mesh.panel_count,
        condition_number=rad.condition_number,
    )


def compute_hemisphere_radiation_study(
    radius: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    subdivisions: int = 3,
    ka_values: Sequence[float] = (0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0),
) -> dict[str, Any]:
    """Full radiation study: BEM vs Hulme (1982) across frequency range.

    Parameters
    ----------
    radius : float
        Hemisphere radius (m).
    density : float
        Fluid density (kg/m³).
    subdivisions : int
        Mesh refinement level.
    ka_values : sequence
        Non-dimensional wavenumbers to evaluate.

    Returns
    -------
    dict
        Contains 'analytical_a33_zero_freq', 'levels' (list of per-ka results).
    """
    mesh = hemisphere_hull_mesh(subdivisions, radius=radius)
    displaced_volume = (2.0 / 3.0) * math.pi * radius ** 3
    a33_zero = density * displaced_volume

    levels: list[dict[str, Any]] = []
    for ka in ka_values:
        ka = float(ka)
        nu = ka / radius
        ref = hulme_hemisphere_radiation(ka, radius=radius, density=density)
        ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")
        rad = solve_radiation_3d(
            mesh, nu, modes=(0, 2), density=density, gravity=_DEFAULT_GRAVITY,
            operators=ops,
        )

        a33_bem = float(rad.added_mass[1, 1])
        b33_bem = float(rad.damping[1, 1])
        a33_dimless = a33_bem / (density * displaced_volume)
        b33_dimless = b33_bem / (density * ref.omega * displaced_volume)

        levels.append({
            "ka": ka,
            "omega": ref.omega,
            "a33_bem_dimless": a33_dimless,
            "a33_hulme_dimless": ref.heave_added_mass_dimless,
            "a33_rel_error": abs(a33_dimless - ref.heave_added_mass_dimless) / ref.heave_added_mass_dimless,
            "b33_bem_dimless": b33_dimless,
            "b33_hulme_dimless": ref.heave_damping_dimless,
            "b33_rel_error": abs(b33_dimless - ref.heave_damping_dimless) / max(ref.heave_damping_dimless, 1e-12),
            "condition_number": rad.condition_number,
        })

    return {
        "analytical_a33_zero_freq": a33_zero,
        "displaced_volume": displaced_volume,
        "levels": levels,
    }
