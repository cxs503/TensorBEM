"""Schiller-Naumann and Clift-Grace-Weber sphere drag correlations.

References
----------
- Schiller, L. & Naumann, A. (1935). "Über die grundlegenden Berechnungen
  bei der Schwerkraftaufbereitung". Zeitschrift des Vereins Deutscher
  Ingenieure, 77(12), 318-320.
- Clift, R., Grace, J.R. & Weber, M.E. (1978). "Bubbles, Drops, and
  Particles". Academic Press. (Standard reference for sphere drag)
- White, F.M. (2006). "Viscous Fluid Flow", 3rd ed. McGraw-Hill.

Schiller-Naumann correlation (valid for 0.1 < Re < 1000):
    C_D = (24/Re) * (1 + 0.15 * Re^0.687)

This is the standard engineering correlation for sphere drag in the
subcritical regime. For Re < 0.1, Stokes' law applies (C_D = 24/Re).
For Re > 1000, the drag coefficient approaches a constant (~0.44) in
the Newton's law regime.

The Clift-Grace-Weber book provides a more comprehensive correlation
that covers a wider range of Reynolds numbers.
"""

from __future__ import annotations

import math
from typing import Tuple


def reynolds_number(
    velocity: float, diameter: float, kinematic_viscosity: float
) -> float:
    """Compute Reynolds number Re = U * D / nu.

    Parameters
    ----------
    velocity : float
        Free-stream velocity (m/s).
    diameter : float
        Sphere diameter (m).
    kinematic_viscosity : float
        Kinematic viscosity of the fluid (m²/s).

    Returns
    -------
    float
        Reynolds number (dimensionless).
    """
    if kinematic_viscosity <= 0:
        raise ValueError("kinematic_viscosity must be positive")
    if diameter <= 0:
        raise ValueError("diameter must be positive")
    if velocity < 0:
        raise ValueError("velocity must be non-negative")
    return velocity * diameter / kinematic_viscosity


def schiller_naumann_cd(reynolds: float) -> float:
    """Schiller-Naumann drag coefficient correlation.

    C_D = (24/Re) * (1 + 0.15 * Re^0.687)

    Valid for 0.1 < Re < 1000. Outside this range, the correlation
    extrapolates but should be used with caution.

    Parameters
    ----------
    reynolds : float
        Reynolds number.

    Returns
    -------
    float
        Drag coefficient C_D (dimensionless).

    Notes
    -----
    For Re < 0.1, Stokes' law (C_D = 24/Re) is more accurate.
    For Re > 1000, the drag coefficient approaches ~0.44 (Newton's regime).
    """
    if reynolds <= 0:
        raise ValueError("reynolds must be positive")
    return (24.0 / reynolds) * (1.0 + 0.15 * reynolds**0.687)


def schiller_naumann_valid_range() -> Tuple[float, float]:
    """Return the valid Reynolds number range for Schiller-Naumann.

    Returns
    -------
    tuple
        (Re_min, Re_max) = (0.1, 1000)
    """
    return (0.1, 1000.0)


def clift_grace_weber_cd(reynolds: float) -> float:
    """Clift-Grace-Weber comprehensive drag coefficient correlation.

    This is a piecewise correlation covering a wider range of Reynolds
    numbers than Schiller-Naumann:

    - Re < 0.01: Stokes' law, C_D = 24/Re
    - 0.01 <= Re < 20: C_D = (24/Re) * (1 + 0.1315 * Re^(0.82 - 0.05*log10(Re)))
    - 20 <= Re < 260: C_D = (24/Re) * (1 + 0.1935 * Re^0.6305)
    - 260 <= Re < 1000: C_D = (24/Re) * (1 + 0.15 * Re^0.687)  [Schiller-Naumann]
    - 1000 <= Re < 5000: C_D = 0.44 (approximately)

    Parameters
    ----------
    reynolds : float
        Reynolds number.

    Returns
    -------
    float
        Drag coefficient C_D (dimensionless).

    Reference
    ---------
    Clift, R., Grace, J.R. & Weber, M.E. (1978). "Bubbles, Drops, and
    Particles", Academic Press, Table 5.2.
    """
    if reynolds <= 0:
        raise ValueError("reynolds must be positive")

    if reynolds < 0.01:
        # Stokes' law
        return 24.0 / reynolds
    elif reynolds < 20:
        # Intermediate regime 1
        log_re = math.log10(reynolds)
        exponent = 0.82 - 0.05 * log_re
        return (24.0 / reynolds) * (1.0 + 0.1315 * reynolds**exponent)
    elif reynolds < 260:
        # Intermediate regime 2
        return (24.0 / reynolds) * (1.0 + 0.1935 * reynolds**0.6305)
    elif reynolds < 1000:
        # Schiller-Naumann regime
        return schiller_naumann_cd(reynolds)
    elif reynolds < 5000:
        # Newton's regime (approximate)
        return 0.44
    else:
        # High Re, drag crisis region (simplified)
        return 0.2  # Approximate post-crisis value


def drag_force_from_cd(
    cd: float, velocity: float, diameter: float, density: float
) -> float:
    """Compute drag force from drag coefficient.

    F_D = 0.5 * rho * U² * A * C_D
    where A = pi * D² / 4 (projected area of sphere)

    Parameters
    ----------
    cd : float
        Drag coefficient (dimensionless).
    velocity : float
        Free-stream velocity (m/s).
    diameter : float
        Sphere diameter (m).
    density : float
        Fluid density (kg/m³).

    Returns
    -------
    float
        Drag force (N).
    """
    if cd < 0:
        raise ValueError("cd must be non-negative")
    if velocity < 0:
        raise ValueError("velocity must be non-negative")
    if diameter <= 0:
        raise ValueError("diameter must be positive")
    if density <= 0:
        raise ValueError("density must be positive")
    area = math.pi * diameter**2 / 4.0
    return 0.5 * density * velocity**2 * area * cd


def benchmark_table(
    reynolds_values: list[float] | None = None,
) -> list[dict[str, float]]:
    """Generate a benchmark table of drag coefficients vs Reynolds number.

    Parameters
    ----------
    reynolds_values : list of float, optional
        Reynolds numbers to evaluate. If None, uses a standard set.

    Returns
    -------
    list of dict
        Each dict has keys: reynolds, cd_sn (Schiller-Naumann),
        cd_cgw (Clift-Grace-Weber).
    """
    if reynolds_values is None:
        reynolds_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    results = []
    for re in reynolds_values:
        results.append(
            {
                "reynolds": re,
                "cd_schiller_naumann": schiller_naumann_cd(re),
                "cd_clift_grace_weber": clift_grace_weber_cd(re),
            }
        )
    return results
