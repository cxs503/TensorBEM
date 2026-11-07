"""Analytically reduced Maruo far-field mean drift force (m5296).

The un-reduced numerical flux integral oscillates with the radiation radius kR
because the interference term contains exp(ikR cos θ) which doesn't converge
under numerical quadrature. The analytically reduced Maruo (1957) formula
evaluates the interference integral via stationary phase, giving a kR-independent
expression:

    F̄_x = (ρω²/4k) ∫₀²π |K(θ)|² (1 - cos θ) dθ

where K(θ) is the Kochin function of the scattered wave. This is the standard
far-field formula for the mean drift force on a fixed body in head seas (wave
propagating in +x direction), deep water.

References
----------
Maruo, H. (1957). "On the theory of ship drift in waves." J. Soc. Naval Arch. Japan, 102.
"""

from __future__ import annotations

import math
import numpy as np

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.81

__all__ = [
    "maruo_mean_drift_force",
    "maruo_mean_drift_force_general",
]


def maruo_mean_drift_force(
    kochin_values,
    azimuths,
    wavenumber: float,
    *,
    n_theta: int = 8192,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict:
    """Maruo mean drift force in head seas (wave in +x direction).

    Parameters
    ----------
    kochin_values :
        Complex Kochin function K(θ) of the scattered wave, sampled at ``azimuths``.
    azimuths :
        Azimuthal angles θ (radians) at which ``kochin_values`` are given.
    wavenumber :
        Wave number k (rad/m).
    n_theta :
        Number of quadrature points for the azimuthal integral.
    density :
        Fluid density (kg/m³).
    gravity :
        Gravitational acceleration (m/s²).

    Returns
    -------
    result : dict
        Dictionary with keys:
        - ``mean_drift_force_x``: mean drift force in x-direction (N)
        - ``flux_integral``: the azimuthal integral ∫|K|²(1-cos θ)dθ
        - ``wavenumber``: k
        - ``n_theta``: number of quadrature points

    Notes
    -----
    The formula is:

        F̄_x = (ρω²/4k) ∫₀²π |K(θ)|² (1 - cos θ) dθ

    where ω = √(gk) is the wave frequency. The factor (1 - cos θ) comes from
    the momentum flux in the x-direction: the scattered wave carries momentum
    in all directions, but only the component opposite to the incident wave
    (i.e., in the -x direction) contributes to the drift force.
    """
    k = float(wavenumber)
    if k <= 0:
        raise ValueError("wavenumber must be positive")
    g = float(gravity)
    if g <= 0:
        raise ValueError("gravity must be positive")
    rho = float(density)
    if rho <= 0:
        raise ValueError("density must be positive")

    omega = math.sqrt(g * k)
    n_t = int(n_theta)
    if n_t < 128:
        raise ValueError("n_theta must be >= 128")

    # Quadrature points
    theta = 2.0 * math.pi * np.arange(n_t) / n_t
    dtheta = 2.0 * math.pi / n_t

    # Interpolate Kochin function to quadrature points
    koch = np.asarray(kochin_values, dtype=np.complex128)
    ang = np.mod(np.asarray(azimuths, dtype=float), 2.0 * math.pi)
    order = np.argsort(ang)
    ang, koch = ang[order], koch[order]
    # Close the periodic table
    ang = np.concatenate((ang, ang[:1] + 2.0 * math.pi))
    koch = np.concatenate((koch, koch[:1]))
    # Interpolate
    koch_interp = np.interp(theta, ang, koch.real) + 1j * np.interp(theta, ang, koch.imag)

    # Maruo integral: ∫|K|²(1 - cos θ)dθ
    integrand = np.abs(koch_interp) ** 2 * (1.0 - np.cos(theta))
    flux_integral = float(np.sum(integrand * dtheta))

    # Mean drift force
    force_x = (rho * omega * omega / (4.0 * k)) * flux_integral

    return {
        "mean_drift_force_x": force_x,
        "flux_integral": flux_integral,
        "wavenumber": k,
        "omega": omega,
        "n_theta": n_t,
    }


def maruo_mean_drift_force_general(
    kochin_values,
    azimuths,
    wavenumber: float,
    wave_angle: float = 0.0,
    *,
    n_theta: int = 8192,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict:
    """Maruo mean drift force for arbitrary wave direction.

    Parameters
    ----------
    kochin_values :
        Complex Kochin function K(θ) of the scattered wave.
    azimuths :
        Azimuthal angles θ (radians).
    wavenumber :
        Wave number k (rad/m).
    wave_angle :
        Wave direction β (radians). 0 = head seas (+x), π/2 = beam seas (+y).
    n_theta :
        Number of quadrature points.
    density :
        Fluid density (kg/m³).
    gravity :
        Gravitational acceleration (m/s²).

    Returns
    -------
    result : dict
        Dictionary with keys:
        - ``mean_drift_force_x``: x-component of mean drift force (N)
        - ``mean_drift_force_y``: y-component of mean drift force (N)
        - ``flux_integral_x``: azimuthal integral for x-component
        - ``flux_integral_y``: azimuthal integral for y-component

    Notes
    -----
    For arbitrary wave direction β the far-field formula is the vector
    statement "incident momentum flux minus scattered momentum flux":

        F̄_x = (ρω²/4k) ∫₀²π |K(θ)|² (cos β - cos θ) dθ
        F̄_y = (ρω²/4k) ∫₀²π |K(θ)|² (sin β - sin θ) dθ

    where ``θ`` is the **absolute** azimuth of the radiation direction
    (measured from +x, the same frame as the sampled ``azimuths``) and ``β`` is
    the absolute direction the incident wave propagates in.  The subtracted
    term is therefore the fixed unit vector ``e_θ = (cos θ, sin θ)``, *not* a
    vector rotated with the heading: the scattered wave leaves along ``e_θ``
    regardless of where it came from, and only the incident flux rotates with
    ``β``.

    Two consistency checks fix this convention (both are asserted in
    ``tests/test_maruo_drift.py``):

    * ``wave_angle = 0`` reproduces :func:`maruo_mean_drift_force` exactly,
      ``(cos β - cos θ, sin β - sin θ) -> (1 - cos θ, - sin θ)``;
    * **rotational invariance** -- rotating the body by ``γ``
      (``|K|²(θ) -> |K|²(θ - γ)``) *and* the heading by ``γ`` must rotate the
      force vector by ``γ``: ``F̄(γ, |K|²_γ) = R(γ) F̄(0, |K|²)``.  For an
      isotropic scatterer (``|K|² = const``) this forces
      ``F̄ = 2π|K|²(ρω²/4k) e_β``, i.e. the force is along the incident
      direction.

    .. note::
       Before 2026-11-07 this function subtracted ``e_{θ-β}`` instead of
       ``e_θ``.  The two agree only at ``β = 0``, so the head-seas function was
       right and the general one was not: for a ``|K|² = 1 + 0.5 cos(θ - γ)``
       scatterer at beam seas (``β = γ``) it reported a transverse force of
       ``3.95e3 N`` where the exact invariant answer is zero.  See
       ``docs/m6287/maruo_general_heading_fix.md``.
    """
    k = float(wavenumber)
    if k <= 0:
        raise ValueError("wavenumber must be positive")
    g = float(gravity)
    if g <= 0:
        raise ValueError("gravity must be positive")
    rho = float(density)
    if rho <= 0:
        raise ValueError("density must be positive")

    omega = math.sqrt(g * k)
    n_t = int(n_theta)
    if n_t < 128:
        raise ValueError("n_theta must be >= 128")

    # Quadrature points
    theta = 2.0 * math.pi * np.arange(n_t) / n_t
    dtheta = 2.0 * math.pi / n_t

    # Interpolate Kochin function
    koch = np.asarray(kochin_values, dtype=np.complex128)
    ang = np.mod(np.asarray(azimuths, dtype=float), 2.0 * math.pi)
    order = np.argsort(ang)
    ang, koch = ang[order], koch[order]
    ang = np.concatenate((ang, ang[:1] + 2.0 * math.pi))
    koch = np.concatenate((koch, koch[:1]))
    koch_interp = np.interp(theta, ang, koch.real) + 1j * np.interp(theta, ang, koch.imag)

    # Maruo integrals
    k_sq = np.abs(koch_interp) ** 2
    # Incident unit vector e_beta minus the (absolute) radiation direction e_theta.
    integrand_x = k_sq * (math.cos(wave_angle) - np.cos(theta))
    integrand_y = k_sq * (math.sin(wave_angle) - np.sin(theta))

    flux_x = float(np.sum(integrand_x * dtheta))
    flux_y = float(np.sum(integrand_y * dtheta))

    coeff = rho * omega * omega / (4.0 * k)
    force_x = coeff * flux_x
    force_y = coeff * flux_y

    return {
        "mean_drift_force_x": force_x,
        "mean_drift_force_y": force_y,
        "flux_integral_x": flux_x,
        "flux_integral_y": flux_y,
        "wavenumber": k,
        "omega": omega,
        "wave_angle": wave_angle,
        "n_theta": n_t,
    }
