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

Conventions
-----------
``theta`` is measured in the same frame as the wave direction ``beta``
(0 = +x), so the result is a *vector* that must be frame covariant:

    F_j = (rho*omega^2/4k) Int |K(theta)|^2 [ e_beta_j - e_theta_j ] dtheta

with ``e_beta = (cos beta, sin beta)`` the direction of wave propagation and
``e_theta = (cos theta, sin theta)`` the radiation direction of the scattered
wave.  The head-sea entry point is the ``beta = 0`` case of this weight,
``e_beta - e_theta = (1 - cos theta, -sin theta)``.  The identical weight is
used by the verified far-field routine in :mod:`tensorbem.drift`
(``far_field_drift_3d``), which derives it from the control-surface momentum
flux; the ``e_beta`` term is the isotropic incident-flux part and the
``e_theta`` term the radiated momentum-flux moment.  Because ``omega^2 = gk``
the prefactor reduces to ``rho*g/4``, independent of the wave number.

The normalisation of ``K`` (wave amplitude included or not) is the caller's
convention: only ``|K|^2`` enters and no amplitude is assumed here, so the
returned force is in newtons for ``K`` in metre units.

References
----------
Maruo, H. (1957). "On the theory of ship drift in waves." J. Soc. Naval Arch. Japan, 102.
Maruo, H. (1960). "The drift of a body floating on waves." J. Ship Research, 4(3), 1-10.
Newman, J.N. (1977). Marine Hydrodynamics, MIT Press, section 7.3 (far-field drift force).
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
    if np.ndim(kochin_values) != 1 or np.ndim(azimuths) != 1:
        raise ValueError("kochin_values and azimuths must be one-dimensional")
    if len(kochin_values) != len(azimuths):
        raise ValueError(
            f"kochin_values ({len(kochin_values)}) and azimuths "
            f"({len(azimuths)}) must have the same length"
        )
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
    For arbitrary wave direction β, the Maruo formula generalizes to:

        F̄_x = (ρω²/4k) ∫₀²π |K(θ)|² (cos β - cos θ) dθ
        F̄_y = (ρω²/4k) ∫₀²π |K(θ)|² (sin β - sin θ) dθ

    Equivalently ``F̄ = (ρω²/4k) ∫ |K(θ)|² (e_β - e_θ) dθ`` with
    ``e_θ = (cos θ, sin θ)``.  Note that the subtracted radiation direction is
    ``e_θ`` itself, not ``e_{θ-β}``: the weight must be frame covariant, i.e.
    rotating body and incident waves together by ``α`` rotates ``F̄`` by
    ``α``.  The ``e_θ - e_{θ-β}`` variant satisfies the head-sea limit but
    breaks that covariance and mis-predicts oblique/beam-sea lateral forces
    (see ``docs/maruo_drift.md`` and ``tests/test_maruo_drift.py``).
    """
    k = float(wavenumber)
    if np.ndim(kochin_values) != 1 or np.ndim(azimuths) != 1:
        raise ValueError("kochin_values and azimuths must be one-dimensional")
    if len(kochin_values) != len(azimuths):
        raise ValueError(
            f"kochin_values ({len(kochin_values)}) and azimuths "
            f"({len(azimuths)}) must have the same length"
        )
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
    # Frame-covariant Maruo weight e_beta - e_theta (see module docstring).
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
