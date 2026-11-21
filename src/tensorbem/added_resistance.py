"""Sectional radiated-energy approximation for a ship in regular head seas."""

import numpy as np


def radiated_resistance(x, mass, damping, incident_w, motion, *, k, omega_e, speed):
    """Gerritsma–Beukelman head-sea mean force in N, with exp(-i omega_e t).

    At increasing x stations in metres (positive forward), mass and damping
    are heave radiation coefficients per unit length at encounter frequency.
    incident_w holds the complex incident vertical water velocity at each
    station. motion contains the complex heave (m, up) and pitch (rad,
    bow down) at the same wave amplitude and rotation origin. The wave
    number k is intrinsic, speed is in m/s, and omega_e is in rad/s.
    """
    x, mass, damping, incident_w = [
        np.asarray(v) for v in (x, mass, damping, incident_w)
    ]
    if (
        x.ndim != 1
        or len(x) < 2
        or any(v.shape != x.shape for v in (mass, damping, incident_w))
        or not all(np.all(np.isfinite(v)) for v in (x, mass, damping, incident_w))
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError("finite station arrays on a strictly increasing grid required")
    if not (
        np.isfinite(k)
        and k > 0
        and np.isfinite(omega_e)
        and omega_e > 0
        and np.isfinite(speed)
        and speed >= 0
    ):
        raise ValueError(
            "positive wavenumber and encounter frequency, nonnegative speed required"
        )
    if np.shape(motion) != (2,) or not np.all(np.isfinite(motion)):
        raise ValueError("finite complex heave and pitch displacements required")
    heave, pitch = motion
    relative = -1j * omega_e * (heave - x * pitch) + speed * pitch - incident_w
    # Integration by parts for a pointed hull. This avoids differentiation of
    # computed added mass, and requires vanishing heave added mass at both ends.
    if mass[0] != 0 or mass[-1] != 0:
        raise ValueError("Expected two closed ends")
    v2 = abs(relative) ** 2
    # Integrate piecewise-linear N, m and |v|². The derivative term is exact
    # on each interval under that interpolation.
    dx = np.diff(x)
    radiation = np.sum(
        dx
        * (
            2 * damping[:-1] * v2[:-1]
            + damping[:-1] * v2[1:]
            + damping[1:] * v2[:-1]
            + 2 * damping[1:] * v2[1:]
        )
        / 6
    )
    convection = speed * np.sum(0.5 * (mass[1:] + mass[:-1]) * np.diff(v2))
    return k / (2 * omega_e) * (radiation + convection)


def bow_reflection(waterline, *, k, draft, froude, density, gravity, amplitude=1.0):
    """Fujii–Takahashi head-sea reflection, N; ITTC 27th Seakeeping Committee eqs. 12–14.

    waterline is the exposed *half* bow waterline (x, half-breadth) from stem
    to maximum beam. The factor of two accounts for port and starboard.
    """
    from scipy.special import i1e, k1e

    points = np.asarray(waterline, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or not np.isfinite(points).all()
    ):
        raise ValueError("finite half-bow waterline required")
    if not all(
        np.isfinite(v) and v > 0 for v in (k, draft, density, gravity)
    ) or not all(np.isfinite(v) and v >= 0 for v in (froude, amplitude)):
        raise ValueError("invalid wave or fluid parameters")
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dy = np.diff(points[:, 1])
    if np.any(ds <= 0) or np.any(dy < -1e-10):
        raise ValueError("use a monotone exposed half-bow")
    projected = 2 * np.sum(dy**3 / ds**2)
    kd = k * draft
    # Exponentially scaled functions avoid overflow in the short-wave limit.
    num = np.pi * i1e(kd)
    den = np.hypot(num, k1e(kd) * np.exp(-2 * kd))
    return (
        0.5
        * density
        * gravity
        * amplitude**2
        * projected
        * (num / den) ** 2
        * (1 + 5 * np.sqrt(froude))
    )
