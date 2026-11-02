"""Hydrodynamic pressure and surface-load integration for potential-flow results.

TensorBEM solves boundary-value problems for the velocity potential ``phi``;
ship hydrodynamics needs the *loads*.  This module is the bridge: it converts a
solved potential flow into a Bernoulli pressure field and integrates that
pressure over the wetted surface into a resultant force and a moment.

Conventions (they matter, and they are checked in the test suite)
------------------------------------------------------------------
* Panel normals ``n`` of a :class:`~tensorbem.geometry3d.TriangleMesh` point
  **out of the body into the fluid** (outward normals, positive signed
  volume).
* ``integrate_pressure_loads`` returns the raw pressure integrals in that
  convention, ``F_out = ∫ p n dS``.  The force **on the body** is
  ``F_body = -F_out`` because the fluid pushes along ``-n``; it is exposed as
  :attr:`PressureLoads.force_on_body`.
* The moment is taken about ``reference_center``:
  ``M_out = ∫ (x - c) x (p n) dS`` and ``M_body = -M_out``.
* ``steady_pressure_coefficient`` and ``bernoulli_pressure`` use the **total**
  velocity ``v = U + grad(phi_d)`` (free stream plus disturbance), for which
  steady Bernoulli gives the exact identity

      p - p_inf = (1/2) * rho * (U**2 - |v|**2)      (potential flow)

  and unsteady Bernoulli adds the ``-rho * dphi/dt`` term.  ``U`` is the
  magnitude of the free-stream velocity, so the free-stream direction never
  enters the pressure: it is already contained in ``v``.

Quadrature
----------
Constant-panel pressure is integrated exactly on each flat triangle
(``∫_T p dS = area * p``).  For a *linear* pressure field the centroid rule is
still exact because the centroid is the area-weighted mean of a linear
function over a triangle and the panel normal is constant.  Consequently a
hydrostatic pressure field ``p = rho * g * (z_fs - z)`` reproduces Archimedes'
result ``F = rho * g * V * z_hat`` to machine precision on **any** closed
faceted mesh, independently of the mesh resolution; this is what the
verification tests use as an exact reference for the integrator itself.

The module deliberately contains no free-surface, forward-speed (Munk moment),
or viscous physics.  It is a verified post-processing layer, not a
ship-hydrodynamics resistance model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tensorbem.geometry3d import TriangleMesh

FloatArray = np.ndarray

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


def _as_points(points: Any, name: str = "points") -> FloatArray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (M, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must all be finite")
    return array


def _positive_finite(value: float, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.floating, np.integer)
    ):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < 0.0 or (number == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return number


@dataclass(frozen=True)
class PressureLoads:
    """Pressure resultant and moment about a reference point.

    ``force`` is ``∫ p n dS`` with the outward panel normal of the mesh and
    ``moment`` is ``∫ (x - c) x (p n) dS``; the corresponding loads **on the
    body** are the negatives (:attr:`force_on_body`, :attr:`moment_on_body`).
    ``reference_pressure`` is removed from the pressure before integration and
    is recorded so the result is reproducible.
    """

    force: FloatArray
    moment: FloatArray
    reference_center: FloatArray
    wetted_area: float
    reference_pressure: float
    density: float

    @property
    def force_on_body(self) -> FloatArray:
        """Fluid force acting on the body, ``-∫ p n dS``."""
        return -self.force

    @property
    def moment_on_body(self) -> FloatArray:
        """Fluid moment acting on the body about the reference centre."""
        return -self.moment

    def coefficients(
        self,
        density: float,
        speed: float,
        area: float,
        length: float = 1.0,
    ) -> dict[str, float]:
        """Non-dimensional force and moment coefficients on the body.

        ``C = F_body / (0.5 * density * speed**2 * area)`` for the force and
        ``C_m = M_body / (0.5 * density * speed**2 * area * length)`` for the
        moment.  All components are reported separately so the caller can pick
        the drag/lift/side axis of its own frame.
        """
        rho = _positive_finite(density, "density")
        velocity = _positive_finite(speed, "speed")
        reference_area = _positive_finite(area, "area")
        reference_length = _positive_finite(length, "length")
        scale = 0.5 * rho * velocity**2 * reference_area
        force = self.force_on_body / scale
        moment = self.moment_on_body / (scale * reference_length)
        magnitude = float(np.linalg.norm(self.force_on_body)) / scale
        return {
            "cx": float(force[0]),
            "cy": float(force[1]),
            "cz": float(force[2]),
            "magnitude": magnitude,
            "cmx": float(moment[0]),
            "cmy": float(moment[1]),
            "cmz": float(moment[2]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "force": [float(value) for value in self.force],
            "moment": [float(value) for value in self.moment],
            "force_on_body": [float(value) for value in self.force_on_body],
            "moment_on_body": [float(value) for value in self.moment_on_body],
            "reference_center": [float(value) for value in self.reference_center],
            "wetted_area": float(self.wetted_area),
            "reference_pressure": float(self.reference_pressure),
            "density": float(self.density),
        }


def steady_pressure_coefficient(
    velocity: FloatArray, free_stream_speed: float
) -> FloatArray:
    """Pressure coefficient ``Cp = 1 - |v|**2 / U**2`` of a steady potential flow.

    ``velocity`` holds the **total** velocity (free stream plus disturbance) at
    the query points, shape ``(M, 3)``.
    """
    speed = _positive_finite(free_stream_speed, "free_stream_speed")
    v = _as_points(velocity, "velocity")
    return 1.0 - np.sum(v * v, axis=1) / (speed * speed)


def bernoulli_pressure(
    velocity: FloatArray,
    density: float = _DEFAULT_DENSITY,
    free_stream_speed: float = 0.0,
    free_stream_pressure: float = 0.0,
    potential_rate: FloatArray | None = None,
) -> FloatArray:
    """Bernoulli pressure of a potential flow (steady or unsteady).

    Returns ``p = p_inf + (1/2) * rho * (U**2 - |v|**2) - rho * dphi/dt`` with
    ``velocity`` the total velocity and ``U = free_stream_speed``.  For a body
    in a fluid at rest (``U = 0``) this reduces to the classical
    ``p = p_inf - rho * dphi/dt - (1/2) * rho * |v|**2``.

    ``potential_rate`` is the time derivative of the potential at the same
    points.  It is optional: ``None`` means a steady flow.
    """
    rho = _positive_finite(density, "density")
    speed = _positive_finite(free_stream_speed, "free_stream_speed", allow_zero=True)
    if not math.isfinite(float(free_stream_pressure)):
        raise ValueError("free_stream_pressure must be finite")
    v = _as_points(velocity, "velocity")
    pressure = (
        float(free_stream_pressure)
        + 0.5 * rho * (speed * speed - np.sum(v * v, axis=1))
    )
    if potential_rate is not None:
        rate = np.asarray(potential_rate, dtype=float)
        if rate.shape != (v.shape[0],):
            raise ValueError(f"potential_rate must have shape ({v.shape[0]},)")
        if not np.isfinite(rate).all():
            raise ValueError("potential_rate must all be finite")
        pressure = pressure - rho * rate
    return pressure


def hydrostatic_pressure(
    points: FloatArray,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    free_surface_z: float = 0.0,
) -> FloatArray:
    """Linear hydrostatic pressure ``p = rho * g * (z_fs - z)``.

    No clipping is applied: points above the free surface return negative
    gauge pressure, exactly as the linear field extrapolates.  The caller is
    responsible for restricting the integral to the wetted surface when a
    waterline is present.
    """
    rho = _positive_finite(density, "density")
    g = _positive_finite(gravity, "gravity")
    if not math.isfinite(float(free_surface_z)):
        raise ValueError("free_surface_z must be finite")
    x = _as_points(points)
    return rho * g * (float(free_surface_z) - x[:, 2])


def integrate_pressure_loads(
    mesh: TriangleMesh,
    pressure: FloatArray,
    reference_center: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    reference_pressure: float = 0.0,
    density: float = _DEFAULT_DENSITY,
) -> PressureLoads:
    """Integrate a per-panel pressure into a resultant force and moment.

    ``pressure`` is constant per panel (shape ``(F,)``), or given at the panel
    centroids of a linear field, in which case the centroid rule is exact on
    each flat triangle.  A uniform ``reference_pressure`` is removed first, so
    the result is the gauge load (the only part a closed body can feel).
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    values = np.asarray(pressure, dtype=float)
    panel_count = mesh.panel_count
    if values.shape != (panel_count,):
        raise ValueError(f"pressure must have shape ({panel_count},)")
    if not np.isfinite(values).all():
        raise ValueError("pressure must all be finite")
    if not math.isfinite(float(reference_pressure)):
        raise ValueError("reference_pressure must be finite")
    rho = _positive_finite(density, "density")
    center = np.asarray(reference_center, dtype=float)
    if center.shape != (3,):
        raise ValueError("reference_center must have shape (3,)")
    if not np.isfinite(center).all():
        raise ValueError("reference_center must be finite")

    gauge = values - float(reference_pressure)
    panel_force = (gauge * mesh.areas)[:, None] * mesh.normals
    force = np.sum(panel_force, axis=0)
    offsets = mesh.centroids - center[None, :]
    moment = np.sum(np.cross(offsets, panel_force), axis=0)
    return PressureLoads(
        force=force,
        moment=moment,
        reference_center=center,
        wetted_area=float(np.sum(mesh.areas)),
        reference_pressure=float(reference_pressure),
        density=rho,
    )


def steady_pressure_loads(
    mesh: TriangleMesh,
    velocity: FloatArray,
    free_stream_velocity: tuple[float, float, float] | FloatArray,
    density: float = _DEFAULT_DENSITY,
    reference_center: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    reference_pressure: float = 0.0,
) -> tuple[FloatArray, PressureLoads]:
    """Steady Bernoulli pressure coefficient and loads from a surface velocity.

    ``velocity`` is the **total** velocity at the panel centroids, shape
    ``(F, 3)``.  Returns ``(cp, loads)`` where ``cp = 1 - |v|**2/U**2`` and the
    loads are integrated with ``p - p_inf = (1/2) * rho * U**2 * cp``.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    v = _as_points(velocity, "velocity")
    panel_count = mesh.panel_count
    if v.shape != (panel_count, 3):
        raise ValueError(f"velocity must have shape ({panel_count}, 3)")
    free_stream = np.asarray(free_stream_velocity, dtype=float)
    if free_stream.shape != (3,):
        raise ValueError("free_stream_velocity must have shape (3,)")
    if not np.isfinite(free_stream).all():
        raise ValueError("free_stream_velocity must be finite")
    speed = float(np.linalg.norm(free_stream))
    rho = _positive_finite(density, "density")
    if not math.isfinite(float(reference_pressure)):
        raise ValueError("reference_pressure must be finite")
    cp = steady_pressure_coefficient(v, speed)
    pressure = float(reference_pressure) + 0.5 * rho * speed * speed * cp
    loads = integrate_pressure_loads(
        mesh,
        pressure,
        reference_center=reference_center,
        reference_pressure=float(reference_pressure),
        density=rho,
    )
    return cp, loads