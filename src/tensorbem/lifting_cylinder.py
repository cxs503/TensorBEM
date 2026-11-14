"""Prescribed-circulation (rotating / Magnus) cylinder: closed form and BEM solve.

Why this module exists
----------------------
:mod:`tensorbem.lifting` lifts the exterior-Neumann solver by *solving* for the
circulation from a Kutta condition at a cusp; :mod:`tensorbem.solver` otherwise
handles flows whose circulation is zero by construction (the exterior Neumann
problem with zero net flux has a unique solution).  A rotating cylinder is the
third, simplest well-posed case: the circulation is *prescribed*, the body has no
cusp, and the whole flow has a one-line complex potential, so the numerical lift
and the stagnation-point shift can be checked against closed form instead of
against another solver.

What is solved and what is analytic
-----------------------------------
The disturbance produced by the body is ``phi' = phi_d - Gamma*theta/(2*pi)``:
a single-valued doublet ``phi_d = U a^2 cos(theta)/r`` that decays at infinity
plus the vortex potential, whose gradient is single valued but whose *potential*
is not.  The exterior Green representation only applies to a single-valued
trace, so the multivalued part is carried analytically (its velocity is added
back at the end); the BEM solves for ``phi_d``, the single-valued decaying part.
Two constant-panel Neumann problems are assembled -- the doublet data and the
unit-vortex response -- and on the inscribed-polygon ladder of this case the
second one is *identically zero* (the vortex is tangent at every panel
midpoint, see :func:`vortex_flux`), so the circulation is prescribed
analytically and only the faceted doublet solve is numerical.  That is the
honest statement of what a rotating cylinder tests: no Kutta condition is
solved here.

The decomposition used by the solve:

* the Neumann data imposed on the solver is the total fluid normal derivative
  ``q_e = (U e_x + grad(phi_d) + Gamma v_unit) . n_e``;
* the solve then returns the trace of ``phi_d`` (that is the unique decaying,
  single-valued harmonic function with that data, because the difference
  ``phi' - phi_d`` has zero normal derivative and is exactly the vortex);
* the total velocity is reassembled as ``U e_x + grad(phi_d) + Gamma v_unit``,
  with ``v_unit = (1/(2*pi*r)) e_theta`` the counter-clockwise unit vortex.

Sign conventions (mixing them reads as a 100 % error, see
``docs/lifting_kutta_closure.md`` for the same trap):

* ``circulation`` is the counter-clockwise contour integral ``closed_int(v.ds)``;
* Kutta-Joukowski then gives ``L' = -rho U Gamma`` (a positive, i.e.
  counter-clockwise, circulation produces *downward* lift), ``rho = 1`` here;
* the surface speed is ``v_theta = Gamma/(2*pi*a) - 2 U sin(theta)`` and
  ``Cp = 1 - (v_theta/U)**2``.

Reference provenance is closed form only: ``hand_typed_numbers = 0``, every
reference value is evaluated at the same points as the discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh
from .kernels import REFERENCE_LENGTH
from .lifting import vortex_velocity_field
from .solver import (
    LaplaceDirichletResult,
    evaluate_exterior_field,
    solve_laplace_exterior_neumann,
)

FloatArray = NDArray[np.float64]
TWO_PI = 2.0 * math.pi


def _positive(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


# --------------------------------------------------------------------------- #
# closed-form reference
# --------------------------------------------------------------------------- #
def exact_velocity(
    points: FloatArray,
    radius: float = 1.0,
    speed: float = 1.0,
    circulation: float = 0.0,
) -> FloatArray:
    """Velocity of ``W(z) = U (z + a^2/z) - i Gamma/(2 pi) ln(z/a)``.

    Returned as ``(u, v)`` from ``dW/dz = u - i v``; every point must be
    outside the cylinder.
    """
    a = _positive("radius", radius)
    u_free = _positive("speed", speed)
    gamma = _finite("circulation", circulation)
    coordinates = np.atleast_2d(np.asarray(points, dtype=float))
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    z = coordinates[:, 0] + 1j * coordinates[:, 1]
    modulus = np.abs(z)
    if np.any(modulus <= a * (1.0 - 1e-12)):
        raise ValueError("exact_velocity is only defined outside the cylinder")
    deriv = u_free * (1.0 - a**2 / z**2) - 1j * gamma / (TWO_PI * z)
    return np.column_stack((deriv.real, -deriv.imag))


def exact_perturbation_velocity(
    points: FloatArray,
    radius: float = 1.0,
    speed: float = 1.0,
    circulation: float = 0.0,
) -> FloatArray:
    """``exact_velocity`` minus the free stream ``U e_x``."""
    free = np.array([_positive("speed", speed), 0.0])
    return exact_velocity(points, radius, speed, circulation) - free


def exact_surface_speed(
    theta: FloatArray,
    radius: float = 1.0,
    speed: float = 1.0,
    circulation: float = 0.0,
) -> FloatArray:
    """``|v_theta|`` on the cylinder, ``v_theta = Gamma/(2 pi a) - 2 U sin(theta)``."""
    a = _positive("radius", radius)
    u_free = _positive("speed", speed)
    gamma = _finite("circulation", circulation)
    angles = np.asarray(theta, dtype=float)
    return np.abs(gamma / (TWO_PI * a) - 2.0 * u_free * np.sin(angles))


def exact_pressure_coefficient(
    theta: FloatArray,
    radius: float = 1.0,
    speed: float = 1.0,
    circulation: float = 0.0,
) -> FloatArray:
    """``Cp = 1 - (v_theta/U)**2`` (steady, incompressible, no gravity term)."""
    u_free = _positive("speed", speed)
    speed_theta = exact_surface_speed(theta, radius, speed, circulation)
    return 1.0 - (speed_theta / u_free) ** 2


def stagnation_angles(
    radius: float = 1.0,
    speed: float = 1.0,
    circulation: float = 0.0,
) -> list[float]:
    """Angles (radians, in ``[0, 2 pi)``) where the surface speed vanishes.

    ``v_theta = 0`` gives ``sin(theta) = Gamma/(4 pi U a)``.  With
    ``|Gamma| < 4 pi U a`` there are two stagnation points, ``theta_s`` and
    ``pi - theta_s``; at ``|Gamma| = 4 pi U a`` they merge at the top (or
    bottom); beyond that the flow has no stagnation point on the body and the
    surface velocity never reverses, which is outside this module's scope.
    """
    a = _positive("radius", radius)
    u_free = _positive("speed", speed)
    gamma = _finite("circulation", circulation)
    s = gamma / (4.0 * math.pi * u_free * a)
    if abs(s) > 1.0:
        raise ValueError(
            "|Gamma| >= 4 pi U a leaves no stagnation point on the cylinder; "
            "this module only covers the two-stagnation-point regime"
        )
    theta = math.asin(s)
    angles = [theta if theta >= 0.0 else theta + TWO_PI, math.pi - theta]
    return [float(angle % TWO_PI) for angle in angles]


def joukowski_lift(
    radius: float = 1.0, speed: float = 1.0, circulation: float = 0.0
) -> float:
    """``L' = -rho U Gamma`` with ``rho = 1`` (per unit span, ``rho U^2 a`` units)."""
    _positive("radius", radius)
    return -_positive("speed", speed) * _finite("circulation", circulation)


# --------------------------------------------------------------------------- #
# boundary-element solve
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RotatingCylinderResult:
    """BEM solution of the prescribed-circulation cylinder flow.

    All velocities are normalised by ``U`` and lengths by the nominal radius
    ``a``; ``rho = 1``.  ``circulation_ccw`` is the counter-clockwise contour
    integral of the total surface velocity, i.e. the value the solve actually
    produced (it is compared against the prescribed one by the caller).
    """

    mesh: PanelMesh
    speed: float
    radius: float
    prescribed_circulation: float
    circulation_ccw: float
    surface_velocity: FloatArray
    surface_pressure_coefficient: FloatArray
    lift: float
    drag: float
    neumann_result: LaplaceDirichletResult
    correction_result: LaplaceDirichletResult
    condition_number: float
    relative_residual_l2: float
    h_mean: float

    @property
    def lift_relative_error_vs_joukowski(self) -> float:
        exact = joukowski_lift(self.radius, self.speed, self.prescribed_circulation)
        if exact == 0.0:
            return abs(self.lift)
        return abs(self.lift - exact) / abs(exact)


def _doublet_velocity(points: FloatArray, radius: float, speed: float) -> FloatArray:
    """``grad(U a^2 cos(theta)/r)`` = ``grad(U a^2 x / (x^2+y^2))``, closed form."""
    coordinates = np.atleast_2d(np.asarray(points, dtype=float))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    r2 = x * x + y * y
    r4 = r2 * r2
    scale = speed * radius**2
    return np.column_stack((scale * (r2 - 2.0 * x * x) / r4, -2.0 * scale * x * y / r4))


def _unit_vortex(points: FloatArray) -> FloatArray:
    """Counter-clockwise unit vortex at the origin (``Gamma = 1``)."""
    return vortex_velocity_field(points, (0.0, 0.0), 1.0)


def doublet_flux(
    mesh: PanelMesh, radius: float = 1.0, speed: float = 1.0
) -> FloatArray:
    """``q_e = grad(phi_d) . n_e`` of the doublet ``phi_d = U a^2 cos(theta)/r``.

    This is the Neumann data of the faceted-body problem: the collocation
    points are panel midpoints, so the *exact* normal derivative is imposed
    there and the solve returns the single-valued potential that carries it.
    ``n_e = -n_body`` is the fluid-domain normal used by
    :func:`tensorbem.solver.solve_laplace_exterior_neumann`.
    """
    _positive("radius", radius)
    u_free = _positive("speed", speed)
    fluid_normal = -mesh.normals
    return np.sum(_doublet_velocity(mesh.centers, radius, u_free) * fluid_normal, axis=1)


def vortex_flux(mesh: PanelMesh) -> FloatArray:
    """Neumann data that cancels the unit vortex on the wall: ``q_e = v_unit . n_body``.

    ``grad(phi_c) . n_body = -v_unit . n_body`` is what removes the vortex's
    normal velocity, so the correction solve is fed ``+v_unit . n_body`` in the
    solver's fluid-normal convention.

    For a mesh whose panel normals are radial at the midpoints -- a regular
    polygon inscribed in the circle, which is what this case uses -- this data
    is *identically zero* (measured ~1e-16), because the vortex is tangent at
    every midpoint. The response solve is kept anyway: it is the term that
    makes the construction valid on a mesh where the midpoint normal is not
    radial (e.g. :func:`tensorbem.geometry.ellipse_mesh`), and
    ``tests/test_lifting_cylinder.py`` both pins the zero for the polygon and
    demonstrates the non-zero case. Consequence worth knowing: on the cylinder
    ladder the circulation is carried *analytically* by the vortex, and the BEM
    solve is the doublet Neumann problem -- what is verified numerically is the
    external field, the wall velocity, the pressure integral and the stagnation
    points, not a circulation solve (that is the Kutta family,
    :func:`tensorbem.lifting.solve_lifting_flow`).
    """
    unit_vortex = _unit_vortex(mesh.centers)
    return np.sum(unit_vortex * mesh.normals, axis=1)


def _surface_gradient(
    mesh: PanelMesh, flux: FloatArray, trace: FloatArray
) -> FloatArray:
    """Wall gradient from the solved trace: ``flux * n_e`` plus ``d(trace)/ds * t``."""
    tangents = mesh.vectors / mesh.lengths[:, None]
    spacing = 0.5 * (
        np.roll(mesh.lengths, 1) + 2.0 * mesh.lengths + np.roll(mesh.lengths, -1)
    )
    derivative = (np.roll(trace, -1) - np.roll(trace, 1)) / spacing
    return flux[:, None] * (-mesh.normals) + derivative[:, None] * tangents


def solve_rotating_cylinder(
    mesh: PanelMesh,
    radius: float = 1.0,
    speed: float = 1.0,
    circulation: float = 0.0,
    reference_length: float = REFERENCE_LENGTH,
) -> RotatingCylinderResult:
    """Solve the exterior flow about a cylinder with prescribed ``Gamma``.

    Two genuinely separate constant-panel Neumann solves, exactly the
    decomposition used by the verified lifting solve of
    :mod:`tensorbem.lifting` (there the circulation is fixed by a Kutta
    condition; here it is prescribed, which is what a rotating cylinder does):

    * *base*: the doublet Neumann data ``q_e = grad(phi_d).n_e`` -> ``phi_d``;
    * *response*: the unit-vortex flux ``q_e = v_unit.n_body`` -> ``phi_c``,
      the single-valued correction that makes the vortex tangent to the
      faceted wall (its net circulation is zero, so the combined field
      ``v_unit + grad(phi_c)`` carries exactly unit circulation).

    The total velocity is then ``U e_x + grad(phi_d) + Gamma (v_unit + grad(phi_c))``
    with normal components taken from the imposed data and tangential ones from
    the arc-length derivative of the solved trace.
    """
    _positive("radius", radius)
    u_free = _positive("speed", speed)
    gamma = _finite("circulation", circulation)
    if mesh.panel_count < 3:
        raise ValueError("mesh needs at least three panels")

    base_flux = doublet_flux(mesh, radius, u_free)
    base = solve_laplace_exterior_neumann(mesh, base_flux, reference_length)
    response_flux = vortex_flux(mesh)
    response = solve_laplace_exterior_neumann(mesh, response_flux, reference_length)

    doublet_surface = _surface_gradient(mesh, base_flux, base.boundary_values)
    correction_surface = _surface_gradient(mesh, response_flux, response.boundary_values)
    unit_vortex = _unit_vortex(mesh.centers)
    free = np.array([u_free, 0.0])
    surface_velocity = (
        free + doublet_surface + gamma * (unit_vortex + correction_surface)
    )
    speed_squared = np.sum(surface_velocity**2, axis=1)
    pressure = 1.0 - speed_squared / u_free**2
    # F' = -closed_int(p n_body ds), p = 0.5 rho U^2 Cp, rho = 1.
    force = -0.5 * u_free**2 * np.array(
        [np.sum(pressure * mesh.normals[:, k] * mesh.lengths) for k in (0, 1)]
    )
    tangents = mesh.vectors / mesh.lengths[:, None]
    circulation_ccw = float(
        np.sum(np.sum(surface_velocity * tangents, axis=1) * mesh.lengths)
    )

    return RotatingCylinderResult(
        mesh=mesh,
        speed=u_free,
        radius=float(radius),
        prescribed_circulation=gamma,
        circulation_ccw=circulation_ccw,
        surface_velocity=surface_velocity,
        surface_pressure_coefficient=pressure,
        lift=float(force[1]),
        drag=float(force[0]),
        neumann_result=base,
        correction_result=response,
        condition_number=max(
            float(base.condition_number), float(response.condition_number)
        ),
        relative_residual_l2=max(
            float(base.relative_residual_l2), float(response.relative_residual_l2)
        ),
        h_mean=float(np.mean(mesh.lengths)),
    )


def exterior_velocity_field(
    solution: RotatingCylinderResult,
    points: FloatArray,
    minimum_clearance_factor: float = 0.0,
) -> FloatArray:
    """Total velocity at fluid points.

    ``U e_x + grad(phi_d) + Gamma (v_unit + grad(phi_c))``, where both panel
    gradients come from the closed-form single/double-layer gradient integrals
    of the exterior Green representation
    (:func:`tensorbem.solver.evaluate_exterior_field`), so no finite-difference
    step enters the reported field.
    """
    doublet = evaluate_exterior_field(
        solution.mesh,
        solution.neumann_result,
        points,
        minimum_clearance_factor=minimum_clearance_factor,
    ).gradient
    correction = evaluate_exterior_field(
        solution.mesh,
        solution.correction_result,
        points,
        minimum_clearance_factor=minimum_clearance_factor,
    ).gradient
    free = np.array([solution.speed, 0.0])
    return (
        free
        + doublet
        + solution.prescribed_circulation * (_unit_vortex(points) + correction)
    )


def stagnation_angles_from_surface(
    mesh: PanelMesh,
    surface_velocity: FloatArray,
    tol: float = 1.0e-12,
) -> list[float]:
    """Locate the zeros of the tangential surface velocity by linear interpolation.

    The sign of ``v_t`` is sampled at panel midpoints; a sign change between two
    adjacent midpoints is interpolated linearly in the panel angle, which is
    second-order accurate for the zero location.  A tangential velocity that
    does not change sign anywhere means the prescribed ``|Gamma|`` is at or
    beyond the critical value ``4 pi U a`` and an empty list is returned.
    """
    centers = mesh.centers
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    tangents = mesh.vectors / mesh.lengths[:, None]
    tangential = np.sum(surface_velocity * tangents, axis=1)
    if np.max(np.abs(tangential)) <= tol:  # pragma: no cover - degenerate mesh
        return []
    zeros: list[float] = []
    count = len(tangential)
    for i in range(count):
        a_value = tangential[i]
        b_value = tangential[(i + 1) % count]
        if a_value == 0.0:
            zeros.append(float(theta[i] % TWO_PI))
            continue
        if a_value * b_value < 0.0:
            a_angle = theta[i]
            b_angle = theta[(i + 1) % count]
            if b_angle < a_angle:
                b_angle += TWO_PI
            weight = a_value / (a_value - b_value)
            zeros.append(float((a_angle + weight * (b_angle - a_angle)) % TWO_PI))
    return zeros


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, in ``[0, pi]``."""
    delta = (a - b) % TWO_PI
    return float(min(delta, TWO_PI - delta))