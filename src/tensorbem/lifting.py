"""Analytic reference and verification harness for steady 2-D lifting flows.

Motivation
----------
Every solver in this package is *non-lifting*: the Dirichlet and zero-net-flux
Neumann solvers have unique solutions with no circulation, and the free-surface
solvers are radiation/diffraction problems of the same kind. A lifting flow is a
different well-posed problem -- the circulation is fixed by the Kutta condition
at the trailing edge, not by the boundary data -- and README roadmap item 6
lists "Kutta-condition lifting hydrofoils" as not implemented.

This module supplies the closed-form solution of the symmetric Joukowski family
in steady incompressible flow, meshes that sit exactly on that analytic surface,
and - since the point-vortex closure was re-derived and verified against that
exact solution - a lifting boundary-element solve,
:func:`solve_lifting_flow`.  See ``docs/lifting_kutta_closure.md`` for the
verification numbers and ``docs/lifting_foil.md`` for the history of the
closure that was previously (and wrongly) reported as sign-inverted.

Circulation sign convention
--------------------------
Two conventions coexist in this module and mixing them reads as a 100% error:

* :meth:`JoukowskiAirfoil.exact_circulation` returns ``Gamma = 4 pi a U sin(alpha)``,
  the value that enters the complex potential below together with
  ``C_L = 2 Gamma / (U c)``.  Call that the *clockwise* value: it is minus the
  contour integral of the velocity taken counter clockwise.
* :func:`vortex_velocity_field` and :func:`solve_lifting_flow` use the
  counter-clockwise positive circulation ``Gamma_ccw = closed_integral(v . ds)``
  of a point vortex.  The lift follows Kutta-Joukowski as ``L' = -rho U Gamma_ccw``,
  i.e. ``C_L = -2 Gamma_ccw / (U c)``.

For the Kutta solution of this module the two are related by
``Gamma_ccw = -exact_circulation()``; the earlier closure attempt compared them
without the sign change and concluded that its circulation was inverted.

Analytic solution
-----------------
The map ``z = zeta + b**2/zeta`` sends the circle ``|zeta + m| = a`` with
``a = m + b`` onto a symmetric aerofoil whose trailing edge is the cusp
``z = 2b``. With ``zeta0 = -m`` and uniform stream ``U = U e^{i alpha}``,

    F(zeta) = U [ (zeta - zeta0) e^{-i alpha} + a**2 e^{i alpha} / (zeta - zeta0) ]
              + (i Gamma / 2 pi) ln(zeta - zeta0)

is the complex potential. Requiring ``dF/dzeta = 0`` at the trailing edge
(stagnation point on the cusp: the Kutta condition for a cusp trailing edge)
gives

    Gamma = 4 pi a U sin(alpha),      C_L = 2 Gamma / (U c),

and the surface velocity follows from ``W = (dF/dzeta) / (dz/dzeta)`` with
``C_p = 1 - |W|**2/U**2``. Because ``F`` carries the constant
``-U zeta0 e^{-i alpha}`` while the physical stream potential is
``Re[U z e^{-i alpha}]``, the *decaying* disturbance potential of the exterior
Neumann problem is

    phi_d = Re[F] - Re[U z e^{-i alpha}] - U m cos(alpha),

i.e. it differs from the naive difference by the offset ``U m cos(alpha)``.
That offset is not cosmetic: without it the reference field does not decay, and
:func:`disturbance_potential` subtracts it so that the analytic reference can be
compared with :func:`tensorbem.solver.solve_laplace_exterior_neumann` directly.
The offset was confirmed numerically against the solver to 2e-3 absolute.

Limitations
-----------
* The lifting solve is 2-D, steady, inviscid and unbounded; see
  :func:`solve_lifting_flow` for the closure it uses.
* Symmetric (zero-camber) Joukowski sections only.
* Steady, inviscid, incompressible, infinite domain; no free surface, no
  forward speed, no boundary layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh, _validate_panel_count
from .kernels import REFERENCE_LENGTH
from .solver import LaplaceDirichletResult, solve_laplace_exterior_neumann

FloatArray = NDArray[np.float64]

_TWO_PI = 2.0 * math.pi
_ARC_SAMPLES = 20001


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class JoukowskiAirfoil:
    """Symmetric Joukowski aerofoil with its closed-form flow solution.

    ``thickness = 0.1`` and ``transform_radius = 1.0`` give a section with
    ``t/c = 0.1179`` and chord ``c = 4.0333``.
    """

    thickness: float = 0.1
    transform_radius: float = 1.0
    alpha_deg: float = 4.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "thickness", _positive("thickness", self.thickness))
        object.__setattr__(
            self, "transform_radius", _positive("transform_radius", self.transform_radius)
        )
        object.__setattr__(self, "alpha_deg", _finite("alpha_deg", self.alpha_deg))
        if abs(self.alpha_deg) >= 90.0:
            raise ValueError("alpha_deg must satisfy |alpha_deg| < 90")

    # -- geometry -----------------------------------------------------------
    @property
    def circle_centre(self) -> tuple[float, float]:
        return (-self.thickness, 0.0)

    @property
    def circle_radius(self) -> float:
        return self.thickness + self.transform_radius

    @property
    def alpha_rad(self) -> float:
        return math.radians(self.alpha_deg)

    @property
    def trailing_edge(self) -> tuple[float, float]:
        return (2.0 * self.transform_radius, 0.0)

    @property
    def leading_edge(self) -> tuple[float, float]:
        zeta = -(self.circle_radius + self.thickness)
        return (zeta + self.transform_radius**2 / zeta, 0.0)

    @property
    def chord(self) -> float:
        return self.trailing_edge[0] - self.leading_edge[0]

    def map_point(self, zeta: complex) -> complex:
        """Apply the Joukowski map to a complex circle-plane point."""
        return zeta + self.transform_radius**2 / zeta

    def circle_angles(self, panel_count: int) -> FloatArray:
        """Circle-plane angles giving nearly uniform arc length on the surface.

        Uniform ``theta`` spacing concentrates panels quadratically at the cusped
        trailing edge (``dz/dtheta = 0`` there), leaving panels three orders of
        magnitude shorter than the chord. The angles returned here invert the
        arc-length map so the panels are (near) equal length; the cusp is where
        a panel ends, and ``theta = 0`` (the trailing edge) stays a vertex.
        """
        count = _validate_panel_count(panel_count)
        if count < 8:
            raise ValueError("lifting meshes need at least 8 panels")
        fine = np.linspace(0.0, _TWO_PI, _ARC_SAMPLES)
        zeta = self.circle_centre[0] + self.circle_radius * np.exp(1j * fine)
        z = self.map_point(zeta)
        segment = np.abs(np.diff(np.concatenate((z, z[:1]))))
        cumulative = np.concatenate(([0.0], np.cumsum(segment)))
        targets = cumulative[-1] * np.arange(count) / count
        angles = np.concatenate((fine, [_TWO_PI]))
        return np.interp(targets, cumulative, angles)

    def vertices(self, panel_count: int) -> FloatArray:
        """Counter-clockwise equal-arc polygon with a vertex at the TE."""
        theta = self.circle_angles(panel_count)
        zeta = self.circle_centre[0] + self.circle_radius * np.exp(1j * theta)
        z = self.map_point(zeta)
        vertices = np.column_stack((z.real, z.imag))
        vertices[0] = np.array(self.trailing_edge)
        return vertices

    def mesh(self, panel_count: int) -> PanelMesh:
        """Return the aerofoil as a :class:`PanelMesh` with the TE as a vertex."""
        return PanelMesh(self.vertices(panel_count))

    def thickness_ratio(self) -> float:
        """Maximum thickness over chord, from a dense sampling of the surface."""
        theta = np.linspace(0.0, _TWO_PI, 4001)
        zeta = self.circle_centre[0] + self.circle_radius * np.exp(1j * theta)
        z = self.map_point(zeta)
        return float(2.0 * np.max(np.abs(z.imag)) / self.chord)

    def _inverse_map(self, points: FloatArray) -> "np.ndarray":
        """Circle-plane pre-image of physical points, keeping the outer branch."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        z = pts[:, 0] + 1j * pts[:, 1]
        radius = self.transform_radius
        disc = np.sqrt(z * z - 4.0 * radius * radius + 0.0j)
        root_a = 0.5 * (z + disc)
        root_b = 0.5 * (z - disc)
        centre = complex(*self.circle_centre)
        return np.where(
            np.abs(root_a - centre) > np.abs(root_b - centre), root_a, root_b
        )

    # -- closed-form solution ----------------------------------------------
    def exact_circulation(self, speed: float = 1.0) -> float:
        """Kutta circulation ``4 pi a U sin(alpha)`` (counter clockwise, lift up)."""
        speed = _positive("speed", speed)
        return 4.0 * math.pi * self.circle_radius * speed * math.sin(self.alpha_rad)

    def exact_lift_coefficient(self, speed: float = 1.0) -> float:
        """``C_L = 2 Gamma / (U c)`` from the Kutta circulation."""
        speed = _positive("speed", speed)
        return 2.0 * self.exact_circulation(speed) / (speed * self.chord)

    def exact_thin_aerofoil_lift_coefficient(self) -> float:
        """Thin-aerofoil limit ``2 pi (a/b) sin(alpha)`` of the same section."""
        return 2.0 * math.pi * (self.circle_radius / self.transform_radius) * math.sin(
            self.alpha_rad
        )

    def _exact_velocity(self, zeta: "np.ndarray", speed: float) -> "np.ndarray":
        centre = complex(*self.circle_centre)
        radius = self.circle_radius
        alpha = self.alpha_rad
        gamma = self.exact_circulation(speed)
        rel = zeta - centre
        dF = speed * (np.exp(-1j * alpha) - radius**2 * np.exp(1j * alpha) / rel**2)
        dF = dF + 1j * gamma / (_TWO_PI * rel)
        dz = 1.0 - self.transform_radius**2 / zeta**2
        return dF / dz

    def exact_surface_speed(self, points: FloatArray, speed: float = 1.0) -> FloatArray:
        """``|W|`` at points on (or very near) the aerofoil surface."""
        speed = _positive("speed", speed)
        zeta = self._inverse_map(points)
        return np.abs(self._exact_velocity(zeta, speed))

    def exact_pressure_coefficient(
        self, points: FloatArray, speed: float = 1.0
    ) -> FloatArray:
        """``C_p = 1 - |W|**2 / U**2`` at points on the aerofoil surface."""
        speed = _positive("speed", speed)
        return 1.0 - (self.exact_surface_speed(points, speed) / speed) ** 2

    def decay_offset(self, speed: float = 1.0) -> float:
        """Constant ``U m cos(alpha)`` removed to make the disturbance decay."""
        speed = _positive("speed", speed)
        return speed * self.thickness * math.cos(self.alpha_rad)

    def disturbance_potential(self, points: FloatArray, speed: float = 1.0) -> FloatArray:
        """Decaying disturbance potential of the exact field at ``points``.

        This is the reference for :func:`tensorbem.solver.solve_laplace_exterior_neumann`
        driven with the tangency flux: compare the returned boundary values with
        this function directly, with no constant fitted by hand.
        """
        speed = _positive("speed", speed)
        pts = np.asarray(points, dtype=float)
        zeta = self._inverse_map(pts)
        z = pts[:, 0] + 1j * pts[:, 1]
        alpha = self.alpha_rad
        centre = complex(*self.circle_centre)
        radius = self.circle_radius
        rel = zeta - centre
        potential = speed * np.real(
            rel * np.exp(-1j * alpha) + radius**2 * np.exp(1j * alpha) / rel
        )
        return potential - np.real(speed * z * np.exp(-1j * alpha)) - self.decay_offset(
            speed
        )


def panel_tangents(mesh: PanelMesh) -> FloatArray:
    """Unit contour tangents, consistent with :attr:`PanelMesh.normals`."""
    normals = mesh.normals
    return np.column_stack((-normals[:, 1], normals[:, 0]))


def stream_tangential_velocity(
    mesh: PanelMesh, alpha_deg: float, speed: float = 1.0
) -> FloatArray:
    """Free-stream component along the contour tangent, ``U . t``."""
    alpha = math.radians(_finite("alpha_deg", alpha_deg))
    speed = _positive("speed", speed)
    stream = speed * np.array([math.cos(alpha), math.sin(alpha)])
    return np.sum(panel_tangents(mesh) * stream, axis=1)


def tangency_flux(mesh: PanelMesh, alpha_deg: float, speed: float = 1.0) -> FloatArray:
    """Fluid-normal flux of the disturbance that enforces no penetration.

    The disturbance ``phi`` satisfies ``d(phi)/dn_body = -U . n_body``, and the
    exterior Neumann solver takes the fluiddomain normal ``n_e = -n_body``, so
    the flux to pass is ``U . n_body``.
    """
    alpha = math.radians(_finite("alpha_deg", alpha_deg))
    speed = _positive("speed", speed)
    stream = speed * np.array([math.cos(alpha), math.sin(alpha)])
    return mesh.normals @ stream


@dataclass(frozen=True)
class NonLiftingFieldComparison:
    """Agreement between the BEM disturbance field and the analytic reference."""

    potential_max_abs_error: float
    potential_rms_error: float
    speed_rms_error: float
    speed_max_abs_error: float
    relative_potential_error: float
    relative_speed_error: float
    condition_number: float
    relative_residual_l2: float
    result: LaplaceDirichletResult
    reference_potential: FloatArray
    surface_speed: FloatArray
    exact_surface_speed: FloatArray


def compare_nonlifting_field(
    foil: JoukowskiAirfoil,
    panel_count: int,
    speed: float = 1.0,
    reference_length: float = REFERENCE_LENGTH,
) -> NonLiftingFieldComparison:
    """Solve the non-lifting flow about ``foil`` and compare it to the exact field.

    The surface velocity of the boundary-element solution is the free-stream
    tangential component plus ``d(phi)/ds``, differenced with a three-point rule
    on the panel-centre samples; that is the same first/second-order
    reconstruction the rest of the package uses for surface quantities.
    """
    speed = _positive("speed", speed)
    mesh = foil.mesh(panel_count)
    result = solve_laplace_exterior_neumann(
        mesh, tangency_flux(mesh, foil.alpha_deg, speed), reference_length
    )
    reference = foil.disturbance_potential(mesh.centers, speed)
    potential_error = result.boundary_values - reference

    arc = surface_arc_coordinates(mesh)
    slope = _three_point_slopes(arc, result.boundary_values)
    surface_speed = np.abs(stream_tangential_velocity(mesh, foil.alpha_deg, speed) + slope)
    exact_speed = np.abs(
        foil.exact_surface_speed(mesh.centers, speed)
    )
    amplitude = float(np.max(np.abs(reference)))
    speed_scale = float(np.max(exact_speed))
    return NonLiftingFieldComparison(
        potential_max_abs_error=float(np.max(np.abs(potential_error))),
        potential_rms_error=float(np.sqrt(np.mean(potential_error**2))),
        speed_rms_error=float(np.sqrt(np.mean((surface_speed - exact_speed) ** 2))),
        speed_max_abs_error=float(np.max(np.abs(surface_speed - exact_speed))),
        relative_potential_error=float(np.max(np.abs(potential_error)) / amplitude),
        relative_speed_error=float(np.sqrt(np.mean((surface_speed - exact_speed) ** 2)) / speed_scale),
        condition_number=result.condition_number,
        relative_residual_l2=result.relative_residual_l2,
        result=result,
        reference_potential=reference,
        surface_speed=surface_speed,
        exact_surface_speed=exact_speed,
    )


def surface_arc_coordinates(mesh: PanelMesh) -> FloatArray:
    """Arc length at each panel centre, measured counter clockwise from panel 0."""
    lengths = mesh.lengths
    return np.concatenate(
        ([0.5 * lengths[0]], np.cumsum(lengths)[:-1] + 0.5 * lengths[1:])
    )


def _three_point_slopes(samples: FloatArray, values: FloatArray) -> FloatArray:
    """Along-contour derivative of panel-centre data with a three-point rule.

    Interior samples use the central stencil; the two end samples use a
    one-sided triple inside the array, which keeps the trailing-edge panels from
    mixing the two sides of the cusp.
    """
    n = len(values)
    if n < 4:
        raise ValueError("at least four samples are needed")
    slopes = np.empty(n)
    indices = [0] + list(range(1, n - 1)) + [n - 1]
    for index in indices:
        if index == 0:
            stencil, target = (0, 1, 2), 0
        elif index == n - 1:
            stencil, target = (n - 3, n - 2, n - 1), 2
        else:
            stencil, target = (index - 1, index, index + 1), 1
        s0, s1, s2 = (float(samples[i]) for i in stencil)
        u0, u1, u2 = (float(values[i]) for i in stencil)
        s_target = (s0, s1, s2)[target]
        total = 0.0
        for (sa, ua), (sb, sc) in (
            ((s0, u0), (s1, s2)),
            ((s1, u1), (s0, s2)),
            ((s2, u2), (s0, s1)),
        ):
            total += ua * (2.0 * s_target - sb - sc) / ((sa - sb) * (sa - sc))
        slopes[index] = total
    return slopes


def vortex_velocity_field(
    points: FloatArray,
    center: tuple[float, float] | FloatArray,
    circulation: float = 1.0,
) -> FloatArray:
    """Velocity of a point vortex, ``Gamma`` counter-clockwise positive.

    ``v = Gamma/(2 pi) * (-(y-y0), (x-x0)) / r**2``.  The vector field is
    single valued everywhere except at ``center``; the potential is not, which
    is exactly what carries the circulation of the lifting flow.
    """
    coordinates = np.atleast_2d(np.asarray(points, dtype=float))
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    origin = np.asarray(center, dtype=float).reshape(2)
    if not np.isfinite(origin).all():
        raise ValueError("center must be finite")
    if not math.isfinite(circulation):
        raise ValueError("circulation must be finite")
    delta = coordinates - origin
    radius_squared = np.sum(delta * delta, axis=1)
    if np.any(radius_squared <= 0.0):
        raise ValueError("a target point coincides with the vortex centre")
    scale = circulation / (_TWO_PI * radius_squared)
    return np.column_stack((-delta[:, 1] * scale, delta[:, 0] * scale))


@dataclass(frozen=True)
class LiftingFlowResult:
    """Lifting boundary-element solution for a symmetric Joukowski section.

    All velocities are normalised by the free-stream speed ``U``; ``rho = 1``.
    The circulation is the *counter-clockwise* value, so the lift coefficient is
    ``-2 * circulation_ccw / (U * chord)``.
    """

    mesh: PanelMesh
    alpha_deg: float
    speed: float
    panel_count: int
    vortex_center: FloatArray
    vortex_fraction: float
    circulation_ccw: float
    lift_coefficient_from_circulation: float
    lift_coefficient_from_pressure: float
    surface_speed: FloatArray
    surface_tangential_velocity: FloatArray
    boundary_values: FloatArray
    base_tangential_velocity: FloatArray
    vortex_tangential_velocity: FloatArray
    kutta_base_sum: float
    kutta_response_sum: float
    condition_number: float
    relative_residual_l2: float

    @property
    def kutta_residual(self) -> float:
        """Residual of the two-panel Kutta condition, normalised by the peak speed.

        It vanishes by construction (the circulation is solved for it); it is
        reported so a caller can see the closure really was imposed.
        """
        peak = float(np.max(np.abs(self.surface_tangential_velocity)))
        if peak == 0.0:
            return 0.0
        mismatch = self.surface_tangential_velocity[0] + self.surface_tangential_velocity[-1]
        return abs(float(mismatch)) / peak


def solve_lifting_flow(
    foil: JoukowskiAirfoil,
    panel_count: int,
    speed: float = 1.0,
    vortex_fraction: float = 0.25,
    reference_length: float = REFERENCE_LENGTH,
) -> LiftingFlowResult:
    """Lift the exterior-Neumann solver with an explicit Kutta closure.

    The disturbance is split as ``phi = phi_U + Gamma phi_v``:

    * ``phi_U`` is the non-lifting disturbance already used by
      :func:`compare_nonlifting_field` (flux ``U . n_body``);
    * ``phi_v`` is the response to a unit point vortex placed inside the section
      at ``leading_edge + vortex_fraction * chord``, plus the Neumann
      correction that cancels its normal velocity on the wall
      (flux ``v_vortex . n_body``).  By construction its circulation around the
      section is exactly ``1``: the correction is single valued on the contour,
      so ``closed_integral(grad(phi_v) . ds) = 1``.

    ``Gamma`` then follows from the Kutta condition at the cusp trailing edge:
    the two panels adjacent to the cusp must carry the *same* velocity vector,
    and because their tangents are almost exactly opposite that is

        ``v_t[0] + v_t[-1] = 0``,

    one linear equation in ``Gamma``.  Surface velocities are reconstructed
    from the boundary potential with :func:`_three_point_slopes` (one-sided
    stencils at the ends, so the two sides of the cusp never mix).

    The Kutta value of this family is a *stagnation* point at the cusp, so a
    converged solution has both trailing-edge panels slow, not fast.

    Notes
    -----
    A layer-gradient reconstruction of the same surface velocity was tried and
    rejected: for a circle it is an order of magnitude more accurate than the
    three-point rule, but the piecewise-constant single-layer density is
    logarithmically singular at panel ends, and at the cusp the two panels that
    define the closure sit right in that singularity.  It returned a
    circulation 30% too large at every mesh (see ``docs/lifting_kutta_closure.md``).
    """
    if not isinstance(foil, JoukowskiAirfoil):
        raise TypeError("foil must be a JoukowskiAirfoil")
    count = _validate_panel_count(panel_count)
    speed = _positive("speed", speed)
    fraction = float(vortex_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("vortex_fraction must lie strictly between 0 and 1")

    mesh = foil.mesh(count)
    tangents = panel_tangents(mesh)
    alpha = math.radians(foil.alpha_deg)
    stream = speed * np.array([math.cos(alpha), math.sin(alpha)])

    base = solve_laplace_exterior_neumann(
        mesh, tangency_flux(mesh, foil.alpha_deg, speed), reference_length
    )
    centre = np.asarray(foil.leading_edge, dtype=float) + fraction * (
        np.asarray(foil.trailing_edge, dtype=float) - np.asarray(foil.leading_edge, dtype=float)
    )
    vortex = vortex_velocity_field(mesh.centers, centre, 1.0)
    response = solve_laplace_exterior_neumann(
        mesh, np.sum(mesh.normals * vortex, axis=1), reference_length
    )

    arc = surface_arc_coordinates(mesh)
    base_tangential = tangents @ stream + _three_point_slopes(arc, base.boundary_values)
    vortex_tangential = np.sum(tangents * vortex, axis=1) + _three_point_slopes(
        arc, response.boundary_values
    )

    kutta_base = float(base_tangential[0] + base_tangential[-1])
    kutta_response = float(vortex_tangential[0] + vortex_tangential[-1])
    if not math.isfinite(kutta_response) or abs(kutta_response) < 1.0e-12:
        raise ValueError("the vortex response is Kutta-degenerate on this mesh")
    circulation = -kutta_base / kutta_response

    tangential = base_tangential + circulation * vortex_tangential
    surface_speed = np.abs(tangential)
    pressure_coefficient = 1.0 - (surface_speed / speed) ** 2
    # L' = -closed_integral(p n_y ds) with p = 0.5 rho U^2 C_p, rho = 1
    lift_pressure = -0.5 * speed**2 * float(
        np.sum(pressure_coefficient * mesh.normals[:, 1] * mesh.lengths)
    )
    lift_coefficient_pressure = lift_pressure / (0.5 * speed**2 * foil.chord)

    return LiftingFlowResult(
        mesh=mesh,
        alpha_deg=float(foil.alpha_deg),
        speed=speed,
        panel_count=count,
        vortex_center=centre,
        vortex_fraction=fraction,
        circulation_ccw=circulation,
        lift_coefficient_from_circulation=-2.0 * circulation / (speed * foil.chord),
        lift_coefficient_from_pressure=lift_coefficient_pressure,
        surface_speed=surface_speed,
        surface_tangential_velocity=tangential,
        boundary_values=base.boundary_values + circulation * response.boundary_values,
        base_tangential_velocity=base_tangential,
        vortex_tangential_velocity=vortex_tangential,
        kutta_base_sum=kutta_base,
        kutta_response_sum=kutta_response,
        condition_number=response.condition_number,
        relative_residual_l2=max(base.relative_residual_l2, response.relative_residual_l2),
    )
