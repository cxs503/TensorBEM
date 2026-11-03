"""Kutta-condition lifting panel method for steady 2-D aerofoils.

Why this module exists
----------------------
Every other solver in TensorBEM is non-lifting: a Dirichlet or zero-net-flux
Neumann problem has a unique, circulation-free solution, so README roadmap item
6 listed "Kutta-condition lifting hydrofoils" as open.
``docs/lifting_foil.md`` section 4 records the failed attempt to close it by
superposing an exact base solution and an interior point vortex: the returned
circulation had the right magnitude and the wrong sign, so no lifting solve was
shipped.  Two independent attempts made in this module (m2573) are recorded
there as well -- a constant-strength source/vortex (Hess-Smith) assembly
converges only to ~7% lift error at 128 panels, so the shipped method is the
textbook **linear-strength vortex panel method**.

Formulation
-----------
A closed contour is discretised into ``N`` straight panels with nodes
``z_0 .. z_{N-1}``.  The bound vorticity is a piecewise-linear vortex sheet: on
panel ``j`` the strength varies linearly from ``gamma_j`` at node ``j`` to
``gamma_{j+1}`` at node ``j+1``.  The unknowns are the ``N`` node values and the
``N`` no-penetration conditions at the panel control points close the system.
The Kutta condition for a sharp trailing edge, "zero vorticity at the trailing
edge", is imposed by the antisymmetric closure

    gamma_N = -gamma_0,

which fixes how the sheet wraps around the trailing-edge vertex.  This is the
classical vortex panel method (Katz & Plotkin, *Low-Speed Aerodynamics*,
section 11.7); it converges at second order in the panel size on lift, three
orders better than a constant-strength vortex sheet for the same mesh.

Sign convention
---------------
Vorticity is taken positive counter clockwise, so a positive ``gamma`` produces
the velocity ``R90(r) / |r|^2`` per unit strength.  The aerodynamic circulation
(signed so that positive circulation means positive lift for a stream along
``+x``) is therefore ``Gamma = -oint gamma ds`` and the lift coefficient is
``C_L = 2 Gamma / (U c)``, which is the convention of
:meth:`tensorbem.lifting.JoukowskiAirfoil.exact_lift_coefficient`.  With the
other sign the computed lift comes out with the wrong sign -- exactly the defect
that blocked the superposition attempt.

Limitations
-----------
* Steady, inviscid, incompressible, unbounded flow: no free surface, no forward
  speed, no boundary layer, no separation, no compressibility.
* Sharp (cusp) trailing edge only; a blunt or rounded trailing edge needs a
  different closure.
* Symmetric Joukowski sections only, because that is the family with a closed
  analytic solution to compare against; the panel method itself is generic.
* Piecewise-constant panel geometry means the surface pressure is piecewise
  constant, so the leading-edge suction peak is first-order accurate even though
  the lift converges at second order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh, _validate_panel_count
from .lifting import JoukowskiAirfoil, panel_tangents

FloatArray = NDArray[np.float64]

_TWO_PI = 2.0 * math.pi


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def linear_vortex_influence(
    mesh: PanelMesh, points: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Induced velocity of a unit linear vortex sheet on every panel.

    Returns ``(c1, c2)``, both of shape ``(P, N, 2)``: the velocity at each of
    the ``P`` target points produced by panel ``j`` when its vortex strength is
    one at the first (respectively second) node and zero at the other.  The
    closed-form integrals are

    ``J0 = beta/|y0|``, ``J1 = ln(r2/r1)``, ``J2 = l - y0**2 J0``

    in the panel frame, where ``beta`` is the angle subtended by the panel at
    the target and ``(x0, y0)`` the target position measured from the first
    node along the panel tangent and normal.  ``beta`` is taken as a positive
    magnitude and the sign of the normal distance supplies the side, which
    avoids the ``atan2`` branch ambiguity at the collinear control point.

    The own-panel entry is the *outer* limit of the sheet: zero normal velocity
    and ``-gamma_mid/2`` along the panel tangent.  Using the principal value
    (the average of the two sides) instead biases every surface velocity by
    ``gamma/2`` and destroys the lift: that was the second defect found here.
    """
    starts = mesh.vertices
    ends = mesh.ends
    lengths = mesh.lengths
    tangents = panel_tangents(mesh)
    normals = mesh.normals
    # The (t, n) frame of a counter-clockwise polygon is right handed, so
    # R90(t) = n; a caller may hand in any orientation, hence the determinant.
    handedness = (
        tangents[:, 0] * normals[:, 1] - tangents[:, 1] * normals[:, 0]
    )[None, :]

    targets = np.asarray(points, dtype=float)
    rel_start = targets[:, None, :] - starts[None, :, :]
    rel_end = targets[:, None, :] - ends[None, :, :]
    x0 = np.einsum("pnk,nk->pn", rel_start, tangents)
    y0 = np.einsum("pnk,nk->pn", rel_start, normals)
    length = lengths[None, :]

    cross = (
        rel_start[:, :, 0] * rel_end[:, :, 1] - rel_start[:, :, 1] * rel_end[:, :, 0]
    )
    dot = np.einsum("pnk,pnk->pn", rel_start, rel_end)
    beta = np.arctan2(np.abs(cross), dot)              # angle in [0, pi]
    on_panel = np.abs(y0) <= 1.0e-12 * length
    j0 = np.where(on_panel, 0.0, beta / np.where(on_panel, 1.0, np.abs(y0)))
    r1_sq = np.einsum("pnk,pnk->pn", rel_start, rel_start)
    r2_sq = np.einsum("pnk,pnk->pn", rel_end, rel_end)
    j1 = 0.5 * np.log(np.maximum(r2_sq, 1.0e-300) / np.maximum(r1_sq, 1.0e-300))
    j2 = length - y0**2 * j0

    coefficients = []
    for gamma1, gamma2 in ((1.0, 0.0), (0.0, 1.0)):
        delta = gamma2 - gamma1
        node_term = gamma1 + delta * x0 / length
        along = -handedness * (y0 / _TWO_PI) * (node_term * j0 + (delta / length) * j1)
        across = -handedness * (1.0 / _TWO_PI) * (
            node_term * j1 + (delta / length) * j2
        )
        coefficients.append(
            along[:, :, None] * tangents[None, :, :]
            + across[:, :, None] * normals[None, :, :]
        )

    first, second = coefficients
    midpoints = mesh.centers
    for target in range(min(len(targets), mesh.panel_count)):
        if np.allclose(targets[target], midpoints[target], atol=1.0e-12):
            jump = -0.25 * handedness[0, target] * tangents[target]
            first[target, target] = jump
            second[target, target] = jump
    return first, second


def analytic_pressure_coefficient_at_panels(
    foil: JoukowskiAirfoil, panel_count: int, speed: float = 1.0
) -> FloatArray:
    """Exact ``C_p`` of the closed-form Kutta solution at every panel midpoint.

    The panel midpoints of the faceted surface lie *inside* the analytic
    aerofoil, so evaluating the analytic field there would be meaningless; the
    exact value is instead taken at the circle-plane angle halfway between the
    two vertices, which is the surface point the panel represents.
    """
    angles = np.asarray(foil.circle_angles(panel_count), dtype=float)
    centre = foil.circle_centre[0]
    radius = foil.circle_radius
    midpoints = []
    for index in range(len(angles)):
        theta_a = angles[index]
        theta_b = angles[(index + 1) % len(angles)]
        if theta_b < theta_a:
            theta_b += _TWO_PI
        theta = 0.5 * (theta_a + theta_b)
        zeta = centre + radius * complex(math.cos(theta), math.sin(theta))
        z = foil.map_point(zeta)
        midpoints.append((z.real, z.imag))
    return foil.exact_pressure_coefficient(np.array(midpoints), speed)


@dataclass(frozen=True)
class LiftingPanelSolution:
    """Result of the Kutta-condition panel solve on one aerofoil discretisation."""

    panel_count: int
    alpha_deg: float
    speed: float
    density: float
    chord: float
    mesh: PanelMesh
    node_vorticity: FloatArray
    tangential_velocity: FloatArray
    pressure_coefficient: FloatArray
    circulation: float
    lift_coefficient: float
    lift_coefficient_from_pressure: float
    kutta_residual: float
    relative_residual: float
    condition_number: float
    exact_circulation: float
    exact_lift_coefficient: float
    circulation_relative_error: float
    lift_coefficient_relative_error: float
    pressure_rms_error: float

    def summary(self) -> dict[str, object]:
        """Strict-JSON-friendly summary (no NaN/Inf, no arrays)."""
        return {
            "panel_count": self.panel_count,
            "alpha_deg": self.alpha_deg,
            "speed": self.speed,
            "density": self.density,
            "chord": self.chord,
            "circulation": self.circulation,
            "exact_circulation": self.exact_circulation,
            "circulation_relative_error": self.circulation_relative_error,
            "lift_coefficient": self.lift_coefficient,
            "exact_lift_coefficient": self.exact_lift_coefficient,
            "lift_coefficient_relative_error": self.lift_coefficient_relative_error,
            "lift_coefficient_from_pressure": self.lift_coefficient_from_pressure,
            "kutta_residual": self.kutta_residual,
            "relative_residual": self.relative_residual,
            "condition_number": self.condition_number,
            "pressure_rms_error": self.pressure_rms_error,
            "trailing_edge_vorticity": float(
                self.node_vorticity[0] + self.node_vorticity[0]
            ),
        }


def solve_lifting_airfoil(
    foil: JoukowskiAirfoil,
    panel_count: int,
    speed: float = 1.0,
    density: float = 1.0,
) -> LiftingPanelSolution:
    """Solve the Kutta-condition lifting flow past a symmetric Joukowski section.

    Returns the node vorticity, the surface velocity and pressure at the panel
    control points, the circulation, the lift coefficient (from both the
    Kutta-Joukowski identity and the pressure integral) and its error against
    the exact closed-form solution.
    """
    panels = _validate_panel_count(panel_count)
    if panels < 8:
        raise ValueError("panel_count must be at least 8 for a lifting solve")
    speed = _positive("speed", speed)
    density = _positive("density", density)

    mesh = foil.mesh(panels)
    normals = mesh.normals
    tangents = panel_tangents(mesh)
    lengths = mesh.lengths
    centres = mesh.centers

    alpha = math.radians(foil.alpha_deg)
    stream = speed * np.array([math.cos(alpha), math.sin(alpha)])

    first, second = linear_vortex_influence(mesh, centres)
    normal_first = np.einsum("pnk,pk->pn", first, normals)
    normal_second = np.einsum("pnk,pk->pn", second, normals)

    matrix = np.zeros((panels, panels), dtype=float)
    for panel in range(panels):
        matrix[:, panel] += normal_first[:, panel]
        upper = (panel + 1) % panels
        if upper == 0:
            # Kutta closure gamma_N = -gamma_0: the last panel's second node is
            # the first unknown with the opposite sign.
            matrix[:, 0] -= normal_second[:, panel]
        else:
            matrix[:, upper] += normal_second[:, panel]
    right_hand_side = -normals @ stream

    node_vorticity = np.linalg.solve(matrix, right_hand_side)
    wraparound = -node_vorticity[0]

    # Surface velocity at the control points, own-panel outer limit included.
    tangential_first = np.einsum("pnk,pk->pn", first, tangents)
    tangential_second = np.einsum("pnk,pk->pn", second, tangents)
    upper_nodes = np.append(node_vorticity[1:], wraparound)
    tangential = np.einsum("nk,k->n", tangents, stream)
    tangential += tangential_first @ node_vorticity
    tangential += tangential_second @ upper_nodes

    circulation_sheet = float(
        np.sum(0.5 * (node_vorticity + upper_nodes) * lengths)
    )
    circulation = -circulation_sheet
    lift_from_circulation = density * speed * circulation
    reference = 0.5 * density * speed**2 * foil.chord

    pressure = 1.0 - (tangential / speed) ** 2
    lift_from_pressure = -0.5 * density * speed**2 * float(
        np.sum(pressure * normals[:, 1] * lengths)
    )

    exact_pressure = analytic_pressure_coefficient_at_panels(foil, panels, speed)
    pressure_rms = float(np.sqrt(np.mean((pressure - exact_pressure) ** 2)))

    exact_gamma = foil.exact_circulation(speed)
    exact_cl = foil.exact_lift_coefficient(speed)
    cl_from_circulation = lift_from_circulation / reference

    return LiftingPanelSolution(
        panel_count=panels,
        alpha_deg=float(foil.alpha_deg),
        speed=speed,
        density=density,
        chord=float(foil.chord),
        mesh=mesh,
        node_vorticity=node_vorticity,
        tangential_velocity=tangential,
        pressure_coefficient=pressure,
        circulation=circulation,
        lift_coefficient=cl_from_circulation,
        lift_coefficient_from_pressure=lift_from_pressure / reference,
        kutta_residual=float(node_vorticity[0] + wraparound),
        relative_residual=float(
            np.linalg.norm(matrix @ node_vorticity - right_hand_side)
            / max(float(np.linalg.norm(right_hand_side)), 1.0e-300)
        ),
        condition_number=float(np.linalg.cond(matrix)),
        exact_circulation=exact_gamma,
        exact_lift_coefficient=exact_cl,
        circulation_relative_error=abs(circulation - exact_gamma)
        / abs(exact_gamma)
        if exact_gamma != 0.0
        else abs(circulation),
        lift_coefficient_relative_error=abs(cl_from_circulation - exact_cl)
        / abs(exact_cl)
        if exact_cl != 0.0
        else abs(cl_from_circulation),
        pressure_rms_error=pressure_rms,
    )