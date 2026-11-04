"""Multiply-connected 2-D Laplace domains: the concentric annulus.

Why this module exists
----------------------
``tensorbem.solver`` assembles one :class:`~tensorbem.geometry.PanelMesh`, and
:class:`~tensorbem.geometry.PanelMesh` describes *one* closed polygon. The
verification case-selection table in ``docs/verification.md`` therefore listed
the annulus (``u = A log r + B``, two boundary components with opposite
normals) as *"后续：需多边界组件"* rather than as a verified capability.

The assembly itself is component-agnostic: every matrix entry is a pair
single/double-layer panel integral, and the free term ``c = 1/2`` only needs a
smooth collocation point, which a straight-panel midpoint always is. What the
annulus actually needs is the **orientation convention**, and that is the trap
this module exists to close:

* the 2-D interior formulation assumes ``q`` is the derivative along the
  outward normal *of the fluid domain*;
* for a hole, "outward from the fluid" points **into the hole**, so the inner
  loop must be traversed clockwise while the outer loop is counter-clockwise;
* if both loops are given counter-clockwise, the assembled system is still
  well-posed and its residual is still at rounding level (the panels are the
  same set, only the normal signs of one component flip) — the answer is simply
  wrong. This is a *silent* failure, so the convention is enforced here at
  construction and re-checked at solve time instead of being left to the user.

The reference solution is closed-form: ``u = A log r + B`` is harmonic away
from the origin and constant on every circle, so both the Dirichlet data and
the exact normal derivative at each panel midpoint are known analytically. The
flux through each component is exactly ``±2 pi A`` for *any* refinement, because
for a regular polygon the chord integral ``integral ds / r**2`` telescopes; that
makes the per-loop flux a discretisation-free acceptance criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh
from .kernels import REFERENCE_LENGTH
from .solver import LaplaceDirichletResult, solve_laplace_dirichlet

FloatArray = NDArray[np.float64]

__all__ = [
    "Annulus",
    "annulus_mesh",
    "log_potential",
    "check_loop_orientation",
    "log_potential_boundary_data",
    "log_potential_normal_derivative",
    "solve_annulus_dirichlet",
    "loop_fluxes",
]


def _validate_integer(name: str, value: object, minimum: int = 3) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    count = int(value)
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return count


def _validate_radius(name: str, value: object) -> float:
    radius = float(value)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"{name} must be a finite positive length")
    return radius


@dataclass(frozen=True)
class Annulus:
    """A concentric annulus discretised into two oppositely wound loops.

    ``mesh.vertices`` concatenates the outer loop (counter-clockwise) and the
    inner loop (clockwise), so every panel normal of ``mesh`` points *away from
    the fluid*: the outer normals away from the centre, the inner normals
    towards it. The fluid domain is ``inner_radius < r < outer_radius``.
    """

    mesh: PanelMesh
    outer_panels: int
    inner_panels: int
    outer_radius: float
    inner_radius: float

    @property
    def panel_count(self) -> int:
        return self.mesh.panel_count

    @property
    def outer_slice(self) -> slice:
        return slice(0, self.outer_panels)

    @property
    def inner_slice(self) -> slice:
        return slice(self.outer_panels, self.panel_count)

    @property
    def loops(self) -> tuple[tuple[str, slice], tuple[str, slice]]:
        """``(("outer", slice), ("inner", slice))`` — the two components."""
        return (("outer", self.outer_slice), ("inner", self.inner_slice))

    @property
    def panel_radii(self) -> FloatArray:
        """Distance of every panel midpoint from the common centre."""
        return np.linalg.norm(self.mesh.centers, axis=1)


def annulus_mesh(
    outer_panels: int,
    inner_panels: int,
    outer_radius: float = 1.0,
    inner_radius: float = 0.5,
    inner_phase: float = 0.0,
) -> Annulus:
    """Discretise ``inner_radius < r < outer_radius`` into an :class:`Annulus`.

    Vertices lie on the two circles and the panels are the chords between them
    (an inscribed polygon per loop), so every collocation point is on its
    circle — the geometry error is the chord error alone and is not hidden by
    the panel-constant trace.

    Args:
        outer_panels: panels on the outer loop (counter-clockwise).
        inner_panels: panels on the inner loop (wound clockwise on purpose).
        outer_radius: radius of the outer circle.
        inner_radius: radius of the inner circle, strictly smaller.
        inner_phase: angular offset of the inner loop. A non-zero value breaks
            the radial alignment of the two discretisations, which is how the
            phase-independence of the answer can be checked.
    """
    outer_count = _validate_integer("outer_panels", outer_panels)
    inner_count = _validate_integer("inner_panels", inner_panels)
    r_outer = _validate_radius("outer_radius", outer_radius)
    r_inner = _validate_radius("inner_radius", inner_radius)
    if not r_inner < r_outer:
        raise ValueError("inner_radius must be strictly smaller than outer_radius")
    phase = float(inner_phase)
    if not math.isfinite(phase):
        raise ValueError("inner_phase must be finite")

    theta_outer = 2.0 * math.pi * np.arange(outer_count) / outer_count
    theta_inner = -2.0 * math.pi * np.arange(inner_count) / inner_count + phase
    outer_vertices = r_outer * np.column_stack(
        (np.cos(theta_outer), np.sin(theta_outer))
    )
    inner_vertices = r_inner * np.column_stack(
        (np.cos(theta_inner), np.sin(theta_inner))
    )
    vertices = np.vstack((outer_vertices, inner_vertices))
    return Annulus(
        mesh=PanelMesh(vertices),
        outer_panels=outer_count,
        inner_panels=inner_count,
        outer_radius=r_outer,
        inner_radius=r_inner,
    )


def check_loop_orientation(annulus: Annulus) -> None:
    """Raise unless the outer loop is CCW and the inner loop is CW.

    A flipped inner loop keeps the system square and its residual at rounding
    level while flipping the sign of every inner normal, so the solve silently
    returns a wrong ``q``. Checking the signed areas is the cheapest place to
    catch it, and it is checked again inside :func:`solve_annulus_dirichlet`
    rather than trusted from construction.

    The normal test is geometric and independent of the parameterisation: on a
    pair of concentric circles the fluid lies *outside* the inner polygon, so a
    fluid-outward normal must satisfy ``n · (x_c - centre) > 0`` on the outer
    loop and ``< 0`` on the inner loop.
    """
    mesh = annulus.mesh
    vertices = mesh.vertices
    for name, index in annulus.loops:
        points = vertices[index]
        # Shoelace area of this component, in its own traversal direction.
        signed = 0.5 * float(
            np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - points[:, 1] * np.roll(points[:, 0], -1)
            )
        )
        expected_positive = name == "outer"
        if expected_positive and signed <= 0.0:
            raise ValueError(
                "the outer loop of a multiply-connected domain must be wound "
                "counter-clockwise (signed area > 0) so that its panel normals "
                f"point out of the fluid; got signed area {signed:+.6e}"
            )
        if not expected_positive and signed >= 0.0:
            raise ValueError(
                "the inner loop (a hole) must be wound clockwise so that its "
                "panel normals point away from the fluid, i.e. into the hole; "
                f"got signed area {signed:+.6e} -- with the opposite winding the "
                "system stays solvable and its residual stays at rounding level, "
                "so the result would be silently wrong"
            )
        normals = mesh.normals[index]
        centres = mesh.centers[index]
        radial = centres - centres.mean(axis=0)
        # Each panel normal must be antiparallel to the fluid-side radial sense.
        projection = np.sum(normals * radial, axis=1)
        points_outward = bool(
            np.all(projection > 0.0) if expected_positive else np.all(projection < 0.0)
        )
        if not points_outward:
            raise ValueError(
                f"the {name} loop has a panel whose normal does not point away "
                "from the fluid domain"
            )


def log_potential(radius: FloatArray | float, amplitude: float = 1.0, offset: float = 0.0):
    """The exact harmonic reference ``u = A log r + B`` (``r = |x|``)."""
    scale = float(amplitude)
    shift = float(offset)
    if not math.isfinite(scale) or not math.isfinite(shift):
        raise ValueError("amplitude and offset must be finite")
    r = np.asarray(radius, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("the logarithmic potential needs r > 0")
    values = scale * np.log(r) + shift
    return float(values) if values.ndim == 0 else values


def log_potential_boundary_data(
    annulus: Annulus, amplitude: float = 1.0, offset: float = 0.0
) -> FloatArray:
    """Exact Dirichlet data of ``u = A log r + B`` at every panel midpoint."""
    return np.asarray(
        log_potential(annulus.panel_radii, amplitude=amplitude, offset=offset),
        dtype=float,
    )


def log_potential_normal_derivative(
    annulus: Annulus, amplitude: float = 1.0
) -> FloatArray:
    """Exact ``du/dn`` of ``u = A log r`` along the fluid-outward panel normals.

    Because the mesh normals point away from the fluid, the sign flips between
    the two components: ``+A/r`` on the outer loop, ``-A/r`` on the inner one.
    """
    scale = float(amplitude)
    radii = annulus.panel_radii
    q = np.empty(annulus.panel_count, dtype=float)
    q[annulus.outer_slice] = scale / radii[annulus.outer_slice]
    q[annulus.inner_slice] = -scale / radii[annulus.inner_slice]
    return q


def solve_annulus_dirichlet(
    annulus: Annulus,
    boundary_values: FloatArray | None = None,
    reference_length: float = REFERENCE_LENGTH,
) -> LaplaceDirichletResult:
    """Solve the interior Dirichlet problem on an annulus.

    ``boundary_values`` defaults to the exact logarithmic reference. The
    returned ``normal_derivative`` follows each panel's fluid-outward normal,
    i.e. ``+A/r`` on the outer loop and ``-A/r`` on the inner loop for the
    default data.
    """
    if not isinstance(annulus, Annulus):
        raise TypeError("annulus must be an Annulus")
    check_loop_orientation(annulus)
    values = (
        log_potential_boundary_data(annulus)
        if boundary_values is None
        else np.asarray(boundary_values, dtype=float)
    )
    return solve_laplace_dirichlet(
        annulus.mesh, values, reference_length=reference_length
    )


def loop_fluxes(annulus: Annulus, normal_derivative: FloatArray) -> dict[str, float]:
    """Panel-constant flux ``sum_j q_j * length_j`` over each loop, plus the net.

    For ``u = A log r`` the closed-form flux is ``+2 pi A`` on the outer loop
    and ``-2 pi A`` on the inner one. That value is exact for a *polygonal*
    loop too, because a chord at distance ``d`` from the centre carries
    ``A d * integral ds / r**2 = 2 pi A / n`` and the ``n`` chords of a loop sum
    to the circle's value; the panel-midpoint rule used here reproduces that
    integral up to its own ``O(h**2)`` quadrature error, so the deviation is a
    measure of the trace error, not of geometry. The net flux vanishes for any
    source-free field, which double-checks that the two components were
    assembled with consistent normal senses.
    """
    q = np.asarray(normal_derivative, dtype=float)
    if q.shape != (annulus.panel_count,):
        raise ValueError(
            f"normal_derivative must have shape ({annulus.panel_count},)"
        )
    lengths = annulus.mesh.lengths
    fluxes = {
        name: float(np.sum(q[index] * lengths[index]))
        for name, index in annulus.loops
    }
    fluxes["net"] = fluxes["outer"] + fluxes["inner"]
    return fluxes