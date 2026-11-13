"""2-D Laplace BEM on multiply connected domains (several boundary loops).

Why this module exists
----------------------
``docs/verification.md`` §1 registers the annular example ``u = A log r + B``
as *not implemented*: "多连通边界、内外法向 | 后续：需多边界组件" (needs a
multi-component boundary).  ``tensorbem.solver`` and ``tensorbem.geometry``
both hard-wire a **single** counter-clockwise closed polygon, so an annular
domain -- whose boundary is two disjoint loops with opposite orientation -- could
not be assembled at all.

This module supplies the missing component and nothing else:

* :class:`MultiLoopMesh` concatenates panel data from several loops while
  keeping each loop's closing segment (and therefore its normals) local;
* :func:`annulus_mesh` builds the annulus from two counter-clockwise polygons and
  marks the inner one as a hole, which negates its normals: the inner loop's
  domain-outward normal then points into the hole (``-r``).  ``PanelMesh`` can
  only store counter-clockwise polygons, so the orientation convention lives in
  :class:`MultiLoopMesh` -- the difference matters, see the negative control in
  ``tests/test_multiloop_annulus.py``;
* :func:`solve_laplace_multiloop` reuses the existing analytically integrated
  single- and double-layer assembly (:func:`tensorbem.solver.assemble_matrices`)
  and the already validated direct formulation ``S q = (1/2 I + H) g``, so no
  new kernel is introduced and the annular result is not a second, separately
  trusted code path;
* :func:`evaluate_multiloop_field` evaluates the Green representation at points
  of the annular domain.  The representation is a sum over *all* panels, so a
  target that lies inside the outer loop and outside the inner loop is handled
  by the same formula with no topological bookkeeping.

Conventions match the rest of the package: lengths in metres, ``G =
-(1/(2*pi)) log(|x-y|/L)`` with the package reference length ``L``, constant
panels, midpoint collocation, ``c = 1/2``.  The returned ``normal_derivative``
is ``du/dn`` along the **domain-outward** normal, so on the inner loop of an
annulus it is the derivative pointing *into the hole* (i.e. ``-du/dr``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh, _validate_length, _validate_panel_count
from .kernels import REFERENCE_LENGTH, validate_reference_length
from .solver import (
    LaplaceDirichletResult,
    _MAX_CONDITION,
    assemble_matrices,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "MultiLoopMesh",
    "annulus_mesh",
    "solve_laplace_multiloop",
    "evaluate_multiloop_field",
    "domain_membership",
    "boundary_flux",
]


@dataclass(frozen=True)
class MultiLoopMesh:
    """A set of closed CCW loops that together bound one domain.

    Each loop is a :class:`tensorbem.geometry.PanelMesh`, i.e. a
    counter-clockwise polygon whose normals point *away* from the loop's
    interior.  For a hole, that is exactly the domain-outward normal of the
    surrounding material, which is why the inner loop is stored as a CCW polygon
    around the hole (a clockwise run around the outer domain).
    """

    loops: tuple[PanelMesh, ...]
    hole_loops: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        loops = tuple(self.loops)
        if len(loops) < 2:
            raise ValueError("a multi-loop mesh requires at least two loops")
        for loop in loops:
            if not isinstance(loop, PanelMesh):
                raise TypeError("every loop must be a PanelMesh")
        holes = tuple(int(index) for index in self.hole_loops)
        if len(set(holes)) != len(holes):
            raise ValueError("hole_loops must not repeat an index")
        for index in holes:
            if not 0 <= index < len(loops):
                raise ValueError("hole_loops index out of range")
        object.__setattr__(self, "loops", loops)
        object.__setattr__(self, "hole_loops", holes)

    # -- per-loop fan-out -------------------------------------------------
    @property
    def panel_count(self) -> int:
        return int(sum(loop.panel_count for loop in self.loops))

    @property
    def vertices(self) -> FloatArray:
        return np.vstack([loop.vertices for loop in self.loops])

    @property
    def ends(self) -> FloatArray:
        # roll each loop separately so no spurious segment joins two loops
        return np.vstack([loop.ends for loop in self.loops])

    @property
    def vectors(self) -> FloatArray:
        return self.ends - self.vertices

    @property
    def lengths(self) -> FloatArray:
        return np.concatenate([loop.lengths for loop in self.loops])

    @property
    def centers(self) -> FloatArray:
        return np.vstack([loop.centers for loop in self.loops])

    @property
    def normals(self) -> FloatArray:
        """Domain-outward normals: negated on hole loops (see module docstring)."""
        parts = []
        for index, loop in enumerate(self.loops):
            normal = loop.normals
            if index in self.hole_loops:
                normal = -normal
            parts.append(normal)
        return np.vstack(parts)

    @property
    def perimeter(self) -> float:
        return float(sum(loop.perimeter for loop in self.loops))

    @property
    def loop_slices(self) -> tuple[slice, ...]:
        """Panel index slice of each loop, in ``loops`` order."""
        slices: list[slice] = []
        start = 0
        for loop in self.loops:
            stop = start + loop.panel_count
            slices.append(slice(start, stop))
            start = stop
        return tuple(slices)


def annulus_mesh(
    outer_panels: int = 64,
    inner_panels: int = 32,
    outer_radius: float = 1.0,
    inner_radius: float = 0.5,
) -> MultiLoopMesh:
    """Return the annulus ``inner_radius < r < outer_radius`` as two loops.

    The outer loop is a CCW inscribed polygon at ``outer_radius``; the inner
    loop is a CCW inscribed polygon at ``inner_radius`` (hence a clockwise run
    around the *material*), so its normals point into the hole.
    """
    outer_panels = _validate_panel_count(outer_panels)
    inner_panels = _validate_panel_count(inner_panels)
    outer_radius = _validate_length("outer_radius", outer_radius)
    inner_radius = _validate_length("inner_radius", inner_radius)
    if not inner_radius < outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")
    theta_o = 2.0 * np.pi * np.arange(outer_panels) / outer_panels
    theta_i = 2.0 * np.pi * np.arange(inner_panels) / inner_panels
    outer = PanelMesh(outer_radius * np.column_stack((np.cos(theta_o), np.sin(theta_o))))
    inner = PanelMesh(inner_radius * np.column_stack((np.cos(theta_i), np.sin(theta_i))))
    return MultiLoopMesh(loops=(outer, inner), hole_loops=(1,))


def solve_laplace_multiloop(
    mesh: MultiLoopMesh,
    boundary_values: FloatArray,
    reference_length: float = REFERENCE_LENGTH,
) -> LaplaceDirichletResult:
    """Solve ``Laplace(u)=0`` in a multiply connected domain, Dirichlet data.

    Unknown: ``du/dn`` along the domain-outward normal on every loop.  Uses the
    same direct collocation formulation and the same analytically integrated
    matrices as :func:`tensorbem.solver.solve_laplace_dirichlet`; the only new
    ingredient is that the boundary trace is assembled from several loops.
    """
    if not isinstance(mesh, MultiLoopMesh):
        raise TypeError("mesh must be a MultiLoopMesh")
    values = np.asarray(boundary_values, dtype=float)
    n = mesh.panel_count
    if values.shape != (n,):
        raise ValueError(f"boundary_values must have shape ({n},)")
    if not np.isfinite(values).all():
        raise ValueError("boundary_values must all be finite")
    reference = validate_reference_length(reference_length)

    single, double = assemble_matrices(mesh, reference_length=reference)  # type: ignore[arg-type]
    rhs = 0.5 * values + double @ values
    condition = float(np.linalg.cond(single))
    if not math.isfinite(condition) or condition > _MAX_CONDITION:
        raise ValueError(
            "the single-layer matrix is numerically singular "
            f"(condition number {condition:.3e}); in 2-D this happens when a loop's "
            "logarithmic capacity approaches reference_length "
            f"(currently {reference:g}). Pass a different reference_length."
        )
    q = np.linalg.solve(single, rhs)
    residual = float(np.linalg.norm(single @ q - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    return LaplaceDirichletResult(
        normal_derivative=q,
        boundary_values=values.copy(),
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_number=condition,
        residual_l2=residual,
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0.0 else residual,
        domain="interior",
        reference_length=reference,
    )


def _point_in_polygon(points: FloatArray, polygon: FloatArray) -> NDArray[np.bool_]:
    """Even-odd ray casting: True where a point lies inside the closed polygon."""
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        crosses = (y1 > y) != (y2 > y)
        if not bool(crosses.any()):
            continue
        x_at = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < x_at)
    return inside


def domain_membership(mesh: "MultiLoopMesh", points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Where targets lie in a hole loop and where they lie outside the outer loop.

    Green's representation summed over *all* loops is only the physical field in
    the annular region.  Inside a hole the same sum evaluates the analytic
    continuation (the ``c`` indicator functions no longer cancel), and outside the
    outer loop it is neither; both would look plausible and be wrong, so the
    evaluation below refuses them instead of returning a number.
    """
    targets = np.asarray(points, dtype=float)
    in_hole = np.zeros(len(targets), dtype=bool)
    for index in mesh.hole_loops:
        in_hole |= _point_in_polygon(targets, mesh.loops[index].vertices)
    outside = ~_point_in_polygon(targets, mesh.loops[0].vertices)
    return in_hole, outside


def evaluate_multiloop_field(
    mesh: MultiLoopMesh, result: LaplaceDirichletResult, points: FloatArray
) -> FloatArray:
    """Evaluate the potential at points of the multiply connected domain.

    Green's representation is summed over every panel of every loop.  A target
    inside the outer loop and outside the hole contributes with ``c = 1``, which
    is exactly what the plain sum of layer potentials gives (the indicator
    functions of the two loops cancel in the annular region).
    """
    if not isinstance(mesh, MultiLoopMesh):
        raise TypeError("mesh must be a MultiLoopMesh")
    if not isinstance(result, LaplaceDirichletResult):
        raise TypeError("result must be a LaplaceDirichletResult")
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")
    if result.normal_derivative.shape != (mesh.panel_count,):
        raise ValueError("result boundary traces must match the mesh panel count")

    in_hole, outside = domain_membership(mesh, targets)
    if bool(np.any(in_hole)):
        raise ValueError(
            "points must lie in the fluid domain: "
            f"{int(np.sum(in_hole))} target(s) fall inside a hole loop, where the "
            "Green representation evaluates the analytic continuation and not the "
            "field of the problem"
        )
    if bool(np.any(outside)):
        raise ValueError(
            "points must lie in the fluid domain: "
            f"{int(np.sum(outside))} target(s) fall outside the outer loop"
        )

    from .kernels import integrate_double_layer_panel, integrate_single_layer_panel

    lengths = mesh.lengths
    vertices = mesh.vertices
    tangents = mesh.vectors / lengths[:, None]
    normals = mesh.normals
    reference = result.reference_length
    values = np.zeros(len(targets), dtype=float)
    for i, target in enumerate(targets):
        total = 0.0
        for j in range(mesh.panel_count):
            total += (
                integrate_single_layer_panel(
                    target, vertices[j], tangents[j], float(lengths[j]), reference
                )
                * result.normal_derivative[j]
            )
            total -= (
                integrate_double_layer_panel(
                    target,
                    vertices[j],
                    tangents[j],
                    normals[j],
                    float(lengths[j]),
                )
                * result.boundary_values[j]
            )
        values[i] = total
    return values


def boundary_flux(
    mesh: MultiLoopMesh, normal_derivative: FloatArray
) -> tuple[float, ...]:
    """Return ``∮ q ds`` over each loop, in loops order.

    On a multiply connected domain the outer loop's flux equals minus the sum of
    the holes' fluxes whenever the field is harmonic and the boundary data is
    compatible, which is the discrete counterpart of ``∮ q ds = 0`` over the
    whole boundary.
    """
    q = np.asarray(normal_derivative, dtype=float)
    if q.shape != (mesh.panel_count,):
        raise ValueError(f"normal_derivative must have shape ({mesh.panel_count},)")
    lengths = mesh.lengths
    return tuple(
        float(np.sum(q[sl] * lengths[sl])) for sl in mesh.loop_slices
    )