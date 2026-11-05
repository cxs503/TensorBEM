"""Multi-component (multi-connected) 2-D Laplace domains.

Every solver entry point in :mod:`tensorbem.solver` is duck-typed on the *panel
arrays* of :class:`tensorbem.geometry.PanelMesh`, but ``PanelMesh`` itself only
accepts a **single** counter-clockwise polygon: it stores ``vertices`` as one
cyclic array, so the closing segment ``N-1 -> 0`` is implicit and a second loop
cannot be appended without inventing a spurious edge between the two loops.

That is the gap registered in ``docs/verification.md`` as

    | 圆环径向导热，u = A log r + B | 多连通边界、内外法向 | 后续：需多边界组件 |

This module closes it on the geometry side only:

* :class:`MultiComponentMesh` holds several closed loops, keeps each loop's
  cyclic closure local to that loop, and exposes exactly the panel arrays the
  assembler consumes (``vertices``/``ends``/``vectors``/``lengths``/``centers``/
  ``normals``/``panel_count``).  The normals are the *fluid-domain outward*
  normals: on a hole loop they point into the hole, which is what the interior
  boundary integral equation requires.
* :func:`build_multicomponent_mesh` takes the outer loop and any holes as
  ordinary CCW ``PanelMesh`` polygons and reverses the hole traversal, so a
  caller never has to think about orientation by hand.
* :func:`solve_multicomponent_dirichlet` / :func:`evaluate_fluid_field` reuse
  the existing collocation solver and field evaluator unchanged, but validate
  that field points really lie in the fluid region (outside every hole).  A
  silent evaluation inside a hole would look plausible and be wrong, which is
  exactly the failure mode this guard removes.

No solver code is copied: the analytic single- and double-layer panel integrals,
the first-kind Dirichlet operator and the condition-number guard are the ones
already tested by ``tests/test_solver.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh, _validate_length, _validate_panel_count
from .kernels import REFERENCE_LENGTH
from .solver import (
    LaplaceDirichletResult,
    evaluate_interior_field,
    solve_laplace_dirichlet,
)

FloatArray = NDArray[np.float64]


def _loop_arrays(vertices: FloatArray) -> dict[str, FloatArray]:
    """Panel arrays of one closed loop, closure kept inside the loop."""
    points = np.asarray(vertices, dtype=float)
    ends = np.roll(points, -1, axis=0)
    vectors = ends - points
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors / lengths[:, None]
    signed_area = 0.5 * float(
        np.sum(points[:, 0] * ends[:, 1] - points[:, 1] * ends[:, 0])
    )
    return {
        "vertices": points,
        "ends": ends,
        "vectors": vectors,
        "lengths": lengths,
        "centers": 0.5 * (points + ends),
        "normals": np.column_stack((tangents[:, 1], -tangents[:, 0])),
        "signed_area": np.array(signed_area),
    }


def _point_in_polygon(points: FloatArray, polygon: FloatArray) -> NDArray[np.bool_]:
    """Even-odd ray casting (vectorised over ``points``, loop over edges)."""
    inside = np.zeros(len(points), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Count crossings of the +x ray; the (y1 > y) != (y2 > y) test keeps
        # vertices from being counted twice.
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < x_at)
    return inside


@dataclass(frozen=True)
class MultiComponentMesh:
    """Panels of several closed loops whose normals point out of the fluid.

    ``loop_vertices`` are already oriented for the interior boundary integral
    equation: the outer loop runs counter-clockwise (normals outward), each hole
    loop runs clockwise (normals into the hole, i.e. out of the fluid).  Use
    :func:`build_multicomponent_mesh` instead of constructing this directly.
    """

    loop_vertices: tuple[FloatArray, ...]
    loop_labels: tuple[str, ...] = ()
    reference_length: float = REFERENCE_LENGTH
    enforce_orientation: bool = True
    _arrays: dict[str, FloatArray] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        loops = tuple(np.asarray(loop, dtype=float) for loop in self.loop_vertices)
        if not loops:
            raise ValueError("at least one boundary loop is required")
        labels = self.loop_labels or tuple(
            "outer" if i == 0 else f"hole{i}" for i in range(len(loops))
        )
        if len(labels) != len(loops):
            raise ValueError("loop_labels must match the number of loops")
        reference = _validate_length("reference_length", self.reference_length)

        collected: dict[str, list[FloatArray]] = {
            key: [] for key in ("vertices", "ends", "vectors", "lengths", "centers", "normals")
        }
        slices: list[slice] = []
        signed_areas: list[float] = []
        offset = 0
        for label, loop in zip(labels, loops):
            if loop.ndim != 2 or loop.shape[1] != 2:
                raise ValueError(f"loop {label!r} must have shape (N, 2)")
            _validate_panel_count(len(loop))
            if not np.isfinite(loop).all():
                raise ValueError(f"loop {label!r} must be finite")
            arrays = _loop_arrays(loop)
            if np.any(arrays["lengths"] <= 1.0e-14):
                raise ValueError(f"loop {label!r} has a zero-length panel")
            if abs(float(arrays["signed_area"])) <= 1.0e-12:
                raise ValueError(f"loop {label!r} is degenerate (zero signed area)")
            for key in collected:
                collected[key].append(arrays[key])
            slices.append(slice(offset, offset + len(loop)))
            signed_areas.append(float(arrays["signed_area"]))
            offset += len(loop)

        if self.enforce_orientation and signed_areas[0] <= 0.0:
            raise ValueError(
                "the first (outer) loop must be counter-clockwise so its normals "
                "point out of the fluid"
            )
        for label, area in zip(labels[1:], signed_areas[1:]):
            if self.enforce_orientation and area >= 0.0:
                raise ValueError(
                    f"hole loop {label!r} must be clockwise (build it with "
                    "build_multicomponent_mesh, which reverses the hole traversal)"
                )

        object.__setattr__(self, "loop_vertices", loops)
        object.__setattr__(self, "loop_labels", tuple(labels))
        object.__setattr__(self, "reference_length", reference)
        object.__setattr__(
            self,
            "_arrays",
            {
                **{key: np.concatenate(parts) for key, parts in collected.items()},
                "loop_slices": tuple(slices),
                "signed_areas": np.asarray(signed_areas),
            },
        )

    # -- arrays consumed by the assembler and the field evaluator ---------
    @property
    def vertices(self) -> FloatArray:
        return self._arrays["vertices"]

    @property
    def ends(self) -> FloatArray:
        return self._arrays["ends"]

    @property
    def vectors(self) -> FloatArray:
        return self._arrays["vectors"]

    @property
    def lengths(self) -> FloatArray:
        return self._arrays["lengths"]

    @property
    def centers(self) -> FloatArray:
        return self._arrays["centers"]

    @property
    def normals(self) -> FloatArray:
        return self._arrays["normals"]

    @property
    def panel_count(self) -> int:
        return int(len(self._arrays["vertices"]))

    # -- multi-component specific ----------------------------------------
    @property
    def loop_slices(self) -> tuple[slice, ...]:
        return tuple(self._arrays["loop_slices"])

    @property
    def outer(self) -> FloatArray:
        return self.loop_vertices[0]

    @property
    def holes(self) -> tuple[FloatArray, ...]:
        return tuple(self.loop_vertices[1:])

    @property
    def panel_counts_per_loop(self) -> tuple[int, ...]:
        return tuple(len(loop) for loop in self.loop_vertices)

    @property
    def perimeter(self) -> float:
        return float(np.sum(self._arrays["lengths"]))

    @property
    def fluid_area(self) -> float:
        """Signed sum of the loop areas: outer CCW minus holes (CW)."""
        return float(np.sum(self._arrays["signed_areas"]))

    def fluid_contains(self, points: FloatArray) -> NDArray[np.bool_]:
        """True where a point lies inside the outer loop and outside every hole."""
        targets = np.asarray(points, dtype=float)
        inside = _point_in_polygon(targets, self.loop_vertices[0])
        for hole in self.holes:
            inside &= ~_point_in_polygon(targets, hole)
        return inside

    def contains_point(self, points: FloatArray) -> NDArray[np.bool_]:
        """:meth:`fluid_contains` under the name the field evaluator expects.

        :func:`tensorbem.solver.evaluate_interior_field` screens targets with
        ``mesh.contains_point``; for a mesh with holes the fluid region is the
        outer polygon *minus* the holes, so delegating here keeps "inside the
        domain" meaning the same thing to the solver and to
        :func:`evaluate_fluid_field`.
        """
        return self.fluid_contains(points)

    def boundary_clearance(self, points: FloatArray) -> FloatArray:
        """Minimum Euclidean distance from each point to any wetted panel.

        Same segment-projection definition as
        :meth:`tensorbem.geometry.PanelMesh.boundary_clearance`, applied to all
        loops at once: a target next to a hole is exactly as close to the
        boundary as one next to the outer loop, which is what the clearance
        guard in :func:`tensorbem.solver.evaluate_interior_field` assumes.
        """
        targets = np.asarray(points, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("points must have shape (M, 2)")
        if not np.isfinite(targets).all():
            raise ValueError("points must all be finite")
        clearance = np.full(len(targets), np.inf, dtype=float)
        lengths_squared = np.sum(self.vectors**2, axis=1)
        for start, segment, length_squared in zip(
            self.vertices, self.vectors, lengths_squared
        ):
            projection = np.clip(
                (targets - start) @ segment / length_squared, 0.0, 1.0
            )
            closest = start + projection[:, None] * segment
            clearance = np.minimum(
                clearance, np.linalg.norm(targets - closest, axis=1)
            )
        return clearance


def build_multicomponent_mesh(
    outer: PanelMesh,
    holes: Sequence[PanelMesh] = (),
    reference_length: float = REFERENCE_LENGTH,
) -> MultiComponentMesh:
    """Assemble an outer CCW polygon plus CCW hole polygons.

    ``outer`` and each hole are ordinary :class:`PanelMesh` objects, i.e. given
    counter-clockwise as *solids*.  The builder reverses every hole so that the
    resulting panel normal points from the fluid into the hole.
    """
    if not isinstance(outer, PanelMesh):
        raise TypeError("outer must be a PanelMesh")
    loops = [outer.vertices]
    labels = ["outer"]
    for index, hole in enumerate(holes, start=1):
        if not isinstance(hole, PanelMesh):
            raise TypeError("holes must be PanelMesh instances")
        loops.append(hole.vertices[::-1])
        labels.append(f"hole{index}")
    mesh = MultiComponentMesh(
        loop_vertices=tuple(loops),
        loop_labels=tuple(labels),
        reference_length=reference_length,
    )
    # Every hole vertex must lie strictly inside the outer loop, and no hole may
    # swallow another hole: a nested or overlapping hole is a caller error, not
    # a number that should be silently integrated over.
    for label, hole in zip(labels[1:], mesh.holes):
        if not bool(_point_in_polygon(hole, outer.vertices).all()):
            raise ValueError(f"hole loop {label!r} leaves the outer loop")
    for i, hole_a in enumerate(mesh.holes):
        for hole_b in mesh.holes[i + 1 :]:
            if bool(_point_in_polygon(hole_a, hole_b).any()) or bool(
                _point_in_polygon(hole_b, hole_a).any()
            ):
                raise ValueError("hole loops must not overlap or nest")
    return mesh


def annulus_mesh(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    panels_per_loop: int = 64,
    reference_length: float = REFERENCE_LENGTH,
) -> MultiComponentMesh:
    """Annulus ``r_inner <= r <= r_outer`` built from two inscribed polygons."""
    inner = _validate_length("inner_radius", inner_radius)
    outer = _validate_length("outer_radius", outer_radius)
    if not inner < outer:
        raise ValueError("inner_radius must be smaller than outer_radius")
    count = _validate_panel_count(panels_per_loop)
    return build_multicomponent_mesh(
        outer=_circle_polygon(outer, count),
        holes=(_circle_polygon(inner, count),),
        reference_length=reference_length,
    )


def _circle_polygon(radius: float, panel_count: int) -> PanelMesh:
    theta = 2.0 * np.pi * np.arange(panel_count) / panel_count
    return PanelMesh(radius * np.column_stack((np.cos(theta), np.sin(theta))))


# ---------------------------------------------------------------------------
# Analytic fields, expressed through a complex potential
# ---------------------------------------------------------------------------
def _as_complex(points: FloatArray) -> NDArray[np.complex128]:
    """``array of (x, y)`` -> ``x + iy``; a plain 1-D complex array passes through."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    return array[:, 0] + 1j * array[:, 1]


@dataclass(frozen=True)
class AnalyticField:
    """Harmonic field ``u = Re f(z)`` with gradient ``(Re f', -Im f')``."""

    name: str
    potential: Callable[[NDArray[np.complex128]], NDArray[np.float64]]
    derivative: Callable[[NDArray[np.complex128]], NDArray[np.complex128]]

    def values(self, points: FloatArray) -> FloatArray:
        return np.real(self.potential(_as_complex(points)))

    def gradient(self, points: FloatArray) -> FloatArray:
        dz = self.derivative(_as_complex(points))
        return np.column_stack((np.real(dz), -np.imag(dz)))

    def normal_derivative(self, mesh: MultiComponentMesh) -> FloatArray:
        """``grad(u) . n`` with ``n`` the fluid-outward panel normal."""
        return np.sum(self.gradient(mesh.centers) * mesh.normals, axis=1)


def log_field(amplitude: float = 1.0, offset: float = 0.0) -> AnalyticField:
    """``u = A log r + B`` -- the annulus radial-conduction reference."""
    a = _validate_length("amplitude", abs(amplitude)) * math.copysign(1.0, amplitude)

    def potential(z: NDArray[np.complex128]) -> NDArray[np.float64]:
        return a * np.log(z) + offset

    def derivative(z: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return a / z

    return AnalyticField(name=f"log(A={a:g},B={offset:g})", potential=potential, derivative=derivative)


def power_field(degree: int, inner_weight: float = 1.0, outer_weight: float = 1.0) -> AnalyticField:
    """``u = Re(c z^n + d z^-n)``, ``n >= 1`` -- angular harmonics in an annulus.

    ``z^-n`` is harmonic and single valued in the annulus but singular at the
    origin, which is *outside* the fluid; the pair therefore probes the hole
    boundary as strongly as the outer boundary.
    """
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    n = int(degree)
    c = float(outer_weight)
    d = float(inner_weight)
    if c == 0.0 and d == 0.0:
        raise ValueError("at least one of outer_weight/inner_weight must be non-zero")

    def potential(z: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return c * z**n + d * z ** (-n)

    def derivative(z: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return n * c * z ** (n - 1) - n * d * z ** (-n - 1)

    return AnalyticField(
        name=f"power(n={n},c={c:g},d={d:g})", potential=potential, derivative=derivative
    )


# ---------------------------------------------------------------------------
# Solve / evaluate with the multi-component guards
# ---------------------------------------------------------------------------
def solve_multicomponent_dirichlet(
    mesh: MultiComponentMesh,
    boundary_values: FloatArray,
    reference_length: float | None = None,
) -> LaplaceDirichletResult:
    """Interior Dirichlet solve on a multi-connected domain.

    Thin wrapper over :func:`tensorbem.solver.solve_laplace_dirichlet`: the
    multi-component mesh already carries fluid-outward normals, which is the
    only thing the single-loop solver needed.
    """
    if not isinstance(mesh, MultiComponentMesh):
        raise TypeError("mesh must be a MultiComponentMesh")
    reference = mesh.reference_length if reference_length is None else reference_length
    return solve_laplace_dirichlet(mesh, boundary_values, reference_length=reference)


def evaluate_fluid_field(
    mesh: MultiComponentMesh,
    result: LaplaceDirichletResult,
    points: FloatArray,
) -> FloatArray:
    """Field evaluation that refuses points outside the fluid region.

    ``evaluate_interior_field`` cannot know about holes; a probe accidentally
    placed inside a hole would return a finite, plausible and wrong number.
    """
    if not isinstance(mesh, MultiComponentMesh):
        raise TypeError("mesh must be a MultiComponentMesh")
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    inside = mesh.fluid_contains(targets)
    if not inside.all():
        raise ValueError(
            f"{int(np.count_nonzero(~inside))} of {len(targets)} points lie outside "
            "the fluid region (inside a hole or outside the outer loop)"
        )
    return evaluate_interior_field(mesh, result, targets)


# ---------------------------------------------------------------------------
# Verification study
# ---------------------------------------------------------------------------
def _probes(inner_radius: float, outer_radius: float, count_per_ring: int = 6) -> FloatArray:
    rings = (0.15, 0.5, 0.85)
    points = []
    for ring, fraction in enumerate(rings):
        radius = inner_radius + fraction * (outer_radius - inner_radius)
        for k in range(count_per_ring):
            angle = 2.0 * np.pi * (k + 0.5 * ring) / count_per_ring
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
    return np.asarray(points)


def run_annulus_case(
    field_spec: Callable[[], AnalyticField],
    panels_per_loop: int,
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    reference_length: float = REFERENCE_LENGTH,
    probes: FloatArray | None = None,
) -> dict[str, object]:
    """One (case, mesh) evaluation of the annulus Dirichlet problem."""
    field = field_spec()
    mesh = annulus_mesh(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        panels_per_loop=panels_per_loop,
        reference_length=reference_length,
    )
    values = field.values(mesh.centers)
    result = solve_multicomponent_dirichlet(mesh, values)
    q_bem = result.normal_derivative
    q_ref = field.normal_derivative(mesh)
    lengths = mesh.lengths

    denominator = float(np.sum(lengths * q_ref**2))
    flux_rel_l2 = float(
        math.sqrt(float(np.sum(lengths * (q_bem - q_ref) ** 2)) / denominator)
    )
    scale = float(np.max(np.abs(q_ref)))
    flux_max_abs = float(np.max(np.abs(q_bem - q_ref)))
    flux_max_rel = flux_max_abs / scale if scale > 0.0 else flux_max_abs

    point_targets = _probes(inner_radius, outer_radius) if probes is None else probes
    u_ref = field.values(point_targets)
    u_bem = evaluate_fluid_field(mesh, result, point_targets)
    u_scale = float(np.max(np.abs(u_ref))) if len(u_ref) else 1.0
    field_rmse = float(np.sqrt(np.mean((u_bem - u_ref) ** 2)))
    field_max_rel = float(np.max(np.abs(u_bem - u_ref))) / u_scale if u_scale > 0.0 else 0.0

    # Independent of the analytic field: constant panels on a closed loop give
    # an exactly computable per-loop flux, and the total must vanish because no
    # source sits in the fluid region.
    per_loop_flux = [
        float(np.sum(q_bem[sl] * lengths[sl])) for sl in mesh.loop_slices
    ]
    per_loop_flux_ref = [
        float(np.sum(q_ref[sl] * lengths[sl])) for sl in mesh.loop_slices
    ]
    total_flux = float(np.sum(q_bem * lengths))
    total_flux_scale = float(np.sum(np.abs(q_ref) * lengths))

    return {
        "case": field.name,
        "panels_per_loop": int(panels_per_loop),
        "panels_total": int(mesh.panel_count),
        "flow_area": mesh.fluid_area,
        "polygon_area": float(np.pi * (outer_radius**2 - inner_radius**2)),
        "flux_rel_l2": flux_rel_l2,
        "flux_max_rel": flux_max_rel,
        "field_rmse": field_rmse,
        "field_max_rel": field_max_rel,
        "condition_number": float(result.condition_number),
        "relative_residual_l2": float(result.relative_residual_l2),
        "net_flux": total_flux,
        "net_flux_reference": float(np.sum(per_loop_flux_ref)),
        "net_flux_relative": total_flux / total_flux_scale if total_flux_scale > 0.0 else total_flux,
        "per_loop_flux": per_loop_flux,
        "per_loop_flux_reference": per_loop_flux_ref,
        "probe_points": point_targets.tolist(),
        "probe_values_bem": u_bem.tolist(),
        "probe_values_reference": u_ref.tolist(),
    }


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Two-grid observed order ``p = log(e_coarse/e_fine)/log(ratio)``."""
    if fine <= 0.0 or coarse <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(ratio))


def run_wrong_orientation_control(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    panels_per_loop: int = 64,
) -> dict[str, float]:
    """Same annulus, hole normals pointing *into* the fluid (wrong sign)."""
    field = log_field(amplitude=1.0, offset=0.5)
    good = annulus_mesh(inner_radius, outer_radius, panels_per_loop)
    wrong_mesh = _hole_normal_flipped(
        good.loop_vertices, inner_radius, outer_radius
    )

    values = field.values(wrong_mesh.centers)
    result = solve_multicomponent_dirichlet(wrong_mesh, values)
    q_bem = result.normal_derivative
    q_ref = field.normal_derivative(wrong_mesh)
    lengths = wrong_mesh.lengths
    denominator = float(np.sum(lengths * q_ref**2))
    flux_rel_l2 = float(
        math.sqrt(float(np.sum(lengths * (q_bem - q_ref) ** 2)) / denominator)
    )
    return {
        "panels_per_loop": int(panels_per_loop),
        "flux_rel_l2": flux_rel_l2,
        "condition_number": float(result.condition_number),
        "relative_residual_l2": float(result.relative_residual_l2),
    }


def _hole_normal_flipped(
    loops: Sequence[FloatArray], inner_radius: float, outer_radius: float
) -> MultiComponentMesh:
    """Mesh whose hole normals point *into* the fluid: the deliberate control.

    ``enforce_orientation=False`` is the documented escape hatch that lets the
    negative control exist at all; nothing in the verification path uses it.
    """
    outer = _circle_polygon(outer_radius, len(loops[0]))
    hole = _circle_polygon(inner_radius, len(loops[1]))
    return MultiComponentMesh(
        loop_vertices=(outer.vertices, hole.vertices),
        loop_labels=("outer", "hole_wrong_orientation"),
        reference_length=REFERENCE_LENGTH,
        enforce_orientation=False,
    )