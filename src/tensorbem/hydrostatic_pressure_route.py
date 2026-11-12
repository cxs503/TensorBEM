"""Hydrostatic restoring matrix by direct pressure integration (second route).

Why this module exists
----------------------
:func:`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` obtains the
6x6 hydrostatic restoring matrix ``C`` from *waterplane integrals*: the wetted
waterline loops are traced, ``A``, ``S_x``, ``S_y``, ``I_xx``, ``I_yy``,
``I_xy`` are integrated over the plane ``z = 0``, and the entries follow the
closed-form hydrostatic formulae (Journee & Massie, *Offshore Hydromechanics*,
section 3.2.2).  That is one route only, so an error in it cannot be attributed:
a wrong loop trace, a wrong waterplane moment and a wrong formula look the same
from the outside, and ``C`` feeds every GM and RAO number downstream.

This module implements a *structurally different* route for exactly the same
quantity.  It never touches the waterplane:

1. start from the **closed** body surface (no truncation at the waterline),
2. move the body rigidly by a motion vector ``xi`` (6 components),
3. clip the moved triangles against the calm free surface ``z = 0``,
4. integrate the hydrostatic pressure field ``p = rho*g*(z_fs - z)`` over the
   clipped surface to get the resultant force and moment,
5. differentiate the resultant load with respect to ``xi`` by central
   differences: ``C = -d(load)/d(xi)``.

Because a linear pressure field integrated with the centroid rule is exact on
planar triangles, the only approximations left are the facet discretisation of
the body and the ``O(step**2)`` truncation of the central difference.  For a
body whose wetted surface is exactly representable by its facets (a rectangular
barge) the two routes must therefore agree to round-off apart from the
truncation term, which is the sharpest available falsification test of the
waterplane formulae; for a curved body (spheroid) the two routes must converge
to each other under mesh refinement.

Conventions (identical to :mod:`tensorbem.hydrostatic`)
-------------------------------------------------------
* ``z`` up, calm free surface ``z = 0``, fluid in ``z <= 0``.
* Motion vector ``xi = (xi1, xi2, xi3, xi4, xi5, xi6)`` =
  ``(surge, sway, heave, roll, pitch, yaw)``.  Translations are in metres,
  rotations in radians, right-handed about ``+x``, ``+y``, ``+z``.
* The rotation is applied about ``rotation_center`` (default: the coordinate
  origin, which is the reference point of the ``C`` matrix), then the
  translation is added.  The finite rotation matrix
  ``R = Rz(xi6) @ Ry(xi5) @ Rx(xi4)`` is used, not a small-angle expansion, so
  the central difference is the only linearisation in the route.
* Outward panel normals point from the body into the fluid.  The load on the
  body from pressure is ``F = -int p n dS`` and ``M = -int p (r x n) dS`` with
  ``r`` measured from the coordinate origin, so a submerged closed body feels
  ``F_z = +rho*g*Nabla`` (Archimedes) and ``M = rho*g*Nabla*(y_B, -x_B, 0)``.
* ``C = -d(load)/d(xi)`` with ``load`` the *total* load (hydrostatic pressure
  plus the constant body weight acting at the centre of gravity, whose moment
  changes as the body rotates).  This is the matrix that multiplies ``xi`` in
  the frequency-domain equation ``[-w**2 (M + A) + i w B + C] xi = F``; the
  weight term is what makes the sway-roll and surge-pitch hydrostatic entries
  ``rho*g*Nabla`` cancel, exactly as in the waterplane route.

Limitations
-----------
* Linear (Airy) hydrostatics: ``p = rho*g*(z_fs - z)``, no wave elevation, no
  dynamic pressure.  The pressure *route* is linear in the pressure field but
  exact in the rigid-body rotation; only the differentiation is linearised.
* The body must be a closed triangle mesh that pierces ``z = 0``.  A body that
  is fully submerged at equilibrium has no waterplane and no restoring matrix.
* The route differentiates a discrete load, so it is a verification tool, not a
  cheaper replacement for the waterplane route: it costs ``12`` clippings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry3d import WettedHullMesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665

DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _as_vertices(vertices: ArrayLike) -> FloatArray:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if not np.isfinite(verts).all():
        raise ValueError("vertices must all be finite")
    return verts


def _as_faces(faces: ArrayLike, vertex_count: int) -> IntArray:
    tris = np.asarray(faces)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if tris.size == 0:
        raise ValueError("a closed body surface requires at least one face")
    if not np.issubdtype(tris.dtype, np.integer):
        raise ValueError("faces must be an integer array")
    tris = np.asarray(tris, dtype=np.int64)
    if np.any(tris < 0) or np.any(tris >= vertex_count):
        raise ValueError("face indices must be in range [0, vertex_count)")
    if np.any(
        (tris[:, 0] == tris[:, 1])
        | (tris[:, 1] == tris[:, 2])
        | (tris[:, 2] == tris[:, 0])
    ):
        raise ValueError("each triangle must use three distinct vertices")
    return tris


def _positive_finite(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return out


def _as_motion(motion: ArrayLike) -> FloatArray:
    xi = np.asarray(motion, dtype=float)
    if xi.shape != (6,):
        raise ValueError("motion must have shape (6,)")
    if not np.isfinite(xi).all():
        raise ValueError("motion must all be finite")
    return xi


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> FloatArray:
    """``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (right-handed, ``z`` up)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def rigid_transform(
    vertices: ArrayLike,
    motion: ArrayLike,
    *,
    rotation_center: Sequence[float] = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Move ``vertices`` by the rigid-body ``motion`` (see module docstring)."""
    verts = _as_vertices(vertices)
    xi = _as_motion(motion)
    center = np.asarray(rotation_center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("rotation_center must be a finite 3-tuple")
    rot = _rotation_matrix(xi[3], xi[4], xi[5])
    return (verts - center[None, :]) @ rot.T + center[None, :] + xi[None, :3]


# ---------------------------------------------------------------------------
# clipping
# ---------------------------------------------------------------------------
def clip_to_fluid(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    free_surface_z: float = 0.0,
) -> tuple[FloatArray, IntArray]:
    """Clip a closed surface against ``z <= free_surface_z``.

    Triangles entirely above the plane are dropped, triangles entirely below are
    kept, and crossing triangles are cut by the Sutherland-Hodgman rule and
    re-triangulated as a fan.  Vertex order is preserved, so outward normals and
    the surface orientation survive the clip.  Cut points on a shared edge are
    computed from the same edge parametrisation, so neighbouring triangles agree
    to round-off and the clipped surface stays watertight along the waterline.
    """
    verts = _as_vertices(vertices)
    tris = _as_faces(faces, verts.shape[0])
    if not math.isfinite(float(free_surface_z)):
        raise ValueError("free_surface_z must be finite")
    plane = float(free_surface_z)

    heights = verts[:, 2] - plane
    tri_heights = heights[tris]
    inside = tri_heights <= 0.0
    fully_inside = inside.all(axis=1)
    fully_outside = (~inside).all(axis=1)

    out_verts: list[FloatArray] = []
    out_faces: list[tuple[int, int, int]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}

    def weld(point: FloatArray) -> int:
        key = (
            round(float(point[0]), 12),
            round(float(point[1]), 12),
            round(float(point[2]), 12),
        )
        index = vertex_map.get(key)
        if index is None:
            index = len(out_verts)
            vertex_map[key] = index
            out_verts.append(np.asarray(point, dtype=float))
        return index

    for index in np.flatnonzero(fully_inside):
        tri = tris[index]
        out_faces.append((weld(verts[tri[0]]), weld(verts[tri[1]]), weld(verts[tri[2]])))

    crossing = np.flatnonzero(~fully_inside & ~fully_outside)
    for index in crossing:
        tri = tris[index]
        polygon: list[FloatArray] = []
        for k in range(3):
            current = verts[tri[k]]
            following = verts[tri[(k + 1) % 3]]
            current_in = current[2] - plane <= 0.0
            following_in = following[2] - plane <= 0.0
            if current_in:
                polygon.append(current)
            if current_in != following_in:
                t = (plane - current[2]) / (following[2] - current[2])
                cut = current + t * (following - current)
                cut = np.array([cut[0], cut[1], plane])
                polygon.append(cut)
        deduped: list[FloatArray] = []
        for point in polygon:
            if deduped and np.allclose(deduped[-1], point, rtol=0.0, atol=1e-13):
                continue
            deduped.append(point)
        if len(deduped) < 3:
            continue
        area_vector = np.zeros(3)
        for k in range(1, len(deduped) - 1):
            area_vector += np.cross(
                deduped[k] - deduped[0], deduped[k + 1] - deduped[0]
            )
        if float(np.linalg.norm(area_vector)) <= 1e-15:
            continue  # degenerate sliver on the waterline
        indices = [weld(point) for point in deduped]
        for k in range(1, len(indices) - 1):
            tri = (indices[0], indices[k], indices[k + 1])
            if len(set(tri)) == 3:  # welding can collapse a sliver onto one vertex
                out_faces.append(tri)

    if not out_faces:
        raise ValueError("the surface has no part below the free surface")
    return (
        np.asarray(out_verts, dtype=float).reshape(-1, 3),
        np.asarray(out_faces, dtype=np.int64).reshape(-1, 3),
    )


def wetted_hull_from_closed(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    free_surface_z: float = 0.0,
) -> WettedHullMesh:
    """Return the :class:`WettedHullMesh` of a closed surface cut at ``z = 0``.

    The point is to run the waterplane route and this pressure route on the
    *same* facets, so that a difference between the two matrices cannot come
    from the mesh.
    """
    clipped_verts, clipped_faces = clip_to_fluid(
        vertices, faces, free_surface_z=free_surface_z
    )
    return WettedHullMesh(clipped_verts, clipped_faces)


# ---------------------------------------------------------------------------
# pressure integration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PressureLoad:
    """Resultant hydrostatic load of a wetted surface, moments about the origin."""

    force: FloatArray
    moment: FloatArray
    wetted_area: float
    immersed_volume: float
    free_surface_z: float = 0.0

    @property
    def load(self) -> FloatArray:
        return np.concatenate([self.force, self.moment])


def _triangle_geometry(vertices: FloatArray, faces: IntArray) -> tuple[FloatArray, FloatArray]:
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    double_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * double_area
    normals = np.zeros_like(cross)
    good = double_area > 0.0
    normals[good] = cross[good] / double_area[good, None]
    centroids = (p0 + p1 + p2) / 3.0
    return areas, normals, centroids


def pressure_profile(
    points: ArrayLike,
    *,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    free_surface_z: float = 0.0,
) -> FloatArray:
    """Gauge hydrostatic pressure ``rho*g*(z_fs - z)`` at ``(N, 3)`` points."""
    pts = _as_vertices(points)
    rho = _positive_finite(density, "density")
    g = _positive_finite(gravity, "gravity")
    return rho * g * (float(free_surface_z) - pts[:, 2])


def pressure_loads(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    free_surface_z: float = 0.0,
) -> PressureLoad:
    """Integrate the hydrostatic pressure field over an already wetted surface.

    ``F = -int p n dS`` and ``M = -int p (r x n) dS`` (``r`` from the coordinate
    origin) with the centroid rule, which is exact for a linear pressure field on
    planar triangles.  The immersed volume is obtained from the same clipped
    surface by the divergence theorem, ``Nabla = (1/3) sum (c . n) A`` — the
    waterplane lid contributes nothing because ``z = 0`` there — so the volume
    used for the weight is *not* taken from the pressure integral.
    """
    verts = _as_vertices(vertices)
    tris = _as_faces(faces, verts.shape[0])
    rho = _positive_finite(density, "density")
    g = _positive_finite(gravity, "gravity")
    plane = float(free_surface_z)

    areas, normals, centroids = _triangle_geometry(verts, tris)
    panel_pressure = rho * g * (plane - centroids[:, 2])
    panel_force = -(panel_pressure * areas)[:, None] * normals
    force = np.sum(panel_force, axis=0)
    moment = np.sum(np.cross(centroids, panel_force), axis=0)
    volume = float(np.sum(np.sum(centroids * normals, axis=1) * areas) / 3.0)
    return PressureLoad(
        force=force,
        moment=moment,
        wetted_area=float(np.sum(areas)),
        immersed_volume=volume,
        free_surface_z=plane,
    )


def immersed_volume(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    free_surface_z: float = 0.0,
) -> float:
    """Immersed volume of a closed body cut by ``z = free_surface_z``."""
    clipped_verts, clipped_faces = clip_to_fluid(
        vertices, faces, free_surface_z=free_surface_z
    )
    areas, normals, centroids = _triangle_geometry(clipped_verts, clipped_faces)
    return float(np.sum(np.sum(centroids * normals, axis=1) * areas) / 3.0)


def _gravity_load(
    center_of_gravity: FloatArray,
    motion: FloatArray,
    weight: float,
    *,
    rotation_center: FloatArray,
) -> FloatArray:
    """Weight load ``(0, 0, -W)`` acting at the *moved* centre of gravity."""
    moved = rigid_transform(
        center_of_gravity.reshape(1, 3), motion, rotation_center=rotation_center
    )[0]
    force = np.array([0.0, 0.0, -weight])
    moment = np.cross(moved, force)
    return np.concatenate([force, moment])


def _total_load(
    verts: FloatArray,
    tris: IntArray,
    motion: FloatArray,
    *,
    center_of_gravity: FloatArray,
    weight: float,
    density: float,
    gravity: float,
    free_surface_z: float,
    rotation_center: FloatArray,
) -> FloatArray:
    moved = rigid_transform(verts, motion, rotation_center=rotation_center)
    clipped_verts, clipped_faces = clip_to_fluid(
        moved, tris, free_surface_z=free_surface_z
    )
    hydro = pressure_loads(
        clipped_verts,
        clipped_faces,
        density=density,
        gravity=gravity,
        free_surface_z=free_surface_z,
    )
    return hydro.load + _gravity_load(
        center_of_gravity, motion, weight, rotation_center=rotation_center
    )


@dataclass(frozen=True)
class PressureRouteResult:
    """Restoring matrix from the pressure route plus its diagnostics."""

    matrix: FloatArray
    immersed_volume: float
    buoyancy: float
    weight: float
    center_of_buoyancy: FloatArray
    free_surface_z: float = 0.0
    steps: FloatArray = field(default_factory=lambda: np.zeros(0))

    @property
    def displaced_weight(self) -> float:
        return self.buoyancy


def restoring_matrix_pressure_route(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    center_of_gravity: Sequence[float] = (0.0, 0.0, 0.0),
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    step: float = 1.0e-4,
    free_surface_z: float = 0.0,
    rotation_center: Optional[Sequence[float]] = None,
) -> PressureRouteResult:
    """6x6 hydrostatic restoring matrix by central differences of the pressure load.

    ``step`` is the finite-difference increment: metres for the three
    translations, radians for the three rotations (the two are dimensionally
    different, but both are small perturbations of the same linearised problem;
    use :func:`richardson_restoring_matrix` to remove the ``O(step**2)`` bias).
    ``weight = rho*g*Nabla`` is taken from the *equilibrium* immersed volume and
    held constant, as for a rigid body.
    """
    verts = _as_vertices(vertices)
    tris = _as_faces(faces, verts.shape[0])
    rho = _positive_finite(density, "density")
    g = _positive_finite(gravity, "gravity")
    delta = _positive_finite(step, "step")
    plane = float(free_surface_z)
    cog = np.asarray(center_of_gravity, dtype=float)
    if cog.shape != (3,) or not np.isfinite(cog).all():
        raise ValueError("center_of_gravity must be a finite 3-tuple")
    center = (
        np.zeros(3)
        if rotation_center is None
        else np.asarray(rotation_center, dtype=float)
    )
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("rotation_center must be a finite 3-tuple")

    clipped_verts, clipped_faces = clip_to_fluid(verts, tris, free_surface_z=plane)
    equilibrium = pressure_loads(
        clipped_verts,
        clipped_faces,
        density=rho,
        gravity=g,
        free_surface_z=plane,
    )
    volume = equilibrium.immersed_volume
    if volume <= 0.0:
        raise ValueError(
            "immersed volume is not positive: check the surface orientation "
            "(normals must point from the body into the fluid)"
        )
    weight = rho * g * volume
    areas, normals, centroids = _triangle_geometry(clipped_verts, clipped_faces)
    panel_pressure = rho * g * (plane - centroids[:, 2])
    panel_force = -(panel_pressure * areas)[:, None] * normals
    buoyancy = np.sum(panel_force, axis=0)
    fz = float(buoyancy[2])
    if abs(fz) <= 1e-12 * weight:
        raise ValueError("vertical buoyancy vanishes: the surface is not wetted")
    # M = r_B x F with F = (0, 0, F_z):  M_x = F_z*y_B,  M_y = -F_z*x_B
    center_of_buoyancy = np.array(
        [
            -float(equilibrium.moment[1]) / fz,
            float(equilibrium.moment[0]) / fz,
            0.0,
        ]
    )

    matrix = np.zeros((6, 6))
    for k in range(6):
        plus = np.zeros(6)
        minus = np.zeros(6)
        plus[k] = delta
        minus[k] = -delta
        load_plus = _total_load(
            verts,
            tris,
            plus,
            center_of_gravity=cog,
            weight=weight,
            density=rho,
            gravity=g,
            free_surface_z=plane,
            rotation_center=center,
        )
        load_minus = _total_load(
            verts,
            tris,
            minus,
            center_of_gravity=cog,
            weight=weight,
            density=rho,
            gravity=g,
            free_surface_z=plane,
            rotation_center=center,
        )
        matrix[:, k] = -(load_plus - load_minus) / (2.0 * delta)

    return PressureRouteResult(
        matrix=matrix,
        immersed_volume=volume,
        buoyancy=float(buoyancy[2]),
        weight=weight,
        center_of_buoyancy=center_of_buoyancy,
        free_surface_z=plane,
        steps=np.array([delta]),
    )


def richardson_restoring_matrix(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    steps: Sequence[float] = (2.0e-3, 1.0e-3, 5.0e-4),
    **kwargs,
) -> PressureRouteResult:
    """Pressure-route matrix with a Richardson extrapolation in ``step``.

    The central difference error is even in ``step``, so three increments are
    fitted with ``C(step) = C0 + a*step**2`` (least squares over the three
    points) and ``C0`` is returned.  This removes the truncation bias that would
    otherwise be indistinguishable from a route-to-route discrepancy.
    """
    increments = [float(s) for s in steps]
    if len(increments) < 2:
        raise ValueError("at least two steps are required")
    results = [
        restoring_matrix_pressure_route(vertices, faces, step=s, **kwargs)
        for s in increments
    ]
    if len(increments) == 2:
        c0 = (increments[1] ** 2 * results[0].matrix - increments[0] ** 2 * results[1].matrix) / (
            increments[1] ** 2 - increments[0] ** 2
        )
    else:
        a = np.array([[1.0, s * s] for s in increments])
        pinv = np.linalg.pinv(a)
        stack = np.stack([r.matrix.ravel() for r in results], axis=0)
        coeffs = pinv @ stack
        c0 = coeffs[0].reshape(6, 6)
    base = results[-1]
    return PressureRouteResult(
        matrix=c0,
        immersed_volume=base.immersed_volume,
        buoyancy=base.buoyancy,
        weight=base.weight,
        center_of_buoyancy=base.center_of_buoyancy,
        free_surface_z=base.free_surface_z,
        steps=np.asarray(increments),
    )


__all__ = [
    "DOF_NAMES",
    "PressureLoad",
    "PressureRouteResult",
    "clip_to_fluid",
    "immersed_volume",
    "pressure_loads",
    "pressure_profile",
    "restoring_matrix_pressure_route",
    "richardson_restoring_matrix",
    "rigid_transform",
    "wetted_hull_from_closed",
]