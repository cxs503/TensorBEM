"""Three-dimensional triangular surface geometry and validation.

The mesh is a closed, consistently oriented triangulation with outward normals.
Collocation is performed at triangle centroids, which lie on flat panel faces;
this keeps the boundary integral equation at a smooth surface point (c = 1/2).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TriangleMesh:
    """A closed, consistently oriented triangular surface mesh.

    Parameters
    ----------
    vertices:
        Array of shape ``(V, 3)`` with finite vertex coordinates.
    faces:
        Integer array of shape ``(F, 3)``. Each row lists a triangle in an
        order consistent with the outward normal (right-hand rule). The mesh
        must be closed: every directed edge appears exactly once and its
        reverse also appears exactly once. The signed volume must be positive.
    """

    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if vertices.shape[0] < 4:
            raise ValueError("a closed surface mesh requires at least four vertices")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices must all be finite")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if faces.shape[0] < 4:
            raise ValueError("a closed surface mesh requires at least four faces")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("faces must be an integer array")
        faces = np.asarray(faces, dtype=np.int64)
        vertex_count = vertices.shape[0]
        if np.any(faces < 0) or np.any(faces >= vertex_count):
            raise ValueError("face indices must be in range [0, vertex_count)")
        if np.any(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        ):
            raise ValueError("each triangle must use three distinct vertices")

        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        double_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        scale = max(1.0, float(np.max(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1))))
        if np.any(double_areas <= 1.0e-14 * scale * scale):
            raise ValueError("degenerate triangles are not allowed")

        directed = np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        )
        keys = directed[:, 0] * vertex_count + directed[:, 1]
        _, counts = np.unique(keys, return_counts=True)
        if np.any(counts != 1):
            raise ValueError(
                "each directed edge must appear exactly once; the mesh must be "
                "closed and consistently oriented"
            )
        reverse = directed[:, 1] * vertex_count + directed[:, 0]
        if not np.isin(reverse, keys).all():
            raise ValueError("the mesh is not closed: an edge has no opposite partner")

        signed_volume = float(
            np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0
        )
        if signed_volume <= 0.0:
            raise ValueError("faces must be ordered so that the signed volume is positive (outward normals)")

        object.__setattr__(self, "vertices", vertices.copy())
        object.__setattr__(self, "faces", faces.copy())

    # ----- basic arrays -------------------------------------------------
    @property
    def v0(self) -> FloatArray:
        return self.vertices[self.faces[:, 0]]

    @property
    def v1(self) -> FloatArray:
        return self.vertices[self.faces[:, 1]]

    @property
    def v2(self) -> FloatArray:
        return self.vertices[self.faces[:, 2]]

    @property
    def panel_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def edge_vectors(self) -> FloatArray:
        """Edge vectors ``v1-v0``, ``v2-v1``, ``v0-v2`` with shape ``(F, 3, 3)``."""
        return np.stack((self.v1 - self.v0, self.v2 - self.v1, self.v0 - self.v2), axis=1)

    @property
    def edge_lengths(self) -> FloatArray:
        return np.linalg.norm(self.edge_vectors, axis=2)

    @property
    def double_areas(self) -> FloatArray:
        return np.linalg.norm(np.cross(self.v1 - self.v0, self.v2 - self.v0), axis=1)

    @property
    def areas(self) -> FloatArray:
        return 0.5 * self.double_areas

    @property
    def centroids(self) -> FloatArray:
        return (self.v0 + self.v1 + self.v2) / 3.0

    @property
    def normals(self) -> FloatArray:
        """Outward unit normals with shape ``(F, 3)``."""
        return np.cross(self.v1 - self.v0, self.v2 - self.v0) / self.double_areas[:, None]

    @property
    def surface_area(self) -> float:
        return float(np.sum(self.areas))

    @property
    def volume(self) -> float:
        """Signed volume of the closed surface (positive for outward normals)."""
        return float(np.sum(np.einsum("ij,ij->i", self.v0, np.cross(self.v1, self.v2))) / 6.0)

    @property
    def h_max(self) -> float:
        return float(np.max(self.edge_lengths))

    @property
    def h_mean(self) -> float:
        return float(np.mean(self.edge_lengths))

    @property
    def quality(self) -> dict[str, float]:
        """Deterministic mesh-quality diagnostics for JSON reports."""
        lengths = self.edge_lengths
        shortest = float(np.min(lengths))
        longest = float(np.max(lengths))
        areas = self.areas
        # Interior angles via the law of cosines at each vertex.
        e01 = self.v1 - self.v0
        e02 = self.v2 - self.v0
        e12 = self.v2 - self.v1
        e10 = self.v0 - self.v1
        e20 = self.v0 - self.v2
        e21 = self.v1 - self.v2
        angles = np.stack(
            (
                np.arccos(np.clip(
                    np.einsum("ij,ij->i", e01, e02)
                    / (np.linalg.norm(e01, axis=1) * np.linalg.norm(e02, axis=1)),
                    -1.0,
                    1.0,
                )),
                np.arccos(np.clip(
                    np.einsum("ij,ij->i", e12, e10)
                    / (np.linalg.norm(e12, axis=1) * np.linalg.norm(e10, axis=1)),
                    -1.0,
                    1.0,
                )),
                np.arccos(np.clip(
                    np.einsum("ij,ij->i", e20, e21)
                    / (np.linalg.norm(e20, axis=1) * np.linalg.norm(e21, axis=1)),
                    -1.0,
                    1.0,
                )),
            ),
            axis=1,
        )
        max_edge = np.max(lengths, axis=1)
        aspect = max_edge * max_edge / (2.0 * areas)
        return {
            "min_edge_length": shortest,
            "max_edge_length": longest,
            "edge_length_ratio": longest / shortest,
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "area_ratio": float(np.max(areas) / np.min(areas)),
            "min_angle_degrees": float(np.degrees(np.min(angles))),
            "max_angle_degrees": float(np.degrees(np.max(angles))),
            "max_aspect_ratio": float(np.max(aspect)),
        }


def icosphere(subdivisions: int, radius: float = 1.0) -> TriangleMesh:
    """Return a geodesic sphere obtained by subdividing an icosahedron.

    The panel count is ``20 * 4**subdivisions``. Vertices are projected onto
    the sphere after each subdivision, so every vertex has the requested
    radius while the flat triangular panels lie inside the sphere.
    """
    if isinstance(subdivisions, (bool, np.bool_)) or not isinstance(
        subdivisions, (int, np.integer)
    ):
        raise ValueError("subdivisions must be a non-negative integer")
    if subdivisions < 0:
        raise ValueError("subdivisions must be a non-negative integer")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number")

    phi = (1.0 + 5.0 ** 0.5) / 2.0
    base_vertices = [
        (-1.0, phi, 0.0), (1.0, phi, 0.0), (-1.0, -phi, 0.0), (1.0, -phi, 0.0),
        (0.0, -1.0, phi), (0.0, 1.0, phi), (0.0, -1.0, -phi), (0.0, 1.0, -phi),
        (phi, 0.0, -1.0), (phi, 0.0, 1.0), (-phi, 0.0, -1.0), (-phi, 0.0, 1.0),
    ]
    vertices = [
        tuple(float(value) for value in np.array(vertex) / np.linalg.norm(vertex))
        for vertex in base_vertices
    ]
    faces: list[tuple[int, int, int]] = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    for _ in range(int(subdivisions)):
        cache: dict[tuple[int, int], int] = {}
        refined: list[tuple[int, int, int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                midpoint_vector = np.array(vertices[a]) + np.array(vertices[b])
                midpoint_vector /= np.linalg.norm(midpoint_vector)
                vertices.append(tuple(float(value) for value in midpoint_vector))
                cache[key] = len(vertices) - 1
            return cache[key]

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined

    coordinates = radius * np.asarray(vertices, dtype=float)
    return TriangleMesh(coordinates, np.asarray(faces, dtype=np.int64))


def _validate_semi_axes(semi_axes: FloatArray) -> FloatArray:
    axes = np.asarray(semi_axes, dtype=float)
    if axes.shape != (3,):
        raise ValueError("semi_axes must have shape (3,)")
    if not np.isfinite(axes).all():
        raise ValueError("semi_axes must all be finite")
    if np.any(axes <= 0.0):
        raise ValueError("semi_axes must all be positive")
    return axes


def ellipsoid_mesh(subdivisions: int, semi_axes: FloatArray) -> TriangleMesh:
    """Return a flat-faceted ellipsoid mesh with the icosphere topology.

    The unit icosphere is scaled axis by axis, so the connectivity, the
    outward orientation and the closure checks are those of
    :func:`icosphere`; the vertices lie on
    ``x**2/a**2 + y**2/b**2 + z**2/c**2 = 1`` and the flat panels lie inside
    it. The panel count is ``20 * 4**subdivisions``.

    Scaling by a positive diagonal has positive determinant, so the signed
    volume stays positive and the outward normals are preserved. A mesh built
    this way is *not* an equi-angular parametrisation of the ellipsoid, which
    is deliberate: it keeps a controlled, isotropic-in-index distribution of
    panels across the three principal directions.
    """
    axes = _validate_semi_axes(semi_axes)
    base = icosphere(subdivisions, radius=1.0)
    return TriangleMesh(base.vertices * axes, base.faces)


def project_to_ellipsoid(points: FloatArray, semi_axes: FloatArray) -> FloatArray:
    """Project points radially from the origin onto an ellipsoid.

    Parameters
    ----------
    points:
        Array of shape ``(M, 3)``. The origin is not a valid direction and is
        rejected.
    semi_axes:
        Positive semi-axes of shape ``(3,)``.

    Returns
    -------
    numpy.ndarray
        Points of shape ``(M, 3)`` on the ellipsoid, each along the ray from
        the origin through the corresponding input point.
    """
    axes = _validate_semi_axes(semi_axes)
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError("points must all be finite")
    norms = np.linalg.norm(coordinates, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("points must not contain the origin")
    directions = coordinates / norms[:, None]
    scale = 1.0 / np.sqrt(np.sum((directions / axes[None, :]) ** 2, axis=1))
    return scale[:, None] * directions


@dataclass(frozen=True)
class WettedHullMesh:
    """Waterline-truncated 3-D wetted hull surface in ``z <= 0``.

    Unlike :class:`TriangleMesh`, which enforces a closed watertight surface,
    ``WettedHullMesh`` represents the submerged hull panels of a floating body
    piercing the calm free surface ``z = 0``. Every interior edge must be
    shared by two oppositely oriented triangles, while every unmatched boundary
    half-edge must lie on the calm waterplane ``z = 0`` and form one or more
    closed oriented waterline loops. Panel normals point outward from the body
    into the fluid, so closing the hull with the planar waterplane ``z = 0``
    encloses a positive displaced volume.
    """

    vertices: FloatArray
    faces: IntArray
    waterline_edges: IntArray

    def __init__(
        self,
        vertices: FloatArray,
        faces: IntArray,
        *,
        waterline_tol: float = 1.0e-12,
    ) -> None:
        verts = np.asarray(vertices, dtype=float)
        tris = np.asarray(faces)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if not np.isfinite(verts).all():
            raise ValueError("vertices must all be finite")
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if tris.shape[0] < 3:
            raise ValueError("a wetted hull mesh requires at least three faces")
        if not np.issubdtype(tris.dtype, np.integer):
            raise ValueError("faces must be an integer array")
        tris = np.asarray(tris, dtype=np.int64)
        vertex_count = verts.shape[0]
        if np.any(tris < 0) or np.any(tris >= vertex_count):
            raise ValueError("face indices must be in range [0, vertex_count)")
        if np.any(
            (tris[:, 0] == tris[:, 1])
            | (tris[:, 1] == tris[:, 2])
            | (tris[:, 2] == tris[:, 0])
        ):
            raise ValueError("each triangle must use three distinct vertices")

        scale = max(
            1.0,
            float(np.max(np.linalg.norm(verts - verts.mean(axis=0), axis=1))),
        )
        tol = float(waterline_tol) * scale
        if np.any(verts[:, 2] > tol):
            raise ValueError("all wetted hull vertices must satisfy z <= 0")
        verts = verts.copy()
        on_waterline = np.abs(verts[:, 2]) <= tol
        verts[on_waterline, 2] = 0.0

        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        double_areas = np.linalg.norm(cross, axis=1)
        if np.any(double_areas <= 1.0e-14 * scale * scale):
            raise ValueError("degenerate triangles are not allowed")
        centroids = (v0 + v1 + v2) / 3.0
        if np.any(centroids[:, 2] >= -1.0e-14 * scale):
            raise ValueError(
                "wetted hull panels must lie strictly below z = 0 at their centroids "
                "(do not include horizontal waterplane lid panels)"
            )

        directed = np.concatenate(
            (tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]), axis=0
        )
        keys = directed[:, 0] * vertex_count + directed[:, 1]
        _, counts = np.unique(keys, return_counts=True)
        if np.any(counts != 1):
            raise ValueError(
                "each directed edge must appear at most once with consistent orientation"
            )
        reverse = directed[:, 1] * vertex_count + directed[:, 0]
        unmatched = ~np.isin(reverse, keys)
        boundary = directed[unmatched]
        if boundary.shape[0] > 0:
            if not (
                np.all(verts[boundary[:, 0], 2] == 0.0)
                and np.all(verts[boundary[:, 1], 2] == 0.0)
            ):
                raise ValueError(
                    "open hull boundary edges must lie on the calm waterplane z = 0"
                )
            out_verts, out_counts = np.unique(boundary[:, 0], return_counts=True)
            in_verts, in_counts = np.unique(boundary[:, 1], return_counts=True)
            if not (
                np.array_equal(out_verts, in_verts)
                and np.array_equal(out_counts, in_counts)
            ):
                raise ValueError("waterline boundary edges must form closed loop(s)")

        signed_volume = float(
            np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0
        )
        normals = cross / double_areas[:, None]
        areas = 0.5 * double_areas
        volume_z = float(np.sum(areas * centroids[:, 2] * normals[:, 2]))
        if signed_volume <= 0.0 or volume_z <= 0.0:
            raise ValueError(
                "faces must be ordered so outward normals point into the fluid "
                "(positive displaced volume)"
            )
        if abs(signed_volume - volume_z) > 1.0e-10 * max(1.0, signed_volume):
            raise ValueError("inconsistent displaced volume across hull panels")

        object.__setattr__(self, "vertices", verts)
        object.__setattr__(self, "faces", tris.copy())
        object.__setattr__(
            self, "waterline_edges", np.asarray(boundary, dtype=np.int64)
        )

    @property
    def v0(self) -> FloatArray:
        return self.vertices[self.faces[:, 0]]

    @property
    def v1(self) -> FloatArray:
        return self.vertices[self.faces[:, 1]]

    @property
    def v2(self) -> FloatArray:
        return self.vertices[self.faces[:, 2]]

    @property
    def panel_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def edge_vectors(self) -> FloatArray:
        return np.stack(
            (self.v1 - self.v0, self.v2 - self.v1, self.v0 - self.v2), axis=1
        )

    @property
    def edge_lengths(self) -> FloatArray:
        return np.linalg.norm(self.edge_vectors, axis=2)

    @property
    def double_areas(self) -> FloatArray:
        return np.linalg.norm(np.cross(self.v1 - self.v0, self.v2 - self.v0), axis=1)

    @property
    def areas(self) -> FloatArray:
        return 0.5 * self.double_areas

    @property
    def centroids(self) -> FloatArray:
        return (self.v0 + self.v1 + self.v2) / 3.0

    @property
    def normals(self) -> FloatArray:
        """Unit normals pointing out of the body into the fluid."""
        return np.cross(self.v1 - self.v0, self.v2 - self.v0) / self.double_areas[:, None]

    @property
    def surface_area(self) -> float:
        return float(np.sum(self.areas))

    @property
    def volume(self) -> float:
        """Displaced volume enclosed by the hull and the ``z = 0`` waterplane."""
        return float(
            np.sum(np.einsum("ij,ij->i", self.v0, np.cross(self.v1, self.v2))) / 6.0
        )

    @property
    def waterplane_area(self) -> float:
        """Area of the ``z = 0`` waterplane cut by the hull."""
        if self.waterline_edges.shape[0] == 0:
            return 0.0
        va = self.vertices[self.waterline_edges[:, 0]]
        vb = self.vertices[self.waterline_edges[:, 1]]
        return float(-0.5 * np.sum(va[:, 0] * vb[:, 1] - vb[:, 0] * va[:, 1]))

    @property
    def h_max(self) -> float:
        return float(np.max(self.edge_lengths))

    @property
    def h_mean(self) -> float:
        return float(np.mean(self.edge_lengths))

    @property
    def quality(self) -> dict[str, float]:
        lengths = self.edge_lengths
        shortest = float(np.min(lengths))
        longest = float(np.max(lengths))
        areas = self.areas
        e01 = self.v1 - self.v0
        e02 = self.v2 - self.v0
        e12 = self.v2 - self.v1
        e10 = self.v0 - self.v1
        e20 = self.v0 - self.v2
        e21 = self.v1 - self.v2
        angles = np.stack(
            (
                np.arccos(
                    np.clip(
                        np.einsum("ij,ij->i", e01, e02)
                        / (np.linalg.norm(e01, axis=1) * np.linalg.norm(e02, axis=1)),
                        -1.0,
                        1.0,
                    )
                ),
                np.arccos(
                    np.clip(
                        np.einsum("ij,ij->i", e12, e10)
                        / (np.linalg.norm(e12, axis=1) * np.linalg.norm(e10, axis=1)),
                        -1.0,
                        1.0,
                    )
                ),
                np.arccos(
                    np.clip(
                        np.einsum("ij,ij->i", e20, e21)
                        / (np.linalg.norm(e20, axis=1) * np.linalg.norm(e21, axis=1)),
                        -1.0,
                        1.0,
                    )
                ),
            ),
            axis=1,
        )
        max_edge = np.max(lengths, axis=1)
        aspect = max_edge * max_edge / (2.0 * areas)
        return {
            "min_edge_length": shortest,
            "max_edge_length": longest,
            "edge_length_ratio": longest / shortest,
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "area_ratio": float(np.max(areas) / np.min(areas)),
            "min_angle_degrees": float(np.degrees(np.min(angles))),
            "max_angle_degrees": float(np.degrees(np.max(angles))),
            "max_aspect_ratio": float(np.max(aspect)),
        }


SurfaceMesh3D = TriangleMesh | WettedHullMesh


def hemisphere_hull_mesh(
    subdivisions: int,
    radius: float = 1.0,
    center_xy: tuple[float, float] = (0.0, 0.0),
) -> WettedHullMesh:
    """Return a quasi-uniform floating hemisphere hull in ``z <= 0``.

    Subdivides the four lower faces of a unit octahedron and projects vertices
    radially onto the hemisphere ``r = radius, z <= 0``. The panel count is
    ``4 * 4**subdivisions``, all waterline vertices lie strictly on ``z = 0``,
    and panel normals point outward into the fluid.
    """
    if isinstance(subdivisions, (bool, np.bool_)) or not isinstance(
        subdivisions, (int, np.integer)
    ):
        raise ValueError("subdivisions must be a non-negative integer")
    if subdivisions < 0:
        raise ValueError("subdivisions must be a non-negative integer")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number")
    xy = np.asarray(center_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("center_xy must be a finite 2-tuple")

    vertices: list[tuple[float, float, float]] = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    ]
    faces: list[tuple[int, int, int]] = [
        (0, 4, 1),
        (1, 4, 2),
        (2, 4, 3),
        (3, 4, 0),
    ]
    for _ in range(int(subdivisions)):
        cache: dict[tuple[int, int], int] = {}
        refined: list[tuple[int, int, int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                mid = np.array(vertices[a]) + np.array(vertices[b])
                mid /= np.linalg.norm(mid)
                if abs(mid[2]) < 1.0e-15:
                    mid[2] = 0.0
                vertices.append((float(mid[0]), float(mid[1]), float(mid[2])))
                cache[key] = len(vertices) - 1
            return cache[key]

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined

    coords = radius * np.asarray(vertices, dtype=float)
    coords[:, 0] += float(xy[0])
    coords[:, 1] += float(xy[1])
    return WettedHullMesh(coords, np.asarray(faces, dtype=np.int64))


def submerged_sphere_mesh(
    subdivisions: int,
    radius: float = 1.0,
    submergence: float = 1.5,
    center_xy: tuple[float, float] = (0.0, 0.0),
) -> TriangleMesh:
    """Return a closed geodesic sphere of ``radius`` centered at ``z = -submergence``."""
    if not math.isfinite(submergence) or submergence <= radius:
        raise ValueError("submergence must be finite and strictly greater than radius")
    xy = np.asarray(center_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("center_xy must be a finite 2-tuple")
    base = icosphere(subdivisions, radius=radius)
    offset = np.array([float(xy[0]), float(xy[1]), -float(submergence)])
    return TriangleMesh(base.vertices + offset, base.faces)


def floating_sphere_hull_mesh(
    subdivisions: int,
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, -0.5),
) -> WettedHullMesh:
    """Return the wetted cap of a floating sphere truncated at ``z = 0``.

    The body is the sphere of ``radius`` centred at ``center = (x0, y0, z0)``
    with ``-radius < z0 < 0`` (centre below the calm waterplane, sphere
    piercing it). The returned :class:`WettedHullMesh` contains the part of
    the sphere with ``z <= 0``; its open boundary is the circular waterline of
    radius ``sqrt(radius**2 - z0**2)`` centred at ``(x0, y0)``.

    The geodesic sphere of :func:`icosphere` is clipped against the plane
    ``z = 0`` triangle by triangle (Sutherland-Hodgman). Points are welded by
    coordinates rounded to ``1e-12`` of the radius, which both closes the
    waterline (cut points on a shared edge agree up to round-off) and keeps a
    below-water vertex distinct from a waterline point above it. Clipping
    preserves the counter-clockwise ordering of the source panels, hence
    panel normals stay outward (into the fluid).
    """
    if isinstance(subdivisions, (bool, np.bool_)) or not isinstance(
        subdivisions, (int, np.integer)
    ):
        raise ValueError("subdivisions must be a non-negative integer")
    if subdivisions < 0:
        raise ValueError("subdivisions must be a non-negative integer")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number")
    c = np.asarray(center, dtype=float)
    if c.shape != (3,) or not np.isfinite(c).all():
        raise ValueError("center must be a finite 3-tuple")
    if not (-radius < c[2] < 0.0):
        raise ValueError(
            "center z must lie strictly between -radius and 0 "
            "(the sphere must pierce the calm waterplane)"
        )

    base = icosphere(subdivisions, radius=radius)
    verts = base.vertices + c
    faces = base.faces

    def key_of(point: np.ndarray) -> tuple[float, float, float]:
        scale = 1.0e-12 * radius
        return (
            round(float(point[0]) / scale) * scale,
            round(float(point[1]) / scale) * scale,
            round(float(point[2]) / scale) * scale,
        )

    new_verts: list[tuple[float, float, float]] = []
    point_map: dict[tuple[float, float, float], int] = {}

    def weld(point: np.ndarray) -> int:
        key = key_of(point)
        if key not in point_map:
            point_map[key] = len(new_verts)
            new_verts.append(
                (float(point[0]), float(point[1]), 0.0 if abs(point[2]) < 1.0e-15 else float(point[2]))
            )
        return point_map[key]

    new_faces: list[tuple[int, int, int]] = []
    for a, b, cc in faces:
        z = verts[[a, b, cc], 2]
        if np.all(z > 0.0):
            continue  # panel entirely above the waterplane
        if np.all(z <= 0.0):
            new_faces.append((weld(verts[a]), weld(verts[b]), weld(verts[cc])))
            continue
        # Clip the triangle against z <= 0 (Sutherland-Hodgman).
        order = (a, b, cc)
        polygon: list[np.ndarray] = []
        for i in range(3):
            current = verts[order[i]]
            following = verts[order[(i + 1) % 3]]
            current_in = current[2] <= 0.0
            following_in = following[2] <= 0.0
            if current_in:
                polygon.append(current)
            if current_in != following_in:
                t = -current[2] / (following[2] - current[2])
                if 0.0 < t < 1.0:
                    polygon.append(current + t * (following - current))
        # Drop consecutive duplicates (a vertex sitting exactly on the
        # waterplane is emitted both as a kept vertex and as a cut).
        deduped: list[np.ndarray] = []
        for point in polygon:
            if deduped and np.array_equal(deduped[-1], point):
                continue
            deduped.append(point)
        if len(deduped) < 3:
            continue
        coords = np.asarray(deduped, dtype=float)
        area = 0.0
        for k in range(1, len(coords) - 1):
            tri = np.array([coords[0], coords[k], coords[k + 1]])
            area += 0.5 * float(
                np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            )
        if area <= 1.0e-14 * radius * radius:
            continue  # sliver on the waterplane: no fluid-side area
        indices = [weld(p) for p in coords]
        for k in range(1, len(coords) - 1):
            new_faces.append((indices[0], indices[k], indices[k + 1]))

    return WettedHullMesh(
        np.asarray(new_verts, dtype=float), np.asarray(new_faces, dtype=np.int64)
    )


def _oriented_triangle(
    p0: FloatArray,
    p1: FloatArray,
    p2: FloatArray,
    outward: FloatArray,
) -> tuple[int, int, int] | None:
    """Return index order (0,1,2)/(0,2,1) whose normal agrees with ``outward``."""
    normal = np.cross(p1 - p0, p2 - p0)
    if not np.isfinite(normal).all() or float(np.linalg.norm(normal)) == 0.0:
        return None
    if float(np.dot(normal, outward)) >= 0.0:
        return (0, 1, 2)
    return (0, 2, 1)


def _wall_strip(
    bottom_loop: IntArray,
    top_loop: IntArray,
    coords: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    """Append an outward-oriented vertical wall strip between two closed loops.

    ``bottom_loop`` and ``top_loop`` hold vertex indices of the same closed
    polygon at two z levels (``top_loop`` strictly above). Triangle orientation
    is fixed per quad by testing the geometric normal against the horizontal
    outward direction, which is derived from the loop's signed area so that the
    same helper serves clockwise and counter-clockwise loops.
    """
    xy = np.array([[coords[i][0], coords[i][1]] for i in bottom_loop], dtype=float)
    x_next = np.roll(xy[:, 0], -1)
    y_next = np.roll(xy[:, 1], -1)
    twice_area = float(np.sum(xy[:, 0] * y_next - x_next * xy[:, 1]))
    sign = 1.0 if twice_area > 0.0 else -1.0
    n = len(bottom_loop)
    for k in range(n):
        k2 = (k + 1) % n
        b0 = np.array(coords[bottom_loop[k]], dtype=float)
        b1 = np.array(coords[bottom_loop[k2]], dtype=float)
        t0 = np.array(coords[top_loop[k]], dtype=float)
        t1 = np.array(coords[top_loop[k2]], dtype=float)
        d = b1 - b0
        outward = np.array([sign * d[1], -sign * d[0], 0.0], dtype=float)
        if float(np.linalg.norm(outward)) == 0.0:
            raise ValueError("degenerate vertical wall edge (zero length in xy)")
        outward /= np.linalg.norm(outward)
        corners = (b0, t0, t1, b1)
        for tri in ((0, 1, 2), (0, 2, 3)):
            order = _oriented_triangle(
                corners[tri[0]], corners[tri[1]], corners[tri[2]], outward
            )
            if order is None:
                raise ValueError("degenerate vertical wall triangle")
            quad = (bottom_loop[k], top_loop[k], top_loop[k2], bottom_loop[k2])
            faces.append(tuple(int(quad[tri[order[j]]]) for j in range(3)))


def rectangular_barge_hull_mesh(
    length: float,
    beam: float,
    draft: float,
    nx: int = 1,
    ny: int = 1,
) -> WettedHullMesh:
    """Return a wall-sided rectangular barge hull in ``z <= 0``.

    The wetted surface is the flat bottom at ``z = -draft`` plus the four
    vertical sides up to the calm waterline ``z = 0``. Cell counts ``nx``/``ny``
    refine the bottom and the sides while keeping the geometry exactly
    wall-sided, so every hydrostatic quantity for this hull has a closed-form
    reference: ``V = L B T``, ``A_wp = L B``, ``I_T = L B^3 / 12``,
    ``I_L = B L^3 / 12``, ``KB = T / 2`` and ``BM_T = B^2 / (12 T)``.
    """
    for name, value in (("length", length), ("beam", beam), ("draft", draft)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if int(nx) != nx or int(ny) != ny:
        raise ValueError("nx and ny must be integers")
    nx, ny = int(nx), int(ny)
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be at least 1")

    half_l, half_b = 0.5 * float(length), 0.5 * float(beam)
    coords: list[tuple[float, float, float]] = []
    bottom = np.zeros((nx + 1, ny + 1), dtype=np.int64)
    for i in range(nx + 1):
        x = -half_l + float(length) * i / nx
        for j in range(ny + 1):
            y = -half_b + float(beam) * j / ny
            bottom[i, j] = len(coords)
            coords.append((x, y, -float(draft)))

    # Boundary loop in clockwise order seen from above, matching the ordering
    # used by the bottom-face triangulation below.
    loop: list[tuple[int, int]] = []
    loop.extend((0, j) for j in range(ny + 1))
    loop.extend((i, ny) for i in range(1, nx + 1))
    loop.extend((nx, j) for j in range(ny - 1, -1, -1))
    loop.extend((i, 0) for i in range(nx - 1, 0, -1))

    top_of: dict[tuple[int, int], int] = {}
    for ij in loop:
        top_of[ij] = len(coords)
        coords.append((coords[bottom[ij]][0], coords[bottom[ij]][1], 0.0))

    faces: list[tuple[int, int, int]] = []
    down = np.array([0.0, 0.0, -1.0])
    for i in range(nx):
        for j in range(ny):
            q = (
                bottom[i, j],
                bottom[i + 1, j],
                bottom[i + 1, j + 1],
                bottom[i, j + 1],
            )
            pts = [np.array(coords[v], dtype=float) for v in q]
            for tri in ((0, 1, 2), (0, 2, 3)):
                order = _oriented_triangle(pts[tri[0]], pts[tri[1]], pts[tri[2]], down)
                if order is None:
                    raise ValueError("degenerate bottom triangle")
                faces.append(tuple(q[tri[order[k]]] for k in range(3)))

    bottom_loop = np.array([bottom[ij] for ij in loop], dtype=np.int64)
    top_loop = np.array([top_of[ij] for ij in loop], dtype=np.int64)
    _wall_strip(bottom_loop, top_loop, coords, faces)

    return WettedHullMesh(np.asarray(coords, dtype=float), np.asarray(faces, dtype=np.int64))


def elliptic_cylinder_hull_mesh(
    semi_axis_x: float,
    semi_axis_y: float,
    draft: float,
    panels: int = 64,
    phase: float = 0.0,
) -> WettedHullMesh:
    """Return a wall-sided elliptic-cylinder hull in ``z <= 0``.

    The waterplane is a ``panels``-sided polygon inscribed in the ellipse, so
    the exact reference values are the polygon values ``A_wp = (n/2) ab
    sin(2 pi / n)`` and ``V = A_wp * draft``, while the ellipse values
    ``pi a b`` / ``(pi/4) a b^3`` bound the discretization error as
    ``O(n^-2)``. Useful as a curved-waterplane convergence case.
    """
    for name, value in (
        ("semi_axis_x", semi_axis_x),
        ("semi_axis_y", semi_axis_y),
        ("draft", draft),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if int(panels) != panels or int(panels) < 8:
        raise ValueError("panels must be an integer >= 8")

    coords: list[tuple[float, float, float]] = []
    n = int(panels)
    angles = np.arange(n) * (2.0 * math.pi / n) + float(phase)
    xs = float(semi_axis_x) * np.cos(angles)
    ys = float(semi_axis_y) * np.sin(angles)
    bottom = []
    for x, y in zip(xs, ys):
        bottom.append(len(coords))
        coords.append((float(x), float(y), -float(draft)))
    top = []
    for x, y in zip(xs, ys):
        top.append(len(coords))
        coords.append((float(x), float(y), 0.0))

    faces: list[tuple[int, int, int]] = []
    down = np.array([0.0, 0.0, -1.0])
    center_bottom = len(coords)
    coords.append((0.0, 0.0, -float(draft)))
    for k in range(n):
        k2 = (k + 1) % n
        p0 = np.array(coords[bottom[k]], dtype=float)
        p1 = np.array(coords[bottom[k2]], dtype=float)
        p2 = np.array(coords[center_bottom], dtype=float)
        order = _oriented_triangle(p0, p1, p2, down)
        if order is None:
            raise ValueError("degenerate bottom triangle")
        tri = (bottom[k], bottom[k2], center_bottom)
        faces.append(tuple(tri[order[j]] for j in range(3)))

    _wall_strip(
        np.asarray(bottom, dtype=np.int64),
        np.asarray(top, dtype=np.int64),
        coords,
        faces,
    )
    return WettedHullMesh(np.asarray(coords, dtype=float), np.asarray(faces, dtype=np.int64))


def _weld_quads_to_wetted_hull(
    points: FloatArray,
    quads: list[tuple[int, int, int, int]],
    outward: FloatArray,
) -> WettedHullMesh:
    """Triangulate a quad sheet, make the winding outward, and build the hull.

    ``outward`` holds, per quad, a direction the outward panel normal must have
    a positive projection on (a face normal for the barge, the sheet normal
    ``(0, +-1, 0)`` for the Wigley hull, which is a height field over
    ``(x, z)``). This orients every panel without relying on a body centroid.
    """
    verts: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}

    def vid(k: int) -> int:
        p = points[k]
        key = (float(p[0]), float(p[1]), float(p[2]))
        key = (key[0] + 0.0, key[1] + 0.0, key[2] + 0.0)
        if key not in index:
            index[key] = len(verts)
            verts.append(key)
        return index[key]

    faces: list[tuple[int, int, int]] = []
    for k, (a, b, c, d) in enumerate(quads):
        pa, pb, pc, pd = (points[j] for j in (a, b, c, d))
        n = np.cross(pb - pa, pc - pa) + np.cross(pc - pa, pd - pa)
        tri = [(a, b, c), (a, c, d)]
        if float(np.dot(n, outward[k])) < 0.0:
            tri = [(a, c, b), (a, d, c)]
        for t in tri:
            ia, ib, ic = (vid(j) for j in t)
            if ia == ib or ib == ic or ic == ia:
                continue
            faces.append((ia, ib, ic))
    return WettedHullMesh(
        np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)
    )


def box_hull_mesh(
    length: float = 10.0,
    beam: float = 4.0,
    draft: float = 2.0,
    *,
    n_length: int = 4,
    n_beam: int = 3,
    n_draft: int = 3,
) -> WettedHullMesh:
    """Return the wetted surface of a rectangular barge (``L x B x T``).

    The submerged surface is the flat bottom plus four flat sides; the
    waterplane lid is omitted, as required by :class:`WettedHullMesh`. Because
    every panel is planar and exactly on the analytic surface, the discretely
    displaced volume, waterplane area and waterplane second moments are
    *exact* to round-off for any subdivision. This makes the barge the
    reference case that separates discretisation error from implementation
    error in the hydrostatic module.
    """
    for name, value in (("length", length), ("beam", beam), ("draft", draft)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    for name, value in (
        ("n_length", n_length),
        ("n_beam", n_beam),
        ("n_draft", n_draft),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")

    xs = np.linspace(-0.5 * length, 0.5 * length, int(n_length) + 1)
    ys = np.linspace(-0.5 * beam, 0.5 * beam, int(n_beam) + 1)
    zs = np.linspace(-draft, 0.0, int(n_draft) + 1)

    points: list[tuple[float, float, float]] = []
    pos: dict[tuple[float, float, float], int] = {}

    def pid(p: tuple[float, float, float]) -> int:
        if p not in pos:
            pos[p] = len(points)
            points.append(p)
        return pos[p]

    quads: list[tuple[int, int, int, int]] = []
    outward: list[tuple[float, float, float]] = []
    # bottom, z = -draft
    for i in range(int(n_length)):
        for j in range(int(n_beam)):
            quads.append(
                (
                    pid((float(xs[i]), float(ys[j]), -draft)),
                    pid((float(xs[i + 1]), float(ys[j]), -draft)),
                    pid((float(xs[i + 1]), float(ys[j + 1]), -draft)),
                    pid((float(xs[i]), float(ys[j + 1]), -draft)),
                )
            )
            outward.append((0.0, 0.0, -1.0))
    # side shells: y = -B/2, y = +B/2, x = -L/2, x = +L/2
    for i in range(int(n_length)):
        for k in range(int(n_draft)):
            for y_side, sgn in ((-0.5 * beam, -1.0), (0.5 * beam, 1.0)):
                quads.append(
                    (
                        pid((float(xs[i]), y_side, float(zs[k]))),
                        pid((float(xs[i + 1]), y_side, float(zs[k]))),
                        pid((float(xs[i + 1]), y_side, float(zs[k + 1]))),
                        pid((float(xs[i]), y_side, float(zs[k + 1]))),
                    )
                )
                outward.append((0.0, sgn, 0.0))
    for j in range(int(n_beam)):
        for k in range(int(n_draft)):
            for x_side, sgn in ((-0.5 * length, -1.0), (0.5 * length, 1.0)):
                quads.append(
                    (
                        pid((x_side, float(ys[j]), float(zs[k]))),
                        pid((x_side, float(ys[j + 1]), float(zs[k]))),
                        pid((x_side, float(ys[j + 1]), float(zs[k + 1]))),
                        pid((x_side, float(ys[j]), float(zs[k + 1]))),
                    )
                )
                outward.append((sgn, 0.0, 0.0))
    return _weld_quads_to_wetted_hull(
        np.asarray(points, dtype=float), quads, np.asarray(outward, dtype=float)
    )


def spheroid_hull_mesh(
    semi_axes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    subdivisions: int = 3,
) -> WettedHullMesh:
    """Return the wetted half-ellipsoid (spheroid) hull in ``z <= 0``.

    The body is the lower half of the ellipsoid with semi-axes
    ``(a, b, c)`` (length, beam, draft), i.e. the ellipsoid is truncated by the
    calm waterplane through its centre. Every hydrostatic property of this body
    has a closed form (see :mod:`tensorbem.benchmarks.hydrostatic_benchmarks`),
    and the waterplane is an ellipse rather than a rectangle, so the hull is a
    ship-like verification case for the curved-waterline integration path.

    The mesh is the geodesic hemisphere of :func:`hemisphere_hull_mesh` scaled
    anisotropically, which keeps panel normals outward and the waterline on
    ``z = 0`` for any positive semi-axes.
    """
    axes = np.asarray(semi_axes, dtype=float)
    if axes.shape != (3,):
        raise ValueError("semi_axes must be a 3-tuple (a, b, c)")
    if not np.isfinite(axes).all() or np.any(axes <= 0.0):
        raise ValueError("semi_axes must all be positive and finite")
    base = hemisphere_hull_mesh(subdivisions)
    verts = base.vertices * axes
    verts[np.abs(verts[:, 2]) <= 1.0e-12, 2] = 0.0
    return WettedHullMesh(verts, base.faces)


def capsule_mesh(
    length: float,
    radius: float = 1.0,
    *,
    panel_size: float | None = None,
    axial_rings: int | None = None,
    circumferential: int | None = None,
    submergence: float = 0.5,
) -> TriangleMesh:
    """Return a closed submerged capsule: cylinder of ``length`` plus two caps.

    The body is the surface of revolution about the ``x`` axis swept by a
    stadium profile: a cylinder of radius ``radius`` and axial length ``length``
    closed by two hemispheres. Its volume ``pi a^2 L + 4/3 pi a^3`` and area
    ``2 pi a L + 4 pi a^2`` are closed form, which makes the mesh directly
    auditable, and the rounded ends keep every panel well shaped, so the
    discretisation error stays a property of the *size* of the panels rather
    than of a sharp edge. The axis sits at ``z = -submergence`` (fully
    submerged, ``submergence > radius``).

    It exists to probe the 2-D limit of the 3-D solver: as ``length/radius``
    grows the end contribution is a fixed number of ``O(radius^3)`` while the
    wetted area grows like ``radius * length``, so any per-unit-span quantity
    must approach the value of the infinite cylinder.

    ``panel_size`` sets the target edge length on the meridian;
    ``axial_rings`` / ``circumferential`` override the ring counts directly
    (needed when the caller wants a specific refinement family).
    """
    a = float(radius)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("radius must be positive and finite")
    cyl = float(length)
    if not math.isfinite(cyl) or cyl < 0.0:
        raise ValueError("length must be finite and non-negative")
    if not math.isfinite(submergence) or submergence <= a:
        raise ValueError("submergence must be finite and strictly greater than radius")

    meridian = cyl + math.pi * a
    if axial_rings is None:
        if panel_size is None:
            raise ValueError("either panel_size or axial_rings must be given")
        n_rings = max(3, int(round(meridian / float(panel_size))))
    else:
        n_rings = int(axial_rings)
    if circumferential is None:
        if panel_size is None:
            raise ValueError("either panel_size or circumferential must be given")
        n_phi = max(6, int(round(2.0 * math.pi * a / float(panel_size))))
    else:
        n_phi = int(circumferential)
    if n_rings < 3:
        raise ValueError("need at least 3 axial rings")
    if n_phi < 6:
        raise ValueError("need at least 6 circumferential panels")

    def meridian_point(s: float) -> tuple[float, float]:
        half = 0.5 * cyl
        cap = 0.5 * math.pi * a
        if s <= 0.0:
            return -half - a, 0.0
        if s >= meridian:
            return half + a, 0.0
        if s < cap:
            theta = -0.5 * math.pi + s / a
            return -half + a * math.sin(theta), a * math.cos(theta)
        if s < cap + cyl:
            return -half + (s - cap), a
        tail = s - cap - cyl
        theta = tail / a
        return half + a * math.sin(theta), a * math.cos(theta)

    phis = 2.0 * math.pi * np.arange(n_phi, dtype=float) / n_phi
    cos_phi = np.cos(phis)
    sin_phi = np.sin(phis)

    rings = np.empty((n_rings, n_phi, 3), dtype=float)
    for i in range(n_rings):
        s = meridian * (i + 0.5) / n_rings
        x, r = meridian_point(s)
        rings[i, :, 0] = x
        rings[i, :, 1] = r * cos_phi
        rings[i, :, 2] = r * sin_phi

    vertices = [ring for ring in rings]
    faces: list[list[int]] = []
    for i in range(n_rings - 1):
        base0, base1 = i * n_phi, (i + 1) * n_phi
        for j in range(n_phi):
            j2 = (j + 1) % n_phi
            faces.append([base0 + j, base1 + j, base1 + j2])
            faces.append([base0 + j, base1 + j2, base0 + j2])
    verts = np.vstack(vertices)
    starboard = len(verts)
    verts = np.vstack([verts, [[-0.5 * cyl - a, 0.0, 0.0], [0.5 * cyl + a, 0.0, 0.0]]])
    last = (n_rings - 1) * n_phi
    for j in range(n_phi):
        j2 = (j + 1) % n_phi
        faces.append([starboard, j, j2])
        faces.append([starboard + 1, last + j2, last + j])
    face_arr = np.asarray(faces, dtype=np.int64)
    verts[:, 2] -= float(submergence)

    tri = verts[face_arr]
    six_volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum())
    if six_volume < 0.0:
        face_arr = face_arr[:, ::-1]
    return TriangleMesh(verts, face_arr)
