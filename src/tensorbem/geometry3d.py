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
