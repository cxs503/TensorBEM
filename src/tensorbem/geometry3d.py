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
