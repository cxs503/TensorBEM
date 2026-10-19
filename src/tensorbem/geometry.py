"""Boundary panel geometry and validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PanelMesh:
    """A closed, counter-clockwise polygon discretized into straight panels.

    Vertices have shape (N, 2); the closing segment from vertex N-1 to vertex 0
    is implicit. For counter-clockwise boundaries, normals point outward.
    """

    vertices: FloatArray

    def __post_init__(self) -> None:
        points = np.asarray(self.vertices, dtype=float)
        object.__setattr__(self, "vertices", points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2)")
        if len(points) < 3:
            raise ValueError("a closed polygon requires at least three vertices")
        if not np.isfinite(points).all():
            raise ValueError("vertices must all be finite")
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths <= 1.0e-14):
            raise ValueError("zero-length boundary panels are not allowed")
        area2 = np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                       - points[:, 1] * np.roll(points[:, 0], -1))
        if area2 <= 1.0e-14:
            raise ValueError("vertices must describe a non-degenerate CCW polygon")

    @property
    def panel_count(self) -> int:
        return int(len(self.vertices))

    @property
    def ends(self) -> FloatArray:
        return np.roll(self.vertices, -1, axis=0)

    @property
    def vectors(self) -> FloatArray:
        return self.ends - self.vertices

    @property
    def lengths(self) -> FloatArray:
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def centers(self) -> FloatArray:
        return 0.5 * (self.vertices + self.ends)

    @property
    def normals(self) -> FloatArray:
        tangent = self.vectors / self.lengths[:, None]
        return np.column_stack((tangent[:, 1], -tangent[:, 0]))

    @property
    def perimeter(self) -> float:
        return float(np.sum(self.lengths))


def regular_polygon(panel_count: int, radius: float = 1.0) -> PanelMesh:
    """Return a counter-clockwise regular polygon approximating a circle."""
    if panel_count < 3:
        raise ValueError("panel_count must be at least 3")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number")
    theta = 2.0 * np.pi * np.arange(panel_count) / panel_count
    vertices = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    return PanelMesh(vertices)
