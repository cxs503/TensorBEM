"""Boundary panel geometry and validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _validate_panel_count(panel_count: int) -> int:
    """Return a validated integer panel count.

    Floats, booleans and non-integral values are rejected rather than silently
    truncated: a report that claims ``panels: 3.5`` while the mesh has four
    panels is not reproducible.
    """
    if isinstance(panel_count, (bool, np.bool_)) or not isinstance(
        panel_count, (int, np.integer)
    ):
        raise ValueError("panel_count must be an integer >= 3")
    count = int(panel_count)
    if count < 3:
        raise ValueError("panel_count must be at least 3")
    return count


def _validate_length(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


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

    @property
    def area(self) -> float:
        """Enclosed area of the polygon."""
        points = self.vertices
        return float(
            0.5
            * np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - points[:, 1] * np.roll(points[:, 0], -1)
            )
        )


def regular_polygon(panel_count: int, radius: float = 1.0) -> PanelMesh:
    """Return a counter-clockwise regular polygon approximating a circle."""
    count = _validate_panel_count(panel_count)
    radius = _validate_length("radius", radius)
    theta = 2.0 * np.pi * np.arange(count) / count
    vertices = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    return PanelMesh(vertices)


def ellipse_mesh(
    panel_count: int, semi_major: float = 1.5, semi_minor: float = 0.75
) -> PanelMesh:
    """Return a counter-clockwise inscribed polygon approximating an ellipse.

    Vertices lie on ``x**2/semi_major**2 + y**2/semi_minor**2 = 1`` at uniform
    angle steps. Uniform angle steps do not give uniform arc length on an
    ellipse; that is intentional, because a benchmark that only ever sees
    equal-length panels cannot expose a length-dependent assembly error.
    """
    count = _validate_panel_count(panel_count)
    major = _validate_length("semi_major", semi_major)
    minor = _validate_length("semi_minor", semi_minor)
    theta = 2.0 * np.pi * np.arange(count) / count
    vertices = np.column_stack((major * np.cos(theta), minor * np.sin(theta)))
    return PanelMesh(vertices)
