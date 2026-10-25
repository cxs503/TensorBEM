"""Mesh quality diagnostics for triangular surface meshes.

Provides detailed mesh quality metrics including:
- Area ratios (min/max)
- Interior angles (minimum angle per triangle)
- Aspect ratios (edge length ratio)
- Skewness measures

These diagnostics help identify poorly-shaped elements that may degrade
BEM solution accuracy or cause numerical issues.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MeshQualityReport:
    """Comprehensive mesh quality diagnostics.

    Attributes
    ----------
    mesh_panels : int
        Number of panels in the mesh.
    min_area : float
        Minimum panel area.
    max_area : float
        Maximum panel area.
    area_ratio : float
        Ratio of max to min area (1.0 = uniform).
    min_angle_deg : float
        Minimum interior angle across all triangles (degrees).
    max_angle_deg : float
        Maximum interior angle across all triangles (degrees).
    mean_min_angle_deg : float
        Mean of minimum angles per triangle (degrees).
    min_aspect_ratio : float
        Minimum aspect ratio (longest edge / shortest edge).
    max_aspect_ratio : float
        Maximum aspect ratio.
    mean_aspect_ratio : float
        Mean aspect ratio.
    panels_below_30_deg : int
        Number of panels with minimum angle < 30 degrees.
    panels_above_5_aspect : int
        Number of panels with aspect ratio > 5.
    """

    mesh_panels: int
    min_area: float
    max_area: float
    area_ratio: float
    min_angle_deg: float
    max_angle_deg: float
    mean_min_angle_deg: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    mean_aspect_ratio: float
    panels_below_30_deg: int
    panels_above_5_aspect: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "mesh_panels": int(self.mesh_panels),
            "min_area": float(self.min_area),
            "max_area": float(self.max_area),
            "area_ratio": float(self.area_ratio),
            "min_angle_deg": float(self.min_angle_deg),
            "max_angle_deg": float(self.max_angle_deg),
            "mean_min_angle_deg": float(self.mean_min_angle_deg),
            "min_aspect_ratio": float(self.min_aspect_ratio),
            "max_aspect_ratio": float(self.max_aspect_ratio),
            "mean_aspect_ratio": float(self.mean_aspect_ratio),
            "panels_below_30_deg": int(self.panels_below_30_deg),
            "panels_above_5_aspect": int(self.panels_above_5_aspect),
        }


def _compute_triangle_angles(vertices: FloatArray, faces: NDArray) -> FloatArray:
    """Compute interior angles for all triangles.

    Parameters
    ----------
    vertices : ndarray
        Vertex coordinates of shape (V, 3).
    faces : ndarray
        Face indices of shape (F, 3).

    Returns
    -------
    angles : ndarray
        Interior angles in degrees, shape (F, 3).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Edge vectors
    e01 = v1 - v0
    e02 = v2 - v0
    e10 = v0 - v1
    e12 = v2 - v1
    e20 = v0 - v2
    e21 = v1 - v2

    # Edge lengths
    l01 = np.linalg.norm(e01, axis=1)
    l02 = np.linalg.norm(e02, axis=1)
    l12 = np.linalg.norm(e12, axis=1)

    # Angles using dot product: cos(angle) = (a . b) / (|a| * |b|)
    # Angle at vertex 0
    cos_a0 = np.sum(e01 * e02, axis=1) / (l01 * l02)
    cos_a0 = np.clip(cos_a0, -1.0, 1.0)
    angle0 = np.degrees(np.arccos(cos_a0))

    # Angle at vertex 1
    cos_a1 = np.sum(e10 * e12, axis=1) / (l01 * l12)
    cos_a1 = np.clip(cos_a1, -1.0, 1.0)
    angle1 = np.degrees(np.arccos(cos_a1))

    # Angle at vertex 2
    cos_a2 = np.sum(e20 * e21, axis=1) / (l02 * l12)
    cos_a2 = np.clip(cos_a2, -1.0, 1.0)
    angle2 = np.degrees(np.arccos(cos_a2))

    return np.column_stack([angle0, angle1, angle2])


def _compute_aspect_ratios(vertices: FloatArray, faces: NDArray) -> FloatArray:
    """Compute aspect ratio (longest edge / shortest edge) for each triangle.

    Parameters
    ----------
    vertices : ndarray
        Vertex coordinates of shape (V, 3).
    faces : ndarray
        Face indices of shape (F, 3).

    Returns
    -------
    aspect_ratios : ndarray
        Aspect ratio for each triangle, shape (F,).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Edge lengths
    l01 = np.linalg.norm(v1 - v0, axis=1)
    l12 = np.linalg.norm(v2 - v1, axis=1)
    l20 = np.linalg.norm(v0 - v2, axis=1)

    # Stack and find min/max per triangle
    edges = np.column_stack([l01, l12, l20])
    min_edges = np.min(edges, axis=1)
    max_edges = np.max(edges, axis=1)

    # Avoid division by zero
    min_edges = np.maximum(min_edges, 1e-14)

    return max_edges / min_edges


def diagnose_mesh_quality(mesh: TriangleMesh) -> MeshQualityReport:
    """Compute comprehensive mesh quality diagnostics.

    Parameters
    ----------
    mesh : TriangleMesh
        The surface mesh to analyze.

    Returns
    -------
    MeshQualityReport
        Detailed quality metrics.
    """
    n = mesh.panel_count
    areas = mesh.areas

    # Area statistics
    min_area = float(np.min(areas))
    max_area = float(np.max(areas))
    area_ratio = max_area / min_area if min_area > 1e-14 else float("inf")

    # Angle statistics
    angles = _compute_triangle_angles(mesh.vertices, mesh.faces)
    min_angles_per_tri = np.min(angles, axis=1)

    min_angle_deg = float(np.min(angles))
    max_angle_deg = float(np.max(angles))
    mean_min_angle_deg = float(np.mean(min_angles_per_tri))

    # Aspect ratio statistics
    aspect_ratios = _compute_aspect_ratios(mesh.vertices, mesh.faces)
    min_aspect = float(np.min(aspect_ratios))
    max_aspect = float(np.max(aspect_ratios))
    mean_aspect = float(np.mean(aspect_ratios))

    # Count problematic panels
    panels_below_30 = int(np.sum(min_angles_per_tri < 30.0))
    panels_above_5 = int(np.sum(aspect_ratios > 5.0))

    return MeshQualityReport(
        mesh_panels=n,
        min_area=min_area,
        max_area=max_area,
        area_ratio=area_ratio,
        min_angle_deg=min_angle_deg,
        max_angle_deg=max_angle_deg,
        mean_min_angle_deg=mean_min_angle_deg,
        min_aspect_ratio=min_aspect,
        max_aspect_ratio=max_aspect,
        mean_aspect_ratio=mean_aspect,
        panels_below_30_deg=panels_below_30,
        panels_above_5_aspect=panels_above_5,
    )


def identify_low_quality_panels(
    mesh: TriangleMesh,
    min_angle_threshold: float = 30.0,
    max_aspect_threshold: float = 5.0,
) -> list[int]:
    """Identify panels with poor quality metrics.

    Returns indices of panels that have either:
    - Minimum interior angle < min_angle_threshold degrees
    - Aspect ratio > max_aspect_threshold

    Parameters
    ----------
    mesh : TriangleMesh
        The surface mesh.
    min_angle_threshold : float
        Minimum angle threshold in degrees.
    max_aspect_threshold : float
        Maximum aspect ratio threshold.

    Returns
    -------
    list of int
        Indices of problematic panels.
    """
    angles = _compute_triangle_angles(mesh.vertices, mesh.faces)
    min_angles = np.min(angles, axis=1)
    aspect_ratios = _compute_aspect_ratios(mesh.vertices, mesh.faces)

    bad_angle = min_angles < min_angle_threshold
    bad_aspect = aspect_ratios > max_aspect_threshold

    problematic = np.where(bad_angle | bad_aspect)[0]
    return problematic.tolist()


def mesh_quality_summary(mesh: TriangleMesh) -> str:
    """Generate a human-readable mesh quality summary.

    Parameters
    ----------
    mesh : TriangleMesh
        The surface mesh.

    Returns
    -------
    str
        Formatted quality report.
    """
    report = diagnose_mesh_quality(mesh)

    lines = [
        f"Mesh Quality Report ({report.mesh_panels} panels)",
        "=" * 50,
        f"Area range: [{report.min_area:.4e}, {report.max_area:.4e}]",
        f"Area ratio (max/min): {report.area_ratio:.2f}",
        "",
        f"Angle range: [{report.min_angle_deg:.2f}°, {report.max_angle_deg:.2f}°]",
        f"Mean min angle per triangle: {report.mean_min_angle_deg:.2f}°",
        f"Panels with min angle < 30°: {report.panels_below_30_deg}",
        "",
        f"Aspect ratio range: [{report.min_aspect_ratio:.2f}, {report.max_aspect_ratio:.2f}]",
        f"Mean aspect ratio: {report.mean_aspect_ratio:.2f}",
        f"Panels with aspect ratio > 5: {report.panels_above_5_aspect}",
    ]

    # Quality assessment
    if report.panels_below_30_deg > 0 or report.panels_above_5_aspect > 0:
        lines.append("")
        lines.append("⚠️  Warning: Mesh contains low-quality panels")
    elif report.area_ratio > 10:
        lines.append("")
        lines.append("⚠️  Warning: High area variation (non-uniform mesh)")
    else:
        lines.append("")
        lines.append("✓ Mesh quality is good")

    return "\n".join(lines)
