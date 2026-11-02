"""Waterplane properties for multi-loop (multi-hull) wetted surfaces.

A ``WettedHullMesh`` may pierce the calm free surface in one or more closed
waterline loops; its own contract explicitly allows "one or more closed oriented
waterline loops". A catamaran / twin-pontoon hull is the ship-architecture case
with two loops, and its hydrostatics must use the union of the loops: the heave
stiffness ``C[2,2] = rho g A_wp`` and the pitch stiffness
``C[4,4] = rho g grad GM_L`` are integrals over the whole waterplane.

The reference values below are computed independently of the library by
fan-triangulating each waterline polygon and integrating the exact triangle
moments, which is a different algorithm from the edge-wise Green's theorem sums
used in :func:`tensorbem.hydrostatic.compute_waterplane_properties`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry3d import WettedHullMesh, hemisphere_hull_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)

RADIUS = 1.0
PONTOON_OFFSET = 2.5
SUBDIVISIONS = 3
DENSITY = 1025.0
GRAVITY = 9.80665


def _merge(hulls: list[WettedHullMesh]) -> WettedHullMesh:
    """Merge disjoint wetted hulls into a single multi-loop wetted mesh."""
    vertices = []
    faces = []
    offset = 0
    for hull in hulls:
        vertices.append(hull.vertices)
        faces.append(hull.faces + offset)
        offset += hull.vertices.shape[0]
    return WettedHullMesh(np.vstack(vertices), np.vstack(faces))


def _twin_hull(offset: float = PONTOON_OFFSET) -> WettedHullMesh:
    return _merge(
        [
            hemisphere_hull_mesh(
                subdivisions=SUBDIVISIONS, radius=RADIUS, center_xy=(-offset, 0.0)
            ),
            hemisphere_hull_mesh(
                subdivisions=SUBDIVISIONS, radius=RADIUS, center_xy=(offset, 0.0)
            ),
        ]
    )


def _loop_polygons(hull: WettedHullMesh) -> list[np.ndarray]:
    """Group waterline vertices into polygons by connected component."""
    edges = np.asarray(hull.waterline_edges, dtype=int)
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        parent.setdefault(int(a), int(a))
        parent.setdefault(int(b), int(b))
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, set[int]] = {}
    for a, b in edges:
        groups.setdefault(find(int(a)), set()).update((int(a), int(b)))

    polygons = []
    for members in groups.values():
        points = hull.vertices[sorted(members)][:, :2]
        centre = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
        polygons.append(points[np.argsort(angles)])
    return polygons


def _fan_reference(polygons: list[np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact waterplane (area, centroid, [I_xx, I_yy, I_xy]) of the loop union."""
    area = 0.0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for polygon in polygons:
        p0 = polygon[0]
        for i in range(1, len(polygon) - 1):
            tri = np.array([p0, polygon[i], polygon[i + 1]])
            x = tri[:, 0]
            y = tri[:, 1]
            signed = 0.5 * (
                (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            )
            area += signed
            sx += signed * x.sum() / 3.0
            sy += signed * y.sum() / 3.0
            sxx += signed / 6.0 * (np.sum(x**2) + x[0] * x[1] + x[1] * x[2] + x[2] * x[0])
            syy += signed / 6.0 * (np.sum(y**2) + y[0] * y[1] + y[1] * y[2] + y[2] * y[0])
            sxy += signed / 12.0 * (x.sum() * y.sum() + np.sum(x * y))
    centroid = np.array([sx / area, sy / area])
    moments = np.array(
        [
            syy - area * centroid[1] ** 2,
            sxx - area * centroid[0] ** 2,
            sxy - area * centroid[0] * centroid[1],
        ]
    )
    return area, centroid, moments


def test_single_loop_waterplane_matches_fan_reference() -> None:
    hull = hemisphere_hull_mesh(subdivisions=SUBDIVISIONS, radius=RADIUS)
    area, centroid, moments = compute_waterplane_properties(hull)
    ref_area, ref_centroid, ref_moments = _fan_reference(_loop_polygons(hull))

    assert area == pytest.approx(ref_area, rel=1.0e-12)
    assert centroid == pytest.approx(ref_centroid, abs=1.0e-12)
    assert moments == pytest.approx(ref_moments, abs=1.0e-12)


def test_single_loop_waterplane_matches_closed_form_circle() -> None:
    hull = hemisphere_hull_mesh(subdivisions=4, radius=RADIUS)
    area, centroid, moments = compute_waterplane_properties(hull)
    # The waterline is a regular n-gon inscribed in the circle.
    n = int(np.asarray(hull.waterline_edges).shape[0])
    exact_area = 0.5 * n * RADIUS**2 * math.sin(2.0 * math.pi / n)
    assert area == pytest.approx(exact_area, rel=1.0e-12)
    assert area == pytest.approx(math.pi * RADIUS**2, rel=1.0e-2)
    assert centroid == pytest.approx([0.0, 0.0], abs=1.0e-12)
    # Transverse second moment of a circle: pi R^4 / 4 (inscribed polygon is 1-2% low)
    assert moments[0] == pytest.approx(math.pi * RADIUS**4 / 4.0, rel=2.0e-2)
    assert moments[1] == pytest.approx(moments[0], rel=1.0e-12)
    assert abs(moments[2]) < 1.0e-12


def test_twin_pontoon_waterplane_uses_both_loops() -> None:
    single = hemisphere_hull_mesh(subdivisions=SUBDIVISIONS, radius=RADIUS)
    twin = _twin_hull()

    area_single, _, moments_single = compute_waterplane_properties(single)
    area_twin, centroid_twin, moments_twin = compute_waterplane_properties(twin)

    # Superposition of two identical, translated waterplanes.
    assert area_twin == pytest.approx(2.0 * area_single, rel=1.0e-12)
    assert centroid_twin == pytest.approx([0.0, 0.0], abs=1.0e-12)
    assert moments_twin[0] == pytest.approx(2.0 * moments_single[0], rel=1.0e-12)
    assert moments_twin[1] == pytest.approx(
        2.0 * (moments_single[1] + area_single * PONTOON_OFFSET**2), rel=1.0e-12
    )
    assert abs(moments_twin[2]) < 1.0e-12

    # Independent fan-triangulation reference for the same union.
    ref_area, ref_centroid, ref_moments = _fan_reference(_loop_polygons(twin))
    assert area_twin == pytest.approx(ref_area, rel=1.0e-12)
    assert centroid_twin == pytest.approx(ref_centroid, abs=1.0e-12)
    assert moments_twin == pytest.approx(ref_moments, abs=1.0e-12)


def test_twin_pontoon_restoring_matrix_uses_pooled_waterplane() -> None:
    single = hemisphere_hull_mesh(subdivisions=SUBDIVISIONS, radius=RADIUS)
    twin = _twin_hull()

    props_single = compute_hydrostatic_restoring_matrix(
        single, mass=DENSITY * 2.0 * math.pi * RADIUS**3 / 3.0, center_of_gravity=[0, 0, -0.4]
    )
    props_twin = compute_hydrostatic_restoring_matrix(
        twin, mass=DENSITY * 2.0 * math.pi * RADIUS**3 / 3.0, center_of_gravity=[0, 0, -0.4]
    )

    # Heave stiffness is proportional to the waterplane area of the whole vessel.
    assert props_twin.restoring_matrix[2, 2] == pytest.approx(
        2.0 * props_single.restoring_matrix[2, 2], rel=1.0e-12
    )
    # Pitch stiffness follows GM_L built from the pooled longitudinal second moment.
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(twin)
    z_min = np.min(twin.vertices[:, 2])
    gm_l = (
        center_of_buoyancy[2]
        - z_min
        + props_twin.waterplane_moments[1] / volume
        - (-0.4 - z_min)
    )
    assert props_twin.restoring_matrix[4, 4] == pytest.approx(
        DENSITY * GRAVITY * volume * gm_l, rel=1.0e-12
    )
    # Twin-pontoon pitch stiffness must exceed the single hull by far more than 2x
    # because the longitudinal second moment picks up the A*d^2 offset term.
    assert props_twin.restoring_matrix[4, 4] > 10.0 * props_single.restoring_matrix[4, 4]


def test_non_manifold_waterline_is_rejected() -> None:
    hull = hemisphere_hull_mesh(subdivisions=1, radius=RADIUS)
    edges = np.asarray(hull.waterline_edges, dtype=np.int64)
    broken = object.__new__(WettedHullMesh)
    object.__setattr__(broken, "vertices", hull.vertices)
    object.__setattr__(broken, "faces", hull.faces)
    object.__setattr__(
        broken, "waterline_edges", np.vstack([edges, edges[:1] + 0])
    )
    with pytest.raises(ValueError, match="closed loops"):
        compute_waterplane_properties(broken)