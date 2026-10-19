"""Three-dimensional Laplace boundary-integral kernels and triangle quadrature.

Conventions
-----------
``G(x, y) = 1 / (4*pi*|x-y|)`` satisfies ``-Laplacian(G) = delta``.
The double-layer kernel is ``dG/dn_y = (x-y).n_y / (4*pi*|x-y|**3)``.
Panel vertices are ordered consistently with the outward unit normal.

The double-layer integral over a flat triangle is evaluated analytically from
its signed solid angle. The single-layer self-panel integral is evaluated by
polar integration about the centroid, which removes the logarithmic-type
singularity. Non-self single-layer panels use a Duffy-transformed tensor
Gauss rule.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_FOUR_PI = 4.0 * np.pi


def fundamental_solution_3d(distance: FloatArray | float) -> FloatArray:
    r"""Return ``G(r) = 1/(4*pi*r)``, satisfying ``-Laplacian(G)=delta``."""
    r = np.asarray(distance, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("distance must be positive; handle singular panel integrals separately")
    return 1.0 / (_FOUR_PI * r)


def double_layer_kernel_3d(
    target: FloatArray, source: FloatArray, source_normal: FloatArray
) -> float:
    r"""Return ``dG/dn_y = (x-y).n_y/(4*pi*|x-y|**3)`` for ``x != y``."""
    delta = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    r2 = float(np.dot(delta, delta))
    if r2 <= 0.0:
        raise ValueError("double-layer point kernel is singular at coincident points")
    return float(np.dot(delta, source_normal) / (_FOUR_PI * r2 * math.sqrt(r2)))


def _validate_vertices(vertices: FloatArray) -> FloatArray:
    points = np.asarray(vertices, dtype=float)
    if points.shape != (3, 3):
        raise ValueError("triangle vertices must have shape (3, 3)")
    if not np.isfinite(points).all():
        raise ValueError("triangle vertices must all be finite")
    area = 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
    if area <= 0.0:
        raise ValueError("triangle must be non-degenerate")
    return points


def triangle_area(vertices: FloatArray) -> float:
    """Return the area of one triangle."""
    points = _validate_vertices(vertices)
    return 0.5 * float(
        np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
    )


def _validate_order(order: int) -> int:
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("quadrature order must be an integer >= 2")
    if order < 2:
        raise ValueError("quadrature order must be an integer >= 2")
    return int(order)


def duffy_quadrature(vertices: FloatArray, order: int = 8) -> tuple[FloatArray, FloatArray]:
    """Return Duffy-transformed Gauss points and weights on one triangle.

    The map is ``y = v0 + u*((1-v)*(v1-v0) + v*(v2-v0))`` with Jacobian
    ``2*area*u``. The returned weights integrate a constant exactly.
    """
    points = _validate_vertices(vertices)
    order = _validate_order(order)
    v0, v1, v2 = points
    area = 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
    nodes, gauss_weights = leggauss(order)
    u = 0.5 * (nodes + 1.0)
    v = 0.5 * (nodes + 1.0)
    du = 0.5 * gauss_weights
    dv = 0.5 * gauss_weights
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    samples = (
        v0
        + grid_u[:, :, None] * (1.0 - grid_v)[:, :, None] * (v1 - v0)
        + grid_u[:, :, None] * grid_v[:, :, None] * (v2 - v0)
    )
    weights = (
        2.0
        * area
        * grid_u
        * du[:, None]
        * dv[None, :]
    )
    return samples.reshape(-1, 3), weights.reshape(-1)


def duffy_quadrature_batch(
    v0: FloatArray, v1: FloatArray, v2: FloatArray, order: int = 8
) -> tuple[FloatArray, FloatArray]:
    """Vectorized Duffy quadrature for ``F`` triangles.

    Returns arrays of shape ``(F, order**2, 3)`` and ``(F, order**2)``.
    """
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if v0.ndim != 2 or v0.shape[1] != 3 or v1.shape != v0.shape or v2.shape != v0.shape:
        raise ValueError("v0, v1, and v2 must all have shape (F, 3)")
    if not (np.isfinite(v0).all() and np.isfinite(v1).all() and np.isfinite(v2).all()):
        raise ValueError("triangle vertices must all be finite")
    if np.any(
        np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) <= 0.0
    ):
        raise ValueError("triangles must be non-degenerate")
    order = _validate_order(order)
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    nodes, gauss_weights = leggauss(order)
    u = 0.5 * (nodes + 1.0)
    v = 0.5 * (nodes + 1.0)
    du = 0.5 * gauss_weights
    dv = 0.5 * gauss_weights
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    samples = (
        v0[:, None, None, :]
        + grid_u[None, :, :, None] * (1.0 - grid_v)[None, :, :, None] * (v1 - v0)[:, None, None, :]
        + grid_u[None, :, :, None] * grid_v[None, :, :, None] * (v2 - v0)[:, None, None, :]
    )
    weights = (
        2.0
        * areas[:, None, None]
        * grid_u[None, :, :]
        * du[None, :, None]
        * dv[None, None, :]
    )
    panel_count = v0.shape[0]
    return samples.reshape(panel_count, -1, 3), weights.reshape(panel_count, -1)


def _plane_geometry(
    target: FloatArray, vertices: FloatArray
) -> tuple[float, FloatArray, float]:
    """Return area, unit normal, and signed plane distance for one triangle."""
    points = _validate_vertices(vertices)
    x = np.asarray(target, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("target must be a finite array of shape (3,)")
    v0, v1, v2 = points
    cross = np.cross(v1 - v0, v2 - v0)
    double_area = float(np.linalg.norm(cross))
    area = 0.5 * double_area
    normal = cross / double_area
    distance = float(np.dot(x - v0, normal))
    return area, normal, distance


def _barycentric_in_plane(target: FloatArray, vertices: FloatArray, normal: FloatArray) -> FloatArray:
    """Barycentric coordinates of the plane projection of ``target``."""
    v0, v1, v2 = vertices
    projected = target - np.dot(target - v0, normal) * normal
    e1 = v1 - v0
    e2 = v2 - v0
    delta = projected - v0
    c11 = float(np.dot(e1, e1))
    c12 = float(np.dot(e1, e2))
    c22 = float(np.dot(e2, e2))
    d1 = float(np.dot(delta, e1))
    d2 = float(np.dot(delta, e2))
    determinant = c11 * c22 - c12 * c12
    if determinant <= 0.0:
        raise ValueError("degenerate triangle in barycentric projection")
    b1 = (d1 * c22 - d2 * c12) / determinant
    b2 = (d2 * c11 - d1 * c12) / determinant
    return np.array([1.0 - b1 - b2, b1, b2])


def integrate_single_layer_triangle(
    target: FloatArray, vertices: FloatArray, order: int = 8
) -> float:
    """Integrate ``G`` over a flat triangle that does not contain the target.

    A target lying in the panel plane and inside the triangle is singular and
    rejected; use :func:`integrate_single_layer_triangle_self` for that case.
    """
    vertices = _validate_vertices(vertices)
    area, normal, distance = _plane_geometry(target, vertices)
    if abs(distance) <= 1.0e-12 * math.sqrt(area):
        barycentric = _barycentric_in_plane(np.asarray(target, dtype=float), vertices, normal)
        if np.all(barycentric >= -1.0e-10):
            raise ValueError(
                "single-layer panel integral is singular inside the panel; "
                "use integrate_single_layer_triangle_self for the collocation panel"
            )
    points, weights = duffy_quadrature(vertices, order=order)
    distances = np.linalg.norm(np.asarray(target, dtype=float) - points, axis=1)
    return float(np.sum(weights / (_FOUR_PI * distances)))


def integrate_single_layer_triangle_self(vertices: FloatArray, order: int = 64) -> float:
    r"""Integrate ``G`` over a flat triangle at its centroid.

    The triangle is split into three polar sectors about the centroid. In each
    sector ``dS = r dr dtheta`` and the ``1/r`` kernel reduces to an inner
    integral over ``dr`` equal to the radial extent ``R(theta)``, so the only
    numerical quadrature is one-dimensional and smooth.
    """
    points = _validate_vertices(vertices)
    order = _validate_order(order)
    centroid = points.mean(axis=0)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal /= np.linalg.norm(normal)
    nodes, weights = leggauss(order)
    total = 0.0
    for index in range(3):
        a = points[index] - centroid
        b = points[(index + 1) % 3] - centroid
        e1 = a / np.linalg.norm(a)
        e2 = np.cross(normal, e1)
        a2 = np.array([np.dot(a, e1), np.dot(a, e2)])
        b2 = np.array([np.dot(b, e1), np.dot(b, e2)])
        angle_a = math.atan2(a2[1], a2[0])
        angle_b = math.atan2(b2[1], b2[0])
        dtheta = (angle_b - angle_a + math.pi) % (2.0 * math.pi) - math.pi
        edge = b2 - a2
        edge_normal = np.array([edge[1], -edge[0]])
        offset = float(np.dot(edge_normal, a2))
        if offset < 0.0:
            edge_normal = -edge_normal
            offset = -offset
        if offset <= 0.0:
            raise ValueError("centroid does not lie strictly inside the triangle")
        theta = angle_a + 0.5 * dtheta * (nodes + 1.0)
        cosine = np.cos(theta) * edge_normal[0] + np.sin(theta) * edge_normal[1]
        radial_extent = offset / cosine
        total += 0.5 * dtheta * float(np.dot(weights, radial_extent))
    return total / _FOUR_PI


def solid_angle_triangle(target: FloatArray, vertices: FloatArray) -> float:
    """Return the signed solid angle of one oriented triangle seen from ``target``.

    The sign follows the vertex ordering: it is positive when the vertices are
    counter-clockwise as seen from the target. A coplanar target gives zero or
    a degenerate ``2*pi`` value; the double-layer principal value in that case
    is handled separately.
    """
    points = _validate_vertices(vertices)
    x = np.asarray(target, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("target must be a finite array of shape (3,)")
    a = points[0] - x
    b = points[1] - x
    c = points[2] - x
    determinant = float(np.dot(a, np.cross(b, c)))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    norm_c = float(np.linalg.norm(c))
    denominator = (
        norm_a * norm_b * norm_c
        + float(np.dot(a, b)) * norm_c
        + float(np.dot(b, c)) * norm_a
        + float(np.dot(c, a)) * norm_b
    )
    return 2.0 * math.atan2(determinant, denominator)


def solid_angle_triangles_batch(
    target: FloatArray, v0: FloatArray, v1: FloatArray, v2: FloatArray
) -> FloatArray:
    """Signed solid angles of ``F`` triangles seen from one target point."""
    x = np.asarray(target, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("target must be a finite array of shape (3,)")
    if v0.ndim != 2 or v0.shape[1] != 3 or v1.shape != v0.shape or v2.shape != v0.shape:
        raise ValueError("v0, v1, and v2 must all have shape (F, 3)")
    if not (np.isfinite(v0).all() and np.isfinite(v1).all() and np.isfinite(v2).all()):
        raise ValueError("triangle vertices must all be finite")
    a = v0 - x
    b = v1 - x
    c = v2 - x
    determinant = np.einsum("ij,ij->i", a, np.cross(b, c))
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    norm_c = np.linalg.norm(c, axis=1)
    denominator = (
        norm_a * norm_b * norm_c
        + np.einsum("ij,ij->i", a, b) * norm_c
        + np.einsum("ij,ij->i", b, c) * norm_a
        + np.einsum("ij,ij->i", c, a) * norm_b
    )
    return 2.0 * np.arctan2(determinant, denominator)


def integrate_double_layer_triangle(target: FloatArray, vertices: FloatArray) -> float:
    """Integrate ``dG/dn_y`` exactly over a flat triangle.

    For a non-coplanar target this equals ``-Omega/(4*pi)``. For a target in
    the panel plane the principal value is zero.
    """
    vertices = _validate_vertices(vertices)
    area, _, distance = _plane_geometry(target, vertices)
    if abs(distance) <= 1.0e-12 * math.sqrt(area):
        return 0.0
    return -solid_angle_triangle(target, vertices) / _FOUR_PI
