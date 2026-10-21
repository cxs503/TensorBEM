"""Constant-panel collocation solver for 3-D Laplace boundary-value problems.

Boundary integral equation at a smooth collocation point (panel centroid):
    c*u = integral(G*q dS) - integral((dG/dn_y)*u dS),  c = 1/2.

With ``G = 1/(4*pi*r)`` and outward panel normals, the single-layer matrix
``S`` and double-layer matrix ``H`` satisfy

    interior Dirichlet:  S q = (1/2 I + H) u
    exterior Dirichlet:  S q = (1/2 I - H) u

where for the exterior problem ``q`` is the derivative along the fluid-domain
normal (opposite to the body-outward normal). The double-layer entries are
computed analytically from signed solid angles; the single-layer self terms
use a polar quadrature about the centroid and all other entries use a
Duffy-transformed Gauss rule. The dense solve is intended for verification
meshes, not large production models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh
from .kernels3d import (
    duffy_quadrature_batch,
    gradient_single_layer_triangle_self,
    gradient_single_layer_triangles_batch,
    gradient_solid_angle_triangles_batch,
    integrate_single_layer_triangle_self,
    solid_angle_triangles_batch,
)

FloatArray = NDArray[np.float64]
_FOUR_PI = 4.0 * np.pi
_DOMAINS = ("interior", "exterior")


@dataclass(frozen=True)
class LaplaceSurfaceResult:
    """Solution and matrices from a constant-triangle collocation solve."""

    normal_derivative: FloatArray
    boundary_values: FloatArray
    single_layer_matrix: FloatArray
    double_layer_matrix: FloatArray
    condition_number: float
    residual_l2: float
    relative_residual_l2: float
    double_layer_row_sum_error: float
    domain: str
    quadrature_order: int


def _validate_domain(domain: str) -> str:
    if domain not in _DOMAINS:
        raise ValueError(f"domain must be one of {_DOMAINS}")
    return domain


def _validate_order(order: int) -> int:
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("quadrature_order must be an integer >= 2")
    if order < 2:
        raise ValueError("quadrature_order must be an integer >= 2")
    return int(order)


def assemble_surface_matrices(
    mesh: TriangleMesh, quadrature_order: int = 8
) -> tuple[FloatArray, FloatArray]:
    """Assemble analytically integrated surface single- and double-layer matrices.

    The double-layer matrix is exact for flat constant panels (signed solid
    angles). The single-layer matrix uses exact polar integration on each
    collocation panel and Duffy quadrature elsewhere.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    order = _validate_order(quadrature_order)
    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    centroids = mesh.centroids
    normal_points, weights = duffy_quadrature_batch(v0, v1, v2, order=order)
    panel_count = mesh.panel_count
    single = np.empty((panel_count, panel_count), dtype=float)
    double = np.empty((panel_count, panel_count), dtype=float)
    self_integrals = np.array(
        [
            integrate_single_layer_triangle_self(mesh.vertices[mesh.faces[index]])
            for index in range(panel_count)
        ]
    )
    for index, target in enumerate(centroids):
        delta = target[None, None, :] - normal_points
        distances = np.linalg.norm(delta, axis=2)
        single[index] = np.sum(weights / (_FOUR_PI * distances), axis=1)
        solid_angles = solid_angle_triangles_batch(target, v0, v1, v2)
        double[index] = -solid_angles / _FOUR_PI
        single[index, index] = self_integrals[index]
        double[index, index] = 0.0
    return single, double


def solve_laplace_surface(
    mesh: TriangleMesh,
    boundary_values: FloatArray,
    domain: str = "interior",
    quadrature_order: int = 8,
) -> LaplaceSurfaceResult:
    """Solve the interior or exterior 3-D Laplace Dirichlet problem.

    ``boundary_values`` are constant per triangle, prescribed at centroids.
    ``domain='exterior'`` returns the trace of ``dG/dn`` along the fluid-domain
    normal, i.e. the negative of the body-outward normal derivative.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    domain = _validate_domain(domain)
    values = np.asarray(boundary_values, dtype=float)
    panel_count = mesh.panel_count
    if values.shape != (panel_count,):
        raise ValueError(f"boundary_values must have shape ({panel_count},)")
    if not np.isfinite(values).all():
        raise ValueError("boundary_values must all be finite")

    single, double = assemble_surface_matrices(mesh, quadrature_order=quadrature_order)
    if domain == "interior":
        rhs = 0.5 * values + double @ values
    else:
        rhs = 0.5 * values - double @ values
    condition = float(np.linalg.cond(single))
    q = np.linalg.solve(single, rhs)
    residual = float(np.linalg.norm(single @ q - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    return LaplaceSurfaceResult(
        normal_derivative=q,
        boundary_values=values.copy(),
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_number=condition,
        residual_l2=residual,
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0.0 else residual,
        double_layer_row_sum_error=float(np.max(np.abs(np.sum(double, axis=1) + 0.5))),
        domain=domain,
        quadrature_order=quadrature_order,
    )


def _minimum_distance_to_mesh(mesh: TriangleMesh, point: FloatArray) -> float:
    """Distance from a point to the closed triangulated surface."""
    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    normals = mesh.normals
    e1 = v1 - v0
    e2 = v2 - v0
    delta = point - v0
    c11 = np.einsum("ij,ij->i", e1, e1)
    c12 = np.einsum("ij,ij->i", e1, e2)
    c22 = np.einsum("ij,ij->i", e2, e2)
    d1 = np.einsum("ij,ij->i", delta, e1)
    d2 = np.einsum("ij,ij->i", delta, e2)
    determinant = c11 * c22 - c12 * c12
    b1 = (d1 * c22 - d2 * c12) / determinant
    b2 = (d2 * c11 - d1 * c12) / determinant
    b0 = 1.0 - b1 - b2
    inside = (b0 >= -1.0e-12) & (b1 >= -1.0e-12) & (b2 >= -1.0e-12)
    plane_distance = np.abs(np.einsum("ij,ij->i", delta, normals))

    shortest = np.full(mesh.panel_count, np.inf)
    lower = 0.0
    for start, end in ((v0, v1), (v1, v2), (v2, v0)):
        edge = end - start
        edge_length_squared = np.einsum("ij,ij->i", edge, edge)
        parameter = np.einsum("ij,ij->i", point - start, edge) / edge_length_squared
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = start + parameter[:, None] * edge
        shortest = np.minimum(shortest, np.linalg.norm(point - closest, axis=1))
    shortest = np.where(inside, plane_distance, shortest)
    return float(np.min(shortest))


def evaluate_surface_field(
    mesh: TriangleMesh,
    result: LaplaceSurfaceResult,
    points: FloatArray,
    domain: str | None = None,
    quadrature_order: int = 8,
    minimum_clearance_factor: float = 0.05,
) -> FloatArray:
    """Evaluate the harmonic field at points away from the closed surface.

    Interior points use ``u = S q - H u_boundary``; exterior points use
    ``u = S q + H u_boundary``. The caller is responsible for supplying points
    that lie in the stated domain (inside for interior, outside for exterior);
    domain membership is not checked. Points closer than
    ``minimum_clearance_factor * h_max`` to the surface are rejected because
    the fixed-order single-layer quadrature is not a near-singular scheme.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if not isinstance(result, LaplaceSurfaceResult):
        raise TypeError("result must be a LaplaceSurfaceResult")
    if domain is None:
        domain = result.domain
    domain = _validate_domain(domain)
    if domain != result.domain:
        raise ValueError("domain must match the result's boundary-value problem")
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")
    if (
        result.normal_derivative.shape != (mesh.panel_count,)
        or result.boundary_values.shape != (mesh.panel_count,)
    ):
        raise ValueError("result boundary traces must match the mesh panel count")
    order = _validate_order(quadrature_order)
    points_quad, weights = duffy_quadrature_batch(
        mesh.v0, mesh.v1, mesh.v2, order=order
    )
    clearance = minimum_clearance_factor * mesh.h_max
    values = np.empty(len(targets), dtype=float)
    for index, target in enumerate(targets):
        if _minimum_distance_to_mesh(mesh, target) < clearance:
            raise ValueError(
                "evaluation point is too close to the boundary for the fixed-order "
                "field quadrature; choose a point farther from the surface"
            )
        delta = target[None, None, :] - points_quad
        distances = np.linalg.norm(delta, axis=2)
        single_row = np.sum(weights / (_FOUR_PI * distances), axis=1)
        solid_angles = solid_angle_triangles_batch(target, mesh.v0, mesh.v1, mesh.v2)
        double_row = -solid_angles / _FOUR_PI
        contribution = single_row @ result.normal_derivative
        if domain == "interior":
            contribution -= double_row @ result.boundary_values
        else:
            contribution += double_row @ result.boundary_values
        values[index] = contribution
    return values


def evaluate_surface_velocity(
    mesh: TriangleMesh,
    result: LaplaceSurfaceResult,
    points: FloatArray,
    domain: str | None = None,
    quadrature_order: int = 8,
    minimum_clearance_factor: float = 0.05,
) -> FloatArray:
    """Evaluate the harmonic velocity ``grad(phi)`` at points away from the surface.

    The velocity is obtained by differentiating the Green representation used
    by :func:`evaluate_surface_field`:

    * the single-layer term contributes
      ``sum_j q_j * integral_{T_j} grad_x G dS``, whose kernel
      ``-(x - y)/(4*pi*|x - y|**3)`` is integrable and is evaluated with the
      same fixed-order Duffy rule;
    * the double-layer term contributes ``sum_j u_j * grad(Omega_j)/(4*pi)``
      for the interior and its negative for the exterior, where ``Omega_j`` is
      the signed solid angle of panel ``j``.  The hypersingular kernel
      ``grad_x(dG/dn_y)`` is not integrable by ordinary quadrature, but the
      integral of a constant density over a flat panel is the gradient of the
      panel's closed-form solid angle, which is what is used here.

    Returns an array of shape ``(M, 3)``.  Points closer than
    ``minimum_clearance_factor * h_max`` to the surface are rejected, for the
    same reason as in :func:`evaluate_surface_field`.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if not isinstance(result, LaplaceSurfaceResult):
        raise TypeError("result must be a LaplaceSurfaceResult")
    if domain is None:
        domain = result.domain
    domain = _validate_domain(domain)
    if domain != result.domain:
        raise ValueError("domain must match the result's boundary-value problem")
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")
    if (
        result.normal_derivative.shape != (mesh.panel_count,)
        or result.boundary_values.shape != (mesh.panel_count,)
    ):
        raise ValueError("result boundary traces must match the mesh panel count")
    order = _validate_order(quadrature_order)
    clearance = minimum_clearance_factor * mesh.h_max
    values = np.empty((len(targets), 3), dtype=float)
    for index, target in enumerate(targets):
        if _minimum_distance_to_mesh(mesh, target) < clearance:
            raise ValueError(
                "evaluation point is too close to the boundary for the fixed-order "
                "field quadrature; choose a point farther from the surface"
            )
        single_gradients = gradient_single_layer_triangles_batch(
            target, mesh.v0, mesh.v1, mesh.v2, order=order
        )
        contribution = result.normal_derivative @ single_gradients
        solid_angle_gradients = gradient_solid_angle_triangles_batch(
            target, mesh.v0, mesh.v1, mesh.v2
        )
        double_term = (result.boundary_values @ solid_angle_gradients) / _FOUR_PI
        if domain == "interior":
            contribution += double_term
        else:
            contribution -= double_term
        values[index] = contribution
    return values


def evaluate_surface_velocity_at_panels(
    mesh: TriangleMesh,
    result: LaplaceSurfaceResult,
    quadrature_order: int = 8,
) -> FloatArray:
    """Evaluate the velocity on each panel centroid, from the stated domain's side.

    The single-layer self term is evaluated with the polar rule that removes
    the logarithmic-type singularity of the ``1/r`` kernel; the double-layer
    self term uses the closed-form solid-angle gradient, which is continuous
    across the panel plane.

    The constant-panel trace makes the reconstructed gradient degrade as the
    evaluation point approaches the surface, so this quantity is a diagnostic
    rather than an accuracy claim; see the module notes in
    :mod:`tensorbem.benchmarks.sphere_velocity`.

    Returns an array of shape ``(F, 3)``.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if not isinstance(result, LaplaceSurfaceResult):
        raise TypeError("result must be a LaplaceSurfaceResult")
    order = _validate_order(quadrature_order)
    sign = 1.0 if result.domain == "interior" else -1.0
    panel_count = mesh.panel_count
    values = np.empty((panel_count, 3), dtype=float)
    for index in range(panel_count):
        target = mesh.centroids[index]
        single_gradients = gradient_single_layer_triangles_batch(
            target, mesh.v0, mesh.v1, mesh.v2, order=order
        )
        single_gradients[index] = gradient_single_layer_triangle_self(
            mesh.vertices[mesh.faces[index]]
        )
        solid_angle_gradients = gradient_solid_angle_triangles_batch(
            target, mesh.v0, mesh.v1, mesh.v2
        )
        values[index] = (
            result.normal_derivative @ single_gradients
            + sign * (result.boundary_values @ solid_angle_gradients) / _FOUR_PI
        )
    return values
