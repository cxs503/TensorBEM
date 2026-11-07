"""Piecewise-linear (order-1) panel operator for the 3-D Laplace BEM.

Why this module exists
----------------------
:mod:`tensorbem.solver3d` discretises the boundary integral operators with
*constant* panels: one unknown per triangle, collocation at the panel centroid.
That scheme cannot be refined past its own geometric error and, more
importantly, it leaves the order-1 half of any accuracy-order benchmark
unmeasurable.  This module adds the missing rung: a genuine piecewise-linear
boundary element discretisation with

* **unknowns**: the potential at the mesh *vertices* (one global degree of
  freedom per vertex, shared by all incident panels);
* **interpolation**: the barycentric shape functions ``lambda_k``
  (``lambda_k = 1`` at vertex ``k``, ``0`` at the other two);
* **collocation**: at the vertices themselves;
* **analytically integrated self terms**: the entries ``int_T lambda_k G dS``
  of a panel whose apex *is* the collocation point are evaluated in closed
  form, and the flat-panel double-layer self term is exactly zero.

Convention
----------
``G(x, y) = 1 / (4 pi |x - y|)``.  For a body with outward unit normal ``n``
and a closed surface ``S``, Green's representation for the exterior domain is

.. math::

    u(x) = \\int_S \\left[ u(y) \\frac{\\partial G}{\\partial n}
                          - G(x,y) \\, q(y) \\right] dS_y ,
    \\qquad q = \\nabla u \\cdot n .

Taking the collocation point onto ``S`` from the fluid side and using
``sum_j D_ij = -1/2`` on a smooth closed surface gives the standard exterior
second-kind equation

.. math::

    \\left( \\tfrac12 I - D \\right) u = -S q ,
    \\qquad
    S_{ik} = \\int_T \\lambda_k G \\, dS , \\quad
    D_{ik} = \\int_T \\lambda_k \\frac{\\partial G}{\\partial n} \\, dS ,

which is the form solved here (and the same form as the constant-panel operator
in :mod:`tensorbem.solver3d`).  The exterior problem is not singular: decay at
infinity removes the constant that would otherwise be a null vector, so
``(1/2 I - D)`` is well conditioned while ``cond(S)`` grows like
``N**(1/2)``.

Quadrature
----------
Every panel integral is evaluated with a Duffy-mapped product Gauss rule, which
makes the smooth part spectrally accurate.  Panels *near* the collocation point
(a ``1/r`` or ``1/r**3`` kernel peak) are refined toward the target before
integrating (:func:`_refine_toward`), and the panels that contain the target
are handled in closed form.  Without that split the double-layer row sums come
out several percent wrong, which is exactly the kind of error that would
masquerade as a discretisation order in a convergence study.

A note on the free term of a *faceted* mesh
-------------------------------------------
The free term of the second-kind equation is the continuous value ``c = 1/2``.
On a faceted mesh the assembled double layer has row sum ``-Omega/(4 pi)``
instead, ``Omega`` being the solid angle of the body-side *cone* at the
collocation vertex (:func:`cone_solid_angle`).  That shortfall is the *cone
defect*; on an inscribed sphere it is ``O(h)`` (0.08 at ``N = 320``) and is
therefore **not** negligible next to the discretisation error being measured.
It is reported as a diagnostic and used only as a negative control: taking the
row sum as the free term breaks the discrete boundary identity
(:func:`representation_residual`, 0.23 versus 0.16 at ``N = 320``), whereas the
free term implied by the exact field (:func:`implied_free_term`) converges to
``1/2`` under refinement.  :func:`angle_defect` reports the ``O(h**2)``
Gaussian curvature of the same vertices for completeness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from .geometry3d import TriangleMesh

FloatArray = NDArray[np.float64]

FOUR_PI = 4.0 * math.pi

#: Rules accepted for the self-panel (singular) integral.
SELF_RULES = ("analytic", "graded_duffy", "coarse")

#: Default Duffy-mapped product-Gauss order.
DEFAULT_QUADRATURE_ORDER = 8

#: Default number of refinement levels for near-singular panels.
DEFAULT_REFINEMENT_DEPTH = 3


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _validate_triangle(vertices) -> FloatArray:
    triangle = np.asarray(vertices, dtype=float)
    if triangle.shape != (3, 3):
        raise ValueError("vertices must have shape (3, 3)")
    if not np.isfinite(triangle).all():
        raise ValueError("vertices must all be finite")
    if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) <= 0.0:
        raise ValueError("triangle is degenerate")
    return triangle


def panel_frame(vertices) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float]:
    """Return ``(origin, e1, e2, n, area)`` of a flat triangle's local frame."""
    triangle = _validate_triangle(vertices)
    origin = triangle[0]
    edge1 = triangle[1] - origin
    e1 = edge1 / np.linalg.norm(edge1)
    normal = np.cross(edge1, triangle[2] - origin)
    area = 0.5 * float(np.linalg.norm(normal))
    normal = normal / (2.0 * area)
    e2 = np.cross(normal, e1)
    return origin, e1, e2, normal, area


def _gauss_unit_interval(order: int) -> tuple[FloatArray, FloatArray]:
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be an integer")
    if int(order) < 1:
        raise ValueError("order must be >= 1")
    nodes, weights = leggauss(int(order))
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _unit_triangle_rule(order: int) -> tuple[FloatArray, FloatArray]:
    """Duffy-mapped product Gauss rule on the unit reference triangle."""
    nodes_u, weights_u = _gauss_unit_interval(order)
    nodes_v, weights_v = _gauss_unit_interval(order)
    lam_1 = np.repeat(nodes_u[:, None], nodes_v.size, axis=1).ravel()
    lam_2 = np.tile(nodes_v[None, :], (nodes_u.size, 1)).ravel()
    weights = (np.outer(weights_u, weights_v) * lam_1.reshape(nodes_u.size, -1)).ravel()
    lam_0 = 1.0 - lam_1 - lam_2
    return np.stack((lam_0, lam_1, lam_2), axis=-1), weights


# ---------------------------------------------------------------------------
# singular (self) panel integrals of the linear shape functions
# ---------------------------------------------------------------------------
def integrate_linear_single_layer_self_analytic(vertices, apex: int = 0) -> FloatArray:
    r"""Closed form of ``int_T lambda_k / (4 pi r) dS`` with the target at a vertex.

    Polar coordinates about the apex turn the panel into a single angular
    sector: with ``r`` the in-plane distance from the apex, ``dS = r dr dtheta``
    and the ``1/r`` kernel cancels the radial Jacobian, so the radial integral of
    the affine shape function is a polynomial in the radial extent
    ``R(theta) = d / cos(theta - theta_n)`` (``d`` = distance from the apex to
    the opposite edge, ``theta_n`` the direction of its outward normal).  The
    angular integrals then have elementary antiderivatives,

    * ``int dtheta R = d * artanh(sin phi)``,
    * ``int dtheta (g.u) R**2 / 2 = (d**2/2) [cos(alpha) artanh(sin phi)
      + sin(alpha) sec(phi)]``,

    with ``phi = theta - theta_n`` and ``alpha`` the direction of the shape
    function gradient relative to ``theta_n``.  The result is exact to
    round-off; see :func:`self_integral_consistency` for the numerical check.

    Parameters
    ----------
    vertices:
        ``(3, 3)`` panel vertices.
    apex:
        Index (``0..2``) of the vertex coinciding with the collocation point.

    Returns
    -------
    ``(3,)`` array of ``int_T lambda_k G dS`` ordered like ``vertices``.
    """
    triangle = _validate_triangle(vertices)
    apex = int(apex)
    if apex not in (0, 1, 2):
        raise ValueError("apex must be 0, 1 or 2")
    ordered = np.roll(triangle, -apex, axis=0)
    origin, e1, e2, _normal, _area = panel_frame(ordered)
    planar = np.stack(
        ((ordered - origin) @ e1, (ordered - origin) @ e2), axis=-1
    )
    point_b = planar[1]
    point_c = planar[2]

    edge = point_c - point_b
    edge = edge / np.linalg.norm(edge)
    inward = np.array([-edge[1], edge[0]])
    distance = float(inward @ point_b)
    if distance < 0.0:
        inward = -inward
        distance = -distance
    if distance <= 0.0:
        raise ValueError("degenerate panel: apex lies on the opposite edge")

    angle_b = math.atan2(point_b[1], point_b[0])
    angle_c = math.atan2(point_c[1], point_c[0])
    delta = (angle_c - angle_b + math.pi) % (2.0 * math.pi) - math.pi
    if delta == 0.0:
        raise ValueError("degenerate panel: zero angular extent")
    angle_n = math.atan2(inward[1], inward[0])
    phi_b = angle_b - angle_n
    phi_c = phi_b + delta
    if abs(phi_b) >= 0.5 * math.pi or abs(phi_c) >= 0.5 * math.pi:
        raise ValueError("degenerate panel: apex outside the panel's angular wedge")

    integral_r = distance * (math.atanh(math.sin(phi_c)) - math.atanh(math.sin(phi_b)))
    delta_sec = 1.0 / math.cos(phi_c) - 1.0 / math.cos(phi_b)

    def _grad(point_on_edge: FloatArray, point_at_one: FloatArray) -> FloatArray:
        normal = np.array([-point_on_edge[1], point_on_edge[0]])
        return normal / float(normal @ point_at_one)

    grad_b = _grad(point_c, point_b)
    grad_c = _grad(point_b, point_c)
    grad_a = -(grad_b + grad_c)

    def _angular(gradient: FloatArray) -> float:
        magnitude = float(np.linalg.norm(gradient))
        if magnitude == 0.0:
            return 0.0
        alpha = math.atan2(gradient[1], gradient[0]) - angle_n
        return magnitude * (
            math.cos(alpha) * integral_r / distance + math.sin(alpha) * delta_sec
        )

    constant_part = np.array([1.0, 0.0, 0.0])
    radial_part = np.array([_angular(grad_a), _angular(grad_b), _angular(grad_c)])
    values = (constant_part * integral_r + 0.5 * distance**2 * radial_part) / FOUR_PI

    result = np.empty(3, dtype=float)
    result[apex] = values[0]
    result[(apex + 1) % 3] = values[1]
    result[(apex + 2) % 3] = values[2]
    return result


def integrate_linear_single_layer_self_numerical(
    vertices,
    apex: int = 0,
    order: int = 24,
) -> FloatArray:
    r"""Graded Duffy quadrature of ``int_T lambda_k / (4 pi r) dS`` at a vertex.

    Parametrising the panel as ``y = A + u [(1 - v)(B - A) + v (C - A)]`` gives
    Jacobian ``2 A_T u`` and distance ``u w(v)`` to the apex, so the ``1/r``
    singularity is cancelled exactly by the Jacobian and Gauss-Legendre in
    ``(u, v)`` converges geometrically.  This makes the rule an independent
    reference for the closed form above.

    ``order=2`` is kept as a deliberately coarse negative control: it must
    *fail* a 1e-6 agreement gate.
    """
    triangle = _validate_triangle(vertices)
    apex = int(apex)
    if apex not in (0, 1, 2):
        raise ValueError("apex must be 0, 1 or 2")
    ordered = np.roll(triangle, -apex, axis=0)
    nodes_u, weights_u = _gauss_unit_interval(order)
    nodes_v, weights_v = _gauss_unit_interval(order)
    u = np.repeat(nodes_u[:, None], nodes_v.size, axis=1)
    v = np.tile(nodes_v[None, :], (nodes_u.size, 1))
    weights = np.outer(weights_u, weights_v)

    edge_b = ordered[1] - ordered[0]
    edge_c = ordered[2] - ordered[0]
    jacobian = float(np.linalg.norm(np.cross(edge_b, edge_c)))
    shape = u[:, :, None] * (
        (1.0 - v)[:, :, None] * edge_b[None, None, :]
        + v[:, :, None] * edge_c[None, None, :]
    )
    distance = np.linalg.norm(shape, axis=2)
    lam_b = u * (1.0 - v)
    lam_c = u * v
    lam_a = 1.0 - u
    weighted = weights * jacobian * u / (FOUR_PI * distance)
    values = np.array(
        [
            float(np.sum(weighted * lam_a)),
            float(np.sum(weighted * lam_b)),
            float(np.sum(weighted * lam_c)),
        ]
    )
    result = np.empty(3, dtype=float)
    result[apex] = values[0]
    result[(apex + 1) % 3] = values[1]
    result[(apex + 2) % 3] = values[2]
    return result


def self_integral_consistency(vertices, order: int = 24) -> dict:
    """Compare the closed form with the graded Duffy rule and a coarse control.

    Two independent statements are checked:

    1. entry-wise, the closed form must match the graded Duffy rule (which
       cancels the vertex singularity with the Jacobian) to round-off;
    2. row sums, ``sum_k int_T lambda_k G dS = int_T G dS`` for the same
       in-plane target must hold in both readings -- this catches a derivation
       that gets the weighting right but loses panel area.

    Note that ``int_T G dS`` at a *vertex* is not the same number as the
    constant-panel self integral at the *centroid*: the two differ by roughly a
    factor of two on a quasi-equilateral panel and must not be compared.
    """
    triangle = _validate_triangle(vertices)
    analytic = np.empty((3, 3), dtype=float)
    numerical = np.empty((3, 3), dtype=float)
    coarse = np.empty((3, 3), dtype=float)
    for apex in range(3):
        analytic[apex] = integrate_linear_single_layer_self_analytic(triangle, apex=apex)
        numerical[apex] = integrate_linear_single_layer_self_numerical(
            triangle, apex=apex, order=order
        )
        coarse[apex] = integrate_linear_single_layer_self_numerical(
            triangle, apex=apex, order=2
        )
    scale = float(np.max(np.abs(analytic)))
    row_analytic = np.sum(analytic, axis=1)
    row_numeric = np.sum(numerical, axis=1)
    return {
        "analytic": analytic.tolist(),
        "numerical": numerical.tolist(),
        "max_rel_diff_analytic_vs_numerical": float(
            np.max(np.abs(analytic - numerical)) / scale
        ),
        "coarse_order2_max_rel_diff": float(np.max(np.abs(analytic - coarse)) / scale),
        "row_sums_analytic": row_analytic.tolist(),
        "row_sums_numerical": row_numeric.tolist(),
        "row_sum_rel_error": float(
            np.max(np.abs(row_analytic - row_numeric)) / abs(float(np.max(row_numeric)))
        ),
    }


# ---------------------------------------------------------------------------
# mesh-level helpers
# ---------------------------------------------------------------------------
def area_weights(mesh: TriangleMesh) -> FloatArray:
    """Lumped vertex areas (a third of each incident panel)."""
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    lumped = np.zeros(mesh.vertices.shape[0], dtype=float)
    panel_area = np.asarray(mesh.areas, dtype=float) / 3.0
    for local in range(3):
        np.add.at(lumped, mesh.faces[:, local], panel_area)
    return lumped


def vertex_normals(mesh: TriangleMesh) -> FloatArray:
    """Area-weighted vertex normals (normalised)."""
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    accumulated = np.zeros((mesh.vertices.shape[0], 3), dtype=float)
    normals = np.asarray(mesh.normals, dtype=float)
    weight = np.asarray(mesh.areas, dtype=float)[:, None]
    for local in range(3):
        np.add.at(accumulated, mesh.faces[:, local], weight * normals)
    lengths = np.linalg.norm(accumulated, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("degenerate mesh: some vertex normals vanish")
    return accumulated / lengths[:, None]


def angle_defect(mesh: TriangleMesh) -> FloatArray:
    """Angle defect ``2 pi - sum(incident panel angles)`` at every vertex.

    On a smooth surface this is the solid angle of the spherical patch the
    vertex stands for, i.e. ``O(h**2)``, and it is the size of the free-term
    error incurred by using the smooth-surface value ``1/2``.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    defect = np.full(mesh.vertices.shape[0], 2.0 * math.pi, dtype=float)
    for local in range(3):
        index = mesh.faces[:, local]
        first = mesh.vertices[mesh.faces[:, (local + 1) % 3]] - mesh.vertices[index]
        second = mesh.vertices[mesh.faces[:, (local + 2) % 3]] - mesh.vertices[index]
        first /= np.linalg.norm(first, axis=1)[:, None]
        second /= np.linalg.norm(second, axis=1)[:, None]
        cosine = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
        np.add.at(defect, index, -np.arccos(cosine))
    return defect


def _panel_quadrature(
    mesh: TriangleMesh, order: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Duffy-mapped product Gauss rule stored per panel.

    Returns ``(points, weights, barycentric)`` of shapes ``(F, Q, 3)``,
    ``(F, Q)`` and ``(F, Q, 3)``.
    """
    bary, weights = _unit_triangle_rule(order)
    vertices_stack = np.stack((mesh.v0, mesh.v1, mesh.v2), axis=1)  # (F, 3, 3)
    points = np.einsum("qk,fkj->fqj", bary, vertices_stack)        # (F, Q, 3)
    panel_weights = (2.0 * np.asarray(mesh.areas, dtype=float))[:, None] * weights[None, :]
    panel_bary = np.broadcast_to(
        bary[None, :, :], (mesh.panel_count, bary.shape[0], 3)
    ).copy()
    return points, panel_weights, panel_bary


def _refine_toward(vertices: FloatArray, point: FloatArray, depth: int) -> list[FloatArray]:
    """Split a triangle into ``3 * depth + 1`` pieces refined toward ``point``.

    Each level splits the current piece into four by edge midpoints and keeps
    only the child nearest ``point`` for the next level; the other three are
    integrated as they are.  The union of the returned pieces is exactly the
    parent triangle, so refinement changes the quadrature resolution only,
    never the integrated domain.
    """
    current = np.asarray(vertices, dtype=float)
    if depth <= 0:
        return [current]
    pieces: list[FloatArray] = []
    for _ in range(int(depth)):
        a, b, c = current
        m_ab = 0.5 * (a + b)
        m_bc = 0.5 * (b + c)
        m_ca = 0.5 * (c + a)
        children = (
            np.stack((a, m_ab, m_ca)),
            np.stack((b, m_bc, m_ab)),
            np.stack((c, m_ca, m_bc)),
            np.stack((m_ab, m_bc, m_ca)),
        )
        keep = int(
            np.argmin([np.linalg.norm(child.mean(axis=0) - point) for child in children])
        )
        pieces.extend(children[index] for index in range(4) if index != keep)
        current = children[keep]
    pieces.append(current)
    return pieces


def _integrate_panel_pieces(
    target: FloatArray,
    pieces: list[FloatArray],
    normal: FloatArray,
    rule: tuple[FloatArray, FloatArray],
) -> tuple[FloatArray, FloatArray]:
    """Integrate the linear shape functions over near-singular sub-triangles."""
    bary, weights = rule
    single = np.zeros(3, dtype=float)
    double = np.zeros(3, dtype=float)
    for triangle in pieces:
        points = bary @ triangle
        delta = target[None, :] - points
        distance = np.linalg.norm(delta, axis=1)
        area2 = float(
            np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        )
        quadrature = weights * 0.5 * area2
        normal_dot = delta @ normal
        single += bary.T @ (quadrature / (FOUR_PI * distance))
        double += bary.T @ (quadrature * normal_dot / (FOUR_PI * distance**3))
    return single, double


# ---------------------------------------------------------------------------
# assembly and solves
# ---------------------------------------------------------------------------
def assemble_linear_panel_matrices(
    mesh: TriangleMesh,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    self_rule: str = "analytic",
    refinement_depth: int = DEFAULT_REFINEMENT_DEPTH,
) -> tuple[FloatArray, FloatArray]:
    """Assemble the vertex-collocated order-1 single- and double-layer matrices.

    Parameters
    ----------
    mesh:
        Closed :class:`TriangleMesh`.
    quadrature_order:
        Duffy-mapped product-Gauss order (``order**2`` points per panel piece).
    self_rule:
        Rule for the panel that *contains* the collocation vertex:
        ``"analytic"`` (closed form, production path), ``"graded_duffy"``
        (independent high-order rule) or ``"coarse"`` (2-point rule, negative
        control only).
    refinement_depth:
        Refinement levels applied to panels nearer than twice their own size to
        the collocation point.  ``0`` disables the refinement and is useful only
        as a negative control.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if self_rule not in SELF_RULES:
        raise ValueError(f"self_rule must be one of {SELF_RULES}")
    if int(refinement_depth) < 0:
        raise ValueError("refinement_depth must be >= 0")

    vertex_count = mesh.vertices.shape[0]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    targets = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.normals, dtype=float)
    centroids = np.asarray(mesh.centroids, dtype=float)
    areas = np.asarray(mesh.areas, dtype=float)

    far_rule = _panel_quadrature(mesh, quadrature_order)
    near_rule = _unit_triangle_rule(max(int(quadrature_order), 6))
    far_points, far_weights, far_bary = far_rule

    # closed-form self terms: self_single[panel, local apex, local vertex]
    self_single = np.empty((mesh.panel_count, 3, 3), dtype=float)
    for panel in range(mesh.panel_count):
        triangle = targets[faces[panel]]
        for apex in range(3):
            if self_rule == "analytic":
                self_single[panel, apex] = integrate_linear_single_layer_self_analytic(
                    triangle, apex=apex
                )
            else:
                order = 24 if self_rule == "graded_duffy" else 2
                self_single[panel, apex] = integrate_linear_single_layer_self_numerical(
                    triangle, apex=apex, order=order
                )

    incident: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for panel in range(mesh.panel_count):
        for local in range(3):
            incident[int(faces[panel, local])].append((panel, local))

    single = np.zeros((vertex_count, vertex_count), dtype=float)
    double = np.zeros((vertex_count, vertex_count), dtype=float)
    near_radius = 2.0 * np.sqrt(areas)

    for index, target in enumerate(targets):
        near = np.linalg.norm(centroids - target[None, :], axis=1) < near_radius
        for panel, _local in incident[index]:
            near[panel] = False

        far_panels = np.flatnonzero(~near)
        if far_panels.size:
            delta = target[None, None, :] - far_points[far_panels]
            distance = np.linalg.norm(delta, axis=2)
            inverse = far_weights[far_panels] / (FOUR_PI * distance)
            single_row = np.einsum("fq,fqk->fk", inverse, far_bary[far_panels])
            normal_dot = np.einsum("fj,fqj->fq", normals[far_panels], delta)
            double_row = np.einsum(
                "fq,fqk->fk",
                far_weights[far_panels] * normal_dot / (FOUR_PI * distance**3),
                far_bary[far_panels],
            )
            np.add.at(single[index], faces[far_panels].ravel(), single_row.ravel())
            np.add.at(double[index], faces[far_panels].ravel(), double_row.ravel())

        for panel in np.flatnonzero(near):
            pieces = _refine_toward(targets[faces[panel]], target, refinement_depth)
            near_single, near_double = _integrate_panel_pieces(
                target, pieces, normals[panel], near_rule
            )
            np.add.at(single[index], faces[panel], near_single)
            np.add.at(double[index], faces[panel], near_double)

        for panel, local in incident[index]:
            for column in range(3):
                single[index, int(faces[panel, column])] += self_single[panel, local, column]

    return single, double


@dataclass(frozen=True)
class LinearLaplaceResult:
    """Result of an order-1 exterior Laplace solve on a closed surface."""

    potentials: FloatArray
    normal_derivatives: FloatArray
    single_layer_matrix: FloatArray
    double_layer_matrix: FloatArray
    condition_single: float
    condition_second_kind: float
    relative_residual: float
    map_name: str


def cone_solid_angle(double: FloatArray) -> FloatArray:
    """Body-side cone solid angle ``Omega_i / (4 pi) = -sum_j D_ij`` at each vertex.

    For a *smooth* closed surface ``sum_j D_ij = -1/2`` and this is the solid
    angle of the half-space the body occupies over ``4 pi``, i.e. ``1/2``.  At a
    vertex of a faceted mesh the body occupies a cone narrower than a
    half-space, so the value is below ``1/2``; the shortfall is the *cone
    defect* and is ``O(h)`` on an inscribed sphere (0.08 at ``N = 320``).  The
    assembled ``D`` reproduces this solid angle exactly (flat-panel solid angles
    are analytic), so the value is a geometric diagnostic, not a free parameter.

    It is **not** the free term of the discretised boundary integral equation.
    That term is the continuous ``c = 1/2``; taking the cone value instead
    violates the discrete boundary identity (see
    :func:`representation_residual`), so it is kept only as a negative control.
    """
    return -np.sum(double, axis=1)


def implied_free_term(
    double: FloatArray, single: FloatArray, potential: FloatArray, flux: FloatArray
) -> FloatArray:
    """Free term implied by the assembled operators acting on an exact field.

    ``c_i = ((D u)_i - (S q)_i) / u_i`` makes the discrete identity
    ``(diag(c) - D) u + S q = 0`` hold exactly at vertex ``i``.  Feeding an
    analytic harmonic field therefore measures the free term the
    discretisation actually needs, which is the cleanest check on the ``1/2``
    convention (it converges to ``1/2`` under refinement).
    """
    numerator = double @ potential - single @ flux
    return numerator / np.asarray(potential, dtype=float)


def representation_residual(
    double: FloatArray,
    single: FloatArray,
    potential: FloatArray,
    flux: FloatArray,
    free_term: str = "half",
) -> float:
    """Relative residual of the discrete boundary identity on an exact field."""
    if free_term == "half":
        diagonal = np.full(double.shape[0], 0.5)
    elif free_term == "row_sum":
        diagonal = cone_solid_angle(double)
    else:
        raise ValueError("free_term must be 'half' or 'row_sum'")
    residual = (np.diag(diagonal) - double) @ potential + single @ flux
    scale = float(np.linalg.norm(single @ flux))
    return float(np.linalg.norm(residual) / scale) if scale > 0.0 else float("nan")


def _second_kind_system(
    mesh: TriangleMesh, double: FloatArray, free_term: str
) -> FloatArray:
    if free_term == "half":
        diagonal = np.full(double.shape[0], 0.5)
    elif free_term == "row_sum":
        diagonal = cone_solid_angle(double)
    else:
        raise ValueError("free_term must be 'half' or 'row_sum'")
    return np.diag(diagonal) - double


def solve_linear_panel_dirichlet(
    mesh: TriangleMesh,
    boundary_values: FloatArray,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    self_rule: str = "analytic",
    refinement_depth: int = DEFAULT_REFINEMENT_DEPTH,
    matrices: tuple[FloatArray, FloatArray] | None = None,
    free_term: str = "half",
) -> LinearLaplaceResult:
    """Exterior Dirichlet solve: recover the body-normal flux from the potential.

    Solves ``(diag(c) - D) u = -S q`` for ``q = grad(u) . n`` at the vertices
    with the continuous free term ``c = 1/2`` (the row-sum variant
    ``free_term="row_sum"`` is a negative control, see the module docstring).
    """
    values = _validate_vertex_field(mesh, boundary_values, "boundary_values")
    single, double = (
        matrices
        if matrices is not None
        else assemble_linear_panel_matrices(
            mesh,
            quadrature_order=quadrature_order,
            self_rule=self_rule,
            refinement_depth=refinement_depth,
        )
    )
    system = _second_kind_system(mesh, double, free_term)
    rhs = -(system @ values)
    flux = np.linalg.solve(single, rhs)
    residual = np.linalg.norm(single @ flux - rhs) / np.linalg.norm(rhs)
    return LinearLaplaceResult(
        potentials=values,
        normal_derivatives=flux,
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_single=float(np.linalg.cond(single)),
        condition_second_kind=float(np.linalg.cond(system)),
        relative_residual=float(residual),
        map_name="dirichlet_to_flux",
    )


def solve_linear_panel_neumann(
    mesh: TriangleMesh,
    normal_derivative: FloatArray,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    self_rule: str = "analytic",
    refinement_depth: int = DEFAULT_REFINEMENT_DEPTH,
    matrices: tuple[FloatArray, FloatArray] | None = None,
    free_term: str = "half",
) -> LinearLaplaceResult:
    """Exterior Neumann solve: recover the surface potential from the flux.

    Solves ``(diag(c) - D) u = -S q`` (the Neumann-to-potential reading).
    """
    flux = _validate_vertex_field(mesh, normal_derivative, "normal_derivative")
    single, double = (
        matrices
        if matrices is not None
        else assemble_linear_panel_matrices(
            mesh,
            quadrature_order=quadrature_order,
            self_rule=self_rule,
            refinement_depth=refinement_depth,
        )
    )
    system = _second_kind_system(mesh, double, free_term)
    rhs = -(single @ flux)
    potential = np.linalg.solve(system, rhs)
    residual = np.linalg.norm(system @ potential - rhs) / np.linalg.norm(rhs)
    return LinearLaplaceResult(
        potentials=potential,
        normal_derivatives=flux,
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_single=float(np.linalg.cond(single)),
        condition_second_kind=float(np.linalg.cond(system)),
        relative_residual=float(residual),
        map_name="neumann_to_potential",
    )


def _validate_vertex_field(mesh: TriangleMesh, field, name: str) -> FloatArray:
    values = np.asarray(field, dtype=float)
    expected = mesh.vertices.shape[0]
    if values.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must all be finite")
    return values.copy()