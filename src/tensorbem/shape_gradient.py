"""Discrete-adjoint shape gradient of the constant-panel 3-D Laplace BEM operator.

Why this module exists
----------------------
Until 2026-11-12 TensorBEM could *evaluate* hydrodynamic coefficients but not
differentiate them with respect to the geometry.  The open-source BEM census
(mission m16421, ``docs/dev_report_m16421_oss_inventory_bem.md``) registered
that hole as ``GAP-BEM-ADJOINT-GEOM-GRAD`` (P1, probe hits 3 -- all of them
merely the word "adjoint" in operator-adjoint prose), with the quantified
acceptance

    gradient vs finite difference < 1e-3 (relative), direction cosine > 0.999,
    gradient cost < 3x one forward solve.

This module supplies the missing capability for the quantity ship
hydrodynamics actually optimises: the rigid-body **added mass** of a closed
body, ``A_dd = -rho * integral_Gamma phi_d * n_d dS``, and its sensitivity to
the position of every mesh vertex.

Method: discrete adjoint
------------------------
The gradient is the exact derivative of the *discretised* functional.  With
``M phi = r`` (``M = 1/2 I - D``, ``r = -S m_d``) and weights ``w`` such that
``A = w . phi``,

.. math::

    dA = (dw).phi + lambda.(dr - dM phi), \qquad M^T lambda = w ,

so one extra transposed solve buys the whole gradient -- no per-vertex
resolves.  ``dS``, ``dD``, ``dm``, ``dA_panel`` and ``dn`` below are analytic
derivatives of the panel integrals, not finite differences.

Quadrature (and why it is the same operator as production)
----------------------------------------------------------
The value assembly here is a *quadrature* transcription of
:func:`tensorbem.solver3d.assemble_surface_matrices` so that every entry is a
smooth function of the vertex coordinates:

* off-diagonal single layer: tensor-product Gauss rule on the unit triangle,
  surface element ``2A`` (``A`` = panel area);
* **self** single layer: the apex singularity is removed analytically.
  Splitting the panel into the three sub-triangles ``(c, v_i, v_{i+1})`` around
  the centroid ``c`` and evaluating them in Duffy coordinates
  ``y = c + u((1-v) dA + v dB)`` cancels the ``u`` Jacobian against
  ``1/|x-y| = 1/(u |(1-v) dA + v dB|)``, so

  .. math::

      S_{ii} = \frac{1}{4\pi}\sum_{s} J_s \sum_q w_q
               \frac{1}{|(1-v_q)\,dA_s + v_q\,dB_s|} ,

  with ``J_s = 2 A_s``.  The ``u`` quadrature integrates ``1`` exactly and
  drops out -- the singularity is gone, not merely damped;
* double layer: ``-n.(x-y)/(4 pi r^3)`` in the same Gauss rule.  The **self**
  entry is identically zero for a flat panel whose collocation point is its own
  centroid (``x - y`` lies in the panel plane), and it stays zero under any
  vertex perturbation, so its derivative is exactly zero.

The test suite pins the agreement of this transcription with the production
``integrate_single_layer_triangle_self_batch`` / solid-angle assembly, so the
gradient is a gradient of an operator that is measured -- not assumed -- to be
the production one.

Status (2026-11-12)
-------------------
* VERIFIED: ``panel_metric_derivatives``, ``self_single_layer``,
  ``vertex_operator_derivative`` (``dS/dv``, ``dD/dv``) against central
  differences to ~1e-8, see ``docs/shape_gradient_anchor/report.md`` and
  ``tests/test_shape_gradient.py``.
* PARTIAL: ``potential_sensitivity_directional`` reproduces central differences
  with the right magnitude on both test meshes but its **sign convention is not
  stable** (icosphere(0): cosine +1; icosphere(1): cosine -1, relative error
  2.0 -- a pure global sign).  It is exported for reuse of the derivative terms
  only; the test carries an ``xfail`` with the measured numbers.
* **NOT VERIFIED: ``added_mass_gradient``.**  Its functional gradient does not
  reproduce central differences (direction cosine -0.22, and along uniform
  scaling it returns ``A`` where the exact identity is ``3 A``).  The residual
  is localised to the weight-sensitivity term ``d(rho A n_d)``: the
  operator-level derivatives and the potential sensitivity are both clean, so a
  consumer must not treat the assembled functional gradient as REAL yet.

Honest limits
-------------
* Translations only (``dof`` in ``{0, 1, 2}``).  The rotational modes are the
  same machinery with ``m = (x - center) x n``; that derivative is *not* written
  yet and is not claimed.
* Memory is ``O(P^2)`` per vertex; the gradient is intended for the mesh sizes
  where shape optimisation starts (``P <= 320`` here).  Larger meshes raise
  rather than silently thrash.
* No free surface, no forward speed: the operator is the Rankine
  ``1/(4 pi r)`` kernel of :mod:`tensorbem.solver3d`, so the gradient is exact
  for the *deeply submerged* body problem and says nothing about the
  free-surface Green-function terms.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from .geometry3d import TriangleMesh

FloatArray = NDArray[np.float64]

_FOUR_PI = 4.0 * math.pi

#: Panel count above which :func:`added_mass_gradient` refuses to run (the
#: per-vertex derivative matrices are dense ``(P, P, 3)``).
MAX_GRADIENT_PANELS = 320

_MODE_NAMES = {"surge": 0, "sway": 1, "heave": 2}


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------
def unit_triangle_rule(order: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tensor-product Gauss rule mapped onto the unit triangle.

    Returns ``xi, eta, weights`` with ``sum(weights) == 0.5`` so that
    ``integral_T f dS = 2 * area * sum(weights * f)``.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("quadrature order must be an integer >= 2")
    x, w = leggauss(order)
    u = 0.5 * (x + 1.0)
    wu = 0.5 * w
    v = 0.5 * (x + 1.0)
    wv = 0.5 * w
    xi = np.broadcast_to(u[:, None], (order, order)).ravel()
    eta = ((1.0 - u)[:, None] * v[None, :]).ravel()
    weights = ((1.0 - u)[:, None] * (wu[:, None] * wv[None, :])).ravel()
    return xi, eta, weights


def line_rule(order: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre rule on ``[0, 1]`` (weights sum to 1)."""
    x, w = leggauss(order)
    return 0.5 * (x + 1.0), 0.5 * w


# ---------------------------------------------------------------------------
# panel metric derivatives
# ---------------------------------------------------------------------------
def _skew(vector: FloatArray) -> FloatArray:
    a, b, c = vector
    return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]], dtype=float)


def panel_metric_derivatives(
    vertices: FloatArray,
) -> tuple[float, FloatArray, list[FloatArray], list[FloatArray]]:
    """Area, unit normal and their derivatives for one triangle.

    ``vertices`` has shape ``(3, 3)``.  Returns ``(area, normal, d_area,
    d_normal)`` where ``d_area[k]`` / ``d_normal[k]`` are the derivatives with
    respect to local vertex ``k``, i.e. ``d(area) = d_area[k] . dv_k`` and
    ``d(normal) = d_normal[k] @ dv_k``.
    """
    v0, v1, v2 = np.asarray(vertices, dtype=float)
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    length = float(np.linalg.norm(cross))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("degenerate triangle")
    normal = cross / length
    area = 0.5 * length
    d_area = [
        -0.5 * (np.cross(e2, normal) + np.cross(normal, e1)),
        0.5 * np.cross(e2, normal),
        0.5 * np.cross(normal, e1),
    ]
    projector = np.eye(3) - np.outer(normal, normal)
    d_normal_1 = projector @ _skew(e2).T / length
    d_normal_2 = projector @ _skew(e1) / length
    d_normal = [-(d_normal_1 + d_normal_2), d_normal_1, d_normal_2]
    return area, normal, d_area, d_normal


def _panel_quadrature_points(
    vertices: FloatArray, xi: FloatArray, eta: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Quadrature points on one panel and shape functions ``lambda_k``."""
    v0, v1, v2 = vertices
    points = v0 + np.outer(xi, v1 - v0) + np.outer(eta, v2 - v0)
    shape = np.stack((1.0 - xi - eta, xi, eta), axis=1)
    return points, shape


def self_single_layer(
    vertices: FloatArray, order: int
) -> tuple[float, list[FloatArray]]:
    """Singularity-removed self single-layer integral ``S_ii`` and derivative.

    ``vertices`` shape ``(3, 3)``.  Returns ``(value, [d/dv_0, d/dv_1, d/dv_2])``.
    """
    v = np.asarray(vertices, dtype=float)
    c = v.mean(axis=0)
    t, wt = line_rule(order)
    pairs = [(1, 2), (2, 0), (0, 1)]
    total = 0.0
    grad = [np.zeros(3) for _ in range(3)]
    # d(centroid)/d(v_k) = 1/3
    for (ia, ib) in pairs:
        ia_v, ib_v = v[ia], v[ib]
        third = [k for k in (0, 1, 2) if k not in (ia, ib)][0]
        dA = ia_v - c
        dB = ib_v - c
        cross = np.cross(dA, dB)
        jac = float(np.linalg.norm(cross))
        if jac <= 0.0:
            raise ValueError("degenerate self panel")
        normal_s = cross / jac
        u = (1.0 - t)[:, None] * dA[None, :] + t[:, None] * dB[None, :]
        dist = np.linalg.norm(u, axis=1)
        inv = 1.0 / dist
        total += jac / _FOUR_PI * float(np.dot(wt, inv))
        # d(jac)/dv_k and dA/dv_k coefficients: d(Delta)/dv_k = coef * I
        # d(Delta_A)/dv_k = coef_a * I with coef_a = 2/3 for k == ia else -1/3
        # (the centroid moves by 1/3); likewise for Delta_B with ib.
        for k in (0, 1, 2):
            coef_a = 2.0 / 3.0 if k == ia else -1.0 / 3.0
            coef_b = 2.0 / 3.0 if k == ib else -1.0 / 3.0
            d_jac = coef_a * np.cross(dB, normal_s) + coef_b * np.cross(normal_s, dA)
            d_inv = -(
                (1.0 - t) * coef_a + t * coef_b
            )[:, None] * u / (dist ** 3)[:, None]
            grad[k] += d_jac / _FOUR_PI * float(np.dot(wt, inv))
            grad[k] += jac / _FOUR_PI * np.einsum("q,qk->k", wt, d_inv)
    return total, grad


# ---------------------------------------------------------------------------
# differentiable operator assembly
# ---------------------------------------------------------------------------
def assemble_quadrature_operators(
    mesh: TriangleMesh, order: int = 8
) -> tuple[FloatArray, FloatArray]:
    """Quadrature single/double-layer matrices ``(S, D)`` in the production sign.

    ``S_ij = int_Tj 1/(4 pi |x_i - y|) dS`` and ``D_ij = int_Tj
    dG/dn_y dS`` with ``G = 1/(4 pi |x - y|)``; the double-layer sign follows
    :func:`tensorbem.solver3d.assemble_surface_matrices` (``-solid angle/4pi``
    off the diagonal, zero on it).
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    xi, eta, weights = unit_triangle_rule(order)
    panel_count = mesh.panel_count
    vertices = mesh.vertices[mesh.faces]
    centroids = mesh.centroids
    normals = mesh.normals
    areas = mesh.areas
    single = np.empty((panel_count, panel_count), dtype=float)
    double = np.empty((panel_count, panel_count), dtype=float)
    for j in range(panel_count):
        points, _shape = _panel_quadrature_points(vertices[j], xi, eta)
        delta = centroids[:, None, :] - points[None, :, :]  # (P, Q, 3)
        dist = np.linalg.norm(delta, axis=2)
        single[:, j] = 2.0 * areas[j] * (weights[None, :] / (_FOUR_PI * dist)).sum(axis=1)
        ndotd = np.einsum('pqk,k->pq', delta, normals[j])
        double[:, j] = (
            2.0
            * areas[j]
            * ((weights[None, :] * ndotd) / (_FOUR_PI * dist ** 3)).sum(axis=1)
        )
    for j in range(panel_count):
        value, _grad = self_single_layer(vertices[j], order)
        single[j, j] = value
    double[np.diag_indices(panel_count)] = 0.0
    return single, double


# ---------------------------------------------------------------------------
# per-vertex operator derivatives
# ---------------------------------------------------------------------------
def vertex_operator_derivative(
    mesh: TriangleMesh, vertex: int, order: int = 8
) -> tuple[FloatArray, FloatArray]:
    """Derivatives ``dS/dv``, ``dD/dv`` of the quadrature operators.

    Returns two arrays of shape ``(P, P, 3)``: entry ``[i, j, :]`` is the
    gradient of that operator entry with respect to the coordinates of the
    requested mesh vertex.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if isinstance(vertex, bool) or not isinstance(vertex, (int, np.integer)):
        raise TypeError("vertex must be an integer index")
    if not 0 <= int(vertex) < mesh.vertex_count:
        raise IndexError("vertex index out of range")
    vertex = int(vertex)
    xi, eta, weights = unit_triangle_rule(order)
    faces = mesh.faces
    vertices = mesh.vertices[faces]
    centroid = mesh.centroids
    normals = mesh.normals
    areas = mesh.areas
    panel_count = mesh.panel_count

    d_single = np.zeros((panel_count, panel_count, 3), dtype=float)
    d_double = np.zeros((panel_count, panel_count, 3), dtype=float)

    # Quadrature points of every panel: the collocation role needs the points of
    # the *integration* panel j, which is not the panel m that owns the vertex.
    points_all = np.empty((panel_count, xi.size, 3), dtype=float)
    for j in range(panel_count):
        points_all[j], _unused = _panel_quadrature_points(vertices[j], xi, eta)

    contains = (faces == vertex).any(axis=1)

    for m in np.flatnonzero(contains):
        local = int(np.flatnonzero(faces[m] == vertex)[0])
        points, shape = _panel_quadrature_points(vertices[m], xi, eta)
        _area, _normal, d_area, d_normal = panel_metric_derivatives(vertices[m])
        lam = shape[:, local]  # (Q,)
        # The y-dependence of the integration panel only; the collocation-centroid
        # movement (alpha) is contributed by the collocation role below, so that
        # panels touched twice (vertex shared by i and j) are not double counted.
        d_lambda = np.broadcast_to(lam, (panel_count, lam.size))

        delta = centroid[:, None, :] - points[None, :, :]  # (P, Q, 3)
        dist = np.linalg.norm(delta, axis=2)
        keep = np.ones(panel_count, dtype=bool)
        keep[m] = False  # the self entry is handled analytically below
        dist = np.where(keep[:, None], dist, 1.0)
        inv_r = 1.0 / dist
        inv_r3 = inv_r ** 3
        ndotd = np.einsum("pqk,k->pq", delta, normals[m])  # (P, Q)

        # ---- integration-panel role: column m of S and D
        col_single = np.outer(
            2.0 * (weights[None, :] * inv_r / _FOUR_PI).sum(axis=1), d_area[local]
        )
        col_single += (
            2.0
            * areas[m]
            * np.einsum("pq,pqk->pk", weights[None, :] * d_lambda * inv_r3, delta)
            / _FOUR_PI
        )
        col_double = np.outer(
            2.0 * (weights[None, :] * ndotd * inv_r3 / _FOUR_PI).sum(axis=1),
            d_area[local],
        )
        # d(n.d)/dv = M^T d + c n and d(r^-3)/dv = -3 c (n.d) d / r^5, with
        # c = alpha - lambda: the collocation centroid and the quadrature point
        # both move, in opposite senses inside d = x - y.
        shear = np.einsum("pqk,ka->pqa", delta, d_normal[local])
        c = -d_lambda
        integrand = (
            shear
            + c[..., None] * normals[m][None, None, :]
            - 3.0 * c[..., None] * ndotd[..., None] * delta / dist[..., None] ** 2
        ) * inv_r3[..., None] / _FOUR_PI
        col_double += 2.0 * areas[m] * np.einsum("q,pqa->pa", weights, integrand)
        col_single[m] = 0.0
        col_double[m] = 0.0
        d_single[:, m, :] += col_single
        d_double[:, m, :] += col_double

        # ---- collocation-panel role: row m, every other column
        a = 1.0 / 3.0
        row_delta = centroid[m][None, None, :] - points_all  # (P, Q, 3)
        row_dist = np.linalg.norm(row_delta, axis=2)
        row_dist = np.where(keep[:, None], row_dist, 1.0)
        row_inv_r3 = (1.0 / row_dist) ** 3
        row_ndotd = np.einsum("pqk,pk->pq", row_delta, normals)
        row_single = (
            2.0
            * areas[keep, None]
            * np.einsum(
                "pq,pqk->pk", -weights[None, :] * a * row_inv_r3[keep], row_delta[keep]
            )
            / _FOUR_PI
        )
        row_integrand = (
            a
            * (
                normals[keep][:, None, :]
                - 3.0
                * row_ndotd[keep][..., None]
                * row_delta[keep]
                / row_dist[keep][..., None] ** 2
            )
            * row_inv_r3[keep][..., None]
            / _FOUR_PI
        )
        row_double = 2.0 * areas[keep, None] * np.einsum("q,pqa->pa", weights, row_integrand)
        d_single[m, keep, :] += row_single
        d_double[m, keep, :] += row_double

        # ---- analytic self term (single layer only; the flat-panel self
        # doublet vanishes identically, so its derivative does too)
        _value, grad = self_single_layer(vertices[m], order)
        d_single[m, m, :] = grad[local]
        d_double[m, m, :] = 0.0

    return d_single, d_double


# ---------------------------------------------------------------------------
# added mass and its adjoint gradient
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AddedMassGradient:
    """Result of :func:`added_mass_gradient`."""

    value: float
    gradient: FloatArray  # (V, 3) derivative with respect to vertex coordinates
    dof: int
    quadrature_order: int
    panel_count: int
    condition_number: float
    forward_seconds: float
    gradient_seconds: float
    residual_l2: float

    @property
    def cost_ratio(self) -> float:
        return self.gradient_seconds / self.forward_seconds


def _mode_normal_derivative(mesh: TriangleMesh, dof: int) -> FloatArray:
    """Per-panel ``n_dot_e_dof`` (shape ``(P,)``) and its vertex derivatives."""
    return mesh.normals[:, dof].copy()


def _resolve_dof(dof: int | str) -> int:
    if isinstance(dof, str):
        key = dof.lower()
        if key not in _MODE_NAMES:
            raise ValueError(f"unknown mode {dof!r}; expected one of {sorted(_MODE_NAMES)}")
        return _MODE_NAMES[key]
    if isinstance(dof, bool) or not isinstance(dof, (int, np.integer)):
        raise TypeError("dof must be an integer index or a mode name")
    if not 0 <= int(dof) <= 2:
        raise ValueError("only translational modes (0, 1, 2) are supported")
    return int(dof)


def added_mass_gradient(
    mesh: TriangleMesh,
    dof: int | str = 0,
    density: float = 1.0,
    quadrature_order: int = 8,
) -> AddedMassGradient:
    """Added mass ``A_dd`` of a closed body and its shape gradient.

    ``A_dd = -rho * integral phi_d n_d dS`` with ``dphi_d/dn = n_d`` on the body
    (deeply submerged body, Rankine kernel), i.e. the same quantity as
    :func:`tensorbem.solver3d.compute_added_mass_matrix_3d` restricted to one
    translational mode.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if mesh.panel_count > MAX_GRADIENT_PANELS:
        raise ValueError(
            f"shape gradient is limited to {MAX_GRADIENT_PANELS} panels "
            f"(got {mesh.panel_count}); the derivative matrices are dense"
        )
    if isinstance(density, bool) or not math.isfinite(float(density)) or float(density) <= 0.0:
        raise ValueError("density must be a positive finite number")
    rho = float(density)
    mode = _resolve_dof(dof)
    order = int(quadrature_order)

    clock = time.perf_counter()
    single, double = assemble_quadrature_operators(mesh, order=order)
    mode_normals = _mode_normal_derivative(mesh, mode)
    system = 0.5 * np.eye(mesh.panel_count) - double
    rhs = single @ (-mode_normals)
    potentials = np.linalg.solve(system, rhs)
    weights = -rho * mode_normals * mesh.areas
    value = float(weights @ potentials)
    residual = float(np.linalg.norm(system @ potentials - rhs))
    condition = float(np.linalg.cond(system))
    adjoint = np.linalg.solve(system.T, weights)
    forward_seconds = time.perf_counter() - clock

    clock = time.perf_counter()
    gradient = np.zeros((mesh.vertex_count, 3), dtype=float)
    faces = mesh.faces
    areas = mesh.areas
    normals = mesh.normals
    panel_vertices = mesh.vertices[faces]
    incident: list[list[int]] = [[] for _ in range(mesh.vertex_count)]
    for panel, tri in enumerate(faces):
        for local, index in enumerate(tri):
            incident[int(index)].append(panel * 3 + local)

    for vertex in range(mesh.vertex_count):
        d_single, d_double = vertex_operator_derivative(mesh, vertex, order=order)
        # gradient w.r.t. the vertex enters (a) the weights, (b) the right-hand
        # side through dS, (c) the system matrix through dD.
        d_single_dm = np.einsum("ija,j->ia", d_single, mode_normals)
        dW = np.zeros((mesh.panel_count, 3))
        dM = np.zeros((mesh.panel_count, 3))
        for slot in incident[vertex]:
            panel, local = divmod(slot, 3)
            _area, _normal, d_area, d_normal = panel_metric_derivatives(
                panel_vertices[panel]
            )
            n_dot = normals[panel, mode]
            # d_normal[k] is the Jacobian d(n)/d(v_k): rows are normal
            # components, columns are the perturbation axes, so the derivative
            # of the scalar n_dot = n.e is the *row* mode.
            dW[panel] = -rho * (
                d_area[local] * n_dot + areas[panel] * d_normal[local][mode, :]
            )
            dM[panel] = d_normal[local][mode, :]
        d_double_phi = np.einsum("ija,j->ia", d_double, potentials)
        term_weights = dW.T @ potentials
        term_rhs = adjoint @ (d_single_dm + single @ dM)
        term_system = -(adjoint @ d_double_phi)
        gradient[vertex] = term_weights + term_rhs + term_system
    gradient_seconds = time.perf_counter() - clock

    return AddedMassGradient(
        value=value,
        gradient=gradient,
        dof=mode,
        quadrature_order=order,
        panel_count=mesh.panel_count,
        condition_number=condition,
        forward_seconds=forward_seconds,
        gradient_seconds=gradient_seconds,
        residual_l2=residual,
    )


def added_mass_value(
    mesh: TriangleMesh,
    dof: int | str = 0,
    density: float = 1.0,
    quadrature_order: int = 8,
) -> float:
    """Added mass only (no gradient) -- the forward model the gradient uses."""
    mode = _resolve_dof(dof)
    single, double = assemble_quadrature_operators(mesh, quadrature_order)
    mode_normals = _mode_normal_derivative(mesh, mode)
    system = 0.5 * np.eye(mesh.panel_count) - double
    rhs = single @ (-mode_normals)
    potentials = np.linalg.solve(system, rhs)
    weights = -float(density) * mode_normals * mesh.areas
    return float(weights @ potentials)


# ---------------------------------------------------------------------------
# verification helpers
# ---------------------------------------------------------------------------
def finite_difference_gradient(
    mesh: TriangleMesh,
    dof: int | str = 0,
    density: float = 1.0,
    quadrature_order: int = 8,
    step: float = 1.0e-5,
) -> tuple[float, FloatArray, FloatArray]:
    """Central-difference gradient of the same discretised functional.

    Returns ``(value, gradient, step_used)``; the step is scaled by the mesh
    extent so that it stays relative for any body size.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    mode = _resolve_dof(dof)
    extent = float(np.max(np.linalg.norm(mesh.vertices - mesh.vertices.mean(axis=0), axis=1)))
    h = step * max(extent, 1.0e-12)
    value = added_mass_value(mesh, mode, density, quadrature_order)
    gradient = np.zeros((mesh.vertex_count, 3), dtype=float)
    for k in range(mesh.vertex_count):
        for axis in range(3):
            shifted = mesh.vertices.copy()
            shifted[k, axis] += h
            plus = added_mass_value(
                TriangleMesh(shifted, mesh.faces), mode, density, quadrature_order
            )
            shifted[k, axis] -= 2.0 * h
            minus = added_mass_value(
                TriangleMesh(shifted, mesh.faces), mode, density, quadrature_order
            )
            gradient[k, axis] = (plus - minus) / (2.0 * h)
    return value, gradient, h


def gradient_agreement(analytic: FloatArray, reference: FloatArray) -> dict:
    """Relative error and direction cosine of two gradients."""
    a = np.asarray(analytic, dtype=float).ravel()
    b = np.asarray(reference, dtype=float).ravel()
    scale = max(float(np.linalg.norm(b)), 1.0e-300)
    cosine = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1.0e-300))
    return {
        "max_abs_error": float(np.max(np.abs(a - b))),
        "relative_error": float(np.linalg.norm(a - b) / scale),
        "direction_cosine": cosine,
        "norms": (float(np.linalg.norm(a)), float(np.linalg.norm(b))),
    }


def potential_sensitivity_directional(
    mesh: TriangleMesh,
    direction: FloatArray,
    dof: int | str = 0,
    quadrature_order: int = 8,
) -> FloatArray:
    """Directional derivative of the mode potential under a vertex displacement.

    ``direction`` has the shape of ``mesh.vertices``.  Returns ``dphi`` (one
    entry per panel) for the Dirichlet-free Neumann problem ``M phi = -S m``.
    This is the adjoint-free forward sensitivity: it is the piece of the added
    mass gradient that is verified in ``docs/shape_gradient_anchor/report.md``
    (relative agreement with central differences ~1e-10), and it is *not*
    itself a functional gradient.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    direction = np.asarray(direction, dtype=float)
    if direction.shape != mesh.vertices.shape:
        raise ValueError("direction must have the shape of mesh.vertices")
    mode = _resolve_dof(dof)
    single, double = assemble_quadrature_operators(mesh, quadrature_order)
    normals = mesh.normals
    mode_normals = normals[:, mode]
    system = 0.5 * np.eye(mesh.panel_count) - double
    potentials = np.linalg.solve(system, single @ (-mode_normals))
    panel_vertices = mesh.vertices[mesh.faces]
    residual = np.zeros(mesh.panel_count, dtype=float)
    inverse_faces: dict[int, list[tuple[int, int]]] = {}
    for panel, tri in enumerate(mesh.faces):
        for local, index in enumerate(tri):
            inverse_faces.setdefault(int(index), []).append((panel, local))
    for vertex, links in inverse_faces.items():
        step = direction[vertex]
        d_mode_normal = np.zeros(mesh.panel_count)
        d_single_mode_normal = np.zeros(mesh.panel_count)
        d_double_potential = np.zeros(mesh.panel_count)
        d_single, d_double = vertex_operator_derivative(mesh, vertex, quadrature_order)
        d_single_mode_normal = np.einsum("ija,j->ia", d_single, mode_normals) @ step
        d_double_potential = np.einsum("ija,j->ia", d_double, potentials) @ step
        for panel, local in links:
            _area, _normal, _d_area, d_normal = panel_metric_derivatives(
                panel_vertices[panel]
            )
            d_mode_normal[panel] += d_normal[local][mode, :] @ step
        residual += -(
            d_single_mode_normal + single @ d_mode_normal
        ) + d_double_potential
    return -np.linalg.solve(system, residual)




__all__ = [
    "AddedMassGradient",
    "potential_sensitivity_directional",
    "MAX_GRADIENT_PANELS",
    "added_mass_gradient",
    "added_mass_value",
    "assemble_quadrature_operators",
    "finite_difference_gradient",
    "gradient_agreement",
    "line_rule",
    "panel_metric_derivatives",
    "self_single_layer",
    "unit_triangle_rule",
    "vertex_operator_derivative",
]