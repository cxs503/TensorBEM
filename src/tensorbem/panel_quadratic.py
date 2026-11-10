"""Quadratic (6-node) curved-triangle panel operator for 3-D Laplace BEM.

Why this module exists
----------------------
The constant-panel solver (:mod:`tensorbem.solver3d`) has an ``O(h)``
convergence floor set by the piecewise-constant geometry approximation.
The piecewise-linear solver (:mod:`tensorbem.panel_linear`) raises the
convergence to ``O(h**2)`` for smooth fields but still uses flat panels,
so the geometric error remains ``O(h**2)``.  This module adds the next
rung: genuine *quadratic* (order-2) boundary elements with

* **6-node triangles**: 3 vertex nodes + 3 edge-midpoint nodes;
* **isoparametric mapping**: geometry and field use the same quadratic
  shape functions, so curved surfaces are captured to ``O(h**3)``;
* **analytical self-influence**: the singular diagonal integrals are
  evaluated by Duffy-transformed polar quadrature that cancels the
  ``1/r`` singularity exactly;
* **vertex collocation**: unknowns live at the 6 nodes, shared between
  incident elements, giving a continuous boundary representation.

Shape functions
---------------
On the reference triangle with barycentric coordinates ``(L1, L2, L3)``::

    Vertex 1 (L1=1):  N1 = L1*(2*L1 - 1)
    Vertex 2 (L2=1):  N2 = L2*(2*L2 - 1)
    Vertex 3 (L3=1):  N3 = L3*(2*L3 - 1)
    Edge 12 (L3=0):   N4 = 4*L1*L2
    Edge 23 (L1=0):   N5 = 4*L2*L3
    Edge 31 (L2=0):   N6 = 4*L3*L1

Convention
----------
``G(x, y) = 1 / (4 pi |x - y|)``.  The exterior second-kind equation is

    (1/2 I - D) u = -S q

with ``S_ik = int N_k G dS`` and ``D_ik = int N_k dG/dn dS``.

Quadrature
----------
Non-singular panels use a high-order Gauss rule on the reference triangle.
Self-panels (target at a node of the source panel) use a Duffy transform
that cancels the ``1/r`` singularity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    # Constants
    "NODES_PER_ELEMENT",
    "NODE_LABELS",
    "NODE_BARYCENTRIC",
    "FOUR_PI",
    # Shape functions
    "quadratic_shape_functions",
    "quadratic_shape_gradients_ref",
    "quadratic_shape_functions_array",
    "quadratic_shape_gradients_ref_array",
    "partition_of_unity_check",
    # Quadrature
    "gauss_triangle_rule",
    "duffy_self_quadrature",
    "triangle_apex_quadrature",
    # Geometry
    "QuadraticTriangle",
    "QuadraticSurfaceMesh",
    "icosphere_quadratic",
    "ellipsoid_quadratic",
    # BEM operators
    "assemble_quadratic_matrices",
    "solve_laplace_dirichlet_quadratic",
    "solve_neumann_from_dirichlet_quadratic",
    "QuadraticBEMResult",
    "QuadraticNeumannResult",
    # Diagnostics
    "self_influence_analytical_vs_numerical",
]

FOUR_PI = 4.0 * math.pi

#: Number of nodes per quadratic triangle element.
NODES_PER_ELEMENT = 6

#: Node ordering: vertices 0,1,2 then edge midpoints 01,12,20.
NODE_LABELS = ("v0", "v1", "v2", "mid01", "mid12", "mid20")


# ---------------------------------------------------------------------------
# Quadratic shape functions on the reference triangle
# ---------------------------------------------------------------------------
def quadratic_shape_functions(L1: float, L2: float) -> FloatArray:
    """Return the 6 quadratic shape functions at barycentric (L1, L2, L3=1-L1-L2).

    Parameters
    ----------
    L1, L2 : float
        First two barycentric coordinates. L3 = 1 - L1 - L2.

    Returns
    -------
    (6,) array of shape function values [N1..N6].
    """
    L3 = 1.0 - L1 - L2
    return np.array([
        L1 * (2.0 * L1 - 1.0),   # vertex 0
        L2 * (2.0 * L2 - 1.0),   # vertex 1
        L3 * (2.0 * L3 - 1.0),   # vertex 2
        4.0 * L1 * L2,           # edge 0-1
        4.0 * L2 * L3,           # edge 1-2
        4.0 * L3 * L1,           # edge 2-0
    ])


def quadratic_shape_gradients_ref(L1: float, L2: float) -> FloatArray:
    """Return gradients of shape functions w.r.t. (L1, L2) on reference triangle.

    Returns
    -------
    (6, 2) array: dN_i/dL1, dN_i/dL2.
    """
    L3 = 1.0 - L1 - L2
    # dL3/dL1 = dL3/dL2 = -1, so d(4*L2*L3)/dL1 = -4*L2 etc.
    return np.array([
        [4.0 * L1 - 1.0, 0.0],                    # dN1/dL1, dN1/dL2
        [0.0, 4.0 * L2 - 1.0],                    # dN2
        [-4.0 * L3 + 1.0, -4.0 * L3 + 1.0],       # dN3 (chain rule: dL3/dL1=-1, dL3/dL2=-1)
        [4.0 * L2, 4.0 * L1],                     # dN4 = 4*L1*L2
        [-4.0 * L2, 4.0 * (L3 - L2)],             # dN5 = 4*L2*L3
        [4.0 * (L3 - L1), -4.0 * L1],             # dN6 = 4*L3*L1
    ])


def partition_of_unity_check(L1: float, L2: float, tol: float = 1e-14) -> bool:
    """Verify that shape functions sum to 1 (partition of unity)."""
    N = quadratic_shape_functions(L1, L2)
    return abs(np.sum(N) - 1.0) < tol


# ---------------------------------------------------------------------------
# Quadrature rules on the reference triangle
# ---------------------------------------------------------------------------
def gauss_triangle_rule(order: int) -> tuple[FloatArray, FloatArray]:
    """Return Gauss-Legendre quadrature nodes and weights on the reference triangle.

    Uses a Duffy transform from the unit square to the triangle:
        L1 = u, L2 = v*(1-u), with Jacobian (1-u).

    Parameters
    ----------
    order : int
        Number of Gauss points per direction (total = order^2).

    Returns
    -------
    nodes : (order^2, 2) array of (L1, L2) coordinates.
    weights : (order^2,) array of quadrature weights (sum = 0.5 = area of ref triangle).
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    nodes_1d, weights_1d = leggauss(order)
    # Map from [-1,1] to [0,1]
    u = 0.5 * (nodes_1d + 1.0)
    w_u = 0.5 * weights_1d
    v = 0.5 * (nodes_1d + 1.0)
    w_v = 0.5 * weights_1d

    # Duffy transform: L1 = u, L2 = v*(1-u), Jacobian = (1-u)
    L1 = np.repeat(u, order)
    L2 = np.tile(v, order) * (1.0 - L1)
    weights = np.outer(w_u, w_v).ravel() * (1.0 - L1)

    return np.column_stack([L1, L2]), weights


def duffy_self_quadrature(
    apex_node: int, order: int = 12
) -> tuple[FloatArray, FloatArray]:
    """Quadrature rule for singular integrals with target at a vertex node.

    The Duffy transform from the unit square to the triangle, with the
    singularity at the apex, cancels the 1/r kernel singularity:
        r = u * w(v), dS = 2*A*u*du*dv, so 1/r * dS = 2*A*w(v)/w(v) * du*dv = 2*A*du*dv.

    Parameters
    ----------
    apex_node : int
        Index (0, 1, or 2) of the vertex where the target sits.
    order : int
        Gauss order per direction.

    Returns
    -------
    nodes : (order^2, 2) barycentric coordinates.
    weights : (order^2,) weights with the 1/r singularity already cancelled.
    """
    if apex_node not in (0, 1, 2):
        raise ValueError("apex_node must be 0, 1, or 2 (a vertex)")
    nodes_1d, weights_1d = leggauss(order)
    u = 0.5 * (nodes_1d + 1.0)
    w_u = 0.5 * weights_1d
    v = 0.5 * (nodes_1d + 1.0)
    w_v = 0.5 * weights_1d

    # Standard Duffy: L1 = u, L2 = v*(1-u)
    L1_duffy = np.repeat(u, order)
    L2_duffy = np.tile(v, order) * (1.0 - L1_duffy)
    # Jacobian factor (1-u) already in weights; the 1/r cancellation gives
    # an additional factor that we handle in the integration routine.
    weights_duffy = np.outer(w_u, w_v).ravel() * (1.0 - L1_duffy)

    # Permute barycentric coordinates so the apex is at the right vertex
    if apex_node == 0:
        L1, L2 = L1_duffy, L2_duffy
    elif apex_node == 1:
        L1, L2 = L2_duffy, 1.0 - L1_duffy - L2_duffy
    else:  # apex_node == 2
        L1, L2 = 1.0 - L1_duffy - L2_duffy, L1_duffy

    return np.column_stack([L1, L2]), weights_duffy


# ---------------------------------------------------------------------------
# Node bookkeeping and vectorised shape functions
# ---------------------------------------------------------------------------
#: Barycentric coordinates (L1, L2) of the six element nodes (see NODE_LABELS).
NODE_BARYCENTRIC: FloatArray = np.array(
    [
        [1.0, 0.0],  # v0
        [0.0, 1.0],  # v1
        [0.0, 0.0],  # v2
        [0.5, 0.5],  # mid01
        [0.0, 0.5],  # mid12
        [0.5, 0.0],  # mid20
    ]
)

#: Reference-triangle vertices in (L1, L2) coordinates.
_REF_VERTICES: FloatArray = NODE_BARYCENTRIC[:3]


def quadratic_shape_functions_array(L: FloatArray) -> FloatArray:
    """Vectorised quadratic shape functions.

    Parameters
    ----------
    L : (Q, 2) barycentric coordinates (L1, L2).

    Returns
    -------
    (Q, 6) shape-function values.
    """
    L = np.atleast_2d(np.asarray(L, dtype=float))
    L1, L2 = L[:, 0], L[:, 1]
    L3 = 1.0 - L1 - L2
    return np.column_stack(
        [
            L1 * (2.0 * L1 - 1.0),
            L2 * (2.0 * L2 - 1.0),
            L3 * (2.0 * L3 - 1.0),
            4.0 * L1 * L2,
            4.0 * L2 * L3,
            4.0 * L3 * L1,
        ]
    )


def quadratic_shape_gradients_ref_array(L: FloatArray) -> FloatArray:
    """Vectorised reference-coordinate gradients of the shape functions.

    Returns
    -------
    (Q, 6, 2) array of dN/d(L1, L2).
    """
    L = np.atleast_2d(np.asarray(L, dtype=float))
    L1, L2 = L[:, 0], L[:, 1]
    L3 = 1.0 - L1 - L2
    zero = np.zeros_like(L1)
    return np.stack(
        [
            np.column_stack([4.0 * L1 - 1.0, zero]),
            np.column_stack([zero, 4.0 * L2 - 1.0]),
            np.column_stack([1.0 - 4.0 * L3, 1.0 - 4.0 * L3]),
            np.column_stack([4.0 * L2, 4.0 * L1]),
            np.column_stack([-4.0 * L2, 4.0 * (L3 - L2)]),
            np.column_stack([4.0 * (L3 - L1), -4.0 * L1]),
        ],
        axis=1,
    )


def triangle_apex_quadrature(
    apex_L: Sequence[float], order: int = 16
) -> tuple[FloatArray, FloatArray]:
    """Singularity-cancelling quadrature on the reference triangle.

    The reference triangle is split into sub-triangles that all carry the
    apex (the singular point) as a vertex; on each sub-triangle the standard
    Duffy map ``x = apex + u*((1-v)*a + v*b)`` is applied.  Its Jacobian
    contains a factor ``u`` and the distance to the apex behaves like
    ``r = u*|...|``, so the ``1/r`` factor of the single-layer kernel (and
    the ``1/r`` behaviour of the curved double-layer kernel) is cancelled
    exactly and the transformed integrand is smooth.

    Parameters
    ----------
    apex_L : sequence of 2 floats
        Barycentric (L1, L2) of the singular point (element node, or an
        arbitrary interior point).
    order : int
        Gauss order per direction on each sub-triangle.

    Returns
    -------
    nodes : (Q, 2) reference coordinates.
    weights : (Q,) weights with respect to the reference-triangle measure
        (``sum(weights) == 0.5``, the area of the reference triangle).
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    apex = np.asarray(apex_L, dtype=float).reshape(2)

    # Locate the apex: reference vertex, edge point, or interior point.
    sub_triangles: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    on_vertex = [k for k in range(3) if np.allclose(apex, _REF_VERTICES[k], atol=1e-12)]
    if on_vertex:
        k = on_vertex[0]
        others = [m for m in range(3) if m != k]
        sub_triangles.append((apex, _REF_VERTICES[others[0]], _REF_VERTICES[others[1]]))
    else:
        on_edge: tuple[int, int, int] | None = None
        for k in range(3):
            p, q = _REF_VERTICES[k], _REF_VERTICES[(k + 1) % 3]
            edge = q - p
            s = float((apex - p) @ edge) / float(edge @ edge)
            if -1e-12 <= s <= 1.0 + 1e-12:
                closest = p + s * edge
                if np.linalg.norm(apex - closest) < 1e-10:
                    on_edge = (k, (k + 1) % 3, (k + 2) % 3)
                    break
        if on_edge is not None:
            i, j, o = on_edge
            sub_triangles.append((apex, _REF_VERTICES[i], _REF_VERTICES[o]))
            sub_triangles.append((apex, _REF_VERTICES[o], _REF_VERTICES[j]))
        else:
            for k in range(3):
                sub_triangles.append(
                    (apex, _REF_VERTICES[k], _REF_VERTICES[(k + 1) % 3])
                )

    nodes_1d, weights_1d = leggauss(order)
    u = 0.5 * (nodes_1d + 1.0)
    w_u = 0.5 * weights_1d
    v = 0.5 * (nodes_1d + 1.0)
    w_v = 0.5 * weights_1d
    U, V = np.meshgrid(u, v, indexing="ij")
    U = U.ravel()
    V = V.ravel()
    w_uv = np.outer(w_u, w_v).ravel()

    nodes_out: list[FloatArray] = []
    weights_out: list[FloatArray] = []
    for apex_pt, a_pt, b_pt in sub_triangles:
        e1 = a_pt - apex_pt
        e2 = b_pt - apex_pt
        area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        if area <= 0.0:
            raise ValueError("degenerate sub-triangle in apex quadrature")
        pts = (
            apex_pt[None, :]
            + (U * (1.0 - V))[:, None] * e1[None, :]
            + (U * V)[:, None] * e2[None, :]
        )
        nodes_out.append(pts)
        weights_out.append(w_uv * U * (2.0 * area))

    return np.vstack(nodes_out), np.concatenate(weights_out)


# ---------------------------------------------------------------------------
# Quadratic triangle geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuadraticTriangle:
    """A single 6-node curved triangular element.

    Parameters
    ----------
    nodes : (6, 3) array
        Physical coordinates of the 6 nodes:
        [vertex0, vertex1, vertex2, mid01, mid12, mid20].
    """
    nodes: FloatArray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        if nodes.shape != (6, 3):
            raise ValueError("nodes must have shape (6, 3)")
        if not np.isfinite(nodes).all():
            raise ValueError("nodes must all be finite")
        # Check non-degeneracy via the flat triangle formed by vertices
        v0, v1, v2 = nodes[0], nodes[1], nodes[2]
        area = 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
        if area <= 0.0:
            raise ValueError("quadratic triangle is degenerate (zero vertex area)")
        object.__setattr__(self, "nodes", nodes.copy())

    @property
    def vertex_area(self) -> float:
        """Area of the flat triangle formed by the 3 vertex nodes."""
        v0, v1, v2 = self.nodes[0], self.nodes[1], self.nodes[2]
        return 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))

    def map_to_physical(self, L1: float, L2: float) -> FloatArray:
        """Map reference triangle coordinates to physical space."""
        N = quadratic_shape_functions(L1, L2)
        return N @ self.nodes

    def jacobian(self, L1: float, L2: float) -> tuple[FloatArray, float]:
        """Compute the Jacobian of the isoparametric mapping.

        Returns
        -------
        normal_unnorm : (3,) unnormalized normal (= dx/dL1 x dx/dL2).
        det : float |dx/dL1 x dx/dL2| (surface metric determinant).
        """
        dNdL = quadratic_shape_gradients_ref(L1, L2)  # (6, 2)
        dx_dL1 = dNdL[:, 0] @ self.nodes  # (3,)
        dx_dL2 = dNdL[:, 1] @ self.nodes  # (3,)
        cross = np.cross(dx_dL1, dx_dL2)
        det = float(np.linalg.norm(cross))
        return cross, det

    def _geometry_at(self, L: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Physical points, shape functions, metric determinant and unit normals.

        Returns
        -------
        X : (Q, 3) physical quadrature points.
        N : (Q, 6) shape-function values.
        det : (Q,) surface metric (``dS = det * dL1 * dL2``).
        normals : (Q, 3) outward unit normals of the isoparametric surface.
        """
        N = quadratic_shape_functions_array(L)
        X = N @ self.nodes
        dN = quadratic_shape_gradients_ref_array(L)  # (Q, 6, 2)
        dx_dL1 = dN[:, :, 0] @ self.nodes
        dx_dL2 = dN[:, :, 1] @ self.nodes
        cross = np.cross(dx_dL1, dx_dL2)
        det = np.linalg.norm(cross, axis=1)
        if np.any(det <= 0.0):
            raise ValueError("degenerate isoparametric mapping (zero metric)")
        return X, N, det, cross / det[:, None]

    def integrate_single_layer_at_node(
        self, local_node: int, order: int = 16
    ) -> FloatArray:
        """Single-layer self integral ``int_T N_k G(x_node, y) dS`` at a node.

        Works for all six nodes because the singular quadrature
        (:func:`triangle_apex_quadrature`) handles a singular point that is a
        vertex *or* an edge midpoint.

        Parameters
        ----------
        local_node : int
            Local node index 0..5 (0-2 vertices, 3-5 edge midpoints).
        order : int
            Gauss order per direction on each Duffy sub-triangle.

        Returns
        -------
        (6,) array of ``int_T N_k / (4*pi*r) dS`` for each shape function.
        """
        if not 0 <= local_node < NODES_PER_ELEMENT:
            raise ValueError("local_node must be in 0..5")
        target = self.nodes[local_node]
        L, w = triangle_apex_quadrature(NODE_BARYCENTRIC[local_node], order=order)
        X, N, det, _ = self._geometry_at(L)
        r = np.linalg.norm(X - target, axis=1)
        scale = max(1.0, float(np.max(np.abs(self.nodes))))
        ok = r > 1e-14 * scale
        if not np.any(ok):
            raise ValueError("no quadrature point away from the singular node")
        kern = w[ok] * det[ok] / (FOUR_PI * r[ok])
        return kern @ N[ok]

    def integrate_double_layer_at_node(
        self, local_node: int, order: int = 16
    ) -> FloatArray:
        """Double-layer self integral ``int_T N_k dG/dn dS`` at a node.

        The kernel ``K(x, y) = dG/dn_y = +n(y).(x-y) / (4*pi*|x-y|**3)`` is
        evaluated with the *exact* isoparametric normal ``n(y)``.  For a flat
        panel ``(x-y).n == 0`` for every target in the panel plane, so the
        self contribution vanishes identically; for a curved panel the kernel
        behaves like ``1/r`` and the Duffy map of
        :func:`triangle_apex_quadrature` cancels it.

        Returns
        -------
        (6,) array of principal-value ``int_T N_k dG/dn dS`` (the free term
        ``-/+ 1/2 delta_ik`` is *not* included).
        """
        if not 0 <= local_node < NODES_PER_ELEMENT:
            raise ValueError("local_node must be in 0..5")
        target = self.nodes[local_node]
        L, w = triangle_apex_quadrature(NODE_BARYCENTRIC[local_node], order=order)
        X, N, det, normals = self._geometry_at(L)
        delta = target - X  # (x - y): the sign convention of dG/dn_y
        r = np.linalg.norm(delta, axis=1)
        scale = max(1.0, float(np.max(np.abs(self.nodes))))
        ok = r > 1e-14 * scale
        if not np.any(ok):
            raise ValueError("no quadrature point away from the singular node")
        r3 = r[ok] ** 3
        K = np.einsum("qj,qj->q", normals[ok], delta[ok]) / (FOUR_PI * r3)
        kern = w[ok] * det[ok] * K
        return kern @ N[ok]

    def node_self_matrices(self, order: int = 16) -> tuple[FloatArray, FloatArray]:
        """Self-panel blocks ``S[i, k]``, ``D[i, k]`` for the six node targets."""
        S = np.empty((NODES_PER_ELEMENT, NODES_PER_ELEMENT))
        D = np.empty((NODES_PER_ELEMENT, NODES_PER_ELEMENT))
        for i in range(NODES_PER_ELEMENT):
            S[i] = self.integrate_single_layer_at_node(i, order=order)
            D[i] = self.integrate_double_layer_at_node(i, order=order)
        return S, D

    def integrate_single_layer_self(
        self, apex_node: int, order: int = 16
    ) -> FloatArray:
        """Single-layer self integral at a *vertex* node (legacy entry point).

        Kept for backwards compatibility; equivalent to
        :meth:`integrate_single_layer_at_node` for ``apex_node`` in 0..2.
        """
        if apex_node not in (0, 1, 2):
            raise ValueError("apex_node must be 0, 1, or 2")
        return self.integrate_single_layer_at_node(apex_node, order=order)

    def target_blocks(
        self, targets: FloatArray, order: int = 8
    ) -> tuple[FloatArray, FloatArray]:
        """Kernel blocks of this element against *many* collocation targets.

        Vectorised over the quadrature points and the targets (used by the
        assembler to avoid a Python loop over node/element pairs).

        Parameters
        ----------
        targets : (T, 3) collocation points, generally *not* on this element.
        order : int Gauss order.

        Returns
        -------
        S_blk, D_blk : (T, 6) arrays with
            ``S_blk[t, k] = int_T N_k G(x_t, y) dS`` and
            ``D_blk[t, k] = int_T N_k dG/dn_y(x_t, y) dS``.
            Entries whose target lies on the element (the singular self rows)
            are returned as zero; those must be filled with
            :meth:`node_self_matrices`.
        """
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        L, w = gauss_triangle_rule(order)
        X, N, det, normals = self._geometry_at(L)
        diff = targets[:, None, :] - X[None, :, :]  # (T, Q, 3)
        r = np.linalg.norm(diff, axis=2)            # (T, Q)
        scale = max(1.0, float(np.max(np.abs(self.nodes))))
        singular = r <= 1e-12 * scale
        r_safe = np.where(singular, 1.0, r)
        wgt = w * det / FOUR_PI                     # (Q,)
        S_blk = (1.0 / r_safe) @ (wgt[:, None] * N)
        radial = np.einsum("tqj,qj->tq", diff, normals) / r_safe ** 3
        D_blk = (radial * wgt[None, :]) @ N
        if np.any(singular):
            S_blk[singular] = 0.0
            D_blk[singular] = 0.0
        return S_blk, D_blk

    def integrate_single_layer(
        self, target: FloatArray, order: int = 8
    ) -> FloatArray:
        """Integrate N_k * G over this triangle for a non-singular target.

        Parameters
        ----------
        target : (3,) point not on this element.
        order : int Gauss order.

        Returns
        -------
        (6,) array of int_T N_k / (4*pi*|x-y|) dS.
        """
        target = np.asarray(target, dtype=float)
        L, w = gauss_triangle_rule(order)
        X, N, det, _ = self._geometry_at(L)
        r = np.linalg.norm(X - target, axis=1)
        if np.any(r < 1e-15):
            raise ValueError("target lies on the element; use self-integral")
        kern = w * det / (FOUR_PI * r)
        return kern @ N

    def integrate_double_layer(
        self, target: FloatArray, order: int = 8
    ) -> FloatArray:
        """Integrate N_k * dG/dn over this triangle for a non-singular target.

        Returns
        -------
        (6,) array of int_T N_k dG/dn dS.
        """
        target = np.asarray(target, dtype=float)
        L, w = gauss_triangle_rule(order)
        X, N, det, normals = self._geometry_at(L)
        delta = target - X  # (x - y)
        r = np.linalg.norm(delta, axis=1)
        if np.any(r < 1e-15):
            raise ValueError("target lies on the element; use self-integral")
        K = np.einsum("qj,qj->q", normals, delta) / (FOUR_PI * r ** 3)
        kern = w * det * K
        return kern @ N


# ---------------------------------------------------------------------------
# Quadratic surface mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuadraticSurfaceMesh:
    """A closed surface mesh of 6-node quadratic triangles.

    Parameters
    ----------
    vertices : (V, 3) array of vertex coordinates.
    faces : (F, 6) array of node indices per element.
        Columns 0-2 are vertex indices, columns 3-5 are edge midpoint indices.
        Edge midpoints are stored as separate nodes to allow curved geometry.

    Notes
    -----
    The outward-normal convention is enforced on construction: the
    isoparametric normal follows the face winding, so faces whose winding
    makes the normal point toward the vertex centroid are re-wound
    (``v1 <-> v2`` with the edge midpoints carried along).  The assembled
    operators therefore always use outward normals for closed star-shaped
    surfaces, including mirrored copies of valid meshes.
    """
    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.array(self.faces, dtype=np.int64)  # private copy (may be re-wound)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices must all be finite")
        if faces.ndim != 2 or faces.shape[1] != 6:
            raise ValueError("faces must have shape (F, 6)")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("faces must be integer")
        node_count = vertices.shape[0]
        if np.any(faces < 0) or np.any(faces >= node_count):
            raise ValueError("face indices out of range")
        # Enforce the outward-normal convention.  The isoparametric normal is
        # the cross product dx/dL1 x dx/dL2, so its direction follows the face
        # winding: a mesh whose faces are wound inward (e.g. a mirrored copy
        # of a valid mesh) would silently flip D and the recovered flux even
        # though every identity on the original mesh still holds.  For a
        # closed star-shaped surface the outward direction at an element is
        # approximated by (element_centre - vertex_centroid); re-wind any face
        # whose flat normal points the other way.  Re-winding swaps v1 <-> v2
        # and carries each edge midpoint with its edge:
        # (a, b, c, m_ab, m_bc, m_ca) -> (a, c, b, m_ca, m_bc, m_ab).
        # Limitation: the heuristic assumes the surface is star-shaped with
        # respect to the vertex centroid (all built-in generators are).
        centre = vertices.mean(axis=0)
        for f in range(faces.shape[0]):
            row = faces[f]
            v0, v1, v2 = vertices[row[0]], vertices[row[1]], vertices[row[2]]
            if float(np.dot(np.cross(v1 - v0, v2 - v0),
                            vertices[row].mean(axis=0) - centre)) < 0.0:
                faces[f] = row[[0, 2, 1, 5, 4, 3]]
        object.__setattr__(self, "vertices", vertices.copy())
        object.__setattr__(self, "faces", faces)

    @property
    def element_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.vertices.shape[0])

    def element(self, index: int) -> QuadraticTriangle:
        """Return the quadratic triangle for element ``index``."""
        node_ids = self.faces[index]
        return QuadraticTriangle(self.vertices[node_ids])

    @classmethod
    def from_flat_triangle_mesh(
        cls, vertices: FloatArray, faces: IntArray
    ) -> "QuadraticSurfaceMesh":
        """Build a quadratic mesh from a flat (3-node) triangle mesh.

        Edge midpoints are computed as the average of the two endpoint vertices.
        This gives a valid quadratic mesh but with flat geometry (midpoints
        lie on the straight edges). For curved surfaces, use
        :meth:`from_flat_with_projection` to project midpoints onto the true surface.
        """
        vertices = np.asarray(vertices, dtype=float)
        faces_flat = np.asarray(faces, dtype=np.int64)
        if faces_flat.shape[1] != 3:
            raise ValueError("flat faces must have shape (F, 3)")

        # Build edge midpoint lookup
        edge_to_mid: dict[tuple[int, int], int] = {}
        midpoints: list[list[float]] = []
        new_faces: list[list[int]] = []

        for f in range(faces_flat.shape[0]):
            v0, v1, v2 = int(faces_flat[f, 0]), int(faces_flat[f, 1]), int(faces_flat[f, 2])
            edge_nodes = []
            for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
                key = (min(a, b), max(a, b))
                if key not in edge_to_mid:
                    mid = 0.5 * (vertices[a] + vertices[b])
                    edge_to_mid[key] = len(vertices) + len(midpoints)
                    midpoints.append(mid.tolist())
                edge_nodes.append(edge_to_mid[key])
            new_faces.append([v0, v1, v2] + edge_nodes)

        all_vertices = np.vstack([vertices, np.array(midpoints)])
        return cls(all_vertices, np.array(new_faces, dtype=np.int64))

    @classmethod
    def from_flat_with_projection(
        cls,
        vertices: FloatArray,
        faces: IntArray,
        project_fn,
    ) -> "QuadraticSurfaceMesh":
        """Build a quadratic mesh with midpoints projected onto a curved surface.

        Parameters
        ----------
        vertices : (V, 3) vertex coordinates (should lie on the true surface).
        faces : (F, 3) flat triangle connectivity.
        project_fn : callable(x) -> x_projected
            Function that projects a point onto the true surface.
        """
        base = cls.from_flat_triangle_mesh(vertices, faces)
        # Project midpoints
        V_orig = vertices.shape[0]
        projected = base.vertices.copy()
        for i in range(V_orig, projected.shape[0]):
            projected[i] = project_fn(projected[i])
        return cls(projected, base.faces)


# ---------------------------------------------------------------------------
# Icosphere with quadratic nodes
# ---------------------------------------------------------------------------
def icosphere_quadratic(subdivisions: int, radius: float = 1.0) -> QuadraticSurfaceMesh:
    """Create a quadratic mesh of a sphere using icosphere topology.

    The vertex nodes lie on the sphere; edge midpoints are computed as the
    average of their endpoints and then projected onto the sphere, giving
    a genuinely curved (isoparametric) representation.
    """
    from .geometry3d import icosphere
    flat_mesh = icosphere(subdivisions, radius=radius)
    return QuadraticSurfaceMesh.from_flat_with_projection(
        flat_mesh.vertices,
        flat_mesh.faces,
        project_fn=lambda x: radius * x / np.linalg.norm(x),
    )


def ellipsoid_quadratic(
    subdivisions: int,
    axes: Sequence[float] = (1.0, 1.0, 1.0),
    centre: Sequence[float] = (0.0, 0.0, 0.0),
) -> QuadraticSurfaceMesh:
    """Create a quadratic mesh of an axis-aligned ellipsoid.

    The mesh is the *affine image* of the icosphere mesh of
    :func:`icosphere_quadratic`: ``x = centre + diag(axes) * x_sphere``.  Every
    node therefore lies exactly on the ellipsoid
    ``|diag(axes)^-1 (x - centre)| = 1`` and each element is the exact image of
    a curved sphere patch, i.e. a genuinely curved *non-spherical* isoparametric
    panel.  (Anisotropic ``axes`` distort the elements; that distortion is a
    property of the geometry, not of the mesh generator.)

    Parameters
    ----------
    subdivisions : int
        Icosphere refinement level (same meaning as :func:`icosphere_quadratic`).
    axes : sequence of 3 floats
        Semi-axis lengths ``(a, b, c)`` along x, y, z.
    centre : sequence of 3 floats
        Centre of the ellipsoid.

    Returns
    -------
    QuadraticSurfaceMesh
        Closed quadratic mesh of the ellipsoid.
    """
    axes_arr = np.asarray(axes, dtype=float).reshape(3)
    centre_arr = np.asarray(centre, dtype=float).reshape(3)
    if np.any(axes_arr <= 0.0):
        raise ValueError("axes must all be positive")

    sphere = icosphere_quadratic(subdivisions)
    vertices = centre_arr + sphere.vertices * axes_arr
    return QuadraticSurfaceMesh(vertices, sphere.faces.copy())


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------
def assemble_quadratic_matrices(
    mesh: QuadraticSurfaceMesh,
    quadrature_order: int = 8,
    self_order: int = 16,
) -> tuple[FloatArray, FloatArray]:
    """Assemble single-layer (S) and double-layer (D) matrices for quadratic BEM.

    The system has ``mesh.node_count`` unknowns (one per node, shared by the
    incident elements)::

        S_ik = sum_elements int_{T_e} N_k G(x_i, y) dS_y
        D_ik = sum_elements int_{T_e} N_k dG/dn_y(x_i, y) dS_y

    ``D`` is the **principal-value** double-layer operator: the free term
    (``-1/2`` jump for a smooth boundary, in the convention of the module
    docstring) is *not* included.  Consequently, for a closed surface the row
    sums of ``D`` satisfy the solid-angle identity

        sum_k D_ik = PV int_S dG/dn dS = -1/2        (exact identity)

    which is asserted by the operator tests.

    Both matrices use the exact isoparametric normal of the curved element in
    the double-layer kernel.  The singular (self-panel) blocks — the target is
    one of the six nodes of the source panel, including the edge midpoints —
    are integrated with the Duffy-type :func:`triangle_apex_quadrature`;
    non-singular blocks use a Gauss rule on the reference triangle.

    Parameters
    ----------
    mesh : QuadraticSurfaceMesh
    quadrature_order : int
        Gauss order for non-singular panels.
    self_order : int
        Gauss order per direction for the singular self-panel quadrature.

    Returns
    -------
    S : (node_count, node_count) single-layer matrix.
    D : (node_count, node_count) principal-value double-layer matrix.
    """
    n_nodes = mesh.node_count
    n_elems = mesh.element_count
    S = np.zeros((n_nodes, n_nodes))
    D = np.zeros((n_nodes, n_nodes))

    all_targets = mesh.vertices
    for e in range(n_elems):
        elem = mesh.element(e)
        node_ids = mesh.faces[e]

        # Non-singular part: element e against every collocation node.
        S_blk, D_blk = elem.target_blocks(all_targets, order=quadrature_order)
        # Singular part: the six nodes of e are collocated on e itself, so
        # their rows are replaced by the accurate Duffy self quadrature
        # (including the edge midpoints, whose integrals are also singular).
        S_self, D_self = elem.node_self_matrices(order=self_order)
        S_blk[node_ids] = S_self
        D_blk[node_ids] = D_self

        S[:, node_ids] += S_blk
        D[:, node_ids] += D_blk

    return S, D


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuadraticBEMResult:
    """Solution from a quadratic-panel BEM solve."""
    nodal_values: FloatArray
    single_layer_matrix: FloatArray
    condition_number: float
    residual_l2: float
    relative_residual_l2: float


def solve_laplace_dirichlet_quadratic(
    mesh: QuadraticSurfaceMesh,
    boundary_values: FloatArray,
    quadrature_order: int = 8,
    self_order: int = 16,
) -> QuadraticBEMResult:
    """Solve the exterior 3-D Laplace Dirichlet problem with quadratic panels.

    Uses the indirect (source density) formulation:
        S q = u_prescribed
    where S is the single-layer operator and q is the source density.

    Parameters
    ----------
    mesh : QuadraticSurfaceMesh
    boundary_values : (node_count,) prescribed Dirichlet data at nodes.
    quadrature_order : int
    self_order : int

    Returns
    -------
    QuadraticBEMResult with the solved source density q.
    """
    values = np.asarray(boundary_values, dtype=float)
    if values.shape != (mesh.node_count,):
        raise ValueError(
            f"boundary_values must have shape ({mesh.node_count},)"
        )
    if not np.isfinite(values).all():
        raise ValueError("boundary_values must all be finite")

    S, _ = assemble_quadratic_matrices(
        mesh, quadrature_order=quadrature_order, self_order=self_order
    )
    condition = float(np.linalg.cond(S))
    q = np.linalg.solve(S, values)
    residual = float(np.linalg.norm(S @ q - values))
    rhs_norm = float(np.linalg.norm(values))
    return QuadraticBEMResult(
        nodal_values=q,
        single_layer_matrix=S,
        condition_number=condition,
        residual_l2=residual,
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0 else residual,
    )


@dataclass(frozen=True)
class QuadraticNeumannResult:
    """Normal-derivative recovery from Dirichlet data via the second-kind BIE.

    Attributes
    ----------
    normal_derivative : (node_count,) recovered outward ``q = du/dn``.
    single_layer_matrix, double_layer_matrix : assembled operators.
    condition_number : cond(S).
    relative_residual_l2 : ``||(1/2 I - D)u + S q|| / ||(1/2 I - D)u||``.
    operator_row_sum_error : ``max_i |sum_k D_ik + 1/2|`` (solid-angle
        identity residual; a pure operator diagnostic, no solution involved).
    """

    normal_derivative: FloatArray
    single_layer_matrix: FloatArray
    double_layer_matrix: FloatArray
    condition_number: float
    relative_residual_l2: float
    operator_row_sum_error: float


def solve_neumann_from_dirichlet_quadratic(
    mesh: QuadraticSurfaceMesh,
    boundary_values: FloatArray,
    quadrature_order: int = 8,
    self_order: int = 16,
) -> QuadraticNeumannResult:
    """Recover the outward normal derivative from Dirichlet data (direct BIE).

    The exterior second-kind boundary integral equation (module docstring) is::

        (1/2 I - D) u = -S q        =>        q = -S^{-1} (1/2 I - D) u

    This is an *independent* route to the boundary flux compared with the
    indirect single-layer solve of :func:`solve_laplace_dirichlet_quadratic`
    (``S q = u``): it exercises the double-layer operator ``D`` and therefore
    fails if ``D`` (or its singular self-panel quadrature) is wrong.

    Parameters
    ----------
    mesh : QuadraticSurfaceMesh
    boundary_values : (node_count,) prescribed Dirichlet data at the nodes.
    quadrature_order, self_order : int
        Quadrature orders, see :func:`assemble_quadratic_matrices`.

    Returns
    -------
    QuadraticNeumannResult
    """
    values = np.asarray(boundary_values, dtype=float)
    if values.shape != (mesh.node_count,):
        raise ValueError(f"boundary_values must have shape ({mesh.node_count},)")
    if not np.isfinite(values).all():
        raise ValueError("boundary_values must all be finite")

    S, D = assemble_quadratic_matrices(
        mesh, quadrature_order=quadrature_order, self_order=self_order
    )
    n = mesh.node_count
    second_kind = 0.5 * np.eye(n) - D
    rhs = second_kind @ values
    q = -np.linalg.solve(S, rhs)
    residual = float(np.linalg.norm(second_kind @ values + S @ q))
    rhs_norm = float(np.linalg.norm(rhs))
    row_sum_error = float(np.max(np.abs(D.sum(axis=1) + 0.5)))
    return QuadraticNeumannResult(
        normal_derivative=q,
        single_layer_matrix=S,
        double_layer_matrix=D,
        condition_number=float(np.linalg.cond(S)),
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0.0 else residual,
        operator_row_sum_error=row_sum_error,
    )


# ---------------------------------------------------------------------------
# Self-influence analytical vs numerical comparison
# ---------------------------------------------------------------------------
def self_influence_analytical_vs_numerical(
    mesh: QuadraticSurfaceMesh,
    quadrature_orders: Sequence[int] = (8, 12, 16, 24, 32),
) -> dict:
    """Compare analytical (Duffy) vs numerical self-influence for convergence.

    For each vertex node, computes the self-influence integral using
    increasing quadrature orders and reports the relative difference
    from the highest-order reference.

    Returns
    -------
    dict with keys:
        'max_relative_diff': max relative difference across all nodes
        'orders': list of quadrature orders tested
        'convergence': list of max relative differences per order
    """
    n_elems = mesh.element_count
    results = {"orders": list(quadrature_orders), "convergence": []}

    # Reference: highest order
    ref_order = max(quadrature_orders)
    ref_values = []
    for e in range(n_elems):
        elem = mesh.element(e)
        for apex in range(3):
            val = elem.integrate_single_layer_self(apex_node=apex, order=ref_order)
            ref_values.append(np.sum(np.abs(val)))

    ref_values = np.array(ref_values)

    for order in quadrature_orders:
        test_values = []
        for e in range(n_elems):
            elem = mesh.element(e)
            for apex in range(3):
                val = elem.integrate_single_layer_self(apex_node=apex, order=order)
                test_values.append(np.sum(np.abs(val)))
        test_values = np.array(test_values)

        # Relative difference
        mask = ref_values > 1e-15
        if np.any(mask):
            rel_diff = np.max(np.abs(test_values[mask] - ref_values[mask]) / ref_values[mask])
        else:
            rel_diff = 0.0
        results["convergence"].append(float(rel_diff))

    results["max_relative_diff"] = results["convergence"][0] if results["convergence"] else 0.0
    return results
