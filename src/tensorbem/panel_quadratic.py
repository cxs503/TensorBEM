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
    # Derivations (dL3/dL1 = dL3/dL2 = -1):
    #   N5 = 4 L2 L3      -> dN5/dL1 = -4 L2,      dN5/dL2 = 4 (L3 - L2)
    #   N6 = 4 L3 L1      -> dN6/dL1 = 4 (L3 - L1), dN6/dL2 = -4 L1
    return np.array([
        [4.0 * L1 - 1.0, 0.0],                    # dN1/dL1, dN1/dL2
        [0.0, 4.0 * L2 - 1.0],                    # dN2
        [-4.0 * L3 + 1.0, -4.0 * L3 + 1.0],      # dN3 (chain rule: dL3/dL1=-1, dL3/dL2=-1)
        [4.0 * L2, 4.0 * L1],                     # dN4
        [-4.0 * L2, 4.0 * (L3 - L2)],             # dN5
        [4.0 * (L3 - L1), -4.0 * L1],             # dN6
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


#: Barycentric coordinates ``(L1, L2, L3)`` of the six element nodes.
NODE_BARYCENTRIC: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),   # v0
    (0.0, 1.0, 0.0),   # v1
    (0.0, 0.0, 1.0),   # v2
    (0.5, 0.5, 0.0),   # mid01
    (0.0, 0.5, 0.5),   # mid12
    (0.5, 0.0, 0.5),   # mid20
)

#: Reference sub-triangles that tile the element with the node as a common
#: corner.  A vertex is a corner of the whole triangle (one patch); an edge
#: midpoint lies on the boundary, so the element is split into two patches by
#: the segment from that midpoint to the opposite vertex.
_SINGULAR_PATCHES: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((1, 2),),
    1: ((2, 0),),
    2: ((0, 1),),
    3: ((0, 2), (2, 1)),
    4: ((1, 0), (0, 2)),
    5: ((2, 1), (1, 0)),
}


def _duffy_patch_quadrature(
    singular_node: int,
    corner_a: int,
    corner_b: int,
    order: int = 12,
) -> tuple[FloatArray, FloatArray]:
    """Duffy quadrature on one reference sub-triangle with the singularity at a corner.

    The sub-triangle has vertices ``singular_node``, ``corner_a`` and
    ``corner_b`` (all element node indices).  It is parametrised as

        P(u, v) = S + u * [(1 - v) (A - S) + v (B - S)],   u, v in [0, 1],

    so the distance to the singular corner behaves like ``r ~ u * const`` while
    the area element carries a factor ``u``; their product is smooth and Gauss
    quadrature converges geometrically.

    Returns
    -------
    nodes : (order**2, 2) array of ``(L1, L2)`` barycentric coordinates.
    weights : (order**2,) weights such that
        ``int_patch f dS ~ sum(weights * jacobian(L1, L2) * f)``.
    """
    if singular_node not in range(6):
        raise ValueError("singular_node must be in 0..5")
    if order < 1:
        raise ValueError("order must be >= 1")

    S = np.array(NODE_BARYCENTRIC[singular_node])
    A = np.array(NODE_BARYCENTRIC[corner_a])
    B = np.array(NODE_BARYCENTRIC[corner_b])

    # Jacobian of (u, v) -> (L1, L2) is u * |det[(A-S), (B-S)]| in the L1/L2 plane.
    det2 = abs(
        (A[0] - S[0]) * (B[1] - S[1]) - (A[1] - S[1]) * (B[0] - S[0])
    )
    if det2 <= 0.0:
        raise ValueError("degenerate singular patch")

    nodes_1d, weights_1d = leggauss(order)
    u = 0.5 * (nodes_1d + 1.0)
    w_u = 0.5 * weights_1d
    v = 0.5 * (nodes_1d + 1.0)
    w_v = 0.5 * weights_1d

    U, V = np.meshgrid(u, v, indexing="ij")
    U = U.ravel()
    V = V.ravel()
    weights = (np.outer(w_u, w_v).ravel()) * U * det2

    L = (
        S[None, :]
        + U[:, None] * ((1.0 - V)[:, None] * (A - S)[None, :] + V[:, None] * (B - S)[None, :])
    )
    nodes = np.column_stack([L[:, 0], L[:, 1]])
    return nodes, weights


def _shape_matrix(L1: FloatArray, L2: FloatArray) -> FloatArray:
    """Return the 6 shape functions at each of ``N`` barycentric points.

    Vectorised twin of :func:`quadratic_shape_functions`.
    """
    L1 = np.asarray(L1, dtype=float)
    L2 = np.asarray(L2, dtype=float)
    L3 = 1.0 - L1 - L2
    return np.column_stack([
        L1 * (2.0 * L1 - 1.0),
        L2 * (2.0 * L2 - 1.0),
        L3 * (2.0 * L3 - 1.0),
        4.0 * L1 * L2,
        4.0 * L2 * L3,
        4.0 * L3 * L1,
    ])


def _shape_gradients_matrix(L1: FloatArray, L2: FloatArray) -> FloatArray:
    """Return ``dN_k/dL1`` and ``dN_k/dL2`` at each of ``N`` points.

    Shape ``(N, 6, 2)``; vectorised twin of
    :func:`quadratic_shape_gradients_ref`.
    """
    L1 = np.asarray(L1, dtype=float)
    L2 = np.asarray(L2, dtype=float)
    L3 = 1.0 - L1 - L2
    g = np.empty(L1.shape + (6, 2))
    g[..., 0, 0] = 4.0 * L1 - 1.0
    g[..., 0, 1] = 0.0
    g[..., 1, 0] = 0.0
    g[..., 1, 1] = 4.0 * L2 - 1.0
    g[..., 2, 0] = 1.0 - 4.0 * L3
    g[..., 2, 1] = 1.0 - 4.0 * L3
    g[..., 3, 0] = 4.0 * L2
    g[..., 3, 1] = 4.0 * L1
    g[..., 4, 0] = -4.0 * L2
    g[..., 4, 1] = 4.0 * (L3 - L2)
    g[..., 5, 0] = 4.0 * (L3 - L1)
    g[..., 5, 1] = -4.0 * L1
    return g


def duffy_self_quadrature(
    apex_node: int, order: int = 12
) -> tuple[FloatArray, FloatArray]:
    """Quadrature rule for the singular integral with the target at an element node.

    The ``1/r`` singularity is cancelled by the Duffy (polar) parametrisation of
    every reference sub-triangle that has the target node as a corner
    (:func:`_duffy_patch_quadrature`): ``r ~ u`` and ``dS ~ u du dv``.

    Parameters
    ----------
    apex_node : int
        Element node index in ``0..5`` where the collocation point sits.
    order : int
        Gauss order per direction.

    Returns
    -------
    nodes : (k, 2) array of ``(L1, L2)`` barycentric coordinates; for a vertex
        node this is a single patch, for an edge midpoint the concatenation of
        the two patches that tile the element.
    weights : (k,) weights already containing the Duffy Jacobian; integrate with
        ``sum(weights * jacobian(L1, L2) * f)``.
    """
    if apex_node not in range(6):
        raise ValueError("apex_node must be in 0..5 (an element node)")
    blocks = [
        _duffy_patch_quadrature(apex_node, a, b, order=order)
        for (a, b) in _SINGULAR_PATCHES[apex_node]
    ]
    nodes = np.vstack([b[0] for b in blocks])
    weights = np.concatenate([b[1] for b in blocks])
    return nodes, weights


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

    def integrate_single_layer_self(
        self, apex_node: int, order: int = 16
    ) -> FloatArray:
        """Singular self-influence integral ``int_T N_k G dS`` at an element node.

        The collocation point sits *on* the element -- at a vertex (nodes 0-2)
        or on an edge midpoint (nodes 3-5) -- so the kernel ``1/(4 pi r)`` is
        weakly singular.  Each reference sub-triangle carrying the singularity
        at a corner is integrated with the Duffy rule of
        :func:`_duffy_patch_quadrature`, which removes the singularity
        analytically.  Before 2026-11-09 only vertices were treated (midpoints
        were silently routed through the non-singular rule, and the apex
        permutation for nodes 1 and 2 pointed at the wrong vertex); both
        defects are pinned by ``tests/test_panel_quadratic_operator.py``.

        Parameters
        ----------
        apex_node : int
            Element node index in ``0..5`` where the collocation point sits.
        order : int
            Gauss quadrature order per direction.

        Returns
        -------
        (6,) array of ``int_T N_k / (4*pi*r) dS`` for each shape function.
        """
        if apex_node not in range(6):
            raise ValueError("apex_node must be in 0..5")

        target = self.nodes[apex_node]
        nodes_ref, weights_ref = duffy_self_quadrature(apex_node, order=order)

        L1 = nodes_ref[:, 0]
        L2 = nodes_ref[:, 1]
        N_q = _shape_matrix(L1, L2)                        # (Q, 6)
        x_q = N_q @ self.nodes                             # (Q, 3)
        r = np.linalg.norm(x_q - target, axis=1)           # (Q,)
        grad = _shape_gradients_matrix(L1, L2)             # (Q, 6, 2)
        dx_dL1 = np.einsum("qk,kx->qx", grad[:, :, 0], self.nodes)
        dx_dL2 = np.einsum("qk,kx->qx", grad[:, :, 1], self.nodes)
        det = np.linalg.norm(np.cross(dx_dL1, dx_dL2), axis=1)

        safe = r > 1e-15          # the singular corner itself has measure zero
        scale = np.zeros_like(r)
        scale[safe] = weights_ref[safe] * det[safe] / (FOUR_PI * r[safe])
        return scale @ N_q

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
        nodes_ref, weights = gauss_triangle_rule(order)
        result = np.zeros(6)
        for q in range(len(weights)):
            L1, L2 = nodes_ref[q]
            x_q = self.map_to_physical(L1, L2)
            r = float(np.linalg.norm(x_q - target))
            if r < 1e-15:
                raise ValueError("target lies on the element; use self-integral")
            N_q = quadratic_shape_functions(L1, L2)
            cross, det = self.jacobian(L1, L2)
            result += weights[q] * N_q * det / (FOUR_PI * r)
        return result


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
    """
    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces, dtype=np.int64)
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
        object.__setattr__(self, "vertices", vertices.copy())
        object.__setattr__(self, "faces", faces.copy())

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


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------
def assemble_quadratic_matrices(
    mesh: QuadraticSurfaceMesh,
    quadrature_order: int = 8,
    self_order: int = 16,
) -> tuple[FloatArray, FloatArray]:
    """Assemble the single-layer and double-layer matrices for quadratic BEM.

    The collocation system has ``mesh.node_count`` unknowns (one per node) and
    the single-layer matrix is

        S_ik = int_S N_k(y) G(x_i, y) dS_y,    G = 1 / (4 pi |x - y|),

    where ``N_k`` is the global 6-node basis function of node ``k`` and ``x_i``
    is node ``i``.  ``N_k`` is supported only on the elements incident to node
    ``k``, so the row ``i`` receives a contribution from **every element** -- the
    local basis functions of an element that does not touch ``x_i`` still enter
    row ``i`` through their own nodes.  Iterating ``(element, own nodes)`` only
    therefore assembles a *patch* operator, not the boundary operator; that was
    the state of this function until 2026-11-09 (the residual of the analytic
    eigenfunction test in ``docs/quadratic_panel_verification.md`` was 2.84
    instead of decreasing with ``h``).  The full row is now assembled:

    * targets that are nodes of the element (distance zero) use the Duffy
      singular rule :meth:`QuadraticTriangle.integrate_single_layer_self`;
    * every other node is integrated with the vectorised Gauss rule over the
      curved element.

    .. warning::
       ``D`` is **not** assembled: the double-layer matrix is returned as all
       zeros.  It was documented as a "flat-panel approximation" from the start
       while the loop never filled it, and nothing consumed it (the shipped
       solver :func:`solve_laplace_dirichlet_quadratic` uses the indirect
       single-layer formulation only).  The zero matrix is kept for signature
       compatibility and pinned by ``tests/test_panel_quadratic_operator.py``;
       a real double-layer assembly (free term ``-1/2`` plus a curved-panel
      solid-angle self term) is an open item in
       ``docs/quadratic_panel_verification.md``.

    Parameters
    ----------
    mesh : QuadraticSurfaceMesh
    quadrature_order : int for the regular (non-singular) element integrals
    self_order : int for the singular (target-on-element) integrals

    Returns
    -------
    S : (node_count, node_count) single-layer matrix
    D : (node_count, node_count) double-layer matrix (currently zeros, see above)
    """
    n_nodes = mesh.node_count
    n_elems = mesh.element_count
    S = np.zeros((n_nodes, n_nodes))
    D = np.zeros((n_nodes, n_nodes))

    nodes_ref, weights_ref = gauss_triangle_rule(quadrature_order)
    L1 = nodes_ref[:, 0]
    L2 = nodes_ref[:, 1]
    N_q = _shape_matrix(L1, L2)                       # (Q, 6)
    grad_q = _shape_gradients_matrix(L1, L2)          # (Q, 6, 2)
    all_nodes = np.arange(n_nodes)

    for e in range(n_elems):
        elem = mesh.element(e)
        node_ids = mesh.faces[e]

        # --- singular rows: collocation node is a node of this element ------
        for local_i in range(6):
            s_self = elem.integrate_single_layer_self(
                apex_node=local_i, order=self_order
            )
            S[node_ids[local_i], node_ids] += s_self

        # --- regular rows: every other node of the mesh --------------------
        on_element = np.zeros(n_nodes, dtype=bool)
        on_element[node_ids] = True
        others = all_nodes[~on_element]
        if others.size == 0:
            continue

        x_q = N_q @ elem.nodes                          # (Q, 3)
        dx_dL1 = np.einsum("qk,kx->qx", grad_q[:, :, 0], elem.nodes)
        dx_dL2 = np.einsum("qk,kx->qx", grad_q[:, :, 1], elem.nodes)
        det_q = np.linalg.norm(np.cross(dx_dL1, dx_dL2), axis=1)   # (Q,)
        w_q = weights_ref * det_q / FOUR_PI                         # (Q,)

        delta = mesh.vertices[others][:, None, :] - x_q[None, :, :]  # (T, Q, 3)
        r = np.linalg.norm(delta, axis=2)                            # (T, Q)
        if np.any(r <= 0.0):
            raise ValueError("non-singular target coincides with a quadrature point")
        S[np.ix_(others, node_ids)] += (w_q[None, :] / r) @ N_q

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
