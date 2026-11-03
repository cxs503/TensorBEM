"""Symmetry-plane (half-model) reduction for the 3-D Laplace boundary-element solver.

Why this module exists
----------------------
A ship hull, a spheroid, a barge and a hemisphere are all symmetric about a
longitudinal plane, so half of the wetted surface carries redundant unknowns:
the solution on the hidden half follows from a reflection.  TensorBEM had no
way to exploit that, so every symmetric model paid twice the panel count, four
times the assembly work and eight times the dense-solve work of the half
model.

The reduction is exact, not approximate, for the *whole* exterior Neumann
problem, provided the boundary data respect the reflection.  Let ``R`` be the
reflection in the plane ``x = 0`` and ``p`` the parity of the mode, defined by
``phi(R x) = p phi(x)`` with ``p = +1`` (even, e.g. sway/heave/roll) or
``p = -1`` (odd, e.g. surge/pitch/yaw).  Then the hidden half carries the same
density multiplied by ``p``, and the collocation rows on the visible half
become

    (1/2 I - H_half - p H_image) u_half = (S_half + p S_image) q_half ,

where ``H_image`` and ``S_image`` are the usual layer matrices of the visible
panels evaluated at the *mirrored* panels.  That is the entire reduction: pass
reflected geometry to the same kernels, fold with the parity, and solve an
``n x n`` system instead of ``2n x 2n``.  Because no kernel is re-derived, the
half model reproduces the full model on the same triangulation to roundoff --
the verification in ``tests/test_symmetry_half_model.py`` and
``benchmarks/half_model_symmetry.py`` asserts exactly that, so a factor of four
in dense-solve work costs nothing in accuracy.

Parity of the six rigid-body modes (about a centre on the plane)
--------------------------------------------------------------
For translation the mode normal is the unit vector ``e_i``, for rotation it is
``(x - c) x n``; requiring ``v(R x) = p R v(x)`` gives

===========  =======  =======
mode         parity   reason
===========  =======  =======
surge        ``-1``   ``v = e_x``, ``R v = -v``
sway         ``+1``   ``v = e_y``, ``R v = +v``
heave        ``+1``   ``v = e_z``, ``R v = +v``
roll         ``+1``   ``v = (0, -z, y)`` is reflection-invariant
pitch        ``-1``   ``v = (z, 0, -x)`` flips sign under ``R``
yaw          ``-1``   ``v = (-y, x, 0)`` flips sign under ``R``
===========  =======  =======

Consequently the added-mass entries coupling modes of opposite parity vanish
identically, and the implementation returns them as exact zeros rather than as
roundoff-level noise.

Scope and limitations
---------------------
* Unbounded-fluid (Laplace) exterior Neumann problems only.  The free-surface
  solvers have no half-model path: their Green function already breaks the
  fore-aft symmetry, and their image construction would need the free-surface
  condition checked on the mirror too.
* One symmetry plane per model.  Two planes (quarter model) would need the
  parity to be a group representation and are not implemented.
* The visible half must be a topological disk whose only open boundary lies in
  the symmetry plane; :class:`HalfSurface` validates that.
* ``center`` for the rotational modes must lie in the symmetry plane.
* Near the plane the image panel of a panel touching the plane is a *distinct*
  panel at ``O(h)`` distance, so the folded entries are quadratured by the same
  Duffy rule as any other near neighbour and inherit its accuracy; this is the
  same arithmetic the full model performs, which is why the involution is
  exact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh
from .kernels3d import (
    duffy_quadrature_batch,
    integrate_single_layer_triangle_self,
    solid_angle_triangles_batch,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_FOUR_PI = 4.0 * np.pi

#: Parity ``p`` of the six rigid-body mode potentials under ``x -> -x``.
MODE_PARITIES: tuple[float, float, float, float, float, float] = (
    -1.0,  # surge
    1.0,  # sway
    1.0,  # heave
    1.0,  # roll
    -1.0,  # pitch
    -1.0,  # yaw
)


def _reflection(axis: int) -> FloatArray:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be 0, 1 or 2")
    if int(axis) not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    matrix = np.eye(3)
    matrix[int(axis), int(axis)] = -1.0
    return matrix


def _validate_order(order: int) -> int:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError("quadrature_order must be an integer")
    value = int(order)
    if not 1 <= value <= 32:
        raise ValueError("quadrature_order must lie in [1, 32]")
    return value


@dataclass(frozen=True)
class HalfSurface:
    """An open triangular surface lying on one side of a symmetry plane.

    Parameters
    ----------
    vertices:
        ``(V, 3)`` finite coordinates.
    faces:
        ``(F, 3)`` triangles ordered consistently with the outward normal of
        the body.  The surface must be a topological disk whose open boundary
        lies entirely in the plane ``vertices[:, axis] == 0``.  It is *not*
        closed, so it cannot be a :class:`~tensorbem.geometry3d.TriangleMesh`;
        :meth:`closed_mesh` mirrors it into one.
    axis:
        Symmetry-plane normal direction, ``0`` (``x = 0``), ``1`` (``y = 0``)
        or ``2`` (``z = 0``).
    side:
        ``+1`` when the surface lies in ``axis coordinate >= 0``, ``-1``
        otherwise.  Mirrored surfaces report the opposite side.
    """

    vertices: FloatArray
    faces: IntArray
    axis: int = 0
    side: int = 1

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces)
        _reflection(self.axis)
        if self.side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if vertices.shape[0] < 3:
            raise ValueError("a surface requires at least three vertices")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices must all be finite")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if faces.shape[0] < 1:
            raise ValueError("faces must contain at least one triangle")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("faces must be an integer array")
        faces = np.asarray(faces, dtype=np.int64)
        if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
            raise ValueError("face indices must be in range [0, vertex_count)")
        if np.any(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        ):
            raise ValueError("each triangle must use three distinct vertices")

        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        double_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        scale = max(1.0, float(np.max(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1))))
        if np.any(double_areas <= 1.0e-14 * scale * scale):
            raise ValueError("degenerate triangles are not allowed")

        axis = int(self.axis)
        tolerance = 1.0e-9 * scale
        centroids = (v0 + v1 + v2) / 3.0
        if np.any(self.side * centroids[:, axis] < -tolerance):
            raise ValueError(
                "all panel centroids must lie on the declared side of the symmetry plane"
            )

        directed = np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        )
        vertex_count = int(vertices.shape[0])
        keys = directed[:, 0] * vertex_count + directed[:, 1]
        if np.unique(keys).size != keys.size:
            raise ValueError(
                "a directed edge appears more than once: the half surface must be a "
                "single consistently oriented disk"
            )
        reverse = directed[:, 1] * vertex_count + directed[:, 0]
        boundary = ~np.isin(reverse, keys)
        if not np.any(boundary):
            raise ValueError("the surface is closed: no symmetry-plane boundary found")
        boundary_vertices = np.unique(directed[boundary])
        if np.any(np.abs(vertices[boundary_vertices, axis]) > tolerance):
            raise ValueError("the open boundary of the half surface must lie in the symmetry plane")

        object.__setattr__(self, "vertices", vertices.copy())
        object.__setattr__(self, "faces", faces.copy())
        object.__setattr__(self, "axis", axis)

    # ----- basic arrays -------------------------------------------------
    @property
    def v0(self) -> FloatArray:
        return self.vertices[self.faces[:, 0]]

    @property
    def v1(self) -> FloatArray:
        return self.vertices[self.faces[:, 1]]

    @property
    def v2(self) -> FloatArray:
        return self.vertices[self.faces[:, 2]]

    @property
    def panel_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def double_areas(self) -> FloatArray:
        return np.linalg.norm(np.cross(self.v1 - self.v0, self.v2 - self.v0), axis=1)

    @property
    def areas(self) -> FloatArray:
        return 0.5 * self.double_areas

    @property
    def centroids(self) -> FloatArray:
        return (self.v0 + self.v1 + self.v2) / 3.0

    @property
    def normals(self) -> FloatArray:
        """Unit normals pointing out of the body into the fluid."""
        return np.cross(self.v1 - self.v0, self.v2 - self.v0) / self.double_areas[:, None]

    # ----- transformations ---------------------------------------------
    def mirrored(self) -> "HalfSurface":
        """The reflected surface, with winding flipped to keep outward normals."""
        reflection = _reflection(self.axis)
        vertices = self.vertices @ reflection.T
        faces = self.faces[:, [0, 2, 1]]
        return HalfSurface(vertices, faces, axis=self.axis, side=-self.side)

    def closed_mesh(self, weld_decimals: int = 9) -> TriangleMesh:
        """Mirror and weld the half surface into a closed :class:`TriangleMesh`."""
        mirrored = self.mirrored()
        axis = int(self.axis)
        scale = max(1.0, float(np.max(np.linalg.norm(self.vertices, axis=1))))
        tolerance = 1.0e-9 * scale
        on_plane = np.abs(self.vertices[:, axis]) <= tolerance
        keys = {
            tuple(np.round(self.vertices[index], weld_decimals)): index
            for index in np.nonzero(on_plane)[0]
        }
        offset = self.vertex_count
        remap = np.arange(mirrored.vertex_count, dtype=np.int64) + offset
        for index in range(mirrored.vertex_count):
            if abs(mirrored.vertices[index, axis]) > tolerance:
                continue
            key = tuple(np.round(mirrored.vertices[index], weld_decimals))
            if key in keys:
                remap[index] = keys[key]
        vertices = np.vstack((self.vertices, mirrored.vertices))
        faces = np.vstack((self.faces, remap[mirrored.faces]))
        used, compacted = np.unique(faces, return_inverse=True)
        return TriangleMesh(vertices[used], compacted.reshape(faces.shape))


def half_ellipsoid_surface(
    semi_axes: FloatArray | tuple[float, float, float],
    n_phi: int = 8,
    n_theta: int = 16,
) -> HalfSurface:
    """Half (``x >= 0``) of a centred ellipsoid, as a structured ring mesh.

    ``phi`` runs from the north pole to the south pole and ``theta`` from one
    point of the symmetry-plane cut to the other, so the open boundary is the
    ellipse ``x = 0``.  Both pole rings collapse to a single vertex, which is
    capped with triangles instead of degenerate quads.

    The full ellipsoid is ``(x/a)^2 + (y/b)^2 + (z/c)^2 = 1``; its mirror
    closure is exactly the full surface, so ``surface.closed_mesh()`` is a
    valid closed mesh for the ordinary
    :func:`tensorbem.solver3d.compute_added_mass_matrix_3d`.
    """
    axes = np.asarray(semi_axes, dtype=float).reshape(-1)
    if axes.shape != (3,):
        raise ValueError("semi_axes must be a 3-vector")
    if not np.isfinite(axes).all() or np.any(axes <= 0.0):
        raise ValueError("semi_axes must be positive and finite")
    if isinstance(n_phi, bool) or not isinstance(n_phi, (int, np.integer)):
        raise ValueError("n_phi must be an integer")
    if isinstance(n_theta, bool) or not isinstance(n_theta, (int, np.integer)):
        raise ValueError("n_theta must be an integer")
    rings, sectors = int(n_phi), int(n_theta)
    if rings < 2 or sectors < 2:
        raise ValueError("n_phi and n_theta must both be at least 2")

    a, b, c = float(axes[0]), float(axes[1]), float(axes[2])
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, c), (0.0, 0.0, -c)]

    def ring_index(i: int, j: int) -> int:
        return 2 + (i - 1) * (sectors + 1) + j

    for i in range(1, rings):
        phi = math.pi * i / rings
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        for j in range(sectors + 1):
            theta = math.pi * j / sectors
            vertices.append(
                (
                    a * sin_phi * math.sin(theta),
                    b * sin_phi * math.cos(theta),
                    c * cos_phi,
                )
            )

    faces: list[tuple[int, int, int]] = []
    for j in range(sectors):
        faces.append((0, ring_index(1, j), ring_index(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(sectors):
            faces.append((ring_index(i, j), ring_index(i + 1, j), ring_index(i + 1, j + 1)))
            faces.append((ring_index(i, j), ring_index(i + 1, j + 1), ring_index(i, j + 1)))
    for j in range(sectors):
        faces.append((1, ring_index(rings - 1, j + 1), ring_index(rings - 1, j)))

    vertex_array = np.asarray(vertices, dtype=float)
    face_array = np.asarray(faces, dtype=np.int64)
    v0 = vertex_array[face_array[:, 0]]
    v1 = vertex_array[face_array[:, 1]]
    v2 = vertex_array[face_array[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0
    outward = np.column_stack(
        (
            centroids[:, 0] / (a * a),
            centroids[:, 1] / (b * b),
            centroids[:, 2] / (c * c),
        )
    )
    alignment = np.einsum("ij,ij->i", normals, outward)
    if np.all(alignment < 0.0):
        face_array = face_array[:, [0, 2, 1]]
        v0 = vertex_array[face_array[:, 0]]
        v1 = vertex_array[face_array[:, 1]]
        v2 = vertex_array[face_array[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        alignment = -alignment
    if not np.all(alignment > 0.0):
        raise RuntimeError("internal error: half-ellipsoid triangulation is not consistently oriented")
    return HalfSurface(vertex_array, face_array, axis=0, side=1)


def _layer_matrices(
    targets: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    order: int,
) -> tuple[FloatArray, FloatArray]:
    """Single- and double-layer matrices for arbitrary targets and triangles."""
    points, weights = duffy_quadrature_batch(v0, v1, v2, order=order)
    single = np.empty((targets.shape[0], v0.shape[0]), dtype=float)
    double = np.empty_like(single)
    for index, target in enumerate(targets):
        delta = target[None, None, :] - points
        distances = np.linalg.norm(delta, axis=2)
        single[index] = np.sum(weights / (_FOUR_PI * distances), axis=1)
        double[index] = -solid_angle_triangles_batch(target, v0, v1, v2) / _FOUR_PI
    return single, double


def assemble_half_surface_matrices(
    surface: HalfSurface,
    parity: float,
    quadrature_order: int = 8,
) -> tuple[FloatArray, FloatArray]:
    """Fold the image half into the single- and double-layer matrices.

    ``parity`` is ``+1`` for a reflection-even field and ``-1`` for an odd one.
    The returned matrices act on the *visible* half's unknowns only.
    """
    if not isinstance(surface, HalfSurface):
        raise TypeError("surface must be a HalfSurface")
    if parity not in (1.0, -1.0):
        raise ValueError("parity must be +1 or -1")
    order = _validate_order(quadrature_order)
    centroids = surface.centroids

    single, double = _layer_matrices(centroids, surface.v0, surface.v1, surface.v2, order)
    self_integrals = np.array(
        [
            integrate_single_layer_triangle_self(surface.vertices[surface.faces[index]])
            for index in range(surface.panel_count)
        ]
    )
    single[np.diag_indices_from(single)] = self_integrals
    double[np.diag_indices_from(double)] = 0.0

    image = surface.mirrored()
    image_single, image_double = _layer_matrices(
        centroids, image.v0, image.v1, image.v2, order
    )
    return single + parity * image_single, double + parity * image_double


@dataclass(frozen=True)
class HalfModelAddedMass3DResult:
    """Added mass of a symmetric 3-D body from the half model.

    ``added_mass_matrix`` is the full ``(6, 6)`` tensor of the *whole* body;
    entries coupling modes of opposite parity are exact zeros by symmetry.
    """

    added_mass_matrix: FloatArray
    mode_potentials: FloatArray
    mode_normals: FloatArray
    parities: FloatArray
    half_panel_count: int
    full_panel_count: int
    condition_number_even: float
    condition_number_odd: float
    max_relative_residual_l2: float
    off_parity_max: float
    symmetry_error: float
    center: tuple[float, float, float]
    quadrature_order: int


def compute_added_mass_matrix_3d_half(
    surface: HalfSurface,
    density: float = 1.0,
    center: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    quadrature_order: int = 8,
) -> HalfModelAddedMass3DResult:
    """Half-model ``(6, 6)`` added-mass matrix of a symmetric 3-D body.

    Solves the exterior Neumann problem on the visible half only, for the six
    rigid-body modes about ``center`` (which must lie in the symmetry plane),
    and folds the image half back into the boundary-energy integral.
    """
    if not isinstance(surface, HalfSurface):
        raise TypeError("surface must be a HalfSurface")
    if isinstance(density, (bool, np.bool_)) or not math.isfinite(float(density)) or float(density) <= 0.0:
        raise ValueError("density must be a positive finite number")
    rho = float(density)
    origin = np.asarray(center, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("center must be a finite 3-vector")
    axis = int(surface.axis)
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.linalg.norm(surface.vertices, axis=1))))
    if abs(origin[axis]) > tolerance:
        raise ValueError("center must lie in the symmetry plane")
    order = _validate_order(quadrature_order)

    rel = surface.centroids - origin[None, :]
    normals = surface.normals
    mode_normals = np.hstack((normals, np.cross(rel, normals)))
    parities = np.asarray(MODE_PARITIES, dtype=float)

    panel_count = surface.panel_count
    potentials = np.zeros((panel_count, 6), dtype=float)
    condition: dict[float, float] = {}
    residuals = np.zeros(6, dtype=float)
    for parity in (1.0, -1.0):
        single, double = assemble_half_surface_matrices(
            surface, parity, quadrature_order=order
        )
        system = 0.5 * np.eye(panel_count) - double
        condition[parity] = float(np.linalg.cond(system))
        modes = np.nonzero(parities == parity)[0]
        rhs = single @ (-mode_normals[:, modes])
        solution = np.linalg.solve(system, rhs)
        potentials[:, modes] = solution
        difference = system @ solution - rhs
        rhs_norms = np.linalg.norm(rhs, axis=0)
        residuals[modes] = np.where(
            rhs_norms > 0.0, np.linalg.norm(difference, axis=0) / rhs_norms, 0.0
        )

    weighted = (mode_normals * surface.areas[:, None])
    half_integral = weighted.T @ potentials
    fold = 1.0 + np.outer(parities, parities)
    added_mass = -rho * fold * half_integral
    off_parity = np.outer(parities, parities) < 0.0
    return HalfModelAddedMass3DResult(
        added_mass_matrix=added_mass,
        mode_potentials=potentials,
        mode_normals=mode_normals,
        parities=parities,
        half_panel_count=panel_count,
        full_panel_count=2 * panel_count,
        condition_number_even=condition[1.0],
        condition_number_odd=condition[-1.0],
        max_relative_residual_l2=float(np.max(residuals)),
        off_parity_max=float(np.max(np.abs(added_mass[off_parity]))),
        symmetry_error=float(np.max(np.abs(added_mass - added_mass.T))),
        center=(float(origin[0]), float(origin[1]), float(origin[2])),
        quadrature_order=order,
    )


__all__ = [
    "HalfModelAddedMass3DResult",
    "HalfSurface",
    "MODE_PARITIES",
    "assemble_half_surface_matrices",
    "compute_added_mass_matrix_3d_half",
    "half_ellipsoid_surface",
]