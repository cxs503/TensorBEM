"""High- and zero-frequency rigid-body added mass of a surface-piercing body.

Why this module exists
----------------------
A frequency sweep of the radiation problem gives ``A(omega)`` and ``B(omega)``
only on a *finite* band, but two pieces of information that the band cannot
supply are needed again and again:

* ``A_inf = A(omega -> infinity)`` -- the asymptotic added mass.  The
  Kramers-Kronig relation that predicts damping from added mass carries
  ``A - A_inf``, so ``A_inf`` has to come from somewhere else, and extrapolating
  it from the noisy tail of a panel sweep is ill-conditioned exactly where the
  added mass is nearly flat in frequency (documented in
  ``docs/dev_report_m20594_cummins_g3_conditioning.md``: the two-term tail fit
  returns physically impossible values, e.g. a negative sway added mass).
* ``A(0) = A(omega -> 0)`` -- the zero-frequency limit, which a sweep that
  starts at ``omega = 0.2`` rad/s approaches from the wrong side.

Both limits are *boundary-value problems in the lower half space*, not
frequency-domain radiation problems, and both follow from the linearised free
surface condition

    d(phi)/dz = (omega**2 / g) phi   at   z = 0 ,

because the two ends of the frequency range degenerate:

============================  ==========================================
omega -> 0                    d(phi)/dz = 0   (rigid wall)
omega -> infinity             phi = 0         (pressure release)
============================  ==========================================

Method: antisymmetric/symmetric images, no truncated free-surface patch
-----------------------------------------------------------------------
A floating hull is only panelised below ``z = 0`` (`WettedHullMesh`), so the
mirror plane is not a boundary of the discretisation at all: reflecting the
wetted surface in ``z = 0`` closes it into the *doubled body* (the region
between the hull and its image, enclosing ``2 * displaced volume``).  Solving
the **infinite-fluid** Neumann problem for that closed body with boundary data
extended by the reflection

    q'(M x) = s * q(x),      s = -1 for the Dirichlet (omega -> infinity) limit,
                             s = +1 for the rigid-wall (omega -> 0) limit,

produces a potential that is odd (``s = -1``) or even (``s = +1``) in ``z``,
so the free-surface condition holds *exactly on the whole plane* by
construction.  No truncation radius and no free-surface panels are needed; the
only discretisation is the hull.  Restricting the Green integral to the
physical (lower) half gives

    A_ij = -rho * integral_{hull} phi_j * n_i dS .

Why the extension rule is ``q'(Mx) = s q(x)``
---------------------------------------------
The reflection ``M = diag(1, 1, -1)`` is orthogonal with ``det M = -1``, so the
image of a rigid motion ``(U, omega)`` is ``(M U, -M omega)``.  With
``n' = M n`` and ``(Ma) x (Mb) = -M (a x b)`` the image velocity satisfies
``v'(Mx) = M v(x)``, hence ``q' = v' . n' = v . n = q``: the mirrored panels
carry the *same* normal-velocity numbers, with the potential parity ``s``
fixing the sign.  The rotation centre may lie off ``z = 0``: the rule
``q'(Mx) = s q(x)`` holds for any centre because it is derived from the
potential parity, not from a rigid-body identity of the doubled body.

Verification ladder (see ``tests/test_added_mass_limit.py``)
-----------------------------------------------------------
* the doubled mesh is closed and encloses ``2 x`` the hull volume;
* the computed potential satisfies the imposed free-surface condition on
  ``z = 0`` (``max |phi|`` for ``s = -1``, ``max |d(phi)/dz|`` for ``s = +1``);
* the ``omega -> 0`` limit agrees with the *independent*, already verified
  frequency-domain Green-function solver (:func:`solve_radiation_3d`) at the
  lowest swept frequency, which is the end-to-end check of the conventions
  (normals, sign of the jump, half-body integral);
* ``A`` is symmetric, positive diagonal, and refines with the mesh.

Nothing here shares code with the radiation solver except the analytically
integrated single/double layer operators of :mod:`tensorbem.solver3d`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh, WettedHullMesh
from .solver3d import assemble_surface_matrices

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "ADDED_MASS_LIMITS",
    "AddedMassLimitResult",
    "doubled_body_mesh",
    "generalized_normals_6dof",
    "added_mass_limit",
]

ADDED_MASS_LIMITS = ("infinite_frequency", "zero_frequency")
LimitName = Literal["infinite_frequency", "zero_frequency"]

_PARITY = {"infinite_frequency": -1.0, "zero_frequency": 1.0}
_DEFAULT_DENSITY = 1025.0


@dataclass(frozen=True)
class AddedMassLimitResult:
    """Outcome of one high- or zero-frequency added-mass solve.

    Attributes
    ----------
    limit:
        ``"infinite_frequency"`` (Dirichlet free surface) or
        ``"zero_frequency"`` (rigid-wall free surface).
    parity:
        ``-1`` (odd potential) or ``+1`` (even potential) used for the image.
    added_mass_matrix:
        ``(6, 6)`` added mass ``[kg, kg m, kg m**2]`` about ``center``.
    mode_potentials:
        Potential at the *physical* (lower) panel centroids, ``(M, 6)``.
    """

    limit: str
    parity: float
    added_mass_matrix: FloatArray
    mode_potentials: FloatArray
    mode_normals: FloatArray
    center: tuple[float, float, float]
    density: float
    modes: tuple
    lower_panel_count: int
    doubled_panel_count: int
    doubled_volume: float
    hull_volume: float
    condition_number: float
    max_relative_residual_l2: float
    symmetry_error: float
    double_layer_row_sum_error: float
    quadrature_order: int
    diagnostics: dict = field(default_factory=dict)

    def diagonal(self) -> FloatArray:
        """Diagonal of the added-mass matrix (the six diagonal coefficients)."""
        return np.diag(np.asarray(self.added_mass_matrix, dtype=float))


def doubled_body_mesh(mesh: WettedHullMesh) -> tuple[TriangleMesh, IntArray]:
    """Close a waterline-truncated hull by reflecting it in ``z = 0``.

    Returns ``(doubled_mesh, lower_faces)`` where ``lower_faces`` are the panel
    indices of the *physical* hull inside ``doubled_mesh`` (the first block).
    Vertices on the waterline are welded (``z == 0`` maps onto itself), so the
    result is a closed, consistently oriented surface with outward normals and
    a positive signed volume of twice the displaced volume.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh.vertices must have shape (V, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh.vertices must all be finite")
    if np.any(vertices[:, 2] > 1.0e-12):
        raise ValueError("a wetted hull must lie in z <= 0")

    # Weld the mirror image onto the original on the waterplane.
    index = {(round(float(x), 12) + 0.0, round(float(y), 12) + 0.0,
              round(float(z), 12) + 0.0): i
             for i, (x, y, z) in enumerate(vertices)}
    merged: list[tuple[float, float, float]] = [tuple(map(float, v)) for v in vertices]
    mirror_map = np.empty(vertices.shape[0], dtype=np.int64)
    for i, (x, y, z) in enumerate(vertices):
        key = (round(float(x), 12) + 0.0, round(float(y), 12) + 0.0,
               round(float(-z), 12) + 0.0)
        if key in index:
            mirror_map[i] = index[key]
        else:
            index[key] = len(merged)
            merged.append((key[0], key[1], key[2]))
            mirror_map[i] = len(merged) - 1

    mirrored_faces = mirror_map[faces][:, ::-1]  # reflection flips orientation
    all_faces = np.vstack((faces, mirrored_faces)).astype(np.int64)
    doubled = TriangleMesh(np.asarray(merged, dtype=float), all_faces)
    lower = np.arange(faces.shape[0], dtype=np.int64)
    return doubled, lower


def generalized_normals_6dof(
    mesh: TriangleMesh | WettedHullMesh, center: Sequence[float] = (0.0, 0.0, 0.0)
) -> FloatArray:
    """Generalized normals ``(n_1..n_3, (x - center) x n)`` of shape ``(M, 6)``."""
    origin = np.asarray(center, dtype=float).reshape(-1)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("center must be a finite 3-vector")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    double_area = np.linalg.norm(cross, axis=1)
    if np.any(double_area <= 0.0):
        raise ValueError("degenerate panel in mesh")
    normals = cross / double_area[:, None]
    centroids = (v0 + v1 + v2) / 3.0
    rel = centroids - origin[None, :]
    return np.hstack((normals, np.cross(rel, normals)))


def added_mass_limit(
    mesh: WettedHullMesh,
    limit: LimitName = "infinite_frequency",
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    density: float = _DEFAULT_DENSITY,
    quadrature_order: int = 8,
    modes: Sequence[int] | None = None,
) -> AddedMassLimitResult:
    """Added-mass matrix in one of the two spectral limits.

    ``limit="infinite_frequency"`` imposes ``phi = 0`` on the calm waterplane
    (odd image), ``limit="zero_frequency"`` imposes ``d(phi)/dz = 0`` (even
    image).  Both are solved as the exterior Neumann problem of the doubled
    body for the rigid-body modes ``0..5`` (``surge, sway, heave, roll, pitch,
    yaw``) about ``center``.
    """
    if limit not in _PARITY:
        raise ValueError(f"limit must be one of {ADDED_MASS_LIMITS}")
    if not isinstance(mesh, WettedHullMesh):
        raise TypeError("mesh must be a WettedHullMesh")
    if isinstance(density, (bool, np.bool_)) or not math.isfinite(float(density)) \
            or float(density) <= 0.0:
        raise ValueError("density must be a positive finite number")
    rho = float(density)
    parity = float(_PARITY[limit])

    doubled, lower = doubled_body_mesh(mesh)
    if modes is None:
        mode_arr = np.arange(6, dtype=np.int64)
    else:
        mode_arr = np.asarray(list(modes), dtype=np.int64)
        if mode_arr.ndim != 1 or mode_arr.size == 0:
            raise ValueError("modes must be a non-empty 1-D sequence")
        if np.any((mode_arr < 0) | (mode_arr > 5)):
            raise ValueError("each mode index must be in range 0..5")

    lower_normals = generalized_normals_6dof(mesh, center=center)
    # Image panels carry the same numbers, scaled by the potential parity.
    full_normals = np.vstack((lower_normals, parity * lower_normals))
    q_full = full_normals[:, mode_arr]

    single, double = assemble_surface_matrices(doubled, quadrature_order=quadrature_order)
    system = 0.5 * np.eye(doubled.panel_count) - double
    rhs = single @ (-q_full)
    condition = float(np.linalg.cond(system))
    potentials = np.linalg.solve(system, rhs)
    residuals = np.linalg.norm(system @ potentials - rhs, axis=0)
    rhs_norms = np.linalg.norm(rhs, axis=0)
    rel = np.where(rhs_norms > 0.0, residuals / rhs_norms, residuals)

    lower_areas = mesh_panel_areas(doubled)[lower]
    weighted = lower_areas[:, None] * lower_normals[:, mode_arr]
    added_mass = -rho * (weighted.T @ potentials[lower])
    added_mass = 0.5 * (added_mass + added_mass.T)

    doubled_volume = float(abs(doubled.volume))
    hull_volume = float(abs(mesh.volume))
    return AddedMassLimitResult(
        limit=limit,
        parity=parity,
        added_mass_matrix=added_mass,
        mode_potentials=potentials[lower],
        mode_normals=lower_normals[:, mode_arr],
        modes=tuple(int(m) for m in mode_arr),
        center=(float(center[0]), float(center[1]), float(center[2])),
        density=rho,
        lower_panel_count=int(lower.size),
        doubled_panel_count=int(doubled.panel_count),
        doubled_volume=doubled_volume,
        hull_volume=hull_volume,
        condition_number=condition,
        max_relative_residual_l2=float(np.max(rel)),
        symmetry_error=float(np.max(np.abs(added_mass - added_mass.T))),
        double_layer_row_sum_error=float(
            np.max(np.abs(np.sum(double, axis=1) + 0.5))
        ),
        quadrature_order=int(quadrature_order),
    )


def mesh_panel_areas(mesh: TriangleMesh) -> FloatArray:
    """Panel areas of a triangulated surface, shape ``(M,)``."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
