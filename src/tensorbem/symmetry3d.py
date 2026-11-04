"""Plane-of-symmetry reduction for the 3-D Kelvin free-surface BEM solver.

Why it is exact
---------------
The deep-water Kelvin Green function of :mod:`tensorbem.free_surface3d` is built
from the horizontal distance ``R = hypot(dx, dy)`` and the vertical coordinates
``(z, z0)`` only, so it is *exactly* invariant under the reflection
``M: (x, y, z) -> (x, -y, z)``::

    G(M P, M Q) = G(P, Q)

and, because the kernel gradient is contracted with a mirrored normal, the
normal derivative obeys the same invariance.  Consequently, for a body symmetric
about ``y = 0`` -- every ship hull -- a potential with definite parity
``phi(M x) = s phi(x)``, ``s = +-1``, satisfies the boundary integral equation on
the retained half alone::

    [1/2 I - H_half - s H_mirror] phi_half = [S_half + s S_mirror] q_half

with ``H_mirror``/``S_mirror`` the operators of the mirrored half surface (its
own outward normals) evaluated at the retained half collocation points, and
``q_half(M x) = s q_half(x)`` for ``q = -d(phi)/dn_body`` (from
``grad(phi o M) = s M grad(phi)`` and ``n(M x) = M n(x)``).

Mode parity
-----------
The six rigid-body modes split cleanly for a body symmetric about ``y = 0``:

===========  ======  ======================================================
mode         s       reason (normal velocity ``n_i`` at the mirrored panel)
===========  ======  ======================================================
surge  (0)   +1      ``n_x``                     even
sway   (1)   -1      ``n_y``                     odd
heave  (2)   +1      ``n_z``                     even
roll   (3)   -1      ``(r x n)_x``               odd
pitch  (4)   +1      ``(r x n)_y``               even (``z n_x - x n_z``)
yaw    (5)   -1      ``(r x n)_z``               odd
===========  ======  ======================================================

The parity of the *generalized* normals is asserted numerically, not assumed:
``n_i(M x) = s_i n_i(x)`` is checked by
:func:`verify_generalized_normal_parity`.

Consequences
------------
* Radiation: one solve per distinct parity (at most two, each of half the panels)
  yields the full ``6 x 6`` added-mass and damping matrices.  The folding weight
  ``(1 + s_i s_j)`` makes a coupling between modes of *different* parity
  identically zero -- a property the full-mesh solver only reproduces to
  discretisation error.
* Diffraction: the incident Airy wave is split into its even and odd halves
  under ``y -> -y``, each solved once with the matching parity, and the
  exciting loads are recombined with the same folding weight.

Cost
----
For ``N`` panels on the full hull the reduced systems have ``N/2`` unknowns, so
one parity costs ``2 (N/2)^2 = N^2/2`` kernel entries (half of the full ``N^2``)
and a dense solve that is ``8x`` smaller; two parities cost ``N^3/4`` flops
instead of ``N^3``.

Limitations
-----------
* Only the ``y = 0`` plane is supported and the retained half is ``y >= 0``.
  The surface must have no panel straddling the symmetry plane: mirroring never
  splits a panel, so a mesh whose seam is not a panel edge is rejected.
* The expansion point must lie on the symmetry plane (``center[1] == 0``),
  otherwise the generalized normals have no definite parity.
* Far-field post-processing (:func:`tensorbem.free_surface_solver3d.\
far_field_damping_matrix_3d`, :func:`...haskind_excitation_force_3d`) is not
  folded here; those helpers expect a full-surface potential.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .free_surface_solver3d import (
    Diffraction3DResult,
    FreeSurface3DOperators,
    Radiation3DResult,
    _auto_chief_points_3d,
    _build_target_operators_3d,
    _solve_system,
    _validate_positive,
    incident_airy_wave_3d,
)
from .geometry3d import SurfaceMesh3D, TriangleMesh, WettedHullMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665
_PLANE_TOL = 1.0e-12

#: Parity ``s_i`` of the six rigid-body modes for a body symmetric about ``y = 0``.
MODE_PARITY: dict[int, float] = {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0, 4: 1.0, 5: -1.0}


@dataclass(frozen=True)
class PanelPatch:
    """An *open* triangular surface patch carrying the mesh attribute surface.

    ``TriangleMesh`` refuses watertight-open geometry and ``WettedHullMesh``
    refuses interior boundary edges that are not on ``z = 0``; a half hull cut by
    a symmetry plane is neither.  ``PanelPatch`` therefore performs no topology
    validation: it is a bag of triangles exposing exactly what the BEM assembly
    consumes (``v0``, ``v1``, ``v2``, ``areas``, ``normals``, ``centroids``,
    ``panel_count``).
    """

    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces)
        if verts.ndim != 2 or verts.shape[1] != 3 or not np.isfinite(verts).all():
            raise ValueError("vertices must be a finite array of shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            raise ValueError("faces must be a non-empty integer array of shape (F, 3)")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("faces must be an integer array")
        tris = np.asarray(faces, dtype=np.int64)
        if np.any(tris < 0) or np.any(tris >= verts.shape[0]):
            raise ValueError("face indices must be in range [0, vertex_count)")
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        double_areas = np.linalg.norm(cross, axis=1)
        if np.any(double_areas <= 0.0):
            raise ValueError("degenerate triangles are not allowed")
        object.__setattr__(self, "vertices", verts)
        object.__setattr__(self, "faces", tris)
        object.__setattr__(self, "_v0", v0)
        object.__setattr__(self, "_v1", v1)
        object.__setattr__(self, "_v2", v2)
        object.__setattr__(self, "_double_areas", double_areas)
        object.__setattr__(self, "_centroids", (v0 + v1 + v2) / 3.0)
        object.__setattr__(self, "_normals", cross / double_areas[:, None])

    # -- mesh-like read-only interface -----------------------------------
    @property
    def v0(self) -> FloatArray:
        return self._v0

    @property
    def v1(self) -> FloatArray:
        return self._v1

    @property
    def v2(self) -> FloatArray:
        return self._v2

    @property
    def double_areas(self) -> FloatArray:
        return self._double_areas

    @property
    def areas(self) -> FloatArray:
        return 0.5 * self._double_areas

    @property
    def centroids(self) -> FloatArray:
        return self._centroids

    @property
    def normals(self) -> FloatArray:
        return self._normals

    @property
    def panel_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def edge_lengths(self) -> FloatArray:
        e0 = np.linalg.norm(self._v1 - self._v0, axis=1)
        e1 = np.linalg.norm(self._v2 - self._v1, axis=1)
        e2 = np.linalg.norm(self._v0 - self._v2, axis=1)
        return np.stack((e0, e1, e2), axis=1)

    @property
    def h_max(self) -> float:
        return float(np.max(self.edge_lengths))

    @property
    def h_mean(self) -> float:
        return float(np.mean(self.edge_lengths))

    @property
    def surface_area(self) -> float:
        return float(np.sum(self.areas))


def _as_vertices_faces(mesh: SurfaceMesh3D | PanelPatch) -> tuple[FloatArray, IntArray]:
    if isinstance(mesh, PanelPatch):
        return mesh.vertices, mesh.faces
    if isinstance(mesh, (TriangleMesh, WettedHullMesh)):
        return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)
    raise ValueError("mesh must be a TriangleMesh, WettedHullMesh or PanelPatch")


def split_patch(
    mesh: SurfaceMesh3D | PanelPatch, *, plane: Literal["y"] = "y"
) -> tuple[PanelPatch, PanelPatch]:
    """Split a ``y = 0`` symmetric surface into its ``y >= 0`` and ``y <= 0`` halves.

    Panels that straddle the symmetry plane are rejected, because mirroring must
    not require splitting a panel.  Vertices within ``1e-12`` of the plane
    (relative to the mesh size) are snapped exactly onto it so both halves share
    the seam bit-for-bit -- that exactness is what makes the half and full
    systems algebraically identical rather than merely close.
    """
    if plane != "y":
        raise ValueError('only plane="y" is supported')
    verts, faces = _as_vertices_faces(mesh)
    scale = max(1.0, float(np.max(np.linalg.norm(verts - verts.mean(axis=0), axis=1))))
    tol = _PLANE_TOL * scale
    snapped = verts.copy()
    snapped[np.abs(snapped[:, 1]) <= tol, 1] = 0.0

    tri_y = snapped[faces][:, :, 1]
    upper = np.all(tri_y >= -tol, axis=1)
    lower = np.all(tri_y <= tol, axis=1)
    straddle = ~(upper | lower)
    if np.any(straddle):
        raise ValueError(
            f"{int(np.sum(straddle))} panel(s) straddle the y = 0 symmetry plane; "
            "the half surface must be cut along panel edges"
        )
    return _sub_patch(snapped, faces, upper), _sub_patch(snapped, faces, lower)


def _sub_patch(verts: FloatArray, faces: IntArray, mask: NDArray[np.bool_]) -> PanelPatch:
    sub_faces = faces[mask]
    if sub_faces.shape[0] == 0:
        raise ValueError("the requested half of the symmetry split contains no panels")
    used, inverse = np.unique(sub_faces, return_inverse=True)
    return PanelPatch(verts[used], inverse.reshape(-1, 3))


def mirror_patch(patch: PanelPatch) -> PanelPatch:
    """Reflect ``patch`` in ``y = 0`` keeping outward normals.

    The reflection reverses the orientation of every triangle, so the second and
    third vertices are swapped: the mirrored normal is then
    ``(n_x, -n_y, n_z)``, still pointing out of the body.
    """
    verts = patch.vertices.copy()
    verts[:, 1] = -verts[:, 1]
    faces = patch.faces[:, [0, 2, 1]]
    return PanelPatch(np.ascontiguousarray(verts), np.ascontiguousarray(faces))


def generalized_normals_3d(
    patch: PanelPatch, center: ArrayLike | None = None
) -> FloatArray:
    """Rigid-body generalized normals ``(n, (x - center) x n)`` of shape ``(F, 6)``."""
    ref = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
    if ref.shape != (3,) or not np.isfinite(ref).all():
        raise ValueError("center must be a finite 3-element coordinate")
    rel = patch.centroids - ref[None, :]
    return np.hstack((patch.normals, np.cross(rel, patch.normals)))


def verify_generalized_normal_parity(
    patch: PanelPatch, center: ArrayLike | None = None, *, tolerance: float = 1.0e-12
) -> FloatArray:
    """Return ``max_p |n_i(mirror p) - s_i n_i(p)|`` for each of the six modes.

    ``patch`` is compared panel-by-panel against :func:`mirror_patch`, so the
    returned vector is the numerical proof (or refutation) of the parity table in
    the module docstring for the mesh at hand.
    """
    ref = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
    if abs(float(ref[1])) > tolerance:
        raise ValueError("center must lie on the symmetry plane y = 0")
    n6 = generalized_normals_3d(patch, ref)
    n6_mirror = generalized_normals_3d(mirror_patch(patch), ref)
    parity = np.array([MODE_PARITY[k] for k in range(6)], dtype=float)
    return np.max(np.abs(n6_mirror - parity[None, :] * n6), axis=0)


@dataclass(frozen=True)
class SymmetricOperatorBlocks:
    """Parity-independent operator blocks of a half hull at one wavenumber.

    ``*_half`` acts from the retained panels, ``*_mirror`` from their ``y = 0``
    images; a parity ``s`` is folded in later
    (:func:`operators_for_parity`), so both parities share one assembly.
    """

    wavenumber: float
    single_half: ComplexArray
    double_half: ComplexArray
    single_mirror: ComplexArray
    double_mirror: ComplexArray
    chief_points: FloatArray
    chief_single_half: ComplexArray
    chief_double_half: ComplexArray
    chief_single_mirror: ComplexArray
    chief_double_mirror: ComplexArray
    panel_count: int

    def assembly_entries(self) -> int:
        """Kernel matrix entries built: ``2 (N/2)^2 = N^2/2`` for ``N`` full panels."""
        return int(self.single_half.size + self.single_mirror.size)


def assemble_symmetric_blocks(
    patch: PanelPatch,
    nu: float,
    *,
    chief_points: ArrayLike | Literal["auto"] | None = None,
    rankine_order: int = 5,
) -> SymmetricOperatorBlocks:
    """Assemble the half/mirror operator blocks of a ``y = 0`` symmetric patch."""
    if not isinstance(patch, PanelPatch):
        raise ValueError("patch must be a PanelPatch (use split_patch to obtain one)")
    freq = _validate_positive("nu", nu)
    if (
        isinstance(rankine_order, (bool, np.bool_))
        or not isinstance(rankine_order, (int, np.integer))
        or int(rankine_order) < 2
    ):
        raise ValueError("rankine_order must be an integer >= 2")
    r_order = int(rankine_order)

    mirror = mirror_patch(patch)
    s_half, h_half = _build_target_operators_3d(
        patch.centroids, patch, freq, self_interaction=True, rankine_order=r_order
    )
    s_mirror, h_mirror = _build_target_operators_3d(
        patch.centroids, mirror, freq, self_interaction=False, rankine_order=r_order
    )

    if chief_points is None:
        chief = np.zeros((0, 3), dtype=float)
    elif isinstance(chief_points, str):
        if chief_points != "auto":
            raise ValueError("chief_points must be None, 'auto', or an array of shape (K, 3)")
        chief = _auto_chief_points_3d(patch)
    else:
        chief = np.asarray(chief_points, dtype=float)
        if chief.ndim != 2 or chief.shape[1] != 3 or not np.isfinite(chief).all():
            raise ValueError("chief_points must be a finite array of shape (K, 3)")
        if chief.shape[0] > 0 and np.any(chief[:, 2] >= 0.0):
            raise ValueError("3-D CHIEF interior points must satisfy z < 0")

    if chief.shape[0] > 0:
        cs_half, ch_half = _build_target_operators_3d(
            chief, patch, freq, self_interaction=False, rankine_order=r_order
        )
        cs_mirror, ch_mirror = _build_target_operators_3d(
            chief, mirror, freq, self_interaction=False, rankine_order=r_order
        )
    else:
        empty = np.zeros((0, patch.panel_count), dtype=np.complex128)
        cs_half = ch_half = cs_mirror = ch_mirror = empty

    return SymmetricOperatorBlocks(
        wavenumber=freq,
        single_half=s_half,
        double_half=h_half,
        single_mirror=s_mirror,
        double_mirror=h_mirror,
        chief_points=chief,
        chief_single_half=cs_half,
        chief_double_half=ch_half,
        chief_single_mirror=cs_mirror,
        chief_double_mirror=ch_mirror,
        panel_count=patch.panel_count,
    )


def operators_for_parity(
    blocks: SymmetricOperatorBlocks, parity: float
) -> FreeSurface3DOperators:
    """Fold the blocks of :func:`assemble_symmetric_blocks` at one parity ``s``."""
    if parity not in (1.0, -1.0):
        raise ValueError("parity must be +1 or -1")
    s = float(parity)
    single = blocks.single_half + s * blocks.single_mirror
    double = blocks.double_half + s * blocks.double_mirror
    system = 0.5 * np.eye(blocks.panel_count, dtype=np.complex128) - double
    if blocks.chief_points.shape[0] > 0:
        chief_single = blocks.chief_single_half + s * blocks.chief_single_mirror
        chief_double = blocks.chief_double_half + s * blocks.chief_double_mirror
        system = np.vstack((system, -chief_double))
    else:
        chief_single = np.zeros((0, blocks.panel_count), dtype=np.complex128)
        chief_double = np.zeros((0, blocks.panel_count), dtype=np.complex128)
    return FreeSurface3DOperators(
        wavenumber=blocks.wavenumber,
        single_layer=single,
        double_layer=double,
        chief_points=blocks.chief_points,
        chief_single_layer=chief_single,
        chief_double_layer=chief_double,
        system_matrix=system,
        condition_number=float(np.linalg.cond(system)),
    )


def assemble_symmetric_operators_3d(
    patch: PanelPatch,
    nu: float,
    parity: float,
    *,
    chief_points: ArrayLike | Literal["auto"] | None = None,
    rankine_order: int = 5,
) -> FreeSurface3DOperators:
    """Convenience wrapper: assemble the blocks and fold them at a single parity."""
    blocks = assemble_symmetric_blocks(
        patch, nu, chief_points=chief_points, rankine_order=rankine_order
    )
    return operators_for_parity(blocks, parity)


def _validate_modes(modes: Sequence[int] | None) -> IntArray:
    if modes is None:
        return np.arange(6, dtype=np.int64)
    mode_arr = np.asarray(list(modes), dtype=np.int64)
    if mode_arr.ndim != 1 or mode_arr.size == 0:
        raise ValueError("modes must be a non-empty 1-D sequence of integers in 0..5")
    if np.any((mode_arr < 0) | (mode_arr > 5)):
        raise ValueError("each mode index must be in range 0..5")
    return mode_arr


def _center_on_plane(center: ArrayLike | None, *, atol: float = 1.0e-12) -> FloatArray:
    ref = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
    if ref.shape != (3,) or not np.isfinite(ref).all():
        raise ValueError("center must be a finite 3-element coordinate")
    if abs(float(ref[1])) > atol:
        raise ValueError(
            "the symmetry expansion point must lie on the symmetry plane (center[1] == 0)"
        )
    return ref


def _resolve_blocks(
    patch: PanelPatch,
    freq: float,
    blocks: SymmetricOperatorBlocks | None,
    chief_points: ArrayLike | Literal["auto"] | None,
    rankine_order: int,
) -> SymmetricOperatorBlocks:
    blk = (
        blocks
        if blocks is not None
        else assemble_symmetric_blocks(
            patch, freq, chief_points=chief_points, rankine_order=rankine_order
        )
    )
    if abs(blk.wavenumber - freq) > 1.0e-14 * max(1.0, freq):
        raise ValueError("blocks.wavenumber does not match nu")
    if blk.panel_count != patch.panel_count:
        raise ValueError("blocks were assembled for a different patch")
    return blk


def solve_radiation_3d_symmetric(
    patch: PanelPatch,
    nu: float,
    *,
    modes: Sequence[int] | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    chief_points: ArrayLike | Literal["auto"] | None = None,
    rankine_order: int = 5,
    blocks: SymmetricOperatorBlocks | None = None,
) -> Radiation3DResult:
    """6-DOF 3-D wave radiation on a ``y = 0`` symmetric half hull.

    Returns the same :class:`tensorbem.free_surface_solver3d.Radiation3DResult`
    as the full-mesh solver; ``potentials`` and ``normal_velocities`` refer to the
    retained half panels (the mirrored half is ``s_j`` times that).
    """
    if not isinstance(patch, PanelPatch):
        raise ValueError("patch must be a PanelPatch (use split_patch to obtain one)")
    freq = _validate_positive("nu", nu)
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    ref = _center_on_plane(center)
    omega = math.sqrt(g * freq)
    mode_arr = _validate_modes(modes)
    blk = _resolve_blocks(patch, freq, blocks, chief_points, rankine_order)

    n6 = generalized_normals_3d(patch, ref)
    n_modes = n6[:, mode_arr]
    parities = np.array([MODE_PARITY[int(k)] for k in mode_arr], dtype=float)

    potentials = np.zeros((patch.panel_count, mode_arr.size), dtype=np.complex128)
    worst_residual = 0.0
    worst_condition = 0.0
    for s in (1.0, -1.0):
        cols = np.nonzero(parities == s)[0]
        if cols.size == 0:
            continue
        ops = operators_for_parity(blk, s)
        q_ext = (-n_modes[:, cols]).astype(np.complex128)
        sol, rel_res = _solve_system(ops, q_ext)
        potentials[:, cols] = sol
        worst_residual = max(worst_residual, rel_res)
        worst_condition = max(worst_condition, ops.condition_number)

    weighted = patch.areas[:, None] * n_modes
    # Folding weight: a load/field-mode pair survives only when the parities match.
    fold = 1.0 + np.outer(parities, parities)
    impedance = -rho * fold * (weighted.T @ potentials)
    added_mass = np.asarray(impedance.real, dtype=float)
    damping = np.asarray(omega * impedance.imag, dtype=float)

    return Radiation3DResult(
        modes=mode_arr,
        potentials=potentials,
        normal_velocities=n_modes,
        added_mass=added_mass,
        damping=damping,
        wavenumber=freq,
        omega=omega,
        chief_points_used=blk.chief_points,
        condition_number=worst_condition,
        relative_residual=worst_residual,
    )


def solve_diffraction_3d_symmetric(
    patch: PanelPatch,
    nu: float,
    *,
    amplitude: float = 1.0,
    wave_angle: float = 0.0,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    chief_points: ArrayLike | Literal["auto"] | None = None,
    rankine_order: int = 5,
    blocks: SymmetricOperatorBlocks | None = None,
) -> Diffraction3DResult:
    """3-D regular-wave scattering and 6-DOF exciting loads on a half hull.

    The incident Airy wave is split into its even/odd parts under ``y -> -y``,
    each part is solved with the matching parity, and the half-mesh potentials are
    summed.  ``total_potential``, ``scattered_potential``, ``incident_potential``
    and ``incident_normal_velocity`` refer to the retained half panels; the
    exciting force is the full-surface integral.
    """
    if not isinstance(patch, PanelPatch):
        raise ValueError("patch must be a PanelPatch (use split_patch to obtain one)")
    freq = _validate_positive("nu", nu)
    amp = _validate_positive("amplitude", amplitude)
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    ref = _center_on_plane(center)
    omega = math.sqrt(g * freq)
    beta = float(wave_angle)
    if not math.isfinite(beta):
        raise ValueError("wave_angle must be finite")
    blk = _resolve_blocks(patch, freq, blocks, chief_points, rankine_order)

    mirror = mirror_patch(patch)
    phi_inc_half, grad_half = incident_airy_wave_3d(
        patch.centroids, freq, amplitude=amp, wave_angle=beta, gravity=g
    )
    phi_inc_mirror, grad_mirror = incident_airy_wave_3d(
        mirror.centroids, freq, amplitude=amp, wave_angle=beta, gravity=g
    )
    q_half = np.einsum("pk,pk->p", grad_half, patch.normals)
    q_mirror = np.einsum("pk,pk->p", grad_mirror, mirror.normals)

    n6 = generalized_normals_3d(patch, ref)
    weighted = patch.areas[:, None] * n6
    parity = np.array([MODE_PARITY[k] for k in range(6)], dtype=float)

    scattered = np.zeros(patch.panel_count, dtype=np.complex128)
    flux = np.zeros((2, 6), dtype=np.complex128)  # (parity, mode) surface integral
    worst_residual = 0.0
    worst_condition = 0.0
    for index, s in enumerate((1.0, -1.0)):
        ops = operators_for_parity(blk, s)
        # The scattered problem has q_e = -d(phi_S)/dn = +d(phi_I)/dn, matching
        # the convention of :func:`tensorbem.free_surface_solver3d.solve_diffraction_3d`.
        q_ext = (0.5 * (q_half + s * q_mirror))[:, None].astype(np.complex128)
        sol, rel_res = _solve_system(ops, q_ext)
        phi_s = sol[:, 0]
        scattered += phi_s
        phi_inc_s = 0.5 * (phi_inc_half + s * phi_inc_mirror)
        flux[index] = weighted.T @ (phi_inc_s + phi_s)
        worst_residual = max(worst_residual, rel_res)
        worst_condition = max(worst_condition, ops.condition_number)

    fold = 1.0 + np.outer(parity, np.array([1.0, -1.0]))  # (mode, parity)
    forces = -1.0j * omega * rho * np.einsum("ip,pi->i", fold, flux)

    return Diffraction3DResult(
        total_potential=phi_inc_half + scattered,
        scattered_potential=scattered,
        incident_potential=phi_inc_half,
        incident_normal_velocity=q_half,
        excitation_force=np.asarray(forces, dtype=np.complex128),
        wavenumber=freq,
        omega=omega,
        wave_angle=beta,
        amplitude=amp,
        chief_points_used=blk.chief_points,
        condition_number=worst_condition,
        relative_residual=worst_residual,
    )