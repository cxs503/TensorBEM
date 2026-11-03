"""Mirror-symmetry (``xOz`` plane) reduction for 3-D free-surface BEM.

Why this module exists
----------------------
Every 3-D assembly in :mod:`tensorbem.free_surface_solver3d` builds a dense
``(N, N)`` complex matrix, so the cost of a panel-convergence study grows as
``N**2`` in memory and ``N**3`` in the direct solve.  A large class of marine
benchmarks -- floating hemispheres, submerged spheres, symmetric hulls, and
the Hulme (1982) / Wang-Linton geometries used throughout the verification
chain -- are symmetric about the vertical plane ``y = 0`` (``xOz``).  For those
bodies the source strength on a mirror pair of panels is either identical
(symmetric modes) or opposite (antisymmetric modes), so the ``2 N`` unknowns of
the full surface collapse to the ``N`` unknowns of one half.

The reduction is *exact*, not an approximation: with the mirror image treated
as an extra source set, the folded system is algebraically identical to the
full one, and the two agree to round-off.  The only reason to keep the full
path around is that not every geometry is symmetric.

Physical basis
--------------
Let ``R`` be the reflection ``(x, y, z) -> (x, -y, z)``.  The zero-speed
deep-water Green function used here depends on the horizontal separation
``R_h = hypot(x - xi, y - eta)`` and on ``z, zeta`` only, hence
``G(R p, q) = G(p, R q)``.  If the wetted surface ``Gamma`` is invariant under
``R``, write ``Gamma = Gamma_+ u R Gamma_+``.  For a potential with definite
parity ``phi(R p) = s phi(p)``, ``s = +-1``, the boundary-integral equation

    (1/2) phi(p) - int_Gamma dG/dn(q, p) phi(q) dS_q = int_Gamma G(p, q) sigma(q) dS_q

restricted to ``p in Gamma_+`` only involves the folded kernels

    H_eff(p, q) = H(p, q) + s H(p, R q),     S_eff(p, q) = S(p, q) + s S(p, R q),

i.e. the ``n`` half-panels with the ``2 n`` original+mirrored sources as the
column set.  That is ``2 n**2`` kernel evaluations instead of ``4 n**2``.

Parity bookkeeping
------------------
For the six rigid-body modes and ``xOz`` reflection (``y -> -y``):

===============  ==========================  =========
mode index       motion                      parity s
===============  ==========================  =========
0                surge (translation along x)   +1
1                sway  (translation along y)   -1
2                heave (translation along z)   +1
3                roll  (rotation about x)      -1
4                pitch (rotation about y)      +1
5                yaw   (rotation about z)      -1
===============  ==========================  =========

A pressure-type integral of mode ``j`` against the normal of mode ``i`` over
the full surface picks up the factor ``(1 + s_i s_j)``: it doubles inside a
parity class and vanishes across classes.  That factor is what makes the
off-parity coupling entries come out identically zero instead of merely small,
exactly as the geometry demands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .free_surface_solver3d import (
    Diffraction3DResult,
    FreeSurface3DOperators,
    Radiation3DResult,
    _build_target_operators_3d,
    _resolve_chief_points_3d,
    _validate_mesh_3d,
    _validate_positive,
    incident_airy_wave_3d,
    rigid_body_normals_3d,
)
from .geometry3d import SurfaceMesh3D

__all__ = [
    "MODE_PARITY_XOZ",
    "HalfMeshSplit",
    "assemble_free_surface_operators_3d_half",
    "full_surface_factor",
    "solve_diffraction_3d_symmetric",
    "solve_radiation_3d_symmetric",
    "split_symmetric_mesh",
]

#: Parity of the six rigid-body modes under ``xOz`` reflection ``y -> -y``.
MODE_PARITY_XOZ: tuple[int, int, int, int, int, int] = (1, -1, 1, -1, 1, -1)

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665
_DECIMALS = 10


@dataclass(frozen=True)
class _PanelBundle:
    """Minimal panel arrays consumed by ``_build_target_operators_3d``."""

    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    areas: np.ndarray
    normals: np.ndarray
    centroids: np.ndarray
    panel_count: int


@dataclass(frozen=True)
class HalfMeshSplit:
    """One symmetric half of a mirror-symmetric 3-D wetted surface.

    Attributes
    ----------
    half_to_full:
        Index into the full mesh of every half panel (row order of the folded
        system).
    mirror_of_half:
        Index into the full mesh of the mirror image of every half panel; the
        column order of the source bundle is ``[half, mirror]``.
    """

    full_panel_count: int
    half_to_full: np.ndarray
    mirror_of_half: np.ndarray
    half: _PanelBundle
    sources: _PanelBundle

    @property
    def half_panel_count(self) -> int:
        return int(self.half.panel_count)

    @property
    def centroids(self) -> np.ndarray:
        return self.half.centroids

    @property
    def areas(self) -> np.ndarray:
        return self.half.areas

    @property
    def normals(self) -> np.ndarray:
        return self.half.normals

    def mirror_full_indices(self) -> np.ndarray:
        """Full-mesh index permutation induced by the reflection."""
        perm = np.empty(self.full_panel_count, dtype=np.int64)
        perm[self.half_to_full] = self.mirror_of_half
        perm[self.mirror_of_half] = self.half_to_full
        return perm

    def expand(self, values: np.ndarray, parity: int) -> np.ndarray:
        """Expand half-surface panel values to full-mesh ordering."""
        arr = np.asarray(values)
        out = np.empty((self.full_panel_count,) + arr.shape[1:], dtype=arr.dtype)
        out[self.half_to_full] = arr
        out[self.mirror_of_half] = parity * arr
        return out


def full_surface_factor(parity_i: int, parity_j: int) -> float:
    """Multiplier turning a half-surface integral into a full-surface one."""
    return float(1 + int(parity_i) * int(parity_j))


def _triangle_key(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple:
    pts = sorted(
        tuple(np.round(np.asarray(p, dtype=float), _DECIMALS).tolist()) for p in (a, b, c)
    )
    return tuple(pts)


def _panel_arrays(mesh: SurfaceMesh3D, index: np.ndarray) -> _PanelBundle:
    v0 = np.asarray(mesh.v0, dtype=float)[index]
    v1 = np.asarray(mesh.v1, dtype=float)[index]
    v2 = np.asarray(mesh.v2, dtype=float)[index]
    return _PanelBundle(
        v0=v0,
        v1=v1,
        v2=v2,
        areas=np.asarray(mesh.areas, dtype=float)[index],
        normals=np.asarray(mesh.normals, dtype=float)[index],
        centroids=np.asarray(mesh.centroids, dtype=float)[index],
        panel_count=int(index.size),
    )


def split_symmetric_mesh(
    mesh: SurfaceMesh3D,
    *,
    plane: Literal["xOz"] = "xOz",
    mirror_tol: float = 1.0e-9,
) -> HalfMeshSplit:
    """Split a wetted surface that is exactly invariant under ``y -> -y``.

    Raises
    ------
    ValueError
        If the geometry is not a ``TriangleMesh``/``WettedHullMesh``, if the
        requested plane is not ``xOz``, if some panel straddles the symmetry
        plane, or if the panel set is not closed under reflection.
    """
    if plane != "xOz":
        raise ValueError("only the vertical 'xOz' symmetry plane is supported")
    _validate_mesh_3d(mesh)

    n_full = int(mesh.panel_count)
    if n_full % 2 != 0:
        raise ValueError(
            "a mesh symmetric about xOz must have an even panel count, "
            f"got {n_full}"
        )

    v0 = np.asarray(mesh.v0, dtype=float)
    v1 = np.asarray(mesh.v1, dtype=float)
    v2 = np.asarray(mesh.v2, dtype=float)
    centroids = np.asarray(mesh.centroids, dtype=float)

    keys: list[tuple] = [
        _triangle_key(v0[i], v1[i], v2[i]) for i in range(n_full)
    ]
    lookup = {key: idx for idx, key in enumerate(keys)}
    if len(lookup) != n_full:
        raise ValueError("duplicate panels encountered; cannot build a mirror map")

    reflect = np.array([1.0, -1.0, 1.0], dtype=float)
    mirror_index = np.empty(n_full, dtype=np.int64)
    for i in range(n_full):
        a, b, c = v0[i], v1[i], v2[i]
        # reflection reverses orientation, so swap two vertices to keep the
        # outward normal convention of the parent mesh
        key = _triangle_key(a * reflect, c * reflect, b * reflect)
        j = lookup.get(key)
        if j is None:
            raise ValueError(
                f"panel {i} has no mirror image under y -> -y; the mesh is not "
                "symmetric about xOz"
            )
        mirror_index[i] = j

    if not np.array_equal(mirror_index[mirror_index], np.arange(n_full)):
        raise ValueError("the reflection does not induce an involution on the mesh")

    if np.any(np.abs(centroids[:, 1]) < mirror_tol):
        raise ValueError(
            "no panel may straddle the xOz symmetry plane (|y| < mirror_tol at "
            "a panel centroid); cut the mesh along the plane instead"
        )

    half_to_full = np.flatnonzero(centroids[:, 1] > 0.0)
    if half_to_full.size != n_full // 2:
        raise ValueError(
            "expected exactly half of the panels above the symmetry plane, got "
            f"{half_to_full.size} of {n_full}"
        )
    mirror_of_half = mirror_index[half_to_full]
    if np.any(centroids[mirror_of_half, 1] >= 0.0):
        raise ValueError("mirror images of the half panels must lie in y < 0")
    if not np.array_equal(
        np.sort(np.concatenate((half_to_full, mirror_of_half))), np.arange(n_full)
    ):
        raise ValueError("half panels and their images do not partition the mesh")

    half = _panel_arrays(mesh, half_to_full)
    image = _panel_arrays(mesh, mirror_of_half)
    # columns [n:2n] are the mirror partners themselves, i.e. panels already
    # located in y < 0 with the outward normals of the parent mesh
    sources = _PanelBundle(
        v0=np.concatenate((half.v0, image.v0)),
        v1=np.concatenate((half.v1, image.v1)),
        v2=np.concatenate((half.v2, image.v2)),
        areas=np.concatenate((half.areas, image.areas)),
        normals=np.concatenate((half.normals, image.normals)),
        centroids=np.concatenate((half.centroids, image.centroids)),
        panel_count=2 * half.panel_count,
    )
    # the mirrored triangles carry reflected vertices: their geometric normal
    # must equal the reflected outward normal of the parent panel
    cross = np.cross(sources.v1 - sources.v0, sources.v2 - sources.v0)
    geo = cross / np.linalg.norm(cross, axis=1)[:, None]
    if not np.allclose(geo, sources.normals, atol=1.0e-9, rtol=0.0):
        raise ValueError("panel orientation is inconsistent with the stored normals")
    ys = sources.centroids[:, 1]
    if np.any(ys[: half.panel_count] <= 0.0) or np.any(ys[half.panel_count :] >= 0.0):
        raise ValueError("source bundle must pair y > 0 half panels with y < 0 images")

    return HalfMeshSplit(
        full_panel_count=n_full,
        half_to_full=np.asarray(half_to_full, dtype=np.int64),
        mirror_of_half=np.asarray(mirror_of_half, dtype=np.int64),
        half=half,
        sources=sources,
    )


def _validate_parity(parity: int) -> int:
    if isinstance(parity, (bool, np.bool_)) or not isinstance(parity, (int, np.integer)):
        raise ValueError("parity must be the integer +1 or -1")
    if int(parity) not in (1, -1):
        raise ValueError("parity must be +1 (symmetric) or -1 (antisymmetric)")
    return int(parity)


def _half_chief_points(
    mesh: SurfaceMesh3D,
    chief_points: ArrayLike | Literal["auto"] | None,
    mirror_tol: float,
) -> np.ndarray:
    points = _resolve_chief_points_3d(mesh, chief_points)
    if points.shape[0] == 0:
        return points
    # Only one representative per mirror pair may be retained: a interior point
    # at (x, -y, z) carries the *same* equation up to the factor -parity, so
    # feeding it back into the folded system would over-determine it
    # inconsistently.  Points sitting exactly on the symmetry plane are dropped
    # as they carry no information beyond their mirror image.
    keep = points[points[:, 1] > mirror_tol]
    if keep.shape[0] == 0:
        raise ValueError(
            "no CHIEF point lies strictly off the xOz symmetry plane; supply "
            "chief_points with y != 0 or use the full (non-symmetric) solver"
        )
    return np.ascontiguousarray(keep, dtype=float)


def assemble_free_surface_operators_3d_half(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    parity: int = 1,
    chief_points: ArrayLike | Literal["auto"] | None = "auto",
    rankine_order: int = 5,
    split: HalfMeshSplit | None = None,
) -> FreeSurface3DOperators:
    """Assemble the folded ``(n, n)`` Kelvin operators for one parity class.

    The returned matrices act on the half-surface unknowns; ``mesh`` is the
    *full* (mirror-symmetric) wetted surface.  Entries are

        ``S_eff = S(p, q) + parity * S(p, R q)``,  ``H_eff = H(p, q) + parity * H(p, R q)``

    with ``p, q`` running over the half panels and ``R`` the ``xOz`` reflection.
    """
    _validate_mesh_3d(mesh)
    freq = _validate_positive("nu", nu)
    s = _validate_parity(parity)
    if isinstance(rankine_order, (bool, np.bool_)) or not isinstance(
        rankine_order, (int, np.integer)
    ) or int(rankine_order) < 2:
        raise ValueError("rankine_order must be an integer >= 2")
    r_order = int(rankine_order)

    half = split if split is not None else split_symmetric_mesh(mesh)
    if half.full_panel_count != int(mesh.panel_count):
        raise ValueError("split was produced for a different mesh")

    s_raw, h_raw = _build_target_operators_3d(
        half.centroids,
        half.sources,
        freq,
        self_interaction=True,
        rankine_order=r_order,
    )
    n = half.half_panel_count
    single_layer = s_raw[:, :n] + s * s_raw[:, n:]
    double_layer = h_raw[:, :n] + s * h_raw[:, n:]
    system_matrix = 0.5 * np.eye(n, dtype=np.complex128) - double_layer

    chief_pts = _half_chief_points(mesh, chief_points, 1.0e-9)
    if chief_pts.shape[0] > 0:
        cs_raw, ch_raw = _build_target_operators_3d(
            chief_pts,
            half.sources,
            freq,
            self_interaction=False,
            rankine_order=r_order,
        )
        chief_single = cs_raw[:, :n] + s * cs_raw[:, n:]
        chief_double = ch_raw[:, :n] + s * ch_raw[:, n:]
        system_matrix = np.vstack((system_matrix, -chief_double))
    else:
        chief_single = np.zeros((0, n), dtype=np.complex128)
        chief_double = np.zeros((0, n), dtype=np.complex128)

    return FreeSurface3DOperators(
        wavenumber=freq,
        single_layer=single_layer,
        double_layer=double_layer,
        chief_points=chief_pts,
        chief_single_layer=chief_single,
        chief_double_layer=chief_double,
        system_matrix=system_matrix,
        condition_number=float(np.linalg.cond(system_matrix)),
    )


def _solve_folded(
    ops: FreeSurface3DOperators, q_ext: np.ndarray
) -> tuple[np.ndarray, float]:
    """Solve the folded system and return ``(solution, relative residual)``."""
    rhs_body = ops.single_layer @ q_ext
    if ops.chief_points.shape[0] > 0:
        rhs = np.vstack((rhs_body, ops.chief_single_layer @ q_ext))
        sol, *_ = np.linalg.lstsq(ops.system_matrix, rhs, rcond=None)
    else:
        rhs = rhs_body
        sol = np.linalg.solve(ops.system_matrix, rhs)
    denom = max(float(np.linalg.norm(rhs)), 1.0e-30)
    rel_res = float(np.linalg.norm(ops.system_matrix @ sol - rhs) / denom)
    return np.asarray(sol, dtype=np.complex128), rel_res


def solve_radiation_3d_symmetric(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    modes: Sequence[int] | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    chief_points: ArrayLike | Literal["auto"] | None = "auto",
    rankine_order: int = 5,
) -> Radiation3DResult:
    """Radiation problem on a mirror-symmetric body, solved half-surface-wise.

    Returns the same :class:`Radiation3DResult` as
    :func:`tensorbem.free_surface_solver3d.solve_radiation_3d`, with the
    potentials expanded back onto the *full* panel ordering so that downstream
    Kochin / pressure post-processing can use the original mesh.
    """
    _validate_mesh_3d(mesh)
    freq = _validate_positive("nu", nu)
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    omega = math.sqrt(g * freq)

    if modes is None:
        mode_arr = np.arange(6, dtype=np.int64)
    else:
        mode_arr = np.asarray(list(modes), dtype=np.int64)
        if mode_arr.ndim != 1 or mode_arr.size == 0:
            raise ValueError("modes must be a non-empty 1-D sequence of integers in 0..5")
        if np.any((mode_arr < 0) | (mode_arr > 5)):
            raise ValueError("each mode index must be in range 0..5")

    half = split_symmetric_mesh(mesh)
    n6_half = rigid_body_normals_3d(mesh, center=center)[half.half_to_full]
    n_modes_half = n6_half[:, mode_arr]
    parities = np.array([MODE_PARITY_XOZ[int(m)] for m in mode_arr], dtype=np.int64)

    cache: dict[int, FreeSurface3DOperators] = {}
    half_potentials = np.empty((half.half_panel_count, mode_arr.size), dtype=np.complex128)
    residuals: list[float] = []
    conds: list[float] = []
    for s in (1, -1):
        sel = np.flatnonzero(parities == s)
        if sel.size == 0:
            continue
        ops = assemble_free_surface_operators_3d_half(
            mesh,
            freq,
            parity=s,
            chief_points=chief_points,
            rankine_order=rankine_order,
            split=half,
        )
        cache[s] = ops
        sol, rel_res = _solve_folded(ops, (-n_modes_half[:, sel]).astype(np.complex128))
        half_potentials[:, sel] = sol
        residuals.append(rel_res)
        conds.append(ops.condition_number)

    potentials = np.empty((int(mesh.panel_count), mode_arr.size), dtype=np.complex128)
    for j, s in enumerate(parities):
        potentials[:, j] = half.expand(half_potentials[:, j], int(s))

    weighted_half = half.areas[:, None] * n_modes_half
    impedance_half = -rho * (weighted_half.T @ half_potentials)
    factor = np.array(
        [[full_surface_factor(parities[i], parities[j]) for j in range(mode_arr.size)]
         for i in range(mode_arr.size)],
        dtype=float,
    )
    impedance = factor * impedance_half
    added_mass = np.asarray(impedance.real, dtype=float)
    damping = np.asarray(omega * impedance.imag, dtype=float)

    chief_used = (
        cache[int(parities[0])].chief_points if cache else np.zeros((0, 3), dtype=float)
    )
    return Radiation3DResult(
        modes=mode_arr,
        potentials=potentials,
        normal_velocities=rigid_body_normals_3d(mesh, center=center)[:, mode_arr],
        added_mass=added_mass,
        damping=damping,
        wavenumber=freq,
        omega=omega,
        chief_points_used=chief_used,
        condition_number=float(max(conds)),
        relative_residual=float(max(residuals)),
    )


def solve_diffraction_3d_symmetric(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    amplitude: float = 1.0,
    wave_angle: float = 0.0,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    chief_points: ArrayLike | Literal["auto"] | None = "auto",
    rankine_order: int = 5,
) -> Diffraction3DResult:
    """Diffraction on a mirror-symmetric body for waves along the ``x`` axis.

    Only ``wave_angle = 0`` or ``pi`` (head or following seas, i.e.
    ``|sin(wave_angle)| ~ 0``) keeps the incident potential symmetric about
    ``xOz``; oblique incidence has no definite parity and must use the full
    solver.
    """
    _validate_mesh_3d(mesh)
    freq = _validate_positive("nu", nu)
    amp = _validate_positive("amplitude", amplitude)
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    beta = float(wave_angle)
    if not math.isfinite(beta):
        raise ValueError("wave_angle must be finite")
    if abs(math.sin(beta)) > 1.0e-12:
        raise ValueError(
            "solve_diffraction_3d_symmetric requires |sin(wave_angle)| ~ 0 "
            "(waves along the x axis); use solve_diffraction_3d otherwise"
        )
    omega = math.sqrt(g * freq)

    half = split_symmetric_mesh(mesh)
    ops = assemble_free_surface_operators_3d_half(
        mesh, freq, parity=1, chief_points=chief_points, rankine_order=rankine_order
    )

    phi_inc_half, grad_inc_half = incident_airy_wave_3d(
        half.centroids, freq, amplitude=amp, wave_angle=beta, gravity=g
    )
    dphi_inc_dn = np.einsum("pk,pk->p", grad_inc_half, half.normals)
    sol, rel_res = _solve_folded(ops, dphi_inc_dn[:, None].astype(np.complex128))
    phi_scat_half = sol[:, 0]
    phi_tot_half = phi_inc_half + phi_scat_half

    n6_half = rigid_body_normals_3d(mesh, center=center)[half.half_to_full]
    weighted_half = half.areas[:, None] * n6_half
    forces_half = -1.0j * omega * rho * (weighted_half.T @ phi_tot_half)
    # incident / total potential is symmetric (s_j = +1); the factor therefore
    # doubles symmetric normals and annihilates antisymmetric ones
    factors = np.array(
        [full_surface_factor(int(MODE_PARITY_XOZ[i]), 1) for i in range(6)],
        dtype=float,
    )
    forces = factors * forces_half

    n6 = rigid_body_normals_3d(mesh, center=center)
    return Diffraction3DResult(
        total_potential=half.expand(phi_tot_half, 1),
        scattered_potential=half.expand(phi_scat_half, 1),
        incident_potential=incident_airy_wave_3d(
            np.asarray(mesh.centroids, dtype=float),
            freq,
            amplitude=amp,
            wave_angle=beta,
            gravity=g,
        )[0],
        incident_normal_velocity=np.einsum(
            "pk,pk->p",
            incident_airy_wave_3d(
                np.asarray(mesh.centroids, dtype=float),
                freq,
                amplitude=amp,
                wave_angle=beta,
                gravity=g,
            )[1],
            np.asarray(mesh.normals, dtype=float),
        ),
        excitation_force=np.asarray(forces, dtype=np.complex128),
        wavenumber=freq,
        omega=omega,
        wave_angle=beta,
        amplitude=amp,
        chief_points_used=ops.chief_points,
        condition_number=ops.condition_number,
        relative_residual=rel_res,
    )