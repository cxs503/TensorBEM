"""Three-dimensional frequency-domain wave radiation and diffraction BEM solver.

Formulation
-----------
In three dimensions, both the Rankine kernel ``1/(4*pi*r)`` and outgoing cylindrical
free-surface waves ``O(R**(-1/2))`` satisfy the decay and Sommerfeld radiation
conditions required for Green's third identity to close at infinity (unlike 2-D,
where ``log r`` and non-decaying surface waves require the indirect source
formulation of :mod:`tensorbem.free_surface_solver`). For any exterior velocity
potential ``phi`` satisfying ``nabla^2 phi = 0`` in ``Omega_f subset {z <= 0}``,
``d(phi)/dz - nu*phi = 0`` on ``z = 0``, and outgoing radiation at infinity,
direct collocation on the wetted hull ``Gamma_B`` gives

    (1/2 I - H_nu) phi = S_nu q_e,      q_e = -d(phi)/dn_body,

where ``n_body`` points *out of the body into the fluid*, ``S_nu`` is the
single-layer operator of :func:`tensorbem.free_surface3d.green_function_3d`,
and ``H_nu`` is its source-normal double-layer operator.

Singular & mirror-image treatment
---------------------------------
For surface-piercing hulls (:class:`tensorbem.geometry3d.WettedHullMesh`), panel
vertices touch ``z = 0`` so the image distance ``r_1`` becomes nearly singular
as well as ``r``. Splitting

    G = 1/(4*pi*r) + 1/(4*pi*r_1) + G_W

allows *both* ``1/(4*pi*r)`` and ``1/(4*pi*r_1)`` to be integrated with exact
Van Oosterom-Strackee solid angles (for ``H``) and polar-sector / Duffy rules
(for ``S``) on the hull panels and their ``z = 0`` mirror-reflected triangles,
leaving only the smooth wave kernel ``G_W`` for 3-point degree-2 triangle
quadrature.

Irregular-frequency suppression (3-D CHIEF)
-------------------------------------------
On surface-piercing hulls, the unregularized exterior BIE suffers from discrete
irregular frequencies ``nu_k`` equal to eigenvalues of the interior waterplane
problem (John 1950; Hulme 1982). Applying Green's second identity in ``Omega_f``
with field points ``x_int`` strictly inside the hull below ``z = 0`` yields the
exact null-field equations

    -H_nu(x_int, Gamma_B) phi = S_nu(x_int, Gamma_B) q_e

without discretizing any waterplane lid panels. Stacking a small set of
asymmetric interior CHIEF rows (`chief_points="auto"`) and solving in a
least-squares sense removes the irregular-frequency spikes and restores
positive radiation damping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from .free_surface3d import pure_wave_value_and_source_gradient_3d
from .geometry3d import SurfaceMesh3D, TriangleMesh, WettedHullMesh
from .kernels3d import integrate_single_layer_triangle_self

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]

_FOUR_PI = 4.0 * math.pi
_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0
_BARY_DEG2 = np.array(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class FreeSurface3DOperators:
    """Pre-assembled 3-D Kelvin BEM matrices at one wavenumber ``nu``."""

    wavenumber: float
    single_layer: ComplexArray
    double_layer: ComplexArray
    chief_points: FloatArray
    chief_single_layer: ComplexArray
    chief_double_layer: ComplexArray
    system_matrix: ComplexArray
    condition_number: float


@dataclass(frozen=True)
class Radiation3DResult:
    """Multi-DOF 3-D wave-radiation BEM solution."""

    modes: IntArray
    potentials: ComplexArray
    normal_velocities: FloatArray
    added_mass: FloatArray
    damping: FloatArray
    wavenumber: float
    omega: float
    chief_points_used: FloatArray
    condition_number: float
    relative_residual: float


@dataclass(frozen=True)
class Diffraction3DResult:
    """3-D regular-wave diffraction BEM solution and 6-DOF exciting loads."""

    total_potential: ComplexArray
    scattered_potential: ComplexArray
    incident_potential: ComplexArray
    incident_normal_velocity: ComplexArray
    excitation_force: ComplexArray
    wavenumber: float
    omega: float
    wave_angle: float
    amplitude: float
    chief_points_used: FloatArray
    condition_number: float
    relative_residual: float


def _validate_positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite real number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _validate_mesh_3d(mesh: SurfaceMesh3D) -> SurfaceMesh3D:
    if not isinstance(mesh, (TriangleMesh, WettedHullMesh)):
        raise ValueError("mesh must be a TriangleMesh or WettedHullMesh")
    if isinstance(mesh, TriangleMesh):
        if np.any(mesh.vertices[:, 2] > 1.0e-13) or np.any(mesh.centroids[:, 2] >= 0.0):
            raise ValueError(
                "closed TriangleMesh must lie in z <= 0 with panel centroids strictly below z = 0"
            )
    return mesh


def _validate_center(center: ArrayLike | None) -> FloatArray:
    if center is None:
        return np.zeros(3, dtype=float)
    arr = np.asarray(center, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError("center must be a finite 3-element coordinate")
    return arr


def rigid_body_normals_3d(
    mesh: SurfaceMesh3D, center: ArrayLike | None = None
) -> FloatArray:
    """Return generalized 6-DOF body normals ``(n, (x - center) x n)`` of shape ``(F, 6)``."""
    _validate_mesh_3d(mesh)
    ref = _validate_center(center)
    normals = mesh.normals
    rel = mesh.centroids - ref[None, :]
    rot = np.cross(rel, normals)
    return np.hstack((normals, rot))


def _vectorized_solid_angle_dl(
    targets: FloatArray, v0: FloatArray, v1: FloatArray, v2: FloatArray
) -> FloatArray:
    """Exact double-layer integral ``int_T (x - y).n_y / (4*pi*|x-y|^3) dS``."""
    m_cnt = targets.shape[0]
    p_cnt = v0.shape[0]
    out = np.empty((m_cnt, p_cnt), dtype=float)
    for start in range(0, m_cnt, 128):
        stop = min(start + 128, m_cnt)
        tgt = targets[start:stop, None, :]
        a = v0[None, :, :] - tgt
        b = v1[None, :, :] - tgt
        c = v2[None, :, :] - tgt
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        cross_bc = np.cross(b, c)
        num = np.einsum("mpk,mpk->mp", a, cross_bc)
        ab = np.einsum("mpk,mpk->mp", a, b)
        bc = np.einsum("mpk,mpk->mp", b, c)
        ca = np.einsum("mpk,mpk->mp", c, a)
        den = la * lb * lc + ab * lc + bc * la + ca * lb
        coplanar = np.abs(num) <= 1.0e-14 * (la * lb * lc + 1.0e-30)
        omega = np.where(coplanar, 0.0, 2.0 * np.arctan2(num, den))
        out[start:stop] = -omega / _FOUR_PI
    return out


def _vectorized_single_layer_rankine(
    targets: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    areas: FloatArray,
    *,
    self_diagonal: bool,
    order: int,
) -> FloatArray:
    """Vectorized Duffy single-layer integral ``int_T 1/(4*pi*|x - y|) dS``."""
    nodes, gw = leggauss(order)
    u = 0.5 * (nodes + 1.0)
    du = 0.5 * gw
    gu, gv = np.meshgrid(u, u, indexing="ij")
    gu_flat = gu.ravel()
    gv_flat = gv.ravel()
    w_ref = 2.0 * gu_flat * (du[:, None] * du[None, :]).ravel()

    e1 = v1 - v0
    e2 = v2 - v0
    qpts = v0[:, None, :] + gu_flat[None, :, None] * (
        (1.0 - gv_flat)[None, :, None] * e1[:, None, :]
        + gv_flat[None, :, None] * e2[:, None, :]
    )

    m_cnt = targets.shape[0]
    p_cnt = v0.shape[0]
    out = np.empty((m_cnt, p_cnt), dtype=float)
    for start in range(0, m_cnt, 64):
        stop = min(start + 64, m_cnt)
        diff = targets[start:stop, None, None, :] - qpts[None, :, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=3), 1.0e-30)
        out[start:stop] = (areas[None, :] / _FOUR_PI) * ((1.0 / dist) @ w_ref)

    if self_diagonal:
        for idx in range(p_cnt):
            tri = np.stack((v0[idx], v1[idx], v2[idx]), axis=0)
            out[idx, idx] = integrate_single_layer_triangle_self(tri)
    return out


def _build_target_operators_3d(
    targets: FloatArray,
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    self_interaction: bool,
    rankine_order: int,
) -> tuple[ComplexArray, ComplexArray]:
    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    areas = mesh.areas
    normals = mesh.normals
    p_cnt = mesh.panel_count
    m_cnt = targets.shape[0]
    if m_cnt == 0:
        empty = np.zeros((0, p_cnt), dtype=np.complex128)
        return empty, empty

    refl = np.array([1.0, 1.0, -1.0], dtype=float)
    v0_img = v0 * refl
    v1_img = v2 * refl
    v2_img = v1 * refl

    s_rankine = _vectorized_single_layer_rankine(
        targets, v0, v1, v2, areas, self_diagonal=self_interaction, order=rankine_order
    ) + _vectorized_single_layer_rankine(
        targets, v0_img, v1_img, v2_img, areas, self_diagonal=False, order=rankine_order
    )

    h_rankine = _vectorized_solid_angle_dl(targets, v0, v1, v2)
    if self_interaction:
        np.fill_diagonal(h_rankine, 0.0)
    h_rankine = h_rankine + _vectorized_solid_angle_dl(targets, v0_img, v1_img, v2_img)

    qpts = (
        _BARY_DEG2[None, :, 0, None] * v0[:, None, :]
        + _BARY_DEG2[None, :, 1, None] * v1[:, None, :]
        + _BARY_DEG2[None, :, 2, None] * v2[:, None, :]
    ).reshape(-1, 3)
    q_areas = np.repeat(areas / 3.0, 3)
    q_normals = np.repeat(normals, 3, axis=0)

    s_wave = np.empty((m_cnt, p_cnt), dtype=np.complex128)
    h_wave = np.empty((m_cnt, p_cnt), dtype=np.complex128)
    row_chunk = max(1, 262144 // max(1, qpts.shape[0]))
    for start in range(0, m_cnt, row_chunk):
        stop = min(start + row_chunk, m_cnt)
        tgt_sub = targets[start:stop, None, :]
        src_sub = qpts[None, :, :]
        gw, grad_w = pure_wave_value_and_source_gradient_3d(tgt_sub, src_sub, nu)
        dg_dn = np.einsum("mqk,qk->mq", grad_w, q_normals)
        sub_len = stop - start
        s_wave[start:stop] = (gw * q_areas[None, :]).reshape(sub_len, p_cnt, 3).sum(axis=2)
        h_wave[start:stop] = (dg_dn * q_areas[None, :]).reshape(sub_len, p_cnt, 3).sum(axis=2)

    return s_rankine + s_wave, h_rankine + h_wave


def _auto_chief_points_3d(mesh: SurfaceMesh3D) -> FloatArray:
    """Generate 5 deterministic asymmetric interior CHIEF points in ``z < 0``."""
    areas = mesh.areas
    centroids = mesh.centroids
    hull_mean = np.sum(areas[:, None] * centroids, axis=0) / np.sum(areas)
    if isinstance(mesh, WettedHullMesh) and mesh.waterline_edges.shape[0] > 0:
        wl_verts = np.unique(mesh.waterline_edges.ravel())
        xy_center = np.mean(mesh.vertices[wl_verts, :2], axis=0)
        base = np.array([xy_center[0], xy_center[1], 0.65 * hull_mean[2]], dtype=float)
    else:
        base = hull_mean.copy()

    min_panel_dist = float(np.min(np.linalg.norm(centroids - base[None, :], axis=1)))
    depth_margin = abs(float(base[2]))
    radius = 0.35 * min(min_panel_dist, 0.8 * depth_margin)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.55, 0.31, 0.18],
            [-0.49, 0.53, -0.14],
            [0.34, -0.58, 0.22],
            [-0.57, -0.36, -0.25],
        ],
        dtype=float,
    )
    points = base[None, :] + radius * offsets
    if np.any(points[:, 2] >= -1.0e-12):
        raise ValueError("automatic CHIEF points failed to stay strictly below z = 0")
    return points


def _resolve_chief_points_3d(
    mesh: SurfaceMesh3D, chief_points: ArrayLike | Literal["auto"] | None
) -> FloatArray:
    if chief_points is None:
        return np.zeros((0, 3), dtype=float)
    if isinstance(chief_points, str):
        if chief_points != "auto":
            raise ValueError("chief_points must be None, 'auto', or an array of shape (K, 3)")
        return _auto_chief_points_3d(mesh)
    pts = np.asarray(chief_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("chief_points must be a finite array of shape (K, 3)")
    if pts.shape[0] > 0 and np.any(pts[:, 2] >= 0.0):
        raise ValueError("3-D CHIEF interior points must satisfy z < 0")
    return pts


def assemble_free_surface_operators_3d(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    chief_points: ArrayLike | Literal["auto"] | None = None,
    rankine_order: int = 5,
) -> FreeSurface3DOperators:
    """Assemble 3-D Kelvin BEM single/double-layer matrices and CHIEF rows."""
    _validate_mesh_3d(mesh)
    freq = _validate_positive("nu", nu)
    if isinstance(rankine_order, (bool, np.bool_)) or not isinstance(
        rankine_order, (int, np.integer)
    ) or int(rankine_order) < 2:
        raise ValueError("rankine_order must be an integer >= 2")
    r_order = int(rankine_order)

    chief_pts = _resolve_chief_points_3d(mesh, chief_points)
    s_body, h_body = _build_target_operators_3d(
        mesh.centroids,
        mesh,
        freq,
        self_interaction=True,
        rankine_order=r_order,
    )
    sys_body = 0.5 * np.eye(mesh.panel_count, dtype=np.complex128) - h_body

    if chief_pts.shape[0] > 0:
        s_chief, h_chief = _build_target_operators_3d(
            chief_pts,
            mesh,
            freq,
            self_interaction=False,
            rankine_order=r_order,
        )
        system_matrix = np.vstack((sys_body, -h_chief))
    else:
        s_chief = np.zeros((0, mesh.panel_count), dtype=np.complex128)
        h_chief = np.zeros((0, mesh.panel_count), dtype=np.complex128)
        system_matrix = sys_body

    cond = float(np.linalg.cond(system_matrix))
    return FreeSurface3DOperators(
        wavenumber=freq,
        single_layer=s_body,
        double_layer=h_body,
        chief_points=chief_pts,
        chief_single_layer=s_chief,
        chief_double_layer=h_chief,
        system_matrix=system_matrix,
        condition_number=cond,
    )


def _solve_system(
    ops: FreeSurface3DOperators, q_ext: ComplexArray
) -> tuple[ComplexArray, float]:
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


def solve_radiation_3d(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    modes: Sequence[int] | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    chief_points: ArrayLike | Literal["auto"] | None = None,
    rankine_order: int = 5,
    operators: FreeSurface3DOperators | None = None,
) -> Radiation3DResult:
    """Solve 3-D wave radiation for up to 6 rigid-body degrees of freedom.

    Modes are indexed ``0..5`` for ``(surge, sway, heave, roll, pitch, yaw)``
    about ``center`` (default ``(0, 0, 0)``). With ``exp(-i*omega*t)`` and unit
    velocity amplitude, ``A_ij + (i/omega)*B_ij = -rho * int_{Gamma_B} phi_j n_i dS``.
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

    ops = (
        operators
        if operators is not None
        else assemble_free_surface_operators_3d(
            mesh, freq, chief_points=chief_points, rankine_order=rankine_order
        )
    )
    if abs(ops.wavenumber - freq) > 1.0e-14 * max(1.0, freq):
        raise ValueError("operators.wavenumber does not match nu")

    n6 = rigid_body_normals_3d(mesh, center=center)
    n_modes = n6[:, mode_arr]
    q_ext = (-n_modes).astype(np.complex128)
    potentials, rel_res = _solve_system(ops, q_ext)

    weighted_normals = mesh.areas[:, None] * n_modes
    impedance = -rho * (weighted_normals.T @ potentials)
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
        chief_points_used=ops.chief_points,
        condition_number=ops.condition_number,
        relative_residual=rel_res,
    )


def incident_airy_wave_3d(
    points: ArrayLike,
    nu: float,
    *,
    amplitude: float = 1.0,
    wave_angle: float = 0.0,
    gravity: float = _DEFAULT_GRAVITY,
) -> tuple[ComplexArray, ComplexArray]:
    """Incident 3-D deep-water Airy wave potential ``phi_I`` and ``nabla phi_I``."""
    freq = _validate_positive("nu", nu)
    amp = _validate_positive("amplitude", amplitude)
    g = _validate_positive("gravity", gravity)
    beta = float(wave_angle)
    if not math.isfinite(beta):
        raise ValueError("wave_angle must be finite")

    pts = np.asarray(points, dtype=float)
    if pts.ndim < 1 or pts.shape[-1] != 3 or not np.isfinite(pts).all():
        raise ValueError("points must be a finite array with last dimension 3")
    if np.any(pts[..., 2] > 1.0e-13):
        raise ValueError("points must satisfy z <= 0")

    omega = math.sqrt(g * freq)
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    phase = np.exp(
        freq * np.minimum(pts[..., 2], 0.0)
        + 1.0j * freq * (pts[..., 0] * cos_b + pts[..., 1] * sin_b)
    )
    phi_inc = -(1.0j * g * amp / omega) * phase
    grad_inc = np.stack(
        (
            omega * amp * cos_b * phase,
            omega * amp * sin_b * phase,
            -1.0j * omega * amp * phase,
        ),
        axis=-1,
    )
    return phi_inc, grad_inc


def solve_diffraction_3d(
    mesh: SurfaceMesh3D,
    nu: float,
    *,
    amplitude: float = 1.0,
    wave_angle: float = 0.0,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    chief_points: ArrayLike | Literal["auto"] | None = None,
    rankine_order: int = 5,
    operators: FreeSurface3DOperators | None = None,
) -> Diffraction3DResult:
    """Solve 3-D regular-wave scattering and compute 6-DOF wave exciting loads."""
    _validate_mesh_3d(mesh)
    freq = _validate_positive("nu", nu)
    amp = _validate_positive("amplitude", amplitude)
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    beta = float(wave_angle)
    omega = math.sqrt(g * freq)

    ops = (
        operators
        if operators is not None
        else assemble_free_surface_operators_3d(
            mesh, freq, chief_points=chief_points, rankine_order=rankine_order
        )
    )
    if abs(ops.wavenumber - freq) > 1.0e-14 * max(1.0, freq):
        raise ValueError("operators.wavenumber does not match nu")

    phi_inc, grad_inc = incident_airy_wave_3d(
        mesh.centroids, freq, amplitude=amp, wave_angle=beta, gravity=g
    )
    dphi_inc_dn = np.einsum("pk,pk->p", grad_inc, mesh.normals)
    phi_scat_2d, rel_res = _solve_system(ops, dphi_inc_dn[:, None])
    phi_scat = phi_scat_2d[:, 0]
    phi_tot = phi_inc + phi_scat

    n6 = rigid_body_normals_3d(mesh, center=center)
    forces = -1.0j * omega * rho * ((mesh.areas[:, None] * n6).T @ phi_tot)

    return Diffraction3DResult(
        total_potential=phi_tot,
        scattered_potential=phi_scat,
        incident_potential=phi_inc,
        incident_normal_velocity=dphi_inc_dn,
        excitation_force=np.asarray(forces, dtype=np.complex128),
        wavenumber=freq,
        omega=omega,
        wave_angle=beta,
        amplitude=amp,
        chief_points_used=ops.chief_points,
        condition_number=ops.condition_number,
        relative_residual=rel_res,
    )


def kochin_function_3d(
    mesh: SurfaceMesh3D,
    potential: ArrayLike,
    normal_velocity_body: ArrayLike,
    nu: float,
    azimuth: ArrayLike,
) -> ComplexArray:
    """Evaluate the 3-D far-field Kochin function ``K(theta)``.

    With ``phi(R, theta, z) ~ (i*nu/2) * H_0^{(1)}(nu*R) * exp(nu*z) * K(theta)``
    as ``R -> inf``, Green's representation yields

        K(theta) = int_{Gamma_B} [-d(phi)/dn_body + nu * phi * (n_z - i*(n_x cos theta + n_y sin theta))]
                   * exp(nu*zeta - i*nu*(xi cos theta + eta sin theta)) dS,

    integrated over each triangle with the same 3-point degree-2 rule used to
    assemble the smooth wave operators.
    """
    _validate_mesh_3d(mesh)
    freq = _validate_positive("nu", nu)
    phi = np.asarray(potential, dtype=np.complex128)
    vn = np.asarray(normal_velocity_body, dtype=np.complex128)
    if phi.shape != (mesh.panel_count,) or vn.shape != (mesh.panel_count,):
        raise ValueError("potential and normal_velocity_body must have shape (panel_count,)")

    theta = np.asarray(azimuth, dtype=float)
    orig_shape = theta.shape
    th_flat = theta.ravel()
    cos_t = np.cos(th_flat)
    sin_t = np.sin(th_flat)

    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    qpts = (
        _BARY_DEG2[None, :, 0, None] * v0[:, None, :]
        + _BARY_DEG2[None, :, 1, None] * v1[:, None, :]
        + _BARY_DEG2[None, :, 2, None] * v2[:, None, :]
    )  # (P, 3, 3)
    phase = np.exp(
        freq * qpts[:, :, 2, None]
        - 1.0j
        * freq
        * (
            qpts[:, :, 0, None] * cos_t[None, None, :]
            + qpts[:, :, 1, None] * sin_t[None, None, :]
        )
    )
    avg_phase = np.mean(phase, axis=1)  # (P, T)

    normals = mesh.normals
    dir_factor = normals[:, 2, None] - 1.0j * (
        normals[:, 0, None] * cos_t[None, :] + normals[:, 1, None] * sin_t[None, :]
    )
    integrand = -vn[:, None] + freq * phi[:, None] * dir_factor
    values = np.sum(mesh.areas[:, None] * avg_phase * integrand, axis=0)
    return values.reshape(orig_shape)


def far_field_damping_matrix_3d(
    mesh: SurfaceMesh3D,
    radiation: Radiation3DResult,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_azimuth: int = 72,
) -> FloatArray:
    """Compute far-field radiation damping ``B_ij^{far}`` from 3-D Kochin functions."""
    _validate_mesh_3d(mesh)
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    if isinstance(n_azimuth, (bool, np.bool_)) or not isinstance(
        n_azimuth, (int, np.integer)
    ) or int(n_azimuth) < 16:
        raise ValueError("n_azimuth must be an integer >= 16")

    n_az = int(n_azimuth)
    freq = radiation.wavenumber
    omega = math.sqrt(g * freq)
    azimuths = (2.0 * math.pi / n_az) * np.arange(n_az, dtype=float)
    dtheta = 2.0 * math.pi / n_az

    m_cnt = radiation.modes.size
    kochin_rows = np.empty((m_cnt, n_az), dtype=np.complex128)
    for idx in range(m_cnt):
        kochin_rows[idx] = kochin_function_3d(
            mesh,
            radiation.potentials[:, idx],
            radiation.normal_velocities[:, idx],
            freq,
            azimuths,
        )

    prefactor = (omega * rho * freq / _FOUR_PI) * dtheta
    hermitian = kochin_rows @ np.conj(kochin_rows.T)
    return np.asarray(prefactor * hermitian.real, dtype=float)


def haskind_excitation_force_3d(
    mesh: SurfaceMesh3D,
    radiation: Radiation3DResult,
    *,
    amplitude: float = 1.0,
    wave_angle: float = 0.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, ComplexArray]:
    """Compute wave exciting loads from radiation potentials via Haskind & Kochin.

    Returns both the near-field boundary Haskind integral
    ``-i*omega*rho * int_{Gamma_B} (phi_I * n_j - phi_j * d(phi_I)/dn_body) dS``
    and the far-field Kochin reciprocity formula ``+rho * g * A * K_j(beta + pi)``
    for every mode present in ``radiation``.
    """
    _validate_mesh_3d(mesh)
    amp = _validate_positive("amplitude", amplitude)
    rho = _validate_positive("density", density)
    g = _validate_positive("gravity", gravity)
    beta = float(wave_angle)
    freq = radiation.wavenumber
    omega = math.sqrt(g * freq)

    phi_inc, grad_inc = incident_airy_wave_3d(
        mesh.centroids, freq, amplitude=amp, wave_angle=beta, gravity=g
    )
    dphi_inc_dn = np.einsum("pk,pk->p", grad_inc, mesh.normals)

    m_cnt = radiation.modes.size
    boundary_forces = np.empty(m_cnt, dtype=np.complex128)
    kochin_forces = np.empty(m_cnt, dtype=np.complex128)
    for idx in range(m_cnt):
        phi_j = radiation.potentials[:, idx]
        n_j = radiation.normal_velocities[:, idx]
        boundary_forces[idx] = (
            -1.0j
            * omega
            * rho
            * np.sum(mesh.areas * (phi_inc * n_j - phi_j * dphi_inc_dn))
        )
        k_back = complex(
            kochin_function_3d(mesh, phi_j, n_j, freq, beta + math.pi)
        )
        kochin_forces[idx] = rho * g * amp * k_back

    return {
        "boundary_haskind": boundary_forces,
        "kochin_haskind": kochin_forces,
    }
