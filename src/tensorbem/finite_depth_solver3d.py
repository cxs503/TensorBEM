"""Three-dimensional finite-depth wave-radiation BEM for closed bodies.

Why a new solver
----------------
``docs/finite_depth3d.md`` §6 recorded the state of the art in this repository
as *"kernel only, no BEM solver yet"*: the John (1950) eigenseries kernel
(:mod:`tensorbem.finite_depth3d`) was verified to machine precision, but nothing
consumed it, so no added mass and no damping could be obtained in water of
finite depth in three dimensions.  This module is that consumer.  It is the 3-D
counterpart of :mod:`tensorbem.finite_depth_solver` (the 2-D section solver,
which uses the *indirect* single-layer form because a 2-D radiated wave does not
decay).

Formulation
-----------
In three dimensions both the Rankine kernel ``1/(4*pi*r)`` and the outgoing
cylindrical wave ``O(R**(-1/2))`` decay at infinity, so Green's third identity
closes without an auxiliary surface and the *direct* collocation form used by
the deep-water solver :mod:`tensorbem.free_surface_solver3d` carries over
verbatim -- only the kernel changes::

    (1/2 I - H_nu,h) phi = S_nu,h q_e,      q_e = -d(phi)/dn_body

with ``S`` the single-layer and ``H`` the source-normal double-layer operator of
:func:`tensorbem.finite_depth3d.green_function_3d` at ``nu = omega**2/g`` and
depth ``h``.  For unit velocity amplitude in mode ``j``

    A_ij + (i/omega) B_ij = -rho * int_{Gamma_B} phi_j n_i dS.

Scope: closed, fully submerged bodies
-------------------------------------
The kernel satisfies the free-surface and sea-bed conditions by construction, so
the body is the only surface to discretise -- no waterplane lid, and therefore no
interior-waterplane eigenproblem and no irregular frequencies (John 1950; Hulme
1982) to suppress.  That is exactly why this increment can ship before the
surface-piercing case: a surface-piercing hull in finite depth needs the
CHIEF/lid treatment re-derived against the *strip* interior eigenproblem, which
is left as the documented next step (``docs/finite_depth_solver3d.md``).  Meshes
that reach the free surface or the sea bed are rejected rather than silently
mis-integrated.

Singularity split
-----------------
``G_fd`` carries the same ``1/(4*pi*r)`` singularity as the deep-water kernel, so
the split is the deep-water one::

    G_fd = 1/(4*pi*r) + G_smooth

The Rankine part is integrated in closed form on every triangle (Duffy polar
rule, with the exact self-panel integral) and only ``G_smooth`` is quadratured.
``H`` is the exact Van Oosterom-Strackee solid angle plus the quadratured
remainder of ``dG/dn_source``, the analytic source gradient coming from
:func:`tensorbem.finite_depth3d.green_gradient_source_3d`.

Mode truncation
---------------
The eigenseries is truncated where the omitted evanescent modes have decayed by
``exp(-_MODE_DECAY)`` over the *closest target/quadrature-point separation*, not
over the closest panel centroid; the degree-4 rule used here has no point at the
panel centroid, which also keeps the mode count (and hence the cost) down.
``mode_count`` may be pinned explicitly for truncation studies.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .finite_depth3d import green_function_3d, green_gradient_source_3d
from .free_surface_solver3d import (
    _validate_mesh_3d,
    _vectorized_single_layer_rankine,
    _vectorized_solid_angle_dl,
    rigid_body_normals_3d,
)
from .geometry3d import SurfaceMesh3D
from .kernels3d import (
    integrate_single_layer_triangle_self,
    integrate_single_layer_triangle_self_batch,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_FOUR_PI = 4.0 * math.pi
_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0

#: Exponent of the omitted evanescent tail at the nearest target/quad pair.
_MODE_DECAY = 10.0
#: Hard cap so a pathological mesh cannot ask for a multi-million-mode operator.
_MODES_CAP = 4096
#: Element budget of one row block of the quadratured remainder.
_BLOCK_ELEMENTS = 65536

# Six-point, degree-four symmetric triangle rule (Dunavant).  Unlike the
# degree-2 rule used by the deep-water solver it has no point at the centroid,
# so the self-panel remainder is never evaluated at ``r = 0``.
_DEG4_BARY = np.array(
    [
        [0.816847572980459, 0.091576213509771, 0.091576213509771],
        [0.091576213509771, 0.816847572980459, 0.091576213509771],
        [0.091576213509771, 0.091576213509771, 0.816847572980459],
        [0.108103018168070, 0.445948490915965, 0.445948490915965],
        [0.445948490915965, 0.108103018168070, 0.445948490915965],
        [0.445948490915965, 0.445948490915965, 0.108103018168070],
    ],
    dtype=float,
)
_DEG4_WEIGHTS = np.array(
    [
        0.109951743655322,
        0.109951743655322,
        0.109951743655322,
        0.223381589678011,
        0.223381589678011,
        0.223381589678011,
    ],
    dtype=float,
)


@dataclass
class FiniteDepth3DOperators:
    """Boundary-element operators of the finite-depth 3-D radiation problem."""

    wavenumber: float
    depth: float
    mode_count: int
    single_layer: ComplexArray
    double_layer: ComplexArray
    system_matrix: ComplexArray
    condition_number: float
    kernel: str = "eigenseries"
    #: Rows whose per-row mode requirement (``_modes_for`` at the nearest
    #: target/quadrature separation) exceeded ``mode_count``, i.e. rows the cap
    #: truncated below the requested ``exp(-_MODE_DECAY)`` tail.  Non-zero means
    #: the operator is *under-resolved* by construction and the result is a
    #: truncation study artefact rather than a converged solve.
    mode_cap_bound_rows: int = 0
    #: Largest ``requirement / mode_count`` ratio over the rows (1.0 = no cap).
    mode_shortfall: float = 1.0


@dataclass
class FiniteDepth3DResult:
    """Added mass / damping of a closed body in water of finite depth.

    Attribute names match
    :class:`tensorbem.free_surface_solver3d.Radiation3DResult` so downstream
    consumers can treat the two interchangeably.
    """

    modes: NDArray[np.int64]
    potentials: ComplexArray
    normal_velocities: FloatArray
    added_mass: FloatArray
    damping: FloatArray
    wavenumber: float
    omega: float
    depth: float
    mode_count: int
    condition_number: float
    relative_residual: float


def _positive(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _validate_depth_mesh(mesh: SurfaceMesh3D, depth: float) -> float:
    h = _positive("depth", depth)
    _validate_mesh_3d(mesh)
    if np.any(mesh.vertices[:, 2] < -h - 1.0e-9):
        raise ValueError(
            "every mesh vertex must lie inside the fluid, z >= -depth "
            f"(deepest vertex z = {float(mesh.vertices[:, 2].min()):.6g}, depth = {h:.6g})"
        )
    return h


def _resolve_panel_areas(
    mesh: SurfaceMesh3D, panel_areas: ArrayLike | None
) -> FloatArray:
    """Return the surface measure attached to every panel.

    ``None`` keeps the flat-triangle areas of the mesh.  A supplied vector must be
    positive and of length ``mesh.panel_count``; it then replaces the flat measure
    in every integral that carries a weight (Rankine quadrature, eigenseries
    remainder quadrature and the force/moment integral), while the panels, their
    normals and the collocation points stay unchanged.  That is the
    exact-surface-measure route of
    :func:`tensorbem.geometry3d.spherical_patch_areas`, with the same semantics as
    the argument of the same name in
    :func:`tensorbem.free_surface_solver3d.solve_radiation_3d`.
    """
    if panel_areas is None:
        return mesh.areas
    areas = np.asarray(panel_areas, dtype=float)
    if areas.shape != (mesh.panel_count,):
        raise ValueError(
            f"panel_areas must have shape ({mesh.panel_count},), got {areas.shape}"
        )
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("panel_areas must be finite and strictly positive")
    return areas


def quadrature_points(
    mesh: SurfaceMesh3D, areas: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(points, weights, normals)`` of the 6-point triangle rule.

    ``weights`` already include the panel measure, so ``sum(weights)`` equals the
    wetted area; entries are ordered panel-major (panel 0 rule points first).
    ``areas`` overrides the flat-triangle measure, see
    :func:`_resolve_panel_areas`.
    """
    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    bary = _DEG4_BARY
    pts = (
        bary[None, :, 0, None] * v0[:, None, :]
        + bary[None, :, 1, None] * v1[:, None, :]
        + bary[None, :, 2, None] * v2[:, None, :]
    ).reshape(-1, 3)
    measure = _resolve_panel_areas(mesh, areas)
    weights = (measure[:, None] * _DEG4_WEIGHTS[None, :]).reshape(-1)
    normals = np.repeat(mesh.normals, bary.shape[0], axis=0)
    return pts, weights, normals


def _modes_for(depth: float, r_min: float) -> int:
    spacing = math.pi * max(float(r_min), 1.0e-9)
    wanted = int(math.ceil(_MODE_DECAY * depth / spacing)) + 2
    return int(min(max(wanted, 8), _MODES_CAP))


def _row_mode_requirements(mesh: SurfaceMesh3D, depth: float) -> NDArray[np.int64]:
    """Per-row evanescent mode requirement of the quadratured remainder.

    One entry per panel row: the mode count ``_modes_for`` asks for at that
    row's smallest target/quadrature *horizontal* separation.  The near field
    (small separation) needs thousands of modes on a 320-panel sphere, so a
    pinned ``mode_count`` below this vector silently truncates the operator.
    """
    qpts, _, _ = quadrature_points(mesh)
    horiz = np.hypot(
        mesh.centroids[:, None, 0] - qpts[None, :, 0],
        mesh.centroids[:, None, 1] - qpts[None, :, 1],
    )
    with np.errstate(invalid="ignore"):
        row_min = np.where(horiz > 0.0, horiz, np.inf).min(axis=1)
    return np.array([_modes_for(depth, r) for r in row_min], dtype=np.int64)


def mode_count_for(mesh: SurfaceMesh3D, depth: float) -> int:
    """Evanescent modes needed for an ``exp(-_MODE_DECAY)`` tail at the nearest pair."""
    qpts, _, _ = quadrature_points(mesh)
    horiz = np.hypot(
        mesh.centroids[:, None, 0] - qpts[None, :, 0],
        mesh.centroids[:, None, 1] - qpts[None, :, 1],
    )
    positive = horiz[horiz > 0.0]
    r_min = float(positive.min()) if positive.size else 1.0
    return _modes_for(depth, r_min)


def _rankine_operators(
    targets: FloatArray,
    mesh: SurfaceMesh3D,
    *,
    self_interaction: bool,
    order: int,
    areas: ArrayLike | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Exact single-layer / solid-angle operators of ``1/(4*pi*r)``."""
    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    single = _vectorized_single_layer_rankine(
        targets, v0, v1, v2, _resolve_panel_areas(mesh, areas),
        self_diagonal=False, order=order,
    ).astype(np.complex128)
    double = _vectorized_solid_angle_dl(targets, v0, v1, v2).astype(np.complex128)
    if self_interaction:
        # Only a row/column pair whose target *is* the panel centroid needs the
        # exact self integral; _vectorized_solid_angle_dl already returns zero
        # for a coplanar (hence self) triangle.
        diagonal = integrate_single_layer_triangle_self_batch(
            np.stack((v0, v1, v2), axis=1)
        )
        for idx in range(min(mesh.panel_count, targets.shape[0])):
            if np.array_equal(targets[idx], mesh.centroids[idx]):
                single[idx, idx] = diagonal[idx]
    return single, double


#: Free-surface kernels the solver can be assembled on.
KERNELS: tuple[str, ...] = ("eigenseries", "quadrature")


def kernel_functions(
    kernel: str, nu: float, depth: float, mode_count: int, *, nodes: int = 16
):
    """``(value, gradient)`` callables of the chosen free-surface kernel.

    ``"eigenseries"`` is the John (1950) modal kernel of
    :mod:`tensorbem.finite_depth3d` -- the package default.  ``"quadrature"``
    is the independent wavenumber-integral kernel of
    :mod:`tensorbem.finite_depth3d_quadrature`, which is a different
    representation of the same boundary-value problem; running the solver on it
    is an end-to-end check that a kernel defect shows up in added mass and
    damping, not only in a pointwise kernel test.
    """
    if kernel == "eigenseries":

        def value(tgt, src):
            return green_function_3d(tgt, src, nu, depth, modes=mode_count)

        def gradient(tgt, src):
            return green_gradient_source_3d(tgt, src, nu, depth, modes=mode_count)

        return value, gradient
    if kernel == "quadrature":
        from .finite_depth3d_quadrature import (
            green_function_finite_depth_3d,
            green_gradient_source_finite_depth_3d,
        )

        def value(tgt, src):
            return green_function_finite_depth_3d(tgt, src, nu, depth, nodes=nodes)

        def gradient(tgt, src):
            return green_gradient_source_finite_depth_3d(
                tgt, src, nu, depth, nodes=nodes
            )

        return value, gradient
    raise ValueError(f"kernel must be one of {KERNELS}")


def _smooth_remainder(
    targets: FloatArray,
    mesh: SurfaceMesh3D,
    nu: float,
    depth: float,
    mode_count: int,
    *,
    order: int,
    kernel: str = "eigenseries",
    kernel_nodes: int = 16,
    group_rows: int = 8,
    areas: ArrayLike | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Quadratured single- and double-layer operators of ``G_fd - 1/(4*pi*r)``.

    The exact Rankine operators are added in closed form; the eigenseries
    remainder is quadratured with the 6-point triangle rule.  The evanescent tail
    decays with the *horizontal* target-source separation, so rows are grouped by
    how many modes they actually need (``_modes_for``) instead of paying the
    diagonal-row mode count everywhere; ``mode_count`` stays the hard cap, which
    keeps the truncation study reproducible.
    """
    p_cnt = mesh.panel_count
    m_cnt = targets.shape[0]
    if m_cnt == 0:
        empty = np.zeros((0, p_cnt), dtype=np.complex128)
        return empty, empty

    qpts, qweights, qnormals = quadrature_points(mesh, areas)
    per_panel = _DEG4_BARY.shape[0]

    single, double = _rankine_operators(
        targets, mesh, self_interaction=True, order=order, areas=areas
    )

    value_fn, grad_fn = kernel_functions(
        kernel, nu, depth, mode_count, nodes=kernel_nodes
    )
    if kernel == "eigenseries":
        # The eigenseries cost is set by the per-row evanescent mode count, so
        # rows are grouped by how many modes they need.  The quadrature kernel
        # costs one wave integral per (target, unique source depth) pair, which
        # grouping cannot change, so all rows go in one block.
        horiz = np.hypot(
            targets[:, None, 0] - qpts[None, :, 0],
            targets[:, None, 1] - qpts[None, :, 1],
        )
        with np.errstate(invalid="ignore"):
            row_min = np.where(horiz > 0.0, horiz, np.inf).min(axis=1)
        row_modes = np.array(
            [min(mode_count, _modes_for(depth, r)) for r in row_min], dtype=np.int64
        )
        order_rows = np.argsort(-row_modes, kind="stable")
        step = max(1, int(group_rows))
        blocks = [
            order_rows[start : start + step] for start in range(0, m_cnt, step)
        ]
    else:
        row_modes = np.full(m_cnt, int(mode_count), dtype=np.int64)
        blocks = np.arange(m_cnt)[None, :].tolist()

    n_qp = int(qpts.shape[0])
    for rows in blocks:
        rows = np.asarray(rows, dtype=np.int64)
        # Every row of a block is evaluated with the *same* mode count (the
        # largest requirement inside it), so the block must be small enough that
        # ``rows * n_qp * modes`` stays inside the element budget: without this
        # the temporaries grow as ``group_rows * n_qp * mode_count`` and a
        # converged (large) mode count exhausts memory.
        block_modes = int(row_modes[rows].max())
        per_block = max(1, _BLOCK_ELEMENTS // max(1, n_qp * block_modes))
        for start in range(0, rows.size, per_block):
            _quadrature_remainder_block(
                rows[start : start + per_block], targets, qpts, qweights, qnormals,
                per_panel, single, double, nu, depth, block_modes, kernel,
                kernel_nodes,
            )
    return single, double


def _quadrature_remainder_block(
    rows: NDArray[np.int64],
    targets: FloatArray,
    qpts: FloatArray,
    qweights: FloatArray,
    qnormals: FloatArray,
    per_panel: int,
    single: ComplexArray,
    double: ComplexArray,
    nu: float,
    depth: float,
    block_modes: int,
    kernel: str,
    kernel_nodes: int,
) -> None:
    """Add one row block of the quadratured ``G_fd - 1/(4*pi*r)`` operators."""
    value_fn, grad_fn = kernel_functions(
        kernel, nu, depth, block_modes, nodes=kernel_nodes
    )
    tgt = targets[rows][:, None, :]
    src = qpts[None, :, :]

    diff = tgt - src
    r = np.maximum(np.linalg.norm(diff, axis=2), 1.0e-300)
    inv = 1.0 / (_FOUR_PI * r)

    value = value_fn(tgt, src)
    single[rows] += ((value - inv) * qweights[None, :]).reshape(
        rows.size, -1, per_panel
    ).sum(axis=2)

    grad = grad_fn(tgt, src)
    dn_wave = np.einsum("mqk,qk->mq", grad, qnormals)
    dn_rankine = np.einsum("mqk,qk->mq", diff, qnormals) / (_FOUR_PI * r**3)
    double[rows] += ((dn_wave - dn_rankine) * qweights[None, :]).reshape(
        rows.size, -1, per_panel
    ).sum(axis=2)


def assemble_finite_depth_operators_3d(
    mesh: SurfaceMesh3D,
    nu: float,
    depth: float,
    *,
    mode_count: int | None = None,
    rankine_order: int = 5,
    kernel: str = "eigenseries",
    kernel_nodes: int = 16,
    panel_areas: ArrayLike | None = None,
    warn_on_truncation: bool = False,
) -> FiniteDepth3DOperators:
    """Assemble the finite-depth 3-D single/double-layer operators and system.

    ``kernel`` selects the free-surface Green function: ``"eigenseries"``
    (default, the John modal kernel) or ``"quadrature"`` (the independent
    wavenumber-integral kernel).  The quadrature kernel needs strictly submerged
    panels, because its wave integral requires ``z + zeta < 0``;
    ``kernel_nodes`` is its Gauss-Legendre node count per wavenumber panel
    (ignored by the eigenseries kernel).
    """
    h = _validate_depth_mesh(mesh, depth)
    freq = _positive("nu", nu)
    if (
        isinstance(rankine_order, (bool, np.bool_))
        or not isinstance(rankine_order, (int, np.integer))
        or int(rankine_order) < 2
    ):
        raise ValueError("rankine_order must be an integer >= 2")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    if (
        isinstance(kernel_nodes, (bool, np.bool_))
        or not isinstance(kernel_nodes, (int, np.integer))
        or int(kernel_nodes) < 4
    ):
        raise ValueError("kernel_nodes must be an integer >= 4")
    if kernel == "quadrature":
        qpts, _, _ = quadrature_points(mesh)
        highest = max(float(mesh.vertices[..., 2].max()), float(qpts[:, 2].max()))
        if highest >= 0.0:
            raise ValueError(
                "kernel='quadrature' requires strictly submerged panels "
                f"(found z = {highest:.3g} on the waterline); "
                "use kernel='eigenseries' for surface-piercing bodies"
            )
    count = mode_count_for(mesh, h) if mode_count is None else int(mode_count)
    if count < 1:
        raise ValueError("mode_count must be >= 1")

    single, double = _smooth_remainder(
        mesh.centroids, mesh, freq, h, count,
        order=int(rankine_order), kernel=kernel, kernel_nodes=int(kernel_nodes),
        areas=panel_areas,
    )
    system = 0.5 * np.eye(mesh.panel_count, dtype=np.complex128) - double
    need = _row_mode_requirements(mesh, h)
    bound = int(np.count_nonzero(need > count))
    shortfall = float(need.max() / count) if count else 1.0
    if bound and warn_on_truncation:
        warnings.warn(
            f"mode_count={count} truncates {bound}/{need.size} rows below their "
            f"e^{-_MODE_DECAY:.0f} tail requirement (max requirement {need.max()}); "
            "the operator is under-resolved by construction",
            RuntimeWarning,
            stacklevel=2,
        )
    return FiniteDepth3DOperators(
        wavenumber=freq,
        depth=h,
        mode_count=count,
        single_layer=single,
        double_layer=double,
        system_matrix=system,
        condition_number=float(np.linalg.cond(system)),
        kernel=kernel,
        mode_cap_bound_rows=bound,
        mode_shortfall=shortfall,
    )


def solve_radiation_finite_depth_3d(
    mesh: SurfaceMesh3D,
    nu: float,
    depth: float,
    *,
    modes: Sequence[int] | None = None,
    mode_count: int | None = None,
    center: ArrayLike | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    rankine_order: int = 5,
    kernel: str = "eigenseries",
    kernel_nodes: int = 16,
    operators: FiniteDepth3DOperators | None = None,
    panel_areas: ArrayLike | None = None,
) -> FiniteDepth3DResult:
    """Solve 3-D wave radiation in water of finite depth for up to 6 DOF.

    Modes are ``0..5`` = ``(surge, sway, heave, roll, pitch, yaw)`` about
    ``center`` (default ``(0, 0, 0)``).  With ``exp(-i*omega*t)`` and unit
    velocity amplitude, ``A_ij + (i/omega)*B_ij = -rho * int phi_j n_i dS``.
    """
    h = _validate_depth_mesh(mesh, depth)
    freq = _positive("nu", nu)
    rho = _positive("density", density)
    g = _positive("gravity", gravity)
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
        else assemble_finite_depth_operators_3d(
            mesh, freq, h, mode_count=mode_count, rankine_order=rankine_order,
            kernel=kernel, kernel_nodes=kernel_nodes, panel_areas=panel_areas,
        )
    )
    if abs(ops.wavenumber - freq) > 1.0e-14 * max(1.0, freq):
        raise ValueError("operators.wavenumber does not match nu")
    if abs(ops.depth - h) > 1.0e-12 * max(1.0, h):
        raise ValueError("operators.depth does not match depth")

    n6 = rigid_body_normals_3d(mesh, center=center)
    n_modes = n6[:, mode_arr]
    q_ext = (-n_modes).astype(np.complex128)

    rhs = ops.single_layer @ q_ext
    solution = np.linalg.solve(ops.system_matrix, rhs)
    denom = max(float(np.linalg.norm(rhs)), 1.0e-30)
    residual = float(np.linalg.norm(ops.system_matrix @ solution - rhs) / denom)

    weighted_normals = _resolve_panel_areas(mesh, panel_areas)[:, None] * n_modes
    impedance = -rho * (weighted_normals.T @ solution)
    return FiniteDepth3DResult(
        modes=mode_arr,
        potentials=np.asarray(solution, dtype=np.complex128),
        normal_velocities=n_modes,
        added_mass=np.asarray(impedance.real, dtype=float),
        damping=np.asarray(omega * impedance.imag, dtype=float),
        wavenumber=freq,
        omega=omega,
        depth=h,
        mode_count=ops.mode_count,
        condition_number=ops.condition_number,
        relative_residual=residual,
    )


def gravity_wave_wavenumber(nu: float, depth: float) -> float:
    """Propagating wavenumber ``k0`` of the strip, re-exported for convenience."""
    from .finite_depth3d import finite_depth_wavenumber

    return finite_depth_wavenumber(_positive("nu", nu), _positive("depth", depth))


__all__ = [
    "KERNELS",
    "FiniteDepth3DOperators",
    "FiniteDepth3DResult",
    "assemble_finite_depth_operators_3d",
    "gravity_wave_wavenumber",
    "kernel_functions",
    "mode_count_for",
    "quadrature_points",
    "solve_radiation_finite_depth_3d",
]