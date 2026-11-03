"""Mixed Dirichlet/Neumann boundary conditions for the 2-D Laplace panel solver.

:mod:`tensorbem.solver` solves the *all-Dirichlet* interior problem and the
exterior Dirichlet/Neumann problems.  Many verification cases (a square
conductor with insulated sides, a linearised free surface that is kinematic in
one region and dynamic in another, patchy Dirichlet data on a ship section)
need the two boundary-condition types *on the same closed curve*, which the
all-Dirichlet formulation cannot express.  :func:`solve_laplace_mixed` closes
that gap with one square system, without new kernels and without touching the
Dirichlet path.

Formulation
-----------
Constant panels, midpoint collocation, analytically integrated kernels.  At
every collocation point the interior direct boundary integral identity is

    sum_j S_ij q_j = o_i u_i + sum_j H_ij u_j,     o_i = 1/2,

with ``S`` the single-layer and ``H`` the double-layer matrix returned by
:func:`tensorbem.solver.assemble_matrices`.  There is one equation per panel,
but *which* trace is unknown depends on that panel's boundary condition, so the
mixed problem is a single ``n x n`` system in the ``n`` unknowns
``{q_j : j in D} union {u_k : k in N}``:

    + sum_{j in D} S_ij q_j  - sum_{k in N} (o_i delta_ik + H_ik) u_k
        = sum_{j in D} (o_i delta_ij + H_ij) g_j  -  sum_{k in N} S_ik qhat_k,

where ``g`` are the prescribed Dirichlet values and ``qhat`` the prescribed
normal derivatives along the polygon's outward normal.  The kernel matrices are
reused verbatim, so with ``N`` empty the system is algebraically identical to
:func:`tensorbem.solver.solve_laplace_dirichlet` (asserted by the test suite).

Pure Neumann
------------
With ``D`` empty the system is rank deficient by one: the constant is a null
vector of ``o I + H`` on a closed curve, which mirrors the physical
non-uniqueness of ``u`` up to an additive constant.  ``pure_neumann_gauge =
"mean_zero"`` (default) replaces the last collocation row with the discrete
mean constraint ``mean(u) = 0`` over panel midpoints, so the returned trace is
the unique zero-mean representative of the solution set; ``"raise"`` refuses to
solve instead of silently returning an arbitrary member.  The compatibility
condition ``oint qhat ds = 0`` is *not* enforced on input: an incompatible
Neumann data set makes the system inconsistent, and the reported
``relative_residual_l2`` exposes it rather than hiding it.

Scope
-----
Interior (bounded) domains only, smooth-panel midpoint collocation, and no
corner-specific free-term correction: the free term is ``1/2`` at every
collocation point because collocation points are panel midpoints of straight
panels.  Panel-constant boundary data limits the trace accuracy to
``O(h)`` pointwise / ``O(h^2)`` for midpoint superconvergent quantities; see
``docs/mixed_boundary_conditions.md`` for the measured orders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh
from .kernels import (
    REFERENCE_LENGTH,
    integrate_double_layer_panel,
    integrate_single_layer_panel,
    validate_reference_length,
)
from .solver import assemble_matrices

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
_MAX_CONDITION = 1.0e13
_GAUGES = ("mean_zero", "raise")


@dataclass(frozen=True)
class MixedBoundary:
    """Panel-wise Dirichlet and Neumann data for :func:`solve_laplace_mixed`.

    ``dirichlet_panels`` / ``neumann_panels`` are disjoint index arrays that
    together cover every panel of the mesh exactly once, so no panel is left
    without a boundary condition.
    """

    dirichlet_panels: IntArray
    dirichlet_values: FloatArray
    neumann_panels: IntArray
    neumann_values: FloatArray


@dataclass(frozen=True)
class LaplaceMixedResult:
    """Boundary traces and diagnostics of a mixed boundary-condition solve."""

    boundary_values: FloatArray
    normal_derivative: FloatArray
    dirichlet_panels: IntArray
    neumann_panels: IntArray
    pure_neumann_gauge: str | None
    condition_number: float
    residual_l2: float
    relative_residual_l2: float
    reference_length: float = REFERENCE_LENGTH
    domain: str = "interior"

    @property
    def is_pure_neumann(self) -> bool:
        """True when the problem had no Dirichlet panel (gauge applied)."""
        return self.dirichlet_panels.size == 0


def _as_index_array(name: str, values: object) -> IntArray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.empty(0, dtype=np.int_)
    if arr.dtype == bool or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be an integer index array")
    flat = arr.ravel()
    if flat.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return flat.astype(np.int_, copy=False)


def _as_value_array(name: str, values: object) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.empty(0, dtype=float)
    flat = arr.ravel()
    if not np.isfinite(flat).all():
        raise ValueError(f"{name} must all be finite")
    return flat


def _validate_boundary(
    mesh: PanelMesh, boundary: MixedBoundary
) -> tuple[IntArray, FloatArray, IntArray, FloatArray]:
    if not isinstance(boundary, MixedBoundary):
        raise TypeError("boundary must be a MixedBoundary instance")
    n = mesh.panel_count
    d_idx = _as_index_array("dirichlet_panels", boundary.dirichlet_panels)
    n_idx = _as_index_array("neumann_panels", boundary.neumann_panels)
    d_val = _as_value_array("dirichlet_values", boundary.dirichlet_values)
    n_val = _as_value_array("neumann_values", boundary.neumann_values)
    if d_val.size != d_idx.size:
        raise ValueError(
            "dirichlet_values must have one entry per dirichlet panel "
            f"({d_idx.size}); got {d_val.size}"
        )
    if n_val.size != n_idx.size:
        raise ValueError(
            "neumann_values must have one entry per neumann panel "
            f"({n_idx.size}); got {n_val.size}"
        )
    for name, idx in (("dirichlet_panels", d_idx), ("neumann_panels", n_idx)):
        if idx.size == 0:
            continue
        if idx.min() < 0 or idx.max() >= n:
            raise ValueError(f"{name} entries must lie in [0, {n - 1}]")
        if np.unique(idx).size != idx.size:
            raise ValueError(f"{name} contains duplicate panel indices")
    if d_idx.size and n_idx.size:
        shared = np.intersect1d(d_idx, n_idx)
        if shared.size:
            raise ValueError(
                "a panel cannot be both Dirichlet and Neumann; overlap at "
                f"{shared.tolist()}"
            )
    covered = np.concatenate((d_idx, n_idx))
    if covered.size != n:
        missing = np.setdiff1d(np.arange(n), covered)
        raise ValueError(
            "every panel needs exactly one boundary condition; unassigned "
            f"panels: {missing.tolist()}"
        )
    return d_idx, d_val, n_idx, n_val


def solve_laplace_mixed(
    mesh: PanelMesh,
    boundary: MixedBoundary,
    reference_length: float = REFERENCE_LENGTH,
    pure_neumann_gauge: str = "mean_zero",
) -> LaplaceMixedResult:
    """Solve the interior 2-D Laplace problem with per-panel Dirichlet/Neumann data.

    Parameters
    ----------
    mesh:
        Closed counter-clockwise :class:`~tensorbem.geometry.PanelMesh`.
    boundary:
        :class:`MixedBoundary` covering every panel exactly once.
    reference_length:
        Kernel reference length passed through to
        :func:`tensorbem.solver.assemble_matrices`.
    pure_neumann_gauge:
        ``"mean_zero"`` (default) or ``"raise"``; only consulted when no panel
        carries a Dirichlet condition.
    """
    if pure_neumann_gauge not in _GAUGES:
        raise ValueError(f"pure_neumann_gauge must be one of {_GAUGES}")
    d_idx, d_val, n_idx, n_val = _validate_boundary(mesh, boundary)
    reference = validate_reference_length(reference_length)
    single, double = assemble_matrices(mesh, reference_length=reference)

    n = mesh.panel_count
    total = 0.5 * np.eye(n) + double
    # RHS: known Dirichlet data acts through the double-layer row, known
    # Neumann data through the single-layer row (moved to the right-hand side).
    rhs = total[:, d_idx] @ d_val - single[:, n_idx] @ n_val
    system = np.empty((n, n), dtype=float)
    system[:, : d_idx.size] = single[:, d_idx]
    system[:, d_idx.size :] = -total[:, n_idx]

    gauge: str | None = None
    if d_idx.size == 0:
        if pure_neumann_gauge == "raise":
            raise ValueError(
                "the problem has no Dirichlet panel: the interior Neumann "
                "problem determines u only up to an additive constant; pass "
                "pure_neumann_gauge='mean_zero' to fix the gauge"
            )
        system[-1, :] = 1.0
        rhs[-1] = 0.0
        gauge = "mean_zero"

    condition = float(np.linalg.cond(system))
    if not math.isfinite(condition) or condition > _MAX_CONDITION:
        raise ValueError(
            "the mixed-boundary system is numerically singular "
            f"(condition number {condition:.3e})"
        )
    solution = np.linalg.solve(system, rhs)
    residual = float(np.linalg.norm(system @ solution - rhs))
    rhs_norm = float(np.linalg.norm(rhs))

    normal_derivative = np.empty(n, dtype=float)
    boundary_values = np.empty(n, dtype=float)
    normal_derivative[d_idx] = solution[: d_idx.size]
    boundary_values[n_idx] = solution[d_idx.size :]
    boundary_values[d_idx] = d_val
    normal_derivative[n_idx] = n_val

    return LaplaceMixedResult(
        boundary_values=boundary_values,
        normal_derivative=normal_derivative,
        dirichlet_panels=d_idx,
        neumann_panels=n_idx,
        pure_neumann_gauge=gauge,
        condition_number=condition,
        residual_l2=residual,
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0.0 else residual,
        reference_length=reference,
    )


def evaluate_mixed_interior_field(
    mesh: PanelMesh, result: LaplaceMixedResult, points: FloatArray
) -> FloatArray:
    """Evaluate the harmonic potential of a mixed solve at interior points.

    Uses the same exactly integrated panel kernels as
    :func:`tensorbem.solver.evaluate_interior_field`; the Green representation
    only needs the two boundary traces, which both boundary-condition types
    supply after the solve has filled in the unknown one.
    """
    if not isinstance(result, LaplaceMixedResult):
        raise TypeError("result must be a LaplaceMixedResult")
    if result.domain != "interior":
        raise ValueError("evaluate_mixed_interior_field supports interior results only")
    if (
        result.normal_derivative.shape != (mesh.panel_count,)
        or result.boundary_values.shape != (mesh.panel_count,)
    ):
        raise ValueError("result boundary traces must match the mesh panel count")
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")

    lengths = mesh.lengths
    tangents = mesh.vectors / lengths[:, None]
    normals = mesh.normals
    reference = result.reference_length
    values = np.zeros(len(targets), dtype=float)
    for i, target in enumerate(targets):
        for j in range(mesh.panel_count):
            single = integrate_single_layer_panel(
                target, mesh.vertices[j], tangents[j], float(lengths[j]), reference
            )
            double = integrate_double_layer_panel(
                target,
                mesh.vertices[j],
                tangents[j],
                normals[j],
                float(lengths[j]),
            )
            values[i] += (
                single * result.normal_derivative[j]
                - double * result.boundary_values[j]
            )
    return values