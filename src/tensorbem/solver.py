"""Constant-panel collocation solver for the 2-D Laplace Dirichlet problem.

Boundary equation at a smooth collocation point:
    c*u = integral(G*q ds) - integral((dG/dn_y)*u ds)
with G = -(1/(2*pi))*log(r/L), -Delta(G)=delta, and c=1/2.
Thus Gmat*q = (0.5*I + Hmat)*u for a bounded domain, and
Gmat*q_e = (0.5*I - Hmat)*u for the exterior domain, where q_e is the
derivative along the fluid-domain normal (opposite the outward normal).

All straight-panel matrix entries are integrated analytically. This removes
the off-diagonal midpoint-quadrature error and improves near-panel accuracy.
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

FloatArray = NDArray[np.float64]
_DOMAINS = ("interior", "exterior")
# Beyond this condition number the dense solve is no longer meaningful in
# double precision; it is the logarithmic-capacity degeneracy in practice.
_MAX_CONDITION = 1.0e13


@dataclass(frozen=True)
class LaplaceDirichletResult:
    """Solution and matrices from a constant-panel collocation solve."""

    normal_derivative: FloatArray
    boundary_values: FloatArray
    single_layer_matrix: FloatArray
    double_layer_matrix: FloatArray
    condition_number: float
    residual_l2: float
    relative_residual_l2: float = 0.0
    domain: str = "interior"
    reference_length: float = REFERENCE_LENGTH


def _validate_domain(domain: str) -> str:
    if domain not in _DOMAINS:
        raise ValueError(f"domain must be one of {_DOMAINS}")
    return domain


def assemble_matrices(
    mesh: PanelMesh, reference_length: float = REFERENCE_LENGTH
) -> tuple[FloatArray, FloatArray]:
    """Assemble analytically integrated single- and double-layer matrices."""
    reference = validate_reference_length(reference_length)
    centers = mesh.centers
    starts = mesh.vertices
    tangents = mesh.vectors / mesh.lengths[:, None]
    normals = mesh.normals
    lengths = mesh.lengths
    n = mesh.panel_count
    single = np.empty((n, n), dtype=float)
    double = np.empty((n, n), dtype=float)

    for i, target in enumerate(centers):
        for j in range(n):
            single[i, j] = integrate_single_layer_panel(
                target, starts[j], tangents[j], float(lengths[j]), reference
            )
            double[i, j] = integrate_double_layer_panel(
                target, starts[j], tangents[j], normals[j], float(lengths[j])
            )
    return single, double


def _solve(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    domain: str,
    reference_length: float,
) -> LaplaceDirichletResult:
    values = np.asarray(boundary_values, dtype=float)
    n = mesh.panel_count
    if values.shape != (n,):
        raise ValueError(f"boundary_values must have shape ({n},)")
    if not np.isfinite(values).all():
        raise ValueError("boundary_values must all be finite")
    reference = validate_reference_length(reference_length)

    single, double = assemble_matrices(mesh, reference_length=reference)
    sign = 1.0 if domain == "interior" else -1.0
    rhs = 0.5 * values + sign * (double @ values)
    condition = float(np.linalg.cond(single))
    if not math.isfinite(condition) or condition > _MAX_CONDITION:
        raise ValueError(
            "the single-layer matrix is numerically singular "
            f"(condition number {condition:.3e}); for a closed 2-D boundary this "
            "happens when its logarithmic capacity approaches reference_length "
            f"(currently {reference:g}). Pass a different reference_length."
        )
    q = np.linalg.solve(single, rhs)
    residual = float(np.linalg.norm(single @ q - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    return LaplaceDirichletResult(
        normal_derivative=q,
        boundary_values=values.copy(),
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_number=condition,
        residual_l2=residual,
        relative_residual_l2=residual / rhs_norm if rhs_norm > 0.0 else residual,
        domain=domain,
        reference_length=reference,
    )


def solve_laplace_dirichlet(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    reference_length: float = REFERENCE_LENGTH,
) -> LaplaceDirichletResult:
    """Solve for the unknown outward normal derivative given boundary values.

    The baseline targets smooth, closed, simply connected 2-D domains.
    Boundary values are piecewise constant at panel midpoints. The returned
    ``normal_derivative`` is the derivative along the polygon's outward normal.
    """
    return _solve(mesh, boundary_values, "interior", reference_length)


def solve_laplace_exterior_dirichlet(
    mesh: PanelMesh,
    boundary_values: FloatArray,
    reference_length: float = REFERENCE_LENGTH,
) -> LaplaceDirichletResult:
    """Solve the exterior 2-D Laplace Dirichlet problem.

    The fluid occupies the unbounded complement of the closed polygon. The
    returned ``normal_derivative`` is ``q_e = grad(u) . n_e`` with ``n_e`` the
    fluid-domain normal, i.e. the negative of the polygon's outward normal,
    matching the three-dimensional convention.

    The integral equation is of the first kind,

        S q_e = (1/2 I - H) u,

    so ``S`` must be well conditioned. Integrating a constant density over a
    closed curve gives ``a*log(reference_length/a)`` for a circle of radius
    ``a``, so ``S`` approaches rank deficiency as the curve's logarithmic
    capacity approaches ``reference_length``: the unshifted kernel (capacity 1)
    is degenerate on a unit circle, and the default ``reference_length = 2`` is
    degenerate on a circle of radius 2. For a fixed inscribed polygon the
    capacity is slightly below the circumscribed radius, so the system is
    ill-conditioned rather than exactly singular, and the condition number
    grows with refinement. A numerically singular system raises ``ValueError``
    instead of being solved.

    The far field is not imposed explicitly: the single-layer representation
    with a decaying density gives a decaying potential, so the boundary
    equation plus the decay of the density is the formulation. Unlike the
    3-D solver there is no explicit far-field or irregular-frequency
    treatment, and the solve is intended for verification meshes.
    """
    return _solve(mesh, boundary_values, "exterior", reference_length)


def evaluate_interior_field(
    mesh: PanelMesh, result: LaplaceDirichletResult, points: FloatArray
) -> FloatArray:
    """Evaluate the harmonic potential at interior points using exact panel integrals.

    Points must be strictly inside the domain and not on the boundary. The
    representation integrates each constant-panel kernel analytically rather
    than sampling the kernel at the panel midpoint.
    """
    if not isinstance(result, LaplaceDirichletResult):
        raise TypeError("result must be a LaplaceDirichletResult")
    if result.domain != "interior":
        raise ValueError(
            "evaluate_interior_field requires a result from "
            "solve_laplace_dirichlet; the exterior formulation is not implemented "
            "for field evaluation"
        )
    targets = np.asarray(points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")
    if not np.isfinite(targets).all():
        raise ValueError("points must all be finite")
    if (
        result.normal_derivative.shape != (mesh.panel_count,)
        or result.boundary_values.shape != (mesh.panel_count,)
    ):
        raise ValueError("result boundary traces must match the mesh panel count")

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
