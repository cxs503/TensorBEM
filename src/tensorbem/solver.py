"""Constant-panel collocation solver for the 2-D Laplace Dirichlet problem.

Boundary equation at a smooth collocation point:
    c*u = integral(G*q ds) - integral((dG/dn_y)*u ds)
with G = -(1/(2*pi))*log(r/2), -Delta(G)=delta, and c=1/2.
Thus Gmat*q = (0.5*I + Hmat)*u. The diagonal H principal-value
contribution is zero for a straight panel collocated at its midpoint.

All straight-panel matrix entries are integrated analytically. This removes
the off-diagonal midpoint-quadrature error and improves near-panel accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh
from .kernels import (
    integrate_double_layer_panel,
    integrate_single_layer_panel,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LaplaceDirichletResult:
    """Solution and matrices from a constant-panel collocation solve."""

    normal_derivative: FloatArray
    boundary_values: FloatArray
    single_layer_matrix: FloatArray
    double_layer_matrix: FloatArray
    condition_number: float
    residual_l2: float


def assemble_matrices(mesh: PanelMesh) -> tuple[FloatArray, FloatArray]:
    """Assemble analytically integrated single- and double-layer matrices."""
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
                target, starts[j], tangents[j], float(lengths[j])
            )
            double[i, j] = integrate_double_layer_panel(
                target, starts[j], tangents[j], normals[j], float(lengths[j])
            )
    return single, double


def solve_laplace_dirichlet(
    mesh: PanelMesh, boundary_values: FloatArray
) -> LaplaceDirichletResult:
    """Solve for the unknown outward normal derivative given boundary values.

    The baseline targets smooth, closed, simply connected 2-D domains.
    Boundary values are piecewise constant at panel midpoints.
    """
    values = np.asarray(boundary_values, dtype=float)
    n = mesh.panel_count
    if values.shape != (n,):
        raise ValueError(f"boundary_values must have shape ({n},)")
    if not np.isfinite(values).all():
        raise ValueError("boundary_values must all be finite")

    single, double = assemble_matrices(mesh)
    rhs = 0.5 * values + double @ values
    condition = float(np.linalg.cond(single))
    q = np.linalg.solve(single, rhs)
    residual = float(np.linalg.norm(single @ q - rhs))
    return LaplaceDirichletResult(
        normal_derivative=q,
        boundary_values=values.copy(),
        single_layer_matrix=single,
        double_layer_matrix=double,
        condition_number=condition,
        residual_l2=residual,
    )


def evaluate_interior_field(
    mesh: PanelMesh, result: LaplaceDirichletResult, points: FloatArray
) -> FloatArray:
    """Evaluate the harmonic potential at interior points using exact panel integrals.

    Points must be strictly inside the domain and not on the boundary. The
    representation integrates each constant-panel kernel analytically rather
    than sampling the kernel at the panel midpoint.
    """
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
    values = np.zeros(len(targets), dtype=float)
    for i, target in enumerate(targets):
        for j in range(mesh.panel_count):
            single = integrate_single_layer_panel(
                target, mesh.vertices[j], tangents[j], float(lengths[j])
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
