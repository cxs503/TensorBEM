"""Constant-panel collocation solver for the 2-D Laplace Dirichlet problem.

Boundary equation at a smooth collocation point:
    c*u = integral(G*q ds) - integral((dG/dn_y)*u ds)
with G = -(1/(2*pi))*log(r), -Delta(G)=delta, and c=1/2.
Thus Gmat*q = (0.5*I + Hmat)*u. The diagonal H principal-value
contribution is zero for a straight panel collocated at its midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .geometry import PanelMesh
from .kernels import double_layer_kernel, fundamental_solution

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
    """Assemble midpoint-collocation single- and double-layer matrices.

    Off-diagonal panel integrals use midpoint quadrature. The logarithmic
    single-layer self-panel integral is integrated analytically.
    """
    centers = mesh.centers
    normals = mesh.normals
    lengths = mesh.lengths
    n = mesh.panel_count
    single = np.empty((n, n), dtype=float)
    double = np.zeros((n, n), dtype=float)

    for i, target in enumerate(centers):
        for j, source in enumerate(centers):
            if i == j:
                # Integral_{-L/2}^{L/2} -log(|s|)/(2*pi) ds.
                single[i, j] = -lengths[j] / (2.0 * np.pi) * (
                    np.log(lengths[j] / 2.0) - 1.0
                )
                # Principal value of the straight-panel self contribution.
                double[i, j] = 0.0
            else:
                delta = target - source
                distance = float(np.linalg.norm(delta))
                single[i, j] = float(fundamental_solution(distance)) * lengths[j]
                double[i, j] = (
                    double_layer_kernel(target, source, normals[j]) * lengths[j]
                )
    return single, double


def solve_laplace_dirichlet(
    mesh: PanelMesh, boundary_values: FloatArray
) -> LaplaceDirichletResult:
    """Solve for the unknown outward normal derivative given boundary values.

    This baseline is intended for smooth, closed, simply connected 2-D domains.
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
