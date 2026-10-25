"""Condition number estimation for BEM operators.

Provides condition number estimation for the single-layer potential operator S
and the double-layer operator (1/2 I - H) used in exterior Dirichlet problems.
For small matrices, uses numpy.linalg.cond directly. For larger matrices,
uses power iteration to estimate the largest and smallest singular values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .geometry3d import TriangleMesh
from .solver3d import solve_laplace_surface

FloatArray = NDArray[np.float64]

# Threshold for switching from direct to iterative estimation
_LARGE_MATRIX_THRESHOLD = 500


@dataclass(frozen=True)
class ConditioningReport:
    """Condition number diagnostics for BEM operators.

    Attributes
    ----------
    mesh_panels : int
        Number of panels in the mesh.
    single_layer_condition : float
        Condition number of the single-layer operator S.
    double_layer_condition : float
        Condition number of the double-layer operator (1/2 I - H).
    estimation_method : str
        Method used: 'direct' (numpy.linalg.cond) or 'power_iteration'.
    power_iterations : int
        Number of power iterations used (0 if direct method).
    """

    mesh_panels: int
    single_layer_condition: float
    double_layer_condition: float
    estimation_method: Literal["direct", "power_iteration"]
    power_iterations: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return {
            "mesh_panels": int(self.mesh_panels),
            "single_layer_condition": float(self.single_layer_condition),
            "double_layer_condition": float(self.double_layer_condition),
            "estimation_method": self.estimation_method,
            "power_iterations": int(self.power_iterations),
        }


def _power_iteration_largest(matrix: FloatArray, iterations: int = 50) -> float:
    """Estimate the largest singular value using power iteration on A^T A."""
    n = matrix.shape[1]
    # Random initial vector
    rng = np.random.default_rng(seed=42)
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    ata = matrix.T @ matrix
    for _ in range(iterations):
        w = ata @ v
        sigma = np.linalg.norm(w)
        if sigma < 1e-14:
            return 0.0
        v = w / sigma

    # sigma_max^2 is the Rayleigh quotient
    return float(np.sqrt(sigma))


def _power_iteration_smallest(
    matrix: FloatArray, iterations: int = 50, tol: float = 1e-10
) -> float:
    """Estimate the smallest singular value using inverse power iteration.

    Uses LU decomposition for efficiency. Falls back to direct solve if
    matrix is singular.
    """
    n = matrix.shape[1]
    rng = np.random.default_rng(seed=42)
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    try:
        # Use LU decomposition for repeated solves
        from scipy.linalg import lu_factor, lu_solve

        lu, piv = lu_factor(matrix)
        for _ in range(iterations):
            w = lu_solve((lu, piv), v)
            sigma = np.linalg.norm(w)
            if sigma < tol:
                return 0.0
            v = w / sigma
        return float(1.0 / sigma)
    except Exception:
        # Fall back to direct solve
        for _ in range(iterations):
            try:
                w = np.linalg.solve(matrix, v)
            except np.linalg.LinAlgError:
                return 0.0
            sigma = np.linalg.norm(w)
            if sigma < tol:
                return 0.0
            v = w / sigma
        return float(1.0 / sigma)


def estimate_condition_number(
    matrix: FloatArray,
    method: Literal["auto", "direct", "power_iteration"] = "auto",
    power_iterations: int = 50,
) -> tuple[float, str, int]:
    """Estimate the condition number of a matrix.

    Parameters
    ----------
    matrix : ndarray
        Square matrix to analyze.
    method : str
        'auto': use direct for small matrices, power iteration for large
        'direct': always use numpy.linalg.cond
        'power_iteration': always use power iteration
    power_iterations : int
        Number of iterations for power method.

    Returns
    -------
    condition : float
        Estimated condition number (ratio of largest to smallest singular value).
    method_used : str
        Actual method used ('direct' or 'power_iteration').
    iterations : int
        Number of power iterations performed (0 for direct method).
    """
    n = matrix.shape[0]

    if method == "auto":
        if n <= _LARGE_MATRIX_THRESHOLD:
            method = "direct"
        else:
            method = "power_iteration"

    if method == "direct":
        cond = float(np.linalg.cond(matrix))
        return cond, "direct", 0
    else:
        sigma_max = _power_iteration_largest(matrix, power_iterations)
        sigma_min = _power_iteration_smallest(matrix, power_iterations)
        if sigma_min < 1e-14:
            cond = float("inf")
        else:
            cond = sigma_max / sigma_min
        return cond, "power_iteration", power_iterations


def estimate_operator_conditions(
    mesh: TriangleMesh,
    boundary_values: FloatArray | None = None,
    domain: str = "exterior",
    method: Literal["auto", "direct", "power_iteration"] = "auto",
    power_iterations: int = 50,
) -> ConditioningReport:
    """Estimate condition numbers for BEM operators on a mesh.

    Solves the Laplace surface problem to obtain the S and H matrices,
    then estimates their condition numbers.

    Parameters
    ----------
    mesh : TriangleMesh
        The surface mesh.
    boundary_values : ndarray, optional
        Boundary values for the solve. If None, uses unit values.
    domain : str
        'interior' or 'exterior' problem.
    method : str
        Condition number estimation method.
    power_iterations : int
        Number of iterations for power method.

    Returns
    -------
    ConditioningReport
        Condition number diagnostics.
    """
    n = mesh.panel_count

    # Use simple boundary values if not provided
    if boundary_values is None:
        boundary_values = np.ones(n)

    # Solve to get the matrices
    result = solve_laplace_surface(mesh, boundary_values, domain=domain)

    S = result.single_layer_matrix
    H = result.double_layer_matrix

    # Double-layer operator is (1/2 I - H) for exterior, (1/2 I + H) for interior
    if domain == "exterior":
        DL_operator = 0.5 * np.eye(n) - H
    else:
        DL_operator = 0.5 * np.eye(n) + H

    # Estimate condition numbers
    s_cond, s_method, s_iter = estimate_condition_number(S, method, power_iterations)
    dl_cond, dl_method, dl_iter = estimate_condition_number(
        DL_operator, method, power_iterations
    )

    # Use the more expensive method's iteration count
    max_iter = max(s_iter, dl_iter)

    return ConditioningReport(
        mesh_panels=n,
        single_layer_condition=s_cond,
        double_layer_condition=dl_cond,
        estimation_method=s_method,
        power_iterations=max_iter,
    )


def condition_number_trend(
    mesh_generator,
    panel_counts: list[int],
    method: Literal["auto", "direct", "power_iteration"] = "auto",
    power_iterations: int = 50,
) -> list[ConditioningReport]:
    """Compute condition numbers for a sequence of mesh refinements.

    Parameters
    ----------
    mesh_generator : callable
        Function that takes panel count (or similar) and returns a TriangleMesh.
        Should accept a keyword argument to control mesh density.
    panel_counts : list of int
        Target panel counts for each refinement level.
    method : str
        Condition number estimation method.
    power_iterations : int
        Number of iterations for power method.

    Returns
    -------
    list of ConditioningReport
        Reports for each mesh refinement level.
    """
    reports = []
    for count in panel_counts:
        try:
            mesh = mesh_generator(panels=count)
        except TypeError:
            # Try subdivisions-based generator
            mesh = mesh_generator(count)

        report = estimate_operator_conditions(
            mesh, method=method, power_iterations=power_iterations
        )
        reports.append(report)

    return reports
