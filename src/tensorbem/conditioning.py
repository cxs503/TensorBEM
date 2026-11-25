"""Condition number estimation for BEM operators.

Provides condition number estimation for the single-layer potential operator S
and the double-layer operator (1/2 I - H) used in exterior Dirichlet problems.
For small matrices, uses numpy.linalg.cond directly. For larger matrices,
uses power iteration to estimate the largest and smallest singular values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve, norm

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


def _largest_singular_value(apply, adjoint, size, iterations):
    """Alternate normalized left and right singular-vector iterations."""
    v = np.random.default_rng(42).standard_normal(size)
    v /= norm(v)
    for _ in range(iterations):
        u = apply(v)
        sigma = norm(u)
        if sigma == 0:
            return 0.
        v = adjoint(u / sigma)
        v /= norm(v)
    return float(sigma)


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
        if power_iterations < 1:
            raise ValueError("power_iterations must be positive")
        matrix = np.asarray(matrix)
        if matrix.shape != (n, n) or not np.isfinite(matrix).all():
            raise ValueError("power iteration requires a finite square matrix")
        scale = np.max(np.abs(matrix))
        if scale == 0:
            return float("inf"), "power_iteration", power_iterations
        matrix = matrix / scale
        # cond_2(A) = ||A||_2 ||A^-1||_2. Both norms require the adjoint
        # action, even when every eigenvalue of A happens to be identical.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                factors = lu_factor(matrix)
        except LinAlgWarning:
            return float("inf"), "power_iteration", power_iterations
        adjoint = matrix.conj().T
        sigma_max = _largest_singular_value(lambda v: matrix @ v, lambda v: adjoint @ v,
                                            n, power_iterations)
        inverse_norm = _largest_singular_value(lambda v: lu_solve(factors, v),
                                               lambda v: lu_solve(factors, v, trans=2),
                                               n, power_iterations)
        return sigma_max * inverse_norm, "power_iteration", power_iterations


def condition_number_with_fallback(matrix: FloatArray) -> tuple[float, str]:
    """Condition number of ``matrix`` with a non-SVD fallback.

    ``numpy.linalg.cond`` is singular-value based.  Some LAPACK builds return
    ``inf`` (or ``nan``) for perfectly well conditioned panels; the loongarch64
    compute nodes did exactly that for the 128-panel single-layer matrix whose
    true condition number is ~76 (``docs/dev_report_m21296_cylinder_near_wall_keyerror.md``).
    Callers that treat the condition number as a hard "is this system singular"
    gate then reject a solvable problem.  A non-finite SVD result is therefore
    re-estimated with the LU-based power iteration of
    :func:`estimate_condition_number`, which never calls into the SVD.

    Parameters
    ----------
    matrix : ndarray
        Square matrix to analyse.

    Returns
    -------
    (condition, source) : tuple[float, str]
        ``source`` is ``"svd"`` when ``numpy.linalg.cond`` returned a finite
        value and ``"power_iteration"`` when the fallback was needed.
    """
    condition = float(np.linalg.cond(matrix))
    if np.isfinite(condition):
        return condition, "svd"
    # Thin QR preserves the singular values without squaring the condition
    # number. For a wide matrix, factor its adjoint.
    square = np.asarray(matrix)
    rectangular = square.shape[0] != square.shape[1]
    if rectangular:
        columns = square if square.shape[0] > square.shape[1] else square.conj().T
        square = np.linalg.qr(columns, mode="r")
    estimate, method, _ = estimate_condition_number(square, method="power_iteration")
    return float(estimate), f"{method}_qr" if rectangular else method


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
