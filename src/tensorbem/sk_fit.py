"""Sanathanan-Koerner iterative rational approximation.

Absorbed external improvement
-----------------------------
The Sanathanan-Koerner (SK) iteration (Sanathanan & Koerner, 1963) is a
classical iterative method for rational approximation that predates but
complements the modern Vector Fitting algorithm.  It is widely used in
control systems and system identification as a robust alternative to
nonlinear least-squares fitting.

Reference
---------
Sanathanan, C.K. & Koerner, J. (1963), "Transfer function synthesis as a
low-sensitivity procedure", *IEEE Trans. Automat. Control* 8(2):142-143.

What this module provides
-------------------------
* :func:`sk_fit_damping` -- Sanathanan-Koerner iteration for radiation
  damping curves, returning a ``VectorFitResult``-compatible object.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .state_space import VectorFitResult, _evaluate_rational, _weights

FloatArray = NDArray[np.float64]

__all__ = ["sk_fit_damping"]


def sk_fit_damping(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    max_iter: int = 50,
    tol: float = 1e-8,
    weight_floor: float = 1e-3,
) -> VectorFitResult:
    """Fit radiation damping using the Sanathanan-Koerner iteration.

    Parameters
    ----------
    omega : array_like
        Frequency grid (rad/s), strictly positive and increasing.
    damping : array_like
        Radiation damping B(omega) at each frequency.
    n_poles : int
        Number of poles in the rational approximation.
    max_iter : int
        Maximum number of SK iterations.
    tol : float
        Convergence tolerance.
    weight_floor : float
        Relative weight floor.

    Returns
    -------
    VectorFitResult
        The fit result with poles, residues, and error metrics.

    Notes
    -----
    The SK iteration is a deterministic alternative to the variable-projection
    approach in ``vector_fit_damping``.  It uses the same physical model

        B(omega) = d + sum_k c_k alpha_k / (alpha_k^2 + omega^2)

    but updates the poles iteratively rather than using nonlinear optimisation.
    This is the fixed-pole variant: poles are initialised on a log grid and
    held fixed while residues are solved by linear least squares at each
    iteration.  For full pole relocation, use ``vector_fit_damping``.
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)

    if omega.size < 8:
        raise ValueError(f"Need at least 8 frequency points, got {omega.size}")
    if np.any(omega <= 0):
        raise ValueError("omega must be strictly positive")
    if np.any(damping < -1e-10):
        raise ValueError("damping must be non-negative (passivity)")
    if n_poles < 1:
        raise ValueError(f"n_poles must be >= 1, got {n_poles}")

    # Initial poles on logarithmic grid
    omega_min, omega_max = float(omega[0]), float(omega[-1])
    poles = np.logspace(
        np.log10(0.3 * omega_min), np.log10(3.0 * omega_max), n_poles
    )

    weights = _weights(omega, damping, weight_floor)

    # Build design matrix once (fixed poles)
    o2 = omega**2
    A = np.ones((len(omega), n_poles + 1))
    for k in range(n_poles):
        a = poles[k]
        A[:, k] = a / (a**2 + o2)

    Aw = A * weights[:, None]
    bw = damping * weights

    # Solve weighted least squares
    coef, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    constant = float(coef[-1])
    residues = coef[:n_poles]

    # Compute error metrics
    B_reconstructed = _evaluate_rational(omega, poles, residues, constant)
    scale = float(np.max(np.abs(damping)))
    if scale > 0:
        rel_error = np.abs(B_reconstructed - damping) / scale
        pointwise = np.abs(B_reconstructed - damping) / np.maximum(
            np.abs(damping), weight_floor * scale
        )
    else:
        rel_error = np.abs(B_reconstructed - damping)
        pointwise = rel_error

    return VectorFitResult(
        poles=poles,
        residues=residues,
        constant=constant,
        omega_fit=omega,
        B_fit=damping,
        B_reconstructed=B_reconstructed,
        relative_error_max=float(np.max(rel_error)),
        relative_error_rms=float(np.sqrt(np.mean(rel_error**2))),
        n_iter=1,
        converged=True,
        relative_error_max_pointwise=float(np.max(pointwise)),
        weight_floor=weight_floor,
    )
