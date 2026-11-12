"""Sanathanan-Koerner iterative rational approximation of radiation damping.

Background
----------
Radiation damping of a floating body is a rational function of ``omega**2``::

    B(omega) = d + sum_k c_k * alpha_k / (alpha_k**2 + omega**2)

with real decay rates ``alpha_k > 0`` (the "relaxation spectrum" form).  The
time-domain bridge (Cummins equation) needs this form because the rational
denominator makes the radiation memory kernel an exact sum of exponentials,
``K(t) = sum_k c_k exp(-alpha_k t)`` for ``t > 0`` --- see
:func:`tensorbem.time_domain.radiation_memory_kernel` for the quadrature route
that this closes against.

Fitting that model is a *nonlinear* problem in ``alpha``.  The classical
Sanathanan-Koerner iteration (Sanathanan & Koerner, 1963) turns it into a
sequence of *linear* least-squares problems: at each iteration the denominator
``D(x)`` (``x = omega**2``) is represented in the localized form

    D(x) = 1 + sum_j rt_j / (x + beta_j)

and the linearised equation

    sum_k c_k g_k(w) + d - B(w) * sum_j rt_j / (x + beta_j) = B(w)

is solved with weights ``1/|D_prev(x)|`` (the SK weights).  The zeros of
``D`` are the relocated poles for the next iteration --- this is the pole
relocation step of vector fitting (Gustavsen & Semlyen, 1999), which is SK
iteration plus a rational re-basing.  Because the linearised problem describes
*2p* poles (``p`` current + ``p`` relocated), the residues are re-solved at the
relocated poles before the model misfit is measured; the returned cost is the
misfit of the p-pole model, not of the linearised auxiliary system.

History / what changed
----------------------
The first version of this module carried the SK name but performed a **single**
fixed-pole weighted least squares (poles frozen on a log grid), reporting
``n_iter=1`` and ``converged=True`` unconditionally while accepting ``max_iter``
and ``tol`` arguments that were never read.  That behaviour is preserved as
:func:`fixed_pole_fit_damping` (for baseline comparisons) and is what
:func:`sk_fit_damping` used to be; :func:`sk_fit_damping` now performs the
iteration it advertises.

Reference
---------
Sanathanan, C.K. & Koerner, J. (1963), "Transfer function synthesis as a
low-sensitivity procedure", *IEEE Trans. Automat. Control* 8(2):142-143.

Gustavsen, B. & Semlyen, A. (1999), "Rational approximation of frequency
domain responses by vector fitting", *IEEE Trans. Power Delivery* 14(3):1052-1061.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .state_space import VectorFitResult, _evaluate_rational, _weights

FloatArray = NDArray[np.float64]

__all__ = ["sk_fit_damping", "fixed_pole_fit_damping"]


# ---------------------------------------------------------------------------
# basis / weights
# ---------------------------------------------------------------------------
def _relaxation_basis(omega: FloatArray, poles: FloatArray) -> FloatArray:
    """Columns ``alpha_k / (alpha_k^2 + omega^2)``."""
    a = np.asarray(poles, dtype=float)[None, :]
    return a / (a * a + omega[:, None] ** 2)


def _denominator(omega: FloatArray, poles: FloatArray, rt: FloatArray) -> FloatArray:
    """``D(x) = 1 + sum_j rt_j / (x + beta_j)`` evaluated at ``x = omega**2``."""
    x = np.asarray(omega, dtype=float) ** 2
    local = 1.0 / (x[:, None] + np.asarray(poles, dtype=float)[None, :])
    return 1.0 + local @ np.asarray(rt, dtype=float)


def _solve_residues(
    omega: FloatArray, damping: FloatArray, poles: FloatArray, weights: FloatArray
) -> tuple[float, FloatArray, float]:
    """Weighted linear solve for ``(d, c)`` at fixed poles; cost ``d >= 0``."""
    design = np.concatenate(
        [np.ones((omega.size, 1)), _relaxation_basis(omega, poles)], axis=1
    )
    weighted = design * weights[:, None]
    column = np.linalg.norm(weighted, axis=0)
    column[column == 0.0] = 1.0
    coef = np.linalg.lstsq(weighted / column, damping * weights, rcond=None)[0] / column
    if coef[0] < 0.0:
        # Negative d means negative damping at infinity (an active system):
        # re-solve with the constant pinned at the physical bound d = 0.
        rest = np.linalg.lstsq(
            (weighted / column)[:, 1:], damping * weights, rcond=None
        )[0] / column[1:]
        coef = np.concatenate([[0.0], rest])
    residual = (design @ coef - damping) * weights
    return float(coef[0]), coef[1:], float(residual @ residual)


def _relocate(poles: FloatArray, rt: FloatArray) -> tuple[FloatArray, int]:
    """Zeros of ``D`` become the new poles; returns ``(poles, n_complex_roots)``.

    ``D(x) = prod_l (x + beta_l) + sum_j rt_j prod_{l != j} (x + beta_l)`` is a
    degree-p polynomial in ``x = omega**2``; its roots are ``-alpha_k**2``.
    Roots with a non-negligible imaginary part mean the target is outside the
    real-pole model class; they are folded onto the real axis and counted so
    the caller can report the misfit honestly instead of hiding it.
    """
    poles = np.asarray(poles, dtype=float)
    numerator = np.poly(poles)
    for j in range(poles.size):
        others = np.delete(poles, j)
        numerator = numerator + rt[j] * np.concatenate([[0.0], np.poly(others)])
    roots = np.roots(numerator)
    n_complex = int(np.sum(np.abs(roots.imag) > 1.0e-6 * np.abs(roots.real) + 1.0e-12))
    return np.sort(np.sqrt(np.abs(roots.real))), n_complex


def _fit_result(
    omega: FloatArray,
    damping: FloatArray,
    poles: FloatArray,
    residues: FloatArray,
    constant: float,
    weight_floor: float,
    *,
    n_iter: int,
    converged: bool,
) -> VectorFitResult:
    """Assemble a :class:`VectorFitResult` with the module-wide error metrics."""
    order = np.argsort(poles)
    poles = np.asarray(poles, dtype=float)[order]
    residues = np.asarray(residues, dtype=float)[order]
    reconstructed = _evaluate_rational(omega, poles, residues, float(constant))
    scale = float(np.max(np.abs(damping)))
    if scale > 0.0:
        rel_error = np.abs(reconstructed - damping) / scale
        pointwise = np.abs(reconstructed - damping) / np.maximum(
            np.abs(damping), weight_floor * scale
        )
    else:  # pragma: no cover - degenerate all-zero curve
        rel_error = np.abs(reconstructed - damping)
        pointwise = rel_error
    return VectorFitResult(
        poles=poles,
        residues=residues,
        constant=float(constant),
        omega_fit=omega,
        B_fit=damping,
        B_reconstructed=reconstructed,
        relative_error_max=float(np.max(rel_error)),
        relative_error_rms=float(np.sqrt(np.mean(rel_error**2))),
        n_iter=int(n_iter),
        converged=bool(converged),
        relative_error_max_pointwise=float(np.max(pointwise)),
        weight_floor=weight_floor,
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _validate(omega: Sequence[float], damping: Sequence[float], n_poles: int) -> None:
    if n_poles < 1:
        raise ValueError(f"n_poles must be >= 1, got {n_poles}")


def fixed_pole_fit_damping(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    weight_floor: float = 1e-3,
) -> VectorFitResult:
    """Pre-fix behaviour: one weighted linear solve with poles frozen on a log grid.

    Kept only as the baseline for the SK verification benchmark
    (:mod:`tensorbem.benchmarks.sk_fit_verification`); it does not relocate
    poles, so it cannot recover decay rates that fall between grid points.
    """
    omega_arr = np.asarray(omega, dtype=float)
    damping_arr = np.asarray(damping, dtype=float)
    _validate(omega_arr, damping_arr, n_poles)
    poles = np.logspace(
        np.log10(0.3 * float(omega_arr[0])), np.log10(3.0 * float(omega_arr[-1])), n_poles
    )
    weights = _weights(omega_arr, damping_arr, weight_floor)
    constant, residues, _ = _solve_residues(omega_arr, damping_arr, poles, weights)
    return _fit_result(
        omega_arr,
        damping_arr,
        poles,
        residues,
        constant,
        weight_floor,
        n_iter=1,
        converged=True,
    )


def sk_fit_damping(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    max_iter: int = 50,
    tol: float = 1e-8,
    weight_floor: float = 1e-3,
) -> VectorFitResult:
    """Fit radiation damping with the Sanathanan-Koerner iteration.

    Parameters
    ----------
    omega : array_like
        Frequency grid (rad/s), strictly positive.  A strictly increasing grid is
        not required, but the fit is a plain least-squares problem on the samples
        given.
    damping : array_like
        Radiation damping ``B(omega)`` at each frequency (non-negative).
    n_poles : int
        Number of real poles ``alpha_k`` in the model.
    max_iter : int
        Maximum number of SK iterations (pole relocations).  ``1`` reproduces the
        single-shot behaviour of :func:`fixed_pole_fit_damping` up to the
        relocation step.
    tol : float
        Relative decrease of the weighted model misfit that counts as converged.
    weight_floor : float
        Floor (relative to ``max|B|``) of the per-sample relative weights, so a
        curve spanning decades does not let the near-zero tail dominate.

    Returns
    -------
    VectorFitResult
        ``poles``/``residues``/``constant`` of the fitted relaxation spectrum,
        ``B_reconstructed``, the error metrics, and the *actual* iteration
        count / convergence flag (``n_iter``, ``converged``).

    Notes
    -----
    The iteration is deterministic: poles start on the logarithmic grid
    ``[0.3*omega_min, 3*omega_max]`` and only ever move by pole relocation, so
    repeated calls give bit-identical results (no optimizer, no random restart).

    The model class has real poles only.  When the target curve is not in the
    class (e.g. a 2-D damping curve with a long-wave logarithmic branch), the
    relocated polynomial develops complex roots; those are folded onto the real
    axis and the iteration stops as soon as the model misfit stops decreasing,
    with ``converged=False``.  Resonance-free band-limited curves also leave the
    individual pole positions poorly observable --- the misfit of ``B`` (and of
    the derived memory kernel) is the number to check, not the pole values.
    """
    omega_arr = np.asarray(omega, dtype=float)
    damping_arr = np.asarray(damping, dtype=float)

    if omega_arr.size < 8:
        raise ValueError(f"Need at least 8 frequency points, got {omega_arr.size}")
    if np.any(omega_arr <= 0):
        raise ValueError("omega must be strictly positive")
    if np.any(damping_arr < -1e-10):
        raise ValueError("damping must be non-negative (passivity)")
    _validate(omega_arr, damping_arr, n_poles)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    weights = _weights(omega_arr, damping_arr, weight_floor)
    poles = np.logspace(
        np.log10(0.3 * float(omega_arr[0])),
        np.log10(3.0 * float(omega_arr[-1])),
        n_poles,
    )
    relocation = np.zeros(n_poles)
    constant, residues, cost = _solve_residues(omega_arr, damping_arr, poles, weights)
    best = (cost, poles.copy(), residues.copy(), constant, 0)

    n_iter = 0
    converged = False
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        sk_weights = weights / np.maximum(
            np.abs(_denominator(omega_arr, poles, relocation)), 1e-300
        )
        # Linearised SK system: unknowns (c_1..c_p, d, rt_1..rt_p).
        local = 1.0 / (
            omega_arr[:, None] ** 2 + np.asarray(poles, dtype=float)[None, :]
        )
        design = np.concatenate(
            [
                _relaxation_basis(omega_arr, poles),
                np.ones((omega_arr.size, 1)),
                -damping_arr[:, None] * local,
            ],
            axis=1,
        )
        weighted = design * sk_weights[:, None]
        column = np.linalg.norm(weighted, axis=0)
        column[column == 0.0] = 1.0
        coef = np.linalg.lstsq(
            weighted / column, damping_arr * sk_weights, rcond=None
        )[0] / column
        relocation = coef[n_poles + 1 :]

        relocated, _ = _relocate(poles, relocation)
        trial_constant, trial_residues, trial_cost = _solve_residues(
            omega_arr, damping_arr, relocated, weights
        )
        if trial_cost > cost and not np.allclose(relocated, poles):
            # Full relocation overshot: take the geometric (log-space) midpoint
            # step on the poles before giving up on this iteration.
            blended = np.sqrt(poles * relocated)
            b_constant, b_residues, b_cost = _solve_residues(
                omega_arr, damping_arr, blended, weights
            )
            if b_cost < trial_cost:
                relocated, trial_constant, trial_residues, trial_cost = (
                    blended,
                    b_constant,
                    b_residues,
                    b_cost,
                )

        if trial_cost < best[0]:
            best = (
                trial_cost,
                relocated.copy(),
                trial_residues.copy(),
                trial_constant,
                iteration,
            )
        if trial_cost > cost * (1.0 + 1e-12):
            # Relocation left the basin: stop on the best p-pole model so far.
            break
        decrease = (cost - trial_cost) / max(cost, 1e-300)
        poles, constant, residues, cost = relocated, trial_constant, trial_residues, trial_cost
        if decrease <= tol:
            converged = True
            break

    cost, poles, residues, constant, _ = best
    return _fit_result(
        omega_arr,
        damping_arr,
        poles,
        residues,
        constant,
        weight_floor,
        n_iter=n_iter,
        converged=converged,
    )