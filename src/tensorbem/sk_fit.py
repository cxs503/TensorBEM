"""Sanathanan-Koerner (SK) rational fit of radiation-damping curves.

What changed in this round (and why)
------------------------------------
The pre-existing ``sk_fit_damping`` carried the name of the Sanathanan-Koerner
iteration (Sanathanan & Koerner, 1963) but contained **no iteration at all**: it
placed the decay rates on a fixed logarithmic grid, solved one weighted linear
least-squares problem, accepted ``max_iter``/``tol`` and ignored both, and
returned ``n_iter=1, converged=True``.  Measured consequence (this repository,
``docs/sk_fit_sanathanan_koerner.md``): on a curve that *is* a sum of three real
decay rates, the fixed-grid fit misses it by 6.95e-01 (69 % of the peak) at the
same order at which the SK iteration of this module recovers the three decay
rates, the three residues and the asymptote to 6.4e-14.

The module now implements the actual iteration.  Rational approximation in
``u = omega**2`` is written as ``B(u) ~ A(u)/D(u)`` with
``D(u) = 1 + d_1 u + ... + d_M u**M``; the SK step solves the *linearised*
problem ``A(u_j) - B_j * (D(u_j) - 1) = B_j`` at every frequency point ``j``,
weighted by the data weights and by ``1/|D_{i-1}(u_j)|``, and re-solves until
the denominator stops moving (or the weighted residual stops improving).  The
denominator roots are then projected onto the *real decay-rate* basis that
:mod:`tensorbem.state_space` and :mod:`tensorbem.time_domain` use, so the result
is a :class:`~tensorbem.state_space.VectorFitResult` and stays physically
admissible (real poles, non-negative asymptotic damping).

Two properties are deliberately not assumed:

* **SK is not automatically better on measured curves.** A measured 3-D
  radiation-damping curve is *not* a finite sum of real exponentials (the
  free-surface memory kernel has an algebraic tail), so the unconstrained
  rational model can be orders of magnitude more accurate than any real-pole
  projection of it.  The pole set is therefore chosen by *measured* weighted
  residual between the SK projection and the fixed grid of
  :func:`fixed_grid_damping_fit` (the pre-rewrite behaviour, kept verbatim as
  the baseline), and the returned fit is never worse than that baseline at the
  same order.
* **Convergence is reported, not asserted.** The SK iteration can limit-cycle on
  a stiff measured curve; the best iterate (lowest weighted residual) is
  returned and ``converged`` says whether the stopping rule was met.

References
----------
Sanathanan, C.K. & Koerner, J. (1963), "Transfer function synthesis as a
low-sensitivity procedure", *IEEE Trans. Automat. Control* 8(2):142-143.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .state_space import (
    VectorFitResult,
    _evaluate_rational,
    _initial_poles,
    _solve_linear,
    _weights,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "SKFitReport",
    "sk_fit_damping",
    "sk_fit_damping_report",
    "fixed_grid_damping_fit",
]


@dataclass(frozen=True)
class SKFitReport:
    """What the fit actually did, so the caller does not have to trust it.

    Attributes
    ----------
    order : int
        Number of decay rates requested (``n_poles``).
    pole_source : str
        ``"sk"`` when the SK-relocated pole set won the residual comparison,
        ``"grid"`` when the fixed log grid did.  The returned fit is always the
        better of the two by weighted residual.
    sk_iterations, sk_converged : int, bool
        Iterations performed and whether the stopping rule was met.
    residual_history : tuple[float, ...]
        Weighted residual norm of the rational model at each SK iteration.
    rational_rel_error : float
        Scale-normalised max error of the *unconstrained* rational model
        ``A/D``; this is the ceiling the real-pole basis cannot exceed.
    sk_pole_rel_error, grid_pole_rel_error : float
        Scale-normalised max error of the two candidate real-pole fits.
    real_root_count, complex_pair_count : int
        Split of the ``order`` denominator roots: real negative roots map
        exactly onto decay rates, complex pairs are projected onto their
        modulus (an unavoidable model-class error, quantified by
        ``rational_rel_error``).
    """

    order: int
    pole_source: str
    sk_iterations: int
    sk_converged: bool
    residual_history: tuple[float, ...]
    rational_rel_error: float
    sk_pole_rel_error: float
    grid_pole_rel_error: float
    real_root_count: int
    complex_pair_count: int

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["residual_history"] = [float(x) for x in self.residual_history]
        return data


def _validate(omega: FloatArray, damping: FloatArray, n_poles: int) -> None:
    if omega.size < 8:
        raise ValueError(f"Need at least 8 frequency points, got {omega.size}")
    if np.any(omega <= 0):
        raise ValueError("omega must be strictly positive")
    if np.any(damping < -1e-10):
        raise ValueError("damping must be non-negative (passivity)")
    if n_poles < 1:
        raise ValueError(f"n_poles must be >= 1, got {n_poles}")
    if omega.size < 2 * n_poles + 1:
        raise ValueError(
            "the SK rational system has "
            f"{2 * n_poles + 1} unknowns; need at least that many frequency "
            f"points, got {omega.size} (lower n_poles)"
        )


def _rational_model(coef: FloatArray, u: FloatArray, order: int) -> FloatArray:
    numerator = np.polyval(coef[: order + 1][::-1], u)
    denominator = np.polyval(np.concatenate([[1.0], coef[order + 1 :]])[::-1], u)
    return numerator / denominator


def _sk_iterate(
    omega: FloatArray,
    damping: FloatArray,
    order: int,
    max_iter: int,
    tol: float,
    weight_floor: float,
) -> tuple[FloatArray, float, int, bool, FloatArray]:
    """Run the Sanathanan-Koerner iteration on the rational model ``A(u)/D(u)``.

    Returns ``(denominator_coefficients, u_reference, iterations, converged,
    residual_history)`` where the denominator is ``[1, d_1 ... d_order]``.
    """
    u_reference = float(np.median(omega**2))
    u = omega**2 / u_reference
    weights = _weights(omega, damping, weight_floor)

    powers = [u**k for k in range(order + 1)]
    # Unknowns: a_0 ... a_order (numerator), d_1 ... d_order (denominator).
    design = np.column_stack(powers + [-(damping * u**k) for k in range(1, order + 1)])

    denominator = np.zeros(order + 1)
    denominator[0] = 1.0
    best_denominator = denominator.copy()
    best_residual = np.inf
    history: list[float] = []
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        previous = np.polyval(denominator[::-1], u)
        sk_weights = weights / np.maximum(
            np.abs(previous), 1e-12 * max(float(np.max(np.abs(previous))), 1e-300)
        )
        scaled = design * sk_weights[:, None]
        column_norms = np.linalg.norm(scaled, axis=0)
        column_norms[column_norms == 0.0] = 1.0
        coef = np.linalg.lstsq(scaled / column_norms, damping * sk_weights, rcond=None)[0]
        coef = coef / column_norms

        new_denominator = np.concatenate([[1.0], coef[order + 1 :]])
        model = _rational_model(coef, u, order)
        residual = float(np.sqrt(np.mean((weights * (model - damping)) ** 2)))
        history.append(residual)
        if np.isfinite(residual) and residual < best_residual:
            best_residual = residual
            best_denominator = new_denominator.copy()

        shift = float(np.max(np.abs(new_denominator - denominator))) / max(
            1.0, float(np.max(np.abs(new_denominator)))
        )
        previous_residual = history[-2] if len(history) > 1 else np.inf
        residual_shift = (
            abs(history[-1] - previous_residual) / max(abs(previous_residual), 1e-300)
            if np.isfinite(previous_residual)
            else np.inf
        )
        denominator = new_denominator
        if shift <= tol or residual_shift <= 1e-12:
            converged = True
            break

    return best_denominator, u_reference, iterations, converged, np.asarray(history)


def _real_pole_projection(denominator: FloatArray, u_reference: float) -> tuple[FloatArray, int]:
    """Project the denominator roots onto real decay rates.

    A real negative root ``u_k = -alpha**2/u_reference`` is represented exactly.
    A complex pair ``u = r exp(+-i theta)`` cannot be, and is mapped to the
    modulus ``|u|``; the count of such pairs is returned so the caller can see
    how much of the residual is model class rather than fit.
    """
    roots = np.roots(denominator[::-1])
    scale = max(float(np.max(np.abs(roots))), 1e-300)
    real_negative = (np.abs(roots.imag) <= 1e-9 * scale) & (roots.real < 0.0)
    poles = np.empty(roots.size)
    poles[real_negative] = np.sqrt(-roots.real[real_negative] * u_reference)
    complex_roots = ~real_negative
    if np.any(complex_roots):
        poles[complex_roots] = np.sqrt(np.abs(roots[complex_roots]) * u_reference)
    poles = np.sort(np.abs(poles))
    poles[poles <= 0.0] = 1e-12
    pairs = int(np.sum(complex_roots)) // 2
    return poles, pairs


def _result(
    omega: FloatArray,
    damping: FloatArray,
    poles: FloatArray,
    weight_floor: float,
    n_iter: int,
    converged: bool,
) -> VectorFitResult:
    weights = _weights(omega, damping, weight_floor)
    coef, _ = _solve_linear(omega, damping, weights, poles)
    constant = float(coef[0])
    residues = coef[1:]
    reconstructed = _evaluate_rational(omega, poles, residues, constant)
    scale = float(np.max(np.abs(damping)))
    if scale > 0:
        rel_error = np.abs(reconstructed - damping) / scale
        pointwise = np.abs(reconstructed - damping) / np.maximum(
            np.abs(damping), weight_floor * scale
        )
    else:
        rel_error = np.abs(reconstructed - damping)
        pointwise = rel_error
    return VectorFitResult(
        poles=poles,
        residues=residues,
        constant=constant,
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


def fixed_grid_damping_fit(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    weight_floor: float = 1e-3,
) -> VectorFitResult:
    """The pre-rewrite behaviour, kept as the measured baseline.

    Decay rates sit on the logarithmic grid of ``state_space._initial_poles``,
    the residues are solved by one weighted linear least-squares pass.  Retained
    because the verification study must be able to measure what the SK iteration
    changed; it is not the recommended entry point.
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)
    _validate(omega, damping, n_poles)
    poles = _initial_poles(omega, n_poles)
    return _result(omega, damping, poles, weight_floor, 1, True)


def sk_fit_damping_report(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    max_iter: int = 50,
    tol: float = 1e-8,
    weight_floor: float = 1e-3,
) -> tuple[VectorFitResult, SKFitReport]:
    """SK fit of radiation damping, plus a report of what the fit did.

    See :func:`sk_fit_damping` for the model; this variant returns the
    :class:`SKFitReport` alongside the fit so the caller can see the residual
    history, which pole set won and how much headroom the real-pole basis left.
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)
    _validate(omega, damping, n_poles)

    denominator, u_reference, iterations, converged, history = _sk_iterate(
        omega, damping, n_poles, max_iter, tol, weight_floor
    )
    sk_poles, complex_pairs = _real_pole_projection(denominator, u_reference)
    sk_result = _result(omega, damping, sk_poles, weight_floor, iterations, converged)
    grid_result = _result(
        omega, damping, _initial_poles(omega, n_poles), weight_floor, 1, True
    )

    u = omega**2 / u_reference
    weights = _weights(omega, damping, weight_floor)
    sk_weights = weights / np.maximum(np.abs(np.polyval(denominator[::-1], u)), 1e-12)
    powers = [u**k for k in range(n_poles + 1)]
    design = np.column_stack(
        powers + [-(damping * u**k) for k in range(1, n_poles + 1)]
    )
    scaled = design * sk_weights[:, None]
    norms = np.linalg.norm(scaled, axis=0)
    norms[norms == 0.0] = 1.0
    numerator_coef = np.linalg.lstsq(
        scaled / norms, damping * sk_weights, rcond=None
    )[0] / norms
    rational = _rational_model(numerator_coef, u, n_poles)
    scale = float(np.max(np.abs(damping))) if damping.size else 0.0
    rational_error = (
        float(np.max(np.abs(rational - damping)) / scale) if scale > 0 else np.inf
    )

    if sk_result.relative_error_max <= grid_result.relative_error_max:
        result, source = sk_result, "sk"
    else:
        result, source = grid_result, "grid"

    report = SKFitReport(
        order=int(n_poles),
        pole_source=source,
        sk_iterations=int(iterations),
        sk_converged=bool(converged),
        residual_history=tuple(float(x) for x in history),
        rational_rel_error=rational_error,
        sk_pole_rel_error=float(sk_result.relative_error_max),
        grid_pole_rel_error=float(grid_result.relative_error_max),
        real_root_count=int(n_poles - 2 * complex_pairs),
        complex_pair_count=int(complex_pairs),
    )
    return result, report


def sk_fit_damping(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    max_iter: int = 50,
    tol: float = 1e-8,
    weight_floor: float = 1e-3,
) -> VectorFitResult:
    """Fit radiation damping with the Sanathanan-Koerner iteration.

    Model (unchanged, so the result feeds ``state_space`` / ``time_domain``)::

        B(omega) = d + sum_k c_k alpha_k / (alpha_k**2 + omega**2)

    Parameters
    ----------
    omega, damping : array_like
        Frequency grid (strictly positive) and damping values (non-negative).
    n_poles : int
        Number of decay rates.  The SK system needs at least ``2*n_poles + 1``
        frequency points.
    max_iter, tol : int, float
        SK iteration cap and stopping tolerance on the relative denominator
        change (a relative weighted-residual change of 1e-12 also stops it).
    weight_floor : float
        Relative weight floor, see :func:`tensorbem.state_space._weights`.

    Returns
    -------
    VectorFitResult
        Fit on the real decay-rate basis; ``n_iter`` is the number of SK
        iterations actually performed and ``converged`` reports the stopping
        rule, not an assumption.
    """
    result, _ = sk_fit_damping_report(
        omega,
        damping,
        n_poles=n_poles,
        max_iter=max_iter,
        tol=tol,
        weight_floor=weight_floor,
    )
    return result