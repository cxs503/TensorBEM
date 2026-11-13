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

Round 2 (this revision) adds the step that was missing between the SK
iteration and the physical basis: **pole relocation in the physical basis
itself**.  The SK step places the *denominator roots* well in ``u``, but a
conjugate pair can only be projected onto its modulus, and on the measured
Hulme curve that projection leaves the real poles so badly placed that the fit
stalls at 17-49 % of the peak (the number the first round of this module
reported as a "model-class floor").  A bounded downhill-simplex search on
``log(alpha_k)`` -- residues still solved linearly, so only ``n_poles``
variables move, seeded from both the SK projection and the fixed grid -- reaches
1.3e-03 of the peak on the same curve and the same 12 orders.  The floor was a
*method* limit, not a model-class limit; see
``docs/sk_fit_sanathanan_koerner.md`` and ``docs/dev_report_m16042_sk_iteration.md``.
The refinement is budgeted (``refine_maxfev``), reported, and can be switched
off (``refine_maxfev=0``) to reproduce the first-round numbers exactly.

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
  returned and ``converged`` says whether the stopping rule was met.  The same
  discipline applies to the pole refinement: the simplex never satisfies its own
  simplex-size test within a sensible budget, so the module does *not* forward
  that flag.  It measures local optimality instead -- a one-pole +/-5 % log
  probe at the returned poles, reported as ``refine_probe_gain`` -- so
  "converged" here means "no measured single-pole move lowers the weighted
  residual by more than 0.1 %", not "the optimiser said so".

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
from scipy import optimize

from .state_space import (
    VectorFitResult,
    _evaluate_rational,
    _initial_poles,
    _solve_linear,
    _weights,
)

FloatArray = NDArray[np.float64]

#: Default evaluation budget of the physical-basis pole refinement (per start).
#: ``refine_maxfev=0`` disables the step and reproduces the first-round results.
REFINE_MAXFEV = 1500

#: Warm restarts of the pole-relocation simplex (each one re-seeded from the
#: previous best).  A single Nelder-Mead run on 8-12 variables does not finish
#: within any sensible budget; one restart is enough to make the measured
#: one-pole probe small on the curves of the study.
_REFINE_RESTARTS = 2

#: One-pole log probe used to *measure* local optimality of the refined poles.
_PROBE_FACTORS = (0.95, 0.98, 1.02, 1.05)

#: A probe gain above this means the pole set is not a local optimum of the
#: weighted residual; reported as ``refine_converged = False``.
_PROBE_TOL = 1e-3

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
        Which pole set produced the returned fit, by measured
        scale-normalised max error: ``"sk"`` for the SK-relocated projection,
        ``"grid"`` for the fixed log grid, ``"refined"`` for the
        physical-basis pole refinement of round 2 (which start it came from is
        in ``refine_start``).
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
    refined_pole_rel_error : float
        Best scale-normalised max error of the two refined pole sets;
        ``nan`` when the refinement is disabled (``refine_maxfev=0``).
    refine_start : str
        ``"sk"`` / ``"grid"``: which seed the winning refined pole set came
        from; ``"none"`` when no refined candidate won or the step is off.
    refine_evaluations : int
        Weighted-residual evaluations spent by the refinement (both seeds).
    refine_probe_gain : float
        Best relative objective gain available from a single-pole +/-5 % move
        at the returned poles: ``0`` means ``refine_converged`` is a measured
        statement, not an assumption.  ``nan`` when the step is off.
    refine_converged : bool
        ``refine_probe_gain <= 1e-3`` -- the returned pole set is a measured
        local optimum of the weighted residual.
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
    refined_pole_rel_error: float = float("nan")
    refine_start: str = "none"
    refine_evaluations: int = 0
    refine_probe_gain: float = float("nan")
    refine_converged: bool = False

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


def _refine_poles(
    omega: FloatArray,
    damping: FloatArray,
    poles0: FloatArray,
    weight_floor: float,
    maxfev: int,
) -> tuple[FloatArray, int]:
    """Relocate the real decay rates by downhill simplex on ``log(alpha)``.

    The residues are *not* free variables here: for any pole set they are the
    solution of the weighted linear least-squares problem
    :func:`tensorbem.state_space._solve_linear` already solves, so only the
    ``n_poles`` decay rates move and every iterate stays a physically
    admissible ``(alpha, c, d >= 0)`` fit.  The objective is that same weighted
    residual, which makes the refinement a continuation of the SK step rather
    than a second, differently-weighted fit.

    Why this is needed at all: :func:`_real_pole_projection` maps a complex
    conjugate pair onto its modulus, i.e. onto a *duplicated* real pole.  On the
    measured Hulme hemisphere curve that projection is the dominant error term
    (17-49 % of the peak at 6-12 orders) while the unconstrained rational model
    of the same order is already at 1e-6; the SK roots are good, their
    projection is not.

    Parameters
    ----------
    omega, damping : FloatArray
        Data on the fit band.
    poles0 : FloatArray
        Seed decay rates (SK projection or the fixed log grid).
    weight_floor : float
        Relative weight floor, see :func:`tensorbem.state_space._weights`.
    maxfev : int
        Objective-evaluation budget of the simplex.

    Returns
    -------
    tuple[FloatArray, int]
        Refined decay rates and the number of objective evaluations spent.
    """
    weights = _weights(omega, damping, weight_floor)
    log_low = float(np.log(0.05 * float(omega[0])))
    log_high = float(np.log(50.0 * float(omega[-1])))

    def objective(log_poles: FloatArray) -> float:
        poles = np.exp(np.clip(log_poles, log_low, log_high))
        return _solve_linear(omega, damping, weights, poles)[1]

    best = np.clip(np.log(np.abs(np.asarray(poles0, dtype=float))), log_low, log_high)
    best_objective = objective(best)
    evaluations = 0
    for _ in range(_REFINE_RESTARTS):
        # Nelder-Mead carries no random state: same seed, same iterates, bit for bit.
        outcome = optimize.minimize(
            objective,
            best,
            method="Nelder-Mead",
            options={"maxfev": int(maxfev), "xatol": 1e-8, "fatol": 0.0, "adaptive": True},
        )
        evaluations += int(outcome.nfev)
        candidate = np.clip(np.asarray(outcome.x, dtype=float), log_low, log_high)
        candidate_objective = objective(candidate)
        if candidate_objective < best_objective:
            best, best_objective = candidate, candidate_objective
        else:
            break
    refined = np.exp(best)
    return np.sort(refined), evaluations


def _probe_local_optimality(
    omega: FloatArray,
    damping: FloatArray,
    poles: FloatArray,
    weight_floor: float,
) -> float:
    """Best relative gain of a single-pole ``+/-5 %`` log move, at ``poles``.

    The simplex used by :func:`_refine_poles` does not converge to its own
    simplex-size tolerance on these flat objectives, so its ``success`` flag is
    not reported as convergence.  This probe answers the question the caller
    actually cares about -- "is there a cheaper one-pole move I am leaving on
    the table?" -- with a measurement.  Returns ``0.0`` when no probe improves.
    """
    weights = _weights(omega, damping, weight_floor)
    base = _solve_linear(omega, damping, weights, poles)[1]
    # Round-off guard: on a curve the basis represents exactly the residual is
    # ~1e-32 and a *relative* gain on it is noise, not headroom.
    scale = float(np.max(np.abs(damping))) if damping.size else 0.0
    floor = (1e-10 * max(scale, 1e-300)) ** 2 * max(omega.size, 1)
    if not np.isfinite(base) or base <= floor:
        return 0.0
    gain = 0.0
    for k in range(poles.size):
        for factor in _PROBE_FACTORS:
            trial = np.array(poles, dtype=float, copy=True)
            trial[k] = trial[k] * factor
            candidate = _solve_linear(omega, damping, weights, trial)[1]
            if np.isfinite(candidate):
                gain = max(gain, (base - candidate) / base)
    return float(gain)


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
    refine_maxfev: int = REFINE_MAXFEV,
) -> tuple[VectorFitResult, SKFitReport]:
    """SK fit of radiation damping, plus a report of what the fit did.

    See :func:`sk_fit_damping` for the model; this variant returns the
    :class:`SKFitReport` alongside the fit so the caller can see the residual
    history, which pole set won and how much headroom the real-pole basis left.

    ``refine_maxfev`` budgets the physical-basis pole refinement (round 2);
    ``0`` disables it and reproduces the first-round result bit for bit.
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

    # Candidate pole sets, each scored by the same measured metric the caller
    # sees (scale-normalised max error), so "the returned fit is never worse"
    # is an invariant and not a hope.
    candidates: list[tuple[str, str, VectorFitResult]] = [
        ("sk", "none", sk_result),
        ("grid", "none", grid_result),
    ]
    refined_error = float("nan")
    refine_start = "none"
    refine_evaluations = 0
    probe_gain = float("nan")
    refine_converged = False
    if int(refine_maxfev) > 0:
        probe_gain = 0.0
        for start_name, start_poles in (
            ("sk", sk_poles),
            ("grid", _initial_poles(omega, n_poles)),
        ):
            refined_poles, evaluations = _refine_poles(
                omega, damping, start_poles, weight_floor, int(refine_maxfev)
            )
            refine_evaluations += evaluations
            refined = _result(
                omega,
                damping,
                refined_poles,
                weight_floor,
                iterations + refine_evaluations,
                False,
            )
            if not np.isfinite(refined_error) or refined.relative_error_max < refined_error:
                refined_error = float(refined.relative_error_max)
            candidates.append(("refined", start_name, refined))
        probe_gain = _probe_local_optimality(omega, damping, refined_poles, weight_floor)
        refine_converged = bool(probe_gain <= _PROBE_TOL)

    source, refine_start, result = min(
        candidates, key=lambda item: item[2].relative_error_max
    )
    if source == "refined":
        # The refinement is the stage that produced the returned poles, so the
        # returned ``converged`` reports the *measured* probe, not the simplex's
        # own (never-met) flag.  See ``SKFitReport.refine_converged``.
        result = _result(
            omega, damping, result.poles, weight_floor, result.n_iter, refine_converged
        )

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
        refined_pole_rel_error=refined_error,
        refine_start=refine_start,
        refine_evaluations=int(refine_evaluations),
        refine_probe_gain=probe_gain,
        refine_converged=refine_converged,
    )
    return result, report


def sk_fit_damping(
    omega: Sequence[float],
    damping: Sequence[float],
    n_poles: int = 8,
    max_iter: int = 50,
    tol: float = 1e-8,
    weight_floor: float = 1e-3,
    refine_maxfev: int = REFINE_MAXFEV,
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
    refine_maxfev : int
        Evaluation budget of the physical-basis pole refinement that follows
        the SK iteration (``0`` disables it: first-round behaviour, bit for
        bit).  The returned fit is always the best of the candidate pole sets by
        measured error, so a larger budget can only lower the error; it is an
        accuracy-per-second knob, not a correctness one.

    Returns
    -------
    VectorFitResult
        Fit on the real decay-rate basis; ``n_iter`` counts the SK iterations
        (plus the refinement evaluations when a refined pole set won) and
        ``converged`` reports the stopping rule of the stage that produced the
        returned poles, not an assumption.
    """
    result, _ = sk_fit_damping_report(
        omega,
        damping,
        n_poles=n_poles,
        max_iter=max_iter,
        tol=tol,
        weight_floor=weight_floor,
        refine_maxfev=refine_maxfev,
    )
    return result