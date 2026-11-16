"""Decidability of a code-to-code cross-validation verdict.

Why this module exists
----------------------
A cross-validation headline ("TensorBEM is within 3 % of capytaine") is a
statement about *two* discretisations: the study's own panel ladder and the
reference's.  Both carry a remaining discretisation error, and both are finite.
When the sum of those two uncertainties is a sizeable fraction of the tolerance
being judged, the deviation is a statement about *meshes*, not about solvers --
and a ``passed=False`` at that tolerance is not a failure of the solver, it is a
failure of the study to be *decidable* at that tolerance.

Reporting that as PASS/FAIL is the defect this module removes: every judged
quantity gets a three-valued verdict

    PASS         deviation within tolerance, and the tolerance is resolvable
    FAIL         deviation outside tolerance, and the tolerance is resolvable
    UNDECIDABLE  the two uncertainties do not let the tolerance be resolved
                 (or their size is unknown, which is *not* zero)

Rules (fixed here, before any study is looked at)
------------------------------------------------
1. **Uncertainty of a ladder's reported value.**  With a monotone power-law
   ladder ``v(N) = v_inf + C N^-p`` the *finest rung* carries a remaining
   relative error of ``last_rung_change / (r^p - 1)`` where ``r`` is the panel
   refinement ratio and ``p`` the measured order -- this is the classical
   a-posteriori (Richardson/GCI) estimate and needs no extra runs.
2. **Extrapolation is not credited for free.**  A Richardson extrapolant is
   only as trustworthy as the asymptotic range it was fitted in; on ladders that
   leave that range the fit has been observed to move a value several times the
   last rung change (``docs/xval_bem_finite_depth_crossval_plan.md``).  So the
   extrapolant inherits the *finest-rung* bound above unless a fourth rung
   measures its own band (drop-coarsest re-extrapolation); then the larger of
   the measured band and the model bound is used.
3. **Non-asymptotic ladders fall back to the last rung change.**  If the measured
   order is missing or below the study's pre-registered floor, the power-law
   model does not hold; the last rung change is then used as the bound and the
   basis says so (it is a weaker statement, never a smaller one).
4. **The budget is a third of the tolerance**, i.e. the classical "uncertainty
   well below the acceptance band" rule, matching
   :data:`tensorbem.benchmarks.finite_depth_crossval_plan.EXTRAPOLATION_BAND_FRACTION`.
   A tolerance is *resolvable* when ``u_reference + u_solver <= tolerance / 3``.
   The sum is deliberately linear (conservative) instead of quadrature.
5. **Unknown is not zero.**  A missing bound on either side makes the verdict
   ``UNDECIDABLE``; it can never produce a PASS.  A deviation whose *lower*
   bound already exceeds the tolerance is still a FAIL -- that conclusion is
   robust to the unknown size of the uncertainty.

6. **Model bounds are inflated, measurements are not.**  A model bound carries
   ``GCI_SAFETY_FACTOR`` (1.25) because it is an estimate of an error it cannot
   see; an error measured against an exact anchor is used as it is.

Everything below is pure arithmetic on numbers a study already reports, so a
frozen artifact can be re-judged without re-running a solver.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

#: Fraction of a tolerance the discretisation budget may consume (rule 4).
BUDGET_SHARE = 1.0 / 3.0

#: Safety factor applied to a *model* bound when it is used in place of a
#: measurement (Roache's GCI factor, 1.25).  ``docs/xval_bem_finite_depth_crossval_plan.md``
#: records the empirical justification: on the six finite-depth cells where an
#: exact anchor exists, the raw last-rung-change proxy under-estimated the
#: reference's true error by up to 2.1x, so even 1.25 is not a guarantee --
#: it is the floor a bound must be inflated by, and a *measured* error (see
#: ``scripts/finite_depth_reference_truth_audit.py``) always supersedes it.
GCI_SAFETY_FACTOR = 1.25

#: Verdicts.  ``UNDECIDABLE`` is a first-class outcome, not an error state.
PASS = "PASS"
FAIL = "FAIL"
UNDECIDABLE = "UNDECIDABLE"


def ladder_uncertainty(
    *,
    last_rung_change: float | None,
    order: float | None = None,
    refinement: float | None = None,
    min_order: float = 0.0,
    measured_band: float | None = None,
    measured_error: float | None = None,
    safety_factor: float = 1.0,
) -> dict[str, Any]:
    """Bound the remaining relative error of one ladder's reported value.

    Parameters
    ----------
    last_rung_change:
        ``|v_last - v_prev| / |v_last|`` -- the measured change between the two
        finest rungs.  ``None`` means "no ladder" and yields an unknown bound.
    order:
        Measured observed order ``p`` of the ladder (``None`` if the ladder is
        non-monotone / has fewer than three rungs).
    refinement:
        Panel refinement ratio between the two finest rungs (``N_last/N_prev``).
    min_order:
        The study's pre-registered floor below which the power-law model is not
        trusted (rule 3).
    measured_band:
        Independently measured uncertainty of the reported value, if any (e.g.
        a drop-coarsest-rung band on a four-rung ladder).
    measured_error:
        The ladder's error measured against an *exact* anchor (closed-form /
        semi-analytic).  A measurement supersedes every model bound here: the
        anchor has no discretisation of its own, so the residual belongs to the
        ladder alone.
    safety_factor:
        Inflation applied to a *model* bound (rule 6); ignored for measured
        quantities, which do not need one.

    Returns
    -------
    dict with ``value`` (the bound or ``None``), ``basis`` (how it was obtained),
    ``model_bound``, ``asymptotic`` and ``inputs`` for auditability.
    """
    out: dict[str, Any] = {
        "value": None,
        "basis": "unknown (no rung change recorded)",
        "model_bound": None,
        "measured_band": measured_band,
        "asymptotic": None,
        "inputs": {
            "last_rung_change": last_rung_change,
            "order": order,
            "refinement": refinement,
            "min_order": min_order,
        },
    }
    if measured_error is not None:
        out["value"] = abs(float(measured_error))
        out["basis"] = "measured against an exact anchor (no model bound needed)"
        out["model_bound"] = None
        out["asymptotic"] = None
        out["measured_error"] = abs(float(measured_error))
        return out
    if last_rung_change is None:
        if measured_band is not None:
            out["value"] = float(measured_band)
            out["basis"] = "measured band only (no rung change recorded)"
        return out
    change = abs(float(last_rung_change))
    asymptotic = (
        order is not None
        and refinement is not None
        and float(refinement) > 1.0
        and float(order) > 0.0
        and float(order) >= float(min_order)
    )
    out["asymptotic"] = bool(asymptotic)
    r_p = None
    if (
        order is not None
        and refinement is not None
        and float(refinement) > 1.0
        and float(order) > 0.0
    ):
        r_p = float(refinement) ** float(order)
    if r_p is not None and r_p > 1.0:
        # The power-law estimate is used even when the measured order sits
        # below the study's asymptotic floor: the audit of the finite-depth
        # reference (`scripts/finite_depth_reference_truth_audit.py`) shows the
        # raw last-rung change under-estimates the true error by up to 2.1x, so
        # the *smaller* of the two available estimates is never the safe one.
        out["model_bound"] = float(change / (r_p - 1.0))
        bound = float(out["model_bound"])
        basis = (
            f"last rung change / (r^p - 1), r={float(refinement):g}, "
            f"p={float(order):.3f}"
        )
        if not asymptotic:
            basis += (
                f" (p below the pre-registered floor {float(min_order):g}: "
                "outside the asymptotic range, hence not credited as small)"
            )
    else:
        bound = change
        basis = "last rung change (no usable measured order or rung ratio)"
    if measured_band is not None:
        # A fourth rung measures the extrapolant itself (drop-coarsest
        # re-extrapolation).  That measurement replaces the finest-rung model
        # bound -- otherwise the four-rung confirmation cells could never
        # improve a verdict, which is their whole purpose.
        basis = (
            "measured drop-coarsest band on a four-rung ladder (the "
            "extrapolant is verified, rule 2)"
        )
        bound = float(measured_band)
        out["model_bound"] = out["model_bound"]
    if safety_factor != 1.0 and float(safety_factor) > 0.0:
        bound = bound * float(safety_factor)
        basis = basis + f"; x{float(safety_factor):.2f} safety factor (rule 6)"
    out["value"] = float(bound)
    out["basis"] = basis
    return out


def total_uncertainty(
    reference: Mapping[str, Any] | float | None,
    solver: Mapping[str, Any] | float | None,
) -> dict[str, Any]:
    """Conservative (linear) sum of the two sides' bounds, or unknown."""
    def _value(side: Mapping[str, Any] | float | None) -> float | None:
        if side is None:
            return None
        if isinstance(side, Mapping):
            value = side.get("value")
            return None if value is None else float(value)
        return float(side)

    u_ref = _value(reference)
    u_sol = _value(solver)
    total = None if (u_ref is None or u_sol is None) else u_ref + u_sol
    return {
        "reference": u_ref,
        "solver": u_sol,
        "total": total,
        "known": total is not None,
        "combination": "linear (conservative) sum, rule 4",
    }


def decide(
    *,
    deviation: float,
    tolerance: float,
    reference: Mapping[str, Any] | float | None,
    solver: Mapping[str, Any] | float | None,
    share: float = BUDGET_SHARE,
) -> dict[str, Any]:
    """Three-valued verdict for one deviation at one tolerance.

    ``deviation`` is the relative difference of the reported values (signed is
    accepted; the magnitude is judged).
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not (share > 0.0 and share < 1.0):
        raise ValueError("share must lie strictly between 0 and 1")
    dev = abs(float(deviation))
    unc = total_uncertainty(reference, solver)
    budget = float(share) * float(tolerance)
    total = unc["total"]
    out: dict[str, Any] = {
        "deviation": float(deviation),
        "deviation_abs": dev,
        "tolerance": float(tolerance),
        "budget": budget,
        "budget_share": float(share),
        "uncertainty": unc,
        "budget_exceeded": None,
        "decidable": False,
        "verdict": UNDECIDABLE,
        "reason": "",
        "decidable_from_tolerance": None,
        "headroom": None,
        "binding_side": None,
        "remedy": None,
    }
    if total is None:
        missing = [
            name
            for name in ("reference", "solver")
            if unc[name] is None
        ]
        out["reason"] = (
            "uncertainty unknown on the "
            + "/".join(missing)
            + " side: unknown is not zero (rule 5)"
        )
        out["remedy"] = (
            "record a bound on the " + "/".join(missing) + " side (a fourth rung "
            "measures the extrapolant's own band; a missing ladder has no bound)"
        )
        return out
    out["budget_exceeded"] = bool(total > budget)
    out["binding_side"] = (
        "solver" if (unc["solver"] or 0.0) >= (unc["reference"] or 0.0) else "reference"
    )
    if out["budget_exceeded"]:
        out["remedy"] = (
            f"resolve at this tolerance needs {out['binding_side']} uncertainty "
            f"<= {float(share) * float(tolerance) / 2:.2%} (half the budget); "
            f"the {out['binding_side']} side is the binding one"
        )
    out["headroom"] = float(budget - total)
    out["decidable_from_tolerance"] = float(total / float(share))
    if dev - total > float(tolerance):
        # The lower bound of the deviation already violates the tolerance: no
        # possible value of the uncertainty can rescue a PASS, so the FAIL is
        # robust even when the budget is blown.
        out["decidable"] = True
        out["verdict"] = FAIL
        out["reason"] = (
            f"deviation {dev:.2%} exceeds the tolerance {float(tolerance):.2%} "
            f"by more than its own uncertainty {total:.2%}: FAIL is robust"
        )
        return out
    if out["budget_exceeded"]:
        out["reason"] = (
            f"uncertainty {total:.2%} exceeds the budget {budget:.2%} "
            f"({float(share):.2f} x {float(tolerance):.2%})"
        )
        return out
    out["decidable"] = True
    out["verdict"] = PASS if dev <= float(tolerance) else FAIL
    out["reason"] = (
        f"uncertainty {total:.2%} within budget {budget:.2%}; deviation "
        f"{dev:.2%} against tolerance {float(tolerance):.2%}"
    )
    return out


def required_refinement(
    uncertainty: Mapping[str, Any] | float | None,
    *,
    target: float,
    order: float | None,
    refinement: float | None,
) -> dict[str, Any]:
    """How much finer a ladder must be to bring its bound down to ``target``.

    Uses the same power-law model as :func:`ladder_uncertainty`:
    ``u(N) ~ N^-p``, so the panel count must grow by ``(u/target)^(1/p)``.  When
    the measured order is missing or non-positive the requirement is reported as
    unknown rather than infinite-by-assumption.
    """
    value = (
        None
        if uncertainty is None
        else (
            uncertainty.get("value")
            if isinstance(uncertainty, Mapping)
            else float(uncertainty)
        )
    )
    out: dict[str, Any] = {
        "current": None if value is None else float(value),
        "target": float(target),
        "order": order,
        "refinement_panels": None,
        "feasible": None,
        "note": "",
    }
    if value is None:
        out["note"] = "current bound unknown"
        return out
    if target <= 0.0:
        raise ValueError("target must be positive")
    if value <= target:
        out["refinement_panels"] = 1.0
        out["feasible"] = True
        out["note"] = "already within target"
        return out
    if order is None or float(order) <= 0.0:
        out["note"] = (
            "no positive measured order: the refinement needed to reach the "
            "target cannot be extrapolated from this ladder"
        )
        return out
    factor = (float(value) / float(target)) ** (1.0 / float(order))
    out["refinement_panels"] = float(factor)
    out["feasible"] = bool(math.isfinite(factor))
    out["note"] = (
        f"panels x{factor:.1f} at the measured order {float(order):.3f}"
        + (
            f" (current finest-rung ratio {float(refinement):g})"
            if refinement is not None
            else ""
        )
    )
    return out


def summarise(
    decisions: list[Mapping[str, Any]],
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-quantity decisions into one study-level statement."""
    counts = {PASS: 0, FAIL: 0, UNDECIDABLE: 0}
    for decision in decisions:
        counts[decision["verdict"]] = counts.get(decision["verdict"], 0) + 1
    decidable = counts[PASS] + counts[FAIL]
    return {
        "tolerance": tolerance,
        "counts": counts,
        "decisions": len(decisions),
        "decidable": bool(decisions) and counts[UNDECIDABLE] == 0,
        "all_pass": bool(decisions) and counts[PASS] == len(decisions),
        "any_fail": counts[FAIL] > 0,
        "note": (
            "a study is only decidable at this tolerance when no judged "
            "quantity is UNDECIDABLE"
        )
        if decidable or not decisions
        else "no quantity is decidable at this tolerance",
    }


__all__ = [
    "BUDGET_SHARE",
    "GCI_SAFETY_FACTOR",
    "PASS",
    "FAIL",
    "UNDECIDABLE",
    "ladder_uncertainty",
    "total_uncertainty",
    "decide",
    "required_refinement",
    "summarise",
]