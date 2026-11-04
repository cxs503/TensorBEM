"""Verification & validation uncertainty quantification (GCI, error bars, CIs).

Motivation
----------
Every benchmark in this package reports *point estimates*: an observed order of
convergence ``p``, a Richardson-extrapolated value, a percentage error against a
reference.  A point estimate is not an uncertainty statement.  Reports that quote
"MAPE = 58.3 %" without an interval leave the reader unable to tell a systematic
bias from a five-point sampling accident, and the platform's W9 gate flags such
material as ``UNCERT`` (missing uncertainty / error bars / GCI).

This module supplies the missing layer with four independent, standard tools:

1. :func:`richardson_extrapolate` / :func:`grid_convergence_index` -- Roache's
   Grid Convergence Index (Roache 1994, 1998; ASME V&V 20 / Celik et al. 2008)
   on a three-level refinement study with constant refinement ratio ``r``:
   observed order ``p``, extrapolated value, ``GCI_fine`` at a factor of safety
   ``Fs``, the asymptotic-range ratio, and an oscillatory-convergence flag.
2. :func:`observed_order` -- the order of accuracy extracted from two levels.
3. :func:`bootstrap_ci` / :func:`bootstrap_mape` -- non-parametric percentile
   confidence interval for an aggregate error metric over a *small* sample of
   comparison points (``n`` of order ten, where normal-theory intervals are not
   defensible).
4. :func:`perturbation_band` -- Monte-Carlo propagation of a stated uncertainty
   in the *reference* values onto the error metric, which is how the
   "reference values themselves are uncertain" case is quantified without
   pretending the reference is exact.

Conventions
-----------
* Relative quantities are dimensionless fractions, not percents; the helper
  :func:`as_percent` converts for reporting.
* ``r`` is the refinement ratio ``h_coarse / h_fine``; all three levels share
  the same ``r`` (uniform refinement), which the GCI formula assumes.
* A negative or complex-looking ``p`` is *reported*, never silently repaired:
  :class:`GCIResult.converged` is ``False`` and the caller must not quote the
  extrapolated value as a converged answer.

References (no empirical coefficients are fitted here; ``Fs`` and ``p`` are the
only free inputs and both have published defaults):
* P. J. Roache, *Verification and Validation in Computational Science and
  Engineering*, Hermosa, 1998 (GCI, ``Fs = 1.25`` for three-grid studies).
* I. B. Celik et al., "Procedure for estimation and reporting of uncertainty due
  to discretization in CFD applications", ASME J. Fluids Eng. 130 (2008).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "GCIResult",
    "grid_convergence_index",
    "observed_order",
    "richardson_extrapolate",
    "percentile_ci",
    "bootstrap_ci",
    "bootstrap_mape",
    "mape",
    "perturbation_band",
    "as_percent",
]

DEFAULT_SAFETY_FACTOR = 1.25


# --------------------------------------------------------------------------- #
# Richardson extrapolation and the Grid Convergence Index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GCIResult:
    """Outcome of a three-level GCI study for one quantity.

    Attributes
    ----------
    fine, medium, coarse:
        The three solutions, finest mesh first.
    refinement_ratio:
        ``r = h_coarse / h_fine`` (uniform over the three levels).
    observed_order:
        ``p`` from the three-level Richardson formula; ``nan`` if it cannot be
        formed (e.g. two differences are exactly zero).
    extrapolated:
        Richardson-extrapolated (``h -> 0``) value, ``f1 + (f1-f2)/(r**p - 1)``.
    extrapolated_uncertainty:
        ``|f_ext - f_fine|``, i.e. how far the fine solution still is from the
        extrapolated limit; a direct estimate of the remaining discretization
        error of the fine level.
    gci_fine, gci_medium:
        Relative GCI (fraction of the respective value) at the factor of safety.
    asymptotic_ratio:
        ``GCI_medium / (r**p * GCI_fine)``; ``~1`` means the three levels are in
        the asymptotic range and ``p`` may be trusted.  Because the GCI is
        normalised by the local value, the exact identity for a pure power law is
        ``f_fine / f_medium``, so the tolerance is on ``|ratio - 1|``, not on
        equality.
    oscillatory:
        ``True`` when the consecutive differences change sign (``R < 0``); the
        observed order is then extracted from the magnitude ratio and the
        extrapolation is quoted but flagged.
    monotone, converged:
        ``monotone`` = successive differences of the same sign; ``converged`` is
        ``monotone and asymptotic_ratio`` within :data:`ASYMPTOTIC_TOL`.
    safety_factor:
        ``Fs`` used (Roache's 1.25 default for three grids).
    quantity, label:
        Free-form tags carried through to JSON for provenance.
    """

    fine: float
    medium: float
    coarse: float
    refinement_ratio: float
    observed_order: float
    extrapolated: float
    extrapolated_uncertainty: float
    gci_fine: float
    gci_medium: float
    asymptotic_ratio: float
    oscillatory: bool
    monotone: bool
    converged: bool
    safety_factor: float
    quantity: str = ""
    label: str = ""

    @property
    def band95(self) -> tuple[float, float]:
        """Symmetric 95 %-coverage interval around the fine solution.

        ``GCI`` at ``Fs = 1.25`` is already reported by Roache as a ~95 %
        coverage interval for a three-grid study, so the band is
        ``f_fine * (1 -/+ GCI_fine)``.
        """
        return (self.fine * (1.0 - self.gci_fine), self.fine * (1.0 + self.gci_fine))

    def as_dict(self) -> dict:
        d = asdict(self)
        lo, hi = self.band95
        d["band95"] = [lo, hi]
        return d


ASYMPTOTIC_TOL = 0.10


def observed_order(coarse: float, medium: float, fine: float,
                   refinement_ratio: float = 2.0) -> float:
    """Two-level observed order of accuracy from the difference ratio.

    ``p = ln|(f_medium - f_fine)/(f_coarse - f_medium)| / ln(r)``.
    Returns ``nan`` when the denominator vanishes (no measurable change) or when
    the three values are identical (already converged to machine precision).
    """
    d_fine = medium - fine
    d_coarse = coarse - medium
    if d_fine == 0.0:
        return float("nan")
    # consecutive differences shrink as r**-p, so the coarse/fine ratio gives p
    ratio = abs(d_coarse / d_fine)
    if ratio == 0.0:
        return float("nan")
    return math.log(ratio) / math.log(refinement_ratio)


def richardson_extrapolate(fine: float, medium: float, coarse: float,
                           refinement_ratio: float = 2.0) -> tuple[float, float]:
    """Return ``(f_extrapolated, p_observed)`` for three levels, finest first."""
    p = observed_order(coarse, medium, fine, refinement_ratio)
    if not math.isfinite(p):
        return (fine, p)
    denom = refinement_ratio ** p - 1.0
    if abs(denom) < 1e-14:
        return (fine, p)
    return (fine + (fine - medium) / denom, p)


def grid_convergence_index(fine: float, medium: float, coarse: float,
                           refinement_ratio: float = 2.0,
                           safety_factor: float = DEFAULT_SAFETY_FACTOR,
                           quantity: str = "", label: str = "") -> GCIResult:
    """Roache GCI for one quantity from a three-level, uniformly refined study.

    ``fine`` must be the most accurate (smallest ``h``) member.  The function is
    deliberately total: degenerate inputs yield ``nan`` order and
    ``converged = False`` rather than an exception, because a benchmark that
    spins on a non-monotone quantity should *report* that fact.
    """
    fine = float(fine)
    medium = float(medium)
    coarse = float(coarse)
    r = float(refinement_ratio)
    if r <= 1.0:
        raise ValueError("refinement_ratio must be > 1")

    d_fine = medium - fine
    d_coarse = coarse - medium
    oscillatory = (d_fine * d_coarse) < 0.0
    monotone = not oscillatory and (d_fine != 0.0) and (d_coarse != 0.0)

    if d_coarse != 0.0 and d_fine != 0.0:
        p = math.log(abs(d_coarse / d_fine)) / math.log(r)
    else:
        p = float("nan")

    if math.isfinite(p) and abs(r ** p - 1.0) > 1e-14:
        extrapolated = fine + (fine - medium) / (r ** p - 1.0)
        denom = r ** p - 1.0
        gci_fine = safety_factor * abs(d_fine / fine) / abs(denom) if fine else float("nan")
        gci_medium = safety_factor * abs(d_coarse / medium) / abs(denom) if medium else float("nan")
    else:
        extrapolated = fine
        gci_fine = float("nan")
        gci_medium = float("nan")

    if all(math.isfinite(v) for v in (gci_fine, gci_medium, p)) and gci_fine > 0.0:
        asymptotic_ratio = gci_medium / (r ** p * gci_fine)
    else:
        asymptotic_ratio = float("nan")

    converged = bool(
        monotone
        and math.isfinite(asymptotic_ratio)
        and abs(asymptotic_ratio - 1.0) <= ASYMPTOTIC_TOL
        and math.isfinite(p)
    )

    return GCIResult(
        fine=fine,
        medium=medium,
        coarse=coarse,
        refinement_ratio=r,
        observed_order=p,
        extrapolated=float(extrapolated),
        extrapolated_uncertainty=abs(float(extrapolated) - fine),
        gci_fine=float(gci_fine),
        gci_medium=float(gci_medium),
        asymptotic_ratio=float(asymptotic_ratio),
        oscillatory=bool(oscillatory),
        monotone=bool(monotone),
        converged=converged,
        safety_factor=float(safety_factor),
        quantity=quantity,
        label=label,
    )


# --------------------------------------------------------------------------- #
# Confidence intervals for aggregate error metrics
# --------------------------------------------------------------------------- #
def percentile_ci(samples: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    """Percentile confidence interval of ``samples`` (no distribution assumed)."""
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(arr, [100.0 * (1.0 - level) / 2.0, 100.0 * (1.0 + level) / 2.0])
    return (float(lo), float(hi))


def bootstrap_ci(data: Sequence[float], statistic: Callable[[np.ndarray], float],
                 n_boot: int = 4000, level: float = 0.95,
                 seed: int = 20261103) -> dict:
    """Non-parametric bootstrap percentile CI of ``statistic(data)``.

    Percentile (not BCa) because the sample sizes here are ~10 and the
    statistics are bounded ratios; the percentile interval is the conservative
    choice and needs no bias correction estimate that ten points cannot support.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    point = float(statistic(arr)) if n else float("nan")
    if n < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n": n, "n_boot": 0, "level": level, "seed": seed}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.array([statistic(arr[i]) for i in idx], dtype=float)
    lo, hi = percentile_ci(stats, level)
    return {"point": point, "lo": lo, "hi": hi, "n": n, "n_boot": int(n_boot),
            "level": float(level), "seed": int(seed)}


def mape(values: Sequence[float], reference: Sequence[float]) -> float:
    """Mean absolute percentage error (in percent) of ``values`` vs ``reference``.

    Points whose reference is zero are excluded (a relative error against an
    exact zero is undefined, not infinite).
    """
    v = np.asarray(values, dtype=float)
    r = np.asarray(reference, dtype=float)
    if v.shape != r.shape:
        raise ValueError("values and reference must have the same shape")
    mask = np.isfinite(v) & np.isfinite(r) & (r != 0.0)
    if not np.any(mask):
        return float("nan")
    return float(100.0 * np.mean(np.abs((v[mask] - r[mask]) / r[mask])))


def bootstrap_mape(values: Sequence[float], reference: Sequence[float],
                   n_boot: int = 4000, level: float = 0.95,
                   seed: int = 20261103) -> dict:
    """Bootstrap CI of the MAPE (percent) over paired points."""
    v = np.asarray(values, dtype=float)
    r = np.asarray(reference, dtype=float)
    mask = np.isfinite(v) & np.isfinite(r) & (r != 0.0)
    v, r = v[mask], r[mask]
    base = bootstrap_ci(np.arange(v.size), lambda idx: mape(v[idx.astype(int)], r[idx.astype(int)]),
                        n_boot=n_boot, level=level, seed=seed)
    # bootstrap over index arrays keeps the pairing
    return {"point": mape(v, r), "lo": base["lo"], "hi": base["hi"],
            "n": int(v.size), "n_boot": int(n_boot), "level": float(level),
            "seed": int(seed)}


def perturbation_band(values: Sequence[float], reference: Sequence[float],
                      rel_sigma: float = 0.05, distribution: str = "uniform",
                      n_samples: int = 20000, level: float = 0.95,
                      seed: int = 20261103) -> dict:
    """Monte-Carlo band of the MAPE under a stated uncertainty in the reference.

    The reference values of a digitised propeller chart are not exact; the
    caller states the belief (``rel_sigma`` as a uniform half-width or a normal
    standard deviation) and this returns the induced 95 % band of the error
    metric.  It answers "how much of the reported error could be reference
    error?" -- a question a point estimate cannot answer.

    Returns a dict with the nominal MAPE, the band, and the median MAPE over the
    samples (a first-order sensitivity check: a large pull of the median away
    from the nominal value means the metric is biased by reference uncertainty).
    """
    v = np.asarray(values, dtype=float)
    r = np.asarray(reference, dtype=float)
    if v.shape != r.shape:
        raise ValueError("values and reference must have the same shape")
    mask = np.isfinite(v) & np.isfinite(r) & (r != 0.0)
    v, r = v[mask], r[mask]
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        noise = rng.uniform(-rel_sigma, rel_sigma, size=(n_samples, r.size))
    elif distribution == "normal":
        noise = rng.normal(0.0, rel_sigma, size=(n_samples, r.size))
    else:
        raise ValueError("distribution must be 'uniform' or 'normal'")
    ref_s = r[None, :] * (1.0 + noise)
    err = np.abs((v[None, :] - ref_s) / ref_s)
    stats = 100.0 * np.mean(err, axis=1)
    lo, hi = percentile_ci(stats, level)
    return {
        "nominal": mape(v, r),
        "lo": lo,
        "hi": hi,
        "median": float(np.median(stats)),
        "rel_sigma": float(rel_sigma),
        "distribution": distribution,
        "n_samples": int(n_samples),
        "n_points": int(r.size),
        "level": float(level),
        "seed": int(seed),
    }


def as_percent(fraction: float) -> float:
    """Convert a dimensionless fraction to percent (``nan`` passes through)."""
    return 100.0 * fraction if isinstance(fraction, (int, float)) else float("nan")