"""Distribution-free upper bound on the Kramers-Kronig tail above the band top.

``tensorbem.causality.high_band_truncation_budget`` certifies a KK closure by
continuing the damping as ``B(s) = C s^-p`` and evaluating the *principal-value*
tail integral in closed form.  That closed form is only invoked when the
measured tail exponent clears ``p > 2.05``; below that bar the budget is
reported as infinite, and every sample of surge / sway / yaw on
CASE-BEM-CUMMINS-WIGLEY is therefore judged undecidable even though the KK
residuals there are 2.6 % / 2.8 % / 3.6 %.

"Infinite" is true of the *unconstrained* integral, but the tail is not
unconstrained: the damping is measured to decay at least as fast as the local
exponent at the band top.  Taking the sup of the kernel factor over the omitted
band gives a bound that needs nothing beyond ``p > 1``:

    |Delta(omega)| = (2/pi) |int_{s_max}^inf B(s)/(s^2 - omega^2) ds|
                  <= (2/pi) B(s_max) s_max^p / (s_max^2 - omega^2) * int_{s_max}^inf s^-p ds
                   = (2/pi) B(s_max) s_max / ((p - 1)(s_max^2 - omega^2)),

using ``1/(s^2 - omega^2) <= 1/(s_max^2 - omega^2)`` for ``s >= s_max > omega``.
No free parameter enters: ``B(s_max)`` is the last tabulated sample and ``p`` is
the measured exponent.

The bound is deliberately loose -- it prices the kernel at its worst value over
the whole omitted band instead of its actual decay -- so it is a *sufficient*
certificate, never an optimistic one: a residual that clears this bar is
attributable to the tabulated band alone.  It does not remove the modelling
assumption that the measured exponent persists above ``s_max``; that assumption
is what a wider sweep (``scripts/case_cummins_wigley_extended_band.py``) tests.

Because ``p`` is estimated from a finite window of tabulated data, the envelope is
rigorous only up to that estimation error; callers should read a bound far above
the tolerance (the horizontal modes land at 0.6-3.5 x max|A|) as the honest
signal "this DOF is not certified", which is exactly how it is used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tensorbem.causality import _as_sweep, _default_samples, local_tail_exponent
from tensorbem.causality import FloatArray  # re-use the causality module type alias

__all__ = ["SupTailBudget", "kernel_sup_tail_budget"]


@dataclass(frozen=True)
class SupTailBudget:
    """Upper bound on the omitted KK tail integral, per sample frequency."""

    sample_omegas: FloatArray
    budget: FloatArray
    exponent: float
    amplitude: float
    start: float
    admissible: bool
    reason: str

    @property
    def max_relative(self) -> float:
        """Worst budget divided by a caller-supplied scale is applied outside."""
        finite = self.budget[np.isfinite(self.budget)]
        return float(np.max(finite)) if finite.size else math.inf


def kernel_sup_tail_budget(
    omegas: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    sample_omegas: Sequence[float] | FloatArray | None = None,
    *,
    window: int = 3,
    min_exponent: float = 1.05,
) -> SupTailBudget:
    """Rigorous bound on ``(2/pi) int_{s_max}^inf B(s)/(s^2-omega^2) ds``.

    ``min_exponent`` guards the ``int s^-p ds`` convergence (``p > 1``) with a
    small margin; it is *not* a tolerance on the physics and is four times lower
    than the closed-form route's ``p > 2.05``.
    """
    grid, data = _as_sweep(omegas, damping, "damping")
    samples = (
        _default_samples(grid)
        if sample_omegas is None
        else np.asarray(list(sample_omegas), dtype=float).reshape(-1)
    )
    amplitude = float(abs(data[-1]))
    start = float(grid[-1])
    exponent, _ = local_tail_exponent(grid, data, window=window)
    if not (math.isfinite(exponent) and exponent > float(min_exponent)):
        return SupTailBudget(
            sample_omegas=samples,
            budget=np.full(samples.shape, math.inf, dtype=float),
            exponent=exponent,
            amplitude=amplitude,
            start=start,
            admissible=False,
            reason=(
                f"measured tail exponent {exponent:.3g} is not > {float(min_exponent):g}: "
                "the envelope int s^-p ds diverges, so no amplitude-only bound exists"
            ),
        )
    budget = np.empty(samples.shape, dtype=float)
    for index, sample in enumerate(samples):
        value = float(sample)
        if not value < start:
            budget[index] = math.inf
            continue
        budget[index] = (
            2.0
            / math.pi
            * amplitude
            * start
            / ((exponent - 1.0) * (start * start - value * value))
        )
    return SupTailBudget(
        sample_omegas=samples,
        budget=budget,
        exponent=exponent,
        amplitude=amplitude,
        start=start,
        admissible=True,
        reason=(
            f"kernel-sup envelope B(s) <= B(s_max) (s/s_max)^-{exponent:.3g} over s > "
            f"{start:g}; bound (2/pi) B(s_max) s_max /((p-1)(s_max^2-omega^2))"
        ),
    )