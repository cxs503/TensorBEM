"""Certification of the Kramers-Kronig truncation budget used by gate ``C3``.

The KK closure of a band-limited sweep is judged on the samples where
:func:`tensorbem.causality.high_band_truncation_budget` -- the closed-form
continuation ``B(s) = C s^-p`` above ``s_max`` -- predicts that the omitted
integral cannot explain the residual on its own.  That budget is an **upper**
bound on the omission only under the hypothesis

    (H)   ``B(s) <= B(s_max) (s/s_max)^-p``  for every ``s > s_max``,

with ``p`` the log-log slope of the last tabulated samples.  ``H`` is *not*
implied by band-limited data: nothing in ``[s_0, s_max]`` excludes a second
resonance or a slow re-growth immediately above the band top, and where that
happens the power-law integral is a lower bound, not an upper one.  The gate's
certification is therefore conditional, and this module turns the condition into
two measured numbers per sample frequency:

* **allowance** ``kappa = tolerance / budget_relative`` -- how much larger than
  its own continuation the real damping above ``s_max`` may be before the
  omitted band alone could move the residual past the case tolerance.  ``kappa``
  is ``0`` where the budget is unbounded (no certification at all) and grows as
  the continuation shrinks.
* **measured omission ratio** -- the omitted integral actually carried by a
  *wider-band probe*, normalised by the same budget.  The probe is integrated
  exactly on the frequencies it tabulates and continued in closed form beyond
  its own top, so the value reported as ``omission_lower`` is a strict *lower*
  bound on the true omission (it drops the band above the probe).

A certification survives iff the measured ratio stays below ``kappa`` *and* the
measured omission alone stays below the case tolerance; both are reported, so a
pass cannot be attributed to causality where the un-solved band could produce it
by itself.

Nothing here changes :func:`~tensorbem.causality.high_band_truncation_budget`;
this is the audit of what that bound is worth on real data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .causality import (
    FloatArray,
    _default_samples,
    high_band_truncation_budget,
    local_tail_exponent,
    power_tail_principal_value,
)

__all__ = [
    "TailOmission",
    "TailBudgetCertificate",
    "tail_principal_value_quadrature",
    "tail_omission",
    "certify_tail_budget",
]

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(24)


def _as_1d(values: Sequence[float] | FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 3:
        raise ValueError(f"{name} needs at least three samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def tail_principal_value_quadrature(
    omega: float,
    coefficient: float,
    power: float,
    start: float,
    *,
    panels: int = 400,
) -> float:
    """Independent quadrature of ``PV integral_start^inf C s^-p/(s^2-omega^2) ds``.

    Exists to be an oracle: the production path
    (:func:`tensorbem.causality.power_tail_principal_value`) evaluates the same
    integral as a series, and a series checked against itself is not checked.
    Here the pole at ``s = omega`` never enters the domain (``omega < start``)
    and the remaining integrand is smooth on ``[start, inf)`` after
    ``s = omega/t``, so composite Gauss-Legendre converges geometrically.
    """
    value = float(omega)
    coefficient = float(coefficient)
    power = float(power)
    upper = float(start)
    if not (value > 0.0 and math.isfinite(value)):
        raise ValueError("omega must be positive and finite")
    if not upper > value:
        raise ValueError("the tail must start above the sample frequency")
    if not power > 2.0:
        raise ValueError("the power tail must decay faster than s^-2")
    if panels < 1:
        raise ValueError("panels must be positive")
    t_max = value / upper
    edges = np.linspace(0.0, t_max, int(panels) + 1)
    total = 0.0
    for lower, upper_edge in zip(edges[:-1], edges[1:]):
        half = 0.5 * (upper_edge - lower)
        centre = 0.5 * (upper_edge + lower)
        t = half * _GL_NODES + centre
        # s = omega/t maps [start, inf) to t in (0, omega/start] with
        #   C s^-p ds/(s^2-omega^2) = -C omega^(-p-1) t^p/(1-t^2) dt,
        # and ``t <= omega/start < 1`` so the integrand is smooth on the panel.
        integrand = t**power / (1.0 - t * t)
        total += half * float(np.dot(_GL_WEIGHTS, integrand))
    return coefficient * value ** (-power - 1.0) * total


@dataclass(frozen=True)
class TailOmission:
    """Omitted KK integral measured from a wider-band probe (a lower bound)."""

    sample_omegas: FloatArray
    measured: FloatArray
    beyond_probe: FloatArray
    probe_band: tuple[float, float]
    probe_exponent: float
    probe_start: float
    c2_admissible_top: float

    @property
    def lower(self) -> FloatArray:
        """Measured part only: a strict lower bound on the true omission."""
        return self.measured

    @property
    def total(self) -> FloatArray:
        """Measured part plus the probe's own continuation (``inf`` if unbounded)."""
        return self.measured + self.beyond_probe

    def relative(self, scale: float, *, total: bool = False) -> FloatArray:
        floor = max(float(scale), 1e-300)
        values = self.total if total else self.lower
        return np.asarray(values, dtype=float) / floor


def tail_omission(
    omega: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    probe_omega: Sequence[float] | FloatArray,
    probe_damping: Sequence[float] | FloatArray,
    sample_omegas: Sequence[float] | FloatArray | None = None,
    *,
    window: int = 3,
    panels: int = 2000,
    gravity: float = 9.80665,
    panel_scale: float | None = None,
) -> TailOmission:
    """Integrate the omitted band ``[s_max, probe_top]`` from real probe damping.

    The probe must start at (or below) the top of the committed band and share
    its normalisation -- it is the same solver on a wider frequency grid.  On
    ``[s_max, probe_top]`` the interpolation is log-log between the probe nodes
    (the damping there is a smooth algebraic decay), so the reported value is
    quadrature error only.  Above ``probe_top`` the probe's own local slope is
    continued in closed form: if that slope is not ``> 2`` the omission is not
    bounded at all and ``beyond_probe`` is ``inf``, which is reported instead of
    silently assuming decay.

    ``c2_admissible_top`` records the highest frequency the probe mesh can
    actually resolve (``k*sqrt(A_panel) = 1``): above it the probe damping is
    under-resolved, so the measurement is an indication, not evidence.
    """
    grid = _as_1d(omega, "omega")
    data = _as_1d(damping, "damping")
    p_grid = _as_1d(probe_omega, "probe_omega")
    p_data = _as_1d(probe_damping, "probe_damping")
    if p_grid.size != p_data.size:
        raise ValueError("probe_omega and probe_damping must have the same length")
    if np.any(p_grid <= 0.0) or np.any(p_data <= 0.0):
        raise ValueError("probe frequencies and damping must be positive")
    if np.any(np.diff(p_grid) <= 0.0):
        raise ValueError("probe_omega must be increasing")
    start = float(grid[-1])
    probe_top = float(p_grid[-1])
    if not (p_grid[0] <= start * (1.0 + 1e-12) and probe_top > start):
        raise ValueError(
            "the probe must start at or below the top of the committed band and "
            "extend above it"
        )
    samples = (
        _default_samples(grid)
        if sample_omegas is None
        else _as_1d(sample_omegas, "sample_omegas")
    )
    keep = samples < min(start, probe_top)
    if not np.all(keep):
        raise ValueError("sample frequencies must lie inside the committed band")

    # Composite Gauss-Legendre in log(s): the damping is smooth in log-log and
    # the pole at s = omega lies below s_max, so no singularity is integrated.
    edges = np.logspace(math.log10(start), math.log10(probe_top), int(panels) + 1)
    measured = np.zeros(samples.size, dtype=float)
    for index, sample in enumerate(samples):
        value = float(sample)
        total = 0.0
        for lower, upper_edge in zip(edges[:-1], edges[1:]):
            half = 0.5 * (upper_edge - lower)
            centre = 0.5 * (upper_edge + lower)
            s = half * _GL_NODES + centre
            b = np.exp(np.interp(np.log(s), np.log(p_grid), np.log(p_data)))
            total += half * float(np.dot(_GL_WEIGHTS, b / (s * s - value * value)))
        measured[index] = 2.0 / math.pi * total

    exponent, _ = local_tail_exponent(p_grid, p_data, window=window)
    beyond = np.empty(samples.size, dtype=float)
    for index, sample in enumerate(samples):
        if math.isfinite(exponent) and exponent > 2.0:
            beyond[index] = 2.0 / math.pi * power_tail_principal_value(
                float(sample), float(p_data[-1]) * probe_top**exponent, exponent, probe_top
            )
        else:
            beyond[index] = math.inf
    c2_top = (
        math.sqrt(float(gravity) / float(panel_scale))
        if panel_scale is not None and float(panel_scale) > 0.0
        else math.inf
    )
    return TailOmission(
        sample_omegas=samples,
        measured=measured,
        beyond_probe=beyond,
        probe_band=(float(p_grid[0]), probe_top),
        probe_exponent=float(exponent),
        probe_start=start,
        c2_admissible_top=c2_top,
    )


@dataclass(frozen=True)
class TailBudgetCertificate:
    """What the ``C3`` decidable window is worth once the tail model is audited."""

    sample_omegas: FloatArray
    budget: FloatArray
    budget_relative: FloatArray
    omission_lower_relative: FloatArray
    omission_total_relative: FloatArray
    ratio: FloatArray
    allowance: FloatArray
    exponent: float
    tolerance: float
    budget_tolerance: float
    verdict: str
    reason: str

    @property
    def n_declared_decidable(self) -> int:
        finite = np.isfinite(self.budget_relative)
        return int(np.count_nonzero(finite & (self.budget_relative <= self.budget_tolerance)))

    @property
    def worst_ratio(self) -> float:
        finite = self.ratio[np.isfinite(self.ratio)]
        return float(np.max(finite)) if finite.size else math.inf

    @property
    def min_allowance(self) -> float:
        finite = self.allowance[np.isfinite(self.allowance)]
        return float(np.min(finite)) if finite.size else 0.0


def certify_tail_budget(
    omega: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    scale: float,
    *,
    sample_omegas: Sequence[float] | FloatArray | None = None,
    probe_omega: Sequence[float] | FloatArray | None = None,
    probe_damping: Sequence[float] | FloatArray | None = None,
    window: int = 3,
    tolerance: float = 0.10,
    budget_tolerance: float = 0.05,
    gravity: float = 9.80665,
    panel_scale: float | None = None,
) -> TailBudgetCertificate:
    """Audit one DOF's ``C3`` window: allowance, measured omission, verdict.

    ``verdict`` is the **worst** class over the samples of the committed band:

    ``certified``
        every sample with a finite budget also has a measured omission at or
        below its budget -- the continuation bounded the real tail on the
        evidence of the probe.
    ``certified_within_allowance``
        the measured omission exceeds the budget somewhere, but stays below the
        case tolerance, i.e. the certification survives the measured tail-model
        error (``ratio <= allowance``).
    ``falsified``
        the measured omission alone exceeds the case tolerance at some sample:
        the omitted band could produce the residual by itself, so the pass at
        that sample is not evidence of causality.
    ``uncertified``
        the budget is unbounded (damping not in algebraic decay at the band
        top), so no sample is certified; ``omission_lower_relative`` then only
        states how much of the scale the resolved tail already carries.
    """
    grid = _as_1d(omega, "omega")
    data = _as_1d(damping, "damping")
    samples = (
        _default_samples(grid)
        if sample_omegas is None
        else _as_1d(sample_omegas, "sample_omegas")
    )
    budget = high_band_truncation_budget(grid, data, samples, window=window)
    budget_relative = budget.relative(scale)
    allowance = np.where(
        np.isfinite(budget_relative) & (budget_relative > 0.0),
        float(tolerance) / np.where(budget_relative > 0.0, budget_relative, 1.0),
        0.0,
    )
    if probe_omega is None or probe_damping is None:
        omission = None
        lower_rel = np.full(samples.shape, np.nan)
        total_rel = np.full(samples.shape, np.nan)
    else:
        omission = tail_omission(
            grid,
            data,
            probe_omega,
            probe_damping,
            samples,
            window=window,
            gravity=gravity,
            panel_scale=panel_scale,
        )
        lower_rel = omission.relative(scale)
        total_rel = omission.relative(scale, total=True)
    ratio = np.where(
        np.isfinite(budget_relative) & (budget_relative > 0.0) & np.isfinite(lower_rel),
        lower_rel / np.where(budget_relative > 0.0, budget_relative, np.inf),
        np.nan,
    )

    bounded = np.isfinite(budget_relative)
    if not np.any(bounded):
        verdict = "uncertified"
        reason = (
            "the measured damping has not entered algebraic decay at s_max, so the "
            "power-law continuation is unbounded and no sample is certified; the "
            "measured tail omission is reported as a magnitude only"
        )
    elif np.any(np.isfinite(lower_rel) & (lower_rel > tolerance)):
        verdict = "falsified"
        reason = (
            "the probe-measured omitted integral alone exceeds the case tolerance "
            "at some sample, so the residual there cannot be attributed to causality"
        )
    elif np.any(np.isfinite(ratio) & (ratio > 1.0 + 1e-9)):
        verdict = "certified_within_allowance"
        reason = (
            "the measured tail exceeds the power-law budget somewhere but stays "
            "below the case tolerance: the certification survives the measured "
            "tail-model error"
        )
    else:
        verdict = "certified"
        reason = (
            "wherever a budget exists it also bounds the probe-measured tail, so "
            "the continuation is conservative on this evidence"
        )
    return TailBudgetCertificate(
        sample_omegas=samples,
        budget=budget.budget,
        budget_relative=budget_relative,
        omission_lower_relative=lower_rel,
        omission_total_relative=total_rel,
        ratio=ratio,
        allowance=allowance,
        exponent=float(budget.exponent),
        tolerance=float(tolerance),
        budget_tolerance=float(budget_tolerance),
        verdict=verdict,
        reason=reason,
    )