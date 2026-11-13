"""Attribute a Kramers-Kronig residual to asymptote bias versus shape error.

Why this module exists
----------------------
The hydrodynamic Kramers-Kronig (KK) relation

    A(w) - A_inf = (2/pi) PV int_0^inf B(s) / (s^2 - w^2) ds                (1)

is the causality test used by the case-pool anchor ``CASE-BEM-CUMMINS-WIGLEY``,
and its residual is dominated by a quantity that a frequency sweep does *not*
contain: the added mass at infinite frequency, ``A_inf``.  ``A_inf`` is
extrapolated from the upper tail of the sweep, and that extrapolation is biased
whenever the top of the band is not yet asymptotic.  Because the residual is
normalised by ``max |A - A_inf|`` -- a *difference* -- a bias of a few percent
of ``A`` becomes tens of percent of the reported residual.

The consequence is a diagnostic trap: a large KK residual is routinely read as
"the panel solution is not causal", while the same residual can be produced by
an exactly causal pair whose ``A_inf`` was mis-extrapolated.  Two quantities
therefore have to be separated:

* **shape residual** -- the part no choice of ``A_inf`` can remove.  A constant
  added to ``A`` is invisible to eq. (1) (it integrates to zero against the
  principal-value kernel), so *this* is the physical causality statement.
* **offset bias** -- the constant the sweep implies for ``A_inf`` minus the
  constant the tail fit supplied.  It belongs to the asymptote estimator, not
  to the solver.

:func:`kkr_decomposition` reports both.  The offset is the Chebyshev (minimax)
centre of the pointwise residual, so ``residual_shape`` is the least
infinity-norm residual attainable by *any* constant asymptote -- an
estimator-independent floor -- and it is reported against the same normaliser
the declared gate uses, so the two numbers are directly comparable.

:func:`null_calibration` closes the loop with an *exactly causal* control: a
damped-oscillator memory pair, fitted to the measured damping and sampled on the
same frequency grid, is pushed through the same estimator chain.  Whatever
residual it produces is the estimator floor and cannot be blamed on the solver.
For ``K(s) = c exp(-a s) cos(b s)`` both KK integrals are exact in closed form,

    B(w)     = c/2 [ a/(a^2+(w-b)^2) + a/(a^2+(w+b)^2) ]
    A(w)-A_inf = -c/(2w) [ (w+b)/(a^2+(w+b)^2) + (w-b)/(a^2+(w-b)^2) ] ,

and the family (unlike a pure exponential sum) can reproduce a *peaked*
damping curve, which is what a resonant roll/heave sweep looks like.

:func:`sum_rule_asymptote` supplies a second, independent estimate of ``A_inf``
from the ``w -> inf`` limit of eq. (1),

    A(w) - A_inf = -(2/pi) w^-2 int_0^inf B(s) ds + O(w^-4),                (2)

i.e. the ``1/w^2`` coefficient of the added-mass tail is fixed by the damping
sweep alone.  Fitting only the constant against that prescribed slope is a
one-parameter problem and is far better conditioned than the two-parameter tail
fit.  It is not used to re-judge a gate silently: the caller gets both numbers
and the residual each one produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem.causality import (
    band_integral,
    fit_added_mass_asymptote,
    principal_value_integral,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "KKRAttribution",
    "chebyshev_centre",
    "fit_causal_surrogate",
    "kkr_decomposition",
    "null_calibration",
    "null_grid_convergence",
    "oscillator_kernel_pair",
    "sum_rule_asymptote",
]

_MINIMUM_POINTS = 8


def _validate(
    omegas: Sequence[float] | FloatArray,
    values: Sequence[float] | FloatArray,
    name: str,
) -> tuple[FloatArray, FloatArray]:
    grid = np.asarray(omegas, dtype=float).reshape(-1)
    data = np.asarray(values, dtype=float).reshape(-1)
    if grid.size != data.size:
        raise ValueError(f"omegas and {name} must have the same length")
    if grid.size < _MINIMUM_POINTS:
        raise ValueError(f"at least {_MINIMUM_POINTS} frequencies are required")
    if not (np.all(np.isfinite(grid)) and np.all(np.isfinite(data))):
        raise ValueError(f"omegas and {name} must be finite")
    if np.any(grid <= 0.0):
        raise ValueError("omegas must be strictly positive")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("omegas must be strictly increasing")
    return grid, data


def _below_band_term(
    grid: FloatArray, beta: FloatArray, omega: float, model: str
) -> float:
    """``int_0^omega_min B(s)/(s^2-omega^2) ds`` under a stated low-frequency model.

    ``"constant"`` is the closure :func:`tensorbem.causality.principal_value_integral`
    applies (holds ``B`` at its first tabulated value); it is the right model for
    a memory kernel with non-zero mean, and it is the wrong one for a floating
    body, whose damping vanishes at zero frequency.  ``"zero"`` assumes
    ``B ~ s^n`` with ``n >= 1``, for which the band contributes nothing to this
    integrand.  ``"linear"`` uses ``B(s) = B0 * s / s0``; the singular-free
    integral is evaluated by Gauss-Legendre because ``omega > omega_min``
    everywhere the gate samples.
    """
    first = float(grid[0])
    if model == "constant":
        return 0.5 * float(beta[0]) / omega * math.log(abs(first - omega) / (first + omega))
    if model == "zero":
        return 0.0
    if model == "linear":
        nodes, weights = np.polynomial.legendre.leggauss(64)
        s = 0.5 * first * (nodes + 1.0)
        w = 0.5 * first * weights
        values = float(beta[0]) * s / first
        return float(np.sum(w * values / (s**2 - omega**2)))
    raise ValueError(f"unknown below-band model {model!r}")


def chebyshev_centre(deviation: Sequence[float] | FloatArray) -> float:
    """Minimax constant for ``deviation``: the one that halves its range.

    Subtracting ``chebyshev_centre(d)`` leaves a vector whose largest magnitude
    is exactly half the peak-to-peak range of ``d``, which is the smallest
    attainable infinity norm for a free constant.
    """
    data = np.asarray(deviation, dtype=float).reshape(-1)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("deviation must be a non-empty finite vector")
    return 0.5 * (float(data.max()) + float(data.min()))


def oscillator_kernel_pair(
    omegas: Sequence[float] | FloatArray,
    rates: Sequence[float] | FloatArray,
    frequencies: Sequence[float] | FloatArray,
    coefficients: Sequence[float] | FloatArray,
    *,
    added_mass_inf: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Exact causal pair of the damped-oscillator memory family.

    The memory kernel ``K(s) = sum_j c_j exp(-a_j s) cos(b_j s)`` (``c_j > 0``,
    ``a_j > 0``, ``b_j >= 0``) is real, causal and decaying, so its radiation
    coefficients are one Kramers-Kronig pair by construction:

        B(w)       = sum_j c_j/2 [ a_j/(a_j^2+(w-b_j)^2) + a_j/(a_j^2+(w+b_j)^2) ]
        A(w)-A_inf = -sum_j c_j/(2w) [ (w+b_j)/(a_j^2+(w+b_j)^2)
                                      + (w-b_j)/(a_j^2+(w-b_j)^2) ] .

    ``b_j = 0`` recovers the pure exponential-sum family.  Every term is a
    strictly positive bump in ``B`` around ``w = b_j``, so a peaked damping curve
    is representable -- which a monotone exponential sum cannot do.
    """
    grid = np.asarray(omegas, dtype=float).reshape(-1)
    rate = np.asarray(rates, dtype=float).reshape(-1)
    freq = np.asarray(frequencies, dtype=float).reshape(-1)
    coef = np.asarray(coefficients, dtype=float).reshape(-1)
    if not (rate.size == freq.size == coef.size):
        raise ValueError("rates, frequencies and coefficients must have equal length")
    if np.any(grid <= 0.0) or np.any(rate <= 0.0) or np.any(freq < 0.0):
        raise ValueError("omegas and rates must be positive, frequencies >= 0")
    upper = grid[:, None] + freq[None, :]
    lower = grid[:, None] - freq[None, :]
    damping = 0.5 * (
        (rate[None, :] / (rate[None, :] ** 2 + upper**2))
        + (rate[None, :] / (rate[None, :] ** 2 + lower**2))
    ) @ coef
    odd = 0.5 * (
        upper / (rate[None, :] ** 2 + upper**2)
        + lower / (rate[None, :] ** 2 + lower**2)
    ) @ coef
    added = float(added_mass_inf) - odd / grid
    return added, damping


def fit_causal_surrogate(
    omegas: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    *,
    rates: Sequence[float] | None = None,
    frequencies: Sequence[float] | None = None,
) -> dict:
    """Fit the exact causal family to a measured damping curve.

    A non-negative least-squares fit (so ``B >= 0``: physically admissible
    damping) over a fixed grid of ``(a_j, b_j)`` pairs.  The added mass is then
    *not* fitted -- it follows from the family identity -- and ``A_inf`` is a
    free constant that the caller prescribes.  The result is the calibration
    vehicle of :func:`null_calibration`.
    """
    grid, data = _validate(omegas, damping, "damping")
    rate_grid = (np.geomspace(0.15, 16.0, 9) if rates is None
                 else np.asarray(rates, dtype=float).reshape(-1))
    freq_grid = (np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.8, 3.6, 4.5, 5.5, 7.0, 9.0,
                           11.0, 13.0]) if frequencies is None
                 else np.asarray(frequencies, dtype=float).reshape(-1))
    if rate_grid.size == 0 or freq_grid.size == 0:
        raise ValueError("the surrogate needs a non-empty rate/frequency grid")
    rate = np.repeat(rate_grid, freq_grid.size)
    freq = np.tile(freq_grid, rate_grid.size)
    upper = grid[:, None] + freq[None, :]
    lower = grid[:, None] - freq[None, :]
    design = 0.5 * (
        rate[None, :] / (rate[None, :] ** 2 + upper**2)
        + rate[None, :] / (rate[None, :] ** 2 + lower**2)
    )
    from scipy.optimize import nnls

    coeff, norm = nnls(design, data)
    model = design @ coeff
    scale = max(float(np.max(np.abs(data))), 1e-300)
    active = coeff > 1e-12 * max(float(coeff.max()), 1e-300)
    return {
        "rates": rate[active].tolist(),
        "frequencies": freq[active].tolist(),
        "coefficients": coeff[active].tolist(),
        "n_basis": int(rate.size),
        "n_active": int(np.count_nonzero(active)),
        "relative_residual": float(np.max(np.abs(model - data)) / scale),
        "l2_residual": float(norm / max(np.linalg.norm(data), 1e-300)),
        "model_damping": model.tolist(),
        "omega": grid.tolist(),
    }


@dataclass(frozen=True)
class KKRAttribution:
    """Split of a KK residual into an asymptote offset and a shape residual.

    ``residual_total`` is the declared gate quantity (``A_inf`` from the tail
    fit, normaliser ``max|A - A_inf|``).  ``residual_shape`` uses the *same*
    normaliser and the minimax-optimal constant, so

        residual_total**2 >= residual_shape**2 + residual_offset**2

    is not implied -- the identity that does hold is that the pointwise total is
    the pointwise shape plus the offset.  ``scale_implied`` is reported because
    the dynamic range of ``A`` depends on which asymptote one believes.
    """

    omega: FloatArray
    added_mass_measured: FloatArray
    predicted_offsetless: FloatArray
    added_mass_inf_used: float
    added_mass_inf_implied: float
    offset_bias: float
    scale_used: float
    scale_implied: float
    residual_total: float
    residual_shape: float
    residual_offset: float
    pointwise_total: FloatArray
    pointwise_shape: FloatArray

    @property
    def residual_shape_on_implied_scale(self) -> float:
        return float(np.max(np.abs(self.pointwise_shape)) / self.scale_implied)

    @property
    def residual_offset_on_implied_scale(self) -> float:
        return float(abs(self.offset_bias) / self.scale_implied)

    @property
    def bias_over_scale_used(self) -> float:
        return float(self.offset_bias / self.scale_used)

    def as_dict(self) -> dict:
        return {
            "omega_query": self.omega.tolist(),
            "added_mass_measured": self.added_mass_measured.tolist(),
            "predicted_offsetless": self.predicted_offsetless.tolist(),
            "added_mass_inf_used": float(self.added_mass_inf_used),
            "added_mass_inf_implied": float(self.added_mass_inf_implied),
            "offset_bias": float(self.offset_bias),
            "bias_over_scale_used": self.bias_over_scale_used,
            "scale_used": float(self.scale_used),
            "scale_implied": float(self.scale_implied),
            "residual_total": float(self.residual_total),
            "residual_shape": float(self.residual_shape),
            "residual_shape_on_implied_scale": self.residual_shape_on_implied_scale,
            "residual_offset": float(self.residual_offset),
            "residual_offset_on_implied_scale": self.residual_offset_on_implied_scale,
            "pointwise_total": self.pointwise_total.tolist(),
            "pointwise_shape": self.pointwise_shape.tolist(),
        }


def kkr_decomposition(
    omegas: Sequence[float] | FloatArray,
    added_mass: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    *,
    added_mass_inf: float | None = None,
    tail_fraction: float = 0.25,
    below_band_model: str = "constant",
) -> KKRAttribution:
    """KK residual at the sweep midpoints, split into offset and shape.

    ``added_mass_inf=None`` reproduces the estimator the declared gate uses (the
    two-parameter tail fit of :func:`tensorbem.causality.fit_added_mass_asymptote`);
    pass a value to test a different asymptote.  The offset is the Chebyshev
    centre of the pointwise residual, so ``residual_shape`` is the least
    infinity-norm residual attainable by *any* constant asymptote.
    """
    grid, added = _validate(omegas, added_mass, "added_mass")
    _, beta = _validate(omegas, damping, "damping")
    if added_mass_inf is None:
        added_mass_inf = float(
            fit_added_mass_asymptote(grid, added, tail_fraction=tail_fraction)
            .added_mass_inf
        )
    a_inf = float(added_mass_inf)
    if not math.isfinite(a_inf):
        raise ValueError("added_mass_inf must be finite")

    query = 0.5 * (grid[:-1] + grid[1:])
    measured = 0.5 * (added[:-1] + added[1:])
    predicted = np.array([
        2.0 / math.pi * (
            principal_value_integral(grid, beta, float(w))
            - _below_band_term(grid, beta, float(w), "constant")
            + _below_band_term(grid, beta, float(w), below_band_model)
        )
        for w in query
    ])
    pointwise_total = a_inf + predicted - measured
    bias = chebyshev_centre(pointwise_total)
    pointwise_shape = pointwise_total - bias
    a_inf_implied = a_inf - bias

    scale_used = max(float(np.max(np.abs(measured - a_inf))), 1e-300)
    scale_implied = max(float(np.max(np.abs(measured - a_inf_implied))), 1e-300)
    return KKRAttribution(
        omega=query,
        added_mass_measured=measured,
        predicted_offsetless=predicted,
        added_mass_inf_used=a_inf,
        added_mass_inf_implied=float(a_inf_implied),
        offset_bias=float(bias),
        scale_used=scale_used,
        scale_implied=scale_implied,
        residual_total=float(np.max(np.abs(pointwise_total)) / scale_used),
        residual_shape=float(np.max(np.abs(pointwise_shape)) / scale_used),
        residual_offset=float(abs(bias) / scale_used),
        pointwise_total=pointwise_total,
        pointwise_shape=pointwise_shape,
    )


def null_calibration(
    omegas: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    *,
    added_mass_inf: float,
    rates: Sequence[float] | None = None,
    frequencies: Sequence[float] | None = None,
    tail_fraction: float = 0.25,
    below_band_model: str = "constant",
) -> dict:
    """Estimator floor: the same chain applied to an *exactly* causal pair.

    The damped-oscillator surrogate is fitted to the measured damping (same
    magnitude, same peak location), its added mass follows analytically, and its
    ``A_inf`` is known because the caller prescribes it.  Residuals reported
    here are properties of the grid, the quadrature and the tail fit alone.
    """
    grid, _ = _validate(omegas, damping, "damping")
    surrogate = fit_causal_surrogate(
        grid, damping, rates=rates, frequencies=frequencies)
    added_sur, damping_sur = oscillator_kernel_pair(
        grid, surrogate["rates"], surrogate["frequencies"],
        surrogate["coefficients"], added_mass_inf=float(added_mass_inf),
    )
    with_fit = kkr_decomposition(
        grid, added_sur, damping_sur, tail_fraction=tail_fraction,
        below_band_model=below_band_model)
    with_oracle = kkr_decomposition(
        grid, added_sur, damping_sur, added_mass_inf=float(added_mass_inf),
        below_band_model=below_band_model)
    true_scale = max(
        float(np.max(np.abs(added_sur - float(added_mass_inf)))), 1e-300)
    return {
        "surrogate": {
            "n_basis": surrogate["n_basis"],
            "n_active": surrogate["n_active"],
            "damping_relative_residual": surrogate["relative_residual"],
            "damping_l2_residual": surrogate["l2_residual"],
            "rates": surrogate["rates"],
            "frequencies": surrogate["frequencies"],
            "coefficients": surrogate["coefficients"],
            "added_mass_inf_true": float(added_mass_inf),
            "added_mass_range": [
                float(np.min(added_sur)), float(np.max(added_sur))],
        },
        "tail_fit": {
            "added_mass_inf_fitted": float(with_fit.added_mass_inf_used),
            "bias": float(with_fit.added_mass_inf_used - float(added_mass_inf)),
            "bias_over_true_scale": float(
                (with_fit.added_mass_inf_used - float(added_mass_inf)) / true_scale),
            "residual_total": float(with_fit.residual_total),
            "residual_shape": float(with_fit.residual_shape),
            "residual_offset": float(with_fit.residual_offset),
        },
        "oracle_asymptote": {
            "added_mass_inf_used": float(added_mass_inf),
            "residual_total": float(with_oracle.residual_total),
            "residual_shape": float(with_oracle.residual_shape),
            "residual_offset": float(with_oracle.residual_offset),
        },
        "grid": {
            "n": int(grid.size),
            "omega_min": float(grid[0]),
            "omega_max": float(grid[-1]),
        },
    }


def sum_rule_asymptote(
    omegas: Sequence[float] | FloatArray,
    added_mass: Sequence[float] | FloatArray,
    damping: Sequence[float] | FloatArray,
    *,
    tail_points: int = 6,
) -> dict:
    """``A_inf`` from the ``w -> inf`` sum rule, with the slope taken from ``B``.

    Eq. (2) fixes the leading ``1/w^2`` coefficient of the added-mass tail at
    ``c = -(2/pi) int_0^inf B ds``, which contains no added-mass information at
    all.  Holding that slope and solving for the single remaining unknown gives
    an asymptote that is independent of the two-parameter tail fit; the residual
    it produces on the declared KK quantity is returned next to it.
    """
    grid, added = _validate(omegas, added_mass, "added_mass")
    _, beta = _validate(omegas, damping, "damping")
    count = int(min(max(int(tail_points), 2), grid.size))
    coefficient = -2.0 / math.pi * band_integral(grid, beta)
    window = grid[-count:]
    a_inf = float(np.mean(added[-count:] - coefficient / window**2))
    decomposition = kkr_decomposition(grid, added, beta, added_mass_inf=a_inf)
    tail_fit = float(fit_added_mass_asymptote(grid, added).added_mass_inf)
    return {
        "tail_coefficient": float(coefficient),
        "tail_points": count,
        "tail_window": [float(window[0]), float(window[-1])],
        "added_mass_inf_sum_rule": a_inf,
        "added_mass_inf_tail_fit": tail_fit,
        "added_mass_inf_difference": float(a_inf - tail_fit),
        "added_mass_inf_implied_by_kk": float(decomposition.added_mass_inf_implied),
        "residual_total": float(decomposition.residual_total),
        "residual_shape": float(decomposition.residual_shape),
        "residual_offset": float(decomposition.residual_offset),
    }

def null_grid_convergence(
    rates: Sequence[float] | FloatArray,
    frequencies: Sequence[float] | FloatArray,
    coefficients: Sequence[float] | FloatArray,
    *,
    added_mass_inf: float,
    node_counts: Sequence[int] = (22, 44, 88, 176),
    band: tuple[float, float] = (0.3, 12.5),
    tail_fraction: float = 0.25,
    below_band_model: str = "constant",
) -> dict:
    """How the estimator floor shrinks as the *frequency grid* is refined.

    The surrogate pair is exact, so anything the decomposition reports at a given
    node count is quadrature and asymptote estimation.  The sweep of node counts
    separates the two: with the true ``A_inf`` supplied ("oracle") the residual
    is pure principal-value quadrature; with the tail fit it also carries the
    asymptote bias.  This is the calibration behind the claim that a KK residual
    above tolerance does not, by itself, mean the panel solution is
    non-causal -- and it is also the cheapest remedy, because refining the
    frequency grid costs BEM solves while refining the mesh does not touch it.
    """
    low, high = (float(v) for v in band)
    if not (0.0 < low < high):
        raise ValueError("band must satisfy 0 < low < high")
    rows = []
    for count in node_counts:
        if int(count) < _MINIMUM_POINTS:
            raise ValueError(f"node_counts must be >= {_MINIMUM_POINTS}")
        grid = np.geomspace(low, high, int(count))
        added, damping = oscillator_kernel_pair(
            grid, rates, frequencies, coefficients,
            added_mass_inf=float(added_mass_inf))
        oracle = kkr_decomposition(
            grid, added, damping, added_mass_inf=float(added_mass_inf),
            below_band_model=below_band_model)
        fitted = kkr_decomposition(
            grid, added, damping, tail_fraction=tail_fraction,
            below_band_model=below_band_model)
        rows.append({
            "nodes": int(count),
            "omega_min": float(grid[0]),
            "omega_max": float(grid[-1]),
            "shape_residual_oracle": float(oracle.residual_shape),
            "offset_residual_oracle": float(oracle.residual_offset),
            "total_residual_tail_fit": float(fitted.residual_total),
            "shape_residual_tail_fit": float(fitted.residual_shape),
            "offset_bias": float(fitted.offset_bias),
            "added_mass_inf_fitted": float(fitted.added_mass_inf_used),
            "added_mass_inf_true": float(added_mass_inf),
        })
    return {
        "band": [low, high],
        "added_mass_inf_true": float(added_mass_inf),
        "rows": rows,
    }
