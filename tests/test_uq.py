"""Tests for :mod:`tensorbem.uq` (GCI / Richardson / confidence intervals).

Every check here is against a *manufactured* answer whose value is known
independently of the implementation:

* a synthetic sequence ``f(h) = f_exact + C h^p`` with known ``p`` -- the GCI
  implementation must recover ``p`` and ``f_exact``;
* a sample drawn from a known distribution -- the percentile CI must cover the
  population statistic at the stated rate;
* a multiplicative offset of exactly 10 % -- the MAPE must return 10 % and its
  bootstrap interval must bracket it.

None of these tests compares a quantity with a number produced by the same code
path that computed it (the circularity trap that report 3578 documents for the
propeller table chain).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.uq import (
    GCIResult,
    as_percent,
    bootstrap_ci,
    bootstrap_mape,
    grid_convergence_index,
    mape,
    observed_order,
    percentile_ci,
    perturbation_band,
    richardson_extrapolate,
)


# --------------------------------------------------------------------------- #
# Richardson extrapolation and GCI
# --------------------------------------------------------------------------- #
def test_richardson_recovers_manufactured_order_and_limit():
    """f(h) = 1.5 + 0.8 h^2, h = 1, 1/2, 1/4 -> p = 2 and f_ext = 1.5."""
    exact, c, p = 1.5, 0.8, 2.0

    def f(h):
        return exact + c * h ** p

    coarse, medium, fine = f(0.25), f(0.125), f(0.0625)   # r = 2, finest last
    ext, p_obs = richardson_extrapolate(fine, medium, coarse, refinement_ratio=2.0)
    assert p_obs == pytest.approx(p, rel=1e-10)
    assert ext == pytest.approx(exact, rel=1e-10)


def test_gci_band_asymptotic_and_converged_flags():
    """A pure power law is in the asymptotic range: ratio ~ 1, band symmetric."""
    exact, c = 2.0, 1.0
    vals = [exact + c * h ** 1.5 for h in (0.4, 0.2, 0.1)]
    res = grid_convergence_index(vals[2], vals[1], vals[0], refinement_ratio=2.0,
                                 quantity="KT", label="manufactured")
    assert isinstance(res, GCIResult)
    assert res.observed_order == pytest.approx(1.5, rel=1e-8)
    # the ratio equals f_fine / f_medium for a pure power law: the relative
    # normalisation of GCI is the only reason it is not exactly one
    assert res.asymptotic_ratio == pytest.approx(res.fine / res.medium, rel=1e-8)
    assert res.converged is True
    assert res.monotone is True and res.oscillatory is False
    assert res.extrapolated == pytest.approx(exact, rel=1e-8)
    lo, hi = res.band95
    assert lo < res.fine < hi
    assert (res.fine - lo) == pytest.approx(hi - res.fine)
    # GCI scales as Fs * |error of the fine level| / (r^p - 1) -> positive
    assert res.gci_fine > 0.0
    # GCI_medium uses the (coarse, medium) pair, i.e. the larger error estimate:
    # GCI_medium / GCI_fine = r**p * f_fine / f_medium > 1
    assert res.gci_medium > res.gci_fine
    assert res.gci_medium / res.gci_fine == pytest.approx(
        2.0 ** 1.5 * res.fine / res.medium, rel=1e-8)


def test_gci_safety_factor_scales_linearly():
    vals = [1.0 + 0.3 * h ** 2 for h in (0.4, 0.2, 0.1)]
    a = grid_convergence_index(vals[2], vals[1], vals[0], safety_factor=1.25)
    b = grid_convergence_index(vals[2], vals[1], vals[0], safety_factor=2.50)
    assert b.gci_fine == pytest.approx(2.0 * a.gci_fine, rel=1e-12)


def test_gci_flags_oscillatory_convergence():
    """Alternating differences must be reported, not repaired."""
    res = grid_convergence_index(1.00, 1.01, 0.99, refinement_ratio=2.0)
    assert res.oscillatory is True
    assert res.monotone is False
    assert res.converged is False


def test_gci_degenerate_inputs_do_not_raise():
    """Three identical values: no measurable change -> nan order, not converged."""
    res = grid_convergence_index(3.0, 3.0, 3.0)
    assert not math.isfinite(res.observed_order)
    assert res.converged is False
    assert res.extrapolated == pytest.approx(3.0)


def test_gci_rejects_non_refinement_ratio():
    with pytest.raises(ValueError):
        grid_convergence_index(1.0, 2.0, 3.0, refinement_ratio=1.0)


def test_observed_order_two_level_matches_power_law():
    """h = 1, 1/2 with error ~ h^3: the two-level order must return 3."""
    f = lambda h: 10.0 + 0.2 * h ** 3
    assert observed_order(f(1.0), f(0.5), f(0.25), 2.0) == pytest.approx(3.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def test_percentile_ci_of_uniform_sample_covers_the_mean():
    rng = np.random.default_rng(7)
    sample = rng.uniform(-1.0, 1.0, size=4000)
    lo, hi = percentile_ci(sample, level=0.95)
    assert lo < 0.0 < hi
    assert lo == pytest.approx(-0.95, abs=0.06)
    assert hi == pytest.approx(0.95, abs=0.06)


def test_bootstrap_ci_of_the_mean_has_empirical_coverage():
    """Across 200 synthetic samples the 95 % interval must cover 0 about 95 %."""
    rng = np.random.default_rng(11)
    covered = 0
    trials = 200
    for t in range(trials):
        data = rng.normal(0.0, 1.0, size=12)
        res = bootstrap_ci(data, np.mean, n_boot=600, seed=1000 + t)
        covered += int(res["lo"] <= 0.0 <= res["hi"])
    rate = covered / trials
    # binomial(200, 0.95) -> sd ~ 1.5 %; allow a generous window
    assert 0.88 <= rate <= 0.99, f"empirical coverage {rate:.3f}"


def test_bootstrap_ci_brackets_the_point_estimate():
    data = [0.10, 0.22, 0.31, 0.19, 0.05, 0.44]
    res = bootstrap_ci(data, np.mean, n_boot=2000)
    assert res["lo"] <= res["point"] <= res["hi"]
    assert res["n"] == len(data)


def test_mape_is_exact_on_a_known_offset():
    reference = [1.0, 2.0, 4.0, 8.0]
    values = [v * 1.10 for v in reference]
    assert mape(values, reference) == pytest.approx(10.0, rel=1e-12)
    # an exact-zero reference carries no relative information and is excluded
    assert mape([0.0, 2.2], [0.0, 2.0]) == pytest.approx(10.0, rel=1e-12)


def test_bootstrap_mape_intervals_are_ordered_and_scale_with_bias():
    reference = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    small = bootstrap_mape([r * 1.02 for r in reference], reference, n_boot=2000)
    large = bootstrap_mape([r * 1.30 for r in reference], reference, n_boot=2000)
    assert small["point"] == pytest.approx(2.0, rel=1e-9)
    assert large["point"] == pytest.approx(30.0, rel=1e-9)
    assert small["lo"] <= small["point"] <= small["hi"]
    assert large["lo"] <= large["point"] <= large["hi"]
    assert (large["hi"] - large["lo"]) > (small["hi"] - small["lo"])


def test_perturbation_band_collapses_at_zero_reference_uncertainty():
    reference = [0.2, 0.3, 0.4]
    values = [0.22, 0.33, 0.44]
    res = perturbation_band(values, reference, rel_sigma=0.0, n_samples=200)
    assert res["nominal"] == pytest.approx(10.0, rel=1e-9)
    assert res["lo"] == pytest.approx(10.0, rel=1e-9)
    assert res["hi"] == pytest.approx(10.0, rel=1e-9)


def test_perturbation_band_widens_with_reference_uncertainty():
    reference = [0.2, 0.3, 0.4]
    values = [0.22, 0.33, 0.44]
    narrow = perturbation_band(values, reference, rel_sigma=0.02, n_samples=4000)
    wide = perturbation_band(values, reference, rel_sigma=0.10, n_samples=4000)
    assert (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"])
    # a 10 % reference uncertainty swamps a 10 % model error: the band must
    # contain the nominal value and reach down towards zero
    assert wide["lo"] < wide["nominal"] < wide["hi"]


def test_perturbation_band_rejects_unknown_distribution():
    with pytest.raises(ValueError):
        perturbation_band([1.0], [1.0], distribution="cauchy")


def test_as_percent():
    assert as_percent(0.0325) == pytest.approx(3.25)
    assert math.isnan(as_percent(float("nan")))


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        mape([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        perturbation_band([1.0, 2.0], [1.0])