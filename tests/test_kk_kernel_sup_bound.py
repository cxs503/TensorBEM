"""Unit tests for the distribution-free KK tail bound (``kk_kernel_sup_bound``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import numpy as np

from tensorbem.causality import power_tail_principal_value
from tensorbem.kk_kernel_sup_bound import kernel_sup_tail_budget



def _power_law(exponent: float, coefficient: float = 2.0, count: int = 21):
    omega = np.geomspace(0.2, 12.0, count)
    return omega, coefficient * omega ** (-exponent)


def test_bound_never_underbids_the_closed_form_tail() -> None:
    """Where the closed-form PV tail exists (p > 2), the envelope must dominate it."""
    for exponent in (2.5, 3.0):
        omega, damping = _power_law(exponent)
        samples = np.array([0.5, 1.0, 3.0, 8.0])
        result = kernel_sup_tail_budget(omega, damping, samples, window=6)
        assert result.admissible
        assert result.exponent == pytest.approx(exponent, rel=1e-6)
        for value, bound in zip(samples, result.budget):
            # ``power_tail_principal_value`` refuses p <= 2; that refusal is the
            # closed-form route's usability bar, which is why the envelope route
            # needs its own reference-free argument instead of a numerical one.
            exact = abs(
                2.0
                / math.pi
                * power_tail_principal_value(
                    float(value),
                    float(damping[-1]) * float(omega[-1]) ** exponent,
                    exponent,
                    float(omega[-1]),
                )
            )
            assert bound >= exact * (1.0 - 1e-6)
            assert bound <= 20.0 * exact


def test_bound_is_finite_below_the_closed_form_bar() -> None:
    """p = 1.5 gives no closed form (the helper raises) but still a finite envelope."""
    omega, damping = _power_law(1.5)
    result = kernel_sup_tail_budget(omega, damping, np.array([1.0]))
    assert result.admissible
    assert math.isfinite(float(result.budget[0])) and float(result.budget[0]) > 0.0
    with pytest.raises(ValueError):
        power_tail_principal_value(1.0, 1.0, 1.5, float(omega[-1]))


def test_bound_diverges_when_the_envelope_is_not_integrable() -> None:
    omega, damping = _power_law(0.6)
    result = kernel_sup_tail_budget(omega, damping)
    assert not result.admissible
    assert math.isinf(result.max_relative)
    assert "diverges" in result.reason


def test_bound_is_monotone_in_the_tail_exponent() -> None:
    """Faster decay must never produce a larger bound."""
    bounds = []
    for exponent in (1.2, 1.8, 2.4, 3.0):
        omega, damping = _power_law(exponent)
        result = kernel_sup_tail_budget(omega, damping, np.array([2.0]), window=6)
        bounds.append(float(result.budget[0]))
    assert bounds == sorted(bounds, reverse=True)


def test_bound_scales_linearly_and_carries_no_free_parameter() -> None:
    """Doubling the damping doubles the bound; nothing is fitted to the samples."""
    omega, damping = _power_law(1.7)
    samples = np.array([0.7, 4.0])
    single = kernel_sup_tail_budget(omega, damping, samples, window=6)
    double = kernel_sup_tail_budget(omega, 2.0 * damping, samples, window=6)
    assert double.exponent == pytest.approx(single.exponent, rel=1e-6)
    np.testing.assert_allclose(double.budget, 2.0 * single.budget, rtol=1e-12)


def test_bound_is_blind_to_a_persisting_but_hidden_peak() -> None:
    """Honesty check: the bound only uses the band top, so a rising tail passes it.

    A damping sweep whose last decade rises instead of decaying is reported
    inadmissible rather than silently bounded -- the failure mode this test pins
    is *not* caught by the envelope magnitude, only by its exponent.
    """
    omega = np.geomspace(0.2, 12.0, 21)
    damping = np.ones_like(omega)  # flat: exponent 0
    result = kernel_sup_tail_budget(omega, damping)
    assert not result.admissible