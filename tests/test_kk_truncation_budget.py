"""Kramers-Kronig truncation budget and the C3 decidable-window semantics.

The residual of a band-limited KK closure is *not* a causality statement at every
sample frequency: the missing integral above ``s_max`` is positive for every
``omega < s_max`` and unbounded as ``omega -> s_max``.  ``high_band_truncation_budget``
bounds it from the measured local decay exponent; these tests pin the bound's
analytic value on an exact power law, its invalidity on a damping that does not
decay, and the way gate C3 reports (and does not silently hide) the samples it
cannot certify.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.causality import (
    fit_damping_power_tail,
    high_band_truncation_budget,
    kk_added_mass_closure,
    local_tail_exponent,
    power_tail_principal_value,
)


def _exact_tail_sweep(power: float = 4.0, band=(0.2, 12.0), n: int = 41):
    grid = np.logspace(math.log10(band[0]), math.log10(band[1]), n)
    grid[0], grid[-1] = band
    return grid, 3.0 * grid ** (-power)


def test_budget_matches_the_closed_form_tail_on_an_exact_power_law() -> None:
    grid, damping = _exact_tail_sweep(4.0)
    samples = np.array([0.5, 1.0, 3.0, 6.0])
    budget = high_band_truncation_budget(grid, damping, samples)
    assert budget.admissible
    assert budget.exponent == pytest.approx(4.0, rel=1e-6)
    for index, omega in enumerate(samples):
        reference = 2.0 / math.pi * power_tail_principal_value(
            float(omega), 3.0, 4.0, float(grid[-1])
        )
        assert budget.budget[index] == pytest.approx(reference, rel=1e-6)


def test_budget_grows_towards_the_band_top_and_shrinks_with_the_exponent() -> None:
    grid, gentle = _exact_tail_sweep(3.0)
    _, steep = _exact_tail_sweep(6.0)
    samples = np.array([0.5, 2.0, 6.0, 11.0])
    gentle_budget = high_band_truncation_budget(grid, gentle, samples)
    steep_budget = high_band_truncation_budget(grid, steep, samples)
    assert np.all(np.diff(gentle_budget.budget) > 0.0)
    assert np.all(steep_budget.budget < gentle_budget.budget)
    assert gentle_budget.exponent == pytest.approx(3.0, rel=1e-6)
    assert steep_budget.exponent == pytest.approx(6.0, rel=1e-6)


def test_budget_is_unbounded_when_the_damping_does_not_decay() -> None:
    grid = np.logspace(math.log10(0.2), math.log10(12.0), 41)
    rising = 1.0 + grid  # damping still growing at the top of the band
    budget = high_band_truncation_budget(grid, rising)
    assert not budget.admissible
    assert np.all(np.isinf(budget.budget))
    assert "not >" in budget.reason


def test_budget_is_infinite_at_and_above_the_band_top() -> None:
    grid, damping = _exact_tail_sweep(4.0)
    budget = high_band_truncation_budget(grid, damping, np.array([11.0, 12.0, 15.0]))
    assert math.isfinite(budget.budget[0])
    assert math.isinf(budget.budget[1]) and math.isinf(budget.budget[2])


def test_local_tail_exponent_recovers_a_known_slope() -> None:
    grid, damping = _exact_tail_sweep(3.5)
    exponent, coefficient = local_tail_exponent(grid, damping)
    assert exponent == pytest.approx(3.5, rel=1e-6)
    assert coefficient == pytest.approx(3.0, rel=1e-6)


def _fake_sweep(w, mass, b, label="medium"):
    return case.Sweep(
        label=label,
        n_sections=20,
        n_levels=6,
        panel_count=480,
        max_panel_scale=0.0645,
        omega=w,
        added_mass=mass,
        damping=b,
        solver_relative_residual=np.zeros(w.size),
        elapsed=0.0,
    )


def _causal_pair(band=(0.2, 12.0), n=41):
    """Exact KK pair with B ~ s^-3 tail: A - A_inf = -sum c/(a^2+w^2)."""
    poles = np.array([0.9, 2.7, 6.5])
    weights = np.array([2.0, -1.1, 0.6])
    w = np.logspace(math.log10(band[0]), math.log10(band[1]), n)
    w[0], w[-1] = band
    b = sum(c * a / (a**2 + w**2) for c, a in zip(weights, poles))
    mass = sum(-c / (a**2 + w**2) for c, a in zip(weights, poles))
    full = np.zeros((w.size, 6, 6))
    damp = np.zeros((w.size, 6, 6))
    for index in range(6):
        full[:, index, index] = mass * (index + 1)
        damp[:, index, index] = b * (index + 1)
    return w, full, damp


def test_c3_passes_on_an_exact_causal_pair_and_says_how_much_it_certified() -> None:
    w, mass, damp = _causal_pair()
    gate = case.kk_gate(_fake_sweep(w, mass, damp))
    assert gate["passed"] is True
    assert gate["values"]["worst_relative_residual_all_samples"] < case.TOLERANCES["kk"]
    assert isinstance(gate["fully_certified"], bool)
    assert "certif" in gate["status"]


def test_c3_flags_every_sample_undecided_but_does_not_turn_that_into_a_failure() -> None:
    # damping still rising at the band top (a sweep that stops inside a
    # resonance): the omitted integral cannot be bounded, so the pass decision
    # falls back to the measured residual, annotated as not certified.
    w = np.logspace(math.log10(0.2), math.log10(12.0), 21)
    mass = np.zeros((w.size, 6, 6))
    damp = np.zeros((w.size, 6, 6))
    for index in range(6):
        damp[:, index, index] = 0.5 * w**2
    gate = case.kk_gate(_fake_sweep(w, mass, damp))
    assert gate["values"]["all_dof_fully_decidable"] is False
    for row in gate["values"]["per_dof"]:
        assert row["n_decidable"] == 0
        assert math.isinf(row["truncation_budget_max_relative"])
    assert "not certified" in gate["status"]
    assert gate["passed"] == gate["raw_gate_passed"]


def test_negative_control_is_judged_on_the_clean_window_and_still_bites() -> None:
    w, mass, damp = _causal_pair()
    sweep = _fake_sweep(w, mass, damp)
    clean = case.kk_gate(sweep)
    window = [
        all(row["decidable_mask"][k] for row in clean["values"]["per_dof"])
        for k in range(len(clean["values"]["per_dof"][0]["decidable_mask"]))
    ]
    corrupt = case.kk_gate(sweep, corrupt=True, window=window)
    assert corrupt["id"] == "C6b"
    # the control is judged on exactly the samples the clean sweep was judged on
    for row in corrupt["values"]["per_dof"]:
        assert row["decidable_mask"] == window
    # and the corruption does move the residual somewhere on that window
    assert (
        corrupt["values"]["worst_relative_residual_all_samples"]
        > clean["values"]["worst_relative_residual_all_samples"]
    )
