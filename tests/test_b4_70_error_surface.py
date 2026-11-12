"""Guard rails for the W25 B4-70 error-surface driver.

These tests pin the *machinery* (Shapley identity, level-selection rule,
anchor-versus-kernel scoring, numerical-stability rule) on cheap
configurations; the full sweep is produced by
``python -m tensorbem.benchmarks.b4_70_error_surface`` and lands in
``papers/artifacts/w25/propeller-openwater-bem/``.
"""

from __future__ import annotations

import itertools
import math

import pytest

from tensorbem.benchmarks.b4_70_error_surface import (
    B4_70_CONFIG,
    KQ_PLAUSIBLE_MAX,
    KT_PLAUSIBLE_MAX,
    KP505_CONFIG,
    WAKE_BASE,
    WAKE_FIDELITY,
    WAKE_FINE,
    _kp505_j0_from_anchor,
    _rel,
    decompose_b4_70,
    score_point,
    select_fine_configuration,
    shapley_binary,
    solve,
    stability_probe,
)
from tensorbem.wageningen import b4_70


# ---------------------------------------------------------------------------
# Shapley machinery
# ---------------------------------------------------------------------------


def test_shapley_is_exact_and_additive():
    factors = ["a", "b", "c"]
    values = {
        frozenset(): 1.0,
        frozenset({"a"}): 3.0,
        frozenset({"b"}): 2.5,
        frozenset({"c"}): -0.5,
        frozenset({"a", "b"}): 6.0,
        frozenset({"a", "c"}): 2.0,
        frozenset({"b", "c"}): 1.5,
        frozenset({"a", "b", "c"}): 5.0,
    }
    phi = shapley_binary(values, factors)
    total = values[frozenset(factors)] - values[frozenset()]
    assert sum(phi.values()) == pytest.approx(total, abs=1e-12)


def test_shapley_reduces_to_marginal_when_effects_are_additive():
    """With an additive value function every Shapley value is the main effect."""

    effects = {"a": 2.0, "b": -1.0, "c": 0.5}
    factors = list(effects)
    values = {}
    for size in range(len(factors) + 1):
        for subset in itertools.combinations(factors, size):
            values[frozenset(subset)] = 10.0 + sum(effects[f] for f in subset)
    phi = shapley_binary(values, factors)
    for factor, effect in effects.items():
        assert phi[factor] == pytest.approx(effect, abs=1e-12)


def test_shapley_splits_a_pure_interaction_evenly():
    """With no main effects and a symmetric interaction, each share is equal."""

    factors = ["a", "b"]
    values = {
        frozenset(): 0.0,
        frozenset({"a"}): 0.0,
        frozenset({"b"}): 0.0,
        frozenset({"a", "b"}): 4.0,
    }
    phi = shapley_binary(values, factors)
    assert phi["a"] == pytest.approx(2.0)
    assert phi["b"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Level selection (stability-driven, no cherry-picking)
# ---------------------------------------------------------------------------


def _combination(n_radial, n_chord, wake_model, length, steps, stable, unstable=()):
    return {
        "n_radial": n_radial,
        "n_chord": n_chord,
        "n_panels_per_blade": n_radial * n_chord,
        "wake_model": wake_model,
        "wake_length": length,
        "wake_steps": steps,
        "stable_at_all_J": stable,
        "unstable_points": list(unstable),
    }


def test_selection_prefers_fidelity_but_requires_all_lattice_stability():
    probe = {
        "combinations": [
            # induced is stable only on the coarsest lattice -> not separable
            _combination(8, 1, "induced", 2.0, 48, True),
            _combination(24, 2, "induced", 2.0, 48, False, [{"J": 0.1, "KT": -1e5}]),
            # geometric is stable everywhere
            _combination(8, 1, "geometric", 1.0, 24, True),
            _combination(24, 2, "geometric", 1.0, 24, True),
        ]
    }
    selection = select_fine_configuration(probe)
    assert selection["wake"]["wake_model"] == "geometric"
    assert selection["mesh"] == {"n_radial": 24, "n_chord": 2}
    unstable = selection["unstable_combinations"]
    assert unstable and unstable[0]["wake_model"] == "induced"


def test_selection_respects_the_panel_budget():
    probe = {
        "combinations": [
            _combination(8, 1, "geometric", 2.0, 48, True),
            _combination(24, 2, "geometric", 2.0, 48, True),
        ]
    }
    tight = select_fine_configuration(probe, panel_budget=40)
    assert tight["mesh"] == {"n_radial": 8, "n_chord": 1}
    roomy = select_fine_configuration(probe, panel_budget=400)
    assert roomy["mesh"] == {"n_radial": 24, "n_chord": 2}


def test_selection_fails_loudly_when_nothing_is_stable():
    probe = {
        "combinations": [
            _combination(8, 1, "induced", 2.0, 48, False, [{"J": 0.1, "KT": -1e3}]),
            _combination(8, 1, "geometric", 1.0, 24, False, [{"J": 0.1, "KT": -1e3}]),
        ]
    }
    with pytest.raises(RuntimeError):
        select_fine_configuration(probe)


def test_wake_fidelity_order_is_declared_and_ordered_by_model_quality():
    models = [entry[0] for entry in WAKE_FIDELITY]
    assert models[0] == "induced" and models[-1] == "geometric"
    assert ("geometric", 2.0, 48) in WAKE_FIDELITY


# ---------------------------------------------------------------------------
# Scoring against the anchor
# ---------------------------------------------------------------------------


def test_relative_deviation_guard_handles_zero_anchor():
    assert _rel(1.0, 0.0) != _rel(1.0, 0.0)  # nan
    assert _rel(1.1, 1.0) == pytest.approx(10.0)


def test_score_point_matches_the_anchor_evaluator():
    series = b4_70(1.0)
    row = score_point(B4_70_CONFIG, 0.3, series, series.kt(0.0), series.kq(0.0))
    assert row["KT_anchor"] == pytest.approx(series.kt(0.3))
    assert row["KQ_anchor"] == pytest.approx(series.kq(0.3))
    assert row["dKT_pct"] == pytest.approx(100.0 * (row["KT"] - row["KT_anchor"]) / row["KT_anchor"])
    # Reynolds matching puts the 0.75 R section on the regression's reference Re
    assert row["reynolds_075R"] == pytest.approx(2e6, rel=1e-6)
    assert row["in_criterion_range"] is True


def test_criterion_range_flag_follows_the_mission_definition():
    series = b4_70(1.0)
    low = score_point(B4_70_CONFIG, 0.8, series, series.kt(0.0), series.kq(0.0))
    high = score_point(B4_70_CONFIG, 0.8333, series, series.kt(0.0), series.kq(0.0))
    assert low["in_criterion_range"] is True
    assert high["in_criterion_range"] is False


# ---------------------------------------------------------------------------
# Numerical-stability envelope of the kernel (regression pin for a real defect)
# ---------------------------------------------------------------------------


def test_stability_rule_flags_a_known_breakdown():
    """The kinematic helix at low J returns a garbage K_T: the rule must catch it."""

    row = solve(
        B4_70_CONFIG.__class__(**{**B4_70_CONFIG.__dict__, **WAKE_BASE}),
        0.5,
    )
    assert abs(row["KT"]) <= KT_PLAUSIBLE_MAX  # geometric helix is fine at J = 0.5

    broken = solve(
        B4_70_CONFIG.__class__(**{**B4_70_CONFIG.__dict__, "wake_model": "kinematic"}),
        0.1,
    )
    assert not (abs(broken["KT"]) <= KT_PLAUSIBLE_MAX and abs(broken["KQ"]) <= KQ_PLAUSIBLE_MAX)


def test_decomposition_is_exact_and_reports_a_remainder():
    series = b4_70(1.0)
    entry = decompose_b4_70(
        0.5,
        series,
        mesh_fine={"n_radial": 12, "n_chord": 2},
        wake_fine={"wake_model": "geometric", "wake_length": 2.0, "wake_steps": 48},
    )
    for quantity, block in entry["factors"].items():
        scale = max(abs(block["deviation_fine"]), abs(block["model_form_remainder"]), 1e-12)
        assert abs(block["sum_minus_total"]) / scale < 1e-9
        assert set(block["contributions"]) == {"mesh", "wake", "friction"}
        assert "none" in block["coalition_values"]


# ---------------------------------------------------------------------------
# KP505 anchor helpers
# ---------------------------------------------------------------------------


def test_kp505_zero_thrust_extrapolation_is_physical():
    j0 = _kp505_j0_from_anchor()
    assert 0.99 < j0["J0"] < 1.10
    assert "extrapolation" in j0["method"]
    assert j0["slope_dKT_dJ"] < 0.0


def test_kp505_config_uses_published_particulars():
    assert KP505_CONFIG.blades == 5
    assert KP505_CONFIG.expanded_area_ratio == pytest.approx(0.80)
    assert KP505_CONFIG.pitch_ratio == pytest.approx(0.95)
    assert KP505_CONFIG.hub_ratio == pytest.approx(0.18)


def test_probe_constants_are_sane():
    assert KT_PLAUSIBLE_MAX > 0 and KQ_PLAUSIBLE_MAX > 0
    assert WAKE_FINE["wake_length"] > WAKE_BASE["wake_length"]