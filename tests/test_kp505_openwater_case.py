"""Contract tests for the CASE-BEM-KP505-OPENWATER driver (mission m17207).

What is pinned here
-------------------
* the anchor used by the case is the *transcribed* Table-2 experimental column of
  Iftikhar & Takinaci (2023, p. 10) imported from the existing benchmark module --
  no digit is typed into the case driver;
* the 5 % scoring logic and the numerical-admissibility gate behave as declared
  (a blown-up lattice solution is flagged, not silently reported as a model
  error);
* :func:`tensorbem.propeller.open_water_point` warns when a near-singular solve
  produces a value outside the physical envelope (regression: the API default
  ``wake_model="kinematic"`` used to return ``K_T`` of order 1e5 in silence).
"""

from __future__ import annotations

import math

import pytest

from tensorbem.benchmarks.kp505_openwater_bem import (
    ANCHOR,
    TABLE_2_ROWS_VERBATIM,
)
from tensorbem.benchmarks.kp505_openwater_case import (
    TOLERANCE_PCT,
    _summary,
    evaluate,
    slope_diagnosis,
)
from tensorbem.propeller import BladeGeometry, open_water_point


def test_anchor_is_the_parsed_verbatim_table():
    """The case scores against the p.10 experimental column, parsed not typed."""

    assert len(ANCHOR) == 10
    assert len(TABLE_2_ROWS_VERBATIM) == 10
    for row in TABLE_2_ROWS_VERBATIM:
        parts = row.split()
        advance = float(parts[0])
        assert ANCHOR[advance]["KT"] == float(parts[3])
        assert ANCHOR[advance]["10KQ"] == float(parts[4])
    assert ANCHOR[0.8]["KT"] == pytest.approx(0.141)
    assert ANCHOR[0.8]["10KQ"] == pytest.approx(0.270)


def test_point_scoring_matches_the_recorded_canonical_row():
    """J = 0.5 of the canonical configuration: K_T inside 5 %, 10K_Q outside."""

    row = evaluate(0.5)
    assert row["admissible"] is True
    assert row["n_panels"] == 16 * 2
    assert abs(row["error_pct"]["KT_pct"]) < TOLERANCE_PCT
    assert row["pass_within_5pct"]["KT_pct"] is True
    assert abs(row["error_pct"]["10KQ_pct"]) > TOLERANCE_PCT
    assert row["pass_within_5pct"]["10KQ_pct"] is False
    # the anchor column is carried, never recomputed from the measurement
    assert row["KT_anchor"] == ANCHOR[0.5]["KT"]


def test_blown_up_solution_is_flagged_not_scored():
    """The kinematic helix at J = 0.2 is near singular -> gate says inadmissible."""

    row = evaluate(0.2, wake_model="kinematic")
    assert row["admissible"] is False
    assert abs(row["KT"]) > 1.0
    # the kernel's own diagnosis is carried into the artefact, not printed away
    assert row["kernel_warnings"]
    assert "unphysical solution" in row["kernel_warnings"][0]


def test_default_wake_model_warns_when_it_breaks_down():
    """Regression: the API default must not return a 1e5 K_T in silence."""

    geometry = BladeGeometry()
    with pytest.warns(RuntimeWarning, match="unphysical solution"):
        result = open_water_point(geometry, 0.5, wake_model="kinematic",
                                  wake_length=2.0, wake_steps=24)
    assert abs(result.thrust) > 1.0

    # and it must stay quiet on a healthy point
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        healthy = open_water_point(geometry, 0.5, wake_model="geometric",
                                   wake_length=1.0, wake_steps=24)
    assert 0.0 < healthy.thrust < 1.0


def test_summary_counts_are_scoped_to_the_gate():
    rows = [
        {"J": 0.1, "admissible": True,
         "error_pct": {"KT_pct": 1.0, "10KQ_pct": 1.0, "eta0_pct": 1.0},
         "pass_within_5pct": {"KT_pct": True, "10KQ_pct": True, "eta0_pct": True}},
        {"J": 0.2, "admissible": True,
         "error_pct": {"KT_pct": -9.0, "10KQ_pct": 1.0, "eta0_pct": 1.0},
         "pass_within_5pct": {"KT_pct": False, "10KQ_pct": True, "eta0_pct": True}},
        {"J": 0.3, "admissible": False,
         "error_pct": {"KT_pct": 400.0, "10KQ_pct": -400.0, "eta0_pct": 900.0},
         "pass_within_5pct": {"KT_pct": False, "10KQ_pct": False, "eta0_pct": False}},
    ]
    summary = _summary(rows)
    assert summary["n_points"] == 3
    assert summary["n_admissible"] == 2
    assert summary["n_pass_all_three_5pct"] == 1
    assert summary["n_pass_all_three_5pct_admissible"] == 1
    assert summary["n_pass_by_quantity"]["KT_pct"] == 1
    assert summary["worst"]["eta0_pct"]["J"] == 0.3


def test_slope_diagnosis_separates_offset_from_slope():
    """A pure vertical offset must leave the slope ratio at 1.0."""

    rows = [{"J": 0.1 * i, "KT": 0.5 - 0.5 * 0.1 * i,
             "KT_anchor": 0.4 - 0.5 * 0.1 * i} for i in range(1, 10)]
    diagnosis = slope_diagnosis(rows)
    assert diagnosis["slope_ratio_measured_over_reference"] == pytest.approx(1.0)
    assert math.isclose(diagnosis["measured"]["zero_thrust_J"], 1.0, rel_tol=1e-9)
    assert math.isclose(diagnosis["reference"]["zero_thrust_J"], 0.8, rel_tol=1e-9)