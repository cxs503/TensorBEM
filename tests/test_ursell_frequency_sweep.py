"""Guards for the Ursell frequency/submergence sweep.

``docs/ursell_error_budget.md`` recorded that the accuracy of the Ursell panel
solution had been established at one ``(ka, a/f)`` point only. The sweep in
:mod:`tensorbem.benchmarks.ursell_frequency_sweep` extends it to a grid; these
tests keep the extension honest:

1. the closed-form reference must be truncation-insensitive at every grid point,
   otherwise the comparison measures the reference and not the solver;
2. the panel error must shrink with mesh refinement, at first order, and stay
   below the 3 % acceptance threshold at the finest quoted mesh;
3. the near-field damping must still equal the far-field (radiation) value off
   the original design point;
4. the committed sweep record must be internally consistent and regenerated from
   the committed reference, not typed by hand.

Tests are kept on coarse meshes so the suite stays fast; the fine-mesh numbers
live in ``docs/ursell_frequency_sweep/results.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks.ursell_frequency_sweep import (
    DEFAULT_A_OVER_F,
    DEFAULT_KA_VALUES,
    DEFAULT_PANEL_COUNTS,
    panel_coefficients,
    reference_coefficients,
    sweep,
    truncation_sensitivity,
)

_REPO = Path(__file__).resolve().parents[1]
_RECORD = _REPO / "docs" / "ursell_frequency_sweep" / "results.json"


def test_defaults_are_physical_and_span_a_ka_greater_than_one():
    """The grid must actually cover the case the debt called out (ka >= 1)."""
    assert min(DEFAULT_KA_VALUES) > 0.0
    assert max(DEFAULT_KA_VALUES) >= 1.0
    assert all(0.0 < ratio < 1.0 for ratio in DEFAULT_A_OVER_F)
    assert DEFAULT_PANEL_COUNTS == tuple(sorted(DEFAULT_PANEL_COUNTS))
    assert len(set(DEFAULT_PANEL_COUNTS)) == len(DEFAULT_PANEL_COUNTS)


@pytest.mark.parametrize(
    ("ka", "a_over_f"), [(0.2, 0.25), (0.6, 0.5), (1.0, 0.75), (1.5, 0.75)]
)
def test_reference_truncation_is_negligible_against_the_panel_error(ka, a_over_f):
    """40 terms must already be converged, across the whole grid extent."""
    sensitivity = truncation_sensitivity(ka, a_over_f, terms=(40, 80, 120))
    assert sensitivity["added_mass"] < 1.0e-10
    assert sensitivity["damping"] < 1.0e-10


@pytest.mark.parametrize(("ka", "a_over_f"), [(0.6, 0.5), (1.2, 0.75)])
def test_panel_error_is_first_order_and_below_three_percent(ka, a_over_f):
    """Convergence is first order away from the original design point too."""
    coarse = panel_coefficients(32, ka, a_over_f)
    fine = panel_coefficients(64, ka, a_over_f)
    reference = reference_coefficients(ka, a_over_f)

    coarse_error = abs(coarse["added_mass"] - reference["added_mass"]) / abs(
        reference["added_mass"]
    )
    fine_error = abs(fine["added_mass"] - reference["added_mass"]) / abs(
        reference["added_mass"]
    )
    damping_error = abs(fine["damping"] - reference["damping"]) / abs(
        reference["damping"]
    )

    assert fine_error < coarse_error
    assert fine_error < 3.0e-2
    assert damping_error < 3.0e-2
    # First order in panel size, i.e. the N * error product is roughly constant.
    order = math.log2(coarse_error / fine_error)
    assert 0.5 < order < 1.5
    assert 0.5 * fine["panels"] * fine_error < 1.0


def test_near_field_damping_matches_far_field_off_the_design_point():
    """The radiation condition is not a coincidence of ka = 0.6."""
    result = panel_coefficients(48, 1.2, 0.75)
    assert math.isfinite(result["near_over_far"])
    assert abs(result["near_over_far"] - 1.0) < 5.0e-2
    assert result["relative_residual"] < 1.0e-8


def test_invalid_point_is_rejected():
    with pytest.raises(ValueError):
        reference_coefficients(0.0, 0.5)
    with pytest.raises(ValueError):
        reference_coefficients(0.6, 1.0)
    with pytest.raises(ValueError):
        panel_coefficients(8, 0.6, 0.0)


def test_sweep_record_shape_and_orders():
    """A tiny in-test sweep reproduces the record structure and the summaries."""
    result = sweep(
        ka_values=(0.6,),
        a_over_f_values=(0.5,),
        panel_counts=(32, 64),
        reference_terms_check=(40, 80),
    )
    rows = result["rows"]
    summary = result["summary"]
    assert len(rows) == 2
    assert rows[0]["order_added_mass"] is None
    assert rows[1]["order_added_mass"] is not None
    assert summary["finest_panels"] == 64
    assert summary["points"] == 1
    finest_rows = [row for row in rows if row["panels"] == summary["finest_panels"]]
    assert summary["max_error_added_mass"] == max(
        row["error_added_mass"] for row in finest_rows
    )
    assert summary["all_within_three_percent"] == (
        rows[1]["error_added_mass"] < 3.0e-2 and rows[1]["error_damping"] < 3.0e-2
    )


@pytest.mark.skipif(not _RECORD.exists(), reason="sweep record not committed")
def test_committed_record_matches_the_committed_reference():
    """Every committed row must agree with the closed-form reference.

    This is a cheap hand-copy detector: it recomputes the analytical column only
    (no panel solve) and re-derives the summary from the rows.
    """
    record = json.loads(_RECORD.read_text())
    rows = record["rows"]
    settings = record["settings"]
    assert len(rows) == (
        len(settings["ka_values"])
        * len(settings["a_over_f_values"])
        * len(settings["panel_counts"])
    )
    for row in rows:
        reference = reference_coefficients(
            row["ka"],
            row["a_over_f"],
            terms=int(row["reference_terms"]),
            radius=settings["radius"],
            density=settings["density"],
            gravity=settings["gravity"],
        )
        assert row["reference_added_mass"] == pytest.approx(
            reference["added_mass"], rel=1.0e-12
        )
        assert row["reference_damping"] == pytest.approx(
            reference["damping"], rel=1.0e-12
        )
        assert row["error_added_mass"] == pytest.approx(
            abs(row["panel_added_mass"] - row["reference_added_mass"])
            / abs(row["reference_added_mass"]),
            rel=1.0e-12,
        )

    finest = max(settings["panel_counts"])
    finest_rows = [row for row in rows if row["panels"] == finest]
    summary = record["summary"]
    assert summary["finest_panels"] == finest
    assert summary["max_error_added_mass"] == max(
        row["error_added_mass"] for row in finest_rows
    )
    assert summary["all_within_three_percent"] == all(
        row["error_added_mass"] < 3.0e-2 and row["error_damping"] < 3.0e-2
        for row in finest_rows
    )
    assert summary["max_truncation_added_mass"] < 1.0e-9


@pytest.mark.skipif(not _RECORD.exists(), reason="sweep record not committed")
def test_committed_record_reproduces_the_published_single_point_number():
    """The sweep's ka = 0.6, a/f = 0.5, 128-panel row is the number in the docs."""
    record = json.loads(_RECORD.read_text())
    matches = [
        row
        for row in record["rows"]
        if row["ka"] == 0.6 and row["a_over_f"] == 0.5 and row["panels"] == 128.0
    ]
    assert len(matches) == 1
    row = matches[0]
    assert row["error_added_mass"] == pytest.approx(8.513e-3, rel=2.0e-3)
    assert row["error_damping"] == pytest.approx(1.336e-2, rel=2.0e-3)