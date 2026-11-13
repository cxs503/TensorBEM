"""Machine gate for CASE-BEM-IRREG-FREQ-CYL.

Every acceptance of
:mod:`tensorbem.benchmarks.irregular_frequency_bessel_zero_case` is re-derived
here from the solver and the committed reference table, so neither the artifact
nor the case report can drift away from the code.  The heavy bare-BEM band sweep
(Route C) is not repeated: it is a coarse locator and lives in the artifact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from tensorbem.benchmarks.irregular_frequency_bessel_zero_case import (
    MODE_LABELS,
    TOLERANCE_REL,
    audit_reference_zeros,
    draft_invariance,
    grade_case,
    load_reference_table,
    panel_sensitivity,
    predict_case,
    reference_zero,
    scale_sensitivity,
    waterplane_radius_model,
)
from tensorbem.benchmarks.irregular_frequency_study import dirichlet_zeros

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "artifacts/case_bem_irreg_freq_cyl/results.json"


@pytest.fixture(scope="module")
def table():
    rows, meta = load_reference_table()
    return rows, meta


@pytest.fixture(scope="module")
def headline(table):
    rows, _ = table
    case = predict_case(1.0, n_theta=64, draft_ratio=1.0)
    return case, grade_case(case, rows)


# ---------------------------------------------------------------------------
# The reference anchor
# ---------------------------------------------------------------------------


def test_reference_table_carries_its_page_pin(table):
    rows, meta = table
    assert len(rows) == 20
    assert "Table 9.5" in meta["anchor"]["citation"]
    assert "370" in meta["anchor"]["section_page_pin"]
    assert "Mathematical Functions" in meta["anchor"]["citation"]
    # only the J_0 / J_1 families may claim the handbook as their source
    for row in rows:
        if row.source == "handbook":
            assert row.order in (0, 1)
        else:
            assert row.source == "computed"


def test_every_literal_survives_the_three_route_digit_audit(table):
    rows, meta = table
    audit = audit_reference_zeros(rows, tolerance_absolute=meta["audit_tolerance_absolute"])
    assert audit["all_pass"]
    assert audit["worst_abs_diff_mpmath"] <= meta["audit_tolerance_absolute"]
    # scipy differs from the literal by at most the literal's own 10-decimal
    # rounding, so the cross-witness bound is the audit tolerance
    assert audit["worst_abs_diff_scipy"] <= meta["audit_tolerance_absolute"]
    assert audit["worst_abs_diff_repo_dirichlet_zeros"] <= meta["audit_tolerance_absolute"]


def test_dirichlet_zeros_is_correct_for_order_at_least_four():
    """Regression for the recurrence blow-up found by this case (was garbage)."""
    mp.mp.dps = 40
    for order in (4, 5):
        for index in (1, 2):
            exact = float(mp.besseljzero(order, index))
            got = dirichlet_zeros(order, index)[-1]
            assert got == pytest.approx(exact, rel=1e-12)


# ---------------------------------------------------------------------------
# The case acceptance and its two error sources
# ---------------------------------------------------------------------------


def test_headline_case_meets_the_two_percent_acceptance(headline):
    _, graded = headline
    for item in graded:
        assert abs(item["err_total_rel"]) <= TOLERANCE_REL
        assert item["pass_2pct"]


def test_error_split_is_an_identity(headline):
    _, graded = headline
    for item in graded:
        assert item["split_residual"] == pytest.approx(0.0, abs=1e-15)


def test_bessel_constants_are_exact_to_the_reconstruction_floor(headline, table):
    rows, _ = table
    _, graded = headline
    for item in graded:
        # the predictor's own constant vs the handbook zero
        assert abs(item["err_const_rel"]) <= 1e-10
        assert item["ref_j"] == reference_zero(rows, item["order"], 1).value


def test_waterplane_bias_follows_the_polygon_model():
    """Eq. (2): err_geom = sqrt(x/sin x) - 1 for x = 2 pi / N, to 1e-3."""
    for n_theta in (8, 12, 17, 32, 64):
        case = predict_case(1.0, n_theta=n_theta)
        model = waterplane_radius_model(1.0, n_theta) - 1.0
        assert case["radius_bias"] == pytest.approx(model, abs=1e-12)
        x = 2.0 * math.pi / n_theta
        assert case["radius_bias"] == pytest.approx(math.sqrt(math.sin(x) / x) - 1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Sensitivities
# ---------------------------------------------------------------------------


def test_panel_sweep_puts_the_acceptance_crossing_at_thirteen(table):
    rows, _ = table
    sweep = {row["n_theta"]: row for row in panel_sensitivity(rows, panels=(8, 12, 13, 16, 32, 64))}
    assert sweep[8]["pass_2pct_all_modes"] is False
    assert sweep[12]["pass_2pct_all_modes"] is False
    assert sweep[13]["pass_2pct_all_modes"] is True
    assert sweep[32]["pass_2pct_all_modes"] is True
    for row in sweep.values():
        assert abs(row["bias_model_error"]) <= 1e-3
    crossing_model = int(math.ceil(math.pi / math.sqrt(3.0 * TOLERANCE_REL)))
    assert crossing_model == 13


def test_prediction_is_scale_invariant(table):
    rows, _ = table
    result = scale_sensitivity(rows, radii=(0.5, 1.0, 3.0), n_theta=64)
    assert result["pass"]
    assert max(result["got_j_spread_rel"]) <= 1e-12


def test_prediction_is_draft_blind(table):
    rows, _ = table
    result = draft_invariance(rows, draft_ratios=(0.5, 1.0, 2.0), n_theta=32)
    assert result["pass"]
    assert max(result["max_rel_spread"]) <= 1e-12


# ---------------------------------------------------------------------------
# Artifact consistency (skipped when the artifact has not been generated)
# ---------------------------------------------------------------------------


def test_artifact_matches_the_module(headline):
    if not ARTIFACT.exists():
        pytest.skip("case artifact not generated in this checkout")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["case"] == "CASE-BEM-IRREG-FREQ-CYL"
    assert payload["verdict"]["pass"] is True
    assert payload["verdict"]["headline_max_abs_err_rel"] <= TOLERANCE_REL
    assert [g["mode"] for g in payload["graded"]] == [label for label, _ in MODE_LABELS]
    case, graded = headline
    for got, artifact_row in zip(graded, payload["graded"]):
        assert got["err_total_rel"] == pytest.approx(artifact_row["err_total_rel"], rel=1e-6)
    assert np.isfinite(payload["verdict"]["headline_max_abs_err_rel"])