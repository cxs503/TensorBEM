"""Tests for CASE-BEM-HULME-HEMISPHERE band edge analysis (m22067).

Verifies:
- band_edge_analysis.json exists and has correct structure
- convergence orders are physically reasonable (1.5 < p < 4.0)
- extrapolation predictions are consistent with input data
- band_current_code results are self-consistent
"""
import json
import math
from pathlib import Path

import pytest

# Use the test file's location to find the repo root
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
CASE_DIR = REPO_ROOT / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
ANALYSIS_PATH = CASE_DIR / "band_edge_analysis.json"
BAND_CURRENT_PATH = CASE_DIR / "band_current_code" / "results.json"


@pytest.fixture(scope="module")
def analysis():
    assert ANALYSIS_PATH.exists(), f"band_edge_analysis.json not found at {ANALYSIS_PATH}"
    return json.loads(ANALYSIS_PATH.read_text())


@pytest.fixture(scope="module")
def band_current():
    assert BAND_CURRENT_PATH.exists(), f"band_current_code/results.json not found at {BAND_CURRENT_PATH}"
    return json.loads(BAND_CURRENT_PATH.read_text())


# --- Artifact integrity ---

def test_analysis_file_exists():
    assert ANALYSIS_PATH.exists(), f"band_edge_analysis.json must exist at {ANALYSIS_PATH}"


def test_analysis_has_required_keys(analysis):
    required = {"case_code", "driver", "analysis", "criteria", "verdict", "metadata"}
    for key in required:
        assert key in analysis, f"Missing key: {key}"


def test_analysis_case_code(analysis):
    assert analysis["case_code"] == "BEM-HULME-HEMISPHERE"


def test_analysis_driver(analysis):
    assert analysis["driver"] == "band_edge_analysis"


# --- Convergence order sanity ---

def test_ka4_convergence_order_reasonable(analysis):
    p = analysis["analysis"]["convergence"]["ka_4.0"]["order_p"]
    assert 1.5 < p < 4.0, f"ka=4.0 convergence order p={p} outside reasonable range [1.5, 4.0]"


def test_ka3_convergence_order_reasonable(analysis):
    p = analysis["analysis"]["convergence"]["ka_3.0"]["order_p"]
    assert 1.5 < p < 4.0, f"ka=3.0 convergence order p={p} outside reasonable range [1.5, 4.0]"


def test_convergence_orders_consistent(analysis):
    """ka=3.0 and ka=4.0 should have similar convergence orders (both constant panel BEM)."""
    p3 = analysis["analysis"]["convergence"]["ka_3.0"]["order_p"]
    p4 = analysis["analysis"]["convergence"]["ka_4.0"]["order_p"]
    assert abs(p3 - p4) / max(p3, p4) < 0.25, f"Orders too different: p3={p3}, p4={p4}"


# --- Extrapolation consistency ---

def test_ka4_extrapolation_math(analysis):
    """Verify extrapolation formula: e_4096 = e_1024 * (1/2)^p"""
    ka4 = analysis["analysis"]["convergence"]["ka_4.0"]
    e1024 = ka4["band_max_1024"]
    p = ka4["order_p"]
    predicted = e1024 * (0.5) ** p
    assert abs(predicted - ka4["predicted_4096"]) < 0.001, \
        f"Extrapolation mismatch: computed {predicted:.4f} vs stored {ka4['predicted_4096']:.4f}"


def test_ka3_extrapolation_math(analysis):
    """Verify extrapolation formula for ka=3.0."""
    ka3 = analysis["analysis"]["convergence"]["ka_3.0"]
    e1024 = ka3["band_max_1024"]
    p = ka3["order_p"]
    predicted = e1024 * (0.5) ** p
    assert abs(predicted - ka3["predicted_4096"]) < 0.001, \
        f"Extrapolation mismatch: computed {predicted:.4f} vs stored {ka3['predicted_4096']:.4f}"


def test_ka4_predicted_passes_5pct(analysis):
    """ka=4.0 extrapolated to 4096 panels should pass 5% gate."""
    pred = analysis["analysis"]["convergence"]["ka_4.0"]["predicted_4096"]
    assert pred < 0.05, f"Predicted ka=4.0 band_max at 4096 = {pred*100:.2f}% >= 5%"


# --- Band current code consistency ---

def test_band_current_has_cells(band_current):
    cells = band_current.get("cells", [])
    assert len(cells) > 0, "band_current_code/results.json has no cells"


def test_band_current_ka4_1024_fails(band_current):
    """ka=4.0 on 1024 panels should have band_max > 5%."""
    for c in band_current.get("cells", []):
        if abs(c.get("ka", 0) - 4.0) < 0.01 and c.get("panels") == 1024:
            rel_errors = [abs(v) for k, v in c.items() if k.startswith("rel_error_")]
            band_max = max(rel_errors)
            assert band_max > 0.05, f"ka=4.0 1024-panel band_max = {band_max*100:.2f}% should be > 5%"
            return
    pytest.fail("ka=4.0 1024-panel cell not found in band_current_code")


def test_band_current_ka35_1024_passes(band_current):
    """ka=3.5 on 1024 panels should have band_max < 5%."""
    for c in band_current.get("cells", []):
        if abs(c.get("ka", 0) - 3.5) < 0.01 and c.get("panels") == 1024:
            rel_errors = [abs(v) for k, v in c.items() if k.startswith("rel_error_")]
            band_max = max(rel_errors)
            assert band_max < 0.05, f"ka=3.5 1024-panel band_max = {band_max*100:.2f}% should be < 5%"
            return
    pytest.fail("ka=3.5 1024-panel cell not found in band_current_code")


# --- Criteria ---

def test_criteria_E1_fails(analysis):
    assert analysis["criteria"]["E1_extension_band_error_below_5pct_1024"]["pass"] is False


def test_criteria_E2_passes(analysis):
    assert analysis["criteria"]["E2_geometry_precondition"]["pass"] is True


def test_criteria_E3_passes(analysis):
    assert analysis["criteria"]["E3_refinement_does_not_worsen"]["pass"] is True


def test_criteria_E4_passes(analysis):
    assert analysis["criteria"]["E4_reference_self_converged_up_to_ka4"]["pass"] is True


def test_criteria_E5_passes(analysis):
    assert analysis["criteria"]["E5_chief_keeps_irregular_frequency_inside_gate"]["pass"] is True


def test_criteria_E6_passes(analysis):
    assert analysis["criteria"]["E6_ka4_is_discretisation_not_method_limit"]["pass"] is True


# --- Verdict ---

def test_verdict_is_pass_with_extrapolation(analysis):
    assert analysis["verdict"] == "PASS_WITH_EXTRAPOLATION"


def test_verdict_note_mentions_discretisation(analysis):
    note = analysis.get("verdict_note", "")
    assert "discretisation" in note.lower() or "discretization" in note.lower(), \
        f"Verdict note should mention discretisation limit: {note}"
