"""Machine checks for the ``acoustic-array-beamforming`` external-anchor artifact.

These tests prove the measurable claims of
``benchmarks/anchors/acoustic-array-beamforming.json`` and prove that the anchor is
*not* claimed as verified (nothing was transcribed, so nothing may be cited).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ANCHOR_DIR = REPO / "benchmarks" / "anchors" / "acoustic_array_beamforming"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pekeris_anchor", ANCHOR_DIR / "pekeris_anchor.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pekeris_anchor"] = mod   # dataclass needs the module registered
    spec.loader.exec_module(mod)
    return mod


pa = _load()


@pytest.fixture(scope="module")
def wg():
    return pa.Waveguide()


@pytest.fixture(scope="module")
def derivation():
    return pa.derive_dispersion()


def test_branch_equation_derived_symbolically(derivation):
    assert derivation["sympy_residual_is_zero"] is True
    assert derivation["interface_elimination_residual"] == "0"
    assert derivation["helmholtz_residual_layer"] == "0"
    assert derivation["helmholtz_residual_halfspace"] == "0"


def test_dimensional_homogeneity(derivation):
    assert derivation["dimensional_checks_all_homogeneous"] is True
    assert all(c["homogeneous"] for c in derivation["dimensional_checks"])


def test_dimensional_checker_rejects_inconsistent_quantity():
    bad = pa.dim_of_check("k_r", "dimensionless", pa.DIM["inv_length"])
    assert bad["homogeneous"] is False


def test_solver_residual_and_trapped_window(wg):
    roots = pa.pekeris_roots(wg, 100.0)
    assert len(roots) == pa.analytic_mode_count(wg, 100.0) >= 1
    for r in roots:
        assert r.residual_F_rel < 1e-12
        assert wg.critical_ratio < r.s < 1.0
        # internal consistency: s, grazing angle and phase velocity agree
        import math
        assert r.s == pytest.approx(math.cos(math.radians(r.grazing_angle_deg)), rel=1e-12)
        assert r.phase_velocity_m_s == pytest.approx(wg.c_water_m_s / r.s, rel=1e-12)
        assert r.k_z == pytest.approx(math.sqrt((wg.k1(100.0) ** 2) - r.k_r ** 2), rel=1e-12)


def test_mode_count_matches_closed_form_over_band(wg):
    for f in (20.0, 50.0, 100.0, 200.0, 300.0, 400.0):
        assert len(pa.pekeris_roots(wg, f)) == pa.analytic_mode_count(wg, f)


def test_independent_fem_agrees_with_solver(wg):
    f = 100.0
    analytic = [r.k_r for r in pa.pekeris_roots(wg, f)]
    coarse = pa.fem_branch_roots(wg, f, n_layer=80)[: len(analytic)]
    fine = pa.fem_branch_roots(wg, f, n_layer=160)[: len(analytic)]
    err_coarse = max(abs(a - b) / a for a, b in zip(analytic, coarse))
    err_fine = max(abs(a - b) / a for a, b in zip(analytic, fine))
    assert err_fine < 1e-2
    assert err_fine < err_coarse          # converging towards the analytic roots


def test_conversion_chain_round_trip_below_1e9(wg):
    worst = 0.0
    for f in (25.0, 100.0, 275.0):
        for r in pa.pekeris_roots(wg, f):
            reps = {
                "k_r_rad_m": r.k_r,
                "c_ph_m_s": r.phase_velocity_m_s,
                "k_r_times_H": r.k_r * wg.depth_m,
                "grazing_deg": r.grazing_angle_deg,
            }
            for key, val in reps.items():
                out = pa.to_case_frame(wg, f, {key: val})
                assert out["s_kr_over_k1"] == pytest.approx(r.s, rel=1e-12)
                worst = max(worst, out["round_trip_max_abs_residual"])
    assert worst < 1e-9


def test_conversion_rejects_ambiguous_representation(wg):
    with pytest.raises(ValueError):
        pa.to_case_frame(wg, 100.0, {"k_r_rad_m": 0.4, "c_ph_m_s": 1500.0})
    with pytest.raises(ValueError):
        pa.to_case_frame(wg, 100.0, {})


def test_quote_gate_rejects_untranscribed_records():
    # a value with no quote/locator must never become an eligible anchor
    gate = pa.validate_quote_record(
        {"source_id": "X", "quantity": "k_r", "value": 0.4, "units": "rad/m"})
    assert gate["verdict"] == "REJECTED"
    assert any(r.startswith("missing_field:verbatim_quote") for r in gate["reasons"])
    assert pa.anchor_status([])["status"] == "WITHHELD"


def test_quote_gate_accepts_a_complete_record():
    rec = {
        "source_id": "doi:10.1000/example",
        "verbatim_quote": "the roots of the branch equation are tabulated below",
        "locator": "Table 5.1, p. 340",
        "table_or_page": "p.340",
        "quantity": "k_r",
        "value": 0.4189,
        "units": "rad/m",
    }
    assert pa.validate_quote_record(rec)["verdict"] == "ELIGIBLE"
    assert pa.anchor_status([rec])["status"] == "VERIFIED"


def test_artifact_reports_withheld_anchor_and_honest_gap():
    art = json.loads((REPO / "benchmarks" / "anchors" / "acoustic-array-beamforming.json")
                     .read_text())
    crit = {c["id"]: c for c in art["criteria"]}
    assert set(crit) == {"C1", "C2", "C3", "C4"}
    # C1 (verbatim transcription) failed, therefore the anchor is withheld ...
    assert crit["C1"]["status"] == "NOT_MET"
    assert art["external_anchor"]["status"] == "WITHHELD"
    assert art["conclusion"]["external_anchor_status"] == "WITHHELD"
    # ... and no external number is registered to support a claim
    assert art["external_anchor"]["n_records"] == 0
    # the machine-checkable criteria are met and carry measured numbers
    assert crit["C2"]["measured"]["max_round_trip_abs_residual"] < 1e-9
    assert crit["C3"]["measured"]["max_branch_residual_F_rel"] < 1e-12
    assert crit["C3"]["measured"]["mode_count_mismatches"] == 0
    assert art["conclusion"]["honest_verdict"] == "HONEST_FAIL"