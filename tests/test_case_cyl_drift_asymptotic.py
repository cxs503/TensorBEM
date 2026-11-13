"""Tests for the CASE-BEM-CYL-DRIFT-ASYMP case-pool benchmark driver.

The analytic gates run always (seconds).  The BEM gates are read from the
committed artifact ``artifacts/case_cyl_drift_asymptotic/results.json`` and are
skipped when the artifact is absent (the BEM run needs ~17 min of compute and is
submitted to the cluster, not to the test suite).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem import cylinder_drift as cd
from tensorbem.benchmarks import case_cyl_drift_asymptotic as case

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts/case_cyl_drift_asymptotic/results.json"


def test_literature_coefficient_identity():
    """5 pi^2/16 evaluated from the recorded expression == the module's closed form."""
    lit = case.literature_coefficient()
    assert lit == pytest.approx(5.0 * math.pi**2 / 16.0, rel=0.0, abs=0.0)
    assert lit == pytest.approx(3.0842513753404246, rel=1e-15)
    assert lit == pytest.approx(float(cd.long_wave_drift_coefficient()), rel=1e-15)


def test_long_wave_law_holds_at_small_ka():
    """Law value (5 pi^2/16 rho g A^2 a (ka)^3) vs exact closed form in the asymptotic band."""
    for ka, tol in ((0.02, 0.01), (0.05, 0.01), (0.10, 0.01), (0.20, 0.05)):
        exact = cd.far_field_cylinder_drift(ka)["mean_drift_force_x"]
        law = case.literature_coefficient() * case.SCALE * ka**3
        assert abs(exact - law) / law < tol, (ka, exact, law)


def test_long_wave_law_breaks_above_band():
    """The law is asymptotic: at ka = 0.3 it must already miss the 5 % criterion.

    This is the honest half of the gate -- it fixes the band edge reported in the
    case (`law_valid_ka`) instead of leaving it as a free parameter.
    """
    exact = cd.far_field_cylinder_drift(0.3)["mean_drift_force_x"]
    law = case.literature_coefficient() * case.SCALE * 0.3**3
    assert abs(exact - law) / law > 0.05


def test_law_valid_band_and_slope():
    lw = case.long_wave()
    assert lw["law_valid_ka"] >= 0.20
    assert lw["coefficient_abs_diff"] < 1e-12
    sl = case.slope()
    assert abs(sl["fitted_slope"] - 3.0) <= 0.05


def test_two_closed_form_routes_agree():
    wide = case.wide_ka()
    assert wide["max_near_vs_far"] < 1e-6


def test_second_order_amplitude_scaling():
    sc = case.scaling()
    assert sc["ratio"] == pytest.approx(4.0, rel=1e-12)


def test_gate_structure_analytic_only():
    res = case.build(skip_bem=True)
    gates = res["gates"]
    for name in ("G1_law_at_ka002", "G2_law_band", "G3_coefficient_identity",
                 "G4_slope", "G5_near_vs_far", "G6_amplitude_square"):
        assert gates[name]["pass"], (name, gates[name])
    assert gates["all_pass"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="BEM artifact not present")
def test_artifact_bem_gates():
    res = json.loads(ARTIFACT.read_text())
    gates = res["gates"]
    assert gates["G7_bem_far_vs_closed_form"]["got"] < 0.05
    assert gates["G8_bem_near_converged"]["got"] < 0.05
    assert gates["G10_azimuth"]["got"] < 0.01
    assert gates["all_pass"], {k: v for k, v in gates.items() if isinstance(v, dict) and not v["pass"]}


@pytest.mark.skipif(not ARTIFACT.exists(), reason="BEM artifact not present")
def test_artifact_provenance():
    res = json.loads(ARTIFACT.read_text())
    assert res["run"]["case"] == "CASE-BEM-CYL-DRIFT-ASYMP"
    assert "case_cyl_drift_asymptotic" in res["run"]["repro_cmd"]
    assert res["literature"]["long_wave_law"]["expression"] == "5*pi**2/16"
    # literature transcriptions carry a provenance hash of the fetched PDF
    for key in ("maccamy_fuchs_1954", "maruo_1960"):
        sha = res["literature"][key]["transcription"]["pdf_sha256"]
        assert len(sha) == 64 and int(sha, 16) >= 0