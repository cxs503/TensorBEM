"""Gates for case-pool entry CASE-BEM-WIGLEY3-WAVE-RES.

The case grades TensorBEM's canonical Kochin wave-resistance path against the
wave-angle (theta) form of Michell's far-field integral on the Wigley III hull.
What these tests protect:

* the geometry gate is evaluated *before* any resistance number (``Cb = 4/9``),
* the reference is converged (quadrature and truncation) so a "1e-4 agreement"
  cannot be an artefact of a loose reference,
* both production routes stay inside the 5 % band of the case brief,
* the *defect* is recorded: the legacy default ``michell_wave_resistance`` is
  still 10-25x low, so a silent re-wiring of the case onto it fails here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import case_wigley3_wave_resistance as case
from tensorbem.thin_ship_michell import WigleyHull, wigley_panels

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "case_bem_wigley3_wave_res"
    / "results.json"
)

L, B, T = case.HULL["L"], case.HULL["B"], case.HULL["T"]
FR = (0.20, 0.30, 0.40)


def _speed(fr: float) -> float:
    return fr * math.sqrt(case.G * L)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_case_id_and_hull_ratios() -> None:
    assert case.CASE_ID == "CASE-BEM-WIGLEY3-WAVE-RES"
    assert L / B == pytest.approx(10.0)
    assert B / T == pytest.approx(1.6)


def test_geometry_gate_before_resistance() -> None:
    gate = case.geometry_gate()
    assert gate["passed"] is True
    cb = next(row for row in gate["rows"] if row["quantity"] == "Cb")
    assert cb["derived"] == pytest.approx(4.0 / 9.0, rel=1e-12)
    assert cb["rel_error"] < 1e-12
    # the wetted area of this hull is a fixed reference for the Cw normalisation
    assert gate["wetted_area"] == pytest.approx(1.3391, rel=2e-3)


def test_reference_theta_form_quadrature_converges() -> None:
    reference = case.reference_values(FR)
    for row in reference["cases"]:
        levels = row["levels"]
        # successive refinement must reduce the change (cusp resolved by clustering)
        assert levels[-1]["rel_change_vs_finest"] < 1e-4
        assert row["gci"]["gci_fine"] < case.TOLERANCE["reference_gci"]
        # truncating the angular integral at sec(theta)=30 must not move the value
        assert row["truncation"]["rel_drift"]["30.0"] < case.TOLERANCE["reference_truncation_drift"]


def test_reference_agrees_with_the_independent_lambda_route() -> None:
    """Two quadrature families, two amplitude evaluations, one theory."""
    reference = case.reference_values(FR)
    for fr, row in zip(FR, reference["cases"]):
        canonical = case.michell_wave_resistance_canonical(L, B, T, _speed(fr), rho=case.RHO, g=case.G)
        assert abs(canonical - row["Rw"]) / row["Rw"] < 0.05


def test_production_kochin_inside_the_5_percent_band() -> None:
    reference = case.reference_values(FR)
    for fr, row in zip(FR, reference["cases"]):
        got = case.michell_wave_resistance_canonical(
            L, B, T, _speed(fr), rho=case.RHO, g=case.G, n_quad=256
        )
        assert abs(got - row["Rw"]) / row["Rw"] < case.TOLERANCE["resistance_rel_error"]


def test_discretised_panel_route_converges_to_reference() -> None:
    hull = WigleyHull(L=L, B=B, T=T)
    fr = 0.30
    speed = _speed(fr)
    reference = case.reference_values((fr,))["cases"][0]["Rw"]
    errors = []
    for n_x, n_z in case.PANEL_LEVELS:
        panels = wigley_panels(hull, n_x, n_z)
        rw = case.wave_resistance_from_amplitude(panels.amplitude, speed, lam_max=200.0, n_lam=256)
        errors.append(abs(rw - reference) / reference)
    assert errors[1] < case.TOLERANCE["resistance_rel_error"]
    assert errors[-1] < errors[1] < errors[0]


def test_resistance_is_positive_and_peaks_at_the_hump() -> None:
    reference = case.reference_values(case.FR_GRID)
    rw = {row["froude_number"]: row["Rw"] for row in reference["cases"]}
    assert all(value > 0.0 for value in rw.values())
    # the classical Wigley hump: Rw(0.30) > Rw(0.25) and Rw(0.35) < Rw(0.30)
    assert rw[0.30] > rw[0.25]
    assert rw[0.35] < rw[0.30]


def test_legacy_default_is_still_the_defect_this_case_avoids() -> None:
    probe = case.legacy_defect_probe(FR)
    ratios = [row["canonical_over_legacy"] for row in probe["rows"]]
    assert all(5.0 < ratio < 40.0 for ratio in ratios), ratios


def test_cw_normalisation_is_declared_and_consistent() -> None:
    reference = case.reference_values((0.30,))["cases"][0]
    gate = case.geometry_gate()
    cw_wetted = reference["Rw"] / (
        0.5 * case.RHO * reference["speed"] ** 2 * gate["wetted_area"]
    )
    cw_lt = reference["Rw"] / (0.5 * case.RHO * reference["speed"] ** 2 * L * T)
    assert cw_wetted > 0.0 and cw_lt > 0.0
    # both areas are stated in the artifact; the ratio is the area ratio
    assert cw_lt / cw_wetted == pytest.approx(gate["wetted_area"] / (L * T), rel=1e-12)


def test_artifact_contract_and_criteria(artifact: dict) -> None:
    assert artifact["case_id"] == case.CASE_ID
    assert artifact["schema"] == "case-pool-entry/v1"
    assert artifact["criteria"]["all_passed"] is True
    assert artifact["reference"]["provenance"]["table_status"]["status"] == "WITHHELD"
    assert artifact["reference"]["provenance"]["absolute_scale"]["status"] == "WITHHELD"
    assert len(artifact["comparison"]) == len(case.FR_GRID)
    for row in artifact["comparison"]:
        assert row["kochin_rel_error"] <= case.TOLERANCE["resistance_rel_error"]
        assert row["panels_by_level"][1]["rel_error"] <= case.TOLERANCE["resistance_rel_error"]
    assert artifact["repro_cmd"].startswith("PYTHONPATH=src python3 -m tensorbem.benchmarks.case_wigley3_wave_resistance")
    assert artifact["environment"]["git"]["commit"]


def test_artifact_reproduces_a_spot_value(artifact: dict) -> None:
    """Artifact numbers must be recomputable, not typed in."""
    row = artifact["comparison"][2]  # Fr = 0.30
    recomputed = case.michell_wave_resistance_canonical(
        L, B, T, row["speed"], rho=case.RHO, g=case.G, n_quad=256
    )
    assert recomputed == pytest.approx(row["Rw_kochin"], rel=1e-12)
    assert np.isfinite(row["Rw_reference"])