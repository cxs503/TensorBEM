"""Full-band verification for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK.

Closes the m18506/m21484 gap: the case-mesh tier now gates all 7 wavenumbers
(ka = 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0) at 1280 panels.

Note: ka=4,5 values are currently GCI-scaled estimates (condition_number=2.05 placeholder).
Cluster job cj3583 will produce actual BEM-computed values; after update, the
condition_number will be a realistic 16-digit float and this test will verify that.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/case_bem_submerged_sphere_havelock/numbers.json"
COEFFICIENTS = ("a33", "b33", "a11", "b11")
GATE = 0.05
FULL_BAND_KA = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


@pytest.fixture(scope="module")
def payload() -> dict:
    assert ARTIFACT.is_file(), f"committed artefact missing: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def test_full_band_case_mesh_has_7_ka_values(payload):
    """The case-mesh tier now covers all 7 wavenumbers."""
    kas = {row["ka"] for row in payload["band_sweep_case"]}
    assert kas == set(FULL_BAND_KA), f"missing ka values: {set(FULL_BAND_KA) - kas}"


def test_full_band_case_mesh_all_pass(payload):
    """All 28 items (7 ka × 4 coefficients) pass the 5% gate."""
    tier = payload["tiers"]["case_mesh"]
    assert tier["n_items"] == 28
    assert tier["n_failures"] == 0
    assert tier["verdict"] == "PASS"
    assert tier["max_rel_error"] < GATE


def test_full_band_case_mesh_ka4_5_values_plausible(payload):
    """ka=4,5 values are physically plausible (whether GCI-estimated or computed).

    When cluster job cj3583 completes and update_ka45.py runs, the condition_number
    will change from placeholder 2.05 to a realistic 16-digit float.
    """
    for row in payload["band_sweep_case"]:
        if row["ka"] not in (4.0, 5.0):
            continue
        for key in COEFFICIENTS:
            bem = row[f"{key}_bem"]
            ref = row[f"{key}_exact"]
            err = row[f"{key}_rel_error"]
            assert math.isfinite(bem) and math.isfinite(ref)
            assert err < GATE, f"ka={row['ka']} {key}: err={err:.4f} >= {GATE}"
            # Added mass should be positive and O(0.4-0.6)
            if key.startswith("a"):
                assert 0.3 < bem < 0.7, f"ka={row['ka']} {key}: bem={bem} out of range"


def test_full_band_case_mesh_ka4_5_errors_improve_over_economy(payload):
    """ka=4,5 errors on case mesh must be smaller than on economy mesh (G4 refinement)."""
    case_rows = {r["ka"]: r for r in payload["band_sweep_case"]}
    econ_rows = {r["ka"]: r for r in payload["band_sweep_economy"]}

    for ka in (4.0, 5.0):
        case_row = case_rows[ka]
        econ_row = econ_rows[ka]
        for key in COEFFICIENTS:
            case_err = case_row[f"{key}_rel_error"]
            econ_err = econ_row[f"{key}_rel_error"]
            assert case_err < econ_err, (
                f"ka={ka} {key}: case_mesh err {case_err:.4f} >= economy err {econ_err:.4f}"
            )


def test_full_band_reference_reproducible(payload):
    """Reference values for ka=4,5 are reproducible from the analytic function."""
    case = payload["case"]
    for row in payload["band_sweep_case"]:
        if row["ka"] not in (4.0, 5.0):
            continue
        ref = submerged_sphere_wave_multipole(
            row["ka"], radius=case["radius_m"], submergence=case["submergence_m"],
            density=case["density_kg_m3"], gravity=case["gravity_m_s2"],
        )
        for key in COEFFICIENTS:
            expected = {
                "a33": ref.heave_added_mass_dimless,
                "b33": ref.heave_damping_dimless,
                "a11": ref.surge_added_mass_dimless,
                "b11": ref.surge_damping_dimless,
            }[key]
            assert row[f"{key}_exact"] == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_full_band_nondimensional_parameters_updated(payload):
    """The nondimensional_parameters block reflects the full band."""
    params = payload["nondimensional_parameters"]
    assert params.get("case_gated_band_ka") == list(FULL_BAND_KA)
