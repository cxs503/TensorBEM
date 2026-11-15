"""Provenance guards for the case-mesh rows of CASE-BEM-SUBMERGED-SPHERE-HAVELOCK.

Round-4 adversarial audit (m21580, major finding #2): the ``ka = 4, 5`` rows of
the case-mesh tier were **GCI-scaled estimates** taken from the 320-panel
economy mesh -- with a placeholder ``condition_number = 2.05`` -- while the case
verdict claimed the whole band was a computed PASS.  These tests make that
state unrepresentable:

* every row of ``band_sweep_case`` must carry a ``provenance`` marker, and all
  of them must be ``computed_case_mesh_1280*``;
* the two former estimate rows must retain their superseded estimates under
  ``superseded_gci_estimate`` (audit trail, nothing silently overwritten);
* the committed values must agree with the recorded compute artefact
  (``ka45_computation.estimate_vs_computed``), which is itself bound to the
  script ``scripts/havelock_ka45_case_mesh_compute.py`` and guarded by a
  route-identity control at ``ka = 3.0``;
* the reference side is re-derived from the analytic multipole series (no
  transcription);
* the declared 5 % criterion is still applied to all 28 items, and the
  degenerate high-ka damping items are flagged explicitly instead of being
  silently counted as informative PASSes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/case_bem_submerged_sphere_havelock/numbers.json"
COMPUTED = ROOT / "docs/case_bem_submerged_sphere_havelock/ka45_case_mesh_computed.json"
COEFFICIENTS = ("a33", "b33", "a11", "b11")
GATE = 0.05
FULL_BAND_KA = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
REPLACED_KA = (4.0, 5.0)


def _significant_digits(value: float) -> int:
    """Number of significant decimal digits of ``value`` (0 for round placeholders)."""
    mantissa = f"{abs(float(value)):.17g}"
    if "e" in mantissa:
        mantissa = mantissa.split("e")[0]
    mantissa = mantissa.replace("-", "").replace(".", "").lstrip("0").rstrip("0")
    return len(mantissa)


@pytest.fixture(scope="module")
def payload() -> dict:
    assert ARTIFACT.is_file(), f"committed artefact missing: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def computed() -> dict:
    assert COMPUTED.is_file(), f"compute artefact missing: {COMPUTED}"
    return json.loads(COMPUTED.read_text())


def test_every_case_mesh_row_is_marked_computed(payload):
    """No row of the case-mesh tier may be an unmarked estimate."""
    for row in payload["band_sweep_case"]:
        assert "provenance" in row, f"ka={row['ka']}: row has no provenance marker"
        assert row["provenance"].startswith("computed_case_mesh_1280"), (
            f"ka={row['ka']}: provenance={row['provenance']!r} is not a 1280-panel computation")


def test_replaced_rows_keep_audit_trail(payload):
    """The superseded GCI estimates stay visible on the replaced rows."""
    rows = {float(r["ka"]): r for r in payload["band_sweep_case"]}
    for ka in REPLACED_KA:
        row = rows[ka]
        assert row["n_panels"] == 1280
        superseded = row.get("superseded_gci_estimate")
        assert superseded, f"ka={ka}: superseded estimate not recorded"
        for key in COEFFICIENTS:
            assert "bem" in superseded[key] and "rel_error" in superseded[key]
        assert superseded["condition_number"] == pytest.approx(2.05), (
            "the superseded row should document the 2.05 placeholder condition number")


def test_condition_numbers_are_not_placeholders(payload):
    """Every case-mesh row carries a full-precision system condition number."""
    for row in payload["band_sweep_case"]:
        cond = row.get("condition_number")
        assert cond is not None and math.isfinite(cond), f"ka={row['ka']}: bad condition_number"
        assert _significant_digits(cond) >= 10, (
            f"ka={row['ka']}: condition_number={cond!r} looks like a placeholder")


def test_committed_values_match_compute_artefact(payload, computed):
    """Committed ka=4,5 numbers are the ones the compute artefact recorded."""
    rows = {float(r["ka"]): r for r in payload["band_sweep_case"]}
    recorded = {float(e["ka"]): e for e in computed["estimate_vs_computed"]}
    for ka in REPLACED_KA:
        row, rec = rows[ka], recorded[ka]
        for key in COEFFICIENTS:
            assert row[f"{key}_bem"] == pytest.approx(rec["quantities"][key]["computed"], rel=1e-12)
            assert row[f"{key}_rel_error"] == pytest.approx(
                rec["quantities"][key]["computed_rel_error"], rel=1e-9)


def test_route_control_binds_numbers_to_the_table(payload, computed):
    """ka=3.0 must be reproduced by the same run -> the new rows share the route."""
    control = computed["route_control"]
    assert float(control["ka"]) == 3.0
    assert control["route_identical"], (
        f"route control failed: max rel diff {control['max_rel_diff']:.3e}")
    assert computed["geometry_audit"]["verdict"] == "PASS"
    assert computed["geometry_audit"]["n_panels"] == 1280
    assert payload["ka45_computation"]["route_control"]["route_identical"]


def test_reference_side_rederived_from_multipoles(payload):
    """Reference values at ka=4,5 re-derived from the Havelock multipole series."""
    case = payload["case"]
    for row in payload["band_sweep_case"]:
        if row["ka"] not in REPLACED_KA:
            continue
        ref = submerged_sphere_wave_multipole(
            row["ka"], radius=case["radius_m"], submergence=case["submergence_m"],
            density=case["density_kg_m3"], gravity=case["gravity_m_s2"])
        expected = {"a33": ref.heave_added_mass_dimless, "b33": ref.heave_damping_dimless,
                    "a11": ref.surge_added_mass_dimless, "b11": ref.surge_damping_dimless}
        for key in COEFFICIENTS:
            assert row[f"{key}_exact"] == pytest.approx(expected[key], rel=1e-12, abs=1e-15)
            assert row[f"{key}_rel_error"] == pytest.approx(
                abs(row[f"{key}_bem"] - row[f"{key}_exact"]) / abs(row[f"{key}_exact"]), rel=1e-9)


def test_declared_gate_still_applies_to_all_28_items(payload):
    """The pre-declared 5 % criterion is unchanged; all 28 items pass after the swap."""
    tier = payload["tiers"]["case_mesh"]
    assert tier["n_items"] == 28 == len(payload["band_sweep_case"]) * 4
    assert tier["gate"] == GATE
    assert tier["n_failures"] == 0
    assert tier["max_rel_error"] < GATE
    assert {float(r["ka"]) for r in payload["band_sweep_case"]} == set(FULL_BAND_KA)


def test_degenerate_items_flagged_without_relaxing_gate(payload):
    """Small-|ref| high-ka damping items are flagged, not silently counted."""
    block = payload["degenerate_items"]
    flagged = {(it["item"], it["quantity"]) for it in block["items"]}
    assert flagged, "expected some noise-dominated items at ka>=4"
    assert all(float(it["ref"]) < 1e-4 for it in block["items"])
    assert ("ka=5", "b11") in flagged or ("ka=5", "b33") in flagged
    # the scale-normalised error must never be used to flip the verdict
    assert "criterion" in block["note"] and "unchanged" in block["note"]
    assert payload["tiers"]["case_mesh"]["verdict"] == "PASS"


def test_verdict_note_reports_computed_band(payload):
    """The honest note must state that ka=4,5 are computed and the economy FAIL kept."""
    note = payload["verdict"]["honest_note"]
    assert "computed" in note
    assert "ka=4,5" in note or "ka=4" in note
    assert "economy" in note.lower()
    assert payload["verdict"]["economy_mesh"] == "FAIL"
    assert payload["verdict"]["overall"] == "PASS"