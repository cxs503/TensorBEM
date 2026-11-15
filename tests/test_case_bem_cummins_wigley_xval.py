"""Tests for gate C10: the external (capytaine) anchor of the Cummins/Wigley case.

The reference is a frozen artifact, so every test here runs without capytaine
installed.  Two invariants are the point of the suite:

* the reference was built on the *same panels* the committed sweep used
  (recomputed fingerprint, not a stored claim);
* the gate **bites**: a reference that is off by more than the case tolerance
  must fail, otherwise "PASS" says nothing.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.benchmarks import case_bem_cummins_wigley_xval as xval


@pytest.fixture(scope="module")
def reference() -> dict:
    return xval.load_reference()


@pytest.fixture(scope="module")
def sweep() -> case.Sweep:
    return case.load_sweep(xval.SWEEP_PATH)


def test_reference_is_frozen_with_provenance(reference) -> None:
    assert reference["code"] == "capytaine"
    for key in ("capytaine_version", "platform", "python"):
        assert reference["provenance"][key]
    assert reference["settings"]["dof_order"] == list(case.DOF_NAMES)
    assert reference["settings"]["rotation_center"] == list(case.CENTER)
    assert np.allclose(reference["settings"]["band_rad_s"], case.BAND)


def test_reference_sits_on_the_committed_panels(reference) -> None:
    """Same vertices, same triangles: the gate's whole premise."""
    assert xval.geometry_matches(reference, "medium")
    assert xval.geometry_matches(reference, "coarse")
    # the fingerprint must reject a different mesh, not just echo a field
    tampered = json.loads(json.dumps(reference))
    tampered["levels"]["medium"]["geometry"]["sha256"] = "0" * 64
    assert not xval.geometry_matches(tampered, "medium")


def test_gate_passes_and_reports_its_numbers(reference, sweep) -> None:
    gate = xval.external_anchor_gate(sweep, reference)
    assert gate["id"] == "C10"
    assert gate["passed"] is True
    assert gate["status"] == "passed"
    values = gate["values"]
    assert values["identical_panels"] is True
    assert values["scored_cells"] > 0
    assert 0.0 <= values["worst"]["relative"] <= gate["tolerance"]
    assert values["dof_order" if False else "worst"]["dof_to"] in case.DOF_NAMES
    # the 3 % bar must be reported next to the 10 % criterion, not conflated
    assert values["target_fraction"] == pytest.approx(xval.TARGET_FRACTION)
    assert 0 <= values["cells_within_target"] <= values["scored_cells"]


def test_gate_bites_on_a_perturbed_reference(reference, sweep) -> None:
    """A 15 %-off foreign reference must fail the 10 % gate."""
    tampered = json.loads(json.dumps(reference))
    block = tampered["levels"]["medium"]["variants"][xval.PRIMARY_VARIANT]
    added = np.asarray(block["added_mass"], dtype=float)
    added[:, 2, 2] *= 1.15
    block["added_mass"] = added.tolist()
    gate = xval.external_anchor_gate(sweep, tampered)
    assert gate["passed"] is False
    assert gate["values"]["worst"]["dof_to"] == "heave"


def test_gate_fails_when_the_reference_geometry_is_not_the_sweep_geometry(reference, sweep) -> None:
    tampered = json.loads(json.dumps(reference))
    tampered["levels"]["medium"]["geometry"]["sha256"] = "0" * 64
    gate = xval.external_anchor_gate(sweep, tampered)
    assert gate["values"]["identical_panels"] is False
    assert gate["passed"] is False


def test_null_cells_are_excluded_not_scored(reference, sweep) -> None:
    rows = xval.compare(sweep, reference)
    scored = np.asarray(rows["relative_matrices"]["added_mass"])
    assert np.all(scored >= 0.0)
    assert len(rows["null_cells"]) > 0, "the 6x6 matrix must have coupling zeros to classify"
    amplitude = np.max(np.abs(np.asarray(sweep.added_mass, dtype=float)), axis=0)
    damping_amplitude = np.max(np.abs(np.asarray(sweep.damping, dtype=float)), axis=0)
    for cell in rows["null_cells"]:
        i = case.DOF_NAMES.index(cell["dof_to"])
        j = case.DOF_NAMES.index(cell["dof_from"])
        reference_scale = amplitude if cell["quantity"] == "added_mass" else damping_amplitude
        assert cell["amplitude"] == pytest.approx(reference_scale[i, j])


def test_frequency_grid_mismatch_is_refused(reference, sweep) -> None:
    tampered = json.loads(json.dumps(reference))
    tampered["omega"] = [float(w) * 1.01 for w in tampered["omega"]]
    with pytest.raises(ValueError):
        xval.compare(sweep, tampered)


def test_attribution_is_falsifiable(reference, sweep) -> None:
    attr = xval.attribution(sweep, reference)
    assert attr["flagged"] == len(attr["flagged_frequencies_rad_s"])
    assert "testable" in attr and "lid_effect_by_level" in attr
    if not attr["testable"] or attr["flagged"] == 0:
        assert attr["supported"] is False
    else:
        assert attr["supported"] == (
            attr["repaired"] >= max(1, int(0.5 * attr["flagged"]))
        )
    # the lid must be reported with the number of panels it actually contributed
    for level, effect in attr["lid_effect_by_level"].items():
        assert effect["lid_panel_count"] >= 0
        assert effect["added_mass"] >= 0.0


def test_document_carries_the_verdict(reference, sweep) -> None:
    study = xval.run_study(reference, sweep)
    text = xval.render_document(study, figure="fig.png")
    assert "C10" in text
    assert study["gate"]["status"].upper() in text
    assert f"{100.0 * study['gate']['values']['worst']['relative']:.2f}%" in text
    assert "fig.png" in text


def test_xval_modules_never_import_the_foreign_code() -> None:
    """capytaine is optional: the case, its gate and its tests must not need it."""
    modules = [Path(xval.__file__), Path(case.__file__)]
    for path in modules:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("import capytaine", "from capytaine")), path


def test_write_outputs_round_trip(reference, sweep, tmp_path) -> None:
    study = xval.run_study(reference, sweep)
    paths = xval.write_outputs(study, root=tmp_path)
    assert paths["json"].exists() and paths["md"].exists()
    reloaded = json.loads(paths["json"].read_text())
    assert reloaded["gate"]["status"] == study["gate"]["status"]


def test_load_reference_rejects_a_foreign_producer(tmp_path) -> None:
    payload = json.loads(xval.REFERENCE_PATH.read_text())
    payload["code"] = "another-code"
    target = tmp_path / "fake.json"
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        xval.load_reference(target)