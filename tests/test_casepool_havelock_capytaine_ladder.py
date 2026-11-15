"""Guards for the capytaine refinement study of the Havelock submerged-sphere case.

The study (``scripts/case_havelock_capytaine_refinement_ladder.py``,
``docs/case_bem_submerged_sphere_havelock/capytaine_refinement_ladder.json``)
answers how far the third-party cross-check can certify the multipole reference.
These tests bind the committed artefact to its own claims and to the note, so a
later edit cannot silently upgrade a FAIL into a PASS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DOCS = Path("docs/case_bem_submerged_sphere_havelock")
LADDER = DOCS / "capytaine_refinement_ladder.json"
CROSSCHECK = DOCS / "capytaine_crosscheck.json"
PROBE = DOCS / "gf_singularity_probe.json"
NOTE = DOCS / "capytaine_resolution_note.md"

COEFFICIENTS = ("a33", "b33", "a11", "b11")


@pytest.fixture(scope="module")
def ladder() -> dict:
    return json.loads(LADDER.read_text())


@pytest.fixture(scope="module")
def block(ladder: dict) -> dict:
    return ladder["settings"]["low_freq"]


def test_ladder_levels_are_self_similar(block: dict) -> None:
    """The four levels must be 80/320/1280/5120 panels with a shrinking volume deficit."""
    levels = sorted(block["ladder"], key=lambda lv: lv["n_panels"])
    assert [lv["n_panels"] for lv in levels] == [80, 320, 1280, 5120]
    deficits = [lv["volume_deficit"] for lv in levels]
    assert all(a > b for a, b in zip(deficits, deficits[1:])), deficits
    # an inscribed polyhedron of the sphere loses O(h^2): the deficit must fall ~4x
    for a, b in zip(deficits, deficits[1:]):
        assert 3.5 < a / b < 4.5, (a, b)


def test_items_are_internally_consistent(block: dict) -> None:
    items = block["items"]
    assert len(items) == 28
    for it in items:
        assert it["quantity"] in COEFFICIENTS
        assert len(it["per_level"]) == 4
        finest = it["per_level"][-1]
        mid = it["per_level"][-2]
        assert it["finest_signed_deviation"] == pytest.approx(finest["signed_rel_deviation"])
        assert it["finest_abs_deviation"] == pytest.approx(abs(finest["signed_rel_deviation"]))
        assert it["decay_ratio"] == pytest.approx(
            it["finest_abs_deviation"] / it["mid_abs_deviation"])
        assert it["decay_ratio"] > 0.0
        assert mid["n_panels"] == 1280


def test_predeclared_criteria_and_honest_verdict(block: dict) -> None:
    """The 1 % criterion is NOT met: the artefact must keep saying FAIL."""
    summary = block["summary"]
    assert summary["n_items"] == 28
    assert summary["decay_fraction"] >= 0.9, "decay is the evidence for 'converges to the reference'"
    assert summary["worst_active_abs_deviation"] > 0.01, "if this ever passes, update the note"
    assert summary["worst_active_abs_deviation"] < 0.02
    assert block["verdict"] == "FAIL"
    assert json.loads(LADDER.read_text())["verdict"] == "FAIL"


def test_single_mesh_crosscheck_is_floored_by_capytaine(ladder: dict, block: dict) -> None:
    """The two-mesh crosscheck is strictly worse than the refined ladder -- it is
    a capytaine discretisation statement, not a reference-quality statement."""
    crosscheck = json.loads(CROSSCHECK.read_text())
    assert crosscheck["verdict"] == "FAIL"
    assert crosscheck["worst"]["rel_error"] > 0.05
    assert crosscheck["worst"]["rel_error"] > block["summary"]["worst_finest_abs_deviation"]
    # the artefact has no free-text claim field; the note carries the explanation
    note = NOTE.read_text()
    assert "离散" in note and "320" in note


def test_gf_singularity_hypothesis_is_recorded_as_falsified() -> None:
    probe = json.loads(PROBE.read_text())
    assert probe["verdict"] == "falsified"
    rows = probe["signed_deviation_vs_multipole_ref_percent"]
    for level, variants in rows.items():
        a, b = variants["low_freq"], variants["low_freq_with_rankine_part"]
        for key in COEFFICIENTS:
            assert a[key] == pytest.approx(b[key], abs=1e-3), (level, key)


def test_tensorbem_agrees_with_the_third_party_fine_mesh(ladder: dict) -> None:
    block = ladder["tensorbem_vs_capytaine_finest"]
    assert block["n_items"] == 28
    assert block["n_over_pool_tolerance"] == 0
    assert block["max_abs_rel_error"] < 0.05
    # and the honest ordering: mutual agreement is coarser than the 1.43 % self-report
    assert block["max_abs_rel_error"] > 0.0143


def test_note_states_the_unachieved_certification() -> None:
    text = NOTE.read_text()
    assert "C1" in text and "FAIL" in text
    assert "约 2%" in text or "~2%" in text