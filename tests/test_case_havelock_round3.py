"""CI guard for round 3 of ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK``.

Round 3 exists to close three gaps that round 2 declared honestly:

* the seven-point ``ka`` band was never recomputed end to end;
* the 1280-panel refinement was never produced under the spherical-patch measure;
* the reference-integrity section was injected as ``nan`` / empty because the
  generator read a key that does not exist (``truncation.ka_values`` -- the real
  key is ``truncation.rows``).

These tests guard the *mechanism*, not just the numbers:

* the assembled artefact in-repo is internally consistent and every item is
  recomputable from its own got/ref pair;
* the non-finite / empty-table guards actually fire on the round-2 failure mode;
* the refinement pairing refuses mismatched ``ka``;
* the document renderer refuses an artefact containing NaN and emits one row per
  judged item (no silent dropping);
* one live solve through the public API still reproduces the committed
  ``ka=1.0`` rung, so the artefact cannot rot away from the solver.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from tensorbem.benchmarks import case_havelock_submerged_sphere as case
from tensorbem.benchmarks import submerged_sphere_havelock as library

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "case_bem_submerged_sphere_havelock" / "round3_numbers.json"
DOC = ROOT / "docs" / "case_submerged_sphere_havelock_round3.md"
COEFFICIENTS = ("a33", "b33", "a11", "b11")
GATE = 0.05

sys.path.insert(0, str(ROOT / "tools"))
import case_havelock_round3_doc as docgen  # noqa: E402
import case_havelock_round3_stage as stage  # noqa: E402


@pytest.fixture(scope="module")
def art() -> dict:
    assert ARTIFACT.is_file(), "committed round-3 artefact missing: %s" % ARTIFACT
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# artefact consistency
# ---------------------------------------------------------------------------

def test_band_is_the_declared_seven_points(art):
    assert [row["ka"] for row in art["frequency_sweep"]] == [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [row["ka"] for row in art["control_frequency_sweep"]] == [
        row["ka"] for row in art["frequency_sweep"]]


def test_every_rel_error_recomputes_from_its_own_pair(art):
    """The stored error must follow from the stored got/ref -- no hand editing."""
    for block in ("frequency_sweep", "control_frequency_sweep", "refinement_sweep_1280"):
        for row in art[block]:
            for key in COEFFICIENTS:
                got, ref = row["%s_bem" % key], row["%s_exact" % key]
                expected = abs(got - ref) / abs(ref)
                assert row["%s_rel_error" % key] == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_production_and_refinement_meshes_are_the_declared_ones(art):
    assert {row["n_panels"] for row in art["frequency_sweep"]} == {320}
    if art.get("refinement_status") != "completed":
        assert art["refinement_sweep_1280"] == []
        pytest.skip("1280-panel refinement honestly absent in this artefact")
    assert {row["n_panels"] for row in art["refinement_sweep_1280"]} == {1280}


def test_primary_measure_passes_the_predeclared_gate(art):
    verdict = art["judgement"]["production_sweep_spherical"]
    assert verdict["tolerance"] == GATE
    assert verdict["n_items"] == 28, "7 frequencies x 4 coefficients"
    assert verdict["passed"] is True, verdict["failures"]
    assert verdict["worst_item"]["rel_error"] < GATE


def test_flat_measure_control_is_reported_not_hidden(art):
    """Round 1's FAIL must stay visible as the control arm."""
    verdict = art["judgement"]["production_sweep_flat_control"]
    assert verdict["n_failures"] > 0
    assert verdict["passed"] is False
    assert max(row["b33_rel_error"] for row in art["control_frequency_sweep"]) > GATE


def test_artifact_contains_no_non_finite_value(art):
    assert stage._find_nan(art) == []


# ---------------------------------------------------------------------------
# the round-2 failure modes are now caught
# ---------------------------------------------------------------------------

def test_nan_guard_catches_the_round2_injection_defect():
    """Round 2 shipped ``nan``; the assembler/renderer must refuse such a payload."""
    poisoned = {"reference_integrity": {"truncation": {"max_truncation_rel_delta": float("nan")}}}
    assert stage._find_nan(poisoned) == [".reference_integrity.truncation.max_truncation_rel_delta"]
    assert stage._find_nan({"ok": 1.0, "rows": [{"x": 2.0}]}) == []


def test_integrity_block_is_keyed_on_rows_not_a_missing_key(art):
    """The exact bug: ``truncation.ka_values`` never existed."""
    truncation = art["reference_integrity"]["truncation"]
    assert "ka_values" not in truncation
    assert [row["ka"] for row in truncation["rows"]] == [0.1, 1.0, 5.0]
    for row in truncation["rows"]:
        assert math.isfinite(row["max_truncation_rel_delta"])
        assert row["max_truncation_rel_delta"] < 1e-9
        assert row["haskind_error_z"] < 1e-12


def test_deep_submergence_limit_returns_to_lamb(art):
    rows = art["reference_integrity"]["deep_submergence_limit"]["rows"]
    deepest = max(rows, key=lambda row: row["submergence_ratio"])
    assert deepest["submergence_ratio"] >= 18.0
    assert deepest["a33_excess_over_lamb"] < 1e-3
    assert deepest["inertia_coefficient_cm"] == pytest.approx(1.5, abs=1e-3)


@pytest.mark.parametrize("payload", [
    {"ka": "bad"},
])
def test_refinement_pairing_refuses_mismatched_ka(payload):
    coarse = [{"ka": 0.5, "n_panels": 320, "a33_rel_error": 0.01, "b33_rel_error": 0.01,
               "a11_rel_error": 0.01, "b11_rel_error": 0.01}]
    fine = [{"ka": 1.0, "n_panels": 1280, "a33_rel_error": 0.005, "b33_rel_error": 0.005,
             "a11_rel_error": 0.005, "b11_rel_error": 0.005}]
    with pytest.raises(AssertionError):
        stage._refine_order({"sub2": coarse, "sub3": fine})


def test_missing_shard_is_an_error_not_a_silent_empty_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "SHARD_DIR", tmp_path)
    with pytest.raises(SystemExit):
        stage.collect_shards(library.AREA_MEASURE_SPHERICAL, 2, (9.99,))


def test_refinement_error_ratios_are_computed_per_ka(art):
    if art.get("refinement_status") != "completed":
        assert art["refinement_error_ratios"] == []
        pytest.skip("no refinement shards, so no error ratios")
    for entry in art["refinement_error_ratios"]:
        assert entry["n_panels_coarse"] == 320 and entry["n_panels_fine"] == 1280
        for key in COEFFICIENTS:
            assert math.isfinite(entry["%s_error_ratio" % key])


# ---------------------------------------------------------------------------
# document renderer
# ---------------------------------------------------------------------------

def test_renderer_refuses_a_nan_artifact(art, tmp_path):
    bad = json.loads(json.dumps(art))
    bad["judgement"]["production_sweep_spherical"]["worst_item"]["rel_error"] = float("nan")
    with pytest.raises(SystemExit):
        docgen.render(bad, tmp_path)


def test_renderer_emits_one_verdict_row_per_judged_item(art, tmp_path):
    text = docgen.render(art, tmp_path)
    n_items = art["judgement"]["production_sweep_spherical"]["n_items"]
    section = text.split("## ⑤ 误差判定")[1].split("**结论**")[0]
    assert section.count("| 是 |") + section.count("| 否 |") == n_items
    # prose may *mention* nan when explaining round 2's defect; rendered values may not be nan
    assert "| nan" not in text.lower()
    assert "nan%" not in text.lower()


def test_published_document_matches_the_artifact(art):
    assert DOC.is_file(), "round-3 document missing: %s" % DOC
    text = DOC.read_text()
    for heading in ("## ① 案例介绍", "## ② 怎么做的", "## ③ 计算结果",
                    "## ④ 结果图", "## ⑤ 误差判定", "## ⑥ 注意点与局限"):
        assert heading in text, heading
    worst = art["judgement"]["production_sweep_spherical"]["worst_item"]["rel_error"]
    assert docgen._pct(worst) in text, "worst-case error is not injected into the document"


# ---------------------------------------------------------------------------
# live solver call (integration)
# ---------------------------------------------------------------------------

def test_live_solve_reproduces_the_committed_ka1_rung(art):
    """One 320-panel solve through the public API against the stored row."""
    stored = next(row for row in art["frequency_sweep"] if row["ka"] == 1.0)
    live = library.run_frequency_sweep(
        ka_values=(1.0,), radius=1.0, submergence=2.0, density=1025.0, gravity=9.80665,
        subdivision=2, area_measure=library.AREA_MEASURE_SPHERICAL,
    )[0]
    for key in COEFFICIENTS:
        assert live["%s_bem" % key] == pytest.approx(stored["%s_bem" % key], rel=1e-9)
        assert live["%s_rel_error" % key] < GATE
    assert case.CASE_TOLERANCE == GATE