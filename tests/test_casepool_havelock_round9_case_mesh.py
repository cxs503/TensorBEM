"""Round 9 guards: the recovered case-mesh measurement must stay auditable.

Anchor ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK``.  These tests do not re-solve
anything; they re-derive the published numbers from the recovered raw rows and
fail if the artefact, the adjudication or the provenance ever drift apart.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "docs" / "case_bem_submerged_sphere_havelock_round9"
REC = ROUND / "recovered"
COEFFICIENTS = ("a33", "b33", "a11", "b11")
GATE = 0.05


def _study() -> dict:
    return json.loads((REC / "numbers_study_full.json").read_text())


def _adj() -> dict:
    return json.loads((ROUND / "adjudicate.json").read_text())


def _rows(tier: str) -> list[dict]:
    return list(_study()["tiers"][tier]["rows"])


def test_provenance_hashes_are_pinned() -> None:
    prov = json.loads((REC / "PROVENANCE.json").read_text())
    for rel, expected in prov["sha256"].items():
        actual = hashlib.sha256((REC / rel.replace("recovered/", "")).read_bytes()).hexdigest()
        assert actual == expected, f"{rel} drifted from the pinned recovery hash"
    assert prov["source_commit"] == "6d3725781e4122e374c706dc8bc988a117895014"


def test_gate_rederived_per_item() -> None:
    """The 5 % gate is recomputed here from got/ref, independent of any stored verdict."""
    adj = _adj()
    for tier, per_measure in adj["all_tier_verdicts"].items():
        for measure, published in per_measure.items():
            sub = [r for r in _rows(tier) if r["measure"] == measure]
            errors = [
                abs(float(r[c]) - float(r[f"{c}_ref"])) / abs(float(r[f"{c}_ref"]))
                for r in sub
                for c in COEFFICIENTS
            ]
            assert published["n_items"] == len(errors)
            assert published["n_items_over_gate"] == sum(1 for e in errors if e > GATE)
            assert published["band_max_rel_error"] == pytest.approx(max(errors), rel=1e-12)
            assert published["verdict"] == ("PASS" if max(errors) <= GATE else "FAIL")


def test_case_mesh_is_the_gated_mesh_and_passes() -> None:
    adj = _adj()
    t3 = adj["case_mesh_verdicts"]
    assert adj["all_tier_verdicts"]["T3:case_s3_band"]["flat"]["n_items"] == 12
    for measure in ("flat", "patch", "uniform"):
        assert t3[measure]["verdict"] == "PASS"
        assert t3[measure]["band_max_rel_error"] < GATE
    # the intervention must reduce the case-mesh band-max by at least a factor 2 (P1)
    assert adj["predeclared_P1_P2_P3"]["P1"]["holds"] is True
    assert (
        t3["patch"]["band_max_rel_error"] * 2.0
        <= t3["flat"]["band_max_rel_error"]
    )


def test_routing_identity_against_gated_table() -> None:
    """Recovered flat rows must equal the published case table (same code path)."""
    committed = {
        round(float(r["ka"]), 6): r
        for r in json.loads(
            (ROOT / "docs" / "case_bem_submerged_sphere_havelock" / "numbers.json").read_text()
        )["band_sweep_case"]
    }
    checked = 0
    for row in _rows("T3:case_s3_band"):
        if row["measure"] != "flat":
            continue
        ka = round(float(row["ka"]), 6)
        if ka not in committed:
            continue
        checked += 1
        for c in COEFFICIENTS:
            ref = float(committed[ka][f"{c}_bem"])
            assert abs(float(row[c]) - ref) / abs(ref) < 1e-12
    assert checked == 3
    assert _adj()["checks"]["routing_identity_R1"]["status"] == "PASS"


def test_registered_prediction_is_judged_not_extrapolated() -> None:
    ra = _adj()["registered_prediction_RA4"]
    assert ra["verdict"] == "FALSIFIED"  # measured/predicted = 2.52 > 2
    assert ra["measured_over_predicted"] > 2.0
    # the measurement exists: the verdict must not be a not_run placeholder
    assert ra["measured_patch_band_max"] > 0.0


def test_honest_not_run_is_declared() -> None:
    adj = _adj()
    assert adj["bands"]["full_band_not_measured"] == [0.5, 2.0, 4.0, 5.0]
    assert any("widened" in s for s in adj["not_run"])


def test_uniform_control_is_not_distinguishable_on_case_mesh() -> None:
    """Honest reading: on the case mesh the per-panel measure ~= scalar rescaling."""
    pv = _adj()["patch_vs_uniform_control"]
    assert pv["max_abs_rel_gap_any_cell"] < 0.10
    t3 = _adj()["case_mesh_verdicts"]
    assert abs(t3["patch"]["band_max_rel_error"] - t3["uniform"]["band_max_rel_error"]) < 1e-5


def test_recovered_files_are_byte_identical_to_the_source_branch() -> None:
    """Skip (not fake-pass) when the source branch is not present in this clone."""
    prov = json.loads((REC / "PROVENANCE.json").read_text())
    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", prov["source_branch"]],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except Exception:  # pragma: no cover - depends on clone layout
        pytest.skip(f"branch {prov['source_branch']} not available in this clone")
    for rel, src in prov["source_artifacts"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{prov['source_branch']}:{src}"], cwd=ROOT
        )
        assert hashlib.sha256(blob).hexdigest() == prov["sha256"][rel]