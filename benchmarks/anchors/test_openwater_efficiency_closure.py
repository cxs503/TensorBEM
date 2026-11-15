"""Regression tests for the w30 open-water anchor closure (mission m21514).

These tests assert *invariants*, not "the script ran":

1. every registered anchor has >= 5 advance ratios, a non-empty machine-extracted
   quotation per row, and a sha256 that re-hashes to the declared value;
2. the efficiency identity ``eta0 = J KT / (2 pi KQ)`` closes on every anchor row
   to better than the predeclared 0.5 %;
3. the ten-significant-digit reproduction of the Oosterveld & van Oossanen
   closed form from the in-repo coefficients still matches the registered B4-70
   curve (i.e. the coefficients were not silently edited);
4. the eta_R audit still finds the platform's ``relative_rot_eff`` constant and
   still puts 9 of 11 states outside the admissible window when inverted against
   the resistance module -- if either count moves, the artefact must be
   regenerated and re-reviewed rather than silently drifting;
5. the archived artefact on disk agrees with a fresh regeneration.

Run::

    PYTHONPATH=src python -m pytest benchmarks/anchors/test_openwater_efficiency_closure.py -q
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).resolve().parent / "build_openwater_efficiency_closure.py"
ARTIFACT = REPO / "papers" / "artifacts" / "w30" / "propeller-openwater-bem" / "openwater_efficiency_closure.json"


def _load():
    spec = importlib.util.spec_from_file_location("openwater_closure", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def anchors(mod):
    return mod.ovo_anchor(), mod.kp505_anchor()


def test_anchor_point_counts_and_quotes(anchors):
    ovo, kp505 = anchors
    for anchor in (ovo, kp505):
        assert anchor["n_points"] >= 5
        assert anchor["n_points"] == len(anchor["points"])
        for row in anchor["points"]:
            assert row["quote"].strip(), "every anchor row must carry a quotation"
            assert row["quote_status"] in {
                "PUBLISHED_TABLE_NOT_IN_WORKTREE",
                "VERBATIM_TRANSCRIPT_IN_WORKTREE",
            }


def test_kp505_sha256_rehashes_to_declared(anchors):
    _, kp505 = anchors
    checks = kp505["carrier"]["sha256_checks"]
    assert checks["transcript_sha256_matches_declared"] is True
    assert checks["pdf_sha256_matches_declared"] is True


def test_eta0_identity_closes_on_every_anchor_row(anchors, mod):
    ovo, kp505 = anchors
    for anchor in (ovo, kp505):
        for row in anchor["points"]:
            assert abs(row["eta0_residual_pct"]) <= mod.ETA0_RESIDUAL_MAX_PCT, row


def test_eta0_identity_formula_is_not_a_copy(anchors, mod):
    """The independent path must be fed KQ = 10KQ/10, so a 10x convention slip shows up."""

    _, kp505 = anchors
    row = kp505["points"][0]
    wrong = mod.eta0_from(row["KT"], row["10KQ"], row["J"])   # feeds 10KQ where KQ belongs
    assert abs(mod.rel_pct(wrong, row["eta0_registered"])) > 50.0


def test_ovo_coefficients_still_reproduce_registered_curve(anchors):
    ovo, _ = anchors
    for row in ovo["points"]:
        assert abs(row["KT_reproduction_pct"]) < 1e-6
        assert abs(row["KQ_reproduction_pct"]) < 1e-6


def test_etar_audit_counts_are_frozen(mod):
    audit = mod.etar_audit()
    summary = audit["summary"]
    assert summary["n_states"] == 11
    assert summary["distinct_etaR_returned"] == [1.0], "platform eta_R is expected to be a constant"
    assert summary["n_etaR_returned_out_of_window"] == 0
    assert summary["n_etaR_cross_module_out_of_window"] == 9
    for row in audit["states"]:
        assert math.isfinite(row["etaR_cross_module_implied"])
        # the identity-implied value is 1.0 by construction: documented, not asserted as physics
        assert abs(row["etaR_identity_implied"] - 1.0) < 0.02


def test_archived_artifact_matches_regeneration(mod, tmp_path):
    assert ARTIFACT.exists(), "run benchmarks/anchors/build_openwater_efficiency_closure.py first"
    # the builder records repo-relative paths, so the regeneration target must live
    # inside the worktree (tmp_path is outside it)
    scratch = REPO / "data" / "w30" / "_regen_check"
    fresh = mod.build(scratch)
    archived = json.loads(ARTIFACT.read_text())
    for key in ("criteria", "eta0_identity", "solver_vs_anchor", "etaR_audit"):
        assert json.dumps(fresh[key], sort_keys=True) == json.dumps(archived[key], sort_keys=True), key
    for name in ("solver_vs_anchor.csv", "etar_audit.csv", "eta0_identity.csv"):
        (scratch / name).unlink(missing_ok=True)
    (scratch / "openwater_efficiency_closure.json").unlink(missing_ok=True)