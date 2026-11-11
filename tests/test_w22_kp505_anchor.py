"""Tests for the W22 KP505 anchor recheck / five-J open-water study.

The tests pin the *plumbing* the artefacts depend on: the reachability pass
really hashes what it found, the transcription really parses its source (so the
numbers cannot drift from the text), the AN-ITTC57 evaluation is the formula and
not a memorised value, and the GCI table reports a known-order synthetic ladder
correctly.  They deliberately do not re-run the L3 lattice -- that belongs to
the driver, not to the test suite.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks import kp505_w22 as w22
from tensorbem.benchmarks.kp505_openwater_bem import ANCHOR as PINNED_ANCHOR

ARTPATH = Path("benchmarks/verified/kp505_anchor_w22")


# --------------------------------------------------------------------------- #
# A1: reachability
# --------------------------------------------------------------------------- #
def test_reachability_hashes_every_hit_file():
    reach = w22.reachability()
    assert reach["n_hit_files"] >= 1, "the KP505 anchor must be reachable in-repo"
    assert reach["glob_hit_files"] >= 1
    assert reach["anchor_table_source_present"] is True
    for record in reach["hit_files"]:
        assert len(record["sha256"]) == 64
        assert all(ch in "0123456789abcdef" for ch in record["sha256"])
        path = w22.REPO_ROOT / record["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"], f"sha256 mismatch for {record['file']}"


def test_reachability_classifies_layers():
    reach = w22.reachability()
    assert reach["layer_summary"].get("table", 0) >= 1
    assert set(reach["layer_summary"]) <= {"table", "figure", "geometry", "text"}


# --------------------------------------------------------------------------- #
# A2: transcription
# --------------------------------------------------------------------------- #
def test_transcription_every_field_carries_provenance():
    anchor = w22.transcribe_anchor()
    assert anchor["available"] is True
    assert anchor["honest_fail"] is False
    assert anchor["n_rows"] == 10
    assert anchor["n_fields"] == anchor["n_rows"] * len(w22._FIELD_ORDER)
    assert anchor["n_fields_with_source"] == anchor["n_fields"]
    source_sha = anchor["sha256"]
    for field in anchor["fields"]:
        assert field["file"] == w22.ANCHOR_TABLE_SOURCE
        assert field["sha256"] == source_sha
        assert field["line"] >= 1
        assert field["verbatim"]
        assert field["reference_type"] in {"experiment", "numerical", "analytic"}


def test_transcription_reproduces_the_pinned_anchor_table():
    anchor = w22.transcribe_anchor()
    assert anchor["cross_check_pinned_table_mismatches"] == []
    by_j = {row["J"]: row for row in anchor["experimental_curve"]}
    for advance, pinned in PINNED_ANCHOR.items():
        assert by_j[advance]["KT"] == pytest.approx(pinned["KT"], abs=0.0)
        assert by_j[advance]["10KQ"] == pytest.approx(pinned["10KQ"], abs=0.0)


def test_transcription_design_point_is_present():
    anchor = w22.transcribe_anchor()
    design = anchor["design_point"]
    assert design["J"] == pytest.approx(w22.DESIGN_J)
    assert design["present"] is True
    # eta0 must be derived from the transcribed columns, not typed
    expected = design["J"] * design["KT"] / (2.0 * math.pi * design["KQ"])
    assert design["eta0"] == pytest.approx(expected, rel=1e-12)


def test_source_self_consistency_is_recomputed():
    anchor = w22.transcribe_anchor()
    block = anchor["source_self_consistency"]
    assert block["n_checks"] == anchor["n_rows"] * 2
    # the printed error columns are the source's own, so the recomputation must
    # agree within the *row-specific* round-off bound (the table prints 3
    # decimals, so the smallest K_T already carries ~1.7 % relative rounding)
    assert block["max_ratio_to_roundoff_bound"] <= 1.0
    for row in block["rows"]:
        assert row["abs_diff_pct_points"] == pytest.approx(
            abs(row["recomputed_err_pct"] - row["printed_err_pct"]), abs=1e-12
        )
        assert row["abs_diff_pct_points"] <= row["roundoff_bound_pct_points"] + 1e-12


def test_ittc57_is_evaluated_from_the_formula():
    for re_value in (1.0e5, 1.0e6, 1.0e7):
        expected = w22.ITTC57["numerator"] / (
            math.log10(re_value) - w22.ITTC57["log10_shift"]
        ) ** 2
        assert float(w22.ittc57_cf(re_value)) == pytest.approx(expected, rel=1e-14)
    # monotone decreasing in Re
    values = [float(w22.ittc57_cf(r)) for r in (1.0e5, 1.0e6, 1.0e7)]
    assert values[0] > values[1] > values[2]


# --------------------------------------------------------------------------- #
# A4: GCI machinery on a synthetic ladder with a known order
# --------------------------------------------------------------------------- #
def _synthetic_points(order: float = 2.0, ratio: float = 2.0):
    """Three-level ladder whose differences shrink like ratio**(-order)."""

    exact = {"KT": 0.5000, "KQ": 0.0500, "eta0": 0.7000}
    # h shrinks by `ratio` per level; the error is C h**order
    levels = []
    for index, level in enumerate(w22.LADDER):
        h = ratio ** (-index)
        values = {key: value * (1.0 + 1.0e-2 * h**order) for key, value in exact.items()}
        levels.append({**level, **values})
    return [{"J": 0.7, "levels": levels, "reference_type": "experiment"}]


def test_gci_table_recovers_a_known_order():
    rows = w22.gci_table(_synthetic_points())
    assert len(rows) == 3
    for row in rows:
        assert row["observed_order"] == pytest.approx(2.0, abs=1e-6)
        assert row["asymptotic_range"] is True
        # Roache with safety factor 1.25: GCI = 1.25 |(f2-f1)/f1| / |r^p - 1|
        expected = 100.0 * 1.25 * abs(row["L2"] - row["L3"]) / row["L3"] / (
            row["refinement_ratio"] ** 2 - 1.0
        )
        assert row["gci_fine_pct"] == pytest.approx(expected, rel=1e-9)


def test_gci_table_flags_a_non_asymptotic_ladder():
    points = _synthetic_points()
    # reverse the medium level -> oscillatory differences, hence no order
    levels = points[0]["levels"]
    levels[1]["KT"] = levels[2]["KT"] * 1.1
    rows = [r for r in w22.gci_table(points) if r["quantity"] == "KT"]
    assert rows[0]["asymptotic_range"] is False
    assert rows[0]["oscillatory"] is True


# --------------------------------------------------------------------------- #
# Delivered artefacts (skipped when the driver has not been run in this tree)
# --------------------------------------------------------------------------- #
def _load(name: str):
    path = w22.REPO_ROOT / ARTPATH / name
    if not path.is_file():
        pytest.skip(f"{path} not produced in this checkout")
    return json.loads(path.read_text())


def test_transcription_artifact_matches_source_hash():
    artefact = _load(w22.TRANSCRIPTION_NAME)
    source = w22.REPO_ROOT / w22.ANCHOR_TABLE_SOURCE
    assert artefact["anchor"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    verdict = artefact["verdict"]
    assert verdict["anchor_reachable"] is True
    assert verdict["hit_files"] >= 1
    assert verdict["transcribed_fields"] == artefact["anchor"]["n_fields"]
    assert verdict["sha256_of_hit_files_recorded"] == verdict["hit_files"]


def test_compare_artifact_is_complete():
    artefact = _load(w22.COMPARE_NAME)
    assert artefact["blocked"] is False
    assert [p["J"] for p in artefact["points"]] == list(w22.MISSION_J)
    for point in artefact["points"]:
        assert point["reference_type"] in {"experiment", "numerical", "analytic"}
        for key in ("dKT_pct", "dKQ_pct", "deta0_pct"):
            assert math.isfinite(point[key])
        assert len(point["levels"]) == len(w22.LADDER)
    ids = [row["id"] for row in artefact["criteria"]]
    assert ids == [
        "A1-anchor-reachability-machine-decided",
        "A2-five-J-table-with-deviation-and-reference-type",
        "A3-design-point-KT-within-5pct",
        "A4-panel-density-gci",
    ]
    # A1/A2/A4 are completeness criteria; only A3 is an accuracy criterion
    by_id = {row["id"]: row for row in artefact["criteria"]}
    assert by_id["A1-anchor-reachability-machine-decided"]["pass"] is True
    assert by_id["A2-five-J-table-with-deviation-and-reference-type"]["pass"] is True
    assert by_id["A4-panel-density-gci"]["pass"] is True
    design = artefact["design_point"]
    assert design["J"] == pytest.approx(w22.DESIGN_J)
    assert design["within_5pct"] == (
        abs(design["dKT_pct"]) < w22.THRESHOLD_DESIGN_DKT_PCT
    )
    # the verdict must not claim more than the measured design-point value
    assert artefact["verdict"]["design_point_threshold_met"] == design["within_5pct"]


def test_compare_artifact_gci_rows_cover_the_mission_grid():
    artefact = _load(w22.COMPARE_NAME)
    rows = [r for r in artefact["gci"] if r["quantity"] in {"KT", "KQ"}]
    assert len(rows) == 2 * len(w22.MISSION_J)
    for row in rows:
        assert row["n_levels"] >= 2
        assert "observed_order" in row
        assert "asymptotic_range" in row