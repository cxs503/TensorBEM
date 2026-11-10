"""Contract tests for the W20 KP505 anchor transcription artefact.

The artefact is only worth anything if every number in it can be re-read from a
source that is committed next to it.  These tests do three things:

1. re-run :func:`tools.build_w20_kp505_transcription.verify`, which re-reads the
   transcript file, re-parses the row each number claims to come from and
   re-derives every efficiency value;
2. rebuild the artefact and compare the payload field by field, so a hand edit
   of the JSON shows up as a diff;
3. pin the *negative* cross-repository finding, so the fact that the circulating
   TensorLBM constants are not the source-table values cannot silently
   disappear from the artefact.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = REPO_ROOT / "papers" / "data" / "w20-anchors" / "kp505_transcription.json"
EXCERPT = (
    REPO_ROOT
    / "papers"
    / "data"
    / "w20-anchors"
    / "sources"
    / "lbm_bench_propeller_kp505_excerpt.txt"
)
BUILDER = REPO_ROOT / "tools" / "build_w20_kp505_transcription.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("w20_kp505_transcription", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def artefact() -> dict:
    assert ARTEFACT.is_file(), f"missing artefact {ARTEFACT}"
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


def test_every_number_carries_full_provenance(artefact: dict) -> None:
    numbers = artefact["numbers"]
    assert len(numbers) >= 60, "table should carry 7 numbers per row over >= 9 rows"
    transcript_sha = artefact["transcript"]["sha256"]
    for record in numbers:
        source = record["source"]
        assert source["file"] == artefact["transcript"]["file"]
        assert source["sha256"] == transcript_sha
        assert isinstance(source["line"], int) and source["line"] >= 1
        assert source["field"] == record["field"]
        assert source["verbatim"].strip(), "verbatim source line must be stored"
        assert source["table"].startswith("Table 2")


def test_machine_reread_reproduces_every_number(artefact: dict, builder) -> None:
    report = builder.verify(artefact)
    assert report["problems"] == []
    assert report["ok"] is True
    assert report["n_numbers_checked"] >= len(artefact["numbers"])


def test_rebuild_matches_committed_artefact(artefact: dict, builder, tmp_path) -> None:
    out = tmp_path / "kp505_transcription.json"
    rebuilt = builder.build(lbm_repo=builder.LBM_DEFAULT, out_path=out)
    for key in (
        "numbers",
        "anchor_points",
        "source_consistency_checks",
        "anchor_table_agreement",
        "criteria",
        "transcript",
    ):
        assert rebuilt[key] == artefact[key], (
            f"field {key!r} drifted from the committed artefact"
        )
    assert (
        rebuilt["cross_repository_backcheck"] == artefact["cross_repository_backcheck"]
    )


def test_efficiency_is_derived_not_typed(artefact: dict) -> None:
    for point in artefact["anchor_points"]:
        expected = 5.0 * point["J"] * point["KT"] / (math.pi * point["10KQ"])
        assert point["eta0"] == pytest.approx(expected, abs=1e-12)
        assert point["eta0_formula"] == "eta0 = 5 * J * KT / (pi * 10KQ)"


def test_source_error_columns_reproduce_within_roundoff(artefact: dict) -> None:
    consistency = artefact["source_consistency_checks"]
    assert consistency["all_pass"] is True
    assert consistency["max_diff_beyond_roundoff_pct_points"] <= 0.0
    # the largest raw disagreement is reported, not hidden
    assert consistency["max_abs_diff_pct_points"] > 1.0
    assert consistency["worst_raw_check"]["J"] == 1.0
    assert consistency["worst_raw_check"]["within_roundoff"] is True


def test_design_point_present(artefact: dict) -> None:
    js = [point["J"] for point in artefact["anchor_points"]]
    assert 0.7 in js
    assert len(js) == 10
    design = next(point for point in artefact["anchor_points"] if point["J"] == 0.7)
    assert design["KT"] == pytest.approx(0.193, abs=1e-12)
    assert design["10KQ"] == pytest.approx(0.342, abs=1e-12)


def test_circulating_constants_are_not_source_values(artefact: dict) -> None:
    """Pin the negative finding: the TensorLBM constants are not the table's."""

    block = artefact["cross_repository_backcheck"]
    assert block["available"] is True, (
        "the back-check must have run for this finding to hold"
    )
    assert block["summary"]["identical_to_source_table"] is False
    assert block["summary"]["max_abs_dKT_pct"] > 10.0
    by_J = {row["J_circulating"]: row for row in block["comparison"]}
    # every advance ratio of the circulating table sits outside the anchor by
    # more than the 3 % design criterion, in both unit readings of column 2
    for row in block["comparison"]:
        if not row["J_exact_match"]:
            continue
        assert abs(row["dKT_pct"]) > 3.0
        assert min(abs(row["d10KQ_pct_as_is"]), abs(row["d10KQ_pct_times_ten"])) > 3.0
    assert by_J[0.9]["dKT_pct"] > 90.0, (
        "the J = 0.9 row is the worst offender and must stay pinned"
    )


def test_circulating_table_uses_an_offgrid_advance_ratio(artefact: dict) -> None:
    block = artefact["cross_repository_backcheck"]
    offgrid = [
        row["J_circulating"] for row in block["comparison"] if not row["J_exact_match"]
    ]
    assert offgrid == [1.1], (
        "the circulating table quotes J = 1.1, outside the source table"
    )


def test_in_repo_excerpt_matches_transcribed_entries(artefact: dict) -> None:
    assert EXCERPT.is_file()
    lines = EXCERPT.read_text(encoding="utf-8").splitlines()
    for entry in artefact["cross_repository_backcheck"]["entries"]:
        assert entry["verbatim"] in lines


def test_criteria_and_verdict_are_consistent(artefact: dict) -> None:
    passes = {c["id"]: c["pass"] for c in artefact["criteria"]}
    assert passes["T1-every-number-has-source"] is True
    assert passes["T3-source-self-consistency"] is True
    assert passes["T5-anchor-table-agreement"] is True
    assert passes["T6-circulating-constants-are-source-values"] is False
    verdict = artefact["verdict"]
    assert verdict["transcription_quality_all_pass"] is True
    assert verdict["circulating_constants_are_source_values"] is False
    assert verdict["all_criteria_pass"] is False, (
        "the negative cross-repository finding must keep the overall verdict honest"
    )
