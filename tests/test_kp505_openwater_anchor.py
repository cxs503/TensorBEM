"""Contract tests for benchmarks/anchors/kp505_openwater.json (paper-exp-propeller-openwater-bem-1).

The artefact is machine-generated; these tests make sure the acceptance criteria of the
mission are actually encoded in it and that the committed artefact still matches its
sources (PDF sha256, transcript sha256) and its own arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART_PATH = REPO / "benchmarks" / "anchors" / "kp505_openwater.json"
SRC = REPO / "benchmarks" / "anchors" / "sources"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def art():
    assert ART_PATH.exists(), "missing artefact: %s" % ART_PATH
    return json.loads(ART_PATH.read_text())


def test_criteria_are_encoded_and_pass(art):
    crit = {c["id"]: c for c in art["criteria"]}
    assert set(crit) == {
        "C1-quote-and-table-number",
        "C2-points-and-design-J",
        "C3-eta0-self-consistency",
    }
    for c in crit.values():
        assert c["pass"] is True, (c["id"], c["measured"])


def test_every_point_carries_a_quote_and_table_number(art):
    for row in art["open_water_table"]:
        assert row["source_line"].strip(), "row without a verbatim source line"
    assert art["source"]["table"].startswith("Table 2")
    assert "Table 2." in art["source"]["quote"]
    assert art["source"]["page"] >= 1


def test_point_count_and_design_advance_ratio(art):
    js = [row["J"] for row in art["open_water_table"]]
    assert len(js) >= 8
    assert 0.70 in js, "design advance ratio J = 0.70 missing"
    assert js == sorted(js)


def test_eta0_is_self_consistent_with_kt_and_10kq(art):
    for row in art["open_water_table"]:
        kq = row["10KQ"] / 10.0
        recomputed = row["J"] * row["KT"] / (2.0 * math.pi * kq)
        assert abs(row["eta0"] - recomputed) < 1e-6, row
        assert abs(row["eta0_recomputed"] - recomputed) < 1e-12
        assert row["eta0_residual"] < 1e-6


def test_peak_efficiency_j_is_recorded_and_is_the_table_maximum(art):
    peak = art["peak_efficiency"]
    table_max = max(art["open_water_table"], key=lambda r: r["eta0"])
    assert peak["J"] == table_max["J"]
    assert abs(peak["eta0"] - table_max["eta0"]) < 1e-12
    assert 0.0 < peak["eta0"] < 1.0


def test_design_point_matches_the_table(art):
    dp = art["design_point"]
    row = [r for r in art["open_water_table"] if abs(r["J"] - 0.70) < 1e-12][0]
    for key in ("J", "KT", "10KQ", "eta0"):
        assert dp[key] == row[key]


def test_source_sha_and_transcript_reproduce_the_pdf_bytes(art):
    pdf = SRC / art["source"]["pdf"]
    transcript = SRC / art["source"]["transcript"]
    assert pdf.exists() and transcript.exists()
    assert sha256(pdf) == art["source"]["pdf_sha256"]
    assert sha256(transcript) == art["source"]["transcript_sha256"]
    text = transcript.read_text()
    assert "Table 2. Comparison of CFD and Experimental Results" in text
    # every transcribed data row must still be present verbatim in the transcript
    for row in art["open_water_table"]:
        assert row["source_line"] in text


def test_geometry_fields_state_their_evidence(art):
    fields = art["geometry"]["fields"]
    for name in ("diameter_m", "blade_count_Z", "P_over_D", "AE_over_A0"):
        assert name in fields
        assert fields[name]["status"]
        assert fields[name]["evidence"]
    # values that are present must carry a quotation from a cited page
    for name in ("diameter_m", "blade_count_Z"):
        spec = fields[name]
        assert spec["value"] is not None
        assert re.search(r"page \d+", spec["evidence"])
    # withheld values are declared, never silently filled
    for name in ("P_over_D", "AE_over_A0"):
        assert fields[name]["value"] is None
        assert fields[name]["status"] == "WITHHELD"
    assert art["geometry"]["spec"]["diameter_m"] == pytest.approx(0.25)
    assert art["geometry"]["spec"]["blade_count_Z"] == 5
    assert art["verdict"]["geometry_complete"] is False  # honest: two fields withheld


def test_honest_notes_declare_the_secondary_provenance(art):
    notes = " ".join(art["verdict"]["honest_notes"])
    assert "SECONDARY" in notes.upper()
    assert art["source"]["kind"] == "SECONDARY_REPRODUCTION"


def test_artefact_is_regenerable(tmp_path):
    """Running the builder reproduces the committed JSON byte-for-byte."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kp505_builder", REPO / "tools" / "build_kp505_openwater_anchor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rebuilt = mod.build()
    committed = json.loads(ART_PATH.read_text())
    assert json.dumps(rebuilt, sort_keys=True) == json.dumps(committed, sort_keys=True)