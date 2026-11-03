"""Guards for the 3-D documentation numbers.

``docs/three_dimensional_bem.md`` quotes four result tables.  Until now nothing
in CI checked that the numbers in them are still produced by the code.  These
tests do three things:

1. structural: the tables are still parseable and still have the expected rows;
2. positive: every number in the coarse tables (levels 0-1) is reproduced within
   the pin tolerance, recomputed from the library, not from a hand copy;
3. negative control: a single edited digit in the document must be reported as
   drift, so a green suite cannot mean "the checker is asleep".

The heavy rows (320 and 1280 panels) are covered by
``python -m tensorbem.benchmarks.doc_pins_3d --check`` rather than by CI.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from tensorbem.benchmarks.doc_pins_3d import (
    COLUMNS,
    DEFAULT_RECORD,
    DOC_PATH,
    RELATIVE_TOLERANCE,
    TABLES,
    check,
    decode_number,
    parse_doc_tables,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_decode_number_handles_document_formats() -> None:
    assert decode_number(r"\(1.4600\times10^{-1}\)") == pytest.approx(0.146)
    assert decode_number("1.0171e-1") == pytest.approx(0.10171)
    assert decode_number(r"\(9.13\)") == pytest.approx(9.13)
    assert decode_number(r"\(1.1\times10^{-16}\)") == pytest.approx(1.1e-16)
    assert decode_number("—") is None
    assert decode_number("") is None


def test_tables_parse_with_expected_rows() -> None:
    doc_text = (REPO_ROOT / DOC_PATH).read_text(encoding="utf-8")
    tables = parse_doc_tables(doc_text)
    assert set(tables) == {spec.table_id for spec in TABLES}
    expected_rows = {
        "interior_laplace": [20, 80, 320, 1280],
        "exterior_added_mass": [20, 80, 320, 1280],
        "exterior_constant_dirichlet": [20, 80, 320, 1280],
        "exterior_velocity": [80, 320, 1280],
    }
    for table_id, panels in expected_rows.items():
        assert sorted(tables[table_id]) == panels, table_id
        for row in tables[table_id].values():
            assert len(row) == len(COLUMNS[table_id])


def test_coarse_rows_match_the_recomputed_benchmarks() -> None:
    results, notes = check(REPO_ROOT / DOC_PATH, max_level=1)
    summary = summarise(results)
    assert notes == []
    assert summary["pins"] >= 25
    assert summary["failed"] == 0, summary["failures"]


def test_single_edited_digit_is_reported_as_drift() -> None:
    doc_text = (REPO_ROOT / DOC_PATH).read_text(encoding="utf-8")
    # Corrupt the 20-panel interior flux error (documented 1.4600e-1).
    corrupted = doc_text.replace(
        r"\(1.4600\times10^{-1}\)", r"\(1.9600\times10^{-1}\)", 1
    )
    assert corrupted != doc_text
    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "three_dimensional_bem.md"
        doc.write_text(corrupted, encoding="utf-8")
        results, _ = check(doc, max_level=0, tolerance=RELATIVE_TOLERANCE)
    failures = [item for item in results if not item.passed]
    assert failures, "a corrupted documented number must be reported"
    assert any(item.table_id == "interior_laplace" and item.panels == 20
               for item in failures)


def test_committed_record_matches_the_document() -> None:
    record_path = REPO_ROOT / DEFAULT_RECORD
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["schema"] == "tensorbem.doc_pins_3d.v1"
    assert record["relative_tolerance"] == RELATIVE_TOLERANCE
    doc_text = (REPO_ROOT / DOC_PATH).read_text(encoding="utf-8")
    tables = parse_doc_tables(doc_text)
    index = {}
    for table_id, rows in tables.items():
        symbols = [c.symbol if c is not None else None for c in COLUMNS[table_id]]
        for panels, values in rows.items():
            for symbol, value in zip(symbols, values):
                if symbol is None or value is None:
                    continue
                index[(table_id, panels, symbol)] = value
    assert record["pins"], "the record must contain pins"
    for pin in record["pins"]:
        key = (pin["table"], pin["panels"], pin["column"])
        assert key in index, key
        assert index[key] == pytest.approx(pin["declared"], rel=1e-12)
    # the record must not contain a stale failure either
    assert record["summary"]["failed"] == 0
    assert math.isfinite(record["summary"]["worst"]["drift"])