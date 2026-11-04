"""Guards the numbers printed in ``docs/finite_depth3d.md``.

``docs/finite_depth3d.md`` quotes 64 values across five tables (dispersion
ladder, kernel boundary conditions, deep-water limit, far field, singularity and
truncation).  Until this test existed nothing read them: a coverage audit
(:mod:`tensorbem.benchmarks.doc_claim_audit`) classified the document as
``UNGUARDED``, i.e. exactly the state in which the Ursell ``7.7e-3`` and the
Wigley displacement numbers rotted unnoticed.

The gates below are, in order:

1. **live reproduction** -- every parsed value is recomputed from
   :mod:`tensorbem.finite_depth3d` / :mod:`tensorbem.free_surface3d` and must
   agree within the tolerance implied by its printed precision;
2. **non-vacuity** -- the parser must actually have found the five tables and at
   least 60 values, so a formatting change cannot silently empty the gate;
3. **negative control** -- a mutated copy of the document (one digit changed)
   must produce a ``DRIFT``, i.e. the gate really reads the document and does
   not just print PASS.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tensorbem.benchmarks import finite_depth3d_pins as pins

DOC = Path(pins.DOC_PATH)


@pytest.fixture(scope="module")
def rows():
    return pins.collect(DOC)


def test_document_exists():
    assert DOC.is_file(), "docs/finite_depth3d.md is part of the verification record"


def test_non_vacuous_parse(rows):
    """The parser must see all five tables; an empty gate must not look green."""
    tables = {r.table for r in rows}
    assert tables == {"3.1", "3.2", "3.3", "3.4", "3.5"}, tables
    assert len(rows) >= 60, f"only {len(rows)} values parsed"
    per_table = {}
    for row in rows:
        per_table[row.table] = per_table.get(row.table, 0) + 1
    for table, count in per_table.items():
        assert count >= 5, f"table {table} contributed only {count} values"


def test_every_number_reproduces(rows):
    """Live recomputation of every quoted number; drift is a red build."""
    drift = [r for r in rows if r.verdict == "DRIFT"]
    detail = "\n".join(
        f"{r.table} {r.key} {r.quantity}: declared={r.declared:.6e} "
        f"recomputed={r.recomputed:.6e} rel={r.rel_deviation:.2e} rtol={r.rtol:.1e}"
        for r in drift
    )
    assert not drift, f"{len(drift)} documented numbers no longer reproduce:\n{detail}"


def test_worst_deviation_is_small(rows):
    """The largest deviation over the whole document, reported as evidence."""
    worst = max(rows, key=lambda r: r.rel_deviation)
    # 1e-2 is the '3 significant digits printed' floor; the measured worst case
    # is the truncation ladder, whose entries are differences of order 1e-11.
    assert worst.rel_deviation < 1.0e-2, (
        f"worst pinned deviation {worst.rel_deviation:.2e} in "
        f"{worst.table}/{worst.key}/{worst.quantity}"
    )


def test_mutated_document_is_caught(tmp_path, rows):
    """Negative control: change one printed digit, the gate must go red."""
    text = DOC.read_text(encoding="utf-8")
    target = next(r for r in rows if r.table == "3.3" and r.quantity == "max_rel_error")
    printed = f"{target.declared:.3e}"
    assert printed in text, f"{printed} not found in the document"
    mutated = tmp_path / "finite_depth3d.md"
    mutated.write_text(text.replace(printed, f"{target.declared * 1.5:.3e}", 1), encoding="utf-8")
    mutated_rows = pins.collect(mutated)
    drift = [r for r in mutated_rows if r.verdict == "DRIFT"]
    assert len(drift) == 1, [f"{r.table}/{r.key}/{r.quantity}" for r in drift]
    assert drift[0].quantity == "max_rel_error"


def test_missing_table_is_not_silently_green(tmp_path):
    """A document whose tables were dropped yields few rows -> the parser is honest."""
    trimmed = tmp_path / "finite_depth3d.md"
    text = DOC.read_text(encoding="utf-8")
    head = text.split("## 3. Results")[0]
    trimmed.write_text(head + "\n## 6. Limitations\n", encoding="utf-8")
    trimmed_rows = pins.collect(trimmed)
    assert len(trimmed_rows) == 0


def test_cli_artifact_roundtrip(tmp_path):
    """--strict exits 0 on a clean document and writes a JSON artifact."""
    out = tmp_path / "check.json"
    code = pins.main(["--json", str(out)])
    assert code == 0
    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["drift"] == 0
    assert payload["summary"]["rows"] == len(payload["rows"])
    assert payload["summary"]["schema"] == pins.SCHEMA