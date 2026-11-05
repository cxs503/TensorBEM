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


def test_roundoff_rows_are_identified(rows):
    """The free-surface residuals are the only sub-floor rows, and they pass."""
    sub = [r for r in rows if r.is_roundoff_zero]
    assert len(sub) == 3, [(r.table, r.key, r.declared) for r in sub]
    assert all(r.table == "3.2" and r.quantity == "free_surface_residual" for r in sub)
    assert all(r.verdict == "PASS" for r in sub)


def test_roundoff_floor_absorbs_ulp_noise_only():
    """Negative controls: the 1e-12 floor must not turn a real residual green.

    The m3409 rewrite of the propagating mode (``cosh`` -> log-scaled
    exponentials) is mathematically neutral but moves the free-surface
    residual by 1-3 ulp: 1.26e-15 -> 1.48e-15.  That is machine zero, so the
    row passes; a residual that actually broke the boundary condition
    (O(1e-10) and worse) is still a DRIFT, and the printed exact zeros keep the
    much tighter 5e-16 floor.
    """
    res = "free_surface_residual"
    # ulp-level move of a roundoff residual -- machine zero either way
    assert pins._judge(1.26e-15, 1.476491e-15, 5e-3, res)[0] == "PASS"
    assert pins._judge(7.847e-15, 8.097902e-15, 5e-3, res)[0] == "PASS"
    # a genuine free-surface residual regression is far above the floor
    assert pins._judge(1.26e-15, 4.0e-10, 5e-3, res)[0] == "DRIFT"
    assert pins._judge(1.26e-15, 1.0e-8, 5e-3, res)[0] == "DRIFT"
    # printed exact zeros stay as strict as before (ZERO_FLOOR, not NOISE_FLOOR)
    assert pins._judge(0.0, 0.0, 5e-3, "bottom_residual")[0] == "PASS"
    assert pins._judge(0.0, 1.0e-15, 5e-3, "bottom_residual")[0] == "DRIFT"
    # a small *physical* value keeps its relative tolerance: the floor is not a
    # blanket exemption for tiny numbers
    assert pins._judge(4.6e-20, 4.6e-20, 5e-2, "evanescent_share")[0] == "PASS"
    assert pins._judge(4.6e-20, 9.2e-20, 5e-2, "evanescent_share")[0] == "DRIFT"
    # ordinary rows are still judged relative to the printed precision
    assert pins._judge(3.481518, 3.4815177, 1e-6, "k0")[0] == "PASS"
    assert pins._judge(3.481518, 3.49, 1e-6, "k0")[0] == "DRIFT"


def test_worst_deviation_ignores_roundoff_rows(rows):
    """The reported worst case is a physical row, not 1e-15 cancellation noise."""
    worst = max((r for r in rows if abs(r.declared) > pins.NOISE_FLOOR),
                key=lambda r: r.rel_deviation)
    assert not worst.is_roundoff_zero
    assert worst.rel_deviation < 1.0e-2
