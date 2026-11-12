"""Gate: every number in ``docs/m14568_quadrature_kernel_solver.md`` is recomputed.

The document ``m14568_quadrature_kernel_solver.md`` is the paper-level report of
the two-kernel (John eigenseries vs. wavenumber integral) cross-check of the 3-D
finite-depth radiation solver.  Its 1111 number tokens were unguarded -- the
coverage audit ``tensorbem.benchmarks.doc_claim_audit`` listed it as the largest
``UNGUARDED`` paper-level document in the tree -- so the engineering claim it
rests on ("the two kernels agree on A33/B33 to 1e-4") could decay silently.

These tests run the pin module and, crucially, prove that the gate *bites*:

* a hand edit of one printed kernel value must produce a ``DRIFT``;
* a three-decade edit of a round-off error column must produce a ``DRIFT``;
* deleting a whole table must collapse the pinned-row count for that table to
  zero (an empty parse must not be a green run).
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

from tensorbem.benchmarks import m14568_quadrature_kernel_solver_pins as pins

#: The document this gate guards (named literally so the coverage audit
#: ``tensorbem.benchmarks.doc_claim_audit`` sees it as a *reading* test and
#: promotes m14568_quadrature_kernel_solver.md from UNGUARDED to PINNED).
DOC_NAME = "m14568_quadrature_kernel_solver.md"
DOC_PATH = pins.REPO_ROOT / "docs" / DOC_NAME


@pytest.fixture(scope="module")
def data():
    """Library-side numbers, computed once (~20 s solver cross-check)."""
    return pins.recompute_data(include_slow=False)


@pytest.fixture(scope="module")
def report(data):
    summary, rows = pins.check(data=data)
    return summary, rows


def test_document_exists_and_is_the_guarded_one():
    assert DOC_PATH.is_file(), f"missing {DOC_NAME}"
    assert DOC_PATH.name == DOC_NAME


def test_every_pinned_value_reproduces(report):
    summary, rows = report
    drift = [r for r in rows if r.verdict == "DRIFT"]
    assert not drift, [(r.table, r.key, r.quantity, r.declared, r.recomputed) for r in drift]
    # Coverage: all five tables of the report are read, and the pinned count is
    # of the right order (59 kernel + 60 gradient + 12 solver + 4 frobenius +
    # 10 verdict = 145-300 depending on how many cells a row carries).
    assert summary["pinned"] >= 300
    assert set(summary["tables_covered"]) == {"3.1", "3.2", "3.3", "3.3-frobenius", "6"}


def test_solver_level_gate_is_pinned_tightly(report):
    """The engineering claim of the report: two kernels agree on A33/B33."""
    _, rows = report
    a33 = [r for r in rows if r.table == "3.3" and r.quantity == "A33_got"
           and r.verdict != "UNPINNED"]
    b33 = [r for r in rows if r.table == "3.3" and r.quantity == "B33_got"
           and r.verdict != "UNPINNED"]
    assert a33 and b33, "solver-level rows must be pinned by default"
    for row in a33 + b33:
        assert row.rtol == pins.VALUE_RTOL
        assert row.verdict == "OK"
        assert row.rel_deviation < pins.VALUE_RTOL
    # the doc's headline agreement numbers
    err_a = [r for r in rows if r.table == "3.3" and r.quantity == "err_A33"][0]
    err_b = [r for r in rows if r.table == "3.3" and r.quantity == "err_B33"][0]
    assert err_a.recomputed < 0.03 and err_b.recomputed < 0.03


def test_under_three_percent_claims_are_recomputed(report):
    _, rows = report
    claims = [r for r in rows if r.quantity == "under_3_percent_claim"]
    assert len(claims) == 5, "the §6 verdict table has five rows"
    assert all(r.verdict == "OK" for r in claims)


def test_omissions_are_declared_not_silent(report):
    summary, rows = report
    unpinned = [r for r in rows if r.verdict == "UNPINNED"]
    assert unpinned, "timings / slow rows exist and must be reported as UNPINNED"
    assert summary["unpinned"] == len(unpinned)
    assert any("wall clock" in r.note for r in unpinned)
    assert all(r.note for r in unpinned)


def _mutated_copy(tmp_path: pathlib.Path, old: str, new: str) -> pathlib.Path:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert old in text, f"fixture string {old!r} no longer in the document"
    target = tmp_path / DOC_NAME
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return target


def test_hand_edit_of_a_kernel_value_trips_the_gate(tmp_path, data):
    """Change one printed digit: the value row must go DRIFT."""
    mutated = _mutated_copy(tmp_path, "9.549336e-02", "9.549336e-01")
    _, rows = pins.check(mutated, data=data)
    drift = [r for r in rows if r.verdict == "DRIFT"]
    assert drift, "a 10x edit of a printed kernel value must be a DRIFT"
    assert any(r.quantity == "G_ref_re" for r in drift)


def test_hand_edit_of_a_roundoff_column_trips_the_gate(tmp_path, data):
    """The decade-band pin still bites across decades (round-off columns)."""
    mutated = _mutated_copy(tmp_path, "2.233e-11", "2.233e-08")
    _, rows = pins.check(mutated, data=data)
    drift = [r for r in rows if r.verdict == "DRIFT"]
    assert drift, "three decades of drift must not pass a one-decade band"
    assert any(r.quantity == "rel_err" for r in drift)


def test_deleting_the_kernel_table_cannot_pass(tmp_path, data):
    """An empty parse must not be a green run (no silent '0 rows = 0 drift')."""
    lines = DOC_PATH.read_text(encoding="utf-8").splitlines()
    header = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("| target | source | h [m] |")
    )
    # drop the header + separator + the contiguous body of the §3.1 table
    end = header
    while end < len(lines) and lines[end].strip().startswith("|"):
        end += 1
    trimmed = "\n".join(lines[:header] + lines[end:]) + "\n"
    target = tmp_path / DOC_NAME
    target.write_text(trimmed, encoding="utf-8")

    summary, rows = pins.check(target, data=data)
    assert not [r for r in rows if r.table == "3.1"]
    assert summary["pinned"] < 500, "losing a table must shrink the pinned set"


def test_gate_is_rerunnable_and_stable(data, tmp_path):
    """Same document, same numbers: the check is deterministic."""
    first, _ = pins.check(data=data)
    second, _ = pins.check(data=data)
    assert first["drift"] == second["drift"] == 0
    assert first["pinned"] == second["pinned"]