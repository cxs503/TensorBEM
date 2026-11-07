"""Gates for the capability coverage ratchet and the drift-force entry points.

``tensorbem.capability_evidence`` grades every capability it knows about, but
nothing in the suite noticed until m6280 that half of the core package was in no
row at all: the table was not wrong about those modules, it was silent, and
silence reads as out of scope.  ``capability_coverage`` turns that silence into
a failing gate; the tests here are its negative control (a registry with the
m6280 rows removed must actually fail it) plus the live anchor for
``maruo_drift``, which was the one core physics module with no static test
reference at all -- that module is recorded in ``UNTESTED_MODULES`` with the debt
stated, rather than covered by a made-up assertion.
"""

from __future__ import annotations

import pytest

from tensorbem import capability_evidence as ce


def test_every_core_module_is_graded_or_explained() -> None:
    coverage = ce.capability_coverage()
    assert coverage["unowned"] == [], coverage["unowned"]
    assert coverage["stale_reasons"] == [], coverage["stale_reasons"]
    assert len(coverage["graded"]) + len(coverage["auxiliary"]) == len(coverage["core_modules"])


def test_coverage_gate_discriminates_against_the_pre_m6280_registry() -> None:
    """Negative control: without the m6280 rows the gate must go red."""

    from tensorbem.benchmarks.capability_coverage_study import M6280_KEYS

    before_rows = tuple(cap for cap in ce.CAPABILITIES if cap.key not in M6280_KEYS)
    before = ce.capability_coverage(capabilities=before_rows)
    after = ce.capability_coverage()
    assert len(before["unowned"]) >= 20, before["unowned"]
    assert before["unowned"] and not after["unowned"]
    assert set(after["graded"]) - set(before["graded"]) == set(before["unowned"])


def test_every_m6280_row_is_real_with_several_named_nodes() -> None:
    from tensorbem.benchmarks.capability_coverage_study import M6280_KEYS

    rows = {row.capability.key: row for row in ce.audit()}
    for key in M6280_KEYS:
        row = rows[key]
        assert row.level == ce.REAL, (key, row.level, row.reasons)
        assert row.test_nodes >= 3, (key, row.test_nodes)


def test_study_driver_reports_zero_unowned_after_and_some_before() -> None:
    from tensorbem.benchmarks import capability_coverage_study as study

    coverage = study.coverage_before_after()
    assert coverage["unowned_after"] == 0
    assert coverage["unowned_before"] == coverage["graded_after"] - coverage["graded_before"]
    assert coverage["rows_after"] - coverage["rows_before"] == len(study.M6280_KEYS)
