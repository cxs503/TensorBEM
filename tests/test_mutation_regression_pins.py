"""Executable pins for the mutation-audit survivors.

``docs/mutation_audit_tensorbem.md`` §5 left four coverage gaps and
``docs/mutation_gaps_closure.md`` backfilled them.  These pins keep that work
from rotting: each pinned mutant is re-applied to a shadow copy of ``src`` and
the one test that is supposed to notice it must fail.  A pin that can no longer
be located fails the first test, so a refactor of the numerical code forces a
re-audit instead of silently reopening the hole.
"""

from __future__ import annotations

import pytest

from tensorbem.benchmarks.mutation_regression import PINS, locate, verify


def test_every_pin_still_locates_exactly_one_mutant() -> None:
    for pin in PINS:
        mutant = locate(pin)
        assert mutant.function == pin.function
        assert mutant.original == pin.original
        assert mutant.replacement != mutant.original


def test_pinned_mutants_are_killed_by_their_named_test() -> None:
    payload = verify()
    unforgiven = [
        row for row in payload["rows"] if row["verdict"] != "killed"
    ]
    assert not unforgiven, unforgiven
    assert payload["killed"] == payload["pins"] == len(PINS)
    for row in payload["rows"]:
        # The kill must come from the intended test, not from an import error.
        assert row["failing_test"] == row["pin"]["test"], row
