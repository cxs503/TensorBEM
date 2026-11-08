"""Executable pins for ``docs/ursell_convergence_study.md`` (mission m6436).

The m2198 convergence study was the last paper-level Ursell contract with no
executable guard: :mod:`tensorbem.benchmarks.doc_claim_audit` graded it
``UNGUARDED`` (93 number tokens, no declared generator, no reading test).  Its
digits had drifted -- a wrong angular frequency, three transcription errors
inside single rows, a near/far column that used a second definition of the same
ratio and a README comparison against a claim the README had already dropped.

These tests close that debt in the same way the other Ursell documents are
closed: :mod:`tensorbem.benchmarks.ursell_convergence_doc` recomputes every
table from the shipped solver and re-derives every printed number from the
document text.  A solver change that moves a value, or a prose edit that
changes a digit, turns this file red.

Cost: the study solves 16/32/64/128 panels once (~29 s, the 128-panel row being
~22 s of it) and is shared by every test through a module-scope fixture.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tensorbem.benchmarks import ursell_convergence_doc as doc_pins
from tensorbem.benchmarks.ursell_convergence_table import reference_coefficients

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / doc_pins.DOC_NAME
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def study():
    return doc_pins.study()


@pytest.fixture(scope="module")
def failures(study):
    return doc_pins.check_document(DOC, study)


def test_document_exists_and_is_the_guarded_contract():
    assert DOC.is_file(), f"{doc_pins.DOC_NAME} must exist for the pins to bind"
    text = DOC.read_text(encoding="utf-8")
    for name, tag in doc_pins.MARKERS.items():
        assert f"<!-- {tag}:begin -->" in text, f"block '{name}' lost its begin marker"
        assert f"<!-- {tag}:end -->" in text, f"block '{name}' lost its end marker"


def test_every_table_reproduces_from_the_shipped_solver(failures):
    assert not failures, "document no longer reproduces:\n  " + "\n  ".join(failures)


def test_regeneration_is_byte_identical(study):
    """The committed text must equal a fresh render, digit for digit."""
    text = DOC.read_text(encoding="utf-8")
    regenerated = doc_pins.apply(text, study)
    assert regenerated == text, "generated blocks drifted from the solver output"


def test_omega_is_consistent_with_the_parameters_it_is_printed_next_to():
    """The defect this mission found: 1.905 rad/s beside ka=0.6, a=1, g=9.80665."""
    text = DOC.read_text(encoding="utf-8")
    match = re.search(r"\|\s*角频率\s*\|[^|]*\|\s*([0-9.]+)\s*\|\s*rad/s", text)
    assert match, "the parameter table lost its angular-frequency row"
    printed = float(match.group(1))
    expected = math.sqrt(0.6 * 9.80665 / 1.0)
    assert printed == pytest.approx(expected, rel=1e-6), (
        f"omega={printed} contradicts sqrt(ka*g/a)={expected:.6f}"
    )
    assert doc_pins.SUPERSEDED_OMEGA != pytest.approx(printed, rel=1e-6), (
        "the superseded 1.905 rad/s value must not come back"
    )


def test_rows_are_internally_consistent(study):
    """A value and the relative error printed beside it must imply each other.

    This is the check that would have caught the 2697.5 / 2851.7 / 2587.6
    transcription errors without recomputing anything.
    """
    reference = study["reference"]
    for row in study["rows"]:
        implied = reference["added_mass"] * (1 + row["added_mass_relative_error"])
        assert implied == pytest.approx(row["added_mass"], rel=2e-4)
        implied = reference["damping"] * (1 + row["damping_relative_error"])
        assert implied == pytest.approx(row["damping"], rel=2e-4)


def test_reference_matches_the_analytic_multipole_solution(study):
    """The reference column is Ursell's closed form, not a panel result."""
    live = reference_coefficients()
    assert live["added_mass"] == pytest.approx(study["reference"]["added_mass"], rel=1e-12)
    assert live["damping"] == pytest.approx(study["reference"]["damping"], rel=1e-12)
    # published in the document's §1.3 and in docs/ursell_verification_debt_2026-11-02.md
    assert study["reference"]["added_mass"] == pytest.approx(2545.1672, rel=1e-6)
    assert study["reference"]["damping"] == pytest.approx(2718.1540, rel=1e-6)


def test_panel_errors_clear_the_three_percent_criterion(study):
    """The verdict §6 certifies: every key quantity under 3 % at 128 panels."""
    row = next(r for r in study["rows"] if r["panels"] == doc_pins.VERDICT_PANELS)
    assert row["added_mass_relative_error"] < 0.03
    assert row["damping_relative_error"] < 0.03
    assert row["near_far_relative_difference"] < 0.03
    assert study["mfs"]["added_mass_relative_error"] < 0.03
    assert study["mfs"]["damping_relative_error"] < 0.03


def test_convergence_is_first_order_and_monotone(study):
    """Observed orders approach 1 (constant panels, logarithmic kernel)."""
    errors = [r["added_mass_relative_error"] for r in study["rows"]]
    assert errors == sorted(errors, reverse=True), "added-mass error is not decreasing"
    finest = study["orders"][-1]
    assert 0.9 < finest["added_mass_order"] < 1.1
    assert 0.85 < finest["damping_order"] < 1.1


def test_readme_claims_quoted_by_the_document_still_exist(study):
    """§3.3 compares against README text; a README edit must not orphan it."""
    text = README.read_text(encoding="utf-8")
    for quantity, claim in doc_pins.README_CLAIMS.items():
        assert claim in text, f"README no longer states {quantity} as {claim}"
        assert doc_pins.README_DECLARED[quantity] > 0


def test_superseded_digits_are_recorded_in_the_audit_trail(study):
    """Nothing is silently overwritten: the old digits stay in §3.4."""
    text = DOC.read_text(encoding="utf-8")
    trail = text.split("### 3.4", 1)
    assert len(trail) == 2, "the document lost its audit-trail section"
    trail = trail[1].split("### 3.5", 1)[0]
    for panels, old in doc_pins.SUPERSEDED.items():
        assert f"{panels}" in trail
        assert f"{old['damping']}" in trail or f"{old['added_mass']}" in trail
    assert f"{doc_pins.SUPERSEDED_OMEGA}" in trail