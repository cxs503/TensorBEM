"""The numbers in ``docs/lifting_foil.md`` must still be the numbers the code produces.

``benchmarks/lifting_doc_pins`` parses the document and recomputes every claim;
these tests run that gate and, just as important, prove that the gate is not
vacuous: a mutated document cell, a mutated measurement and a re-worded order
claim all have to be reported as ``DRIFT``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from tensorbem.benchmarks import lifting_doc_pins as pins


@pytest.fixture(scope="module")
def document() -> str:
    return pins.DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declared(document: str) -> dict[str, pins.Claim]:
    return pins.parse_document(document)


@pytest.fixture(scope="module")
def measured(document: str, declared: dict[str, pins.Claim]) -> dict[str, pins.Claim]:
    return pins.measure(declared)


# ---------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------
def test_every_documented_number_reproduces(declared, measured):
    """The gate: all parsed claims PASS against a fresh computation."""
    verdicts = pins.compare(declared, measured)
    drifted = [v.as_dict() for v in verdicts if v.status != "PASS"]
    assert not drifted, f"documented numbers drifted: {drifted}"
    assert len(verdicts) >= 40  # two 4x4 tables plus the prose claims


def test_claim_inventory_is_exactly_the_tables_and_the_prose(declared):
    """A cell deleted from the document must shrink the inventory, not vanish."""
    expected = {
        f"analytic.{alpha}.{column}"
        for alpha in pins.ANGLES
        for column in pins.ANALYTIC_COLUMNS
    } | {
        f"refinement.{panels}.{column}"
        for panels in pins.PANEL_COUNTS
        for column in pins.REFINEMENT_COLUMNS
    } | {
        "prose.thin_aerofoil_cl",
        "prose.thin_aerofoil_section_cl",
        "prose.thin_aerofoil_gap_percent",
        "prose.closure_gap_percent",
        "prose.lift_below_exact_percent",
        "prose.singular_from_fraction",
        "prose.singular_to_fraction",
        "prose.singular_variation_percent",
        "prose.layer_route_error_percent",
        "prose.layer_route_panels",
        "prose.three_point_error_percent",
        "prose.decay_offset",
        "prose.boundary_offset",
        "prose.boundary_spread",
        "prose.boundary_spread_panels",
        "prose.rms_order_word",
        "prose.rms_order_32_64",
        "prose.rms_order_64_128",
        "prose.rms_order_128_256",
        "prose.rms_order_mean",
    }
    assert set(declared) == expected


def test_deleting_a_table_row_removes_a_claim(document, declared):
    """Dropping a ladder row must remove exactly that row's claims."""
    victim = "| 256 | 1.68% | 7.36e-4 | 8.2% | 205 |"
    assert victim in document
    reduced = pins.parse_document(document.replace(victim, ""))
    lost = set(declared) - set(reduced)
    assert lost == {
        f"refinement.256.{column}" for column in pins.REFINEMENT_COLUMNS
    }


def test_settings_are_reported_from_the_document(declared):
    """The singular-point fractions and the probe mesh come from the note."""
    assert declared["prose.singular_from_fraction"].value == pytest.approx(0.15)
    assert declared["prose.singular_to_fraction"].value == pytest.approx(0.75)
    assert declared["prose.layer_route_panels"].value == 64
    assert declared["prose.boundary_spread_panels"].value == 128


# ---------------------------------------------------------------------------
# negative controls: the gate has to be able to fail
# ---------------------------------------------------------------------------
def test_a_perturbed_document_cell_is_reported_as_drift(document, measured):
    """Editing one printed digit (4.67% -> 4.68%) must break the build."""
    mutated = document.replace("| 32 | 4.67% |", "| 32 | 4.68% |")
    assert mutated != document
    verdicts = pins.check(doc_text=mutated, measurements=measured)
    drifted = [v.name for v in verdicts if v.status != "PASS"]
    assert drifted == ["refinement.32.relative_potential_error_percent"]


def test_a_perturbed_measurement_is_reported_as_drift(declared, measured):
    """A code change that moves a number must break the build."""
    mutated = dict(measured)
    original = measured["refinement.128.condition_number"]
    mutated["refinement.128.condition_number"] = pins.Claim(
        token=repr(original.value * 1.2), value=original.value * 1.2, tol=0.0
    )
    verdicts = pins.check(measurements=mutated)
    drifted = [v.name for v in verdicts if v.status != "PASS"]
    assert drifted == ["refinement.128.condition_number"]


def test_second_order_wording_is_rejected(document, measured):
    """The corrected prose claim is load-bearing, not decoration."""
    mutated = document.replace("(first order in the panel",
                              "(second order in the panel")
    assert mutated != document
    verdicts = pins.check(doc_text=mutated, measurements=measured)
    drifted = [v.name for v in verdicts if v.status != "PASS"]
    assert drifted == ["prose.rms_order_word"]


def test_upper_bound_claim_is_one_sided(declared, measured):
    """``less than 1%`` is judged against the bound, not against equality."""
    bound = declared["prose.singular_variation_percent"]
    assert bound.kind == pins.UPPER_BOUND
    assert measured["prose.singular_variation_percent"].value < bound.value
    verdicts = {v.name: v for v in pins.compare(declared, measured)}
    assert verdicts["prose.singular_variation_percent"].status == "PASS"


# ---------------------------------------------------------------------------
# parser and tolerance units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("0.4824", 5e-5),
        ("4.80e-3", 5e-6),
        ("7.36e-4", 5e-7),
        ("61", 0.5),
        ("0.0998", 5e-5),
    ],
)
def test_tolerance_follows_the_printed_precision(printed, expected):
    assert pins.half_unit_in_last_place(printed) == pytest.approx(expected)


def test_claim_parser_handles_percent_signs_and_bounds():
    claim = pins.parse_claim("4.67%")
    assert claim.value == pytest.approx(4.67)
    assert claim.tol == pytest.approx(0.005)
    assert claim.unit == "%"
    bound = pins.parse_claim("<1%")
    assert bound.kind == pins.UPPER_BOUND
    assert bound.tol == 0.0


def test_escaped_pipes_in_the_header_do_not_split_cells():
    header = pins._header([r"| panels | max\|dphi\|/max\|phi_d\| | cond |"])
    assert header == ["panels", r"max\|dphi\|/max\|phi_d\|", "cond"]


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------
def test_cli_passes_on_the_real_document(tmp_path):
    out = tmp_path / "ledger.json"
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.lifting_doc_pins",
         "--check", "--json", str(out)],
        capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    ledger = json.loads(out.read_text())
    assert ledger["schema"] == pins.PIN_SCHEMA
    assert ledger["verdict"] == "PASS"
    assert ledger["drifted"] == 0


def test_cli_fails_on_a_corrupted_document(tmp_path, document):
    corrupted = tmp_path / "lifting_foil.md"
    corrupted.write_text(
        document.replace("| 128 | 2.39% |", "| 128 | 2.40% |"), encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.lifting_doc_pins",
         "--check", "--doc", str(corrupted)],
        capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 1
    ledger = json.loads(completed.stdout)
    assert [row["claim"] for row in ledger["drift"]] == [
        "refinement.128.relative_potential_error_percent"
    ]


def test_observed_order_of_the_rms_error_is_first_order(measured):
    """The finding behind the corrected prose, measured rather than asserted."""
    for pair in ("32_64", "64_128", "128_256"):
        order = measured[f"prose.rms_order_{pair}"].value
        assert 0.8 < order < 1.05, f"{pair}: order {order}"
    assert math.isclose(measured["prose.rms_order_mean"].value,
                        sum(measured[f"prose.rms_order_{p}"].value
                            for p in ("32_64", "64_128", "128_256")) / 3.0)