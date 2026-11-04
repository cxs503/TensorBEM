"""Guard for Table 1 of ``docs/three_dimensional_bem.md`` (icosphere mesh quality).

The tests fall into three groups:

* the claim currently reproduces (all 36 printed cells, the two integer closed
  forms and the three prose bounds);
* the guard actually bites -- a single edited digit in the document must turn
  the verdict into ``DRIFT``, and the affected cell must be identified.  A guard
  that cannot fail is not a guard;
* the tolerance is the printed precision, not a magic constant: a claim printed
  to five significant digits must accept a 1e-6 relative move and reject a 1e-3
  one.
"""

from __future__ import annotations

import math
import re

import pytest

from tensorbem.benchmarks import mesh_doc_pins as pins
from tensorbem.benchmarks.three_dimensional_doc_pins import (
    DOC,
    coverage,
    printed_tolerance,
)

DOC_TEXT = DOC.read_text()


def _report(text: str = DOC_TEXT) -> dict:
    return pins.check(text)


# ---------------------------------------------------------------------------
# The claim, as printed
# ---------------------------------------------------------------------------


def test_table_1_reproduces_within_printed_precision() -> None:
    report = _report()
    assert report["verdict"] == "PASS"
    assert report["n_cells"] == 36  # 9 registered columns x 4 levels
    assert report["n_failed"] == 0
    assert report["levels"] == [0, 1, 2, 3]


def test_worst_deviation_is_far_inside_the_printed_precision() -> None:
    report = _report()
    worst = 0.0
    for cell in report["checks"]:
        if cell["kind"] != "value":
            continue
        gap = abs(cell["measured"] - cell["declared"]) / max(
            abs(cell["declared"]), abs(cell["measured"])
        )
        assert gap <= cell["tolerance"], cell
        worst = max(worst, gap / cell["tolerance"])
    # A correctly rounded claim can use up to half a unit of its last printed
    # digit, i.e. ratio ~0.5; the observed worst is 0.56 (the total area, printed
    # to seven significant digits).  Anything at or beyond 1.0 would be drift.
    assert worst < 0.6, worst


def test_refinement_rule_closed_forms_hold() -> None:
    report = _report()
    identities = {entry["id"]: entry for entry in report["identity_checks"]}
    assert len(identities) == 8  # panels + vertices for k = 0..3
    for level in range(4):
        assert identities[f"level_{level}_panels_closed_form"]["measured"] == 20 * 4**level
        assert identities[f"level_{level}_vertices_closed_form"]["measured"] == 10 * 4**level + 2
    assert all(entry["status"] == "PASS" for entry in identities.values())


def test_prose_bounds_hold_with_margin() -> None:
    report = _report()
    bounds = {entry["id"]: entry for entry in report["bound_checks"]}
    assert set(bounds) == {
        "inscribed_area_below_sphere",
        "inscribed_volume_below_sphere",
        "area_monotone_under_refinement",
        "volume_monotone_under_refinement",
        "aspect_ratio_bound",
    }
    assert all(entry["status"] == "PASS" for entry in bounds.values())
    # The inscribed polyhedron approaches the sphere from below and never crosses.
    assert bounds["inscribed_area_below_sphere"]["measured_worst"] < 1.0
    assert bounds["inscribed_volume_below_sphere"]["measured_worst"] < 1.0
    # The aspect bound is read out of the document, so the claim stays the
    # source of truth rather than a copy of today's measurement.
    assert bounds["aspect_ratio_bound"]["documented"] == "< 1.45"
    assert bounds["aspect_ratio_bound"]["measured_worst"] < 1.45


def test_h_mean_counts_each_face_edge_once() -> None:
    """Pin the convention the printed ``h_mean`` column follows.

    The alternative reading (mean of the per-panel longest edge) also looks
    plausible, so it is pinned down here: at ``k = 1`` it would give 0.61803
    instead of the documented 0.58228.
    """
    mesh = pins.icosphere(1, 1.0)
    per_panel_longest = float(mesh.edge_lengths.max(axis=1).mean())
    assert math.isclose(per_panel_longest, 0.6180339887, rel_tol=1e-9)
    assert not math.isclose(per_panel_longest, 0.58228, rel_tol=1e-3)
    assert math.isclose(pins.measure()[1]["h_mean"], 0.5822835, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# The guard must be able to fail
# ---------------------------------------------------------------------------


def test_single_digit_drift_in_area_is_caught() -> None:
    corrupted = DOC_TEXT.replace("| 9.574541 |", "| 9.574641 |", 1)
    assert corrupted != DOC_TEXT
    report = _report(corrupted)
    assert report["verdict"] == "DRIFT"
    failed = [cell for cell in report["checks"] if cell["status"] != "PASS"]
    assert len(failed) == 1
    assert failed[0]["quantity"] == "surface_area"
    assert failed[0]["panels"] == 20


def test_wrong_panel_count_is_caught_even_though_it_is_an_integer() -> None:
    corrupted = DOC_TEXT.replace("| 2 | 320 | 162 |", "| 2 | 320 | 160 |", 1)
    assert corrupted != DOC_TEXT
    report = _report(corrupted)
    assert report["verdict"] == "DRIFT"
    failed = [cell for cell in report["checks"] if cell["status"] != "PASS"]
    assert [cell["quantity"] for cell in failed] == ["vertices"]
    assert failed[0]["panels"] == 320


def test_loosening_the_aspect_bound_is_caught() -> None:
    corrupted = DOC_TEXT.replace("保持在 1.45 以下", "保持在 1.40 以下", 1)
    assert corrupted != DOC_TEXT
    report = _report(corrupted)
    assert report["verdict"] == "DRIFT"
    bounds = {entry["id"]: entry for entry in report["bound_checks"]}
    assert bounds["aspect_ratio_bound"]["status"] == "DRIFT"
    # ... and an unreachably tight bound is "passed" only by a stricter one:
    loosened = DOC_TEXT.replace("保持在 1.45 以下", "保持在 1.50 以下", 1)
    assert {
        entry["id"]: entry["status"] for entry in _report(loosened)["bound_checks"]
    }["aspect_ratio_bound"] == "PASS"


def test_deleting_the_table_breaks_the_parser_loudly() -> None:
    without_table = "\n".join(
        line for line in DOC_TEXT.splitlines() if not line.strip().startswith("|")
    )
    with pytest.raises(pins.ParseError):
        pins.parse_mesh_table(without_table)


# ---------------------------------------------------------------------------
# Tolerance = printed precision
# ---------------------------------------------------------------------------


def test_tolerance_follows_the_number_of_printed_digits() -> None:
    assert printed_tolerance("0.76192") == pytest.approx(0.6e-4)   # 5 digits
    assert printed_tolerance("1.05146") == pytest.approx(0.6e-5)   # 6 digits
    assert printed_tolerance("9.574541") == pytest.approx(0.6e-6)  # 7 digits
    # A six-digit claim accepts a 1e-6 relative move and rejects a 1e-3 one.
    five = printed_tolerance("0.76192")
    assert pins._agrees(0.76192, 0.7619205, five)
    assert not pins._agrees(0.76192, 0.76202, five)


# ---------------------------------------------------------------------------
# Integration with the coverage report of the sibling module
# ---------------------------------------------------------------------------


def test_coverage_counts_the_mesh_table_and_leaves_no_claim_unguarded() -> None:
    report = coverage(DOC_TEXT)
    # The four result tables (86 tokens) plus the 28 decimal cells of Table 1.
    assert report["protected_in_tables"] >= 114
    assert report["protected_fraction"] > 0.45
    unguarded = report["unguarded_by_section"]
    # Two numeric tokens of the section stay outside the table-line coverage
    # proxy: the heading number "3.1" (not a claim at all) and the prose decimal
    # 1.45.  The latter *is* executed -- bound_checks() parses it out of the
    # document and fails if the mesh reaches it -- so the section has no real
    # unguarded claim left, while the line-based proxy cannot know that.
    assert unguarded.get("3.1 二十面体球面网格", 0) <= 2


def test_generated_section_is_the_fresh_render() -> None:
    """The injected block must be byte-identical to a fresh render.

    This is what makes the section's own numbers executable: editing them by hand
    (the usual route of drift) fails here instead of surviving until an audit.
    """
    section = pins.extract_section(DOC_TEXT)
    assert section is not None, "the generated section is missing from the document"
    assert section == pins.render_section(_report())


def test_generated_section_is_excluded_from_the_coverage_denominator() -> None:
    """Numbers rendered from a report are not transcriptions, so they do not count.

    Verified against an independent recount rather than a frozen total, so the
    test does not go stale when prose is added elsewhere in the document.
    """
    block = pins.generated_line_range(DOC_TEXT)
    assert block is not None
    start, end = block
    number = re.compile(r"\b\d+\.\d+(?:\\times10\^\{[-+]?\d+\})?(?:e[-+]?\d+)?\b")
    lines = DOC_TEXT.splitlines()
    inside = sum(len(number.findall(line)) for line in lines[start - 1 : end])
    assert inside > 0, "the generated block carries no numbers, so the rule is moot"
    outside = sum(
        len(number.findall(line))
        for index, line in enumerate(lines, start=1)
        if not start <= index <= end
    )
    assert coverage(DOC_TEXT)["numeric_tokens_total"] == outside


def test_table_lines_are_reported_for_the_coverage_hook() -> None:
    lines = pins.table_lines(DOC_TEXT)
    assert len(lines) == 4
    assert all(line > 0 for line in lines)
    assert pins.table_lines("no table here") == set()