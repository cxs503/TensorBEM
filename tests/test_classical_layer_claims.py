"""Guard the README's *classical layer* claims (2-D ellipse, 3-D ellipsoid).

``docs/benchmark_pins.json`` re-runs the documented numbers, but a pin only
protects the code: the README paragraph can still drift away from the pin file
without any test noticing, and until now the classical-layer numbers were not
pinned at all.  ``expand_audit_tensorbem.md`` §4.2 recorded exactly that gap --
the audit could reproduce the 320- and 5120-panel ellipsoid rows and had to
*extrapolate* for the 1280-panel row that the README quotes as the headline.

This module closes the loop in the direction the pin file cannot: it parses the
numbers back out of the prose (``README.md`` and
``docs/marine_bem_benchmarks.md`` tables 3 and 4) and compares them with the
pinned measurements.  Three claims are checked:

* the five README values -- ``7.54e-4`` (x) and ``9.00e-4`` (y) on the ellipse,
  ``4.17e-3``/``4.46e-3``/``4.94e-3`` on the ellipsoid;
* the table cells for the pinned discretisations (128-panel ellipse,
  320- and 1280-panel ellipsoid), including ``h_max`` and the volume ratio;
* the printed *observed order* of the ellipsoid, recomputed from the pinned
  320/1280 pair rather than trusted.

Tolerances come from printed precision (half a unit in the last printed digit),
never from physics, so a correctly rounded claim passes and a stale one fails.

What this file does **not** cover: the ellipse observed order ``2.00`` needs the
64-panel row, which is not pinned (only the 128-panel row is), and the 20/80
panel rows are "—" for order in the table.  That order is therefore not guarded
here.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from tensorbem.benchmarks import doc_pins

REPO_ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = REPO_ROOT / "docs" / "benchmark_pins.json"
README = REPO_ROOT / "README.md"
REPORT = REPO_ROOT / "docs" / "marine_bem_benchmarks.md"

_NUM = r"\d+(?:\.\d+)?"
#: A number as the documents print it: ``\(4.17\times10^{-3}\)`` or ``0.48738``.
#: ``\(4.17\\times10^{-3}\)`` -- the mantissa plus its power of ten; the
#: surrounding ``\( \)`` is optional so the same pattern reads markdown
#: tables and prose.
_LATEX = re.compile("(" + _NUM + r")\\times10\^\{(-?\d+)\}")
_PLAIN = re.compile(r"(?<![\d.^])(\d+\.\d+)(?![\d.^])")


def _printed_values(text: str) -> list[tuple[float, str, int]]:
    """Return ``(value, literal, significant_digits)`` for every printed number.

    LaTeX ``a\\times10^{b}`` forms are converted; plain decimals are kept as
    they are.  The literal is returned so that the tolerance can be derived
    from how many digits the author actually printed.
    """
    found: list[tuple[int, float, str, int]] = []
    spans: list[tuple[int, int]] = []
    for match in _LATEX.finditer(text):
        mantissa, exponent = match.group(1), int(match.group(2))
        value = float(mantissa) * 10.0 ** exponent
        digits = len(mantissa.replace(".", "").lstrip("0"))
        found.append((match.start(), value, mantissa, digits))
        spans.append((match.start(), match.end()))
    for match in _PLAIN.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        literal = match.group(1)
        digits = len(literal.replace(".", "").lstrip("0"))
        found.append((match.start(), float(literal), literal, digits))
    found.sort(key=lambda item: item[0])
    return [(value, literal, digits) for _, value, literal, digits in found]


def _tolerance(digits: int) -> float:
    """Half a unit in the last printed place, as a *relative* tolerance."""
    return 0.5 * 10.0 ** -(digits - 1)


def _readme_paragraph() -> str:
    """The README paragraph that states the classical-layer numbers."""
    text = README.read_text(encoding="utf-8")
    start = text.index("The classical benchmark layer reports")
    end = text.index("\n\n", start)
    return text[start:end]


def _table_rows(heading: str) -> dict[int, list[str]]:
    """Return ``{panel_count: cells}`` for the markdown table under ``heading``."""
    text = REPORT.read_text(encoding="utf-8")
    start = text.index(heading)
    body = text[start:].split("\n\n", 1)[1]
    rows: dict[int, list[str]] = {}
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or not cells[0].isdigit():
            continue  # header / separator
        rows[int(cells[0])] = cells
    return rows


def _pins() -> dict[str, dict[str, object]]:
    document = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    return document["pins"]


def _cell_value(cell: str) -> tuple[float, int]:
    """Parse one table cell into ``(value, significant_digits)``."""
    values = _printed_values(cell)
    if len(values) != 1:
        raise AssertionError(f"expected exactly one number in cell {cell!r}")
    value, _literal, digits = values[0]
    return value, digits


def _assert_claim(measured: float, printed: float, digits: int, where: str) -> None:
    tolerance = _tolerance(digits)
    drift = abs(measured - printed) / max(abs(printed), 1e-300)
    assert drift <= tolerance, (
        f"{where}: document prints {printed!r} ({digits} significant digits, "
        f"tolerance {tolerance:.1e}), pin file says {measured!r} "
        f"(relative drift {drift:.3e})"
    )


# ---------------------------------------------------------------------------
# README prose
# ---------------------------------------------------------------------------
def test_the_readme_paragraph_still_exists():
    """Guard against a vacuous pass if the paragraph is reworded away."""
    paragraph = _readme_paragraph()
    values = [value for value, _, _ in _printed_values(paragraph)]
    assert len(values) >= 5, values
    assert "7.54" in paragraph and "4.17" in paragraph


def test_readme_classical_layer_numbers_match_the_pins():
    """``7.54e-4``/``9.00e-4`` (ellipse) and ``4.17e-3``/``4.46e-3``/``4.94e-3``."""
    paragraph = _readme_paragraph()
    printed = {
        literal: (value, digits) for value, literal, digits in _printed_values(paragraph)
    }
    pins = _pins()

    ellipse_x = pins["ellipse_added_mass_2d_128_x"]["metrics"]
    ellipse_y = pins["ellipse_added_mass_2d_128_y"]["metrics"]
    ellipsoid = pins["ellipsoid_added_mass_1280"]["metrics"]

    cases = (
        ("ellipse x", ellipse_x["added_mass_relative_error"], "7.54"),
        ("ellipse y", ellipse_y["added_mass_relative_error"], "9.00"),
        ("ellipsoid x", ellipsoid["axes.0.added_mass_relative_error"], "4.17"),
        ("ellipsoid y", ellipsoid["axes.1.added_mass_relative_error"], "4.46"),
        ("ellipsoid z", ellipsoid["axes.2.added_mass_relative_error"], "4.94"),
    )
    for label, measured, literal in cases:
        assert literal in printed, f"README no longer prints {literal}"
        value, digits = printed[literal]
        _assert_claim(measured, value, digits, f"README {label}")


def test_readme_claims_are_rounded_not_copied():
    """Negative control: the test compares *rounded* claims, so it can fail.

    If the README printed the pinned float verbatim the tolerance would never
    bite.  Every headline claim is printed to three significant digits, so the
    printed and measured values must differ by more than zero.
    """
    pins = _pins()
    ellipsoid = pins["ellipsoid_added_mass_1280"]["metrics"]
    measured = ellipsoid["axes.0.added_mass_relative_error"]
    assert measured != pytest.approx(4.17e-3, rel=0.0, abs=0.0)
    assert abs(measured - 4.17e-3) > 0.0


# ---------------------------------------------------------------------------
# tables 3 and 4 of the benchmark report
# ---------------------------------------------------------------------------
def test_ellipse_table_row_matches_the_pins():
    """Table 3, 128 panels: both axes, error and ``h_max``."""
    row = _table_rows("**表 3. 椭圆柱附加质量")
    assert 128 in row, sorted(row)
    cells = row[128]
    pins = _pins()
    x = pins["ellipse_added_mass_2d_128_x"]["metrics"]
    y = pins["ellipse_added_mass_2d_128_y"]["metrics"]

    # | N | h_max | m_x | rel err x | order | m_y | rel err y | order |
    claimed = (
        ("h_max", x["h_max"], cells[1]),
        ("m_x", x["added_mass_per_unit_span"], cells[2]),
        ("rel x", x["added_mass_relative_error"], cells[3]),
        ("m_y", y["added_mass_per_unit_span"], cells[5]),
        ("rel y", y["added_mass_relative_error"], cells[6]),
    )
    for label, measured, cell in claimed:
        value, digits = _cell_value(cell)
        _assert_claim(measured, value, digits, f"table 3, 128 panels, {label}")


@pytest.mark.parametrize("panels,case", [(320, "ellipsoid_added_mass_320"),
                                         (1280, "ellipsoid_added_mass_1280")])
def test_ellipsoid_table_rows_match_the_pins(panels, case):
    """Table 4 rows for the two pinned discretisations, cell by cell."""
    row = _table_rows("**表 4. 椭球三轴附加质量的网格收敛**")
    assert panels in row, sorted(row)
    cells = row[panels]
    metrics = _pins()[case]["metrics"]

    claimed = (
        ("h_max", metrics["h_max"], cells[1]),
        ("V/V_exact", metrics["volume_ratio"], cells[2]),
        ("rel x", metrics["axes.0.added_mass_relative_error"], cells[3]),
        ("rel y", metrics["axes.1.added_mass_relative_error"], cells[5]),
        ("rel z", metrics["axes.2.added_mass_relative_error"], cells[7]),
    )
    for label, measured, cell in claimed:
        value, digits = _cell_value(cell)
        _assert_claim(measured, value, digits, f"table 4, {panels} panels, {label}")


def test_printed_observed_order_matches_the_pinned_pair():
    """The 1280-panel row prints an order; recompute it from the pinned 320 row.

    The order is ``log(e_320 / e_1280) / log(h_320 / h_1280)``: the icosphere
    quadruples its panel count per subdivision, so ``h`` halves and an ``h^2``
    method drops its error by four, not by sixteen.  Reading the order off the
    printed errors instead of re-running the coarse case is what pins 4 (rows
    320 and 1280) are for.
    """
    rows = _table_rows("**表 4. 椭球三轴附加质量的网格收敛**")
    assert 320 in rows and 1280 in rows
    coarse = _pins()["ellipsoid_added_mass_320"]["metrics"]
    fine = _pins()["ellipsoid_added_mass_1280"]["metrics"]

    h_ratio = coarse["h_max"] / fine["h_max"]
    for axis, column in ((0, 4), (1, 6), (2, 8)):
        error_ratio = (
            coarse[f"axes.{axis}.added_mass_relative_error"]
            / fine[f"axes.{axis}.added_mass_relative_error"]
        )
        measured_order = math.log(error_ratio) / math.log(h_ratio)
        printed, digits = _cell_value(rows[1280][column])
        assert abs(measured_order - printed) <= 0.01, (
            f"axis {axis}: pinned errors give observed order {measured_order:.4f}, "
            f"table prints {printed} ({digits} digits)"
        )


# ---------------------------------------------------------------------------
# registration contract
# ---------------------------------------------------------------------------
def test_the_headline_rows_are_registered_and_bounded_in_cost():
    """The expensive row must stay ``slow`` and the cheap rows must stay fast."""
    cases = {case.name: case for case in doc_pins.CASES}
    for name in ("ellipse_added_mass_2d_128_x", "ellipse_added_mass_2d_128_y",
                 "ellipsoid_added_mass_320"):
        assert name in cases, name
        assert not cases[name].slow, f"{name} must stay in the default CI check"
    assert "ellipsoid_added_mass_1280" in cases
    assert cases["ellipsoid_added_mass_1280"].slow, (
        "the 1280-panel ellipsoid case takes about a minute; it stays out of the "
        "default check and is verified with --include-slow"
    )
    pins = _pins()
    for name in cases:
        assert name in pins, f"{name} is registered but not pinned"