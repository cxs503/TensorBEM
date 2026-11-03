"""Bind the published 3-D benchmark tables to the pinned values.

``docs/marine_bem_benchmarks.md`` tables 2 (translating sphere) and 4
(translating ellipsoid) quote four mesh-refinement rows each: ``h_max``, the
numerical added mass, its relative error, the per-panel flux error, and the
*observed order of convergence* computed between consecutive rows.  Until now
only the 2-D circle and the Ursell cylinder were pinned in
:mod:`tensorbem.benchmarks.doc_pins`; the whole 3-D no-free-surface section was
prose, so the table could drift while every test stayed green.

Two independent things are checked here:

1. **Table digits vs pins.**  Every number printed in the two tables is parsed
   straight out of the markdown and compared with the corresponding pinned
   measurement (``docs/benchmark_pins.json``).  Nothing is retyped, so a stale
   table or a re-typed digit is caught by the parse.
2. **The convergence-order column is re-derived.**  The order is a *derived*
   quantity (``log(e_coarse/e_fine)/log(h_coarse/h_fine)``), so it is recomputed
   from the pinned errors instead of being pinned itself, and compared with the
   printed order.  A pin on the order alone would hide a compensating error.

The expensive half of the claim -- that the finest (1280-panel) rows still
*reproduce* on the current code -- is a live re-run guarded by
``TENSORBEM_SLOW=1``; it is the row the 2026-10-24 expansion audit flagged as
never directly reproduced (see ``docs/marine_bem_benchmarks.md`` section 4.2/4.4
and ``artifacts/m2638/``).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pytest

from tensorbem.benchmarks import doc_pins

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "marine_bem_benchmarks.md"
PINS = REPO / "docs" / "benchmark_pins.json"

SLOW = os.environ.get("TENSORBEM_SLOW") == "1"
slow = pytest.mark.skipif(not SLOW, reason="set TENSORBEM_SLOW=1 to re-run the 1280-panel rows")

_SUPERSCRIPT = re.compile(r"^\\\((.+?)\\times10\^\{(-?\d+)\}\\\)$")


def parse_cell(cell: str) -> float | None:
    """Parse one published table cell, e.g. ``0.16465`` or ``\\(4.4674\\times10^{-3}\\)``."""
    text = cell.strip()
    if text in {"", "—", "-", "–"}:
        return None
    match = _SUPERSCRIPT.match(text)
    if match:
        return float(match.group(1)) * 10.0 ** int(match.group(2))
    try:
        return float(text)
    except ValueError:  # a header cell such as "\(N\)" or a unit label
        return None


def _table_rows(marker: str) -> list[list[float | None]]:
    """Return the numeric rows of the markdown table that starts at ``marker``."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index(marker)
    rows: list[list[float | None]] = []
    for line in text[start:].splitlines()[1:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        cells = [cell for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= set(":- ") for cell in cells):  # header separator
            continue
        parsed = [parse_cell(cell) for cell in cells]
        if parsed and parsed[0] is None:  # header row
            continue
        rows.append(parsed)
    return rows


def sphere_rows() -> list[list[float | None]]:
    return _table_rows("**表 2. 球体附加质量基准的网格收敛**")


def ellipsoid_rows() -> list[list[float | None]]:
    return _table_rows("**表 4. 椭球三轴附加质量的网格收敛**")


def pinned(cases: dict[str, dict[str, float]], name: str) -> dict[str, float]:
    if name not in cases:
        raise AssertionError(f"{name} is in the documentation but not pinned in {PINS}")
    return cases[name]["metrics"]


@pytest.fixture(scope="module")
def pins() -> dict[str, dict[str, float]]:
    document = json.loads(PINS.read_text(encoding="utf-8"))
    return document["pins"]


def test_tables_are_parsed_completely() -> None:
    assert [row[0] for row in sphere_rows()] == [20, 80, 320, 1280]
    assert [row[0] for row in ellipsoid_rows()] == [20, 80, 320, 1280]


@pytest.mark.parametrize("panel_index", range(4))
def test_sphere_table_digits_match_pins(panel_index: int, pins: dict[str, dict[str, float]]) -> None:
    row = sphere_rows()[panel_index]
    panels, h_max, added_mass, error, flux_error = row[:5]
    values = pinned(pins, f"sphere_added_mass_3d_{int(panels)}")
    published = {
        "h_max": h_max,
        "added_mass": added_mass,
        "added_mass_relative_error": error,
        "normal_derivative_relative_l2_error": flux_error,
    }
    for metric, printed in published.items():
        assert printed is not None, (panels, metric)
        assert values[metric] == pytest.approx(printed, rel=5.0e-5), (panels, metric)


@pytest.mark.parametrize("panel_index", range(4))
def test_ellipsoid_table_digits_match_pins(
    panel_index: int, pins: dict[str, dict[str, float]]
) -> None:
    row = ellipsoid_rows()[panel_index]
    panels, h_max, volume_ratio = row[0], row[1], row[2]
    values = pinned(pins, f"ellipsoid_added_mass_3d_{int(panels)}")
    assert values["h_max"] == pytest.approx(float(h_max), rel=5.0e-5)
    assert values["volume_ratio"] == pytest.approx(float(volume_ratio), rel=5.0e-5)
    for axis in (0, 1, 2):
        printed = row[3 + 2 * axis]
        assert printed is not None
        assert values[f"axes.{axis}.added_mass_relative_error"] == pytest.approx(
            float(printed), rel=5.0e-5
        ), (panels, axis)


def _order(coarse: float, fine: float, h_coarse: float, h_fine: float) -> float:
    return math.log(coarse / fine) / math.log(h_coarse / h_fine)


@pytest.mark.parametrize("row_index", range(1, 4))
def test_published_sphere_orders_are_reproduced(pins: dict[str, dict[str, float]], row_index: int) -> None:
    rows = sphere_rows()
    coarse, fine = rows[row_index - 1], rows[row_index]
    order_published = fine[5]
    assert order_published is not None
    values = [
        pinned(pins, f"sphere_added_mass_3d_{int(row[0])}") for row in (coarse, fine)
    ]
    recomputed = _order(
        values[0]["added_mass_relative_error"],
        values[1]["added_mass_relative_error"],
        values[0]["h_max"],
        values[1]["h_max"],
    )
    assert recomputed == pytest.approx(float(order_published), rel=2.0e-3)


@pytest.mark.parametrize("row_index", range(1, 4))
def test_published_ellipsoid_orders_are_reproduced(
    pins: dict[str, dict[str, float]], row_index: int
) -> None:
    rows = ellipsoid_rows()
    coarse, fine = rows[row_index - 1], rows[row_index]
    values = [
        pinned(pins, f"ellipsoid_added_mass_3d_{int(row[0])}") for row in (coarse, fine)
    ]
    for axis in (0, 1, 2):
        published = fine[4 + 2 * axis]
        assert published is not None, (fine[0], axis)
        recomputed = _order(
            values[0][f"axes.{axis}.added_mass_relative_error"],
            values[1][f"axes.{axis}.added_mass_relative_error"],
            values[0]["h_max"],
            values[1]["h_max"],
        )
        assert recomputed == pytest.approx(float(published), rel=5.0e-3), (fine[0], axis)


def test_every_documented_row_is_pinned() -> None:
    """A row added to either table must come with a pin, otherwise the check is blind."""
    for row in sphere_rows():
        doc_pins.CASES_BY_NAME[f"sphere_added_mass_3d_{int(row[0])}"]
    for row in ellipsoid_rows():
        doc_pins.CASES_BY_NAME[f"ellipsoid_added_mass_3d_{int(row[0])}"]
    documented = {f"sphere_added_mass_3d_{int(r[0])}" for r in sphere_rows()}
    documented |= {f"ellipsoid_added_mass_3d_{int(r[0])}" for r in ellipsoid_rows()}
    pinned_names = {case.name for case in doc_pins.CASES}
    assert documented <= pinned_names


def test_panel_counts_follow_the_subdivision_ladder() -> None:
    for case in doc_pins.CASES:
        if not case.name.startswith(("sphere_added_mass_3d_", "ellipsoid_added_mass_3d_")):
            continue
        panels = int(case.name.rsplit("_", 1)[1])
        subdivisions = int(case.settings["subdivisions"])
        assert panels == 20 * 4**subdivisions, case.name


@slow
@pytest.mark.parametrize(
    "module, case_name, table_row, columns",
    [
        (
            "sphere",
            "sphere_added_mass_3d_1280",
            sphere_rows,
            ("h_max", "added_mass", "added_mass_relative_error", "normal_derivative_relative_l2_error"),
        ),
        (
            "ellipsoid",
            "ellipsoid_added_mass_3d_1280",
            ellipsoid_rows,
            ("h_max", "volume_ratio", "axes.0.added_mass_relative_error",
             "axes.1.added_mass_relative_error", "axes.2.added_mass_relative_error"),
        ),
    ],
)
def test_finest_row_reproduces_on_current_code(
    module: str, case_name: str, table_row, columns: tuple[str, ...]
) -> None:
    """Directly re-run the 1280-panel row (the audit's open item) and compare."""
    case = doc_pins.CASES_BY_NAME[case_name]
    measured = case.evaluate()
    published_row = table_row()[-1]
    if module == "sphere":
        printed = [published_row[1], published_row[2], published_row[3], published_row[4]]
    else:
        printed = [published_row[1], published_row[2], published_row[3],
                   published_row[5], published_row[7]]
    for metric, value in zip(columns, printed):
        assert value is not None
        assert measured[metric] == pytest.approx(float(value), rel=1.0e-5), metric