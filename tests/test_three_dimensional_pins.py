"""Machine-checked pins for the three-dimensional numbers in the documentation.

``docs/marine_bem_benchmarks.md`` tables 2-5 and ``docs/three_dimensional_bem.md``
quote 3-D errors (sphere added mass, ellipse added mass, ellipsoid three axes,
interior solid harmonics, constant Dirichlet capacitance). Before this module
those numbers were prose only: ``docs/benchmark_pins.json`` pinned the 2-D
cylinder and the Ursell cylinder, so a solver change could move every 3-D table
without any test failing -- exactly the failure mode that was found for the
Ursell 128-panel row (``docs/ursell_quadrature_provenance.md``).

These tests run the cheap 3-D cases (20/80 panels, 2-D ellipse at 128 panels)
on every test run and keep the expensive 1280-panel rows out of the default
selection. The heavy rows are still pinned; they are recomputed by
``python -m tensorbem.benchmarks.doc_pins --write`` and by
``pytest --slow``-style manual runs gated on ``TENSORBEM_SLOW_TESTS``.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tensorbem.benchmarks.doc_pins import (
    CASES,
    CASES_BY_NAME,
    DEFAULT_PIN_PATH,
    check,
    render_three_dimensional_tables,
)

#: Every documented three-dimensional / ellipse row that must stay pinned.
THREE_DIMENSIONAL_CASES = (
    "sphere_added_mass_3d_20",
    "sphere_added_mass_3d_80",
    "sphere_added_mass_3d_320",
    "sphere_added_mass_3d_1280",
    "constant_dirichlet_3d_1280",
    "ellipse_added_mass_2d_128_x",
    "ellipse_added_mass_2d_128_y",
    "ellipsoid_added_mass_3d_20",
    "ellipsoid_added_mass_3d_1280",
    "interior_harmonic_degree1_1280",
    "interior_harmonic_degree2_1280",
    "interior_harmonic_degree3_1280",
)

#: Cases cheap enough for the default test run (a few seconds in total).
FAST_CASES = tuple(
    CASES_BY_NAME[name] for name in THREE_DIMENSIONAL_CASES if not CASES_BY_NAME[name].slow
)
SLOW_CASES = tuple(
    CASES_BY_NAME[name] for name in THREE_DIMENSIONAL_CASES if CASES_BY_NAME[name].slow
)


def _pins() -> dict:
    document = json.loads(Path(DEFAULT_PIN_PATH).read_text(encoding="utf-8"))
    return document["pins"]


def test_every_three_dimensional_table_row_is_registered():
    registered = {case.name for case in CASES}
    missing = [name for name in THREE_DIMENSIONAL_CASES if name not in registered]
    assert not missing, f"3-D rows not registered as pins: {missing}"


def test_three_dimensional_pins_are_in_the_pin_file():
    pins = _pins()
    missing = [name for name in THREE_DIMENSIONAL_CASES if name not in pins]
    assert not missing, f"3-D rows absent from {DEFAULT_PIN_PATH}: {missing}"


def test_three_dimensional_pins_record_their_documentation_source():
    for name in THREE_DIMENSIONAL_CASES:
        case = CASES_BY_NAME[name]
        assert case.doc, f"{name}: no documentation citation"
        assert case.metrics, f"{name}: no pinned metric"
        assert 0.0 < case.rtol < 1.0e-2, f"{name}: suspicious tolerance {case.rtol}"


def test_three_dimensional_pins_are_all_finite_and_positive():
    pins = _pins()
    for name in THREE_DIMENSIONAL_CASES:
        for metric, value in pins[name]["metrics"].items():
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(value), f"{name}.{metric} = {value!r}"
            assert value > 0.0, f"{name}.{metric} = {value!r}"


def test_fast_three_dimensional_pins_still_reproduce():
    assert FAST_CASES, "the fast 3-D tier must not be empty"
    problems, measured = check(cases=FAST_CASES)
    assert not problems, "documented 3-D numbers drifted:\n" + "\n".join(problems)
    assert set(measured) == {case.name for case in FAST_CASES}


def test_the_default_check_skips_the_1280_panel_rows():
    problems, measured = check()
    assert not problems, "\n".join(problems)
    assert not set(measured) & {case.name for case in SLOW_CASES}


@pytest.mark.skipif(
    not os.environ.get("TENSORBEM_SLOW_TESTS"),
    reason="recomputes 1280-panel 3-D solves; set TENSORBEM_SLOW_TESTS=1 to run",
)
def test_slow_three_dimensional_pins_still_reproduce():
    problems, measured = check(cases=SLOW_CASES)
    assert not problems, "documented 1280-panel numbers drifted:\n" + "\n".join(problems)
    assert set(measured) == {case.name for case in SLOW_CASES}


def test_three_dimensional_tables_render_from_the_pins():
    table = render_three_dimensional_tables()
    assert table.count("**Table") == 4
    for marker in ("Table 2", "Table 3", "Table 4", "Table 5"):
        assert marker in table
    assert "—" in table or "nan" not in table.lower()
    # The rendered rows must carry numbers, not placeholders.
    numbers = [chunk for chunk in table.split() if chunk.startswith("$")]
    assert len(numbers) >= 12, numbers