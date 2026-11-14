"""Locks the measured band-edge exponents of the Wigley III radiation sweep.

The panel data these tests read were solved by job ``cj3099`` (medium level,
``scripts/run_cummins_wigley_closure_full.py --stage solve --grid extended``)
and are committed under
``artifacts/case_bem_cummins_wigley_ext/points/``.  The assertions are the
claims the round's report makes; each one is a number the sweep decides, not a
restatement of the code.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorbem.benchmarks import cummins_wigley_kk_causality as case

POINTS = (pathlib.Path(__file__).resolve().parents[1]
          / "artifacts" / "case_bem_cummins_wigley_ext" / "points")
EDGES = POINTS.parent / "band_edges.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not POINTS.exists():
        pytest.skip("extended-band point cache not present in this checkout")
    edges = POINTS.parent / "band_edges.json"
    if not edges.exists():
        pytest.skip("band_edges.json not generated (run scripts/cummins_wigley_band_edges.py)")
    return json.loads(edges.read_text())


def test_low_frequency_nodes_are_present_and_below_the_round2_band() -> None:
    if not POINTS.exists():
        pytest.skip("extended-band point cache not present in this checkout")
    points = case.load_points(POINTS, "medium")
    below = sorted(w for w in points if w < case.OMEGA_GRID[0])
    assert below == [pytest.approx(0.10), pytest.approx(0.15),
                     pytest.approx(0.20), pytest.approx(0.25)]
    for payload in points.values():
        assert payload["rotation_center"] == pytest.approx(list(case.ROTATION_CENTER))


def test_measured_low_frequency_exponent_is_heave_three(report: dict) -> None:
    """Heave ``B ~ w^3`` (the dipole closure the kernel ships) on real panel data."""
    row = {r["dof"]: r for r in report["rows"]}["heave"]
    q = row["low_exponent_q"]
    assert q["exponent"] == pytest.approx(3.0, abs=0.05)
    assert q["r_squared"] > 0.999


def test_low_frequency_exponents_are_integers_per_dof(report: dict) -> None:
    """The measured exponents are the classical ones (``w^7``, ``w^3``, ``w^11``)."""
    expected = {"surge": 7.0, "sway": 7.0, "heave": 3.0,
                "roll": 7.0, "pitch": 7.0, "yaw": 11.0}
    for row in report["rows"]:
        q = row["low_exponent_q"]
        assert q["exponent"] == pytest.approx(expected[row["dof"]], abs=0.05)
        assert q["r_squared"] > 0.995


def test_sub_band_closure_matters_less_than_a_tenth_of_a_percent(report: dict) -> None:
    """``K(0) = (2/pi) int B dw``: the band below 0.10 rad/s is measured, not modelled."""
    for row in report["rows"]:
        assert row["K_zero_below_power_fraction"] < 1e-3
        assert row["K_zero_below_constant_fraction"] < 1e-3


def test_high_frequency_chord_is_not_yet_an_asymptote(report: dict) -> None:
    """Honest limit: 4.7 -> 12.5 rad/s is a chord, and it does not decay for 4 DOF.

    The docstring of :func:`tensorbem.causality.kkr_added_mass` drops the tail
    claiming the damping "vanishes exponentially fast".  With the highest
    available node at 12.5 rad/s and no 15 / 20 rad/s nodes yet, the chord
    exponent cannot support that claim -- so the test pins the *measurement*,
    not a pass.
    """
    rows = {r["dof"]: r for r in report["rows"]}
    growing = [dof for dof, r in rows.items()
               if r["high_decay_exponent_p"]["exponent"] < 0.5]
    assert growing, "if this ever fails the 15/20 rad/s nodes were solved: re-do §5"
    assert np.isfinite(rows["pitch"]["high_decay_exponent_p"]["exponent"])


def test_geometry_gate_precedes_every_number(report: dict) -> None:
    geometry = case.geometry_gate()
    assert geometry["passed"] is True
    assert geometry["Cb_abs_error"] < 1e-9
    assert report["n_nodes"] == len(report["nodes"]) >= 14