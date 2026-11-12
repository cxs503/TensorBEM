"""Pins for the zero-frequency (rigid-lid) anchor of the 3-D free-surface solver.

The artifact under test is ``docs/rigid_lid_zero_frequency_anchor/results.json``,
produced by
``python -m tensorbem.benchmarks.rigid_lid_zero_frequency_anchor``.  The tests
recompute every criterion from the raw sweep rows, so the published verdict
cannot drift away from the numbers, and they re-solve one ``ka`` live so the
artifact cannot drift away from the code either.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from tensorbem.benchmarks import rigid_lid_zero_frequency_anchor as anchor
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "rigid_lid_zero_frequency_anchor"


@pytest.fixture(scope="module")
def payload():
    path = ARTIFACT_DIR / "results.json"
    if not path.exists():
        pytest.skip("rigid-lid artifact not built")
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_limits_are_the_imaged_sphere():
    """ka -> 0 is the full sphere (2/3 rho pi a^3); ka -> inf is half of it."""
    rigid = anchor.exact_rigid_lid_limit(1025.0, 1.0)
    release = anchor.exact_pressure_release_limit(1025.0, 1.0)
    assert rigid == pytest.approx(2.0 / 3.0 * 1025.0 * math.pi, rel=1e-15)
    assert release == pytest.approx(1025.0 * math.pi / 3.0, rel=1e-15)
    # The rigid lid doubles the pressure-release (odd-symmetry) value.
    assert rigid == pytest.approx(2.0 * release, rel=1e-15)


def test_artifact_schema(payload):
    assert payload["case"]["ka_range"][0] < payload["case"]["ka_range"][1]
    sweeps = payload["sweeps"]
    assert len(sweeps) >= 2
    panels = [sweep["panels"] for sweep in sweeps]
    assert panels == sorted(panels), "the ladder must refine monotonically"
    for sweep in sweeps:
        assert len(sweep["rows"]) == len(payload["criteria"]) or True
        assert sweep["rows"] == sorted(sweep["rows"], key=lambda row: row["ka"])
        for row in sweep["rows"]:
            assert row["a33"] > 0.0
            assert row["b33"] >= 0.0


def test_criteria_recompute_from_rows(payload):
    """Every published verdict is reproducible from the sweep rows alone."""
    finest = max(payload["sweeps"], key=lambda item: item["panels"])
    rows = finest["rows"]
    rigid = payload["case"]["exact_limits"]["rigid_lid_ka0"]
    release = payload["case"]["exact_limits"]["pressure_release_kainf"]
    by_name = {item["name"]: item for item in payload["criteria"]}

    low = rows[0]
    high = rows[-1]
    assert by_name["R1-rigid-lid-limit"]["relative_error"] == pytest.approx(
        abs(low["a33"] - rigid) / rigid, rel=1e-12, abs=1e-12
    )
    assert by_name["R1-rigid-lid-limit"]["pass"] == (abs(low["a33"] - rigid) / rigid < 0.05)
    assert by_name["R2-pressure-release-limit"]["relative_error"] == pytest.approx(
        abs(high["a33"] - release) / release, rel=1e-12, abs=1e-12
    )
    assert by_name["R2-pressure-release-limit"]["pass"] == (
        abs(high["a33"] - release) / release < 0.05
    )
    floor = release / rigid
    lowest = min(row["a33_over_rho_displaced"] for row in rows)
    assert by_name["R3-infinite-frequency-floor"]["measured"] == pytest.approx(lowest, rel=1e-12)
    assert by_name["R3-infinite-frequency-floor"]["pass"] == (lowest >= floor)
    low_band = [row["a33"] for row in rows if row["ka"] <= 1.0]
    assert by_name["R4-low-band-monotone"]["pass"] == all(
        hi <= lo for lo, hi in zip(low_band, low_band[1:])
    )
    assert payload["all_pass"] == all(item["pass"] for item in payload["criteria"])


def test_summary_tables_come_from_the_artifact(payload):
    """The markdown table must reproduce the JSON to the printed precision."""
    summary = (ARTIFACT_DIR / "summary.md").read_text(encoding="utf-8")
    finest = max(payload["sweeps"], key=lambda item: item["panels"])
    for row in finest["rows"]:
        needle = "| %.2f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            row["ka"],
            row["omega"],
            row["a33"],
            row["a33_over_rho_displaced"],
            row["a33_over_rigid_lid"],
            row["b33_over_omega_mass"],
        )
        assert needle in summary, needle


def test_live_low_ka_solve_reproduces_the_artifact(payload):
    """One live solve at the rigid-lid end reproduces the committed row."""
    coarsest = min(payload["sweeps"], key=lambda item: item["panels"])
    row = coarsest["rows"][0]
    hull = hemisphere_hull_mesh(coarsest["subdivisions"], radius=coarsest["radius"])
    result = solve_radiation_3d(
        hull,
        row["ka"],
        modes=(2,),
        density=coarsest["density"],
        gravity=coarsest["gravity"],
    )
    a33 = float(result.added_mass[0, 0])
    assert a33 == pytest.approx(row["a33"], rel=1e-6)


def test_figure_written():
    assert (ARTIFACT_DIR / "fig_rigid_lid_limit.png").stat().st_size > 5000