"""Verification tests for the rigid-lid hemisphere attribution study.

The study (``tensorbem.benchmarks.rigid_lid_hemisphere_attribution``) refutes the
reference value used by report #15809 (``A33(ka->0) = rho*nabla``) and shows the
3-D radiation solver agrees with the even-parity double-body series instead.

These tests recompute every criterion from the committed payload *and* re-derive
the reference from the series, so a drift between code, artifacts and the
documented numbers fails here.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.benchmarks.rigid_lid_hemisphere_attribution import (
    even_mode_added_mass_dimless,
    pressure_release_added_mass_dimless,
    render_summary,
    run_study,
)
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

ARTIFACTS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "bem_rigid_lid_hemisphere"
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads((ARTIFACTS / "results.json").read_text())


def test_even_mode_series_is_machine_converged() -> None:
    """The rigid-lid value must not depend on the truncation degree."""
    values = [
        even_mode_added_mass_dimless(degree_max=degree)["rigid_lid_added_mass_dimless"]
        for degree in (60, 80, 120)
    ]
    assert max(values) - min(values) < 1e-6
    # Refutes the "mirror image = translating sphere => 1.0" claim of report 15809.
    assert 0.82 < values[-1] < 0.84
    assert abs(values[-1] - 1.0) > 0.15


def test_pressure_release_endpoint_is_the_odd_single_mode() -> None:
    """``ka -> infinity`` keeps anchor A2: mu = P_1 is already one odd mode."""
    assert pressure_release_added_mass_dimless() == 0.5
    series = even_mode_added_mass_dimless(degree_max=40)
    assert series["pressure_release_added_mass_dimless"] == pytest.approx(0.5, abs=1e-12)


def test_rigid_lid_value_differs_from_pressure_release_by_more_than_60_percent() -> None:
    """The two endpoints are genuinely different (they were conflated before)."""
    series = even_mode_added_mass_dimless(degree_max=60)
    ratio = series["rigid_lid_added_mass_dimless"] / 0.5
    assert 1.6 < ratio < 1.7


def test_solver_approaches_the_even_mode_value_at_zero_frequency() -> None:
    """Independent confirmation: the panel solver must converge to 0.831, not 1.0."""
    mesh = hemisphere_hull_mesh(3, radius=1.0)
    volume = (2.0 / 3.0) * math.pi
    result = solve_radiation_3d(mesh, 1.0e-3, modes=[2], density=1025.0)
    ratio = float(result.added_mass[0, 0]) / (1025.0 * volume)
    reference = even_mode_added_mass_dimless(degree_max=60)["rigid_lid_added_mass_dimless"]
    assert ratio == pytest.approx(reference, rel=1.0e-2)
    assert ratio < 0.9  # would fail if the report's 1.0 reference were right


def test_artifact_rows_recompute_and_match_the_documented_criteria(payload: dict) -> None:
    """Every row of the payload must be internally consistent (zero hand copies)."""
    volume = (2.0 / 3.0) * math.pi
    reference = mathematically_expected = payload["exact_reference"][
        "rigid_lid_added_mass_dimless"
    ]
    assert mathematically_expected == pytest.approx(0.8309541, abs=1e-6)
    finest = max(payload["mesh_levels"])
    for row in payload["rows"]:
        assert row["added_mass_dimless"] == pytest.approx(
            row["added_mass"] / (1025.0 * volume), rel=1e-12
        )
        assert row["rel_error_vs_rigid_lid_reference"] == pytest.approx(
            row["added_mass_dimless"] / reference - 1.0, rel=1e-12
        )
        assert row["rel_error_vs_report_15809_reference"] == pytest.approx(
            row["added_mass_dimless"] - 1.0, rel=1e-12
        )
        assert row["relative_residual"] < 1e-10
        if row["subdivisions"] == finest:
            assert abs(row["rel_error_vs_hulme_added_mass"]) < 0.03


def test_report_15809_number_is_reproduced_row_for_row(payload: dict) -> None:
    """The m-session report quoted 0.8564 at ka = 0.1 / 256 panels (14.4 % low)."""
    row = next(
        item
        for item in payload["rows"]
        if item["subdivisions"] == 3 and abs(item["ka"] - 0.1) < 1e-12
    )
    assert row["added_mass_dimless"] == pytest.approx(0.85636, abs=5e-5)
    assert row["rel_error_vs_report_15809_reference"] == pytest.approx(-0.1436, abs=1e-3)


def test_all_criteria_pass_and_summary_is_rendered(payload: dict) -> None:
    assert payload["verdict"] == "ALL PASS"
    assert all(crit["pass"] for crit in payload["criteria"].values())
    text = render_summary(payload)
    assert "REFUTED" not in text  # the summary states the correction, not a verdict word
    assert "0.830954" in text
    assert (ARTIFACTS / "figure.png").stat().st_size > 10_000


def test_study_is_reproducible_on_a_smaller_mesh() -> None:
    """A single-level rerun must give the same numbers as the committed rows."""
    fresh = run_study(mesh_levels=(2,), endpoint_ka=(0.1,), band_ka=(1.0,))
    committed = json.loads((ARTIFACTS / "results.json").read_text())
    for new_row in fresh["rows"]:
        old_row = next(
            row
            for row in committed["rows"]
            if row["subdivisions"] == 2 and abs(row["ka"] - new_row["ka"]) < 1e-12
        )
        assert new_row["added_mass_dimless"] == pytest.approx(
            old_row["added_mass_dimless"], rel=1e-10
        )
    assert np.isfinite(fresh["criteria"]["R1_even_mode_reference_refutes_rho_nabla"]["measured"])


def test_hulme_reference_itself_differs_from_rho_nabla_at_zero_frequency() -> None:
    """The repository's own Hulme series already contradicted the 1.0 reference."""
    hulme = hulme_hemisphere_radiation(1.0e-3, radius=1.0)
    assert 0.80 < hulme.heave_added_mass_dimless < 0.87