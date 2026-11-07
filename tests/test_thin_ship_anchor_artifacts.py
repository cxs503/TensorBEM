"""The committed verification anchors must stay reproducible and honest.

``benchmarks/verified/{bem_wigley_hydrostatics,bem_ellipsoid_wave,
bem_multihull_interference}`` hold the acceptance record of the thin-ship
anchors: ``results.json`` (every number, produced by
:mod:`tensorbem.benchmarks.thin_ship_verification`), ``report.md`` and a figure.
This test re-derives a representative number of each study from the library and
compares it with the committed record, so the artifacts cannot drift, and it
checks that the acceptance flags in the record are consistent with the
thresholds the campaign node asked for.
"""

import json
import pathlib

import numpy as np
import pytest

from tensorbem.benchmarks.thin_ship_verification import (
    ELLIPSOID_B,
    ELLIPSOID_L,
    ELLIPSOID_T,
    FROUDE_NUMBERS,
    TOLERANCE,
    WIGLEY_B,
    WIGLEY_L,
    WIGLEY_T,
    run_wigley_hydrostatics,
)
from tensorbem.hydrostatic import compute_displaced_volume_and_buoyancy
from tensorbem.thin_ship_michell import (
    EllipsoidHull,
    WigleyHull,
    ellipsoid_amplitude_function,
    michell_wave_resistance,
    tandem_amplitude,
    wave_resistance_from_amplitude,
    wigley_amplitude_function,
    wigley_panels,
)
from tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "benchmarks" / "verified"
STUDIES = ("bem_wigley_hydrostatics", "bem_ellipsoid_wave", "bem_multihull_interference")
G = 9.80665


def _load(study: str) -> dict:
    path = VERIFIED / study / "results.json"
    if not path.exists():  # pragma: no cover - missing artifact is a hard failure
        pytest.fail(f"missing verification anchor artifact {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("study", STUDIES)
def test_artifact_set_is_complete(study: str) -> None:
    payload = _load(study)
    directory = VERIFIED / study
    assert (directory / "report.md").is_file()
    assert (directory / "figures" / f"{study}.png").is_file()
    assert payload["study"] == study
    assert payload["artifacts"]["report_md"].endswith("report.md")
    # the report is the paper-level summary: it must carry the verdict
    report = (directory / "report.md").read_text()
    assert "Acceptance" in report


@pytest.mark.parametrize("study", STUDIES)
def test_all_criteria_passed(study: str) -> None:
    payload = _load(study)
    assert payload["all_criteria_passed"] is True
    if "criteria" in payload:
        for name, item in payload["criteria"].items():
            assert item["passed"] is True, name
    for case in payload.get("cases", []):
        for name, item in case.get("criteria", {}).items():
            assert item["passed"] is True, (case.get("froude_number"), name)


@pytest.mark.parametrize("study", STUDIES)
def test_thresholds_match_the_campaign_node(study: str) -> None:
    payload = _load(study)
    limits = set()
    if "criteria" in payload:
        limits.update(item["limit"] for item in payload["criteria"].values())
    for case in payload.get("cases", []):
        limits.update(item["limit"] for item in case.get("criteria", {}).values())
    # the far-field trend is a trend band (10 %), not a discretisation tolerance
    accepted = set(TOLERANCE.values()) | {1.0e-9, 1.0e-6, 1.0e-3, 0.1}
    assert limits <= accepted


def test_hydrostatic_record_reproduces() -> None:
    payload = _load("bem_wigley_hydrostatics")
    production = payload["production_mesh"]
    hull = wigley_wetted_hull_mesh(
        WIGLEY_L, WIGLEY_B, WIGLEY_T, production["n_sections"], production["n_levels"]
    )
    volume, centre = compute_displaced_volume_and_buoyancy(hull)
    row = next(
        item
        for item in payload["levels"]
        if item["n_sections"] == production["n_sections"]
        and item["n_levels"] == production["n_levels"]
    )
    assert volume == pytest.approx(row["displaced_volume"], rel=1.0e-10)
    assert centre[2] + WIGLEY_T == pytest.approx(row["KB"], rel=1.0e-10)
    assert row["rel_error_Cb"] <= TOLERANCE["Cb"]
    assert row["rel_error_volume"] <= TOLERANCE["volume"]
    assert row["rel_error_wetted_area"] <= TOLERANCE["wetted_area"]


def test_hydrostatic_runner_is_deterministic() -> None:
    first = run_wigley_hydrostatics(refinements=((16, 8), (32, 16), (64, 32)))
    second = run_wigley_hydrostatics(refinements=((16, 8), (32, 16), (64, 32)))
    assert first["levels"] == second["levels"]


def test_ellipsoid_reference_reproduces() -> None:
    payload = _load("bem_ellipsoid_wave")
    hull = EllipsoidHull(ELLIPSOID_L, ELLIPSOID_B, ELLIPSOID_T)
    amplitude = ellipsoid_amplitude_function(hull)
    case = payload["cases"][1]  # Fr = 0.30
    assert case["froude_number"] == FROUDE_NUMBERS[1]
    speed = case["froude_number"] * (G * ELLIPSOID_L) ** 0.5
    method = payload["method"]
    reference = michell_wave_resistance(
        amplitude,
        speed,
        lam_max=method["lam_max"],
        n_lam=method["n_lam"],
        extrapolate_tail=False,
    )
    assert reference["Rw"] == pytest.approx(case["Rw_reference"], rel=1.0e-6)
    assert case["levels"][-1]["rel_error_vs_reference"] <= TOLERANCE["wave_resistance"]


def test_multihull_record_reproduces() -> None:
    payload = _load("bem_multihull_interference")
    hull = WigleyHull(WIGLEY_L, WIGLEY_B, WIGLEY_T)
    amplitude = wigley_amplitude_function(hull)
    panels = wigley_panels(hull, payload["panel_grid"]["n_x"], payload["panel_grid"]["n_z"])
    case = payload["cases"][0]
    speed = case["froude_number"] * (G * WIGLEY_L) ** 0.5
    distance = case["centre_distance_over_L"] * WIGLEY_L
    single = michell_wave_resistance(
        amplitude, speed, lam_max=40.0, n_lam=300, extrapolate_tail=False
    )["Rw"]
    pair = michell_wave_resistance(
        tandem_amplitude(amplitude, distance),
        speed,
        lam_max=40.0,
        n_lam=300,
        extrapolate_tail=False,
    )["Rw"]
    assert pair / (2.0 * single) == pytest.approx(case["sigma_reference"], rel=1.0e-6)

    def pair_panels(k, alpha):
        return panels.amplitude(k, alpha) + panels.amplitude(k, alpha, x_offset=distance)

    single_panel = wave_resistance_from_amplitude(panels.amplitude, speed, lam_max=40.0, n_lam=300)
    pair_panel = wave_resistance_from_amplitude(pair_panels, speed, lam_max=40.0, n_lam=300)
    assert pair_panel / (2.0 * single_panel) == pytest.approx(case["sigma_panel"], rel=1.0e-6)
    assert case["sigma_abs_difference"] <= TOLERANCE["sigma_absolute"]


def test_multihull_limits_are_analytic() -> None:
    payload = _load("bem_multihull_interference")
    limits = payload["limits"]
    assert limits["sigma_coincident"] == pytest.approx(2.0, abs=1.0e-6)
    assert abs(limits["sigma_far_60L"] - 1.0) <= 0.1
    sweep = payload["spacing_sweep"]["points"]
    gaps = [point["gap_over_L"] for point in sweep]
    assert gaps == sorted(gaps)
    assert np.all([point["sigma_reference"] > 0.0 for point in sweep])