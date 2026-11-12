"""Gate for the sphere-BEM accuracy envelope (``CASE-ACOUSTIC-PULSE``).

The envelope driver must (a) reproduce the physics it claims -- refinement
reduces the error, wavenumber increases it, and the analytic reference is the
closed-form pulsating-sphere pressure -- and (b) publish only injected numbers.
Each assertion below fails if the driver silently stops measuring what it says
it measures.
"""

from __future__ import annotations

import math

import pytest

from tensorbem.benchmarks import acoustic3d_pulse_envelope as env

SMALL_KA = (1.0, 2.0)
SMALL_SUBDIVISIONS = (1, 2)


@pytest.fixture(scope="module")
def small_payload() -> dict:
    return env.run_study(SMALL_KA, SMALL_SUBDIVISIONS)


def test_error_falls_with_panel_refinement(small_payload: dict) -> None:
    """On every ka row, 320 panels must beat 80 panels (no false convergence)."""

    for entry in small_payload["per_ka"]:
        errors = [row["relative_error"] for row in entry["rows"]]
        assert errors[1] < errors[0], (entry["ka"], errors)
        assert errors[1] < 0.08, (entry["ka"], errors[1])


def test_error_grows_with_dimensionless_wavenumber(small_payload: dict) -> None:
    """At a fixed 80-panel mesh, ka = 2 is harder than ka = 1."""

    coarse = {
        entry["ka"]: entry["rows"][0]["relative_error"]
        for entry in small_payload["per_ka"]
    }
    assert coarse[2.0] > coarse[1.0]


def test_panels_per_wavelength_uses_the_area_equivalent_panel_size() -> None:
    mesh = env.icosphere(1, radius=env.RADIUS)
    h = math.sqrt(4.0 * math.pi * env.RADIUS**2 / mesh.panel_count)
    assert env.panel_size(env.RADIUS, mesh.panel_count) == pytest.approx(h)
    wavelength = 2.0 * math.pi * env.RADIUS / 2.0
    assert env.panels_per_wavelength(2.0, env.RADIUS, mesh.panel_count) == \
        pytest.approx(wavelength / h)


def test_coarsest_passing_selects_the_minimal_mesh_or_reports_none() -> None:
    rows = [
        {"panel_count": 80, "relative_error": 0.5, "panels_per_wavelength": 3.0},
        {"panel_count": 320, "relative_error": 0.06, "panels_per_wavelength": 6.0},
        {"panel_count": 1280, "relative_error": 0.01, "panels_per_wavelength": 12.0},
    ]
    assert env.coarsest_passing(rows, 0.08)["panel_count"] == 320
    assert env.coarsest_passing(rows, 0.03)["panel_count"] == 1280
    assert env.coarsest_passing(rows, 1.0e-6) is None


def test_requirements_are_recorded_for_every_threshold(small_payload: dict) -> None:
    for entry in small_payload["per_ka"]:
        thresholds = [item["threshold"] for item in entry["requirements"]]
        assert thresholds == list(env.THRESHOLDS)
        for item in entry["requirements"]:
            if item["unreachable_on_this_ladder"]:
                assert item["min_panel_count"] is None
            else:
                assert item["min_panels_per_wavelength"] > 0.0


def test_markdown_is_injected_from_the_payload(small_payload: dict) -> None:
    text = env.render_markdown(small_payload)
    assert text.startswith(env.DOC_ANCHOR)
    for row in small_payload["rows"]:
        assert f"{row['panel_count']}" in text
        assert f"{row['relative_error']:.3%}" in text
    assert "never retyped" in text


def test_driver_rejects_degenerate_inputs(tmp_path) -> None:
    with pytest.raises(ValueError):
        env.panel_size(0.0, 10)
    with pytest.raises(ValueError):
        env.panels_per_wavelength(0.0, 1.0, 80)
    with pytest.raises(ValueError):
        env.run_study((), SMALL_SUBDIVISIONS)
    with pytest.raises(ValueError):
        env.harmonic_error(-1.0, 1)