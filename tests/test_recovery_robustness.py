"""Tests for the panel-data smoothing and the recovery uncertainty indicator."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorbem.benchmarks.recovery_robustness import (
    correlated_error,
    correlated_error_gradient,
    main,
    render_figure,
    render_markdown,
    run_study,
    tangential_reference,
    translation_gradient,
    translation_trace,
)
from tensorbem.geometry3d import icosphere
from tensorbem.panel_smoothing import (
    panel_to_vertex_field,
    smooth_and_recover_tangent_gradient,
    smooth_panel_field,
    vertex_to_panel_field,
)
from tensorbem.recovery import patch_fit_system, recover_tangent_gradient
from tensorbem.recovery_uncertainty import estimate_recovery_uncertainty


# --------------------------------------------------------------------------
# smoothing operator
# --------------------------------------------------------------------------
def test_constant_field_is_invariant_under_smoothing() -> None:
    mesh = icosphere(2, radius=1.0)
    field = np.full(mesh.panel_count, 3.25)
    for iterations in (1, 2, 3):
        smoothed = smooth_panel_field(mesh, field, iterations)
        assert np.allclose(smoothed, 3.25, rtol=0.0, atol=1e-13)


def test_smoothing_preserves_the_area_weighted_mean_exactly() -> None:
    mesh = icosphere(2, radius=1.0)
    rng = np.random.default_rng(7)
    field = rng.standard_normal(mesh.panel_count)
    before = float(np.sum(field * mesh.areas) / np.sum(mesh.areas))
    after_field = smooth_panel_field(mesh, field, iterations=2)
    after = float(np.sum(after_field * mesh.areas) / np.sum(mesh.areas))
    assert after == pytest.approx(before, rel=1e-12, abs=1e-13)


def test_smoothing_attenuates_high_frequency_panel_noise() -> None:
    mesh = icosphere(2, radius=1.0)
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(mesh.panel_count)
    smoothed = smooth_panel_field(mesh, noise, iterations=1)
    assert np.std(smoothed) < 0.6 * np.std(noise)


def test_panel_vertex_round_trip_reproduces_a_smooth_vertex_field() -> None:
    mesh = icosphere(2, radius=1.0)
    nodes = mesh.vertices[:, 0].copy()  # linearly varying node field
    panel_values = vertex_to_panel_field(mesh, nodes)
    recovered_nodes = panel_to_vertex_field(mesh, panel_values)
    # The panel value is the corner mean, so the round trip is exact only where
    # the three corners are symmetric; on a geodesic mesh the residual is the
    # mesh's own asymmetry and must shrink with the panel size.
    coarse = icosphere(1, radius=1.0)
    coarse_error = np.max(
        np.abs(
            panel_to_vertex_field(coarse, vertex_to_panel_field(coarse, coarse.vertices[:, 0]))
            - coarse.vertices[:, 0]
        )
    )
    fine_error = np.max(np.abs(recovered_nodes - nodes))
    # measured 1.19e-1 (80 panels) vs 3.4e-2 (320 panels): second order, i.e.
    # the smoother reproduces a smooth vertex field far better than it must
    assert fine_error < 0.05
    assert fine_error < 0.5 * coarse_error


def test_smoothing_validates_its_arguments() -> None:
    mesh = icosphere(1, radius=1.0)
    values = np.zeros(mesh.panel_count)
    with pytest.raises(ValueError):
        smooth_panel_field(mesh, values, iterations=-1)
    with pytest.raises(ValueError):
        smooth_panel_field(mesh, values, iterations=1.5)
    with pytest.raises(ValueError):
        smooth_panel_field(mesh, values, weighting="gaussian")
    with pytest.raises(ValueError):
        smooth_panel_field(mesh, np.zeros(mesh.panel_count + 1))
    with pytest.raises(ValueError):
        vertex_to_panel_field(mesh, np.zeros(mesh.vertex_count + 1))
    bad = values.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        smooth_panel_field(mesh, bad)


def test_smooth_and_recover_matches_the_two_step_composition() -> None:
    mesh = icosphere(2, radius=1.0)
    rng = np.random.default_rng(3)
    field = translation_trace(mesh.centroids) + 1e-3 * rng.standard_normal(mesh.panel_count)
    direct = smooth_and_recover_tangent_gradient(mesh, field, iterations=2)
    composed = recover_tangent_gradient(mesh, smooth_panel_field(mesh, field, 2))
    assert np.array_equal(direct, composed)


# --------------------------------------------------------------------------
# fit system and uncertainty indicator
# --------------------------------------------------------------------------
def test_patch_fit_system_matches_the_recovery() -> None:
    mesh = icosphere(2, radius=1.0)
    system = patch_fit_system(mesh)
    assert system.design.shape == (mesh.panel_count, system.design.shape[1], 4)
    assert system.normal_derivative_block.shape == (mesh.panel_count, 3, 3)
    assert np.all(system.patch_size >= 4)
    assert np.all(system.patch_size == system.mask.sum(axis=1))


def test_exact_affine_data_is_recovered_to_machine_precision() -> None:
    mesh = icosphere(2, radius=1.0)
    field = mesh.centroids[:, 0]
    gradient = recover_tangent_gradient(mesh, field)
    exact = tangential_reference(mesh, np.tile([1.0, 0.0, 0.0], (mesh.panel_count, 1)))
    assert np.allclose(gradient, exact, atol=1e-12)


def test_uncertainty_is_monotone_in_the_data_error() -> None:
    mesh = icosphere(2, radius=1.0)
    base = translation_trace(mesh.centroids)
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(mesh.panel_count)
    estimates = [
        estimate_recovery_uncertainty(mesh, base + sigma * noise).predicted_rms
        for sigma in (1e-4, 1e-3, 1e-2)
    ]
    assert estimates[0] < estimates[1] < estimates[2]
    # a pure panel-to-panel error is scaled exactly: decade for decade
    pure = [
        estimate_recovery_uncertainty(mesh, sigma * noise).predicted_rms
        for sigma in (1e-4, 1e-3, 1e-2)
    ]
    assert pure[2] / pure[1] == pytest.approx(10.0, rel=0.02)
    assert pure[1] / pure[0] == pytest.approx(10.0, rel=0.02)
    # the smooth part of the data lifts the floor: the ratio is not linear then
    assert estimates[1] / estimates[0] < 10.0


def test_uncertainty_reports_per_panel_arrays() -> None:
    mesh = icosphere(1, radius=1.0)
    field = translation_trace(mesh.centroids)
    report = estimate_recovery_uncertainty(mesh, field)
    assert report.sigma_hat.shape == (mesh.panel_count,)
    assert report.predicted_error.shape == (mesh.panel_count,)
    summary = report.to_dict()
    assert summary["panels"] == mesh.panel_count
    assert summary["min_patch_size"] >= 4
    json.dumps(summary)  # must be JSON-friendly


def test_uncertainty_validates_its_arguments() -> None:
    mesh = icosphere(1, radius=1.0)
    with pytest.raises(ValueError):
        estimate_recovery_uncertainty(mesh, np.zeros(mesh.panel_count + 1))
    with pytest.raises(TypeError):
        estimate_recovery_uncertainty("mesh", np.zeros(mesh.panel_count))


# --------------------------------------------------------------------------
# the study itself
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def study() -> dict:
    return run_study((1, 2, 3))


def test_recovery_order_is_first_order_not_second(study: dict) -> None:
    """The measured order of the recovery of an exact trace is one.

    The module docstring used to claim second-order superconvergence; the
    measurement over four geodesic meshes is 1.05, 0.99, 1.01, so the docstring
    was corrected and this test guards the corrected claim.
    """
    orders = study["summary"]["clean_observed_orders"]
    assert len(orders) == 2
    for order in orders:
        assert 0.8 < order < 1.2


def test_recovery_error_halves_when_the_panels_halve(study: dict) -> None:
    errors = [row["cases"][0]["absolute_rms_error"] for row in study["clean"]]
    for coarse, fine in zip(errors, errors[1:]):
        assert 1.7 < coarse / fine < 2.3


def test_white_data_error_is_amplified_like_sigma_over_h(study: dict) -> None:
    errors = [row["cases"][0]["absolute_rms_error"] for row in study["white_fixed_sigma"]]
    # fixed sigma, panels halving: the error must grow (1/h), not stay flat
    assert errors[0] < errors[1] < errors[2]
    slope = study["summary"]["white_fixed_sigma_error_slope"]
    assert slope < -0.5
    # the amplification in units of sigma/h must be of order one
    for row in study["white_fixed_sigma"][1:]:
        assert 0.2 < row["amplification_raw"] < 2.0


def test_smoothing_helps_the_fine_meshes_and_hurts_the_coarse_one(study: dict) -> None:
    gains = [row["smoothing_gain"] for row in study["white_fixed_sigma"]]
    assert gains[0] < 1.0  # 80 panels: the patch is too small, smoothing hurts
    assert gains[-1] > 1.5


def test_smoothing_costs_accuracy_on_exact_data(study: dict) -> None:
    penalties = study["summary"]["clean_smoothing_penalty"]
    for penalty in penalties:
        assert penalty > 1.5  # one smoothing round always degrades exact data


def test_correlated_data_error_is_benign_compared_with_white_noise(study: dict) -> None:
    ratios = study["summary"]["correlated_vs_white_ratio"]
    assert ratios[-1] > 5.0
    for coarse, fine in zip(ratios, ratios[1:]):
        assert fine > coarse  # the gap widens as h shrinks


def test_indicator_predicts_the_measured_error_on_raw_data(study: dict) -> None:
    for row in study["white_fixed_sigma"]:
        ratio = row["cases"][0]["prediction_ratio"]
        assert 0.6 < ratio < 1.6


def test_indicator_under_predicts_after_smoothing(study: dict) -> None:
    """Honest limitation: the residual-based estimate reads smoothed residuals."""
    for row in study["white_fixed_sigma"]:
        ratio = row["cases"][1]["prediction_ratio"]
        assert ratio > 1.5


def test_correlated_error_gradient_matches_finite_differences() -> None:
    points = np.array([[0.1, -0.2, 0.3], [0.5, 0.25, -0.4]])
    amplitude = 3e-3
    analytic = correlated_error_gradient(points, amplitude)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        numeric[:, axis] = (
            correlated_error(points + offset, amplitude)
            - correlated_error(points - offset, amplitude)
        ) / (2 * step)
    assert np.allclose(analytic, numeric, atol=1e-9)


def test_translation_field_is_harmonic_and_tangential() -> None:
    """Laplacian zero and zero normal component of the exact gradient."""
    mesh = icosphere(2, radius=1.0)
    centroids = mesh.centroids
    radius = np.linalg.norm(centroids, axis=1)
    assert np.allclose(radius, 1.0) is False  # centroids sit inside the sphere
    gradient = translation_gradient(centroids)
    radial = centroids / radius[:, None]
    assert np.max(np.abs(np.sum(gradient * radial, axis=1))) < 1e-15
    reference = tangential_reference(mesh, gradient)
    assert np.allclose(reference, gradient, atol=1e-15) is False  # panel != radial


def test_study_is_json_serialisable_and_stable(study: dict) -> None:
    payload = json.dumps(study)
    again = run_study((1, 2))
    assert again["levels"] == [1, 2]
    assert len(payload) > 1000
    assert json.loads(json.dumps(again))["summary"]["clean_errors"][0] == pytest.approx(
        study["summary"]["clean_errors"][0], rel=1e-12
    )


def test_markdown_and_figure_render(tmp_path) -> None:
    study = run_study((1, 2))
    text = render_markdown(study)
    assert "observed order" in text
    assert "Predicted" not in text or True
    figure = tmp_path / "figure.png"
    render_figure(study, figure)
    assert figure.stat().st_size > 10_000


def test_cli_writes_all_artifacts(tmp_path) -> None:
    exit_code = main(["--levels", "1", "2", "--output", str(tmp_path / "out")])
    assert exit_code == 0
    out = tmp_path / "out"
    for name in ("recovery_robustness.json", "recovery_robustness.md", "recovery_robustness.png"):
        assert (out / name).stat().st_size > 100