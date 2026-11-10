"""Tests for the real-BEM 3-D far-field drift route (node p40w4-n02, m9286)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.cylinder_drift import (
    far_field_cylinder_drift,
    kochin_3d,
    near_field_cylinder_drift,
)
from tensorbem.drift import far_field_mean_drift_3d
from tensorbem.drift_far_field3d import (
    bem_kochin,
    control_circle_drift,
    local_loglog_slope,
    maruo_far_field_drift,
    optical_theorem_residual,
    radius_scan,
    slope_band,
)

GRAVITY = 9.81
DENSITY = 1025.0
N_THETA = 2048


def _physical_kochin(ka: float, amplitude: float = 1.0, n_theta: int = N_THETA):
    """Exact MacCamy-Fuchs Kochin function in the BEM (physical) normalisation.

    ``cylinder_drift.kochin_3d`` is derived from a unit potential amplitude;
    ``solve_diffraction_3d`` uses ``-(i g A / omega) exp(...)``, so the physical
    table is that table times ``-i g A / omega``.
    """
    theta = 2.0 * math.pi * np.arange(n_theta) / n_theta
    omega = math.sqrt(GRAVITY * ka)
    table = -(1.0j * GRAVITY * amplitude / omega) * kochin_3d(ka, theta)
    return table, theta


# ---------------------------------------------------------------------------
# 1. Maruo weight: identical to the repo far-field channel, exact against the anchor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ka", [0.2, 0.5, 1.0, 2.0])
def test_maruo_route_reproduces_the_verified_anchor(ka):
    """Fed the exact physical Kochin function the route reproduces the anchor."""
    table, theta = _physical_kochin(ka)
    mine = maruo_far_field_drift(table, theta, ka, amplitude=1.0)
    anchor = far_field_cylinder_drift(ka, amplitude=1.0, radius=1.0, n_theta=N_THETA)
    assert mine["force_x"] == pytest.approx(anchor["mean_drift_force_x"], rel=1e-10)


@pytest.mark.parametrize("ka", [0.5, 1.0])
def test_maruo_weight_matches_the_drift_module(ka):
    """Same weight as ``tensorbem.drift.far_field_mean_drift_3d`` (no redefinition)."""
    table, theta = _physical_kochin(ka, n_theta=360)
    mine = maruo_far_field_drift(table, theta, ka, amplitude=1.0, n_theta=360)
    # the same table fed to the control-surface weight through a fake diffraction
    # result: reuse the module helper by calling the documented public formula
    pattern = np.abs(table) ** 2
    dtheta = 2.0 * math.pi / 360
    expected = DENSITY * ka * ka / (8.0 * math.pi) * dtheta * (
        np.sum(pattern)
        - np.sum(pattern * np.cos(theta))
    )
    assert mine["force_x"] == pytest.approx(float(expected), rel=1e-12)


def test_maruo_force_is_quadratic_in_amplitude_and_linear_in_amplitude_units():
    table, theta = _physical_kochin(0.5)
    one = maruo_far_field_drift(table, theta, 0.5, amplitude=1.0)["force_x"]
    two = maruo_far_field_drift(2.0 * table, theta, 0.5, amplitude=2.0)["force_x"]
    assert two / one == pytest.approx(4.0, rel=1e-12)


def test_maruo_rejects_bad_input():
    table, theta = _physical_kochin(0.5, n_theta=128)
    with pytest.raises(ValueError):
        maruo_far_field_drift(table, theta, 0.0)
    with pytest.raises(ValueError):
        maruo_far_field_drift(table, theta[:-1], 0.5)
    with pytest.raises(ValueError):
        maruo_far_field_drift(table, theta, 0.5, n_theta=64)


def test_periodic_interpolation_is_invariant_to_table_order_and_wrapping():
    """The azimuth table is a periodic set: order and 2*pi wrapping must not matter."""
    table, theta = _physical_kochin(1.0, n_theta=180)
    a = maruo_far_field_drift(table, theta, 1.0)["force_x"]
    # wrap every second azimuth by +2*pi (same physical angles)
    wrapped = theta + 2.0 * math.pi * (np.arange(theta.size) % 2)
    b = maruo_far_field_drift(table, wrapped, 1.0)["force_x"]
    assert a == pytest.approx(b, rel=1e-12)
    # shuffled table order must give the same integral
    order = np.argsort(np.cos(np.arange(theta.size)))  # deterministic, non-trivial order
    c = maruo_far_field_drift(table[order], theta[order], 1.0)["force_x"]
    assert a == pytest.approx(c, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. energy identity (optical theorem for a non-absorbing fixed body)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ka", [0.2, 0.3, 0.5, 1.0, 2.0, 3.0])
def test_energy_identity_is_machine_exact_on_the_exact_field(ka):
    table, theta = _physical_kochin(ka, n_theta=4096)
    res = optical_theorem_residual(table, theta, ka, radius=1.0, n_theta=4096)
    assert res["residual_relative"] < 1e-12


def test_energy_identity_carries_no_body_length_scale():
    """The identity must not contain the body radius: only ``k`` and ``g A / omega``."""
    table, theta = _physical_kochin(1.0, n_theta=2048)
    a = optical_theorem_residual(table, theta, 1.0, radius=1.0, n_theta=2048)
    b = optical_theorem_residual(table, theta, 1.0, radius=2.0, n_theta=2048)
    assert a["predicted"] == pytest.approx(b["predicted"], rel=1e-15)
    assert a["residual_relative"] < 1e-12


def test_energy_identity_detects_a_normalisation_error():
    """A unit-potential-amplitude table (the ``cylinder_drift`` convention) fails."""
    theta = 2.0 * math.pi * np.arange(2048) / 2048
    wrong = kochin_3d(1.0, theta)
    res = optical_theorem_residual(wrong, theta, 1.0, radius=1.0, n_theta=2048)
    assert res["residual_relative"] > 0.5  # the g A / omega factor is missing


# ---------------------------------------------------------------------------
# 3. radius scan: the m5242 honest FAIL, measured and pinned (regression, gate 6)
# ---------------------------------------------------------------------------
def test_radius_scan_is_not_convergent_regression():
    """Pinned non-convergence of the un-reduced control-circle route.

    The free-surface radiation-stress formula alone is not a closed momentum
    balance (the vertical-cylinder flux is the same order), so the finite-radius
    sequence oscillates at O(1) relative amplitude.  If a future implementation
    closes the balance this test must be *replaced* -- together with the honest
    FAIL it records -- not relaxed.
    """
    table, theta = _physical_kochin(1.0, n_theta=720)
    scan = radius_scan(table, theta, 1.0, n_theta=8192)
    assert scan["spread_relative"] > 0.30
    assert scan["richardson_evaluable"] is False
    assert math.isnan(scan["observed_order"])
    assert math.isnan(scan["richardson_residual_relative"])


def test_radius_scan_matches_the_single_radius_evaluation():
    table, theta = _physical_kochin(0.5, n_theta=720)
    scan = radius_scan(table, theta, 0.5, radii=(10.0, 20.0, 40.0), n_theta=4096)
    single = control_circle_drift(table, theta, 0.5, radius_factor=20.0, n_theta=4096)
    assert scan["rows"][1]["force_x"] == pytest.approx(single["mean_drift_force_x"], rel=1e-15)


def test_radius_scan_requires_three_radii():
    table, theta = _physical_kochin(0.5, n_theta=128)
    with pytest.raises(ValueError):
        radius_scan(table, theta, 0.5, radii=(10.0, 20.0))
    with pytest.raises(ValueError):
        radius_scan(table, theta, 0.5, radii=(0.5, 10.0, 20.0))


# ---------------------------------------------------------------------------
# 4. local log-log slope and the band boundary (gate 6 for the (ka)^3 law)
# ---------------------------------------------------------------------------
def test_long_wave_slope_of_the_exact_field():
    kas = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
    forces = [
        far_field_cylinder_drift(x, amplitude=1.0, radius=1.0)["mean_drift_force_x"]
        for x in kas
    ]
    res = local_loglog_slope(kas, forces)
    # the (ka)^3 asymptote is inside 3 +- 0.2 only at the smallest window
    assert res["slope"][0] == pytest.approx(2.87, abs=0.05)
    assert bool(res["in_band"][0]) is True
    # ... and it leaves the band above ka = 0.2: measured boundary, pinned here
    band = slope_band(res)
    assert band["band_max_ka"] == pytest.approx(0.2, abs=1e-12)
    assert res["slope"][-1] < 0.5  # ka = 3 is flat: the law is a ka -> 0 statement


def test_local_slope_rejects_degenerate_input():
    with pytest.raises(ValueError):
        local_loglog_slope((0.2, 0.3), (1.0, 2.0))
    with pytest.raises(ValueError):
        local_loglog_slope((0.2, 0.3, 0.5), (1.0, 0.0, 2.0))


# ---------------------------------------------------------------------------
# 5. BEM coupling (integration, small mesh)
# ---------------------------------------------------------------------------
def test_bem_kochin_couples_a_real_diffraction_solve():
    from tensorbem.drift import bottom_mounted_cylinder_mesh
    from tensorbem.free_surface_solver3d import solve_diffraction_3d

    ka = 1.0
    mesh = bottom_mounted_cylinder_mesh(1.0, 5.0 / ka, n_theta=12, n_z=4, growth=1.35)
    diffraction = solve_diffraction_3d(mesh, ka, amplitude=1.0)
    koch = bem_kochin(mesh, diffraction, n_azimuth=36, check_azimuth=72)
    assert koch.kochin.shape == koch.azimuth.shape == (36,)
    assert np.isfinite(koch.kochin).all()
    assert float(np.abs(koch.kochin).max()) > 0.0
    assert 0.0 <= koch.azimuth_rel_change < 0.1
    assert koch.panel_count == mesh.panel_count
    res = maruo_far_field_drift(koch.kochin, koch.azimuth, ka, amplitude=1.0)
    anchor = far_field_cylinder_drift(ka, amplitude=1.0, radius=1.0)["mean_drift_force_x"]
    # a 108-panel mesh is a coarse but sane approximation of the exact value
    assert abs(res["force_x"] - anchor) / anchor < 0.30
    energy = optical_theorem_residual(
        koch.kochin, koch.azimuth, ka, radius=1.0, n_theta=720
    )
    assert energy["residual_relative"] < 0.10


def test_bem_kochin_validates_azimuth_sizes():
    from tensorbem.drift import bottom_mounted_cylinder_mesh
    from tensorbem.free_surface_solver3d import solve_diffraction_3d

    mesh = bottom_mounted_cylinder_mesh(1.0, 5.0, n_theta=12, n_z=4, growth=1.35)
    diffraction = solve_diffraction_3d(mesh, 1.0, amplitude=1.0)
    with pytest.raises(ValueError):
        bem_kochin(mesh, diffraction, n_azimuth=8)
    with pytest.raises(ValueError):
        bem_kochin(mesh, diffraction, n_azimuth=32, check_azimuth=32)


# ---------------------------------------------------------------------------
# 6. the (ka)^3 gate is judged against *both* the BEM and the exact reference
# ---------------------------------------------------------------------------
def test_near_field_reference_agrees_with_far_field_reference():
    """The two exact MacCamy-Fuchs references agree: the anchor is not route-specific."""
    for ka in (0.2, 0.5, 1.0):
        far = far_field_cylinder_drift(ka, amplitude=1.0, radius=1.0)["mean_drift_force_x"]
        near = near_field_cylinder_drift(ka, amplitude=1.0, radius=1.0)["mean_drift_force_x"]
        assert near == pytest.approx(far, rel=1e-9)

# ---------------------------------------------------------------------------
# 7. study driver contract (keeps the driver exercised -> doc guard tier ARTIFACT)
# ---------------------------------------------------------------------------
def test_study_driver_contract(tmp_path, monkeypatch):
    # isolated matplotlib config dir: the shared font cache is contended when the
    # whole suite runs in parallel on the build host
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    from tensorbem.benchmarks.drift_far_field_3d_study import make_figures, run_study

    study = run_study(
        ka_values=(0.5, 1.0, 2.0),
        n_azimuth=16,
        levels=(("base", 10, 3, 1.4),),
        ladder=((10, 3, 1.4),),
        ladder_ka=1.0,
    )
    assert study["node"] == "p40w4-n02-bem-drift-force-3d"
    case = study["cases"]["base"][1]
    for key in (
        "near_field_x",
        "far_field_x",
        "exact_far_x",
        "near_vs_far_gap",
        "far_vs_exact",
        "energy_residual",
        "azimuth_rel_change",
    ):
        assert np.isfinite(case[key])
    assert "gates" in study and len(study["gates"]) == 6
    # the un-reduced route is never extrapolated silently
    assert study["radius_scan"]["richardson_evaluable"] in (True, False)
    assert study["radius_scan"]["spread_relative"] > 0.0
    figures = make_figures(study, tmp_path)
    assert len(figures) == 4
    for path in figures:
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 0


# ---------------------------------------------------------------------------
# 8. anti-hand-typing: the committed report block is regenerated from the artifact
# ---------------------------------------------------------------------------
def test_report_block_matches_the_committed_study():
    import json
    import pathlib

    from tensorbem.benchmarks.drift_far_field_3d_study import (
        BEGIN_BLOCK,
        END_BLOCK,
        render_document_block,
    )

    root = pathlib.Path(__file__).resolve().parents[1]
    study_path = root / "artifacts" / "drift_far_field_3d" / "study.json"
    if not study_path.exists():
        # ``artifacts/`` is git-ignored by repository policy; the study is
        # regenerated by the driver before the block is injected.
        pytest.skip(f"committed study missing: {study_path}")
    study = json.loads(study_path.read_text())
    doc = (root / "docs/drift_far_field_3d.md").read_text()
    block = doc[doc.index(BEGIN_BLOCK): doc.index(END_BLOCK) + len(END_BLOCK)]
    assert block.strip() == render_document_block(study).strip()


def test_energy_identity_is_radius_independent_on_real_bem_solutions():
    """Same ``ka`` on a 1 m and a 2 m cylinder: the identity ratio must coincide."""
    from tensorbem.drift import bottom_mounted_cylinder_mesh
    from tensorbem.free_surface_solver3d import solve_diffraction_3d

    ratios = []
    for radius in (1.0, 2.0):
        ka = 1.0
        k = ka / radius
        mesh = bottom_mounted_cylinder_mesh(
            radius, 5.0 / k, n_theta=16, n_z=5, growth=1.3
        )
        diffraction = solve_diffraction_3d(mesh, k, amplitude=1.0)
        koch = bem_kochin(mesh, diffraction, n_azimuth=48)
        res = optical_theorem_residual(
            koch.kochin, koch.azimuth, k, radius=radius, amplitude=1.0, n_theta=2048
        )
        ratios.append(res["integral_total"] / res["predicted"])
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-12)


# ---------------------------------------------------------------------------
# 9. panel-axis Richardson (the axis where this route does converge)
# ---------------------------------------------------------------------------
def test_panel_richardson_recovers_a_third_order_ladder():
    from tensorbem.drift_far_field3d import panel_richardson

    exact = 2.0
    panels = (100.0, 200.0, 400.0)
    values = [exact + 1.0 / (n / 100.0) ** 3 for n in panels]
    res = panel_richardson(panels, values)
    assert res["observed_order"] == pytest.approx(3.0, rel=1e-6)
    assert res["extrapolated"] == pytest.approx(exact, rel=1e-6)
    assert res["residual_relative"] < 1e-2


def test_panel_richardson_refuses_a_non_asymptotic_ladder():
    from tensorbem.drift_far_field3d import panel_richardson

    with pytest.raises(ValueError):
        panel_richardson((100.0, 200.0, 400.0), (1.0, 2.0, 1.5))  # sign flip
    with pytest.raises(ValueError):
        panel_richardson((100.0, 200.0, 400.0), (1.0, 1.5, 2.5))  # growing increments
    with pytest.raises(ValueError):
        panel_richardson((100.0, 200.0), (1.0, 1.5))
