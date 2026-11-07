"""Tests for :mod:`tensorbem.drift_force` (m5242).

Pins the three-way verification of the second-order mean drift force on the 2-D
vertical-wall anchor (``Fbar = rho*g*a**2/2``).  The 3-D Maruo far-field route is
pinned as a *documented limitation*: its un-reduced flux integral is not
converged in the radiation radius, which is why the campaign's ``(ka)^3``
asymptotic gate is reported as an honest FAIL.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.drift_force import (
    decompose_traveling_waves_2d,
    maruo_far_field_drift_3d,
    near_field_drift_wall,
    near_field_surface_waterline_2d,
    richardson_extrapolate,
    two_d_far_field_drift,
    wall_first_order_field,
    wall_mean_drift_analytic,
)

A = 1.0
OMEGA = 1.0


def test_analytic_anchor_value():
    anchor = wall_mean_drift_analytic(A, 1025.0, 9.81)
    assert anchor == pytest.approx(0.5 * 1025.0 * 9.81 * A * A, rel=1e-15)
    with pytest.raises(ValueError):
        wall_mean_drift_analytic(0.0)


def test_wall_field_satisfies_free_surface_and_wall_conditions():
    field = wall_first_order_field(A, OMEGA)
    times = np.linspace(0.0, field.period, 17)[:-1]
    # no normal flow through the wall
    for t in times:
        assert abs(float(field.dphi_dx(0.0, -0.5, t))) < 1e-12
    # kinematic free-surface condition at z = 0, x = 0: zeta = -(1/g) phi_t
    for t in times:
        zeta = float(field.elevation(0.0, t))
        assert zeta == pytest.approx(-float(field.phi_t(0.0, 0.0, t)) / 9.81, rel=1e-13)
    # antinode at the wall: zeta = 2a cos(omega t)
    for t in times:
        assert float(field.elevation(0.0, t)) == pytest.approx(
            2.0 * A * math.cos(OMEGA * t), abs=1e-12
        )


def test_near_field_matches_anchor():
    """Route 1: second-order pressure integral on the instantaneous surface."""
    anchor = wall_mean_drift_analytic(A)
    force = near_field_drift_wall(A, OMEGA, n_panels=200, n_time=64)
    assert abs(force / anchor - 1.0) < 0.01
    fine = near_field_drift_wall(A, OMEGA, n_panels=1600, n_time=64)
    assert abs(fine / anchor - 1.0) < 3e-3
    assert fine == pytest.approx(5026.65, rel=2e-4)


def test_surface_waterline_form_matches_anchor():
    """Same field, textbook surface + waterline splitting (algebraic check)."""
    anchor = wall_mean_drift_analytic(A)
    parts = near_field_surface_waterline_2d(A, OMEGA)
    assert parts["surface_term"] == pytest.approx(-0.5 * 1025.0 * 9.81, rel=1e-5)
    assert parts["waterline_term"] == pytest.approx(+1025.0 * 9.81, rel=1e-5)
    assert abs(parts["total"] / anchor - 1.0) < 1e-5


def test_far_field_matches_anchor():
    """Route 2: radiation-stress balance from measured wave amplitudes."""
    anchor = wall_mean_drift_analytic(A)
    field = wall_first_order_field(A, OMEGA)
    for stations in ((-30.0, -40.0), (-60.0, -80.0)):
        dec = decompose_traveling_waves_2d(field.elevation, stations, OMEGA, n_time=64)
        assert dec["amplitude_plus"] == pytest.approx(A, rel=1e-9)
        assert dec["amplitude_minus"] == pytest.approx(A, rel=1e-9)
        assert dec["fit_residual"] < 1e-9
        force = two_d_far_field_drift(A, dec["amplitude_minus"], 0.0)
        assert abs(force / anchor - 1.0) < 1e-9


def test_far_field_is_quadratic_in_amplitude():
    for amp in (0.3, 1.0, 2.0):
        assert two_d_far_field_drift(amp, amp, 0.0) == pytest.approx(
            0.5 * 1025.0 * 9.81 * amp**2, rel=1e-14
        )
    # a fully transmitting body produces no drift force
    assert two_d_far_field_drift(1.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        two_d_far_field_drift(1.0, -0.5, 0.3)


def test_panel_convergence_gate():
    """Three-level Richardson extrapolation, residual gate 2 %."""
    anchor = wall_mean_drift_analytic(A)
    counts = (50, 100, 200)
    values = [near_field_drift_wall(A, OMEGA, n_panels=n, n_time=64) for n in counts]
    res = richardson_extrapolate(counts, values)
    assert res["residual_relative"] < 0.02
    assert 1.5 < res["observed_order"] < 2.5  # second-order pressure quadrature
    assert abs(res["extrapolated"] / anchor - 1.0) < 3e-3


def test_richardson_validation():
    with pytest.raises(ValueError):
        richardson_extrapolate((10, 20), [1.0, 1.0])
    with pytest.raises(ValueError):
        richardson_extrapolate((10, 20, 30), [1.0, 2.0, 3.0])  # non-uniform


def test_maruo_zero_kochin_gives_zero_force():
    theta = 2.0 * math.pi * np.arange(1024) / 1024
    res = maruo_far_field_drift_3d(np.zeros(1024), theta, 1.0, 1.0)
    assert abs(res["mean_drift_force_x"]) < 1e-12


@pytest.mark.parametrize("kR", [100.0, 400.0])
def test_maruo_far_field_is_not_converged_in_radius(kR):
    """Documented limitation: the un-reduced flux route oscillates with ``kR``.

    Pinned so that a future analytically reduced (Maruo 1+cos kernel)
    implementation cannot silently inherit this behaviour: replace this test when
    the reduction lands.
    """
    n = 8192
    theta = 2.0 * math.pi * np.arange(n) / n
    kochin = 0.01 * np.cos(theta)
    res = maruo_far_field_drift_3d(kochin, theta, 1.0, 1.0, radius_factor=kR)
    assert abs(res["flux_scattered"]) < 1e-12  # symmetric dipole: no net |K|^2 flux
    means = [
        maruo_far_field_drift_3d(
            kochin, theta, 1.0, 1.0, radius_factor=r
        )["mean_drift_force_x"]
        for r in (100.0, 200.0, 400.0, 800.0)
    ]
    spread = (max(map(abs, means)) - min(map(abs, means))) / abs(np.mean(means))
    assert spread > 0.5  # the route is not converged -- honest FAIL recorded


def test_study_runner_contract(tmp_path):
    from tensorbem.benchmarks.drift_force_study import run_study

    study = run_study(n_time=32, n_theta=1024)
    verdict = study["verdict"]
    assert verdict["near_field_pass"] is True
    assert verdict["far_field_pass"] is True
    assert verdict["surface_waterline_pass"] is True
    assert verdict["convergence_gate_pass"] is True
    assert verdict["cylinder_asymptotic_pass"] is False
    assert study["near_field"]["richardson_gate"]["residual_relative"] < 0.02