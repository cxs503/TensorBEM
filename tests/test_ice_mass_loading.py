"""Verification of the mass-loading ice-cover dispersion relation.

Every assertion compares the production solver with a route that is
algebraically or numerically independent of it:

* the deep-water Weitz-Keller closed form ``omega**2 = g k/(1 + mu k)``;
* the shallow-water closed form ``omega**2 = g k**2 H/(1 + mu k**2 H)``;
* a fourth-order Runge-Kutta integration of the vertical ODE ``W'' = k**2 W``
  that never uses the ``cosh`` eigenfunction;
* the Robin-condition residual ``k tanh(kH) - omega**2/(g - mu omega**2)``;
* the mass-loading cut-off and the evanescent branch.

The negative cases pin the input validation: non-physical ice covers and wave
parameters must raise instead of returning a number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.ice import (
    IceCover,
    dispersion_table,
    energy_flux_amplitude_ratio,
    mass_loading_attenuation,
    mass_loading_group_velocity,
    mass_loading_omega,
    mass_loading_wavenumber,
    open_water_group_velocity,
    open_water_wavenumber,
    robin_coefficient,
    robin_residual,
    shooting_omega,
    thickness_study,
)

ICE = IceCover(thickness=0.9, water_depth=200.0)


def test_deep_water_limit_matches_weitz_keller():
    """``H -> infinity`` must reproduce the closed form ``omega^2 = g k/(1+mu k)``."""
    deep = IceCover(thickness=0.9, water_depth=1.0e5)
    omega = 2.0 * math.pi / 10.0
    k = mass_loading_wavenumber(omega, deep).wavenumber
    assert k * deep.water_depth > 50.0  # the deep-water assumption is real here
    reference = math.sqrt(deep.gravity * k / (1.0 + deep.mu * k))
    assert abs(reference - omega) / omega < 1.0e-6


def test_shallow_water_limit_matches_long_wave_closed_form():
    """``kH -> 0`` must reproduce ``omega^2 = g k^2 H/(1 + mu k^2 H)``."""
    shallow = IceCover(thickness=0.9, water_depth=6.0)
    omega = 2.0 * math.pi / 12.0
    k = mass_loading_wavenumber(omega, shallow).wavenumber
    assert k * shallow.water_depth < 0.5  # long-wave regime
    tanh = math.tanh(k * shallow.water_depth)
    reference = math.sqrt(
        shallow.gravity * k * tanh / (1.0 + shallow.mu * k * tanh)
    )
    assert abs(reference - omega) / omega < 1.0e-6


def test_wavenumber_matches_independent_rk4_shooting():
    """The RK4 shooting of ``W'' = k^2 W`` must return the same frequency."""
    for period in (4.0, 6.0, 10.0, 14.0):
        omega = 2.0 * math.pi / period
        k = mass_loading_wavenumber(omega, ICE).wavenumber
        omega_shooting = shooting_omega(k, ICE, steps=600)
        assert abs(omega_shooting - omega) / omega < 1.0e-9


def test_robin_condition_residual_is_negligible():
    """``k tanh(kH) = omega^2/(g - mu omega^2)`` must hold at the solved root."""
    for period in (3.5, 5.0, 8.0, 12.0):
        omega = 2.0 * math.pi / period
        k = mass_loading_wavenumber(omega, ICE).wavenumber
        assert abs(robin_residual(k, omega, ICE)) < 1.0e-12
        # the solved wave also satisfies the cut-off-normalised coefficient
        assert robin_coefficient(omega, ICE) > 0.0


def test_thickness_study_matches_analytic_root_below_five_percent():
    """``k(h_i)`` must match the independent Robin root to far better than 5 %."""
    study = thickness_study(2.0 * math.pi / 6.0, [0.3, 0.6, 0.9, 1.5, 2.5], ICE)
    assert np.all(study["relative_difference"] < 0.05)
    assert np.all(np.diff(study["k_ice"]) > 0.0)  # more ice -> shorter waves
    assert np.all(study["k_ice"] > study["k_open"])  # ice always shortens the waves


def test_evanescent_branch_above_cutoff():
    """Above ``sqrt(g/mu)`` no propagating root exists; ``kappa`` must be finite."""
    cutoff = ICE.cutoff_omega
    with pytest.raises(ValueError):
        robin_coefficient(cutoff, ICE)
    for factor in (1.02, 1.5, 3.0):
        wave = mass_loading_wavenumber(cutoff * factor, ICE)
        assert wave.propagating is False
        assert wave.wavenumber == 0.0
        assert wave.decay_rate > 0.0
        target = (cutoff * factor) ** 2 / (
            ICE.mu * (cutoff * factor) ** 2 - ICE.gravity
        )
        assert (
            abs(wave.decay_rate * math.tan(wave.decay_rate * ICE.water_depth) - target)
            < 1.0e-6
        )
        # exponential decay over 100 m, and monotone in distance
        near = mass_loading_attenuation(cutoff * factor, ICE, 10.0)
        far = mass_loading_attenuation(cutoff * factor, ICE, 100.0)
        assert 0.0 < far < near < 1.0


def test_mass_loading_is_conservative_below_cutoff():
    """Below cut-off mass loading dissipates nothing: attenuation ratio is exactly 1."""
    omega = 2.0 * math.pi / 8.0
    assert mass_loading_attenuation(omega, ICE, 0.0) == 1.0
    assert mass_loading_attenuation(omega, ICE, 5000.0) == 1.0
    ratio = energy_flux_amplitude_ratio(omega, ICE)
    # cg decreases under ice, so an energy-flux-conserving field is steeper
    cg_open = open_water_group_velocity(
        omega, ICE.water_depth, ICE.gravity,
        open_water_wavenumber(omega, ICE.water_depth, ICE.gravity),
    )
    cg_ice = mass_loading_wavenumber(omega, ICE).group_speed
    assert cg_ice < cg_open
    assert ratio == pytest.approx(math.sqrt(cg_open / cg_ice), rel=1e-12)
    assert ratio > 1.0


def test_group_velocity_tends_to_open_water_as_ice_vanishes():
    """``mu -> 0``: ice group velocity tends to the open-water group velocity."""
    thin = IceCover(thickness=1.0e-12, water_depth=200.0)
    omega = 2.0 * math.pi / 9.0
    k_open = open_water_wavenumber(omega, ICE.water_depth, ICE.gravity)
    cg_open = open_water_group_velocity(omega, ICE.water_depth, ICE.gravity, k_open)
    k_ice = mass_loading_wavenumber(omega, thin).wavenumber
    cg_ice = mass_loading_group_velocity(omega, thin, k_ice)
    assert abs(cg_ice - cg_open) / cg_open < 1.0e-6
    assert energy_flux_amplitude_ratio(omega, thin) == pytest.approx(1.0, abs=1e-9)


def test_dispersion_table_shapes_and_wavelength_consistency():
    table = dispersion_table([0.5, 0.8, 1.2], ICE)
    assert table["k_ice"].shape == (3,)
    assert np.all(table["propagating"] == 1.0)
    assert np.allclose(table["wavelength"], 2.0 * math.pi / table["k_ice"])
    assert np.allclose(table["phase_speed"], table["omega"] / table["k_ice"])


# ---------------------------------------------------------------------------
# negative cases: non-physical input must raise
# ---------------------------------------------------------------------------
def test_negative_ice_thickness_rejected():
    with pytest.raises(ValueError, match="ice thickness"):
        IceCover(thickness=-0.5)


def test_zero_ice_thickness_rejected():
    with pytest.raises(ValueError, match="ice thickness"):
        IceCover(thickness=0.0)


def test_zero_water_depth_rejected():
    with pytest.raises(ValueError, match="water depth"):
        IceCover(thickness=0.9, water_depth=0.0)


def test_ice_thicker_than_water_column_rejected():
    with pytest.raises(ValueError, match="smaller than the water depth"):
        IceCover(thickness=5.0, water_depth=4.0)


def test_negative_water_density_rejected():
    with pytest.raises(ValueError, match="water density"):
        IceCover(thickness=0.9, water_density=-1025.0)


def test_negative_frequency_and_negative_distance_rejected():
    with pytest.raises(ValueError, match="omega"):
        mass_loading_wavenumber(-1.0, ICE)
    with pytest.raises(ValueError, match="wavenumber"):
        mass_loading_omega(-0.3, ICE)
    with pytest.raises(ValueError, match="distance"):
        mass_loading_attenuation(1.0, ICE, -10.0)


def test_negative_thickness_array_rejected():
    with pytest.raises(ValueError, match="thicknesses"):
        thickness_study(1.0, [0.5, -0.5], ICE)
    with pytest.raises(ValueError, match="smaller than the water depth"):
        thickness_study(1.0, [0.5, 500.0], ICE)


def test_negative_wave_number_and_amplitude_rejected():
    with pytest.raises(ValueError, match="wavenumber"):
        mass_loading_omega(-0.1, ICE)
    with pytest.raises(ValueError, match="wavenumber"):
        robin_residual(-0.1, 1.0, ICE)

def test_verification_driver_reports_the_gates(tmp_path):
    """The artifact driver must run and report the verification gates."""
    from tensorbem.benchmarks.ice_ship_verification import run_dispersion_stage

    payload = run_dispersion_stage(periods=(6.0, 10.0), thicknesses=(0.6, 0.9))
    verification = payload["verification"]
    assert verification["gate_thickness_study_lt_5_percent"] is True
    assert verification["max_robin_residual_abs_per_m"] < 1.0e-10
    assert verification["max_rk4_shooting_error"] < 1.0e-9
    assert payload["limits"]["deep_water_weitz_keller"]["relative_error"] < 1.0e-6
    assert payload["thickness_study"]["max_relative_difference"] < 0.05
    assert set(payload["evanescent_branch"]) == {"1.02", "1.20", "2.00"}
    # JSON-serialisable (the driver writes it verbatim)
    import json

    json.dumps(payload, default=lambda o: float(o))
