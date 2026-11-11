"""Verification tests for the unified flexural-gravity + mass-loading dispersion.

The implemented relation (eq. 3 of ``tensorbem.flexural_gravity``) is

    omega^2 = (g k + D k^5/rho_w) tanh(k H) / (1 + mu k tanh(k H)),
    mu = rho_i h_ice / rho_w,

which keeps both the bending stiffness of the ice sheet (eq. 2, rigidity only)
and the inertia of the ice sheet (``tensorbem.ice``, mass loading only).

Each test pins one of the criteria V1..V10 of
``scripts/verify_flexural_gravity_mass_loading.py``:

V1  mu -> 0 recovers the rigidity-only relation (2)
V2  short-wave deep-water limit -> classical plate-in-water D k^5/(rho_w + rho_i h k)
V3  long-wave shallow-water limit -> g H k^2/(1 + mu H k^2)
V4  cross-module consistency: D -> 0 reproduces ``tensorbem.ice`` mass loading
V5  closed-form crossover wavenumber k_c = (g mu/a)^(1/3)
V6  analytic group velocity vs central difference
V7  wavenumber solver round trip
V8  ice-edge energy conservation with plate inertia
V9  magnitude of the inertia correction in the ocean-wave band
V10 sign of (lambda_ice - lambda_open) flips exactly at the crossover
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.flexural_gravity import (
    IcePlate,
    crossover_wavenumber,
    deep_water_crossover_wavenumber,
    flexural_gravity_group_velocity_mass_loading,
    flexural_gravity_omega,
    flexural_gravity_omega2,
    flexural_gravity_omega_mass_loading,
    flexural_gravity_omega2_mass_loading,
    flexural_gravity_wavenumber,
    flexural_gravity_wavenumber_mass_loading,
    ice_edge_coefficients,
    mass_loading_parameter,
    open_water_wavenumber,
    phase_speed_ratio_mass_loading,
    wavelength_ratio_mass_loading,
)
from tensorbem.ice import IceCover, mass_loading_omega

TWO_PI = 2.0 * math.pi


@pytest.fixture
def plate():
    return IcePlate(thickness=1.0, water_depth=200.0)


# ---------------------------------------------------------------------------
# V1  inertia -> 0 recovers equation (2)
# ---------------------------------------------------------------------------
def test_v1_zero_ice_density_recovers_equation_two(plate):
    light = IcePlate(thickness=1.0, water_depth=200.0, ice_density=1.0e-9)
    for k in np.logspace(-4, 2, 31):
        a = flexural_gravity_omega_mass_loading(float(k), light)
        b = flexural_gravity_omega(float(k), plate)
        assert abs(a - b) / b < 1.0e-9


def test_v1_mass_loading_parameter_from_densities():
    p = IcePlate(thickness=0.8, ice_density=900.0, water_density=1000.0)
    assert mass_loading_parameter(p) == pytest.approx(0.72, rel=0, abs=1e-15)


# ---------------------------------------------------------------------------
# V2  short-wave deep-water limit: classical plate-in-water dispersion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [5.0, 10.0, 20.0])
def test_v2_short_wave_limit_matches_plate_in_water(plate, k):
    mu = mass_loading_parameter(plate)
    measured = flexural_gravity_omega2_mass_loading(k, plate)
    reference = plate.flexural_rigidity * k ** 5 / (
        plate.water_density * (1.0 + mu * k)
    )
    assert abs(measured - reference) / reference < 1.0e-6


# ---------------------------------------------------------------------------
# V3  long-wave shallow-water limit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h_ice", [0.1, 0.5, 1.0, 2.0])
def test_v3_long_wave_shallow_limit(h_ice):
    p = IcePlate(thickness=h_ice, water_depth=200.0)
    k = 1.0e-3 / p.water_depth
    mu = mass_loading_parameter(p)
    measured = flexural_gravity_omega2_mass_loading(k, p)
    reference = p.gravity * k ** 2 * p.water_depth / (
        1.0 + mu * p.water_depth * k ** 2
    )
    assert abs(measured - reference) / reference < 1.0e-5


# ---------------------------------------------------------------------------
# V4  cross-module consistency with tensorbem.ice (D -> 0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h_ice", [0.1, 0.5, 1.0, 2.0])
def test_v4_soft_limit_matches_ice_module(h_ice):
    soft = IcePlate(thickness=h_ice, water_depth=200.0, young_modulus=1.0e-6)
    cover = IceCover(thickness=h_ice, water_depth=200.0)
    assert mass_loading_parameter(soft) == pytest.approx(cover.mu, rel=1e-15)
    for k in (1.0e-4, 1.0e-3, 1.0e-2, 0.1):
        a = flexural_gravity_omega_mass_loading(k, soft)
        b = mass_loading_omega(k, cover)
        assert abs(a - b) / b < 1.0e-9


def test_v4_rigidity_changes_the_answer(plate):
    """Sanity: the unified relation is *not* the mass-loading relation."""
    cover = IceCover(thickness=plate.thickness, water_depth=plate.water_depth)
    stiff = flexural_gravity_omega_mass_loading(0.2, plate)
    massload = mass_loading_omega(0.2, cover)
    assert stiff > 1.5 * massload


# ---------------------------------------------------------------------------
# V5  closed-form crossover wavenumber
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h_ice", [0.1, 0.5, 1.0, 2.0])
def test_v5_crossover_closed_form_matches_phase_speed(h_ice):
    p = IcePlate(thickness=h_ice, water_depth=200.0)
    k_c = deep_water_crossover_wavenumber(p)
    # at k_c the ice-covered phase speed equals the open-water phase speed
    omega = flexural_gravity_omega_mass_loading(k_c, p)
    c_ice = omega / k_c
    c_open = math.sqrt(p.gravity * math.tanh(k_c * p.water_depth) / k_c)
    assert abs(c_ice - c_open) / c_open < 1.0e-4


@pytest.mark.parametrize("h_ice", [0.1, 0.5, 1.0, 2.0])
def test_v5_finite_depth_crossover_close_to_closed_form(h_ice):
    p = IcePlate(thickness=h_ice, water_depth=200.0)
    closed = deep_water_crossover_wavenumber(p)
    finite = crossover_wavenumber(p, deep_water=False)
    # the closed form is the deep-water (k H >> 1) limit; for H = 200 m the
    # tanh correction is < 1e-3 even for the thickest plate
    assert abs(closed - finite) / finite < 1.0e-3


def test_v5_crossover_scales_with_young_modulus():
    soft = IcePlate(thickness=1.0, young_modulus=1.0e9)
    stiff = IcePlate(thickness=1.0, young_modulus=8.0e9)
    # k_c ~ (mu rho g / D)^(1/3) ~ E^(-1/3)
    ratio = deep_water_crossover_wavenumber(soft) / deep_water_crossover_wavenumber(stiff)
    assert ratio == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-12)


# ---------------------------------------------------------------------------
# V6  analytic group velocity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("omega", [0.2, 0.6, 2.0, 6.0])
def test_v6_group_velocity_matches_central_difference(plate, omega):
    k = flexural_gravity_wavenumber_mass_loading(omega, plate)
    analytic = flexural_gravity_group_velocity_mass_loading(omega, plate, k)
    h = 1.0e-6 * k
    numeric = (
        flexural_gravity_omega_mass_loading(k + h, plate)
        - flexural_gravity_omega_mass_loading(k - h, plate)
    ) / (2.0 * h)
    assert abs(analytic - numeric) / abs(numeric) < 1.0e-6


def test_v6_group_velocity_of_mass_loading_limit(plate):
    """In the D -> 0 limit cg -> c, i.e. the shallow long-wave speed."""
    soft = IcePlate(thickness=1.0, water_depth=20.0, young_modulus=1.0e-6)
    omega = 0.05
    k = flexural_gravity_wavenumber_mass_loading(omega, soft)
    cg = flexural_gravity_group_velocity_mass_loading(omega, soft, k)
    # k H = O(0.07): the shallow-water group velocity equals the phase speed
    assert cg == pytest.approx(omega / k, rel=2e-2)


# ---------------------------------------------------------------------------
# V7  wavenumber round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h_ice", [0.1, 0.5, 1.0, 2.0])
def test_v7_round_trip(h_ice):
    p = IcePlate(thickness=h_ice, water_depth=200.0)
    for omega in np.logspace(-2, 1.5, 13):
        k = flexural_gravity_wavenumber_mass_loading(float(omega), p)
        back = flexural_gravity_omega_mass_loading(k, p)
        assert abs(back - float(omega)) / float(omega) < 1.0e-9


def test_v7_wavenumber_below_open_water_where_bending_dominates(plate):
    omega = TWO_PI * 0.3
    k_ice = flexural_gravity_wavenumber_mass_loading(omega, plate)
    k_open = open_water_wavenumber(omega, plate.water_depth, plate.gravity)
    assert k_ice < k_open          # bending lengthens the wave


# ---------------------------------------------------------------------------
# V8  ice-edge energy conservation with plate inertia
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frequency", [0.05, 0.1, 0.2, 0.5, 1.0])
def test_v8_energy_conservation_with_inertia(plate, frequency):
    res = ice_edge_coefficients(TWO_PI * frequency, plate, include_plate_inertia=True)
    assert abs(res.energy_residual) < 1.0e-12
    assert res.energy_residual == pytest.approx(0.0, abs=1e-12)


def test_v8_default_flag_is_backward_compatible(plate):
    """include_plate_inertia defaults to False: identical to the old call."""
    a = ice_edge_coefficients(1.5, plate)
    b = ice_edge_coefficients(1.5, plate, include_plate_inertia=False)
    assert a == b


def test_v8_inertia_changes_wavenumbers_but_not_energy(plate):
    legacy = ice_edge_coefficients(1.5, plate)
    unified = ice_edge_coefficients(1.5, plate, include_plate_inertia=True)
    assert unified.k_ice != legacy.k_ice
    assert unified.k_open == legacy.k_open
    assert abs(unified.energy_residual) < 1.0e-12
    assert abs(legacy.energy_residual) < 1.0e-12


# ---------------------------------------------------------------------------
# V9  magnitude of the inertia correction in the ocean-wave band
# ---------------------------------------------------------------------------
def test_v9_inertia_correction_stays_small_in_ocean_band():
    worst = 0.0
    for young in (1.0e9, 5.0e9, 10.0e9):
        for h_ice in (0.5, 1.0, 2.0):
            p = IcePlate(thickness=h_ice, water_depth=200.0, young_modulus=young)
            for frequency in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
                omega = TWO_PI * frequency
                unified = wavelength_ratio_mass_loading(omega, p)
                legacy = (
                    open_water_wavenumber(omega, p.water_depth, p.gravity)
                    / flexural_gravity_wavenumber(omega, p)
                )
                worst = max(worst, abs(unified / legacy - 1.0))
    assert worst < 0.10


# ---------------------------------------------------------------------------
# V10  the sign of (lambda_ice - lambda_open) flips at the crossover
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h_ice", [0.5, 1.0, 2.0])
def test_v10_slower_below_crossover_faster_above(h_ice):
    p = IcePlate(thickness=h_ice, water_depth=200.0)
    k_c = deep_water_crossover_wavenumber(p)
    f_c = flexural_gravity_omega_mass_loading(k_c, p) / TWO_PI
    below = wavelength_ratio_mass_loading(TWO_PI * 0.5 * f_c, p)
    above = wavelength_ratio_mass_loading(TWO_PI * 2.0 * f_c, p)
    assert below < 1.0          # inertia dominated: wave shortened
    assert above > 1.0          # bending dominated: wave lengthened


def test_v10_ratio_equals_phase_speed_ratio(plate):
    """lambda_ice/lambda_open and c_ice/c_open are the same number (lambda=2pi/k)."""
    for frequency in (0.05, 0.2, 0.5):
        omega = TWO_PI * frequency
        assert wavelength_ratio_mass_loading(omega, plate) == pytest.approx(
            phase_speed_ratio_mass_loading(omega, plate), rel=1e-15
        )


# ---------------------------------------------------------------------------
# input validation / negative controls
# ---------------------------------------------------------------------------
def test_negative_control_zero_rigidity_and_zero_mass_is_open_water():
    p = IcePlate(thickness=1.0e-9, water_depth=200.0)
    for k in (0.01, 0.1, 1.0):
        measured = flexural_gravity_omega2_mass_loading(k, p)
        reference = p.gravity * k * math.tanh(k * p.water_depth)
        assert abs(measured - reference) / reference < 1.0e-6


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_wavenumber_rejected(bad):
    p = IcePlate(thickness=1.0)
    with pytest.raises(ValueError):
        flexural_gravity_omega2_mass_loading(bad, p)
    with pytest.raises(ValueError):
        flexural_gravity_omega_mass_loading(bad, p)


@pytest.mark.parametrize("bad", [0.0, -2.0, float("nan")])
def test_invalid_omega_rejected(bad):
    p = IcePlate(thickness=1.0)
    with pytest.raises(ValueError):
        flexural_gravity_wavenumber_mass_loading(bad, p)
    with pytest.raises(ValueError):
        phase_speed_ratio_mass_loading(bad, p)


def test_mass_loading_parameter_is_positive_and_monotone_in_thickness():
    values = [mass_loading_parameter(IcePlate(thickness=h)) for h in (0.1, 0.5, 1.0, 2.0)]
    assert all(v > 0.0 for v in values)
    assert values == sorted(values)