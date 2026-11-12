"""Tests for the W25 RAO truncation helpers and the Wigley III KB fix.

The truncation study of mission 16712 rests on three self-contained pieces of
arithmetic: the ITTC two-parameter spectrum, the truncated band statistics of a
sampled RAO curve, and the Wigley III geometry integrals (volume, centre of
buoyancy, inertia).  Each is anchored to a closed form here, so the study cannot
silently drift away from its analytic references.
"""

import math

import numpy as np
import pytest

from tensorbem.benchmarks.rao_truncation import (
    band_statistics,
    curve_peak,
    doubling_pairs,
    ittc_spectral_peak_omega,
    ittc_two_parameter_spectrum,
    omega_from_wavelength,
    relative_change_pct,
    relative_deviation_pct,
    truncation_radius_omega_min,
    wigley3_body_inertia,
)
from tensorbem.wigley3_hull import (
    WIGLEY3_ALPHA,
    WIGLEY3_B,
    WIGLEY3_L,
    WIGLEY3_T,
    compute_wigley3_properties,
)


# ---------------------------------------------------------------------------
# ITTC spectrum
# ---------------------------------------------------------------------------

def test_spectrum_zeroth_moment_is_hs_squared_over_sixteen():
    """ITTC (1978) two-parameter spectrum: m0 = A/(4B) = Hs^2/16."""
    hs, t1 = 2.5, 7.0
    omega = np.linspace(1e-3, 20.0, 400001)
    spec = ittc_two_parameter_spectrum(omega, hs, t1)
    m0 = float(np.trapezoid(spec, omega))
    # 20 rad/s leaves a negligible tail; m0 = A/(4B) = Hs^2/16 exactly.
    assert m0 == pytest.approx(hs**2 / 16.0, rel=2e-3)


def test_spectrum_peak_matches_closed_form():
    t1 = 1.4
    omega = np.linspace(0.05, 12.0, 200001)
    spec = ittc_two_parameter_spectrum(omega, 1.0, t1)
    grid_peak = float(omega[int(np.argmax(spec))])
    assert grid_peak == pytest.approx(ittc_spectral_peak_omega(t1), rel=2e-3)


def test_spectrum_rejects_non_positive_frequency():
    with pytest.raises(ValueError):
        ittc_two_parameter_spectrum(np.array([0.0, 1.0]), 1.0, 1.4)


def test_dispersion_and_truncation_radius_are_consistent():
    L = 3.0
    wavelength = omega_from_wavelength(7.5)
    omega = omega_from_wavelength(7.5)
    assert omega == pytest.approx(wavelength)
    assert truncation_radius_omega_min(2.5, L) == pytest.approx(
        omega_from_wavelength(2.5 * L)
    )


# ---------------------------------------------------------------------------
# curve peak / band statistics
# ---------------------------------------------------------------------------

def test_curve_peak_refines_a_gaussian_between_nodes():
    omega = np.linspace(0.5, 6.0, 111)
    center, width = 3.14159, 0.37
    values = np.exp(-((omega - center) ** 2) / (2.0 * width**2))
    peak = curve_peak(omega, values)
    assert peak.refined
    assert peak.omega == pytest.approx(center, rel=5e-3)
    assert peak.value == pytest.approx(1.0, rel=5e-3)


def test_curve_peak_keeps_end_maximum_unrefined():
    omega = np.linspace(1.0, 5.0, 9)
    values = np.linspace(0.0, 1.0, 9)
    peak = curve_peak(omega, values)
    assert not peak.refined
    assert peak.omega == pytest.approx(5.0)


def test_band_statistics_full_band_has_unit_area_fraction():
    omega = np.linspace(0.5, 12.0, 240)
    rao = np.ones_like(omega)
    spec = ittc_two_parameter_spectrum(omega, 1.0, 1.4)
    stats = band_statistics(omega, rao, spec, 0.0, math.inf)
    assert stats.area_fraction == pytest.approx(1.0, abs=1e-12)
    assert stats.sigma_band == pytest.approx(stats.sigma_full, rel=1e-12)


def test_band_statistics_matches_analytic_for_flat_integrand():
    omega = np.linspace(1.0, 3.0, 201)
    rao = np.full_like(omega, 2.0)
    spec = np.full_like(omega, 3.0)
    stats = band_statistics(omega, rao, spec, 1.2, 2.4)
    assert stats.m0_band == pytest.approx(12.0 * (2.4 - 1.2), rel=1e-12)
    assert stats.area_fraction == pytest.approx((2.4 - 1.2) / (3.0 - 1.0), rel=1e-12)


def test_band_statistics_is_monotone_in_the_upper_edge():
    omega = np.linspace(0.5, 12.0, 240)
    rao = np.exp(-((omega - 4.0) ** 2))
    spec = ittc_two_parameter_spectrum(omega, 1.0, 1.4)
    fractions = [
        band_statistics(omega, rao, spec, 0.0, edge).area_fraction
        for edge in (2.0, 4.0, 8.0, 12.0)
    ]
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    assert fractions[-1] == pytest.approx(1.0, rel=1e-12)


def test_band_statistics_rejects_a_band_without_nodes():
    omega = np.linspace(1.0, 2.0, 5)
    with pytest.raises(ValueError):
        band_statistics(omega, np.ones_like(omega), np.ones_like(omega), 5.0, 6.0)


# ---------------------------------------------------------------------------
# doubling ladders and percentage helpers
# ---------------------------------------------------------------------------

def test_doubling_pairs_picks_only_exact_doublings():
    pairs = doubling_pairs((1.5, 3.0, 6.0, 12.0))
    assert pairs == [(1.5, 3.0), (3.0, 6.0), (6.0, 12.0)]
    pairs = doubling_pairs((1.0, 1.25, 2.0, 2.5))
    assert pairs == [(1.0, 2.0), (1.25, 2.5)]


def test_percentage_helpers_are_symmetric():
    assert relative_change_pct(1.0, 2.0) == pytest.approx(50.0)
    assert relative_change_pct(2.0, 1.0) == pytest.approx(50.0)
    assert relative_deviation_pct(0.95, 1.0) == pytest.approx(5.0)
    assert math.isnan(relative_deviation_pct(1.0, 0.0))


# ---------------------------------------------------------------------------
# Wigley III geometry and the centre-of-buoyancy fix
# ---------------------------------------------------------------------------

def test_wigley3_kb_at_alpha_zero_is_five_eighths_of_draft():
    """Classical result for the parabolic Wigley hull: KB = 5 T / 8."""
    props = compute_wigley3_properties(alpha=0.0)
    assert props.KB / WIGLEY3_T == pytest.approx(5.0 / 8.0, rel=1e-12)


def test_wigley3_kb_matches_independent_quadrature():
    props = compute_wigley3_properties()
    inertia = wigley3_body_inertia(
        WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_ALPHA, 1025.0
    )
    kb_quadrature = WIGLEY3_T + inertia["z_centroid"]
    assert kb_quadrature == pytest.approx(props.KB, rel=1e-3)


def test_wigley3_volume_matches_closed_form():
    props = compute_wigley3_properties()
    inertia = wigley3_body_inertia(
        WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_ALPHA, 1025.0
    )
    assert inertia["volume"] == pytest.approx(props.volume, rel=1e-4)


def test_wigley3_inertia_radii_of_gyration_are_ship_like():
    """Sanity band for a Wigley form: k_yy/L ~ 0.2-0.3, k_xx/B ~ 0.2-0.4."""
    inertia = wigley3_body_inertia(
        WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_ALPHA, 1025.0
    )
    mass = 1025.0 * inertia["volume"]
    k_yy = math.sqrt(inertia["Iyy"] / mass)
    k_xx = math.sqrt(inertia["Ixx"] / mass)
    assert 0.15 < k_yy / WIGLEY3_L < 0.35
    assert 0.15 < k_xx / WIGLEY3_B < 0.45
    assert inertia["Izz"] > inertia["Iyy"]
