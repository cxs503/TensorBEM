"""Three-route cross-validation of the ellipsoid Kochin function and wave resistance.

The ellipsoid_kochin module provides three independent evaluations of the Kochin
function H(k) for a half-submerged ellipsoid:
  1. kochin_analytic   — Bessel closed form (one spectral integral in u)
  2. kochin_quadrature — definition quadrature on elliptical projection
  3. kochin_panels     — thin-ship Rankine-source discretisation

This test cross-validates all three routes at multiple wavenumbers and checks
wave resistance consistency across routes.

Key finding: the 'panels' route is a thin-ship approximation, so it carries both
discretisation error AND modelling error O((b/a)^2). The 'analytic' and 'quadrature'
routes compute the exact ellipsoid Kochin function and agree to < 0.1%.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from tensorbem.ellipsoid_kochin import (
    EllipsoidHull,
    kochin_analytic,
    kochin_quadrature,
    kochin_panels,
    wave_resistance_analytic,
    wave_resistance_panels,
)


@pytest.fixture
def ellipsoid():
    """Standard test ellipsoid: a=5, b=1, c=0.5 (b/a=0.2 thin-ship regime)."""
    return EllipsoidHull(a=5.0, b=1.0, c=0.5)


@pytest.fixture
def wavenumbers():
    """Range of wavenumbers covering long-wave to short-wave."""
    return [0.3, 0.5, 1.0, 2.0, 4.0]


class TestExactRoutesAgreement:
    """The two exact routes (analytic and quadrature) must agree to high precision."""

    def test_analytic_vs_quadrature_magnitudes(self, ellipsoid, wavenumbers):
        """|H_analytic - H_quadrature| / |H_analytic| < 0.1% for all k."""
        for k in wavenumbers:
            H_a = kochin_analytic(k, ellipsoid)
            H_q = kochin_quadrature(k, ellipsoid)
            rel_err = abs(H_a - H_q) / abs(H_a)
            assert rel_err < 0.001, f"k={k}: rel_err={rel_err:.4e}"

    def test_analytic_vs_quadrature_phases(self, ellipsoid, wavenumbers):
        """Phase difference < 0.1 degree for all k."""
        for k in wavenumbers:
            H_a = kochin_analytic(k, ellipsoid)
            H_q = kochin_quadrature(k, ellipsoid)
            phase_a = np.angle(H_a)
            phase_q = np.angle(H_q)
            phase_diff = abs(phase_a - phase_q)
            if phase_diff > math.pi:
                phase_diff = 2 * math.pi - phase_diff
            assert phase_diff < 0.001, f"k={k}: phase_diff={math.degrees(phase_diff):.4f} deg"


class TestThinShipPanelsBounded:
    """The thin-ship panels route has modelling error O((b/a)^2) = 4%.

    These tests verify that the panels route is *bounded* — not that it matches
    the exact solution. The discrepancy is modelling error, not a code bug.
    """

    def test_panels_source_bounded(self, ellipsoid, wavenumbers):
        """|H_analytic - H_panels(source)| / |H_analytic| < 20% for all k."""
        for k in wavenumbers:
            H_a = kochin_analytic(k, ellipsoid)
            H_p = kochin_panels(k, ellipsoid, n_r=32, n_phi=32, scheme="source")
            rel_err = abs(H_a - H_p) / abs(H_a)
            assert rel_err < 0.20, f"k={k}: rel_err={rel_err:.4e}"

    def test_panels_source_integrated_bounded(self, ellipsoid, wavenumbers):
        """|H_analytic - H_panels(source_integrated)| / |H_analytic| < 20%."""
        for k in wavenumbers:
            H_a = kochin_analytic(k, ellipsoid)
            H_p = kochin_panels(k, ellipsoid, n_r=32, n_phi=32, scheme="source_integrated")
            rel_err = abs(H_a - H_p) / abs(H_a)
            assert rel_err < 0.20, f"k={k}: rel_err={rel_err:.4e}"

    def test_panels_ordinate_bounded(self, ellipsoid, wavenumbers):
        """|H_analytic - H_panels(ordinate)| / |H_analytic| < 50% (coarse at high k)."""
        for k in wavenumbers:
            H_a = kochin_analytic(k, ellipsoid)
            H_p = kochin_panels(k, ellipsoid, n_r=32, n_phi=32, scheme="ordinate")
            rel_err = abs(H_a - H_p) / abs(H_a)
            assert rel_err < 0.50, f"k={k}: rel_err={rel_err:.4e}"


class TestWaveResistanceConsistency:
    """Cross-validate wave resistance from different routes."""

    def test_analytic_vs_panels_resistance(self, ellipsoid):
        """R_W from wave_resistance_analytic vs wave_resistance_panels (thin-ship)."""
        for Fn in [0.2, 0.3, 0.5, 0.7]:
            R_a = wave_resistance_analytic(ellipsoid, Fn)
            R_p = wave_resistance_panels(ellipsoid, Fn, n_r=32, n_phi=32)
            
            if abs(R_a) > 1e-6:
                rel_err = abs(R_a - R_p) / abs(R_a)
                # Thin-ship modelling error O((b/a)^2) = 4%, plus discretisation
                assert rel_err < 0.20, f"Fn={Fn}: rel_err={rel_err:.4e}"


class TestConvergenceStudy:
    """Verify convergence of discretisation routes."""

    def test_panels_converge_to_analytic(self, ellipsoid):
        """Error decreases with panel count (finest better than coarsest)."""
        k = 1.0
        H_ref = kochin_analytic(k, ellipsoid)
        
        errors = []
        for n in [8, 16, 32, 64]:
            H_p = kochin_panels(k, ellipsoid, n_r=n, n_phi=n, scheme="source")
            err = abs(H_ref - H_p) / abs(H_ref)
            errors.append(err)
        
        # Finest should be better than coarsest
        assert errors[-1] < errors[0], f"Finest not better: errors={errors}"

    def test_quadrature_converges_to_analytic(self, ellipsoid):
        """kochin_quadrature converges rapidly to kochin_analytic."""
        k = 2.0
        H_ref = kochin_analytic(k, ellipsoid, n_quad=256)
        
        errors = []
        for n_psi in [24, 48, 96]:
            H_q = kochin_quadrature(k, ellipsoid, n_psi=n_psi, n_phi=2 * n_psi)
            err = abs(H_ref - H_q) / abs(H_ref)
            errors.append(err)
        
        # Should converge
        assert errors[-1] < errors[0], f"Quadrature not converging: errors={errors}"
        # High-order quadrature should be very accurate
        assert errors[-1] < 0.001, f"Quadrature error {errors[-1]:.4e} exceeds 0.1%"
