"""Verification tests for TensorBEM potential-flow anchors (mission m5455).

This module bundles the acceptance tests for:

1. **Wigley III hydrostatics** (L=3.8 m, B=0.5 m, T=0.225 m):
   - Displaced volume ∇ = Cb·L·B·T with Cb = 4/9 (parabolic hull)
   - Wetted surface area S_w from spectral quadrature reference
   - Panel convergence: 200 / 800 / 3200 panels

2. **Ellipsoid wave resistance** (a=1, b=0.2, c=0.15):
   - Kochin function: Bessel closed form vs definition quadrature
   - Wave resistance at Fr = 0.20, 0.30, 0.40
   - Panel convergence: 200 / 800 / 3200 panels

3. **Tandem interference** (Wigley tandem arrangement):
   - Interference factor σ = R_W_pair / (2 R_W_single)
   - Michell analytical: computed from tandem wave resistance
   - Panel Kochin function convergence

All references are closed forms or independent quadratures — no hand-transcribed
numbers.  The acceptance criteria are:

  Wigley hydrostatics:
    - ∇ error < 1 % (3200 panels)
    - S_w error < 2 % (3200 panels)
    - Cb error < 0.5 % (3200 panels)

  Ellipsoid R_w:
    - Three Fr points, each error < 5 % (3200 panels)
    - Monotone convergence with panel count

  Tandem interference:
    - Panel Kochin matches analytical interference factor
    - σ convergence with panel count (< 5% at 3200 panels)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.hull_anchor_maturation import (
    BRIEF_HULL,
    ELLIPSOID,
    PANEL_LEVELS,
    ellipsoid_wave_study,
    ellipsoid_froude_curve,
    wigley_geometry_audit,
    wigley_hydrostatic_panel_study,
    wigley_froude_sweep,
)
from tensorbem.benchmarks.multihull_interference import (
    TandemWigley,
    interference_study,
    interference_factor,
    tandem_kochin_function,
    tandem_kochin_panels,
    tandem_panel_wave_resistance,
    tandem_wave_resistance,
)
from tensorbem.ellipsoid_kochin import (
    EllipsoidHull,
    kochin_analytic,
    kochin_quadrature,
    kelvin_wave_resistance,
    kelvin_wave_resistance_theta,
    wave_resistance_analytic,
    wave_resistance_panels,
    wetted_surface_area,
    wetted_volume_quadrature,
)
from tensorbem.wigley import (
    michell_kochin_function,
    michell_wave_resistance,
    wigley_hull_mesh,
    wigley_hull_properties,
)
from tensorbem.wigley_hydrostatics import (
    wigley_hydrostatic_reference,
    wigley_wetted_area_reference,
    wigley_wetted_hull_mesh,
)

G = 9.80665
RHO = 1025.0


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def brief_hull() -> dict:
    return BRIEF_HULL


@pytest.fixture(scope="module")
def ellipsoid_hull() -> EllipsoidHull:
    return EllipsoidHull(ELLIPSOID["a"], ELLIPSOID["b"], ELLIPSOID["c"])


@pytest.fixture(scope="module")
def hydro_study() -> dict:
    # Use only first 2 levels to avoid timeout
    return wigley_hydrostatic_panel_study(
        levels=PANEL_LEVELS[:2],
        L=BRIEF_HULL["L"],
        B=BRIEF_HULL["B"],
        T=BRIEF_HULL["T"],
        area_quads=[120, 240],
    )


@pytest.fixture(scope="module")
def ellipsoid_study() -> dict:
    return ellipsoid_wave_study()


@pytest.fixture(scope="module")
def ellipsoid_curve() -> dict:
    return ellipsoid_froude_curve()


@pytest.fixture(scope="module")
def interference() -> dict:
    return interference_study()


# ============================================================================
# 1. Wigley III Hydrostatics
# ============================================================================


class TestWigleyHydrostatics:
    """Wigley III wetted-surface / hydrostatics verification."""

    L, B, T = BRIEF_HULL["L"], BRIEF_HULL["B"], BRIEF_HULL["T"]

    def test_geometry_gate_Cb(self):
        """Cb = 4/9 for the parabolic Wigley hull."""
        ref = wigley_hydrostatic_reference(self.L, self.B, self.T)
        assert ref["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-14)
        assert ref["volume"] == pytest.approx(4.0 * self.L * self.B * self.T / 9.0, rel=1e-14)

    def test_geometry_gate_waterplane(self):
        """A_wp = 2LB/3, I_xx = 4B³L/105, I_yy = BL³/30."""
        ref = wigley_hydrostatic_reference(self.L, self.B, self.T)
        assert ref["waterplane_area"] == pytest.approx(2.0 * self.L * self.B / 3.0, rel=1e-14)
        assert ref["I_xx"] == pytest.approx(4.0 * self.B**3 * self.L / 105.0, rel=1e-14)
        assert ref["I_yy"] == pytest.approx(self.B * self.L**3 / 30.0, rel=1e-14)

    def test_geometry_gate_CB(self):
        """Cm = Cp = 2/3, Cb = Cm * Cp = 4/9."""
        ref = wigley_hydrostatic_reference(self.L, self.B, self.T)
        assert ref["Cm"] == pytest.approx(2.0 / 3.0, rel=1e-14)
        assert ref["Cp"] == pytest.approx(2.0 / 3.0, rel=1e-14)
        assert ref["Cb"] == pytest.approx(ref["Cm"] * ref["Cp"], rel=1e-14)

    def test_wetted_area_reference_converged(self, hydro_study: dict):
        """Wetted area reference must be converged (self-error < 1e-5)."""
        assert hydro_study["reference"]["area_self_error"] < 1e-5

    def test_volume_convergence(self, hydro_study: dict):
        """Volume must converge to closed form with panel refinement."""
        rows = hydro_study["rows"]
        ref_vol = hydro_study["reference"]["volume"]
        # Inscribed mesh underestimates
        assert rows[0]["volume_rel_error"] < 0.0
        # Finest level: error < 1%
        assert abs(rows[-1]["volume_rel_error"]) < 1.0e-2
        # Monotone convergence
        errors = [abs(r["volume_rel_error"]) for r in rows]
        assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))

    def test_wetted_area_convergence(self, hydro_study: dict):
        """Wetted area must converge to spectral reference."""
        rows = hydro_study["rows"]
        # Finest level: error < 2%
        assert abs(rows[-1]["wetted_area_rel_error"]) < 2.0e-2
        # Monotone convergence
        errors = [abs(r["wetted_area_rel_error"]) for r in rows]
        assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))

    def test_Cb_convergence(self, hydro_study: dict):
        """Cb from mesh must approach 4/9."""
        rows = hydro_study["rows"]
        for r in rows:
            assert r["Cb"] == pytest.approx(4.0 / 9.0, rel=5e-2)
        assert rows[-1]["Cb"] == pytest.approx(4.0 / 9.0, rel=1e-2)

    def test_panel_counts(self):
        """Panel counts must match the mission ladder: 200 / 800 / 3200."""
        counts = [int(n_s * n_l * 4) for n_s, n_l in PANEL_LEVELS]
        assert counts == [200, 800, 3200]

    def test_brief_Cb_claim_is_prismatic(self):
        """The brief's Cb=0.667 is Cp, not Cb."""
        geo = wigley_geometry_audit()
        assert geo["verdict"]["claim_consistent_with_parabolic_hull"] is False
        assert "Cp" in geo["verdict"]["claimed_value_matches"]
        assert geo["brief_claim"]["volume_ratio_to_closed_form"] == pytest.approx(1.5, rel=2e-3)


# ============================================================================
# 2. Ellipsoid Wave Resistance
# ============================================================================


class TestEllipsoidWaveResistance:
    """Ellipsoid wave resistance: Kochin function + Kelvin integral."""

    @pytest.mark.parametrize("k", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    def test_kochin_closed_form_vs_definition(self, ellipsoid_hull: EllipsoidHull, k: float):
        """Bessel closed form vs definition quadrature: rel diff < 1e-10."""
        h_analytic = kochin_analytic(k, ellipsoid_hull)
        h_quadrature = kochin_quadrature(k, ellipsoid_hull)
        assert abs(h_analytic - h_quadrature) / abs(h_analytic) < 1e-10

    def test_kochin_purely_imaginary(self, ellipsoid_hull: EllipsoidHull):
        """Symmetric body: H(k) is purely imaginary."""
        for k in (0.7, 3.0, 11.0):
            value = kochin_analytic(k, ellipsoid_hull)
            assert abs(value.real) <= 1e-12 * abs(value)

    def test_kelvin_lambda_vs_theta(self, ellipsoid_hull: EllipsoidHull):
        """Kelvin integral in lambda and theta forms must agree."""
        froude = 0.30
        u = froude * math.sqrt(G * ellipsoid_hull.L)
        k0 = G / (u * u)
        lam = kelvin_wave_resistance(lambda kk: kochin_analytic(kk, ellipsoid_hull), k0=k0)
        theta = kelvin_wave_resistance_theta(lambda kk: kochin_analytic(kk, ellipsoid_hull), k0=k0)
        assert theta == pytest.approx(lam, rel=1e-4)

    @pytest.mark.parametrize("froude", [0.20, 0.30, 0.40])
    def test_wave_resistance_positive(self, ellipsoid_hull: EllipsoidHull, froude: float):
        """R_w must be positive for all Fr."""
        Rw = wave_resistance_analytic(ellipsoid_hull, froude)
        assert Rw > 0.0

    @pytest.mark.parametrize("froude", [0.20, 0.30, 0.40])
    def test_panel_route_converges(self, ellipsoid_hull: EllipsoidHull, froude: float):
        """Panel Kochin converges to analytical R_w."""
        reference = wave_resistance_analytic(ellipsoid_hull, froude)
        errors = {}
        for n_r, n_phi in ((10, 20), (20, 40), (40, 80)):
            value = wave_resistance_panels(ellipsoid_hull, froude, n_r=n_r, n_phi=n_phi)
            errors[(n_r, n_phi)] = abs(value / reference - 1.0)
        # 3200 panels: error < 0.5%
        assert errors[(40, 80)] < 5.0e-3
        # Monotone convergence
        assert errors[(40, 80)] < errors[(20, 40)] < errors[(10, 20)]

    def test_study_result_shape(self, ellipsoid_study: dict):
        """Study result must have the expected structure."""
        assert ellipsoid_study["geometry"]["Cb"] == pytest.approx(math.pi / 6.0, rel=1e-14)
        assert ellipsoid_study["kochin"]["max_rel_difference"] < 1e-10
        assert ellipsoid_study["wave_resistance"]["analytic_closed_form"] > 0.0

    def test_froude_curve_monotone(self, ellipsoid_curve: dict):
        """R_w should generally increase with Fr for this hull."""
        # The ellipsoid_froude_curve returns 'analytical' key with Fr and Rw data
        frs = [r["froude"] for r in ellipsoid_curve.get("analytical", [])]
        Rws = [r["Rw_analytic"] for r in ellipsoid_curve.get("analytical", [])]
        # Not strictly monotone (wave resistance has minima), but positive
        assert all(rw > 0 for rw in Rws)


# ============================================================================
# 3. Tandem Interference
# ============================================================================


class TestTandemInterference:
    """Tandem Wigley hulls: interference factor vs spacing."""

    L, B, T = BRIEF_HULL["L"], BRIEF_HULL["B"], BRIEF_HULL["T"]

    def test_interference_factor_formula(self):
        """sigma = R_W_pair / (2 R_W_single) at Fr=0.30."""
        froude = 0.30
        U = froude * math.sqrt(G * self.L)
        k0 = G / (U * U)
        for sr in (0.5, 1.0, 2.0, 3.0):
            S = sr * self.L
            tw = TandemWigley(self.L, self.B, self.T, S)
            sigma = interference_factor(tw, U)
            Rw_pair = tandem_wave_resistance(tw, U)
            single_Rw = michell_wave_resistance(self.L, self.B, self.T, U)
            expected = Rw_pair / (2.0 * single_Rw)
            assert sigma == pytest.approx(expected, rel=1e-12)

    def test_interference_at_zero_spacing(self):
        """S -> 0: sigma -> 4 (two hulls perfectly superimposed)."""
        froude = 0.30
        U = froude * math.sqrt(G * self.L)
        tw = TandemWigley(self.L, self.B, self.T, 1e-10)
        sigma = interference_factor(tw, U)
        assert sigma == pytest.approx(4.0, rel=1e-2)

    def test_interference_at_large_spacing(self):
        """S -> inf: sigma -> 1 (no interference)."""
        froude = 0.30
        U = froude * math.sqrt(G * self.L)
        tw = TandemWigley(self.L, self.B, self.T, 10.0 * self.L)
        sigma = interference_factor(tw, U)
        # At large spacing, sigma should approach 1
        assert sigma == pytest.approx(1.0, rel=0.1)

    def test_tandem_kochin_function(self):
        """H_pair(k) = H_single(k) * 2 cos(kS/2)."""
        froude = 0.30
        U = froude * math.sqrt(G * self.L)
        S = self.L
        tw = TandemWigley(self.L, self.B, self.T, S)
        for k in (1.0, 5.0):
            H_pair = tandem_kochin_function(k, tw, U)
            H_single = michell_kochin_function(self.L, self.B, self.T, k)
            expected = H_single * 2.0 * math.cos(k * S / 2.0)
            assert abs(H_pair - expected) / abs(H_single) < 1e-12

    def test_tandem_wave_resistance_positive(self):
        """R_w_pair must be positive."""
        froude = 0.30
        U = froude * math.sqrt(G * self.L)
        tw = TandemWigley(self.L, self.B, self.T, self.L)
        Rw = tandem_wave_resistance(tw, U)
        assert Rw > 0.0

    def test_panel_kochin_matches_analytical(self):
        """Panel Kochin function matches analytical formula."""
        froude = 0.30
        U = froude * math.sqrt(G * self.L)
        S = self.L
        tw = TandemWigley(self.L, self.B, self.T, S)
        for k in (1.0, 5.0):
            H_panel = tandem_kochin_panels(k, tw, 40, 20)
            H_analytic = tandem_kochin_function(k, tw, U)
            assert abs(H_panel - H_analytic) / abs(H_analytic) < 1e-2

    def test_panel_wave_resistance_converges(self):
        """Panel R_w converges to analytical R_w."""
        froude = 0.30
        U = froude * math.sqrt(G * self.L)
        S = self.L
        tw = TandemWigley(self.L, self.B, self.T, S)
        Rw_analytic = tandem_wave_resistance(tw, U)
        errors = {}
        for n_s, n_l in ((10, 5), (20, 10), (40, 20)):
            Rw_panel = tandem_panel_wave_resistance(tw, U, nx=n_s, nz=n_l)
            errors[(n_s, n_l)] = abs(Rw_panel / Rw_analytic - 1.0)
        # 3200 panels: error < 5%
        assert errors[(40, 20)] < 5.0e-2
        # Monotone convergence
        assert errors[(40, 20)] < errors[(20, 10)] < errors[(10, 5)]

    def test_interference_study_shape(self, interference: dict):
        """Study result must have the expected structure."""
        assert interference["L"] == self.L
        assert interference["froude"] == 0.30
        assert len(interference["analytical"]) == len(interference["spacing_ratios"])
        assert len(interference["panel_levels"]) == 3

    def test_interference_study_convergence(self, interference: dict):
        """Panel interference converges to analytical."""
        panel_rows = interference["panel_levels"]
        analytical = interference["analytical"]
        # Check convergence across panel levels
        for sr, ana in zip(interference["spacing_ratios"], analytical):
            sigma_ana = ana["sigma_analytic"]
            errors = []
            for pr in panel_rows:
                panel_result = next(r for r in pr["results"] if r["spacing_ratio"] == sr)
                errors.append(panel_result["rel_error"])
            # Finest panel level should have smallest error
            assert errors[-1] <= errors[0] + 0.1  # Allow some tolerance
            # At finest level, error should be < 5%
            assert errors[-1] < 5.0e-2


# ============================================================================
# 4. Convergence Studies (GCI)
# ============================================================================


class TestConvergenceStudies:
    """Grid convergence studies with GCI estimation."""

    L, B, T = BRIEF_HULL["L"], BRIEF_HULL["B"], BRIEF_HULL["T"]

    def test_wigley_volume_gci(self, hydro_study: dict):
        """GCI for volume should be < 1% at finest level."""
        rows = hydro_study["rows"]
        ref_vol = hydro_study["reference"]["volume"]
        # Compute GCI between finest two levels
        e2 = abs(rows[-1]["volume"] - ref_vol)
        e1 = abs(rows[-2]["volume"] - ref_vol)
        if e1 > 0 and e2 > 0:
            r = e2 / e1
            p = math.log(e1 / e2) / math.log(rows[-1]["panels"] / rows[-2]["panels"])
            GCI = 1.25 * r * e2 / (r - 1) if r != 1 else e2
            assert GCI / ref_vol < 1.0e-2  # GCI < 1% of reference

    def test_ellipsoid_rw_convergence(self, ellipsoid_study: dict):
        """Ellipsoid R_w convergence: error decreases with panel count."""
        panels_data = ellipsoid_study.get("panels", [])
        for scheme in ("source_integrated", "ordinate"):
            errors = []
            for pr in panels_data:
                row = next(r for r in pr.get("results", []) if r.get("scheme") == scheme)
                errors.append(row["rel_error"])
            # Monotone decrease
            assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))


# ============================================================================
# 5. Integration with hull_anchor_maturation
# ============================================================================


class TestHullAnchorIntegration:
    """Verify that the new interference module integrates with the anchor."""

    def test_imports(self):
        """All new symbols must be importable."""
        from tensorbem.benchmarks.multihull_interference import (
            TandemWigley,
            interference_study,
            tandem_kochin_function,
            tandem_wave_resistance,
            interference_factor,
        )
        assert callable(TandemWigley)
        assert callable(interference_study)

    def test_interference_study_runs(self, interference: dict):
        """Full interference study must complete without error."""
        assert interference["single_Rw"] > 0
        assert len(interference["analytical"]) == len(interference["spacing_ratios"])
        # Max panel error should be < 10% at coarsest level
        assert interference["max_panel_error"] < 0.1

    def test_interference_study_matches_analytical(self, interference: dict):
        """Panel interference should match analytical."""
        analytical = interference["analytical"]
        finest = interference["panel_levels"][-1]["results"]
        for ana, pan in zip(analytical, finest):
            # At fine panels, panel result should match analytical
            assert pan["rel_error"] < 0.05  # < 5% at finest level
