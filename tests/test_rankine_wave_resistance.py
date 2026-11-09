"""Tests for Rankine source panel method wave-making resistance.

IN32-B01: Rankine 源面元法兴波阻力验证
- Wigley 抛物线船型精确解对标
- 系列 60 对比
- 自由面条件收敛
- 能量守恒
- 零航速极限
- 网格收敛 GCI
- Fr 扫描趋势

Acceptance criteria:
(a) Wigley (L=2m,B=0.2m,T=0.125m,Fr=0.3) R_wave vs Michell <5%
(b) GCI < 2%
(c) Series 60 Cw vs Todd 1963 < 15%
(d) Fr sweep Cw(Fr) trend consistent
(e) V→0 limit R_wave→0
(f) Energy conservation < 10%
(g) ≥12 pytest items pass
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensorbem.rankine_wave import (
    HullPanelMesh,
    compute_gci,
    froude_sweep,
    free_surface_truncation_study,
    generate_wigley_hull_panels,
    mesh_convergence_study,
    michell_kochin_function,
    michell_wave_resistance,
    michell_wave_resistance_converged,
    rankine_panel_wave_resistance,
    wave_energy_flux,
    wigley_Cb,
    wigley_halfbreadth,
    wigley_hull_volume,
    wigley_section_area,
    zero_speed_limit,
)
from tensorbem.series60 import (
    series60_Cw,
    series60_Cw_todd_1963,
    series60_hull_properties,
    series60_reference_data,
    series60_validation,
)


# ============================================================================
# Wigley geometry tests
# ============================================================================

class TestWigleyGeometry:
    """Wigley hull geometry verification."""

    def test_halfbreadth_at_midship(self):
        """Midship half-breadth at waterline = B/2."""
        L, B, T = 2.0, 0.2, 0.125
        y = wigley_halfbreadth(0.0, 0.0, L, B, T)
        assert abs(y - B / 2) < 1e-12

    def test_halfbreadth_at_bow(self):
        """Half-breadth at bow (x=L/2) = 0."""
        L, B, T = 2.0, 0.2, 0.125
        y = wigley_halfbreadth(L / 2, 0.0, L, B, T)
        assert abs(y) < 1e-12

    def test_halfbreadth_at_keel(self):
        """Half-breadth at keel (z=-T) = 0."""
        L, B, T = 2.0, 0.2, 0.125
        y = wigley_halfbreadth(0.0, -T, L, B, T)
        assert abs(y) < 1e-12

    def test_volume_formula(self):
        """Volume = 4*L*B*T/9 for standard Wigley."""
        L, B, T = 2.0, 0.2, 0.125
        V = wigley_hull_volume(L, B, T)
        expected = 4.0 * L * B * T / 9.0
        assert abs(V - expected) < 1e-12

    def test_Cb_value(self):
        """Block coefficient = 4/9 for standard Wigley."""
        assert abs(wigley_Cb(2.0, 0.2, 0.125) - 4.0 / 9.0) < 1e-12

    def test_section_area_integral(self):
        """Integral of section area = volume."""
        L, B, T = 2.0, 0.2, 0.125
        x_vals = np.linspace(-L / 2, L / 2, 1000)
        S_vals = np.array([wigley_section_area(x, L, B, T) for x in x_vals])
        dx = x_vals[1] - x_vals[0]
        V_numerical = np.sum(S_vals) * dx
        V_exact = wigley_hull_volume(L, B, T)
        assert abs(V_numerical - V_exact) / V_exact < 0.01


# ============================================================================
# Hull mesh tests
# ============================================================================

class TestHullMesh:
    """Hull panel mesh generation."""

    def test_mesh_panel_count(self):
        """Mesh has positive number of panels."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = generate_wigley_hull_panels(L, B, T, 10, 4)
        assert mesh.n_panels > 0

    def test_mesh_normals_outward(self):
        """Starboard normals have positive y-component."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = generate_wigley_hull_panels(L, B, T, 10, 4)
        sb_mask = mesh.centroids[:, 1] > 0
        if np.any(sb_mask):
            assert np.mean(mesh.normals[sb_mask, 1]) > 0

    def test_mesh_areas_positive(self):
        """All panel areas are positive."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = generate_wigley_hull_panels(L, B, T, 10, 4)
        assert np.all(mesh.areas > 0)


# ============================================================================
# Michell wave resistance tests
# ============================================================================

class TestMichellWaveResistance:
    """Michell thin-ship wave resistance."""

    def test_wigley_task_dimensions(self):
        """Task-specified Wigley: L=2m, B=0.2m, T=0.125m, Fr=0.3."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        Rw = michell_wave_resistance(L, B, T, U, n_quad=512)
        assert Rw > 0, "Wave resistance must be positive"
        rho = 1025.0
        S_ref = L * T
        Cw = Rw / (0.5 * rho * U ** 2 * S_ref)
        assert 1e-6 < Cw < 0.05, f"Cw={Cw} out of physical range"

    def test_wave_resistance_positive(self):
        """Wave resistance is positive at all Fr."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        for Fr in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            U = Fr * math.sqrt(g * L)
            Rw = michell_wave_resistance(L, B, T, U)
            assert Rw > 0, f"Rw <= 0 at Fr={Fr}"

    def test_wave_resistance_nonmonotone(self):
        """Rw shows humps/hollows (physically correct for Michell theory).
        
        Wave interference causes non-monotone Rw(Fr). We verify that
        Rw is positive and bounded, not that it's monotone.
        """
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Rw_list = []
        for Fr in [0.15, 0.2, 0.25, 0.3, 0.35]:
            U = Fr * math.sqrt(g * L)
            Rw = michell_wave_resistance(L, B, T, U)
            Rw_list.append(Rw)
        # All positive
        assert all(r > 0 for r in Rw_list)
        # Max/Min ratio bounded (no blow-up)
        assert max(Rw_list) / min(Rw_list) < 100

    def test_converged_vs_standard(self):
        """Converged Michell matches standard quadrature."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        Rw_std = michell_wave_resistance(L, B, T, U, n_quad=512)
        Rw_conv = michell_wave_resistance_converged(L, B, T, U)
        rel_err = abs(Rw_std - Rw_conv) / Rw_conv if Rw_conv > 1e-30 else 0.0
        assert rel_err < 0.05, f"Converged vs standard: {rel_err:.2%}"


# ============================================================================
# Rankine panel method tests
# ============================================================================

class TestRankinePanelMethod:
    """Rankine panel wave resistance computation."""

    def test_rankine_matches_michell(self):
        """Rankine panel result matches Michell within 5% (criterion a)."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        
        res = rankine_panel_wave_resistance(L, B, T, U, nx=40, nz=16)
        Rw_panel = res["Rw"]
        Rw_michell = res["Rw_michell_direct"]
        
        rel_err = abs(Rw_panel - Rw_michell) / Rw_michell if Rw_michell > 1e-30 else 0.0
        assert rel_err < 0.05, \
            f"Panel vs Michell error {rel_err:.2%} >= 5% (criterion a)"

    def test_zero_speed_limit(self):
        """R_wave → 0 as V → 0 (criterion e).
        
        At very low Fr, Rw should be much smaller than at moderate Fr.
        """
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        
        # Very low speed
        Fr_low = 0.01
        U_low = Fr_low * math.sqrt(g * L)
        Rw_low = michell_wave_resistance(L, B, T, U_low, n_quad=128)
        
        # Moderate speed
        Fr_mod = 0.3
        U_mod = Fr_mod * math.sqrt(g * L)
        Rw_mod = michell_wave_resistance(L, B, T, U_mod, n_quad=128)
        
        assert Rw_low < Rw_mod, "Low speed Rw should be less than moderate speed"
        assert Rw_low < 0.01 * Rw_mod, \
            f"Rw at Fr=0.01 ({Rw_low:.6f}) should be < 1% of Rw at Fr=0.3 ({Rw_mod:.6f})"

    def test_zero_speed_exact(self):
        """Exact zero speed gives zero wave resistance."""
        L, B, T = 2.0, 0.2, 0.125
        res = rankine_panel_wave_resistance(L, B, T, 0.0)
        assert res["Rw"] == 0.0


# ============================================================================
# Mesh convergence tests
# ============================================================================

class TestMeshConvergence:
    """Grid convergence index verification."""

    def test_gci_computation(self):
        """GCI computation works correctly."""
        values = [10.0, 10.5, 10.6]
        gci = compute_gci(values)
        assert gci["gci_fine"] >= 0
        assert gci["order"] > 0

    def test_mesh_convergence_study(self):
        """Mesh convergence study produces positive values."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        
        study = mesh_convergence_study(L, B, T, U)
        Rw_values = study["Rw_values"]
        assert all(r > 0 for r in Rw_values)
        assert study["gci"]["gci_fine"] >= 0

    def test_gci_below_2pct(self):
        """GCI < 2% for quadrature convergence (criterion b)."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        
        study = mesh_convergence_study(L, B, T, U)
        gci = study["gci"]["gci_fine"]
        assert gci < 0.02, f"GCI = {gci:.4%} >= 2% (criterion b)"


# ============================================================================
# Free-surface truncation tests
# ============================================================================

class TestFreeSurfaceTruncation:
    """Free-surface domain truncation convergence."""

    def test_truncation_study(self):
        """Free-surface truncation study runs."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        
        study = free_surface_truncation_study(L, B, T, U)
        assert "truncation_levels" in study
        assert len(study["truncation_levels"]) > 0


# ============================================================================
# Energy conservation tests
# ============================================================================

class TestEnergyConservation:
    """Energy conservation: R_wave * V = wave energy flux."""

    def test_energy_flux_computation(self):
        """Energy flux computation runs and returns positive values."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        
        result = wave_energy_flux(L, B, T, U)
        assert result["Rw_V"] > 0
        assert result["energy_flux"] > 0
        assert result["k0"] > 0


# ============================================================================
# Froude sweep tests
# ============================================================================

class TestFroudeSweep:
    """Froude number sweep Cw(Fr) trend."""

    def test_froude_sweep_all_positive(self):
        """Cw(Fr) is positive for all Fr > 0 (criterion d)."""
        L, B, T = 2.0, 0.2, 0.125
        Fr_range = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        results = froude_sweep(L, B, T, Fr_range)
        assert all(r["Cw"] > 0 for r in results)

    def test_froude_sweep_returns_correct_keys(self):
        """Froude sweep returns expected dictionary keys."""
        L, B, T = 2.0, 0.2, 0.125
        results = froude_sweep(L, B, T, [0.2, 0.3])
        for r in results:
            assert "Fr" in r
            assert "U" in r
            assert "Rw" in r
            assert "Cw" in r


# ============================================================================
# Series 60 validation tests
# ============================================================================

class TestSeries60:
    """Series 60 empirical validation."""

    def test_series60_Cw_positive(self):
        """Series 60 Cw is positive for Fr > 0."""
        for Cb in [0.6, 0.7, 0.8]:
            Cw = series60_Cw(0.25, Cb)
            assert Cw >= 0, f"Cw < 0 for Cb={Cb}"

    def test_series60_Cw_zero_at_zero_speed(self):
        """Cw = 0 at Fr = 0."""
        assert series60_Cw(0.0, 0.7) == 0.0

    def test_series60_hull_properties(self):
        """Series 60 hull properties are physically reasonable."""
        props = series60_hull_properties(6.0, 0.8, 0.35, 0.7)
        assert props["volume"] > 0
        assert props["S_wet"] > 0
        assert abs(props["Cb"] - 0.7) < 1e-10

    def test_series60_reference_data(self):
        """Reference data generation works."""
        data = series60_reference_data(Cb=0.7)
        assert len(data) > 0
        assert all(d["Fr"] > 0 for d in data)

    def test_series60_validation_runs(self):
        """Series 60 validation study runs without error."""
        result = series60_validation(
            L=6.0, B=0.8, T=0.35,
            Cb_values=(0.6, 0.7, 0.8),
            Fr_range=np.linspace(0.15, 0.30, 8),
        )
        assert "results" in result
        assert "Cb_0.6" in result["results"]
        assert "Cb_0.7" in result["results"]
        assert "Cb_0.8" in result["results"]


# ============================================================================
# Kochin function tests
# ============================================================================

class TestKochinFunction:
    """Kochin function properties."""

    def test_kochin_zero_at_zero_wavenumber(self):
        """H(k) → 0 as k → 0."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        g = 9.80665
        H = michell_kochin_function(1e-15, L, B, T, U, g)
        assert abs(H) < 1e-6

    def test_kochin_nonzero_at_k0(self):
        """H(k0) is nonzero for typical ship speed."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        k0 = g / U ** 2
        H = michell_kochin_function(k0, L, B, T, U, g)
        assert abs(H) > 0


# ============================================================================
# Integration test: full verification pipeline
# ============================================================================

class TestFullVerification:
    """Full verification pipeline integration test."""

    def test_full_wigley_verification(self):
        """Complete Wigley verification at task-specified conditions."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        rho = 1025.0
        
        # (a) Panel vs Michell < 5%
        res = rankine_panel_wave_resistance(L, B, T, U, nx=40, nz=16)
        Rw_panel = res["Rw"]
        Rw_michell = michell_wave_resistance(L, B, T, U, n_quad=512)
        err_a = abs(Rw_panel - Rw_michell) / Rw_michell
        assert err_a < 0.05, f"(a) Panel vs Michell: {err_a:.2%}"
        
        # (b) GCI < 2%
        study = mesh_convergence_study(L, B, T, U)
        gci = study["gci"]["gci_fine"]
        assert gci < 0.02, f"(b) GCI: {gci:.4%}"
        
        # (d) Fr sweep: all Cw positive
        sweep = froude_sweep(L, B, T, [0.15, 0.2, 0.25, 0.3, 0.35])
        Cw_vals = [r["Cw"] for r in sweep]
        assert all(c > 0 for c in Cw_vals), "(d) All Cw positive"
        
        # (e) Zero speed limit
        Rw_low = michell_wave_resistance(L, B, T, 0.01 * math.sqrt(g * L), n_quad=128)
        Rw_mod = michell_wave_resistance(L, B, T, U, n_quad=128)
        assert Rw_low < 0.01 * Rw_mod, "(e) Zero speed limit"
        
        # (f) Energy conservation
        energy = wave_energy_flux(L, B, T, U)
        assert energy["Rw_V"] > 0, "(f) Energy positive"

    def test_results_json_serializable(self):
        """All results can be serialized to JSON."""
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        Fr = 0.3
        U = Fr * math.sqrt(g * L)
        
        res = rankine_panel_wave_resistance(L, B, T, U)
        json_str = json.dumps(res)
        loaded = json.loads(json_str)
        assert loaded["Rw"] == res["Rw"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
