"""Tests for Wigley hull wave resistance computation."""

import math
import numpy as np
import pytest

from tensorbem.benchmarks.wigley_verification import converged_wave_resistance
from tensorbem.wigley import (
    WigleyHullMesh,
    WigleyWaveResistanceResult,
    compute_wigley_wave_resistance,
    mesh_convergence_study,
    michell_kochin_function,
    michell_wave_resistance,
    wave_resistance_kochin,
    wave_resistance_pressure,
    wigley_halfbreadths,
    wigley_hull_mesh,
    wigley_hull_properties,
)


class TestWigleyHullMesh:
    """Tests for Wigley hull mesh generation."""

    def test_mesh_creation(self):
        """Test that mesh can be created."""
        mesh = wigley_hull_mesh(2.0, 0.2, 0.125, 8, 4)
        assert isinstance(mesh, WigleyHullMesh)
        assert mesh.panel_count > 0
        assert mesh.surface_area > 0

    def test_mesh_properties(self):
        """Test mesh geometric properties."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = wigley_hull_mesh(L, B, T, 8, 4)
        
        # Volume should be positive
        assert mesh.volume > 0
        
        # Surface area should be reasonable
        # Approximate: S ~ 2*L*T + pi*B*T/2
        S_approx = 2 * L * T + math.pi * B * T / 2
        assert 0.5 * S_approx < mesh.surface_area < 2.0 * S_approx

    def test_hull_properties(self):
        """Test analytical hull properties.

        Volume = 2 * (B/2) * (2L/3) * (2T/3) = 4LBT/9 and Cb = 4/9: the
        two normalised factors of the Wigley surface each integrate to 2/3,
        so 2/3 is Cp (and Cm), not Cb.  Verified independently against the
        meshed hull in ``tests/test_wigley_verification.py``.
        """
        L, B, T = 2.0, 0.2, 0.125
        props = wigley_hull_properties(L, B, T)

        expected_volume = 4.0 * L * B * T / 9.0
        assert abs(props["volume"] - expected_volume) < 1e-12

        assert abs(props["Cb"] - 4.0 / 9.0) < 1e-12
        assert abs(props["Cm"] - 2.0 / 3.0) < 1e-12
        assert abs(props["Cp"] - 2.0 / 3.0) < 1e-12
        assert abs(props["waterplane_area"] - 2.0 * L * B / 3.0) < 1e-12

    def test_halfbreadths(self):
        """Test Wigley half-breadth formula."""
        L, B, T = 2.0, 0.2, 0.125
        
        # At midship (x=0), waterline (z=0): y = B/2
        y = wigley_halfbreadths(0.0, 0.0, L, B, T)
        assert abs(y - B / 2) < 1e-10
        
        # At bow (x=L/2): y = 0
        y = wigley_halfbreadths(L / 2, 0.0, L, B, T)
        assert abs(y) < 1e-10
        
        # At keel (z=-T): y = 0
        y = wigley_halfbreadths(0.0, -T, L, B, T)
        assert abs(y) < 1e-10


class TestMichellTheory:
    """Tests for Michell's thin-ship theory."""

    def test_kochin_function_symmetry(self):
        """Test that Kochin function is purely imaginary for symmetric hull."""
        L, B, T = 2.0, 0.2, 0.125
        k = 5.0
        
        Hk = michell_kochin_function(L, B, T, k)
        
        # For symmetric Wigley hull, H(k) should be purely imaginary
        assert abs(Hk.real) < 1e-10 * abs(Hk.imag)

    def test_kochin_function_decay(self):
        """Test that Kochin function decays for large k."""
        L, B, T = 2.0, 0.2, 0.125
        
        Hk1 = michell_kochin_function(L, B, T, 5.0)
        Hk2 = michell_kochin_function(L, B, T, 10.0)
        
        # |H(k)| should decay as k increases
        assert abs(Hk2) < abs(Hk1)

    def test_wave_resistance_positive(self):
        """Test that wave resistance is positive."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        
        Rw = michell_wave_resistance(L, B, T, U)
        assert Rw > 0

    def test_wave_resistance_zero_at_zero_speed(self):
        """Test that wave resistance is zero at zero speed."""
        L, B, T = 2.0, 0.2, 0.125
        U = 0.0
        
        Rw = michell_wave_resistance(L, B, T, U)
        assert Rw == 0.0


class TestWaveResistance:
    """Tests for wave resistance computation."""

    def test_two_paths_exist(self):
        """Test that both wave resistance paths can be computed."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        
        Rw_kochin = wave_resistance_kochin(L, B, T, U)
        mesh = wigley_hull_mesh(L, B, T, 8, 4)
        Rw_pressure = wave_resistance_pressure(mesh, L, B, T, U)
        
        assert Rw_kochin > 0
        # Pressure path may be negative due to thin-ship approximation limitations
        assert math.isfinite(Rw_pressure)

    def test_compute_result(self):
        """Test high-level compute function."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        
        result = compute_wigley_wave_resistance(L, B, T, U, nx=8, nz=4)
        
        assert isinstance(result, WigleyWaveResistanceResult)
        assert result.froude_number > 0
        assert result.wave_resistance_kochin > 0
        assert math.isfinite(result.wave_resistance_pressure)
        assert math.isfinite(result.relative_difference)

class TestMeshConvergence:
    """Tests for mesh convergence study."""

    def test_convergence_study(self):
        """Test that convergence study runs."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        mesh_levels = [(4, 2), (8, 4), (16, 8)]
        
        results = mesh_convergence_study(L, B, T, U, mesh_levels)
        
        assert len(results) == 3
        for r in results:
            assert "panel_count" in r
            assert "h_mean" in r
            assert "Cw_kochin" in r
            assert "Cw_pressure" in r

    def test_mesh_refinement(self):
        """Test that mesh refinement increases panel count."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        mesh_levels = [(4, 2), (8, 4), (16, 8)]
        
        results = mesh_convergence_study(L, B, T, U, mesh_levels)
        
        # Panel count should increase with refinement
        for i in range(1, len(results)):
            assert results[i]["panel_count"] > results[i - 1]["panel_count"]


class TestBenchmarkComparison:
    """Quantitative benchmarks for the Kochin (Michell) path.

    The earlier version of these tests only asserted ``0 < Cw < 1e-2``, which
    any implementation that returns a small positive number would pass, and the
    "published data" it referred to carried neither a reference area nor a
    source page.  They now pin the *computed* value to the converged
    quadrature reference (``benchmarks.wigley_verification``), i.e. they test
    the implementation instead of the plausibility of its magnitude.  The
    published-reference comparison is registered separately as an
    UNSUPPORTED claim in ``tests/test_wigley_verification.py`` because its
    normalization is not reproducible from the document.
    """

    @pytest.mark.parametrize("froude", [0.20, 0.30, 0.40])
    def test_kochin_matches_converged_reference(self, froude):
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        U = froude * math.sqrt(g * L)

        result = compute_wigley_wave_resistance(L, B, T, U, nx=16, nz=8)
        reference = converged_wave_resistance(L, B, T, U)

        assert result.wave_resistance_kochin == pytest.approx(reference, rel=1e-4)
        assert result.froude_number == pytest.approx(froude, rel=1e-12)

    def test_froude_number(self):
        """Fr = U / sqrt(g L)."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        g = 9.80665
        result = compute_wigley_wave_resistance(L, B, T, U, nx=16, nz=8)
        assert abs(result.froude_number - U / math.sqrt(g * L)) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
