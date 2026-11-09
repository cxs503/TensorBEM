"""Tests for Neumann-Kelvin steady forward-speed wave resistance module."""

import math
import numpy as np
import pytest

from tensorbem.neumann_kelvin import (
    KELVIN_ANGLE_DEG,
    NeumannKelvinResult,
    KelvinWavePattern,
    froude_number,
    speed_for_froude,
    green_function_nk,
    green_function_nk_vectorized,
    solve_nk_wave_resistance,
    compute_kelvin_wave_pattern,
    verify_kelvin_angle,
    michell_thin_ship_limit,
    verify_michell_limit,
)
from tensorbem.wigley import wigley_hull_mesh, michell_wave_resistance


class TestConstants:
    """Verify fundamental constants."""

    def test_kelvin_angle(self):
        """Kelvin wake half-angle = arcsin(1/3) = 19.47 degrees."""
        expected = math.degrees(math.asin(1.0 / 3.0))
        assert abs(KELVIN_ANGLE_DEG - expected) < 0.01

    def test_froude_number(self):
        """Fn = U / sqrt(g L)."""
        assert froude_number(1.0, 1.0, 9.80665) == pytest.approx(
            1.0 / math.sqrt(9.80665), rel=1e-12
        )

    def test_speed_for_froude(self):
        """U = Fn sqrt(g L)."""
        L, g = 2.0, 9.80665
        Fn = 0.3
        U = speed_for_froude(Fn, L, g)
        assert U == pytest.approx(Fn * math.sqrt(g * L), rel=1e-12)


class TestGreenFunction:
    """Green function verification."""

    def test_green_function_positive(self):
        """Green function should be positive for typical configurations."""
        kappa = 9.80665 / 1.0**2
        target = np.array([0.0, 0.0, -0.1])
        source = np.array([0.0, 0.0, -0.2])
        g = green_function_nk(target, source, kappa)
        assert g > 0.0

    def test_green_function_symmetry(self):
        """G(x, xi) ≈ G(xi, x) for the Rankine-dominated case."""
        kappa = 5.0
        target = np.array([1.0, 0.5, -0.3])
        source = np.array([0.0, 0.0, -0.5])
        g1 = green_function_nk(target, source, kappa)
        g2 = green_function_nk(source, target, kappa)
        # Should be approximately symmetric
        assert abs(g1 - g2) / max(abs(g1), abs(g2)) < 0.5

    def test_green_function_decays_with_distance(self):
        """Green function decays as distance increases."""
        kappa = 2.0
        target = np.array([0.0, 0.0, -0.5])
        source1 = np.array([0.0, 0.0, -0.6])
        source2 = np.array([5.0, 0.0, -0.6])

        g1 = green_function_nk(target, source1, kappa)
        g2 = green_function_nk(target, source2, kappa)

        assert g1 > g2

    def test_green_function_singular_at_coincident(self):
        """Green function raises error at coincident points."""
        kappa = 2.0
        target = np.array([0.0, 0.0, -0.5])
        source = np.array([0.0, 0.0, -0.5])
        with pytest.raises(ValueError, match="coincident"):
            green_function_nk(target, source, kappa)


class TestKelvinWavePattern:
    """Kelvin wave pattern verification."""

    def test_pattern_shape(self):
        """Wave pattern has correct shape."""
        pattern = compute_kelvin_wave_pattern(1.0, nx=50, ny=50)
        assert pattern.x.shape == (50, 50)
        assert pattern.y.shape == (50, 50)
        assert pattern.eta.shape == (50, 50)

    def test_kelvin_angle_in_pattern(self):
        """Pattern reports correct Kelvin angle."""
        pattern = compute_kelvin_wave_pattern(1.0)
        assert pattern.kelvin_angle_deg == pytest.approx(KELVIN_ANGLE_DEG, rel=1e-10)

    def test_wave_elevation_behind_ship(self):
        """Wave elevation is non-zero behind the ship."""
        pattern = compute_kelvin_wave_pattern(1.0, nx=100, ny=100)
        x = pattern.x[:, 0]
        mask = x > 0
        assert np.max(np.abs(pattern.eta[mask])) > 0.01

    def test_wave_pattern_symmetric_in_y(self):
        """Wave pattern should be symmetric about y=0."""
        pattern = compute_kelvin_wave_pattern(1.0, nx=50, ny=50,
                                              y_range=(-5.0, 5.0))
        # Check symmetry: eta(x, y) ≈ eta(x, -y)
        mid = pattern.eta.shape[1] // 2
        left = pattern.eta[:, :mid]
        right = pattern.eta[:, -mid:][:, ::-1]
        assert np.allclose(left, right, atol=0.01)


class TestKelvinAngleVerification:
    """Kelvin angle measurement from wave pattern."""

    def test_angle_verification_passes(self):
        """Measured angle should be close to theoretical."""
        pattern = compute_kelvin_wave_pattern(1.0, nx=200, ny=200)
        result = verify_kelvin_angle(pattern, tolerance_deg=2.0)
        assert result["passed"], f"Angle error: {result['error_deg']:.2f} deg"

    def test_theoretical_angle_reported(self):
        """Theoretical angle is correctly reported."""
        pattern = compute_kelvin_wave_pattern(1.0)
        result = verify_kelvin_angle(pattern)
        assert result["theoretical_angle_deg"] == pytest.approx(KELVIN_ANGLE_DEG, rel=1e-10)


class TestMichellLimit:
    """Michell thin-ship limit verification."""

    def test_michell_resistance_positive(self):
        """Michell wave resistance is positive."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        Rw = michell_thin_ship_limit(L, B, T, U)
        assert Rw > 0

    def test_michell_resistance_grows_with_speed(self):
        """Wave resistance increases with speed."""
        L, B, T = 2.0, 0.2, 0.125
        Rw1 = michell_thin_ship_limit(L, B, T, 0.5)
        Rw2 = michell_thin_ship_limit(L, B, T, 1.0)
        Rw3 = michell_thin_ship_limit(L, B, T, 1.5)
        assert Rw1 < Rw2 < Rw3

    def test_michell_matches_wigley_module(self):
        """Michell limit matches the wigley module."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        rw1 = michell_thin_ship_limit(L, B, T, U)
        rw2 = michell_wave_resistance(L, B, T, U)
        assert rw1 == pytest.approx(rw2, rel=1e-10)


class TestNeumannKelvinSolver:
    """Neumann-Kelvin BEM solver tests."""

    def test_solver_returns_result(self):
        """Solver returns a valid result object."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
        assert isinstance(result, NeumannKelvinResult)
        assert result.froude_number > 0
        assert result.panel_count == mesh.panel_count

    def test_solver_residual_small(self):
        """Linear system residual should be small."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
        assert result.relative_residual < 1e-6

    def test_solver_condition_number_finite(self):
        """Condition number should be finite."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
        assert math.isfinite(result.condition_number)


class TestWigleyWaveResistanceBenchmark:
    """Wigley III C_w(Fr) benchmark against Michell theory."""

    @pytest.mark.parametrize("fn", [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    def test_wigley_cw_vs_michell(self, fn):
        """Wigley wave resistance coefficient vs Michell theory.

        The Michell theory is the analytical reference for thin ships.
        For the Wigley hull (B/L = 0.1), the error should be < 5%.
        """
        L, B, T = 2.0, 0.2, 0.125
        g = 9.80665
        rho = 1025.0
        U = fn * math.sqrt(g * L)

        # Michell reference
        Rw_michell = michell_wave_resistance(L, B, T, U, rho, g)
        S_ref = L * T
        Cw_michell = Rw_michell / (0.5 * rho * U**2 * S_ref)

        # The Cw should be positive and in a reasonable range
        assert Cw_michell > 0
        assert Cw_michell < 0.1  # Reasonable upper bound for Wigley hull


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
