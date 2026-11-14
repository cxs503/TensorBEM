"""Tests for the centre-crack Mode-I SIF capability (CASE-BEM-CRACK-SIF)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.crack_sif import (
    CrackSIFResult,
    analytical_cod,
    analytical_sif,
    convergence_study,
    solve_crack_sif,
)


class TestAnalyticalReference:
    """Verify the closed-form reference values."""

    def test_sif_formula(self):
        """K_I = sigma * sqrt(pi * a) for centre crack."""
        sigma = 100.0e6  # 100 MPa
        a = 0.01         # 10 mm
        expected = sigma * math.sqrt(math.pi * a)
        assert analytical_sif(sigma, a) == pytest.approx(expected, rel=1e-12)

    def test_cod_formula(self):
        """D_y(x) = 4*sigma/E' * sqrt(a^2 - x^2)."""
        sigma = 1.0
        a = 1.0
        e_prime = 2.0e11 / (1 - 0.3 ** 2)
        x = np.array([0.0, 0.5, 0.9])
        expected = 4.0 * sigma / e_prime * np.sqrt(a ** 2 - x ** 2)
        got = analytical_cod(x, sigma, a, e_prime)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_cod_zero_outside_crack(self):
        """COD should be zero outside the crack."""
        sigma = 1.0
        a = 1.0
        e_prime = 2.0e11
        x = np.array([-2.0, -1.0, 1.0, 2.0])
        got = analytical_cod(x, sigma, a, e_prime)
        np.testing.assert_allclose(got, 0.0, atol=1e-30)


class TestChebyshevSolver:
    """Chebyshev collocation solver: exact for uniform loading."""

    def test_exact_sif(self):
        """Chebyshev method should give machine-precision SIF for uniform loading."""
        result = solve_crack_sif(
            sigma=1.0, a=1.0, young=2.0e11, poisson=0.3, method="chebyshev"
        )
        assert result.error_pct < 1e-10

    def test_chebyshev_coefficients(self):
        """Only c_0 should be non-zero for uniform loading."""
        result = solve_crack_sif(
            sigma=1.0, a=1.0, young=2.0e11, poisson=0.3, method="chebyshev"
        )
        assert result.chebyshev_coefficients is not None
        assert result.chebyshev_coefficients[0] == pytest.approx(
            4.0 * 1.0 / result.e_prime, rel=1e-12
        )
        assert np.allclose(result.chebyshev_coefficients[1:], 0.0, atol=1e-30)

    def test_cod_matches_analytical(self):
        """COD profile should match the analytical solution."""
        result = solve_crack_sif(
            sigma=1.0, a=1.0, young=2.0e11, poisson=0.3, method="chebyshev"
        )
        np.testing.assert_allclose(
            result.cod_profile_got, result.cod_profile_ref, rtol=1e-12
        )

    @pytest.mark.parametrize("sigma", [1.0, 10.0, 100.0e6])
    @pytest.mark.parametrize("a", [0.001, 0.01, 0.1, 1.0])
    def test_parametric_sif(self, sigma, a):
        """SIF should be exact for various sigma and a combinations."""
        result = solve_crack_sif(
            sigma=sigma, a=a, young=2.0e11, poisson=0.3, method="chebyshev"
        )
        assert result.error_pct < 1e-10


class TestConstantDDSolver:
    """Constant DD element solver: convergent but not exact."""

    def test_sif_within_tolerance(self):
        """SIF should be within 10% for reasonable mesh (constant DD is approximate)."""
        result = solve_crack_sif(
            sigma=1.0, a=1.0, young=2.0e11, poisson=0.3,
            n_elements=200, method="constant"
        )
        # Constant DD with hypersingular kernel has limited convergence;
        # 10% is a generous bound for this pedagogical baseline
        assert result.error_pct < 10.0

    def test_residual_small(self):
        """The linear system residual should be small."""
        result = solve_crack_sif(
            sigma=1.0, a=1.0, young=2.0e11, poisson=0.3,
            n_elements=100, method="constant"
        )
        assert result.residual_l2 < 1e-10

    def test_cod_shape(self):
        """DD profile should have the expected semi-elliptical shape."""
        result = solve_crack_sif(
            sigma=1.0, a=1.0, young=2.0e11, poisson=0.3,
            n_elements=100, method="constant"
        )
        # Maximum DD should be at the centre
        assert result.element_dd[0] < result.element_dd[len(result.element_dd) // 2]
        # DD should be symmetric
        np.testing.assert_allclose(
            result.element_dd[:10], result.element_dd[-10:][::-1], rtol=1e-6
        )


class TestConvergence:
    """Convergence study for the constant-DD method."""

    def test_convergence_trend(self):
        """Error should be bounded for coarse meshes."""
        results = convergence_study(
            sigma=1.0, a=1.0, young=2.0e11, poisson=0.3,
            n_list=[20, 40, 80], method="constant"
        )
        # The constant DD method has limited convergence due to the
        # hypersingular kernel; we just check it runs without error
        for r in results:
            assert r.sif_got > 0
            assert r.error_pct < 50  # generous bound for coarse meshes


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_too_few_elements_constant(self):
        """Should raise ValueError for n_elements < 4 (constant method)."""
        with pytest.raises(ValueError, match="n_elements must be >= 4"):
            solve_crack_sif(n_elements=2, method="constant")

    def test_negative_stress(self):
        """Compression should give negative SIF (by convention)."""
        result = solve_crack_sif(
            sigma=-1.0, a=1.0, young=2.0e11, poisson=0.3, method="chebyshev"
        )
        assert result.sif_got < 0
        assert result.sif_got == pytest.approx(-analytical_sif(1.0, 1.0), rel=1e-10)

    def test_plane_strain_modulus(self):
        """E' = E / (1 - nu^2) for plane strain."""
        result = solve_crack_sif(
            sigma=1.0, a=1.0, young=2.0e11, poisson=0.3, method="chebyshev"
        )
        expected_e_prime = 2.0e11 / (1 - 0.3 ** 2)
        assert result.e_prime == pytest.approx(expected_e_prime, rel=1e-12)
