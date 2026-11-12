"""Extended verification tests for Neumann-Kelvin module.

Fills gaps left by test_neumann_kelvin.py:

1. verify_michell_limit() — the B/L→0 convergence study function was imported
   but never exercised; this suite runs it and checks structural properties.

2. green_function_nk_vectorized vs _wave_integral_steady — documents the
   known discrepancy: the vectorized kernel uses a K_0 asymptotic approximation
   that differs significantly from the scalar quadrature-based version.

3. _wave_integral_steady — direct tests documenting actual behavior (the wave
   integral is negative for typical submerged configurations and oscillatory
   in R due to the Bessel function kernel).

4. Numerical stability audit — documents the known instability of the
   Neumann-Kelvin BEM solver: wave resistance values are orders of magnitude
   too large and oscillate with speed.  Tests verify structural properties
   (non-negativity clamping, finite condition numbers) rather than absolute
   accuracy.

5. Physical consistency — Rw is clamped to ≥ 0, condition numbers are finite,
   residuals are small.

All quantities are SI.  Tests use small meshes to stay lightweight.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.neumann_kelvin import (
    steady_free_surface_residual,
    KELVIN_ANGLE_DEG,
    NeumannKelvinResult,
    KelvinWavePattern,
    _wave_integral_steady,
    froude_number,
    green_function_nk,
    green_function_nk_vectorized,
    solve_nk_wave_resistance,
    verify_michell_limit,
    compute_kelvin_wave_pattern,
    verify_kelvin_angle,
    michell_thin_ship_limit,
)
from tensorbem.wigley import wigley_hull_mesh, michell_wave_resistance


# ============================================================================
# 1. verify_michell_limit — convergence study function
# ============================================================================

class TestVerifyMichellLimit:
    """Exercise the verify_michell_limit() convergence study."""

    def test_returns_dict_with_expected_keys(self):
        """Result dict contains all documented keys."""
        result = verify_michell_limit(L=2.0, T=0.125, U=1.0, B_ratios=(0.1, 0.05))
        for key in ("L", "T", "U", "froude_number", "B_ratios", "results",
                     "converging", "finest_error", "passed", "tolerance"):
            assert key in result, f"Missing key: {key}"

    def test_results_list_has_one_entry_per_B_ratio(self):
        """Each B/L ratio produces one result entry."""
        B_ratios = (0.15, 0.1, 0.05)
        result = verify_michell_limit(L=2.0, T=0.125, U=1.0, B_ratios=B_ratios)
        assert len(result["results"]) == len(B_ratios)

    def test_each_result_has_required_fields(self):
        """Each entry has B_over_L, B, Rw_michell, Rw_nk, rel_error, panel_count."""
        result = verify_michell_limit(L=2.0, T=0.125, U=1.0, B_ratios=(0.1,))
        entry = result["results"][0]
        for key in ("B_over_L", "B", "Rw_michell", "Rw_nk", "rel_error", "panel_count"):
            assert key in entry, f"Missing key in result entry: {key}"

    def test_michell_resistance_positive_in_study(self):
        """Michell reference resistance is positive for all B/L."""
        result = verify_michell_limit(L=2.0, T=0.125, U=1.0,
                                       B_ratios=(0.2, 0.1, 0.05))
        for r in result["results"]:
            assert r["Rw_michell"] > 0, f"Rw_michell <= 0 at B/L={r['B_over_L']}"

    def test_froude_number_matches_direct_call(self):
        """froude_number in result matches froude_number(U, L, g)."""
        L, U, g = 2.0, 1.0, 9.80665
        result = verify_michell_limit(L=L, T=0.125, U=U, g=g)
        expected = froude_number(U, L, g)
        assert result["froude_number"] == pytest.approx(expected, rel=1e-12)

    def test_panel_count_increases_with_B(self):
        """Larger B/L produces more panels (finer mesh)."""
        result = verify_michell_limit(L=2.0, T=0.125, U=1.0,
                                       B_ratios=(0.05, 0.1, 0.2))
        panels = [r["panel_count"] for r in result["results"]]
        for i in range(1, len(panels)):
            assert panels[i] >= panels[i - 1], \
                f"Panel count decreased: {panels[i-1]} -> {panels[i]}"


# ============================================================================
# 2. Vectorized vs scalar Green function — documented discrepancy
# ============================================================================

class TestGreenFunctionConsistency:
    """Scalar and vectorized Green functions: document the known discrepancy.

    The vectorized kernel uses a K_0 asymptotic approximation that differs
    significantly from the scalar quadrature-based _wave_integral_steady.
    """

    def test_both_versions_finite(self):
        """Both versions return finite values for typical inputs."""
        kappa = 5.0
        target = np.array([[0.5, 0.3, -0.2]])
        source = np.array([[0.0, 0.0, -0.5]])
        g_scalar = green_function_nk(target[0], source[0], kappa)
        g_vec = green_function_nk_vectorized(target, source, kappa)[0, 0]
        assert math.isfinite(g_scalar)
        assert math.isfinite(g_vec)

    def test_vectorized_shape(self):
        """Output shape is (n_targets, n_sources)."""
        kappa = 3.0
        targets = np.random.randn(5, 3)
        sources = np.random.randn(3, 3)
        targets[:, 2] = -np.abs(targets[:, 2]) - 0.01
        sources[:, 2] = -np.abs(sources[:, 2]) - 0.01
        g = green_function_nk_vectorized(targets, sources, kappa)
        assert g.shape == (5, 3)

    def test_vectorized_positive_for_close_pairs(self):
        """Vectorized G is positive for close submerged pairs."""
        kappa = 5.0
        target = np.array([[0.1, 0.0, -0.1]])
        source = np.array([[0.0, 0.0, -0.5]])
        g_vec = green_function_nk_vectorized(target, source, kappa)[0, 0]
        assert g_vec > 0, f"Vectorized G <= 0: {g_vec}"

    def test_scalar_can_be_negative(self):
        """Document: scalar G can be negative due to wave integral dominance.

        For certain source/target configurations, the wave integral G_W
        (which is negative) can dominate the Rankine part G_R, making
        the total G negative.  This is a known issue with the scalar
        Green function implementation.
        """
        kappa = 5.0
        target = np.array([0.5, 0.3, -0.2])
        source = np.array([0.0, 0.0, -0.5])
        g_scalar = green_function_nk(target, source, kappa)
        # Document: this can be negative
        assert math.isfinite(g_scalar)


# ============================================================================
# 3. _wave_integral_steady — direct tests (actual behavior)
# ============================================================================

class TestWaveIntegralSteady:
    """Direct tests of the wave integral component.

    Note: The wave integral G_W is typically NEGATIVE for submerged
    source/target pairs because it represents the wave disturbance.
    It is oscillatory in R due to the Bessel function J_0 kernel.
    """

    def test_finite_for_typical_config(self):
        """Wave integral is finite for typical submerged configuration."""
        val = _wave_integral_steady(R=1.0, z_sum=-0.5, kappa=5.0)
        assert math.isfinite(val), f"Got non-finite value: {val}"

    def test_magnitude_decays_at_large_R(self):
        """|Wave integral| is smaller at large R than at small R."""
        kappa = 5.0
        z_sum = -0.5
        v_small = _wave_integral_steady(R=0.5, z_sum=z_sum, kappa=kappa)
        v_large = _wave_integral_steady(R=10.0, z_sum=z_sum, kappa=kappa)
        assert abs(v_large) < abs(v_small), \
            f"Should decay: |{v_large}| vs |{v_small}|"

    def test_magnitude_decays_with_depth(self):
        """|Wave integral| decays as |z+zeta| increases."""
        kappa = 5.0
        R = 1.0
        v_shallow = _wave_integral_steady(R=R, z_sum=-0.2, kappa=kappa)
        v_deep = _wave_integral_steady(R=R, z_sum=-5.0, kappa=kappa)
        assert abs(v_deep) < abs(v_shallow), \
            f"Should decay with depth: |{v_deep}| vs |{v_shallow}|"

    def test_zero_at_coincident(self):
        """Wave integral returns 0 when R=0 and z_sum=0 (singularity handled)."""
        val = _wave_integral_steady(R=0.0, z_sum=0.0, kappa=5.0)
        assert val == 0.0

    def test_finite_for_large_kappa(self):
        """Wave integral remains finite for large kappa."""
        val = _wave_integral_steady(R=0.5, z_sum=-0.3, kappa=100.0)
        assert math.isfinite(val), f"Got non-finite value: {val}"

    def test_oscillatory_in_R(self):
        """Wave integral is oscillatory in R (Bessel J_0 kernel)."""
        kappa = 5.0
        z_sum = -0.5
        Rs = np.linspace(0.1, 5.0, 20)
        vals = [_wave_integral_steady(R, z_sum, kappa) for R in Rs]
        max_val = max(vals)
        min_val = min(vals)
        assert max_val - min_val > 0.05, \
            f"Expected oscillation: range = {max_val - min_val:.4f}"


# ============================================================================
# 4. Structural properties of the solver (not absolute accuracy)
# ============================================================================

class TestSolverStructuralProperties:
    """Structural properties that the solver should satisfy regardless of
    the numerical accuracy of the underlying Green function approximation.
    """

    def test_result_is_dataclass(self):
        """Solver returns a NeumannKelvinResult instance."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
        assert isinstance(result, NeumannKelvinResult)

    def test_rw_non_negative(self):
        """Wave resistance is clamped to >= 0."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        for U in [0.3, 0.5, 1.0, 1.5]:
            result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
            assert result.wave_resistance >= 0, f"Rw < 0 at U={U}"

    def test_cw_non_negative(self):
        """Wave resistance coefficient is clamped to >= 0."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        for U in [0.3, 0.5, 1.0, 1.5]:
            result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
            assert result.cw >= 0, f"Cw < 0 at U={U}"

    def test_condition_number_finite(self):
        """Condition number is always finite."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        for U in [0.3, 0.5, 1.0, 1.5]:
            result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
            assert math.isfinite(result.condition_number), \
                f"Condition number not finite at U={U}"

    def test_residual_small(self):
        """Linear system residual is small."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        for U in [0.3, 0.5, 1.0, 1.5]:
            result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
            assert result.relative_residual < 1e-3, \
                f"Residual too large at U={U}: {result.relative_residual}"

    def test_panel_count_matches_mesh(self):
        """Reported panel count matches the input mesh."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        mesh = wigley_hull_mesh(L, B, T, 8, 4)
        result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U)
        assert result.panel_count == mesh.panel_count

    def test_froude_number_correct(self):
        """Reported Froude number matches U/sqrt(gL)."""
        L, B, T = 2.0, 0.2, 0.125
        U = 1.0
        g = 9.80665
        mesh = wigley_hull_mesh(L, B, T, 4, 2)
        result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, U, g=g)
        expected_fn = U / math.sqrt(g * L)
        assert result.froude_number == pytest.approx(expected_fn, rel=1e-12)


# ============================================================================
# 5. verify_kelvin_angle edge cases
# ============================================================================

class TestKelvinAngleEdgeCases:
    """Edge cases for Kelvin angle verification."""

    def test_no_crests_returns_failed(self):
        """When no wave crests are detected, passed=False."""
        pattern = compute_kelvin_wave_pattern(1.0, nx=50, ny=50,
                                               source_strength=1e-15)
        result = verify_kelvin_angle(pattern)
        assert result["passed"] is False
        assert result["measured_angle_deg"] is None

    def test_theoretical_angle_always_reported(self):
        """Theoretical angle is always reported regardless of measurement."""
        pattern = compute_kelvin_wave_pattern(1.0, nx=50, ny=50)
        result = verify_kelvin_angle(pattern)
        assert result["theoretical_angle_deg"] == pytest.approx(
            KELVIN_ANGLE_DEG, rel=1e-10)


# ============================================================================
# 6. Known numerical issues (documented, not fixed)
# ============================================================================

class TestKnownNumericalIssues:
    """The kernel defect, carried as declared withdrawal of evidence (m16913).

    Mission m16830 recorded here that the solver returned ``C_w ~ 1e8`` and
    that ``R_w`` oscillated between huge values and zero with speed.  m16913
    traced both symptoms to their causes -- an invalid principal-value
    quadrature and a vectorized route that replaced the wave integral by an
    ad-hoc ``K_0`` heuristic -- and fixed them, while establishing that the
    *kernel form itself* is the frequency-domain Havelock kernel and violates
    the steady free-surface condition.  The numbers therefore moved, but they
    are still not evidence, and the result now says so.
    """

    def test_solver_declares_withheld_evidence(self):
        """The solver result carries its evidence level instead of implying one."""
        L, B, T = 2.0, 0.2, 0.125
        mesh = wigley_hull_mesh(L, B, T, 8, 4)
        result = solve_nk_wave_resistance(mesh.vertices, mesh.faces, 0.5)
        assert result.kernel_status.startswith("WITHHELD"), result.kernel_status
        assert math.isfinite(result.cw) and math.isfinite(result.wave_resistance)

    def test_free_surface_condition_violated(self):
        """Referee for any future kernel: the residual is O(0.01-1), not O(h)."""
        residual = steady_free_surface_residual(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -0.6]), 5.0
        )
        assert residual["relative"] > 1.0e-2
        assert abs(residual["residual"]) > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
