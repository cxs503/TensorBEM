"""Pytest tests for the diffraction anchor module.

Tests cover:
1. Single-point diffraction anchor (BEM vs Hulme at Ka=1.0)
2. Frequency sweep across multiple Ka values
3. Mesh convergence study (error decreases with refinement)
4. Haskind reciprocity cross-check
5. Physical consistency (phase, symmetry)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.verification.seakeeping.diffraction_anchor import (
    DiffractionAnchorResult,
    DiffractionConvergenceResult,
    DiffractionSweepResult,
    compute_diffraction_anchor,
    compute_diffraction_convergence,
    compute_diffraction_sweep,
)


# ============================================================================
# 1. Single-point diffraction anchor
# ============================================================================

class TestDiffractionAnchor:
    """Test single-point diffraction anchor."""

    def test_anchor_computes_at_ka1(self):
        """Anchor should compute without errors at Ka=1.0, sub=2."""
        result = compute_diffraction_anchor(ka=1.0, subdivisions=2)

        assert isinstance(result, DiffractionAnchorResult)
        assert result.ka == 1.0
        assert result.subdivisions == 2
        assert result.panel_count > 0
        assert result.condition_number > 0

    def test_heave_error_within_tolerance(self):
        """Heave excitation error should be < 10% at Ka=1.0, sub=2."""
        result = compute_diffraction_anchor(ka=1.0, subdivisions=2)

        assert result.heave_rel_error < 0.10, (
            f"Heave error {result.heave_rel_error:.4f} exceeds 10% "
            f"(BEM={abs(result.bem_fz):.4f}, Hulme={abs(result.hulme_fz):.4f})"
        )

    def test_surge_error_within_tolerance(self):
        """Surge excitation error should be < 30% at Ka=1.0, sub=2.

        Surge convergence is slower than heave because the surge mode has
        a cos(theta) azimuthal dependence that requires finer angular resolution.
        """
        result = compute_diffraction_anchor(ka=1.0, subdivisions=2)

        assert result.surge_rel_error < 0.30, (
            f"Surge error {result.surge_rel_error:.4f} exceeds 30% "
            f"(BEM={abs(result.bem_fx):.4f}, Hulme={abs(result.hulme_fx):.4f})"
        )

    def test_haskind_reciprocity_heave(self):
        """Haskind heave excitation should match Hulme within 5%."""
        result = compute_diffraction_anchor(ka=1.0, subdivisions=2)

        assert result.haskind_heave_rel_error < 0.05, (
            f"Haskind heave error {result.haskind_heave_rel_error:.4f} exceeds 5%"
        )

    def test_haskind_reciprocity_surge(self):
        """Haskind surge excitation should match Hulme within 10%."""
        result = compute_diffraction_anchor(ka=1.0, subdivisions=2)

        assert result.haskind_surge_rel_error < 0.10, (
            f"Haskind surge error {result.haskind_surge_rel_error:.4f} exceeds 10%"
        )

    def test_excitation_force_phase_consistency(self):
        """Excitation force phase should be physically consistent.

        For head-on waves (beta=0), the surge force should be predominantly
        imaginary (90-degree phase lag due to radiation), while the heave
        force should have a significant real component.
        """
        result = compute_diffraction_anchor(ka=1.0, subdivisions=2)

        # Surge should be predominantly imaginary
        assert abs(result.bem_fx.imag) > abs(result.bem_fx.real), (
            f"Surge force phase inconsistent: real={result.bem_fx.real:.4f}, "
            f"imag={result.bem_fx.imag:.4f}"
        )

        # Heave should have significant real component
        assert abs(result.bem_fz.real) > 0.1 * abs(result.bem_fz), (
            f"Heave force real component too small: real={result.bem_fz.real:.4f}, "
            f"mag={abs(result.bem_fz):.4f}"
        )

    def test_anchor_at_low_ka(self):
        """At low Ka, heave excitation should approach hydrostatic limit."""
        result = compute_diffraction_anchor(ka=0.1, subdivisions=2)

        # Hydrostatic limit: Fz -> rho*g*A*pi*a^2
        hydrostatic = 1025.0 * 9.80665 * math.pi * 1.0**2
        bem_mag = abs(result.bem_fz)
        hulme_mag = abs(result.hulme_fz)

        # Both BEM and Hulme should be close to hydrostatic at low Ka
        assert abs(bem_mag - hydrostatic) / hydrostatic < 0.20, (
            f"BEM heave at Ka=0.1: {bem_mag:.2f} vs hydrostatic {hydrostatic:.2f}"
        )
        assert abs(hulme_mag - hydrostatic) / hydrostatic < 0.20, (
            f"Hulme heave at Ka=0.1: {hulme_mag:.2f} vs hydrostatic {hydrostatic:.2f}"
        )

    def test_anchor_invalid_inputs(self):
        """Invalid inputs should raise ValueError."""
        with pytest.raises(ValueError, match="ka must be positive"):
            compute_diffraction_anchor(ka=0.0)
        with pytest.raises(ValueError, match="subdivisions must be >= 1"):
            compute_diffraction_anchor(ka=1.0, subdivisions=0)
        with pytest.raises(ValueError, match="radius must be positive"):
            compute_diffraction_anchor(ka=1.0, radius=0.0)


# ============================================================================
# 2. Frequency sweep
# ============================================================================

class TestDiffractionSweep:
    """Test frequency sweep diffraction study."""

    def test_sweep_computes(self):
        """Sweep should compute across multiple Ka values."""
        result = compute_diffraction_sweep(
            ka_values=(0.5, 1.0, 1.5), subdivisions=1,
        )

        assert isinstance(result, DiffractionSweepResult)
        assert len(result.ka_values) == 3
        assert len(result.levels) == 3

    def test_sweep_errors_reported(self):
        """Sweep should report max and mean errors."""
        result = compute_diffraction_sweep(
            ka_values=(0.5, 1.0, 1.5), subdivisions=1,
        )

        assert result.max_surge_error >= 0
        assert result.max_heave_error >= 0
        assert result.mean_surge_error >= 0
        assert result.mean_heave_error >= 0

    def test_sweep_panel_count_consistent(self):
        """All levels in sweep should have same panel count."""
        result = compute_diffraction_sweep(
            ka_values=(0.5, 1.0, 1.5), subdivisions=1,
        )

        panel_count = result.panel_count
        for level in result.levels:
            assert level.panel_count == panel_count


# ============================================================================
# 3. Mesh convergence
# ============================================================================

class TestDiffractionConvergence:
    """Test mesh convergence diffraction study."""

    def test_convergence_computes(self):
        """Convergence study should compute across refinement levels."""
        result = compute_diffraction_convergence(
            ka=1.0, subdivisions=(1, 2),
        )

        assert isinstance(result, DiffractionConvergenceResult)
        assert len(result.levels) == 2
        assert len(result.panel_counts) == 2

    def test_panel_count_increases(self):
        """Panel count should increase with refinement."""
        result = compute_diffraction_convergence(
            ka=1.0, subdivisions=(1, 2, 3),
        )

        for i in range(1, len(result.panel_counts)):
            assert result.panel_counts[i] > result.panel_counts[i - 1]

    def test_heave_error_decreases(self):
        """Heave error should decrease with mesh refinement."""
        result = compute_diffraction_convergence(
            ka=1.0, subdivisions=(1, 2, 3),
        )

        for i in range(1, len(result.heave_errors)):
            assert result.heave_errors[i] < result.heave_errors[i - 1], (
                f"Heave error did not decrease: "
                f"{result.heave_errors[i-1]:.4f} -> {result.heave_errors[i]:.4f}"
            )

    def test_surge_error_decreases(self):
        """Surge error should decrease with mesh refinement."""
        result = compute_diffraction_convergence(
            ka=1.0, subdivisions=(1, 2, 3),
        )

        for i in range(1, len(result.surge_errors)):
            assert result.surge_errors[i] < result.surge_errors[i - 1], (
                f"Surge error did not decrease: "
                f"{result.surge_errors[i-1]:.4f} -> {result.surge_errors[i]:.4f}"
            )

    def test_observed_order_positive(self):
        """Observed order of convergence should be positive."""
        result = compute_diffraction_convergence(
            ka=1.0, subdivisions=(1, 2, 3),
        )

        if result.observed_order_heave is not None:
            assert result.observed_order_heave > 0, (
                f"Observed heave order {result.observed_order_heave} not positive"
            )
        if result.observed_order_surge is not None:
            assert result.observed_order_surge > 0, (
                f"Observed surge order {result.observed_order_surge} not positive"
            )

    def test_finest_error_reported(self):
        """Finest level errors should be reported."""
        result = compute_diffraction_convergence(
            ka=1.0, subdivisions=(1, 2, 3),
        )

        assert result.finest_surge_error == result.surge_errors[-1]
        assert result.finest_heave_error == result.heave_errors[-1]

    def test_convergence_invalid_inputs(self):
        """Invalid inputs should raise ValueError."""
        with pytest.raises(ValueError, match="at least one subdivision"):
            compute_diffraction_convergence(ka=1.0, subdivisions=[])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
