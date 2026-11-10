"""Pytest tests for the seakeeping verification suite.

Tests cover:
1. Hemisphere zero-frequency added-mass anchor (A₃₃ = 2/3·ρ·π·a³ at ω→0)
2. Froude-Krylov excitation force closed-form vs BEM
3. Wigley III RAO frequency response with limit validation
4. Mesh convergence for Wigley III RAO
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.verification.seakeeping.hemisphere_anchor import (
    compute_hemisphere_zero_frequency_anchor,
    compute_hemisphere_radiation_study,
)
from tensorbem.verification.seakeeping.fk_excitation import (
    fk_heave_force_box_barge,
    compute_fk_vs_bem_comparison,
)
from tensorbem.verification.seakeeping.wigley_rao import (
    compute_wigley_rao_spectrum,
)
from tensorbem.verification.seakeeping.mesh_convergence import (
    run_wigley_mesh_convergence,
)

# Physical constants
_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


# ============================================================================
# 1. Hemisphere zero-frequency added-mass anchor
# ============================================================================

class TestHemisphereAnchor:
    """Test the hemisphere zero-frequency added-mass analytical anchor."""

    def test_zero_frequency_a33_analytical(self):
        """A33/(rho*V) at zero frequency = 1.0 (displaced mass ratio)."""
        radius = 1.0
        density = 1025.0
        expected_volume = (2.0 / 3.0) * math.pi * radius ** 3
        expected_mass = density * expected_volume

        result = compute_hemisphere_zero_frequency_anchor(
            radius=radius, density=density, subdivisions=2, ka_low=0.01,
        )

        assert result.a33_zero_freq_dimless == 1.0
        assert abs(result.displaced_volume - expected_volume) < 1e-10
        assert abs(result.displaced_mass - expected_mass) < 1e-10

    def test_bem_close_to_hulme_at_low_ka(self):
        """BEM at Ka=0.01 should be close to Hulme (1982) reference."""
        result = compute_hemisphere_zero_frequency_anchor(
            radius=1.0, density=1025.0, subdivisions=2, ka_low=0.01,
        )

        # BEM should match Hulme within 5% at Ka=0.01 with subdivision=2
        assert result.a33_vs_hulme_rel_error < 0.05, (
            f"BEM vs Hulme error {result.a33_vs_hulme_rel_error:.4f} exceeds 5%"
        )

    def test_radiation_study_hulme_comparison(self):
        """BEM radiation results should match Hulme (1982) within tolerance."""
        study = compute_hemisphere_radiation_study(
            radius=1.0, density=1025.0, subdivisions=2,
            ka_values=(0.5, 1.0, 2.0),
        )

        for level in study["levels"]:
            # A33 error should be < 10% at Ka=1.0 with subdivision=2
            assert level["a33_rel_error"] < 0.10, (
                f"A33 error {level['a33_rel_error']:.4f} at Ka={level['ka']}"
            )


# ============================================================================
# 2. Froude-Krylov excitation force
# ============================================================================

class TestFKExcitation:
    """Test Froude-Krylov closed-form vs BEM diffraction."""

    def test_fk_closed_form_long_wave_limit(self):
        """At kL → 0, FK heave force → ρ·g·A·B·L·exp(-kT)."""
        length = 50.0
        beam = 10.0
        draft = 4.0
        omega = 0.2  # Low frequency
        amplitude = 1.0
        density = 1025.0
        gravity = 9.80665

        fk = fk_heave_force_box_barge(length, beam, draft, omega, amplitude, density, gravity)
        k = omega ** 2 / gravity

        # At low frequency, sinc(kL/2) ≈ 1 and exp(-kT) ≈ 1
        expected_approx = density * gravity * amplitude * beam * length
        assert abs(abs(fk) - expected_approx) / expected_approx < 0.05

    def test_fk_vs_bem_long_wave_agreement(self):
        """At long waves (kL < 0.3), FK and BEM should agree within 20%."""
        results = compute_fk_vs_bem_comparison(
            length=50.0, beam=10.0, draft=4.0,
            omega_values=(0.2,),
            n_length=10, n_beam=5, n_draft=3,
        )

        for r in results:
            if r["long_wave_limit"]:
                assert r["magnitude_rel_error"] < 0.20, (
                    f"FK vs BEM error {r['magnitude_rel_error']:.4f} at kL={r['kL']:.3f}"
                )

    def test_fk_magnitude_positive(self):
        """FK force magnitude should be positive."""
        fk = fk_heave_force_box_barge(50.0, 10.0, 4.0, 0.5, 1.0)
        assert abs(fk) > 0


# ============================================================================
# 3. Wigley III RAO frequency response
# ============================================================================

class TestWigleyRAO:
    """Test Wigley III RAO with limit behavior validation."""

    def test_rao_spectrum_computes(self):
        """RAO spectrum should compute without errors."""
        study = compute_wigley_rao_spectrum(
            omega_values=[0.1, 0.5, 1.0],
            nx=10, nz=5,
        )

        assert len(study.omegas) == 3
        assert len(study.heave_rao_amplitude) == 3
        assert np.all(study.heave_rao_amplitude >= 0)
        assert np.all(np.isfinite(study.heave_rao_amplitude))

    def test_long_wave_limit_model_scale(self):
        """Heave RAO at long waves (kL < 0.3) should be close to 1.0."""
        study = compute_wigley_rao_spectrum(
            omega_values=[0.05, 0.1, 0.15],
            nx=10, nz=5,
            L=3.0, B=0.3, T=0.1875,
        )

        long_wave_mask = study.kL_values < 0.3
        if np.any(long_wave_mask):
            mean_rao = float(np.mean(study.heave_rao_amplitude[long_wave_mask]))
            assert 0.95 <= mean_rao <= 1.05, (
                f"Long-wave RAO mean = {mean_rao:.4f}, expected in [0.95, 1.05]"
            )

    def test_rao_frequency_dependence(self):
        """RAO should vary with frequency (not constant)."""
        study = compute_wigley_rao_spectrum(
            omega_values=[0.1, 1.0, 3.0],
            nx=10, nz=5,
            L=3.0, B=0.3, T=0.1875,
        )

        rao_std = np.std(study.heave_rao_amplitude)
        assert rao_std > 0.05, "RAO should vary significantly with frequency"


# ============================================================================
# 4. Mesh convergence
# ============================================================================

class TestMeshConvergence:
    """Test mesh convergence for Wigley III RAO."""

    def test_convergence_study_runs(self):
        """Convergence study should complete without errors."""
        study = run_wigley_mesh_convergence(
            omega=1.0,
            nx_values=[8, 12],
            nz_values=[4, 6],
            L=20.0, B=4.0, T=1.5,
        )

        assert len(study.panel_counts) == 2
        assert len(study.heave_rao_amplitudes) == 2
        assert study.max_relative_change is not None

    def test_convergence_criterion_met(self):
        """RAO should converge to within 2% between finest levels."""
        study = run_wigley_mesh_convergence(
            omega=1.0,
            nx_values=[8, 12],
            nz_values=[4, 6],
            L=20.0, B=4.0, T=1.5,
            convergence_threshold=0.02,
        )

        assert study.converged, (
            f"RAO not converged: max relative change = {study.max_relative_change:.4f}"
        )

    def test_panel_count_increases(self):
        """Panel count should increase with refinement."""
        study = run_wigley_mesh_convergence(
            omega=1.0,
            nx_values=[8, 12],
            nz_values=[4, 6],
            L=20.0, B=4.0, T=1.5,
        )

        for i in range(1, len(study.panel_counts)):
            assert study.panel_counts[i] > study.panel_counts[i - 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
