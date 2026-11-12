"""Pytest tests for multihull frequency-dependence benchmark (p106w1-n09).

Verifies all machine criteria from the campaign DAG:
- [zero freq] B(omega->0)/B_max <= 0.01
- [inf freq] |A(omega->inf) - A_geom|/A_geom <= 2%
- [energy] B(omega) >= 0 for all omega
- [distance] |sigma(d) - 1| <= 0.05 for d/L > 2
- [negative] d -> 0 singularity trend (record, not crash)
- [degeneration] omega->0 and d->0 -> single-hull hydrostatic (<=1%)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.multihull_frequency_dependence import (
    MUTUAL_DAMPING_SUPPORTED,
    bem_cross_validation,
    distance_decay_study,
    frequency_scan,
    mutual_interaction_coefficients,
    negative_control_coincident,
    single_cylinder_heave_coefficients,
    verify_asymptotics,
)


# ---------------------------------------------------------------------------
# Single-hull asymptotic limits
# ---------------------------------------------------------------------------

class TestSingleHullAsymptotics:
    """Verify single-hull added mass and damping asymptotic behavior."""

    def test_zero_frequency_damping_vanishes(self):
        """B(ka->0) / B_max <= 0.01 (damping tends to zero)."""
        a_over_f = 0.3 / 0.8  # radius / depth
        # Find B_max
        kas = np.logspace(-2, 1.5, 60)
        B_vals = [single_cylinder_heave_coefficients(float(k), a_over_f)[1] for k in kas]
        B_max = max(B_vals)
        assert B_max > 0, "B_max should be positive"

        # Zero frequency
        _, B_zero = single_cylinder_heave_coefficients(0.001, a_over_f)
        ratio = B_zero / B_max
        assert ratio <= 0.01, f"B(ka->0)/B_max = {ratio:.6e} > 0.01"

    def test_high_frequency_added_mass_is_the_exact_series_value(self):
        """A(ka->inf) is NOT rho*pi*a**2: the free-surface image lowers it 6.8%."""
        a_over_f = 0.3 / 0.8
        A_inf, B_inf = single_cylinder_heave_coefficients(100.0, a_over_f)
        # exact multipole value; the pre-16696 formula hand-patched A = 1 - O(1/ka)
        assert A_inf == pytest.approx(0.931755, rel=2.0e-4)
        assert B_inf < 1.0e-20
        asym = verify_asymptotics(0.3, 0.8)
        assert asym["infinite_frequency"]["pass"] is True
        # the retracted "geometric limit" claim is recorded, not silently dropped
        assert asym["infinite_frequency"]["retracted_claim_error"] > 0.05

    def test_damping_non_negative_all_frequencies(self):
        """B(omega) >= 0 for all omega (energy conservation)."""
        a_over_f = 0.3 / 0.8
        kas = np.logspace(-2, 1.5, 60)
        for ka in kas:
            _, B = single_cylinder_heave_coefficients(float(ka), a_over_f)
            assert B >= -1e-15, f"B < 0 at ka={ka}: B={B}"

    def test_infinite_frequency_damping_vanishes(self):
        """B(ka->inf) -> 0 (no radiation at infinite frequency)."""
        a_over_f = 0.3 / 0.8
        _, B_inf = single_cylinder_heave_coefficients(100.0, a_over_f)
        assert B_inf < 1e-10, f"B(ka->inf) = {B_inf} should be ~0"


# ---------------------------------------------------------------------------
# Multihull distance decay
# ---------------------------------------------------------------------------

class TestDistanceDecay:
    """Verify interference factor sigma(d) -> 1 at large separation."""

    def test_sigma_far_field_degenerates_to_single_hull(self):
        """|sigma_A - 1| <= 5% once the exact (a/d)**2 decay has died out.

        The exact unbounded-fluid mutual added mass decays algebraically, not
        exponentially, so the 5 % band is reached at d/L = 4 (d/a = 8), not at
        the d/L > 2 that the pre-16696 fitted ``exp(-kd/2)`` decay allowed.
        """
        from tensorbem.benchmarks.multihull_frequency_dependence import distance_decay_study

        decay = distance_decay_study(radius=0.3, depth=0.8, ka_fixed=0.5)
        far = [r for r in decay["results"] if r["d_over_L"] >= 4.0]
        assert far, "the ladder must reach d/L = 4"
        for r in far:
            sigma = r["sigma_A"]
            if math.isfinite(sigma):
                assert abs(sigma - 1.0) <= 0.05, (r["d_over_L"], sigma)

    def test_decay_exponent_positive(self):
        """Interference should decay with distance (positive exponent)."""
        decay = distance_decay_study(radius=0.3, depth=0.8, ka_fixed=0.5)
        assert math.isfinite(decay["decay_exponent"])
        assert decay["decay_exponent"] > 0, "Decay exponent should be positive"

    def test_mutual_interaction_envelope_decays(self):
        """Envelope of |A_mutual| should decay with distance.

        Note: the mutual added mass oscillates due to wave interference
        (cos(kd) terms), so we check the envelope decay rather than
        pointwise monotonicity.
        """
        a_over_f = 0.3 / 0.8
        ka = 0.5
        # Compare near-field vs far-field envelope
        d_near = [2.5, 3.0, 3.5]
        d_far = [15.0, 20.0, 30.0]
        
        A_near = [abs(mutual_interaction_coefficients(ka, a_over_f, d)[0]) for d in d_near]
        A_far = [abs(mutual_interaction_coefficients(ka, a_over_f, d)[0]) for d in d_far]
        
        # The far-field values should be smaller than near-field peak
        max_near = max(A_near)
        max_far = max(A_far)
        assert max_far < max_near, (
            f"Far-field envelope ({max_far:.6e}) should be < near-field ({max_near:.6e})"
        )


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------

class TestNegativeControl:
    """Verify singularity behavior at d -> 0."""

    def test_coincident_hulls_no_crash(self):
        """d -> 0 should not crash (record singularity trend)."""
        result = negative_control_coincident(radius=0.3, depth=0.8, ka=0.5)
        assert result["no_crash"] is True

    def test_coincident_hulls_singularity_trend(self):
        """|A_mutual| should grow as d -> 0 (singularity trend)."""
        result = negative_control_coincident(radius=0.3, depth=0.8, ka=0.5)
        assert result["singularity_trend_detected"] is True


# ---------------------------------------------------------------------------
# Frequency scan consistency
# ---------------------------------------------------------------------------

class TestFrequencyScan:
    """Verify frequency scan produces physically consistent results."""

    def test_catamaran_added_mass_positive(self):
        """Catamaran added mass should be positive for all frequencies."""
        rows = frequency_scan(radius=0.3, depth=0.8, separation=1.5)
        for r in rows:
            assert r.A33_catamaran > 0, f"A33_cat < 0 at ka={r.ka}"

    def test_catamaran_damping_non_negative(self):
        """Catamaran damping should be non-negative for all frequencies."""
        rows = frequency_scan(radius=0.3, depth=0.8, separation=1.5)
        for r in rows:
            assert r.B33_catamaran >= -1e-15, f"B33_cat < 0 at ka={r.ka}"

    def test_single_hull_limit_at_large_separation(self):
        """At very large separation, catamaran -> 2 × single hull."""
        rows = frequency_scan(radius=0.3, depth=0.8, separation=100.0)
        for r in rows:
            if math.isfinite(r.sigma_A):
                assert abs(r.sigma_A - 1.0) < 0.01, (
                    f"At d/L=100/0.6, sigma_A={r.sigma_A} should be ~1.0"
                )


# ---------------------------------------------------------------------------
# BEM cross-validation
# ---------------------------------------------------------------------------

class TestBEMCrossValidation:
    """Cross-validate analytical model against 2D BEM solver."""

    def test_bem_runs_without_crash(self):
        """BEM solver should complete for all test frequencies."""
        results = bem_cross_validation(
            radius=0.3, depth=0.8, n_panels=48,
            ka_values=[0.3, 0.5, 1.0],
        )
        for r in results:
            assert "error" not in r, f"BEM failed at ka={r['ka']}: {r.get('error')}"

    def test_bem_agrees_with_the_exact_series(self):
        """BEM vs exact Ursell multipole: measured 2.5% in A, 3.5% in B at N=48."""
        results = bem_cross_validation(
            radius=0.3, depth=0.8, n_panels=48,
            ka_values=[0.3, 0.5, 1.0],
        )
        for r in results:
            assert "error" not in r, f"BEM failed at ka={r['ka']}: {r.get('error')}"
            # the pre-16696 gate was a factor-3 band on A and no gate at all on B,
            # which is why a 200x damping error in the reference went unnoticed.
            assert r["A_rel_error"] < 0.05, (r["ka"], r["A_rel_error"])
            assert r["B_rel_error"] < 0.05, (r["ka"], r["B_rel_error"])



# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------

class TestFullVerification:
    """Run the complete verification and check all criteria."""

    def test_all_criteria_pass(self):
        """All machine criteria must pass."""
        from tensorbem.benchmarks.multihull_frequency_dependence import run_full_study
        result = run_full_study()
        assert result["all_pass"] is True, (
            f"Some criteria failed: {result['criteria']}"
        )

# ---------------------------------------------------------------------------
# Exact-reference provenance (mission 16696)
# ---------------------------------------------------------------------------

class TestExactReferenceProvenance:
    """The "analytic" side is the repo's exact multipole series, and the pieces
    that have no reference are declared unsupported instead of fitted."""

    def test_single_hull_reference_is_the_multipole_series(self):
        from tensorbem.analytic import ursell_cylinder_coefficients

        for ka in (0.01, 0.3, 1.0, 10.0):
            for a_over_f in (0.2, 0.375, 0.6):
                assert single_cylinder_heave_coefficients(ka, a_over_f) == pytest.approx(
                    ursell_cylinder_coefficients(ka, a_over_f), rel=1.0e-12, abs=0.0
                )

    def test_truncation_is_converged(self):
        for ka in (0.01, 0.5, 5.0):
            coarse = single_cylinder_heave_coefficients(ka, 0.375, 40)
            fine = single_cylinder_heave_coefficients(ka, 0.375, 120)
            assert coarse[0] == pytest.approx(fine[0], rel=1.0e-10, abs=0.0)
            assert coarse[1] == pytest.approx(fine[1], rel=1.0e-10, abs=1.0e-14)

    def test_mutual_damping_is_declared_unsupported(self):
        assert MUTUAL_DAMPING_SUPPORTED is False
        A_mut, B_mut = mutual_interaction_coefficients(0.5, 0.375, 5.0)
        assert A_mut > 0.0
        assert B_mut == 0.0  # not claimed, never a fitted number
        assert verify_asymptotics(0.3, 0.8)["mutual_damping_supported"] is False

    def test_mutual_added_mass_is_the_exact_unbounded_fluid_value(self):
        from tensorbem.analytic import two_cylinder_added_mass

        reference = two_cylinder_added_mass(1.0, 5.0, 1.0)
        expected = reference.mutual_added_mass_transverse / reference.displaced_mass_per_span
        assert mutual_interaction_coefficients(0.5, 0.375, 5.0)[0] == pytest.approx(
            expected, rel=1.0e-12
        )
        # an unbounded fluid has no length scale: the value cannot depend on ka
        low = mutual_interaction_coefficients(0.05, 0.375, 5.0)[0]
        high = mutual_interaction_coefficients(5.0, 0.375, 5.0)[0]
        assert low == pytest.approx(high, rel=1.0e-14)

    def test_overlapping_cylinders_are_nan_not_zero(self):
        assert math.isnan(mutual_interaction_coefficients(0.5, 0.375, 1.5)[0])
        assert math.isnan(mutual_interaction_coefficients(0.5, 0.375, 2.0)[1])
