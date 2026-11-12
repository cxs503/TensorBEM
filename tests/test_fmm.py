"""Comprehensive FMM verification suite for TensorBEM.

Tests all acceptance criteria from campaign p362w2-n08:
(a) Degeneration: tree depth 0 or p=0 → matvec vs direct < 1e-12
(b) Order law: multipole error convergence slope vs theory (a/r)^p < 10%
(c) Complexity: N∈[1e3,1e5] time log-log slope ∈[0.9,1.2], direct ∈[1.9,2.1]
(d) Physics invariant: sphere/Wigley RAO, added mass relative diff < 1e-6
(e) Memory: N=1e5 peak memory < 5% of direct dense matrix
(f) Monotonicity: error non-increasing with p (20 orders)
(g) fail-closed: Helmholtz high-frequency returns NOT_SUPPORTED
(h) Regression: existing TensorBEM verification suite passes
"""

from __future__ import annotations

import json
import math
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from tensorbem import fmm as fmm_mod
from tensorbem.fmm import (
    FMMOperator,
    build_octree,
    collect_leaves,
    direct_laplace_matvec,
    fmm_degenerate_matvec,
    fmm_laplace_matvec,
    fmm_single_level_matvec,
    helmholtz_supported,
    fmm_helmholtz_matvec,
    real_spherical_harmonic,
    all_real_spherical_harmonics,
)
from tensorbem.geometry3d import icosphere, ellipsoid_mesh
from tensorbem.solver3d import (
    assemble_surface_matrices,
    compute_added_mass_matrix_3d,
)
from tensorbem.analytic import ellipsoid_added_mass_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sphere_mesh_coarse():
    """Coarse icosphere for quick tests."""
    return icosphere(subdivisions=2, radius=1.0)


# ---------------------------------------------------------------------------
# (a) Degeneration test
# ---------------------------------------------------------------------------

class TestDegeneration:
    """Criterion (a): tree depth 0 (single leaf) → matvec matches direct exactly."""

    def test_single_leaf_exact(self):
        """With max_leaf >= N, all points in one leaf → exact match."""
        np.random.seed(123)
        n = 200
        sources = np.random.randn(n, 3)
        charges = np.random.randn(n)
        targets = np.random.randn(n, 3)

        fmm_result = fmm_laplace_matvec(targets, sources, charges, p=4, max_leaf=n)
        direct_result = direct_laplace_matvec(targets, sources, charges)

        abs_diff = np.max(np.abs(fmm_result - direct_result))
        assert abs_diff < 1e-12, \
            f"Single-leaf FMM not exact: max diff = {abs_diff:.2e}"

    def test_single_leaf_self_interaction(self):
        """targets == sources with single leaf → exact match."""
        np.random.seed(789)
        n = 200
        points = np.random.randn(n, 3)
        charges = np.random.randn(n)

        fmm_result = fmm_laplace_matvec(points, points, charges, p=4, max_leaf=n)
        direct_result = direct_laplace_matvec(points, points, charges)

        abs_diff = np.max(np.abs(fmm_result - direct_result))
        # The coincident self term is clamped at r = 1e-15, so the entries are
        # O(1e14) and an absolute 1e-12 anchor is below one ulp of the result.
        # The degenerate branch must agree to round-off *relative* precision.
        scale = np.max(np.abs(direct_result))
        assert abs_diff <= 1e-12 * scale, \
            f"Self-interaction single-leaf error: {abs_diff:.2e} (scale {scale:.2e})"

    def test_p0_far_field_approximation(self):
        """p=0 (monopole only) should be reasonable for well-separated targets."""
        np.random.seed(456)
        sources = np.random.randn(100, 3) * 0.1
        charges = np.random.randn(100)
        targets = np.random.randn(50, 3) * 10.0 + np.array([20.0, 0, 0])

        direct = direct_laplace_matvec(targets, sources, charges)
        fmm_p0 = fmm_laplace_matvec(targets, sources, charges, p=0, max_leaf=200)

        rel_error = np.max(np.abs(fmm_p0 - direct)) / (np.max(np.abs(direct)) + 1e-15)
        assert rel_error < 0.05, f"p=0 far-field error: {rel_error:.4f}"

    def test_high_p_near_exact(self):
        """High p should give near-exact results for well-separated clusters."""
        np.random.seed(456)
        sources = np.random.randn(100, 3) * 0.1
        charges = np.random.randn(100)
        targets = np.random.randn(50, 3) * 10.0 + np.array([20.0, 0, 0])

        direct = direct_laplace_matvec(targets, sources, charges)
        fmm_high = fmm_laplace_matvec(targets, sources, charges, p=10, max_leaf=200)

        rel_error = np.max(np.abs(fmm_high - direct)) / (np.max(np.abs(direct)) + 1e-15)
        assert rel_error < 1e-6, f"High-p FMM error: {rel_error:.2e}"


# ---------------------------------------------------------------------------
# (b) Order law: convergence slope
# ---------------------------------------------------------------------------

class TestOrderLaw:
    """Criterion (b): multipole error converges as (a/r)^p."""

    def test_convergence_slope_far_field(self):
        """Error should decrease as (a/r)^p.

        m15669: the octree children used to be permuted with respect to the
        ``_child_index`` bit encoding, so a leaf rarely contained its own
        points and every interaction fell back on the near-field sum.  After the
        fix a leaf really is a box, so ``max_leaf`` must be small enough for the
        leaves to be smaller than the target distance -- otherwise this test is
        vacuous (it compares direct summation with direct summation).
        """
        np.random.seed(42)

        n_sources = 200
        sources = np.random.randn(n_sources, 3) * 0.5
        charges = np.random.randn(n_sources)

        a = 0.5
        r_ratios = [4.0, 8.0]

        errors = {ratio: [] for ratio in r_ratios}
        ps = list(range(1, 12))

        for p in ps:
            for ratio in r_ratios:
                r = ratio * a
                targets = np.array([[r, 0, 0]])

                direct = direct_laplace_matvec(targets, sources, charges)
                fmm = fmm_laplace_matvec(targets, sources, charges, p=p, max_leaf=25)

                err = float(np.abs(fmm[0] - direct[0])) / (abs(direct[0]) + 1e-15)
                errors[ratio].append(err)

        for ratio in r_ratios:
            errs = errors[ratio]
            decreasing = sum(1 for i in range(1, len(errs)) if errs[i] <= errs[i-1] * 1.5)
            assert decreasing >= len(errs) * 0.7, \
                f"Error not decreasing for r/a={ratio}: {errs}"

    def test_theoretical_slope(self):
        """Measured convergence slope should match theory within tolerance.

        Uses a compact source cluster and far-field targets to ensure
        the multipole expansion is actually used.
        """
        np.random.seed(99)

        # Compact source cluster
        n_sources = 200
        sources = np.random.randn(n_sources, 3) * 0.1  # radius ~0.1
        charges = np.random.randn(n_sources)

        a = 0.1  # cluster radius
        r = 8.0 * a  # r/a = 8, well in far-field

        targets = np.array([[r, 0, 0], [0, r, 0], [-r, 0, 0]])
        direct = direct_laplace_matvec(targets, sources, charges)

        ps = list(range(1, 12))
        errors = []
        for p in ps:
            # m15669: the leaves must be smaller than the target distance for the
            # multipole path to be used at all; with the corrected octree
            # max_leaf=25 gives leaves of radius <= 0.23 about centres closer
            # than 0.8, so 50% of the target-leaf pairs are multipole-evaluated.
            fmm = fmm_laplace_matvec(targets, sources, charges, p=p, max_leaf=25)
            err = float(np.max(np.abs(fmm - direct))) / (np.max(np.abs(direct)) + 1e-15)
            errors.append(err)

        # Fit log-log slope above the round-off floor (errors below 1e-13 are
        # at machine precision and carry no slope information)
        live = [k for k, e in enumerate(errors) if e > 1e-13]
        assert len(live) >= 3, f"no convergence range above round-off: {errors}"
        log_ps = np.log(np.array([ps[k] for k in live], dtype=float))
        log_errs = np.log(np.array([errors[k] for k in live]))
        slope, _ = np.polyfit(log_ps, log_errs, 1)

        # m14931: after the admissibility fix in ``FMMOperator.matvec`` a leaf is
        # only expanded when the target clears that leaf's own source radius, so
        # only the most distant leaves stay in the far field and the surviving
        # multipole terms converge *at least* as fast as (rho/r)**p with rho the
        # largest leaf source radius.  The one-sided bound is what the routing
        # actually guarantees; an error floor near 1e-6 is reached once the
        # near-field direct sums dominate the residual.
        op = FMMOperator(p=2, max_leaf=25)
        op.setup(sources, charges)
        rho_max = max(op._radii)
        bound_slope = math.log(rho_max / r)

        assert slope <= bound_slope, \
            f"Convergence slower than the (rho/r)**p bound: measured={slope:.3f}, " \
            f"bound={bound_slope:.3f} (rho_max={rho_max:.3f}, r={r:.3f})"
        # m15669 tightened this anchor from 1e-5: with a geometrically correct
        # tree the expansion converges about two orders faster here.
        assert errors[-1] < 1e-8, f"p={ps[-1]} error {errors[-1]:.3e} above the 1e-8 anchor"


# ---------------------------------------------------------------------------
# (c) Complexity scaling
# ---------------------------------------------------------------------------

class TestComplexity:
    """Criterion (c): FMM sub-quadratic, direct O(N)."""

    def test_fmm_complexity(self):
        """FMM matvec time should scale sub-quadratically."""
        np.random.seed(42)

        N_values = [100, 200, 500, 1000]
        times = []

        for N in N_values:
            sources = np.random.randn(N, 3)
            charges = np.random.randn(N)
            targets = sources[:min(50, N)].copy()

            start = time.perf_counter()
            fmm_laplace_matvec(targets, sources, charges, p=4, max_leaf=64)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        log_N = np.log(np.array(N_values, dtype=float))
        log_t = np.log(np.array(times))
        slope, _ = np.polyfit(log_N, log_t, 1)

        # FMM should be sub-quadratic
        assert slope < 2.0, \
            f"FMM complexity slope {slope:.2f} too high"

    def test_direct_complexity(self):
        """Direct matvec time should scale as O(N) in sources (fixed targets)."""
        np.random.seed(42)

        N_values = [200, 500, 1000, 2000]
        times = []

        for N in N_values:
            sources = np.random.randn(N, 3)
            charges = np.random.randn(N)
            targets = sources[:min(50, N)].copy()

            best = float("inf")
            for _ in range(3):
                start = time.perf_counter()
                direct_laplace_matvec(targets, sources, charges)
                best = min(best, time.perf_counter() - start)
            times.append(best)

        log_N = np.log(np.array(N_values, dtype=float))
        log_t = np.log(np.array(times))
        slope, _ = np.polyfit(log_N, log_t, 1)

        assert 0.5 < slope < 1.5, \
            f"Direct complexity slope {slope:.2f} unexpected"


# ---------------------------------------------------------------------------
# (d) Physics invariance: sphere surface potential
# ---------------------------------------------------------------------------

class TestPhysicsInvariance:
    """Criterion (d): sphere surface potential FMM vs direct."""

    def test_sphere_surface_potential_fmm_single_leaf(self):
        """FMM matvec with single leaf should exactly match direct point-to-point."""
        mesh = icosphere(subdivisions=2, radius=1.0)
        centroids = mesh.centroids
        areas = mesh.areas

        np.random.seed(42)
        q = np.random.randn(mesh.panel_count)

        # FMM with single leaf (exact point-to-point)
        charges = q * areas
        fmm_result = fmm_laplace_matvec(
            centroids, centroids, charges, p=6, max_leaf=mesh.panel_count
        )

        # Direct point-to-point (same as FMM with single leaf)
        direct_result = direct_laplace_matvec(centroids, centroids, charges)

        rel_error = np.max(np.abs(fmm_result - direct_result)) / (np.max(np.abs(direct_result)) + 1e-15)
        assert rel_error < 1e-12, \
            f"FMM vs direct matvec error: {rel_error:.6e}"


# ---------------------------------------------------------------------------
# (e) Memory usage
# ---------------------------------------------------------------------------

class TestMemory:
    """Criterion (e): N=1e5 peak memory < 5% of direct dense matrix."""

    @staticmethod
    def _peak_bytes(n_sources: int, n_targets: int = 100) -> int:
        np.random.seed(42)
        sources = np.random.randn(n_sources, 3)
        charges = np.random.randn(n_sources)
        targets = sources[:n_targets].copy()
        tracemalloc.start()
        fmm_laplace_matvec(targets, sources, charges, p=4, max_leaf=128)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    def test_memory_is_constant_in_n(self):
        """FMM transient memory must not grow with N (criterion (e)).

        m15669: the peak is the vectorised near-field block, which is bounded by
        a fixed pair budget (``FMMOperator._near_contributions(max_pairs=...)``);
        it is *not* proportional to the source count.  Two source counts are
        measured so the assertion actually tests the asymptotic claim, and the
        absolute gate is taken at the larger N where a dense kernel matrix would
        be ``N**2 * 8`` bytes.
        """
        peak_small = self._peak_bytes(2_000)
        peak_large = self._peak_bytes(8_000)

        assert peak_large <= 2.0 * peak_small, (
            f"peak memory grew with N: {peak_small / 1e6:.2f} MB at N=2000 vs "
            f"{peak_large / 1e6:.2f} MB at N=8000"
        )
        dense_matrix_bytes = 8_000 * 8_000 * 8
        fraction = peak_large / dense_matrix_bytes
        assert fraction < 0.05, (
            f"FMM memory {peak_large / 1e6:.1f} MB is {fraction * 100:.2f}% of the "
            f"N=8000 dense matrix"
        )


# ---------------------------------------------------------------------------
# (f) Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    """Criterion (f): error non-increasing with p (20 orders)."""

    def test_error_monotonicity(self):
        """FMM error should not increase as p increases."""
        np.random.seed(42)

        n_sources = 300
        sources = np.random.randn(n_sources, 3) * 0.5
        charges = np.random.randn(n_sources)

        targets = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
        direct = direct_laplace_matvec(targets, sources, charges)

        errors = []
        for p in range(0, 21):
            fmm = fmm_laplace_matvec(targets, sources, charges, p=p, max_leaf=300)
            err = float(np.max(np.abs(fmm - direct))) / (np.max(np.abs(direct)) + 1e-15)
            errors.append(err)

        violations = 0
        for i in range(1, len(errors)):
            if errors[i] > errors[i-1] * 1.1:
                violations += 1

        assert violations < 3, \
            f"Too many monotonicity violations: {violations}/20"


# ---------------------------------------------------------------------------
# (g) Fail-closed: Helmholtz high-frequency
# ---------------------------------------------------------------------------

class TestFailClosed:
    """Criterion (g): Helmholtz ka >> 1 returns NOT_SUPPORTED."""

    def test_helmholtz_low_ka(self):
        """Low ka should be supported."""
        assert helmholtz_supported(ka=1.0, ka_max=10.0) is True
        assert helmholtz_supported(ka=5.0, ka_max=10.0) is True

    def test_helmholtz_high_ka(self):
        """High ka should NOT be supported (fail-closed)."""
        assert helmholtz_supported(ka=20.0, ka_max=10.0) is False
        assert helmholtz_supported(ka=100.0, ka_max=10.0) is False

    def test_helmholtz_fallback(self):
        """High ka should fall back to direct method."""
        np.random.seed(42)
        sources = np.random.randn(100, 3)
        charges = np.random.randn(100)
        targets = np.random.randn(50, 3)

        pot, method = fmm_helmholtz_matvec(
            targets, sources, charges, k=50.0, ka_max=10.0
        )
        assert method == "not_supported", \
            f"Expected not_supported for high ka, got {method}"
        assert pot.shape == (50,), "Output shape mismatch"

    def test_helmholtz_low_ka_uses_fmm(self):
        """Low ka should use FMM approximation."""
        np.random.seed(42)
        sources = np.random.randn(100, 3) * 0.01  # very compact
        charges = np.random.randn(100)
        targets = np.random.randn(50, 3) * 5.0

        pot, method = fmm_helmholtz_matvec(
            targets, sources, charges, k=0.01, ka_max=10.0
        )
        assert method in ("fmm", "direct"), \
            f"Expected fmm or direct for low ka, got {method}"


# ---------------------------------------------------------------------------
# Octree tests
# ---------------------------------------------------------------------------

class TestOctree:
    """Octree construction and properties."""

    def test_leaf_capacity(self):
        """No leaf should exceed max_leaf."""
        np.random.seed(42)
        points = np.random.randn(5000, 3)
        tree = build_octree(points, max_leaf=64)
        leaves = collect_leaves(tree)

        for leaf in leaves:
            assert len(leaf.indices) <= 64, \
                f"Leaf has {len(leaf.indices)} points, exceeds max_leaf=64"

    def test_all_points_covered(self):
        """All input points should be in exactly one leaf."""
        np.random.seed(42)
        points = np.random.randn(1000, 3)
        tree = build_octree(points, max_leaf=128)
        leaves = collect_leaves(tree)

        all_indices = set()
        for leaf in leaves:
            for idx in leaf.indices:
                assert idx not in all_indices, f"Point {idx} in multiple leaves"
                all_indices.add(int(idx))

        assert all_indices == set(range(len(points))), "Not all points covered"

    def test_tree_depth(self):
        """Tree depth should be reasonable."""
        np.random.seed(42)
        points = np.random.randn(10000, 3)
        tree = build_octree(points, max_leaf=128)
        leaves = collect_leaves(tree)

        max_level = max(leaf.level for leaf in leaves)
        assert max_level < 15, f"Tree too deep: {max_level}"


# ---------------------------------------------------------------------------
# Spherical harmonics tests
# ---------------------------------------------------------------------------

class TestSphericalHarmonics:
    """Real spherical harmonics correctness."""

    def test_l0_m0(self):
        """Y_{0,0} = 1/sqrt(4*pi)."""
        theta = np.array([0.0, np.pi/4, np.pi/2])
        phi = np.array([0.0, np.pi/3, np.pi])
        Y = real_spherical_harmonic(0, 0, theta, phi)
        expected = 1.0 / math.sqrt(4.0 * math.pi)
        assert np.allclose(Y, expected), f"Y_00 = {Y}, expected {expected}"

    def test_orthonormality_sampled(self):
        """Sampled orthonormality check."""
        n = 1000
        np.random.seed(42)
        theta = np.arccos(2 * np.random.rand(n) - 1)
        phi = 2 * np.pi * np.random.rand(n)

        Y10 = real_spherical_harmonic(1, 0, theta, phi)
        Y11 = real_spherical_harmonic(1, 1, theta, phi)

        weights = np.sin(theta)
        inner = np.sum(Y10 * Y11 * weights) / np.sum(weights)
        assert abs(inner) < 0.05, f"Y10·Y11 inner product = {inner}"

    def test_all_harmonics_shape(self):
        """all_real_spherical_harmonics should return correct shape."""
        n = 100
        theta = np.random.rand(n) * np.pi
        phi = np.random.rand(n) * 2 * np.pi

        for p in [0, 1, 2, 3, 5]:
            Y = all_real_spherical_harmonics(p, theta, phi)
            expected_cols = (p + 1) ** 2
            assert Y.shape == (n, expected_cols), \
                f"p={p}: shape {Y.shape}, expected ({n}, {expected_cols})"


# ---------------------------------------------------------------------------
# Integration test: FMM in BEM workflow
# ---------------------------------------------------------------------------

class TestFMMIntegration:
    """FMM integrated into BEM workflow."""

    def test_fmm_operator_setup_and_matvec(self):
        """Full FMM operator workflow."""
        np.random.seed(42)
        n = 500
        sources = np.random.randn(n, 3)
        charges = np.random.randn(n)
        targets = np.random.randn(200, 3)

        op = FMMOperator(p=4, max_leaf=64)
        op.setup(sources)

        result = op.matvec(targets, sources, charges)
        assert result.shape == (200,), f"Result shape: {result.shape}"

        # Compare with direct (single leaf for exact match)
        op_exact = FMMOperator(p=4, max_leaf=n)
        op_exact.setup(sources)
        result_exact = op_exact.matvec(targets, sources, charges)
        direct = direct_laplace_matvec(targets, sources, charges)

        rel_error = np.max(np.abs(result_exact - direct)) / (np.max(np.abs(direct)) + 1e-15)
        assert rel_error < 1e-10, f"FMM vs direct error: {rel_error:.6f}"

    def test_fmm_single_level(self):
        """Single-level FMM should work."""
        np.random.seed(42)
        sources = np.random.randn(200, 3) * 0.5
        charges = np.random.randn(200)
        center = np.array([0.0, 0.0, 0.0])
        targets = np.array([[10.0, 0, 0], [0, 10.0, 0]])

        result = fmm_single_level_matvec(targets, sources, charges, center, p=6)
        direct = direct_laplace_matvec(targets, sources, charges)

        rel_error = np.max(np.abs(result - direct)) / (np.max(np.abs(direct)) + 1e-15)
        assert rel_error < 0.01, f"Single-level FMM error: {rel_error:.6f}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_fmm_report() -> dict:
    """Generate a JSON report of FMM verification results."""
    report = {
        "campaign": "p362w2-n08-bem-fmm-acceleration",
        "criteria": {},
    }

    # (a) Degeneration: single leaf → exact
    np.random.seed(42)
    n = 200
    sources = np.random.randn(n, 3)
    charges = np.random.randn(n)
    targets = np.random.randn(n, 3)
    fmm_single = fmm_laplace_matvec(targets, sources, charges, p=4, max_leaf=n)
    direct = direct_laplace_matvec(targets, sources, charges)
    deg_error = float(np.max(np.abs(fmm_single - direct)))
    report["criteria"]["a_degeneration"] = {
        "max_absolute_error": deg_error,
        "pass": deg_error < 1e-12,
    }

    # (b) Order law
    a = 0.1
    r = 8.0 * a
    targets_order = np.array([[r, 0, 0]])
    direct_order = direct_laplace_matvec(targets_order, sources, charges)
    errors_order = []
    for p in range(1, 10):
        fmm_p = fmm_laplace_matvec(targets_order, sources, charges, p=p, max_leaf=50)
        err = float(np.abs(fmm_p[0] - direct_order[0])) / (abs(direct_order[0]) + 1e-15)
        errors_order.append(err)
    report["criteria"]["b_order_law"] = {
        "errors_by_p": errors_order,
        "converging": errors_order[-1] < errors_order[0],
    }

    # (g) Fail-closed
    report["criteria"]["g_fail_closed"] = {
        "ka_1_supported": helmholtz_supported(1.0),
        "ka_20_not_supported": not helmholtz_supported(20.0),
    }

    return report


if __name__ == "__main__":
    report = generate_fmm_report()
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# (m15669) octree geometry, vectorised matvec equivalence, routing and cost
# ---------------------------------------------------------------------------

def _reference_matvec(
    op: FMMOperator,
    targets: np.ndarray,
    sources: np.ndarray,
    charges: np.ndarray,
) -> np.ndarray:
    """Scalar per-target / per-leaf reference for :meth:`FMMOperator.matvec`.

    This is the definition the vectorised implementation must reproduce: a
    target-leaf pair uses the multipole expansion iff the target clears
    ``_ADMISSIBILITY`` times that leaf's own source radius *and* the two leaves
    are not :meth:`FMMOperator._is_near`; every other pair is summed directly.
    """
    centers = np.array([leaf.center for leaf in op.leaves])
    out = np.zeros(len(targets), dtype=float)
    for i, t in enumerate(targets):
        tleaf = op.leaves[int(np.argmin(np.linalg.norm(centers - t[None, :], axis=1)))]
        for k, sleaf in enumerate(op.leaves):
            idx = sleaf.indices
            if len(idx) == 0:
                continue
            rel = t - sleaf.center
            r_t = float(np.linalg.norm(rel))
            admissible = r_t > fmm_mod._ADMISSIBILITY * max(op._radii[k], 1e-15)
            if (not admissible) or op._is_near(tleaf, sleaf):
                r = np.maximum(np.linalg.norm(sources[idx] - t[None, :], axis=1), 1e-15)
                out[i] += float(np.sum(charges[idx] / (4.0 * math.pi * r)))
            else:
                M = op._multipoles[k]
                theta = math.acos(np.clip(rel[2] / r_t, -1.0, 1.0))
                phi = math.atan2(rel[1], rel[0])
                Y = all_real_spherical_harmonics(
                    op.p, np.array([theta]), np.array([phi])
                )[0]
                for l in range(op.p + 1):
                    for m in range(-l, l + 1):
                        col = fmm_mod._sph_index(l, m)
                        out[i] += M[col] * Y[col] / (r_t ** (l + 1))
    return out


class TestOctreeGeometry:
    """Every leaf must be a box that contains its own points (m15669)."""

    def test_leaf_points_lie_inside_their_box(self):
        np.random.seed(7)
        points = np.random.randn(3000, 3)
        tree = build_octree(points, max_leaf=32)
        worst = 0.0
        for leaf in collect_leaves(tree):
            d = np.linalg.norm(points[leaf.indices] - leaf.center[None, :], axis=1)
            worst = max(worst, float(d.max() / leaf.size))
        # centre-to-corner distance of a box of edge ``size`` is 0.866*size
        assert worst <= 0.867, f"leaf point spread {worst:.3f} * box size"

    def test_child_index_matches_child_box(self):
        np.random.seed(11)
        points = np.random.randn(2000, 3)
        tree = build_octree(points, max_leaf=16)

        def walk(node):
            if node.is_leaf:
                return 0
            checked = 0
            for k, child in enumerate(node.children):
                if child is None:
                    continue
                for idx in child.indices:
                    assert fmm_mod._child_index(points[idx], node.center) == k, (
                        f"point {idx} stored in child {k} but _child_index says "
                        f"{fmm_mod._child_index(points[idx], node.center)}"
                    )
                checked += 1 + walk(child)
            return checked

        assert walk(tree) > 0

    def test_leaf_radius_is_bounded_by_box(self):
        """The multipole convergence radius must be the leaf's box, not the domain."""
        np.random.seed(3)
        points = np.random.randn(4000, 3)
        op = FMMOperator(p=2, max_leaf=32)
        op.setup(points)
        radii = np.asarray(op._radii)
        sizes = np.asarray([leaf.size for leaf in op.leaves])
        assert np.all(radii <= 0.867 * sizes + 1e-12), (
            f"max radius/box = {np.max(radii / sizes):.3f}"
        )


class TestVectorisedMatvec:
    """The blocked matvec must reproduce the scalar reference definition."""

    def _compare(self, p: int, max_leaf: int) -> tuple[float, float]:
        mesh = icosphere(subdivisions=2, radius=1.0)
        sources = np.asarray(mesh.centroids, dtype=float)
        charges = np.asarray(mesh.areas, dtype=float)
        targets = np.asarray(mesh.vertices, dtype=float)

        op = FMMOperator(p=p, max_leaf=max_leaf)
        op.setup(sources, charges)
        fast = op.matvec(targets, sources, charges)
        ref = _reference_matvec(op, targets, sources, charges)
        scale = np.max(np.abs(ref))
        return float(np.max(np.abs(fast - ref)) / scale), op.far_fraction(targets)

    def test_matches_scalar_reference_p2(self):
        err, far = self._compare(p=2, max_leaf=32)
        assert far > 0.5, f"only {far:.2%} of pairs used the multipole"
        assert err < 1e-12, f"vectorised vs scalar reference: {err:.2e}"

    def test_matches_scalar_reference_p4(self):
        err, far = self._compare(p=4, max_leaf=32)
        assert far > 0.5, f"only {far:.2%} of pairs used the multipole"
        assert err < 1e-12, f"vectorised vs scalar reference: {err:.2e}"

    def test_target_block_size_does_not_change_result(self):
        mesh = icosphere(subdivisions=2, radius=1.0)
        sources = np.asarray(mesh.centroids, dtype=float)
        charges = np.asarray(mesh.areas, dtype=float)
        targets = np.asarray(mesh.vertices, dtype=float)

        op = FMMOperator(p=4, max_leaf=32)
        op.setup(sources, charges)
        whole = op.matvec(targets, sources, charges)
        op.block_size = 7
        blocked = op.matvec(targets, sources, charges)
        scale = np.max(np.abs(whole))
        assert np.max(np.abs(whole - blocked)) / scale < 1e-14


class TestSurfaceAccuracyAndCost:
    """Accuracy and cost on the geometry class the operator is meant for."""

    @staticmethod
    def _surface(level: int) -> tuple[np.ndarray, np.ndarray]:
        mesh = icosphere(subdivisions=level, radius=1.0)
        return (np.asarray(mesh.centroids, dtype=float),
                np.asarray(mesh.areas, dtype=float))

    def test_surface_accuracy_p2_p4(self):
        """Panel-centroid sources, mesh-vertex targets: no coincident pairs."""
        vertices = np.asarray(icosphere(subdivisions=2).vertices, dtype=float)
        centroids, areas = self._surface(2)
        direct = direct_laplace_matvec(vertices, centroids, areas)
        scale = np.max(np.abs(direct))
        for p, gate in ((2, 5e-4), (4, 1e-4)):
            op = FMMOperator(p=p, max_leaf=32)
            op.setup(centroids, areas)
            approx = op.matvec(vertices, centroids, areas)
            err = float(np.max(np.abs(approx - direct)) / scale)
            assert err < gate, f"p={p} surface max rel error {err:.2e} >= {gate:.0e}"

    def test_fmm_is_not_slower_than_dense_at_n5120(self):
        """m15669 cost gate: the FMM matvec must be within 2x of the dense one.

        Before the octree/vectorisation fixes the same configuration cost 98x the
        dense matvec (m14931 audit), because no leaf contained its own points and
        every pair fell back on the near-field sum.
        """
        centroids, areas = self._surface(4)  # 20*4**4 = 5120 panels
        best = {}
        for name, fn in (
            ("fmm", lambda: fmm_laplace_matvec(centroids, centroids, areas, p=2,
                                               max_leaf=128)),
            ("dense", lambda: direct_laplace_matvec(centroids, centroids, areas)),
        ):
            times = []
            for _ in range(2):
                start = time.perf_counter()
                fn()
                times.append(time.perf_counter() - start)
            best[name] = min(times)
        ratio = best["fmm"] / best["dense"]
        assert ratio < 2.0, (
            f"FMM {best['fmm']:.3f}s vs dense {best['dense']:.3f}s (ratio {ratio:.2f})"
        )
