"""Linearity contracts of :class:`tensorbem.fmm.FMMOperator` as a *linear operator*.

Motivation (m16000): ``FMMOperator.matvec`` caches the multipole coefficients
(``_multipole_matrix``) that ``setup`` or the first ``matvec`` call built from a
particular charge vector.  Everything downstream of the cache -- far-field
multipole evaluation, ``_far_contributions`` -- is therefore frozen at the
charges of that first call while the near-field sum always uses the charges
handed to the current call.  The result is an operator that

* gives a machine-precision answer for its *first* charge vector,
* silently returns a wrong answer for every *later* charge vector (measured
  pre-fix relative error 6.2e-2 on a well-separated geometry), and
* violates superposition across calls (measured pre-fix residual 1.4e-1),

which makes it unusable inside an iterative solver or for multiple right-hand
sides.  The tests below pin the contract "``matvec(targets, sources, charges)``
is a linear functional of ``charges``" so the stale-cache defect cannot come
back: every test in this module FAILS on the pre-fix code.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres

from tensorbem import geometry3d
from tensorbem.fmm import (
    FMMOperator,
    direct_laplace_matvec,
    fmm_laplace_matvec,
)


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

def _separated_geometry(seed: int = 42):
    """A compact source cluster plus far targets: multipole path is used.

    ``max_leaf=25`` on 200 sources gives leaf radii << target distance, so a
    third of the target-leaf pairs are multipole-evaluated (``far_fraction``).
    """
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(200, 3)) * 0.5
    targets = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    charges_a = rng.normal(size=200)
    charges_b = rng.normal(size=200)
    return sources, targets, charges_a, charges_b


def _rel_err(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(np.abs(approx - exact)) / np.max(np.abs(exact)))


# ---------------------------------------------------------------------------
# 1. superposition across separate calls (the core linearity contract)
# ---------------------------------------------------------------------------

class TestLinearityAcrossCalls:
    def test_second_call_with_new_charges_matches_direct(self):
        """Call 2 must be as accurate as call 1, for *different* charges."""
        sources, targets, charges_a, charges_b = _separated_geometry()
        op = FMMOperator(p=6, max_leaf=25)
        op.setup(sources)
        assert op.far_fraction(targets) > 0.1, "multipole path must be exercised"

        first = op.matvec(targets, sources, charges_a)
        err_first = _rel_err(first, direct_laplace_matvec(targets, sources, charges_a))
        second = op.matvec(targets, sources, charges_b)
        err_second = _rel_err(second, direct_laplace_matvec(targets, sources, charges_b))

        assert err_first < 1e-6, f"first call error {err_first:.3e}"
        assert err_second < 1e-6, (
            f"second call error {err_second:.3e} (first {err_first:.3e}); "
            "multipole cache was reused for a different charge vector"
        )

    def test_superposition_across_calls(self):
        """S(q1 + q2) == S(q1) + S(q2) when evaluated through the same operator."""
        sources, targets, charges_a, charges_b = _separated_geometry()
        op = FMMOperator(p=6, max_leaf=25)
        op.setup(sources)

        s_a = op.matvec(targets, sources, charges_a)
        s_b = op.matvec(targets, sources, charges_b)
        s_ab = op.matvec(targets, sources, charges_a + charges_b)
        scale = np.max(np.abs(direct_laplace_matvec(targets, sources, charges_a + charges_b)))
        residual = float(np.max(np.abs(s_ab - (s_a + s_b))) / scale)

        assert residual < 1e-6, (
            f"nonlinear operator: |S(q1+q2) - (S q1 + S q2)| / scale = {residual:.3e}"
        )

    def test_scaling_across_calls(self):
        """S(alpha q) == alpha S(q) for alpha != 1 evaluated after another call."""
        sources, targets, charges_a, charges_b = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)

        base = op.matvec(targets, sources, charges_b)
        scaled = op.matvec(targets, sources, 3.5 * charges_b)
        residual = float(np.max(np.abs(scaled - 3.5 * base)) / np.max(np.abs(base)))

        assert residual < 1e-12, f"homogeneity violated: {residual:.3e}"

    def test_repeat_same_charges_is_bitwise_identical(self):
        """Cache reuse for an *unchanged* charge vector stays admissible."""
        sources, targets, charges_a, _ = _separated_geometry()
        op = FMMOperator(p=6, max_leaf=25)
        op.setup(sources)

        first = op.matvec(targets, sources, charges_a)
        second = op.matvec(targets, sources, charges_a)
        assert np.array_equal(first, second)

    def test_in_place_charge_mutation_is_seen(self):
        """A mutated (aliased) charge array must not reuse stale multipoles."""
        sources, targets, charges_a, charges_b = _separated_geometry()
        op = FMMOperator(p=6, max_leaf=25)
        op.setup(sources)

        charges = charges_a.copy()
        op.matvec(targets, sources, charges)
        charges[...] = charges_b  # in-place: same buffer, different values
        got = op.matvec(targets, sources, charges)
        exact = direct_laplace_matvec(targets, sources, charges_b)

        assert _rel_err(got, exact) < 1e-6, "stale cache after in-place mutation"


# ---------------------------------------------------------------------------
# 2. stateless entry point and setup-with-charges consistency
# ---------------------------------------------------------------------------

class TestStatelessEntryPoint:
    def test_forward_call_agrees_with_operator(self):
        """``fmm_laplace_matvec`` (fresh operator per call) == repeated operator."""
        sources, targets, charges_a, charges_b = _separated_geometry()
        op = FMMOperator(p=6, max_leaf=25)
        op.setup(sources)
        op.matvec(targets, sources, charges_a)

        stateless = fmm_laplace_matvec(targets, sources, charges_b, p=6, max_leaf=25)
        assert _rel_err(op.matvec(targets, sources, charges_b), stateless) < 1e-12

    def test_setup_with_charges_then_other_charges(self):
        """``setup(sources, q)`` primes the cache; a later ``matvec`` with other
        charges must not inherit it."""
        sources, targets, charges_a, charges_b = _separated_geometry()
        op = FMMOperator(p=6, max_leaf=25)
        op.setup(sources, charges_a)

        got = op.matvec(targets, sources, charges_b)
        exact = direct_laplace_matvec(targets, sources, charges_b)
        assert _rel_err(got, exact) < 1e-6


# ---------------------------------------------------------------------------
# 3. end-to-end: FMM operator inside GMRES reproduces the dense solve
# ---------------------------------------------------------------------------

class TestGMRESSolve:
    def test_gmres_with_fmm_operator_matches_dense_solve(self):
        """Single-layer system solved by GMRES + FMM operator == dense LU solve.

        This is the capability the caching defect silently broke: an iterative
        BEM solve changes the right-hand side (and the iterate, hence the
        equivalent charge vector) on every matvec call.
        """
        mesh = geometry3d.icosphere(1)
        centroids = mesh.centroids
        areas = mesh.areas
        n = len(centroids)

        op = FMMOperator(p=10, max_leaf=16)
        op.setup(centroids)

        def apply(q: np.ndarray) -> np.ndarray:
            # single-layer operator S_ij = G(x_i, y_j) * area_j
            return op.matvec(centroids, centroids, q * areas)

        diff = centroids[:, None, :] - centroids[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        dense = (1.0 / (4.0 * np.pi * np.maximum(r, 1e-15))) * areas[None, :]
        rhs = np.ones(n)
        q_dense = np.linalg.solve(dense, rhs)

        history: list[float] = []

        lin = LinearOperator((n, n), matvec=apply, dtype=float)
        q_iter, info = gmres(lin, rhs, rtol=1e-12, atol=0.0, restart=n,
                             maxiter=60, callback=history.append,
                             callback_type="pr_norm")
        assert info == 0, f"gmres did not converge (info={info})"

        rel = float(np.linalg.norm(q_iter - q_dense) / np.linalg.norm(q_dense))
        assert rel < 1e-6, (
            f"GMRES+FMM solution differs from dense solve by {rel:.3e} "
            f"({len(history)} iterations)"
        )

    def test_gmres_with_fmm_operator_after_warmup_call(self):
        """Warm-up matvec with another vector must not corrupt the solve.

        A stale multipole cache would make GMRES iterate on a *different*
        operator than the one it assembles, so the residual it reports and the
        solution it returns disagree with the true system.
        """
        mesh = geometry3d.icosphere(1)
        centroids = mesh.centroids
        areas = mesh.areas
        n = len(centroids)

        op = FMMOperator(p=10, max_leaf=16)
        op.setup(centroids, np.ones(n))  # primes the cache with q = 1
        op.matvec(centroids, centroids, np.linspace(1.0, 2.0, n))  # unrelated call

        diff = centroids[:, None, :] - centroids[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        dense = (1.0 / (4.0 * np.pi * np.maximum(r, 1e-15))) * areas[None, :]
        rhs = np.ones(n)
        q_dense = np.linalg.solve(dense, rhs)

        lin = LinearOperator(
            (n, n), matvec=lambda q: op.matvec(centroids, centroids, q * areas),
            dtype=float,
        )
        q_iter, info = gmres(lin, rhs, rtol=1e-12, atol=0.0, restart=n, maxiter=60)
        assert info == 0

        rel = float(np.linalg.norm(q_iter - q_dense) / np.linalg.norm(q_dense))
        assert rel < 1e-6, f"post-warmup GMRES solution drifts by {rel:.3e}"


# ---------------------------------------------------------------------------
# 4. input validation (fail loudly, not with a bare IndexError)
# ---------------------------------------------------------------------------

class TestMatvecValidation:
    def test_charges_length_mismatch_raises_value_error(self):
        sources, targets, charges_a, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)
        with pytest.raises(ValueError):
            op.matvec(targets, sources, charges_a[:-1])