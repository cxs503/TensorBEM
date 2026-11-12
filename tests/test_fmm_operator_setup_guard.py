"""``FMMOperator`` must not answer before ``setup`` -- and ``far_fraction`` must
measure what ``matvec`` actually does.

Motivation (m16156).  ``docs/fmm_operator_linearity/README.md`` (m16000) left one
open contradiction: on the very geometry whose second-call error was ``2.46``
(246 %, i.e. the multipole path *was* exercised), ``far_fraction(targets)`` was
reported as ``0.0`` -- "no far pairs".  The two numbers cannot both describe one
operator.

Root cause found here: the evidence script called

    FMMOperator(p=4, max_leaf=25).far_fraction(targets)

**without** ``setup()``.  ``FMMOperator`` is a dataclass whose tree/leaves are
filled in by ``setup``; an un-set-up instance has ``self.leaves == []``, so

* ``matvec`` returned an array of **zeros** (silently -- no exception),
* ``far_fraction`` returned ``0.0``.

``far_fraction`` returning ``0.0`` was therefore an artefact of a fail-silent
API, not a defect of the near/far routing; after ``setup`` the same geometry
reports ``far_fraction = 0.1739`` and the m16000 error is consistent with it.
The guard below turns "silently wrong answer" into "loud error", and the tests
pin the invariant that ties the diagnostic to the operator semantics:

    ``far_fraction(targets) == 0``  =>  ``matvec`` is the dense direct sum.

Everything in this module FAILS on the pre-m16156 code.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.fmm import FMMOperator, direct_laplace_matvec


def _separated_geometry(seed: int = 42):
    """The m16000 geometry: a compact source cluster plus three far targets."""
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(200, 3)) * 0.5
    targets = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    charges = rng.normal(size=200)
    return sources, targets, charges


# ---------------------------------------------------------------------------
# 1. no silent zeros: an un-set-up operator refuses to answer
# ---------------------------------------------------------------------------

class TestSetupGuard:
    def test_matvec_before_setup_raises(self):
        """Pre-fix this returned ``zeros(3)``: a silent wrong answer."""
        sources, targets, charges = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        with pytest.raises(RuntimeError, match="setup"):
            op.matvec(targets, sources, charges)

    def test_far_fraction_before_setup_raises(self):
        """Pre-fix this returned ``0.0`` -- the m16000 "no far pairs" artefact."""
        _, targets, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        with pytest.raises(RuntimeError, match="setup"):
            op.far_fraction(targets)

    def test_far_work_fraction_before_setup_raises(self):
        _, targets, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        with pytest.raises(RuntimeError, match="setup"):
            op.far_work_fraction(targets)

    def test_guard_clears_after_setup(self):
        sources, targets, charges = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)
        phi = op.matvec(targets, sources, charges)
        assert np.any(phi != 0.0)
        assert op.far_fraction(targets) > 0.0

    def test_setup_rejects_empty_point_set(self):
        """Pre-fix this surfaced as a numpy ``zero-size array to reduction`` error."""
        op = FMMOperator(p=4, max_leaf=25)
        with pytest.raises(ValueError, match="at least one source point"):
            op.setup(np.zeros((0, 3)))


# ---------------------------------------------------------------------------
# 2. the diagnostic must describe the operator (the m16000 contradiction)
# ---------------------------------------------------------------------------

class TestFarFractionDescribesMatvec:
    def test_m16000_geometry_has_far_pairs(self):
        """The geometry of the 2.46-error experiment *does* use the multipoles."""
        sources, targets, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)
        far = op.far_fraction(targets)
        assert far == pytest.approx(4.0 / 23.0, abs=1e-12), far
        assert 0.1 < far < 0.25

    def test_far_fraction_is_independent_of_block_size(self):
        sources, targets, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)
        whole = op.far_fraction(targets)
        op.block_size = 1
        assert op.far_fraction(targets) == whole

    def test_far_fraction_counts_pair_fraction_not_work(self):
        """Pin the published definition: pairs, not source-weighted work."""
        sources, targets, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)

        leaf_of_target = op._route(targets)
        near_pairs = far_pairs = 0
        for start in range(0, len(targets), op.block_size):
            block = targets[start:start + op.block_size]
            _, _, near = op._near_mask(block, leaf_of_target[start:start + op.block_size])
            near_pairs += int(near.sum())
            far_pairs += int((~near).sum())
        assert np.isclose(op.far_fraction(targets), far_pairs / (near_pairs + far_pairs))

        # ... while the source-weighted work fraction is a different number
        leaf_sizes = np.array([len(idx) for idx in op._leaf_indices], dtype=float)
        work = 0.0
        total = 0.0
        for start in range(0, len(targets), op.block_size):
            block = targets[start:start + op.block_size]
            _, _, near = op._near_mask(block, leaf_of_target[start:start + op.block_size])
            work += float(((~near) * leaf_sizes[None, :]).sum())
            total += float((np.ones_like(near) * leaf_sizes[None, :]).sum())
        work_fraction = work / total
        assert np.isclose(op.far_work_fraction(targets), work_fraction)
        assert not np.isclose(work_fraction, op.far_fraction(targets), atol=1e-3)

    def test_work_fraction_equals_pair_fraction_for_equal_leaf_occupancy(self):
        """With one source per leaf the two definitions coincide."""
        sources = np.array([[float(i), 0.0, 0.0] for i in range(8)])
        targets = sources + np.array([50.0, 0.0, 0.0])
        op = FMMOperator(p=4, max_leaf=1)
        op.setup(sources)
        assert np.isclose(op.far_fraction(targets), op.far_work_fraction(targets))

    def test_zero_far_fraction_means_dense_sum(self):
        """The invariant behind the m16000 artefact: no far pairs => direct sum."""
        rng = np.random.default_rng(7)
        sources = rng.normal(size=(40, 3)) * 0.3
        charges = rng.normal(size=40)
        targets = rng.normal(size=(11, 3))
        op = FMMOperator(p=4, max_leaf=len(sources))  # a single leaf
        op.setup(sources)
        assert op.far_fraction(targets) == 0.0

        phi = op.matvec(targets, sources, charges)
        direct = direct_laplace_matvec(targets, sources, charges)
        err = float(np.max(np.abs(phi - direct)) / np.max(np.abs(direct)))
        assert err < 1e-14, f"far_fraction == 0 but matvec != dense: {err:.3e}"

    def test_positive_far_fraction_changes_the_answer(self):
        """Zeroing the multipoles must move the answer iff far pairs exist."""
        sources, targets, charges = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources, charges)
        assert op.far_fraction(targets) > 0.0

        phi = op.matvec(targets, sources, charges).copy()
        op._multipole_matrix[:] = 0.0
        op._cache_charges = np.array(charges, copy=True)
        phi_far_removed = op.matvec(targets, sources, charges)
        assert np.max(np.abs(phi - phi_far_removed)) > 1e-3