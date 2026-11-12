"""Fail-closed setup contract of :class:`tensorbem.fmm.FMMOperator` (m16109).

m16000 limitation #2 registered that ``far_fraction()`` returned 0.0 on the
linearity-bench geometry, contradicting the measured 2.46 far-field-cache
error.  Root cause: the bench called ``far_fraction`` on a *fresh, never
set-up* operator, and the public API was fail-open --

* ``matvec`` on an unset operator silently returned an all-zero vector
  (100% relative error vs the direct sum, no warning),
* ``far_fraction`` on an unset operator silently returned 0.0,
  indistinguishable from "0% of the pairs are far".

Both are now ``RuntimeError``.  Every test in this module FAILS on the
pre-fix code (the two ``raises`` tests observe the silent values instead).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorbem.fmm import FMMOperator, direct_laplace_matvec


def _separated_geometry(seed: int = 42):
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(200, 3)) * 0.5
    targets = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    charges = rng.normal(size=200)
    return sources, targets, charges


class TestUnsetOperatorFailsClosed:
    def test_matvec_before_setup_raises(self):
        sources, targets, charges = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        with pytest.raises(RuntimeError, match="setup"):
            op.matvec(targets, sources, charges)

    def test_far_fraction_before_setup_raises(self):
        """The exact m16000 observation: 0.0 on an unset operator is gone."""
        sources, targets, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        with pytest.raises(RuntimeError, match="setup"):
            op.far_fraction(targets)

    def test_error_message_names_the_method(self):
        sources, targets, charges = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        with pytest.raises(RuntimeError, match="matvec"):
            op.matvec(targets, sources, charges)
        with pytest.raises(RuntimeError, match="far_fraction"):
            op.far_fraction(targets)

    def test_unset_matvec_no_longer_returns_silent_zeros(self):
        """Pre-fix this returned zeros with rel err 1.0 vs the direct sum."""
        sources, targets, charges = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        with pytest.raises(RuntimeError):
            got = op.matvec(targets, sources, charges)
        # the silent answer would have been exactly this:
        assert np.max(np.abs(direct_laplace_matvec(targets, sources, charges))) > 0.0


class TestSetUpOperatorKeepsWorking:
    def test_empty_targets_still_return_empty(self):
        """The legitimate empty case must survive the new guard."""
        sources, _, charges = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)
        got = op.matvec(np.zeros((0, 3)), sources, charges)
        assert got.shape == (0,)

    def test_empty_targets_far_fraction_is_zero(self):
        sources, _, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)
        assert op.far_fraction(np.zeros((0, 3))) == 0.0

    def test_far_fraction_reports_true_fraction_after_setup(self):
        """The m16000 contradiction resolved: the diagnostic now reports the
        actual multipole-pair fraction (>0.1 on the separated geometry)."""
        sources, targets, _ = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)
        assert op.far_fraction(targets) > 0.1

    def test_setup_then_matvec_matches_direct(self):
        sources, targets, charges = _separated_geometry()
        op = FMMOperator(p=4, max_leaf=25)
        op.setup(sources)
        got = op.matvec(targets, sources, charges)
        exact = direct_laplace_matvec(targets, sources, charges)
        rel = float(np.max(np.abs(got - exact)) / np.max(np.abs(exact)))
        assert rel < 1e-5, f"set-up matvec regressed: {rel:.3e}"
