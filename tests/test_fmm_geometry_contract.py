"""Geometry contract of ``tensorbem.fmm.FMMOperator`` (m16265).

The operator derives three things from the point set passed to :meth:`setup`:

* the octree and therefore the routing of a target to a leaf,
* the per-leaf source indices used by the direct near-field sum,
* the leaf centres/radii used by both the admissibility test and the multipole
  expansion.

``matvec(targets, sources, charges)`` used to accept *any* ``sources`` array.
For a point set of the same length but different positions that was **fail-open**:
multipoles were rebuilt about the old leaf centres while the near field indexed
the new points with the old leaf partition, so the answer looked plausible and
was 3.7 %--6.8 % away from the dense sum (see ``scripts/fmm_geometry_contract.py``,
``docs/fmm_geometry_contract/evidence.json``).  A point set of a different length
crashed with a bare ``IndexError`` from inside the near-field loop.

These tests pin the fail-closed contract, and pin that the two things that *are*
allowed to change between calls -- the charges (m16000: multiple right-hand
sides, iterative solvers) and the targets -- still work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.fmm import FMMOperator, direct_laplace_matvec

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "docs" / "fmm_geometry_contract" / "evidence.json"


class _PrefixGeometryOperator(FMMOperator):
    """Pre-m16265 semantics: no geometry validation in ``matvec``.

    The only difference from the shipped class is the validation predicate, so
    it isolates exactly the contract added by m16265.
    """

    def _require_setup_geometry(self, sources):  # noqa: D102
        return None


def _geometry(seed: int = 7, n: int = 200):
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(n, 3)) * 0.5
    charges = rng.normal(size=n)
    targets = np.array([[8.0, 0.0, 0.0], [0.0, 6.0, 0.0], [4.0, 4.0, 0.0]])
    return sources, targets, charges


def _rel(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(np.abs(approx - exact)) / np.max(np.abs(exact)))


def test_mismatched_geometry_is_rejected_instead_of_silently_wrong():
    """A shifted source set used to return a plausible but wrong answer."""
    sources, targets, charges = _geometry()
    shifted = sources + np.array([10.0, 0.0, 0.0])

    prefix = _PrefixGeometryOperator(p=4, max_leaf=25)
    prefix.setup(sources)
    silent = prefix.matvec(targets, shifted, charges)  # pre-fix behaviour
    exact = direct_laplace_matvec(targets, shifted, charges)
    prefix_error = _rel(silent, exact)
    # the defect was silent, i.e. no exception, but O(1e-2) wrong
    assert prefix_error > 1.0e-2, prefix_error

    fixed = FMMOperator(p=4, max_leaf=25)
    fixed.setup(sources)
    with pytest.raises(ValueError, match="differ from the point set"):
        fixed.matvec(targets, shifted, charges)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p * 3.0,
        lambda p: p * 20.0,
        lambda p: p[np.roll(np.arange(len(p)), 1)],
        lambda p: p + 1.0e-9,  # sub-ulp-scale drift is still a different set
    ],
    ids=["scale_x3", "scale_x20", "permuted", "epsilon_shift"],
)
def test_any_different_point_set_is_rejected(mutate):
    sources, targets, charges = _geometry()
    op = FMMOperator(p=4, max_leaf=25)
    op.setup(sources)
    with pytest.raises(ValueError):
        op.matvec(targets, mutate(sources.copy()), charges)


@pytest.mark.parametrize("n", [50, 199, 201])
def test_point_count_mismatch_raises_valueerror_not_indexerror(n):
    """A shorter/longer source set used to raise a bare ``IndexError``."""
    sources, targets, _ = _geometry()
    other, _, charges = _geometry(seed=11, n=n)
    op = FMMOperator(p=4, max_leaf=25)
    op.setup(sources)
    with pytest.raises(ValueError, match="shape"):
        op.matvec(targets, other, charges)


def test_setup_geometry_answers_match_the_dense_sum():
    """Baseline: with the contract respected the operator still works."""
    sources, targets, charges = _geometry()
    op = FMMOperator(p=4, max_leaf=25)
    op.setup(sources)
    got = op.matvec(targets, sources, charges)
    exact = direct_laplace_matvec(targets, sources, charges)
    assert _rel(got, exact) < 1.0e-3


def test_charge_vectors_may_still_change_between_calls():
    """m16000 contract kept: the cache is keyed on (points, charges), not once-only."""
    sources, targets, charges = _geometry()
    _, _, other_charges = _geometry(seed=13)
    op = FMMOperator(p=4, max_leaf=25)
    op.setup(sources)
    for q in (charges, other_charges, charges):
        got = op.matvec(targets, sources, q)
        exact = direct_laplace_matvec(targets, sources, q)
        assert _rel(got, exact) < 1.0e-3


def test_targets_may_still_change_between_calls():
    """Targets are free: routing is recomputed per call from the setup octree."""
    sources, _, charges = _geometry()
    op = FMMOperator(p=4, max_leaf=25)
    op.setup(sources)
    for targets in (
        np.array([[8.0, 0.0, 0.0]]),
        np.array([[0.0, 6.0, 0.0], [4.0, 4.0, 0.0], [-3.0, 0.0, 2.0]]),
    ):
        got = op.matvec(targets, sources, charges)
        exact = direct_laplace_matvec(targets, sources, charges)
        assert _rel(got, exact) < 1.0e-3


def test_unset_operator_still_fails_closed():
    """m16109 regression guard: unset operators raise instead of returning zeros."""
    _, targets, charges = _geometry()
    op = FMMOperator(p=4, max_leaf=25)
    # shapes must be self-consistent, otherwise the charge check fires first
    with pytest.raises(RuntimeError):
        op.matvec(targets, np.zeros((3, 3)), charges[:3])


def test_committed_evidence_reproduces():
    """The numbers quoted in ``docs/fmm_geometry_contract/README.md`` are pinned."""
    if not EVIDENCE.exists():  # pragma: no cover - artifact ships with the change
        pytest.fail(f"missing evidence artifact {EVIDENCE}")
    payload = json.loads(EVIDENCE.read_text(encoding="utf8"))
    sources, targets, charges = _geometry()
    for case in payload["cases"]:
        moved = np.asarray(case["sources"], dtype=float)
        prefix = _PrefixGeometryOperator(p=4, max_leaf=25)
        prefix.setup(np.asarray(payload["setup_points"], dtype=float))
        got = prefix.matvec(np.asarray(case["targets"], dtype=float), moved, charges)
        exact = direct_laplace_matvec(
            np.asarray(case["targets"], dtype=float), moved, charges
        )
        assert _rel(got, exact) == pytest.approx(case["prefix_rel_err"], rel=0.25)