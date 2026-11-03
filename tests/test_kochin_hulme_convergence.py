"""Regression pins for the Kochin/Hulme surge-damping convergence study (m2609).

``test_kochin_validation.py::test_surge_damping_vs_hulme`` was red on main:
at 256 panels the BEM surge radiation damping sat 3.53 % below Hulme (1982),
outside the 2.5 % gate.  The deciding panel study
(:mod:`tensorbem.benchmarks.kochin_hulme_convergence`) showed the offset is a
discretisation error converging to the reference at order ~2 in ``h``
(40.2 % -> 13.0 % -> 3.53 % -> 0.90 % over 16/64/256/1024 panels), so the
gate test moved to 1024 panels.  These pins lock the *behaviour* at the cheap
resolutions so a future change that flips the error's sign, breaks monotone
convergence, or degrades the order is caught without paying the 1024-panel
cost.
"""

from __future__ import annotations

import math

import pytest

from tensorbem.benchmarks.kochin_hulme_convergence import run_study


@pytest.fixture(scope="module")
def study():
    return run_study([2, 3], ka=1.0, radius=1.0)


def _row(study, subdivisions):
    return next(r for r in study["rows"] if r["subdivisions"] == subdivisions)


def test_surge_damping_error_converges_to_hulme(study):
    """B11 error is negative, monotone, and second-order in h (not systematic)."""
    lo, hi = _row(study, 2), _row(study, 3)
    e_lo, e_hi = lo["err_b11_near"], hi["err_b11_near"]
    assert e_lo < 0.0 and e_hi < 0.0, (
        f"surge damping error sign changed (s=2: {e_lo:.4%}, s=3: {e_hi:.4%}); "
        "a sign flip would indicate a normalisation defect, not discretisation"
    )
    assert abs(e_hi) < abs(e_lo), "surge damping error did not shrink on refinement"
    # order in h (h halves per subdivision step)
    order = math.log2(abs(e_lo / e_hi))
    assert 1.5 < order < 2.5, f"observed order in h {order:.2f} outside (1.5, 2.5)"
    # pin the current state: 256 panels must stay in the measured band
    assert 0.02 < abs(e_hi) < 0.05, (
        f"256-panel surge damping error {e_hi:.4%} left the pinned band (2%, 5%)"
    )


def test_heave_damping_error_converges_to_hulme(study):
    """B33 error stays small and shrinks on refinement (control quantity)."""
    lo, hi = _row(study, 2), _row(study, 3)
    assert abs(hi["err_b33_near"]) < abs(lo["err_b33_near"])
    assert abs(hi["err_b33_near"]) < 0.02


def test_hulme_reference_is_self_consistent(study):
    """The reference's own near/far damping offset is far below any gate."""
    ref = study["reference"]
    assert abs(ref["self_offset_b11"]) < 1.0e-4
    assert abs(ref["self_offset_b33"]) < 1.0e-3


def test_far_near_energy_identity_improves_with_refinement(study):
    """Kochin far/near damping ratio approaches 1 from below on refinement."""
    lo, hi = _row(study, 2), _row(study, 3)
    assert lo["ratio_b11"] < hi["ratio_b11"] < 1.0
    assert lo["ratio_b33"] < hi["ratio_b33"] < 1.0
