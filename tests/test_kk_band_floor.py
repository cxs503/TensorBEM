"""Tests for :mod:`tensorbem.benchmarks.kk_band_floor` (KK estimator floor).

The tool exists to separate two error sources in a Kramers-Kronig residual: the
piecewise-linear discretisation of the tabulated sweep and the truncation of the
principal-value integral at the band edges.  These tests pin it on a **known
causal pair** (an exponential-sum memory kernel, whose ``(A, B)`` satisfy both KK
relations identically -- the identity locked down by ``tests/test_causality.py``),
so any residual the tool reports there is estimator error by construction.
"""

from __future__ import annotations

import numpy as np

from tensorbem.benchmarks import kk_band_floor as fb


def _causal_pair(omega, coefficients, rates):
    """``(A, B)`` of ``K(s) = sum_j c_j exp(-a_j s)`` (exactly causal)."""
    a = np.asarray(rates, dtype=float).reshape(1, -1)
    c = np.asarray(coefficients, dtype=float).reshape(1, -1)
    w = np.asarray(omega, dtype=float).reshape(-1, 1)
    denom = a**2 + w**2
    damping = np.sum(c * a / denom, axis=1)
    delta = -np.sum(c / denom, axis=1)
    return delta, damping


def test_exact_causal_pair_has_no_residual_on_a_fine_grid() -> None:
    """Two well-separated relaxation rates: the estimator must see causality."""
    rates = (0.4, 6.0)
    coefficients = (-30.0, 900.0)          # mixed signs: a radiation kernel is not positive
    omega = np.geomspace(0.05, 40.0, 200)
    delta, damping = _causal_pair(omega, coefficients, rates)
    report = fb.kk_residual(omega, delta, damping, added_mass_inf=0.0)
    # 200 geometric nodes over three decades still leave a few percent: the
    # spacing grows with omega, and the estimator's piecewise-linear integral is
    # the limiting model.  What matters is that a causally exact datum stays at
    # the few-percent level instead of the tens of percent the panel sweep shows.
    assert report["max"] < 0.10, report


def test_estimator_floor_separates_discretisation_from_band() -> None:
    """A coarse sampling of the same datum must cost more than a refined one."""
    rates = (0.4, 6.0)
    coefficients = (-30.0, 900.0)
    fine = np.geomspace(0.05, 40.0, 200)
    delta, damping = _causal_pair(fine, coefficients, rates)
    coarse = fine[::10]
    coarse_delta, coarse_damping = _causal_pair(coarse, coefficients, rates)

    floor = fb.estimator_floor(coarse, coarse_delta, coarse_damping,
                               added_mass_inf=0.0, dense_points=601)
    assert floor["residual_coarse"] >= floor["residual_coarse_rms"] >= 0.0
    assert floor["band"] == [float(coarse[0]), float(coarse[-1])]
    # the refinement samples the same analytic datum, so the floor cannot grow
    # away from the band edges; a band that is wide enough to hold both rates
    # leaves the estimator with nothing but a small residual
    assert floor["residual_refined"] < 0.05, floor


def test_extended_band_reports_a_band_attributable_part() -> None:
    """Supplying a wider band of the same DOF turns band truncation into a number."""
    rates = (0.4, 6.0)
    coefficients = (-30.0, 900.0)
    narrow = np.geomspace(0.6, 8.0, 20)
    wide = np.geomspace(0.05, 40.0, 40)
    n_delta, n_damping = _causal_pair(narrow, coefficients, rates)
    w_delta, w_damping = _causal_pair(wide, coefficients, rates)
    floor = fb.estimator_floor(narrow, n_delta, n_damping, added_mass_inf=0.0,
                               extended=(wide, w_delta, w_damping))
    assert floor["band_attributable"] is not None
    assert floor["band_attributable"] >= 0.0
    assert floor["extended_band"]["band"] == [float(wide[0]), float(wide[-1])]


def test_refinement_does_not_invent_extrema() -> None:
    """PCHIP re-expression must not raise or move a single damping peak."""
    omega = np.linspace(0.3, 8.0, 13)
    damping = 1.0 / (1.0 + ((omega - 5.0) / 0.3) ** 2)
    added = np.full(omega.size, 2.0)
    dense, dense_a, dense_b = fb.refine_sweep(omega, added, damping, points=401)
    assert dense.size == 401
    assert np.max(dense_b) <= np.max(damping) + 1.0e-12
    # PCHIP moves nothing: the refined peak sits on the tabulated peak node
    assert abs(float(dense[dense_b.argmax()]) - float(omega[damping.argmax()])) < 0.05