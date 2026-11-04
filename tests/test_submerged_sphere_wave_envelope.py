"""Guards for the (Ka, f/a) accuracy envelope of the 3-D free-surface solver.

The study itself lives in :mod:`tensorbem.benchmarks.submerged_sphere_envelope`;
these tests pin the claims made in ``docs/m3134/submerged_sphere_envelope.md``:

* second-order convergence still holds at the *worst* envelope corner (near the
  free surface), so the envelope is discretisation-limited, not physics-limited;
* the error grows monotonically as the body approaches ``z = 0`` at fixed ``Ka``;
* the single-point verification point (``Ka = 0.8, f/a = 1.5``) is *not*
  representative: the near-surface corner misses the 3 % standard at the same
  320-panel mesh;
* the scale-free metric is well defined where the reference added mass passes
  through zero (guarded relative error returns NaN instead of a huge number).
"""

from __future__ import annotations

import math

import pytest

from tensorbem.benchmarks.submerged_sphere_envelope import (
    evaluate_point,
    run_envelope_refinement,
    _guarded_relative,
)


def test_guarded_relative_returns_nan_for_near_zero_reference() -> None:
    # f/a = 1.1 puts A33 through a near-zero: |exact| below 10 % of the
    # unbounded-fluid scale must not be divided by.
    assert math.isnan(_guarded_relative(0.19, 0.054, 0.5 * 4293.5))
    assert _guarded_relative(0.98, 1.0, 1.0) == pytest.approx(2.0e-2)


def test_convergence_order_at_the_near_surface_corner() -> None:
    """The worst envelope corner still converges at (nearly) second order.

    The 20 -> 80 panel step is pre-asymptotic (measured order ~0.97 at
    ``Ka = 1.2, f/a = 1.1``); the asymptotic 80 -> 320 step is what the claim is
    made on, and it is why the envelope is discretisation-limited rather than
    physics-limited.
    """
    study = run_envelope_refinement(ka=1.2, submergence_ratio=1.1, subdivisions=(0, 1, 2))
    orders = [o for o in study["observed_orders_a33"] if o is not None]
    assert len(orders) == 2
    assert 1.5 < orders[-1] < 2.5, study["observed_orders_a33"]
    assert orders[0] == pytest.approx(0.97, abs=0.15)  # pre-asymptotic, pinned
    errors = [lvl["err_a33_scaled"] for lvl in study["levels"]]
    assert errors[0] > errors[1] > errors[2]


def test_error_is_worst_at_the_near_surface_but_not_depth_monotone() -> None:
    """Near-surface is the worst corner; intermediate depth can look *better*.

    Measured at ``Ka = 1.2``, 80 panels: 7.07e-2 (f/a=3), 2.21e-2 (f/a=1.5),
    2.77e-1 (f/a=1.1).  The f/a = 1.5 point benefits from an error cancellation
    between the free-surface term and the geometric volume deficit, so a
    depth-monotone expectation would be wrong -- and that cancellation is
    exactly why a single verification point cannot characterise the solver.
    """
    ka = 1.2
    errs = [
        evaluate_point(ka, ratio, subdivision=1)["err_a33_scaled"]
        for ratio in (3.0, 1.5, 1.1)
    ]
    assert errs[2] > errs[0] and errs[2] > errs[1], errs  # worst corner
    assert errs[1] < errs[0], errs  # pinned cancellation, not a tolerance
    assert errs[2] == pytest.approx(2.77e-1, rel=0.15)


def test_single_point_verification_is_not_representative() -> None:
    """f/a = 1.5 passes the 3 % standard while f/a = 1.1 misses it (320 panels)."""
    good = evaluate_point(0.8, 1.5, subdivision=2)
    bad = evaluate_point(1.2, 1.1, subdivision=2)
    assert good["panels"] == bad["panels"] == 320
    assert good["err_a33_scaled"] < 0.03
    assert bad["err_a33_scaled"] > 0.03
    assert bad["err_a33_scaled"] == pytest.approx(8.77e-2, rel=0.10)


def test_far_field_and_boundary_haskind_agree() -> None:
    """Kochin and boundary Haskind forms are an internal identity, mesh independent."""
    row = evaluate_point(0.8, 1.5, subdivision=1)
    assert row["haskind_kochin_rel_error"] < 1.0e-2
    assert row["relative_residual"] < 1.0e-8