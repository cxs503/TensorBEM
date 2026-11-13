"""Pins for :mod:`tensorbem.sk_fit` (Sanathanan-Koerner damping fit).

The suite answers five questions with numbers, not with a smoke run:

1. does the iteration solve an inverse problem it can solve at all -- a curve
   that *is* a sum of three real decay rates must come back with those three
   decay rates, the three residues and the asymptote;
2. is that an improvement -- the pre-rewrite fixed-grid fit
   (:func:`tensorbem.sk_fit.fixed_grid_damping_fit`) must be measurably worse on
   the same order, and the SK entry point must never be worse than it anywhere
   in the order scan (the documented "no regression" invariant);
3. does the caller get the truth about convergence -- ``max_iter`` and ``tol``
   must actually change ``n_iter``/``converged``, which the pre-rewrite code
   silently ignored;
4. does the round-2 pole refinement buy measured accuracy -- on the *measured*
   Hulme curve, where the unrefined projection stalls at 17-49 % of the peak,
   the shipped default must reach the platform's 3 % accuracy bar and must beat
   the unrefined fit by at least a decade (and the claim that those 17-49 %
   were a "model-class floor" is pinned as falsified);
5. is the refinement honest and bounded -- it never worsens the returned fit,
   it is deterministic, it preserves the exact recovery on a representable
   curve, and ``refine_maxfev=0`` reproduces the round-1 numbers bit for bit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import sk_fit_verification as study
from tensorbem.sk_fit import (
    fixed_grid_damping_fit,
    sk_fit_damping,
    sk_fit_damping_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "sk_fit_sanathanan_koerner.md"


@pytest.fixture(scope="module")
def representable():
    return study.representable_case()


@pytest.fixture(scope="module")
def hulme():
    return study.hulme_case()


@pytest.fixture(scope="module")
def study_results():
    """One run of the verification study, shared by the artifact pins."""
    return study.run_study()


def test_recovers_an_exact_three_rate_curve(representable):
    """The SK iteration must invert a representable curve, not approximate it."""
    omega, damping, truth = representable
    fit, report = sk_fit_damping_report(omega, damping, n_poles=3, max_iter=200, tol=1e-12)

    assert report.sk_converged
    assert report.sk_iterations <= 200
    assert report.pole_source == "sk"
    # decay rates, residues and the asymptote are recovered to round-off
    np.testing.assert_allclose(
        np.sort(fit.poles), np.sort(truth["decay_rates"]), rtol=1e-9
    )
    np.testing.assert_allclose(
        fit.residues, np.asarray(truth["residues"]), atol=1e-8
    )
    assert fit.constant == pytest.approx(truth["asymptote"], abs=1e-10)
    assert fit.relative_error_max < 1e-12
    # passivity of the fitted model: a non-negative asymptote
    assert fit.constant >= 0.0


def test_the_pre_rewrite_fixed_grid_cannot_represent_the_same_curve(representable):
    """Baseline comparison: the improvement claimed in the report is measured."""
    omega, damping, _ = representable
    baseline = fixed_grid_damping_fit(omega, damping, n_poles=3)
    sk = sk_fit_damping(omega, damping, n_poles=3, max_iter=200, tol=1e-12)

    assert baseline.relative_error_max > 0.5
    assert sk.relative_error_max < 1e-12
    assert sk.relative_error_max < baseline.relative_error_max * 1e-6


@pytest.mark.parametrize("order", [2, 3, 4, 6, 8])
def test_never_worse_than_the_fixed_grid(order, representable, hulme):
    """The returned fit is the best of *all* candidate pole sets, always."""
    for omega, damping in (representable[:2], hulme):
        fit, report = sk_fit_damping_report(omega, damping, n_poles=order, max_iter=120)
        baseline = fixed_grid_damping_fit(omega, damping, n_poles=order)
        unrefined, _ = sk_fit_damping_report(
            omega, damping, n_poles=order, max_iter=120, refine_maxfev=0
        )
        assert fit.relative_error_max <= baseline.relative_error_max * (1.0 + 1e-12)
        assert fit.relative_error_max <= unrefined.relative_error_max * (1.0 + 1e-12)
        assert min(
            report.sk_pole_rel_error,
            report.grid_pole_rel_error,
            report.refined_pole_rel_error,
        ) == pytest.approx(fit.relative_error_max, rel=1e-12, abs=1e-15)
        assert report.pole_source in {"sk", "grid", "refined"}
        assert report.refine_start in {"sk", "grid", "none"}
        assert report.real_root_count + 2 * report.complex_pair_count == order


def test_max_iter_and_tol_are_not_decorative(representable):
    """``max_iter``/``tol`` were accepted and ignored before this rewrite."""
    omega, damping, _ = representable
    # Pole refinement off: this pin is about the SK iteration, not about the
    # (separately pinned) physical-basis relocation that follows it.
    one, report_one = sk_fit_damping_report(
        omega, damping, n_poles=3, max_iter=1, refine_maxfev=0
    )
    many, report_many = sk_fit_damping_report(
        omega, damping, n_poles=3, max_iter=200, tol=1e-10, refine_maxfev=0
    )

    assert report_one.sk_iterations == 1
    assert one.n_iter == 1
    assert not report_one.sk_converged
    assert report_many.sk_iterations > 1
    assert report_many.sk_converged
    # more iterations never leave the iterate worse than the one-shot fit
    assert many.relative_error_max <= one.relative_error_max * (1.0 + 1e-12)
    # the residual history has one entry per iteration
    assert len(report_many.residual_history) == report_many.sk_iterations


def test_unrefined_pole_projection_stalls_on_the_measured_curve(hulme):
    """Why the round-2 refinement exists: the projection alone stalls at O(10 %)."""
    omega, damping = hulme
    _, report = sk_fit_damping_report(
        omega, damping, n_poles=12, max_iter=200, tol=1e-12, refine_maxfev=0
    )
    assert report.rational_rel_error < 1e-4
    assert report.sk_pole_rel_error > 1e-2
    # the measured curve does not admit a real-decay-rate representation at
    # this order: at least one conjugate pair has to be projected, and that
    # projection is what leaves the real poles badly placed.
    assert report.complex_pair_count >= 1
    assert report.refined_pole_rel_error != report.refined_pole_rel_error  # nan: off
    assert report.pole_source in {"sk", "grid"}


@pytest.mark.parametrize("order", [6, 8, 10, 12])
def test_pole_refinement_meets_the_accuracy_bar_on_the_measured_curve(order, hulme):
    """Measured curve, shipped default: the 3 % bar is met from 6 decay rates up.

    Round 1 of this module reported 17-49 % here and concluded that the
    real-decay-rate basis had a model-class floor.  These numbers pincode the
    falsification of that conclusion: same data, same orders, only the pole
    positions were allowed to move.
    """
    omega, damping = hulme
    fit, report = sk_fit_damping_report(
        omega, damping, n_poles=order, max_iter=200, tol=1e-12
    )
    assert report.pole_source == "refined"
    assert fit.relative_error_max <= study.ACCURACY_BAR
    # and the gain over the *unrefined* SK fit is at least a decade on the same
    # data at the same order: this is the round-2 claim, measured.
    unrefined = min(report.sk_pole_rel_error, report.grid_pole_rel_error)
    assert unrefined / fit.relative_error_max >= 10.0


def test_pole_refinement_round_two_numbers(hulme):
    """Pinned round-2 numbers (order 12), so the report cannot drift silently."""
    omega, damping = hulme
    fit, report = sk_fit_damping_report(omega, damping, n_poles=12, max_iter=200, tol=1e-12)
    assert min(report.sk_pole_rel_error, report.grid_pole_rel_error) > 1e-1
    assert fit.relative_error_max < 1e-3
    assert report.refine_evaluations > 0
    assert report.pole_source == "refined"


def test_refinement_preserves_the_exact_recovery(representable):
    """Pole refinement must not disturb a curve the basis already represents."""
    omega, damping, truth = representable
    fit, report = sk_fit_damping_report(omega, damping, n_poles=3, max_iter=200, tol=1e-12)
    np.testing.assert_allclose(fit.poles, np.sort(truth["decay_rates"]), rtol=1e-9)
    np.testing.assert_allclose(fit.residues, np.asarray(truth["residues"]), atol=1e-8)
    assert fit.constant == pytest.approx(truth["asymptote"], abs=1e-10)
    assert fit.relative_error_max < 1e-12


def test_refine_budget_zero_reproduces_round_one(hulme):
    """``refine_maxfev=0`` is the round-1 estimator, bit for bit."""
    omega, damping = hulme
    off, report_off = sk_fit_damping_report(
        omega, damping, n_poles=8, max_iter=200, tol=1e-12, refine_maxfev=0
    )
    assert report_off.pole_source in {"sk", "grid"}
    assert report_off.refine_evaluations == 0
    assert report_off.refine_start == "none"
    assert math.isnan(report_off.refined_pole_rel_error)
    baseline = fixed_grid_damping_fit(omega, damping, n_poles=8)
    assert off.relative_error_max == pytest.approx(
        min(report_off.sk_pole_rel_error, baseline.relative_error_max), rel=1e-12
    )


def test_refinement_is_deterministic(hulme):
    """No RNG anywhere: two calls give the same poles bit for bit."""
    omega, damping = hulme
    first, report_first = sk_fit_damping_report(
        omega, damping, n_poles=6, max_iter=200, tol=1e-12
    )
    second, report_second = sk_fit_damping_report(
        omega, damping, n_poles=6, max_iter=200, tol=1e-12
    )
    assert np.array_equal(first.poles, second.poles)
    assert np.array_equal(first.residues, second.residues)
    assert report_first.refine_evaluations == report_second.refine_evaluations
    assert report_first.refine_probe_gain == report_second.refine_probe_gain


def test_refinement_convergence_is_measured_not_asserted(hulme):
    """``converged`` on a refined fit reports the one-pole probe, not the simplex flag."""
    omega, damping = hulme
    fit, report = sk_fit_damping_report(omega, damping, n_poles=8, max_iter=200, tol=1e-12)
    assert report.pole_source == "refined"
    assert report.refine_probe_gain >= 0.0
    assert report.refine_converged == (report.refine_probe_gain <= 1e-3)
    assert fit.converged == report.refine_converged


def test_validation_rejects_impossible_inputs():
    omega = np.logspace(-1, 1, 20)
    damping = np.ones_like(omega)
    with pytest.raises(ValueError):
        sk_fit_damping(omega[:5], damping[:5], n_poles=2)  # too few points
    with pytest.raises(ValueError):
        sk_fit_damping(np.r_[-1.0, omega[1:]], damping, n_poles=2)  # non-positive omega
    with pytest.raises(ValueError):
        sk_fit_damping(omega, -damping, n_poles=2)  # negative damping
    with pytest.raises(ValueError):
        sk_fit_damping(omega, damping, n_poles=0)
    with pytest.raises(ValueError):
        sk_fit_damping(omega, damping, n_poles=30)  # 61 unknowns > 20 points


def test_published_table_matches_a_fresh_run(study_results):
    """The document's table is regenerated from the study, not transcribed."""
    if not DOC.exists():
        pytest.skip("report not present in this checkout")
    text = DOC.read_text()
    begin, end = "<!-- sk-fit-table:begin -->", "<!-- sk-fit-table:end -->"
    assert begin in text and end in text
    published = text.split(begin)[1].split(end)[0].strip()
    regenerated = study.render_markdown(study_results).strip()
    assert published == regenerated


def test_artifact_matches_the_study(study_results):
    artifact = REPO_ROOT / "docs" / "sk_fit_study" / "results.json"
    if not artifact.exists():
        pytest.skip("artifact not present in this checkout")
    stored = json.loads(artifact.read_text())
    fresh = study_results
    assert stored["mission"] == fresh["mission"]
    assert stored["sk_refine_maxfev"] == fresh["sk_refine_maxfev"]
    for name, case in fresh["cases"].items():
        assert stored["cases"][name]["regression_free"] == case["regression_free"]
        assert (
            stored["cases"][name]["never_worse_than_unrefined"]
            == case["never_worse_than_unrefined"]
        )
        assert (
            stored["cases"][name]["orders_meeting_accuracy_bar"]
            == case["orders_meeting_accuracy_bar"]
        )
        for stored_row, fresh_row in zip(stored["cases"][name]["rows"], case["rows"]):
            for key in (
                "order",
                "pole_source",
                "refine_start",
                "refine_evaluations",
                "meets_accuracy_bar",
                "refine_converged",
            ):
                assert stored_row[key] == fresh_row[key]
            for key in (
                "returned_rel_error",
                "unrefined_rel_error",
                "refined_rel_error",
                "baseline_rel_error",
            ):
                assert stored_row[key] == pytest.approx(
                    fresh_row[key], rel=1e-9, abs=1e-15
                )
            assert stored_row["pole_source"] == fresh_row["pole_source"]