"""Pins for :mod:`tensorbem.sk_fit` (Sanathanan-Koerner damping fit).

The suite answers three questions with numbers, not with a smoke run:

1. does the iteration solve an inverse problem it can solve at all -- a curve
   that *is* a sum of three real decay rates must come back with those three
   decay rates, the three residues and the asymptote;
2. is that an improvement -- the pre-rewrite fixed-grid fit
   (:func:`tensorbem.sk_fit.fixed_grid_damping_fit`) must be measurably worse on
   the same order, and the SK entry point must never be worse than it anywhere
   in the order scan (the documented "no regression" invariant);
3. does the caller get the truth about convergence -- ``max_iter`` and ``tol``
   must actually change ``n_iter``/``converged``, which the pre-rewrite code
   silently ignored.
"""

from __future__ import annotations

import json
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
    """The returned fit is the better of the two candidate pole sets, always."""
    for omega, damping in (representable[:2], hulme):
        fit, report = sk_fit_damping_report(omega, damping, n_poles=order, max_iter=120)
        baseline = fixed_grid_damping_fit(omega, damping, n_poles=order)
        assert fit.relative_error_max <= baseline.relative_error_max * (1.0 + 1e-12)
        assert min(report.sk_pole_rel_error, report.grid_pole_rel_error) == pytest.approx(
            fit.relative_error_max, rel=1e-12, abs=1e-15
        )
        assert report.pole_source in {"sk", "grid"}
        assert report.real_root_count + 2 * report.complex_pair_count == order


def test_max_iter_and_tol_are_not_decorative(representable):
    """``max_iter``/``tol`` were accepted and ignored before this rewrite."""
    omega, damping, _ = representable
    one, report_one = sk_fit_damping_report(omega, damping, n_poles=3, max_iter=1)
    many, report_many = sk_fit_damping_report(omega, damping, n_poles=3, max_iter=200, tol=1e-10)

    assert report_one.sk_iterations == 1
    assert one.n_iter == 1
    assert not report_one.sk_converged
    assert report_many.sk_iterations > 1
    assert report_many.sk_converged
    # more iterations never leave the iterate worse than the one-shot fit
    assert many.relative_error_max <= one.relative_error_max * (1.0 + 1e-12)
    # the residual history has one entry per iteration
    assert len(report_many.residual_history) == report_many.sk_iterations


def test_measured_curve_shows_the_real_pole_model_class_floor(hulme):
    """Honest limit: the rational ceiling is far below any real-pole fit here."""
    omega, damping = hulme
    _, report = sk_fit_damping_report(omega, damping, n_poles=12, max_iter=200, tol=1e-12)
    assert report.rational_rel_error < report.sk_pole_rel_error
    assert report.rational_rel_error < 1e-4
    assert report.sk_pole_rel_error > 1e-2
    # the measured curve does not admit a real-decay-rate representation at
    # this order: at least one conjugate pair has to be projected
    assert report.complex_pair_count >= 1


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


def test_published_table_matches_a_fresh_run():
    """The document's table is regenerated from the study, not transcribed."""
    if not DOC.exists():
        pytest.skip("report not present in this checkout")
    text = DOC.read_text()
    begin, end = "<!-- sk-fit-table:begin -->", "<!-- sk-fit-table:end -->"
    assert begin in text and end in text
    published = text.split(begin)[1].split(end)[0].strip()
    regenerated = study.render_markdown(study.run_study()).strip()
    assert published == regenerated


def test_artifact_matches_the_study():
    artifact = REPO_ROOT / "docs" / "sk_fit_study" / "results.json"
    if not artifact.exists():
        pytest.skip("artifact not present in this checkout")
    stored = json.loads(artifact.read_text())
    fresh = study.run_study()
    for name, case in fresh["cases"].items():
        assert stored["cases"][name]["regression_free"] == case["regression_free"]
        for stored_row, fresh_row in zip(stored["cases"][name]["rows"], case["rows"]):
            assert stored_row["order"] == fresh_row["order"]
            assert stored_row["returned_rel_error"] == pytest.approx(
                fresh_row["returned_rel_error"], rel=1e-9, abs=1e-15
            )
            assert stored_row["pole_source"] == fresh_row["pole_source"]