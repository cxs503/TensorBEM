"""The KK lower-band tail cannot explain the Cummins-Wigley causality residual.

Rounds 2-5 of ``CASE-BEM-CUMMINS-WIGLEY`` recorded 20-80 % Kramers-Kronig
residuals on the panel sweep and attributed them to band truncation, in
particular to the un-solved band below the lowest frequency.  That attribution
is testable on a case with an exact answer: an exponential memory kernel has a
closed-form ``(A, B)`` pair, so any residual of the closure is a statement about
the closure alone.

``scripts/cummins_wigley_kk_lowtail_bound.py`` builds such a pair constrained to
``sum c/a = 0`` (damping onset ``B ~ w^2``, as a 3-D radiator has), samples the
predeclared ``EXTENDED_GRID``, and compares

* the shipped closure (damping held constant over ``(0, s_min)``), and
* a closure whose sub-band below ``s_min`` is integrated *exactly* from a fitted
  ``B = C s^q`` tail,

against the analytic ``A(w)``.  The numbers below are read from the artifact of
that run; the test does not re-derive them and does not touch BEM.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ARTIFACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "docs"
    / "case_bem_cummins_wigley_lowtail_bound.json"
)


@pytest.fixture(scope="module")
def evidence() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"artifact not generated: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


def test_low_frequency_onset_of_the_synthetic_pair_is_quadratic(evidence):
    # the constraint sum c/a = 0 must give B ~ w^q with q ~ 2, i.e. the low
    # band of a 3-D radiator -- otherwise the synthetic pair is not the
    # situation the attribution claim is about
    assert evidence["kernel"]["sum_c_over_a"] == pytest.approx(0.0, abs=1e-10)
    exponent = evidence["low_frequency_fit"]["exponent_q"]
    assert 1.5 < exponent < 2.6, exponent


def test_exact_lower_tail_does_not_buy_the_recorded_residual(evidence):
    constant = evidence["closure_constant_tail"]["max_relative_residual"]
    exact = evidence["lower_tail_bound"]["max_relative_residual"]
    # the shipped closure is already far inside the 10 % case tolerance on an
    # exactly causal pair, so the lower band tail is not where 20-80 % can come
    # from: the two closures differ by well under one percentage point
    assert constant < 0.03, constant
    assert exact < 0.03, exact
    assert abs(exact - constant) < 0.01, (constant, exact)
    assert evidence["verdict"]["lower_tail_is_the_culprit"] is False