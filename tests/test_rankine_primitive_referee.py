"""Guards the closed-form Rankine panel integral and its referee's resolution limit.

History: ``_evaluate_source_potential`` integrates the ``log|P-Q|`` part of the
single-layer potential with a closed-form primitive.  The guard for that change
compared it against a *composite uniform Gauss rule* refined until "converged",
with a ``1e-8`` tolerance.  On a coplanar panel the composite rule converges only
like ``O(1/N)`` (``7.5e-6`` at 400 pieces), so the test could never pass -- the
referee was the inexact side, not the primitive.

These tests replace the unattainable criterion with two checkable ones:

* the primitive is certified against an arbitrary-precision referee
  (``mpmath.quad`` at 40 digits, singular abscissa passed as a breakpoint) and
  against a numpy-only referee, so the certificate survives a numpy-only
  install;
* the retired criterion is *quantified*: the observed order of the uniform
  referee, its error at 400/1600 pieces and the single-panel straddling bias are
  pinned, so "why a uniform rule cannot be the reference" is an executable fact.

``docs/m2966/outputs/m2966/rankine_primitive_check.json`` holds the same
measurements; ``python -m tensorbem.benchmarks.rankine_primitive_referee
--check`` re-derives it, and :func:`test_committed_artifact_reproduces` keeps the
document and the code from drifting apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import rankine_primitive_referee as referee

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "docs" / "m2966" / "outputs" / "m2966" / "rankine_primitive_check.json"
DOC = REPO_ROOT / "docs" / "rankine_primitive_referee_study.md"


@pytest.fixture(scope="module")
def payload() -> dict:
    return referee.measure()


def test_cases_cover_both_sides_of_the_coplanar_switch():
    """``_log_panel_primitive`` switches branch at ``h <= 1e-14``."""
    heights = {h for _, _, h, _ in referee.CASES}
    assert any(h == 0.0 for h in heights), "no coplanar case"
    assert any(0.0 < h < 1.0e-14 for h in heights), "no case inside the switch band"
    assert any(h > 1.0e-2 for h in heights), "no ordinary smooth case"


def test_closed_form_matches_arbitrary_precision_referee(payload):
    """The primitive is exact to machine precision against a 40-digit referee."""
    pytest.importorskip(
        "mpmath", reason="needs an arbitrary-precision referee for the closed form"
    )
    assert payload["mpmath_available"]
    for row in payload["cases"]:
        reference = row["mpmath_referee"]
        assert reference is not None, row["label"]
        assert math.isclose(
            row["closed_form"], reference, rel_tol=1.0e-13, abs_tol=1.0e-13
        ), (row["label"], row["closed_form"], reference)


def test_closed_form_matches_the_numpy_only_referee(payload):
    """Same certificate without mpmath: elementary / graded quadrature referee."""
    for row in payload["cases"]:
        assert row["abs_error_numpy_referee"] < 1.0e-12, (row["label"], row)


def test_coplanar_branch_reproduces_the_elementary_integral():
    """``int_0^L log|s-xi| ds = G(L-xi) - G(-xi)`` with ``G(t)=t log|t|-t``.

    Independent of the branch under test: this is the elementary antiderivative
    evaluated at the panel ends, valid for the singularity anywhere on or off the
    panel, including the end points where the code takes its ``coplanar`` branch.
    """
    for xi in (-0.3, 0.0, 0.25, 0.5, 1.0, 1.4):
        closed = referee.panel_integral(xi, 0.0, 1.0)
        elementary = referee.elementary_coplanar(xi, 1.0)
        assert closed == pytest.approx(elementary, rel=0.0, abs=1.0e-15), xi


def test_uniform_gauss_referee_has_a_first_order_resolution_limit(payload):
    """Pin the retired ``1e-8`` criterion's referee: ``O(1/N)``, never converged."""
    assert 0.8 < payload["observed_order_edge"] < 1.2, payload["observed_order_edge"]
    assert 0.8 < payload["observed_order_inside"] < 1.2, payload["observed_order_inside"]

    coplanar = [
        row for row in payload["convergence"] if row["xi"] == 0.5 and row["h"] == 0.0
    ]
    errors = {row["pieces"]: row["err_sing_on_edge"] for row in coplanar}
    assert sorted(errors) == list(referee.PIECES)
    # Halving the error needs doubling the piece count, so 400 pieces is still
    # 1e-6 away and even 1600 pieces cannot reach the retired 1e-8 threshold.
    assert errors[400] > 1.0e-6, errors[400]
    assert errors[1600] > 1.0e-7, errors[1600]
    # Monotone decay: the alignment does not matter at this order.
    ordered = [errors[n] for n in referee.PIECES]
    assert all(later < earlier for earlier, later in zip(ordered, ordered[1:])), ordered
    # A single uniform rule straddling the self-panel singularity (the scheme the
    # closed form replaced) is two orders of magnitude worse than the primitive.
    assert payload["single_panel_bias_24pt"] > 1.0e-4, payload["single_panel_bias_24pt"]
    assert payload["single_panel_bias_20pt"] > 1.0e-4


def test_smooth_panels_converge_to_machine_precision(payload):
    """The ``O(1/N)`` limit is a coplanar-only effect, not a property of the rule."""
    smooth = [row for row in payload["convergence"] if row["label"].startswith("smooth")]
    assert smooth, "no smooth case in the convergence study"
    assert max(row["err_sing_on_edge"] for row in smooth) < 1.0e-10


def test_committed_artifact_reproduces(payload):
    """The evidence JSON must be what this code measures, not a transcript."""
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert referee.compare(payload, artifact) == []


def test_study_note_quotes_the_measured_numbers(payload):
    """The note is rendered from the payload; a stale note is a red build."""
    text = DOC.read_text(encoding="utf-8")
    assert f"{payload['observed_order_edge']:.2f}" in text
    assert f"{payload['observed_order_inside']:.2f}" in text
    assert referee._sci(payload["max_abs_error_mpmath"]) in text
    assert referee._sci(payload["single_panel_bias_24pt"]) in text
    assert ARTIFACT.name in text


def test_check_mode_accepts_a_fresh_artifact_and_rejects_a_tampered_one(tmp_path, payload):
    """``--check`` is the CI/CLI form of the artifact comparison."""
    target = tmp_path / "evidence.json"
    referee.write_artifacts(target, None, None, payload=payload)
    assert referee.main(["--check", "--json", str(target)]) == 0

    tampered = json.loads(target.read_text(encoding="utf-8"))
    tampered["cases"][0]["closed_form"] += 1.0e-6
    target.write_text(json.dumps(tampered), encoding="utf-8")
    assert referee.main(["--check", "--json", str(target)]) == 1


def test_figure_helper_is_importable_and_optional(tmp_path, payload):
    """The figure is an optional extra: matplotlib absence must not be fatal."""
    pytest.importorskip("matplotlib")
    path = referee.make_figure(payload, tmp_path / "figure.png")
    assert path is not None and path.stat().st_size > 0


def test_numpy_referee_does_not_use_the_primitive_under_test():
    """Guard against a tautological referee (a graded rule built on the closed form)."""
    source = Path(referee.__file__).read_text(encoding="utf-8")
    body = source.split("def graded_gauss(")[1].split("\ndef ")[0]
    assert "_log_panel_primitive" not in body
    assert np.isfinite(referee.graded_gauss(0.5, 1.0e-3, 1.0))