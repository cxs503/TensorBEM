"""Pin ``docs/audit_xval_r2_degeneracy.md`` to the archived results.

Failure mode this test exists for
---------------------------------
The audit document is the only prose that states *what the rotational-block
verdict means* for the anchor ``XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE``.  It was
hand-written for the first scoring round (m20476, 256/270 panels) and stayed
that way after m22034 re-ran the same four bodies on 1024/1026 panels and
replaced ``results.json``; the document therefore kept reporting a worst
deviation of 87.93 % and 16 exact-zero bound violations for an archive whose
actual numbers are 6.71 % and 0.  Nothing failed, because no test read the
document.

The document is now a rendered derivative of the archive, and this module
requires

* byte equality between the checked-in document and the render, and
* that the numbers quoted in the document are the numbers in ``results.json``
  (spot-checked field by field, in case the renderer ever starts carrying its
  own literals).

The superseded round is *not* deleted: it lives in ``degeneracy_round_m20476``
.json, and the tests below require that it still describes the coarser mesh
and the worse verdict.  History that cannot be told apart from the current
round would be indistinguishable from hand-copied numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2
from tensorbem.benchmarks import xval_r2_degeneracy_audit as audit

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / r2.ARTIFACT_ROOT / r2.RESULT_FILE
HISTORY = ROOT / audit.ARTIFACT_ROOT / audit.HISTORY_FILE
DOC = ROOT / audit.DOC_PATH


@pytest.fixture(scope="module")
def study() -> dict:
    assert RESULT.exists(), f"archive missing: {RESULT}"
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def history() -> dict:
    assert HISTORY.exists(), f"superseded-round summary missing: {HISTORY}"
    return json.loads(HISTORY.read_text())


@pytest.fixture(scope="module")
def current(study: dict) -> dict:
    return audit.summarise(study)


def test_document_is_the_render(current: dict, history: dict) -> None:
    assert DOC.exists(), f"audit document missing: {DOC}"
    assert DOC.read_text() == audit.render_document(current, history)


def test_render_is_deterministic(current: dict, history: dict) -> None:
    assert audit.render_document(current, history) == audit.render_document(
        current, history
    )


def test_quoted_percentages_are_the_archived_ones(current: dict) -> None:
    text = DOC.read_text()
    for name, case in current["cases"].items():
        quoted = f"{case['worst_physics_relative']:.2%}"
        assert quoted in text, f"{name}: {quoted} not quoted in the audit document"
    worst = max(c["worst_physics_relative"] for c in current["cases"].values())
    assert f"{worst:.2%}" in text
    assert f"{current['centre_transfer']['worst_residual']:.1e}" in text


def test_verdicts_match_the_archive(current: dict, study: dict) -> None:
    text = DOC.read_text()
    assert current["comparison"]["verdict"] == study["comparison"]["verdict"]
    for name, case in current["cases"].items():
        assert (
            case["physics_verdict"]
            == study["invariant_scoring"]["cases"][name]["physics_verdict"]
        )
        assert (
            case["invariant_verdict"]
            == study["invariant_scoring"]["cases"][name]["invariant_verdict"]
        )
    # the archive's own two layers must both appear, or a reader cannot tell
    # "internal band FAIL" from "acceptance FAIL"
    assert "内部" in text and "验收线" in text


def test_invariant_violations_are_reported_from_data(current: dict) -> None:
    total = sum(c["counts"]["invariant_violated"] for c in current["cases"].values())
    text = DOC.read_text()
    consistent = sum(
        c["counts"]["invariant_consistent"] for c in current["cases"].values()
    )
    assert f"越界数降为 {total}" in text
    assert f"全部 {consistent} 个精确零项一致" in text


def test_acceptance_layer_is_evaluated_not_asserted(current: dict) -> None:
    worst = max(c["worst_physics_relative"] for c in current["cases"].values())
    text = DOC.read_text()
    expected = "PASS" if worst < 0.15 else "FAIL"
    line = next(ln for ln in text.splitlines() if "验收线" in ln and "owner" in ln)
    assert f"| {expected} |" in line, line
    # the <3% question must be answered per case, not waved through
    below_three = [
        name
        for name, case in current["cases"].items()
        if case["worst_physics_relative"] < 0.03
    ]
    assert below_three == [], (
        "the archive now has cases under 3%: the document's honest-FAIL "
        "paragraph must be revisited before this test is relaxed"
    )


def test_history_round_is_the_coarser_mesh(history: dict, current: dict) -> None:
    assert history["round"] == "m20476"
    assert history["provenance"]["source_revision"].startswith("4123e585")
    assert set(history["cases"]) == set(
        current["cases"]
    ), "the superseded round must cover the same four bodies"
    for name, case in history["cases"].items():
        assert (
            case["tensorbem_faces"] < current["cases"][name]["tensorbem_faces"]
        ), f"{name}: the historical round must stay the coarser one"
    old_failures = sum(c["n_physics_failures"] for c in history["cases"].values())
    new_failures = sum(c["n_physics_failures"] for c in current["cases"].values())
    assert (
        old_failures > new_failures
    ), "the refinement is supposed to reduce the number of out-of-band entries"
    old_zero_violations = sum(
        c["counts"]["invariant_violated"] for c in history["cases"].values()
    )
    assert old_zero_violations > 0, (
        "the historical round's invariant violations are the reason the "
        "structural bound was introduced; if they are gone the narrative needs "
        "rewriting"
    )


def test_historical_numbers_carry_their_round_label(current: dict) -> None:
    """A stale number must never appear under the current-round label."""
    text = DOC.read_text()
    label = audit.ROUND_LABEL["current"]
    current_lines = [ln for ln in text.splitlines() if label in ln]
    assert current_lines, "the current round must be labelled in the tables"
    stale = {
        f"{case['worst_physics_relative']:.2%}" for case in current["cases"].values()
    }
    for line in current_lines:
        quoted = set(re.findall(r"\d+\.\d+%", line))
        assert quoted and quoted <= stale | {"15%"}, (line, quoted)


def test_render_rejects_an_incomplete_history(current: dict, history: dict) -> None:
    partial = json.loads(json.dumps(history))
    victim = next(iter(partial["cases"]))
    partial["cases"].pop(victim)
    with pytest.raises(KeyError):
        audit.render_document(current, partial)


def test_cli_check_mode_passes() -> None:
    assert audit.main(["--check", "--root", str(ROOT)]) == 0
