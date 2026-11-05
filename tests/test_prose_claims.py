"""Tests for the prose-claim ledger of ``docs/three_dimensional_bem.md``.

The ledger exists to make the *quoted* numbers executable, so the tests have two
jobs: prove the ledger passes on the committed document, and prove it **bites**
when the document is edited.  The second half is done by mutating the document
text in memory (never on disk): a criterion that cannot fail is not a criterion,
and this repository has been bitten three times by numbers that no build checked.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tensorbem.benchmarks import prose_claims as ledger
from tensorbem.benchmarks.three_dimensional_doc_pins import DOC, REPO_ROOT


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text()


@pytest.fixture(scope="module")
def report(text):
    return ledger.check(text=text)


# ---------------------------------------------------------------------------
# The ledger passes on the committed document
# ---------------------------------------------------------------------------


def test_every_anchor_occurs_verbatim_in_the_document(text):
    missing = [claim.id for claim in ledger.PROSE_CLAIMS if claim.anchor not in text]
    assert not missing, f"anchors no longer present: {missing}"


def test_claim_ids_are_unique():
    ids = [claim.id for claim in ledger.PROSE_CLAIMS]
    assert len(ids) == len(set(ids))


def test_ledger_passes_on_the_committed_document(report):
    assert report["verdict"] == "PASS", json.dumps(
        [item for item in report["findings"] if item["status"] != "PASS"], ensure_ascii=False
    )
    assert report["drift"] == 0
    assert report["claims"] == len(ledger.PROSE_CLAIMS)


def test_each_restatement_is_judged_against_its_own_measurement(report):
    # Eight sentences, seven distinct printed numbers; every cell must have a
    # measured value, i.e. nothing is silently skipped.
    for finding in report["findings"]:
        assert finding["cells"], finding["id"]
        for cell in finding["cells"]:
            assert "measured" in cell, (finding["id"], cell)
            assert cell["relative_deviation"] <= cell["tolerance"]


def test_restatements_of_the_same_fact_agree_with_each_other(report):
    # The classic drift is "the table got regenerated, four sentences did not".
    # Where two anchors quote the same group, their printed tokens must match.
    by_group: dict[str, set[tuple[str, ...]]] = {}
    for finding in report["findings"]:
        by_group.setdefault(str(finding["group"]), set()).add(tuple(finding["tokens"]))
    band_groups = {group: groups for group, groups in by_group.items() if "band" in group}
    assert band_groups, by_group
    for group, variants in band_groups.items():
        assert len(variants) == 1, f"{group} is quoted inconsistently: {variants}"


def test_qualitative_claims_are_measured_not_asserted(report):
    by_id = {item["id"]: item for item in report["qualitative"]}
    assert set(by_id) == {name for name, _, _, _ in ledger.QUALITATIVE_CLAIMS}
    assert all(item["status"] == "PASS" for item in by_id.values())
    # The representation path really does stall: both cases stay far below first
    # order, which is what "does not decrease under refinement" means numerically.
    orders = ledger.representation_orders()
    assert len(orders) == 4
    assert max(orders) < 0.5


# ---------------------------------------------------------------------------
# The measurements are recomputed, not transcribed
# ---------------------------------------------------------------------------


def test_band_is_the_span_of_the_quoted_refinement_levels():
    low, high = ledger.representation_band()
    rows = {row["panels"]: row for row in ledger.representation_ladder()}
    assert set(rows) == {20, 80, 320, 1280}
    assert low == pytest.approx(100.0 * rows[1280]["representation_relative_l2_error"])
    assert high == pytest.approx(100.0 * rows[80]["representation_relative_l2_error"])
    # The 20-panel level (81%) is printed in the table but never quoted; if the
    # band ever swallowed it, the claim would become false.
    assert low > 50.0
    assert 100.0 * rows[20]["representation_relative_l2_error"] > high


def test_near_body_layer_probe_is_a_live_solve():
    layer = ledger.near_body_layer()
    assert layer["panels"] == 80
    # Exact interior velocity of the harmonic extension of u = x is (1, 0, 0).
    assert layer["exact_velocity"] == 1.0
    # Inside the O(h) layer: tens of percent wrong.  At the centre: percent level.
    assert 40.0 < layer["layered_percent_error"] < 60.0
    assert 0.0 < layer["centre_percent_error"] < 5.0
    assert layer["layered_velocity"] < layer["centre_velocity"] < 1.0


def test_layer_probe_is_deterministic():
    first = ledger.near_body_layer()
    second = ledger.near_body_layer()
    assert first == second


def test_recovered_path_converges_while_representation_stalls():
    rows = ledger.representation_ladder()
    recovered = [row["recovered_relative_l2_error"] for row in rows]
    represented = [row["representation_relative_l2_error"] for row in rows]
    assert ledger.recovered_error_decreases()
    assert recovered[-1] < 0.02
    assert represented[-1] > 0.5
    # The whole point of the diagnosis: refinement does not rescue the
    # representation path, it rescues the recovered one.
    assert represented[0] / represented[-1] < 2.0
    assert recovered[0] / recovered[-1] > 10.0


# ---------------------------------------------------------------------------
# Negative controls: an edited document must trip the ledger
# ---------------------------------------------------------------------------


def test_edited_headline_number_is_caught(text):
    mutated = text.replace("53%–65%", "43%–65%")
    assert mutated != text
    verdict = ledger.check(text=mutated)
    assert verdict["verdict"] == "DRIFT"
    flagged = {item["id"] for item in verdict["findings"] if item["status"] != "PASS"}
    # Every band restatement must notice, not just the first one.
    assert flagged == {"abstract-band", "s55-band", "s64-band", "s7-band", "s9-band", "s11-index"}


def test_edited_layer_number_is_caught(text):
    # The anchors carry their numbers verbatim, so editing one is reported as a
    # lost anchor rather than as a per-cell value drift.  What matters is that it
    # is caught, and caught at the right claim.
    mutated = text.replace("速度为 0.518 而正确值为 1.000", "速度为 0.418 而正确值为 1.000")
    assert mutated != text
    verdict = ledger.check(text=mutated)
    finding = next(item for item in verdict["findings"] if item["id"] == "s55-layer")
    assert finding["status"] == "DRIFT"
    assert "anchor missing" in finding["reason"]
    untouched = next(item for item in verdict["findings"] if item["id"] == "s55-centre")
    assert untouched["status"] == "PASS"


def test_registry_tokens_are_standalone_numbers_in_their_own_anchor():
    # Guards the ``token not standalone`` branch: registering "1.6" against an
    # anchor that prints "1.65" would silently judge the wrong number.
    for claim in ledger.PROSE_CLAIMS:
        for token in claim.tokens:
            assert ledger._token_in(token, claim.anchor), (claim.id, token)


def test_reworded_sentence_is_caught(text):
    # Dropping one character from the anchor must not pass silently: the sentence
    # is the audit trail, so a reworded claim is an untracked claim.
    mutated = text.replace("在体心处则为 0.984（1.6%）", "在体心处为 0.984（1.6%）")
    assert mutated != text
    verdict = ledger.check(text=mutated)
    finding = next(item for item in verdict["findings"] if item["id"] == "s55-centre")
    assert finding["status"] == "DRIFT"
    assert "anchor missing" in finding["reason"]


def test_deleted_sentence_is_caught(text):
    mutated = text.replace("紧贴表面误差 48%，体心处 1.6%", "")
    assert mutated != text
    verdict = ledger.check(text=mutated)
    flagged = {item["id"] for item in verdict["findings"] if item["status"] != "PASS"}
    assert {"s7-layer", "s9-layer"} <= flagged


def test_qualitative_claim_is_caught_when_reworded(text):
    mutated = text.replace("恢复路径没有这一缺陷", "恢复路径同样有缺陷")
    verdict = ledger.check(text=mutated)
    item = next(q for q in verdict["qualitative"] if q["id"] == "recovered-path-has-no-defect")
    assert item["status"] == "DRIFT"


def test_partial_update_is_caught_even_when_each_number_reproduces(text):
    # Update the abstract only: the other seven restatements now disagree with it.
    mutated = text.replace("重构误差达 53%–65%", "重构误差达 53%–64%")
    assert mutated != text
    verdict = ledger.check(text=mutated)
    assert verdict["verdict"] == "DRIFT"
    flagged = {item["id"] for item in verdict["findings"] if item["status"] != "PASS"}
    assert flagged == {"abstract-band"}


# ---------------------------------------------------------------------------
# Coverage, rendering, CLI
# ---------------------------------------------------------------------------


def test_coverage_counts_prose_that_was_unguarded(text):
    payload = ledger.coverage(text=text)
    assert payload["bound_by_prose_claims"] == len(payload["bound_tokens"])
    assert payload["bound_tokens"] == ["53", "65", "0.518", "1.000", "48", "0.984", "1.6"]
    assert payload["guarded_total"] == payload["protected_in_tables"] + 7
    assert payload["unguarded_after"] == payload["unguarded_before"] - 7
    assert 0.0 < payload["guarded_fraction"] <= 1.0


def test_rendered_section_is_stable_and_all_pass():
    first = ledger.render_section()
    assert first == ledger.render_section()
    assert "DRIFT" not in first
    assert first.count("\n") > len(ledger.PROSE_CLAIMS)


def test_cli_exits_zero_on_the_committed_document():
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.prose_claims", "--check", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "OPENBLAS_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout)
    assert payload["verdict"] == "PASS"


def test_cli_coverage_mode_runs():
    assert ledger.main(["--coverage", "--json"]) == 0