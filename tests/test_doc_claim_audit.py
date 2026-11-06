"""Gates for the doc-claim coverage audit (:mod:`tensorbem.benchmarks.doc_claim_audit`).

The audit answers the inverse of every existing gate: *which documents quote
numbers that no test reads?*  These tests keep the answer honest and, above
all, keep the answer from silently getting worse:

* the classifier is exercised on synthetic fixtures (pinned / artifact /
  unguarded) so a refactor cannot turn every document green;
* the number-token counter is checked against known strings;
* the ratchet fails when a document *loses* its guard, and passes when the
  unguarded set shrinks (improvement is always allowed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.benchmarks import doc_claim_audit as audit

BASELINE = Path(audit.DOCS) / "doc_claim_audit" / "baseline.json"


@pytest.fixture(scope="module")
def report():
    return audit.audit()


def test_report_shape(report):
    counts = report["counts"]
    assert counts["documents"] >= 30
    assert counts["pinned"] + counts["artifact"] + counts["unguarded"] == counts["documents"]
    assert 0.0 <= report["coverage"]["guarded_number_fraction"] <= 1.0
    tiers = {r["tier"] for r in report["records"]}
    assert tiers <= {"PINNED", "ARTIFACT", "UNGUARDED"}


def test_number_counter():
    text = "values 8.51e-3, -0.87202 and 2566.83; not v1.0"
    # the regex counts decimal tokens; 'v1.0' is preceded by a letter and must be
    # excluded by the lookbehind
    assert len(audit.NUM_RE.findall(text)) == 3


def test_classifier_pinned_fixture(tmp_path, monkeypatch):
    """A document whose name appears in tests/ must be classified PINNED."""
    docs = tmp_path / "docs"
    docs.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    doc = docs / "sample_note.md"
    doc.write_text("error 1.5e-3 at 128 panels\n", encoding="utf-8")
    (tests / "test_sample.py").write_text(
        'ROOT / "docs" / "sample_note.md"\n', encoding="utf-8"
    )
    record = audit.classify(doc, {"test_sample.py": (tests / "test_sample.py").read_text()})
    assert record.tier == "PINNED"
    assert record.number_tokens == 1
    assert record.reading_tests == ("test_sample.py",)


def test_classifier_unguarded_fixture(tmp_path):
    """A generator no test ever runs stays UNGUARDED, and says why."""
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "orphan_note.md"
    doc.write_text(
        "Run `python -m tensorbem.benchmarks.orphan`; error 2.5e-4.\n", encoding="utf-8"
    )
    record = audit.classify(doc, {})
    assert record.tier == "UNGUARDED"
    assert record.generators == ("tensorbem.benchmarks.orphan",)
    assert "never exercised" in record.note


def test_classifier_artifact_fixture(tmp_path):
    """Generator exercised by a test but the document text unread -> ARTIFACT."""
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "artifact_note.md"
    doc.write_text(
        "Run `python -m tensorbem.benchmarks.orphan`; error 2.5e-4.\n", encoding="utf-8"
    )
    tests = {"test_orphan.py": "from tensorbem.benchmarks.orphan import run\n"}
    record = audit.classify(doc, tests)
    assert record.tier == "ARTIFACT"
    assert record.modules_exercised == ("test_orphan.py",)


def test_historical_documents_are_separated():
    assert audit.is_historical("dev_report_tensorbem_declared_numbers.md")
    assert audit.is_historical("project_review_2026-10-22.md")
    assert not audit.is_historical("finite_depth3d.md")
    assert not audit.is_historical("marine_bem_benchmarks.md")

def test_mission_prefixed_reports_are_historical():
    """A report whose mission id leads the name is a dated work product.

    Mission ids used to appear as a suffix (``_m####``) or behind a prefix
    (``dev_report_``, ``report_m``); newer reports put the id first
    (``m####_ci_fix_report.md``).  The spelling must not decide the tier: a
    snapshot of one run is historical, so it cannot be counted as a paper-level
    contract that lost its guard.  The synthetic names below keep this test from
    mentioning a real document, because a bare file name in ``tests/`` pins that
    document: ``classify`` treats a bare file name in ``tests/`` as a reading
    test (see :func:`test_classifier_pinned_fixture`).
    """
    assert audit.is_historical("m9999_ci_fix_report.md")
    assert audit.is_historical("m9999_pr1_merge_triage.md")
    assert audit.MISSION_RE.search("m9999_ci_fix_report.md")
    # A name that only looks like a mission id (no separator after the digits) is not one.
    assert not audit.is_historical("mid3_report.md")


def test_paper_level_exemptions_survive_the_mission_prefix_rule():
    """Every exempted work-product name keeps the paper-level gate."""
    for name in audit.PAPER_LEVEL_EXTRA:
        assert audit.MISSION_RE.search(name), name
        assert not audit.is_historical(name), name


def test_historical_documents_are_still_registered_in_the_baseline(report):
    """Negative control: freezing a report does not exempt it from registration.

    The mission-prefix rule only moves documents out of the paper-level half of
    the ratchet.  The ``UNGUARDED`` half must still see them, so every frozen
    unguarded document has to be registered in the baseline before the gate is
    green.
    """
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    registered = set(baseline["unguarded"])
    frozen = {
        r["doc"]
        for r in report["records"]
        if r["tier"] == "UNGUARDED" and audit.is_historical(r["doc"])
    }
    assert frozen, "expected at least one unguarded historical document"
    assert frozen - registered == set()


def test_ratchet_against_baseline(report):
    """No document may lose its guard; the recorded baseline is the floor."""
    assert BASELINE.exists(), (
        "docs/doc_claim_audit/baseline.json is missing; regenerate with "
        "python -m tensorbem.benchmarks.doc_claim_audit --write-baseline"
    )
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert audit.check(report, baseline) == []


def test_ratchet_detects_regression(report):
    """Negative control: a document dropping out of the guard set is a failure."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    artificial = dict(baseline)
    artificial["unguarded"] = list(baseline.get("unguarded", []))
    artificial["unguarded_paper_level"] = list(baseline.get("unguarded_paper_level", []))
    victim = next(
        (r["doc"] for r in report["records"] if r["tier"] == "UNGUARDED"),
        None,
    )
    assert victim is not None, "no unguarded document left to test the ratchet with"
    artificial["unguarded"] = [doc for doc in artificial["unguarded"] if doc != victim]
    failures = audit.check(report, artificial)
    assert any(victim in f for f in failures), failures


def test_renderers_are_non_empty(report):
    assert "coverage" in audit.render_markdown(report).lower()
    assert "UNGUARDED" in audit.render_table(report, limit=5)