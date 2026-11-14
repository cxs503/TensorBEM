"""Contract of the case-pool verdict extractor (``registry._verdict_of``).

The registry is the machine-readable answer to "what did this anchor last
verdict?", so its extractor has to cope with every schema the pool actually
writes -- and it must never answer ``None`` (silence) or turn a recorded failure
into a pass.  These tests pin both halves of that contract on the four shapes in
use, plus the whole registered pool (a case whose artefact stops yielding a
verdict turns this file red).
"""

from __future__ import annotations

import pytest

from tensorbem.benchmarks import case_pool_registry as registry
from tensorbem.benchmarks.case_pool_registry import _verdict_of


def test_scalar_top_level_verdict():
    assert _verdict_of({"overall": "PASS", "rows": []}) == "PASS"


def test_nested_verdict_string():
    # Hulme hemisphere: {"verdict": {"criteria": {...}, "verdict": "PASS"}}
    payload = {"verdict": {"criteria": {"G1": {"pass": True}}, "verdict": "PASS"}}
    assert _verdict_of(payload) == "PASS"


def test_nested_verdict_block_without_scalar_still_reports_failure():
    # a nested block whose only signal is a per-item criterion list
    payload = {"verdict": {"criteria": {"G1": {"pass": True}, "G2": {"pass": False}}}}
    verdict = _verdict_of(payload)
    assert verdict["passed"] is False
    assert verdict["failed_checks"] == ["G2"]


def test_boolean_flag_inside_verdict():
    payload = {"verdict": {"passed": True, "tolerance": 0.03, "checks": []}}
    assert _verdict_of(payload) == {"passed": True, "failed_checks": []}


def test_boolean_flag_inside_verdict_records_failed_checks():
    payload = {"verdict": {"passed": False, "failed_checks": ["G3"]}}
    assert _verdict_of(payload) == {"passed": False, "failed_checks": ["G3"]}


def test_acceptance_block():
    payload = {"acceptance": {"passed": False, "failed_checks": ["entry_0_0"]}}
    assert _verdict_of(payload) == {"passed": False, "failed_checks": ["entry_0_0"]}


def test_criteria_block_without_booleans_is_not_fabricated():
    # a criteria block that only carries prose must not invent a verdict ...
    assert _verdict_of({"criteria": {"G1": "see section 4"}}) == "see artifact"
    # ... while one with a boolean does answer
    assert _verdict_of({"criteria": {"G1": True, "G2": False}})["passed"] is False


def test_tiers_schema():
    payload = {"tiers": {"case": {"verdict": "PASS"}, "economy": {"verdict": "FAIL"}}}
    assert _verdict_of(payload) == {"case": "PASS", "economy": "FAIL"}


def test_empty_payload_never_returns_none():
    assert _verdict_of({}) == "see artifact"


@pytest.mark.parametrize("case_code", [e.case_code for e in registry.list_cases()])
def test_every_registered_case_yields_a_verdict(case_code):
    """No registered anchor may answer ``None``: silence is a contract violation."""
    verdict = registry.artifact_verdict(case_code)
    assert verdict is not None
    assert verdict != ""
    assert verdict != "unreadable artifact"


def test_repair_producers_only_name_registered_cases():
    """``scripts/repair_case_pool_evidence.py`` may only repair real anchors."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "repair_case_pool_evidence.py"
    spec = importlib.util.spec_from_file_location("_repair_case_pool_evidence", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert set(module.PRODUCERS) <= {e.case_code for e in registry.list_cases()}