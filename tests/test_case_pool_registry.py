"""Guard for :mod:`tensorbem.benchmarks.case_pool_registry` and its CLI caller.

The registry is the machine answer to "which case-pool anchors can this
checkout run, with which entry, and what did they last verdict?".  A registry
that silently rots (moved entry, deleted artefact, unreadable JSON) is worse
than no registry, so every registered path must resolve and every registered
artefact must be readable and carry a verdict.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tensorbem.benchmarks import case_pool_registry as registry

ROOT = Path(__file__).resolve().parents[1]


def test_every_registered_case_resolves_its_paths():
    missing = registry.missing_paths()
    assert missing == {}, f"registry points at paths that do not exist: {missing}"


def test_normalise_accepts_anchor_and_pool_codes():
    assert registry.normalise("CASE-BEM-SUBMERGED-SPHERE-HAVELOCK") == \
        "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
    assert registry.normalise("BEM-SUBMERGED-SPHERE-HAVELOCK") == \
        "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
    assert registry.normalise("bem-submerged-sphere-havelock") == \
        "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
    with pytest.raises(KeyError):
        registry.normalise("CASE-BEM-NOT-A-CASE")


def test_entries_declare_a_runnable_surface():
    for entry in registry.list_cases():
        assert entry.entry.endswith(".py"), entry.case_code
        assert entry.path(entry.entry).is_file(), entry.case_code
        assert entry.tolerance.startswith("<"), entry.case_code
        assert entry.repo == "TensorBEM", entry.case_code
        assert entry.library.startswith("tensorbem."), entry.case_code


@pytest.mark.parametrize("case_code", [e.case_code for e in registry.list_cases()])
def test_artifact_is_readable_and_carries_a_verdict(case_code):
    entry = registry.resolve(case_code)
    if not entry.artifacts:
        pytest.skip(f"{case_code} declares no artefact")
    payload = entry.artifact_payload()
    assert isinstance(payload, dict) and payload
    verdict = registry.artifact_verdict(case_code)
    assert verdict not in (None, "")
    json.dumps(verdict)  # must be serialisable for the platform


def test_cli_lists_and_checks():
    listing = subprocess.run(
        [sys.executable, "scripts/case_pool_run.py", "--list"],
        cwd=ROOT, capture_output=True, text=True, check=False,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert listing.returncode == 0, listing.stderr
    assert "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK" in listing.stdout

    check = subprocess.run(
        [sys.executable, "scripts/case_pool_run.py", "--check", "BEM-SUBMERGED-SPHERE-HAVELOCK"],
        cwd=ROOT, capture_output=True, text=True, check=False,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert check.returncode == 0, check.stderr
    line = json.loads(check.stdout.strip().splitlines()[-1])
    assert line["case_code"] == "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
    assert line["status"] == "OK" and line["missing"] == []


def test_verdict_extraction_handles_known_schemas():
    assert registry._verdict_of({"overall": "PASS"}) == "PASS"
    assert registry._verdict_of({"verdict": {"overall": "FAIL", "x": 1}}) == "FAIL"
    assert registry._verdict_of({"acceptance": {"passed": True}})["passed"] is True
    tiers = {"tiers": {"case_mesh": {"verdict": "PASS"}, "economy_mesh": {"verdict": "FAIL"}}}
    assert registry._verdict_of(tiers) == {"case_mesh": "PASS", "economy_mesh": "FAIL"}
    assert registry._verdict_of({"something": 1}) == "see artifact"