"""Contract test for ``.github/workflows/tests.yml`` (m3229).

The 2026-11-04 CI failure was structural, not numerical: ``pytest`` was the
third step of the job, so the suite ran on a checkout that held *none* of the
benchmark artifacts the job produces later, and the install line omitted the
optional ``visualization`` extra that four plotting tests rely on.  Two of those
tests (``tests/test_ursell_error_budget.py`` and
``tests/test_ursell_band_audit.py``) call ``plot()`` without
``pytest.importorskip``, so they failed with ``ModuleNotFoundError: matplotlib``
while the same code passed locally.

Nobody noticed for a while because the failure looks like a test problem.  This
module makes the ordering itself a test: it parses the workflow file with plain
string scans (no YAML dependency, so it runs anywhere pytest runs) and asserts
that

* the install step pulls the ``visualization`` extra;
* every step that writes evidence under ``artifacts/`` precedes ``pytest``;
* ``pytest`` precedes the artifact upload, so the uploaded reports are the ones
  the suite was checked against.

If a future edit reorders the job again, this test says so in one line instead
of leaving ten unrelated-looking reds on ``main``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"


@pytest.fixture(scope="module")
def workflow_text() -> str:
    assert WORKFLOW.is_file(), f"missing workflow: {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


def _step_names(text: str) -> list[str]:
    """Names of the ``- name:`` entries, in file order."""
    return re.findall(r"^\s*-\s+name:\s*(.+?)\s*$", text, flags=re.MULTILINE)


def _index_of_step(text: str, name: str) -> int:
    match = re.search(rf"^\s*-\s+name:\s*{re.escape(name)}\s*$", text, flags=re.MULTILINE)
    assert match is not None, f"step {name!r} not found in {WORKFLOW.name}"
    return match.start()


def test_install_step_pulls_the_visualization_extra(workflow_text: str) -> None:
    """matplotlib is an optional extra; the job must ask for it explicitly."""
    installs = [
        line.strip()
        for line in workflow_text.splitlines()
        if "pip install" in line and "-e" in line
    ]
    assert installs, "the workflow has no editable install of the package"
    assert any("visualization" in line for line in installs), (
        "the CI install must select the 'visualization' extra: tests/test_ursell_"
        "error_budget.py and tests/test_ursell_band_audit.py call plot() without "
        "pytest.importorskip and otherwise fail with ModuleNotFoundError: matplotlib"
    )


def test_evidence_generation_precedes_the_test_suite(workflow_text: str) -> None:
    """Every artifacts/ producer must run before pytest."""
    pytest_index = _index_of_step(workflow_text, "Run test suite")
    producers = [
        name
        for name in _step_names(workflow_text)
        if name != "Run test suite"
    ]
    late = []
    for name in producers:
        # Only steps that write evidence matter; the checkout/setup/upload steps
        # are handled separately below.
        block = workflow_text[_index_of_step(workflow_text, name):]
        block = block.split("\n      - ", 1)[0]
        writes_artifacts = "artifacts/" in block or "--write" in block
        if writes_artifacts and _index_of_step(workflow_text, name) > pytest_index:
            late.append(name)
    assert not late, (
        "these steps produce the evidence the suite reads but run after pytest: "
        f"{late}"
    )


def test_document_claim_artifact_is_regenerated_before_the_suite(workflow_text: str) -> None:
    """The doc-versus-code gate should see a freshly measured artifact."""
    index = _index_of_step(workflow_text, "Re-measure the Ursell document-claim artifact")
    assert index < _index_of_step(workflow_text, "Run test suite")
    assert "tensorbem.doc_claims --write" in workflow_text


def test_upload_happens_after_the_suite(workflow_text: str) -> None:
    """The uploaded reports must be the ones the suite was checked against."""
    upload = workflow_text.index("actions/upload-artifact")
    assert _index_of_step(workflow_text, "Run test suite") < upload
    assert "artifacts/ursell_doc_claims.json" in workflow_text


def test_the_suite_is_fanned_out_and_the_job_is_bounded(workflow_text: str) -> None:
    """The suite must not hold a runner for the whole serial run (m4796).

    ``ci-loong64`` is a shared pool: a serial run holds the slot for tens of
    minutes and is cancelled as soon as the next commit needs it, and a
    cancelled job is red (main@fa868b64, 2026-11-06: cancelled after 29.8 min
    with no failing step).  Bounding the job turns that silent failure mode
    into an explicit timeout, and the fan-out is what keeps the job inside the
    bound.
    """
    block = workflow_text[_index_of_step(workflow_text, "Run test suite"):]
    block = block.split("\n      - ", 1)[0]
    assert "-n auto" in block, "the suite must fan out over the runner's cores"
    assert "--maxprocesses" in block, (
        "an unbounded fan-out on a many-core runner can exhaust memory"
    )
    assert "loadfile" in block, (
        "grouping by file keeps per-module state on one worker, so the split "
        "cannot change what a test observes"
    )
    assert re.search(r"^\s+timeout-minutes:\s*\d+\s*$", workflow_text,
                     flags=re.MULTILINE), "the job needs an explicit bound"
    assert "pytest-xdist" in workflow_text, (
        "the fan-out must be installable: name the plugin the suite relies on"
    )
