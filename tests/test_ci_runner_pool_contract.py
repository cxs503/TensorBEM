"""Runner-pool contract for ``.github/workflows/tests.yml`` (m4803).

A red gate on ``main`` has two causes that look identical in the checks UI:

* a test failed -- the suite ran, an assertion failed;
* no runner was ever assigned -- the run is ``cancelled`` with
  ``runner_name == ""`` and zero recorded steps, so the suite never started.

The second one happened to ``main@fa868b64`` (run ``42152088640``: queued,
cancelled, no runner) and it is invisible from inside the repository: the code
is fine, the workflow lists the wrong pool.  The evidence is collected
programmatically by ``tools/ci_dispatch_triage.py`` into
``docs/m4803/ci_dispatch_triage.json``.

Because the pool a repository can use changes over time, the workflow must not
hard-code one:

* ``runs-on`` resolves through the repository variable ``TENSORBEM_CI_RUNNER``
  and defaults to the GitHub-hosted image, which is the only pool that can be
  assumed to dispatch;
* ``actions/setup-python`` is guarded, because the self-hosted loong64 runner
  cannot unpack it (``tar: Cannot change mode to rwxr-xr-x``) while the hosted
  image needs it to pin python 3.11.

These are checks on the workflow text, so a future edit that re-pins a single
pool -- and quietly re-breaks main -- is reported in one line.
"""

from __future__ import annotations

import re

from test_ci_workflow_contract import WORKFLOW, workflow_text  # noqa: F401  (fixture)

RUNNER_VARIABLE = "TENSORBEM_CI_RUNNER"


def _runs_on(workflow_text: str) -> str:
    match = re.search(r"^\s*runs-on:\s*(.+?)\s*$", workflow_text, flags=re.MULTILINE)
    assert match is not None, "the job declares no runs-on"
    return match.group(1)


def test_the_runner_pool_is_a_variable_not_a_hard_coded_label(workflow_text: str) -> None:
    """A hard-coded pool is how main went red without a single test running."""
    value = _runs_on(workflow_text)
    assert RUNNER_VARIABLE in value, (
        f"runs-on: {value!r} pins one pool; resolve it through vars.{RUNNER_VARIABLE} "
        "so switching pools does not need a commit"
    )
    assert "||" in value, (
        "runs-on must carry a default: an unset variable would leave runs-on empty "
        "and no runner would ever be assigned"
    )


def test_the_default_pool_is_the_github_hosted_image(workflow_text: str) -> None:
    """The hosted image is the pool that actually dispatched on 2026-11-06."""
    value = _runs_on(workflow_text)
    assert "ubuntu-latest" in value, (
        f"the runs-on default must be 'ubuntu-latest' (got {value!r}); the self-hosted "
        "pool stopped handing out jobs on 2026-11-06 and every run became a cancelled, "
        "no-runner run (docs/m4803/ci_dispatch_triage.json)"
    )


def _setup_python_block(workflow_text: str) -> str:
    """The whole setup-python step, from its name line to the install step."""
    start = workflow_text.index("- name: Set up Python")
    end = workflow_text.index("- name: Install package and test dependencies")
    return workflow_text[start:end]


def test_setup_python_is_skipped_on_a_self_hosted_pool(workflow_text: str) -> None:
    """setup-python cannot unpack on the loong64 runner; it must stay guarded."""
    block = _setup_python_block(workflow_text)
    assert "if:" in block, (
        "actions/setup-python must be conditional: the self-hosted loong64 runner "
        "cannot unpack it and the job dies before the install step"
    )
    assert "startsWith(" in block and "vars." in block, (
        "the guard must key off the runner-selection variable, otherwise it cannot "
        "follow the pool"
    )


def test_the_hosted_pool_still_pins_python(workflow_text: str) -> None:
    """Pinning the interpreter is what made the last green run reproducible."""
    block = _setup_python_block(workflow_text)
    assert 'python-version: "3.11"' in block
    assert "cache: pip" in block


def test_the_triage_evidence_is_not_a_stale_placeholder() -> None:
    """The JSON behind the pool decision must exist and record the outage."""
    import json
    from pathlib import Path

    evidence = (
        Path(__file__).resolve().parents[1] / "docs" / "m4803" / "ci_dispatch_triage.json"
    )
    assert evidence.is_file(), f"missing CI triage evidence: {evidence}"
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["target"]["conclusion"] == "cancelled"
    summary = payload["summary"]
    assert summary["target_runner_assigned"] is False
    assert summary["target_steps_recorded"] == 0
    assert summary["control_all_steps_green"] is True