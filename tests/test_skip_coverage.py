"""Machine gate: no physics assertion may be invisible on a clean checkout.

``docs/skip_coverage/scan.py`` classifies every artifact-guarded test module by
whether a clean checkout can supply the artifact.  This module turns that audit
into a test:

* **no dark modules** -- a module whose guarded artifacts are all untracked and
  not produced by CI skips its assertions everywhere except on the machine that
  generated the file.  That is a deleted check, not a skip.
* **no unlisted untracked artifact** -- a guarded path that a clean checkout
  cannot supply must be reviewed explicitly, in :data:`KNOWN_UNTRACKED`.
* **the committed audit artifact matches a fresh scan** -- the report cannot rot.

Context: ``tests/test_finite_depth_energy_convergence.py`` and
``tests/test_ursell_reference_pinning.py`` used to guard on ``outputs/...json``,
a git-ignored directory.  Five assertions (the convergence tables, the
artifact-vs-code comparison, the deep-water control, the first-order trend, the
pinned panel solution) never executed on CI until the guards learned about the
versioned archives under ``docs/m2498`` and ``docs/m2232``.  See
``docs/skip_coverage/audit.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "docs" / "skip_coverage" / "scan.py"
REPORT = ROOT / "docs" / "skip_coverage" / "report.json"

#: Guarded artifacts a clean checkout cannot supply, with the reason they are
#: tolerated.  Empty means "every guarded artifact is readable on CI".
KNOWN_UNTRACKED: dict[str, str] = {}

#: Modules the audit found guarded.  A new guard appearing here is fine, a guard
#: disappearing should be a deliberate edit -- this list is not a limit, it only
#: documents the audited surface.
AUDITED_MODULES = {
    "test_finite_depth_envelope.py",
    "test_finite_depth_refinement.py",
    "test_mixed_boundary.py",
    "test_near_surface_band.py",
}


def _load_scanner():
    spec = importlib.util.spec_from_file_location("_skip_coverage_scan", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scan() -> dict:
    return _load_scanner().scan()


def test_no_module_hides_its_assertions_behind_an_untracked_artifact() -> None:
    summary = _scan()["summary"]
    assert summary["dark_modules"] == [], (
        "these test modules guard on artifacts a clean checkout cannot produce, "
        "so their assertions never run on CI: "
        f"{summary['dark_modules']}"
    )


def test_every_untracked_guarded_artifact_is_reviewed() -> None:
    summary = _scan()["summary"]
    unlisted = sorted(set(summary["untracked_paths"]) - set(KNOWN_UNTRACKED))
    assert unlisted == [], (
        "guarded artifacts that CI cannot supply and that nobody has reviewed: "
        f"{unlisted}; point the test at a versioned copy or add it to KNOWN_UNTRACKED"
    )


def test_the_audited_modules_still_exist() -> None:
    scanned = set(_scan()["summary"]["modules_with_guards"])
    assert AUDITED_MODULES <= scanned, (
        f"module vanished from the audit: {sorted(AUDITED_MODULES - scanned)}"
    )


def test_committed_audit_report_matches_a_fresh_scan() -> None:
    assert REPORT.exists(), f"missing audit artifact {REPORT}"
    assert json.loads(REPORT.read_text()) == _scan(), (
        "docs/skip_coverage/report.json is stale; regenerate with "
        "`PYTHONPATH=src python3 docs/skip_coverage/scan.py "
        "--output docs/skip_coverage/report.json`"
    )