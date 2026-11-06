"""The m4796 CI wall-clock evidence, and the report that cites it (m4897).

``docs/dev_report_m4796_ci_red_fix.md`` argues that ``main@fa868b64`` was red
because the ``pytest`` job *held a shared runner too long*, not because a test
failed: the check run is ``cancelled`` after 29.8 min, and the 25 most recent
runs of the workflow have a median wall clock of 49.7 min with six runs past
the new 90-minute bound.

Numbers like that are only worth anything if the artifact behind them is in the
tree and the prose can be recomputed from it.  The re-opened submission
(original PR #325 was closed) is exactly where that went wrong on ``main``:
the prose cited ``docs/m4796/ci_duration_evidence.json`` and
``docs/m4796/ci_wallclock.png`` while neither file existed, so every claim in
section 3 was unverifiable from the checkout.

These tests are the gate:

* both artifacts exist, and the figure is a readable PNG of the documented size;
* every number in the report's wall-clock table is recomputed from the JSON by
  ``tools/m4796_ci_evidence.py`` -- the single place the JSON becomes numbers;
* the sample the bound is counted against is the bound the workflow actually
  carries (changing ``timeout-minutes`` must not silently invalidate the report);
* every ``docs/...`` path the report names exists, so citations cannot dangle.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import m4796_ci_evidence as evidence_mod  # noqa: E402

REPORT = ROOT / "docs" / "dev_report_m4796_ci_red_fix.md"
EVIDENCE_JSON = ROOT / "docs" / "m4796" / "ci_duration_evidence.json"
FIGURE = ROOT / "docs" / "m4796" / "ci_wallclock.png"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

#: Pixel size of the committed figure; ``plot()`` pins figsize/dpi to this, so a
#: figure redrawn at another size is caught here rather than in a pull request.
FIGURE_SIZE = (1170, 546)


@pytest.fixture(scope="module")
def summary() -> dict:
    return evidence_mod.summarise(evidence_mod.load(EVIDENCE_JSON))


@pytest.fixture(scope="module")
def report_text() -> str:
    assert REPORT.is_file(), f"missing report: {REPORT}"
    return REPORT.read_text(encoding="utf-8")


def png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    width, height = struct.unpack(">II", raw[16:24])
    return width, height


def test_the_evidence_and_the_figure_are_in_the_tree() -> None:
    """The report cites both by path; a citation to a missing file is not evidence."""
    for path in (EVIDENCE_JSON, FIGURE):
        assert path.is_file(), f"report cites {path.relative_to(ROOT)} but it is missing"
    assert FIGURE.stat().st_size > 10_000, "a 25-bar figure should not be a few hundred bytes"
    assert png_size(FIGURE) == FIGURE_SIZE, (
        f"figure is {png_size(FIGURE)}, expected {FIGURE_SIZE}; redraw it with "
        "`python3 tools/m4796_ci_evidence.py --plot`"
    )


def test_the_target_run_is_recorded_as_cancelled(summary: dict) -> None:
    """The whole diagnosis rests on ``cancelled``, not ``failure``."""
    assert summary["target_sha"].startswith("fa868b64")
    assert summary["target_conclusion"] == "cancelled"
    assert summary["target_wall_min"] == pytest.approx(29.8)
    # Run wall clock and check-run span are two traces of the same job.
    assert summary["target_check_run_min"] == pytest.approx(29.6, abs=0.1)
    assert any("cancel" in note.lower() for note in summary["target_annotations"])


def test_the_sample_is_the_size_the_report_claims(summary: dict) -> None:
    assert summary["n"] == 25
    assert sum(summary["conclusions"].values()) == summary["n"]
    assert summary["conclusions"].get("cancelled") == 10
    assert summary["conclusions"].get("failure") == 11


@pytest.mark.parametrize(
    "label",
    [
        "样本数 n",
        "最短",
        "中位",
        "最长",
        "超过新上界 90 min 的 run 数",
        "`fa868b64`",
        "结论分布",
    ],
)
def test_report_claim_matches_the_evidence(label: str, summary: dict, report_text: str) -> None:
    """One report row per case, so a drifted number names itself."""
    problems = evidence_mod.check_report(summary, report_text)
    blamed = [problem for problem in problems if problem.startswith(repr(label)) or label in problem]
    assert not blamed, blamed[0] if blamed else str(problems)


def test_the_counted_bound_is_the_bound_the_workflow_carries(summary: dict) -> None:
    """``over_bound`` counts runs against ``timeout-minutes``; the two must agree."""
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"^\s+timeout-minutes:\s*(\d+)\s*$", workflow_text, flags=re.MULTILINE)
    assert match is not None, "the pytest job lost its explicit bound"
    assert int(match.group(1)) == summary["bound_min"] == evidence_mod.BOUND_MIN


def test_every_document_path_the_report_names_exists(report_text: str) -> None:
    """This is the defect the re-opened submission fixed: dangling citations."""
    cited = sorted(set(re.findall(r"docs/[A-Za-z0-9_./-]+", report_text)))
    assert cited, "the report cites no documents at all"
    missing = [path for path in cited if not (ROOT / path).exists()]
    assert not missing, f"report cites missing paths: {missing}"


def test_the_figure_is_redrawn_from_the_same_json(tmp_path: Path) -> None:
    """The figure must be a function of the evidence, not a screenshot."""
    pytest.importorskip("matplotlib")
    redrawn = evidence_mod.plot(evidence_mod.load(EVIDENCE_JSON), tmp_path / "redraw.png")
    assert png_size(redrawn) == png_size(FIGURE)