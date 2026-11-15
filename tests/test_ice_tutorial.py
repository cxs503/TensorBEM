"""Gates for the ice-navigation tutorial (``docs/ice_navigation_tutorial``).

Four gates, matching chapter 9 of the tutorial:

1. **criteria recompute** — every ``computable`` criterion of the registry is
   recomputed here and must pass;
2. **rendering fingerprint** — the numeric blocks embedded in the chapter
   markdown must equal a fresh, from-scratch rendering (no hand-edited digits);
3. **reference completeness** — every criteria row resolves to a bibliography
   entry and every chapter carries at least three rows;
4. **chapter self-containment** — each chapter carries the learning objectives,
   prerequisites, criteria table, limitations, references and reproduce
   sections, and the figures it points at exist.

The tutorial is a *teaching* artefact, so these tests deliberately assert
invariants (``mu`` monotone, ``k_ice > k_open``, mesh refinement improves the
error, ``mu -> 0`` reproduces open water) rather than merely executing code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:  # the tutorial harness lives outside src/
    sys.path.insert(0, str(EXAMPLES))

from ice_tutorial import criteria as criteria_module  # noqa: E402
from ice_tutorial import model as model_module  # noqa: E402
from ice_tutorial import render as render_module  # noqa: E402

DOCS = ROOT / "docs" / "ice_navigation_tutorial"

REQUIRED_SECTIONS = (
    "## 0. 学习目标",
    "### 0.1 前置知识",
    "## 3. 判据表",
    "注意点与局限",
    "## 7. 参考文献",
    "## 8. 复现命令",
)


@pytest.fixture(scope="module")
def built():
    """Compute every live chapter once (the BEM points dominate the runtime)."""
    payloads = {
        chapter: model_module.compute(chapter, with_bem=True)
        for chapter in model_module.COMPUTED_CHAPTERS
    }
    verdicts = {
        chapter: criteria_module.evaluate(chapter, payloads.get(chapter, {}).get("metrics", {}))
        for chapter in criteria_module.chapters()
    }
    return payloads, verdicts


def _chapter_path(chapter: str) -> Path:
    return DOCS / render_module.CHAPTER_FILES[chapter]


def _block(text: str, chapter: str, kind: str) -> str:
    start = f"<!-- AUTO-{kind}:{chapter} -->"
    end = f"<!-- /AUTO-{kind}:{chapter} -->"
    i = text.find(start)
    j = text.find(end)
    assert i >= 0 and j > i, f"missing marker pair {kind}:{chapter}"
    return text[i + len(start) : j].strip()


def test_tutorial_has_at_least_eight_chapters() -> None:
    files = [DOCS / name for name in render_module.CHAPTER_FILES.values()]
    present = [f for f in files if f.exists()]
    assert len(present) >= 8, f"only {len(present)} chapters present"
    for chapter, name in render_module.CHAPTER_FILES.items():
        assert (DOCS / name).exists(), f"{chapter} missing: {name}"


@pytest.mark.parametrize("chapter", sorted(render_module.CHAPTER_FILES))
def test_chapters_are_self_contained(chapter: str) -> None:
    text = _chapter_path(chapter).read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"{chapter}: missing section {section!r}"
    # criteria + (for computed chapters) numbers blocks must be filled in
    assert _block(text, chapter, "CRITERIA"), f"{chapter}: empty criteria block"
    if chapter in model_module.COMPUTED_CHAPTERS:
        assert _block(text, chapter, "NUMBERS"), f"{chapter}: empty numbers block"
        assert (DOCS / "figs" / render_module.FIGURE_FILES[chapter]).exists(), (
            f"{chapter}: figure missing"
        )
    for placeholder in ("TODO", "TBD", "待填", "XXX"):
        assert placeholder not in text, f"{chapter}: placeholder {placeholder!r} left in text"


def test_every_criteria_row_has_reference() -> None:
    for chapter in criteria_module.chapters():
        rows = criteria_module.criteria_for(chapter)
        assert len(rows) >= 3, f"{chapter}: only {len(rows)} criteria rows"
        for criterion in rows:
            assert criterion.ref_key in criteria_module.REFS
            assert criterion.ref.strip(), f"{criterion.id}: empty reference text"
            assert criterion.evidence in criteria_module.VALID_EVIDENCE
        verdicts = criteria_module.evaluate(chapter, {})
        for row in verdicts:
            assert row["ref"].strip(), f"{row['id']}: reference missing in rendered row"


def test_registry_covers_nine_chapters() -> None:
    assert criteria_module.chapters() == tuple(sorted(render_module.CHAPTER_FILES))
    ids = [c.id for c in criteria_module.CRITERIA]
    assert len(ids) == len(set(ids)), "duplicate criteria id"


def test_computable_metrics_are_wired(built) -> None:
    payloads, _ = built
    for chapter, payload in payloads.items():
        metrics = payload["metrics"]
        for criterion in criteria_module.criteria_for(chapter):
            if criterion.computable:
                assert criterion.metric in metrics, (
                    f"{criterion.id}: metric {criterion.metric!r} not produced by {chapter}"
                )


def test_computable_criteria_all_pass(built) -> None:
    payloads, verdicts = built
    failures: list[str] = []
    for chapter, rows in verdicts.items():
        for row in rows:
            if row["verdict"] == "未通过":
                failures.append(f"{row['id']} (observed {row['observed']}): {row['statement']}")
    assert not failures, "criteria failed:\n" + "\n".join(failures)
    for chapter in payloads:
        assert any(row["computable"] for row in verdicts[chapter]), (
            f"{chapter}: no computable criterion — a chapter with live computation must gate on numbers"
        )


def test_withheld_capabilities_are_declared(built) -> None:
    """Unverified capabilities must appear as WITHHELD, never silently upgraded."""
    _, verdicts = built
    grades = {row["evidence"] for rows in verdicts.values() for row in rows}
    assert "WITHHELD" in grades, "the tutorial must keep an honest WITHHELD ledger"
    assert "MOCK" not in grades and "STUB" not in grades, "no placeholder may enter a chapter"


def test_rendered_blocks_match_fresh_rendering(built) -> None:
    payloads, verdicts = built
    for chapter, payload in payloads.items():
        text = _chapter_path(chapter).read_text(encoding="utf-8")
        assert _block(text, chapter, "NUMBERS") == render_module.render_numbers(chapter, payload).strip(), (
            f"{chapter}: numbers block drifted from a fresh computation (re-run render.py)"
        )
    for chapter, rows in verdicts.items():
        text = _chapter_path(chapter).read_text(encoding="utf-8")
        assert _block(text, chapter, "CRITERIA") == render_module.render_criteria(chapter, rows).strip(), (
            f"{chapter}: criteria block drifted from a fresh evaluation (re-run render.py)"
        )


def test_artifacts_bundle_is_present() -> None:
    bundle = DOCS / "artifacts" / "tutorial_numbers.json"
    assert bundle.exists(), "artifacts/tutorial_numbers.json missing — run render.py"
    text = bundle.read_text(encoding="utf-8")
    assert "fingerprint" in text and "criteria" in text


def test_render_check_mode_passes() -> None:
    assert render_module.main(["--check", "--no-bem", "--no-figs"]) == 0