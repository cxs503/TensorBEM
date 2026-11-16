"""Pins for ``docs/audit_xval_r2_degeneracy.md``.

The audit document is the *narrative* half of the R2 degeneracy evidence: it is
what a reader cites when asking "was the old 9859.64% FAIL an artefact of
comparing two numerical zeros, and what is the real deviation now?".  Its first
revision was hand written and, once the study was refined from 256 to 1024
TensorBEM panels, it kept quoting ``9.38%`` / ``87.93%`` while the artifact had
moved to ``5.82%`` / ``6.71%`` -- the document silently became unsupported
evidence.

These tests make that impossible: the document is rendered from
``results.json`` by ``render_degeneracy_document`` and every number in it is an
arithmetic function of the artifact.  Nothing here re-runs a solve; the pins are
pure artifact arithmetic, so they run in milliseconds and stay meaningful in CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study() -> dict:
    path = ROOT / r2.ARTIFACT_ROOT / r2.RESULT_FILE
    assert path.exists(), f"{path} missing"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def audit_text(study: dict) -> str:
    path = ROOT / r2.AUDIT_DOC_PATH
    assert path.exists(), f"{path} missing"
    return path.read_text()


# --------------------------------------------------------------------------- 1
def test_audit_document_is_the_arithmetic_of_the_artifact(
    study: dict, audit_text: str
) -> None:
    """Byte equality with a re-render, exactly like the main document's pin."""
    assert audit_text == r2.render_degeneracy_document(study)


def test_audit_document_has_the_six_required_sections(audit_text: str) -> None:
    for heading in (
        "## 1. 案例介绍",
        "## 2. 怎么做的",
        "## 3. 计算结果",
        "## 4. 结果图",
        "## 5. 注意点",
        "## 6. 误差判定",
    ):
        assert heading in audit_text, heading


# --------------------------------------------------------------------------- 2
def test_every_scored_case_quotes_the_artifact_percentage(
    study: dict, audit_text: str
) -> None:
    """The per-case worst physics deviation must appear, and be the artifact's."""
    cases = study["invariant_scoring"]["cases"]
    assert cases, "invariant scoring must record per-case verdicts"
    for case, block in cases.items():
        value = 100.0 * float(block["worst_physics_relative"])
        assert "| {} |".format(case) in audit_text
        assert "{:.2f}%".format(value) in audit_text, (
            f"{case}: the document must quote the artifact's {value:.2f}%"
        )
        # every individual failure line is printed, none merged away
        for failure in block["physics_failures"]:
            assert failure in audit_text


def test_document_never_quotes_a_superseded_deviation(
    study: dict, audit_text: str
) -> None:
    """No stale percentage from the 256-panel ladder may survive.

    The stale revision quoted 9.38% / 87.93% as *verdict-bearing* numbers.  Those
    values are still legitimate as refinement-ladder rungs, so this test only
    forbids them from being the *scored* worst deviation of any case.
    """
    scored = {
        "{:.2f}%".format(100.0 * float(b["worst_physics_relative"]))
        for b in study["invariant_scoring"]["cases"].values()
    }
    # the physics rows of the error table (section 6) must quote the scored ones
    section = audit_text.split("## 6. 误差判定", 1)[1]
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and "最差计分项" in line
    ]
    assert len(rows) == len(study["invariant_scoring"]["cases"]), rows
    for line in rows:
        value = re.search(r"\| (\d+\.\d{2})% \|", line)
        assert value, line
        assert value.group(1) + "%" in scored, (
            f"{value.group(1)}% appears in the error table but is not any case's "
            "scored worst deviation -- the document has drifted from the artifact"
        )


# --------------------------------------------------------------------------- 3
def test_invariant_axes_are_quoted_from_the_audit_block(
    study: dict, audit_text: str
) -> None:
    audit = study["degeneracy"]
    assert audit["verdict"] == "PASS"
    for row in audit["cases"]:
        assert row["case"] in audit_text
        assert "{:.1e}".format(float(row["finest_worst_ratio"])) in audit_text
        for ratio in row["decay_ratios"]:
            assert "{:.2f}".format(float(ratio)) in audit_text
        assert row["monotone_decay"] is True
    transfer = study["centre_transfer"]
    assert transfer["verdict"] == "PASS"
    assert "{:.1e}".format(float(transfer["worst_residual"])) in audit_text


def test_refinement_ladder_is_reproduced_verbatim_from_convergence(
    study: dict, audit_text: str
) -> None:
    """Section 3.3 must be the convergence table, row for row.

    This is the axis the audit uses to argue "discretisation, not an
    implementation error": if the document and ``convergence`` disagree the
    argument loses its evidence.
    """
    cases = study["comparison"]["cases"]
    for case, rows in study["convergence"].items():
        assert case in audit_text
        for row in rows:
            expected = "| {} | {} | {} | {} | {:.2f}% | {:.2f}% |".format(
                case,
                row["level"],
                int(row["n_faces"]),
                int(cases[case]["capytaine_faces"]),
                abs(100.0 * float(row["worst_diagonal"])),
                abs(100.0 * float(row["worst_coupling"])),
            )
            assert expected in audit_text, expected
        diagonal = [abs(float(row["worst_diagonal"])) for row in rows]
        if len(diagonal) > 1:
            assert all(b < a for a, b in zip(diagonal, diagonal[1:])), (
                f"{case}: the document claims monotone shrinkage; convergence "
                f"disagrees: {diagonal}"
            )


# --------------------------------------------------------------------------- 4
def test_acceptance_and_internal_bands_are_not_conflated(
    study: dict, audit_text: str
) -> None:
    """Internal 5%/10% FAIL and owner 15% PASS must both be stated, separated."""
    ledger = study["ledger_acceptance"]
    comparison = study["comparison"]
    assert comparison["verdict"] == "FAIL"
    assert ledger["threshold"] == pytest.approx(0.15)
    assert ledger["satisfied"] is True
    assert ledger["worst_active_relative"] == pytest.approx(
        comparison["worst_active_relative"]
    )
    assert "| 锚点验收（owner 15%） |" in audit_text
    assert "owner 的 15% 验收口径" in audit_text
    # and the honest answer to "is it <3%?" must be 否 for the physics axis
    assert "未通过 <3%" in audit_text


def test_document_declares_the_mesh_mismatch(study: dict, audit_text: str) -> None:
    cases = study["comparison"]["cases"]
    tensorbem = sorted({int(b["tensorbem_faces"]) for b in cases.values()})
    capytaine = sorted({int(b["capytaine_faces"]) for b in cases.values()})
    assert "网格分辨率不对等" in audit_text
    for faces in tensorbem:
        assert str(faces) in audit_text
    for faces in capytaine:
        assert str(faces) in audit_text


# --------------------------------------------------------------------------- 5
def test_write_outputs_keeps_the_audit_document_in_sync(
    study: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``write_outputs`` regenerates the audit document, so a fresh study run
    cannot leave a stale narrative behind."""
    monkeypatch.setattr(r2, "DOC_PATH", tmp_path / "doc.md")
    monkeypatch.setattr(r2, "AUDIT_DOC_PATH", tmp_path / "audit.md")
    figures = [Path(entry) for entry in study.get("figures", [])]
    monkeypatch.setattr(r2, "render_figures", lambda _study: list(figures))
    paths = r2.write_outputs(json.loads(json.dumps(study)), root=tmp_path)
    assert paths["audit_document"] == tmp_path / "audit.md"
    assert paths["audit_document"].read_text() == r2.render_degeneracy_document(study)