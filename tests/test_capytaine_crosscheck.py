"""Acceptance tests for the capytaine cross-code verification anchor.

Anchor **FEAT-OSS-INVENTORY-BEM** (deep-dive wave).  The claim under test is
narrow and falsifiable: the 3-D free-surface BEM solver
(`tensorbem.free_surface_solver3d`) and the independently written panel code
capytaine, driven on the same smooth bodies and frequencies, agree on added
mass and damping to within the band fixed before the run (5 %).

The anchor's *recorded verdict* is whatever the comparison produced -- this file
does not assume PASS.  What it does enforce is that the verdict cannot lie:
every cell outside the band must appear in the failure list, the added-mass
cells (the well-conditioned quantities) must be inside the band, the one
high-frequency damping cell that misses it must be recorded with its measured
deviation, and the deviation must shrink as the mesh is refined.

Checks:

1. **Provenance of the foreign reference** -- `capytaine_ref.json` must name the
   tool, version, git commit, platform and mesh resolutions it was produced
   with.  Without that the comparison is unfalsifiable folklore.
2. **Reproducibility pin** -- re-solving the coarsest level in-process must
   reproduce the committed number to round-off, so the artifact cannot quietly
   drift away from the code.
3. **The comparison itself** -- verdict consistent with the entries, no
   swallowed misses, added mass inside the band.
4. **Prose/artifact agreement** -- `docs/capytaine_crosscheck.md` must equal
   `render_document` of the committed `results.json`.
5. **Optional-dependency hygiene** -- `capytaine` may only be imported by the
   producer script; it must not leak into `src/` or `tests/`.
"""

from __future__ import annotations

import json
import math
import pathlib
import re

import pytest

from tensorbem.benchmarks.capytaine_crosscheck import (
    ARTIFACT_ROOT,
    DOC_PATH,
    REFERENCE_FILE,
    RESULT_FILE,
    TOLERANCE,
    load_reference,
    render_document,
    run_hemisphere,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study() -> dict:
    return json.loads((REPO_ROOT / ARTIFACT_ROOT / RESULT_FILE).read_text())


@pytest.fixture(scope="module")
def reference() -> dict:
    return load_reference(REPO_ROOT / ARTIFACT_ROOT / REFERENCE_FILE)


def test_reference_provenance_complete(reference: dict) -> None:
    meta = reference["metadata"]
    assert meta["tool"] == "capytaine"
    assert re.fullmatch(r"\d+\.\d+[0-9a-z.\-]*", meta["version"]), meta["version"]
    assert meta["git_commit"] and len(meta["git_commit"]) == 40
    assert meta["platform"] and meta["python"]
    assert meta["density"] == pytest.approx(1025.0)
    assert meta["gravity"] == pytest.approx(9.80665)
    for case in ("hemisphere_r1", "sphere_r1_z-0.5"):
        entry = reference["cases"][case]
        assert entry["levels"], f"{case} has no refinement levels"
        faces = [row["n_faces"] for row in entry["levels"]]
        assert faces == sorted(faces) and len(set(faces)) == len(faces)
        assert entry["reference_faces"] == faces[-1]
        assert set(entry["reference"]) == {"0.500", "1.000", "2.000"}
    assert len(reference["cases"]["hemisphere_r1"]["levels"]) >= 4


def test_negative_control_records_its_reason(reference: dict) -> None:
    barge = reference["cases"]["barge_L2_B1_T0.5"]
    assert barge["excluded"] is True
    assert "irregular" in barge["exclusion_reason"].lower()
    # the barge really is unusable: capytaine's own finest level still returns
    # a non-physical (negative) added mass, so no honest reference exists.
    assert min(v["heave"]["added_mass"] for v in barge["reference"].values()) < 0.0


def test_coarsest_level_reproduces_committed_numbers(study: dict) -> None:
    fresh = run_hemisphere((2,))
    stored = study["tensorbem"]["hemisphere_r1"]["levels"][0]
    assert fresh["levels"][0]["n_faces"] == stored["n_faces"] == 64
    for nu_key, per_dof in stored["values"].items():
        for dof, comps in per_dof.items():
            for quantity, expected in comps.items():
                got = fresh["levels"][0]["values"][nu_key][dof][quantity]
                scale = max(abs(expected), 1.0)
                assert abs(got - expected) / scale < 1e-10, (
                    f"{nu_key} {dof} {quantity}: {got} vs {expected}"
                )


def test_verdict_is_consistent_with_entries(study: dict) -> None:
    comp = study["comparison"]
    assert comp["tolerance"] == pytest.approx(TOLERANCE)
    assert comp["verdict"] in {"PASS", "FAIL"}
    compared = {k: v for k, v in comp["cases"].items() if v.get("compared")}
    assert set(compared) == {"hemisphere_r1", "sphere_r1_z-0.5"}
    outside = []
    worst = 0.0
    for name, info in compared.items():
        cells = [e for e in info["entries"] if e.get("relative") is not None]
        assert len(cells) >= 4, f"{name} compares too few cells"
        worst = max(worst, max(abs(e["relative"]) for e in cells))
        for cell in cells:
            if abs(cell["relative"]) > TOLERANCE:
                outside.append((name, cell["nu"], cell["dof"], cell["quantity"]))
        for cell in cells:
            assert cell["tensorbem"] == pytest.approx(
                cell["capytaine"] * (1.0 + cell["relative"]), rel=1e-9
            )
    # the headline verdict follows from the cells; no cell is silently dropped
    assert comp["verdict"] == ("PASS" if not outside else "FAIL")
    assert comp["worst_relative"] == pytest.approx(worst, rel=1e-12)
    recorded = {(f.split()[0], f) for f in comp["failures"]}
    for name, nu, dof, quantity in outside:
        assert any(
            name in text and f"{nu:.2f}" in text and dof in text and quantity in text
            for _, text in recorded
        ), (name, nu, dof, quantity, comp["failures"])


def test_added_mass_inside_band(study: dict) -> None:
    """Added mass is the well-conditioned quantity: it must be within the band."""
    for name, info in study["comparison"]["cases"].items():
        if not info.get("compared"):
            continue
        for cell in info["entries"]:
            if cell.get("relative") is None or cell["quantity"] != "added_mass":
                continue
            assert abs(cell["relative"]) <= TOLERANCE, (name, cell)


def test_damping_inside_band_at_moderate_frequency(study: dict) -> None:
    """Damping at nu <= 1 agrees; the nu = 2 cell is the documented resolution miss."""
    misses = []
    for name, info in study["comparison"]["cases"].items():
        if not info.get("compared"):
            continue
        for cell in info["entries"]:
            if cell.get("relative") is None or cell["quantity"] != "damping":
                continue
            if cell["nu"] <= 1.0 + 1e-9:
                assert abs(cell["relative"]) <= TOLERANCE, (name, cell)
            elif abs(cell["relative"]) > TOLERANCE:
                misses.append((name, cell["nu"], cell["relative"]))
    if misses:
        # an honest miss must be documented in the prose, not only in JSON
        doc = (REPO_ROOT / DOC_PATH).read_text()
        assert "not met" in doc.lower() or "FAIL" in doc


def test_skipped_damping_is_documented(study: dict) -> None:
    sphere = study["comparison"]["cases"]["sphere_r1_z-0.5"]
    skipped = [e for e in sphere["entries"] if e.get("relative") is None]
    assert skipped, "the vanishing sphere heave damping at nu=1 should be skipped"
    for entry in skipped:
        assert entry["nu"] == pytest.approx(1.0)
        assert entry["quantity"] == "damping"
        assert "vanishing" in entry["note"]


def test_convergence_recorded_for_every_level(study: dict) -> None:
    for case, rows in study["convergence"].items():
        assert len(rows) == len(study["tensorbem"][case]["levels"]), case
        faces = [row["n_faces"] for row in rows]
        assert faces == sorted(faces)
        assert all(math.isfinite(row["worst_relative"]) for row in rows)
    rows = study["convergence"]["hemisphere_r1"]
    assert rows[-1]["worst_relative"] < rows[0]["worst_relative"], (
        "refinement must move the solution toward the independent reference"
    )


def test_document_matches_artifacts(study: dict) -> None:
    doc = (REPO_ROOT / DOC_PATH).read_text()
    assert doc == render_document(study)
    assert f"**Verdict: {study['comparison']['verdict']}**" in doc
    for figure in (
        "docs/fig_capytaine_crosscheck_frequency.png",
        "docs/fig_capytaine_crosscheck_convergence.png",
    ):
        assert figure in doc
        assert (REPO_ROOT / figure).exists(), figure


def test_capytaine_not_a_runtime_dependency() -> None:
    offenders = []
    for path in list((REPO_ROOT / "src").rglob("*.py")) + list(
        (REPO_ROOT / "tests").rglob("*.py")
    ):
        text = path.read_text()
        if re.search(r"^\s*(import|from)\s+capytaine", text, re.MULTILINE):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, offenders