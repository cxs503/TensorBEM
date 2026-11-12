"""Gates for the corrected wall-sided-barge cross-check.

The claims under test are narrow and falsifiable:

1. the old negative control was a *geometry* defect, not a physics phenomenon:
   the recorded capytaine barge mesh keeps its waterplane face (sealed box),
   yields a volume of 2.0 m^3 for a case labelled T = 0.5, and produces
   non-physical negative heave damping;
2. the corrected geometry (wetted surface only) is solvable in the tested band
   -- no irregular frequency below nu = 3.7, no sign change in damping;
3. TensorBEM and capytaine, driven on the identical triangulation, agree on
   heave added mass and damping to the band fixed before the run;
4. the committed artifacts cannot drift away from the code or the prose.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from tensorbem.benchmarks.barge_crosscheck import (
    ARTIFACT_ROOT,
    DOC_PATH,
    LEVELS,
    NU_GRID,
    REFERENCE_FILE,
    RESULT_FILE,
    TOLERANCE,
    compare,
    load_reference,
    render_document,
    run_level,
)
from tensorbem.geometry3d import box_hull_mesh
from tensorbem.interior_panels import waterplane_lid_points

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference() -> dict:
    return load_reference(REPO_ROOT / ARTIFACT_ROOT / REFERENCE_FILE)


@pytest.fixture(scope="module")
def study() -> dict:
    return json.loads((REPO_ROOT / ARTIFACT_ROOT / RESULT_FILE).read_text())


def test_reference_provenance_complete(reference: dict) -> None:
    meta = reference["metadata"]
    assert meta["tool"] == "capytaine"
    assert re.fullmatch(r"\d+\.\d+[0-9a-z.\-]*", meta["version"]), meta["version"]
    assert meta["platform"] and meta["python"]
    assert meta["density"] == pytest.approx(1025.0)
    assert meta["gravity"] == pytest.approx(9.80665)
    resolutions = [[int(v) for v in row["resolution"]] for row in reference["levels"]]
    assert resolutions == [list(res) for res in LEVELS]
    panels = [row["panels"] for row in reference["levels"]]
    assert panels == sorted(panels) and len(set(panels)) == len(panels)
    assert all(row["volume"] == pytest.approx(1.0) for row in reference["levels"])
    for row in reference["levels"]:
        assert set(row["bare"]) == {"0.500", "1.000", "2.000"}
        assert set(row["lid"]) == set(row["bare"])


def test_old_negative_control_was_a_geometry_defect(reference: dict) -> None:
    box = reference["sealed_box_reproduction"]
    # the waterplane face survives immersed_part() ...
    assert box["faces_lying_on_waterline"] > 0
    # ... which is why the labelled T=0.5 case carried 2.0 m^3 ...
    assert box["volume"] == pytest.approx(2.0)
    # ... and why the damping came out negative, i.e. non-physical.
    assert min(v["heave"]["damping"] for v in box["values"].values()) < 0.0


def test_corrected_case_is_free_of_irregular_frequencies(reference: dict) -> None:
    estimate = reference["irregular_frequency_estimate"]
    assert estimate["nu_irr"] > max(NU_GRID)
    # every corrected (bare) damping value is positive and the heave added mass
    # decreases monotonically with frequency -- the old "sign change between
    # frequencies" claim does not hold for the real geometry.
    rows = [row["bare"] for row in reference["levels"]]
    for row in rows:
        dampings = [row[f"{nu:.3f}"]["heave"]["damping"] for nu in NU_GRID]
        assert all(value > 0.0 for value in dampings), dampings
        masses = [row[f"{nu:.3f}"]["heave"]["added_mass"] for nu in NU_GRID]
        assert masses[0] > masses[1] > masses[2], masses


def test_lid_generator_handles_rectangular_waterplane() -> None:
    mesh = box_hull_mesh(2.0, 1.0, 0.5, n_length=8, n_beam=4, n_draft=2)
    lid = waterplane_lid_points(mesh)
    assert lid.count > 0
    assert lid.points.shape[1] == 3
    assert (lid.points[:, 2] < 0.0).all()
    # all lid points lie inside the rectangular waterplane (|x| <= 1, |y| <= 0.5)
    assert (abs(lid.points[:, 0]) <= 1.0).all()
    assert (abs(lid.points[:, 1]) <= 0.5).all()
    # a non-watertight / non-star-shaped request is rejected instead of silently
    # placing lid points in the fluid
    with pytest.raises(ValueError):
        waterplane_lid_points(mesh, shrink=1.5)


def test_coarsest_level_reproduces_committed_numbers(study: dict) -> None:
    fresh = run_level(LEVELS[0])
    stored = study["levels"][0]
    assert fresh["panels"] == stored["panels"]
    for route in ("bare", "lid"):
        for nu_key, comps in stored[route].items():
            for quantity in ("added_mass", "damping"):
                expected = comps[quantity]
                got = fresh[route][nu_key][quantity]
                assert abs(got - expected) / max(abs(expected), 1.0) < 1e-10, (
                    route,
                    nu_key,
                    quantity,
                    got,
                    expected,
                )


def test_verdict_follows_from_the_cells(study: dict) -> None:
    comp = study["comparison"]
    assert comp["tolerance"] == pytest.approx(TOLERANCE)
    assert comp["verdict"] in {"PASS", "FAIL"}
    finest = [e for e in comp["entries"] if e["resolution"] == list(LEVELS[-1])]
    assert len(finest) == 2 * 2 * len(NU_GRID)
    outside = [
        e for e in finest if e["relative"] is not None and abs(e["relative"]) > TOLERANCE
    ]
    assert comp["verdict"] == ("PASS" if not outside else "FAIL")
    for entry in finest:
        assert entry["tensorbem"] == pytest.approx(
            entry["capytaine"] * (1.0 + entry["relative"]), rel=1e-9
        )
    # no cell is silently dropped from the failure list
    assert len(comp["failures"]) == len(outside)


def test_added_mass_agrees_with_the_independent_code(study: dict) -> None:
    comp = study["comparison"]
    masses = [
        e
        for e in comp["entries"]
        if e["resolution"] == list(LEVELS[-1]) and e["quantity"] == "added_mass"
    ]
    assert masses
    for entry in masses:
        assert abs(entry["relative"]) < TOLERANCE, entry


def test_document_matches_the_artifact(study: dict) -> None:
    doc = (REPO_ROOT / DOC_PATH).read_text()
    assert doc == render_document(study)
    assert f"{study['comparison']['verdict']}" in doc
    assert "nu_irr" in doc


def test_capytaine_never_leaks_into_the_library() -> None:
    """capytaine is an optional external dependency: only scripts/ may import it."""
    pattern = re.compile(r"^\s*(?:import\s+capytaine\b|from\s+capytaine\b)", re.MULTILINE)
    for directory in ("src", "tests"):
        for path in (REPO_ROOT / directory).rglob("*.py"):
            match = pattern.search(path.read_text())
            assert match is None, f"{path}: {match.group(0) if match else ''}"
