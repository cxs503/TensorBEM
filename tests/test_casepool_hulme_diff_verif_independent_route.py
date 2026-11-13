"""Pin the independent-route convergence evidence of CASE-BEM-HULME-DIFF-VERIF.

Background.  The case `benchmarks/verified/casepool_bem_hulme_diff_verif` grades
the Kelvin-BEM 3-D diffraction force against a re-derived multipole reference
(`tensorbem.analytic.hulme_hemisphere_radiation`; the published table digits of
Hulme 1982 were not available) and cross-checks it with capytaine + exact
Froude-Krylov.  Its one weak number was the independent route at ka = 2.0,
heave: 4.30 % deviation at the frozen ladder's finest rung (1024 faces), i.e.
above the ecosystem 3 % line, with a 3-rung Richardson attempt that made things
worse (7.48 %) and therefore proved nothing.

`independent_route_convergence.json` (produced by the sibling script) refines
that ladder to 1024 / 2160 / 4096 wetted faces at ka = 1.0 / 1.5 / 2.0 and
settles the question: the deviation falls monotonically 4.30 % -> 3.13 % ->
2.17 %, both force components approach the reference monotonically, and the
reference's own multipole truncation sensitivity is 9e-5 -- 25x smaller than
the residual.  So the residual is capytaine's discretisation, and the finest
rung is inside the 3 % line.

These tests freeze that evidence.  They never import capytaine (optional
dependency): they assert on the frozen artefact, so they also guard against a
future edit that silently rewrites the conclusion.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CASE = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "casepool_bem_hulme_diff_verif"
STUDY = CASE / "independent_route_convergence.json"
CASE_RESULTS = CASE / "results.json"

KA_GRID = ("1.0", "1.5", "2.0")
DOFS = ("surge", "heave")


@pytest.fixture(scope="module")
def study() -> dict:
    return json.loads(STUDY.read_text())


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(CASE_RESULTS.read_text())


def test_study_has_three_rungs_and_monotone_mesh(study: dict) -> None:
    """The ladder must actually refine: faces up, mean edge length down."""
    assert len(study["ladder_resolutions"]) == 3
    for key in KA_GRID:
        rungs = study["per_ka"][key]["rungs"]
        faces = [r["wetted_faces"] for r in rungs]
        hs = [r["h_mean"] for r in rungs]
        assert faces == sorted(faces) and len(set(faces)) == 3, faces
        assert hs == sorted(hs, reverse=True), hs


def test_coarsest_rung_reproduces_case_report_cell(study: dict, results: dict) -> None:
    """The new ladder's 1024-face rung must reproduce the case's own G4 numbers.

    This binds the two artefacts: if either the frozen capytaine ladder or the
    refined study were regenerated with a different convention, these cells
    would drift apart by far more than the 0.1 pp band.
    """
    frozen = {(row["ka"], row["dof"]): row["complex_relative_finest"]
              for row in results["independent"]}
    for key in KA_GRID:
        rung = study["per_ka"][key]["rungs"][0]
        assert rung["wetted_faces"] == 1024
        for dof in DOFS:
            mine = rung["relative"][dof]
            theirs = frozen[(float(key), dof)]
            assert abs(mine - theirs) < 1e-3, (
                f"ka={key} {dof}: refined-study coarsest rung {100 * mine:.3f}% vs "
                f"case results {100 * theirs:.3f}% -- the two artefacts disagree"
            )


def test_ka2_heave_residual_is_discretisation_not_reference(study: dict) -> None:
    """The headline claim: the 4.30 % cell shrinks monotonically and ends < 3 %."""
    rungs = study["per_ka"]["2.0"]["rungs"]
    seq = [r["relative"]["heave"] for r in rungs]
    assert seq[0] > 0.04, seq  # the case's weak number really is ~4.3 %
    assert seq[0] > seq[1] > seq[2], f"non-monotone refinement: {seq}"
    assert seq[-1] < 0.03, f"finest rung still {100 * seq[-1]:.2f}% >= 3%"
    # both components move toward the reference at every rung
    assert study["per_ka"]["2.0"]["extrapolation"]["heave"][
        "parts_monotone_toward_reference"
    ] is True
    # ... and the reference's own truncation error cannot explain the residual
    sens = study["per_ka"]["2.0"]["reference_truncation_sensitivity"]["heave"][
        "max_relative_spread"
    ]
    assert sens < 1e-3 and sens < seq[-1] / 10.0, sens


def test_finest_rung_meets_three_percent_everywhere(study: dict) -> None:
    """No cell of the independent route is above the ecosystem 3 % line anymore."""
    worst = max(
        (r["relative"][dof], key, dof)
        for key in KA_GRID
        for r in study["per_ka"][key]["rungs"][-1:]
        for dof in DOFS
    )
    assert worst[0] < 0.03, f"worst finest-rung deviation {100 * worst[0]:.2f}% at {worst[1:]}"
    assert study["verdicts"]["finest_rung_le_3pct"]["verdict"] == "PASS"


def test_extrapolation_claims_are_guarded(study: dict) -> None:
    """Only genuinely asymptotic cells may claim a Richardson limit.

    ka = 2.0 heave has observed order 0.69 (< 0.8): the guard must refuse to
    extrapolate there rather than publish a misleading number.  The two lower
    ka cells are asymptotic and must land within 1 % of the reference.
    """
    ex = {key: study["per_ka"][key]["extrapolation"]["heave"] for key in KA_GRID}
    assert ex["2.0"]["asymptotic"] is False
    assert ex["2.0"]["limit"] is None
    assert 0.4 < ex["2.0"]["observed_order"] < 0.8
    for key in ("1.0", "1.5"):
        assert ex[key]["asymptotic"] is True
        assert ex[key]["observed_order"] >= 0.8
        assert ex[key]["limit_relative_to_reference"] < 0.01


def test_provenance_is_recorded(study: dict) -> None:
    """Foreign-code evidence is only evidence with provenance."""
    prov = study["provenance"]
    assert prov["capytaine"] and prov["platform"] and prov["generated_utc"]
    assert "diffraction-only" in prov["note"]
    assert study["anchor"] == "CASE-BEM-HULME-DIFF-VERIF"