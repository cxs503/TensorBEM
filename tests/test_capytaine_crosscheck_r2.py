"""Acceptance tests for the round-2 capytaine cross-code anchor.

Anchor **XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE**.  The claim under test is narrow
and falsifiable: TensorBEM's **full 6x6** radiation matrix -- rotational block
and translation-rotation couplings included -- agrees with the independent panel
code capytaine to within bands registered *before* the run (5 % diagonal, 10 %
active couplings), and the structural nulls stay null in both codes.

The artifact's *recorded verdict* is whatever the comparison produced -- this
file does not assume PASS.  What it does enforce is that the verdict cannot lie,
that the bands were not relaxed after the fact, and that the prose is the
arithmetic of the committed numbers.

Checks:

1. **Provenance** -- the foreign reference must name tool/version/commit/
   platform and the rotation centre it used; capytaine 3.1 defaults that centre
   to ``nan``, so an unrecorded centre makes the rotational block unfalsifiable.
2. **Coverage** -- all 36 ``(row, column)`` pairs x 2 quantities x 3 frequencies
   per case must be classified exactly once (no entry silently dropped), and the
   classes must partition the set.
3. **Band freeze** -- the tolerances stored in the artifact must equal the
   module constants; a widened band fails here.
4. **Verdict consistency** -- no entry outside its band and no null/non-null
   conflict may be missing from ``failures``; every case verdict must follow.
5. **Structural nulls** -- every entry classified ``null`` must be below the
   floor in *both* codes.
6. **Reproducibility pin** -- re-solving the coarsest level in-process must
   reproduce the committed entry to round-off.
7. **Prose/artifact agreement** -- ``docs/capytaine_crosscheck_r2.md`` must equal
   ``render_document`` of the committed ``results.json``.
8. **Optional-dependency hygiene** -- ``capytaine`` may be imported only by the
   producer script; it must not leak into ``src/`` or ``tests/``.
"""

from __future__ import annotations

import json
import math
import pathlib
import re

import pytest

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / r2.ARTIFACT_ROOT
RESULT = ARTIFACT_DIR / r2.RESULT_FILE
REFERENCE = ARTIFACT_DIR / r2.REFERENCE_FILE


@pytest.fixture(scope="module")
def study() -> dict:
    if not RESULT.exists():
        pytest.skip(f"{RESULT} not produced yet")
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def reference() -> dict:
    if not REFERENCE.exists():
        pytest.skip(f"{REFERENCE} not produced yet")
    return json.loads(REFERENCE.read_text())


# --------------------------------------------------------------------------- 1
def test_reference_provenance_and_rotation_centre(reference: dict) -> None:
    prov = reference["metadata"]
    assert prov["tool"] == "capytaine"
    assert prov["version"], "capytaine version must be recorded"
    assert prov["platform"] and prov["machine"]
    assert prov["density"] == r2.DENSITY and prov["gravity"] == r2.GRAVITY
    assert prov["water_depth"] == "infinite"
    for case, spec in r2.CASES.items():
        block = reference["cases"][case]
        assert block["rotation_center"] == list(spec["centre"]), (
            "the rotation centre used by the referee must be the one TensorBEM uses; "
            "capytaine 3.1 defaults it to nan"
        )
        assert block["reference_faces"] > 0
        assert len(block["levels"]) >= 2, "a single mesh level cannot bound discretisation error"
        assert len(block["dofs"]) == 6


# --------------------------------------------------------------------------- 2
def test_every_entry_is_classified_exactly_once(study: dict) -> None:
    comparison = study["comparison"]
    for case, block in comparison["cases"].items():
        assert block["compared"] is True
        assert block["n_entries"] == 6 * 6 * len(r2.NU_GRID) * 2
        assert block["n_active"] + block["n_null"] + block["n_conflict"] == block["n_entries"]
        seen = {
            (e["nu"], e["quantity"], e["row"], e["column"]) for e in block["entries"]
        }
        assert len(seen) == block["n_entries"], "duplicate entries"
        for quantity in ("added_mass", "damping"):
            for row in r2.DOF_ORDER:
                for column in r2.DOF_ORDER:
                    for nu in r2.NU_GRID:
                        assert (nu, quantity, row, column) in seen, (
                            f"{case}: entry {quantity}[{row},{column}] @ nu={nu} missing"
                        )


# --------------------------------------------------------------------------- 3
def test_bands_are_the_pre_registered_ones(study: dict) -> None:
    comparison = study["comparison"]
    assert comparison["tolerance_diagonal"] == pytest.approx(r2.TOL_DIAGONAL)
    assert comparison["tolerance_offdiagonal"] == pytest.approx(r2.TOL_OFFDIAGONAL)
    assert comparison["null_floor"] == pytest.approx(r2.NULL_FLOOR)
    for block in study["symmetry"].values():
        assert block["tolerance"] == pytest.approx(r2.SYMMETRY_TOL)
    if study.get("deep_submergence"):
        assert study["deep_submergence"]["tolerance"] == pytest.approx(r2.DEEP_TOL)
    assert r2.TOL_DIAGONAL < 0.15 and r2.TOL_OFFDIAGONAL < 0.15, (
        "owner accuracy standard: >15% is unacceptable"
    )


# --------------------------------------------------------------------------- 4
def test_verdict_is_consistent_with_the_entries(study: dict) -> None:
    comparison = study["comparison"]
    failures = comparison["failures"]
    for case, block in comparison["cases"].items():
        case_failures = block["failures"]
        for entry in block["entries"]:
            label = (
                f"{case} nu={entry['nu']:g} {entry['quantity']} "
                f"A[{entry['row']},{entry['column']}]"
            )
            if entry["class"] == "conflict":
                assert any(f.startswith(label) for f in failures), (
                    f"{label}: null/non-null conflict not reported"
                )
            elif entry["class"] == "active":
                assert entry["band"] in (r2.TOL_DIAGONAL, r2.TOL_OFFDIAGONAL)
                if entry["band"] == r2.TOL_DIAGONAL:
                    assert entry["row"] == entry["column"]
                if abs(entry["relative"]) > entry["band"]:
                    assert any(f.startswith(label) for f in failures), (
                        f"{label}: {entry['relative']:+.2%} outside band but not in failures"
                    )
        assert block["verdict"] == ("PASS" if not case_failures else "FAIL")
    assert comparison["verdict"] == ("PASS" if not failures else "FAIL")
    recorded_worst = max(
        (
            b["worst_active_relative"]
            for b in comparison["cases"].values()
            if b.get("compared")
        ),
        default=0.0,
    )
    assert comparison["worst_active_relative"] == pytest.approx(recorded_worst)


# --------------------------------------------------------------------------- 5
def test_structural_nulls_are_null_in_both_codes(study: dict) -> None:
    floor = study["comparison"]["null_floor"]
    n_null = 0
    for block in study["comparison"]["cases"].values():
        for entry in block["entries"]:
            if entry["class"] != "null":
                continue
            n_null += 1
            assert entry["tensorbem_null_ratio"] <= floor
            assert entry["capytaine_null_ratio"] <= floor
    assert n_null > 0, "the symmetry argument predicts structural nulls; none were recorded"
    # the geometries are axisymmetric, so the mirror-forbidden couplings exist
    nulls = {
        (e["row"], e["column"])
        for block in study["comparison"]["cases"].values()
        for e in block["entries"]
        if e["class"] == "null"
    }
    for pair in (("surge", "sway"), ("heave", "roll"), ("surge", "heave")):
        assert pair in nulls, f"expected structural null {pair} was not classified null"


# --------------------------------------------------------------------------- 6
def test_coarsest_level_reproduces_in_process(study: dict) -> None:
    case = "hemisphere_r1"
    level = study["tensorbem"][case]["levels"][0]
    assert level["n_faces"] < 200, "the pin must stay cheap"
    mesh = r2._mesh(case, level["level"])
    nu = r2.NU_GRID[0]
    fresh = r2.solve_matrix(mesh, nu, r2.CASES[case]["centre"])
    committed = level["matrices"][r2._key(nu)]["added_mass"]
    for i in range(6):
        for j in range(6):
            base = committed[i][j]
            tol = 1.0e-9 * max(1.0, abs(base))
            assert abs(fresh["added_mass"][i][j] - base) <= tol, (
                f"committed A[{i},{j}] drifted from the code"
            )


# --------------------------------------------------------------------------- 7
def test_document_is_the_arithmetic_of_the_artifact(study: dict) -> None:
    doc = r2.DOC_PATH
    assert doc.exists(), f"{doc} missing"
    assert doc.read_text() == r2.render_document(study)
    text = doc.read_text()
    worst = study["comparison"]["worst_active_relative"] * 100.0
    assert f"{worst:.2f}%" in text, "worst active deviation not printed in the document"
    for case, block in study["comparison"]["cases"].items():
        assert case in text
        assert str(block["tensorbem_faces"]) in text


# --------------------------------------------------------------------------- 8
def test_capytaine_does_not_leak_into_src_or_tests() -> None:
    """Module-scope imports only: a ``find_spec``-guarded lazy import is legal."""
    pattern = re.compile(r"^(?:import|from)\s+capytaine", re.MULTILINE)
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if pattern.search(path.read_text(errors="ignore")):
            offenders.append(str(path))
    for path in (ROOT / "tests").rglob("*.py"):
        if pattern.search(path.read_text(errors="ignore")):
            offenders.append(str(path))
    assert not offenders, f"capytaine imported outside the producer: {offenders}"


# --------------------------------------------------------------------------- 9
def test_symmetry_invariants_are_recorded_for_both_codes(study: dict) -> None:
    assert study["symmetry"], "symmetry invariants missing"
    for case, block in study["symmetry"].items():
        assert len(block["rows"]) == len(r2.NU_GRID)
        worst = 0.0
        for row in block["rows"]:
            assert row["capytaine_surge_sway"] is not None, (
                "the referee's own anisotropy must be reported, not just ours"
            )
            worst = max(
                worst,
                row["tensorbem_surge_sway"],
                row["tensorbem_roll_pitch"],
                row["tensorbem_reciprocity"],
            )
        assert block["worst"] == pytest.approx(worst)
        assert block["verdict"] == ("PASS" if worst <= block["tolerance"] else "FAIL")


# -------------------------------------------------------------------------- 10
def test_deep_submergence_is_the_analytic_axis(study: dict) -> None:
    """The submerged sphere carries the only *analytic* axis of this round.

    (a) ``A_roll`` about the sphere centre is **exactly zero** (the generalised
        normal ``(r - centre) x n`` vanishes on a sphere), whatever the depth,
        frequency or panel count.  The artifact must show only the discretisation
        floor of that machinery, and both codes must agree that it is a floor.
    (b) ``A_surge`` must approach the Lamb (1932) value ``(2/3) pi rho R^3``.
    """
    deep = study.get("deep_submergence")
    if not deep:
        pytest.skip("deep submergence study not run")
    assert [row["depth"] for row in deep["rows"]] == list(r2.DEEP_DEPTHS)
    assert deep["lamb_surge"] == pytest.approx((2.0 / 3.0) * math.pi * r2.DENSITY)
    assert deep["tolerance"] == pytest.approx(r2.DEEP_TOL)
    assert deep["tolerance_rotation"] == pytest.approx(r2.EXACT_NULL_TOL)
    assert len(deep["levels"]) >= 1, "the deep axis must name the meshes it used"

    worst_rot = 0.0
    faces = set()
    for row in deep["rows"]:
        for values in row["values"].values():
            faces.add(values["n_faces"])
            assert values["surge_vs_lamb"] == pytest.approx(
                (values["surge_added_mass"] - deep["lamb_surge"]) / deep["lamb_surge"]
            )
            assert values["roll_over_scale"] == pytest.approx(
                abs(values["roll_added_inertia"]) / abs(values["surge_added_mass"])
            )
            worst_rot = max(worst_rot, values["roll_over_scale"])
        if "capytaine_roll" in row:
            assert row["capytaine_surge_vs_lamb"] == pytest.approx(
                (row["capytaine_surge"] - deep["lamb_surge"]) / deep["lamb_surge"]
            )
            assert row["capytaine_roll_over_scale"] == pytest.approx(
                abs(row["capytaine_roll"]) / abs(row["capytaine_surge"])
            )
            assert row["capytaine_roll_over_scale"] <= r2.EXACT_NULL_TOL, (
                "the referee must also see the exact rotational null"
            )
    assert len(faces) >= 1
    assert deep["worst_rotation_ratio"] == pytest.approx(worst_rot)
    assert worst_rot <= r2.EXACT_NULL_TOL, (
        "the sphere's roll added inertia is exactly zero; the residual is the "
        "discretisation floor of the rotational normals"
    )
    deepest = deep["rows"][-1]
    finest = max(deepest["values"].values(), key=lambda v: v["n_faces"])
    assert deep["deepest_finest_vs_lamb"] == pytest.approx(finest["surge_vs_lamb"])
    assert deep["deepest_finest_faces"] == finest["n_faces"]
    expected = "PASS" if (
        abs(finest["surge_vs_lamb"]) <= deep["tolerance"]
        and worst_rot <= deep["tolerance_rotation"]
    ) else "FAIL"
    assert deep["verdict"] == expected


def test_document_states_the_analytic_axes(study: dict) -> None:
    text = r2.DOC_PATH.read_text()
    deep = study.get("deep_submergence")
    if not deep:
        pytest.skip("deep submergence study not run")
    assert "精确为零" in text, "the exact rotational null must be stated in prose"
    assert "{:.2f} kg".format(deep["lamb_surge"]) in text
    assert deep["verdict"] in text
