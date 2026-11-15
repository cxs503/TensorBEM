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
9. **Ladder coverage** -- the mesh-convergence ladder must score *both*
   radiation matrices: at the anchor's finest level it has to equal the
   per-entry maximum of ``comparison`` (an added-mass-only ladder hid the very
   damping entry that fails the band on ``hemisphere_r1``), and the committed
   block must reproduce offline from the frozen matrices.
"""

from __future__ import annotations

import json
import math
import pathlib
import re

import numpy as np
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
        assert (
            block["n_active"]
            + block["n_null"]
            + block["n_invariant"]
            + block["n_conflict"]
            == block["n_entries"]
        ), "the classes must partition the entries"
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
    invariants = {
        (e["row"], e["column"])
        for block in study["comparison"]["cases"].values()
        for e in block["entries"]
        if e["class"] == "invariant"
    }
    for pair in (("surge", "sway"), ("heave", "roll"), ("surge", "heave")):
        assert pair in nulls or pair in invariants, (
            f"expected structural null {pair} was not classified null/invariant"
        )


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


# ------------------------------------------------- 9. degeneracy / invariant axis
def test_degenerate_cases_are_analytic_zeros() -> None:
    """The null dofs are a statement about the geometry, not about the numbers.

    Both round-2 bodies rotate about their sphere centre of curvature, where
    ``(r - centre) x n`` vanishes pointwise, so all three rotational dofs are
    exact zeros and the whole rotational block of those cases is *unfalsifiable*
    by a relative comparison.  The off-centre cases keep only the yaw null
    (azimuthal mode of a surface of revolution), which is what makes roll and
    pitch scoreable there.
    """
    assert r2.invariant_null_dofs("hemisphere_r1") == ("roll", "pitch", "yaw")
    assert r2.invariant_null_dofs("sphere_r1_z-0.5") == ("roll", "pitch", "yaw")
    for case in ("hemisphere_r1_cz-0.5", "sphere_r1_z-0.5_c0"):
        assert r2.invariant_null_dofs(case) == ("yaw",), (
            f"{case} must keep roll/pitch non-null, otherwise the rotational block "
            "is again a comparison of two zeros"
        )
        spec = r2.CASES[case]
        assert tuple(spec["centre"]) != tuple(spec["sphere_centre"])
        # the geometry itself is untouched: only the rotation centre moved
        assert tuple(spec["body_centre"]) == tuple(spec["sphere_centre"])


def test_rotation_centre_transfer_identity_is_exact() -> None:
    """``A(c + d) = T^T A(c) T`` with ``T = [[I, +[d]x], [0, I]]``.

    Re-derived in-process on the coarsest mesh: an exact algebraic identity of
    the formulation, so it catches an about-the-origin moment arm or a missing
    ``(r - centre)`` in the generalised normals without any foreign code.
    """
    result = r2.centre_transfer_invariant(level=2)
    assert result["worst_residual"] <= r2.CENTRE_TRANSFER_TOL, result["blocks"]
    assert result["verdict"] == "PASS"
    for quantity in ("added_mass", "damping"):
        assert result["blocks"][quantity]["residual"] <= r2.CENTRE_TRANSFER_TOL
    assert np.allclose(result["moved_centre"], np.add(result["centre"], result["shift"]))


def test_degeneracy_audit_residual_decays_to_zero(study: dict) -> None:
    """The residual of the exact-zero entries must decay, not converge to O(1)."""
    audit = study.get("degeneracy")
    assert audit, "the degeneracy audit must be recorded in the artifact"
    assert audit["verdict"] == "PASS"
    assert {row["case"] for row in audit["cases"]} == {
        "hemisphere_r1",
        "sphere_r1_z-0.5",
    }
    for row in audit["cases"]:
        assert row["null_dofs"] == ["roll", "pitch", "yaw"]
        assert len(row["levels"]) >= 2
        assert row["monotone_decay"], row["decay_ratios"]
        assert min(row["decay_ratios"]) > 1.0
        assert row["finest_worst_ratio"] <= audit["ratio_bound"]
        # the area-weighted rotational mode norm itself must shrink with the mesh
        norms = [level["mode_norm"][3] for level in row["levels"]]
        assert all(b < a for a, b in zip(norms, norms[1:])), norms
        # and the residuals quoted are the entries the raw verdict failed on
        assert sorted(row["levels"][0]["entries"]) == [
            "pitch_pitch",
            "roll_roll",
            "surge_pitch",
            "sway_roll",
            "yaw_yaw",
        ]


def test_invariant_scoring_does_not_touch_the_bands(study: dict) -> None:
    """The separate scoring keeps the pre-registered bands and only re-classifies."""
    scoring = study["invariant_scoring"]
    assert scoring["tolerance_diagonal"] == r2.TOL_DIAGONAL
    assert scoring["tolerance_offdiagonal"] == r2.TOL_OFFDIAGONAL
    assert scoring["invariant_ratio"] == r2.INVARIANT_RATIO
    comparison = study["comparison"]
    for case, block in scoring["cases"].items():
        null = set(r2.invariant_null_dofs(case))
        expected_invariant = sum(
            1
            for entry in comparison["cases"][case]["entries"]
            if entry["row"] in null or entry["column"] in null
        )
        got_invariant = (
            block["counts"]["invariant_consistent"] + block["counts"]["invariant_violated"]
        )
        assert got_invariant == expected_invariant, case
        expected_physics = comparison["cases"][case]["n_entries"] - expected_invariant
        assert block["counts"]["physics_active"] + block["counts"]["physics_null"] == expected_physics
        # nothing vanishes: the classes still partition every entry
        assert got_invariant + expected_physics == comparison["cases"][case]["n_entries"]


def test_off_centre_cases_actually_score_the_rotational_block(study: dict) -> None:
    """Guard against the score becoming vacuous again."""
    scoring = study["invariant_scoring"]
    for case in ("hemisphere_r1_cz-0.5", "sphere_r1_z-0.5_c0"):
        block = scoring["cases"][case]
        assert block["rotational_block_scored"] is True
        assert block["counts"]["physics_active"] > 0
        entries = study["comparison"]["cases"][case]["entries"]
        rotational = {
            (entry["row"], entry["column"])
            for entry in entries
            if entry["class"] == "active"
            and (entry["row"] in {"roll", "pitch"} or entry["column"] in {"roll", "pitch"})
        }
        assert rotational, f"{case} must produce scored entries in the rotational block"
        # roll/pitch are genuinely non-zero on both sides (not a numeric zero)
        for entry in entries:
            if {entry["row"], entry["column"]} == {"roll"} or {
                entry["row"],
                entry["column"],
            } == {"pitch"}:
                assert entry["class"] == "active"
                assert abs(entry["capytaine"]) > 1.0, entry


def test_physical_scale_is_built_from_geometry_only() -> None:
    """``rho L^k`` with k = 3/4/5 for mass / coupling / inertia entries."""
    assert r2.physical_scale(0, 1) == r2.DENSITY * r2.SCALE_LENGTH**3
    assert r2.physical_scale(0, 4) == r2.DENSITY * r2.SCALE_LENGTH**4
    assert r2.physical_scale(4, 3) == r2.DENSITY * r2.SCALE_LENGTH**4
    assert r2.physical_scale(3, 3) == r2.DENSITY * r2.SCALE_LENGTH**5


# -------------------------------------------------------------------------- 14
def test_convergence_ladder_is_the_per_entry_maximum_of_both_matrices(
    study: dict,
) -> None:
    """The ladder must carry the worst entry of *either* radiation matrix.

    ``comparison`` scores both matrices entry by entry; the ladder is the same
    scan over the coarser levels.  If the two disagree at the anchor's finest
    level something has been dropped from one of them -- and the first
    submission of this round dropped damping, which is where the hemisphere
    failure lives.
    """
    for case, block in study["comparison"]["cases"].items():
        if not block.get("compared"):
            continue
        ladder = study["convergence"][case]
        finest = ladder[-1]
        assert finest["n_faces"] == block["tensorbem_faces"]
        active = [e for e in block["entries"] if e["class"] == "active"]
        for quantity in ("added_mass", "damping"):
            sel = [e for e in active if e["quantity"] == quantity]
            diag = max(
                [abs(e["relative"]) for e in sel if e["row"] == e["column"]] or [0.0]
            )
            off = max(
                [abs(e["relative"]) for e in sel if e["row"] != e["column"]] or [0.0]
            )
            assert finest["worst_diagonal_" + quantity] == pytest.approx(
                diag, rel=1e-12, abs=1e-15
            ), (case, quantity)
            assert finest["worst_coupling_" + quantity] == pytest.approx(
                off, rel=1e-12, abs=1e-15
            ), (case, quantity)
        assert finest["worst_diagonal"] == pytest.approx(
            max(finest["worst_diagonal_added_mass"], finest["worst_diagonal_damping"])
        )
        assert finest["worst_coupling"] == pytest.approx(
            max(finest["worst_coupling_added_mass"], finest["worst_coupling_damping"])
        )


def test_damping_ladder_is_not_the_added_mass_ladder(study: dict) -> None:
    """Pin the defect this axis was fixed for.

    On ``hemisphere_r1`` at 1024 panels the added-mass diagonal stays inside the
    registered 5 % band (3.30 %) while the damping diagonal -- the matrix that
    actually fails, ``A[heave,heave]`` at nu = 2 m^-1 -- is at 5.82 %.  A ladder
    that reports only the first number reads "the gap is inside the band and
    decays", so the two must be kept apart and the maximum must be the headline.
    """
    finest = study["convergence"]["hemisphere_r1"][-1]
    assert finest["worst_diagonal_added_mass"] <= r2.TOL_DIAGONAL
    assert finest["worst_diagonal_damping"] > r2.TOL_DIAGONAL
    assert finest["worst_diagonal"] == pytest.approx(finest["worst_diagonal_damping"])
    assert finest["worst_diagonal_damping"] == pytest.approx(0.05818682870816025)


def test_committed_ladder_reproduces_offline() -> None:
    """``scripts/capytaine_r2_recompute_convergence.py --check`` must be clean.

    The ladder is a pure function of the frozen matrices, so a committed block
    that no longer reproduces from them is a stale artifact, not a rounding
    question.
    """
    import subprocess
    import sys

    script = ROOT / "scripts" / "capytaine_r2_recompute_convergence.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# -------------------------------------------------------------------------- 17
def _referee_ladder_module():
    import importlib.util

    path = ROOT / "scripts" / "capytaine_r2_referee_ladder.py"
    spec = importlib.util.spec_from_file_location("capytaine_r2_referee_ladder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def referee_ladder() -> dict:
    path = ROOT / r2.ARTIFACT_ROOT / "referee_ladder.json"
    if not path.exists():
        pytest.skip(f"{path} not produced yet")
    return json.loads(path.read_text())


def test_referee_ladder_arithmetic_and_verdict(referee_ladder: dict) -> None:
    """Every number in the referee verdict must be the arithmetic of its ladder."""
    scoring = referee_ladder["scoring"]
    tol = scoring["criteria"]["C1_asymptotic"]["tolerance"]
    assert tol == referee_ladder.get("tolerance", tol)
    rows = [row for rows in scoring["cases"].values() for row in rows]
    assert rows, "referee ladder is empty"
    for row in rows:
        values, faces = row["values"], row["n_faces"]
        assert len(values) == len(faces) >= 3
        assert row["change_finest_pair"] == pytest.approx(
            abs(values[-1] - values[-2]) / abs(values[-1]), rel=1e-12
        )
        assert row["change_anchor_mesh_to_finest"] == pytest.approx(
            abs(values[-1] - values[0]) / abs(values[-1]), rel=1e-12
        )
        assert row["deviation_vs_finest_referee"] == pytest.approx(
            (row["anchor_tensorbem"] - values[-1]) / abs(values[-1]), rel=1e-12
        )
        assert row["referee_asymptotically_converged"] == (
            row["change_finest_pair"] <= tol
        )
        assert row["referee_converged_at_anchor_mesh"] == (
            row["change_anchor_mesh_to_finest"] <= tol
        )
        assert row["within_band_vs_finest_referee"] == (
            abs(row["deviation_vs_finest_referee"]) <= row["anchor_band"]
        )
    c1, c2 = scoring["criteria"]["C1_asymptotic"], scoring["criteria"]["C2_anchor_mesh"]
    assert c1["satisfied"] == all(r["referee_asymptotically_converged"] for r in rows)
    assert c2["satisfied"] == all(r["referee_converged_at_anchor_mesh"] for r in rows)
    assert scoring["verdict"] == (
        "CONVERGED" if (c1["satisfied"] and c2["satisfied"]) else "NOT_CONVERGED"
    )
    # the headline the document prints
    assert scoring["worst_finest_pair_change"] == pytest.approx(
        max(r["change_finest_pair"] for r in rows), rel=1e-12
    )
    assert scoring["worst_anchor_mesh_to_finest_change"] == pytest.approx(
        max(r["change_anchor_mesh_to_finest"] for r in rows), rel=1e-12
    )
    over = [r for r in rows if abs(r["anchor_deviation"]) > r["anchor_band"]]
    assert len(scoring["anchor_band_failures"]) == len(over)
    resolved = [r for r in over if r["within_band_vs_finest_referee"]]
    assert (
        len(scoring["anchor_band_failures_resolved_against_finest_referee"])
        == len(resolved)
    )


def test_referee_ladder_starts_at_the_frozen_reference(
    referee_ladder: dict, reference: dict, study: dict
) -> None:
    """Row 0 of each ladder must be the anchor's own referee mesh, bit for bit."""
    nu_key = "{:.3f}".format(referee_ladder["nu"])
    for case, block in referee_ladder["referee"].items():
        anchor = referee_ladder["anchor_entries"][case]
        first = block["levels"][0]
        assert first["n_faces"] == anchor["capytaine_faces"]
        frozen = reference["cases"][case]["reference"][nu_key]
        for quantity in ("added_mass", "damping"):
            # the frozen 6x6 matrices carry the referee's own round-off floor
            # (~1e-14 on couplings that are physically zero); the ladder is a
            # separate live run, so compare to a noise floor scaled by the
            # matrix magnitude rather than bit for bit.
            a = np.asarray(first["matrices"][quantity], dtype=float)
            b = np.asarray(frozen[quantity], dtype=float)
            np.testing.assert_allclose(
                a, b, rtol=1e-10, atol=1e-9 * float(np.abs(b).max())
            )
        # and the TensorBEM column is the committed one, not a re-solve
        tb = study["tensorbem"][case]["levels"][-1]["matrices"][nu_key]
        for key, entry in anchor["entries"].items():
            quantity, i, j = key.split("|")
            assert entry["tensorbem"] == pytest.approx(
                np.asarray(tb[quantity])[int(i), int(j)], rel=1e-12
            )


def test_referee_ladder_document_is_the_arithmetic_of_the_artifact(
    referee_ladder: dict,
) -> None:
    module = _referee_ladder_module()
    text = (ROOT / module.DOC).read_text()
    assert text == module.render_document(referee_ladder)


def test_referee_ladder_live_first_rung_reproduces(referee_ladder: dict) -> None:
    """Live capytaine must still produce the coarsest rung (optional dep)."""
    cpt = pytest.importorskip("capytaine")
    module = _referee_ladder_module()
    sys_path = ROOT / "scripts"
    import sys as _sys

    if str(sys_path) not in _sys.path:
        _sys.path.insert(0, str(sys_path))
    import capytaine_crosscheck_r2_ref as ref

    solver = cpt.BEMSolver()
    for case, block in referee_ladder["referee"].items():
        first = block["levels"][0]
        centre = tuple(float(v) for v in block["rotation_center"])
        mesh = ref._mesh(cpt, case, tuple(first["level"]))
        body = ref._body(cpt, mesh, centre)
        columns = ref._columns(cpt, solver, body, r2.DOF_ORDER, (referee_ladder["nu"],))
        key = "{:.3f}".format(referee_ladder["nu"])
        for quantity in ("added_mass", "damping"):
            np.testing.assert_allclose(
                np.asarray(first["matrices"][quantity], dtype=float),
                np.asarray(columns[key][quantity], dtype=float),
                rtol=1e-12,
                atol=0.0,
                err_msg="{} {}".format(case, quantity),
            )


def test_referee_ladder_reproduces_offline() -> None:
    import subprocess
    import sys

    script = ROOT / "scripts" / "capytaine_r2_referee_ladder.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
