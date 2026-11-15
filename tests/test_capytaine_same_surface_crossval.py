"""Acceptance tests for the same-surface capytaine cross-check.

Anchor ``XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE``.  The round-2 study compared
TensorBEM's finest *own* mesh against capytaine's finest own mesh, so its
residual 6.7 % is the sum of two unrelated discretisations.  This artifact
removes the confound: TensorBEM is solved on **capytaine's own panel surface**,
which makes the reported deviation attributable to the formulation.

What these tests enforce (the artifact's verdict is whatever the arithmetic
produced -- nothing here assumes PASS):

1. **Provenance** -- the capytaine column is the committed round-2 reference at
   the same resolution levels, entry for entry; the producer records its own
   capytaine version/platform.
2. **Surface identity** -- the quad-to-triangle conversion must cover exactly
   capytaine's area (relative difference to round-off) and the stored
   triangles must rebuild a valid :class:`WettedHullMesh` (outward normals,
   closed waterline); the tessellation must approach the analytic wetted area
   from below as it refines.
3. **Classification** -- every ``quantity x row x column`` entry is classified
   exactly once, nulls and invariant zeros are not scored as relative
   deviations, and the bands are the frozen module constants.
4. **Refinement signature** -- the same-surface deviation must shrink from the
   coarsest to the finest level for every case.  A formulation mismatch (wrong
   generalised normal, wrong rotation centre) would *not* shrink.
5. **Reproducibility pin** -- re-solving the coarsest stored surface in-process
   reproduces the committed TensorBEM column to round-off.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import WettedHullMesh

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "benchmarks/verified/capytaine_same_surface_crossval"
RESULT = ARTIFACT_DIR / "results.json"
REFERENCE = ROOT / r2.ARTIFACT_ROOT / r2.REFERENCE_FILE

#: Ledger acceptance for ``XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE``.
LEDGER_BAND = 0.15


@pytest.fixture(scope="module")
def study() -> dict:
    if not RESULT.exists():
        pytest.skip(f"{RESULT} not produced yet")
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE.read_text())


def _analytic_area(case: str) -> float:
    """Analytic wetted area of the two bodies (radius 1, unit depth scale)."""
    if case.startswith("hemisphere"):
        return 2.0 * math.pi * 1.0**2
    # floating sphere of radius 1 with centre z = -0.5: cap height 0.5
    return 2.0 * math.pi * 1.0 * 0.5


# --------------------------------------------------------------------------- 1
def test_provenance_names_the_referee(study: dict) -> None:
    prov = study["provenance"]
    assert prov["tool"] == "capytaine"
    assert prov["version"], "capytaine version must be recorded"
    assert prov["machine"] and prov["platform"]
    assert prov["density"] == r2.DENSITY and prov["gravity"] == r2.GRAVITY
    assert tuple(prov["nu_grid"]) == tuple(r2.NU_GRID)
    assert prov["water_depth"] == "infinite"


def test_capytaine_column_is_the_committed_reference(study: dict, reference: dict) -> None:
    """Every scored entry must quote the frozen round-2 reference verbatim."""
    for case, block in study["cases"].items():
        # the capytaine column of a centre variant is the same body's reference
        ref_case = block.get("ref_case") or (
            "sphere_r1_z-0.5" if case.startswith("sphere") else "hemisphere_r1"
        )
        for row in block["levels"]:
            level = tuple(int(v) for v in row["level"])
            ref_level = None
            for candidate in reference["cases"][ref_case]["levels"]:
                if tuple(int(v) for v in candidate["level"]) == level:
                    ref_level = candidate["matrices"]
                    break
            assert ref_level is not None, f"{case}: level {level} missing in reference"
            for key, score in row["scores"].items():
                a_cap = np.asarray(ref_level[key]["added_mass"], dtype=float)
                b_cap = np.asarray(ref_level[key]["damping"], dtype=float)
                for entry in score["entries"]:
                    i = r2.DOF_ORDER.index(entry["row"])
                    j = r2.DOF_ORDER.index(entry["column"])
                    matrix = a_cap if entry["quantity"] == "added_mass" else b_cap
                    assert entry["capytaine"] == pytest.approx(
                        float(matrix[i, j]), rel=0.0, abs=0.0
                    ), (
                        f"{case} level {level} nu={key} "
                        f"{entry['quantity']}[{entry['row']},{entry['column']}] "
                        "does not quote the committed reference"
                    )


# --------------------------------------------------------------------------- 2
def test_quad_split_conserves_the_capytaine_surface(study: dict) -> None:
    for case, block in study["cases"].items():
        for row in block["levels"]:
            assert row["area_relative_difference"] <= 1e-12, (
                f"{case} level {row['level']}: the triangle split changed the area"
            )
            assert row["triangles"] >= row["n_faces"]
            assert row["triangles"] <= 2 * row["n_faces"]


def test_stored_surface_rebuilds_a_valid_wetted_hull(study: dict) -> None:
    """The stored triangles must satisfy the hull invariants, not just exist."""
    for case, block in study["cases"].items():
        for row in block["levels"]:
            hull = WettedHullMesh(
                np.asarray(row["vertices"], dtype=float),
                np.asarray(row["faces"], dtype=np.int64),
            )
            assert hull.faces.shape[0] == row["triangles"]
            verts = np.asarray(row["vertices"], dtype=float)
            assert verts[:, 2].max() <= 1e-12, f"{case}: wetted surface above z = 0"


def test_tessellated_area_approaches_the_analytic_cap(study: dict) -> None:
    for case, block in study["cases"].items():
        analytic = _analytic_area(case)
        areas = [row["capytaine_area"] for row in block["levels"]]
        assert all(a <= analytic * (1.0 + 1e-9) for a in areas), (
            f"{case}: inscribed tessellation exceeded the analytic area"
        )
        assert areas[-1] > areas[0], f"{case}: refinement did not grow the area"
        # a coarse inscribed tessellation is a few per cent short of the analytic
        # cap by construction; the bound only has to catch a broken conversion
        assert abs(areas[-1] - analytic) / analytic <= 0.15, (
            f"{case}: finest tessellated area {areas[-1]:.4f} is not within 15 % of "
            f"the analytic {analytic:.4f}"
        )


# --------------------------------------------------------------------------- 3
def test_every_entry_is_classified_exactly_once(study: dict) -> None:
    for case, block in study["cases"].items():
        for row in block["levels"]:
            for key, score in row["scores"].items():
                assert score["n_entries"] == 6 * 6 * 2
                assert (
                    score["n_active"]
                    + score["n_null"]
                    + score["n_invariant"]
                    + score["n_conflict"]
                    + score["n_unscorable"]
                    == score["n_entries"]
                ), "the classes must partition the entries"
                seen = {
                    (e["quantity"], e["row"], e["column"]) for e in score["entries"]
                }
                assert len(seen) == score["n_entries"], "duplicate entries"
                for entry in score["entries"]:
                    if entry["class"] == "active":
                        assert entry["band"] == (
                            r2.TOL_DIAGONAL
                            if entry["row"] == entry["column"]
                            else r2.TOL_OFFDIAGONAL
                        )


def test_bands_are_the_frozen_constants(study: dict) -> None:
    assert study["tolerance_diagonal"] == pytest.approx(r2.TOL_DIAGONAL)
    assert study["tolerance_offdiagonal"] == pytest.approx(r2.TOL_OFFDIAGONAL)
    assert study["null_floor"] == pytest.approx(r2.NULL_FLOOR)
    for case, block in study["cases"].items():
        for row in block["levels"]:
            for entry in row["scores"]["1.000"]["entries"]:
                if entry["class"] == "active":
                    assert entry["within_band"] == (
                        abs(entry["relative"]) <= entry["band"]
                    )


# --------------------------------------------------------------------------- 4
def test_deviation_shrinks_with_refinement(study: dict) -> None:
    """The signature that separates discretisation error from a formulation bug."""
    for case, block in study["summary"].items():
        coarsest = block["worst_active_coarsest"]
        finest = block["worst_active_finest"]
        assert finest < coarsest, (
            f"{case}: same-surface deviation did not shrink under refinement "
            f"({coarsest:.3%} -> {finest:.3%}); a formulation mismatch does not "
            "converge away"
        )
        assert block["ratio_coarsest_over_finest"] > 1.0


def test_finest_same_surface_deviation_meets_the_ledger_band(study: dict) -> None:
    for case, block in study["summary"].items():
        assert block["worst_active_finest"] <= LEDGER_BAND, (
            f"{case}: {block['worst_active_finest']:.3%} exceeds the ledger band "
            f"{LEDGER_BAND:.0%}"
        )


# --------------------------------------------------------------------------- 5
def test_coarsest_surface_reproduces_in_process(study: dict) -> None:
    case = "hemisphere_r1"
    row = study["cases"][case]["levels"][0]
    hull = WettedHullMesh(
        np.asarray(row["vertices"], dtype=float),
        np.asarray(row["faces"], dtype=np.int64),
    )
    key = f"{r2.NU_GRID[0]:.3f}"
    stored = np.asarray(row["matrices"][key]["tensorbem_added_mass"], dtype=float)
    res = solve_radiation_3d(
        hull,
        r2.NU_GRID[0],
        modes=list(range(6)),
        center=tuple(study["cases"][case]["rotation_center"]),
        density=r2.DENSITY,
        gravity=r2.GRAVITY,
    )
    live = np.asarray(res.added_mass, dtype=float)
    scale = np.abs(stored).max()
    assert np.abs(live - stored).max() / scale <= 1e-10, (
        "the committed TensorBEM column does not reproduce in-process"
    )