"""CASE-BEM-HULME-HEMISPHERE: the 4096-panel rung at the band edge.

The case band stops at ``ka = 2``.  ``band_current_code/`` shows ``2 < ka <= 3.5``
inside the 5 % gate on the 1024-panel case mesh and ``ka = 4.0`` outside it
(8.94 %), with a 256-panel twin at 34.49 %.  Two rungs cannot distinguish a
discretisation limit from a method limit, so
``band_edge_refinement.py`` was predeclared to add the missing 4096-panel rung
at ``ka = 3.5`` and ``ka = 4.0``.

This module asserts the properties the artifact has to have -- it does not
re-solve anything, so it is cheap:

* every cell is machine-provenanced and records its own solve-stage residual;
* the 1024-panel rungs reproduce ``band_current_code/`` bit for bit (R1);
* the ``ka = 4.0`` error falls from 1024 to 4096 panels (R2);
* the ladder is monotone at every ``ka`` where it has three rungs;
* the credited-band statement never widens beyond what the artifact shows,
  i.e. a ``ka`` may only be reported as passing at the panels it actually
  passes at.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
REFINE = CASE / "band_edge_refinement" / "results.json"
CURRENT = CASE / "band_current_code" / "results.json"
COMMITTED = CASE / "band_extension" / "results.json"

GATE = 0.05
QUANTITIES = ("A11", "B11", "A33", "B33", "Fx", "Fz")
EXPECTED_CELLS = {"ka3.5_sub4", "ka3.5_sub5", "ka4_sub4", "ka4_sub5"}
PANELS = {"ka3.5_sub4": 1024, "ka4_sub4": 1024,
          "ka3.5_sub5": 4096, "ka4_sub5": 4096}


def band_max(cell: dict) -> float:
    keys = [k for k in cell if k.startswith("rel_error_")]
    assert keys, f"cell {cell.get('id')} carries no rel_error_* key"
    return max(abs(cell[k]) for k in keys)


@pytest.fixture(scope="module")
def refine() -> dict:
    if not REFINE.exists():
        pytest.fail(f"missing artifact {REFINE} -- the 4096-panel rung was never run")
    return json.loads(REFINE.read_text())


@pytest.fixture(scope="module")
def cells(refine) -> dict[str, dict]:
    return {c["id"]: c for c in refine["cells"]}


# --- provenance / completeness --------------------------------------------
def test_artifact_is_machine_provenanced(refine):
    """Host, numpy and per-module sha256 are mandatory; a git head is not.

    The cells run from a staged copy that carries no ``.git``, so the driver
    records ``git_commit: null``.  Code identity then rests on the per-module
    sha256, which is the stronger pin, and on gate R1 (bit-for-bit reproduction
    of the committed 1024-panel cells).
    """
    meta = refine["metadata"]
    assert meta["host"], "artifact names no host"
    assert meta["numpy"], "artifact records no numpy version"
    assert meta["source_sha256"] and all(
        len(v) == 64 for v in meta["source_sha256"].values()
    ), "per-module sha256 missing"
    assert set(meta["thread_env"]) >= {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
    if meta["git_commit"] is None:
        assert meta["host"] and meta["source_sha256"], (
            "an unprovenanced artifact is not evidence")


def test_cross_environment_rungs_are_pinned_by_r1(refine):
    """The 4096 cells ran in a *different* environment than the case band.

    That is only admissible because the 1024-panel control cells of the same
    artifact reproduce the committed case-band numbers exactly (R1).  If the
    two environments ever disagreed there, the ladder would compare two
    different solvers and this module must fail instead of reporting a
    convergence order.
    """
    meta = refine["metadata"]
    committed = json.loads(CURRENT.read_text())["metadata"]
    if meta["numpy"] == committed["numpy"]:
        pytest.skip("same numpy in both environments; cross-env claim not exercised")
    assert refine["criteria"]["R1_ladder_is_self_consistent"]["pass"] is True
    for cid, cell in {c["id"]: c for c in refine["cells"]}.items():
        if cell["panels"] == 1024:
            assert cell["radiation_relative_residual"] < 1e-2, cid


def test_all_predeclared_cells_returned(refine, cells):
    assert set(cells) == EXPECTED_CELLS, f"cell set drifted: {sorted(cells)}"
    assert refine["missing_cells"] == [], (
        "the artifact declares missing cells; a refinement verdict on a partial "
        f"ladder is not evidence: {refine['missing_cells']}"
    )
    assert refine["absence_reason"] is None


def test_every_cell_records_its_own_solve_stage_output(cells):
    """E0 admissibility: a wrong answer from an unsolved system must be visible."""
    for cid, cell in cells.items():
        res = cell["radiation_relative_residual"]
        assert res is not None, f"{cid} records no radiation residual"
        assert res < 1e-2, f"{cid} radiation residual {res:.3g} -- system not solved"
        assert cell["diffraction_relative_residual"] < 1e-2, cid
        assert cell["radiation_stationarity"] < 1e-10, cid
        assert cell["diffraction_stationarity"] < 1e-10, cid
        assert cell["condition_number"] < 100.0, f"{cid} cond={cell['condition_number']}"
        # Warnings are admissible only if they are the *audited recovery* from the
        # numpy lstsq stationarity defect (m20910): the solution must then come
        # from the fallback path and the audit value must still be tiny.
        for warning in cell["solve_warnings"]:
            assert "stationarity audit" in warning, f"{cid}: unaudited warning {warning!r}"
            assert "lstsq" not in (cell["radiation_solve_method"] or ""), (
                f"{cid} reports a failed lstsq audit but still claims solve method "
                f"{cell['radiation_solve_method']!r}")


def test_geometry_preflight_on_every_rung(cells):
    """A mesh-generation regression must not read as a hydrodynamics result."""
    for cid, cell in cells.items():
        vol_err = abs(cell["mesh_volume"] / cell["nabla_exact"] - 1.0)
        wp_err = abs(cell["mesh_waterplane_area"] / (3.141592653589793) - 1.0)
        assert vol_err < 0.02, f"{cid} volume error {vol_err:.3%}"
        assert wp_err < 0.02, f"{cid} waterplane error {wp_err:.3%}"


# --- R1: the 1024 rungs are a bit-for-bit reproduction --------------------
def test_r1_1024_rungs_reproduce_band_current_code(refine, cells):
    crit = refine["criteria"]["R1_ladder_is_self_consistent"]
    assert crit["pass"] is True
    current = json.loads(CURRENT.read_text())
    ext = {c["ka"]: c for c in current["cells"] if c["kind"] == "extension"}
    for cid, cell in cells.items():
        if cell["panels"] != 1024:
            continue
        ref_cell = ext[cell["ka"]]
        for q in QUANTITIES:
            key = f"rel_error_{q}_dimless"
            assert cell[key] == ref_cell[key], (
                f"{cid} {q}: {cell[key]!r} != committed {ref_cell[key]!r}; the "
                "ladder is not comparable with the case band"
            )


# --- R2: refinement has to move the error ---------------------------------
def test_r2_ka4_error_falls_from_1024_to_4096(refine, cells):
    crit = refine["criteria"]["R2_ka4_error_falls_under_refinement"]
    assert crit["pass"] is True, crit["measured"]
    e4 = band_max(cells["ka4_sub4"])
    e5 = band_max(cells["ka4_sub5"])
    assert e5 < e4, f"ka=4 band-max rose under refinement: {e4:.4f} -> {e5:.4f}"


def test_ladder_is_monotone_where_three_rungs_exist(cells):
    committed = json.loads(COMMITTED.read_text())
    coarse = {c["ka"]: c for c in committed["cells"] if c["kind"] == "ladder_coarse"}
    ladders: dict[float, dict[int, float]] = {}
    for cell in cells.values():
        ladders.setdefault(cell["ka"], {})[cell["panels"]] = band_max(cell)
    for ka, rung in ladders.items():
        if ka in coarse:
            rung[256] = band_max(coarse[ka])
        seq = [rung[n] for n in sorted(rung)]
        assert all(b < a for a, b in zip(seq, seq[1:])), (
            f"ka={ka:g} ladder not monotone: {seq}")


def test_convergence_order_is_positive(refine, cells):
    """Reporting only: a negative order would void the discretisation reading."""
    e4 = band_max(cells["ka4_sub4"])
    e5 = band_max(cells["ka4_sub5"])
    order = math.log(e4 / e5) / math.log(4096 / 1024)
    assert order > 0.0, f"non-converging ladder, panel exponent {order:.2f}"


# --- never overclaim the band --------------------------------------------
def test_credited_band_statement_matches_the_artifact(refine, cells):
    crit = refine["criteria"]["R3_credited_band_edge_localised"]
    obs = crit["measured"]
    ka4_case_mesh = band_max(cells["ka4_sub4"])
    ka4_fine = band_max(cells["ka4_sub5"])
    # credited on the case mesh iff it passes there
    assert (4.0 in obs["credited_ka_on_case_mesh"]) is (ka4_case_mesh < GATE), obs
    # the mesh at which the first failing ka would pass is the finest one that
    # actually passes, or None
    if ka4_fine < GATE:
        assert obs["first_failing_ka_passes_at_panels"] == 4096, obs
    else:
        assert obs["first_failing_ka_passes_at_panels"] is None, obs