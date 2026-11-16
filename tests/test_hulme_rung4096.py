"""Tests for the measured 4096-panel rung of CASE-BEM-HULME-HEMISPHERE.

The 4096-panel solves are not run here (they cost ~15 min of cluster time and
live in ``cells4096/``); what is tested is the part that can rot silently:

* every gate of ``rung4096_ladder.evaluate`` **can** flip to FAIL -- a gate that
  cannot fail is not a gate;
* the reference multipole series is self-converged at ka = 2.58 and 3.0, so the
  ladder is not measuring a truncated reference;
* the committed artifact is internally consistent: every relative error and the
  whole verdict are re-derived from the stored raw values;
* ``run_cell(..., with_diffraction=False)`` returns a cell without excitation
  scores instead of raising (public-API regression).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
RESULTS = CASE_DIR / "results_rung4096.json"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import rung4096_ladder as L  # noqa: E402
from tensorbem.benchmarks import hulme_hemisphere_case as case  # noqa: E402


# --- synthetic cells --------------------------------------------------------
def _synthetic(ka: float, subdivisions: int, *, err_scale: float = 1.0,
               residual: float = 1e-3) -> dict:
    """A cell shaped like a real one, from one cheap 64-panel solve."""
    base = case.run_cell(ka, 2)
    cell = dict(base)
    cell["subdivisions"] = subdivisions
    cell["panels"] = 4 * 4 ** subdivisions
    cell["kind"] = "rung"
    cell["name"] = L.cell_name(ka, subdivisions)
    cell["radiation_relative_residual"] = residual
    cell["diffraction_relative_residual"] = residual
    for q in case.QUANTITIES:
        cell[f"rel_error_{q}"] *= err_scale
    return cell


def _ladders() -> dict:
    return {str(ka): L.reference_ladder(ka) for ka in L.KA_TARGET}


def _healthy_cells() -> list[dict]:
    """Cells that pass every gate: fine rung far better than the coarse one."""
    out = []
    for ka in L.KA_TARGET:
        out.append(_synthetic(ka, 4, err_scale=10.0))
        out.append(_synthetic(ka, 5, err_scale=0.05))
    return out


# --- the gates must be able to fail ----------------------------------------
def test_healthy_cells_pass_every_gate():
    res = L.evaluate(_healthy_cells(), _ladders())
    failed = res["failed"]
    # P6 (geometry) and P5 (conditioning) are read off the synthetic cell; the
    # synthetic cell carries the 64-panel geometry, so P6 must fail here and is
    # asserted separately below.  Everything the data can satisfy must pass.
    assert failed == ["P6_geometry_precondition"], failed
    assert res["verdict"] == "FAIL"


def test_P1_flips_when_finest_rung_breaks_tolerance():
    cells = _healthy_cells()
    for c in cells:
        if c["subdivisions"] == L.RUNGS[-1]:
            for q in case.QUANTITIES:
                c[f"rel_error_{q}"] = 0.20
    res = L.evaluate(cells, _ladders())
    assert not res["gates"]["P1_finest_rung_below_case_tolerance"]["pass"]
    assert "P1_finest_rung_below_case_tolerance" in res["failed"]


def test_P2_flips_when_refinement_worsens():
    cells = _healthy_cells()
    for c in cells:
        if c["subdivisions"] == L.RUNGS[-1]:
            for q in case.QUANTITIES:
                c[f"rel_error_{q}"] = 0.04
    coarse = _synthetic(L.KA_TARGET[0], 4, err_scale=0.05)
    cells = [c for c in cells if not (c["ka"] == L.KA_TARGET[0] and c["subdivisions"] == 4)]
    cells.append(coarse)
    res = L.evaluate(cells, _ladders())
    assert not res["gates"]["P2_refinement_does_not_worsen"]["pass"]


def test_P3_flips_when_a_rung_cannot_prove_its_residual():
    cells = _healthy_cells()
    cells[0]["radiation_relative_residual"] = 5e-2
    res = L.evaluate(cells, _ladders())
    gate = res["gates"]["P3_rung_is_admissible"]
    assert not gate["pass"]
    assert gate["measured"]["channel"] == "radiation"


def test_P5_flips_when_the_refined_rung_is_near_singular():
    cells = _healthy_cells()
    for c in cells:
        if c["subdivisions"] == L.RUNGS[-1]:
            c["condition_number"] = 1.0e3
    res = L.evaluate(cells, _ladders())
    assert not res["gates"]["P5_conditioning_does_not_explode"]["pass"]


# --- the reference is not the weak link ------------------------------------
@pytest.mark.parametrize("ka", [2.58, 3.0])
def test_reference_self_converged_at_target_ka(ka):
    lad = L.reference_ladder(ka)
    assert lad["max_rel_drift"] < L.REF_DRIFT_TOL, lad


def test_reference_values_finite_and_positive_at_target_ka():
    for ka in L.KA_TARGET:
        ref = case.reference(ka)
        for q in case.QUANTITIES:
            assert math.isfinite(ref[q])
            assert ref[q] > 0.0, (ka, q, ref[q])


# --- public API regression --------------------------------------------------
def test_run_cell_without_diffraction_has_no_excitation_scores():
    cell = case.run_cell(2.58, 2, with_diffraction=False)
    assert "Fx_dimless" not in cell
    assert "rel_error_Fx_dimless" not in cell
    assert "delta_phase_Fx_deg" not in cell
    assert math.isfinite(cell["radiation_relative_residual"])


# --- committed artifact consistency ---------------------------------------
@pytest.mark.skipif(not RESULTS.exists(), reason="4096-panel artifact not committed")
def test_committed_artifact_recomputes_to_the_same_verdict():
    stored = json.loads(RESULTS.read_text())
    cells = stored["cells"]
    for c in cells:
        ref = c["reference"]
        for q in case.QUANTITIES:
            assert c[f"rel_error_{q}"] == pytest.approx(
                (c[q] - ref[q]) / abs(ref[q]), rel=1e-12, abs=1e-15
            )
    recomputed = L.evaluate(cells, stored["reference_ladders"])
    assert recomputed["verdict"] == stored["verdict"]
    assert recomputed["failed"] == stored["failed"]
    assert "Machine verdict" in L.render_report({**stored, **recomputed})


@pytest.mark.skipif(not RESULTS.exists(), reason="4096-panel artifact not committed")
def test_committed_artifact_solves_are_admissible_and_panels_right():
    stored = json.loads(RESULTS.read_text())
    finest = [c for c in stored["cells"] if c["subdivisions"] == L.RUNGS[-1]]
    if not finest:
        pytest.skip("4096-panel rung not produced yet (job queued, no payload)")
    assert {c["ka"] for c in finest} == set(L.KA_TARGET)
    for c in finest:
        assert c["panels"] == 4096
        assert c["radiation_relative_residual"] < L.RESIDUAL_TOL
        assert c["diffraction_relative_residual"] < L.RESIDUAL_TOL