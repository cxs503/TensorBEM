"""Integration tests for CASE-BEM-HULME-HEMISPHERE (m17292).

These are the in-repo callers of :mod:`tensorbem.benchmarks.hulme_hemisphere_case`:
they solve a real 3-D free-surface BEM cell, exercise the predeclared gate
evaluation on both passing and deliberately corrupted input, check the
geometry precondition that must hold before a coefficient table is readable,
and re-derive the committed verdict from the committed raw numbers (so a
hand-edited ``results.json`` cannot survive).

The heavy band sweep is *not* run here (it takes minutes); the committed
artefacts under ``benchmarks/verified/casepool_bem_hulme_hemisphere`` are
re-verified instead, and ``tests`` below re-solves one cheap cell end to end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks import hulme_hemisphere_case as case
from tensorbem.geometry3d import hemisphere_hull_mesh

CASE_DIR = (Path(__file__).resolve().parents[1]
            / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere")


def test_geometry_precondition_holds_on_the_case_mesh():
    """Cb/volume/waterplane must be right before any coefficient is believable."""
    mesh = hemisphere_hull_mesh(case.CASE_SUBDIVISIONS, radius=case.RADIUS)
    nabla = 2.0 / 3.0 * math.pi * case.RADIUS**3
    volume_error = mesh.volume / nabla - 1.0
    waterplane_error = mesh.waterplane_area / (math.pi * case.RADIUS**2) - 1.0
    assert mesh.panel_count == 4 * 4**case.CASE_SUBDIVISIONS
    assert abs(volume_error) < case.GEOM_GATE, volume_error
    assert abs(waterplane_error) < case.GEOM_GATE, waterplane_error
    # the coarse rung is deliberately outside the gate: that is why the case
    # mesh had to be refined rather than the gate relaxed
    coarse = hemisphere_hull_mesh(case.COARSE_SUBDIVISIONS, radius=case.RADIUS)
    assert abs(coarse.volume / nabla - 1.0) > case.GEOM_GATE


def test_hulme_reference_series_is_self_converged():
    """96 -> 144 term drift must stay far below the 5 % case tolerance."""
    for ka in (0.2, 1.0, 2.0):
        ladder = case.reference_ladder(ka)
        drift = ladder["max_rel_delta"]["96->144"]
        assert drift < 0.2 * case.GATE, (ka, drift)
        assert drift < ladder["max_rel_delta"]["48->96"]


def test_run_cell_solves_a_real_cell_and_matches_the_reference():
    """One genuine BEM solve on the cheap rung (the suite's end-to-end probe)."""
    cell = case.run_cell(1.0, subdivisions=2)
    ref = case.reference(1.0)
    for key in case.QUANTITIES:
        assert math.isfinite(cell[key])
        assert abs(cell[f"rel_error_{key}"]) < 0.15, (key, cell[f"rel_error_{key}"])
    # passivity: radiation damping cannot be negative for a passive body
    assert cell["B11_dimless"] > 0.0
    assert cell["B33_dimless"] > 0.0
    assert cell["condition_number"] < 100.0
    assert cell["nabla_exact"] == pytest.approx(ref and 2.0 / 3.0 * math.pi)


def test_evaluate_passes_good_rows_and_rejects_out_of_tolerance_rows():
    """The gate must be able to fail -- otherwise it is decoration."""
    nabla = 2.0 / 3.0 * math.pi * case.RADIUS**3
    ref = case.reference(1.0)
    good = {
        "kind": "band",
        "ka": 1.0,
        "subdivisions": case.CASE_SUBDIVISIONS,
        "panels": 1024,
        "chief": True,
        "mesh_volume": nabla,
        "mesh_waterplane_area": math.pi,
        "nabla_exact": nabla,
        "condition_number": 3.0,
        **{key: ref[key] for key in case.QUANTITIES + case.PHASE_KEYS},
        "reference": {key: ref[key] for key in case.QUANTITIES + case.PHASE_KEYS},
    }
    for key in case.QUANTITIES:
        good[f"rel_error_{key}"] = 0.0
    for key in case.PHASE_KEYS:
        good[f"delta_{key}"] = 0.0
    coarse = json.loads(json.dumps(good))
    coarse.update({
        "subdivisions": case.COARSE_SUBDIVISIONS,
        "panels": 4 * 4**case.COARSE_SUBDIVISIONS,
        "mesh_volume": nabla * 0.911,      # the real 64-panel deficit
        "mesh_waterplane_area": math.pi * 0.98,
    })
    coarse["B33_dimless"] = ref["B33_dimless"] * 1.10
    coarse["rel_error_B33_dimless"] = 0.10
    ref_ladders = {"1.0": case.reference_ladder(1.0)}

    verdict = case.evaluate([good, coarse], ref_ladders)
    assert verdict["verdict"] == "PASS"
    assert verdict["criteria"]["G2_geometry_precondition"]["pass"] is True

    # 1) 10 % coefficient error must break G1
    bad = json.loads(json.dumps(good))
    bad["B33_dimless"] = ref["B33_dimless"] * 1.10
    bad["rel_error_B33_dimless"] = 0.10
    assert case.evaluate([bad, coarse], ref_ladders)["verdict"] == "FAIL"
    # a run without the coarse rung cannot evidence h-convergence -> G3 fails
    partial = case.evaluate([good], ref_ladders)
    assert partial["criteria"]["G3_refinement_does_not_worsen"]["pass"] is False

    # 2) a mesh whose volume is off by 3 % must break the geometry gate even
    #    when every coefficient is spot on
    skewed = json.loads(json.dumps(good))
    skewed["mesh_volume"] = nabla * 1.03
    result = case.evaluate([skewed, coarse], ref_ladders)
    assert result["criteria"]["G2_geometry_precondition"]["pass"] is False
    assert result["verdict"] == "FAIL"


@pytest.mark.skipif(not (CASE_DIR / "results.json").is_file(),
                    reason="case artefacts not generated in this checkout")
def test_committed_artefacts_reproduce_their_verdict():
    """Re-run the gate evaluation on the committed raw numbers."""
    payload = json.loads((CASE_DIR / "results.json").read_text(encoding="utf-8"))
    recomputed = case.evaluate(payload["rows"], payload["ref_ladder"])
    assert recomputed["verdict"] == payload["verdict"]["verdict"]
    for key, crit in recomputed["criteria"].items():
        assert crit["pass"] == payload["verdict"]["criteria"][key]["pass"], key

    # the reference column must be the live series, not a transcribed table
    for ka_key, values in payload["reference"].items():
        fresh = case.reference(float(ka_key))
        for q in case.QUANTITIES + case.PHASE_KEYS:
            assert fresh[q] == pytest.approx(float(values[q]), rel=1e-12, abs=1e-12)


@pytest.mark.skipif(not (CASE_DIR / "evidence.json").is_file(),
                    reason="evidence bundle not generated in this checkout")
def test_evidence_bundle_checks_all_pass():
    bundle = json.loads((CASE_DIR / "evidence.json").read_text(encoding="utf-8"))
    failed = [key for key, value in bundle["checks"].items() if not value]
    assert not failed, failed
    assert bundle["levels"] == {
        "L1_artifact": True,
        "L2_implementation": True,
        "L3_integration": True,
        "L4_benchmark": True,
    }
    assert all(row["exists"] for row in bundle["run_artifacts"])
    assert all(row["verified"] for row in bundle["integration_callers"])