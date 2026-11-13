"""Tests for case-pool anchor ``CASE-BEM-HULME-HEMISPHERE`` (floating hemisphere).

These tests pin the *contract* of the case, not just its execution:

* the declared case mesh satisfies the geometry precondition (G2) -- the
  property that made mission 17337 move the case mesh from 256 to 1024 panels;
* the reference is physically admissible and self-converged (G4);
* one cheap solver cell reproduces the reference within a loose band;
* the gate evaluator actually fails a tampered number (a gate that cannot fail
  is not a gate);
* the anchor registry dispatches the case and the historical driver path still
  works;
* the committed evidence bundle is self-consistent (checksums + verdict), which
  is what the L2 "run_artifacts" leg means.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks import case_pool, hulme_hemisphere_case as case
from tensorbem.geometry3d import hemisphere_hull_mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere" / "evidence"
DRIVER = REPO_ROOT / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere" / "run_case.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("casepool_hulme_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- geometry / reference --------------------------------------------------
def test_case_mesh_satisfies_the_geometry_precondition():
    mesh = hemisphere_hull_mesh(case.CASE_SUBDIVISIONS, radius=case.RADIUS)
    assert mesh.panel_count == case.panel_count(case.CASE_SUBDIVISIONS) == 1024
    volume_error = abs(mesh.volume / case.VOLUME - 1.0)
    waterplane_error = abs(mesh.waterplane_area / case.WATERPLANE - 1.0)
    assert volume_error < case.GEOM_GATE
    assert waterplane_error < case.GEOM_GATE


def test_coarse_mesh_violates_the_geometry_precondition():
    """Why the case mesh is 1024 and not 256 panels (regression pin)."""
    mesh = hemisphere_hull_mesh(case.COARSE_SUBDIVISIONS, radius=case.RADIUS)
    assert abs(mesh.volume / case.VOLUME - 1.0) > case.GEOM_GATE


def test_reference_truncation_is_converged():
    ladder = case.reference_ladder(1.0)
    assert ladder["max_rel_delta"]["96->144"] < case.GATE / case.REF_MARGIN
    assert ladder["tail_ratio_A33"] < 1.0  # the series is shrinking


def test_reference_is_physically_admissible():
    for ka in (0.2, 1.0, 2.0):
        ref = case.hulme_reference(ka)
        for key in ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless"):
            assert ref[key] > 0.0
        assert ref["Fx_dimless"] > 0.0 and ref["Fz_dimless"] > 0.0
    # ka -> 0 heave added mass approaches the rigid-lid even-mode value of the
    # double body (0.830954, see benchmarks/verified/bem_rigid_lid_hemisphere).
    low = case.hulme_reference(1e-3)
    assert abs(low["A33_dimless"] - 0.830954) / 0.830954 < 0.02


# --- solver cell -----------------------------------------------------------
def test_coarse_cell_reproduces_the_reference_within_a_loose_band():
    cell = case.panel_cell(1.0, 2)  # 64 panels, ~1 s
    assert cell["panels"] == 64
    assert math.isfinite(cell["condition_number"])
    for quantity in case.QUANTITIES:
        assert math.isfinite(cell[quantity])
        assert abs(cell[f"rel_error_{quantity}"]) < 0.20


def test_richardson_limit_recovers_a_known_field():
    """v(h) = L + C*h**p must be recovered exactly (regression pin: the rework
    ship found a sign/scale error here that pushed one limit below every
    measured value)."""
    limit_true, c_true, p_true = 0.1013, 0.05, 2.0
    points = [(n ** -0.5, limit_true + c_true * n ** -1.0) for n in (784, 1296, 1936)]
    out = case.richardson_limit(points)
    assert out["order"] == pytest.approx(p_true, rel=1e-9)
    assert out["limit"] == pytest.approx(limit_true, abs=1e-12)
    # a decreasing ladder extrapolates BELOW its finest value
    assert out["limit"] < points[-1][1]
    increasing = [(h, 1.0 - (v - limit_true)) for h, v in points]
    assert case.richardson_limit(increasing)["limit"] > increasing[-1][1]


def test_richardson_limit_falls_back_when_the_ladder_is_noisy():
    points = [(n ** -0.5, v) for n, v in ((784, 0.24835), (1296, 0.24815), (1936, 0.24835))]
    out = case.richardson_limit(points)
    assert out["order"] is None
    assert out["limit"] == points[-1][1]
    assert "fallback" in out["method"]


# --- gates -----------------------------------------------------------------
def _synthetic_rows(case_subdiv=4, coarse_subdiv=3, bad=False):
    rows = []
    for subdiv, err in ((coarse_subdiv, 0.06), (case_subdiv, 0.01)):
        for ka in (0.2, 1.0, 2.0):
            row = {
                "kind": "band",
                "ka": ka,
                "subdivisions": subdiv,
                "panels": case.panel_count(subdiv),
                "volume_error": -0.006 if subdiv == case_subdiv else -0.023,
                "waterplane_error": -0.002,
                "condition_number": 3.0,
            }
            for quantity in case.QUANTITIES:
                value = 1.0
                row[quantity] = value
                row["reference"] = {q: 1.0 for q in case.QUANTITIES}
                row[f"rel_error_{quantity}"] = err
            if bad and subdiv == case_subdiv and ka == 2.0:
                row["rel_error_B33_dimless"] = 0.09
            rows.append(row)
    return rows


def test_gate_evaluator_passes_a_clean_run():
    ladders = {f"{ka:g}": case.reference_ladder(ka) for ka in (0.2, 1.0, 2.0)}
    verdict = case.evaluate(_synthetic_rows(), ladders)
    assert verdict["criteria"]["G1_case_mesh_band_error_below_5pct"]["pass"] is True
    assert verdict["criteria"]["G2_geometry_precondition"]["pass"] is True
    assert verdict["criteria"]["G3_refinement_does_not_worsen"]["pass"] is True
    assert verdict["criteria"]["G5_independent_judge_below_5pct"]["pass"] is None


def test_gate_evaluator_fails_a_bad_number():
    ladders = {f"{ka:g}": case.reference_ladder(ka) for ka in (0.2, 1.0, 2.0)}
    verdict = case.evaluate(_synthetic_rows(bad=True), ladders)
    g1 = verdict["criteria"]["G1_case_mesh_band_error_below_5pct"]
    assert g1["pass"] is False
    assert g1["worst_quantity"] == "B33_dimless"
    assert verdict["verdict"] == "FAIL"


def test_gate_evaluator_fails_the_geometry_precondition_of_a_256_panel_mesh():
    """Replay the m17235 finding: 256-panel geometry error 2.32 % > 2 % gate."""
    ladders = {f"{ka:g}": case.reference_ladder(ka) for ka in (0.2, 1.0, 2.0)}
    rows = _synthetic_rows()
    for row in rows:
        if row["subdivisions"] == case.CASE_SUBDIVISIONS:
            row["volume_error"] = -0.0232  # the 256-panel value
    verdict = case.evaluate(rows, ladders)
    assert verdict["criteria"]["G2_geometry_precondition"]["pass"] is False
    assert verdict["verdict"] == "FAIL"


# --- registry / driver -----------------------------------------------------
def test_registry_dispatches_the_anchor():
    module = case_pool.get_anchor(case.CASE_CODE)
    assert module is case
    inventory = {row["case_code"]: row for row in case_pool.list_anchors()}
    assert case.CASE_CODE in inventory
    assert inventory[case.CASE_CODE]["case_pool_id"] == str(case.CASE_POOL_ID)
    assert inventory[case.CASE_CODE]["tolerance"] == "<5%"
    assert case_pool.ANCHORS[case.CASE_CODE].run is case.run


def test_registry_rejects_an_unknown_anchor():
    with pytest.raises(KeyError):
        case_pool.get_anchor("CASE-BEM-NOT-A-CASE")


def test_historical_driver_path_still_dispatches():
    driver = _load_driver()
    assert driver.ANCHOR == case.CASE_CODE
    assert callable(driver.main)


def test_benchmark_table_carries_ref_got_and_independent_columns():
    indep = {
        "ka_grid": [1.0],
        "limits": {
            "1.000": {
                "A33_dimless": {
                    "finest": 0.43,
                    "limit_exact": 0.43,
                    "limit_geometric": 0.43,
                    "bracket": [0.43, 0.43],
                    "method_exact": "3-point Richardson",
                    "order_exact": 2.0,
                    "order_geometric": 2.0,
                    "ladder": [],
                }
            }
        },
    }
    table = case.benchmark_table(_synthetic_rows(), indep)
    g1_rows = [r for r in table if r["ka"] == 1.0]
    assert g1_rows
    a33 = next(r for r in g1_rows if r["quantity"] == "A33_dimless")
    assert a33["ref"] == 1.0 and a33["got"] == 1.0
    assert a33["rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert a33["rel_error_vs_independent"] == pytest.approx((1.0 - 0.43) / 0.43)


# --- committed evidence bundle --------------------------------------------
def test_evidence_bundle_exists_and_is_complete():
    if not EVIDENCE.exists():
        pytest.skip("evidence bundle not present in this checkout")
    for name in (
        "results.json",
        "benchmark.json",
        "reference.json",
        "report.md",
        "figure.png",
        "artifacts_sha256.json",
        "repro.sh",
    ):
        assert (EVIDENCE / name).is_file(), name
    payload = json.loads((EVIDENCE / "results.json").read_text(encoding="utf-8"))
    assert payload["case"]["case_code"] == case.CASE_CODE
    assert payload["case"]["case_mesh_panels"] == case.panel_count(case.CASE_SUBDIVISIONS)
    assert payload["repro_cmd"] and "hulme_hemisphere_case" in payload["repro_cmd"][0]
    assert payload["benchmark"], "benchmark table must not be empty"


def test_evidence_checksums_match_the_files():
    if not EVIDENCE.exists():
        pytest.skip("evidence bundle not present in this checkout")
    digests = json.loads((EVIDENCE / "artifacts_sha256.json").read_text(encoding="utf-8"))["sha256"]
    import hashlib

    for name, digest in digests.items():
        path = REPO_ROOT / name if name.startswith("docs/") else EVIDENCE / name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_committed_run_passes_every_gate():
    """The gate verdict of the committed run, re-read from its artefact."""
    if not EVIDENCE.exists():
        pytest.skip("evidence bundle not present in this checkout")
    verdict = json.loads((EVIDENCE / "results.json").read_text(encoding="utf-8"))["verdict"]
    assert verdict["verdict"] == "PASS"
    for key, crit in verdict["criteria"].items():
        assert crit["pass"] is True, key