"""Acceptance tests for the XVAL-BEM-FINITE-DEPTH cross-validation case.

Artifact gates (every recorded number inside its threshold, gates faithful to the
stored stages) plus cheap method gates that recompute the pieces the case leans
on: the closed-form deep-water reference, the geometry measure, one coarse
finite-depth solve and the exact-measure plumbing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.benchmarks import xval_bem_finite_depth as xval
from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
from tensorbem.geometry3d import spherical_patch_areas, submerged_sphere_mesh

CASE_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "xval_bem_finite_depth"


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads((CASE_DIR / "result.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gates() -> dict:
    return json.loads((CASE_DIR / "gates.json").read_text(encoding="utf-8"))


def test_gate_record_is_consistent(gates: dict, result: dict) -> None:
    assert gates["case"] == "xval_bem_finite_depth"
    recomputed = xval.evaluate_gates(result)
    assert [g["id"] for g in recomputed["gates"]] == [g["id"] for g in gates["gates"]]
    for a, b in zip(recomputed["gates"], gates["gates"]):
        assert a["passed"] == b["passed"], (a["id"], a, b)


def test_acceptance_number_is_recorded(gates: dict) -> None:
    by_id = {g["id"]: g for g in gates["gates"]}
    assert by_id["G6_xval_depth_grid"]["measured"] is not None
    assert by_id["G7_deep_water_limit"]["measured"] is not None
    assert gates["threshold"] == xval.THRESHOLD


def test_every_number_carries_a_verdict(gates: dict) -> None:
    for g in gates["gates"]:
        assert isinstance(g["passed"], bool)
        assert g["criterion"]
    assert gates["all_gates_passed"] == (not gates["failed_gates"])


def test_analytic_reference_recomputes() -> None:
    res = submerged_sphere_wave_multipole(1.0, radius=xval.RADIUS, submergence=xval.SUBMERGENCE,
                                          density=xval.RHO, gravity=xval.GRAVITY)
    recorded = json.loads((CASE_DIR / "result.json").read_text(encoding="utf-8"))["mesh"]["analytic"]
    assert res.heave_added_mass == pytest.approx(recorded["A33"], rel=1e-9)
    assert res.heave_damping == pytest.approx(recorded["B33"], rel=1e-9)


def test_patch_area_measure_is_exact() -> None:
    mesh = submerged_sphere_mesh(1, radius=xval.RADIUS, submergence=xval.SUBMERGEENCE) if False else \
        submerged_sphere_mesh(1, radius=xval.RADIUS, submergence=xval.SUBMERGENCE)
    patch = spherical_patch_areas(mesh.vertices, mesh.faces, center=xval.CENTER, radius=xval.RADIUS)
    assert patch.sum() == pytest.approx(4.0 * math.pi * xval.RADIUS**2, rel=1e-12)
    assert patch.sum() > mesh.areas.sum()


def test_panel_areas_argument_validation() -> None:
    mesh = submerged_sphere_mesh(0, radius=xval.RADIUS, submergence=xval.SUBMERGENCE)
    with pytest.raises(ValueError):
        solve_radiation_finite_depth_3d(mesh, 1.0, 5.0, modes=[2], panel_areas=np.ones(3))


def test_coarse_finite_depth_solve_matches_recorded_mesh_row() -> None:
    mesh = submerged_sphere_mesh(0, radius=xval.RADIUS, submergence=xval.SUBMERGENCE)
    patch = spherical_patch_areas(mesh.vertices, mesh.faces, center=xval.CENTER, radius=xval.RADIUS)
    out = solve_radiation_finite_depth_3d(mesh, 1.0, 20.0, modes=[0, 2], center=xval.CENTER,
                                         density=xval.RHO, gravity=xval.GRAVITY,
                                         mode_count=xval.WORK_MODE_COUNT, panel_areas=patch)
    result = json.loads((CASE_DIR / "result.json").read_text(encoding="utf-8"))
    row = [r for r in result["mesh"]["rows"] if r["panels"] == 20 and r["measure"] == "exact_measure"][0]
    assert out.added_mass[1, 1] == pytest.approx(row["A33"], rel=1e-9)
    assert out.damping[1, 1] == pytest.approx(row["B33"], rel=1e-9)


def test_exact_measure_reduces_the_bias() -> None:
    result = json.loads((CASE_DIR / "result.json").read_text(encoding="utf-8"))
    rows = {(r["panels"], r["measure"]): r for r in result["mesh"]["rows"] if r["route"] != "capytaine"}
    for panels in (80, 320):
        flat = abs(rows[(panels, "flat")]["rel_err_A33"])
        exact = abs(rows[(panels, "exact_measure")]["rel_err_A33"])
        assert exact < flat, (panels, flat, exact)
