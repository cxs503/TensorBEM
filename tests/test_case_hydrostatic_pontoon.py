"""Regression tests for CASE-BEM-HYDROSTATIC-PONTOON (case-pool anchor).

Every assertion is against the textbook closed form of a wall-sided pontoon
(Biran & Lopez-Pulido ch. 3-4; Rawson & Tupper ch. 3-4), whose values are
computed by ``scripts/case_hydrostatic_pontoon.reference_values`` -- no
hard-coded expected digits.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "case_hydrostatic_pontoon.py"

spec = importlib.util.spec_from_file_location("case_hydrostatic_pontoon", SCRIPT)
cp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cp
spec.loader.exec_module(cp)

CASE = cp.CASE
REF = cp.reference_values(
    CASE["length_m"], CASE["beam_m"], CASE["draft_m"], CASE["kg_m"],
    CASE["density_kg_m3"], CASE["gravity_m_s2"],
)
LEVEL = cp.REFINEMENT_LEVELS[1]


@pytest.fixture(scope="module")
def measured():
    wp = cp.measure_waterplane_route(CASE, LEVEL)
    pr = cp.measure_pressure_route(CASE, LEVEL)
    return wp, pr


def test_geometry_gate_before_any_comparison(measured):
    """Cb / L / B / T are checked on the mesh itself, not on the reference."""
    wp, _ = measured
    L, B, T = CASE["length_m"], CASE["beam_m"], CASE["draft_m"]
    assert wp["bbox_got"] == pytest.approx((L, B, T), rel=0, abs=1e-12)
    assert wp["displaced_volume_m3"] / (L * B * T) == pytest.approx(1.0, abs=1e-14)
    assert wp["waterplane_area_m2"] / (L * B) == pytest.approx(1.0, abs=1e-14)


def test_all_closed_form_quantities_within_tolerance(measured):
    """The case criterion: every quantity < 3 % (measured, not asserted loosely)."""
    wp, pr = measured
    rows = cp.build_comparison(CASE, REF, wp, pr)
    worst = max(rows, key=lambda r: r["max_abs_rel_err"])
    assert worst["pass"], worst
    assert worst["max_abs_rel_err"] < CASE["tolerance_rel"]
    # the pontoon is a round-off case: the real bar is far tighter than 3 %
    assert worst["max_abs_rel_err"] < 1e-9


def test_two_routes_agree_on_the_displaced_volume(measured):
    wp, pr = measured
    assert pr["displaced_volume_m3"] == pytest.approx(wp["displaced_volume_m3"], rel=1e-12)


def test_centre_of_buoyancy_is_half_the_draft_on_both_routes(measured):
    wp, pr = measured
    assert wp["kb_m"] == pytest.approx(REF["kb_m"], rel=1e-12)
    assert pr["kb_m"] == pytest.approx(REF["kb_m"], rel=1e-12)
    assert abs(pr["lcb_m"]) < 1e-12


def test_restoring_matrix_against_the_metacentric_closed_form(measured):
    wp, pr = measured
    for matrix in (wp["matrix"], pr["matrix"]):
        assert matrix[2, 2] == pytest.approx(REF["c33_N_per_m"], rel=1e-10)
        assert matrix[3, 3] == pytest.approx(REF["c44_Nm_per_rad"], rel=1e-10)
        assert matrix[4, 4] == pytest.approx(REF["c55_Nm_per_rad"], rel=1e-10)
        scale = np.abs(matrix).max()
        off = np.abs(matrix - np.diag(np.diag(matrix))).max()
        assert off < 1e-10 * scale          # no coupling for a symmetric pontoon
        # surge / sway / yaw carry no hydrostatic restoring at all
        for index in (0, 1, 5):
            assert np.abs(matrix[index, :]).max() < 1e-10 * scale


def test_heel_righting_arm_matches_the_wall_sided_formula():
    steps = cp.heel_check(CASE, REF, angles_deg=(1.0, 2.0, 5.0))
    for row in steps:
        assert row["gz_got_m"] == pytest.approx(row["gz_wall_sided_m"], rel=1e-9)
        assert row["gz_got_m"] > 0.0        # righting, not capsizing


def test_finite_difference_ladder_is_second_order():
    ladder = cp.fd_step_sensitivity(CASE)
    orders = [r["observed_order_c44"] for r in ladder[1:]]
    assert all(order == pytest.approx(2.0, abs=0.05) for order in orders)


def test_pipeline_writes_artifacts_and_passes(tmp_path):
    results = cp.run(tmp_path, figures=False)
    assert results["verdict"]["pass"], results["verdict"]
    assert results["geometry_gate"]["pass"]
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["verdict"]["pass"]
    assert (tmp_path / "report.md").read_text().startswith("# CASE-BEM-HYDROSTATIC-PONTOON")
    assert "✅" in (tmp_path / "report.md").read_text()
    assert "❌" not in (tmp_path / "report.md").read_text()
