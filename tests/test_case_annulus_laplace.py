"""Acceptance tests for the pool case ``CASE-BEM-ANNULUS-LAPLACE``.

The case is a *record*, so the tests check the record rather than re-deriving it:
the reference formulas are harmonic, the predeclared thresholds are met at the
advertised panel budget, the convergence order is second order, and the two
negative controls (wrong loop orientation, probe inside the hole) actually fire.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensorbem.benchmarks.case_annulus_laplace as case_mod
from tensorbem.benchmarks.case_annulus_laplace import (
    BUDGET_PANELS,
    CASE_ID,
    QUICK_LADDER,
    TOLERANCE,
    fitted_order,
    harmonicity_check,
)
from tensorbem.benchmarks.annulus_laplace import CASES
from tensorbem.multiloop import annulus_mesh, evaluate_multiloop_field, solve_laplace_multiloop


@pytest.fixture(scope="module")
def quick_result() -> dict:
    """One ladder up to the advertised budget, sweeps off (test cost)."""
    return case_mod.run_case(ladder=QUICK_LADDER[:3], with_sweeps=False)


def test_case_id_and_ladder_are_declared() -> None:
    assert CASE_ID == "CASE-BEM-ANNULUS-LAPLACE"
    assert BUDGET_PANELS == (64, 32)
    assert TOLERANCE == 0.03
    assert all(pair[0] == 2 * pair[1] for pair in QUICK_LADDER)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_reference_is_harmonic(case) -> None:
    """The closed form must be harmonic, not merely cited."""
    report = harmonicity_check(case)
    assert report["max_abs_laplacian_relative"] < 1.0e-9


def test_reference_table_is_program_generated(quick_result: dict) -> None:
    """48 probes per case, and the table matches the analytic field it came from."""
    for case in quick_result["cases"]:
        rows = case["reference_table"]
        assert len(rows) == 48
        case_object = next(item for item in CASES if item.name == case["name"])
        points = np.array([[row["x_over_a"], row["y_over_a"]] for row in rows])
        values = case_object.potential(points)
        assert np.allclose([row["u_ref"] for row in rows], values, rtol=0.0, atol=0.0)


def test_budget_meets_tolerance(quick_result: dict) -> None:
    """At 64/32 panels every declared quantity must be below 3 %."""
    for case in quick_result["cases"]:
        level = next(
            item
            for item in case["levels"]
            if (item["outer_panels"], item["inner_panels"]) == BUDGET_PANELS
        )
        assert level["q_relative_l2_error"] < TOLERANCE, case["name"]
        assert level["q_relative_linf_error"] < TOLERANCE, case["name"]
        assert level["field_relative_max_error"] < TOLERANCE, case["name"]
        assert level["flux_relative_error"] < TOLERANCE, case["name"]


def test_second_order_convergence(quick_result: dict) -> None:
    """Two independent order estimates must both be second order."""
    for case in quick_result["cases"]:
        orders = case["observed_order_q_relative_l2"]
        assert all(1.8 < order < 2.2 for order in orders), orders
        assert 1.8 < case["fitted_order_q_relative_l2"] < 2.2


def test_panel_requirement_is_ordered(quick_result: dict) -> None:
    """Asking for 1 % must not cost fewer panels than asking for 3 %."""
    for case in quick_result["cases"]:
        first, second = case["panel_requirement"]
        assert first["threshold"] == 0.03 and second["threshold"] == 0.01
        assert first["measured_panel_count"] <= second["measured_panel_count"]
        assert first["extrapolated_panel_count"] < second["extrapolated_panel_count"]


def test_orientation_control_fires() -> None:
    """Wrong inner normals must be a factor 10 worse, with the same residual scale."""
    control = case_mod.orientation_control(CASES[0], *BUDGET_PANELS)
    ratio = (
        control["unflipped_q_relative_l2_error"] / control["correct_q_relative_l2_error"]
    )
    assert ratio > 10.0
    assert control["unflipped_flux_error"] > 10.0 * control["correct_flux_error"]


def test_probe_guard_refuses_hole_and_exterior() -> None:
    """A target in the hole or outside the outer loop is not in the fluid."""
    mesh = annulus_mesh(*BUDGET_PANELS)
    solution = solve_laplace_multiloop(mesh, CASES[0].potential(mesh.centers))
    hole_point = np.array([[0.2, 0.0]])
    outside_point = np.array([[1.5, 0.0]])
    for point in (hole_point, outside_point):
        with pytest.raises(ValueError, match="fluid domain"):
            evaluate_multiloop_field(mesh, solution, point)
    # a point on the fluid side of the inner loop is still accepted
    inside = evaluate_multiloop_field(mesh, solution, np.array([[0.6, 0.0]]))
    assert math.isfinite(float(inside[0]))


def test_verdict_passes_and_reports_failures(quick_result: dict) -> None:
    verdict = quick_result["verdict"]
    assert verdict["passed"], verdict["failed_checks"]
    assert verdict["tolerance"] == TOLERANCE
    names = {check["check"] for check in verdict["checks"]}
    # the sweeps are off in this fixture, so the orientation control is not run
    assert "q_relative_l2_error" in names
    assert "budget_panels_meet_tolerance" in names


def test_markdown_has_the_six_sections(quick_result: dict) -> None:
    text = case_mod.render_markdown(quick_result)
    for heading in (
        "## 1. Problem and reference",
        "## 2. Method",
        "## 3. Parameters (complete)",
        "## 4. Reference vs computed",
    ):
        assert heading in text
    assert "nan |" not in text.lower() and "| nan" not in text.lower()


def test_fitted_order_measures_a_known_slope() -> None:
    levels = [
        {"panel_count": 24, "err": 1.0e-2},
        {"panel_count": 48, "err": 2.5e-3},
        {"panel_count": 96, "err": 6.25e-4},
    ]
    assert fitted_order(levels, "err") == pytest.approx(2.0, abs=1.0e-9)