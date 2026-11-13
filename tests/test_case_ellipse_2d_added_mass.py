"""Acceptance gates for the pool case ``CASE-BEM-ELLIPSE-2D-ADDED-MASS``.

The case is a *record*: these tests check the record and the two numerical
routes behind it, they do not re-derive the closed form.  Three kinds of gate:

1. mathematics of the reference quantities (AGM perimeter against independent
   quadrature, the circle limit ``a = b`` and the interior-extension closed form
   of the negative control);
2. the predeclared thresholds are met at a *reduced* budget (cheap enough for
   CI) and the committed artifact still re-verifies;
3. the registry wiring, so a moved artifact or a renamed entry cannot rot
   silently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import tensorbem.benchmarks.case_ellipse_2d_added_mass as case_mod
from tensorbem.benchmarks import case_pool_registry as registry
from tensorbem.benchmarks.case_ellipse_2d_added_mass import (
    AXES,
    BUDGET_PANELS,
    CASE_ID,
    LADDER,
    ORDER_MIN,
    SEMI_MAJOR,
    SEMI_MINOR,
    TOLERANCE,
    ellipse_perimeter,
    fitted_order,
    geometry_gate,
    interior_extension_control,
    route_dirichlet,
    route_neumann,
    run_case,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks/verified/case_bem_ellipse_2d_added_mass/results.json"
#: reduced ladder for CI: the *threshold* is the same, only the budget is cheaper
QUICK_PANELS = 64


@pytest.fixture(scope="module")
def quick_case() -> dict:
    """Cheap case cycle: reduced ladder, sweeps off (sweep cost is linear in AR)."""
    return run_case(ladder=(16, 32, 64), with_sweeps=False)


def test_case_id_and_declared_thresholds() -> None:
    assert CASE_ID == "CASE-BEM-ELLIPSE-2D-ADDED-MASS"
    assert TOLERANCE == 0.03
    assert ORDER_MIN == 1.5
    assert BUDGET_PANELS == 128
    assert tuple(sorted(LADDER)) == LADDER, "ladder must be monotone increasing"
    assert LADDER[-1] == BUDGET_PANELS


def test_perimeter_matches_independent_quadrature() -> None:
    """AGM ``4aE(e)`` against a Simpson quadrature of the arc element."""
    def simpson(a: float, b: float, n: int = 20001) -> float:
        theta = np.linspace(0.0, 0.5 * math.pi, n)
        integrand = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
        h = theta[1] - theta[0]
        weights = np.ones(n)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        return 4.0 * h / 3.0 * float(np.sum(weights * integrand))

    for major, minor, tol in (
        (1.0, 1.0, 1e-12),
        (1.5, 0.75, 1e-9),
        (1.0, 0.5, 1e-9),
        (3.0, 0.1, 1e-6),  # thin section: the arc integrand peaks, Simpson slows
    ):
        assert ellipse_perimeter(major, minor) == pytest.approx(
            float(simpson(major, minor)), rel=tol
        )
    # the circle limit is exact, not merely quadrature-accurate
    assert ellipse_perimeter(2.0, 2.0) == pytest.approx(4.0 * math.pi, rel=1e-15)


def test_geometry_gate_passes_and_is_an_under_estimate() -> None:
    gate = geometry_gate(BUDGET_PANELS, SEMI_MAJOR, SEMI_MINOR)
    assert gate["passed"] is True
    assert gate["area_ratio"] <= 1.0 and gate["perimeter_ratio"] <= 1.0
    assert gate["gate_area_deficit"] < case_mod.GATE_AREA_DEFICIT
    assert gate["gate_perimeter_deficit"] < case_mod.GATE_PERIMETER_DEFICIT
    assert gate["panel_length_ratio"] > 1.0, (
        "equal-angle ellipse panels are deliberately unequal in arc length"
    )
    exact_area = math.pi * SEMI_MAJOR * SEMI_MINOR
    assert gate["exact_area"] == pytest.approx(exact_area, rel=1e-15)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("route_name", ["dirichlet", "neumann"])
def test_reduced_budget_meets_the_declared_tolerance(route_name: str, axis: str) -> None:
    route = route_dirichlet if route_name == "dirichlet" else route_neumann
    row = route(QUICK_PANELS, SEMI_MAJOR, SEMI_MINOR, 1.0, 1025.0, axis)
    assert row["relative_error"] < TOLERANCE
    # perpendicular semi-axis rule: x motion -> rho*pi*b^2, y motion -> rho*pi*a^2
    perpendicular = SEMI_MINOR if axis == "x" else SEMI_MAJOR
    assert row["exact_added_mass"] == pytest.approx(
        1025.0 * math.pi * perpendicular**2, rel=1e-14
    )
    assert row["relative_residual_l2"] < 1e-10


def test_second_kind_route_is_more_accurate_than_the_first_kind() -> None:
    """Not a criterion, but a documented property worth pinning.

    The first-kind operator carries the 2-D logarithmic-capacity degeneracy, the
    second-kind operator has O(1) condition number; the Neumann route therefore
    lands one to two orders of magnitude closer at the same mesh.
    """
    for axis in AXES:
        d = route_dirichlet(QUICK_PANELS, SEMI_MAJOR, SEMI_MINOR, 1.0, 1025.0, axis)
        n = route_neumann(QUICK_PANELS, SEMI_MAJOR, SEMI_MINOR, 1.0, 1025.0, axis)
        assert n["relative_error"] < d["relative_error"]
        assert n["condition_number"] < 25.0 < d["condition_number"]


def test_interior_extension_control_reproduces_its_closed_form() -> None:
    """The circle-case trap must be measured *and* predicted in closed form."""
    for axis in AXES:
        row = interior_extension_control(
            case_mod.CONTROL_PANELS, SEMI_MAJOR, SEMI_MINOR, 1.0, 1025.0, axis
        )
        assert row["closed_form_matches"], row["closed_form_abs_discrepancy"]
        assert row["closed_form_abs_discrepancy"] < 5e-3
        # 50 % for x motion, 100 % for y motion at a/b = 2
        expected = 0.5 if axis == "x" else 1.0
        assert row["closed_form_relative_error"] == pytest.approx(expected, rel=1e-12)
        assert row["exceeds_tolerance"], "the control must be rejected by the tolerance"


def test_circle_limit_has_unit_added_mass_coefficient() -> None:
    """``a = b = R``: m_A = rho*pi*R**2 (Ca = 1) for both axes and both routes."""
    radius = 0.8
    for axis in AXES:
        for route in (route_dirichlet, route_neumann):
            row = route(64, radius, radius, 1.0, 1000.0, axis)
            assert row["relative_error"] < TOLERANCE
            assert row["exact_added_mass"] == pytest.approx(
                1000.0 * math.pi * radius**2, rel=1e-14
            )


def test_fitted_order_formula_and_guards() -> None:
    # halving the error per doubling of panels is first order
    assert fitted_order(16, 1e-2, 32, 5e-3) == pytest.approx(1.0)
    # quartering is second order
    assert fitted_order(32, 4e-3, 64, 1e-3) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        fitted_order(64, 1e-3, 32, 4e-3)
    with pytest.raises(ValueError):
        fitted_order(16, 0.0, 32, 1e-3)


def test_quick_case_cycle_passes_its_own_criteria(quick_case: dict) -> None:
    assert quick_case["case_id"] == CASE_ID
    assert quick_case["overall"] == "PASS", quick_case["acceptance"]["failed_checks"]
    assert quick_case["acceptance"]["n_checks"] == len(quick_case["criteria"])
    assert all(item["ok"] for item in quick_case["criteria"])
    for item in quick_case["criteria"]:
        if item["name"].startswith("budget_error"):
            assert item["threshold"] == TOLERANCE
    verified = case_mod.verify(quick_case)
    assert verified["passed"] is True
    assert verified["overall"] == "PASS"


def test_committed_artifact_still_verifies() -> None:
    assert ARTIFACT.is_file(), "the case artifact must be committed"
    payload = json.loads(ARTIFACT.read_text())
    assert payload["case_id"] == CASE_ID
    assert payload["overall"] == "PASS"
    assert payload["geometry"]["panel_budget"] == BUDGET_PANELS
    assert payload["geometry_gate"]["passed"] is True
    verified = case_mod.verify(payload)
    assert verified["passed"] is True, verified["failed_checks"]
    worst = max(s["relative_errors"][-1] for s in payload["convergence"].values())
    assert worst < TOLERANCE
    # the artifact must carry the negative control, not just the passing numbers
    assert payload["negative_control"]["at_budget"]
    assert all(
        row["exceeds_tolerance"] for row in payload["negative_control"]["at_budget"]
    )
    assert payload["aspect_ratio_sweep"], "the sweep must be in the artifact"


def test_registry_entry_is_wired_and_resolves() -> None:
    entry = registry.resolve(CASE_ID)
    assert entry.case_code == CASE_ID
    assert entry.pool_code == "BEM-ELLIPSE-2D-ADDED-MASS"
    assert entry.repo == "TensorBEM"
    assert entry.library == "tensorbem.benchmarks.case_ellipse_2d_added_mass"
    assert entry.tolerance == f"<{TOLERANCE:.0%}"
    assert entry.missing() == [], entry.missing()
    assert registry.artifact_verdict(CASE_ID) == "PASS"
    assert registry.normalise("BEM-ELLIPSE-2D-ADDED-MASS") == CASE_ID
    assert registry.normalise("case-bem-ellipse-2d-added-mass") == CASE_ID