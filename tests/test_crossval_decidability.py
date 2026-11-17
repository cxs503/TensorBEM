"""Decidability contract tests (``tensorbem.benchmarks.crossval_decidability``).

The three-valued verdict is the load-bearing idea: a deviation that cannot be
resolved at a tolerance must never be published as PASS *or* as FAIL.  These
tests pin the rules down one by one, plus the two consumer paths
(``capytaine_finite_depth_crossval.compare`` and the frozen reference truth
audit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorbem.benchmarks import capytaine_finite_depth_crossval as study
from tensorbem.benchmarks import crossval_decidability as cd
from tensorbem.benchmarks import finite_depth_crossval_plan as plan

ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = ROOT / "benchmarks" / "verified" / "capytaine_finite_depth"


# --------------------------------------------------------------------------- #
# Rules 1/6: the bound itself
# --------------------------------------------------------------------------- #
def test_power_law_bound_matches_the_classical_gci_value():
    # v(N) = v_inf + C N^-p on a 4x ladder with p = 1: the finest rung still
    # carries 1/3 of the last change... (4^1 - 1 = 3).
    out = cd.ladder_uncertainty(
        last_rung_change=0.03, order=1.0, refinement=4.0
    )
    assert out["value"] == pytest.approx(0.01)
    assert out["asymptotic"] is True


def test_non_asymptotic_order_still_gets_a_model_bound_with_the_floor_noted():
    out = cd.ladder_uncertainty(
        last_rung_change=0.0162, order=0.24, refinement=4.0, min_order=0.3
    )
    assert out["asymptotic"] is False
    # 4^0.24 - 1 = 0.39 -> the model bound is 4x the raw change, i.e. much
    # larger than the change itself, which is exactly why the raw change is not
    # a safe bound.
    assert out["value"] == pytest.approx(0.0162 / (4.0**0.24 - 1.0))
    assert out["value"] > 0.0162
    assert "below the pre-registered floor" in out["basis"]


def test_safety_factor_is_applied_to_model_bounds_only():
    model = cd.ladder_uncertainty(
        last_rung_change=0.01, order=1.0, refinement=4.0,
        safety_factor=cd.GCI_SAFETY_FACTOR,
    )
    measured = cd.ladder_uncertainty(
        last_rung_change=0.01, order=1.0, refinement=4.0,
        measured_error=0.012, safety_factor=cd.GCI_SAFETY_FACTOR,
    )
    assert model["value"] == pytest.approx((0.01 / 3.0) * 1.25)
    # a measurement is used as it is: the exact anchor has no discretisation to
    # inflate
    assert measured["value"] == pytest.approx(0.012)


def test_measured_band_replaces_the_model_bound_on_a_four_rung_ladder():
    out = cd.ladder_uncertainty(
        last_rung_change=0.05, order=0.7, refinement=4.0, measured_band=0.004
    )
    assert out["value"] == pytest.approx(0.004)
    assert "measured drop-coarsest band" in out["basis"]


# --------------------------------------------------------------------------- #
# Rule 5: unknown is not zero
# --------------------------------------------------------------------------- #
def test_unknown_uncertainty_can_never_produce_a_pass():
    decision = cd.decide(
        deviation=0.0, tolerance=0.15, reference=0.01, solver=None
    )
    assert decision["verdict"] == cd.UNDECIDABLE
    assert decision["decidable"] is False
    assert "unknown is not zero" in decision["reason"]


def test_no_ladder_at_all_is_unknown_not_zero():
    out = cd.ladder_uncertainty(last_rung_change=None)
    assert out["value"] is None


# --------------------------------------------------------------------------- #
# Rule 4: the budget
# --------------------------------------------------------------------------- #
def test_budget_blown_is_undecidable_even_when_the_deviation_looks_fine():
    decision = cd.decide(
        deviation=0.001, tolerance=0.03, reference=0.0162, solver=0.05
    )
    assert decision["verdict"] == cd.UNDECIDABLE
    assert decision["budget_exceeded"] is True
    # what tolerance would make this resolvable: (0.0162+0.05)/(1/3)
    assert decision["decidable_from_tolerance"] == pytest.approx(
        (0.0162 + 0.05) / cd.BUDGET_SHARE
    )
    assert decision["binding_side"] == "solver"


def test_a_robust_fail_survives_an_unknown_budget():
    # deviation exceeds the tolerance by more than the total uncertainty: no
    # possible uncertainty value can rescue a PASS, so FAIL is decidable.
    decision = cd.decide(
        deviation=0.30, tolerance=0.05, reference=0.02, solver=0.02
    )
    assert decision["verdict"] == cd.FAIL
    assert decision["decidable"] is True
    assert decision["budget_exceeded"] is True


def test_pass_and_fail_inside_a_resolved_tolerance():
    ok = cd.decide(deviation=0.01, tolerance=0.03, reference=0.004, solver=0.004)
    bad = cd.decide(deviation=0.04, tolerance=0.03, reference=0.004, solver=0.004)
    assert (ok["verdict"], ok["decidable"]) == (cd.PASS, True)
    assert (bad["verdict"], bad["decidable"]) == (cd.FAIL, True)


def test_refinement_requirement_follows_the_measured_order():
    out = cd.required_refinement(0.0181, target=0.01, order=0.2412, refinement=4.0)
    # (1.81)^(1/0.2412) ~ 11.7x, the number mission m23018 quoted in prose
    assert out["refinement_panels"] == pytest.approx(11.7, abs=0.3)
    unknown = cd.required_refinement(0.0181, target=0.01, order=None, refinement=4.0)
    assert unknown["refinement_panels"] is None
    assert unknown["feasible"] is None


def test_summary_is_undecidable_when_any_quantity_is():
    decisions = [
        cd.decide(deviation=0.01, tolerance=0.03, reference=0.002, solver=0.002),
        cd.decide(deviation=0.01, tolerance=0.03, reference=0.02, solver=0.02),
    ]
    summary = cd.summarise(decisions, tolerance=0.03)
    assert summary["decidable"] is False
    assert summary["counts"][cd.UNDECIDABLE] == 1
    assert cd.summarise([], tolerance=0.03)["decidable"] is False


# --------------------------------------------------------------------------- #
# The exact anchor behind the measured reference error
# --------------------------------------------------------------------------- #
def test_truth_audit_artifact_reproduces_from_the_frozen_reference():
    audit_path = REF_ROOT / study.TRUTH_FILE
    if not audit_path.exists():
        pytest.skip("truth audit artifact not frozen")
    frozen = json.loads(audit_path.read_text())
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "finite_depth_reference_truth_audit",
        ROOT / "scripts" / "finite_depth_reference_truth_audit.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recomputed = module.audit(
        json.loads((REF_ROOT / study.REFERENCE_FILE).read_text())
    )
    assert recomputed["summary"]["admitted_cells"] == frozen["summary"][
        "admitted_cells"
    ]
    for got, want in zip(recomputed["cells"], frozen["cells"]):
        assert got["cell"] == want["cell"]
        for slot, quantity in want["quantities"].items():
            assert got["quantities"][slot]["finest_error_abs"] == pytest.approx(
                quantity["finest_error_abs"], rel=1e-12
            )


def test_truth_audit_excludes_the_shallow_cells_by_the_clearance_exponent():
    frozen = json.loads((REF_ROOT / study.TRUTH_FILE).read_text())
    excluded = {row["cell"] for row in frozen["excluded"]}
    # k0 h = 6 at h/a = 3 looks deep, but the clearance is 0.5 m and the image
    # exponent 2 k0 (h - f - a) is only 2 -> the deep-water anchor is not exact
    assert "h_over_a=3.0|ka=2.00" in excluded
    assert all(
        row["clearance_exponent"] < frozen["criterion"]["clearance_exponent"]
        for row in frozen["excluded"]
    )
    # ...while every admitted cell clears the threshold
    assert all(
        row["clearance_exponent"] >= frozen["criterion"]["clearance_exponent"]
        for row in frozen["cells"]
    )


def test_the_last_rung_proxy_is_optimistic_against_measured_truth():
    frozen = json.loads((REF_ROOT / study.TRUTH_FILE).read_text())
    ratios = [
        quantity["proxy_optimism"]
        for row in frozen["cells"]
        for quantity in row["quantities"].values()
    ]
    assert ratios
    # the audit is what justifies GCI_SAFETY_FACTOR: the raw proxy is *below*
    # the measured truth in (almost) every quantity
    assert min(ratios) > 1.0
    assert cd.GCI_SAFETY_FACTOR >= 1.0


# --------------------------------------------------------------------------- #
# Consumer: the frozen verdict carries the verdicts
# --------------------------------------------------------------------------- #
def test_frozen_verdict_carries_a_three_valued_decision_per_quantity():
    result = json.loads((REF_ROOT / study.RESULT_FILE).read_text())
    summary = result["comparison"]["decidability"]
    assert summary["criteria"]["module"].endswith("crossval_decidability")
    decided = 0
    for row in result["comparison"]["rows"]:
        for quantity in row["quantities"].values():
            decision = quantity["decidability"]
            assert decision["verdict"] in (cd.PASS, cd.FAIL, cd.UNDECIDABLE)
            assert decision["tolerance"] == pytest.approx(plan.TOL_ADDED_MASS) or (
                decision["tolerance"] == pytest.approx(plan.TOL_DAMPING)
            )
            assert quantity["uncertainty"]["solver"]["value"] is not None
            decided += 1
    assert decided == summary["added_mass"]["decisions"] + summary["damping"][
        "decisions"
    ]


def test_measured_reference_error_overrides_the_model_bound_on_audited_cells():
    reference = json.loads((REF_ROOT / study.REFERENCE_FILE).read_text())
    ladder = {
        "cells": {
            "h_over_a=10.0|ka=1.00": {
                "cell": "h_over_a=10.0|ka=1.00",
                "h_over_a": 10.0,
                "ka": 1.0,
                "k0h": 10.0,
                "levels": {},
            }
        }
    }
    truth = study.load_reference_truth(REF_ROOT)
    assert "h_over_a=10.0|ka=1.00" in truth
    # a cell with no rungs is still recorded as missing, not compared
    comparison = study.compare(ladder, reference, reference_truth=truth)
    assert comparison["rows"] == []
    assert comparison["cells_without_ladder"] == ["h_over_a=10.0|ka=1.00"]
    assert comparison["decidability"]["measured_reference"]["cells"] == []