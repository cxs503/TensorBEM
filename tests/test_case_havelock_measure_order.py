"""Guards for the round-4 grid-convergence study of CASE-BEM-SUBMERGED-SPHERE-HAVELOCK.

The study's numbers are only as trustworthy as its V&V kernels and its gate logic, so
both are pinned here against *analytically known* ladders rather than against
themselves:

* a synthetic ``f(h) = f* + C h^p`` family must return the injected ``p`` and ``f*``;
* the kernels must agree with the in-repo convention already used by
  ``benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_continuum.py``
  (a second implementation, so the convention is not silently forked);
* a non-monotone or round-off-limited triple must yield *no* order (nan), never a
  lucky positive exponent;
* the gate block must be able to say FAIL and to FALSIFY the patch-order hypothesis
  -- an audit that can only say PASS is not an audit.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import case_havelock_measure_order_study as study  # noqa: E402


def _load_gci_module():
    path = (ROOT / "benchmarks" / "verified"
            / "casepool_bem_submerged_sphere_havelock" / "gci_continuum.py")
    spec = importlib.util.spec_from_file_location("gci_continuum_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0, 3.0])
def test_power_law_recovers_injected_order_and_limit(p_true):
    """f(h) = f* + C h^p on h = 1, 1/2, 1/4 must give back p and f*."""
    f_star, c_const = 0.437, 0.9
    values = [f_star + c_const * h**p_true for h in (1.0, 0.5, 0.25)]
    p = study.observed_order(*values, 2.0)
    assert p == pytest.approx(p_true, rel=1e-10)
    limit = study.richardson_continuum(values[1], values[2], p, 2.0)
    assert limit == pytest.approx(f_star, rel=1e-10)


def test_order_matches_in_repo_convention():
    """Independent second implementation (gci_continuum) must agree exactly."""
    reference = _load_gci_module()
    ladders = [(0.5, 0.51, 0.5125), (1.0, 1.02, 1.005), (0.3, 0.31, 0.3075),
               (0.44116, 0.4412, 0.44116), (1.0, 1.0, 1.0)]
    for ladder in ladders:
        mine = study.observed_order(*ladder, 2.0)
        theirs = reference.observed_order(*ladder, 2.0)
        if math.isnan(mine) or math.isnan(theirs):
            assert math.isnan(mine) and math.isnan(theirs), ladder
        else:
            assert mine == pytest.approx(theirs, rel=1e-12), ladder


def test_non_monotone_ladder_has_no_order():
    assert study.order_regime(0.3, 0.31, 0.3075) == "non_monotone"
    assert math.isnan(study.observed_order(0.3, 0.31, 0.3075, 2.0))
    assert math.isnan(study.richardson_continuum(0.31, 0.3075, float("nan"), 2.0))
    assert math.isnan(study.gci_fine(0.31, 0.3075, float("nan"), 2.0))


def test_round_off_ladder_has_no_order():
    assert study.order_regime(1.0, 1.0, 1.0) == "at_roundoff"
    assert math.isnan(study.observed_order(1.0, 1.0, 1.0, 2.0))


def test_gci_is_positive_and_shrinks_with_finer_difference():
    p = 2.0
    big = study.gci_fine(0.50, 0.512, p, 2.0)
    small = study.gci_fine(0.51, 0.512, p, 2.0)
    assert big > 0.0 and small > 0.0
    assert small < big


def _cell(measure, ka, quantity, values, ref, regime="asymptotic", err_fine=None,
          err_medium=None):
    p = study.observed_order(*values, 2.0)
    limit = study.richardson_continuum(values[1], values[2], p, 2.0)
    return {
        "measure": measure, "ka": ka, "quantity": quantity, "panels": [80, 320, 1280],
        "values": list(values), "reference": ref,
        "successive_differences": [abs(values[1] - values[0]), abs(values[2] - values[1])],
        "order_regime": regime, "resolved": regime == "asymptotic",
        "observed_order": p, "richardson_continuum": limit,
        "gci_fine": study.gci_fine(values[1], values[2], p, 2.0),
        "err_fine_vs_ref": (abs(values[2] - ref) / ref) if err_fine is None else err_fine,
        "err_medium_vs_ref": (abs(values[1] - ref) / ref) if err_medium is None else err_medium,
        "err_coarse_vs_ref": abs(values[0] - ref) / ref,
        "err_continuum_vs_ref": (abs(limit - ref) / ref
                                 if math.isfinite(limit) else float("nan")),
    }


def test_gates_pass_on_a_clean_second_order_ladder():
    ref = 0.437
    values = [0.437 + 0.9 * h**2 for h in (1.0, 0.5, 0.25)]
    cells = [_cell("spherical_patch", ka, q, values, ref)
             for ka in (0.5, 1.0) for q in ("a33", "b33")]
    gates = study._gates(cells)
    assert gates["G1_order_admissible"]["verdict"] == "PASS"
    assert gates["G2_continuum_vs_reference"]["verdict"] == "PASS"
    assert gates["G3_monotone_refinement"]["verdict"] == "PASS"


def test_gates_fail_on_a_wrong_reference():
    """A reference displaced by 5 % must be caught by G2 (the study can say no)."""
    true_ref = 0.437
    bias = 1.05
    values = [true_ref + 0.9 * h**2 for h in (1.0, 0.5, 0.25)]
    cells = [_cell("spherical_patch", 1.0, "a33", values, true_ref * bias)]
    gates = study._gates(cells)
    assert gates["G2_continuum_vs_reference"]["verdict"] == "FAIL"
    assert gates["G2_continuum_vs_reference"]["violations"] == 1


def test_gates_fail_on_non_monotone_refinement():
    cells = [_cell("spherical_patch", 1.0, "a33", (0.437, 0.44, 0.4395), 0.437,
                   regime="non_monotone",
                   err_fine=abs(0.4395 - 0.437) / 0.437,
                   err_medium=abs(0.44 - 0.437) / 0.437)]
    gates = study._gates(cells)
    assert gates["G3_monotone_refinement"]["verdict"] == "PASS"  # err does fall here
    cells[0]["err_fine_vs_ref"] = 0.01
    cells[0]["err_medium_vs_ref"] = 0.005
    gates = study._gates(cells)
    assert gates["G3_monotone_refinement"]["verdict"] == "FAIL"


def test_patch_order_hypothesis_can_be_falsified():
    ref = 0.1
    flat = [_cell("flat", 1.0, "a33", [ref + 0.9 * h**2 for h in (1.0, 0.5, 0.25)], ref)]
    patch = [_cell("spherical_patch", 1.0, "a33",
                   [ref + 0.05 * h**2 for h in (1.0, 0.5, 0.25)], ref)]
    gates = study._gates(flat + patch)
    assert gates["H1_patch_raises_order"]["verdict"] == "FALSIFIED"
    patch_same_order_higher = [_cell("spherical_patch", 1.0, "a33",
                                     [ref + 0.05 * h**3 for h in (1.0, 0.5, 0.25)], ref)]
    gates = study._gates(flat + patch_same_order_higher)
    assert gates["H1_patch_raises_order"]["verdict"] == "SUPPORTED"


def test_non_finite_scan_is_not_vacuous():
    assert study._find_nan_except_orders({"a": [1.0, float("nan")]}) == [".a[1]"]
    assert study._find_nan_except_orders({"a": [1.0, 2.0]}) == []