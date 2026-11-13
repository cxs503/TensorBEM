"""CI guard for the case-pool anchor ``BEM-URSELL-CYL-RAD``.

The heavy band sweep lives in
``benchmarks/verified/casepool_bem_ursell_cylinder/run_case.py`` and is run on
the cluster; these tests keep the *machinery* honest on every commit:

* the Ursell (1949) reference is self-converged in its truncation count (so the
  5 % gate cannot be an artefact of a truncated reference);
* one solver cell is actually executed and lands within a loose envelope;
* the gate evaluator really fails a doctored result (a verifier that cannot
  fail is not a verifier).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

DRIVER = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "casepool_bem_ursell_cylinder"
    / "run_case.py"
)


def _load_driver():
    spec = importlib.util.spec_from_file_location("ursell_case_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    assert DRIVER.is_file(), f"case driver missing: {DRIVER}"
    return _load_driver()


def test_reference_truncation_is_converged(driver):
    """80 -> 100 term drift must be far below the 5 % case gate at every ka."""
    for ka in driver.BAND_KA:
        ladder = driver.reference_ladder(ka)
        drift = ladder["deltas"]["80->100"]["max"]
        assert drift < 1e-6, (ka, drift)
        # the 60 -> 80 step must dominate the 80 -> 100 step (geometric tail)
        assert ladder["deltas"]["60->80"]["max"] >= drift, ka


def test_reference_values_are_physical(driver):
    """Sanity checks on the Ursell reference values.

    For a/f = 0.5, the added mass coefficient mu should be positive and
    of order unity; the damping coefficient nu should also be positive.
    """
    for ka in driver.BAND_KA:
        ref = driver.reference(ka)
        assert ref["mu_dimless"] > 0, f"mu must be positive at ka={ka}"
        assert ref["nu_dimless"] > 0, f"nu must be positive at ka={ka}"
        # mu should be O(1) for a/f = 0.5
        assert 0.1 < ref["mu_dimless"] < 10.0, f"mu out of range at ka={ka}"


def test_coarse_cell_executes_and_is_finite(driver):
    """A real solve on the coarse mesh: finite, within loose envelope."""
    cell = driver._panel_cell(0.6, driver.COARSE_PANELS)
    assert math.isfinite(cell["panel_added_mass"]), "added mass not finite"
    assert math.isfinite(cell["panel_damping"]), "damping not finite"
    assert math.isfinite(cell["rel_error_added_mass"]), "error not finite"
    assert math.isfinite(cell["rel_error_damping"]), "error not finite"
    # Coarse mesh should be within 10% (loose envelope)
    assert cell["rel_error_added_mass"] < 0.10, f"coarse A error too large: {cell['rel_error_added_mass']}"
    assert cell["rel_error_damping"] < 0.15, f"coarse B error too large: {cell['rel_error_damping']}"


def test_case_cell_passes_gate(driver):
    """The case mesh (128 panels) must pass the 5% gate at ka=0.6."""
    cell = driver._panel_cell(0.6, driver.CASE_PANELS)
    assert cell["rel_error_added_mass"] < 0.05, f"case A error: {cell['rel_error_added_mass']}"
    assert cell["rel_error_damping"] < 0.05, f"case B error: {cell['rel_error_damping']}"


def test_convergence_is_monotonic(driver):
    """Error must decrease with mesh refinement: coarse > case > fine."""
    ka = 0.6
    coarse = driver._panel_cell(ka, driver.COARSE_PANELS)
    case = driver._panel_cell(ka, driver.CASE_PANELS)
    fine = driver._panel_cell(ka, driver.FINE_PANELS)
    assert coarse["rel_error_added_mass"] > case["rel_error_added_mass"], "A: coarse not > case"
    assert case["rel_error_added_mass"] > fine["rel_error_added_mass"], "A: case not > fine"
    assert coarse["rel_error_damping"] > case["rel_error_damping"], "B: coarse not > case"
    assert case["rel_error_damping"] > fine["rel_error_damping"], "B: case not > fine"


def test_heave_sway_symmetry(driver):
    """Heave and sway added mass must coincide in deep water."""
    cell = driver._panel_cell(0.6, driver.CASE_PANELS)
    assert cell["heave_sway_diff"] < 1e-8, f"heave/sway diff: {cell['heave_sway_diff']}"


def test_near_far_field_identity(driver):
    """Near-field and far-field damping must agree within 3%."""
    cell = driver._panel_cell(0.6, driver.CASE_PANELS)
    ratio_err = abs(cell["near_over_far"] - 1.0)
    assert ratio_err < 0.03, f"near/far ratio error: {ratio_err}"


def test_gate_evaluator_fails_a_bad_number(driver):
    """Doctor one error and the G1 gate must flip to FAIL."""
    rows = []
    for task in driver.build_tasks():
        if task["kind"] != "case":
            continue
        ref = driver.reference(task["ka"])
        row = {
            "kind": "case",
            "ka": task["ka"],
            "panels": task["panels"],
            "panel_added_mass": ref["added_mass"],
            "panel_damping": ref["damping"],
            "reference_added_mass": ref["added_mass"],
            "reference_damping": ref["damping"],
            "rel_error_added_mass": 0.0,
            "rel_error_damping": 0.0,
            "near_over_far": 1.0,
            "heave_sway_diff": 0.0,
            "panel_residual": 1e-12,
            "condition_number": 10.0,
            "displaced_mass": ref["displaced_mass_per_unit_span"],
        }
        rows.append(row)

    ref_ladders = {f"{ka:g}": driver.reference_ladder(ka) for ka in driver.BAND_KA}
    ok = driver.evaluate(rows, ref_ladders)
    assert ok["criteria"]["G1_case_error_below_5pct"]["pass"] is True

    # Doctor one row
    rows[0]["rel_error_added_mass"] = 0.20
    bad = driver.evaluate(rows, ref_ladders)
    assert bad["criteria"]["G1_case_error_below_5pct"]["pass"] is False
    assert bad["verdict"] == "FAIL"


def test_tasks_are_complete(driver):
    """Task list should cover all ka values and all mesh resolutions."""
    tasks = driver.build_tasks()
    kas = set(t["ka"] for t in tasks)
    assert kas == set(driver.BAND_KA)
    kinds = set(t["kind"] for t in tasks)
    assert kinds == {"case", "coarse", "fine"}
    # 5 ka × 3 kinds = 15 tasks
    assert len(tasks) == 15
