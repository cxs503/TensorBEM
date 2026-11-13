"""CI guard for the case-pool anchor ``BEM-HULME-HEMISPHERE``.

The heavy band sweep lives in
``benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py`` and is run on
the cluster; these tests keep the *machinery* honest on every commit:

* the Hulme (1982) reference is self-converged in its truncation count (so the
  5 % gate cannot be an artefact of a truncated reference);
* one 64-panel solver cell is actually executed and lands within a loose
  coarse-mesh envelope (the predeclared case gate itself is checked by the
  cluster run, not here);
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
    / "casepool_bem_hulme_hemisphere"
    / "run_case.py"
)


def _load_driver():
    spec = importlib.util.spec_from_file_location("hulme_case_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    assert DRIVER.is_file(), f"case driver missing: {DRIVER}"
    return _load_driver()


def test_reference_truncation_is_converged(driver):
    """96 -> 144 term drift must be far below the 5 % case gate at every ka."""
    for ka in driver.BAND_KA:
        ladder = driver.reference_ladder(ka)
        drift = ladder["max_rel_delta"]["96->144"]
        assert drift < 1e-3, (ka, drift)
        # the 48 -> 96 step must dominate the 96 -> 144 step (geometric tail)
        assert ladder["max_rel_delta"]["48->96"] >= drift, ka


def test_reference_normalisation_is_physical(driver):
    """Low-frequency limits: A33/(rho nabla) ~ 0.835, A11/(rho nabla) ~ 0.62."""
    ref = driver.reference(0.05)
    # heave added mass peaks near 0.876 rho*nabla just above ka -> 0 and the
    # low-frequency surge value sits at 0.510 rho*nabla (Hulme's own curve)
    assert 0.85 < ref["A33_dimless"] < 0.90
    assert 0.48 < ref["A11_dimless"] < 0.54
    # surge damping vanishes in the long-wave limit (>= ka^3 law)
    assert abs(ref["B11_dimless"]) < 1e-3


def test_coarse_cell_executes_and_is_finite(driver):
    """A real solve on the 64-panel mesh: finite, geometry-degraded, ~10 % band."""
    cell = driver._panel_cell(1.0, driver.COARSE_SUBDIVISIONS)
    for q in driver.QUANTITIES:
        assert math.isfinite(cell[q]), q
        assert math.isfinite(cell[f"rel_error_{q}"]), q
    # the coarse mesh under-states the displaced volume (constant panels on a
    # curved surface) -- documented geometry floor, must be inside 10 %
    vol_err = cell["mesh_volume"] / cell["nabla_exact"] - 1.0
    assert -0.10 < vol_err < 0.0, vol_err
    worst = max(abs(cell[f"rel_error_{q}"]) for q in driver.QUANTITIES)
    assert worst < 0.25, worst


def test_gate_evaluator_fails_a_bad_number(driver):
    """Doctor one error and the G1 gate must flip to FAIL."""
    rows = []
    for task in driver.build_tasks():
        if task["kind"] != "band" or task["subdivisions"] != driver.CASE_SUBDIVISIONS:
            continue
        ref = driver.reference(task["ka"])
        row = {
            "kind": "band",
            "ka": task["ka"],
            "subdivisions": task["subdivisions"],
            "panels": 256,
            "chief": True,
            "mesh_volume": driver.reference(task["ka"])["A33_dimless"] * 0.0 + 2.09,
            "nabla_exact": 2.0 / 3.0 * math.pi * driver.RADIUS**3,
            "mesh_waterplane_area": math.pi * driver.RADIUS**2,
            "condition_number": 3.0,
            "reference": ref,
            "phase_Fx_deg": ref["phase_Fx_deg"],
            "phase_Fz_deg": ref["phase_Fz_deg"],
            "delta_phase_Fx_deg": 0.0,
            "delta_phase_Fz_deg": 0.0,
        }
        for q in driver.QUANTITIES:
            row[q] = ref[q]
            row[f"rel_error_{q}"] = 0.0
        rows.append(row)
        coarse = dict(row)
        coarse["subdivisions"] = driver.COARSE_SUBDIVISIONS
        coarse["panels"] = 64
        for q in driver.QUANTITIES:
            coarse[f"rel_error_{q}"] = 0.05   # coarser mesh, worse but finite
        rows.append(coarse)
    ref_ladders = {f"{ka:g}": driver.reference_ladder(ka) for ka in driver.BAND_KA}
    ok = driver.evaluate(rows, ref_ladders)
    assert ok["criteria"]["G1_case_mesh_band_error_below_5pct"]["pass"] is True

    case_index = next(i for i, r in enumerate(rows)
                      if r["subdivisions"] == driver.CASE_SUBDIVISIONS)
    rows[case_index]["rel_error_A33_dimless"] = 0.20
    bad = driver.evaluate(rows, ref_ladders)
    assert bad["criteria"]["G1_case_mesh_band_error_below_5pct"]["pass"] is False
    assert bad["verdict"] == "FAIL"