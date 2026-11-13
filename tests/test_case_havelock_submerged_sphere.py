"""CASE-BEM-SUBMERGED-SPHERE-HAVELOCK: reference integrity + case judgement."""
from __future__ import annotations

import math

import pytest

from tensorbem.benchmarks import case_havelock_submerged_sphere as case
from tensorbem.benchmarks import submerged_sphere_havelock_integrity as integrity


def test_multipole_truncation_is_converged():
    """n_terms 20 -> 40 must not move any coefficient (reference is converged)."""
    result = integrity.run_truncation_study(ka_values=(0.1, 1.0, 5.0))
    assert result["worst_truncation_rel_delta"] < 1e-12
    for row in result["rows"]:
        assert row["a33_n20"] == pytest.approx(row["a33_n40"], rel=1e-12)
        assert row["b33_n20"] == pytest.approx(row["b33_n40"], rel=1e-12, abs=1e-15)


def test_multipole_haskind_identity():
    """Radiation/diffraction branches satisfy Newman's Haskind relation exactly."""
    result = integrity.run_truncation_study(ka_values=(0.1, 1.0, 2.0, 5.0))
    assert result["worst_haskind_error"] < 1e-12


def test_deep_submergence_limit_matches_lamb():
    """f/a -> inf recovers Lamb's unbounded-fluid A' = 0.5 and C_M = 1.5."""
    result = integrity.run_deep_submergence_limit(
        submergence_ratios=(2.0, 5.0, 10.0, 20.0)
    )
    last = result["rows"][-1]
    assert last["a33"] == pytest.approx(integrity.UNBOUNDED_FLUID_ADDED_MASS_DIMLESS, abs=1e-4)
    assert last["a11"] == pytest.approx(integrity.UNBOUNDED_FLUID_ADDED_MASS_DIMLESS, abs=1e-4)
    assert last["inertia_coefficient_cm"] == pytest.approx(
        integrity.UNBOUNDED_FLUID_INERTIA_COEFFICIENT, abs=1e-4
    )
    # image dipole -> excess decays like (a/f)**3
    assert 2.5 < result["observed_excess_decay_order_last"] < 3.5


def test_low_frequency_damping_vanishes():
    """B33', B11' -> 0 as ka -> 0 (no radiated energy at zero frequency)."""
    rows = integrity.run_low_frequency_damping(ka_values=(1e-3, 1e-2, 1e-1))
    assert rows[0]["b33"] < rows[1]["b33"] < rows[2]["b33"]
    assert rows[0]["b33"] < 1e-6
    assert abs(rows[0]["a33"] - integrity.UNBOUNDED_FLUID_ADDED_MASS_DIMLESS) < 0.05


def test_verdicts_flag_failures_and_pass():
    """The pre-declared 5% band is applied per item."""
    ok = case._verdicts([{"item": "x", "quantity": "a33", "rel_error": 0.01, "got": 1.0, "ref": 1.0}])
    assert ok["passed"] and ok["n_failures"] == 0
    bad = case._verdicts(
        [
            {"item": "x", "quantity": "a33", "rel_error": 0.01, "got": 1.0, "ref": 1.0},
            {"item": "y", "quantity": "b33", "rel_error": 0.2, "got": 1.2, "ref": 1.0},
        ]
    )
    assert not bad["passed"]
    assert bad["n_failures"] == 1
    assert bad["worst_item"]["item"] == "y"


def test_case_smoke_single_level():
    """End-to-end case at the coarsest mesh: schema and judgement wiring."""
    result = case.run_case(
        subdivisions=(1,),
        ka_values=(1.0,),
        ka_low=(0.1,),
        submergence_ratios=(2.0,),
    )
    assert result["case"] == "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
    assert result["criteria"]["primary"].startswith("per-item relative error")
    assert len(result["frequency_sweep"]) == 1
    assert len(result["submergence_sweep"]) == 1
    assert result["judgement"]["production_mesh"]["n_items"] == 4
    assert result["reference_integrity"]["truncation"]["worst_haskind_error"] < 1e-12
    row = result["frequency_sweep"][0]
    # the frequency sweep always runs on the production mesh (subdivision 2)
    assert row["n_panels"] == 320
    assert math.isfinite(row["a33_bem"]) and math.isfinite(row["b33_bem"])
    # a 80-panel BEM estimate must be physically plausible (A33' ~ 0.4-0.6)
    assert 0.3 < row["a33_bem"] < 0.7

def test_area_deficit_diagnostic_explains_bias():
    """EXPLORATORY: A33 bias == polyhedral area deficit; B33 bias ~ 2.8x it."""
    from tensorbem.benchmarks.submerged_sphere_havelock import run_convergence_study

    conv = run_convergence_study(subdivisions=(1, 2))
    diag = case.run_area_deficit_diagnostic(conv)
    assert diag["status"] == "exploratory_post_hoc"
    for row in diag["rows"]:
        assert 0.85 < row["ratio_a33_over_deficit"] < 1.15
        assert 2.5 < row["ratio_b33_over_deficit"] < 3.3
