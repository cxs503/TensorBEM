"""Regression guards for the two-sided finite-depth convergence study.

Each claim of ``docs/m2952/finite_depth_refinement.md`` is asserted here, so the
claim fails the suite when it stops being true:

* the MFS reference used by ``docs/finite_depth_radiation.md`` is insensitive to
  its own two parameters (source count, source radius) far below the gap it is
  asked to arbitrate;
* the panel -- MFS gap closes at first order in the panel width and extrapolates
  onto the MFS value, i.e. it is panel discretisation and not a defect;
* the near/far-field damping offset also closes at first order;
* the committed artefact is internally consistent (orders recomputed from rows).

Runtimes stay in the tens of seconds: the panel solves are 2-D and the MFS
collocation is cheap.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks.finite_depth_radiation import submerged_circle_mesh
from tensorbem.benchmarks.finite_depth_refinement import (
    far_field_steps,
    mfs_stability,
    observed_order,
    panel_row,
    refine,
    richardson_limit,
)
from tensorbem.finite_depth_solver import radiation_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = (
    REPO_ROOT / "docs" / "m2952" / "outputs" / "finite_depth_refinement" / "benchmark.json"
)
KA = 0.6


def test_observed_order_recovers_first_and_second_order_sequences():
    assert observed_order(2.0, 1.0, 2.0) == pytest.approx(1.0)
    assert observed_order(4.0, 1.0, 2.0) == pytest.approx(2.0)
    # Non-dyadic refinement uses the actual factor, not log2.
    assert observed_order(1.5, 1.0, 1.5) == pytest.approx(1.0)


def test_observed_order_refuses_a_stalled_sequence():
    assert math.isnan(observed_order(1.0, 1.0, 2.0))
    assert math.isnan(observed_order(0.5, 1.0, 2.0))


def test_observed_order_rejects_a_bad_refinement_factor():
    with pytest.raises(ValueError):
        observed_order(2.0, 1.0, 1.0)


def test_richardson_limit_is_exact_for_a_first_order_sequence():
    result = richardson_limit(100.0 + 0.4, 100.0 + 0.2, 2.0, 1.0)
    assert result["limit"] == pytest.approx(100.0, abs=1e-12)
    assert result["limit_shift"] == pytest.approx(0.2 / 100.2, rel=1e-9)


def test_refine_measures_the_order_and_the_limit_of_synthetic_rows():
    rows = [
        {"panels": 64, "value": 100.0 - 4.0, "err": 0.04},
        {"panels": 96, "value": 100.0 - 4.0 * 64 / 96, "err": 0.04 * 64 / 96},
        {"panels": 128, "value": 100.0 - 4.0 * 64 / 128, "err": 0.04 * 64 / 128},
    ]
    payload = refine(rows, "value", "err")
    assert payload["order"] == pytest.approx(1.0)
    assert payload["limit"] == pytest.approx(100.0)
    assert len(payload["steps"]) == 2


def test_far_field_steps_recovers_the_order_of_a_halving_offset():
    rows = [{"panels": n, "far_over_near": 1.0 + 0.4 / n} for n in (40, 80, 160)]
    payload = far_field_steps(rows)
    assert payload["mean_order"] == pytest.approx(1.0)
    assert payload["offsets"] == pytest.approx([0.01, 0.005, 0.0025])


def test_mfs_reference_is_stable_in_its_own_parameters():
    """The reference must not move more than a fraction of the gap it judges."""
    payload = mfs_stability(3.2, "heave", 64, ka=KA)
    assert payload["mass_max_dev_from_default"] < 1.0e-3
    assert payload["damping_max_dev_from_default"] < 1.0e-3
    assert payload["mass_spread"] < 2.0e-3


def test_panel_error_against_mfs_is_first_order():
    rows = [panel_row(3.2, "heave", panels, ka=KA) for panels in (48, 64, 96)]
    damping = [abs(row["damping_rel_diff"]) for row in rows]
    mass = [abs(row["mass_rel_diff"]) for row in rows]
    assert damping[-1] < damping[0]
    assert mass[-1] < mass[0]
    # 96/64 = 1.5 and 64/48 = 4/3: use the real factor.
    order_damping = observed_order(damping[-2], damping[-1], 96 / 64)
    order_mass = observed_order(mass[-2], mass[-1], 96 / 64)
    assert 0.7 < order_damping < 1.4, order_damping
    assert 0.7 < order_mass < 1.4, order_mass


def test_energy_identity_offset_shrinks_with_panels():
    rows = [panel_row(3.2, "heave", panels, ka=KA) for panels in (48, 96)]
    offsets = [abs(row["far_over_near"] - 1.0) for row in rows]
    assert offsets[1] < offsets[0]
    assert offsets[1] < 2.0e-2
    assert far_field_steps(rows)["mean_order"] > 0.6


def test_mfs_reference_rejects_bad_input():
    mesh = submerged_circle_mesh(48)
    with pytest.raises(ValueError):
        radiation_reference(mesh, KA, 3.2, "heave", source_count=0)


@pytest.mark.skipif(not ARTEFACT.exists(), reason="benchmark artefact not present")
def test_committed_artefact_is_internally_consistent():
    payload = json.loads(ARTEFACT.read_text())
    assert payload["benchmark"] == "finite_depth_radiation_two_sided_convergence"
    assert payload["cases"], "no cases stored"
    for case in payload["cases"]:
        rows = sorted(case["rows"], key=lambda row: row["panels"])
        assert [row["panels"] for row in rows] == sorted(row["panels"] for row in rows)
        mass = refine(rows, "added_mass", "mass_rel_diff")
        damping = refine(rows, "damping", "damping_rel_diff")
        assert mass["order"] == pytest.approx(case["mass_order"], nan_ok=True)
        assert damping["order"] == pytest.approx(case["damping_order"], nan_ok=True)
        assert damping["limit"] == pytest.approx(case["damping_richardson"]["limit"], rel=1e-12)
        # The two independent discretisations must close at their limits.
        assert abs(case["mfs_damping_residual_at_limit"]) < 5.0e-3
        assert abs(case["mfs_mass_residual_at_limit"]) < 5.0e-3
        for row in rows:
            assert row["relative_residual"] < 1.0e-10
            assert row["condition_number"] > 1.0
            assert abs(row["far_over_near"] - 1.0) < 5.0e-2