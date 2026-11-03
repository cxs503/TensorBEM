"""Regression pins for the reported Ursell benchmark numbers.

Why this file exists
--------------------
The marine benchmark report quoted a 128-panel added-mass relative error of
`7.72e-3` for the Ursell radiation case.  That number was correct for the code
as it stood when the table was written, but a later refactor of the indirect
free-surface solver changed the panel solution by ~0.08%, and the quoted error
became `8.51e-3` while the table stayed at `7.72e-3`.  Nothing failed: the
existing tests only assert *trends* ("the error decreases and is below 1e-2"),
so a 10.6% drift in a published number went unnoticed until an external audit
re-ran the command.

This module closes that gap in two layers:

* :func:`test_panel_solution_values_are_pinned` records the actual numbers the
  current scheme produces, so any change to the kernel, the quadrature or the
  added-mass post-processing makes the suite fail loudly instead of silently
  invalidating the documentation;
* :func:`test_study_artifact_matches_the_current_code` re-derives the committed
  evidence artifact (:mod:`examples.ursell_reproducibility_study`) so the table
  in the report cannot drift away from the code it claims to describe.

Both use the 64-panel case, which costs a few seconds and shifts in exactly the
same way as the 128/256-panel cases (the scheme is first order, so every
resolution is sensitive to the same discrete operator).
"""

from __future__ import annotations

import json
import math

import pytest

from tensorbem.benchmarks.ursell_cylinder import run

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY_JSON = ROOT / "outputs" / "ursell_reproducibility_study.json"

BASE = dict(radius=1.0, submergence=2.0, ka=0.6, density=1025.0, gravity=9.80665)

# Recorded 2026-11-03 from `run(64, **BASE)`; the closed-form Ursell reference
# in the same run was 2545.167247813704 / 2718.153951572775, which is an
# independent analytical construction and is pinned separately below.
PINNED_64 = {
    "panel_added_mass": 2587.756,
    "panel_damping": 2787.240,
    "panel_added_mass_relative_error": 1.6736e-2,
    "panel_damping_relative_error": 2.5415e-2,
}


@pytest.fixture(scope="module")
def result64() -> dict:
    return run(64, **BASE)


def test_panel_solution_values_are_pinned(result64: dict) -> None:
    """The reported panel-solution numbers must not drift silently."""
    assert result64["panel_added_mass"] == pytest.approx(
        PINNED_64["panel_added_mass"], rel=1.0e-4
    )
    assert result64["panel_damping"] == pytest.approx(
        PINNED_64["panel_damping"], rel=1.0e-4
    )
    assert result64["panel_added_mass_relative_error"] == pytest.approx(
        PINNED_64["panel_added_mass_relative_error"], rel=1.0e-4
    )
    assert result64["panel_damping_relative_error"] == pytest.approx(
        PINNED_64["panel_damping_relative_error"], rel=1.0e-4
    )


def test_reference_is_the_analytical_multipole_solution(result64: dict) -> None:
    """The reference side of the table must never move: it is closed form."""
    assert result64["reference_added_mass"] == pytest.approx(2545.167247813704, rel=1.0e-12)
    assert result64["reference_damping"] == pytest.approx(2718.153951572775, rel=1.0e-12)


def test_smooth_part_quadrature_saturates_above_six_nodes() -> None:
    """`panel_nodes` refines nothing above 6: the smooth part caps at 6 nodes.

    This is a documented design choice (the free-surface remainder is analytic,
    only the logarithmic Rankine part needs an exact primitive), but it means a
    reader cannot improve the panel solution by raising `panel_nodes`.  Pinning
    the saturation keeps the documentation honest if the cap ever changes.
    """
    coarse = run(32, panel_nodes=6, **BASE)
    fine = run(32, panel_nodes=24, **BASE)
    assert coarse["panel_added_mass"] == pytest.approx(
        fine["panel_added_mass"], rel=1.0e-12
    )
    assert coarse["panel_damping"] == pytest.approx(fine["panel_damping"], rel=1.0e-12)


def test_study_artifact_matches_the_current_code(result64: dict) -> None:
    """The committed study artifact must still describe the committed code."""
    if not STUDY_JSON.exists():  # pragma: no cover - artifact is committed
        pytest.skip("reproducibility study artifact not present")
    payload = json.loads(STUDY_JSON.read_text())
    row = next(r for r in payload["panel_sweep"] if r["panels"] == 64)

    assert row["added_mass"] == pytest.approx(result64["panel_added_mass"], rel=1.0e-6)
    assert row["damping"] == pytest.approx(result64["panel_damping"], rel=1.0e-6)
    assert row["added_mass_error"] == pytest.approx(
        result64["panel_added_mass_relative_error"], rel=1.0e-6
    )
    assert row["damping_error"] == pytest.approx(
        result64["panel_damping_relative_error"], rel=1.0e-6
    )
    assert payload["reference_added_mass"] == pytest.approx(2545.167247813704, rel=1.0e-12)
    assert payload["reference_damping"] == pytest.approx(2718.153951572775, rel=1.0e-12)


def test_recorded_convergence_is_first_order() -> None:
    """The study artifact must show the first-order trend the report claims."""
    if not STUDY_JSON.exists():  # pragma: no cover
        pytest.skip("reproducibility study artifact not present")
    payload = json.loads(STUDY_JSON.read_text())
    rows = payload["panel_sweep"]
    panels = [r["panels"] for r in rows]
    assert panels == [64, 128, 256]

    errors = [r["added_mass_error"] for r in rows]
    for coarse, fine in zip(errors, errors[1:]):
        ratio = coarse / fine
        assert 1.7 < ratio < 2.3, f"expected first order, got ratio {ratio:.3f}"

    ratios = [r["near_over_far"] for r in rows]
    assert abs(ratios[-1] - 1.0) < 5.0e-3
    assert abs(ratios[0] - 1.0) > abs(ratios[-1] - 1.0)


def test_accuracy_meets_the_acceptance_standard(result64: dict) -> None:
    """Both reported quantities stay inside the 3% acceptance band."""
    assert result64["panel_added_mass_relative_error"] < 3.0e-2
    assert result64["panel_damping_relative_error"] < 3.0e-2
    assert math.isfinite(result64["near_over_far_field_damping"])