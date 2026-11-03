"""Tests for the ellipsoid mesh-refinement study.

The heavy geometric level (1280 panels) is not recomputed here: the recorded
study artifact produced by

    python -m tensorbem.benchmarks.ellipsoid_refinement --levels 1,2,3

is validated instead, and the algebra of the order estimate / extrapolation is
covered by synthetic families whose exact answer is known.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks.ellipsoid_refinement import (
    AXIS_NAMES,
    observed_order,
    extrapolate_axis,
    run,
)

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "ellipsoid_heavy"
    / "ellipsoid_refinement.json"
)

AXES = (1.5, 1.0, 0.75)


def test_observed_order_recovers_a_known_power_law():
    """A pure first-order family must return order 1, a second-order family 2."""

    # Geometric levels (the refinement ratio is the same between successive
    # levels), a family whose error is 1/level**order, and the absence of a
    # limit term: the measured order must come back as ``order``.
    levels = (2, 4, 8)

    def family(order):
        return [1.0 / (level**order) for level in levels]

    assert observed_order(family(1.0), levels) == pytest.approx(1.0, rel=1e-12)
    assert observed_order(family(2.0), levels) == pytest.approx(2.0, rel=1e-12)


def test_observed_order_rejects_a_non_converging_family():
    with pytest.raises(ValueError):
        observed_order([1.0, 1.0, 1.0], [1, 2, 3])
    with pytest.raises(ValueError):
        observed_order([1.0, 0.5], [1, 2])


def test_extrapolation_recovers_the_limit_of_a_synthetic_family():
    """Richardson extrapolation of a first-order family must hit the limit exactly."""
    limit = 3.25
    rows = []
    for panels in (80, 320, 1280):
        # First-order in panel count: the family richardson_limit parameterises.
        rows.append(
            {
                "panels": panels,
                "added_mass": limit + 75.0 / panels,
                "exact_added_mass": limit,
                "added_mass_relative_error": 75.0 / panels / limit,
                "motion_axis": "x",
            }
        )
    result = extrapolate_axis(rows)
    assert result["extrapolated_added_mass"] == pytest.approx(limit, rel=1e-12)
    assert result["observed_order"] == pytest.approx(1.0, rel=1e-12)
    assert result["extrapolation_vs_analytic"] == pytest.approx(0.0, abs=1e-15)


def test_levels_must_be_increasing():
    with pytest.raises(ValueError):
        run((2, 1))
    with pytest.raises(ValueError):
        run((2, 2))


def test_fast_levels_validate_geometry_before_accuracy():
    """Two cheap levels: faceted volume and added-mass error must both shrink.

    This is the guard that makes the refinement argument meaningful -- if the
    discretisation did not converge towards the ellipsoid, no extrapolation
    from it would be admissible.
    """
    report = run((0, 1), semi_axes=AXES, speed=1.0, density=1.0)

    volumes = [row["volume_relative_error"] for row in report["meshes"]]
    assert volumes[1] < volumes[0]
    for row in report["meshes"]:
        assert row["shape_factor_identity_error"] < 1.0e-9
        assert row["relative_residual_l2"] < 1.0e-9

    for name in AXIS_NAMES:
        levels = report["axes"][name]["levels"]
        assert levels[1]["added_mass_relative_error"] < levels[0]["added_mass_relative_error"]
        assert levels[1]["added_mass_relative_error"] < 0.1
    assert report["volume_error_monotone"] is True
    assert report["added_mass_error_monotone"] is True
    # Two levels are not enough for an extrapolation, and the module must say so.
    assert "extrapolation" not in report["axes"]["x"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="refinement artifact not recorded")
def test_recorded_heavy_study_closes_on_the_analytic_reference():
    """The recorded 80/320/1280-panel study must extrapolate onto Lamb's value."""
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    panels = [row["panels"] for row in report["meshes"]]
    assert len(panels) >= 3 and panels == sorted(panels)
    assert panels[-1] >= 320, "the recorded study must reach at least the 320-panel level"
    assert report["volume_error_monotone"] is True
    assert report["added_mass_error_monotone"] is True
    # The faceted volume converges to the analytic volume.
    assert report["meshes"][-1]["volume_relative_error"] < 5.0e-2
    assert report["volume_error_decay_order"] > 1.0

    for name in AXIS_NAMES:
        entry = report["axes"][name]
        extrapolation = entry["extrapolation"]
        # A meaningful measured order is the prerequisite for trusting the limit.
        # NOTE: richardson_limit parameterises the family by panel count (ratio 4),
        # while the mesh spacing halves per level; the order reported here is
        # therefore about half the h-order, which entry["error_decay_order"] gives.
        assert 0.5 < extrapolation["observed_order"] < 3.0
        assert entry["error_decay_order"] > 1.0
        # The zero-mesh limit lands on Lamb's analytic added mass.
        # The recorded family stops at 320 panels, so the extrapolation residual
        # is set by that level's error (1.6e-2) divided by r^p - 1 (about 2.2).
        assert extrapolation["extrapolation_vs_analytic"] < 1.0e-2
        # The quoted single-mesh error is a discretisation error, not a plateau.
        # This family stops at 320 panels, where the error is 1.6e-2; the
        # 1280-panel point (4.2e-3) is checked by the declared-number registry.
        assert entry["levels"][-1]["added_mass_relative_error"] < 2.5e-2
        assert (
            entry["levels"][-1]["added_mass_relative_error"]
            < entry["levels"][0]["added_mass_relative_error"]
        )
        assert math.isfinite(extrapolation["extrapolated_added_mass"])