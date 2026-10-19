"""Regression tests for the reproducible 2-D refinement evidence."""
import math

from tensorbem.benchmarks.refinement_study import run_study


def test_refinement_study_has_finite_values_and_decreasing_added_mass_error():
    report = run_study([16, 32, 64, 128])
    assert report["panel_counts"] == [16, 32, 64, 128]
    assert report["checks"]["all_outputs_finite"] is True
    assert report["checks"]["cylinder_added_mass_error_decreases"] is True

    cylinder = report["translating_cylinder_added_mass"]
    errors = [row["added_mass_relative_error"] for row in cylinder]
    assert all(math.isfinite(error) and error >= 0.0 for error in errors)
    assert errors[-1] < 0.02
    assert all(b < a for a, b in zip(errors, errors[1:]))

    for row in report["unit_disk_manufactured_solution"]:
        assert math.isfinite(row["residual_l2"])
        assert math.isfinite(row["single_layer_condition_number"])
        assert row["single_layer_condition_number"] > 0.0
