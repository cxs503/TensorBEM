"""Tests for the translating-ellipse added-mass benchmark."""
import json
import math
import subprocess
import sys

import pytest

from tensorbem.benchmarks.ellipse_added_mass import run

MAJOR, MINOR = 1.5, 0.75


def test_added_mass_converges_to_pi_b_squared_for_axial_motion():
    rows = [run(n, MAJOR, MINOR, 1.0, 1.0, "x") for n in (16, 32, 64)]
    errors = [row["added_mass_relative_error"] for row in rows]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 5.0e-3
    assert math.isclose(
        rows[-1]["exact_added_mass_per_unit_span"],
        math.pi * MINOR**2,
        rel_tol=1.0e-15,
    )
    # An inscribed polygon under-predicts the enclosed area, hence the added mass.
    assert rows[-1]["added_mass_per_unit_span"] < rows[-1][
        "exact_added_mass_per_unit_span"
    ]


def test_added_mass_converges_to_pi_a_squared_for_transverse_motion():
    rows = [run(n, MAJOR, MINOR, 1.0, 1.0, "y") for n in (16, 32, 64)]
    errors = [row["added_mass_relative_error"] for row in rows]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 5.0e-3
    assert math.isclose(
        rows[-1]["exact_added_mass_per_unit_span"],
        math.pi * MAJOR**2,
        rel_tol=1.0e-15,
    )


def test_trace_scale_and_area_ratio_are_reported_consistently():
    row = run(64, MAJOR, MINOR, 1.0, 1.0, "x")
    assert math.isclose(row["boundary_trace_scale"], MINOR / MAJOR, rel_tol=1.0e-15)
    assert math.isclose(row["exact_area"], math.pi * MAJOR * MINOR, rel_tol=1.0e-15)
    assert 0.98 < row["area_ratio"] < 1.0
    assert row["relative_residual_l2"] < 1.0e-12
    assert row["single_layer_condition_number"] > 0.0


def test_circle_limit_reproduces_the_cylinder_benchmark():
    # With equal semi-axes the ellipse becomes a circle, and both motion axes
    # must give rho*pi*a**2, matching the existing cylinder benchmark.
    radius, density = 1.0, 1025.0
    for axis in ("x", "y"):
        row = run(64, radius, radius, 1.0, density, axis)
        assert math.isclose(
            row["exact_added_mass_per_unit_span"],
            density * math.pi * radius**2,
            rel_tol=1.0e-15,
        )
        assert row["added_mass_relative_error"] < 5.0e-3


def test_benchmark_is_a_strict_json_document():
    row = run(16, MAJOR, MINOR, 1.0, 1025.0, "x")
    rendered = json.dumps(row, sort_keys=True, allow_nan=False)
    assert json.loads(rendered) == row


def test_benchmark_rejects_invalid_input():
    with pytest.raises(ValueError, match="axis"):
        run(16, MAJOR, MINOR, axis="z")
    with pytest.raises(ValueError, match="speed"):
        run(16, MAJOR, MINOR, speed=0.0)
    with pytest.raises(ValueError, match="density"):
        run(16, MAJOR, MINOR, density=float("nan"))
    with pytest.raises(ValueError, match="panel_count"):
        run(3.5, MAJOR, MINOR)


def test_command_line_writes_a_strict_json_report(tmp_path):
    output = tmp_path / "ellipse.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.ellipse_added_mass",
            "--panels",
            "32",
            "--axis",
            "y",
            "--density",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["benchmark"] == "two_dimensional_translating_ellipse_y"
    assert report["motion_axis"] == "y"
    assert report["panels"] == 32
