"""Tests for the surface-velocity accuracy study."""
from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from tensorbem.benchmarks.surface_velocity_study import run_study


@pytest.fixture(scope="module")
def study() -> dict:
    return run_study([1, 2])


def test_study_reports_a_finite_strict_json_document(study):
    rendered = json.dumps(study, sort_keys=True, allow_nan=False)
    assert json.loads(rendered) == study
    for row in study["cases"]["linear_trace"]["results"]:
        assert math.isfinite(row["representation_relative_l2_error"])
        assert math.isfinite(row["recovered_relative_l2_error"])


def test_study_checks_all_pass(study):
    failed = {name: value for name, value in study["checks"].items() if not value}
    assert not failed, f"failing study checks: {failed}"


def test_constant_boundary_value_gives_an_exact_surface_velocity(study):
    # A boundary value without jumps is represented exactly by the constant
    # panels, so both paths must be exact.
    for row in study["cases"]["constant_trace"]["results"]:
        assert row["recovered_relative_l2_error"] < 1.0e-12
        assert row["representation_relative_l2_error"] < 1.0e-12


def test_representation_error_is_relative_and_not_an_offset(study):
    errors = [
        row["representation_relative_l2_error"]
        for row in study["cases"]["scaled_linear_trace"]["results"]
    ]
    assert max(errors) - min(errors) < 1.0e-9
    amplitudes = [
        row["amplitude"] for row in study["cases"]["scaled_linear_trace"]["results"]
    ]
    assert amplitudes == [1.0, 10.0, 100.0]


def test_representation_error_is_not_a_quadrature_order_effect(study):
    orders = [
        row["quadrature_order"]
        for row in study["cases"]["quadrature_order"]["results"]
    ]
    assert orders == [8, 16, 32, 48]
    errors = [
        row["representation_relative_l2_error"]
        for row in study["cases"]["quadrature_order"]["results"]
    ]
    # Converged to a relative 1e-8, eight orders of magnitude below the
    # error being measured.
    assert max(errors[1:]) - min(errors[1:]) < 1.0e-8


def test_representation_error_does_not_vanish_under_refinement(study):
    errors = [
        row["representation_relative_l2_error"]
        for row in study["cases"]["linear_trace"]["results"]
    ]
    assert errors[-1] > 0.25 * errors[0]
    for order in study["observed_orders"]["representation_linear_trace"]:
        assert abs(order) < 0.5


def test_recovered_error_converges_under_refinement(study):
    errors = [
        row["recovered_relative_l2_error"]
        for row in study["cases"]["linear_trace"]["results"]
    ]
    assert errors[0] > errors[1]
    for order in study["observed_orders"]["recovered_linear_trace"]:
        assert order > 1.0


def test_recovered_path_beats_the_representation_path(study):
    for row in study["cases"]["translating_sphere"]["results"]:
        assert row["recovered_relative_l2_error"] < 0.2
        assert row["recovered_relative_l2_error"] < 0.3 * row[
            "representation_relative_l2_error"
        ]


def test_study_rejects_invalid_levels():
    with pytest.raises(ValueError):
        run_study([1])
    with pytest.raises(ValueError):
        run_study([1, 1])
    with pytest.raises(ValueError):
        run_study([-1, 1])
    with pytest.raises(ValueError):
        run_study([1, 2.5])


def test_study_command_line_writes_a_strict_json_report(tmp_path):
    output = tmp_path / "study.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.surface_velocity_study",
            "--levels",
            "1",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["study"] == "tensorbem_surface_velocity_accuracy"
    assert all(report["checks"].values())
