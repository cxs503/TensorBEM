import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem.benchmarks.sphere_added_mass import run
from tensorbem.benchmarks.three_dimensional_study import plot_study, run_study


def test_sphere_added_mass_converges_to_analytic_value():
    coarse = run(1)
    fine = run(2)
    assert fine["added_mass_relative_error"] < coarse["added_mass_relative_error"]
    assert fine["added_mass_relative_error"] < 2.0e-2
    assert (
        fine["normal_derivative_relative_l2_error"]
        < coarse["normal_derivative_relative_l2_error"]
    )
    assert fine["double_layer_row_sum_error"] < 1.0e-12
    assert 0.9 < fine["surface_area_ratio"] < 1.0
    assert 0.9 < fine["volume_ratio"] < 1.0
    assert fine["relative_residual_l2"] < 1.0e-12


def test_sphere_added_mass_scales_with_density_and_radius_cubed():
    base = run(2, radius=1.0, density=1000.0)
    scaled = run(2, radius=2.0, density=500.0)
    assert np.isclose(
        scaled["exact_added_mass"], 4.0 * base["exact_added_mass"], rtol=1.0e-12
    )
    assert np.isclose(scaled["added_mass"], 4.0 * base["added_mass"], rtol=1.0e-8)


@pytest.mark.parametrize(
    "subdivisions,radius,density,speed",
    [
        (-1, 1.0, 1025.0, 1.0),
        (1.5, 1.0, 1025.0, 1.0),
        (True, 1.0, 1025.0, 1.0),
        (1, 0.0, 1025.0, 1.0),
        (1, 1.0, -1.0, 1.0),
        (1, 1.0, 1025.0, np.nan),
    ],
)
def test_sphere_benchmark_rejects_invalid_parameters(subdivisions, radius, density, speed):
    with pytest.raises(ValueError):
        run(subdivisions, radius=radius, density=density, speed=speed)


def test_sphere_benchmark_cli_writes_strict_json(tmp_path):
    output = tmp_path / "nested" / "sphere.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.sphere_added_mass",
            "--subdivisions",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(completed.stdout)
    assert stdout == json.loads(output.read_text())
    assert stdout["benchmark"] == "three_dimensional_translating_sphere"


@pytest.mark.parametrize(
    "levels",
    [[], [0], [0, 0], [-1, 0], [0, 1.0], [True, 0]],
)
def test_study_rejects_invalid_levels(levels):
    with pytest.raises(ValueError):
        run_study(levels)


def test_study_orders_checks_and_strict_json():
    study = run_study([2, 1])
    assert study["levels"] == [1, 2]
    assert json.loads(json.dumps(study, allow_nan=False)) == study
    assert all(study["checks"].values())
    for case in study["cases"].values():
        assert len(case["results"]) == 2
        for metric in case["checks"]:
            assert case["checks"][metric] is True
    added_mass_rows = study["cases"]["exterior_translating_sphere"]["results"]
    assert added_mass_rows[0]["added_mass_relative_error_order"] is None
    assert added_mass_rows[1]["added_mass_relative_error_order"] is not None
    assert added_mass_rows[1]["added_mass_relative_error"] < 2.0e-2
    assert len(study["interior_probe_points"]) == 81


def test_study_cli_writes_json(tmp_path):
    output = tmp_path / "study.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.three_dimensional_study",
            "--levels",
            "0",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == json.loads(output.read_text())


def test_optional_study_plot(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    output = tmp_path / "figures" / "three_dimensional.svg"
    plot_study(run_study([0, 1]), output)
    assert "<svg" in output.read_text()
