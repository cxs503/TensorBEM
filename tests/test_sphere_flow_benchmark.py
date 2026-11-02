"""Acceptance tests for the uniform-flow-past-a-sphere verification case.

The reference values are the classical closed forms (Lamb 1932, §103):

* ``Cp(theta) = 1 - (9/4) sin(theta)**2`` on the sphere,
* ``Cp = 1`` at the stagnation points, ``Cp = -5/4`` on the equator,
* zero pressure resultant (d'Alembert's paradox).

The force integrator itself is checked against the two analytic identities of
:mod:`tensorbem.pressure`: Archimedes with the signed mesh volume and the
vanishing resultant of a uniform pressure.  No reference here is produced by
the code under test.
"""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem.benchmarks.sphere_flow import (
    exact_pressure_coefficient,
    run,
    run_study,
)

# Measured at the committed mesh levels; thresholds leave margin for BLAS and
# platform round-off but are tight enough to reject a regression of the order
# of the discretisation error itself.
CP_RELATIVE_L2_LEVEL1 = 0.10  # measured 8.0e-2 at 80 panels
CP_RELATIVE_L2_LEVEL2 = 0.035  # measured 2.66e-2 at 320 panels
CP_MAX_ABS_LEVEL2 = 0.08  # measured 5.7e-2 at 320 panels


@pytest.fixture(scope="module")
def level1():
    return run(1)


@pytest.fixture(scope="module")
def level2():
    return run(2)


def test_exact_pressure_coefficient_has_the_classical_extrema():
    assert exact_pressure_coefficient(np.array([0.0]))[0] == pytest.approx(1.0)
    assert exact_pressure_coefficient(np.array([1.0]))[0] == pytest.approx(-1.25)
    grid = np.linspace(0.0, 1.0, 17)
    # Cp must be monotone decreasing in sin(theta) and bounded by the extrema.
    assert np.all(np.diff(exact_pressure_coefficient(grid)) < 0.0)
    with pytest.raises(ValueError):
        exact_pressure_coefficient(np.array([np.nan]))


def test_cp_field_matches_the_closed_form(level1, level2):
    assert level1["cp_relative_l2_error"] < CP_RELATIVE_L2_LEVEL1
    assert level2["cp_relative_l2_error"] < CP_RELATIVE_L2_LEVEL2
    assert level2["cp_max_abs_error"] < CP_MAX_ABS_LEVEL2
    assert level2["boundary_potential_relative_l2_error"] < 5.0e-3


def test_stagnation_and_equator_pressure_coefficients(level2):
    # Cp = 1 at both stagnation points and Cp = -5/4 on the equator.
    assert abs(level2["cp_computed_at_stagnation"] - 1.0) < 0.10
    assert abs(level2["cp_computed_at_equator"] - (-1.25)) < 0.05
    assert level2["cp_error_at_stagnation_points"] < 0.10


def test_dalembert_pressure_resultant_is_negligible(level2):
    coefficients = level2["force_coefficient"]
    # Exact potential-flow value is zero.  The residual is dominated by the
    # faceted-mesh symmetry, so this bounds the integrator/BEM chain rather
    # than certifying the Cp amplitude (which the field test does).
    assert coefficients["magnitude"] < 1.0e-6
    assert level2["force_coefficient_from_exact_cp"]["magnitude"] < 1.0e-9
    for component in ("cx", "cy", "cz"):
        assert abs(coefficients[component]) < 1.0e-6


def test_pressure_integrator_identities_on_the_same_mesh(level1):
    # Archimedes and the uniform-pressure resultant are exact identities of the
    # integrator; they must hold to round-off on a closed faceted mesh.
    assert level1["archimedes_relative_error"] < 1.0e-12
    assert level1["uniform_pressure_force_norm_over_pressure_area"] < 1.0e-12
    assert level1["uniform_pressure_moment_norm_over_pressure_area_length"] < 1.0e-12


def test_cp_error_decreases_under_refinement(level1, level2):
    ratio = level1["cp_relative_l2_error"] / level2["cp_relative_l2_error"]
    h_ratio = level1["h_mean"] / level2["h_mean"]
    observed_order = math.log(ratio) / math.log(h_ratio)
    assert ratio > 2.0
    assert observed_order > 1.2
    assert level2["panels"] > level1["panels"]


def test_conditioning_diagnostics_are_reported(level2):
    # The second-kind Neumann system stays O(1) while the single-layer operator
    # is far worse conditioned: this is the numerical reason the Neumann
    # formulation is used for the surface solve.
    assert level2["second_kind_system_condition_number"] < 10.0
    assert level2["single_layer_condition_number"] > 10.0
    assert level2["single_layer_condition_number"] > level2[
        "second_kind_system_condition_number"
    ]
    assert level2["relative_residual_l2"] < 1.0e-12


def test_run_rejects_invalid_parameters():
    with pytest.raises(TypeError):
        run(True)
    with pytest.raises(ValueError):
        run(-1)
    with pytest.raises(ValueError):
        run(1, radius=0.0)
    with pytest.raises(ValueError):
        run(1, speed=-2.0)
    with pytest.raises(ValueError):
        run(1, density=float("nan"))
    with pytest.raises(TypeError):
        run(1, speed="fast")


def test_study_rejects_invalid_levels():
    with pytest.raises(ValueError):
        run_study([])
    with pytest.raises(ValueError):
        run_study([1, 1])
    with pytest.raises(ValueError):
        run_study([-1, 1])


def test_study_reports_observed_orders():
    study = run_study([0, 1])
    assert study["levels"] == [0, 1]
    assert [row["subdivisions"] for row in study["rows"]] == [0, 1]
    assert len(study["orders"]) == 1
    order = study["orders"][0]
    assert order["panels"] == [20, 80]
    assert order["h_mean_ratio"] > 1.0
    assert math.isfinite(order["cp_error_observed_order"])
    assert order["cp_error_observed_order"] > 0.8


def test_cli_writes_strict_json(tmp_path):
    output = tmp_path / "sphere-flow.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.sphere_flow",
            "--subdivisions",
            "0",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "uniform_stream_past_sphere_study"
    assert payload["rows"][0]["panels"] == 20
    assert json.loads(completed.stdout) == payload
    # NaN/Infinity are not valid JSON and must not be emitted.
    assert "NaN" not in output.read_text()
    assert "Infinity" not in output.read_text()


def test_optional_plot(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from tensorbem.benchmarks.sphere_flow import plot_study

    output = tmp_path / "sphere-flow.png"
    plot_study(run_study([0, 1]), output)
    assert output.exists() and output.stat().st_size > 0