"""Tests for the Ursell submerged-cylinder radiation benchmark.

The reference is Ursell's multipole solution. Three layers are tested:

* the special function and the infinite system on their own, against values
  that do not come from this code (published zeros of the damping, the deep
  submergence limit, the truncation behaviour);
* the free-surface boundary element solve, against that reference;
* the radiation condition, through the identity that the damping obtained from
  the surface pressure equals the damping obtained from the outgoing wave.
"""

import json
import math
import subprocess
import sys

import pytest

from tensorbem.analytic import (
    exponential_integral,
    ursell_cylinder_added_mass,
    ursell_cylinder_coefficients,
)
from tensorbem.benchmarks.ursell_cylinder import run, submerged_circle_mesh
from tensorbem.free_surface_solver import (
    far_field_damping,
    radiation_reference,
    solve_radiation_indirect,
)

RADIUS, SUBMERGENCE = 1.0, 2.0
NU = 0.6
DENSITY = 1025.0
GRAVITY = 9.80665
OMEGA = math.sqrt(NU * GRAVITY / RADIUS)


# --------------------------------------------------------------------------
# The special function
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argument, expected",
    [
        (1.0, 1.8951178163559368),
        (2.0, 4.954234356001890),
        (10.0, 2492.2289762418777),
        (20.0, 25615652.664056599),
        (30.0, 3.6897320940789e11),
    ],
)
def test_exponential_integral_matches_published_values(argument, expected):
    assert math.isclose(
        exponential_integral(argument), expected, rel_tol=1.0e-10
    )


def test_exponential_integral_rejects_bad_arguments():
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            exponential_integral(bad)


# --------------------------------------------------------------------------
# The infinite system
# --------------------------------------------------------------------------


def test_deep_submergence_recovers_the_displaced_mass():
    """With the free surface far away the added mass is simply rho*pi*a**2."""
    shallow_free_surface = ursell_cylinder_added_mass(
        RADIUS, SUBMERGENCE, OMEGA, DENSITY
    )
    deep = ursell_cylinder_added_mass(RADIUS, 100.0, OMEGA, DENSITY)
    displaced = DENSITY * math.pi * RADIUS**2
    assert math.isclose(deep[0], displaced, rel_tol=2.0e-3)
    assert abs(deep[1]) < 1.0e-6
    # The free surface lowers the added mass when the body is close to it.
    assert shallow_free_surface[0] < displaced


def test_truncation_is_converged():
    coarse = ursell_cylinder_coefficients(0.6, 0.5, terms=20)
    fine = ursell_cylinder_coefficients(0.6, 0.5, terms=40)
    assert math.isclose(coarse[0], fine[0], rel_tol=1.0e-12)
    assert math.isclose(coarse[1], fine[1], rel_tol=1.0e-12)


@pytest.mark.parametrize(
    "radius_over_submergence, published_ka",
    [(0.95, 1.50394), (0.9, 2.16016)],
)
def test_published_zero_damping_frequencies(radius_over_submergence, published_ka):
    """Ursell's damping vanishes at frequencies quoted in the literature.

    The values come from McIver & McIver's appendix to "Wave-free motion and
    trapped modes", which tabulates the first zero of the radiation damping for
    a submerged circular cylinder. They are independent of anything here.
    """
    _, damping = ursell_cylinder_coefficients(published_ka, radius_over_submergence)
    assert abs(damping) < 1.0e-8


def test_coefficient_validation():
    with pytest.raises(ValueError):
        ursell_cylinder_coefficients(0.0, 0.5)
    with pytest.raises(ValueError):
        ursell_cylinder_coefficients(0.6, 0.0)
    with pytest.raises(ValueError):
        ursell_cylinder_coefficients(0.6, 1.0)
    with pytest.raises(ValueError):
        ursell_cylinder_coefficients(0.6, 0.5, terms=0)
    with pytest.raises(ValueError):
        ursell_cylinder_added_mass(1.0, 0.5, OMEGA)


# --------------------------------------------------------------------------
# The boundary element solution
# --------------------------------------------------------------------------


def test_panel_added_mass_approaches_the_ursell_value():
    errors = []
    for panels in (32, 64, 128):
        result = run(panels, RADIUS, SUBMERGENCE, NU, DENSITY, GRAVITY)
        errors.append(result["panel_added_mass_relative_error"])
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1.0e-2


def test_radiation_condition_energy_identity():
    """Near-field and far-field damping must agree; that is the radiation test."""
    mesh = submerged_circle_mesh(64, RADIUS, SUBMERGENCE)
    result = solve_radiation_indirect(mesh, NU, "heave", density=DENSITY)
    far = far_field_damping(mesh, result.potential, result.normal_velocity, NU, DENSITY)
    assert math.isclose(result.damping[1], far, rel_tol=2.0e-2)


def test_heave_and_sway_coefficients_coincide_in_infinite_depth():
    """Different fields, same force: the classical two-dimensional symmetry."""
    mesh = submerged_circle_mesh(48, RADIUS, SUBMERGENCE)
    heave = solve_radiation_indirect(mesh, NU, "heave", density=DENSITY)
    sway = solve_radiation_indirect(mesh, NU, "sway", density=DENSITY)
    assert math.isclose(heave.added_mass[1], sway.added_mass[0], rel_tol=1.0e-10)
    assert math.isclose(heave.damping[1], sway.damping[0], rel_tol=1.0e-10)
    # The motions are not the same field: only their forces coincide.
    assert not math.isclose(
        abs(heave.potential[0]), abs(sway.potential[0]), rel_tol=1.0e-6
    )
    # And the force is along the direction of motion.
    assert abs(heave.added_mass[0]) < 1.0e-6 * abs(heave.added_mass[1])


def test_method_of_fundamental_solutions_agrees_with_ursell():
    mesh = submerged_circle_mesh(64, RADIUS, SUBMERGENCE)
    reference = radiation_reference(mesh, NU, "heave", density=DENSITY)
    exact_added_mass, exact_damping = ursell_cylinder_added_mass(
        RADIUS, SUBMERGENCE, OMEGA, DENSITY
    )
    assert math.isclose(reference.added_mass[1], exact_added_mass, rel_tol=3.0e-3)
    assert math.isclose(reference.damping[1], exact_damping, rel_tol=1.0e-2)


# --------------------------------------------------------------------------
# The benchmark entry point
# --------------------------------------------------------------------------


def test_run_reports_a_consistent_picture():
    row = run(64, RADIUS, SUBMERGENCE, NU, DENSITY, GRAVITY)
    assert row["benchmark"] == "ursell_submerged_circular_cylinder_deep_water"
    assert row["radius_over_submergence"] == pytest.approx(0.5)
    assert row["panel_relative_residual"] < 1.0e-10
    assert row["near_far_field_relative_difference"] < 2.0e-2
    assert row["panel_heave_sway_added_mass_relative_difference"] < 1.0e-10
    assert row["reference_solution_added_mass_relative_error"] < 5.0e-3
    assert row["panel_added_mass_over_displaced_mass"] == pytest.approx(
        row["panel_added_mass"] / (DENSITY * math.pi * RADIUS**2)
    )
    json.dumps(row, allow_nan=False)


def test_run_rejects_an_emerged_cylinder():
    with pytest.raises(ValueError):
        run(32, 1.0, 0.5, NU)


def test_cli_writes_strict_json(tmp_path):
    output = tmp_path / "ursell.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.ursell_cylinder",
            "--panels",
            "32",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "benchmark" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["reference_method"] == "ursell_multipole_closed_form_infinite_system"
    assert payload["panel_added_mass"] > 0.0
