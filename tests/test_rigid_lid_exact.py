"""Gates for the exact rigid-lid referee (``tensorbem.rigid_lid``).

Why the referee needs its own gates
-----------------------------------
``CASE-BEM-RIGID-LID-HEMISPHERE`` compares the 3-D radiation solver against the
``ka -> 0`` (even-parity double body) heave added mass of a floating hemisphere.
The solver error under test is a fraction of a percent, so the reference itself
must be resolved far below that.  The case driver carries two formulations of
the double body whose disagreement is ``3e-6`` -- the Gauss-Legendre integration
of the kinked ``|mu|`` near ``mu = 0``, not physics.  :mod:`tensorbem.rigid_lid`
removes that error by integrating ``int_0^1 mu P_n(mu) dmu`` in exact rational
arithmetic, so what is defended here is:

1. the moments are exact (closed forms ``J_0 = 1/2, J_1 = 1/3, J_2 = 1/8``,
   ``J_3 = 0`` reproduced with zero residual);
2. the series converges monotonically to ``0.83095132..`` with a tail below
   ``1e-9`` by ``degree_max = 320``, and agrees with both driver routes within
   ``1e-6``;
3. the odd-parity control (pressure release ``ka -> infinity``) is exactly
   ``0.5``, which is what separates the two limits that report #15809 conflated;
4. the solver module exposes the referee at its own low-frequency limit
   (:func:`tensorbem.free_surface_solver3d.verify_rigid_lid_heave_limit`) -- the
   integration caller of the case, i.e. a live BEM solve scored against it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import pathlib

import pytest

from tensorbem import rigid_lid

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_half_moments_are_exact() -> None:
    half = rigid_lid.half_moments_exact(40)
    assert half["closed_form_residual"] == 0.0
    J = half["J"]
    # J_0 = 1/2, J_1 = 1/3, J_2 = 1/8, J_3 = 0 exactly
    assert [float(J[0]), float(J[1]), float(J[2]), float(J[3])] == pytest.approx(
        [0.5, 1.0 / 3.0, 0.125, 0.0], abs=0.0
    )


def test_series_is_converged_and_monotone() -> None:
    coarse = rigid_lid.rigid_lid_added_mass_dimless(160)
    fine = rigid_lid.rigid_lid_added_mass_dimless(320)
    ladder = [item["value_dimless"] for item in fine["truncation_ladder"]]
    assert ladder == sorted(ladder), "even-mode partial sums must increase"
    assert fine["value_dimless"] - coarse["value_dimless"] < 1.0e-9
    assert fine["value_dimless"] == pytest.approx(0.83095132, abs=1.0e-8)
    assert fine["truncation_tail_relative"] < 1.0e-10


def test_pressure_release_control_is_exactly_one_half() -> None:
    control = rigid_lid.pressure_release_added_mass_dimless()
    assert control["exact"] == "1/2"
    assert control["value_dimless"] == 0.5


def test_exact_route_agrees_with_the_case_driver_routes() -> None:
    from tensorbem.benchmarks import case_bem_rigid_lid_hemisphere as case

    reference = case.rigid_lid_reference(degree_max=60)
    exact = reference["routes"]["exact_rational"]["dimless"]
    assert reference["exact_vs_energy_relative"] < 1.0e-6
    assert abs(exact - reference["value_dimless"]) / exact < 1.0e-6
    assert abs(reference["routes"]["force"]["dimless"] - exact) / exact < 1.0e-5
    assert exact == pytest.approx(rigid_lid.rigid_lid_added_mass_dimless(160)["value_dimless"])


def test_solver_module_exposes_the_referee() -> None:
    from tensorbem import free_surface_solver3d as solver

    assert solver.rigid_lid_heave_added_mass_dimless() == pytest.approx(
        rigid_lid.rigid_lid_added_mass_dimless(160)["value_dimless"], rel=1.0e-15
    )


def test_live_solve_scores_against_the_exact_endpoint() -> None:
    from tensorbem.free_surface_solver3d import verify_rigid_lid_heave_limit

    report = verify_rigid_lid_heave_limit(level=2, ka=1.0e-3)
    assert report["panels"] == 64
    assert report["within_tolerance"] is True, report
    # the 64-panel row of the committed case, re-solved live
    assert report["relative_error"] < 0.0
    assert abs(report["relative_error"]) < case_tolerance()


def case_tolerance() -> float:
    from tensorbem.benchmarks.case_bem_rigid_lid_hemisphere import TOLERANCE

    return TOLERANCE


def test_module_cli_emits_the_reference() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.rigid_lid", "--degrees", "40"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "rigid-lid" in completed.stdout or "CASE-BEM-RIGID-LID-HEMISPHERE" in completed.stdout