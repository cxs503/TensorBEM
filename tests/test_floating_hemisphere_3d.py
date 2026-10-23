"""Verification tests for Hulme (1982) floating-hemisphere radiation, diffraction, and CHIEF."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.benchmarks.floating_hemisphere import run_refinement_study
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh


def test_hulme_reference_limits_tables_and_reciprocity() -> None:
    low = hulme_hemisphere_radiation(0.001, radius=1.0, density=1025.0)
    # Zero-frequency limits (Hulme 1982: A11' -> 0.5, A33' -> ~0.835, F_z -> +rho*g*A*pi*a^2)
    assert abs(low.surge_added_mass_dimless - 0.5) < 5.0e-4
    assert abs(low.heave_added_mass_dimless - 0.835) < 2.0e-3
    assert 0.0 <= low.surge_damping_dimless < 1.0e-6
    assert 0.0 < low.heave_damping_dimless < 3.0e-3
    hydro_fz = 1025.0 * 9.80665 * math.pi
    assert abs(low.excitation_force_z - hydro_fz) / hydro_fz < 2.0e-3

    # Published Hulme (1982) Table 1 (heave) & Table 2 (surge) values at Ka = 1.0 and 2.0
    r1 = hulme_hemisphere_radiation(1.0)
    assert abs(r1.heave_added_mass_dimless - 0.4284) < 3.0e-4
    assert abs(r1.heave_damping_dimless - 0.2484) < 3.0e-4
    assert abs(r1.surge_added_mass_dimless - 0.5740) < 3.0e-4
    assert abs(r1.surge_damping_dimless - 0.3535) < 3.0e-4
    assert abs(r1.surge_damping_dimless - r1.surge_far_field_damping_dimless) < 1.0e-5
    assert abs(r1.heave_damping_dimless - r1.heave_far_field_damping_dimless) < 1.5e-4

    r2 = hulme_hemisphere_radiation(2.0)
    assert abs(r2.heave_added_mass_dimless - 0.3884) < 3.0e-4
    assert abs(r2.heave_damping_dimless - 0.1031) < 3.0e-4
    assert abs(r2.surge_added_mass_dimless - 0.2493) < 3.0e-4
    assert abs(r2.surge_damping_dimless - 0.3424) < 3.0e-4


def test_floating_hemisphere_bem_second_order_convergence_and_haskind() -> None:
    study = run_refinement_study(subdivisions=(1, 2, 3), ka=1.0)
    levels = study["levels"]
    fine = levels[-1]
    assert fine["panel_count"] == 256
    assert fine["observed_order_a33"] is not None and fine["observed_order_a33"] > 1.9
    assert fine["observed_order_b33"] is not None and fine["observed_order_b33"] > 1.85
    assert fine["observed_order_a11"] is not None and fine["observed_order_a11"] > 1.9
    assert fine["observed_order_fz"] is not None and fine["observed_order_fz"] > 1.85
    assert fine["heave_added_mass_rel_error"] < 0.01
    assert fine["heave_damping_rel_error"] < 0.015
    assert fine["surge_added_mass_rel_error"] < 0.016
    assert fine["excitation_force_z_rel_error"] < 0.008
    assert fine["excitation_force_x_rel_error"] < 0.025
    assert fine["far_field_damping_rel_diff_heave"] < 0.002
    assert fine["haskind_reciprocity_rel_diff_x"] < 0.001
    assert fine["haskind_reciprocity_rel_diff_z"] < 0.002


def test_floating_hemisphere_6dof_symmetry_and_oblique_diffraction() -> None:
    mesh = hemisphere_hull_mesh(2, radius=1.0)
    rad_sym = solve_radiation_3d(mesh, 1.0, modes=(0, 1), chief_points=None)
    assert abs(rad_sym.added_mass[0, 0] - rad_sym.added_mass[1, 1]) < 1.0e-13 * rad_sym.added_mass[0, 0]

    ops = assemble_free_surface_operators_3d(mesh, 1.0, chief_points="auto")
    rad = solve_radiation_3d(mesh, 1.0, operators=ops)
    assert rad.added_mass.shape == (6, 6)
    assert rad.damping.shape == (6, 6)
    assert abs(rad.added_mass[0, 0] - rad.added_mass[1, 1]) < 1.0e-5 * rad.added_mass[0, 0]
    assert abs(rad.damping[0, 0] - rad.damping[1, 1]) < 1.0e-5 * rad.damping[0, 0]
    assert np.max(np.abs(rad.added_mass[3:, 3:])) < 1.0e-3 * rad.added_mass[2, 2]

    b_far = far_field_damping_matrix_3d(mesh, rad)
    assert abs(rad.damping[0, 0] - b_far[0, 0]) / rad.damping[0, 0] < 0.015
    assert abs(rad.damping[2, 2] - b_far[2, 2]) / rad.damping[2, 2] < 0.01

    # Oblique wave heading beta = pi / 3: Haskind matches direct diffraction in all 6 DOFs
    beta = math.pi / 3.0
    diff = solve_diffraction_3d(mesh, 1.0, wave_angle=beta, operators=ops)
    hask = haskind_excitation_force_3d(mesh, rad, wave_angle=beta)
    assert (
        np.linalg.norm(diff.excitation_force[:3] - hask["boundary_haskind"][:3])
        / np.linalg.norm(diff.excitation_force[:3])
        < 0.003
    )
    assert (
        np.linalg.norm(diff.excitation_force[:3] - hask["kochin_haskind"][:3])
        / np.linalg.norm(diff.excitation_force[:3])
        < 0.005
    )


def test_floating_hemisphere_irregular_frequency_chief_suppression() -> None:
    mesh = hemisphere_hull_mesh(3, radius=1.0)
    ka = 2.58
    ref = hulme_hemisphere_radiation(ka)
    vol = ref.displaced_volume

    unreg = solve_radiation_3d(mesh, ka, modes=(2,), chief_points=None)
    chief = solve_radiation_3d(mesh, ka, modes=(2,), chief_points="auto")

    b33_unreg = float(unreg.damping[0, 0] / (1025.0 * ref.omega * vol))
    b33_chief = float(chief.damping[0, 0] / (1025.0 * ref.omega * vol))
    a33_chief = float(chief.added_mass[0, 0] / (1025.0 * vol))

    # Unregularized BIE hits the interior waterplane resonance and violates passivity (B33 < 0)
    assert unreg.condition_number > 100.0
    assert b33_unreg < 0.0

    # 5 interior 3-D CHIEF equations cure both conditioning and passivity
    assert chief.chief_points_used.shape == (5, 3)
    assert chief.condition_number < 6.0
    assert b33_chief > 0.0
    assert abs(b33_chief - ref.heave_damping_dimless) / ref.heave_damping_dimless < 0.10
    assert abs(a33_chief - ref.heave_added_mass_dimless) / ref.heave_added_mass_dimless < 0.03


def test_floating_hemisphere_cli_quick_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.floating_hemisphere", "--quick"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["benchmark"] == "hulme_floating_hemisphere_3d_free_surface"
    assert len(payload["refinement"]["levels"]) == 2
