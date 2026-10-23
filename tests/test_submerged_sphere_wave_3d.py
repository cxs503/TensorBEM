"""Verification tests for Wang (1986) / Linton (1991) 3-D submerged-sphere waves."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.benchmarks.submerged_sphere_wave import run_refinement_study
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh


def test_submerged_sphere_multipole_deep_limit_and_exact_haskind() -> None:
    # Deeply submerged sphere recovers Lamb's unbounded-fluid added mass A' -> 0.5
    # and Morison's C_M = 1.5 inertia wave force
    deep = submerged_sphere_wave_multipole(0.05, radius=1.0, submergence=40.0)
    assert abs(deep.surge_added_mass_dimless - 0.5) < 1.0e-5
    assert abs(deep.heave_added_mass_dimless - 0.5) < 1.0e-5
    assert 0.0 <= deep.surge_damping_dimless < 2.0e-5
    assert 0.0 <= deep.heave_damping_dimless < 2.0e-5

    # Under a wave crest at x = 0, water orbital acceleration is downward (-z),
    # so Morison's C_M = 1.5 inertia force on a closed submerged sphere gives
    # F_z -> -1.5 * rho * g * V * nu * exp(-nu * f) and F_x -> -i * 1.5 * ...
    vol = (4.0 / 3.0) * math.pi
    fk_scale = 1.5 * 1025.0 * 9.80665 * vol * 0.05 * math.exp(-0.05 * 40.0)
    assert abs(deep.excitation_force_z - (-fk_scale)) / fk_scale < 5.0e-4
    assert abs(deep.excitation_force_x - (-1.0j * fk_scale)) / fk_scale < 5.0e-4

    # Newman's 3-D axisymmetric Haskind identities hold to ~1e-15 across all (Ka, f/a)
    for sub_ratio in (1.25, 1.5, 2.5):
        for ka in (0.2, 0.8, 1.8):
            res = submerged_sphere_wave_multipole(ka, radius=1.0, submergence=sub_ratio)
            assert res.haskind_error_x < 1.0e-13
            assert res.haskind_error_z < 1.0e-13
            assert res.surge_damping_dimless > 0.0
            assert res.heave_damping_dimless > 0.0

    with pytest.raises(ValueError, match="strictly greater than radius"):
        submerged_sphere_wave_multipole(0.8, radius=1.0, submergence=1.0)


def test_submerged_sphere_bem_second_order_convergence() -> None:
    study = run_refinement_study(subdivisions=(0, 1, 2), ka=0.8, submergence=1.5)
    levels = study["levels"]
    fine = levels[-1]
    assert fine["panel_count"] == 320
    assert fine["observed_order_a33"] is not None and fine["observed_order_a33"] > 1.9
    assert fine["observed_order_b33"] is not None and fine["observed_order_b33"] > 1.85
    assert fine["observed_order_a11"] is not None and fine["observed_order_a11"] > 1.9
    assert fine["observed_order_fz"] is not None and fine["observed_order_fz"] > 1.9
    assert fine["heave_added_mass_rel_error"] < 0.025
    assert fine["surge_added_mass_rel_error"] < 0.025
    assert fine["excitation_force_x_rel_error"] < 0.035
    assert fine["excitation_force_z_rel_error"] < 0.035
    assert fine["rotational_added_mass_max_ratio"] < 1.0e-4
    assert fine["far_field_damping_rel_diff_heave"] < 0.005
    assert fine["haskind_reciprocity_rel_diff_x"] < 0.001
    assert fine["haskind_reciprocity_rel_diff_z"] < 0.001


def test_submerged_sphere_6dof_decoupling_and_oblique_haskind() -> None:
    mesh = submerged_sphere_mesh(1, radius=1.0, submergence=1.5)
    center = (0.0, 0.0, -1.5)
    ops = assemble_free_surface_operators_3d(mesh, 0.9)
    rad = solve_radiation_3d(mesh, 0.9, center=center, operators=ops)
    b_far = far_field_damping_matrix_3d(mesh, rad)

    # Symmetry & off-diagonal decoupling of an axisymmetric submerged sphere
    assert np.allclose(rad.added_mass, rad.added_mass.T, rtol=1.0e-3, atol=1.0)
    assert abs(rad.added_mass[0, 0] - rad.added_mass[1, 1]) / rad.added_mass[0, 0] < 1.0e-3
    assert abs(rad.added_mass[0, 2]) / rad.added_mass[2, 2] < 1.0e-12
    assert abs(rad.damping[2, 2] - b_far[2, 2]) / rad.damping[2, 2] < 0.02

    beta = -0.7
    diff = solve_diffraction_3d(mesh, 0.9, wave_angle=beta, center=center, operators=ops)
    hask = haskind_excitation_force_3d(mesh, rad, wave_angle=beta)
    assert (
        np.linalg.norm(diff.excitation_force[:3] - hask["boundary_haskind"][:3])
        / np.linalg.norm(diff.excitation_force[:3])
        < 0.003
    )


def test_submerged_sphere_cli_quick_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.submerged_sphere_wave", "--quick"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["benchmark"] == "submerged_sphere_3d_free_surface_wave"
    assert len(payload["refinement"]["levels"]) == 2
