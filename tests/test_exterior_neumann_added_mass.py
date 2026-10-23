"""Tests for exterior Neumann solvers and rigid-body rotational/translational added mass."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem import (
    compute_added_mass_matrix_2d,
    compute_added_mass_matrix_3d,
    ellipse_added_mass,
    ellipse_added_mass_matrix,
    ellipse_added_moment_of_inertia,
    ellipse_mesh,
    ellipsoid_added_mass_matrix,
    ellipsoid_added_moment_of_inertia,
    ellipsoid_mesh,
    ellipsoid_volume,
     evaluate_surface_field,
    evaluate_surface_velocity,
    icosphere,
    regular_polygon,
    solve_laplace_exterior_dirichlet,
    solve_laplace_exterior_neumann,
    solve_laplace_surface,
    solve_laplace_surface_neumann,
)
from tensorbem.geometry import PanelMesh
from tensorbem.geometry3d import TriangleMesh


class TestAnalyticalRotationalAddedMass:
    def test_circular_cylinder_has_zero_rotational_added_inertia(self) -> None:
        assert ellipse_added_moment_of_inertia(1.2, 1.2, density=1025.0) == 0.0
        matrix = ellipse_added_mass_matrix(1.2, 1.2, density=1025.0)
        assert matrix.shape == (3, 3)
        assert matrix[2, 2] == 0.0
        assert math.isclose(matrix[0, 0], matrix[1, 1], rel_tol=1.0e-15)

    def test_ellipse_added_moment_matches_lamb_formula_and_plate_limit(self) -> None:
        a, b, rho = 2.0, 0.8, 1025.0
        expected = 0.125 * rho * math.pi * (a * a - b * b) ** 2
        assert math.isclose(
            ellipse_added_moment_of_inertia(a, b, rho), expected, rel_tol=1.0e-15
        )
        # Swap (a, b) leaves rotational inertia about the origin unchanged.
        assert math.isclose(
            ellipse_added_moment_of_inertia(a, b, rho),
            ellipse_added_moment_of_inertia(b, a, rho),
            rel_tol=1.0e-15,
        )
        # Thin flat-plate limit b -> 0 gives (1/8) * rho * pi * a**4.
        plate_limit = 0.125 * rho * math.pi * a**4
        assert math.isclose(
            ellipse_added_moment_of_inertia(a, 1.0e-6, rho),
            plate_limit,
            rel_tol=1.0e-11,
        )
        diag = np.diag(ellipse_added_mass_matrix(a, b, rho))
        np.testing.assert_allclose(
            diag[:2], ellipse_added_mass(a, b, rho), rtol=1.0e-15
        )
        assert math.isclose(diag[2], expected, rel_tol=1.0e-15)

    def test_sphere_and_spheroid_rotational_added_inertia(self) -> None:
        sphere_rot = ellipsoid_added_moment_of_inertia([1.3, 1.3, 1.3], density=1025.0)
        np.testing.assert_array_equal(sphere_rot, np.zeros(3))

        # Prolate spheroid about x-axis: roll I_xx == 0, pitch I_yy == yaw I_zz > 0.
        prolate_rot = ellipsoid_added_moment_of_inertia([2.0, 0.8, 0.8], density=1025.0)
        assert prolate_rot[0] == 0.0
        assert prolate_rot[1] > 0.0
        assert math.isclose(prolate_rot[1], prolate_rot[2], rel_tol=1.0e-14)

    def test_triaxial_ellipsoid_cyclic_permutation_and_slender_limit(self) -> None:
        axes = np.array([1.8, 1.1, 0.65])
        base = ellipsoid_added_moment_of_inertia(axes, density=1000.0)
        cycled = ellipsoid_added_moment_of_inertia(np.roll(axes, -1), density=1000.0)
        np.testing.assert_allclose(cycled, np.roll(base, -1), rtol=1.0e-13)

        # Slender prolate spheroid limit b = c << a: A_55 / ((1/5)*rho*V*a**2) -> 1.
        a = 2.0
        b = 0.01
        rho = 1000.0
        vol = ellipsoid_volume([a, b, b])
        slender_rot = ellipsoid_added_moment_of_inertia([a, b, b], density=rho)
        assert math.isclose(slender_rot[1] / (0.2 * rho * vol * a * a), 1.0, rel_tol=2.0e-3)


class TestExteriorNeumannDualityAndAddedMass:
    def test_2d_neumann_is_discrete_dual_of_exterior_dirichlet(self) -> None:
        mesh = ellipse_mesh(48, 1.4, 0.9)
        centers = mesh.centers
        r2 = np.sum(centers * centers, axis=1)
        phi_exact = centers[:, 0] / r2
        dirichlet = solve_laplace_exterior_dirichlet(mesh, phi_exact)
        neumann = solve_laplace_exterior_neumann(mesh, dirichlet.normal_derivative)
        np.testing.assert_allclose(
            neumann.boundary_values, phi_exact, rtol=1.0e-12, atol=1.0e-12
        )
        assert neumann.relative_residual_l2 < 1.0e-14

    def test_3d_neumann_is_discrete_dual_of_exterior_dirichlet(self) -> None:
        mesh = ellipsoid_mesh(1, [1.4, 1.0, 0.8])
        c = mesh.centroids
        r3 = np.linalg.norm(c, axis=1) ** 3
        phi_exact = c[:, 0] / r3
        dirichlet = solve_laplace_surface(mesh, phi_exact, domain="exterior")
        neumann = solve_laplace_surface_neumann(mesh, dirichlet.normal_derivative)
        np.testing.assert_allclose(
            neumann.boundary_values, phi_exact, rtol=1.0e-12, atol=1.0e-12
        )
        assert neumann.relative_residual_l2 < 1.0e-14

    def test_2d_added_mass_tensor_converges_quadratically_and_obeys_parallel_axis(self) -> None:
        a, b, rho = 1.5, 0.75, 1025.0
        exact_diag = np.diag(ellipse_added_mass_matrix(a, b, rho))
        errors = []
        for panels in (16, 32, 64, 128):
            mesh = ellipse_mesh(panels, a, b)
            res = compute_added_mass_matrix_2d(mesh, density=rho)
            assert res.condition_number < 4.0
            assert res.symmetry_error < 1.0e-11
            diag = np.diag(res.added_mass_matrix)
            errors.append(np.abs(diag - exact_diag) / exact_diag)

        err_arr = np.asarray(errors)
        for mode_idx in range(3):
            assert np.all(np.diff(err_arr[:, mode_idx]) < 0.0)
            ratio = err_arr[-2, mode_idx] / err_arr[-1, mode_idx]
            assert 3.7 < ratio < 4.3
        assert err_arr[-1, 2] < 1.0e-3

        # Circle rotational inertia vanishes.
        circle_res = compute_added_mass_matrix_2d(regular_polygon(64, radius=1.1), density=rho)
        assert abs(circle_res.added_mass_matrix[2, 2]) < 1.0e-11

        # Shifting both the mesh and the reference centre leaves A invariant;
        # keeping center=(0,0) reproduces Newman's exact parallel-axis law.
        shift = np.array([3.5, -2.2])
        base_mesh = ellipse_mesh(64, a, b)
        shifted_mesh = PanelMesh(base_mesh.vertices + shift)
        base_res = compute_added_mass_matrix_2d(base_mesh, density=rho)
        shifted_centred = compute_added_mass_matrix_2d(
            shifted_mesh, density=rho, center=shift
        )
        np.testing.assert_allclose(
            shifted_centred.added_mass_matrix,
            base_res.added_mass_matrix,
            rtol=1.0e-12,
            atol=1.0e-10,
        )
        shifted_origin = compute_added_mass_matrix_2d(
            shifted_mesh, density=rho, center=(0.0, 0.0)
        )
        transform = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-shift[1], shift[0], 1.0],
            ]
        )
        np.testing.assert_allclose(
            shifted_origin.added_mass_matrix,
            transform @ base_res.added_mass_matrix @ transform.T,
            rtol=1.0e-12,
            atol=1.0e-9,
        )

    def test_3d_six_dof_added_mass_tensor_converges_on_triaxial_ellipsoid(self) -> None:
        axes = [1.5, 1.0, 0.75]
        rho = 1025.0
        exact_diag = np.diag(ellipsoid_added_mass_matrix(axes, rho))

        # On the exact regular icosahedron (sub=0) centroids are parallel to face
        # normals so rotational inertia is identically zero; subdivided icospheres
        # decay rapidly toward zero as O(h**3).
        ico0 = compute_added_mass_matrix_3d(icosphere(0, radius=1.0), density=rho)
        np.testing.assert_allclose(
            np.diag(ico0.added_mass_matrix)[3:], 0.0, atol=1.0e-11
        )
        ico1 = compute_added_mass_matrix_3d(icosphere(1, radius=1.0), density=rho)
        ico2 = compute_added_mass_matrix_3d(icosphere(2, radius=1.0), density=rho)
        assert np.max(np.diag(ico2.added_mass_matrix)[3:]) < 0.15 * np.max(
            np.diag(ico1.added_mass_matrix)[3:]
        )

        errors = []
        for sub in (0, 1, 2):
            mesh = ellipsoid_mesh(sub, axes)
            res = compute_added_mass_matrix_3d(mesh, density=rho)
            assert res.condition_number < 3.0
            assert res.max_relative_residual_l2 < 1.0e-14
            diag = np.diag(res.added_mass_matrix)
            errors.append(np.abs(diag - exact_diag) / exact_diag)

        err_arr = np.asarray(errors)
        for mode_idx in range(6):
            assert err_arr[0, mode_idx] > err_arr[1, mode_idx] > err_arr[2, mode_idx]
            ratio = err_arr[1, mode_idx] / err_arr[2, mode_idx]
            assert 3.4 < ratio < 4.5
        assert np.all(err_arr[2, :3] < 0.025)
        assert np.all(err_arr[2, 3:] < 0.05)

        # Translation invariance when reference center shifts with the body.
        shift = np.array([-2.0, 1.5, 0.8])
        m1 = ellipsoid_mesh(1, axes)
        m1_shifted = TriangleMesh(m1.vertices + shift, m1.faces)
        r1 = compute_added_mass_matrix_3d(m1, density=rho)
        r1_shifted = compute_added_mass_matrix_3d(
            m1_shifted, density=rho, center=shift
        )
        np.testing.assert_allclose(
            r1_shifted.added_mass_matrix,
            r1.added_mass_matrix,
            rtol=1.0e-12,
            atol=1.0e-10,
        )

    def test_validation_guards_and_cli_strict_json(self, tmp_path) -> None:
        mesh3d = icosphere(0, radius=1.0)
        res = solve_laplace_surface(
            mesh3d, np.ones(mesh3d.panel_count), domain="exterior"
        )
        with pytest.raises(ValueError, match="minimum_clearance_factor"):
            evaluate_surface_field(mesh3d, res, np.array([[0.0, 0.0, 2.0]]), minimum_clearance_factor=-0.1)
        with pytest.raises(ValueError, match="minimum_clearance_factor"):
            evaluate_surface_velocity(mesh3d, res, np.array([[0.0, 0.0, 2.0]]), minimum_clearance_factor=True)
        # Even with minimum_clearance_factor=0.0, exact collocation centroids are rejected.
        with pytest.raises(ValueError, match="surface"):
            evaluate_surface_field(mesh3d, res, mesh3d.centroids[:1], minimum_clearance_factor=0.0)
        with pytest.raises(ValueError, match="surface"):
            evaluate_surface_velocity(mesh3d, res, mesh3d.centroids[:1], minimum_clearance_factor=0.0)

        out = tmp_path / "rigid_body.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tensorbem.benchmarks.rigid_body_added_mass",
                "--panels",
                "16",
                "32",
                "--subdivisions",
                "0",
                "1",
                "--output",
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(completed.stdout)
        assert parsed == json.loads(out.read_text(encoding="utf-8"))
        assert parsed["benchmark"] == "rigid_body_neumann_added_mass_and_rotational_inertia"
