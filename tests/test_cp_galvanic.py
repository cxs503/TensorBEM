"""Tests for :mod:`tensorbem.cp_galvanic`.

The galvanic cathodic-protection face is verified against

* exact Laplace solutions (parallel plates, coaxial cylinders, exterior sphere),
* the electrochemistry (Butler-Volmer sign/monotonicity, transport limit, Al
  anodic Tafel slope),
* an independent Picard fixed-point solution of the same non-linear problem,
* charge conservation and the DNV-RP-B401 window logic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import cp_galvanic as cp


def test_kinetics_signs_and_transport_limit():
    kinetics = cp.SteelKinetics()
    potentials = np.array([-0.2, -0.6, -0.8, -1.0, -1.2])
    net = kinetics.net_current_density(potentials)
    # above the free-corrosion potential the steel dissolves (negative current)
    assert net[0] < 0.0
    assert abs(net[1]) < 1.0e-12
    # below it the cathodic current grows, and is capped by the transport limit
    assert net[2] > 0.0
    assert np.all(np.diff(net[2:]) > 0.0)
    assert np.all(net < kinetics.i_limiting * (1.0 + 1.0e-12))
    assert kinetics.transport_limited_potential() < -0.80
    # the cathodic Tafel slope of the Butler-Volmer expression is RT/(alpha F)
    expected = math.log(10.0) * cp.GAS_CONSTANT * cp.REFERENCE_TEMPERATURE / (
        kinetics.alpha_c * cp.FARADAY
    )
    assert kinetics.tafel_slope == pytest.approx(expected, rel=1.0e-12)
    assert kinetics.tafel_slope == pytest.approx(0.1183, abs=2.0e-3)


def test_kinetics_derivative_matches_finite_difference():
    kinetics = cp.SteelKinetics()
    points = np.array([-0.35, -0.55, -0.70, -0.90, -1.10])
    step = 1.0e-7
    fd = (
        kinetics.net_current_density(points + step)
        - kinetics.net_current_density(points - step)
    ) / (2.0 * step)
    assert np.allclose(kinetics.derivative(points), fd, rtol=2.0e-5, atol=1.0e-8)
    assert np.all(kinetics.derivative(points) < 0.0)


def test_anode_kinetics_tafel_slope():
    anode = cp.AnodeKinetics()
    assert anode.current_density(np.array([anode.ocp]))[0] == pytest.approx(1.0)
    decade = anode.current_density(np.array([anode.ocp + anode.tafel_slope]))[0]
    assert decade == pytest.approx(10.0, rel=1.0e-12)
    # a decade of current per 0.05 V of polarisation
    assert anode.derivative(np.array([anode.ocp]))[0] == pytest.approx(
        anode.slope * 1.0, rel=1.0e-12
    )


def test_box_hull_mesh_geometry():
    hull = cp.BoxHull()
    mesh, groups = cp.box_hull_mesh(hull, nx=8, ny=3, nz=2)
    assert mesh.panel_count == cp.box_hull_panel_count(8, 3, 2)
    assert mesh.panel_count == sum(group.size for group in groups.values())
    assert mesh.volume == pytest.approx(hull.displacement_volume, rel=1.0e-12)
    assert mesh.surface_area == pytest.approx(
        hull.wetted_area + hull.length * hull.breadth, rel=1.0e-12
    )
    assert groups["waterline"].size == 2 * 8 * 3


def test_partition_covers_surface_and_hits_beta():
    hull = cp.BoxHull()
    mesh, groups = cp.box_hull_mesh(hull, nx=10, ny=4, nz=2)
    partition = cp.partition_surface(mesh, groups, beta=0.05, n_anodes=4)
    total = np.concatenate([partition.bare, partition.coated, partition.anodes, partition.waterline])
    assert total.size == mesh.panel_count
    assert np.unique(total).size == mesh.panel_count
    assert partition.total_area == pytest.approx(hull.wetted_area, rel=1.0e-12)
    assert partition.achieved_beta == pytest.approx(0.05, abs=0.01)
    assert partition.anode_area > 0.0
    assert partition.anode_scale > 0.0


def test_plate_anchor_matches_linear_profile():
    result = cp.plate_linear_benchmark(n_horizontal=8, n_vertical=2)
    assert result["potential_rel_l2"] < 1.0e-2
    assert result["wall_flux_max_abs"] < 1.0e-12
    assert result["relative_residual"] < 1.0e-10


def test_cylinder_anchor_matches_log_profile():
    result = cp.cylinder_log_benchmark(n_theta=32, n_radial=6, n_axial=1)
    assert result["potential_rel_l2"] < 2.0e-2
    assert result["cap_flux_max_abs"] < 1.0e-12
    assert result["relative_residual"] < 1.0e-10


def test_sphere_exterior_anchor():
    result = cp.sphere_exterior_benchmark(subdivisions=2)
    assert result["flux_rel_error"] < 2.0e-2
    assert result["potential_rel_l2"] < 3.0e-2
    assert result["relative_residual"] < 1.0e-10


def test_exterior_mixed_solver_reproduces_uniform_sphere_solution():
    # Dirichlet on the northern hemisphere, exact Neumann elsewhere: the exact
    # solution (uniform potential, uniform flux) must be reproduced everywhere,
    # which tests the exterior mixed formulation used by the protection solve.
    mesh = cp.icosphere(subdivisions=3, radius=1.0)
    north = mesh.centroids[:, 2] > 0.0
    phi = np.full(mesh.panel_count, np.nan)
    flux = np.zeros(mesh.panel_count)
    phi[north] = -1.0
    flux[~north] = -1.0
    result = cp.solve_mixed_laplace(mesh, phi, flux, domain="exterior")
    assert np.max(np.abs(result.potential[~north] + 1.0)) < 5.0e-3
    assert np.max(np.abs(result.flux[north] + 1.0)) < 2.0e-2
    assert abs(float(np.mean(result.flux)) + 1.0) < 5.0e-3


def _small_case(nx: int = 8, ny: int = 3, nz: int = 2):
    hull = cp.BoxHull()
    mesh, groups = cp.box_hull_mesh(hull, nx=nx, ny=ny, nz=nz)
    partition = cp.partition_surface(mesh, groups, beta=0.05, n_anodes=3)
    return mesh, partition


def test_newton_solution_conserves_charge_and_satisfies_bcs():
    mesh, partition = _small_case()
    result = cp.solve_galvanic_protection(mesh, partition)
    assert result.nonlinear_residual < 1.0e-8
    assert result.charge_imbalance < 1.0e-8
    # every panel's current follows its own boundary condition
    kinetics, anode = result.kinetics, result.anode_kinetics
    electrode = result.metal_potential - result.electrolyte_potential
    np.testing.assert_allclose(
        result.current_density[partition.bare],
        -kinetics.net_current_density(electrode[partition.bare]),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    # the discrete anode area is normalised to the design area, so the anode
    # current density carries the normalisation factor
    np.testing.assert_allclose(
        result.current_density[partition.anodes],
        partition.anode_scale * anode.current_density(electrode[partition.anodes]),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert np.all(result.current_density[partition.coated] == 0.0)
    # the anode dissolves and the steel is protected
    assert result.anode_current_total > 0.0
    assert abs(result.steel_cathodic_current_total - result.anode_current_total) < (
        1.0e-6 * result.anode_current_total
    )
    assert cp.dnv_envelope(result)["all_points_protected"]
    assert result.metal_potential < -0.80


def test_newton_jacobian_matches_finite_differences():
    mesh, partition = _small_case()
    single, double = cp.assemble_surface_matrices(mesh)
    kinetics, anode = cp.SteelKinetics(), cp.AnodeKinetics()
    sigma = cp.SEAWATER_CONDUCTIVITY
    phi = np.zeros(mesh.panel_count)
    metal = -1.0
    q, dq_phi, dq_vm = cp._panel_fluxes(
        mesh, partition, metal - phi, kinetics, anode, sigma
    )
    matrix, rhs = cp._assemble_galvanic_system(
        single, double, mesh.areas, q, dq_phi, dq_vm, phi, metal
    )
    step = 1.0e-8

    def system(state: np.ndarray) -> np.ndarray:
        trial_phi = state[:-1]
        trial_metal = state[-1]
        trial_q, _, _ = cp._panel_fluxes(
            mesh, partition, trial_metal - trial_phi, kinetics, anode, sigma
        )
        field = single @ trial_q - 0.5 * trial_phi + double @ trial_phi
        return np.concatenate([field, [mesh.areas @ trial_q]])

    state = np.concatenate([phi, [metal]])
    columns: list[np.ndarray] = []
    for index in (0, 7, mesh.panel_count // 2, mesh.panel_count - 1, mesh.panel_count):
        perturbation = np.zeros(state.size)
        perturbation[index] = step
        columns.append(
            (system(state + perturbation) - system(state - perturbation)) / (2.0 * step)
        )
    fd = np.column_stack(columns)
    analytic = np.column_stack(
        [
            matrix[:, 0],
            matrix[:, 7],
            matrix[:, mesh.panel_count // 2],
            matrix[:, mesh.panel_count - 1],
            matrix[:, mesh.panel_count],
        ]
    )
    assert np.allclose(analytic, fd, rtol=1.0e-5, atol=1.0e-6)


def test_picard_crosscheck_agrees_with_newton():
    mesh, partition = _small_case()
    newton = cp.solve_galvanic_protection(mesh, partition)
    picard = cp.solve_galvanic_picard(mesh, partition, max_iterations=400, relaxation=0.2)
    assert picard.newton_iterations < 400
    assert picard.anode_current_total == pytest.approx(
        newton.anode_current_total, rel=1.0e-4
    )
    assert np.max(np.abs(picard.electrolyte_potential - newton.electrolyte_potential)) < 1.0e-5


def test_gmres_matches_lu_on_newton_system():
    mesh, partition = _small_case(nx=6, ny=3, nz=2)
    single, double = cp.assemble_surface_matrices(mesh)
    kinetics, anode = cp.SteelKinetics(), cp.AnodeKinetics()
    sigma = cp.SEAWATER_CONDUCTIVITY
    phi = np.zeros(mesh.panel_count)
    q, dq_phi, dq_vm = cp._panel_fluxes(
        mesh, partition, -1.0 - phi, kinetics, anode, sigma
    )
    matrix, rhs = cp._assemble_galvanic_system(
        single, double, mesh.areas, q, dq_phi, dq_vm, phi, -1.0
    )
    lu = np.linalg.solve(matrix, rhs)
    gmres, iterations = cp._solve_linear_system(matrix, rhs, "gmres")
    scale = np.linalg.norm(rhs)
    assert np.linalg.norm(matrix @ gmres - rhs) / scale < 1.0e-5
    assert np.linalg.norm(matrix @ lu - rhs) / scale < 1.0e-12
    # the Newton system is ill conditioned by design (stiff Tafel blocks), so the
    # solution vectors may differ more than the residual tolerance
    assert np.linalg.norm(gmres - lu) / np.linalg.norm(lu) < 1.0e-2
    assert iterations > 0


def test_dnv_envelope_window_logic():
    mesh, partition = _small_case()
    result = cp.solve_galvanic_protection(mesh, partition)
    envelope = cp.dnv_envelope(result)
    assert envelope["window_ok"] == (
        envelope["phi_max"] <= cp.PROTECTION_POTENTIAL
        and envelope["phi_min"] >= cp.OVERPROTECTION_POTENTIAL
    )
    assert envelope["literal_reading_satisfiable"] is False
    assert envelope["phi_min_all_including_anodes"] <= envelope["phi_min"]
    assert envelope["phi_max_all_including_anodes"] >= envelope["phi_max"]


def test_grid_convergence_index_recovers_second_order():
    # synthetic O(h^2) convergence towards 2.0 with h halved twice
    values = [2.0 - 0.05 * 0.25**2, 2.0 - 0.05 * 0.5**2, 2.0 - 0.05 * 1.0**2]
    gci = cp.grid_convergence_index(values[0], values[1], values[2], 0.25, 0.5, 1.0)
    assert gci["observed_order"] == pytest.approx(2.0, abs=0.05)
    assert gci["gci_fine_percent"] < 0.5