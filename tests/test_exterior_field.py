"""Verification of the exterior-domain potential reconstruction.

Every assertion here is against a closed-form exterior harmonic of the circle,
``u_m = r**(-m) cos(m theta)``, not against another discretisation. The
companion study is ``python -m tensorbem.benchmarks.exterior_field``; the
numbers quoted in ``docs/exterior_field_verification.md`` come from that JSON.

The last two tests pin the *limitation* the study exposed: constant (non-zero
mean) Dirichlet data have no decaying exterior harmonic extension, so the
exterior Dirichlet equation returns a pure capacity flux and reconstructing the
field from that trace pair yields a logarithmically growing, non-physical
potential. The evaluator must refuse it rather than return a silent non-solution.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem import regular_polygon
from tensorbem.solver import (
    evaluate_exterior_field,
    evaluate_interior_field,
    exterior_dirichlet_decay_ratio,
    exterior_net_flux,
    solve_laplace_dirichlet,
    solve_laplace_exterior_dirichlet,
    solve_laplace_exterior_neumann,
)


def _ring(radius, angles):
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _panel_angle(mesh):
    return np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])


PROBE_ANGLES = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
PROBE_RADII = (1.1, 1.5, 2.0, 4.0, 8.0)


def _probe_points():
    return np.vstack([_ring(r, PROBE_ANGLES) for r in PROBE_RADII])


def _mode_errors(mode, panels_list=(32, 64, 128)):
    points = _probe_points()
    radius = np.hypot(points[:, 0], points[:, 1])
    theta = np.arctan2(points[:, 1], points[:, 0])
    exact = np.cos(mode * theta) / radius**mode
    errors = []
    for panels in panels_list:
        mesh = regular_polygon(panels)
        result = solve_laplace_exterior_dirichlet(mesh, np.cos(mode * _panel_angle(mesh)))
        numerical = evaluate_exterior_field(mesh, result, points)
        errors.append(float(np.max(np.abs(numerical - exact))))
    return errors


def test_dipole_field_matches_the_analytic_exterior_solution_at_second_order():
    errors = _mode_errors(1)
    assert errors[0] > errors[1] > errors[2]
    # Observed order close to the expected 2 for piecewise-constant panels.
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert 1.8 < math.log(coarse / fine, 2.0) < 2.3
    assert errors[-1] < 5.0e-4


def test_quadrupole_field_matches_the_analytic_exterior_solution():
    errors = _mode_errors(2)
    assert errors[0] > errors[1] > errors[2]
    assert 1.8 < math.log(errors[-2] / errors[-1], 2.0) < 2.3
    assert errors[-1] < 1.0e-3


def test_neumann_driven_translating_cylinder_field_is_uniformly_accurate():
    # q_e = -cos(theta) is the translating cylinder with U = 1, u = -cos(theta)/r.
    mesh = regular_polygon(128)
    result = solve_laplace_exterior_neumann(mesh, -np.cos(_panel_angle(mesh)))
    relative = []
    for radius in (1.2, 2.0, 5.0, 10.0):
        points = _ring(radius, PROBE_ANGLES)
        numerical = evaluate_exterior_field(mesh, result, points)
        exact = -np.cos(PROBE_ANGLES) / radius
        relative.append(float(np.max(np.abs(numerical - exact)) / np.max(np.abs(exact))))
    # The representation is exact for the trace pair, so the *relative* error is
    # set only by the piecewise-constant boundary trace and is radius independent.
    assert max(relative) < 5.0e-4
    assert max(relative) - min(relative) < 1.0e-5


def test_far_field_decay_is_one_over_r_with_the_analytic_amplitude():
    mesh = regular_polygon(128)
    result = solve_laplace_exterior_dirichlet(mesh, np.cos(_panel_angle(mesh)))
    radii = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    peak = np.array(
        [
            float(np.max(np.abs(evaluate_exterior_field(mesh, result, _ring(r, PROBE_ANGLES)))))
            for r in radii
        ]
    )
    slope, _ = np.polyfit(np.log(radii), np.log(peak), 1)
    assert abs(slope + 1.0) < 1.0e-3
    scaled = peak * radii
    assert np.all(np.abs(scaled - 1.0) < 1.0e-3)


def test_exterior_field_is_continuous_across_the_boundary():
    # A probe just outside the curve must recover the Dirichlet data to the
    # discretisation error, i.e. the jump of the double layer is applied once.
    mesh = regular_polygon(128)
    result = solve_laplace_exterior_dirichlet(mesh, np.cos(_panel_angle(mesh)))
    angles = _panel_angle(mesh)
    outside = _ring(1.0 + 1.0e-3, angles)
    values = evaluate_exterior_field(mesh, result, outside)
    assert float(np.max(np.abs(values - np.cos(angles)))) < 5.0e-3


def test_interior_and_exterior_evaluators_reject_the_other_domain():
    mesh = regular_polygon(32)
    interior = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
    exterior = solve_laplace_exterior_dirichlet(mesh, np.cos(_panel_angle(mesh)))
    points = np.array([[3.0, 0.0]])
    with pytest.raises(ValueError, match="exterior_field"):
        evaluate_interior_field(mesh, exterior, points)
    with pytest.raises(ValueError, match="exterior solve"):
        evaluate_exterior_field(mesh, interior, points)
    with pytest.raises(TypeError):
        evaluate_exterior_field(mesh, "not a result", points)
    with pytest.raises(ValueError):
        evaluate_exterior_field(mesh, exterior, np.array([[np.inf, 0.0]]))


def test_nondecaying_dirichlet_data_are_refused_instead_of_silently_growing():
    # Constant boundary data have a non-zero mean: the bounded exterior solution
    # is u == 1 with zero flux, but the decaying single-layer representation has
    # no such member, so the solve returns a pure capacity flux.
    mesh = regular_polygon(128)
    result = solve_laplace_exterior_dirichlet(mesh, np.ones(mesh.panel_count))
    assert exterior_dirichlet_decay_ratio(mesh, result) > 0.99
    assert abs(exterior_net_flux(mesh, result)) > 1.0
    with pytest.raises(ValueError, match="non-zero net flux"):
        evaluate_exterior_field(mesh, result, _ring(2.0, PROBE_ANGLES))


def test_the_spurious_mode_is_exactly_the_capacity_flux_and_grows_logarithmically():
    # Opting out shows *why* the guard exists: the reconstruction obeys
    # u(r) - u(r0) = -(Q/2 pi) log(r/r0) to machine precision, so the capability
    # flux is the whole story and nothing else is wrong with the representation.
    mesh = regular_polygon(128)
    result = solve_laplace_exterior_dirichlet(mesh, np.ones(mesh.panel_count))
    net = exterior_net_flux(mesh, result)
    flux = result.normal_derivative
    assert float(np.std(flux)) < 1.0e-10  # a single uniform mode, not noise
    radii = (1.5, 2.0, 4.0, 8.0, 16.0)
    means = np.array(
        [
            float(
                np.mean(
                    evaluate_exterior_field(
                        mesh, result, _ring(r, PROBE_ANGLES), require_decaying=False
                    )
                )
            )
            for r in radii
        ]
    )
    prediction = means[0] - net / (2.0 * math.pi) * np.log(np.asarray(radii) / radii[0])
    assert float(np.max(np.abs(means - prediction))) < 1.0e-9
    assert abs(float(means[-1]) - 1.0) > 1.0  # far from the bounded solution u == 1


def test_zero_flux_neumann_pair_is_accepted_and_stays_bounded():
    # Rigid-body modes always have zero net flux, so the guard must not fire and
    # the field must decay rather than grow.
    mesh = regular_polygon(64)
    result = solve_laplace_exterior_neumann(mesh, -np.cos(_panel_angle(mesh)))
    assert exterior_dirichlet_decay_ratio(mesh, result) < 1.0e-12
    near = evaluate_exterior_field(mesh, result, _ring(2.0, PROBE_ANGLES))
    far = evaluate_exterior_field(mesh, result, _ring(16.0, PROBE_ANGLES))
    assert float(np.max(np.abs(far))) < float(np.max(np.abs(near)))


def test_committed_study_artifact_reproduces_the_headline_numbers():
    path = Path(__file__).resolve().parents[1] / "outputs" / "exterior_field" / "study.json"
    if not path.exists():
        pytest.skip("study artifact not generated in this checkout")
    study = json.loads(path.read_text())
    assert study["dipole_observed_orders"][-1] == pytest.approx(2.0, abs=0.05)
    assert study["quadrupole_observed_orders"][-1] == pytest.approx(2.0, abs=0.05)
    assert study["far_field"]["fitted_decay_exponent"] == pytest.approx(-1.0, abs=1.0e-4)
    assert study["nondecaying"]["decay_ratio"] == pytest.approx(1.0, abs=1.0e-6)
    assert study["nondecaying"]["guard_raised"] == 1.0
    # Recompute the finest dipole error and compare with the stored value.
    errors = _mode_errors(1, panels_list=(128,))
    stored = [r for r in study["dipole"] if r["panels"] == 128][0]["max_abs_error"]
    assert errors[0] == pytest.approx(stored, rel=1.0e-9)