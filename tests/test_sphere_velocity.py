"""Tests for the three-dimensional velocity field and its benchmark.

The velocity is the quantity a potential-flow code needs for Kelvin sources,
Bernoulli pressure, and force integration, so it is checked against the exact
exterior field of a translating sphere.
"""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem.benchmarks.sphere_velocity import (
    analytic_sphere_velocity,
    run,
    velocity_probe_points,
)
from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import (
    evaluate_surface_field,
    evaluate_surface_velocity,
    evaluate_surface_velocity_at_panels,
    solve_laplace_surface,
)


def test_velocity_converges_to_the_analytic_sphere_field():
    coarse = run(1)
    fine = run(2)
    assert fine["velocity_relative_l2_error"] < coarse["velocity_relative_l2_error"]
    assert fine["velocity_relative_l2_error"] < 5.0e-2
    assert fine["velocity_max_abs_error"] < 1.0e-2
    for case in fine["per_probe_radius"]:
        assert case["relative_l2_error"] < 5.0e-2


def test_velocity_error_decreases_at_roughly_second_order():
    levels = [1, 2, 3]
    errors = [run(level)["velocity_relative_l2_error"] for level in levels]
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse
    observed = math.log(errors[-2] / errors[-1]) / math.log(2.0)
    assert 1.6 < observed < 2.4


def test_velocity_agrees_with_a_finite_difference_of_the_potential():
    # The strongest available check: the new gradient must reproduce the
    # finite difference of the already validated potential evaluator.
    mesh = icosphere(1, radius=1.0)
    boundary = mesh.centroids[:, 0].copy()
    result = solve_laplace_surface(mesh, boundary, domain="interior")
    probes = np.array(
        [[0.0, 0.30, 0.0], [0.0, 0.60, 0.0], [0.20, 0.20, 0.20], [0.0, 0.80, 0.0]]
    )
    step = 1.0e-5
    for probe in probes:
        difference = np.zeros(3)
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = step
            difference[axis] = (
                evaluate_surface_field(mesh, result, (probe + offset)[None, :])[0]
                - evaluate_surface_field(mesh, result, (probe - offset)[None, :])[0]
            ) / (2.0 * step)
        gradient = evaluate_surface_velocity(mesh, result, probe[None, :])[0]
        assert np.allclose(gradient, difference, rtol=1.0e-5, atol=1.0e-7)


def test_analytic_sphere_velocity_matches_the_known_surface_tangential_speed():
    # On the sphere the radial velocity is U*cos(theta) and the tangential
    # velocity is U*sin(theta)/2, which is the classic half-added-mass result.
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    velocity = analytic_sphere_velocity(points, radius=1.0, speed=1.0)
    # At the front stagnation point the velocity is U along +x.  At the
    # equator the radial part vanishes and the tangential part is U/2 along
    # the -x direction, which is the classic half-added-mass result.
    assert np.allclose(velocity[0], [1.0, 0.0, 0.0], atol=1.0e-12)
    assert np.allclose(velocity[1], [-0.5, 0.0, 0.0], atol=1.0e-12)
    assert np.allclose(velocity[2], [-0.5, 0.0, 0.0], atol=1.0e-12)


def test_analytic_sphere_velocity_decays_like_one_over_radius_cubed():
    # The potential decays as 1/r**2, so the velocity decays as 1/r**3.
    radius = 1.0
    speed = 1.0
    near = analytic_sphere_velocity(np.array([[2.0, 0.0, 0.0]]), radius, speed)
    far = analytic_sphere_velocity(np.array([[4.0, 0.0, 0.0]]), radius, speed)
    assert np.allclose(far, near / 8.0, atol=1.0e-12)


def test_on_surface_velocity_is_an_accuracy_claim():
    result = run(2)
    # The recovered surface velocity is an accuracy claim: it must be small.
    assert result["surface_velocity_relative_l2_error"] < 0.1
    assert math.isfinite(result["surface_velocity_max_abs_error"])


def test_representation_surface_velocity_is_much_worse_than_the_recovered_one():
    result = run(2)
    # Differentiating the Green formula with the piecewise-constant traces
    # amplifies the O(h) trace error to O(1).  The number stays visible as a
    # diagnostic of that effect.
    assert (
        result["representation_surface_velocity_relative_l2_error"]
        > 10.0 * result["surface_velocity_relative_l2_error"]
    )
    assert math.isfinite(result["representation_surface_velocity_max_abs_error"])


def test_recovered_surface_velocity_matches_the_exact_interior_velocity():
    mesh = icosphere(1, radius=1.0)
    boundary = mesh.centroids[:, 0].copy()
    result = solve_laplace_surface(mesh, boundary, domain="interior")
    on_surface = evaluate_surface_velocity_at_panels(mesh, result)
    # The interior harmonic extension of u = x is the linear function x, so
    # the exact interior velocity is the constant (1, 0, 0).
    exact = np.tile([1.0, 0.0, 0.0], (mesh.panel_count, 1))
    assert np.allclose(on_surface, exact, rtol=5.0e-2, atol=5.0e-2)


def test_representation_velocity_degrades_in_a_boundary_layer():
    # The representation path differentiates the Green formula with the
    # piecewise-constant traces, so it is inaccurate in a layer of thickness
    # O(h) around the body, and accurate away from it.  The recovered surface
    # velocity does not have that defect.
    mesh = icosphere(1, radius=1.0)
    boundary = mesh.centroids[:, 0].copy()
    result = solve_laplace_surface(mesh, boundary, domain="interior")
    exact = np.array([1.0, 0.0, 0.0])

    index = 7
    target = mesh.centroids[index]
    inward = target - 1.0e-3 * mesh.normals[index]
    approaching = evaluate_surface_velocity(
        mesh, result, inward[None, :], minimum_clearance_factor=0.0
    )[0]
    assert not np.allclose(approaching, exact, rtol=0.1, atol=0.1)

    far = evaluate_surface_velocity(
        mesh, result, np.array([[0.0, 0.0, 0.0]]), minimum_clearance_factor=0.0
    )[0]
    assert np.allclose(far, exact, rtol=5.0e-2, atol=5.0e-2)


def test_velocity_evaluation_rejects_points_too_close_to_the_surface():
    mesh = icosphere(1, radius=1.0)
    boundary = mesh.centroids[:, 0].copy()
    result = solve_laplace_surface(mesh, boundary, domain="interior")
    with pytest.raises(ValueError):
        evaluate_surface_velocity(mesh, result, mesh.centroids[:1], domain="interior")


def test_velocity_evaluation_rejects_mismatched_domain_and_shapes():
    mesh = icosphere(1, radius=1.0)
    boundary = mesh.centroids[:, 0].copy()
    result = solve_laplace_surface(mesh, boundary, domain="interior")
    with pytest.raises(ValueError):
        evaluate_surface_velocity(mesh, result, np.zeros((2, 2)), domain="interior")
    with pytest.raises(ValueError):
        evaluate_surface_velocity(
            mesh, result, np.array([[0.0, 0.3, 0.0]]), domain="exterior"
        )
    with pytest.raises(ValueError):
        evaluate_surface_velocity(
            mesh, result, np.array([[np.nan, 0.3, 0.0]]), domain="interior"
        )
    with pytest.raises(TypeError):
        evaluate_surface_velocity("not a mesh", result, np.array([[0.0, 0.3, 0.0]]))


@pytest.mark.parametrize(
    "radius,speed,probe_radii,probe_count",
    [
        (0.0, 1.0, (1.5,), 12),
        (1.0, 0.0, (1.5,), 12),
        (1.0, -1.0, (1.5,), 12),
        (1.0, 1.0, (), 12),
        (1.0, 1.0, (0.5,), 12),
        (1.0, 1.0, (1.5,), 1),
    ],
)
def test_benchmark_rejects_invalid_parameters(radius, speed, probe_radii, probe_count):
    with pytest.raises(ValueError):
        run(1, radius=radius, speed=speed, probe_radii=probe_radii, probe_count=probe_count)


def test_probe_points_are_outside_the_sphere_and_fixed():
    points = velocity_probe_points(1.0, (1.5, 2.5), 12)
    assert points.shape == (24, 3)
    radii = np.linalg.norm(points, axis=1)
    assert np.all(radii > 1.0)
    assert np.isclose(float(np.min(radii)), 1.5)


def test_benchmark_cli_writes_strict_json(tmp_path):
    output = tmp_path / "nested" / "sphere-velocity.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.sphere_velocity",
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
    assert stdout["benchmark"] == "three_dimensional_translating_sphere_velocity"
    json.dumps(stdout, allow_nan=False)
