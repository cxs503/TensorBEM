"""Verification tests for the multiply connected (annular) 2-D Laplace solver.

The oracle is always closed form -- ``u = A log r + B`` and
``u = (r + c^2/r) cos(theta)`` -- never another discretisation.  Four things are
asserted:

* the analytic normal derivatives used as references really are the derivatives
  of the analytic potentials (central differences);
* the annular solve converges to the closed form at the constant-panel rate and
  beats the 3 % acceptance target on the finest mesh of the study artifact;
* the discrete boundary flux is divergence free and matches the analytic
  ``2 pi A`` on each loop, which a solver that silently ignored the hole cannot
  reproduce;
* the *orientation convention* is load bearing: leaving the inner loop's normals
  unflipped destroys the solution, so the convention is pinned by a measurement
  rather than by a comment.

The artifact test re-runs the published refinement levels and compares them with
``docs/annulus_laplace/study.json``, so the numbers quoted in
``docs/annulus_laplace.md`` cannot drift.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import annulus_laplace as study
from tensorbem.multiloop import (
    MultiLoopMesh,
    annulus_mesh,
    boundary_flux,
    evaluate_multiloop_field,
    solve_laplace_multiloop,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "docs" / "annulus_laplace" / "study.json"
QUICK_LADDER = ((16, 8), (32, 16), (64, 32))


def _finite_difference_normal_derivative(potential, points, normals, step=1e-6):
    gradient = np.empty_like(points)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        gradient[:, axis] = (
            potential(points + offset) - potential(points - offset)
        ) / (2.0 * step)
    return np.sum(gradient * normals, axis=1)


def _ring_points(radii=(0.55, 0.75, 0.95), angles=24):
    theta = 2.0 * np.pi * np.arange(angles) / angles
    return np.asarray(
        [
            [radius * math.cos(angle), radius * math.sin(angle)]
            for radius in radii
            for angle in theta
        ]
    )


@pytest.mark.parametrize("case", study.CASES, ids=[case.name for case in study.CASES])
def test_analytic_normal_derivatives_match_finite_differences(case):
    """The reference ``du/dn`` must be the derivative of the reference ``u``."""
    points = _ring_points()
    normals = points / np.linalg.norm(points, axis=1)[:, None]
    numeric = _finite_difference_normal_derivative(case.potential, points, normals)
    analytic = case.normal_derivative(points, normals)
    assert np.max(np.abs(numeric - analytic)) <= 1.0e-6 * np.max(np.abs(analytic))


def test_mesh_geometry_and_normals():
    """Two loops, panel counts preserved, and inner normals point into the hole."""
    mesh = annulus_mesh(32, 16, 1.0, 0.5)
    assert [loop.panel_count for loop in mesh.loops] == [32, 16]
    assert mesh.panel_count == 48
    assert mesh.hole_loops == (1,)
    centers = mesh.centers
    radial = centers / np.linalg.norm(centers, axis=1)[:, None]
    alignment = np.sum(mesh.normals * radial, axis=1)
    outer = slice(0, 32)
    inner = slice(32, 48)
    assert np.all(alignment[outer] > 0.99)  # domain-outward on the outer loop
    assert np.all(alignment[inner] < -0.99)  # into the hole on the inner loop
    np.testing.assert_allclose(np.linalg.norm(centers[outer], axis=1), 1.0, atol=0.01)
    np.testing.assert_allclose(np.linalg.norm(centers[inner], axis=1), 0.5, atol=0.01)


def test_annulus_mesh_validation():
    with pytest.raises(ValueError):
        annulus_mesh(8, 8, 1.0, 1.0)  # inner not smaller than outer
    with pytest.raises(ValueError):
        annulus_mesh(2, 8)  # at least three panels per loop
    with pytest.raises(ValueError):
        MultiLoopMesh(loops=(annulus_mesh(8, 4).loops[0],))  # single loop
    with pytest.raises(ValueError):
        MultiLoopMesh(loops=annulus_mesh(8, 4).loops, hole_loops=(2,))
    with pytest.raises(ValueError):
        MultiLoopMesh(loops=annulus_mesh(8, 4).loops, hole_loops=(0, 0))


@pytest.mark.parametrize("case", study.CASES, ids=[case.name for case in study.CASES])
def test_convergence_to_closed_form(case):
    """Error falls at the constant-panel rate and the finest level beats 3 %."""
    levels = [study.measure_level(case, no, ni) for no, ni in QUICK_LADDER]
    errors = [level["q_relative_l2_error"] for level in levels]
    rmse = [level["field_rmse"] for level in levels]
    assert errors == sorted(errors, reverse=True), errors
    assert rmse == sorted(rmse, reverse=True), rmse
    orders = study.observed_orders(levels, "q_relative_l2_error")
    for order in orders:
        assert 1.5 <= order <= 2.4, orders
    assert errors[-1] < 0.03  # acceptance target of the study artifact
    assert rmse[-1] < 1.0e-3
    for level in levels:
        assert level["relative_residual"] < 1.0e-12


def test_interior_field_matches_closed_form():
    case = study.CASES[0]
    mesh = annulus_mesh(64, 32)
    result = solve_laplace_multiloop(mesh, case.potential(mesh.centers))
    points = _ring_points(radii=(0.52, 0.62, 0.75, 0.88, 0.98))
    numeric = evaluate_multiloop_field(mesh, result, points)
    exact = case.potential(points)
    assert np.max(np.abs(numeric - exact)) < 1.0e-3


def test_boundary_flux_is_divergence_free():
    """Flux in = flux out on the annulus, and the loop fluxes hit the closed form."""
    case = study.CASES[0]
    mesh = annulus_mesh(64, 32)
    result = solve_laplace_multiloop(mesh, case.potential(mesh.centers))
    outer_flux, inner_flux = boundary_flux(mesh, result.normal_derivative)
    exact = 2.0 * np.pi * study.RADIAL_A
    assert abs(outer_flux - exact) / exact < 0.01
    assert abs(inner_flux + exact) / exact < 0.01
    assert abs(outer_flux + inner_flux) / exact < 0.01

    # angular case: a harmonic field with no log term has zero flux through a
    # closed loop, so the discrete fluxes must cancel to roundoff-like level
    angular = study.CASES[1]
    result = solve_laplace_multiloop(mesh, angular.potential(mesh.centers))
    flux = boundary_flux(mesh, result.normal_derivative)
    scale = float(np.sum(np.abs(result.normal_derivative) * mesh.lengths))
    assert abs(flux[0] + flux[1]) / scale < 1.0e-10


def test_inner_loop_orientation_is_load_bearing():
    """Unflipped inner normals must fail by orders of magnitude, not marginally."""
    case = study.CASES[0]
    correct = annulus_mesh(64, 32)
    unflipped = MultiLoopMesh(loops=correct.loops, hole_loops=())
    exact_q = case.normal_derivative(correct.centers, correct.normals)
    good = solve_laplace_multiloop(correct, case.potential(correct.centers))
    bad = solve_laplace_multiloop(unflipped, case.potential(unflipped.centers))
    good_error = np.linalg.norm(good.normal_derivative - exact_q) / np.linalg.norm(exact_q)
    bad_error = np.linalg.norm(bad.normal_derivative - exact_q) / np.linalg.norm(exact_q)
    assert good_error < 0.03
    assert bad_error > 100.0 * good_error
    # the wrong convention is not merely inaccurate: it is O(1) wrong
    assert bad_error > 0.5


def test_artifact_passes_its_own_acceptance_and_reproduces():
    """The committed artifact must pass its declared thresholds and not drift."""
    assert ARTIFACT.exists(), f"missing artifact {ARTIFACT}"
    study_payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    ok, messages = study.verdict(study_payload)
    assert ok, messages
    assert [f"{no}/{ni}" for no, ni in study.LADDER] == study_payload["conventions"]["ladder"]

    # reproduce the committed (64, 32) level from the shipped code
    case = next(
        item for item in study.CASES if item.name == study_payload["cases"][0]["name"]
    )
    committed = study_payload["cases"][0]["levels"][2]
    assert (committed["outer_panels"], committed["inner_panels"]) == (64, 32)
    fresh = study.measure_level(case, 64, 32)
    for key in (
        "q_relative_l2_error",
        "q_relative_linf_error",
        "field_rmse",
        "flux_relative_error",
        "condition_number",
    ):
        assert fresh[key] == pytest.approx(committed[key], rel=1.0e-8, abs=1.0e-14), key


def test_orientation_control_block_of_artifact():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    controls = payload["orientation_control"]
    assert len(controls) == len(study.CASES)
    for control in controls:
        assert (
            control["unflipped_inner_loop_q_relative_l2_error"]
            > 100.0 * control["correct_convention_q_relative_l2_error"]
        )

def test_convergence_figure_renders_from_artifact(tmp_path):
    """The published figure must be reproducible from the artifact alone."""
    from tensorbem.benchmarks import annulus_figures

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    path = annulus_figures.plot_convergence(payload, tmp_path)
    assert path.exists() and path.stat().st_size > 10_000
