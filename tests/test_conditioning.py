"""Tests for conditioning diagnostics module."""

import numpy as np
import pytest

from tensorbem.conditioning import (
    ConditioningReport,
    estimate_condition_number,
    estimate_operator_conditions,
)
from tensorbem.geometry3d import icosphere


def test_conditioning_report_structure():
    """Test that ConditioningReport has expected fields."""
    report = ConditioningReport(
        mesh_panels=100,
        single_layer_condition=50.0,
        double_layer_condition=2.5,
        estimation_method="direct",
        power_iterations=0,
    )
    assert report.mesh_panels == 100
    assert report.single_layer_condition == 50.0
    assert report.double_layer_condition == 2.5
    assert report.estimation_method == "direct"
    assert report.power_iterations == 0


def test_conditioning_report_to_dict():
    """Test JSON serialization of ConditioningReport."""
    report = ConditioningReport(
        mesh_panels=100,
        single_layer_condition=50.0,
        double_layer_condition=2.5,
        estimation_method="direct",
        power_iterations=0,
    )
    d = report.to_dict()
    assert d["mesh_panels"] == 100
    assert d["single_layer_condition"] == 50.0
    assert d["double_layer_condition"] == 2.5
    assert d["estimation_method"] == "direct"
    assert d["power_iterations"] == 0


def test_estimate_condition_number_direct():
    """Test direct condition number estimation."""
    # Identity matrix has condition number 1
    matrix = np.eye(10)
    cond, method, iters = estimate_condition_number(matrix, method="direct")
    assert abs(cond - 1.0) < 1e-10
    assert method == "direct"
    assert iters == 0


def test_estimate_condition_number_power_iteration():
    """Test power iteration condition number estimation."""
    # Diagonal matrix with known condition number
    matrix = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    cond, method, iters = estimate_condition_number(
        matrix, method="power_iteration", power_iterations=100
    )
    # Condition number should be 5/1 = 5
    assert abs(cond - 5.0) < 0.1  # Allow some tolerance for iterative method
    assert method == "power_iteration"
    assert iters == 100


def test_estimate_condition_number_auto_small():
    """Test auto method selection for small matrix."""
    matrix = np.eye(50)
    cond, method, iters = estimate_condition_number(matrix, method="auto")
    assert method == "direct"  # Should use direct for small matrices
    assert iters == 0


def test_estimate_condition_number_auto_large():
    """Test auto method selection for large matrix."""
    matrix = np.eye(600)
    cond, method, iters = estimate_condition_number(matrix, method="auto")
    assert method == "power_iteration"  # Should use power iteration for large matrices
    assert iters == 50


def test_estimate_operator_conditions_sphere():
    """Test operator condition estimation on sphere mesh."""
    mesh = icosphere(2, radius=1.0)
    report = estimate_operator_conditions(mesh)

    assert report.mesh_panels == mesh.panel_count
    assert report.single_layer_condition > 0
    assert report.double_layer_condition > 0
    assert report.estimation_method in ["direct", "power_iteration"]


def test_estimate_operator_conditions_interior_exterior():
    """Test that interior and exterior problems give different results."""
    mesh = icosphere(1, radius=1.0)

    ext_report = estimate_operator_conditions(mesh, domain="exterior")
    int_report = estimate_operator_conditions(mesh, domain="interior")

    # Both should have valid condition numbers
    assert ext_report.single_layer_condition > 0
    assert int_report.single_layer_condition > 0
    # Double-layer operators differ between interior and exterior
    assert ext_report.double_layer_condition != int_report.double_layer_condition


def test_sphere_condition_number_known_value():
    """Test that sphere at 128 panels has condition number near expected value.

    From m238 audit: cylinder 128 panels has κ ≈ 76.11.
    Sphere should have similar conditioning.
    """
    # icosphere subdiv=1 gives 80 panels, subdiv=2 gives 320 panels
    # Use subdiv=1 for closer to 128 panels
    mesh = icosphere(1, radius=1.0)
    report = estimate_operator_conditions(mesh)

    # Single-layer condition number should be in reasonable range
    # (not exactly 76 since this is sphere not cylinder, but similar order)
    assert 10 < report.single_layer_condition < 200
    assert report.mesh_panels == 80


def test_condition_number_increases_with_refinement():
    """Test that condition number generally increases with mesh refinement."""
    conditions = []
    for subdiv in range(3):
        mesh = icosphere(subdiv, radius=1.0)
        report = estimate_operator_conditions(mesh)
        conditions.append(report.single_layer_condition)

    # Condition number should generally increase (not strictly monotonic)
    # Just check that finest mesh has higher condition than coarsest
    assert conditions[2] > conditions[0]


def test_condition_number_with_fallback_uses_the_svd_when_finite():
    """A healthy SVD result is reported unchanged, tagged as the SVD."""
    from tensorbem.conditioning import condition_number_with_fallback

    matrix = np.diag([1.0, 7.0])
    condition, source = condition_number_with_fallback(matrix)
    assert source == "svd"
    assert condition == pytest.approx(7.0, rel=1.0e-12)


def test_condition_number_with_fallback_survives_an_infinite_svd(monkeypatch):
    """A non-finite ``numpy.linalg.cond`` must not be read as "singular".

    ``numpy.linalg.cond`` is SVD based and some LAPACK builds return ``inf``
    for perfectly conditioned matrices (loongarch64 compute nodes).  The
    fallback must produce a finite LU-based estimate instead of propagating
    ``inf`` into the solver guards.
    """
    from tensorbem.conditioning import condition_number_with_fallback

    matrix = np.diag([1.0, 7.0])
    monkeypatch.setattr(np.linalg, "cond", lambda *a, **k: float("inf"))
    condition, source = condition_number_with_fallback(matrix)
    assert source == "power_iteration"
    assert np.isfinite(condition)
    assert condition == pytest.approx(7.0, rel=0.5)


def test_condition_number_with_fallback_still_reports_a_singular_matrix(monkeypatch):
    """A genuinely singular matrix stays singular after the fallback."""
    from tensorbem.conditioning import condition_number_with_fallback

    matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
    monkeypatch.setattr(np.linalg, "cond", lambda *a, **k: float("inf"))
    condition, source = condition_number_with_fallback(matrix)
    assert source == "power_iteration"
    assert not np.isfinite(condition) or condition > 1.0e12


@pytest.mark.parametrize("scale", [1e-150, 1., 1e150])
def test_scaled_nonnormal_shear_condition(scale, record_property):
    shear = 10.
    matrix = scale * np.array([[1., shear], [0., 1.]])
    expected = ((np.hypot(shear, 2.)+shear)/2)**2
    measured, _, _ = estimate_condition_number(matrix, method="power_iteration")
    record_property("relative_error", abs(measured/expected - 1.))
    assert measured == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("wide", [False, True])
def test_rectangular_complex_fallback_preserves_singular_values(wide, monkeypatch):
    from tensorbem.conditioning import condition_number_with_fallback

    matrix = np.array([[1., 2.j], [2.j, -4.], [0., 1e-5]])
    if wide:
        matrix = matrix.conj().T
    expected = np.linalg.cond(matrix)
    monkeypatch.setattr(np.linalg, "cond", lambda a: np.inf)
    value, method = condition_number_with_fallback(matrix)
    assert method == "power_iteration_qr"
    assert value == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("refinement", [1, 2])
def test_ellipsoid_condition_estimates_against_svd(refinement, record_property):
    from tensorbem.geometry3d import ellipsoid_mesh
    from tensorbem.solver3d import assemble_surface_matrices

    mesh = ellipsoid_mesh(refinement, (1.5, 1., .75))
    single, double = assemble_surface_matrices(mesh)
    for name, matrix in (("single", single), ("exterior_double", .5*np.eye(mesh.panel_count)-double)):
        reference = np.linalg.cond(matrix)
        value, _, _ = estimate_condition_number(matrix, method="power_iteration")
        error = abs(value/reference-1)
        record_property(name+"_svd", reference)
        record_property(name+"_relative_error", error)
        assert error < .01
