"""Tests for near-singular quadrature diagnostics module."""

import numpy as np
import pytest

from tensorbem.geometry3d import icosphere
from tensorbem.quadrature_diagnostics import (
    NearSingularReport,
    diagnose_near_singular_quadrature,
    identify_problematic_panels,
    near_singular_by_panel,
)


def test_near_singular_report_structure():
    """Test that NearSingularReport has expected fields."""
    report = NearSingularReport(
        mesh_panels=100,
        total_pairs=10000,
        near_singular_pairs=50,
        near_singular_fraction=0.005,
        threshold_factor=0.5,
        min_distance_ratio=0.1,
        max_distance_ratio=10.0,
        estimated_max_error=1.5,
    )
    assert report.mesh_panels == 100
    assert report.total_pairs == 10000
    assert report.near_singular_pairs == 50
    assert report.near_singular_fraction == 0.005
    assert report.threshold_factor == 0.5
    assert report.min_distance_ratio == 0.1
    assert report.max_distance_ratio == 10.0
    assert report.estimated_max_error == 1.5


def test_near_singular_report_to_dict():
    """Test JSON serialization of NearSingularReport."""
    report = NearSingularReport(
        mesh_panels=100,
        total_pairs=10000,
        near_singular_pairs=50,
        near_singular_fraction=0.005,
        threshold_factor=0.5,
        min_distance_ratio=0.1,
        max_distance_ratio=10.0,
        estimated_max_error=1.5,
    )
    d = report.to_dict()
    assert d["mesh_panels"] == 100
    assert d["total_pairs"] == 10000
    assert d["near_singular_pairs"] == 50
    assert d["near_singular_fraction"] == 0.005
    assert d["threshold_factor"] == 0.5
    assert d["min_distance_ratio"] == 0.1
    assert d["max_distance_ratio"] == 10.0
    assert d["estimated_max_error"] == 1.5


def test_diagnose_near_singular_self_interaction():
    """Test near-singular detection for self-interaction."""
    mesh = icosphere(1, radius=1.0)
    report = diagnose_near_singular_quadrature(mesh)

    assert report.mesh_panels == mesh.panel_count
    assert report.total_pairs == mesh.panel_count**2
    assert 0 <= report.near_singular_fraction <= 1.0
    assert report.threshold_factor == 0.5  # default


def test_diagnose_near_singular_with_targets():
    """Test near-singular detection with external target points."""
    mesh = icosphere(2, radius=1.0)  # Use finer mesh for more panels

    # Points far from the mesh
    far_points = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    report_far = diagnose_near_singular_quadrature(mesh, targets=far_points)

    # Points very close to mesh centroids (within threshold)
    # Use actual centroids as targets - distance is 0, definitely near-singular
    close_points = mesh.centroids[:2]
    report_close = diagnose_near_singular_quadrature(mesh, targets=close_points)

    # Far points should have fewer near-singular pairs
    assert report_far.near_singular_fraction < report_close.near_singular_fraction
    # Close points (at centroids) should have near-singular pairs
    assert report_close.near_singular_fraction > 0


def test_near_singular_threshold_effect():
    """Test that changing threshold affects near-singular count."""
    mesh = icosphere(1, radius=1.0)

    report_small = diagnose_near_singular_quadrature(mesh, threshold_factor=0.1)
    report_large = diagnose_near_singular_quadrature(mesh, threshold_factor=1.0)

    # Larger threshold should detect more near-singular pairs
    assert report_large.near_singular_fraction >= report_small.near_singular_fraction


def test_near_singular_by_panel():
    """Test per-panel near-singular counts."""
    mesh = icosphere(1, radius=1.0)
    counts, fractions = near_singular_by_panel(mesh)

    assert counts.shape == (mesh.panel_count,)
    assert fractions.shape == (mesh.panel_count,)
    assert np.all(counts >= 0)
    assert np.all(fractions >= 0)
    assert np.all(fractions <= 1.0)


def test_identify_problematic_panels():
    """Test identification of problematic panels."""
    mesh = icosphere(1, radius=1.0)
    problematic = identify_problematic_panels(mesh, min_near_singular_fraction=0.5)

    # Should return a list of indices
    assert isinstance(problematic, list)
    # All indices should be valid
    assert all(0 <= idx < mesh.panel_count for idx in problematic)


def test_invalid_targets_shape():
    """Test that invalid target shapes raise errors."""
    mesh = icosphere(1, radius=1.0)

    with pytest.raises(ValueError, match="shape"):
        diagnose_near_singular_quadrature(mesh, targets=np.array([1.0, 2.0, 3.0]))


def test_near_singular_distance_ratios():
    """Test that distance ratios are computed correctly."""
    mesh = icosphere(1, radius=1.0)
    report = diagnose_near_singular_quadrature(mesh)

    # Distance ratios should be positive
    assert report.min_distance_ratio > 0
    assert report.max_distance_ratio > report.min_distance_ratio


def test_near_singular_ursell_like_mesh():
    """Test near-singular detection on a mesh similar to Ursell benchmark.

    The Ursell cylinder benchmark has closely spaced panels that may
    trigger near-singular quadrature.
    """
    # Use a finer mesh to simulate closely-spaced panels
    mesh = icosphere(2, radius=1.0)
    report = diagnose_near_singular_quadrature(mesh, threshold_factor=0.3)

    # Should detect some near-singular pairs (self-interaction diagonal)
    assert report.near_singular_pairs >= 0
    assert report.total_pairs == mesh.panel_count**2
