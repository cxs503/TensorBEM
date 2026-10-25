"""Tests for integrated diagnostics report module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensorbem.diagnostics_report import (
    DiagnosticsReport,
    load_diagnostics_report,
    run_diagnostics,
    save_diagnostics_report,
)
from tensorbem.geometry3d import icosphere, ellipsoid_mesh


def test_diagnostics_report_structure():
    """Test that DiagnosticsReport has expected fields."""
    from tensorbem.conditioning import ConditioningReport
    from tensorbem.quadrature_diagnostics import NearSingularReport
    from tensorbem.mesh_quality import MeshQualityReport

    conditioning = ConditioningReport(
        mesh_panels=100,
        single_layer_condition=50.0,
        double_layer_condition=2.5,
        estimation_method="direct",
        power_iterations=0,
    )
    quadrature = NearSingularReport(
        mesh_panels=100,
        total_pairs=10000,
        near_singular_pairs=50,
        near_singular_fraction=0.005,
        threshold_factor=0.5,
        min_distance_ratio=0.1,
        max_distance_ratio=10.0,
        estimated_max_error=1.5,
    )
    mesh_quality = MeshQualityReport(
        mesh_panels=100,
        min_area=0.01,
        max_area=0.02,
        area_ratio=2.0,
        min_angle_deg=45.0,
        max_angle_deg=90.0,
        mean_min_angle_deg=55.0,
        min_aspect_ratio=1.0,
        max_aspect_ratio=2.0,
        mean_aspect_ratio=1.5,
        panels_below_30_deg=0,
        panels_above_5_aspect=0,
    )

    report = DiagnosticsReport(
        benchmark_name="test_benchmark",
        mesh_description="test mesh",
        conditioning=conditioning,
        quadrature=quadrature,
        mesh_quality=mesh_quality,
    )

    assert report.benchmark_name == "test_benchmark"
    assert report.mesh_description == "test mesh"
    assert report.conditioning.mesh_panels == 100
    assert report.quadrature.mesh_panels == 100
    assert report.mesh_quality.mesh_panels == 100


def test_diagnostics_report_to_dict():
    """Test JSON serialization of DiagnosticsReport."""
    from tensorbem.conditioning import ConditioningReport
    from tensorbem.quadrature_diagnostics import NearSingularReport
    from tensorbem.mesh_quality import MeshQualityReport

    conditioning = ConditioningReport(
        mesh_panels=100,
        single_layer_condition=50.0,
        double_layer_condition=2.5,
        estimation_method="direct",
        power_iterations=0,
    )
    quadrature = NearSingularReport(
        mesh_panels=100,
        total_pairs=10000,
        near_singular_pairs=50,
        near_singular_fraction=0.005,
        threshold_factor=0.5,
        min_distance_ratio=0.1,
        max_distance_ratio=10.0,
        estimated_max_error=1.5,
    )
    mesh_quality = MeshQualityReport(
        mesh_panels=100,
        min_area=0.01,
        max_area=0.02,
        area_ratio=2.0,
        min_angle_deg=45.0,
        max_angle_deg=90.0,
        mean_min_angle_deg=55.0,
        min_aspect_ratio=1.0,
        max_aspect_ratio=2.0,
        mean_aspect_ratio=1.5,
        panels_below_30_deg=0,
        panels_above_5_aspect=0,
    )

    report = DiagnosticsReport(
        benchmark_name="test_benchmark",
        mesh_description="test mesh",
        conditioning=conditioning,
        quadrature=quadrature,
        mesh_quality=mesh_quality,
    )

    d = report.to_dict()
    assert d["benchmark_name"] == "test_benchmark"
    assert d["mesh_description"] == "test mesh"
    assert "conditioning" in d
    assert "quadrature" in d
    assert "mesh_quality" in d


def test_diagnostics_report_to_json():
    """Test JSON string generation."""
    from tensorbem.conditioning import ConditioningReport
    from tensorbem.quadrature_diagnostics import NearSingularReport
    from tensorbem.mesh_quality import MeshQualityReport

    conditioning = ConditioningReport(
        mesh_panels=100,
        single_layer_condition=50.0,
        double_layer_condition=2.5,
        estimation_method="direct",
        power_iterations=0,
    )
    quadrature = NearSingularReport(
        mesh_panels=100,
        total_pairs=10000,
        near_singular_pairs=50,
        near_singular_fraction=0.005,
        threshold_factor=0.5,
        min_distance_ratio=0.1,
        max_distance_ratio=10.0,
        estimated_max_error=1.5,
    )
    mesh_quality = MeshQualityReport(
        mesh_panels=100,
        min_area=0.01,
        max_area=0.02,
        area_ratio=2.0,
        min_angle_deg=45.0,
        max_angle_deg=90.0,
        mean_min_angle_deg=55.0,
        min_aspect_ratio=1.0,
        max_aspect_ratio=2.0,
        mean_aspect_ratio=1.5,
        panels_below_30_deg=0,
        panels_above_5_aspect=0,
    )

    report = DiagnosticsReport(
        benchmark_name="test_benchmark",
        mesh_description="test mesh",
        conditioning=conditioning,
        quadrature=quadrature,
        mesh_quality=mesh_quality,
    )

    json_str = report.to_json()
    # Should be valid JSON
    parsed = json.loads(json_str)
    assert parsed["benchmark_name"] == "test_benchmark"


def test_diagnostics_report_to_markdown():
    """Test Markdown report generation."""
    from tensorbem.conditioning import ConditioningReport
    from tensorbem.quadrature_diagnostics import NearSingularReport
    from tensorbem.mesh_quality import MeshQualityReport

    conditioning = ConditioningReport(
        mesh_panels=100,
        single_layer_condition=50.0,
        double_layer_condition=2.5,
        estimation_method="direct",
        power_iterations=0,
    )
    quadrature = NearSingularReport(
        mesh_panels=100,
        total_pairs=10000,
        near_singular_pairs=50,
        near_singular_fraction=0.005,
        threshold_factor=0.5,
        min_distance_ratio=0.1,
        max_distance_ratio=10.0,
        estimated_max_error=1.5,
    )
    mesh_quality = MeshQualityReport(
        mesh_panels=100,
        min_area=0.01,
        max_area=0.02,
        area_ratio=2.0,
        min_angle_deg=45.0,
        max_angle_deg=90.0,
        mean_min_angle_deg=55.0,
        min_aspect_ratio=1.0,
        max_aspect_ratio=2.0,
        mean_aspect_ratio=1.5,
        panels_below_30_deg=0,
        panels_above_5_aspect=0,
    )

    report = DiagnosticsReport(
        benchmark_name="test_benchmark",
        mesh_description="test mesh",
        conditioning=conditioning,
        quadrature=quadrature,
        mesh_quality=mesh_quality,
    )

    md = report.to_markdown()
    assert "test_benchmark" in md
    assert "test mesh" in md
    assert "Operator Condition Numbers" in md
    assert "Near-Singular Quadrature" in md
    assert "Mesh Quality" in md


def test_run_diagnostics_sphere():
    """Test run_diagnostics on sphere mesh."""
    mesh = icosphere(1, radius=1.0)
    report = run_diagnostics(
        mesh,
        benchmark_name="sphere_test",
        mesh_description="icosphere subdiv=1",
    )

    assert report.benchmark_name == "sphere_test"
    assert report.mesh_description == "icosphere subdiv=1"
    assert report.conditioning.mesh_panels == mesh.panel_count
    assert report.quadrature.mesh_panels == mesh.panel_count
    assert report.mesh_quality.mesh_panels == mesh.panel_count


def test_run_diagnostics_ellipsoid():
    """Test run_diagnostics on ellipsoid mesh."""
    mesh = ellipsoid_mesh(2, semi_axes=(1.5, 1.0, 0.75))
    report = run_diagnostics(
        mesh,
        benchmark_name="ellipsoid_test",
        mesh_description="ellipsoid (1.5, 1.0, 0.75)",
    )

    assert report.benchmark_name == "ellipsoid_test"
    assert report.conditioning.mesh_panels == mesh.panel_count


def test_save_and_load_diagnostics_report():
    """Test saving and loading diagnostics report."""
    mesh = icosphere(1, radius=1.0)
    report = run_diagnostics(
        mesh,
        benchmark_name="test_save_load",
        mesh_description="test mesh",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_diagnostics_report(report, tmpdir, prefix="test_diag")

        # Check files were created
        assert paths["json"].exists()
        assert paths["markdown"].exists()

        # Load and verify
        loaded = load_diagnostics_report(paths["json"])
        assert loaded["benchmark_name"] == "test_save_load"
        assert loaded["conditioning"]["mesh_panels"] == mesh.panel_count


def test_diagnostics_report_json_parseable():
    """Test that JSON output can be parsed by downstream tools."""
    mesh = icosphere(1, radius=1.0)
    report = run_diagnostics(mesh, benchmark_name="json_test")

    json_str = report.to_json()
    # Should be valid JSON
    parsed = json.loads(json_str)

    # Should have all expected keys
    assert "benchmark_name" in parsed
    assert "mesh_description" in parsed
    assert "conditioning" in parsed
    assert "quadrature" in parsed
    assert "mesh_quality" in parsed

    # Conditioning should have expected keys
    assert "single_layer_condition" in parsed["conditioning"]
    assert "double_layer_condition" in parsed["conditioning"]

    # Quadrature should have expected keys
    assert "near_singular_fraction" in parsed["quadrature"]
    assert "total_pairs" in parsed["quadrature"]

    # Mesh quality should have expected keys
    assert "min_angle_deg" in parsed["mesh_quality"]
    assert "area_ratio" in parsed["mesh_quality"]
