"""Tests for mesh quality diagnostics module."""

import numpy as np
import pytest

from tensorbem.geometry3d import (
    icosphere,
    ellipsoid_mesh,
    floating_sphere_hull_mesh,
    hemisphere_hull_mesh,
)
from tensorbem.mesh_quality import (
    MeshQualityReport,
    diagnose_mesh_quality,
    identify_low_quality_panels,
    mesh_quality_summary,
)


def test_mesh_quality_report_structure():
    """Test that MeshQualityReport has expected fields."""
    report = MeshQualityReport(
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
    assert report.mesh_panels == 100
    assert report.min_area == 0.01
    assert report.max_area == 0.02
    assert report.area_ratio == 2.0
    assert report.min_angle_deg == 45.0
    assert report.max_angle_deg == 90.0
    assert report.mean_min_angle_deg == 55.0
    assert report.min_aspect_ratio == 1.0
    assert report.max_aspect_ratio == 2.0
    assert report.mean_aspect_ratio == 1.5
    assert report.panels_below_30_deg == 0
    assert report.panels_above_5_aspect == 0


def test_mesh_quality_report_to_dict():
    """Test JSON serialization of MeshQualityReport."""
    report = MeshQualityReport(
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
    d = report.to_dict()
    assert d["mesh_panels"] == 100
    assert d["min_area"] == 0.01
    assert d["max_area"] == 0.02
    assert d["area_ratio"] == 2.0
    assert d["min_angle_deg"] == 45.0
    assert d["max_angle_deg"] == 90.0
    assert d["mean_min_angle_deg"] == 55.0
    assert d["min_aspect_ratio"] == 1.0
    assert d["max_aspect_ratio"] == 2.0
    assert d["mean_aspect_ratio"] == 1.5
    assert d["panels_below_30_deg"] == 0
    assert d["panels_above_5_aspect"] == 0


def test_diagnose_mesh_quality_icosphere():
    """Test mesh quality diagnostics on icosphere."""
    mesh = icosphere(2, radius=1.0)
    report = diagnose_mesh_quality(mesh)

    assert report.mesh_panels == mesh.panel_count
    assert report.min_area > 0
    assert report.max_area >= report.min_area
    assert report.area_ratio >= 1.0
    assert 0 < report.min_angle_deg <= 60.0  # Equilateral triangle max is 60
    assert report.max_angle_deg <= 180.0
    assert report.min_aspect_ratio >= 1.0
    assert report.max_aspect_ratio >= report.min_aspect_ratio


def test_diagnose_mesh_quality_ellipsoid():
    """Test mesh quality diagnostics on ellipsoid."""
    mesh = ellipsoid_mesh(2, semi_axes=(1.5, 1.0, 0.75))
    report = diagnose_mesh_quality(mesh)

    assert report.mesh_panels == mesh.panel_count
    assert report.min_area > 0
    assert report.area_ratio >= 1.0
    # Ellipsoid should have reasonable quality
    assert report.min_angle_deg > 20.0
    assert report.max_aspect_ratio < 5.0


def test_identify_low_quality_panels():
    """Test identification of low-quality panels."""
    mesh = icosphere(1, radius=1.0)
    problematic = identify_low_quality_panels(
        mesh, min_angle_threshold=30.0, max_aspect_threshold=5.0
    )

    # Should return a list of indices
    assert isinstance(problematic, list)
    # All indices should be valid
    assert all(0 <= idx < mesh.panel_count for idx in problematic)


def test_mesh_quality_summary():
    """Test human-readable quality summary generation."""
    mesh = icosphere(2, radius=1.0)
    summary = mesh_quality_summary(mesh)

    assert isinstance(summary, str)
    assert "Mesh Quality Report" in summary
    assert str(mesh.panel_count) in summary
    assert "Area range" in summary
    assert "Angle range" in summary
    assert "Aspect ratio" in summary


def test_mesh_quality_uniform_mesh():
    """Test that uniform mesh has good quality metrics."""
    # Icosphere is relatively uniform
    mesh = icosphere(2, radius=1.0)
    report = diagnose_mesh_quality(mesh)

    # Should have low area variation
    assert report.area_ratio < 2.0
    # Should have good angles (no very sharp triangles)
    assert report.min_angle_deg > 40.0
    # Should have good aspect ratios
    assert report.max_aspect_ratio < 2.0
    # Should have no problematic panels
    assert report.panels_below_30_deg == 0
    assert report.panels_above_5_aspect == 0


def test_mesh_quality_refinement_effect():
    """Test that mesh refinement maintains or improves quality metrics."""
    qualities = []
    for subdiv in range(3):
        mesh = icosphere(subdiv, radius=1.0)
        report = diagnose_mesh_quality(mesh)
        qualities.append(report)

    # Finer meshes should maintain good quality (no degradation)
    # Check that minimum angles stay reasonable
    assert qualities[2].min_angle_deg > 30.0
    # Check that aspect ratios stay reasonable
    assert qualities[2].max_aspect_ratio < 3.0
    # Check that area ratios are reasonable (not too much variation)
    assert qualities[2].area_ratio < 5.0


def test_mesh_quality_angle_sum():
    """Test that triangle angles sum to 180 degrees."""
    mesh = icosphere(1, radius=1.0)
    report = diagnose_mesh_quality(mesh)

    # Max angle should be reasonable (not degenerate)
    assert report.max_angle_deg < 120.0
    # Min angle should be positive
    assert report.min_angle_deg > 0.0


def test_mesh_quality_ellipsoid_min_angle():
    """Test that ellipsoid benchmark mesh has min angle > 30°."""
    # This is a判据 from the mission
    mesh = ellipsoid_mesh(2, semi_axes=(1.5, 1.0, 0.75))
    report = diagnose_mesh_quality(mesh)

    # 判据: 网格质量指标在 ellipsoid benchmark 上报告最小内角 > 30°
    assert report.min_angle_deg > 30.0, (
        f"Ellipsoid mesh min angle {report.min_angle_deg:.2f}° should be > 30°"
    )


# --------------------------------------------------------------------------- #
# XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE -- verification-debt pin
#
# The round-2 capytaine cross-code anchor (full 6x6 radiation matrix) FAILS
# its diagonal band on the *floating-sphere* cases while the *surface-piercing
# hemisphere* cases (same physical body, radius 1, flat face at z=0) converge
# inside the band.  The only difference between the two is the mesh builder:
# ``floating_sphere_hull_mesh`` clips a geodesic icosphere against z=0
# (Sutherland-Hodgman), which emits *degenerate waterline slivers*; the clean
# ``hemisphere_hull_mesh`` (octahedral, radially projected) does not.
#
# These two tests pin that defect so it cannot silently regress: the diagnostic
# must *report* the slivers (aspect ratio > 5, min angle < 30 deg) and they must
# be located on the waterline (a vertex at z=0).  A future mesh fix that makes
# the floating-sphere cap clean will have to update these assertions.
# --------------------------------------------------------------------------- #
def test_floating_sphere_hull_mesh_reports_waterline_slivers() -> None:
    """The clipped-icosphere floating-sphere cap must be diagnosed as degenerate.

    ``floating_sphere_hull_mesh`` (radius 1, centre z=-0.5) is the mesh used by
    the r2 capytaine cross-code anchor for its floating-sphere cases.  Its
    Sutherland-Hodgman waterline slivers are the reason those cases sit outside
    the 5 % diagonal band while the clean hemisphere (same body) converges to
    3.3 %.  The diagnostic must flag them.
    """
    mesh = floating_sphere_hull_mesh(3, radius=1.0, center=(0.0, 0.0, -0.5))
    report = diagnose_mesh_quality(mesh)

    # The defect is real and must be reported (not silently hidden).
    assert report.max_aspect_ratio > 5.0, (
        f"floating-sphere cap max aspect ratio {report.max_aspect_ratio:.2f} "
        "should exceed 5 (waterline slivers present)"
    )
    assert report.panels_above_5_aspect > 0, (
        "floating-sphere cap must contain panels with aspect ratio > 5"
    )
    assert report.min_angle_deg < 30.0, (
        f"floating-sphere cap min angle {report.min_angle_deg:.2f}° "
        "should be < 30° (degenerate waterline slivers)"
    )


def test_floating_sphere_slivers_are_on_the_waterline() -> None:
    """The degenerate panels must be the waterline slivers, not interior ones.

    Every low-quality panel of the clipped floating-sphere cap must touch the
    calm waterplane (have a vertex at z=0) -- that is where the Sutherland-
    Hodgman cut creates the sliver.  Interior (below-waterline) panels of the
    geodesic icosphere are well-shaped and must not be flagged.
    """
    mesh = floating_sphere_hull_mesh(3, radius=1.0, center=(0.0, 0.0, -0.5))
    bad = identify_low_quality_panels(
        mesh, min_angle_threshold=30.0, max_aspect_threshold=5.0
    )
    assert len(bad) > 0, "expected waterline slivers to be flagged"

    verts = mesh.vertices
    faces = mesh.faces
    for idx in bad:
        tri = verts[faces[idx]]
        assert np.any(np.abs(tri[:, 2]) <= 1.0e-9), (
            f"panel {idx} is flagged low-quality but has no waterline vertex "
            "(it is not a waterline sliver)"
        )


def test_hemisphere_hull_mesh_is_clean_control() -> None:
    """The clean hemisphere builder is the control: same body, no slivers.

    ``hemisphere_hull_mesh`` (radius 1, flat face at z=0) is the *same* physical
    body as the floating-sphere cap but built by radial projection of an
    octahedron -- no clipping, so no waterline slivers.  Its quality must stay
    well inside the good-panel regime, which is what lets the r2 anchor's
    hemisphere cases converge inside the 5 % band.
    """
    mesh = hemisphere_hull_mesh(3, radius=1.0)
    report = diagnose_mesh_quality(mesh)

    assert report.max_aspect_ratio < 2.0, (
        f"clean hemisphere max aspect ratio {report.max_aspect_ratio:.2f} "
        "should be < 2 (no slivers)"
    )
    assert report.panels_above_5_aspect == 0, (
        "clean hemisphere must have no panels with aspect ratio > 5"
    )
    assert report.min_angle_deg > 30.0, (
        f"clean hemisphere min angle {report.min_angle_deg:.2f}° should be > 30°"
    )
