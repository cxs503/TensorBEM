"""Tests for 2D near-singular quadrature diagnostics.

Covers ``tensorbem.quadrature_diagnostics_2d.diagnose_near_singular_quadrature_2d``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry import PanelMesh
from tensorbem.quadrature_diagnostics_2d import diagnose_near_singular_quadrature_2d


def _regular_polygon_mesh(n: int, radius: float = 1.0, center: tuple = (0.0, 0.0)) -> PanelMesh:
    """Create a PanelMesh from a regular n-gon."""
    angles = 2.0 * math.pi * np.arange(n) / n
    vertices = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])
    return PanelMesh(vertices)


def test_unit_circle_near_singular_ratio() -> None:
    """Unit circle mesh: near-singular ratio should be small for uniform panels."""
    mesh = _regular_polygon_mesh(64, radius=1.0)
    result = diagnose_near_singular_quadrature_2d(mesh, threshold=4.0)

    assert result["near_singular_count"] > 0
    assert result["total_pairs"] == 64 * 63
    assert 0.0 < result["near_singular_ratio"] < 0.15
    assert abs(result["max_panel_length"] - result["min_panel_length"]) < 1.0e-10
    assert abs(result["mean_panel_length"] - result["max_panel_length"]) < 1.0e-10


def test_ursell_submerged_cylinder_near_singular() -> None:
    """Ursell submerged cylinder: should detect ~6% near-singular pairs at 128 panels."""
    # Submerged circle centred at (0, -2) with radius 1
    angles = 2.0 * math.pi * np.arange(128) / 128
    vertices = np.column_stack([
        np.cos(angles),
        np.sin(angles) - 2.0,
    ])
    mesh = PanelMesh(vertices)
    result = diagnose_near_singular_quadrature_2d(mesh, threshold=4.0)

    # The audit reported ~6.30% near-singular ratio at 128 panels
    assert result["near_singular_ratio"] > 0.05
    assert result["near_singular_ratio"] < 0.10
    assert result["total_pairs"] == 128 * 127


def test_threshold_sensitivity() -> None:
    """Near-singular count should increase with larger threshold."""
    mesh = _regular_polygon_mesh(32, radius=1.0)
    r_small = diagnose_near_singular_quadrature_2d(mesh, threshold=2.0)
    r_large = diagnose_near_singular_quadrature_2d(mesh, threshold=8.0)

    assert r_large["near_singular_count"] > r_small["near_singular_count"]
    assert r_large["near_singular_ratio"] > r_small["near_singular_ratio"]


def test_single_panel_edge_case() -> None:
    """A mesh with very few panels should handle gracefully."""
    mesh = _regular_polygon_mesh(8, radius=1.0)
    result = diagnose_near_singular_quadrature_2d(mesh, threshold=4.0)

    assert result["total_pairs"] == 8 * 7
    assert result["near_singular_count"] >= 0
    assert 0.0 <= result["near_singular_ratio"] <= 1.0


def test_non_uniform_mesh() -> None:
    """Non-uniform panel lengths should produce different min/max."""
    # Create a mesh with varying panel lengths
    angles = np.array([0.0, 0.3, 0.7, 1.2, 1.8, 2.5, 3.3, 4.2, 5.2, 6.28])
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    mesh = PanelMesh(vertices)
    result = diagnose_near_singular_quadrature_2d(mesh, threshold=4.0)

    assert result["max_panel_length"] > result["min_panel_length"]
    assert result["total_pairs"] == 10 * 9


def test_return_keys_complete() -> None:
    """All expected keys should be present in the result dict."""
    mesh = _regular_polygon_mesh(16, radius=1.0)
    result = diagnose_near_singular_quadrature_2d(mesh)

    expected_keys = {
        "near_singular_count",
        "total_pairs",
        "near_singular_ratio",
        "max_panel_length",
        "min_panel_length",
        "mean_panel_length",
        "threshold",
    }
    assert set(result.keys()) == expected_keys


def test_diagonal_exclusion() -> None:
    """Self-interaction (diagonal) pairs must be excluded from near-singular count."""
    mesh = _regular_polygon_mesh(16, radius=1.0)
    result = diagnose_near_singular_quadrature_2d(mesh, threshold=100.0)

    # Even with huge threshold, diagonal should be excluded
    assert result["near_singular_count"] <= result["total_pairs"]
    assert result["total_pairs"] == 16 * 15  # n*(n-1), no diagonal


def test_panel_length_properties() -> None:
    """Panel lengths should be positive and consistent."""
    mesh = _regular_polygon_mesh(32, radius=2.0)
    result = diagnose_near_singular_quadrature_2d(mesh)

    assert result["min_panel_length"] > 0.0
    assert result["max_panel_length"] >= result["min_panel_length"]
    assert result["mean_panel_length"] >= result["min_panel_length"]
    assert result["mean_panel_length"] <= result["max_panel_length"]
