"""Tests for the closed-form regular-polygon potential flow and its use as a
second reference for the cylinder anchor case."""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks import case_cylinder_stream as anchor
from tensorbem.benchmarks import case_cylinder_polygon_exact as study
from tensorbem.benchmarks import polygon_stream_exact as pe
from tensorbem.geometry import regular_polygon

LADDER = (8, 16, 32, 64, 128)


@pytest.mark.parametrize("panels", LADDER)
def test_map_reproduces_the_polygon_geometry(panels: int) -> None:
    gate = pe.geometry_gate(panels)
    assert gate["edge_length_relative_error"] <= 1.0e-6
    assert gate["midpoint_relative_error"] <= 1.0e-6
    assert gate["vertex_radius_relative_error"] <= 1.0e-6
    assert gate["rotation_symmetry_residual"] <= 1.0e-12
    assert gate["inverse_map_roundtrip"] == 0.0


@pytest.mark.parametrize("panels", LADDER)
def test_closed_form_wall_velocity_satisfies_the_boundary_condition(panels: int) -> None:
    mesh = regular_polygon(panels, 1.0)
    velocity, speed = pe.surface_midpoint_velocity(panels, 1.0, 1.0)
    normal = np.sum(velocity * mesh.normals, axis=1)
    assert np.max(np.abs(normal)) <= 1.0e-12
    theta = pe.surface_midpoint_angles(panels)
    alpha = 2.0 / panels
    assert np.allclose(speed, 2.0 * np.abs(np.sin(theta)) / 2.0**alpha, atol=1e-14)


@pytest.mark.parametrize("panels", LADDER)
def test_closed_form_velocity_is_harmonic_and_lift_free(panels: int) -> None:
    gate = study.exact_solution_gates(panels)
    assert gate["no_flux_residual_over_speed"] <= 1.0e-12
    assert gate["mirror_symmetry_residual_over_speed"] <= 1.0e-12
    assert gate["far_field_velocity_error_over_speed"] <= 1.0e-5
    assert gate["neumann_data_consistency_error_over_speed"] <= 1.0e-5


def test_map_derivative_agrees_with_finite_differences() -> None:
    panels = 16
    w = np.array([1.4 + 0.3j, 2.0 - 0.5j, 1.2 + 0.0j])
    step = 1.0e-6
    finite = (
        pe.map_points(w + step, panels) - pe.map_points(w - step, panels)
    ) / (2.0 * step)
    assert np.allclose(pe.map_derivative(w, panels), finite, rtol=1e-8, atol=1e-10)


def test_wall_speed_is_below_the_circular_cylinder_value() -> None:
    """A polygon inscribed in the circle is a smaller body: slower wall flow."""
    for panels in LADDER:
        _, speed = pe.surface_midpoint_velocity(panels, 1.0, 1.0)
        theta = pe.surface_midpoint_angles(panels)
        cylinder = 2.0 * np.abs(np.sin(theta))
        ratio = speed / cylinder
        assert np.all(ratio < 1.0)
        assert np.allclose(ratio, 2.0 ** (-2.0 / panels), rtol=1e-14)


def test_budget_level_meets_the_anchor_tolerance() -> None:
    level = study.run_level(study.BUDGET_PANELS)
    assert level["neumann"]["surface_velocity_relative_l2"] < study.TOLERANCE
    assert level["neumann"]["cp_relative_l2"] < study.TOLERANCE
    assert level["neumann"]["velocity_relative_l2"] < study.TOLERANCE
    assert level["neumann"]["normal_velocity_residual_over_speed"] <= 0.01
    assert level["dirichlet"]["velocity_relative_l2"] < study.TOLERANCE


def test_faceting_is_larger_than_the_panel_error() -> None:
    """The anchor's circle reference cannot resolve the polygon flow better
    than the polygon's own geometric defect from the circle."""
    level = study.run_level(32)
    assert (
        level["faceting"]["surface_velocity_relative_l2"]
        > level["neumann"]["surface_velocity_relative_l2"]
    )
    assert level["decomposition"]["faceting_share_of_case_surface_error"] > 1.0


def test_panel_error_against_polygon_falls_with_refinement() -> None:
    coarse = study.run_level(16)
    fine = study.run_level(32)
    ratio = (
        coarse["neumann"]["surface_velocity_relative_l2"]
        / fine["neumann"]["surface_velocity_relative_l2"]
    )
    # second order in h means a factor ~4 for a factor 2 in panels
    assert 2.0 < ratio < 8.0


def test_wrong_reference_control_rejects_the_rotated_polygon() -> None:
    control = study.rotated_reference_control(32)
    assert control["relative_l2"] >= study.TOLERANCE
    assert control["discrimination_ratio"] > 5.0


def test_study_artifact_is_self_consistent() -> None:
    artifact = study.run_study((16, 32, 64))
    assert artifact["study_id"] == study.STUDY_ID
    names = [row["name"] for row in artifact["criteria"]]
    assert len(names) == len(set(names))
    assert artifact["overall"] in {"PASS", "FAIL"}
    for row in artifact["criteria"]:
        assert isinstance(row["ok"], bool)
    # the anchor case must be untouched by this study
    assert artifact["anchor_case"] == anchor.CASE_ID