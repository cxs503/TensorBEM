"""Verification of the multi-connected (annulus) Dirichlet solve.

The reference in every case is a closed-form harmonic field, and the checks that
do not use the reference (per-loop circulation, total-flux identity, residual)
are asserted separately so a shared mistake in the reference cannot hide in the
error norm.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.geometry import PanelMesh, regular_polygon
from tensorbem.multicomponent import (
    AnalyticField,
    MultiComponentMesh,
    annulus_mesh,
    build_multicomponent_mesh,
    evaluate_fluid_field,
    log_field,
    observed_order,
    power_field,
    run_annulus_case,
    run_wrong_orientation_control,
    solve_multicomponent_dirichlet,
)
from tensorbem.benchmarks.multicomponent_annulus import field_map, study as annulus_study

INNER = 0.5
OUTER = 1.0
LADDER = (16, 32, 64)


@pytest.fixture(scope="module")
def study() -> dict:
    return annulus_study(ladder=LADDER, degrees=(1, 2))


# ---------------------------------------------------------------------------
# Geometry: two loops, fluid-outward normals, no cross-loop panels
# ---------------------------------------------------------------------------
def test_annulus_mesh_keeps_loop_closure_local() -> None:
    mesh = annulus_mesh(INNER, OUTER, panels_per_loop=32)
    assert mesh.panel_count == 64
    assert mesh.panel_counts_per_loop == (32, 32)
    assert mesh.loop_labels == ("outer", "hole1")
    # A single-polygon PanelMesh would have invented an edge from the last hole
    # vertex back to the first outer vertex. Check that no such edge exists.
    assert np.allclose(mesh.lengths[mesh.loop_slices[0]], 2 * OUTER * math.sin(math.pi / 32))
    assert np.allclose(mesh.lengths[mesh.loop_slices[1]], 2 * INNER * math.sin(math.pi / 32))


def test_normals_point_out_of_the_fluid_on_both_loops() -> None:
    mesh = annulus_mesh(INNER, OUTER, panels_per_loop=24)
    outer = mesh.loop_slices[0]
    hole = mesh.loop_slices[1]
    outer_radial = mesh.centers[outer] / np.linalg.norm(mesh.centers[outer], axis=1)[:, None]
    hole_radial = mesh.centers[hole] / np.linalg.norm(mesh.centers[hole], axis=1)[:, None]
    assert np.allclose(np.sum(mesh.normals[outer] * outer_radial, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.sum(mesh.normals[hole] * hole_radial, axis=1), -1.0, atol=1e-12)


def test_fluid_area_is_the_inscribed_polygon_area() -> None:
    """Exact check: two inscribed N-gons, so no area is lost or double counted."""
    for panels in (16, 64, 256):
        mesh = annulus_mesh(INNER, OUTER, panels_per_loop=panels)
        exact = 0.5 * panels * (OUTER**2 - INNER**2) * math.sin(2 * math.pi / panels)
        assert mesh.fluid_area == pytest.approx(exact, rel=1.0e-14)
        assert mesh.fluid_area < math.pi * (OUTER**2 - INNER**2)


def test_direct_construction_with_a_ccw_hole_is_rejected() -> None:
    outer = regular_polygon(24, OUTER)
    hole_ccw = regular_polygon(24, INNER)
    with pytest.raises(ValueError, match="clockwise"):
        MultiComponentMesh(loop_vertices=(outer.vertices, hole_ccw.vertices))


def test_builder_reverses_the_hole_and_rejects_stray_holes() -> None:
    mesh = build_multicomponent_mesh(regular_polygon(20, OUTER), (regular_polygon(20, INNER),))
    radial = mesh.centers[mesh.loop_slices[1]]
    radial = radial / np.linalg.norm(radial, axis=1)[:, None]
    assert np.allclose(np.sum(mesh.normals[mesh.loop_slices[1]] * radial, axis=1), -1.0)
    with pytest.raises(ValueError, match="leaves the outer loop"):
        build_multicomponent_mesh(
            regular_polygon(20, OUTER), (regular_polygon(20, 2.0),)
        )
    with pytest.raises(ValueError, match="must not overlap or nest"):
        build_multicomponent_mesh(
            regular_polygon(24, OUTER),
            (regular_polygon(24, 0.6), regular_polygon(24, 0.3)),
        )


# ---------------------------------------------------------------------------
# Field-containment guard
# ---------------------------------------------------------------------------
def test_fluid_containment_and_the_field_guard() -> None:
    mesh = annulus_mesh(INNER, OUTER, panels_per_loop=32)
    points = np.array([[0.75, 0.0], [0.0, 0.0], [0.2, 0.0], [1.5, 0.0]])
    assert mesh.fluid_contains(points).tolist() == [True, False, False, False]

    values = log_field(amplitude=1.0, offset=0.5).values(mesh.centers)
    result = solve_multicomponent_dirichlet(mesh, values)
    with pytest.raises(ValueError, match="outside the fluid region"):
        evaluate_fluid_field(mesh, result, points)
    with pytest.raises(TypeError):
        evaluate_fluid_field(regular_polygon(16), result, points[:1])


# ---------------------------------------------------------------------------
# Analytic field machinery
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field_factory", [lambda: log_field(1.0, 0.5), lambda: power_field(2)])
def test_analytic_gradient_matches_central_differences(field_factory) -> None:
    field: AnalyticField = field_factory()
    point = np.array([[0.7, 0.3]])
    h = 1.0e-6
    gradient = field.gradient(point)[0]
    for axis in (0, 1):
        step = np.zeros(2)
        step[axis] = h
        reference = (
            field.values(point + step)[0] - field.values(point - step)[0]
        ) / (2 * h)
        assert gradient[axis] == pytest.approx(reference, rel=1e-6)


def test_observed_order_formula() -> None:
    assert observed_order(4.0e-3, 1.0e-3) == pytest.approx(2.0)
    assert observed_order(2.0e-3, 1.0e-3) == pytest.approx(1.0)
    assert math.isnan(observed_order(1.0e-3, 0.0))


# ---------------------------------------------------------------------------
# Convergence against the closed-form references
# ---------------------------------------------------------------------------
def test_log_case_converges_second_order(study: dict) -> None:
    summary = study["summary"]["log"]
    assert summary["monotone_flux"]
    assert 1.7 < summary["orders"]["flux_rel_l2"] < 2.3
    assert 1.7 < summary["orders"]["field_rmse"] < 2.3
    assert summary["finest_flux_rel_l2"] < 2.0e-3
    # The registered verification target is 1 % flux error; the coarse rung is
    # already inside it, so the case is not sitting on the acceptance boundary.
    assert summary["worst_flux_rel_l2"] < 1.0e-2


@pytest.mark.parametrize("case", ["power1", "power2"])
def test_angular_harmonics_converge_second_order(study: dict, case: str) -> None:
    summary = study["summary"][case]
    assert summary["monotone_flux"]
    assert 1.7 < summary["orders"]["flux_rel_l2"] < 2.3
    assert summary["finest_flux_rel_l2"] < 3.0e-3


def test_condition_number_grows_linearly_and_residual_stays_at_roundoff(study: dict) -> None:
    log_rows = [r for r in study["rows"] if r["case_key"] == "log"]
    conditions = [float(r["condition_number"]) for r in log_rows]
    assert all(b > a for a, b in zip(conditions, conditions[1:]))
    # Doubling the panels doubles kappa (first-kind operator, O(N) growth for
    # this mesh): check every rung pair, not just the ends.
    for coarse, fine in zip(conditions, conditions[1:]):
        assert 1.8 < fine / coarse < 2.2
    assert max(float(r["relative_residual_l2"]) for r in log_rows) < 1.0e-12


# ---------------------------------------------------------------------------
# Checks that do not use the analytic field
# ---------------------------------------------------------------------------
def test_per_loop_circulation_of_the_log_field() -> None:
    """``A log r`` carries +-2 pi A of flux per loop: multi-loop bookkeeping."""
    row = run_annulus_case(lambda: log_field(amplitude=1.0, offset=0.5), panels_per_loop=64)
    outer, hole = row["per_loop_flux"]
    circulation = 2.0 * math.pi
    assert outer == pytest.approx(circulation, rel=2.0e-3)
    assert hole == pytest.approx(-circulation, rel=2.0e-3)
    assert outer + hole == pytest.approx(0.0, abs=1.0e-2)
    # Balances against the analytic per-panel integral on the same polygon.
    reference_outer, reference_hole = row["per_loop_flux_reference"]
    assert abs(outer - reference_outer) / circulation < 2.0e-3
    assert abs(hole - reference_hole) / circulation < 2.0e-3


def test_net_flux_vanishes_for_a_source_free_annulus() -> None:
    row = run_annulus_case(lambda: log_field(amplitude=1.0, offset=0.5), panels_per_loop=64)
    assert abs(float(row["net_flux_relative"])) < 1.0e-3


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------
def test_reversed_hole_normal_produces_a_wrong_answer(study: dict) -> None:
    control = study["orientation_control"]
    assert control["flux_rel_l2"] > 0.1
    assert control["error_ratio"] > 100.0
    # ... while its own linear system is solved to round-off: the residual is
    # not a detector for this mistake.
    assert control["relative_residual_l2"] < 1.0e-12


def test_control_helper_is_consistent_with_the_study() -> None:
    direct = run_wrong_orientation_control(panels_per_loop=32)
    assert direct["flux_rel_l2"] > 0.1


def test_panel_solution_beats_a_wrong_reference_sign() -> None:
    """A sign error in the reference flux would show up as an O(1) error."""
    field = log_field(amplitude=1.0, offset=0.5)
    mesh = annulus_mesh(INNER, OUTER, panels_per_loop=32)
    result = solve_multicomponent_dirichlet(mesh, field.values(mesh.centers))
    q_ref = field.normal_derivative(mesh)
    lengths = mesh.lengths
    signed = float(
        math.sqrt(float(np.sum(lengths * (result.normal_derivative - q_ref) ** 2))
                  / float(np.sum(lengths * q_ref**2)))
    )
    flipped = float(
        math.sqrt(float(np.sum(lengths * (result.normal_derivative + q_ref) ** 2))
                  / float(np.sum(lengths * q_ref**2)))
    )
    assert signed < 1.0e-2
    assert flipped > 1.0


def test_pointwise_field_error_map() -> None:
    field = field_map(panels_per_loop=32, grid=17)
    assert field["max_abs_error"] < 3.0e-3
    assert field["points"] > 100


# ---------------------------------------------------------------------------
# Documentation ledger
# ---------------------------------------------------------------------------
def test_verification_ledger_cell_is_closed() -> None:
    document = (Path(__file__).resolve().parents[1] / "docs" / "verification.md").read_text()
    row = next(line for line in document.splitlines() if "圆环径向导热" in line)
    assert "tensorbem.multicomponent" in row
    assert "后续：需多边界组件" not in row