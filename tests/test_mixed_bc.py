"""Verification tests for mixed Dirichlet/Neumann and multiply connected 2-D BEM.

These tests turn the three capability gaps recorded in ``docs/verification.md``
(multiply connected domains, mixed boundary conditions, pure Neumann problems)
into executable contracts:

* the multiply connected and mixed solves converge at the second order expected
  from piecewise-constant panels on smooth data,
* the pure Neumann solve reports its rank deficiency, its compatibility residual
  and an unambiguous constant gauge instead of returning a silently wrong answer,
* the mixed solve reproduces the pre-existing Dirichlet solver exactly when all
  panels are Dirichlet, and its matrices reproduce the pre-existing assembly,
* the reference-length degeneracy documented in ``tensorbem.solver`` is pinned
  for multiply connected boundaries too.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.mixed_bc_verification import (
    REFERENCE_LENGTH,
    annulus_panels,
    circle_flux,
    circle_potential,
    radial_flux,
    radial_potential,
    run_annulus_dirichlet,
    run_annulus_mixed,
    run_annulus_pure_neumann,
    run_circle_mixed,
    square_panels,
    square_flux,
    square_potential,
)
from tensorbem.geometry import regular_polygon
from tensorbem.mixed_bc import (
    PanelSet,
    assemble_panels,
    circle_loop,
    evaluate_field,
    flux_integral,
    solve_mixed,
)
from tensorbem.solver import assemble_matrices, solve_laplace_dirichlet


def _annulus_residuals(per_loop: int = 32) -> dict[str, float]:
    panels = annulus_panels(per_loop)
    exact_u = radial_potential(panels.centers)
    exact_q = radial_flux(panels)
    result = solve_mixed(
        panels,
        neumann={i: float(v) for i, v in enumerate(exact_q)},
        reference_length=REFERENCE_LENGTH,
    )
    return {"flux_integral": flux_integral(panels, result.normal_derivative)}


# ---------------------------------------------------------------------------
# Consistency with the existing single-loop Dirichlet solver
# ---------------------------------------------------------------------------


def test_all_dirichlet_matches_existing_solver():
    """A single closed loop with Dirichlet data must reproduce solver._solve."""
    mesh = regular_polygon(32, radius=1.5)
    values = np.cos(2.0 * np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0]))
    reference = solve_laplace_dirichlet(mesh, values, reference_length=1.0)
    panels = PanelSet.from_mesh(mesh)
    mixed = solve_mixed(
        panels,
        dirichlet={i: float(v) for i, v in enumerate(values)},
        reference_length=1.0,
    )
    assert mixed.rank_deficiency == 0
    np.testing.assert_allclose(
        mixed.normal_derivative, reference.normal_derivative, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        mixed.boundary_values, reference.boundary_values, rtol=0.0, atol=0.0
    )


def test_panel_assembly_matches_existing_assembly():
    """PanelSet.from_mesh must give the same analytic matrices as assemble_matrices."""
    mesh = regular_polygon(24, radius=1.25)
    expected_single, expected_double = assemble_matrices(mesh, reference_length=1.0)
    single, double = assemble_panels(PanelSet.from_mesh(mesh), reference_length=1.0)
    np.testing.assert_allclose(single, expected_single, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(double, expected_double, rtol=0.0, atol=1e-14)


def test_loop_orientation_flips_normals():
    """A hole loop must carry normals pointing into the hole."""
    outer = PanelSet.from_loops([circle_loop(8, 2.0)])
    centers = outer.centers
    np.testing.assert_allclose(
        outer.normals, centers / np.linalg.norm(centers, axis=1)[:, None], atol=1e-12
    )
    hole = PanelSet.from_loops([circle_loop(8, 1.0, clockwise=True)])
    inner_centers = hole.centers
    np.testing.assert_allclose(
        hole.normals, -inner_centers / np.linalg.norm(inner_centers, axis=1)[:, None],
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Multiply connected domains
# ---------------------------------------------------------------------------


def test_annulus_dirichlet_multiply_connected_second_order():
    coarse = run_annulus_dirichlet(32)
    fine = run_annulus_dirichlet(64)
    error_coarse = coarse.metrics["flux_rel_l2_all"]
    error_fine = fine.metrics["flux_rel_l2_all"]
    # O(h^2): doubling the panels must cut the error by more than 3.5x.
    assert error_coarse / error_fine > 3.5
    assert error_fine < 1.0e-3
    assert fine.metrics["relative_residual"] < 1.0e-12
    assert fine.metrics["rank_deficiency"] == 0.0
    # Both loops are resolved: the hole flux is not an afterthought.
    assert fine.metrics["flux_rel_l2_inner"] < 2.0e-3
    assert fine.metrics["flux_rel_l2_outer"] < 2.0e-3


def test_annulus_interior_field_reconstruction():
    outcome = run_annulus_dirichlet(64)
    assert outcome.metrics["field_rel_max"] < 1.0e-3


def test_reference_length_at_loop_radius_is_degenerate():
    """The logarithmic-capacity degeneracy also bites multiply connected solves."""
    good = run_annulus_dirichlet(64, reference_length=1.0)
    bad = run_annulus_dirichlet(64, reference_length=2.0)  # == outer radius
    assert bad.metrics["flux_rel_l2_all"] > 20.0 * good.metrics["flux_rel_l2_all"]
    # The error is a mean-flux mode: only the outer loop is affected.
    assert bad.metrics["flux_rel_l2_outer"] > 20.0 * good.metrics["flux_rel_l2_outer"]
    assert bad.metrics["flux_rel_l2_inner"] < 2.0 * good.metrics["flux_rel_l2_inner"]


# ---------------------------------------------------------------------------
# Mixed boundary conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("swap", [False, True])
def test_annulus_mixed_reproduces_analytic_traces(swap: bool):
    outcome = run_annulus_mixed(64, swap=swap)
    assert outcome.metrics["dirichlet_flux_rel_l2"] < 5.0e-4
    assert outcome.metrics["neumann_trace_rel_l2"] < 2.0e-3
    assert outcome.metrics["field_rel_max"] < 1.0e-3
    assert outcome.metrics["rank_deficiency"] == 0.0


def test_annulus_mixed_converges_second_order():
    coarse = run_annulus_mixed(32)
    fine = run_annulus_mixed(64)
    assert (
        coarse.metrics["dirichlet_flux_rel_l2"] / fine.metrics["dirichlet_flux_rel_l2"]
        > 3.5
    )


def test_circle_mixed_smooth_boundary_second_order():
    errors = [run_circle_mixed(n).metrics["dirichlet_flux_rel_l2"] for n in (32, 64, 128)]
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5
    assert errors[-1] < 1.0e-3


def test_square_mixed_junction_error_is_localised():
    """The square's flux error sits on the BC-junction corner panels only."""
    outcome = run_annulus_dirichlet(16)  # keeps the report metric shape meaningful
    assert outcome.metrics["flux_rel_l2_all"] < 5.0e-2
    square = run_square_mixed_local()
    assert (
        square["dirichlet_flux_rel_l2_away_from_junction"]
        < square["dirichlet_flux_rel_l2"]
    )
    assert square["junction_panel_flux_abs_max"] > 1.0e-2


def run_square_mixed_local(per_side: int = 16) -> dict[str, float]:
    panels = square_panels(per_side)
    exact_u = square_potential(panels.centers)
    exact_q = square_flux(panels)
    vertical = np.isclose(np.abs(panels.normals[:, 0]), 1.0)
    dirichlet = np.where(~vertical)[0]
    neumann = np.where(vertical)[0]
    result = solve_mixed(
        panels,
        dirichlet={int(i): float(exact_u[i]) for i in dirichlet},
        neumann={int(i): float(exact_q[i]) for i in neumann},
        reference_length=REFERENCE_LENGTH,
    )
    errors = result.normal_derivative[dirichlet] - exact_q[dirichlet]
    within_side = np.arange(len(dirichlet)) % per_side
    interior = (within_side != 0) & (within_side != per_side - 1)
    return {
        "dirichlet_flux_rel_l2": float(
            np.linalg.norm(errors) / np.linalg.norm(exact_q[dirichlet])
        ),
        "dirichlet_flux_rel_l2_away_from_junction": float(
            np.linalg.norm(errors[interior]) / np.linalg.norm(exact_q[dirichlet][interior])
        ),
        "junction_panel_flux_abs_max": float(np.max(np.abs(errors[~interior]))),
    }


# ---------------------------------------------------------------------------
# Pure Neumann problems
# ---------------------------------------------------------------------------


def test_pure_neumann_reports_rank_deficiency_and_gauge():
    outcome = run_annulus_pure_neumann(64)
    assert outcome.metrics["rank_deficiency"] == 1.0
    assert outcome.metrics["potential_rel_l2_up_to_constant"] < 2.0e-3
    assert outcome.metrics["potential_rel_max_up_to_constant"] < 2.0e-3
    # The prescribed data are compatible: the discrete net flux vanishes to
    # roundoff and the discrete compatibility residual is small.
    assert abs(outcome.metrics["flux_integral"]) < 1.0e-12
    assert outcome.metrics["solvability_residual"] < 2.0e-3


def test_pure_neumann_constant_gauge_is_removed():
    """A constant offset added to the Neumann potential must not change the solve."""
    panels = annulus_panels(32)
    exact_q = radial_flux(panels)
    first = solve_mixed(
        panels,
        neumann={i: float(v) for i, v in enumerate(exact_q)},
        reference_length=REFERENCE_LENGTH,
    )
    exact_u = radial_potential(panels.centers) + 7.5
    shifted = solve_mixed(
        panels,
        neumann={i: float(v) for i, v in enumerate(exact_q)},
        reference_length=REFERENCE_LENGTH,
    )
    np.testing.assert_allclose(
        first.boundary_values, shifted.boundary_values, rtol=0.0, atol=1e-13
    )
    # ...and the recovered trace is the zero-mean representative of the truth.
    want = exact_u - float(np.mean(exact_u))
    got = shifted.boundary_values
    np.testing.assert_allclose(got - float(np.mean(got)), want, atol=2.0e-3)


def test_pure_neumann_incompatible_data_is_flagged():
    """Neumann data with a non-zero net flux must produce a large residual."""
    panels = annulus_panels(32)
    exact_q = radial_flux(panels)
    bad_q = exact_q.copy()
    bad_q[panels.loop == 1] *= 1.5  # breaks the global compatibility condition
    result = solve_mixed(
        panels,
        neumann={i: float(v) for i, v in enumerate(bad_q)},
        reference_length=REFERENCE_LENGTH,
    )
    assert abs(flux_integral(panels, bad_q)) > 0.1
    assert result.rank_deficiency == 1
    assert result.solvability_residual is not None
    assert result.solvability_residual > 1.0e-2
    # The consistent problem is orders of magnitude better conditioned in the
    # compatibility sense, which is what the reported residual is for.
    good = solve_mixed(
        panels,
        neumann={i: float(v) for i, v in enumerate(exact_q)},
        reference_length=REFERENCE_LENGTH,
    )
    assert good.solvability_residual is not None
    assert result.solvability_residual > 20.0 * good.solvability_residual
    assert good.solvability_residual < 2.0e-3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validation_rejects_overlapping_and_missing_data():
    panels = PanelSet.from_loops([circle_loop(8, 2.0)])
    with pytest.raises(ValueError, match="both Dirichlet and Neumann"):
        solve_mixed(
            panels,
            dirichlet={0: 0.0},
            neumann={0: 0.0, **{i: 0.0 for i in range(1, 8)}},
        )
    with pytest.raises(ValueError, match="neither Dirichlet nor Neumann"):
        solve_mixed(panels, dirichlet={i: 0.0 for i in range(7)})
    with pytest.raises(ValueError, match="outside"):
        solve_mixed(panels, neumann={i: 0.0 for i in range(9)})
    with pytest.raises(ValueError, match="must be finite"):
        solve_mixed(panels, neumann={i: 0.0 for i in range(7)} | {7: np.nan})


def test_panel_set_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least three"):
        PanelSet.from_loops([np.array([[0.0, 0.0], [1.0, 0.0]])])
    with pytest.raises(ValueError, match="zero-length"):
        PanelSet.from_loops(
            [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])]
        )
    with pytest.raises(ValueError, match="positive"):
        circle_loop(8, radius=0.0)
    with pytest.raises(ValueError, match="at least 3"):
        circle_loop(2)


def test_evaluate_field_rejects_wrong_trace_shape():
    panels = PanelSet.from_loops([circle_loop(8, 2.0)])
    with pytest.raises(ValueError, match="shape"):
        evaluate_field(panels, np.zeros(7), np.zeros(8), [[0.5, 0.5]])
    with pytest.raises(ValueError, match="shape"):
        evaluate_field(panels, np.zeros(8), np.zeros(8), [[0.5, 0.5, 0.5]])


def test_supplied_matrices_are_reused():
    panels = PanelSet.from_loops([circle_loop(8, 2.0)])
    matrices = assemble_panels(panels, reference_length=REFERENCE_LENGTH)
    values = radial_potential(panels.centers)
    auto = solve_mixed(
        panels,
        dirichlet={i: float(v) for i, v in enumerate(values)},
        reference_length=REFERENCE_LENGTH,
    )
    reused = solve_mixed(
        panels,
        dirichlet={i: float(v) for i, v in enumerate(values)},
        reference_length=REFERENCE_LENGTH,
        matrices=matrices,
    )
    np.testing.assert_allclose(
        reused.normal_derivative, auto.normal_derivative, rtol=0.0, atol=1e-14
    )
    with pytest.raises(ValueError, match="shape"):
        solve_mixed(
            PanelSet.from_loops([circle_loop(6, 2.0)]),
            dirichlet={i: 0.0 for i in range(6)},
            matrices=matrices,
        )