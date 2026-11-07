"""Unit and regression tests for the panel-operator accuracy-order benchmark.

``docs/panel_operator_order.md`` claims three things that are cheap enough to
pin in CI:

* the repository's polar-sector self panel integral reproduces the closed form
  derived in :mod:`tensorbem.benchmarks.panel_operator_order` to far better than
  the 1e-6 campaign gate (criterion C3),
* a deliberately coarse quadrature of the same panel integral is *outside* that
  gate, so the gate is a real test and not a tautology (criterion C5),
* the operator errors fall when the sphere is refined (criterion C2), and the
  two readings the module resolves separately are algebraically the equations
  the public solver API already implements (contract pin, so the benchmark can
  never drift away from :mod:`tensorbem.solver3d`).

The four-level sweep that produces the published orders lives in the benchmark
module; running it here would cost minutes, so these tests pin the *behaviour*
at the cheap resolutions instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.panel_operator_order import (
    analytic_normal_derivative,
    analytic_source_potential,
    evaluate_criteria,
    exterior_dirichlet_flux,
    exterior_neumann_potential,
    loglog_slope,
    operator_level_row,
    self_panel_distance_integral,
    self_panel_single_layer_analytic,
    self_panel_single_layer_coarse,
    self_panel_single_layer_polar,
    self_term_error_table,
)
from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import (
    assemble_surface_matrices,
    solve_laplace_surface,
    solve_laplace_surface_neumann,
)

GATE = 1.0e-6
SOURCE = (0.0, 0.0, 0.5)


@pytest.fixture(scope="module")
def sphere_levels():
    """Two cheap refinement levels (80 and 320 panels) for the order tests."""
    return [operator_level_row(1, source=SOURCE), operator_level_row(2, source=SOURCE)]


def test_self_term_matches_closed_form(sphere_levels):
    """C3: the repository's polar rule agrees with the closed form to << 1e-6."""
    table = self_term_error_table(subdivisions=2, panel_stride=16)
    assert table["panels_sampled"] >= 16
    assert table["polar_max_rel_error"] <= GATE, (
        f"polar self panel integral deviates by {table['polar_max_rel_error']:.3e}, "
        f"gate {GATE:.0e}"
    )


def test_closed_form_is_positive_and_scales_with_area():
    """The closed form behaves like an area-scaled 1/h integral on similar panels."""
    small = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.1, 0.0]])
    large = small * 4.0
    small_value = self_panel_distance_integral(small)
    large_value = self_panel_distance_integral(large)
    assert small_value > 0.0
    # int_T dS/r scales linearly with length, so a 4x similar triangle is 4x.
    assert large_value == pytest.approx(4.0 * small_value, rel=1e-12)


def test_coarse_self_term_fails_the_gate():
    """C5: the negative control must be red, otherwise the C3 gate measures nothing."""
    table = self_term_error_table(subdivisions=1, panel_stride=4)
    assert table["coarse_max_rel_error"] > GATE
    assert table["coarse_max_rel_error"] > 1.0e-2, (
        "the coarse control is barely outside the gate; it is no longer a control"
    )
    assert table["polar_max_rel_error"] < table["coarse_max_rel_error"]


def test_dirichlet_flux_matches_public_solver_api():
    """The benchmark's Dirichlet-to-flux map is the public solver's equation."""
    mesh = icosphere(1)
    values = analytic_source_potential(mesh.centroids, SOURCE)
    single, double = assemble_surface_matrices(mesh)
    ours = exterior_dirichlet_flux(single, double, values)
    public = solve_laplace_surface(mesh, values, domain="exterior").normal_derivative
    assert np.allclose(ours, public, rtol=1e-11, atol=0.0)


def test_neumann_potential_matches_public_solver_api():
    """The benchmark's Neumann-to-potential map is the public solver's equation."""
    mesh = icosphere(1)
    body_flux = analytic_normal_derivative(mesh.centroids, mesh.normals, SOURCE)
    fluid_flux = -body_flux
    single, double = assemble_surface_matrices(mesh)
    ours = exterior_neumann_potential(single, double, fluid_flux)
    public = solve_laplace_surface_neumann(mesh, fluid_flux).boundary_values
    assert np.allclose(ours, public, rtol=1e-11, atol=0.0)


def test_loglog_slope_recovers_known_order():
    """The regressor returns the order in h for a synthetic power law."""
    panels = np.array([80.0, 320.0, 1280.0, 5120.0])
    errors = 0.4 * panels ** -0.5  # order 1 in h
    assert loglog_slope(panels, errors) == pytest.approx(1.0, rel=1e-9)
    with pytest.raises(ValueError):
        loglog_slope([80.0], [0.1])
    with pytest.raises(ValueError):
        loglog_slope([80.0, 320.0], [0.1, -0.2])


def test_errors_decrease_on_refinement(sphere_levels):
    """C2: both operator readings improve monotonically from 80 to 320 panels."""
    low, high = sphere_levels
    assert high["panels"] == 4 * low["panels"]
    assert high["err_flux"] < low["err_flux"]
    assert high["err_potential"] < low["err_potential"]
    # the observed order in h must be comfortably positive, i.e. real convergence
    order = math.log(low["err_flux"] / high["err_flux"]) / math.log(2.0)
    assert order > 0.5, f"flux order in h collapsed to {order:.2f}"


def test_conditioning_is_bounded_and_solid_angle_identity_holds(sphere_levels):
    """The second-kind operator stays O(1) conditioned; the closed-mesh identity holds."""
    for row in sphere_levels:
        assert 1.0 < row["condition_double"] < 10.0
        assert row["solid_angle_row_sum_error"] < 1e-12
        assert row["mesh_area_ratio"] < 1.0  # inscribed polyhedron is smaller


def test_criteria_report_is_derived_not_asserted(sphere_levels):
    """Criteria verdicts follow the numbers: a fast slope cannot be reported as PASS."""
    low, high = sphere_levels
    operator = {
        "rows": [low, high],
        "slopes": {
            "err_flux_in_h": math.log(low["err_flux"] / high["err_flux"]) / math.log(2.0),
            "err_potential_in_h": math.log(
                low["err_potential"] / high["err_potential"]
            )
            / math.log(2.0),
            "err_flux_in_N": 0.0,
            "err_potential_in_N": 0.0,
        },
        "steps": [],
        "self_term": {
            "polar_max_rel_error": 1e-15,
            "polar_median_rel_error": 5e-16,
            "panels_sampled": 20,
            "polar_order": 64,
            "coarse_max_rel_error": 0.13,
        },
    }
    criteria = evaluate_criteria(operator, {"rows": []})
    assert criteria["verdicts"]["C3"] == "PASS"
    assert criteria["verdicts"]["C5"] == "PASS"
    assert criteria["verdicts"]["C4"] == "NOT_RUN"
    assert criteria["verdicts"]["C1b"] == "NOT_SUPPORTED"
    # the seeded band is against log N; a fast (steep) scheme is outside it
    assert criteria["verdicts"]["C1a"] == "FAIL"
    assert criteria["C1a"]["inside_target_band"] == {
        "err_flux": False,
        "err_potential": False,
    }


def test_self_panel_rules_reject_bad_input():
    with pytest.raises(ValueError):
        self_panel_single_layer_analytic(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        self_panel_single_layer_polar(np.zeros((3, 3)))  # degenerate
    with pytest.raises(ValueError):
        self_panel_single_layer_coarse(np.zeros((3, 3)))
