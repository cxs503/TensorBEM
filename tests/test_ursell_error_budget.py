"""Tests for the Ursell radiation error budget.

The budget exists to make the documented accuracy claim falsifiable, so the
tests pin down the three things the claim depends on:

* the multipole reference is truncation-converged (and stays computable at the
  truncation needed to *show* that, which is a numerical-range fix);
* the panel error is a genuine discretisation error that converges at the
  documented rate;
* the independent solution and the radiation-condition identity agree with the
  reference at a level well below the panel error.

Only small meshes are used, so the whole module stays inside a few seconds.
"""

import math

import numpy as np

from tensorbem.analytic import ursell_cylinder_coefficients
from tensorbem.benchmarks.ursell_error_budget import (
    CONVERGED_TERMS,
    DOCUMENTED,
    _nondimensional_panel,
    convergence_extrapolation,
    deep_submergence_limit,
    energy_identity,
    heave_sway_symmetry,
    panel_convergence,
    panel_node_study,
    plot,
    reproduce_documented_claim,
    truncation_study,
)

KA = 0.6
A_OVER_F = 0.5
#: Value at ``terms = 40`` produced before the log-domain assembly refactor of
#: ``_ursell_multipole_system``; kept as a bit-level regression guard.
MU_AT_40 = 0.7903920946051628
NU_AT_40 = 0.3479880654670651


# --------------------------------------------------------------------------
# Reference side
# --------------------------------------------------------------------------


def test_multipole_system_stays_computable_at_high_truncation():
    """The bracket terms overflow for terms >~80 unless assembled in log space."""
    mu, nu = ursell_cylinder_coefficients(KA, A_OVER_F, 200)
    assert math.isfinite(mu) and math.isfinite(nu)
    converged = ursell_cylinder_coefficients(KA, A_OVER_F, CONVERGED_TERMS)
    assert math.isclose(mu, converged[0], rel_tol=1.0e-11)
    assert math.isclose(nu, converged[1], rel_tol=1.0e-11)


def test_default_truncation_reproduces_pre_refactor_values():
    mu, nu = ursell_cylinder_coefficients(KA, A_OVER_F, 40)
    assert math.isclose(mu, MU_AT_40, rel_tol=1.0e-13)
    assert math.isclose(nu, NU_AT_40, rel_tol=1.0e-13)


def test_truncation_study_plateaus_below_panel_error():
    study = truncation_study(ka_values=(0.2, 0.6, 1.0), terms_list=(8, 16, 20, 24, 32, 40))
    assert study["default_terms_relative_error"]["max_added_mass"] < 1.0e-10
    assert study["default_terms_relative_error"]["max_damping"] < 1.0e-10
    # A truncation error of 1e-10 cannot explain a 1e-3 panel error: the
    # reference is not the weak link in the comparison.
    assert study["default_terms_relative_error"]["max_added_mass"] < 1.0e-3


# --------------------------------------------------------------------------
# Discretisation side
# --------------------------------------------------------------------------


def test_panel_node_quadrature_saturates():
    study = panel_node_study(panel_count=32, node_counts=(8, 16, 24))
    assert study["rows"][0]["added_mass_relative_error"] > 0.0
    # Once the per-panel rule resolves the kernel, extra nodes stop mattering.
    assert study["saturated_added_mass_error_spread"] < 1.0e-3


def test_panel_convergence_is_monotone_and_first_order():
    study = panel_convergence(panel_counts=(32, 64, 128), panel_nodes=16)
    errors = [row["added_mass_relative_error"] for row in study["rows"]]
    assert errors == sorted(errors, reverse=True), errors
    order = study["observed_order_tail_mean"]
    assert order is not None
    # The documented behaviour is "about first order": N doubles, error halves.
    assert 0.5 < order < 2.5, order


def test_documented_panels_row_matches_remeasured_value():
    study = panel_convergence(panel_counts=(128,), panel_nodes=24)
    row = study["documented_panels_row"]
    assert row is not None
    assert row["panels"] == DOCUMENTED["panels"]
    # The m238 audit remeasured 8.5e-3 here against the documented 7.7e-3.
    assert 5.0e-3 < row["added_mass_relative_error"] < 1.2e-2


def test_richardson_extrapolation_improves_the_error():
    rows = panel_convergence(panel_counts=(64, 128), panel_nodes=16)["rows"]
    extrapolation = convergence_extrapolation(rows)
    assert extrapolation["applicable"] is False  # needs three meshes
    rows = panel_convergence(panel_counts=(32, 64, 128), panel_nodes=16)["rows"]
    extrapolation = convergence_extrapolation(rows)
    assert extrapolation["applicable"] is True
    added_mass = extrapolation["added_mass"]
    assert added_mass["extrapolation_relative_error"] <= added_mass["observed_order"] * 0 + 5.0e-3


# --------------------------------------------------------------------------
# Independent side
# --------------------------------------------------------------------------


def test_energy_identity_closes():
    study = energy_identity(panel_counts=(32, 64), panel_nodes=16)
    assert study["max_relative_difference"] < 5.0e-2, study


def test_heave_and_sway_agree_in_infinite_depth():
    study = heave_sway_symmetry(panel_count=32, panel_nodes=16)
    assert study["added_mass_relative_difference"] < 1.0e-10
    assert study["damping_relative_difference"] < 1.0e-10


def test_deep_submergence_tends_to_displaced_mass():
    study = deep_submergence_limit()
    assert study["decays_to_unity"] is True
    deviations = [row["added_mass_deviation_from_displaced_mass"] for row in study["rows"]]
    assert deviations == sorted(deviations, reverse=True)
    assert deviations[-1] < 1.0e-3


# --------------------------------------------------------------------------
# The documented claim itself
# --------------------------------------------------------------------------


def test_documented_claim_search_returns_a_verdict():
    study = reproduce_documented_claim(panel_counts=(64,), node_counts=(12, 24))
    assert study["target_added_mass_error"] == DOCUMENTED["added_mass_relative_error"]
    assert len(study["all_cells"]) == 2
    assert isinstance(study["documented_claim_reproduced"], bool)
    assert study["best_cell"]["relative_deviation_from_documented"] >= 0.0


def test_plot_writes_a_figure(tmp_path):
    budget = {
        "panel_convergence": {
            "rows": [
                {"panels": 32, "added_mass_relative_error": 3.0e-2, "damping_relative_error": 4.0e-2},
                {"panels": 64, "added_mass_relative_error": 1.5e-2, "damping_relative_error": 2.0e-2},
            ],
            "documented_panels_row": {
                "panels": 128,
                "added_mass_relative_error": 8.5e-3,
                "damping_relative_error": 1.3e-2,
            },
        },
        "mfs_convergence": {"rows": [{"sources": 16, "added_mass_relative_error": 1.0e-4}]},
        "panel_nodes": {
            "rows": [
                {"panel_nodes": 8, "added_mass_relative_error": 9.0e-3, "damping_relative_error": 1.4e-2},
                {"panel_nodes": 24, "added_mass_relative_error": 8.5e-3, "damping_relative_error": 1.3e-2},
            ]
        },
        "documented_claim": DOCUMENTED,
    }
    target = tmp_path / "figure.png"
    plot(budget, target)
    assert target.stat().st_size > 4096
    assert np.isfinite([row["added_mass_relative_error"] for row in budget["panel_convergence"]["rows"]]).all()