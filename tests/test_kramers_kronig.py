"""Tests for the Kramers-Kronig causality relations and their verification study.

The first block validates the principal-value quadrature against closed forms
(the only numerical ingredient that can silently spoil a causality check).  The
second block applies the relations to Ursell's exact multipole solution, which
is the reference the panel solver is judged against.  The third block runs the
same check on the panel solution itself at a resolution cheap enough for the
test suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import ursell_cylinder_coefficients
from tensorbem.benchmarks.kramers_kronig_study import (
    DENSITY,
    DISPLACED,
    GRAVITY,
    RADIUS,
    SUBMERGENCE,
    extrapolated_infinite_added_mass,
    sweep_analytic,
    sweep_panels,
)
from tensorbem.kramers_kronig import (
    added_mass_gap,
    damping_from_added_mass,
    implied_infinite_added_mass,
    kk_residual_metrics,
    principal_value_integral,
    sum_rule_residual,
)

#: Causality-implied infinite-frequency added mass of the study geometry,
#: normalised by the displaced mass; see docs/kramers_kronig_verification.md.
REFERENCE_INFINITE_ADDED_MASS = 0.88329


def _pv_weight(s0: float, s_end: float, omega: float) -> float:
    return (1.0 / (2.0 * omega)) * math.log(
        abs((s_end - omega) * (s0 + omega)) / abs((s_end + omega) * (s0 - omega))
    )


@pytest.mark.parametrize("grid_kind", ["uniform", "logarithmic"])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_principal_value_integral_is_exact_for_polynomials(grid_kind, degree):
    """The quadrature must be exact for ``F(s) = s**degree`` (closed-form check)."""
    s0, s_end, omega = 0.2, 6.0, 1.0
    count = 400
    grid = (
        np.linspace(s0, s_end, count)
        if grid_kind == "uniform"
        else np.geomspace(s0, s_end, count)
    )
    weight = _pv_weight(s0, s_end, omega)
    closed_forms = {
        0: weight,
        1: 0.5 * math.log(abs((s_end**2 - omega**2) / (s0**2 - omega**2))),
        2: (s_end - s0) + omega**2 * weight,
        3: 0.5 * (s_end**2 - s0**2)
        + omega**2 * 0.5 * math.log(abs((s_end**2 - omega**2) / (s0**2 - omega**2))),
    }
    expected = closed_forms[degree]
    computed = principal_value_integral(grid, grid**degree, omega)
    assert computed == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_principal_value_integral_matches_symmetry_identity():
    """``PV int F/(s**2-w**2) = (1/2w) [PV int F/(s-w) - PV int F/(s+w)]``."""
    grid = np.linspace(0.05, 8.0, 800)
    omega = 1.3
    values = np.exp(-grid) * np.cos(grid)
    left = principal_value_integral(grid, values, omega)
    # PV int F/(s-w) ds for a smooth F, evaluated by the same routine with the
    # weight 1/(s**2-w**2) replaced by 1/((s-w)(s+w)) written as two limits.
    split = 0.5 * (1.0 / omega)
    right = split * (
        principal_value_integral(grid, values * (grid + omega), omega)
        - principal_value_integral(grid, values * (grid - omega), omega)
    )
    assert left == pytest.approx(right, rel=1e-8, abs=1e-12)


def test_principal_value_integral_rejects_omega_outside_grid():
    grid = np.linspace(0.2, 6.0, 120)
    values = np.cos(grid)
    for omega in (0.1, 0.2, 6.0, 7.5):
        with pytest.raises(ValueError):
            principal_value_integral(grid, values, omega)


def test_grid_validation_errors():
    grid = np.linspace(0.2, 6.0, 120)
    values = np.cos(grid)
    with pytest.raises(ValueError):
        principal_value_integral(np.linspace(1.0, 0.2, 120), values, 1.0)
    with pytest.raises(ValueError):
        principal_value_integral(grid, values[:-1], 1.0)
    with pytest.raises(ValueError):
        principal_value_integral(grid[:4], values[:4], 1.0)


@pytest.fixture(scope="module")
def analytic_sweep():
    ka = np.geomspace(0.002, 4.0, 60)
    return sweep_analytic(ka)


def test_ursell_multipole_satisfies_causality(analytic_sweep):
    """The closed-form reference must satisfy relation (A) to a few 1e-5."""
    result = implied_infinite_added_mass(
        analytic_sweep["omega"], analytic_sweep["damping"], analytic_sweep["added_mass"]
    )
    limit = result["implied_infinite_added_mass"] / DISPLACED
    assert limit == pytest.approx(REFERENCE_INFINITE_ADDED_MASS, rel=2e-3)
    assert result["implied_added_mass_relative_spread"] < 5e-4
    metrics = kk_residual_metrics(
        result["added_mass_reference"], result["reconstructed_added_mass"]
    )
    assert metrics["max_relative_error"] < 5e-4


def test_relation_a_reproduces_added_mass_differences(analytic_sweep):
    """Relation (A) fixes ``a(w1)-a(w2)`` with no knowledge of ``a(inf)``."""
    omega = analytic_sweep["omega"]
    gap = added_mass_gap(omega, analytic_sweep["damping"], omega[5:-5])
    mass = analytic_sweep["added_mass"][5:-5]
    shift = float(np.mean(mass - gap))
    assert np.max(np.abs(mass - gap - shift)) / DISPLACED < 5e-4


def test_infinite_added_mass_from_extrapolation_agrees_with_causality(analytic_sweep):
    limit = extrapolated_infinite_added_mass()
    causal = implied_infinite_added_mass(
        analytic_sweep["omega"], analytic_sweep["damping"], analytic_sweep["added_mass"]
    )["implied_infinite_added_mass"]
    assert limit["added_mass_infinite"] == pytest.approx(causal, rel=2e-3)


def test_relation_b_is_only_conditionally_convergent_in_two_dimensions(analytic_sweep):
    """Documented limitation: in 2-D ``a(inf)-a = O(1/ka)``, so (B) does not close.

    The residual of relation (B) shrinks like ``1/log(ka_max)`` at best, which is
    why the verification uses relation (A); the assertion pins the *observed*
    behaviour so a future change that silently "fixes" it by accident is caught.
    """
    omega = analytic_sweep["omega"]
    margin = 5
    a_inf = REFERENCE_INFINITE_ADDED_MASS * DISPLACED
    reconstructed = damping_from_added_mass(
        omega, analytic_sweep["added_mass"], a_inf, omega[margin:-margin]
    )
    reference = analytic_sweep["damping"][margin:-margin]
    residual = np.max(np.abs(reconstructed - reference)) / np.max(np.abs(reference))
    assert 0.02 < residual < 0.8


def test_panel_solver_satisfies_causality_at_coarse_resolution():
    """The panel solution passes the same check within its discretisation error."""
    ka = np.geomspace(0.003, 4.0, 12)
    sweep = sweep_panels(ka, 48)
    result = implied_infinite_added_mass(
        sweep["omega"], sweep["damping"], sweep["added_mass"]
    )
    limit = result["implied_infinite_added_mass"] / DISPLACED
    # First-order convergence for this kernel: a coarse mesh lands within a few
    # per cent of the exact limit rather than at 1e-5.
    assert limit == pytest.approx(REFERENCE_INFINITE_ADDED_MASS, rel=0.15)
    assert result["implied_added_mass_relative_spread"] < 0.25


def test_panel_damping_matches_multipole_reference():
    ka = np.array([0.05, 0.6, 3.0])
    panels = sweep_panels(ka, 96)
    exact = sweep_analytic(ka)
    for computed, reference in zip(panels["damping"], exact["damping"]):
        assert computed == pytest.approx(reference, rel=0.05)


def test_sum_rule_and_metrics_are_finite(analytic_sweep):
    value = sum_rule_residual(
        analytic_sweep["omega"], analytic_sweep["added_mass"], 0.88329 * DISPLACED
    )
    assert math.isfinite(value)
    metrics = kk_residual_metrics(np.ones(5), np.ones(5))
    assert metrics["max_relative_error"] == 0.0