"""Verification ladder for the 2-D ice-gate heave added-mass anchor.

Every assertion here is a *baseline* check of
:mod:`tensorbem.benchmarks.case_ice_gate_power_section`: the closed-form values
are exact, the mirror-line residuals identify which free-surface condition is
satisfied, and every negative control is required to be rejected by the gate.
"""

from __future__ import annotations

import math

import numpy as np

from tensorbem.benchmarks.case_ice_gate_power_section import (
    DENSITY,
    LIMIT_INFINITE_FREQUENCY,
    LIMIT_ZERO_FREQUENCY,
    convergence_ladder,
    doubled_rectangle_mesh,
    exact_circle_added_mass,
    half_domain_heave_added_mass,
    mirror_diagnostics,
    negative_controls,
)
from tensorbem.geometry import regular_polygon


def test_full_circle_heave_matches_exact() -> None:
    """Submerged cylinder: A = rho*pi*a**2 (exact)."""
    from tensorbem.solver import compute_added_mass_matrix_2d

    mesh = regular_polygon(64, radius=1.0)
    matrix = compute_added_mass_matrix_2d(mesh, density=DENSITY)
    exact = exact_circle_added_mass(1.0)
    assert abs(matrix.added_mass_matrix[1, 1] - exact) / exact < 1.0e-4


def test_half_immersed_circle_A_inf_is_exactly_half() -> None:
    """The full-circle heave potential is odd in y, so A_inf = 0.5*rho*pi*a**2."""
    mesh = regular_polygon(64, radius=1.0)
    value = half_domain_heave_added_mass(mesh, DENSITY, LIMIT_INFINITE_FREQUENCY)
    exact = 0.5 * exact_circle_added_mass(1.0)
    assert abs(value - exact) / exact < 1.0e-4


def test_mirror_diagnostics_separate_the_two_limits() -> None:
    """The A_inf branch meets phi = 0; the rigid-wall branch meets d(phi)/dz = 0."""
    mesh = doubled_rectangle_mesh(10.0, 6.0, 32)
    infinite = mirror_diagnostics(mesh, LIMIT_INFINITE_FREQUENCY)
    rigid = mirror_diagnostics(mesh, LIMIT_ZERO_FREQUENCY)

    # pressure release: the potential vanishes on the mirror line, net flux ~ 0
    assert infinite["mirror_line_potential_ratio"] < 1.0e-10
    assert abs(infinite["net_flux"]) < 1.0e-10

    # rigid wall: the normal derivative vanishes, but the data are incompatible
    # with a decaying exterior potential (net flux 2*V*B, i.e. 40 m here)
    assert rigid["mirror_line_normal_derivative_ratio"] < 1.0e-10
    assert abs(rigid["net_flux"]) > 1.0


def test_ladder_refines_and_orders_are_physical() -> None:
    """A_inf converges monotonically from above with a positive observed order."""
    limits = convergence_ladder(10.0, 6.0, DENSITY, (8, 16, 32))
    block = limits["limiting_conditions"][LIMIT_INFINITE_FREQUENCY]
    values = block["added_mass_per_metre"]
    assert values[0] > values[1] > values[2]
    assert block["observed_order_finest_three"] > 0.5
    # the Richardson limit must sit below the finest computed value
    assert block["richardson_limit"] < values[2]


def test_negative_controls_are_rejected() -> None:
    """Every mutant is rejected: 3 % for structural slips, 2 % for a locked mesh."""
    limits = convergence_ladder(
        10.0, 6.0, DENSITY, (8, 16, 32), (LIMIT_INFINITE_FREQUENCY, LIMIT_ZERO_FREQUENCY)
    )
    for row in negative_controls(limits, DENSITY, tolerance=0.03):
        if row["name"] == "coarse_mesh_lock":
            continue
        assert row["rejected"], row
    coarse = [
        row
        for row in negative_controls(limits, DENSITY, tolerance=0.02)
        if row["name"] == "coarse_mesh_lock"
    ]
    assert coarse and coarse[0]["rejected"]


def test_unhalved_doubling_is_a_factor_two_error() -> None:
    """The mirror trick halves the doubled-body energy; skipping it doubles A."""
    mesh = doubled_rectangle_mesh(10.0, 6.0, 32)
    half = half_domain_heave_added_mass(mesh, DENSITY, LIMIT_INFINITE_FREQUENCY)
    from tensorbem.solver import compute_added_mass_matrix_2d

    doubled = compute_added_mass_matrix_2d(mesh, density=DENSITY).added_mass_matrix[1, 1]
    assert abs(doubled / half - 2.0) < 1.0e-9
    assert math.isfinite(half) and half > 0.0