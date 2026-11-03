"""Verification of the Kutta-closed lifting boundary-element solve.

The reference is the closed-form Joukowski solution in
:mod:`tensorbem.lifting`, which is independent of the boundary-element code.
The numbers checked here are the ones published in
``docs/lifting_kutta_closure.md``:

* circulation magnitude against ``4 pi a U sin(alpha)`` (2.4% at 128 panels),
* circulation lift against the pressure integral of the *same* discrete field
  (0.1%, i.e. the composite field is self-consistent),
* lift coefficient and surface speed against the exact solution (2.4% / 1.0%),
* first-order convergence in the panel size,
* the sign convention: the counter-clockwise circulation is *minus*
  :meth:`JoukowskiAirfoil.exact_circulation`, because the analytic helper quotes
  the clockwise value that pairs with ``C_L = 2 Gamma / (U c)``.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import pytest

from tensorbem.lifting import (
    JoukowskiAirfoil,
    LiftingFlowResult,
    solve_lifting_flow,
    vortex_velocity_field,
)

THICKNESS = 0.1
RADIUS = 1.0


@lru_cache(maxsize=None)
def _solution(alpha_deg: float, panels: int, fraction: float = 0.25) -> LiftingFlowResult:
    return solve_lifting_flow(
        JoukowskiAirfoil(THICKNESS, RADIUS, alpha_deg), panels, vortex_fraction=fraction
    )


def _exact_speed(result: LiftingFlowResult, alpha_deg: float) -> np.ndarray:
    foil = JoukowskiAirfoil(THICKNESS, RADIUS, alpha_deg)
    return np.abs(foil.exact_surface_speed(result.mesh.centers))


def test_vortex_field_has_unit_circulation_on_a_circle() -> None:
    from tensorbem.geometry import regular_polygon

    mesh = regular_polygon(2048, 2.0)
    velocity = vortex_velocity_field(mesh.centers, (0.0, 0.0), 1.0)
    circulation = float(np.sum(np.sum(velocity * np.column_stack((-mesh.normals[:, 1], mesh.normals[:, 0])), axis=1) * mesh.lengths))
    assert circulation == pytest.approx(1.0, rel=1e-4)


def test_circulation_is_counter_clockwise_and_two_four_percent_low() -> None:
    result = _solution(4.0, 128)
    exact = JoukowskiAirfoil(THICKNESS, RADIUS, 4.0).exact_circulation()

    # sign convention: the solver's circulation is minus the analytic helper's
    assert result.circulation_ccw < 0.0
    assert result.circulation_ccw == pytest.approx(-exact, rel=0.03)
    # and it is a genuine sign change, not a coincidence of magnitudes
    assert abs(result.circulation_ccw + exact) / exact == pytest.approx(0.0237, abs=5e-3)


def test_circulation_lift_agrees_with_the_pressure_integral() -> None:
    result = _solution(4.0, 128)
    gap = abs(
        result.lift_coefficient_from_pressure - result.lift_coefficient_from_circulation
    ) / abs(result.lift_coefficient_from_circulation)
    # The documented failure mode of the earlier closure was a composite field
    # whose pressure lift disagreed with its own circulation by ~200%.
    assert gap < 0.01


def test_lift_coefficient_is_within_three_percent_of_the_exact_value() -> None:
    foil = JoukowskiAirfoil(THICKNESS, RADIUS, 4.0)
    result = _solution(4.0, 128)
    exact = foil.exact_lift_coefficient()
    assert exact == pytest.approx(0.47814, rel=1e-4)
    assert abs(result.lift_coefficient_from_pressure - exact) / exact < 0.03


def test_surface_speed_matches_the_exact_field() -> None:
    result = _solution(4.0, 128)
    exact = _exact_speed(result, 4.0)
    rms = float(np.sqrt(np.mean((result.surface_speed - exact) ** 2))) / float(np.max(exact))
    assert rms < 0.015


def test_zero_incidence_carries_no_circulation() -> None:
    result = _solution(0.0, 128)
    assert result.circulation_ccw == pytest.approx(0.0, abs=1e-6)
    assert result.lift_coefficient_from_pressure == pytest.approx(0.0, abs=1e-6)


def test_negative_incidence_flips_the_circulation() -> None:
    positive = _solution(4.0, 64)
    negative = _solution(-4.0, 64)
    assert negative.circulation_ccw == pytest.approx(-positive.circulation_ccw, rel=1e-9)
    assert negative.lift_coefficient_from_pressure == pytest.approx(
        -positive.lift_coefficient_from_pressure, rel=1e-6
    )


def test_kutta_condition_is_imposed_and_the_cusp_is_stagnating() -> None:
    result = _solution(4.0, 128)
    assert result.kutta_residual == pytest.approx(0.0, abs=1e-12)
    # The exact Kutta solution stagnates *at* the cusp; one panel away the exact
    # speed has already recovered to ~0.91 U, and the discrete field has to
    # follow that decay rather than being fast at the trailing edge.
    exact = _exact_speed(result, 4.0)
    for index in (0, -1):
        measured = float(result.surface_speed[index])
        assert measured == pytest.approx(float(exact[index]), rel=0.02)
        assert measured < 0.65 * float(np.max(result.surface_speed))


def test_circulation_does_not_depend_on_the_singularity_location() -> None:
    near_quarter = _solution(4.0, 128, 0.15)
    mid_chord = _solution(4.0, 128, 0.5)
    assert mid_chord.circulation_ccw == pytest.approx(near_quarter.circulation_ccw, rel=0.01)


def test_circulation_converges_at_first_order() -> None:
    coarse = _solution(4.0, 64)
    fine = _solution(4.0, 128)
    exact = JoukowskiAirfoil(THICKNESS, RADIUS, 4.0).exact_circulation()
    coarse_error = abs(coarse.circulation_ccw + exact) / exact
    fine_error = abs(fine.circulation_ccw + exact) / exact
    assert fine_error < 0.6 * coarse_error


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, float("nan")])
def test_rejects_degenerate_singularity_locations(fraction: float) -> None:
    foil = JoukowskiAirfoil(THICKNESS, RADIUS, 4.0)
    with pytest.raises(ValueError):
        solve_lifting_flow(foil, 32, vortex_fraction=fraction)


def test_rejects_non_positive_speed() -> None:
    foil = JoukowskiAirfoil(THICKNESS, RADIUS, 4.0)
    with pytest.raises(ValueError):
        solve_lifting_flow(foil, 32, speed=0.0)