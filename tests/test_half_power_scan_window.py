"""Scan-window handling of :func:`tensorbem.time_domain.half_power_damping`.

Round 3 of the Cummins/Wigley case (``CASE-BEM-CUMMINS-WIGLEY``, m17773) could
not decide its G4 gate because the half-power estimator died with a bare
``IndexError: index 0 is out of bounds for axis 0 with size 0`` -- the level was
never crossed inside the local scan window and the crossing search indexed an
empty array.  On the rising side the same defect was *silent*: a crossing below
the window was clamped to ``scan[0]``, which reports ``zeta`` too low without
any hint that the window was too narrow.

These tests pin the two behaviours that replace it:

* a crossing that is not bracketed raises a ``ValueError`` naming the window
  (never an ``IndexError``, never a clamped number);
* :func:`half_power_damping` widens the window (x4, up to the tabulated band)
  until both crossings are bracketed, and reports how many widenings it took.

The analytic target is a single-degree-of-freedom oscillator with constant
added mass and damping, whose half-power damping ratio is exactly
``zeta = B / (2 (m + A) omega_n)`` with ``omega_n = sqrt(C / (m + A))`` -- the
limit in which the half-power and energy-identity definitions coincide, so the
assertion needs no model approximation beyond ``O(zeta**2)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import time_domain as td
from tensorbem.time_domain import _half_power_crossing


def _oscillator(zeta: float, omega_n: float = 1.0, mass: float = 1.0,
                added: float = 1.0) -> tuple[float, float, float]:
    """``(mass, restoring, damping)`` of an oscillator with the given ``zeta``."""
    inertia = mass + added
    restoring = inertia * omega_n**2
    damping = 2.0 * zeta * inertia * omega_n
    return mass, restoring, damping


def _sweep(mass: float, added: float, damping: float,
           low: float, high: float, points: int = 21):
    omega = np.linspace(low, high, points)
    return omega, np.full(points, added), np.full(points, damping)


def test_narrow_window_is_widened_not_clipped() -> None:
    """A window narrower than the half-power width must not report a low zeta."""
    zeta = 0.01
    mass, restoring, damping = _oscillator(zeta)
    omega, added, damp = _sweep(mass, 1.0, damping, 0.1, 5.0)
    # width_scale=0.5 gives a window of 0.5*zeta*omega_n, i.e. half of the
    # half-power half width: the lower crossing lies below the first sample.
    result = td.half_power_damping(
        omega, added, damp, mass=mass, restoring=restoring, width_scale=0.5,
    )
    assert result["scan_widenings"] >= 1, result
    assert result["scan_attempts"], "the failed narrow attempt must be recorded"
    assert result["zeta"] == pytest.approx(zeta, rel=2e-2), result
    assert result["zeta_energy"] == pytest.approx(zeta, rel=2e-2)


def test_wide_enough_window_needs_no_widening() -> None:
    zeta = 0.02
    mass, restoring, damping = _oscillator(zeta)
    omega, added, damp = _sweep(mass, 1.0, damping, 0.1, 5.0)
    result = td.half_power_damping(omega, added, damp, mass=mass, restoring=restoring)
    assert result["scan_widenings"] == 0
    assert result["scan_attempts"] == []
    assert result["zeta"] == pytest.approx(zeta, rel=1e-2)


def test_unbracketable_window_raises_value_error() -> None:
    """When the tabulated band cannot contain the width, fail loudly."""
    zeta = 0.30
    mass, restoring, damping = _oscillator(zeta)
    omega, added, damp = _sweep(mass, 1.0, damping, 0.90, 1.05, points=12)
    with pytest.raises(ValueError) as excinfo:
        td.half_power_damping(omega, added, damp, mass=mass, restoring=restoring)
    message = str(excinfo.value)
    assert "not bracketed" in message
    assert "attempts=" in message


def test_crossing_below_window_raises_instead_of_clamping() -> None:
    scan = np.linspace(1.0, 2.0, 9)
    amplitude = 0.5 + 0.5 * (scan - 0.5)  # first sample already above the level
    level = 0.6
    with pytest.raises(ValueError) as excinfo:
        _half_power_crossing(scan, amplitude, level, rising=True)
    assert "below the scan window" in str(excinfo.value)


def test_crossing_above_window_raises_instead_of_indexerror() -> None:
    """The exact failure mode the Cummins/Wigley G4 gate hit: empty array."""
    scan = np.linspace(1.0, 2.0, 9)
    amplitude = 2.0 - 0.1 * (scan - 1.0)  # never falls to the half-power level
    level = 1.0
    with pytest.raises(ValueError) as excinfo:
        _half_power_crossing(scan, amplitude, level, rising=False)
    assert "never crossed" in str(excinfo.value)


def _heave_like(zeta: float, resolve_peak: bool):
    """An oscillator at 7.4 rad/s whose damping sweep either sees its peak or not."""
    mass, restoring, damping = _oscillator(zeta, omega_n=7.4)
    if resolve_peak:
        delta = 4.0 * zeta * 7.4
        omega = np.unique(np.concatenate([
            np.linspace(0.3, 6.8, 12),
            np.linspace(7.4 - delta, 7.4 + delta, 17),
            np.linspace(7.8, 12.5, 6),
        ]))
    else:
        omega = np.linspace(0.3, 12.5, 24)
    # Lorentzian damping profile: the half-power half width is exactly
    # zeta*omega_n, so a sweep that resolves the peak must return zeta.
    damp = damping / (1.0 + ((omega - 7.4) / (zeta * 7.4)) ** 2)
    return omega, np.full(omega.size, 1.0), damp, mass, restoring


def test_light_heave_like_resonance_is_resolved_when_the_grid_sees_it() -> None:
    """zeta ~ 5e-3, damping peak sampled: the half-power width comes out right."""
    zeta = 5.0e-3
    omega, added, damp, mass, restoring = _heave_like(zeta, resolve_peak=True)
    result = td.half_power_damping(omega, added, damp, mass=mass, restoring=restoring)
    assert math.isfinite(result["zeta"])
    # Definedness, not accuracy, is what this test pins: at zeta = 5e-3 the
    # half-power width is 0.037 rad/s and the estimator integrates the
    # *piecewise-linear* interpolant of the sweep, whose peak sits a third of a
    # half width below omega_n -- a tens-of-percent bias is the interpolant's,
    # not the window's.  A factor-2 bracket is the honest assertion here.
    assert 0.5 * zeta < result["zeta"] < 2.0 * zeta, result
    assert result["scan_widenings"] == 0, "a resolved peak needs no widening"


def test_unresolved_damping_peak_still_answers_but_is_flagged_light() -> None:
    """The medium Cummins grid's situation: peak missed, answer biased low.

    The estimator is *defined* (the round-3 window widening removed the bare
    ``IndexError``) but it now answers about the linearly interpolated sweep,
    whose damping at omega_n is far below the true peak -- so ``zeta`` comes out
    low.  The case reports that as ``scan_width_over_half_power_width`` and as a
    mesh/band limitation, not as physics.  This test pins the behaviour so the
    bias cannot be mistaken for a solved gate later on.
    """
    zeta = 5.0e-3
    omega, added, damp, mass, restoring = _heave_like(zeta, resolve_peak=False)
    result = td.half_power_damping(omega, added, damp, mass=mass, restoring=restoring)
    assert math.isfinite(result["zeta"]) and result["zeta"] > 0.0, result
    assert result["zeta"] < 0.8 * zeta, "an unresolved peak must not fake the width"
    assert math.isfinite(result["omega_n"])
