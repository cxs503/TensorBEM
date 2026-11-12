"""Independent-referee tests for the 2-D principal-value kernel (m16829).

The kernel ``tensorbem.free_surface._pv_cosine`` / ``_pv_sine`` used a
fixed-node Gauss-Laguerre tail on ``[2 nu, inf)``.  The tail integrand
oscillates in the integration variable with phase ``nu*|x|`` per unit ``k``, so
the accuracy decayed with horizontal separation: ``1e-6`` at ``nu|x| ~ 3`` and
``O(1)`` (i.e. 100 % relative) at ``nu|x| ~ 9`` once the depth sum became small
(``nu*A = 0.05``).  The audit ``docs/green_family_crosscheck`` registered the
defect at ``6e-2`` for ``nu*A = 0.48``; the sweep below shows it was far worse
at shallow scaled depth.

The tail is now the closed form ``Re/Im{exp(-b) E1(b)}``.  These tests judge the
fix against a quadrature referee that shares no code with it.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from tensorbem import free_surface
from tensorbem.free_surface import principal_value_integral


def referee(nu: float, depth: float, offset: float, *, nper: int = 8,
            n_q: int = 40, decay: float = 40.0) -> float:
    """Panelled Gauss-Legendre PV of the same integral (edges forced at the pole)."""
    x = abs(offset)
    f1 = np.exp(-nu * depth) * np.cos(nu * x)
    df1 = -depth * f1 - x * np.exp(-nu * depth) * np.sin(nu * x)
    k_hi = 2.0 * nu + decay / depth
    period = 2.0 * np.pi / x if x > 1e-9 else np.inf
    width = min(period / nper, 0.25)
    n_pan = max(4, int(np.ceil((k_hi - 2.0 * nu) / width)) + 2)
    edges = sorted(set(np.linspace(0.0, k_hi, n_pan + 1).tolist()) | {nu, 2.0 * nu})
    nodes, weights = leggauss(n_q)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        k = mid + half * nodes
        f = np.exp(-k * depth) * np.cos(k * x)
        num = f - (f1 if b <= 2.0 * nu else 0.0)
        den = k - nu
        safe = np.abs(den) > 1e-9
        integrand = np.where(safe, num / np.where(safe, den, 1.0), df1)
        total += half * float(np.sum(weights * integrand))
    return total


PHASES = (0.0, 1.0, 3.0, 5.3, 9.4, 13.3, 30.0, 60.0, 120.0)
SCALED_DEPTHS = (0.05, 0.48, 1.0, 5.0, 50.0)


@pytest.mark.parametrize("sigma", SCALED_DEPTHS)
def test_pv_cosine_matches_independent_quadrature(sigma):
    """Relative deviation vs the referee, over the full phase x depth grid."""
    worst = 0.0
    for phase in PHASES:
        ref = referee(1.0, sigma, phase)          # nu = 1: depth = sigma, offset = phase
        got = principal_value_integral(1.0, sigma, phase)
        worst = max(worst, abs(got - ref) / max(abs(ref), 1e-300))
    assert worst < 1.0e-10, (sigma, worst)


def test_pv_sine_matches_independent_quadrature():
    """The sine kernel (needed by the source gradient) shares the tail fix."""
    worst = 0.0
    for sigma in (0.05, 0.48, 1.0):
        for phase in (0.5, 3.0, 9.4, 13.3, 60.0):
            got = float(free_surface._pv_sine(1.0, np.array([sigma]),
                                             np.array([phase]))[0])
            # independent referee for the sine PV: same split, sin integrand
            f1 = np.exp(-sigma) * np.sin(phase)
            df1 = -sigma * f1 + phase * np.exp(-sigma) * np.cos(phase)
            k_hi = 2.0 + 40.0 / sigma
            width = min(2.0 * np.pi / (8 * max(phase, 1e-9)), 0.25)
            n_pan = max(4, int(np.ceil((k_hi - 2.0) / width)) + 2)
            edges = sorted(set(np.linspace(0.0, k_hi, n_pan + 1).tolist()) | {1.0, 2.0})
            nodes, weights = leggauss(40)
            ref = 0.0
            for a, b in zip(edges[:-1], edges[1:]):
                half, mid = 0.5 * (b - a), 0.5 * (b + a)
                k = mid + half * nodes
                f = np.exp(-k * sigma) * np.sin(k * phase)
                num = f - (f1 if b <= 2.0 else 0.0)
                den = k - 1.0
                safe = np.abs(den) > 1e-9
                integrand = np.where(safe, num / np.where(safe, den, 1.0), df1)
                ref += half * float(np.sum(weights * integrand))
            worst = max(worst, abs(got - ref) / max(abs(ref), 1e-300))
    assert worst < 1.0e-9, worst


def test_registered_defect_grid_is_repaired():
    """The five-point grid the audit used: 5e-2 -> below 1e-10."""
    sigma = 0.48
    measured = []
    for phase in (1.0, 3.0, 5.3, 9.4, 13.3):
        ref = referee(1.0, sigma, phase)
        got = principal_value_integral(1.0, sigma, phase)
        measured.append(abs(got - ref) / abs(ref))
    assert max(measured) < 1.0e-10, measured


def test_old_laguerre_tail_defect_is_reproducible():
    """Negative control: the discarded rule really was O(1) wrong at nu*A = 0.05.

    Reproducing the defect from the superseded algorithm keeps the fix from
    being a claim about nothing: if this ever stopped failing, the test would no
    longer demonstrate that the closed form was needed.
    """
    from numpy.polynomial.laguerre import laggauss

    sigma, phase = 0.05, 9.4
    u, w = laggauss(160)
    t = u / sigma
    integrand = np.cos((2.0 + t) * phase) / (1.0 + t)
    laguerre_tail = np.exp(-2.0 * sigma) / sigma * np.sum(w * integrand)
    # regular head on [0, 2 nu], k = nu (1 + t)
    nodes, weights = leggauss(80)
    ref_head = np.exp(-sigma) * np.cos(phase)
    k = 1.0 + nodes
    head = np.sum(weights * (np.exp(-k * sigma) * np.cos(k * phase) - ref_head) / nodes)
    old_total = head + laguerre_tail
    ref = referee(1.0, sigma, phase)
    assert abs(old_total - ref) / abs(ref) > 0.5          # > 50 % relative
    assert abs(principal_value_integral(1.0, sigma, phase) - ref) / abs(ref) < 1e-12


def test_head_rule_ladder_resolves_large_phase():
    """The head node count is chosen from the phase, not fixed at 80."""
    rule_small = free_surface._legendre_rule(10.0)
    rule_large = free_surface._legendre_rule(400.0)
    assert rule_small[0].size == free_surface._LEGENDRE_NODES
    assert rule_large[0].size > rule_small[0].size
    for phase in (200.0, 400.0):
        ref = referee(1.0, 0.48, phase)
        got = principal_value_integral(1.0, 0.48, phase)
        assert abs(got - ref) / abs(ref) < 1.0e-9, (phase, got, ref)