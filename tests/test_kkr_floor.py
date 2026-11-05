"""Tests for the Kramers-Kronig floor decomposition (m3051).

These lock in the three findings the audit turns on, so that a later change to
either the transform or the reference solution cannot silently re-label the
floor as a solver defect:

* the equation (2) floor is a *band* effect, not interpolation (node sweep);
* the equation (1) residual is a nearly constant offset whose sign flips when
  the lower band end moves, i.e. no band extension alone closes it;
* the reference pair itself is internally inconsistent at high frequency: the
  damping decays exponentially (rate ``2*k*f = 4`` per ``ka`` for this case)
  while the added-mass deficit decays only algebraically.
"""

from __future__ import annotations


import numpy as np
import pytest

from tensorbem.benchmarks import kkr_floor as module


def _band(rows, *, ka_min=None, ka_max=None):
    for row in rows:
        if ka_min is not None and row["ka_min"] != ka_min:
            continue
        if ka_max is not None and row["ka_max"] != ka_max:
            continue
        return row
    raise AssertionError(f"band row not found: ka_min={ka_min} ka_max={ka_max}")


def test_node_count_does_not_move_the_floor():
    """Interpolation is not the bottleneck: 4x more nodes moves nothing by 2x."""
    rows = {row["points"]: row for row in module.nodes_leg(0.05, 3.0, points=(32, 128))}
    assert 0.1 < rows[32]["damping_floor"] < 1.0
    ratio = rows[128]["damping_floor"] / rows[32]["damping_floor"]
    assert 0.5 < ratio < 2.0, ratio
    mass_ratio = rows[128]["added_mass_floor"] / rows[32]["added_mass_floor"]
    assert 0.5 < mass_ratio < 2.0, mass_ratio


def test_lower_band_controls_the_added_mass_floor():
    """Equation (2): dropping the lower band end 5e-2 -> 1e-2 gains > 10x."""
    rows = module.band_leg(lower=(0.05, 0.01), upper=())
    coarse = _band(rows, ka_min=0.05)
    fine = _band(rows, ka_min=0.01)
    assert coarse["added_mass_floor"] > 5.0e-3
    assert fine["added_mass_floor"] < 1.0e-3
    assert fine["added_mass_floor"] < coarse["added_mass_floor"] / 10.0


def test_equation_one_residual_is_an_offset_whose_sign_flips():
    """Equation (1): the residual is far larger than its own scatter, and its
    sign changes when the lower band end moves -> the constant extension below
    the first node, not the interpolation, is what is being measured."""
    rows = module.band_leg(lower=(0.05, 0.001), upper=())
    high = _band(rows, ka_min=0.05)
    low = _band(rows, ka_min=0.001)
    # the offset dominates its own frequency scatter in both cases
    assert abs(high["damping_offset_mean"]) > 5.0 * high["damping_offset_spread"]
    assert abs(low["damping_offset_mean"]) > 5.0 * low["damping_offset_spread"]
    # ... and it changes sign
    assert high["damping_offset_mean"] > 0.0 > low["damping_offset_mean"]


def test_reference_pair_has_an_algebraic_added_mass_tail():
    """A 1/omega tail cannot belong to a causal kernel: K_i = omega (A-A_inf)
    would stay finite at infinity.  The fitted exponent says how close to that
    pathological case the reference is."""
    tail = module.tail_leg()
    assert 0.85 < tail["added_mass_tail_exponent_8_30"] < 1.15
    assert 0.7 < tail["added_mass_tail_exponent_30_120"] < 1.0
    # while the damping decays at the physical free-surface rate 2*k*f = 4 per ka
    assert 3.3 < tail["damping_decay_rate_per_ka_3_8"] < 4.3
    assert tail["damping"] [-1] < tail["damping"][0] * 1e-12


def test_analytic_deficit_saturates_towards_low_frequency():
    """The constant extension below ka_min is wrong because A - A_inf tends to
    the *nonzero* limit A(0) - A_inf as ka -> 0 (an extended Ursell sweep,
    1e-4..1e-1 with terms 40 vs 120 bit-identical, saturates at ~808 kg/m for
    this case).  The earlier m3051 wording claimed the deficit *grows* towards
    ka -> 0; the reference itself shows it saturates after a hump that peaks
    near ka ~ 0.1, so that claim is corrected here.  What survives -- and what
    the KK floor argument needs -- is that the deficit at the band edge sits
    well above the saturated limit, so extending the last node constantly below
    ka_min misestimates the low-frequency contribution by a bounded but
    non-negligible ~25%."""
    ka, mass, _ = module.analytic_sweep(0.001, 0.1, 8)
    deficit = mass - module.ADDED_MASS_INF
    assert np.all(deficit > 0.0)
    # ka = 1e-3 is already on the saturated plateau A(0) - A_inf ~ 8.1e2
    assert deficit[0] < 1.0e3
    # ...while the value at the ka_min = 5e-2 band edge overestimates the
    # plateau by >20%: a constant extension below the band is measurably wrong
    edge = float(np.interp(0.05, ka, deficit))
    assert edge > 1.2 * deficit[0], (edge, deficit[0])
    assert deficit[-1] > deficit[0]


@pytest.mark.slow
def test_panel_solution_sits_above_the_reference_at_high_ka():
    """Independent cross-check with no causality argument: at ka = 8 the panel
    added mass exceeds the multipole reference by tens of kg/m while the damping
    (1e-8 level here) agrees.  Kept at 32 panels so it runs in seconds."""
    rows = module.cross_leg((32,))
    assert len(rows) == 2
    for row in rows:
        assert row["added_mass_panel"] > row["added_mass_reference"]
        assert row["added_mass_offset"] > 5.0
        # the two solutions agree on the (exponentially small) damping
        assert row["damping_reference"] > 0.0
        assert 0.3 < row["damping_panel"] / row["damping_reference"] < 3.0