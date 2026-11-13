"""Tests of the KK residual attribution machinery.

The point of the module is that a large residual can come from the *asymptote
estimator* rather than from the panel solution, so the tests pin down:

* the damped-oscillator pair really is an exact transform pair (against
  numerical quadrature of the memory kernel),
* a constant shift of the added mass moves the offset and leaves the shape
  residual **exactly** invariant (this is the mathematical statement the whole
  decomposition rests on),
* on a physics-faithful analytic pair (Ursell cylinder, ``A_inf = 0`` known) the
  estimator's shape floor is small compared with the 10 % gate tolerance,
* a genuinely non-causal pair (sign-flipped added mass) is *not* absorbed into
  the offset,
* the sum-rule asymptote is consistent with the tail fit on an exact pair.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem import causality_attribution as ca
from tensorbem.analytic import ursell_cylinder_added_mass

OMEGA = np.geomspace(0.3, 12.5, 22)


def _pair(rates=(0.4, 1.2), freqs=(0.8, 3.0), coeffs=(1.5, 2.5), a_inf=10.0):
    return ca.oscillator_kernel_pair(OMEGA, rates, freqs, coeffs,
                                     added_mass_inf=a_inf)


def test_oscillator_pair_matches_memory_kernel_quadrature():
    c, a, b = 2.0, 0.7, 1.3
    added, damping = ca.oscillator_kernel_pair(
        np.array([0.5, 1.0, 3.0, 8.0]), [a], [b], [c], added_mass_inf=1.0)
    s = np.linspace(0.0, 200.0, 400001)
    memory = c * np.exp(-a * s) * np.cos(b * s)
    omega = np.array([0.5, 1.0, 3.0, 8.0])
    quad_damping = np.trapezoid(
        memory[None, :] * np.cos(omega[:, None] * s[None, :]), s, axis=1)
    quad_added = 1.0 - np.trapezoid(
        memory[None, :] * np.sin(omega[:, None] * s[None, :]), s, axis=1) / omega
    assert np.allclose(damping, quad_damping, rtol=1e-5)
    assert np.allclose(added, quad_added, rtol=1e-5)


def test_chebyshev_centre_halves_the_range():
    data = np.array([-2.0, 0.5, 3.0, 1.0])
    centre = ca.chebyshev_centre(data)
    assert centre == pytest.approx(0.5 * (data.max() + data.min()))
    assert np.max(np.abs(data - centre)) == pytest.approx(
        0.5 * (data.max() - data.min()))


def test_constant_shift_moves_offset_but_not_shape():
    added, damping = _pair()
    base = ca.kkr_decomposition(OMEGA, added, damping, added_mass_inf=10.0)
    shifted = ca.kkr_decomposition(OMEGA, added + 7.5, damping, added_mass_inf=17.5)
    assert shifted.offset_bias == pytest.approx(base.offset_bias, abs=1e-9)
    assert shifted.residual_shape == pytest.approx(base.residual_shape, rel=1e-9)
    assert shifted.residual_total == pytest.approx(base.residual_total, rel=1e-9)
    assert shifted.added_mass_inf_implied == pytest.approx(
        base.added_mass_inf_implied + 7.5, rel=1e-9)


def test_exact_pair_has_small_shape_residual_with_true_asymptote():
    added, damping = _pair()
    decomposition = ca.kkr_decomposition(OMEGA, added, damping, added_mass_inf=10.0)
    # The oscillator family has B(0) > 0, so the below-band closure (which the
    # causality module applies as a constant hold) is the dominant residual
    # here; what the true asymptote must still buy is a *small* offset.
    assert decomposition.residual_shape < 0.20
    assert decomposition.residual_offset < 0.20
    assert decomposition.added_mass_inf_implied == pytest.approx(
        10.0, abs=0.20 * decomposition.scale_used)


def test_sign_flipped_pair_is_not_absorbed_by_the_offset():
    added, damping = _pair()
    flipped = ca.kkr_decomposition(OMEGA, 2.0 * 10.0 - added, damping,
                                   added_mass_inf=10.0)
    honest = ca.kkr_decomposition(OMEGA, added, damping, added_mass_inf=10.0)
    assert flipped.residual_shape > 10.0 * max(honest.residual_shape, 1e-6)
    assert flipped.residual_shape > 0.5


def test_ursell_cylinder_estimator_floor_is_below_the_gate():
    added = np.zeros_like(OMEGA)
    damping = np.zeros_like(OMEGA)
    for i, w in enumerate(OMEGA):
        added[i], damping[i] = ursell_cylinder_added_mass(
            0.09375, 0.35, float(w), 1025.0)
    assert damping.min() >= 0.0 and damping[0] < 1e-3 * damping.max()
    # infinite-frequency limit of a deeply submerged cylinder: rho*pi*a^2
    a_inf = 1025.0 * np.pi * 0.09375**2
    decomposition = ca.kkr_decomposition(OMEGA, added, damping,
                                         added_mass_inf=a_inf)
    assert decomposition.residual_shape < 0.03
    # the offset is only 2 % of A, but that is tens of percent of the (tiny)
    # dynamic range max|A - A_inf| -- the amplification this module exists for
    assert decomposition.residual_offset < 0.5
    assert abs(decomposition.offset_bias) < 0.04 * np.max(np.abs(added))
    # closure choice is immaterial for a body whose damping vanishes at w -> 0
    for model in ("zero", "linear"):
        other = ca.kkr_decomposition(OMEGA, added, damping, added_mass_inf=a_inf,
                                     below_band_model=model)
        assert other.residual_shape == pytest.approx(decomposition.residual_shape,
                                                    rel=1e-3)


def test_sum_rule_asymptote_agrees_with_truth_on_an_exact_pair():
    # exponential-sum special case: the 1/w^2 slope is exactly -(2/pi) int B dw
    added, damping = _pair(rates=(0.5, 1.0), freqs=(0.0, 0.0),
                          coeffs=(2.0, 4.0), a_inf=10.0)
    out = ca.sum_rule_asymptote(OMEGA, added, damping, tail_points=6)
    assert out["added_mass_inf_sum_rule"] == pytest.approx(
        10.0, abs=0.15 * max(abs(added - 10.0).max(), 1.0e-12))
    assert out["tail_coefficient"] < 0.0


def test_null_calibration_reports_a_floor_on_the_same_grid():
    _, damping = _pair()
    null = ca.null_calibration(OMEGA, damping, added_mass_inf=10.0)
    assert null["surrogate"]["damping_relative_residual"] < 0.35
    assert null["oracle_asymptote"]["residual_shape"] < 0.35
    assert null["grid"]["n"] == OMEGA.size


def test_ursell_floor_shrinks_when_the_band_is_sampled_denser():
    counts = (10, 22, 88)
    floors = []
    for count in counts:
        grid = np.geomspace(0.3, 12.5, count)
        added = np.zeros_like(grid)
        damping = np.zeros_like(grid)
        for i, w in enumerate(grid):
            added[i], damping[i] = ursell_cylinder_added_mass(
                0.09375, 0.35, float(w), 1025.0)
        floors.append(ca.kkr_decomposition(
            grid, added, damping,
            added_mass_inf=1025.0 * np.pi * 0.09375**2).residual_shape)
    assert floors[-1] < floors[0]
    assert floors[-1] < 0.02


def test_validation_rejects_bad_sweeps():
    with pytest.raises(ValueError):
        ca.kkr_decomposition([0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        ca.kkr_decomposition(np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) * 0 + 1.0,
                             np.ones(8), np.ones(8))
    with pytest.raises(ValueError):
        ca.chebyshev_centre([1.0, np.nan])