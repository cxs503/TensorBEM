"""Gates for the canonical-spectrum Wigley wave resistance (m16274).

Public routes use the force-valued integral. The explicitly named legacy
route records the dimensional error for archived benchmark reproduction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import thin_ship_michell, wigley

G = 9.80665
RHO = 1025.0
L, B, T = 10.0, 1.0, 0.625
FR = (0.20, 0.25, 0.30, 0.35, 0.40)
FORCE_EXPONENTS = (-1.0, -1.0, 2.0)


def _speed(fr: float, length: float = L) -> float:
    return fr * math.sqrt(G * length)


def _theta(fr: float, l: float = L, b: float = B, t: float = T) -> float:
    hull = thin_ship_michell.WigleyHull(L=l, B=b, T=t)
    return float(
        thin_ship_michell.michell_wave_resistance(
            thin_ship_michell.wigley_amplitude_function(hull),
            _speed(fr, l),
            lam_max=40.0,
            n_lam=240,
            rho=RHO,
            g=G,
        )["Rw"]
    )


def _exponents(func, fr: float = 0.30, l: float = L, b: float = B, t: float = T):
    base = func(l, b, t, _speed(fr, l))
    out = []
    for axis in ("mass", "length", "time"):
        s = 2.0
        sm, sl, st = {"mass": (s, 1.0, 1.0), "length": (1.0, s, 1.0), "time": (1.0, 1.0, s)}[axis]
        val = func(
            l / sl, b / sl, t / sl, _speed(fr, l) * st / sl, RHO * sl**3 / sm, G * st**2 / sl
        )
        out.append(math.log(val / base) / math.log(s))
    return tuple(out)


def test_canonical_resistance_is_a_force() -> None:
    exps = _exponents(wigley.michell_wave_resistance)
    assert np.allclose(exps, FORCE_EXPONENTS, atol=1e-6, rtol=0.0), exps


def test_legacy_is_still_not_a_force() -> None:
    """RATCHET (record of the defect): the explicit legacy route stays reproducible."""
    exps = _exponents(wigley.michell_wave_resistance_legacy)
    assert np.allclose(np.asarray(exps) - np.asarray(FORCE_EXPONENTS), (0.0, 2.0, -4.0),
                       atol=1e-6, rtol=0.0), exps


@pytest.mark.parametrize("fr", FR)
def test_canonical_reproduces_the_independent_theta_form(fr: float) -> None:
    canon = wigley.michell_wave_resistance(L, B, T, _speed(fr))
    theta = _theta(fr)
    assert abs(canon - theta) / theta < 3e-3, (canon, theta)


@pytest.mark.parametrize("fr", FR)
def test_legacy_is_badly_low(fr: float) -> None:
    legacy = wigley.michell_wave_resistance_legacy(L, B, T, _speed(fr))
    canon = wigley.michell_wave_resistance(L, B, T, _speed(fr))
    assert 5.0 < canon / legacy < 40.0, canon / legacy


def test_canonical_matches_the_audit_recipe_value() -> None:
    """``docs/michell_unit_arbitration.md`` §5: 3.891778 N at the m3125 case."""
    l, b, t = 3.0, 0.3, 0.1875
    value = wigley.michell_wave_resistance_canonical(l, b, t, _speed(0.30, l))
    assert value == pytest.approx(3.8917783116, rel=5e-3)


def test_canonical_is_quadrature_converged() -> None:
    coarse = wigley.michell_wave_resistance_canonical(L, B, T, _speed(0.30), n_quad=128)
    fine = wigley.michell_wave_resistance_canonical(L, B, T, _speed(0.30), n_quad=256)
    assert abs(coarse - fine) / fine < 1e-3


def test_kochin_canonical_pair_satisfies_the_free_surface_condition() -> None:
    """``k = alpha**2/k0`` is the only admissible pair for ``exp(k z + i alpha x)``."""
    U = _speed(0.30)
    k0 = G / U**2
    lam = np.array([1.0, 1.5, 2.0, 4.0, 10.0])
    residual = np.abs(lam * k0 - (lam * k0) ** 2 / k0) / (lam * k0)
    assert np.allclose(residual, lam - 1.0, atol=1e-12, rtol=0.0)
    # the canonical pair carries a much smaller |H| (stronger depth decay)
    h_can = [abs(wigley.michell_kochin_function(L, B, T, k0 * l**2, k0 * l)) for l in lam]
    h_leg = [abs(wigley.michell_kochin_function(L, B, T, k0 * l, k0 * l)) for l in lam]
    assert all(c < g for c, g in zip(h_can[1:], h_leg[1:]))


def test_zero_and_negative_speed_return_zero() -> None:
    assert wigley.michell_wave_resistance_canonical(L, B, T, 0.0) == 0.0
    assert wigley.michell_wave_resistance_canonical(L, B, T, -1.0) == 0.0


def test_canonical_grows_with_speed() -> None:
    values = [wigley.michell_wave_resistance_canonical(L, B, T, _speed(fr)) for fr in FR]
    assert all(v > 0.0 for v in values)
    assert values[-1] > values[0]
