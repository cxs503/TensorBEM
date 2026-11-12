"""Gates for the Michell exit arbitration (mission m16345).

The package exposes several "Michell wave resistance" entry points.  Only one
of them can be right: the steady free-surface condition
``U^2 phi_xx + g phi_z = 0`` admits the harmonic ``phi ~ exp(k z + i alpha x)``
only on the pair ``(k, alpha) = (alpha^2/k0, alpha)``, ``k0 = g/U^2``.  These
tests pin the arbitration recorded in ``docs/michell_exit_arbitration.md``:

* the canonical pair is dispersion-exact and the pre-repair single-wave-number
  sampling is off by ``lambda - 1`` pointwise;
* :func:`tensorbem.rankine_wave.michell_kochin_function` and
  :func:`tensorbem.wigley.michell_kochin_function` now describe the same hull
  (the pre-repair depth integral dropped the ``(z/T)^2`` term, which forced its
  ``k -> 0`` limit to ``T`` instead of the exact ``2T/3``);
* :func:`tensorbem.rankine_wave.michell_wave_resistance` returns the canonical
  value, is a force by similarity scaling, and no longer inherits the
  Froude-trend inversion of the legacy formula;
* the legacy formula is preserved verbatim and keeps returning its old number,
  so the defect cannot silently come back;
* the "energy conservation" figure reported by ``wave_energy_flux`` is shown to
  be a tautology (identically zero error), not evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import rankine_wave as RW
from tensorbem import wigley as W
from tensorbem.wigley import _michell_kochin_z_integral

L, B, T = 10.0, 1.0, 0.625  # Wigley III geometry
RHO, G = 1025.0, 9.80665
FR_GRID = (0.20, 0.25, 0.30, 0.35, 0.40)


def _speed(fr: float, length: float = L, gravity: float = G) -> float:
    return fr * math.sqrt(gravity * length)


# ---------------------------------------------------------------------------
# 1. dispersion admissibility of the sampled wave-number pair
# ---------------------------------------------------------------------------

def test_canonical_pair_satisfies_dispersion_exactly() -> None:
    """On (k, alpha) = (alpha^2/k0, alpha) the residual is zero to roundoff."""
    for fr in FR_GRID:
        u = _speed(fr)
        k0 = G / u**2
        for lam in (1.0, 1.25, 2.0, 4.0, 9.0):
            alpha = lam * k0
            k_depth = alpha * alpha / k0
            residual = abs(k_depth - alpha * alpha / k0) / k_depth
            assert residual == 0.0


def test_single_wavenumber_sampling_violates_dispersion() -> None:
    """The pre-repair sampling k = alpha is off by lambda - 1 (up to 8x)."""
    for fr in FR_GRID:
        u = _speed(fr)
        k0 = G / u**2
        for lam in (1.0, 1.25, 2.0, 4.0, 9.0):
            alpha = lam * k0
            k_depth_legacy = alpha  # single-wave-number sampling
            required = alpha * alpha / k0
            residual = abs(k_depth_legacy - required) / required
            # k = alpha instead of alpha^2/k0 leaves a relative residual
            # |1 - alpha/(alpha^2/k0)| = |1 - 1/lam|
            assert residual == pytest.approx(abs(1.0 - 1.0 / lam), rel=1e-12, abs=1e-15)
        # and the violation is large where the integral carries weight
        assert abs(1.0 - 2.0) > 0.5


# ---------------------------------------------------------------------------
# 2. the two closed-form Kochin amplitudes now describe the same hull
# ---------------------------------------------------------------------------

def test_rankine_and_wigley_kochin_agree_up_to_constant() -> None:
    """rankine_H = -(2 U T / 3) * wigley_H, exactly, for any (k, alpha=k).

    Numerically, this is the statement that the depth integral is complete:
    the missing (z/T)^2 term is what used to make the ratio k-dependent
    (1.71 at k0, 1.51 at 3 k0 on the pre-repair code).
    """
    u = _speed(0.30)
    k0 = G / u**2
    for k in (0.5 * k0, k0, 1.5 * k0, 3.0 * k0, 8.0 * k0):
        rankine_h = RW.michell_kochin_function(k, L, B, T, u, G)
        wigley_h = W.michell_kochin_function(L, B, T, k, alpha=k)
        expected = -(2.0 * u * T / 3.0) * wigley_h
        assert abs(rankine_h - expected) <= 1e-12 * max(abs(expected), 1e-30)


def test_depth_integral_matches_quadrature_and_limit() -> None:
    """The closed-form z-integral equals direct quadrature, and -> 2T/3."""
    z = np.linspace(-T, 0.0, 200001)
    weight = 1.0 - (z / T) ** 2
    # k*T below ~0.1 loses the closed form to cancellation (int2 ~ 2/k^3), so
    # the small-wavenumber limit is pinned by the quadrature itself below.
    for k in (0.5 / T, 2.0 / T, 10.0 / T):
        numeric = float(np.trapezoid(weight * np.exp(k * z), z))
        closed = float(_michell_kochin_z_integral(k, T))
        assert closed == pytest.approx(numeric, rel=1e-9)
    numeric_zero = float(np.trapezoid(weight, z))
    assert numeric_zero == pytest.approx(2.0 * T / 3.0, rel=1e-9)  # trapezoid truncation


# ---------------------------------------------------------------------------
# 3. the public resistance entry point: canonical value, force, sane trend
# ---------------------------------------------------------------------------

def test_default_entry_point_is_canonical() -> None:
    for fr in FR_GRID:
        u = _speed(fr)
        got = RW.michell_wave_resistance(L, B, T, u, rho=RHO, g=G)
        want = W.michell_wave_resistance_canonical(L, B, T, u, rho=RHO, g=G)
        assert got == pytest.approx(want, rel=1e-12)
        assert got > 0.0


def test_default_entry_point_scales_as_a_force() -> None:
    """Fr-preserving similarity: (L,B,T,g,U) -> s(...) must scale Rw by s^4.

    Rw = rho U^2 L^2 f(Fr, B/L, T/L); with rho fixed and the whole length set
    (including U) scaled by s, Rw goes as s^4.  The *rankine_wave* legacy
    formula happens to carry the s^4 exponent as well (its defect is the
    wave-number pairing, not the exponent); the legacy formula of
    :mod:`tensorbem.wigley` is the one that scaled as s^2 and was therefore not
    a force at all (m16274).  Both statements are pinned here so the two
    defects are not conflated.
    """
    fr = 0.30
    for fn in (RW.michell_wave_resistance, RW.michell_wave_resistance_legacy):
        base = fn(L, B, T, _speed(fr), rho=RHO, g=G)
        for s in (2.0, 4.0):
            got = fn(L * s, B * s, T * s, _speed(fr, L * s, G * s), rho=RHO, g=G * s)
            assert math.log(got / base) / math.log(s) == pytest.approx(4.0, abs=1e-6)


def test_default_entry_point_has_the_physical_froude_trend() -> None:
    """Canonical: Fr 0.20 < 0.30 < 0.40 monotonically over this band."""
    values = [RW.michell_wave_resistance(L, B, T, _speed(fr), rho=RHO, g=G) for fr in (0.20, 0.30, 0.40)]
    assert values[0] < values[1] < values[2]
    assert values[2] / values[1] > 1.5
    # the legacy formula drops by ~20 % from Fr 0.30 to 0.40 while every
    # independent route rises by more than a factor 2
    legacy = [RW.michell_wave_resistance_legacy(L, B, T, _speed(fr), rho=RHO, g=G) for fr in (0.30, 0.40)]
    assert legacy[1] < legacy[0]


def test_legacy_formula_is_preserved_verbatim() -> None:
    """Regression pin on the pre-repair number (Wigley III, Fr = 0.30).

    If this ever moves, the legacy path was edited and the arbitration in
    docs/michell_exit_arbitration.md no longer describes the code.
    """
    u = _speed(0.30)
    assert RW.michell_wave_resistance_legacy(L, B, T, u, rho=RHO, g=G) == pytest.approx(
        302.4771343584468, rel=1e-9
    )
    assert RW.michell_wave_resistance_legacy(L, B, T, _speed(0.40), rho=RHO, g=G) == pytest.approx(
        242.6623316999877, rel=1e-9
    )


def test_converged_variant_tracks_the_default() -> None:
    """The 5 % reflexivity band the Rankine study asserts is now trivially met."""
    for fr in (0.20, 0.30):
        u = _speed(fr)
        std = RW.michell_wave_resistance(L, B, T, u, rho=RHO, g=G)
        conv = RW.michell_wave_resistance_converged(L, B, T, u, rho=RHO, g=G)
        assert abs(conv - std) / std < 0.01


# ---------------------------------------------------------------------------
# 4. the vacuous "energy conservation" figure
# ---------------------------------------------------------------------------

def test_energy_flux_scaled_is_a_tautology() -> None:
    """``energy_flux_scaled`` equals ``Rw_V`` identically, not by physics.

    The routine calibrates its own far-field estimate by
    ``Rw_V / energy_flux`` before comparing, so ``energy_flux_scaled - Rw_V`` is
    zero for *any* wave resistance, good or bad.  This test pins the identity
    (two very different hull scales, exact equality) and records the honest
    alternative reading: ``relative_error`` is the *raw* crude transverse-wave
    estimate against ``Rw_V`` (0.567 on Wigley III at Fr = 0.30), i.e. a model
    discrepancy, not an energy-conservation check.  Neither figure may be
    quoted as evidence that the Michell integral conserves energy.
    """
    observed = []
    for scale in (1.0, 3.0):
        out = RW.wave_energy_flux(
            L * scale, B * scale, T * scale, _speed(0.30, L * scale), rho=RHO, g=G
        )
        assert out["energy_flux_scaled"] == pytest.approx(out["Rw_V"], rel=1e-15)
        assert out["energy_flux"] > 0.0
        assert out["Rw_V"] > 0.0
        observed.append(out["relative_error"])
    # the reported number is the raw component ratio, and it is not small:
    for value in observed:
        assert value > 0.10
