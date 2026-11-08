"""Verification of the side-by-side (catamaran) interference capability.

Module under test: :mod:`tensorbem.multihull`, ``lateral_*`` functions.

The claim being gated is *not* "the thin-ship modelling is accurate" (it is not:
``B/L = 0.1`` here, and viscous resistance is outside the model).  It is that the
**lateral wave-number phase** and the interference algebra are right, which is
what the analytic limits below pin down:

1. coincident hulls give exactly ``sigma = 2`` (no quadrature involved: the pair
   factor is algebraically ``4 |H|^2``);
2. the initial curvature of ``sigma(b)`` equals ``-k0^2 <psi^2> / 2`` where
   ``psi(lambda) = lambda sqrt(lambda^2 - 1)`` is an analytic moment of the
   hull's own Kelvin weight (:func:`lateral_phase_moments`) -- a wrong phase
   factor changes ``<psi^2>``;
3. the two routes (``lambda`` substitution vs wave-angle quadrature, different
   parameterisation *and* different phase expression) agree;
4. the explicit ``y = +-b/2`` panel route reproduces the factorised phase to
   round-off, and the resistance is invariant to the transverse origin;
5. wide spacing drives ``sigma -> 1`` and the pair resistance never exceeds
   ``4 R_single`` (``|1 + exp(i phi)|^2 <= 4``).
"""

from __future__ import annotations

import cmath
import json
import math
from pathlib import Path

import numpy as np
import pytest

import tensorbem.multihull as multihull
from tensorbem.multihull import (
    analytic_wigley_kochin,
    explicit_lateral_pair_kochin,
    lateral_interference_curve,
    lateral_interference_factor,
    lateral_kochin,
    lateral_phase_moments,
    lateral_wave_resistance,
    lateral_wave_resistance_theta,
    lateral_wavenumber,
    wave_number_k0,
)

L, B, T = 2.0, 0.2, 0.125
FR = 0.3
K0 = wave_number_k0(FR, L)
KOCHIN = analytic_wigley_kochin(L, B, T)
REPO_ROOT = Path(__file__).resolve().parents[1]


def sigma(separation: float, k0: float = K0, kochin=KOCHIN) -> float:
    return lateral_interference_factor(kochin, k0=k0, separation=separation)


# ---------------------------------------------------------------------------
# 1. analytic limits
# ---------------------------------------------------------------------------
def test_coincident_hulls_double_the_resistance() -> None:
    """``|1 + exp(0)|^2 = 4``: sigma(0) = 2 without any quadrature error."""
    assert sigma(0.0) == pytest.approx(2.0, abs=1e-12)
    single, pair = lateral_wave_resistance(KOCHIN, k0=K0, separation=0.0)
    assert pair == pytest.approx(4.0 * single, rel=1e-12)


def test_transverse_wavenumber_vanishes_at_the_transverse_cusp() -> None:
    """``k_y = 0`` for the transverse system (``k = k0``), then grows with ``k``."""
    assert lateral_wavenumber(K0, K0) == 0.0
    assert lateral_wavenumber(0.5 * K0, K0) == 0.0
    ky = [lateral_wavenumber(lam * K0, K0) for lam in (1.0, 1.2, 1.5, 2.0, 5.0)]
    assert all(b > a for a, b in zip(ky, ky[1:]))
    # closed form k_y = k0 lam sqrt(lam^2 - 1) at lam = 2 -> k0 * 2 * sqrt(3)
    assert lateral_wavenumber(2.0 * K0, K0) == pytest.approx(
        K0 * 2.0 * math.sqrt(3.0), rel=1e-14
    )


def test_small_separation_curvature_matches_the_phase_moment() -> None:
    """``sigma = 2 - (k0 b)^2 <psi^2>/2 + O((k0 b)^4)`` -- moment identity."""
    moments = lateral_phase_moments(KOCHIN, k0=K0)
    for k0b in (0.02, 0.05, 0.1):
        measured = sigma(k0b / K0)
        predicted = 2.0 - 0.5 * k0b**2 * moments["m2"]
        next_order = k0b**4 * moments["m4"] / 24.0
        assert abs(measured - predicted) < next_order


def test_long_wave_limit_is_constructive() -> None:
    """``k0 b -> 0`` (hull short against the wavelength) keeps ``sigma`` near 2.

    The phase moment ``<psi^2>`` is a moment of the *hull's* weight at that
    frequency, so it must be recomputed per ``k0``; the bound below is therefore
    stated on ``sigma`` itself rather than on a moment carried over from
    :data:`K0`.
    """
    k0 = 0.5
    value = sigma(0.5, k0=k0)
    assert 1.5 < value <= 2.0
    assert value < sigma(0.5, k0=K0) + 1.0  # still bounded above by the limit 2


def test_wide_separation_approaches_the_isolated_hull() -> None:
    for k0b in (10.0, 20.0, 50.0):
        assert abs(sigma(k0b / K0) - 1.0) < 0.05


# ---------------------------------------------------------------------------
# 2. two independent routes for the same integral
# ---------------------------------------------------------------------------
def test_wave_angle_route_reproduces_the_lambda_route() -> None:
    for ratio in (0.25, 0.5, 1.0):
        b = ratio * L
        single_lam, pair_lam = lateral_wave_resistance(KOCHIN, k0=K0, separation=b)
        single_th, pair_th = lateral_wave_resistance_theta(
            KOCHIN, k0=K0, separation=b, n_quad=400
        )
        assert single_th == pytest.approx(single_lam, rel=1e-6)
        assert pair_th / (2.0 * single_th) == pytest.approx(
            pair_lam / (2.0 * single_lam), rel=2e-3
        )


def test_quadrature_convergence_of_the_interference_factor() -> None:
    coarse = lateral_interference_factor(KOCHIN, k0=K0, separation=0.5 * L, n_quad=128)
    fine = lateral_interference_factor(KOCHIN, k0=K0, separation=0.5 * L, n_quad=384)
    assert coarse == pytest.approx(fine, abs=2e-3)


# ---------------------------------------------------------------------------
# 3. explicit panel route, and invariance to the transverse origin
# ---------------------------------------------------------------------------
def test_explicit_panel_route_recovers_the_factorised_phase() -> None:
    """The ``y = +-b/2`` panel sum equals ``exp(-i k_y b/2)`` times the factor."""
    worst = 0.0
    for k in (K0, 1.2 * K0, 2.0 * K0):
        ky = lateral_wavenumber(k, K0)
        base = explicit_lateral_pair_kochin(
            k, L, B, T, separation=0.0, k0=K0, n_x=40, n_z=20
        ) / lateral_kochin(KOCHIN, k0=K0, separation=0.0)(k)
        for b in (0.5, 1.0, 3.0, 10.0):
            value = explicit_lateral_pair_kochin(
                k, L, B, T, separation=b, k0=K0, n_x=40, n_z=20
            )
            factorised = lateral_kochin(KOCHIN, k0=K0, separation=b)(k)
            residual = value / (cmath.exp(-1j * ky * b / 2.0) * factorised)
            worst = max(worst, abs(residual / base - 1.0))
    assert worst < 1e-12


def test_resistance_is_invariant_to_the_transverse_origin() -> None:
    """A pure ``y`` translation cannot change ``R_W`` -- checked on the two routes."""
    from tensorbem.ellipsoid_kochin import kelvin_wave_resistance

    b = 0.5 * L
    single, pair = lateral_wave_resistance(KOCHIN, k0=K0, separation=b)
    pair_centred = kelvin_wave_resistance(
        lambda k: explicit_lateral_pair_kochin(
            k, L, B, T, separation=b, k0=K0, n_x=60, n_z=30
        ),
        k0=K0,
    )
    assert single == pytest.approx(
        kelvin_wave_resistance(KOCHIN, k0=K0), rel=1e-12
    )
    assert pair_centred == pytest.approx(pair, rel=2e-2)


# ---------------------------------------------------------------------------
# 4. bounds and the curve helper
# ---------------------------------------------------------------------------
def test_interference_factor_is_bounded_and_reaches_both_sides_of_one() -> None:
    separations = np.linspace(0.0, 3.0 * L, 25)
    values = lateral_interference_curve(KOCHIN, k0=K0, separations=separations)
    assert np.all(values <= 2.0 + 1e-9)
    assert np.all(values >= 0.0)
    # the sweep must contain both a beneficial (sigma < 1) and a detrimental
    # (sigma > 1) spacing: a catamaran interference diagram with only one sign
    # would mean the phase oscillates the wrong way round.
    assert values.min() < 0.995
    assert values.max() > 1.05


def test_curve_matches_the_scalar_calls() -> None:
    separations = [0.0, 0.25 * L, L]
    curve = lateral_interference_curve(KOCHIN, k0=K0, separations=separations)
    assert list(curve) == [pytest.approx(sigma(b)) for b in separations]


def test_non_physical_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        lateral_interference_factor(KOCHIN, k0=K0, separation=-1.0)
    with pytest.raises(ValueError):
        lateral_kochin(KOCHIN, k0=0.0, separation=1.0)
    with pytest.raises(ValueError):
        lateral_kochin(KOCHIN, k0=K0, separation=float("nan"))
    with pytest.raises(ValueError):
        lateral_wavenumber(-1.0, K0)
    with pytest.raises(ValueError):
        lateral_wave_resistance_theta(KOCHIN, k0=-1.0, separation=1.0)


# ---------------------------------------------------------------------------
# 5. committed study artifact (gate: JSON must match the module)
# ---------------------------------------------------------------------------
STUDY = REPO_ROOT / "docs" / "multihull_lateral" / "study.json"


def test_study_artifact_reproduces_from_the_module() -> None:
    if not STUDY.exists():  # pragma: no cover - artifact is committed
        pytest.skip("study artifact not built")
    data = json.loads(STUDY.read_text())
    for row in data["sweeps"]:
        k0 = wave_number_k0(row["froude"], L)
        for point in row["points"][::10]:
            assert point["sigma"] == pytest.approx(
                lateral_interference_factor(
                    KOCHIN, k0=k0, separation=point["separation_over_L"] * L
                ),
                abs=1e-9,
            )
    from tensorbem.benchmarks import multihull_lateral_study

    markdown = multihull_lateral_study.render_markdown(data)
    assert "coincident hulls" in markdown
    assert "Side-by-side (catamaran) wave interference" in markdown
    verdict = data["verdict"]
    assert verdict["coincident_sigma_error"] < 1e-12
    assert verdict["angle_vs_lambda_max_relative"] < 2e-3
    assert verdict["explicit_phase_residual"] < 1e-12