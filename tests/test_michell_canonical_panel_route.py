"""Canonical-spectrum Michell resistance for general hulls (m16365).

m16274 added :func:`tensorbem.wigley.michell_wave_resistance_canonical` (the
only implementation that satisfies the steady free-surface condition on the
Wigley closed form) and left the *general* panel route --
``michell.michell_resistance`` / ``wave_resistance_from_panels`` -- on the
single-wavenumber (legacy) spectrum.  m16345 arbitrated the two spectra and
deferred the general route.

This module pins the canonical general route:

* ``CenterplanePanels.kochin_pair(k_depth, alpha)`` -- decoupled wavenumbers,
* ``michell_resistance_canonical`` / ``wave_resistance_from_panels_canonical``
  / ``ellipsoid_wave_resistance_canonical`` -- canonical ``lam^2`` weight and
  ``rho g^2/(pi U^2)`` prefactor,
* ``tandem_interference_factor_canonical`` -- the same interference factor on
  the canonical weight (it is *not* the legacy number; see the pinned table).

The legacy functions are left bit-for-bit intact (their numbers are pinned
across ~8 modules and ``benchmarks/verified``); the tests below pin the
*defect* as well, so a future re-pin cannot silently regress into it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.michell import (
    CenterplanePanels,
    canonical_resistance_nodes,
    ellipsoid_wave_resistance,
    ellipsoid_wave_resistance_canonical,
    michell_resistance,
    michell_resistance_canonical,
    resistance_nodes,
    tandem_interference_factor,
    tandem_interference_factor_canonical,
    tandem_pair_resistance_canonical,
    wave_resistance_from_panels,
    wave_resistance_from_panels_canonical,
    wigley_panels,
)
from tensorbem.wigley import michell_wave_resistance_canonical

G = 9.80665
L, B, T = 1.0, 0.1, 1.0 / 16.0
FRUDES = (0.20, 0.25, 0.30, 0.35, 0.40)


def _speed(fr: float, length: float = L, g: float = G) -> float:
    return fr * math.sqrt(g * length)


def _panels(n_x: int = 64, n_z: int = 32, rule: str = "gauss", scale: float = 1.0):
    return wigley_panels(scale * L, scale * B, scale * T, n_x=n_x, n_z=n_z, rule=rule)


@pytest.fixture(scope="module")
def fine_panels():
    return _panels(64, 32)


def test_panel_route_matches_wigley_closed_form_canonical(fine_panels):
    """Panel discretisation of the Wigley hull vs the verified closed form."""
    for fr in FRUDES:
        u = _speed(fr)
        closed = michell_wave_resistance_canonical(L, B, T, u, g=G)
        panel = wave_resistance_from_panels_canonical(fine_panels, u, T, gravity=G)
        assert closed > 0.0
        assert abs(panel - closed) / closed < 0.03, (fr, panel, closed)


def test_panel_route_spectral_convergence():
    """Gauss panels converge fast: n_x 32 -> 64 collapses the Fr=0.20 error."""
    u = _speed(0.20)
    closed = michell_wave_resistance_canonical(L, B, T, u, g=G)
    errs = {}
    for n_x, n_z in ((16, 8), (32, 16), (64, 32)):
        p = _panels(n_x, n_z)
        errs[n_x] = abs(wave_resistance_from_panels_canonical(p, u, T, gravity=G) - closed) / closed
    assert errs[16] > errs[32] > errs[64]
    assert errs[64] < 1e-3
    assert errs[64] < errs[32] / 10.0


def test_kochin_pair_reduces_to_kochin(fine_panels):
    """``H(k, k)`` is the single-wavenumber amplitude of :meth:`kochin`."""
    ks = np.array([0.5, 2.0, 40.0])
    single = fine_panels.kochin(ks)
    pair = fine_panels.kochin_pair(ks, ks)
    assert pair.shape == single.shape
    assert np.max(np.abs(pair - single)) / np.max(np.abs(single)) < 1e-12
    scalar = fine_panels.kochin_pair(2.0, 2.0)
    assert isinstance(scalar, complex)
    assert abs(scalar - fine_panels.kochin(2.0)) / abs(single[1]) < 1e-12
    # broadcasting: scalar depth against an array of alphas
    assert fine_panels.kochin_pair(2.0, ks).shape == (3,)


def test_canonical_weight_is_lam4_times_legacy_weight():
    """The two spectra differ by exactly ``lam^4`` in the quadrature weight."""
    u = _speed(0.30)
    lam_l, w_l = resistance_nodes(u, T, gravity=G, n_seg=4, n_quad=16)
    lam_c, w_c = canonical_resistance_nodes(u, T, gravity=G, n_seg=4, n_quad=16)
    assert np.allclose(lam_l, lam_c)
    assert np.allclose(w_c / w_l, lam_c**4, rtol=1e-12)
    assert np.all(lam_c > 1.0)


def test_similarity_law_is_s4_canonical_and_s2_legacy(fine_panels):
    """Force scales as ``s^4`` when ``(L, B, T, U, g)`` are all scaled by ``s``.

    The same rescaling leaves ``k0 T`` invariant, so the quadrature is exactly
    self-similar.  The canonical route gives the force exponent 4; the legacy
    route gives 2 -- that mismatch *is* the dimensional defect arbitrated in
    ``docs/michell_unit_arbitration.md``.
    """
    u = _speed(0.30)
    s = 2.0
    base_c = wave_resistance_from_panels_canonical(fine_panels, u, T, gravity=G)
    base_l = wave_resistance_from_panels(fine_panels, u, T, gravity=G)
    scaled = wigley_panels(s * L, s * B, s * T, n_x=64, n_z=32, rule="gauss")
    sc = wave_resistance_from_panels_canonical(scaled, s * u, s * T, gravity=s * G)
    sl = wave_resistance_from_panels(scaled, s * u, s * T, gravity=s * G)
    exp_c = math.log(sc / base_c) / math.log(s)
    exp_l = math.log(sl / base_l) / math.log(s)
    assert abs(exp_c - 4.0) < 1e-6, exp_c
    assert abs(exp_l - 2.0) < 1e-6, exp_l


def test_canonical_quadrature_is_converged(fine_panels):
    """``n_quad`` 32 vs 128 on the canonical route agree to round-off."""
    u = _speed(0.30)
    a = wave_resistance_from_panels_canonical(fine_panels, u, T, gravity=G, n_quad=32)
    b = wave_resistance_from_panels_canonical(fine_panels, u, T, gravity=G, n_quad=128)
    assert abs(a - b) / b < 1e-8


def test_legacy_panel_route_is_low_by_an_order_of_magnitude(fine_panels):
    """Pinned defect: the legacy route under-predicts by ~12-24x (not a force)."""
    for fr in FRUDES:
        u = _speed(fr)
        canonical = wave_resistance_from_panels_canonical(fine_panels, u, T, gravity=G)
        legacy = wave_resistance_from_panels(fine_panels, u, T, gravity=G)
        ratio = legacy / canonical
        assert 0.03 < ratio < 0.11, (fr, ratio)
    # the general-route legacy function is the same quantity as the panel route
    u = _speed(0.30)
    assert abs(
        michell_resistance(fine_panels.kochin, u, T, gravity=G)
        / wave_resistance_from_panels(fine_panels, u, T, gravity=G) - 1.0
    ) < 1e-12


def test_tandem_factored_and_merged_canonical_agree(fine_panels):
    """``sigma`` from the closed-form ``(1 + exp(i alpha s))`` factor vs the
    merged source distribution -- two independent routes, canonical weight."""
    u = _speed(0.30)
    for gap in (0.5, 1.0, 2.0):
        factored = tandem_interference_factor_canonical(
            fine_panels.kochin_pair, u, T, gap * L, gravity=G
        )
        merged = tandem_pair_resistance_canonical(fine_panels, u, T, gap * L, gravity=G)
        sigma_merged = merged / (2.0 * factored["r_single"])
        assert abs(sigma_merged - factored["sigma"]) < 1e-8


def test_tandem_sigma_is_spectrum_dependent(fine_panels):
    """The interference factor moves by up to ~36 % between the two spectra.

    Pinned values; they document that the legacy ``sigma`` (the one claimed by
    the ``michell-general-and-tandem`` capability row) is spectrum-biased and
    therefore not a physical interference factor.
    """
    u = _speed(0.25)
    legacy = tandem_interference_factor(fine_panels.kochin, u, T, 0.5 * L, gravity=G)
    canonical = tandem_interference_factor_canonical(
        fine_panels.kochin_pair, u, T, 0.5 * L, gravity=G
    )
    assert 0.0 <= legacy["sigma"] <= 2.0
    assert 0.0 <= canonical["sigma"] <= 2.0
    assert abs(legacy["sigma"] - 0.4045) < 5e-4
    assert abs(canonical["sigma"] - 0.5513) < 5e-4
    shift = canonical["sigma"] / legacy["sigma"] - 1.0
    assert 0.30 < shift < 0.40, shift


def test_ellipsoid_canonical_route():
    u = _speed(0.30)
    canonical = ellipsoid_wave_resistance_canonical(
        1.0, 0.1, 1.0 / 16.0, u, gravity=G, n_psi=48, n_t=192
    )
    legacy = ellipsoid_wave_resistance(
        1.0, 0.1, 1.0 / 16.0, u, gravity=G, n_psi=48, n_t=192
    )
    assert canonical > 0.0
    assert legacy / canonical < 0.2  # legacy under-predicts by >5x here too
    s = 2.0
    scaled = ellipsoid_wave_resistance_canonical(
        2.0, 0.2, 2.0 / 16.0, s * u, gravity=s * G, n_psi=48, n_t=192
    )
    exponent = math.log(scaled / canonical) / math.log(s)
    assert abs(exponent - 4.0) < 5e-3, exponent


def test_input_validation(fine_panels):
    with pytest.raises(ValueError):
        canonical_resistance_nodes(0.0, T)
    with pytest.raises(ValueError):
        canonical_resistance_nodes(_speed(0.3), 0.0)
    with pytest.raises(ValueError):
        canonical_resistance_nodes(_speed(0.3), T, lambda_max=1.0)
    with pytest.raises(ValueError):
        michell_resistance_canonical(lambda kd, al: np.zeros(3), _speed(0.3), T)
    bogus = CenterplanePanels(np.zeros(1), np.zeros(1), np.ones(1), np.ones(1))
    with pytest.raises(ValueError):
        bogus.kochin_pair(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        tandem_interference_factor_canonical(
            fine_panels.kochin_pair, _speed(0.3), T, -1.0
        )