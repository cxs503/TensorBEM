"""Identity gates for the m17299 real-spherical-harmonic ladder rewrite.

``all_real_spherical_harmonics`` used to evaluate ``real_spherical_harmonic(l, m)``
once per ``(l, m)``, and every one of those calls restarted the
associated-Legendre recurrence at ``P_m^m`` -- ``O(p**3)`` work per target.  The
shipped version runs one recurrence per order ``m`` and reads every ``l`` off it
(``O(p**2)``), which is 1.7x-2.8x faster in the kernel (see
``docs/fmm_cost_audit/README.md`` §8).

A cost rewrite is only allowed to change the cost, so these tests pin the
*numbers*: the ladder output must equal the per-``(l, m)`` reference (which is
still the public ``real_spherical_harmonic``) to machine precision, for every
order and every column.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.fmm import all_real_spherical_harmonics, real_spherical_harmonic


def _reference(l_max: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """The pre-m17299 evaluation order: one harmonic call per ``(l, m)``."""

    out = np.zeros((len(theta), (l_max + 1) ** 2), dtype=float)
    col = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            out[:, col] = real_spherical_harmonic(l, m, theta, phi)
            col += 1
    return out


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 6, 8])
def test_ladder_matches_the_per_lm_reference(p: int) -> None:
    rng = np.random.default_rng(1234 + p)
    theta = np.arccos(rng.uniform(-1.0, 1.0, 257))
    phi = rng.uniform(0.0, 2.0 * math.pi, 257)
    ladder = all_real_spherical_harmonics(p, theta, phi)
    reference = _reference(p, theta, phi)
    assert ladder.shape == reference.shape == (257, (p + 1) ** 2)
    scale = float(np.max(np.abs(reference))) or 1.0
    assert float(np.max(np.abs(ladder - reference))) / scale < 1e-13


def test_ladder_is_exact_at_the_poles() -> None:
    """``theta = 0`` / ``pi`` is the removable-singularity end of the ladder."""

    theta = np.array([0.0, math.pi, 0.0, math.pi])
    phi = np.array([0.0, 0.0, 1.3, -2.1])
    ladder = all_real_spherical_harmonics(4, theta, phi)
    reference = _reference(4, theta, phi)
    scale = float(np.max(np.abs(reference))) or 1.0
    assert float(np.max(np.abs(ladder - reference))) / scale < 1e-13
    # only m = 0 survives at the poles
    assert np.all(np.abs(ladder[:, 0]) > 0.0)


def test_empty_target_set_keeps_the_documented_shape() -> None:
    empty = np.zeros(0, dtype=float)
    for p in (0, 3):
        out = all_real_spherical_harmonics(p, empty, empty)
        assert out.shape == (0, (p + 1) ** 2)


def test_order_zero_is_the_monopole_harmonic() -> None:
    theta = np.linspace(0.1, 3.0, 7)
    phi = np.linspace(-1.0, 1.0, 7)
    y00 = all_real_spherical_harmonics(0, theta, phi)[:, 0]
    assert np.allclose(y00, 1.0 / math.sqrt(4.0 * math.pi), rtol=0.0, atol=1e-15)


def test_column_ordering_is_l_then_m() -> None:
    """``_sph_index(l, m) = l*l + (m + l)`` -- the far-field scale vector relies on it."""

    rng = np.random.default_rng(7)
    theta = np.arccos(rng.uniform(-1.0, 1.0, 5))
    phi = rng.uniform(0.0, 2.0 * math.pi, 5)
    ladder = all_real_spherical_harmonics(3, theta, phi)
    for l in range(4):
        for m in range(-l, l + 1):
            assert np.allclose(ladder[:, l * l + (m + l)],
                               real_spherical_harmonic(l, m, theta, phi),
                               rtol=0.0, atol=1e-15)