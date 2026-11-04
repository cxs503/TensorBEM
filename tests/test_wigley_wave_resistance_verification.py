"""Independent-quadrature verification of the Wigley wave resistance.

Closes the verification debt of the merged forward-speed feature: the
closed-form Michell/Kochin integrals and the port/starboard prefactor are
checked against a direct 2-D Gauss-Legendre evaluation of the defining
integral (no shared code), and the lam = 1 + s^2 substitution is checked
pointwise.
"""

import math

import pytest

from tensorbem.benchmarks.wigley_wave_resistance_verification import (
    kochin_direct_2d,
    run_verification,
    substitution_identity_error,
)
from tensorbem.wigley import michell_wave_resistance

L, B, T = 6.0, 1.0, 0.375
RHO, G = 1025.0, 9.80665


@pytest.fixture(scope="module")
def study():
    return run_verification(froude_numbers=(0.20, 0.30), nx=48, nz=48)


def test_substitution_identity_is_exact():
    assert substitution_identity_error() < 1e-12


def test_closed_form_kochin_matches_direct_quadrature(study):
    assert study.max_kochin_rel_err < 1e-8


def test_closed_form_resistance_matches_direct_kochin(study):
    assert study.max_resistance_rel_err < 1e-6


def test_two_paths_agree_within_three_percent(study):
    assert study.max_two_path_rel_err < 0.03


def test_direct_kochin_is_quadrature_converged():
    k = 2.0
    h96 = kochin_direct_2d(L, B, T, k, nx=96, nz=96)
    h48 = kochin_direct_2d(L, B, T, k, nx=48, nz=48)
    assert abs(h96 - h48) / abs(h96) < 1e-10


def test_resistance_is_positive_and_finite():
    for Fr in (0.15, 0.30, 0.40):
        rw = michell_wave_resistance(L, B, T, Fr * math.sqrt(G * L), RHO, G)
        assert 0.0 < rw < 1e6
