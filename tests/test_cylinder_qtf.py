"""Tests for the closed-form difference-frequency QTF of a bottom-mounted cylinder.

Every gate here is analytic (no panel BEM), so the whole file runs in seconds.
Author: TensorBEM node m17277 (case anchor BEM-QTF-DRIFT).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.cylinder_drift import (
    far_field_cylinder_drift,
    long_wave_drift_coefficient,
    near_field_cylinder_drift,
)
from tensorbem.cylinder_qtf import (
    closed_form_cylinder_qtf,
    pair_integrals,
    qtf_surge_abs,
)
from tensorbem.qtf_drift import DriftQTF

RHO = 1025.0
G = 9.81


@pytest.mark.parametrize("ka", [0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.4])
def test_diagonal_reproduces_the_exact_mean_drift(ka):
    """``T_ii`` must collapse onto the verified near-field mean drift force.

    This is the normalisation pin of the whole construction: the ``1/4``, the
    ``1/(k_i + k_j)`` depth integral and the waterline sign are all fixed by it.
    """
    closed = RHO * pair_integrals(ka, ka, radius=1.0).qtf_surge
    exact = near_field_cylinder_drift(
        ka, radius=1.0, density=RHO, gravity=G
    )["mean_drift_force_x"]
    assert exact > 0.0
    assert abs(closed - exact) / abs(exact) < 1e-10


@pytest.mark.parametrize("ka", [0.2, 0.5, 0.9, 1.4])
def test_diagonal_reproduces_the_reduced_maruo_far_field(ka):
    """Independent route: the same diagonal from the reduced Maruo far field."""
    closed = RHO * pair_integrals(ka, ka, radius=1.0).qtf_surge
    far = far_field_cylinder_drift(ka, radius=1.0, density=RHO, gravity=G)[
        "mean_drift_force_x"
    ]
    assert abs(closed - far) / abs(far) < 1e-3


def test_long_wave_limit_matches_the_classical_cubic_law():
    """``ka -> 0``: ``F_x = (5 pi^2/16) rho g A^2 a (ka)^3`` (Maruo/Vugts asymptote)."""
    coefficient = long_wave_drift_coefficient()
    for ka in (0.01, 0.02, 0.05):
        closed = RHO * pair_integrals(ka, ka, radius=1.0).qtf_surge
        law = coefficient * RHO * G * 1.0 * ka**3
        assert abs(closed - law) / abs(law) < 1e-2


def test_matrix_is_hermitian_and_diagonal_is_real():
    qtf = closed_form_cylinder_qtf((0.3, 0.5, 0.7, 1.0), radius=1.0, density=RHO)
    assert isinstance(qtf, DriftQTF)
    assert np.abs(qtf.matrix - np.conj(qtf.matrix.T)).max() < 1e-10
    assert np.abs(qtf.matrix.diagonal().imag).max() < 1e-10
    assert qtf.route == "cylinder_closed_form"


def test_diagonal_of_the_matrix_is_the_mean_drift_coefficient():
    ka = np.array([0.3, 0.6, 0.9])
    qtf = closed_form_cylinder_qtf(ka, radius=1.0, density=RHO)
    for i, x in enumerate(ka):
        exact = near_field_cylinder_drift(
            float(x), radius=1.0, density=RHO, gravity=G
        )["mean_drift_force_x"]
        assert abs(qtf.diagonal[i] - exact) / abs(exact) < 1e-10


def test_off_diagonal_is_complex_and_continuous_across_the_diagonal():
    """The QTF is complex off the diagonal and continuous onto it.

    Newman's approximation discards the imaginary part entirely, so this test also
    records *why* it can only be a narrow-band approximation: at
    ``(ka_i, ka_j) = (0.4, 0.9)`` the pair already carries
    ``|Im T| / |Re T| ~ 12 %``.  Continuity is what the analytic family must show
    as ``ka_j -> ka_i``.
    """
    base = pair_integrals(0.5, 0.5, radius=1.0)
    near = pair_integrals(0.5, 0.5 * 1.02, radius=1.0)
    assert abs(near.qtf_surge - base.qtf_surge) / abs(base.qtf_surge) < 5e-2
    assert abs(base.qtf_surge_imag) < 1e-12 * abs(base.qtf_surge)
    adjacent = pair_integrals(0.4, 0.9, radius=1.0)
    assert abs(adjacent.qtf_surge_imag) / abs(adjacent.qtf_surge) > 0.05


def test_qtf_scales_linearly_with_the_radius_at_fixed_ka():
    """Non-dimensionalisation check: at fixed ``ka`` the QTF is ``proportional to rho g a``."""
    small = pair_integrals(0.4, 0.9, radius=1.0)
    big = pair_integrals(0.4, 0.9, radius=2.5)
    assert big.qtf_surge / small.qtf_surge == pytest.approx(2.5, rel=1e-11)
    assert big.qtf_surge_imag / small.qtf_surge_imag == pytest.approx(2.5, rel=1e-11)


def test_azimuthal_quadrature_is_exact_for_the_finite_series():
    """The wall fields are finite Fourier series, so the periodic rule is exact."""
    coarse = pair_integrals(0.4, 0.75, radius=1.0, n_theta=256)
    fine = pair_integrals(0.4, 0.75, radius=1.0, n_theta=1024)
    assert abs(coarse.qtf_surge - fine.qtf_surge) <= 1e-10 * abs(fine.qtf_surge)
    assert abs(coarse.qtf_surge_imag - fine.qtf_surge_imag) <= 1e-10 * abs(
        fine.qtf_surge
    )


def test_surge_sway_mirror_symmetry_keeps_the_force_along_the_wave():
    """Only ``F_x`` is non-zero: the wall integrand for ``n_y = sin(theta)`` vanishes."""
    n_theta = 512
    theta = (2.0 * math.pi / n_theta) * np.arange(n_theta, dtype=float)
    from tensorbem.cylinder_drift import cylinder_wall_field

    f_i = cylinder_wall_field(0.5, theta)
    f_j = cylinder_wall_field(0.8, theta)
    prod = f_i["psi"] * np.conj(f_j["psi"])
    sway = float(np.sum(prod * np.sin(theta)).real) * (2.0 * math.pi / n_theta)
    surge = float(np.sum(prod * np.cos(theta)).real) * (2.0 * math.pi / n_theta)
    assert abs(sway) < 1e-12
    assert abs(surge) > 0.1


def test_qtf_is_odd_in_the_frequency_pair_label():
    """``T(omega_i, omega_j) = conj(T(omega_j, omega_i))`` for amplitudes."""
    a = pair_integrals(0.4, 0.9, radius=1.0)
    b = pair_integrals(0.9, 0.4, radius=1.0)
    assert a.qtf_surge == pytest.approx(b.qtf_surge, rel=1e-12, abs=0.0)
    assert a.qtf_surge_imag == pytest.approx(-b.qtf_surge_imag, rel=1e-12, abs=0.0)
    assert qtf_surge_abs(0.4, 0.9, radius=1.0) == pytest.approx(
        qtf_surge_abs(0.9, 0.4, radius=1.0), rel=1e-12
    )
