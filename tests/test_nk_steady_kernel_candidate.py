"""Machine checks for the candidate steady Neumann-Kelvin kernel (mission m17117).

The capability is **WITHHELD**: these tests pin what the candidate does establish
(exact spectral form, real-valued kernel, converged quadrature, correct
asymptotics) and they pin the *failure* that keeps the capability withheld, so a
later mission cannot quietly claim the debt is closed without beating the
recorded referee numbers.
"""

import math

import numpy as np
import pytest

from tensorbem import nk_steady_kernel as nks
from tensorbem.neumann_kelvin import steady_free_surface_residual as shipped_residual

SOURCE = [0.0, 0.0, -1.0]
KAPPA = 1.0
POINTS = [(1.0, 0.0), (0.7, 0.7), (-2.0, 0.5)]


def test_kelvin_pole_wavenumber_is_sec_squared():
    theta = np.array([0.0, 0.4, 1.0, -0.7])
    k0 = nks.kelvin_pole_wavenumber(theta, KAPPA)
    assert np.allclose(k0, KAPPA / np.cos(theta) ** 2, rtol=0, atol=0)


def test_asymptotic_branch_matches_exponential_integral_in_overlap():
    """The two code paths must agree where both are valid (20 < |c| < 25)."""
    for sign in (+1.0, -1.0):
        c = np.array([20.0 + 3.0j, 22.0 - 4.0j, 19.5 + 10.0j]) * sign
        s = np.full(c.shape, sign)
        small = nks._q_small(c, s)
        large = nks._q_large(c, s)
        assert np.allclose(small, large, rtol=2e-3, atol=0)


def test_asymptotic_leading_behaviour():
    """Q(c) -> -2/c for |c| -> infinity."""
    c = np.array([1e4 + 1e3j])
    s = np.array([1.0])
    q = nks.q_kernel(c, s)
    assert abs(q[0] - (-2.0 / c[0])) / abs(2.0 / c[0]) < 1e-3


def test_wave_part_is_real():
    """A real Green function must have a real wave part."""
    for x, y in POINTS:
        value = nks.steady_wave_part(x, y, 0.3, KAPPA, n_theta=1200)
        assert math.isfinite(value)
    # the reduced integral returns float; the imaginary residual of the sum is the
    # quantity that mattered during the derivation, so check it explicitly.
    theta = (np.arange(1200) + 0.5) * (2 * math.pi / 1200)
    a = 0.3 - 1j * 2.0 * np.cos(theta)
    c = nks.kelvin_pole_wavenumber(theta, KAPPA) * a
    integrand = nks.q_kernel(c, np.sign(np.cos(theta))) / a**2
    assert abs(integrand.sum().imag) / abs(integrand.sum()) < 1e-12


def test_quadrature_is_converged():
    """2000 and 12000 theta nodes agree to <1e-4: the failure is not quadrature."""
    for x, y in POINTS:
        coarse = nks.steady_wave_part(x, y, 0.3, KAPPA, n_theta=2000)
        fine = nks.steady_wave_part(x, y, 0.3, KAPPA, n_theta=12000)
        assert abs(coarse - fine) / max(abs(fine), 1e-30) < 1e-4


def test_wave_part_decays_with_depth():
    shallow = abs(nks.steady_wave_part(3.0, 0.0, 0.1, KAPPA, n_theta=800))
    deep = abs(nks.steady_wave_part(3.0, 0.0, 2.0, KAPPA, n_theta=800))
    assert deep < shallow


def test_rankine_limit_recovers_closed_form():
    """With the wave part dropped, the kernel is the Rankine+mirror pair."""
    target = np.array([1.0, 0.0, -0.5])
    dx, dy, dz = target - np.asarray(SOURCE)
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    r1 = math.sqrt(dx * dx + dy * dy + (target[2] + SOURCE[2]) ** 2)
    expected = 1.0 / (4 * math.pi * r) + 1.0 / (4 * math.pi * r1)
    got = nks.steady_green(target, SOURCE, KAPPA, n_theta=600)
    wave = nks.steady_wave_part(dx, dy, -(target[2] + SOURCE[2]), KAPPA, n_theta=600)
    assert abs((got - wave) - expected) < 1e-12


@pytest.mark.parametrize("x,y", POINTS)
@pytest.mark.parametrize("fd_step", [1e-2, 1e-3])
def test_free_surface_referee_is_not_yet_satisfied(x, y, fd_step):
    """Documented FAIL: the candidate does not close the debt (keep WITHHELD)."""
    cand = nks.free_surface_residual([x, y, 0.0], SOURCE, KAPPA, h=fd_step, n_theta=1200)
    assert cand["relative"] > 1e-2, cand
    old = shipped_residual([x, y, 0.0], SOURCE, KAPPA, h=fd_step)
    assert old["relative"] > 1e-2, old


def test_referee_table_shape_and_recorded_baseline():
    """The recorded baseline in KERNEL_MEASUREMENTS is reproducible in magnitude."""
    rows = nks.residual_table([(1.0, 0.0), (0.7, 0.7), (-2.0, 0.5)], SOURCE, KAPPA, steps=(1e-3,))
    assert len(rows) == 3
    recorded = nks.KERNEL_MEASUREMENTS["candidate_relative_residual"]
    by_key = {f"x={r['x']},y={r['y']}": r["candidate_relative"] for r in rows}
    for key, value in recorded.items():
        assert abs(by_key[key] - value) / value < 0.35, (key, by_key[key], value)


def test_status_stays_withheld():
    assert "WITHHELD" in nks.KERNEL_STATUS
    assert "G_z + G_xx" in nks.KERNEL_MEASUREMENTS["referee"]
    assert set(nks.KERNEL_MEASUREMENTS["candidate_relative_residual"]) == set(
        nks.KERNEL_MEASUREMENTS["shipped_frequency_domain_kernel_relative_residual"]
    )