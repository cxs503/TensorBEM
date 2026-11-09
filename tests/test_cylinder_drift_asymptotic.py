"""Tests for the exact cylinder drift-force verification (module + driver)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import cylinder_drift as cd
from tensorbem.benchmarks import cylinder_drift_asymptotic as study

RHO = cd.DEFAULT_DENSITY
G = cd.DEFAULT_GRAVITY


def test_bessel_j_matches_reference_values():
    ref = {
        (0, 1.0): 0.7651976865579666,
        (1, 1.0): 0.4400505857449335,
        (2, 1.0): 0.1149034849319005,
        (0, 2.0): 0.2238907791412357,
        (5, 2.0): 0.0070396297558717,
    }
    for (m, x), expected in ref.items():
        assert cd.bessel_j(m, x) == pytest.approx(expected, rel=1e-12)


def test_bessel_y_matches_reference_values():
    ref = {
        (0, 1.0): 0.0882569642156769,
        (1, 1.0): -0.7812128213002889,
        (2, 1.0): -1.6506826068162548,
        (0, 2.0): 0.5103756726497453,
        (3, 2.0): -1.127783776840428,
    }
    for (m, x), expected in ref.items():
        assert cd.bessel_y(m, x) == pytest.approx(expected, rel=1e-11)


def test_bessel_derivative_identity():
    x = 0.7
    for m in range(4):
        lhs = 0.5 * (cd.bessel_j(m - 1, x) - cd.bessel_j(m + 1, x))
        num = (cd.bessel_j(m, x + 1e-6) - cd.bessel_j(m, x - 1e-6)) / 2e-6
        assert lhs == pytest.approx(num, rel=1e-7)


def test_wall_anchor_reproduces_rho_g_A2_over_2():
    """Fully reflecting wall: the near-field formula must return rho g A^2 / 2."""
    info = study._wall_anchor()
    assert info["near_field_force"] == pytest.approx(0.5 * RHO * G, rel=1e-12)
    assert info["rel_error"] < 1e-12
    # the waterline term is not a correction: it is the same order as the surface
    # term and the two must combine to the anchor
    assert abs(info["waterline_term"]) == pytest.approx(abs(info["surface_term"]) * 2.0, rel=1e-12)


@pytest.mark.parametrize("ka", [0.2, 0.5, 1.0, 1.5, 2.0])
def test_rigid_cylinder_wall_satisfies_neumann(ka):
    theta = np.linspace(0.0, 2.0 * math.pi, 361, endpoint=False)
    field = cd.cylinder_wall_field(ka, theta)
    assert np.max(np.abs(field["dpsi_dr"])) < 1e-12


@pytest.mark.parametrize("ka", [0.2, 0.5, 1.0, 1.5, 2.0])
def test_near_field_and_far_field_agree(ka):
    near = cd.near_field_cylinder_drift(ka)
    far = cd.far_field_cylinder_drift(ka)
    assert near["mean_drift_force_x"] > 0.0
    rel = abs(near["mean_drift_force_x"] - far["mean_drift_force_x"]) / abs(
        far["mean_drift_force_x"]
    )
    assert rel < 1e-10


def test_drift_force_scales_with_amplitude_squared():
    base = cd.far_field_cylinder_drift(1.0)["mean_drift_force_x"]
    doubled = cd.far_field_cylinder_drift(1.0, amplitude=2.0)["mean_drift_force_x"]
    assert doubled == pytest.approx(4.0 * base, rel=1e-12)


def test_drift_force_is_linear_in_radius_at_fixed_ka():
    a = cd.far_field_cylinder_drift(1.0, radius=1.0)["mean_drift_force_x"]
    b = cd.far_field_cylinder_drift(1.0, radius=2.0)["mean_drift_force_x"]
    assert b == pytest.approx(2.0 * a, rel=1e-10)


def test_long_wave_exponent_is_three():
    info = study._long_wave()
    assert info["fitted_slope"] == pytest.approx(3.0, abs=0.2)


def test_long_wave_coefficient_matches_closed_form():
    info = study._long_wave()
    assert info["closed_form_coefficient"] == pytest.approx(5.0 * math.pi**2 / 16.0, rel=1e-12)
    assert info["coefficient_rel_error_at_smallest_ka"] < 3e-3


@pytest.mark.parametrize("ka", [0.5, 1.0, 1.5])
def test_kochin_energy_balance_residual(ka):
    bal = cd.energy_balance(ka, radius_probe=10.0)
    assert bal["residual"] < 1e-9
    assert bal["flux_scattered"] > 0.0
    assert bal["flux_interference"] < 0.0


def test_energy_balance_is_radius_independent():
    residuals = [cd.energy_balance(1.0, radius_probe=r)["residual"] for r in (5.0, 10.0, 20.0)]
    assert max(residuals) < 1e-9


def test_energy_balance_rejects_unresolved_probe_radius():
    with pytest.raises(ValueError, match="k\\*R"):
        cd.energy_balance(2.0, radius_probe=40.0)


def test_kochin_3d_matches_2d_scattering_amplitude():
    theta = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    k = 0.8
    s = cd.kochin_2d(k, theta)
    kk = cd.kochin_3d(k, theta)
    assert np.allclose(kk, -2.0j * s / k)


def test_mode_truncation_is_converged():
    a = cd.far_field_cylinder_drift(2.0, m_max=10)["mean_drift_force_x"]
    b = cd.far_field_cylinder_drift(2.0, m_max=18)["mean_drift_force_x"]
    assert b == pytest.approx(a, rel=1e-10)


def test_azimuthal_resolution_is_converged():
    a = cd.far_field_cylinder_drift(1.0, n_theta=720)["mean_drift_force_x"]
    b = cd.far_field_cylinder_drift(1.0, n_theta=4096)["mean_drift_force_x"]
    assert b == pytest.approx(a, rel=1e-6)


def test_richardson_extrapolate_recovers_polynomial_limit():
    # f(h) = 1 + h^2 with h, h/2 -> Richardson with p = 2 is exact
    f = lambda h: 1.0 + h**2
    assert cd.richardson_extrapolate(f(0.2), f(0.1), 2.0, 2.0) == pytest.approx(1.0, abs=1e-14)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        cd.far_field_cylinder_drift(0.0)
    with pytest.raises(ValueError):
        cd.near_field_cylinder_drift(0.5, n_theta=101)  # odd
    with pytest.raises(ValueError):
        cd.maccamy_fuchs_modes(0.5, m_max=1)


def test_study_gates_and_schema():
    built = study.build_study()
    verdict = built["verdict"]
    assert verdict["all_gates_pass"] is True
    assert verdict["near_far_max_rel_diff"] < 0.05
    assert verdict["long_wave_slope_pass"] is True
    assert verdict["energy_residual_max"] < 0.01
    assert verdict["wall_anchor_pass"] is True
    assert len(built["rows"]) == 8
    for row in built["rows"]:
        assert set(row) >= {"ka", "near_field_force", "far_field_force", "near_far_rel_diff"}
    assert built["repo_convention_audit"]["ratio_repo_over_exact"] == pytest.approx(
        2.0 * math.pi, rel=1e-3
    )