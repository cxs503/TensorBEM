"""Tests for the CASE-BEM-CUMMINS-WIGLEY case module (no BEM solve needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.causality import (
    fit_damping_power_tail,
    kk_added_mass_closure,
    power_tail_principal_value,
)


def _synthetic_pair(band=(0.2, 8.0), n=41):
    """Exact causal pair: B = sum c a/(a^2+w^2), A - A_inf = -sum c/(a^2+w^2)."""
    c = np.array([2.0, -1.3, 0.4])
    a = np.array([0.8, 2.5, 7.0])
    w = np.logspace(math.log10(band[0]), math.log10(band[1]), n)
    b = sum(cj * aj / (aj**2 + w**2) for cj, aj in zip(c, a))
    mass = sum(-cj / (aj**2 + w**2) for cj, aj in zip(c, a))
    return w, mass, b


def _fake_sweep(w, mass, b, label="coarse"):
    ones = np.ones(w.size)
    added = np.zeros((w.size, 6, 6))
    damping = np.zeros((w.size, 6, 6))
    for i in range(6):
        added[:, i, i] = mass
        damping[:, i, i] = b
    return case.Sweep(
        label=label,
        n_sections=12,
        n_levels=4,
        panel_count=192,
        max_panel_scale=0.05,
        omega=w,
        added_mass=added,
        damping=damping,
        solver_relative_residual=ones * 1e-12,
        elapsed=0.0,
    )


def test_power_tail_integral_matches_quadrature() -> None:
    from scipy.integrate import quad

    omega, coefficient, exponent, start = 0.7, 3.1, 3.5, 2.0
    numeric = quad(
        lambda s: coefficient * s ** (-exponent) / (s**2 - omega**2),
        start,
        np.inf,
        limit=400,
    )[0]
    analytic = power_tail_principal_value(omega, coefficient, exponent, start)
    assert abs(analytic - numeric) / abs(numeric) < 1e-10


def test_power_tail_rejects_slow_decay() -> None:
    with pytest.raises(ValueError):
        power_tail_principal_value(0.5, 1.0, 2.0, 3.0)


def test_kk_closure_is_small_for_an_exact_causal_pair() -> None:
    w, mass, b = _synthetic_pair()
    closure = kk_added_mass_closure(w, mass, b)
    assert closure.max_relative_residual < 0.06


def test_corrupted_damping_shape_breaks_the_closure() -> None:
    w, mass, b = _synthetic_pair()
    clean = kk_added_mass_closure(w, mass, b).max_relative_residual
    # a sign flip over the upper half of the band is a non-causal pair
    flipped = kk_added_mass_closure(w, mass, b * np.where(w > 1.0, -1.0, 1.0))
    assert flipped.max_relative_residual > 5.0 * clean


def test_damping_tail_fit_is_valid_on_an_algebraic_decay() -> None:
    w = np.logspace(math.log10(0.5), math.log10(20.0), 30)
    tail = fit_damping_power_tail(w, 12.0 * w ** (-4.0))
    assert tail.valid
    assert tail.exponent == pytest.approx(4.0, rel=1e-6)


def test_geometry_gate_holds_on_the_production_mesh() -> None:
    gate = case.geometry_gate()
    assert gate["passed"] is True
    assert gate["values"]["block_coefficient"] == pytest.approx(4.0 / 9.0, rel=1e-2)
    assert gate["values"]["volume_relative"] < 0.02


def test_panel_adequacy_gate_bites_when_the_mesh_is_too_coarse() -> None:
    w, mass, b = _synthetic_pair()
    sweep = _fake_sweep(w, mass, b)
    assert case.panel_adequacy_gate(sweep)["passed"] is True
    sweep.max_panel_scale = 1.0  # a 1 m panel cannot resolve a 0.1 m wavelength
    assert case.panel_adequacy_gate(sweep)["passed"] is False


def test_decay_zeta_recovers_a_known_logarithmic_decrement() -> None:
    zeta = 0.05
    omega_n = 4.0
    omega_d = omega_n * math.sqrt(1.0 - zeta**2)
    t = np.arange(0.0, 20.0, 0.002)
    x = np.exp(-zeta * omega_n * t) * np.cos(omega_d * t)
    recovered, period, n_peaks = case._decay_zeta(t, x)
    assert n_peaks >= 3
    assert period == pytest.approx(2.0 * math.pi / omega_d, rel=2e-3)
    assert recovered == pytest.approx(zeta, rel=2e-2)


def test_kk_gate_reports_a_corrupted_control_as_failed_gate() -> None:
    w, mass, b = _synthetic_pair()
    sweep = _fake_sweep(w, mass, b)
    clean = case.kk_gate(sweep)
    assert clean["values"]["worst_relative_residual"] < 0.10
    corrupted = case.kk_gate(sweep, corrupt=True)
    # the control gate "passes" only when the underlying KK gate fails
    assert (
        corrupted["values"]["worst_relative_residual"]
        > clean["values"]["worst_relative_residual"]
    )


def test_markdown_render_carries_the_numbers_and_the_verdict() -> None:
    w, mass, b = _synthetic_pair()
    sweep = _fake_sweep(w, mass, b)
    results = {
        "case": case.CASE_SLUG,
        "hull": {
            "form": "Wigley",
            "length_m": case.LENGTH,
            "beam_m": case.BEAM,
            "draft_m": case.DRAFT,
            "block_coefficient_analytic": case.CB_ANALYTIC,
            "displaced_volume_analytic_m3": case.VOLUME_ANALYTIC,
        },
        "settings": {
            "band_rad_s": [0.2, 8.0],
            "n_omega": int(w.size),
            "meshes": {"coarse": [12, 4]},
            "chief_points": "auto",
        },
        "gates": {
            "C1": case.geometry_gate(),
            "C3": case.kk_gate(sweep),
        },
        "verdict": {
            "passed": True,
            "gates_passed": 2,
            "gates_total": 2,
            "failed_list": [],
            "case_tolerance": case.TOLERANCES["kk"],
            "criterion_source": "tensor_case_pool id=145 (tol_policy <10%)",
        },
        "figure": None,
    }
    text = case.render_markdown(results)
    assert case.CASE_SLUG in text
    assert "Reference vs got" in text
    for row in results["gates"]["C3"]["values"]["per_dof"]:
        assert row["dof_name"] in text
    assert "CASE-BEM-CUMMINS-WIGLEY" in text
