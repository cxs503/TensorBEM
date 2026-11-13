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
    # m18249: the previous form of this test demanded a 5x separation, which the
    # data never had (measured 4.3 % -> 18.0 %, ratio 4.2).  The invariant that
    # actually matters for a negative control is that the clean pair sits inside
    # the case tolerance and the corrupted pair sits outside it.
    assert clean < case.TOLERANCES["kk"]
    assert flipped.max_relative_residual > case.TOLERANCES["kk"]
    assert flipped.max_relative_residual > 3.0 * clean


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
# --- m18249: sweep resolution and band coverage ----------------------------


def test_sweep_grid_hits_both_band_ends_and_is_geometric() -> None:
    grid = case.sweep_grid((0.2, 12.0), 41)
    assert grid.size == 41
    assert grid[0] == pytest.approx(0.2, rel=1e-15)
    assert grid[-1] == pytest.approx(12.0, rel=1e-15)
    assert np.all(np.diff(grid) > 0.0)
    ratios = np.diff(np.log(grid))
    assert np.allclose(ratios, ratios[0], rtol=1e-9)


def test_sweep_grid_rejects_a_degenerate_band_or_count() -> None:
    for band in ((0.0, 1.0), (-1.0, 5.0), (2.0, 1.0), (1.0, 1.0)):
        with pytest.raises(ValueError):
            case.sweep_grid(band, 21)
    with pytest.raises(ValueError):
        case.sweep_grid((0.2, 12.0), 2)


def test_refined_grid_targets_the_interval_with_the_largest_damping_jump() -> None:
    grid = case.sweep_grid((0.2, 12.0), 13)
    mass = np.zeros((grid.size, 6, 6))
    damper = np.zeros((grid.size, 6, 6))
    damper[:, 2, 2] = 1.0
    damper[5:, 2, 2] = 5.0  # single step, between node 4 and node 5
    refined = case._refined_grid(grid, mass, damper, n_refine=1)
    assert refined.size == grid.size + 1
    inserted = float(np.min(np.abs(refined - math.sqrt(grid[4] * grid[5]))))
    assert inserted < 1e-9
    # n_refine = 0 is the identity: the production run is unaffected by default
    assert np.array_equal(case._refined_grid(grid, mass, damper, n_refine=0), grid)


def _band_sweep(band: tuple[float, float], n: int = 41) -> "case.Sweep":
    """Synthetic sweep with a smooth, strictly positive damping curve."""
    w = case.sweep_grid(band, n)
    damper = 5.0 * w**2 / (9.0 + w**2)
    added = 50.0 * 9.0 / (9.0 + w**2)
    mass = np.zeros((w.size, 6, 6))
    damping = np.zeros((w.size, 6, 6))
    for dof in range(6):
        mass[:, dof, dof] = added
        damping[:, dof, dof] = damper
    return case.Sweep(
        label="coarse",
        n_sections=12,
        n_levels=4,
        panel_count=192,
        max_panel_scale=0.05,
        omega=w,
        added_mass=mass,
        damping=damping,
        solver_relative_residual=np.zeros(w.size),
        elapsed=0.0,
    )


def test_free_decay_gate_decides_when_the_band_covers_the_natural_frequency() -> None:
    gate = case.free_decay_gate(_band_sweep((0.2, 12.0)))
    values = gate["values"]
    assert values["natural_frequency_in_band"] is True
    assert values["status"] == "decided"
    assert math.isfinite(values["zeta_frequency_domain"])


def test_free_decay_gate_says_undecidable_outside_the_band() -> None:
    """The recorded C5 FAIL was band coverage, not a discrepancy: say so."""
    gate = case.free_decay_gate(_band_sweep((0.2, 4.0)))
    values = gate["values"]
    assert values["natural_frequency_in_band"] is False
    assert values["status"].startswith("undecidable: natural frequency outside")
    assert gate["passed"] is False
    assert math.isnan(values["zeta_frequency_domain"])


def test_production_band_covers_the_highest_natural_frequency() -> None:
    """Documented intent of BAND[1]: gate C5 must be decidable for this hull."""
    restoring, displaced_mass, _ = case._restoring_and_mass(_band_sweep((0.2, 12.0)))
    omega_n_bound = math.sqrt(restoring / displaced_mass)  # zero added mass is an upper bound
    assert omega_n_bound > 8.0  # the recorded failure: 8.15 rad/s above an 8 rad/s band
    assert case.BAND[1] >= 1.2 * omega_n_bound


def test_grid_convergence_study_reports_one_row_per_node_count(monkeypatch) -> None:
    import tensorbem.benchmarks.case_bem_cummins_wigley as module

    calls: list[int] = []

    def fake_solve(label, *, band=module.BAND, n_omega=module.N_OMEGA, jobs=1,
                   omega_grid=None, refine_peaks=0):
        calls.append(int(n_omega))
        return _band_sweep((float(band[0]), float(band[1])), int(n_omega))

    monkeypatch.setattr(module, "solve_sweep", fake_solve)
    study = module.grid_convergence_study(node_counts=(13, 21), jobs=1)
    assert calls == [13, 21]
    assert [row["n_omega"] for row in study["rows"]] == [13, 21]
    for row in study["rows"]:
        assert set(row["kk_residual_per_dof"]) == set(case.DOF_NAMES)
        assert row["kk_residual_max"] >= 0.0
    assert study["tolerance"] == case.TOLERANCES["kk"]

def test_decay_zeta_ignores_a_truncation_ripple_below_the_floor() -> None:
    """Regression (m18249): the kernel is truncated at omega_max, which leaves a
    small ripple at ~omega_max in the decay record. Fitting the decrement across
    that residue gave zeta = 0.027 for a mode whose decrement is 0.13."""
    zeta, omega_n = 0.13, 7.8
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    t = np.arange(0.0, 8.0, 0.002)
    carrier = np.exp(-zeta * omega_n * t) * np.cos(omega_d * t)
    ripple = 2.0e-3 * np.cos(2.0 * math.pi * 1.9 * t)  # ~12 rad/s truncation ripple
    record = carrier + ripple
    got, period, n_peaks = case._decay_zeta(t, record)
    assert n_peaks >= 3
    assert got == pytest.approx(zeta, rel=2e-2)
    assert period == pytest.approx(2.0 * math.pi / omega_d, rel=2e-2)
    # without the floor the same record is fitted across the ripple
    naive, _, _ = case._decay_zeta(t, record, amplitude_floor=0.0)
    assert abs(naive - zeta) > 2.0 * abs(got - zeta)


def test_decay_zeta_is_nan_when_no_peak_clears_the_floor() -> None:
    zeta, omega_n = 0.13, 7.8
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    t = np.arange(0.0, 2.0, 0.002)
    record = 2.0e-3 * np.cos(omega_d * t)  # nothing above 1 % of the excursion
    got, period, n_peaks = case._decay_zeta(t, record, amplitude_floor=0.01)
    assert got != got  # nan: no peak clears the floor (max == excursion here)
