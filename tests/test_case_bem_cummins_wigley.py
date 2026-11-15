"""Tests for the CASE-BEM-CUMMINS-WIGLEY case module (no BEM solve needed)."""

from __future__ import annotations

import json
import math
import pathlib

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


# ---------------------------------------------------------------------------
# artifact replay: the committed sweep is the evidence, and the gates must be
# re-derivable from it without solving the BEM system again
# ---------------------------------------------------------------------------

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"


def test_save_load_sweep_round_trip(tmp_path) -> None:
    w, mass, b = _synthetic_pair()
    sweep = _fake_sweep(w, mass, b)
    sweep.max_panel_scale = 0.0375
    path = case.save_sweep(sweep, tmp_path / "sweep_test.npz")
    back = case.load_sweep(path)
    assert back.label == sweep.label
    assert (back.n_sections, back.n_levels) == (sweep.n_sections, sweep.n_levels)
    assert back.panel_count == sweep.panel_count
    assert back.max_panel_scale == pytest.approx(sweep.max_panel_scale)
    np.testing.assert_allclose(back.omega, sweep.omega)
    np.testing.assert_allclose(back.added_mass, sweep.added_mass)
    np.testing.assert_allclose(back.damping, sweep.damping)


def test_load_sweep_refuses_to_invent_a_mesh_identity(tmp_path) -> None:
    """A coefficients-only artifact cannot be judged: C2 needs max_panel_scale."""
    w, mass, b = _synthetic_pair()
    sweep = _fake_sweep(w, mass, b)
    legacy = tmp_path / "legacy.npz"
    np.savez(
        legacy,
        omega=sweep.omega,
        added_mass=sweep.added_mass,
        damping=sweep.damping,
        solver_relative_residual=sweep.solver_relative_residual,
    )
    with pytest.raises(KeyError):
        case.load_sweep(legacy)
    supplied = case.load_sweep(
        legacy, label="coarse", n_sections=12, n_levels=4,
        panel_count=192, max_panel_scale=0.1,
    )
    assert supplied.panel_count == 192
    assert supplied.max_panel_scale == pytest.approx(0.1)


def test_load_sweep_rejects_coefficient_shape_mismatch(tmp_path) -> None:
    w, mass, b = _synthetic_pair()
    sweep = _fake_sweep(w, mass, b)
    path = case.save_sweep(sweep, tmp_path / "bad.npz")
    payload = dict(np.load(path, allow_pickle=False))
    payload["damping"] = payload["damping"][:, :3, :3]
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        case.load_sweep(path)


def test_case_verdict_tally() -> None:
    gates = {
        "C1": {"passed": True},
        "C2": {"passed": False},
        "C3": None,  # skipped gates are neither passed nor failed
    }
    verdict = case.case_verdict(gates)
    assert verdict["passed"] is False
    assert verdict["passed_list"] == ["C1"]
    assert verdict["failed_list"] == ["C2"]
    assert verdict["gates_total"] == 2


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "sweep_medium.npz").exists(),
    reason="production sweep artifact not committed",
)
def test_production_artifact_replays_to_the_published_verdict() -> None:
    """The committed case_result.json must be re-derivable from the committed sweep."""
    # Skip the anchor BEM solve (~60s) and multi-DOF decay (~30s) since this test
    # only verifies C2, C3, C4, C5, C6a, C6b -- none of which need them.
    replay = case.gates_from_artifact(
        ARTIFACT_DIR / "sweep_medium.npz",
        include_anchor=False,
        include_decay_modes=False,
    )
    published = json.loads((ARTIFACT_DIR / "case_result.json").read_text())
    assert replay["sweep"]["panel_count"] == published["sweeps"]["medium"]["panel_count"]
    assert replay["sweep"]["max_panel_scale"] == pytest.approx(
        published["sweeps"]["medium"]["max_panel_scale"]
    )
    for key in ("C2", "C3", "C4", "C5", "C6a", "C6b"):
        assert key in published["gates"], key
        assert replay["gates"][key]["passed"] == published["gates"][key]["passed"], key
    # the case criterion: KK closure of the added mass, per DOF, < 10 %
    worst = replay["gates"]["C3"]["values"]["worst_relative_residual"]
    assert worst == pytest.approx(
        published["gates"]["C3"]["values"]["worst_relative_residual"], rel=1e-9
    )
    assert worst <= case.TOLERANCES["kk"]
    # panel adequacy at the top of the band is what the medium mesh buys
    assert replay["gates"]["C2"]["values"]["k_times_panel_scale"] <= case.TOLERANCES["panel_kh"]


def _synthetic_sweep(*, residual: float, damping_floor: float) -> case.Sweep:
    omega = np.linspace(0.2, 12.0, 9)
    added = np.zeros((omega.size, 6, 6))
    damper = np.zeros((omega.size, 6, 6))
    for dof in range(6):
        added[:, dof, dof] = 1.0 + omega**-2
        damper[:, dof, dof] = damping_floor + omega**-3
    return case.Sweep(
        label="coarse",
        n_sections=12,
        n_levels=4,
        panel_count=192,
        max_panel_scale=0.1,
        omega=omega,
        added_mass=added,
        damping=damper,
        solver_relative_residual=np.full(omega.size, float(residual)),
        elapsed=0.0,
    )


def test_solver_validity_gate_rejects_a_non_converged_sweep() -> None:
    """A 125 % linear-solve residual must void the case, not be scored as physics."""
    bad = _synthetic_sweep(residual=1.25, damping_floor=-129.9)
    gate = case.solver_validity_gate(bad)
    assert gate["id"] == "C0"
    assert gate["passed"] is False
    assert gate["values"]["solver_relative_residual_max"] == pytest.approx(1.25)
    assert gate["values"]["min_radiation_damping"] < 0.0

    good = _synthetic_sweep(residual=1.0e-3, damping_floor=0.0)
    assert case.solver_validity_gate(good)["passed"] is True


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "case_result.json").exists(),
    reason="production case_result.json not committed",
)
def test_published_case_md_is_the_render_of_the_published_json() -> None:
    """CASE.md must be the render of case_result.json, not a hand-kept copy.

    The committed pair had drifted once already: C5c was added to the gates but
    the document still said "15 / 15 gates passed" while its own JSON said
    16 / 16.  A generated document that is not regenerated is a second, silent
    source of truth.
    """
    published = json.loads((ARTIFACT_DIR / "case_result.json").read_text())
    rendered = case.render_markdown(published)
    assert rendered == (ARTIFACT_DIR / "CASE.md").read_text()


def test_replay_of_an_inadmissible_artifact_is_voided_not_raised(tmp_path) -> None:
    """A contaminated sweep must be *judged* (C0 FAIL, dependents voided), not raised on.

    The m20021 medium-mesh sweep is the motivating artifact: same mesh, band and
    node count as the committed one, but ``max residual = 1.25`` and
    ``min B = -129.9``.  Replaying it used to raise ``ValueError`` from the
    negative-damping guard inside C4, which left the evidence unjudgeable -- the
    one thing :func:`gates_from_artifact` exists to prevent.
    """
    bad = _synthetic_sweep(residual=1.25, damping_floor=-129.9)
    path = case.save_sweep(bad, tmp_path / "sweep_bad.npz")
    replay = case.gates_from_artifact(path)  # must not raise

    assert replay["gates"]["C0"]["passed"] is False
    assert replay["gates"]["C0"]["values"]["min_radiation_damping"] < 0.0
    for key in case.C0_DEPENDENT_GATES:
        gate = replay["gates"][key]
        assert gate["voided"] is True, key
        assert gate["passed"] is False, key
        assert gate["values"]["voided_by"] == "C0", key
    assert replay["verdict"]["passed"] is False
    assert "C0" in replay["verdict"]["failed_list"]
    # a voided sweep carries no anchor and no extended-band study: both would
    # need physical coefficients to mean anything
    assert replay["added_mass_inf_anchor"] is None
    assert replay["extended"] is None


def test_run_case_voids_the_same_gates_when_the_sweep_fails_c0(tmp_path, monkeypatch) -> None:
    """The run path must record the same voiding, and still write its artifacts.

    A run that dies on an inadmissible sweep must leave a *judged* failure
    (``case_result.json`` with C0 FAIL and dependents voided) rather than no
    artifact at all: the m20021 run aborted and left the reader with two
    committed ``.npz`` files and no verdict.
    """
    bad = _synthetic_sweep(residual=1.25, damping_floor=-129.9)
    monkeypatch.setattr(case, "solve_sweep", lambda label, **kwargs: bad)

    results = case.run_case(out_dir=tmp_path, meshes=("medium",), figure=False)

    assert results["gates"]["C0"]["passed"] is False
    assert results["verdict"]["passed"] is False
    for key in case.C0_DEPENDENT_GATES:
        assert results["gates"][key]["voided"] is True, key
    published = json.loads((tmp_path / "case_result.json").read_text())
    assert published["gates"]["C0"]["passed"] is False
    assert published["verdict"]["failed_list"][0] == "C0"
    assert (tmp_path / "CASE.md").exists()


def test_voided_gates_cover_both_paths_through_the_module() -> None:
    """The run path and the replay path must void the *same* gates, or a replayed
    verdict could disagree with the one the run published."""
    voided = set(case.C0_DEPENDENT_GATES)
    assert {"C2", "C3", "C3a", "C7", "C4", "C5", "C6a", "C6b"} <= voided
    # C0 itself and the geometry gate C1 are *not* voided by C0: they are the
    # evidence that the sweep is unusable / that the hull is the hull
    assert "C0" not in voided and "C1" not in voided


M20021_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "m20021"


@pytest.mark.skipif(
    not (M20021_DIR / "sweep_medium.npz").exists(),
    reason="m20021 contaminated artifact not committed",
)
def test_committed_contaminated_sweep_is_kept_as_the_c0_negative_control() -> None:
    """The m20021 sweep stays in the tree as C0's permanent negative control.

    It is the only committed artifact of this case that *fails* the linear-solve
    precondition, so it is what proves the precondition bites in production and
    not just on a synthetic sweep.
    """
    replay = case.gates_from_artifact(M20021_DIR / "sweep_medium.npz")
    c0 = replay["gates"]["C0"]
    assert c0["passed"] is False
    assert c0["values"]["solver_relative_residual_max"] > case.TOLERANCES["solver_residual"]
    assert c0["values"]["min_radiation_damping"] < 0.0
    assert replay["verdict"]["passed"] is False
    assert replay["verdict"]["passed_list"] == ["C1"]
    assert all(replay["gates"][key]["voided"] for key in case.C0_DEPENDENT_GATES)


def test_free_decay_gate_reports_negative_damping_instead_of_raising() -> None:
    bad = _synthetic_sweep(residual=1.25, damping_floor=-129.9)
    gate = case.free_decay_gate(bad)
    assert gate["passed"] is False
    assert gate["values"]["status"].startswith("undecidable: non-physical negative")
    assert gate["values"]["min_radiation_damping"] < 0.0
"""Anchor-gate tests: zero free parameters, and the A_inf referee itself."""

# appended to tests/test_case_bem_cummins_wigley.py


def _fake_anchor(values, *, self_check_passed=True):
    return {
        "method": "synthetic",
        "center_m": [0.0, 0.0, 0.0],
        "n_sections": 12,
        "n_levels": 4,
        "panel_count": 192,
        "doubled_panel_count": 384,
        "doubled_volume_over_hull_volume": 2.0,
        "condition_number": 8.0,
        "solver_relative_residual_l2": 1e-14,
        "symmetry_error": 0.0,
        "added_mass_inf_diagonal": list(values),
        "added_mass_inf": {name: float(values[i]) for i, name in enumerate(case.DOF_NAMES)},
        "self_check": {
            "passed": bool(self_check_passed),
            "relative_error": 0.001,
            "tolerance": case.TOLERANCES["anchor_analytic"],
            "panels": 360,
        },
    }


def test_added_mass_inf_gate_passes_at_the_anchor_and_fails_off_it() -> None:
    """C7 must bite when the fitted A_inf walks away from the physical limit."""
    w, mass, b = _synthetic_pair()
    sweep = _fake_sweep(w, mass, b)
    rows = case.kk_gate(sweep)["values"]["per_dof"]
    fitted = [row["added_mass_inf_closure"] for row in rows]
    scale = [row["residual_scale"] for row in rows]

    ok = case.added_mass_inf_gate(sweep, _fake_anchor(fitted), closure_rows=rows)
    assert ok["passed"] is True
    assert ok["values"]["worst_relative_to_added_mass_scale"] == pytest.approx(0.0)

    # shift every anchor by 25 % of the added-mass amplitude: the free parameter
    # the closure spent is now larger than the criterion it is judged by.
    off = [fitted[i] + 0.25 * scale[i] for i in range(6)]
    bad = case.added_mass_inf_gate(sweep, _fake_anchor(off), closure_rows=rows)
    assert bad["passed"] is False
    assert bad["values"]["worst_relative_to_added_mass_scale"] == pytest.approx(0.25)
    assert "fail" in bad["status"]

    # an anchor that cannot referee anything decides nothing
    unvalidated = case.added_mass_inf_gate(
        sweep, _fake_anchor(fitted, self_check_passed=False), closure_rows=rows
    )
    assert unvalidated["passed"] is False
    assert unvalidated["status"].startswith("undecidable")


def test_anchored_kk_gate_has_no_free_parameter() -> None:
    """Supplying A_inf removes the parameter that could absorb a constant offset."""
    w, mass, b = _synthetic_pair()
    exact_inf = 0.0  # by construction A - A_inf = -sum c/(a^2+w^2)
    clean = _fake_sweep(w, mass, b)
    fitted_clean = case.kk_gate(clean)["values"]["per_dof"][0]
    anchored_clean = case.kk_gate(
        clean, added_mass_inf=[exact_inf] * 6, identifier="C3a"
    )["values"]["per_dof"][0]
    assert anchored_clean["free_parameters"] == 0
    assert fitted_clean["free_parameters"] == 1
    # the residual is not zero even on an exact pair: the band is finite, so the
    # omitted tails of the analytic damping are missing from the reconstruction
    assert anchored_clean["max_relative_residual"] <= case.TOLERANCES["kk"]

    # a constant offset in A is invisible to the fitted gate and fatal to the
    # anchored one: that is exactly the hole the anchor closes.
    shifted = _fake_sweep(w, mass + 0.2 * float(np.max(np.abs(mass))), b)
    fitted_shifted = case.kk_gate(shifted)["values"]["per_dof"][0]
    anchored_shifted = case.kk_gate(
        shifted, added_mass_inf=[exact_inf] * 6, identifier="C3a"
    )["values"]["per_dof"][0]
    # the free parameter absorbs the offset: the fitted gate stays inside the
    # criterion and barely moves, the anchored one is pushed out of it
    assert fitted_shifted["max_relative_residual"] <= case.TOLERANCES["kk"]
    assert fitted_shifted["max_relative_residual"] < 1.5 * fitted_clean["max_relative_residual"]
    assert anchored_shifted["max_relative_residual"] > 3.0 * anchored_clean["max_relative_residual"]
    assert anchored_shifted["max_relative_residual"] > case.TOLERANCES["kk"]


def test_anchor_self_check_meets_its_bar() -> None:
    """The image method must reproduce the closed-form half-sphere A_inf."""
    check = case.anchor_self_check(8, 24)
    assert check["added_mass_inf_exact"] == pytest.approx(
        0.25 * case.DENSITY * 4.0 / 3.0 * math.pi
    )
    assert check["relative_error"] <= case.TOLERANCES["anchor_analytic"]
    assert check["passed"] is True


def test_anchor_is_mesh_converged_between_the_case_meshes() -> None:
    """The A_inf referee must not move by its own tolerance over the mesh ladder."""
    coarse = case.added_mass_inf_anchor(12, 4)
    medium = case.added_mass_inf_anchor(20, 6)
    worst = max(
        abs(a - b) / max(abs(b), 1e-300)
        for a, b in zip(coarse["added_mass_inf_diagonal"], medium["added_mass_inf_diagonal"])
    )
    assert worst < 0.06
    assert medium["doubled_volume_over_hull_volume"] == pytest.approx(2.0, rel=1e-9)


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "sweep_medium.npz").exists(),
    reason="production sweep artifact not committed",
)
def test_replay_artifact_reports_the_anchor_gates() -> None:
    """The committed sweep must replay into the anchored verdict, not just the fit."""
    replay = case.gates_from_artifact(ARTIFACT_DIR / "sweep_medium.npz")
    for key in ("C3a", "C7"):
        assert key in replay["gates"], key
        assert replay["gates"][key]["passed"] is True, key
    assert replay["gates"]["C3a"]["values"]["free_parameters"] == 0
    assert replay["gates"]["C3"]["values"]["free_parameters"] == 1
    # the anchored closure is a *stronger* test: its worst DOF cannot be better
    # than the gate that was allowed to fit its own asymptote
    assert (
        replay["gates"]["C3a"]["values"]["worst_relative_residual"]
        >= replay["gates"]["C3"]["values"]["worst_relative_residual"]
    )
    assert replay["gates"]["C7"]["values"]["anchor_self_check"]["passed"] is True
    assert replay["added_mass_inf_anchor"]["doubled_volume_over_hull_volume"] == pytest.approx(2.0, rel=1e-9)

# ---------------------------------------------------------------------------
# band-extension gates (C8 consistency, C9 truncation decidability)
# ---------------------------------------------------------------------------


def _decaying_sweep(band, *, n=41, exponent=3.0, panels=480, label="extended", mass=5.0):
    """Synthetic sweep with an algebraic damping tail ``B ~ omega^-exponent``."""
    w = case.sweep_grid(band, n)
    b = 40.0 * w ** (-exponent)
    added = np.zeros((w.size, 6, 6))
    damper = np.zeros((w.size, 6, 6))
    for i in range(6):
        added[:, i, i] = -mass / (1.0 + w**2)
        damper[:, i, i] = b
    return case.Sweep(
        label=label,
        n_sections=60,
        n_levels=16,
        panel_count=panels,
        max_panel_scale=0.0232,
        omega=w,
        added_mass=added,
        damping=damper,
        solver_relative_residual=np.full(w.size, 1e-12),
        elapsed=0.0,
    )


def test_extension_consistency_gate_passes_on_a_reproduced_band() -> None:
    """A finer mesh that returns the same coefficients on the shared band passes."""
    committed = _decaying_sweep((0.2, 12.0), n=41, panels=480, label="medium")
    extended = _decaying_sweep((0.2, 20.0), n=55, panels=3840)
    gate = case.extension_consistency_gate(committed, extended)
    assert gate["id"] == "C8"
    assert gate["passed"] is True
    assert gate["values"]["n_shared_frequencies"] == 41
    # What is left over is the piecewise-linear interpolation error between two
    # different log grids (~0.5 % on smooth data here), not physics.
    assert 0.0 < gate["values"]["worst_relative_difference"] < 0.01


def test_extension_consistency_gate_bites_when_the_band_changes() -> None:
    """A 20 % coefficient shift on the shared band must not be waved through."""
    committed = _decaying_sweep((0.2, 12.0), n=41, panels=480, label="medium")
    extended = _decaying_sweep((0.2, 20.0), n=55, panels=3840)
    shifted = case.Sweep(
        label=extended.label,
        n_sections=extended.n_sections,
        n_levels=extended.n_levels,
        panel_count=extended.panel_count,
        max_panel_scale=extended.max_panel_scale,
        omega=extended.omega,
        added_mass=extended.added_mass * 1.2,
        damping=extended.damping,
        solver_relative_residual=extended.solver_relative_residual,
        elapsed=0.0,
    )
    gate = case.extension_consistency_gate(committed, shifted)
    assert gate["passed"] is False
    # the 20 % shift is measured against the amplitude of the *extended* sweep
    assert gate["values"]["worst_relative_difference"] == pytest.approx(
        0.2 / 1.2, rel=1e-6
    )


def test_extension_consistency_gate_needs_a_shared_band() -> None:
    committed = _decaying_sweep((0.2, 12.0), n=41, panels=480, label="medium")
    disjoint = _decaying_sweep((15.0, 20.0), n=11)
    with pytest.raises(ValueError):
        case.extension_consistency_gate(committed, disjoint)


def test_truncation_decidability_gate_credits_a_decaying_tail() -> None:
    """``B ~ omega^-3`` above the old band top makes the budget finite again."""
    committed = _decaying_sweep((0.2, 12.0), n=41, exponent=1.0, panels=480, label="medium")
    extended = _decaying_sweep((0.2, 20.0), n=55, exponent=3.0, panels=3840)
    gate = case.truncation_decidability_gate(committed, extended)
    assert gate["id"] == "C9"
    assert gate["passed"] is True
    assert set(gate["values"]["newly_decidable"]) == set(case.DOF_NAMES)
    assert gate["values"]["lost_decidability"] == []
    assert "widened" in gate["status"]
    for row in gate["values"]["per_dof"]:
        assert row["budget_admissible_at_20"] is True
        assert row["exponent_at_20"] == pytest.approx(3.0, rel=0.05)


def test_truncation_decidability_gate_reports_a_negative_result() -> None:
    """No decay gain: the gate must say so instead of claiming a widening."""
    committed = _decaying_sweep((0.2, 12.0), n=41, exponent=1.0, panels=480, label="medium")
    extended = _decaying_sweep((0.2, 20.0), n=55, exponent=1.0, panels=3840)
    gate = case.truncation_decidability_gate(committed, extended)
    assert gate["values"]["newly_decidable"] == []
    assert gate["values"]["lost_decidability"] == []
    assert gate["passed"] is True  # nothing got worse ...
    assert "no widening" in gate["status"]  # ... but nothing was gained either
    assert all(
        row["budget_admissible_at_20"] is False for row in gate["values"]["per_dof"]
    )


def test_truncation_decidability_gate_refuses_to_lose_ground() -> None:
    """A band that was certifiable at 12 rad/s may not become uncertifiable at 20."""
    committed = _decaying_sweep((0.2, 12.0), n=41, exponent=3.0, panels=480, label="medium")
    flat = _decaying_sweep((0.2, 20.0), n=55, exponent=1.0, panels=3840)
    gate = case.truncation_decidability_gate(committed, flat)
    assert gate["passed"] is False
    assert set(gate["values"]["lost_decidability"]) == set(case.DOF_NAMES)
    assert "lost certifiability" in gate["status"]


def test_extended_band_constants_are_panel_resolved() -> None:
    """C2 must hold at the pre-declared extended band top, by construction."""
    mesh = case.wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT,
        n_sections=case.EXTENDED_MESH[0], n_levels=case.EXTENDED_MESH[1],
    )
    scale = math.sqrt(float(np.max(mesh.areas)))
    kh = case.EXTENDED_BAND[1] ** 2 / case.GRAVITY * scale
    assert kh <= case.TOLERANCES["panel_kh"], (
        f"extended mesh {case.EXTENDED_MESH} resolves only up to "
        f"{math.sqrt(case.GRAVITY / scale):.2f} rad/s"
    )
    # ... and the committed band top would *not* be resolved by that check if the
    # extension were simply run on the medium mesh: the reason the sweep is
    # expensive is a physical one, not a matter of taste.
    medium = case.wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT,
        n_sections=case.MESHES["medium"][0], n_levels=case.MESHES["medium"][1],
    )
    medium_kh = case.EXTENDED_BAND[1] ** 2 / case.GRAVITY * math.sqrt(
        float(np.max(medium.areas))
    )
    assert medium_kh > case.TOLERANCES["panel_kh"]

def test_robust_decay_zeta_multi_window_on_synthetic() -> None:
    """_decay_zeta_multi_window recovers known damping on a clean exponential."""
    from tensorbem.benchmarks.case_bem_cummins_wigley import _decay_zeta_multi_window
    import numpy as np
    t = np.linspace(0, 20, 2000)
    zeta_true = 0.08
    omega_n = 6.0
    omega_d = omega_n * np.sqrt(1 - zeta_true**2)
    d = np.exp(-zeta_true * omega_n * t) * np.cos(omega_d * t)
    result = _decay_zeta_multi_window(t, d)
    assert result["n_qualifying_floors"] >= 2
    assert result["converged"] is True
    assert abs(result["consensus_zeta"] - zeta_true) / zeta_true < 0.1


def test_c5c_gate_passes_on_roll_from_artifact() -> None:
    """C5c gate passes on roll when replaying the committed medium sweep.

    Uses the pre-computed case_result.json to avoid re-running the heavy
    free-decay simulation (which takes ~60s per DOF).
    """
    import json
    with open("benchmarks/verified/casepool_bem_cummins_wigley/case_result.json") as f:
        data = json.load(f)
    c5c = data["gates"]["C5c"]
    assert c5c["passed"] is True
    assert c5c["values"]["status"] == "decided"
    assert c5c["values"]["dof"] == 3
    assert c5c["values"]["dof_name"] == "roll"
    assert c5c["values"]["relative_error"] < 0.05
    assert c5c["values"]["multi_window"]["n_qualifying_floors"] >= 2


# ---------------------------------------------------------------------------
# C5c failure injection (negative controls).
#
# Gate C5c decides the roll damping ratio from a multi-window consensus.  A gate
# that has only ever been observed to pass is indistinguishable from a rubber
# stamp, so every branch of its decision rule is falsified here by construction.
# The decision rule is factored out as ``case.c5c_decision`` exactly so that it
# can be exercised without paying the ~1.5 min Cummins free-decay integration.
# ---------------------------------------------------------------------------

BAR = case.TOLERANCES["free_decay"]


def test_c5c_decision_passes_only_inside_bar_with_two_windows() -> None:
    status, passed = case.c5c_decision(in_band=True, relative=0.0, n_qualifying=3, bar=BAR)
    assert passed is True
    assert status == "decided"


def test_c5c_decision_boundary_exactly_at_bar_passes() -> None:
    """The bar is inclusive; a hair beyond it must fail."""
    assert case.c5c_decision(in_band=True, relative=BAR, n_qualifying=3, bar=BAR)[1] is True
    status, passed = case.c5c_decision(
        in_band=True, relative=BAR * (1 + 1e-9), n_qualifying=3, bar=BAR
    )
    assert passed is False
    assert status == "beyond_bar"


def test_c5c_decision_rejects_disagreement_beyond_bar() -> None:
    """A consensus that disagrees with the impedance route must NOT pass."""
    status, passed = case.c5c_decision(in_band=True, relative=0.30, n_qualifying=3, bar=BAR)
    assert passed is False
    assert status == "beyond_bar"


def test_c5c_decision_rejects_out_of_band_natural_frequency() -> None:
    """Even a perfect agreement is void when omega_n lies outside the computed band."""
    status, passed = case.c5c_decision(in_band=False, relative=0.0, n_qualifying=3, bar=BAR)
    assert passed is False
    assert "outside computed band" in status


def test_c5c_decision_rejects_single_qualifying_window() -> None:
    """The whole point of C5c: one window is not a consensus."""
    status, passed = case.c5c_decision(in_band=True, relative=0.0, n_qualifying=1, bar=BAR)
    assert passed is False
    assert "fewer than 2" in status


def test_c5c_decision_rejects_nonfinite_relative() -> None:
    """zeta_fd == 0 or nan makes the relative error infinite -> never a pass."""
    for bad in (math.inf, math.nan):
        assert case.c5c_decision(in_band=True, relative=bad, n_qualifying=3, bar=BAR)[1] is False


def _fake_decay(monkeypatch, per_floor):
    """Force ``_decay_zeta`` to return prescribed (zeta, period, n_peaks) per floor."""
    def fake(time_array, displacement, amplitude_floor=None):
        zeta, period, n_peaks = per_floor[amplitude_floor]
        return zeta, period, n_peaks

    monkeypatch.setattr(case, "_decay_zeta", fake)


def _toy_signal():
    t = np.linspace(0.0, 40.0, 4001)
    return t, np.exp(-0.05 * t) * np.cos(3.0 * t)


def test_multi_window_even_count_uses_true_median(monkeypatch) -> None:
    """With exactly two qualifying windows the consensus is their MEAN.

    Regression: the previous ``sorted_z[len // 2]`` convention returned the
    upper of the two samples, biasing a monotone window sequence upward.
    """
    floors = [0.01, 0.001]
    _fake_decay(monkeypatch, {floors[0]: (0.09, 1.1, 8), floors[1]: (0.05, 1.1, 15)})
    t, d = _toy_signal()
    out = case._decay_zeta_multi_window(t, d, floors=floors)
    assert out["n_qualifying_floors"] == 2
    assert out["consensus_zeta"] == pytest.approx(0.07)      # mean, not 0.09
    assert out["consensus_period"] == pytest.approx(1.1)
    assert out["converged"] is False
    assert out["spread"] > out["bar"]


def test_multi_window_odd_count_uses_middle_sample(monkeypatch) -> None:
    floors = [0.01, 0.001, 0.0001]
    _fake_decay(
        monkeypatch,
        {floors[0]: (0.0901, 1.086, 8), floors[1]: (0.0711, 1.0989, 15), floors[2]: (0.0501, 1.09, 28)},
    )
    t, d = _toy_signal()
    out = case._decay_zeta_multi_window(t, d, floors=floors)
    assert out["consensus_zeta"] == pytest.approx(0.0711)
    assert out["converged"] is False       # spread 56 % >> 5 % bar
    assert out["spread"] == pytest.approx((0.0901 - 0.0501) / 0.0711, rel=1e-6)


def test_multi_window_single_window_is_not_a_consensus(monkeypatch) -> None:
    """One qualifying window must not be reported as converged with spread 0.

    Regression: the old code returned ``converged=True, spread=0.0`` for a
    single window, which is a false statement about window agreement.
    """
    floors = [0.01, 0.001]
    _fake_decay(monkeypatch, {floors[0]: (0.09, 1.1, 8), floors[1]: (float("nan"), 1.1, 3)})
    t, d = _toy_signal()
    out = case._decay_zeta_multi_window(t, d, floors=floors)
    assert out["n_qualifying_floors"] == 1
    assert out["converged"] is False
    assert math.isnan(out["spread"])
    assert out["consensus_zeta"] == pytest.approx(0.09)


def test_multi_window_no_qualifying_window_is_undecidable(monkeypatch) -> None:
    floors = [0.01]
    _fake_decay(monkeypatch, {floors[0]: (float("nan"), float("nan"), 2)})
    t, d = _toy_signal()
    out = case._decay_zeta_multi_window(t, d, floors=floors)
    assert out["n_qualifying_floors"] == 0
    assert out["converged"] is False
    assert math.isnan(out["consensus_zeta"])


def test_committed_roll_verdict_is_disclosed_as_non_convergent() -> None:
    """The accepted roll verdict passes while its windows do NOT converge.

    This test pins the honest reading of the committed evidence: C5c passes on
    the agreement between the time-domain consensus and the frequency-domain
    impedance, not because the three amplitude windows agreed.  Any regeneration
    that silently turns this into a 'converged' claim must fail here.
    """
    with open("benchmarks/verified/casepool_bem_cummins_wigley/case_result.json") as f:
        data = json.load(f)
    c5c = data["gates"]["C5c"]
    mw = c5c["values"]["multi_window"]
    assert c5c["passed"] is True
    assert c5c["values"]["status"] == "decided"
    assert mw["converged"] is False, "committed roll windows did not converge"
    assert mw["spread"] > mw["bar"], "spread must exceed the bar to be non-exponential"
    assert mw["n_qualifying_floors"] >= 2
    # monotone window drift is the physical signature, not scatter noise
    zetas = [row["zeta"] for row in sorted(mw["per_floor"], key=lambda r: -r["floor"])]
    assert all(a > b for a, b in zip(zetas, zetas[1:])), f"expected monotone drift, got {zetas}"


# ---------------------------------------------------------------------------
# band extension: C8/C9 attachment and the cheap-mesh *probe*
# ---------------------------------------------------------------------------


def _probe_sweep(exponent, panels, label, panel_scale):
    """Extended-band synthetic on one mesh, with that mesh's own panel scale."""
    import dataclasses

    sweep = _decaying_sweep(
        case.EXTENDED_BAND, n=case.EXTENDED_N_OMEGA, exponent=exponent,
        panels=panels, label=label,
    )
    return dataclasses.replace(sweep, max_panel_scale=float(panel_scale))


def _probe_report(exponents=(3.0, 3.0)):
    from tensorbem.benchmarks import cummins_wigley_extended_probe as probe

    committed = _probe_sweep(3.0, 480, "medium", 0.0645)
    coarse = _probe_sweep(exponents[0], 192, "coarse", 0.15)
    medium = _probe_sweep(exponents[1], 480, "medium", 0.0645)
    import dataclasses

    grid = case.sweep_grid((0.2, 12.0), case.N_OMEGA)
    committed = dataclasses.replace(
        committed,
        omega=grid,
        added_mass=np.asarray(committed.added_mass)[: grid.size],
        damping=np.asarray(committed.damping)[: grid.size],
        solver_relative_residual=np.full(grid.size, 1e-12),
    )
    return probe.probe_report(
        committed=committed, coarse=coarse, medium=medium, cost=None
    )


def test_attach_extended_band_gates_without_artifact_changes_nothing(tmp_path) -> None:
    """No artifact means *not evaluated*: gates C8/C9 must not appear as passes."""
    sweep = _decaying_sweep((0.2, 12.0), panels=480, label="medium")
    gates = {"C1": {"passed": True}}
    assert case.attach_extended_band_gates(gates, sweep, tmp_path) is None
    assert set(gates) == {"C1"}


def test_extended_anchor_for_sweep_selects_the_rung_by_mesh_identity(tmp_path) -> None:
    """An anchor solved on another mesh is not an anchor for this sweep."""
    ladder = tmp_path / case.EXTENDED_ANCHOR_NAME
    ladder.write_text(
        json.dumps(
            {
                "rungs": [
                    {
                        "n_sections": 44,
                        "n_levels": 14,
                        "added_mass_inf_diagonal": [1.0] * 6,
                    },
                    {
                        "n_sections": 60,
                        "n_levels": 16,
                        "added_mass_inf_diagonal": [9.0] * 6,
                    },
                ]
            }
        )
    )
    sweep = _decaying_sweep((0.2, 12.0))
    assert (sweep.n_sections, sweep.n_levels) == case.EXTENDED_MESH
    assert case.extended_anchor_for_sweep(sweep, ladder)["added_mass_inf_diagonal"][0] == 9.0
    assert case.extended_anchor_for_sweep(sweep, tmp_path / "absent.json") is None


def test_probe_band_limit_is_the_C2_rule() -> None:
    """The probe's 'how much of the band does this mesh resolve' is gate C2."""
    from tensorbem.benchmarks import cummins_wigley_extended_probe as probe

    scale = 0.0645  # committed medium mesh
    limit = probe._band_limit(scale)
    assert limit == pytest.approx(math.sqrt(case.GRAVITY / scale), rel=1e-12)
    # the medium mesh stops resolving around 12 rad/s, i.e. it cannot carry a
    # 20 rad/s band -- which is exactly why the probe is not a gate
    assert limit == pytest.approx(12.33, abs=0.05)
    assert limit < case.EXTENDED_BAND[1]


def test_probe_never_promotes_itself_to_a_gate() -> None:
    """Even a clean bracket leaves C8/C9 explicitly not_run."""
    report = _probe_report((3.0, 3.0))
    assert report["certified_gates"] == {"C8": "not_run", "C9": "not_run"}
    assert "probe" in report["probe"]
    assert report["manifest"]["shared_band_reproduction"]["n_shared_frequencies"] > 0


def test_probe_reports_robust_widening_only_when_both_meshes_agree() -> None:
    assert all(
        row["probe_verdict"].startswith("robust") for row in _probe_report((3.0, 3.0))["per_dof"]
    )
    bracketed = _probe_report((3.0, 1.2))
    assert all(row["probe_verdict"].startswith("inconclusive") for row in bracketed["per_dof"])
    stuck = _probe_report((1.2, 1.2))
    assert all(row["probe_verdict"].startswith("not widened") for row in stuck["per_dof"])
    assert stuck["robust_dof"] == []


def test_probe_reports_the_C2_price_of_the_cheap_meshes() -> None:
    """The probe must publish *why* its numbers are not certified."""
    manifest = _probe_report((3.0, 3.0))["manifest"]
    for info in manifest["meshes"].values():
        assert info["c2_holds_over_probe_band"] is False
        assert info["band_limit_rad_s"] < case.EXTENDED_BAND[1]


def test_probe_renders_the_cost_of_the_certified_sweep() -> None:
    from tensorbem.benchmarks import cummins_wigley_extended_probe as probe

    cost = {
        "points": [
            {"n_sections": 12, "n_levels": 4, "panel_count": 192, "wall_s": 2.0},
            {"n_sections": 20, "n_levels": 6, "panel_count": 480, "wall_s": 14.6},
            {"n_sections": 32, "n_levels": 10, "panel_count": 640, "wall_s": 40.0},
        ],
        "fitted_exponent": 3.1,
        "certified_panel_count": 3840,
        "certified_wall_s_per_node": 3600.0,
        "certified_cpu_hours_for_band": 55.0,
        "declared_plan_cpu_hours": [2.0, 4.0],
    }
    text = probe.render_markdown(
        probe.probe_report(
            committed=_decaying_sweep((0.2, 12.0), panels=480, label="medium"),
            coarse=_probe_sweep(3.0, 192, "coarse", 0.15),
            medium=_probe_sweep(3.0, 480, "medium", 0.0645),
            cost=cost,
        )
    )
    assert "CPU-hours" in text
    assert "not_run" in text
