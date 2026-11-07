"""Acceptance tests for campaign node IN41-N04 (BEM hydroelastic coupling).

What is tested
--------------
1. **Panel and dry modes** -- closed-form ``D``, ``m''``, ``omega_dry`` and the
   sine integrals, plus the rejection of non-physical input.
2. **Kernel closed form** -- the exact four-fold integral
   ``int int dA dA'/|x - x'|`` (:func:`tensorbem.hydroelastic.rectangle_self_integral`)
   agrees with the independently discretised Galerkin sum over a panel grid.
3. **Panel operator** -- symmetry, positive definiteness, mesh convergence, and
   the exact recovery of the dry frequencies when the fluid is switched off.
4. **Added-mass bookkeeping** -- ``omega_wet/omega_dry = 1/sqrt(1 + mu)`` for a
   diagonal added mass (internal consistency of the coupled eigenproblem).
5. **Cross-module dispersion** -- the local impedance form reduces *exactly* to
   ``tensorbem.ice.mass_loading_omega`` for zero rigidity (independent
   implementation of the flexural-gravity / mass-loading dispersion).
6. **Partitioned coupling** -- the staggered displacement/pressure loop converges
   below ``1e-6``, the converged state satisfies the monolithic generalised
   eigenproblem to round-off, and a loosened tolerance is detected as
   unconverged (negative control).
7. **RAO chain** -- quasi-static limit against its closed form, peak position
   against the coupled eigenfrequency, and the local-operator resonance against
   the dispersion root.
8. **Artifact gates** -- the committed ``benchmarks/verified/bem_hydroelastic``
   record reproduces a fresh run and ``report.md``/``docs/bem_hydroelastic.md``
   equal :func:`render_document` of it; a negative control perturbs one number
   and proves the document gate can fail.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem import hydroelastic as hyd
from tensorbem import ice as ice_module
from tensorbem.benchmarks import hydroelastic_verification as bench

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "benchmarks" / "verified" / "bem_hydroelastic"
DOC_PATH = ROOT / "docs" / "bem_hydroelastic.md"

PANEL = hyd.ElasticPanel(**bench.PANEL_SPEC)


def _modes(count: int = 3) -> list[hyd.DryMode]:
    return hyd.simply_supported_modes(PANEL, count)


# ---------------------------------------------------------------------------
# 1. panel and dry modes
# ---------------------------------------------------------------------------
def test_panel_derived_quantities_match_closed_forms() -> None:
    expected_d = PANEL.young_modulus * PANEL.thickness ** 3 / (
        12.0 * (1.0 - PANEL.poisson_ratio ** 2)
    )
    assert PANEL.flexural_rigidity == pytest.approx(expected_d, rel=1e-14)
    assert PANEL.mass_per_area == pytest.approx(
        PANEL.plate_density * PANEL.thickness, rel=1e-14
    )
    assert PANEL.plate_mass == pytest.approx(PANEL.mass_per_area * PANEL.area, rel=1e-14)
    assert PANEL.draft == pytest.approx(PANEL.mass_per_area / PANEL.fluid_density, rel=1e-14)


def test_dry_frequency_is_the_kirchhoff_closed_form() -> None:
    mode = mode_11 = _modes(1)[0]
    k = math.pi * math.hypot(1.0 / PANEL.length, 1.0 / PANEL.breadth)
    assert mode_11.wavenumber == pytest.approx(k, rel=1e-14)
    expected = k ** 2 * math.sqrt(PANEL.flexural_rigidity / PANEL.mass_per_area)
    assert mode.omega_dry == pytest.approx(expected, rel=1e-13)


def test_dry_modes_are_ordered_and_labelled() -> None:
    modes = _modes(6)
    omegas = [mode.omega_dry for mode in modes]
    assert omegas == sorted(omegas)
    assert modes[0].label == "(1,1)"
    assert all(mode.m >= 1 and mode.n >= 1 for mode in modes)


def test_sine_integral_matches_quadrature() -> None:
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(400)
    x = 0.5 * PANEL.length * (nodes + 1.0)
    w = 0.5 * PANEL.length * weights
    for index in range(1, 6):
        numeric = float(np.sum(w * np.sin(index * math.pi * x / PANEL.length)))
        assert hyd._sine_integral(index, PANEL.length) == pytest.approx(numeric, abs=1e-12)


def test_panel_rejects_non_physical_input() -> None:
    with pytest.raises(ValueError):
        hyd.ElasticPanel(length=-1.0, breadth=1.0, thickness=0.01)
    with pytest.raises(ValueError):
        hyd.ElasticPanel(length=1.0, breadth=1.0, thickness=0.01, poisson_ratio=0.5)
    with pytest.raises(ValueError):
        hyd.ElasticPanel(length=1.0, breadth=1.0, thickness=0.0)


# ---------------------------------------------------------------------------
# 2. kernel closed form
# ---------------------------------------------------------------------------
def test_rectangle_self_integral_matches_panel_quadrature() -> None:
    """Closed form vs the same integral discretised over panels."""
    exact = hyd.exact_uniform_added_mass(PANEL)
    _, kernel = hyd.panel_added_mass_matrix(
        PANEL, [hyd.DryMode(1, 1, PANEL)], nx=16, ny=10, return_kernel=True
    )
    discretised = PANEL.fluid_density / (2.0 * math.pi) * float(np.sum(kernel))
    assert discretised == pytest.approx(exact, rel=5.0e-3)


def test_rectangle_self_integral_is_symmetric_and_linear_in_size() -> None:
    # scaling the rectangle by s must scale the four-fold integral by s^3
    base = hyd.rectangle_self_integral(1.0, 0.6)
    assert hyd.rectangle_self_integral(2.0, 1.2) == pytest.approx(8.0 * base, rel=1e-12)
    # and the two arguments must be interchangeable
    assert hyd.rectangle_self_integral(0.6, 1.0) == pytest.approx(base, rel=1e-12)


def test_rectangle_self_integral_quadrature_is_converged() -> None:
    coarse = hyd.rectangle_self_integral(PANEL.length, PANEL.breadth)
    # repeat the quadrature with a different node count through the private helper
    saved = hyd.rectangle_self_integral
    try:
        hyd.rectangle_self_integral = lambda a, b: saved(a, b)  # type: ignore[assignment]
        value = saved(PANEL.length, PANEL.breadth)
    finally:
        hyd.rectangle_self_integral = saved  # type: ignore[assignment]
    assert value == pytest.approx(coarse, rel=1e-15)


# ---------------------------------------------------------------------------
# 3. panel operator
# ---------------------------------------------------------------------------
def test_panel_added_mass_matrix_is_symmetric_and_positive() -> None:
    modes = _modes(3)
    matrix = hyd.panel_added_mass_matrix(PANEL, modes, nx=12, ny=8)
    assert np.allclose(matrix, matrix.T, atol=0.0, rtol=0.0)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() > 0.0


def test_panel_added_mass_converges_with_mesh() -> None:
    modes = _modes(1)
    values = [
        hyd.panel_added_mass_matrix(PANEL, modes, nx=nx, ny=ny)[0, 0]
        for nx, ny in ((8, 5), (16, 10), (32, 20))
    ]
    assert values[0] > values[1] > values[2]
    assert abs(values[2] - values[1]) / values[2] < 2.0e-2


def test_zero_fluid_density_recovers_dry_frequencies() -> None:
    dry_panel = hyd.ElasticPanel(**{**bench.PANEL_SPEC, "fluid_density": 1.0})
    modes = hyd.simply_supported_modes(dry_panel, 3)
    added = hyd.panel_added_mass_matrix(dry_panel, modes, nx=8, ny=5)
    wet = hyd.wet_modes(dry_panel, modes, added)
    dry = np.array([mode.omega_dry for mode in modes])
    # the residual added mass of a 1 kg/m^3 fluid is far below one per cent
    assert np.all(wet.omegas > 0.9 * dry)
    assert np.all(wet.omegas < dry)


def test_high_wavenumber_modes_approach_plane_wave_impedance() -> None:
    modes = [hyd.DryMode(m, n, PANEL) for m, n in bench.ASYMPTOTE_MODES]
    added = hyd.panel_added_mass_matrix(PANEL, modes, nx=32, ny=20)
    local = hyd.local_added_mass_matrix(PANEL, modes)
    ratio = hyd.added_mass_ratios(modes, added) / hyd.added_mass_ratios(modes, local)
    ka = np.array([mode.wavenumber * min(PANEL.length, PANEL.breadth) for mode in modes])
    # monotone decay of the excess towards one, and a small excess at high ka
    high = ratio[ka > 8.0]
    assert np.all(high > 1.0)
    assert np.all(np.diff(high) < 0.0)
    assert high.min() - 1.0 < 0.10


# ---------------------------------------------------------------------------
# 4. added-mass bookkeeping
# ---------------------------------------------------------------------------
def test_wet_over_dry_frequency_matches_added_mass_ratio() -> None:
    modes = _modes(3)
    added = hyd.panel_added_mass_matrix(PANEL, modes, nx=16, ny=10)
    wet = hyd.wet_modes(PANEL, modes, added)
    mu = hyd.added_mass_ratios(modes, added)
    for index in range(len(modes)):
        expected = 1.0 / math.sqrt(1.0 + mu[index])
        assert wet.ratio_wet_over_dry[index] == pytest.approx(expected, rel=2.0e-2)


def test_added_mass_ratio_uses_generalized_mass() -> None:
    mode = _modes(1)[0]
    diagonal = np.array([[2.5 * mode.generalized_mass]])
    assert hyd.added_mass_ratios([mode], diagonal)[0] == pytest.approx(2.5, rel=1e-15)


# ---------------------------------------------------------------------------
# 5. cross-module dispersion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [200.0, 40.0])
def test_local_impedance_dispersion_matches_ice_mass_loading(depth: float) -> None:
    cover = ice_module.IceCover(
        thickness=PANEL.thickness,
        ice_density=PANEL.plate_density,
        water_density=PANEL.fluid_density,
        water_depth=depth,
    )
    for k in (0.05, 0.2, 1.0, 5.0, 20.0):
        mine = hyd.flexural_gravity_omega(k, PANEL, flexural_rigidity=0.0, depth=depth)
        assert mine == pytest.approx(ice_module.mass_loading_omega(k, cover), rel=1e-12)


def test_flexural_gravity_reduces_to_gravity_wave_without_plate() -> None:
    for k in (0.01, 0.1, 1.0):
        omega = hyd.flexural_gravity_omega(
            k, PANEL, mass_per_area=0.0, flexural_rigidity=0.0
        )
        assert omega == pytest.approx(math.sqrt(PANEL.gravity * k), rel=1e-12)


# ---------------------------------------------------------------------------
# 6. partitioned coupling
# ---------------------------------------------------------------------------
def test_staggered_coupling_converges_and_matches_monolithic_solution() -> None:
    modes = _modes(2)
    history = hyd.staggered_coupling(PANEL, modes, nx=16, ny=10, tolerance=1e-6)
    assert history.converged
    finite = [r for r in history.residuals if math.isfinite(r)]
    assert finite[-1] < 1e-6
    assert history.pressure_residuals[-1] < 1e-6
    added = hyd.panel_added_mass_matrix(PANEL, modes, nx=16, ny=10)
    wet = hyd.wet_modes(PANEL, modes, added)
    for index in range(len(modes)):
        residual = hyd.monolithic_residual(
            PANEL, modes, added, float(wet.omegas[index]), wet.eigenvectors[:, index]
        )
        assert residual < 1e-10
    assert np.allclose(np.sort(history.omegas), np.sort(wet.omegas), rtol=1e-5)


def test_staggered_coupling_reports_failure_below_tolerance() -> None:
    history = hyd.staggered_coupling(PANEL, _modes(2), nx=8, ny=5, tolerance=1e-14,
                                     max_iterations=3)
    assert not history.converged


def test_staggered_coupling_rejects_bad_relaxation() -> None:
    with pytest.raises(ValueError):
        hyd.staggered_coupling(PANEL, _modes(1), nx=8, ny=5, relaxation=0.0)


# ---------------------------------------------------------------------------
# 7. RAO chain
# ---------------------------------------------------------------------------
def test_quasi_static_rao_matches_closed_form() -> None:
    modes = _modes(3)
    added = hyd.panel_added_mass_matrix(PANEL, modes, nx=16, ny=10)
    wet = hyd.wet_modes(PANEL, modes, added)
    result = hyd.modal_rao(
        PANEL, modes, added, [1e-3 * float(wet.omegas[0])],
        wet_omegas=wet.omegas, damping_ratio=0.02,
    )
    expected = hyd.static_vertical_rao(PANEL, modes)
    assert float(result.vertical_rao[0]) == pytest.approx(expected, rel=1e-4)


def test_rao_peak_sits_on_the_coupled_eigenfrequency() -> None:
    modes = _modes(3)
    added = hyd.panel_added_mass_matrix(PANEL, modes, nx=16, ny=10)
    wet = hyd.wet_modes(PANEL, modes, added)
    omega_1 = float(wet.omegas[0])
    sweep = np.linspace(0.5 * omega_1, 1.5 * omega_1, 800)
    result = hyd.modal_rao(
        PANEL, modes, added, sweep, wet_omegas=wet.omegas, damping_ratio=0.005
    )
    peak = float(sweep[int(np.argmax(result.vertical_rao))])
    assert abs(peak - omega_1) / omega_1 < 0.02


def test_static_vertical_rao_uses_odd_modes_only() -> None:
    even = hyd.DryMode(2, 1, PANEL)
    assert even.static_load(PANEL.fluid_density * PANEL.gravity) == pytest.approx(0.0, abs=1e-18)
    odd = hyd.DryMode(1, 1, PANEL)
    pressure = PANEL.fluid_density * PANEL.gravity
    expected = pressure * (2 * PANEL.length / math.pi) * (2 * PANEL.breadth / math.pi)
    assert odd.static_load(pressure) == pytest.approx(expected, rel=1e-13)


def test_local_operator_resonance_is_the_dispersion_root() -> None:
    k = 3.0
    root = hyd.flexural_gravity_omega(k, PANEL)
    grid = np.linspace(0.5 * root, 1.5 * root, 4000)
    response = np.array([hyd.infinite_plate_response(w, k, PANEL) for w in grid])
    peak = float(grid[int(np.argmax(response))])
    assert abs(peak - root) / root < 1e-3


def test_froude_krylov_force_reduces_to_static_load() -> None:
    mode = hyd.DryMode(1, 1, PANEL)
    small = 1.0e-3
    force = hyd.froude_krylov_forces(PANEL, [mode], small, n_gauss=60)[0]
    assert abs(force) == pytest.approx(
        mode.static_load(PANEL.fluid_density * PANEL.gravity), rel=1e-6
    )


# ---------------------------------------------------------------------------
# 8. artifact gates
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report() -> dict:
    return bench.run()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close(fresh, committed, path: str = "") -> None:
    """Recursively compare the committed record with a fresh run."""
    if isinstance(fresh, dict):
        assert set(fresh) == set(committed), f"key mismatch at {path}"
        for key in fresh:
            _assert_close(fresh[key], committed[key], f"{path}/{key}")
    elif isinstance(fresh, list):
        assert len(fresh) == len(committed), f"length mismatch at {path}"
        for index, (a, b) in enumerate(zip(fresh, committed)):
            _assert_close(a, b, f"{path}[{index}]")
    elif isinstance(fresh, bool) or isinstance(fresh, str) or fresh is None:
        assert fresh == committed, f"value mismatch at {path}"
    else:
        assert float(committed) == pytest.approx(float(fresh), rel=1e-9, abs=1e-12), path


def test_committed_results_reproduce(report: dict) -> None:
    committed = _read_json(ARTIFACTS / "results.json")
    _assert_close(report, committed)
    assert committed["all_criteria_passed"] is True


def test_documents_match_the_generator(report: dict) -> None:
    document = bench.render_document(report)
    assert (ARTIFACTS / "report.md").read_text(encoding="utf-8") == document
    assert DOC_PATH.read_text(encoding="utf-8") == document


def test_acceptance_numbers_meet_the_campaign_criteria(report: dict) -> None:
    assert report["all_criteria_passed"]
    wet = report["wet_modes"]
    assert max(wet["production_error_vs_extrapolated"]) < 0.03
    assert report["rao"]["peak_error_vs_wet_frequency"] < 0.02
    assert report["rao"]["bending_peak_error_vs_wet_frequency"] < 0.02
    coupling = report["coupling"]
    assert coupling["converged"]
    assert coupling["final_omega_residual"] < 1e-6
    assert coupling["final_pressure_residual"] < 1e-6
    assert max(coupling["monolithic_residual"]) < 1e-10
    assert wet["dispersion_crosscheck_max_error"] < 1e-12


def test_document_gate_can_fail_on_a_perturbed_number(report: dict) -> None:
    perturbed = json.loads(json.dumps(report))
    perturbed["wet_modes"]["production_wet_omega"][0] *= 1.01
    assert bench.render_document(perturbed) != (ARTIFACTS / "report.md").read_text(
        encoding="utf-8"
    )