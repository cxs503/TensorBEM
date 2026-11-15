"""C5 attribution audit: the estimator is validated before it accuses a route.

The audit claims that gate C5 of ``CASE-BEM-CUMMINS-WIGLEY`` fails because the
free-decay route integrates a hard-truncated memory kernel, not because of the
frequency-domain reference.  These tests hold that claim to its own logic:

* the estimator machinery is exact on analytic problems (D0/D0b),
* the decay measurement is byte-identical to the case module's (no metric drift),
* the causal continuation is inside the in-band budgets (D1) and passes the
  replacement gate (D3),
* the legacy route really is cutoff-dependent (D5) -- documented as a defect
  demonstration, so the day it stops being true the doc must be revisited.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import state_space as ssmod
from tensorbem import time_domain as td
from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.benchmarks import cummins_wigley_causal_decay as audit


@pytest.fixture(scope="module")
def medium() -> dict:
    return audit.run_audit("medium")


def test_estimator_matches_closed_form_on_constant_damping():
    """D0: complex root == B/(2 sqrt((M+A)C)) to machine precision."""
    checks = audit.analytic_oscillator_checks()["constant_damping"]
    for row in checks:
        assert row["relative_error"] < 1e-12, row


def test_estimator_matches_numpy_roots_on_cubic_memory():
    """D0b: Newton root == numpy.roots for a single-exponential memory kernel."""
    row = audit.analytic_oscillator_checks()["single_exponential_memory"]
    assert row["relative_error"] < 1e-10, row
    assert row["zeta_reference"] > 0.0 and row["converged"]


def test_decay_estimator_is_the_case_estimator():
    """The audit must not change the measurement while it changes the model."""
    time = np.linspace(0.0, 20.0, 4001)
    displacement = np.exp(-0.9 * time) * np.cos(7.8 * time)
    ours = audit.decay_zeta(time, displacement)
    theirs = case._decay_zeta(time, displacement)
    assert ours[2] == theirs[2] > 3
    assert ours[0] == pytest.approx(theirs[0], rel=1e-12)
    assert ours[1] == pytest.approx(theirs[1], rel=1e-12)


def test_legacy_c5_is_reproduced(medium):
    """The audit starts from the recorded C5 numbers, not from new ones."""
    legacy = medium["legacy_route"]
    assert legacy["zeta_decay"] == pytest.approx(0.13787, abs=5e-4)
    assert medium["legacy_reference"]["zeta_frequency_domain"] == pytest.approx(0.12875, abs=5e-4)
    assert medium["attribution"]["legacy_C5_relative_error"] == pytest.approx(0.0709, abs=1e-3)


def test_causal_continuation_is_inside_its_band_budgets(medium):
    """D1: the rational continuation fits B in band and keeps A Kramers-Kronig consistent."""
    assert medium["gates"]["D1_fit_band"]["passed"] is True
    assert medium["gates"]["D1_kk_added_mass"]["passed"] is True


def test_causal_kernel_decays_while_truncated_kernel_does_not(medium):
    """D2 + defect demonstration: the two kernels are qualitatively different."""
    assert medium["gates"]["D2_kernel_decay"]["passed"] is True
    assert abs(medium["legacy_route"]["kernel_at_tau_max"]) > 1e-2
    assert medium["causal_model"]["kernel_decay_ratio"] < 1e-5


def test_causal_decay_agrees_with_exact_modal_root(medium):
    """D3: the replacement gate -- time domain vs exact root of the same model."""
    zeta_causal = medium["causal_model"]["zeta_decay_time_domain"]
    zeta_exact = medium["causal_model"]["zeta_exact"]
    assert abs(zeta_causal - zeta_exact) / zeta_exact < 0.05
    assert medium["gates"]["D3_causal_decay"]["passed"] is True


def test_legacy_route_is_the_biased_one(medium):
    """D4/attribution: the truncated-kernel decay is farther from truth than the reference."""
    attribution = medium["attribution"]
    assert attribution["legacy_vs_exact"] > attribution["reference_vs_exact"]
    assert attribution["verdict"].startswith("H-route")
    assert medium["gates"]["D4_legacy_bias"]["passed"] is False  # honest, recorded FAIL


def test_legacy_route_is_not_band_robust(medium):
    """D5 (defect demonstration): the legacy damping ratio follows the cutoff."""
    rows = [r for r in medium["band_top_scan"] if "zeta_decay" in r]
    zeta = [r["zeta_decay"] for r in rows]
    assert max(zeta) / min(zeta) > 10.0, zeta
    # and the sequence has not converged at the production band top either
    assert abs(zeta[-1] - zeta[-2]) / zeta[-1] > 0.02, zeta


def test_exact_modal_root_residual_and_order_convergence(medium):
    """The exact root is a real root of the characteristic equation, order-stable."""
    causal = medium["causal_model"]
    assert causal["residual_relative"] < 1e-8
    zetas = [row["zeta_exact"] for row in medium["order_scan"]
             if row["fit_relative_error_max"] <= 0.03 and row["kk_added_mass_relative"] <= 0.05]
    assert len(zetas) >= 3
    assert (max(zetas) - min(zetas)) / min(zetas) < 0.05


def test_zero_damping_gives_zero_modal_damping():
    """Negative control: a damping-free continuation has no modal damping."""
    poles = np.array([1.0, 4.0])
    realization = ssmod.StateSpaceRealization(
        A_ss=np.diag(-poles), B_ss=np.array([1e-14, 1e-14]), C_ss=np.ones(2),
        poles=poles, residues=np.array([1e-14, 1e-14]), D_ss=0.0,
    )
    root = audit.complex_modal_root(realization, 76.875, 50.93, 6016.0, guess_omega=7.8)
    assert root["zeta"] < 1e-10
    assert root["omega_damped"] == pytest.approx(math.sqrt(6016.0 / (76.875 + 50.93)), rel=1e-3)


def test_attribution_is_mesh_dependent_and_that_is_the_finding():
    """The 5 % C5 comparison is not decidable: the bias follows the mesh.

    Medium mesh -> the truncated-kernel decay is the biased route; coarse mesh ->
    the interpolated energy identity is.  Both errors are of the order of the
    gate itself, so neither estimator is a 5 % reference.  This test pins the
    flip: if a future change makes the two levels agree, the conclusion recorded
    in ``docs/dev_report_m21275_c5_attribution.md`` has to be rewritten, not the
    test silently relaxed.
    """
    coarse = audit.run_audit("coarse")
    medium = audit.run_audit("medium")
    verdicts = {level["attribution"]["verdict"].split(":")[0]
                for level in (coarse, medium)}
    assert verdicts == {"H-route", "H-reference"}, verdicts
    # both routes carry a few per cent on both meshes -- nothing is a 5 % reference
    for result in (coarse, medium):
        attribution = result["attribution"]
        assert min(attribution["legacy_vs_exact"], attribution["reference_vs_exact"]) < 0.05
        assert max(attribution["legacy_vs_exact"], attribution["reference_vs_exact"]) > 0.01
        # the cutoff dependence of the legacy route holds on both meshes
        zeta = [r["zeta_decay"] for r in result["band_top_scan"] if "zeta_decay" in r]
        assert max(zeta) / min(zeta) > 10.0
        # and the causal route stays internally consistent on both meshes
        assert result["gates"]["D3_causal_decay"]["passed"] is True
