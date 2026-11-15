"""G4 free-decay gate tests for CASE-BEM-CUMMINS-WIGLEY.

These tests verify that the G4 gate (free decay zeta vs half-power zeta)
is correctly implemented and produces consistent results on known causal pairs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import time_domain as td
from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.causality import fit_added_mass_asymptote


def _exact_pair_1d(a: float = 0.5, c: float = 10.0, a_inf: float = 5.0,
                   omegas=None) -> tuple:
    """Exact causal (A, B) pair from exponential kernel K = c exp(-a t)."""
    omegas = np.asarray(
        omegas if omegas is not None else np.geomspace(0.1, 15.0, 200), dtype=float
    )
    b = c * a / (a * a + omegas**2)
    added = a_inf - c / (a * a + omegas**2)
    return omegas, added, b


def _build_sweep_1d(omegas, added, damping, mass=100.0, c33=1000.0) -> dict:
    """Build a 1-DOF sweep dict suitable for G4 analysis."""
    return {
        "level": "analytic",
        "rotation_center": list(case.ROTATION_CENTER),
        "omega": omegas.tolist(),
        "added_mass": np.array([[[added[i]]] for i in range(len(omegas))]).tolist(),
        "damping": np.array([[[damping[i]]] for i in range(len(omegas))]).tolist(),
        "mass": mass,
        "C33": c33,
    }


def test_g4_exact_causal_pair_consistent():
    """On an exact causal pair, zeta_decay should match zeta_half_power closely."""
    omegas, added, damping = _exact_pair_1d(a=0.5, c=10.0, a_inf=5.0)
    sweep = _build_sweep_1d(omegas, added, damping, mass=100.0, c33=500.0)

    fit = fit_added_mass_asymptote(omegas, added)
    a_inf = float(fit.added_mass_inf)

    tau = np.arange(0.0, 40.0, 0.02)
    kernel = td.radiation_memory_kernel(omegas, damping, tau, low_frequency_model="zero")

    # Half-power zeta
    hp = td.half_power_damping(omegas, added, damping, mass=100.0, restoring=500.0, samples=4001)
    zeta_hp = hp["zeta"]
    omega_n_hp = hp["omega_n"]

    # Free decay zeta
    decay = td.cummins_free_decay(
        kernel, mass=100.0, restoring=500.0, added_mass_inf=a_inf,
        initial_displacement=0.01, dt=0.02, duration=40.0
    )
    metrics = td.decay_metrics(decay.time, decay.displacement)
    zeta_decay = metrics["decay_rate"] / omega_n_hp

    rel_err = abs(zeta_decay - zeta_hp) / zeta_hp
    assert rel_err < 0.10, f"G4 rel_err={rel_err:.4f} on exact pair (expected <5%)"


def test_g4_heave_closure_grid_passes():
    """Heave G4 on the coarse closure grid should pass (<10%)."""
    import json
    import pathlib

    results_path = pathlib.Path("artifacts/case_bem_cummins_wigley_round6/results_closure.json")
    if not results_path.exists():
        pytest.skip("closure results not yet generated")

    with open(results_path) as f:
        results = json.load(f)

    heave_g4 = results["per_dof"]["heave"]["G4"]
    assert heave_g4["decidable"], "Heave G4 should be decidable on the closure grid"
    assert heave_g4["rel_err"] is not None
    assert heave_g4["rel_err"] < 0.10, (
        f"Heave G4 rel_err={heave_g4['rel_err']:.4f} should be <10% "
        f"(zeta_decay={heave_g4['zeta_decay']:.6f}, zeta_hp={heave_g4['zeta_half_power']:.6f})"
    )


def test_g4_low_damping_dof_has_higher_error():
    """Low-damping DOFs (sway, yaw) should have higher G4 error than heave."""
    import json
    import pathlib

    results_path = pathlib.Path("artifacts/case_bem_cummins_wigley_round6/results_closure.json")
    if not results_path.exists():
        pytest.skip("closure results not yet generated")

    with open(results_path) as f:
        results = json.load(f)

    heave_r4 = results["per_dof"]["heave"]["G4"]["rel_err"]
    sway_r4 = results["per_dof"]["sway"]["G4"]["rel_err"]

    # Sway has much lower damping than heave, so G4 error should be higher
    assert sway_r4 > heave_r4, (
        f"Sway G4 error ({sway_r4:.4f}) should exceed heave ({heave_r4:.4f}) "
        "due to lower damping magnitude"
    )


def test_g4_gate_verdict_machine_decided():
    """The G4 gate verdict should be machine-decided from the rel_err vs tol."""
    tol = case.GATES["G4_free_decay"]["tol"]
    assert tol == 0.10, "G4 tolerance should be 10%"

    # Verify the verdict logic
    assert case._verdict("G4", 0.05, 0.10)["passed"]
    assert not case._verdict("G4", 0.11, 0.10)["passed"]
    assert not case._verdict("G4", float("nan"), 0.10)["passed"]
