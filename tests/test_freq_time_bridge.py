"""Tests for the frequency-domain to time-domain bridge benchmark.

This module tests the freq_time_bridge benchmark workflow:
1. Vector fitting of radiation damping
2. State-space realization
3. Cummins equation solving (SS and IRF paths)
4. Cross-validation between paths
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.state_space import (
    vector_fit_damping,
    state_space_from_fit,
    cummins_state_space,
)
from tensorbem.time_domain import (
    radiation_memory_kernel,
    cummins_free_decay,
)

_DATA = Path(__file__).parent / "data" / "time_domain_hemisphere_sweep.json"


@pytest.fixture(scope="module")
def hemisphere_data():
    """Load Hulme hemisphere radiation data."""
    record = json.loads(_DATA.read_text())
    return {
        "omega": np.asarray(record["omega"], dtype=float),
        "added_mass": np.asarray(record["added_mass"], dtype=float),
        "damping": np.asarray(record["damping"], dtype=float),
        "body": record["body"],
    }


def test_freq_time_bridge_workflow(hemisphere_data):
    """Test the complete freq-time bridge workflow."""
    omega = hemisphere_data["omega"]
    damping = hemisphere_data["damping"]
    body = hemisphere_data["body"]

    # Step 1: Vector fitting
    fit = vector_fit_damping(omega, damping, n_poles=12, max_iter=500)
    assert fit.relative_error_max < 0.01, (
        f"Fit error {fit.relative_error_max*100:.2f}% exceeds 1%"
    )

    # Step 2: State-space realization
    ss = state_space_from_fit(fit)
    assert ss.n_states == len(fit.poles)  # One state per pole in this realization
    assert ss.is_stable(), "State-space realization should be stable"

    # Step 3: Free decay simulation (SS path)
    mass = body["mass"]
    restoring = body["restoring"]
    A_inf = body["added_mass_inf"]

    decay_ss = cummins_state_space(
        ss, mass, A_inf, restoring,
        initial_displacement=0.05,
        initial_velocity=0.0,
        dt=0.02,
        duration=20.0
    )
    assert len(decay_ss.displacement) > 100
    assert decay_ss.displacement[-1] < 0.05  # Should decay

    # Step 4: IRF convolution path
    tau = np.arange(0.0, 30.0 + 1e-12, 0.02)
    kernel = radiation_memory_kernel(omega, damping, tau)

    decay_irf = cummins_free_decay(
        kernel,
        mass=mass,
        restoring=restoring,
        added_mass_inf=A_inf,
        initial_displacement=0.05,
        dt=0.02,
        duration=20.0
    )

    # Step 5: Cross-validation
    n_common = min(len(decay_ss.displacement), len(decay_irf.displacement))
    max_diff = np.max(np.abs(decay_ss.displacement[:n_common] - decay_irf.displacement[:n_common]))
    rel_diff = max_diff / np.max(np.abs(decay_irf.displacement[:n_common]))

    # SS and IRF should agree within 5%
    assert rel_diff < 0.05, (
        f"SS vs IRF relative difference {rel_diff*100:.2f}% exceeds 5%"
    )


def test_state_space_stability(hemisphere_data):
    """Test that state-space realization is stable."""
    omega = hemisphere_data["omega"]
    damping = hemisphere_data["damping"]

    fit = vector_fit_damping(omega, damping, n_poles=12, max_iter=500)
    ss = state_space_from_fit(fit)

    # All eigenvalues should have negative real parts
    eigs = ss.eigenvalues()
    assert np.all(eigs.real < 0), f"Unstable eigenvalues: {eigs[eigs.real >= 0]}"


def test_json_serializable_report(hemisphere_data):
    """Test that the benchmark report is JSON serializable."""
    omega = hemisphere_data["omega"]
    damping = hemisphere_data["damping"]
    body = hemisphere_data["body"]

    fit = vector_fit_damping(omega, damping, n_poles=12, max_iter=500)
    ss = state_space_from_fit(fit)

    # Build a report dictionary similar to the benchmark
    report = {
        "fit": {
            "relative_error_max": float(fit.relative_error_max),
            "converged": bool(fit.converged),
        },
        "state_space": {
            "n_states": int(ss.n_states),
            "stable": bool(ss.is_stable()),
            "min_pole_real_part": float(np.min(ss.eigenvalues().real)),
        },
    }

    # Should not raise
    json_str = json.dumps(report)
    parsed = json.loads(json_str)
    assert parsed["state_space"]["stable"] is True
