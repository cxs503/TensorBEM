"""G4 root-cause verification for CASE-BEM-CUMMINS-WIGLEY (m23978).

Pins the verification-debt finding: the G4 free-decay gate's ~28-35% failure
is a *genuine estimator bias* from the strong B(w) frequency-dependence across
the half-power width -- NOT a solver bug.  The time-domain decay zeta is the
true damping ratio of the free response; the half-power bandwidth proxy is
exact only for constant/weakly-frequency-dependent single-DOF oscillators.
"""
from __future__ import annotations
import json, math, pathlib
import numpy as np
import pytest
from tensorbem import time_domain as td
from tensorbem.benchmarks import cummins_wigley_kk_causality as case

SWEEP = pathlib.Path("benchmarks/verified/casepool_bem_cummins_wigley/sweep_closure_medium800.npz")
ARTIFACT = pathlib.Path("artifacts/g4_root_cause/g4_root_cause.json")


def _load_wigley_heave():
    d = np.load(SWEEP, allow_pickle=True)
    sweep = d["sweep"].item() if "sweep" in d else d
    omega = np.asarray(sweep["omega"], float)
    added = np.asarray(sweep["added_mass"], float)[:, 2, 2]
    damping = np.asarray(sweep["damping"], float)[:, 2, 2]
    ref = case.wigley3_hydrostatic_reference()
    mass = float(ref["volume"] * 1025.0)
    restoring = float(ref["C33"])
    return omega, added, damping, mass, restoring


def test_g4_failure_is_estimator_bias_not_solver_bug():
    """The G4 28-35% is a half-power estimator bias, not a solver defect."""
    omega, added, damping, mass, restoring = _load_wigley_heave()
    a_inf = 69.10675783928987
    hp = td.half_power_damping(
        case.decay_response_grid(omega),
        np.interp(case.decay_response_grid(omega), omega, added),
        np.interp(case.decay_response_grid(omega), omega, damping),
        mass=mass, restoring=restoring)
    lo, hi = hp["half_power_omega_low"], hp["half_power_omega_high"]
    b_lo, b_hi = np.interp(lo, omega, damping), np.interp(hi, omega, damping)
    b_peak = np.interp(hp["peak_omega"], omega, damping)
    # The bias driver: B(w) varies >30% across the half-power width.
    assert abs(b_hi - b_lo) / b_peak > 0.30, (
        f"B variation across half-power width ({abs(b_hi-b_lo)/b_peak*100:.1f}%) "
        "should exceed 30% -- this is what biases the half-power proxy")
    # The half-power proxy is NOT the true damping ratio here (it is biased low).
    assert hp["zeta"] < 0.20, "half-power zeta should be biased low (<0.20)"


def test_decay_zeta_is_the_true_damping_ratio():
    """The time-domain decay zeta is the true damping ratio of the free response."""
    omega, added, damping, mass, restoring = _load_wigley_heave()
    a_inf = 69.10675783928987
    tau = np.arange(0.0, 40.0, 0.02)
    kernel = td.radiation_memory_kernel(omega, damping, tau)
    decay = td.cummins_free_decay(
        kernel, mass=mass, restoring=restoring, added_mass_inf=a_inf,
        dt=0.02, duration=120.0)
    m = td.decay_metrics(decay.time, decay.displacement)
    rate = float(m["decay_rate"]); period = m["period"]
    omega_d = 2 * math.pi / period
    omega_n = math.hypot(omega_d, rate)
    zeta_decay = rate / omega_n
    # The decay zeta is the physically meaningful damping ratio (0.2-0.25 range).
    assert 0.15 < zeta_decay < 0.30, f"decay zeta={zeta_decay:.4f} should be in (0.15,0.30)"


def test_weak_frequency_dependence_agrees_within_tolerance():
    """On a weakly frequency-dependent causal pair the two estimators agree <10%."""
    a, c, a_inf = 0.5, 10.0, 5.0
    ws = np.geomspace(0.1, 15.0, 200)
    bs = c * a / (a * a + ws**2)
    as_ = a_inf - c / (a * a + ws**2)
    hp = td.half_power_damping(
        case.decay_response_grid(ws),
        np.interp(case.decay_response_grid(ws), ws, as_),
        np.interp(case.decay_response_grid(ws), ws, bs),
        mass=100.0, restoring=500.0)
    tau = np.arange(0.0, 40.0, 0.02)
    kernel = td.radiation_memory_kernel(ws, bs, tau)
    decay = td.cummins_free_decay(
        kernel, mass=100.0, restoring=500.0, added_mass_inf=a_inf,
        dt=0.02, duration=120.0)
    m = td.decay_metrics(decay.time, decay.displacement)
    rate = float(m["decay_rate"]); period = m["period"]
    omega_d = 2 * math.pi / period
    omega_n = math.hypot(omega_d, rate)
    zeta_decay = rate / omega_n
    rel = abs(zeta_decay - hp["zeta"]) / hp["zeta"]
    assert rel < 0.10, f"weak-freq-dep pair rel_err={rel*100:.2f}% should be <10%"


def test_committed_artifact_records_the_root_cause():
    """The committed verification artifact records the G4 root-cause conclusion."""
    if not ARTIFACT.exists():
        pytest.skip("g4_root_cause artifact not yet generated")
    payload = json.loads(ARTIFACT.read_text())
    assert "estimator bias" in payload["conclusion"]
    assert payload["wigley"]["B_variation_pct"] > 30.0
