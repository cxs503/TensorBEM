"""Tests for the coupled radiation memory kernel matrix.

Two layers are checked:

* the transform machinery against a **closed-form** Kramers-Kronig pair
  (``K(t) = c exp(-a t) cos(b t)`` has an elementary ``B(omega)`` and
  ``A(omega) - A(inf)``), so the accuracy of the quadratures is measured against
  an exact answer rather than against another implementation of the same idea;
* the *rejections* -- the sampling contract, the passivity rule, the reciprocity
  gate and causality -- which are the part that protects everything downstream.

The boundary-element campaign lives in
``tensorbem.benchmarks.cummins_matrix_closure`` and is exercised by
``docs/cummins_matrix_closure/run_campaign.py``; nothing here solves a BEM
problem, so the suite stays fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import memory_kernel_matrix as mkm

C, A_DECAY, B_FREQ = 1.0, 0.8, 2.0
SCALE = np.array([[1.0, -0.7], [-0.7, 1.6]])


def damping_exact(omega):
    omega = np.asarray(omega, dtype=float)
    return 0.5 * C * (
        A_DECAY / (A_DECAY ** 2 + (omega - B_FREQ) ** 2)
        + A_DECAY / (A_DECAY ** 2 + (omega + B_FREQ) ** 2)
    )


def added_exact(omega):
    """``A(omega) - A(inf)`` for ``K(t) = C exp(-a t) cos(b t)``."""
    omega = np.asarray(omega, dtype=float)
    return -(C / (2.0 * omega)) * (
        (omega - B_FREQ) / (A_DECAY ** 2 + (omega - B_FREQ) ** 2)
        + (omega + B_FREQ) / (A_DECAY ** 2 + (omega + B_FREQ) ** 2)
    )


def kernel_exact(tau, scale=1.0):
    tau = np.asarray(tau, dtype=float)
    return scale * C * np.exp(-A_DECAY * tau) * np.cos(B_FREQ * tau)


@pytest.fixture(scope="module")
def synthetic_sweep():
    omega = np.geomspace(0.005, 60.0, 240)
    added = np.einsum("ij,w->wij", SCALE, added_exact(omega))
    damping = np.einsum("ij,w->wij", SCALE, damping_exact(omega))
    return omega, added, damping


@pytest.fixture(scope="module")
def synthetic_kernel(synthetic_sweep):
    omega, _, damping = synthetic_sweep
    dt = math.pi / 60.0
    tau = np.arange(int(80.0 / dt) + 1) * dt
    return mkm.coupled_memory_kernels(omega, damping, tau, low_frequency_model="zero")


# ---------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------
def test_kernel_matches_closed_form(synthetic_sweep, synthetic_kernel):
    tau = synthetic_kernel.tau
    for i in range(2):
        for j in range(2):
            exact = kernel_exact(tau, SCALE[i, j])
            error = np.max(np.abs(synthetic_kernel.entry(i, j) - exact)) / np.max(np.abs(exact))
            assert error < 2.0e-2, f"entry {i}{j}: relative kernel error {error:.3e}"


def test_jump_at_origin_is_the_dc_damping(synthetic_sweep, synthetic_kernel):
    omega, _, damping = synthetic_sweep
    # K_ij(0+) = (2/pi) int B_ij domega, and for this pair that integral is C*pi/2
    for i in range(2):
        for j in range(2):
            assert synthetic_kernel.entry(i, j)[0] == pytest.approx(
                SCALE[i, j] * C, rel=1.0e-2
            ), "K(0+) must equal the analytic jump"
    del omega, damping


def test_round_trip_recovers_both_coefficients(synthetic_sweep, synthetic_kernel):
    omega, added, damping = synthetic_sweep
    result = mkm.matrix_round_trip(
        omega, added, damping, np.zeros((2, 2)), synthetic_kernel, samples=40
    )
    assert result["passed"], result["worst_relative_error"]
    for entry in result["per_entry"].values():
        assert entry["damping_relative_error"] < 2.0e-2
        assert entry["added_mass_relative_error"] < 2.0e-2


def test_harmonic_force_closes_the_loop(synthetic_sweep, synthetic_kernel):
    omega, added, damping = synthetic_sweep
    infinity = np.zeros((2, 2))
    for value in (1.0, 2.0, 3.0):
        row_a = np.array(
            [[np.interp(value, omega, added[:, i, j]) for j in range(2)] for i in range(2)]
        )
        row_b = np.array(
            [[np.interp(value, omega, damping[:, i, j]) for j in range(2)] for i in range(2)]
        )
        result = mkm.harmonic_radiation_force(
            synthetic_kernel,
            infinity,
            omega=value,
            amplitude=(1.0, 0.6),
            phase=(0.0, 0.7),
            added_mass=row_a,
            damping=row_b,
        )
        assert result["passed"], (value, result["worst_relative_error"])
        assert result["worst_relative_error"] < 1.0e-2


def test_causality_returns_zero_for_negative_time(synthetic_kernel):
    dt = synthetic_kernel.dt
    values = synthetic_kernel.at(np.array([-3.0 * dt, -dt, -1.0e-9]))
    assert np.all(values == 0.0), "K(t < 0) must vanish, not clamp to K(0+)"
    assert np.all(synthetic_kernel.at(np.array([0.0])) != 0.0)


# ---------------------------------------------------------------------------
# the rejections
# ---------------------------------------------------------------------------
def test_sweep_validation_rejects_broken_inputs(synthetic_sweep):
    omega, added, damping = synthetic_sweep
    with pytest.raises(ValueError, match="at least 8 points"):
        mkm.validate_coupled_sweep(omega[:5], added[:5], damping[:5])
    with pytest.raises(ValueError, match="strictly increasing"):
        mkm.validate_coupled_sweep(omega[::-1], added[::-1], damping[::-1])
    asymmetric = damping.copy()
    asymmetric[0, 0, 1] = 10.0 * float(np.max(np.abs(damping)))
    with pytest.raises(ValueError, match="not reciprocal"):
        mkm.validate_coupled_sweep(omega, added, asymmetric)
    indefinite = damping.copy()
    indefinite[0, 0, 1] = 1.0e4
    indefinite[0, 1, 0] = 1.0e4
    with pytest.raises(ValueError, match="positive semidefinite"):
        mkm.validate_coupled_sweep(omega, added, indefinite)


def test_cross_entries_may_be_negative_and_the_scalar_gate_would_refuse():
    """The matrix path must accept a cross entry the scalar path rejects."""
    from tensorbem.time_domain import validate_radiation_sweep

    omega = np.geomspace(0.02, 20.0, 40)
    negative = -damping_exact(omega)
    with pytest.raises(ValueError, match="negative radiation damping"):
        validate_radiation_sweep(omega, np.zeros_like(omega), negative)
    added = np.zeros((omega.size, 2, 2))
    damping = np.stack([np.diag([1.0, 1.0]) * damping_exact(value) for value in omega])
    damping[:, 0, 1] = negative
    damping[:, 1, 0] = negative
    w, a, b = mkm.validate_coupled_sweep(omega, added, damping)
    assert np.all(b[:, 0, 1] < 0.0)
    del w, a


# ---------------------------------------------------------------------------
# the sampling contract
# ---------------------------------------------------------------------------
def test_alias_budget_separates_a_good_grid_from_an_aliased_one(synthetic_sweep):
    omega, _, damping = synthetic_sweep
    fine = mkm.alias_budget(omega, damping, math.pi / omega[-1], threshold=1.0e-2)
    coarse = mkm.alias_budget(omega, damping, 16.0 * math.pi / omega[-1], threshold=1.0e-2)
    assert fine["passed"] and fine["epsilon_tau"] < 1.0e-6
    # a step that folds the damping band onto itself must be refused
    assert not coarse["passed"]
    assert coarse["epsilon_tau"] > 1.0e-2
    steps = [
        mkm.check_alias_budget(omega, damping, factor * math.pi / omega[-1])["epsilon_tau"]
        for factor in (1.0, 16.0, 48.0)
    ]
    assert steps[0] <= steps[1] < steps[2], steps
    assert steps[2] > 0.2, steps
    dt, report = mkm.choose_time_step(omega, damping, threshold=1.0e-2)
    assert report["epsilon_tau"] <= 1.0e-2
    assert dt <= math.pi / omega[-1]


def test_truncation_time_matches_the_half_percent_band(synthetic_kernel):
    truncation = mkm.kernel_truncation(synthetic_kernel, relative=5.0e-3)
    for i in range(2):
        for j in range(2):
            ratio = abs(synthetic_kernel.entry(i, j)[truncation["index"]]) / (
                SCALE[i, j] * C
            )
            assert ratio <= 5.0e-3
    # |K(tau)|/K(0) = exp(-a tau) at the crossing of 5e-3
    analytic = math.log(1.0 / 5.0e-3) / A_DECAY
    assert abs(truncation["tau_max"] - analytic) < 0.5


# ---------------------------------------------------------------------------
# the oscillator
# ---------------------------------------------------------------------------
def _stand(damping_scale=1.0, natural=(0.9, 2.9), mass_value=6.0):
    mass = np.eye(2) * mass_value
    stiffness = np.diag([mass[i, i] * natural[i] ** 2 * damping_scale for i in range(2)])
    return mass, stiffness


def test_negative_control_zero_kernel_gives_zero_decay(synthetic_kernel):
    mass, stiffness = _stand()
    zero = mkm.CoupledKernelMatrix(
        tau=synthetic_kernel.tau,
        values=np.zeros_like(synthetic_kernel.values),
        omega_max=synthetic_kernel.omega_max,
        damping_scale=np.zeros((2, 2)),
        high_frequency_bound=np.zeros((2, 2)),
        low_frequency_bound=np.zeros((2, 2)),
        metadata={},
    )
    frequency, vectors = mkm.mode_shape(mass, np.zeros((2, 2)), stiffness)
    shape = vectors[:, 0]
    period = 2.0 * math.pi / frequency[0]
    result = mkm.coupled_free_decay(
        zero,
        np.zeros((2, 2)),
        mass=mass,
        stiffness=stiffness,
        initial_displacement=shape,
        duration=6.0 * period,
    )
    projection = result["displacement"] @ mass @ shape
    measured = mkm.decay_ratio_from_series(
        result["time"], projection, window=(0.2 * 6 * period, 0.95 * 6 * period)
    )
    assert measured["zeta"] < 1.0e-6, measured


def test_decay_matches_the_kernel_root(synthetic_sweep, synthetic_kernel):
    omega, added, damping = synthetic_sweep
    infinity = np.zeros((2, 2))
    mass, stiffness = _stand()
    roots = mkm.complex_resonances(
        omega,
        added,
        damping,
        infinity,
        mass,
        stiffness,
        omega_window=(0.5, 4.0),
        gamma_window=(0.0, 0.4),
    )
    assert len(roots) == 2, "both modes must be found; a shared root is a silent one-mode test"
    kernel_roots = mkm.kernel_resonances(
        synthetic_kernel, infinity, mass=mass, stiffness=stiffness, guesses=[r["omega"] for r in roots]
    )
    root = roots[0]
    shape = np.asarray(root["mode_shape"], dtype=complex)
    period = 2.0 * math.pi / root["omega"]
    duration = 6.0 * period
    measured = mkm.decay_ratio_richardson(
        synthetic_kernel,
        infinity,
        mass=mass,
        stiffness=stiffness,
        initial_displacement=shape.real,
        initial_velocity=(1j * root["root"] * shape).real,
        duration=duration,
        dt_target=period / 200.0,
        omega_guess=root["omega"],
    )
    relative = abs(measured["zeta"] - kernel_roots[0]["zeta"]) / kernel_roots[0]["zeta"]
    assert relative < 0.03, (measured["zeta"], kernel_roots[0]["zeta"])
    # the unextrapolated pair still carries a few percent of trapezoid bias; the
    # extrapolation is what removes it, so the drift is reported, not minimised
    assert 0.0 <= measured["drift_over_zeta"] < 0.1
    assert measured["fit_residual_over_amplitude"] < 1.0e-2


def test_constant_envelope_is_exact_for_constant_coefficients():
    """A flat sweep means the constant-coefficient substitute is exact."""
    omega = np.geomspace(0.1, 30.0, 60)
    damping = np.zeros((omega.size, 2, 2))
    damping[:, 0, 0] = 2000.0
    damping[:, 1, 1] = 4000.0
    added = np.zeros((omega.size, 2, 2))
    added[:, 0, 0] = 1500.0
    added[:, 1, 1] = 3500.0
    infinity = np.array([[1500.0, 0.0], [0.0, 3500.0]])
    mass = np.eye(2) * 2500.0
    envelope = mkm.constant_matrix_envelope(
        omega, added, damping, infinity, mass, ratios=(1.5, 1.0, 0.7), targets=(1.8, 2.6)
    )
    for row in envelope["rows"]:
        # flat coefficients => the frozen model *is* the system
        assert row["amplitude_error_at_natural"] < 1.0e-9
        assert abs(row["phase_error_degrees"]) < 1.0e-6
        assert row["damping_ratio_error"] < 1.0e-9


# ---------------------------------------------------------------------------
# the campaign helpers that do not need a boundary-element solve
# ---------------------------------------------------------------------------
def test_geometry_gate_accepts_the_case_and_rejects_corrupt_input():
    from tensorbem.benchmarks.cummins_matrix_closure import geometry_gate, section_geometry

    geometry = section_geometry()
    gate = geometry_gate(geometry)
    assert gate["accepted"]
    assert 0.3 < gate["block_coefficient"] <= 1.0
    assert gate["negative_controls_rejected"]
    corrupt = dict(geometry)
    corrupt["displacement"] = geometry["displacement"] * 1.25
    assert not geometry_gate(corrupt)["accepted"]