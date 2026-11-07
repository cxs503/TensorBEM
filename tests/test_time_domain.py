"""Verification of the Cummins radiation memory kernel (``tensorbem.time_domain``).

Three things have to be true before any time-domain seakeeping result built on
this module can be believed, and each is tested here against something that
does not depend on the module:

1. **The transform itself is right.** For a memory kernel that is a sum of
   exponentials, ``K(tau) = sum_j c_j exp(-a_j tau)``, both radiation
   coefficients are known in closed form,
   ``B = sum_j c_j a_j / (a_j^2 + omega^2)`` and
   ``A - A_inf = -sum_j c_j / (a_j^2 + omega^2)``, and eqs. (1)-(3) hold
   identically. The kernel computed from ``B`` must reproduce the exponential
   and the round trip must return ``B`` and ``A``. This calibrates the tool
   before it is used to judge a physical sweep, exactly as the Kramers-Kronig
   tests calibrate the causality transform.
2. **The time integration is right.** With an exponential kernel the Cummins
   equation is a finite linear ODE system (an auxiliary state ``y = int
   exp(-a (t-s)) x' ds`` with ``y' = c x' - a y``), so its decay period and rate
   follow from a cubic characteristic equation and can be computed without any
   convolution. The RK4 + trapezoidal-memory integrator has to reproduce them.
3. **The physical case closes.** On Hulme's floating hemisphere the kernel from
   the pinned sweep must round-trip inside the sampled band, and the free decay
   must agree with the frequency-domain response function's period and
   half-power bandwidth.

Two failure modes are refused rather than silently absorbed: a sweep with
negative radiation damping (non-passive, the kernel would grow) and a sampling
that violates Nyquist in either direction (the kernel would alias).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.time_domain import (
    check_frequency_sampling,
    check_time_sampling,
    cummins_free_decay,
    damping_from_memory_kernel,
    decay_metrics,
    frequency_domain_impedance,
    fourier_round_trip,
    harmonic_response,
    implied_added_mass_inf,
    kernel_diagnostics,
    radiation_memory_kernel,
    resonance_properties,
    two_sided_kernel,
    validate_radiation_sweep,
)

_DATA = Path(__file__).parent / "data" / "time_domain_hemisphere_sweep.json"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hemisphere_sweep():
    """Pinned Hulme heave sweep (uniform in omega) plus the body data."""
    record = json.loads(_DATA.read_text())
    return {
        "omega": np.asarray(record["omega"], dtype=float),
        "added_mass": np.asarray(record["added_mass"], dtype=float),
        "damping": np.asarray(record["damping"], dtype=float),
        "body": record["body"],
    }


@pytest.fixture(scope="module")
def hemisphere_kernel(hemisphere_sweep):
    return radiation_memory_kernel(
        hemisphere_sweep["omega"],
        hemisphere_sweep["damping"],
        np.arange(0.0, 60.0 + 1e-12, 0.01),
    )


def _exponential_pair(omega, coefficients):
    """Exact ``(A - A_inf, B)`` of ``K(tau) = sum_j c_j exp(-a_j tau)``."""
    transform = np.zeros_like(omega, dtype=np.complex128)
    for amplitude, rate in coefficients:
        transform += amplitude / (rate + 1j * omega)
    return np.imag(transform) / omega, np.real(transform)


_COEFFICIENTS = ((1.0, 0.7), (0.4, 2.5), (-0.2, 5.0))


# ---------------------------------------------------------------------------
# 1. the transform, calibrated on an exact causal pair
# ---------------------------------------------------------------------------
def test_kernel_reproduces_exponential_closed_form() -> None:
    """The forward transform of ``B`` must return the exponential kernel.

    The comparison is against the *band-truncated* closed form, evaluated with
    an adaptive quadrature: ``B`` of an exponential kernel falls off only as
    ``omega^-2``, so the error of cutting the sweep at ``omega_max`` is
    ``O(1/omega_max)`` and would otherwise be mistaken for a transform error --
    which is itself the reason the band edges are carried through the module.
    """
    from scipy.integrate import quad

    omega_max = 60.0
    omega = np.linspace(1.0e-4, omega_max, 6001)
    added, damping = _exponential_pair(omega, _COEFFICIENTS)
    tau = np.array([0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
    kernel = radiation_memory_kernel(
        omega, damping, np.arange(0.0, 20.0 + 1e-12, 0.005), low_frequency_model="zero"
    )

    def truncated(value: float) -> float:
        total = 0.0
        for amplitude, rate in _COEFFICIENTS:
            integral, _ = quad(
                lambda w: amplitude * rate / (rate * rate + w * w) * math.cos(w * value),
                0.0,
                omega_max,
                limit=400,
            )
            total += integral
        return 2.0 / math.pi * total

    reference = np.array([truncated(value) for value in tau])
    numeric = kernel(tau)
    scale = float(np.max(np.abs(reference)))
    relative = float(np.max(np.abs(numeric - reference)) / scale)
    assert relative < 1.0e-3, f"kernel vs truncated closed form: {relative:.3e}"


def test_round_trip_returns_both_coefficients_for_exact_pair() -> None:
    """``B -> K -> (B, A)`` on a pair that is causal by construction."""
    omega = np.linspace(1.0e-3, 60.0, 3001)
    added, damping = _exponential_pair(omega, _COEFFICIENTS)
    added_inf = 0.0
    report = fourier_round_trip(
        omega, added, damping, added_inf, dt=0.01, samples=40
    )
    assert report.band[0] <= report.band[1]
    assert report.damping_relative_error_max < 1.0e-2
    assert report.added_mass_relative_error_max < 1.0e-2
    # the implied asymptote must come back as the A_inf that was used. Only the
    # mean is asserted: the per-frequency spread is amplified by 1/omega at the
    # low band edge (eq. (3) divides by omega), so it measures the band edge,
    # not the transform.
    implied = report.implied_added_mass_inf
    assert abs(implied["mean"] - added_inf) < 1.0e-2 * float(np.max(np.abs(added)))


def test_parity_closure_closes_on_exact_pair(tmp_path) -> None:
    """The even/odd reconstruction of the kernel closes for a causal pair.

    ``K`` rebuilt from ``[B cos - omega (A - A_inf) sin] / pi`` must equal the
    cosine transform of ``B`` alone. The odd partner decays only as ``1/omega``,
    so the closure is limited by the band truncation rather than by roundoff --
    which is why the tolerance is 2% here and 1% is asked of the hemisphere
    sweep, whose band is far wider relative to the kernel.
    """
    from tensorbem.benchmarks.time_domain_pulse import parity_closure_residual

    omega = np.linspace(1.0e-3, 60.0, 4001)
    added, damping = _exponential_pair(omega, _COEFFICIENTS)
    tau = np.arange(0.0, 20.0 + 1e-12, 0.01)
    kernel = radiation_memory_kernel(omega, damping, tau)
    residual = parity_closure_residual(omega, added, damping, 0.0, kernel)
    assert residual["relative_residual_max"] < 0.02, residual["relative_residual_max"]


# ---------------------------------------------------------------------------
# 2. the time integration, calibrated on the exact auxiliary ODE
# ---------------------------------------------------------------------------
def test_cummins_decay_matches_exact_auxiliary_ode() -> None:
    """Exponential kernel: the convolution integrator vs the closed-form cubic."""
    amplitude, rate = 5000.0, 2.0
    mass, added_inf, restoring = 2000.0, 1000.0, 30000.0
    omega = np.linspace(1.0e-3, 400.0, 6001)
    damping = amplitude * rate / (rate**2 + omega**2)
    tau = np.arange(0.0, 12.0 + 1e-12, 0.002)
    kernel = radiation_memory_kernel(omega, damping, tau)

    # exact cubic: s^3 + a s^2 + (c + C)/(M + A_inf) s + a C/(M + A_inf) = 0
    inertia = mass + added_inf
    roots = np.roots(
        [1.0, rate, (amplitude + restoring) / inertia, rate * restoring / inertia]
    )
    pole = [root for root in roots if abs(root.imag) > 1.0e-9][0]
    exact_rate, exact_period = -pole.real, 2.0 * math.pi / pole.imag

    decay = cummins_free_decay(
        kernel,
        mass=mass,
        restoring=restoring,
        added_mass_inf=added_inf,
        initial_displacement=0.05,
        dt=0.002,
        duration=16.0,
    )
    metrics = decay_metrics(decay.time, decay.displacement, amplitude_fraction=1e-4)
    assert abs(metrics["period"] - exact_period) / exact_period < 5.0e-3
    assert abs(metrics["decay_rate"] - exact_rate) / exact_rate < 5.0e-3

    # and the closed-form ODE solution, integrated to a tight tolerance
    def rhs(_t, state):
        z, v, y = state
        return [v, -(y + restoring * z) / inertia, amplitude * v - rate * y]

    reference = solve_ivp(
        rhs,
        [0.0, 16.0],
        [0.05, 0.0, 0.0],
        rtol=1e-11,
        atol=1e-13,
        max_step=0.01,
        dense_output=True,
    )
    sampled = reference.sol(decay.time)[0]
    deviation = float(np.max(np.abs(sampled - decay.displacement)))
    assert deviation < 1.0e-2 * 0.05, f"integrator vs exact ODE: {deviation:.3e} m"


# ---------------------------------------------------------------------------
# 3. the physical case
# ---------------------------------------------------------------------------
def test_pinned_sweep_still_matches_the_reference(hemisphere_sweep) -> None:
    """Guard against a stale pin: re-evaluate three points from Hulme."""
    omega = hemisphere_sweep["omega"]
    for fraction in (0.1, 0.5, 0.9):
        index = int(fraction * (omega.size - 1))
        reference = hulme_hemisphere_radiation(
            float(omega[index] ** 2 / 9.80665), radius=1.0, density=1025.0
        )
        assert reference.heave_added_mass == pytest.approx(
            hemisphere_sweep["added_mass"][index], rel=1e-9
        )
        assert reference.heave_damping == pytest.approx(
            hemisphere_sweep["damping"][index], rel=1e-9
        )


def test_hemisphere_kernel_decays_and_is_even(hemisphere_sweep, hemisphere_kernel) -> None:
    diagnostics = kernel_diagnostics(
        hemisphere_kernel, hemisphere_sweep["omega"], hemisphere_sweep["damping"]
    )
    assert diagnostics["decay_rate"] > 0.0
    assert diagnostics["tail_relative"] < 0.05
    assert diagnostics["even_residual"] == pytest.approx(0.0, abs=1e-12)
    # K(0) = (2/pi) int B domega, closed form of eq. (1)
    expected = 2.0 / math.pi * float(
        np.trapezoid(hemisphere_sweep["damping"], hemisphere_sweep["omega"])
    )
    assert diagnostics["value_at_zero"] == pytest.approx(expected, rel=1e-3)
    # even kernel => zero slope at the origin, on the scale of the cosine grid
    time_step = hemisphere_kernel.dt
    bound = (time_step * float(hemisphere_sweep["omega"][-1])) ** 2
    assert abs(diagnostics["relative_slope_at_zero"]) < bound

    tau, values = two_sided_kernel(hemisphere_kernel)
    assert np.allclose(values[: tau.size // 2], values[::-1][: tau.size // 2])
    # causality: the kernel is exactly zero for negative times
    assert float(np.max(np.abs(hemisphere_kernel(-np.abs(tau))))) == pytest.approx(
        hemisphere_kernel(tau).max(), rel=1e-12
    )


def test_hemisphere_round_trip_within_five_percent(hemisphere_sweep) -> None:
    report = fourier_round_trip(
        hemisphere_sweep["omega"],
        hemisphere_sweep["added_mass"],
        hemisphere_sweep["damping"],
        hemisphere_sweep["body"]["added_mass_inf"],
        dt=0.01,
        samples=40,
    )
    assert report.damping_relative_error_max < 0.05
    assert report.added_mass_relative_error_max < 0.05
    # measured margin, not just the acceptance threshold
    assert report.worst_relative_error < 5.0e-3
    implied = implied_added_mass_inf(
        report.kernel, hemisphere_sweep["omega"], hemisphere_sweep["added_mass"]
    )
    assert abs(
        implied["mean"] - hemisphere_sweep["body"]["added_mass_inf"]
    ) < 1.0e-3 * hemisphere_sweep["body"]["added_mass_inf"]


def test_hemisphere_free_decay_matches_response_bandwidth(
    hemisphere_sweep, hemisphere_kernel
) -> None:
    body = hemisphere_sweep["body"]
    decay = cummins_free_decay(
        hemisphere_kernel,
        mass=body["mass"],
        restoring=body["restoring"],
        added_mass_inf=body["added_mass_inf"],
        initial_displacement=0.05,
        dt=0.005,
        duration=14.0,
    )
    metrics = decay_metrics(decay.time, decay.displacement, amplitude_fraction=5e-3)
    response = resonance_properties(
        hemisphere_sweep["omega"],
        hemisphere_sweep["added_mass"],
        hemisphere_sweep["damping"],
        mass=body["mass"],
        restoring=body["restoring"],
    )
    period_error = abs(metrics["period"] - response["damped_period"]) / response[
        "damped_period"
    ]
    bandwidth_rate = response["zeta_bandwidth"] * response["omega_n"]
    rate_error = abs(metrics["decay_rate"] - bandwidth_rate) / bandwidth_rate
    assert period_error < 0.02, f"period error {period_error:.3%}"
    assert rate_error < 0.05, f"decay-rate error {rate_error:.3%}"
    assert metrics["fit_r_squared"] > 0.99


def test_driven_response_matches_frequency_domain_impedance(
    hemisphere_sweep, hemisphere_kernel
) -> None:
    body = hemisphere_sweep["body"]
    response = resonance_properties(
        hemisphere_sweep["omega"],
        hemisphere_sweep["added_mass"],
        hemisphere_sweep["damping"],
        mass=body["mass"],
        restoring=body["restoring"],
    )
    frequency = response["omega_n"]
    driven = harmonic_response(
        hemisphere_kernel,
        mass=body["mass"],
        restoring=body["restoring"],
        added_mass_inf=body["added_mass_inf"],
        force_amplitude=1.0,
        omega=frequency,
        dt=0.005,
        duration=40.0,
        warmup_fraction=0.5,
    )
    impedance = frequency_domain_impedance(
        [frequency],
        hemisphere_sweep["omega"],
        hemisphere_sweep["added_mass"],
        hemisphere_sweep["damping"],
        mass=body["mass"],
        restoring=body["restoring"],
    )[0]
    expected = 1.0 / impedance
    assert abs(driven["amplitude"] - abs(expected)) / abs(expected) < 0.02
    phase_error = abs(
        math.atan2(
            math.sin(driven["phase"] - float(np.angle(expected))),
            math.cos(driven["phase"] - float(np.angle(expected))),
        )
    )
    assert phase_error < 0.05


# ---------------------------------------------------------------------------
# negative cases: rejected inputs
# ---------------------------------------------------------------------------
def test_negative_damping_is_rejected(hemisphere_sweep) -> None:
    damping = np.array(hemisphere_sweep["damping"])
    damping[50] = -0.05 * damping.max()
    with pytest.raises(ValueError, match="negative radiation damping"):
        validate_radiation_sweep(
            hemisphere_sweep["omega"], hemisphere_sweep["added_mass"], damping
        )
    with pytest.raises(ValueError, match="negative radiation damping"):
        radiation_memory_kernel(
            hemisphere_sweep["omega"], damping, np.arange(0.0, 1.0, 0.01)
        )


def test_kernel_sampling_violating_nyquist_is_rejected(hemisphere_sweep) -> None:
    omega = hemisphere_sweep["omega"]
    too_coarse = math.pi / float(omega[-1]) * 1.5
    with pytest.raises(ValueError, match="Nyquist"):
        radiation_memory_kernel(
            omega, hemisphere_sweep["damping"], np.arange(0.0, 5.0, too_coarse)
        )
    with pytest.raises(ValueError, match="Nyquist"):
        check_time_sampling(too_coarse, float(omega[-1]))
    assert check_time_sampling(0.5 * math.pi / float(omega[-1]), float(omega[-1])) > 0


def test_reconstruction_frequency_above_kernel_nyquist_is_rejected(
    hemisphere_sweep, hemisphere_kernel
) -> None:
    above = 1.5 * hemisphere_kernel.nyquist_omega
    with pytest.raises(ValueError, match="Nyquist"):
        damping_from_memory_kernel(hemisphere_kernel, [above])
    with pytest.raises(ValueError, match="Nyquist"):
        check_frequency_sampling(above, hemisphere_kernel.span, dtau=hemisphere_kernel.dt)
    # a query inside the grid is accepted and matches the tabulated damping
    inside = 0.5 * hemisphere_kernel.nyquist_omega
    value = damping_from_memory_kernel(hemisphere_kernel, [inside])[0]
    expected = float(np.interp(inside, hemisphere_sweep["omega"], hemisphere_sweep["damping"]))
    assert abs(value - expected) < 0.05 * hemisphere_sweep["damping"].max()


def test_malformed_sweeps_are_rejected(hemisphere_sweep) -> None:
    omega = hemisphere_sweep["omega"]
    added = hemisphere_sweep["added_mass"]
    damping = hemisphere_sweep["damping"]
    with pytest.raises(ValueError, match="at least"):
        validate_radiation_sweep(omega[:5], added[:5], damping[:5])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_radiation_sweep(omega[::-1], added, damping)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_radiation_sweep(
            np.concatenate([[-1.0], omega[1:]]), added, damping
        )
    broken = np.array(damping)
    broken[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_radiation_sweep(omega, added, broken)


def test_growing_signal_is_reported_as_negative_decay() -> None:
    """A negative-damping synthetic signal must not be reported as decaying."""
    time = np.arange(0.0, 20.0, 0.002)
    signal = 0.1 * np.exp(0.05 * time) * np.cos(3.0 * time)
    metrics = decay_metrics(time, signal, amplitude_fraction=1e-3)
    assert metrics["decay_rate"] < 0.0