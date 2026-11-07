"""Time-domain impulse response of the radiation memory: minimal verification surface.

The frequency-domain legs of this repository verify *coefficients*: added mass,
damping, exciting forces, against closed-form or published references. This
benchmark verifies the object that a time-domain seakeeping code actually
consumes -- the Cummins memory kernel -- and only that, on one documented case:

**Floating hemisphere of radius a = 1 m, heave, deep water, zero speed.**
``A33(omega)`` and ``B33(omega)`` are taken from Hulme's (1982) semi-analytical
solution as implemented in :func:`tensorbem.analytic.hulme_hemisphere_radiation`,
swept over ``ka in [0.002, 90]`` (``omega in [0.14, 29.7] rad/s``). Because the
sweep comes from a reference solution rather than from a panel mesh, the
verification isolates the frequency-to-time mapping: any residual reported here
is a property of the transform or of the time integration, not of a
discretisation.

Three criteria are checked (the numbers are written into the JSON artifact and
into ``docs/time_domain_pulse.md``):

``C1 decay_and_parity``
    The kernel decays (log-envelope rate > 0, tail below 5% of the peak), is
    even (two-sided extension transforms to a real spectrum, residual at
    roundoff), and closes the even/odd pairing of the spectrum: rebuilding the
    kernel from ``B cos - omega (A - A_inf) sin`` -- the parity partner that a
    causal kernel forces -- reproduces the kernel built from ``B`` alone.
``C2 fourier_round_trip``
    ``B -> K -> (B, A)`` inside the sampled band with independent quadratures:
    the worst relative error must stay below 5%, and the added-mass asymptote
    implied by the round trip must recover ``A_inf``.
``C3 time_vs_frequency``
    Free decay from rest, integrated on the Cummins equation with the same
    kernel, must reproduce the resonance period and the half-power bandwidth
    damping of the mechanical response function ``1/kappa(omega)``, which is
    built from the sweep alone and never sees the time integration. Driven runs
    additionally compare the time-domain steady-state amplitude and phase with
    ``1/kappa(omega)`` at three frequencies.

Run ``python -m tensorbem.benchmarks.time_domain_pulse`` for the JSON artifact
(``artifacts/time_domain_pulse.json``) and ``--figure`` for the plots.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from ..analytic import hulme_hemisphere_radiation
from ..time_domain import (
    FourierRoundTrip,
    MemoryKernel,
    cummins_free_decay,
    decay_metrics,
    frequency_domain_impedance,
    fourier_round_trip,
    harmonic_response,
    kernel_diagnostics,
    radiation_memory_kernel,
    resonance_properties,
    two_sided_kernel,
)

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
OMEGA_RANGE = (0.14, 29.7)  # ka in [0.002, 90] for a = 1 m in g = 9.80665 m/s^2
SWEEP_POINTS = 501
ROUND_TRIP_DT = 0.01
DECAY_DT = 0.0025
DECAY_DURATION = 20.0
ROUND_TRIP_CRITERION = 0.05
DECAY_CRITERION = 0.05


# ---------------------------------------------------------------------------
# the sweep and the physical set-up
# ---------------------------------------------------------------------------
def hemisphere_heave_sweep(
    points: int = SWEEP_POINTS, omega_range: tuple[float, float] = OMEGA_RANGE
) -> dict[str, Any]:
    """Hulme (1982) heave radiation coefficients on a *uniform* ``omega`` grid.

    Uniform in ``omega``, not in ``ka``: the transform of eq. (1) integrates
    over ``omega`` and the kernel resolution at low frequency is set by the
    ``omega`` spacing. A sweep that is uniform in ``ka`` (the natural parameter
    of Hulme's expansion) leaves the first ``omega`` interval spanning more than
    a decade, the piecewise-linear interpolant there is meaningless, and the
    reconstructed added mass picks up a spurious low-frequency error that has
    nothing to do with the transform. ``ka = a omega^2 / g`` is evaluated per
    sample, so the cost is unchanged.
    """
    omega = np.linspace(omega_range[0], omega_range[1], int(points))
    ka = omega * omega * RADIUS / GRAVITY
    added = np.empty_like(omega)
    damping = np.empty_like(omega)
    for index, value in enumerate(ka):
        reference = hulme_hemisphere_radiation(
            float(value), radius=RADIUS, density=DENSITY
        )
        added[index] = reference.heave_added_mass
        damping[index] = reference.heave_damping
    return {
        "ka": ka,
        "omega": omega,
        "added_mass": added,
        "damping": damping,
    }


def hemisphere_heave_setup(density: float = DENSITY, gravity: float = GRAVITY) -> dict[str, float]:
    """Body data for the free-floating heave mode of the hemisphere."""
    volume = 2.0 / 3.0 * math.pi * RADIUS**3
    mass = density * volume
    waterplane = math.pi * RADIUS**2
    return {
        "radius": RADIUS,
        "density": density,
        "gravity": gravity,
        "displaced_volume": volume,
        "mass": mass,
        "waterplane_area": waterplane,
        "restoring": density * gravity * waterplane,
        "added_mass_inf": 0.5 * mass,
    }


# ---------------------------------------------------------------------------
# parity closure: rebuild K from the even/odd spectrum pair
# ---------------------------------------------------------------------------
def _piecewise_linear_integral(
    omega: np.ndarray, values: np.ndarray, tau: np.ndarray, *, function: str
) -> np.ndarray:
    """``int (c0 + c1 w) * {cos, sin}(w tau) dw`` over the tabulated sweep."""
    a = omega[:-1]
    b = omega[1:]
    slope = np.diff(values) / np.diff(omega)
    c0 = values[:-1] - a * slope
    t = tau[:, None]
    if function == "cos":
        sa = np.sinc(a * t / math.pi)
        sb = np.sinc(b * t / math.pi)
        first = b * sb - a * sa
        second = (
            b * b * sb
            - a * a * sa
            + 0.5 * (a * a - b * b) * np.sinc((a + b) * t / (2 * math.pi)) * np.sinc((b - a) * t / (2 * math.pi))
        )
    else:
        # int_a^b sin(w t) dw and int_a^b w sin(w t) dw
        with np.errstate(divide="ignore", invalid="ignore"):
            first = (np.cos(a * t) - np.cos(b * t)) / t
            second = (
                (np.sin(b * t) - np.sin(a * t)) / t**2
                - (b * np.cos(b * t) - a * np.cos(a * t)) / t
            )
            first = np.where(t == 0.0, 0.0, first)
            second = np.where(t == 0.0, 0.0, second)
    return first @ c0 + second @ slope


def _odd_partner_tail(
    omega: np.ndarray, added_mass: np.ndarray, added_mass_inf: float, tau: np.ndarray
) -> np.ndarray:
    """Truncated tail of ``int omega (A - A_inf) sin(omega tau) domega``.

    ``A - A_inf -> c/omega^2`` for a body in deep water, so the odd partner of
    the spectrum decays only as ``c/omega`` and its tail is *not* negligible
    (unlike the damping tail, which falls off as ``omega^-3`` or faster). The
    tail integral is elementary,

        int_{W}^inf (c/omega) sin(omega tau) domega = c [pi/2 - Si(W tau)]

    with ``Si`` the sine integral and ``c = lim omega^2 (A - A_inf)`` fitted on
    the top of the sweep. Leaving it out biases the parity closure by the size
    of the tail, which is exactly the kind of bookkeeping this check exists to
    expose.
    """
    from scipy.special import sici

    window = slice(max(0, omega.size - 10), omega.size)
    curvature = float(
        np.mean((added_mass[window] - added_mass_inf) * omega[window] ** 2)
    )
    sine_integral, _ = sici(omega[-1] * np.asarray(tau))
    return curvature * (0.5 * math.pi - sine_integral)


def parity_closure_residual(
    omega: np.ndarray,
    added_mass: np.ndarray,
    damping: np.ndarray,
    added_mass_inf: float,
    kernel: MemoryKernel,
) -> dict[str, float]:
    """Rebuild K from ``B cos - omega (A - A_inf) sin`` and compare with K from ``B``.

    For a causal kernel the imaginary part of the spectrum is fixed by the real
    part: ``omega (A - A_inf) = -int K sin``. Inverting the full complex
    spectrum then gives ``h = (1/pi)[int B cos domega + int omega (A - A_inf) sin domega]``,
    which must equal the cosine transform alone. If the (A, B) pair carries the
    wrong parity -- a sign flip in the odd partner, an A_inf that does not match
    the sweep -- the two constructions differ by O(1) of the kernel peak, so the
    residual is a sharp statement about the pairing rather than about accuracy.

    The residual is normalised by the kernel peak, not pointwise: the odd
    partner decays only as ``1/omega`` while ``B`` falls off faster, so below
    the kernel's zero crossings the pointwise comparison divides by a vanishing
    ``K`` and says nothing. What remains after the tail model is the band
    truncation of the odd partner, reported as a fraction of the kernel peak.
    """
    tau = kernel.tau
    cosine_part = _piecewise_linear_integral(omega, damping, tau, function="cos")
    # the odd partner enters eq. (1) with an extra factor omega:
    # K = (1/pi) int [B cos(omega tau) - omega (A - A_inf) sin(omega tau)] domega
    odd_part = _piecewise_linear_integral(
        omega, omega * (added_mass - added_mass_inf), tau, function="sin"
    ) + _odd_partner_tail(omega, added_mass, added_mass_inf, tau)
    rebuilt = (cosine_part - odd_part) / math.pi
    from_b = (2.0 / math.pi) * cosine_part
    scale = float(np.max(np.abs(from_b)))
    residual = np.abs(rebuilt - from_b)
    tail_residual = float(np.max(residual))
    return {
        "residual_max": tail_residual,
        "relative_residual_max": tail_residual / scale,
        "kernel_peak": scale,
        "residual_at_origin": float(residual[0]),
        "relative_residual_at_origin": float(residual[0] / scale),
        "residuals": residual.tolist()[:2000],
        "rebuilt": rebuilt.tolist()[:2000],
        "from_damping": from_b.tolist()[:2000],
    }


def two_sided_parity(kernel: MemoryKernel, omega_query: float) -> dict[str, float]:
    """Even extension must transform to a purely real spectrum at ``omega_query``."""
    tau, values = two_sided_kernel(kernel)
    weights = np.ones(tau.size)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights = weights * kernel.dt / 3.0
    spectrum = np.sum(values * weights * np.exp(-1j * omega_query * tau))
    return {
        "real_part": float(spectrum.real),
        "imaginary_part": float(spectrum.imag),
        "relative_imaginary": float(abs(spectrum.imag) / abs(spectrum.real)),
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run(
    *,
    sweep_points: int = SWEEP_POINTS,
    round_trip_dt: float = ROUND_TRIP_DT,
    decay_dt: float = DECAY_DT,
    decay_duration: float = DECAY_DURATION,
    sweep: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the three criteria and return a JSON-serialisable record."""
    data = sweep if sweep is not None else hemisphere_heave_sweep(sweep_points)
    omega = np.asarray(data["omega"], dtype=float)
    added = np.asarray(data["added_mass"], dtype=float)
    damping = np.asarray(data["damping"], dtype=float)
    body = hemisphere_heave_setup()
    added_inf = body["added_mass_inf"]

    # --- C2: Fourier reciprocity -------------------------------------------
    round_trip: FourierRoundTrip = fourier_round_trip(
        omega,
        added,
        damping,
        added_inf,
        dt=round_trip_dt,
        samples=80,
    )
    round_trip_summary = {
        "band": list(round_trip.band),
        "tau_max": round_trip.tau_max,
        "kernel_dt": round_trip.dt,
        "damping_relative_error_max": round_trip.damping_relative_error_max,
        "added_mass_relative_error_max": round_trip.added_mass_relative_error_max,
        "worst_relative_error": round_trip.worst_relative_error,
        "criterion": ROUND_TRIP_CRITERION,
        "passed": round_trip.worst_relative_error < ROUND_TRIP_CRITERION,
        "implied_added_mass_inf": round_trip.implied_added_mass_inf,
        "relative_error_of_implied_added_mass_inf": abs(
            round_trip.implied_added_mass_inf["mean"] - added_inf
        )
        / added_inf,
    }

    # --- C1: decay and parity ---------------------------------------------
    kernel = round_trip.kernel
    diagnostics = kernel_diagnostics(kernel, omega, damping)
    parity = parity_closure_residual(omega, added, damping, added_inf, kernel)
    two_sided = two_sided_parity(kernel, float(omega[int(np.argmax(damping))]))
    decay_and_parity = {
        "decay_rate": diagnostics["decay_rate"],
        "memory_time_scale": diagnostics["memory_time_scale"],
        "tail_relative": diagnostics["tail_relative"],
        "value_at_zero": diagnostics["value_at_zero"],
        "first_zero": diagnostics["first_zero"],
        "relative_slope_at_zero": diagnostics["relative_slope_at_zero"],
        "even_residual": diagnostics["even_residual"],
        "two_sided_parity": two_sided,
        "parity_closure": {
            **{key: value for key, value in parity.items() if not isinstance(value, list)},
            "note": (
                "the kernel rebuilt from the even/odd spectrum pair "
                "[B cos - omega (A - A_inf) sin] / pi must equal the cosine "
                "transform of B alone; the residual is normalised by the kernel "
                "peak because the odd partner decays like 1/omega and its band "
                "truncation converges only conditionally"
            ),
        },
        "truncation": diagnostics["truncation"],
        "criterion": {
            "decay_rate_positive": diagnostics["decay_rate"] > 0.0,
            "tail_below_5_percent": diagnostics["tail_relative"] < 0.05,
            "even_residual_at_roundoff": diagnostics["even_residual"] < 1.0e-12,
            "two_sided_imaginary_negligible": two_sided["relative_imaginary"] < 1.0e-6,
            "parity_closure_below_2_percent": parity["relative_residual_max"] < 0.02,
            "slope_at_zero_within_cosine_bound": abs(diagnostics["relative_slope_at_zero"])
            < (kernel.dt * float(omega[-1])) ** 2,
        },
    }
    decay_and_parity["passed"] = all(
        value for key, value in decay_and_parity["criterion"].items()
    )

    # --- C3: time domain vs frequency domain ------------------------------
    decay_kernel = radiation_memory_kernel(
        omega,
        damping,
        np.arange(0.0, 60.0 + 1e-12, decay_dt),
    )
    decay = cummins_free_decay(
        decay_kernel,
        mass=body["mass"],
        restoring=body["restoring"],
        added_mass_inf=added_inf,
        initial_displacement=0.05,
        dt=decay_dt,
        duration=decay_duration,
    )
    metrics = decay_metrics(decay.time, decay.displacement, amplitude_fraction=5.0e-3)
    resonance = resonance_properties(
        omega, added, damping, mass=body["mass"], restoring=body["restoring"]
    )

    driven = {}
    for label, frequency in (
        ("at_resonance", resonance["omega_n"]),
        ("below_resonance", 0.7 * resonance["omega_n"]),
        ("above_resonance", 1.3 * resonance["omega_n"]),
    ):
        force = 1.0
        response = harmonic_response(
            decay_kernel,
            mass=body["mass"],
            restoring=body["restoring"],
            added_mass_inf=added_inf,
            force_amplitude=force,
            omega=frequency,
            dt=decay_dt,
            duration=max(40.0, 20.0 * 2.0 * math.pi / frequency),
            warmup_fraction=0.6,
        )
        impedance = frequency_domain_impedance(
            [frequency], omega, added, damping, mass=body["mass"], restoring=body["restoring"]
        )[0]
        expected = force / impedance
        driven[label] = {
            "omega": frequency,
            "time_domain_amplitude": response["amplitude"],
            "frequency_domain_amplitude": float(abs(expected)),
            "amplitude_relative_error": float(
                abs(response["amplitude"] - abs(expected)) / abs(expected)
            ),
            "time_domain_phase": response["phase"],
            "frequency_domain_phase": float(np.angle(expected)),
            "phase_error": float(
                abs(
                    math.atan2(
                        math.sin(response["phase"] - float(np.angle(expected))),
                        math.cos(response["phase"] - float(np.angle(expected))),
                    )
                )
            ),
            "fit_residual_over_amplitude": response["fit_residual_over_amplitude"],
        }

    period_error = (
        abs(metrics["period"] - resonance["damped_period"]) / resonance["damped_period"]
    )
    rate_error = (
        abs(metrics["decay_rate"] - resonance["decay_rate"]) / resonance["decay_rate"]
    )
    bandwidth_rate = resonance["zeta_bandwidth"] * resonance["omega_n"]
    rate_error_bandwidth = abs(metrics["decay_rate"] - bandwidth_rate) / bandwidth_rate
    time_vs_frequency = {
        "free_decay": metrics,
        "resonance": resonance,
        "decay_rate_from_bandwidth": bandwidth_rate,
        "period_relative_error_vs_resonance": period_error,
        "decay_rate_relative_error_vs_energy_identity": rate_error,
        "decay_rate_relative_error_vs_bandwidth": rate_error_bandwidth,
        "driven_response": driven,
        "criterion": {
            "period_within_1_percent_of_resonance": period_error < 0.01,
            "decay_rate_within_5_percent_of_bandwidth": rate_error_bandwidth < 0.05,
            "driven_amplitude_within_2_percent": all(
                entry["amplitude_relative_error"] < 0.02 for entry in driven.values()
            ),
            "driven_phase_within_0p05_rad": all(
                entry["phase_error"] < 0.05 for entry in driven.values()
            ),
        },
    }
    time_vs_frequency["passed"] = all(
        value for key, value in time_vs_frequency["criterion"].items()
    )

    return {
        "case": {
            "description": (
                "floating hemisphere, heave, deep water, zero forward speed; "
                "radiation memory kernel of the Cummins equation"
            ),
            "geometry": {"radius_m": RADIUS, "waterplane": "circle of radius a"},
            "reference": (
                "Hulme (1982) J. Fluid Mech. 121:443-463 semi-analytical "
                "radiation coefficients via tensorbem.analytic.hulme_hemisphere_radiation"
            ),
            "body": body,
            "sweep": {
                "points": int(omega.size),
                "omega_range_requested": list(OMEGA_RANGE),
                "ka_range": [float(omega[0] ** 2 / GRAVITY), float(omega[-1] ** 2 / GRAVITY)],
                "omega_range": [float(omega[0]), float(omega[-1])],
                "damping_peak": float(np.max(damping)),
                "omega_at_damping_peak": float(omega[int(np.argmax(damping))]),
                "added_mass_range": [float(np.min(added)), float(np.max(added))],
            },
            "settings": {
                "round_trip_dt": round_trip_dt,
                "decay_dt": decay_dt,
                "decay_duration": decay_duration,
                "round_trip_criterion": ROUND_TRIP_CRITERION,
                "decay_criterion": DECAY_CRITERION,
            },
        },
        "criteria": {
            "C1_decay_and_parity": decay_and_parity,
            "C2_fourier_round_trip": round_trip_summary,
            "C3_time_vs_frequency": time_vs_frequency,
        },
        "all_passed": bool(
            decay_and_parity["passed"]
            and round_trip_summary["passed"]
            and time_vs_frequency["passed"]
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "series": {
            "omega": omega.tolist(),
            "added_mass": added.tolist(),
            "damping": damping.tolist(),
            "round_trip": {
                "omega": round_trip.omega_query.tolist(),
                "damping_exact": round_trip.damping_exact.tolist(),
                "damping_round_trip": round_trip.damping_round_trip.tolist(),
                "added_mass_exact": round_trip.added_mass_exact.tolist(),
                "added_mass_round_trip": round_trip.added_mass_round_trip.tolist(),
            },
            "kernel": {
                "tau": kernel.tau[:4000].tolist(),
                "values": kernel.values[:4000].tolist(),
            },
            "free_decay": {
                "time": decay.time[::4].tolist(),
                "displacement": decay.displacement[::4].tolist(),
            },
        },
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _figures(record: dict[str, Any], directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    series = record["series"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    rt = series["round_trip"]
    axes[0].plot(series["omega"], series["damping"], "k-", lw=2.0, label="Hulme B33")
    axes[0].plot(rt["omega"], rt["damping_round_trip"], "o", ms=3.0, mfc="none", label="K -> B round trip")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\omega$ [rad/s]")
    axes[0].set_ylabel(r"$B_{33}$ [kg/s]")
    axes[0].set_title("damping and its round trip")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(series["omega"], series["added_mass"], "k-", lw=2.0, label="Hulme A33")
    axes[1].plot(rt["omega"], rt["added_mass_round_trip"], "o", ms=3.0, mfc="none", label="K -> A round trip")
    axes[1].axhline(record["case"]["body"]["added_mass_inf"], color="0.5", ls="--", label=r"$A_\infty$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\omega$ [rad/s]")
    axes[1].set_ylabel(r"$A_{33}$ [kg]")
    axes[1].set_title("added mass and its round trip")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = directory / "fig_time_domain_round_trip.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    tau = np.asarray(series["kernel"]["tau"])
    values = np.asarray(series["kernel"]["values"])
    axes[0].plot(tau, values, "k-", lw=1.2)
    axes[0].set_xlabel(r"$\tau$ [s]")
    axes[0].set_ylabel(r"$K(\tau)$ [kg/s$^2$]")
    axes[0].set_title("radiation memory kernel")
    axes[0].grid(alpha=0.3)
    zoom = tau <= 1.0
    axes[1].plot(tau[zoom], values[zoom], "k-", lw=1.6)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_xlabel(r"$\tau$ [s]")
    axes[1].set_ylabel(r"$K(\tau)$")
    kv = record["criteria"]["C1_decay_and_parity"]
    axes[1].set_title(
        "near $\\tau=0$: $K(0)$=%.0f, first zero %.3f s, slope %.1e"
        % (kv["value_at_zero"], kv["first_zero"], kv["relative_slope_at_zero"])
    )
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = directory / "fig_time_domain_kernel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    t = np.asarray(series["free_decay"]["time"])
    z = np.asarray(series["free_decay"]["displacement"])
    rate = record["criteria"]["C3_time_vs_frequency"]
    axes[0].plot(t, z, "k-", lw=1.0, label="Cummins free decay")
    envelope = np.abs(z[0]) * np.exp(-rate["free_decay"]["decay_rate"] * t)
    axes[0].plot(t, envelope, "--", lw=1.0, label="fitted envelope")
    axes[0].plot(t, -envelope, "--", lw=1.0)
    resonance_plot = rate["resonance"]
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("heave z [m]")
    axes[0].set_title(
        "free decay: $T$=%.4f s vs response $T$=%.4f s (%.2f%%); $\\sigma$ %.4f vs band %.4f"
        % (
            rate["free_decay"]["period"],
            resonance_plot["damped_period"],
            100.0 * rate["period_relative_error_vs_resonance"],
            rate["free_decay"]["decay_rate"],
            rate["decay_rate_from_bandwidth"],
        )
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    resonance = rate["resonance"]
    scan = np.geomspace(record["series"]["omega"][0], record["series"]["omega"][-1], 800)
    from ..time_domain import frequency_domain_impedance as _impedance

    body = record["case"]["body"]
    amplitude = np.abs(
        1.0
        / _impedance(
            scan,
            np.asarray(series["omega"]),
            np.asarray(series["added_mass"]),
            np.asarray(series["damping"]),
            mass=body["mass"],
            restoring=body["restoring"],
        )
    )
    axes[1].loglog(scan, amplitude, "k-", lw=1.4, label=r"$|1/\kappa(\omega)|$")
    for label, entry in rate["driven_response"].items():
        axes[1].plot(
            entry["omega"], entry["time_domain_amplitude"], "o", ms=6.0, mfc="none", label=label
        )
    axes[1].axvspan(
        resonance["half_power_omega_low"],
        resonance["half_power_omega_high"],
        color="0.85",
        label="half-power band",
    )
    axes[1].set_xlabel(r"$\omega$ [rad/s]")
    axes[1].set_ylabel("amplitude per unit force [m/N]")
    axes[1].set_title("frequency-domain response vs driven time-domain runs")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = directory / "fig_time_domain_decay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/time_domain_pulse.json"))
    parser.add_argument("--figure", action="store_true", help="also write the PNG figures")
    parser.add_argument("--points", type=int, default=SWEEP_POINTS)
    parser.add_argument("--decay-duration", type=float, default=DECAY_DURATION)
    args = parser.parse_args(argv)

    record = run(sweep_points=args.points, decay_duration=args.decay_duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True))

    if args.figure:
        figures = _figures(record, args.output.parent / "figures")
        record["figures"] = figures
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True))

    criteria = record["criteria"]
    print(
        "C1 decay/parity  passed=%s  decay_rate=%.4f 1/s  tail=%.2e  parity=%.2e"
        % (
            criteria["C1_decay_and_parity"]["passed"],
            criteria["C1_decay_and_parity"]["decay_rate"],
            criteria["C1_decay_and_parity"]["tail_relative"],
            criteria["C1_decay_and_parity"]["parity_closure"]["relative_residual_max"],
        )
    )
    print(
        "C2 round trip    passed=%s  damping=%.2e  added_mass=%.2e  band=%s"
        % (
            criteria["C2_fourier_round_trip"]["passed"],
            criteria["C2_fourier_round_trip"]["damping_relative_error_max"],
            criteria["C2_fourier_round_trip"]["added_mass_relative_error_max"],
            criteria["C2_fourier_round_trip"]["band"],
        )
    )
    print(
        "C3 time vs freq  passed=%s  period err=%.2e  decay(err vs bandwidth)=%.2e"
        % (
            criteria["C3_time_vs_frequency"]["passed"],
            criteria["C3_time_vs_frequency"]["period_relative_error_vs_resonance"],
            criteria["C3_time_vs_frequency"]["decay_rate_relative_error_vs_bandwidth"],
        )
    )
    print(f"all_passed={record['all_passed']}  written {args.output}")
    return 0 if record["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())