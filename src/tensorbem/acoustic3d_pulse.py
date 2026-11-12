"""Transient (pulse) acoustics from the frequency-domain 3-D Helmholtz BEM.

Ledger anchor: ``CASE-ACOUSTIC-PULSE`` (case pool, reference = literature,
tolerance ``< 8 %``).

``tensorbem.acoustic3d`` is a *frequency-domain* exterior Helmholtz BEM.  Real
underwater-noise problems are broadband: a machinery knock, a propeller
cavitation collapse or a sonar ping reaches the receiver as a **pulse**, and the
quantity of interest is the waveform (peak level and arrival time), not a single
harmonic amplitude.  Until now the repository had no bridge from the harmonic
solver to the time domain and no transient reference to check such a bridge
against.

This module supplies three things.

1.  **A pulse definition** (:func:`gaussian_velocity_pulse`) -- a smooth
    Gaussian radial-velocity pulse on a closed radiating sphere.  The sphere is
    the only exterior acoustic geometry with a *closed-form* transient solution,
    which is exactly why it is the right benchmark body.
2.  **An exact transient reference** (:func:`exact_monopole_pulse_pressure`).
    The spherically symmetric wave equation has the general solution
    ``p(r, t) = rho * F'(t - r/c) / r`` for every ``r > 0`` -- *no* far-field,
    compact-source or ``ka << 1`` approximation is involved.  Matching the
    prescribed radial velocity at ``r = a`` turns the problem into the first
    order ODE

        ``F'(tau) + (c/a) F(tau) = -c * a * u_a(tau + a/c)``

    solved here on a refined grid with an exponential integrating factor, so
    the reference is accurate for *any* pulse shape and *any* ``ka``.
3.  **The BEM bridge** (:func:`synthesise_bem_pulse`).  The pulse spectrum is
    sampled on the discrete bins ``omega_n = 2*pi*n/T`` of the analysis window;
    each bin is an ordinary :func:`tensorbem.acoustic3d.solve_exterior_neumann`
    solve followed by
    :func:`tensorbem.acoustic3d.field_potential` at the receiver.  Because the
    window contains the whole pulse, the bin synthesis is *exact* for a
    band-limited signal, so the residual error is BEM discretisation error --
    which is what we want to measure.

Waveform metrics (:func:`waveform_errors`) are peak amplitude error, arrival
time shift and the normalised ``L2`` waveform error; the case acceptance uses
the ``L2`` error and the peak error, both against the ``< 8 %`` tolerance.

Conventions (same as :mod:`tensorbem.acoustic3d`): outgoing wave ``exp(+i k r)``,
synthesis from complex amplitudes with ``Re[A exp(-i omega t)]``, pressure
``p = i*omega*rho*phi``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import acoustic3d as ac
from .geometry3d import TriangleMesh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# pulse definition
# ---------------------------------------------------------------------------


def gaussian_velocity_pulse(
    times: ArrayLike,
    *,
    amplitude: float = 1.0,
    centre: float,
    width: float,
) -> FloatArray:
    """Return ``u_a(t) = amplitude * exp(-((t - centre)/width)**2)``.

    The Gaussian is the standard broadband test pulse: it is smooth (so the
    discrete spectrum decays fast and the bin truncation is negligible), it has
    a well defined peak, and its spectral width is set by ``width`` alone.
    """

    t = np.asarray(times, dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be one dimensional")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("width must be positive and finite")
    if not math.isfinite(centre):
        raise ValueError("centre must be finite")
    if not math.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    return float(amplitude) * np.exp(-(((t - float(centre)) / float(width)) ** 2))


# ---------------------------------------------------------------------------
# exact transient reference (retarded potential)
# ---------------------------------------------------------------------------


def exact_monopole_pulse_pressure(
    times: ArrayLike,
    *,
    radius: float,
    velocity: Callable[[FloatArray], FloatArray],
    receiver_distance: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
    substeps: int = 16,
) -> FloatArray:
    """Exact pressure pulse of a radially pulsating sphere.

    Derivation (reproduced in ``docs/acoustic3d_pulse.md``): for the
    spherically symmetric wave equation the pressure is
    ``p(r, t) = rho * F'(t - r/c) / r`` with ``F`` fixed by the surface
    condition ``u_a = d(phi)/dr`` at ``r = a``::

        F'(tau) + (c/a) F(tau) = -c * a * u_a(tau + a/c).

    The ODE is integrated with the exact integrating factor on ``substeps``
    refined intervals per output interval (trapezoid rule for the convolution
    integral), so the discretisation error is ``O(dt**2)`` and is verified
    against a numerical inverse transform in the test suite.

    Parameters
    ----------
    times:
        Uniform output time grid (start time is taken as ``t = 0`` reference
        for the retarded argument).
    radius:
        Sphere radius ``a``.
    velocity:
        Callable ``u_a(t)`` giving the outward radial surface velocity.
    receiver_distance:
        Receiver range ``r`` (must exceed ``radius``).
    substeps:
        Refinement of the integration grid relative to ``times``.
    """

    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 4:
        raise ValueError("times must be a one dimensional array with >= 4 samples")
    dt = float(t[1] - t[0])
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("times must be uniformly increasing")
    if np.any(np.abs(np.diff(t) - dt) > 1e-9 * max(1.0, abs(dt))):
        raise ValueError("times must be uniformly spaced")
    a = float(radius)
    c = float(sound_speed)
    r = float(receiver_distance)
    rho = float(density)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not math.isfinite(r) or r <= a:
        raise ValueError("receiver_distance must be finite and greater than radius")
    if not math.isfinite(c) or c <= 0.0:
        raise ValueError("sound_speed must be positive and finite")
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)):
        raise ValueError("substeps must be an integer")
    if int(substeps) < 1:
        raise ValueError("substeps must be >= 1")

    fine = int(substeps)
    h = dt / fine
    # Integration grid covering the retarded source window
    # tau in [tau_min, tau_max] with tau = t - a/c in the shifted convention:
    # u_a is evaluated at s + a/c, so we need s from t[0] - a/c to t[-1] - a/c.
    tau = t[0] + np.arange(t.size * fine + 1) * h  # argument of F
    source_times = tau + a / c  # physical time at which u_a is sampled
    u_samples = np.asarray(velocity(source_times), dtype=float)
    if u_samples.shape != source_times.shape:
        raise ValueError("velocity must return one value per input time")

    decay = c / a
    # Integrating factor: F(tau_{k+1}) = e^{-decay*h} F(tau_k)
    #                                    - c*a*int u e^{-decay*(tau_{k+1}-s)} ds
    # The integral is evaluated exactly for the piecewise-linear interpolant of
    # u on the refined grid, which keeps the scheme second order and stable for
    # any decay*h (a pure trapezoid rule is used in the limit decay*h -> 0).
    step = math.exp(-decay * h)
    x = decay * h
    integral0 = -math.expm1(-x) / decay  # int_0^h e^{-decay*(h-s)} ds
    integral1 = h * integral0 - (-math.expm1(-x) - x * step) / decay**2  # int_0^h s e^{-...} ds
    weight_prev = integral0 - integral1 / h
    weight_next = integral1 / h
    f = np.empty_like(tau)
    f[0] = 0.0  # the pulse starts from rest: F -> 0 before the pulse
    for k in range(tau.size - 1):
        increment = -c * a * (u_samples[k] * weight_prev + u_samples[k + 1] * weight_next)
        f[k + 1] = step * f[k] + increment
    fprime = -c * a * u_samples - decay * f

    # p(r, t) = rho * F'(t - r/c) / r  -> interpolate F' on the retarded grid
    retarded = t - r / c
    fprime_retarded = np.interp(retarded, tau, fprime, left=0.0, right=0.0)
    # The library pressure convention is p = +i*omega*rho*phi while the physical
    # pressure of a velocity potential with time factor exp(-i*omega*t) is
    # rho*d(phi)/dt = -i*omega*rho*phi, so the library pressure is the negative
    # of rho*F'(t - r/c)/r.  We return the library convention so that the
    # harmonic and transient routes can be compared term by term.
    return -rho * fprime_retarded / r


# ---------------------------------------------------------------------------
# discrete spectrum of a windowed pulse
# ---------------------------------------------------------------------------


def bin_frequencies(duration: float, f_max: float) -> FloatArray:
    """Return the positive DFT bin angular frequencies ``2*pi*n/T``.

    ``n = 1 .. floor(f_max*T)``: the window ``T`` must contain the whole pulse so
    that the bins describe it exactly (periodic extension is then legitimate).
    """

    T = float(duration)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError("duration must be positive and finite")
    top = float(f_max)
    if not math.isfinite(top) or top <= 0.0:
        raise ValueError("f_max must be positive and finite")
    n_max = int(math.floor(top * T))
    if n_max < 1:
        raise ValueError("f_max * duration must be at least 1 bin")
    return 2.0 * math.pi * np.arange(1, n_max + 1, dtype=float) / T


def spectrum(times: ArrayLike, signal: ArrayLike, omegas: ArrayLike) -> ComplexArray:
    """Return ``U(omega) = int u(t) exp(+i omega t) dt`` by the trapezoid rule.

    ``U`` is the complex amplitude multiplying ``exp(-i omega t)`` in the
    expansion of the real signal, ``u(t) = (2/T) Re[sum_n U_n exp(-i omega_n t)]``
    on the DFT bins of the window ``T``.  It is therefore exactly the amplitude
    that :mod:`tensorbem.acoustic3d` expects: its outgoing kernel is
    ``exp(+i k r)`` with time factor ``exp(-i omega t)``.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(signal, dtype=float)
    w = np.asarray(omegas, dtype=float)
    if t.ndim != 1 or x.shape != t.shape:
        raise ValueError("times and signal must be one dimensional and equally long")
    if w.ndim != 1 or w.size == 0:
        raise ValueError("omegas must be a non empty one dimensional array")
    dt = float(t[1] - t[0])
    weights = np.full(t.size, dt)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return np.exp(1j * np.outer(w, t)) @ (x * weights)


def synthesise(
    times: ArrayLike,
    omegas: ArrayLike,
    amplitudes: ArrayLike,
    *,
    duration: float | None = None,
) -> FloatArray:
    """Return ``(2/T) Re[ sum_n A_n exp(-i omega_n t) ]`` for a real signal.

    The amplitudes are the *continuous* Fourier transforms of one period ``T``
    of a signal whose energy is contained in that window, sampled on the DFT
    bins ``omega_n = 2*pi*n/T``.  For such a signal the sum is the exact Fourier
    series of the band-limited signal, so no information is lost by synthesising
    on bins instead of a continuum of frequencies -- which is what makes the
    frequency-domain BEM applicable to a transient pulse at all.
    """

    t = np.asarray(times, dtype=float)
    w = np.asarray(omegas, dtype=float)
    A = np.asarray(amplitudes, dtype=np.complex128)
    if A.shape != w.shape:
        raise ValueError("amplitudes must have one entry per frequency")
    period = float(duration) if duration is not None else float(t[-1] - t[0] + (t[1] - t[0]))
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("duration must be positive and finite")
    return (2.0 / period) * np.real(np.exp(-1j * np.outer(t, w)) @ A)


# ---------------------------------------------------------------------------
# BEM bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BemPulseSolution:
    """Transient solution assembled from frequency-domain BEM bins."""

    times: FloatArray
    pressure: FloatArray
    omegas: FloatArray
    bem_amplitudes: ComplexArray
    exact_amplitudes: ComplexArray
    receiver: FloatArray
    duration: float
    mesh_panel_count: int
    per_bin_relative_error: FloatArray

    @property
    def max_per_bin_error(self) -> float:
        return float(np.max(self.per_bin_relative_error))

    def harmonic_error_table(self) -> list[dict[str, float]]:
        """Per-bin error table (frequency, ka, |p_bem|, error) for archiving."""

        rows: list[dict[str, float]] = []
        for n, omega in enumerate(self.omegas):
            rows.append(
                {
                    "bin": float(n + 1),
                    "frequency_hz": float(omega / (2.0 * math.pi)),
                    "bem_amplitude": float(abs(self.bem_amplitudes[n])),
                    "exact_amplitude": float(abs(self.exact_amplitudes[n])),
                    "relative_error": float(self.per_bin_relative_error[n]),
                }
            )
        return rows


def synthesise_bem_pulse(
    mesh: TriangleMesh,
    times: ArrayLike,
    *,
    radius: float,
    velocity: Callable[[FloatArray], FloatArray],
    receiver: ArrayLike,
    omegas: ArrayLike,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
    n_chief: int = 5,
) -> BemPulseSolution:
    """Radiate a radial-velocity pulse from a sphere with the Helmholtz BEM.

    Every bin ``omega_n`` prescribes the complex surface velocity ``U(omega_n)``
    as Neumann data, solves the exterior problem and evaluates the pressure at
    the receiver.  The bins are then synthesised into the time domain.

    Parameters
    ----------
    mesh:
        Closed triangulated sphere of radius ``radius`` (e.g.
        ``tensorbem.geometry3d.icosphere``).
    times:
        Uniform output time grid.
    radius:
        Physical sphere radius (must match ``mesh``).
    velocity:
        Callable ``u_a(t)`` -- the outward radial surface velocity.
    receiver:
        Receiver point, shape ``(3,)`` or ``(1, 3)``; must be outside the mesh.
    omegas:
        Positive bin angular frequencies, e.g. from :func:`bin_frequencies`.
    """

    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 8:
        raise ValueError("times must be one dimensional with >= 8 samples")
    w = np.asarray(omegas, dtype=float)
    if w.ndim != 1 or w.size == 0 or np.any(w <= 0.0):
        raise ValueError("omegas must be a non empty array of positive frequencies")
    points = np.asarray(receiver, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.shape != (1, 3):
        raise ValueError("receiver must be a single 3-D point")
    if np.linalg.norm(points[0]) <= float(radius):
        raise ValueError("receiver must lie outside the sphere")

    u_samples = np.asarray(velocity(t), dtype=float)
    amplitudes_u = spectrum(t, u_samples, w)

    c = float(sound_speed)
    rho = float(density)
    bem = np.empty(w.size, dtype=np.complex128)
    exact = np.empty(w.size, dtype=np.complex128)
    for n, omega in enumerate(w):
        k = float(omega) / c
        solution = ac.solve_exterior_neumann(
            mesh, k, complex(amplitudes_u[n]), n_chief=n_chief
        )
        phi = ac.field_potential(
            mesh, k, solution.phi, solution.normal_velocity, points
        )
        bem[n] = complex(ac.acoustic_pressure(phi[0], k, density=rho, sound_speed=c))
        exact[n] = complex(
            ac.pulsating_sphere_pressure(
                float(radius), complex(amplitudes_u[n]), k, points,
                density=rho, sound_speed=c,
            )[0]
        )

    pressure = synthesise(t, w, bem, duration=float(t[-1] - t[0] + (t[1] - t[0])))
    scale = np.abs(exact)
    per_bin = np.abs(bem - exact) / np.where(scale > 0.0, scale, 1.0)
    return BemPulseSolution(
        times=t,
        pressure=pressure,
        omegas=w,
        bem_amplitudes=bem,
        exact_amplitudes=exact,
        receiver=points[0].copy(),
        duration=float(t[-1] - t[0]),
        mesh_panel_count=int(mesh.panel_count),
        per_bin_relative_error=per_bin,
    )


# ---------------------------------------------------------------------------
# waveform metrics
# ---------------------------------------------------------------------------


def arrival_time(times: ArrayLike, pressure: ArrayLike) -> float:
    """Return the time of the largest absolute pressure sample."""

    t = np.asarray(times, dtype=float)
    p = np.asarray(pressure, dtype=float)
    if p.shape != t.shape:
        raise ValueError("times and pressure must be equally long")
    return float(t[int(np.argmax(np.abs(p)))])


def waveform_errors(times: ArrayLike, reference: ArrayLike, candidate: ArrayLike) -> dict[str, float]:
    """Compare two waveforms: peak error, arrival shift, normalised L2, correlation.

    ``peak_relative_error`` is ``||cand|_inf - |ref|_inf| / |ref|_inf``;
    ``l2_relative_error`` is ``||cand - ref||_2 / ||ref||_2`` over the whole
    window (the acceptance metric for ``CASE-ACOUSTIC-PULSE``);
    ``arrival_shift`` is the peak-time difference in seconds.
    """

    t = np.asarray(times, dtype=float)
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != t.shape or cand.shape != t.shape:
        raise ValueError("times, reference and candidate must be equally long")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("waveforms must be finite")
    ref_peak = float(np.max(np.abs(ref)))
    if ref_peak <= 0.0:
        raise ValueError("reference waveform is identically zero")
    cand_peak = float(np.max(np.abs(cand)))
    ref_norm = float(np.linalg.norm(ref))
    diff = cand - ref
    correlation = float(
        np.dot(ref, cand) / max(np.linalg.norm(ref) * np.linalg.norm(cand), 1e-300)
    )
    return {
        "peak_relative_error": abs(cand_peak - ref_peak) / ref_peak,
        "l2_relative_error": float(np.linalg.norm(diff)) / ref_norm,
        "arrival_shift": arrival_time(t, cand) - arrival_time(t, ref),
        "correlation": correlation,
    }


def exact_pulse_amplitude(
    omega: float,
    *,
    radius: float,
    velocity_amplitude: complex,
    receiver_distance: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
) -> complex:
    """Closed-form harmonic pressure of a pulsating sphere at range ``r``.

    ``p = -i*omega*rho*a**2*U*exp(i*k*(r-a)) / ((1 - i*k*a) * r)`` -- the
    frequency-domain partner of :func:`exact_monopole_pulse_pressure`, obtained
    from ``p = i*omega*rho*C*exp(i*k*r)/r`` with
    ``C = U*a**2*exp(-i*k*a)/(i*k*a - 1)``.
    """

    k = float(omega) / float(sound_speed)
    r = float(receiver_distance)
    if r <= float(radius):
        raise ValueError("receiver_distance must exceed radius")
    coefficient = complex(velocity_amplitude) * float(radius) ** 2 * np.exp(-1j * k * float(radius)) / (1j * k * float(radius) - 1.0)
    return 1j * float(omega) * float(density) * coefficient * np.exp(1j * k * r) / r


def dominant_frequency(times: ArrayLike, signal: ArrayLike) -> float:
    """Return the frequency of the largest spectral line (Hz)."""

    t = np.asarray(times, dtype=float)
    x = np.asarray(signal, dtype=float)
    if x.shape != t.shape:
        raise ValueError("times and signal must be equally long")
    dt = float(t[1] - t[0])
    spectrum_ = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, dt)
    return float(freqs[int(np.argmax(spectrum_))])


def pulse_energy(pressure: ArrayLike, times: ArrayLike) -> float:
    """Return ``int p**2 dt`` (proportional to pulse energy) in SI units."""

    t = np.asarray(times, dtype=float)
    p = np.asarray(pressure, dtype=float)
    if p.shape != t.shape:
        raise ValueError("times and pressure must be equally long")
    dt = float(t[1] - t[0])
    weights = np.full(t.size, dt)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return float(np.sum(p * p * weights))


def grid(duration: float, samples: int) -> FloatArray:
    """Uniform time grid ``[0, duration)`` with ``samples`` points."""

    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be positive and finite")
    n = int(samples)
    if n < 8:
        raise ValueError("samples must be >= 8")
    return np.arange(n, dtype=float) * (float(duration) / n)


__all__ = [
    "BemPulseSolution",
    "arrival_time",
    "bin_frequencies",
    "dominant_frequency",
    "exact_monopole_pulse_pressure",
    "exact_pulse_amplitude",
    "gaussian_velocity_pulse",
    "grid",
    "pulse_energy",
    "spectrum",
    "synthesise",
    "synthesise_bem_pulse",
    "waveform_errors",
]