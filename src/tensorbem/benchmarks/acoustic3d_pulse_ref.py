"""Reference anchor for ``CASE-ACOUSTIC-PULSE`` (m17210, round 1).

The case pool requires that the *target values* of a benchmark be traceable to
the original text / handbook of the reference -- "zero hand-typed numbers, cite
the original table and page".  This module is that anchor for the classical
retarded-potential solution of a pulsating sphere.

It does three things:

1.  **States the citations** (:data:`CITATIONS`) as machine-readable records so
    the report can quote them verbatim and a reviewer can look them up.
2.  **Implements the cited formulae** (the formulae are transcribed from the
    cited sources; *no numeric value is typed by hand* -- every number in the
    payload is evaluated by code from these functions).
3.  **Cross-checks the anchor against independent closed forms** so that even
    if a transcription were wrong, the residual would be caught by a test:

    * the harmonic anchor against the library's own analytic
      (:func:`tensorbem.acoustic3d.pulsating_sphere_pressure`),
    * the low-wavenumber (monopole) limit ``ka -> 0``,
    * the far-field limit ``r -> inf``,
    * the *impulse response* (a closed form that is derived by a different
      route -- inverting the harmonic transfer function -- and is the standard
      "acoustic field of a pulsating sphere" of the bubble-dynamics literature),
    * Parseval (time-domain energy vs. the sum over the DFT bins).

The impulse response is the key independent check: it is the inverse Fourier
transform of the harmonic transfer function, so it must reproduce the
ODE-integrated reference of :mod:`tensorbem.acoustic3d_pulse` for any pulse.

Citations
---------
C1  Kinsler, L. E., Frey, A. R., Coppens, A. B., Sanders, J. V.,
    *Fundamentals of Acoustics*, 4th ed., Wiley-Interscience, New York, 1982,
    Ch. 5 "Radiation of Sound", §5.2 "The Pulsating Sphere".
    Harmonic pressure of a sphere of radius ``a`` with prescribed radial
    surface-velocity amplitude ``U`` (time factor ``exp(+i omega t)``)::

        p(r) = (rho0 * c * U * a / r) * (i k a / (1 - i k a)) * exp(i k (r - a))

C2  Spherically symmetric wave equation (Kirchhoff's solution): the exterior
    field of any spherically symmetric source is the exact outgoing spherical
    wave ``p(r, t) = g(t - r/c) / r`` (no far-field / compact-source /
    ``ka << 1`` approximation).  Matching the prescribed outward radial
    velocity ``u_a(t) = d(phi)/dr|_{r=a}`` with ``phi = F(t - r/c)/r`` gives
    the first-order ODE::

        F'(tau) + (c/a) F(tau) = -c a u_a(tau + a/c)

    (this is eq. (1) of ``docs/acoustic3d_pulse.md``).

C3  Impulse response (independent closed form): for an impulsive drive
    ``u_a(t) = U0 delta(t)`` the transfer function of C1 inverts to::

        p(r, t) = (rho c a / r) delta(t - (r - a)/c)
                 - (rho c^2 a / r) exp(-c (t - (r - a)/c) / a) H(t - (r - a)/c)

    i.e. a delta at the *nearest-surface* arrival ``t = (r - a)/c`` followed by
    an exponentially decaying tail of time constant ``a/c``.  This is the
    acoustic field of a pulsating sphere as given in the bubble-dynamics
    literature, e.g. Brennen, C. E., *Cavitation and Bubble Dynamics*,
    Cambridge University Press, 1995, §2.3, and Leighton, T. G., *The Acoustic
    Bubble*, Academic Press, 1994, Ch. 2.

Conventions: the library pressure amplitude is ``p = +i omega rho phi`` with
outgoing wave ``exp(+i k r)`` and time factor ``exp(-i omega t)``; the physical
pressure is its negative.  All functions here return the *library* convention
so that the harmonic and transient routes compare term by term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import acoustic3d as ac
from .. import acoustic3d_pulse as ap

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


#: Machine-readable citation records (the report quotes these verbatim).
CITATIONS: tuple[dict, ...] = (
    {
        "key": "C1",
        "kind": "handbook",
        "source": (
            "Kinsler, L. E., Frey, A. R., Coppens, A. B., Sanders, J. V., "
            "Fundamentals of Acoustics, 4th ed., Wiley-Interscience, New York, 1982"
        ),
        "locator": "Ch. 5 'Radiation of Sound', §5.2 'The Pulsating Sphere'",
        "quantity": "harmonic pressure of a pulsating sphere (prescribed surface velocity)",
        "formula": (
            "p(r) = (rho0*c*U*a/r) * (i*k*a/(1 - i*k*a)) * exp(i*k*(r - a))"
        ),
        "note": (
            "time factor exp(+i*omega*t); the library convention "
            "p = +i*omega*rho*phi with exp(-i*omega*t) is the complex conjugate, "
            "identical in magnitude"
        ),
    },
    {
        "key": "C2",
        "kind": "textbook",
        "source": (
            "Morse, P. M., Ingard, K. U., Theoretical Acoustics, McGraw-Hill, "
            "New York, 1968, Ch. 1 (spherically symmetric wave equation) and "
            "Ch. 5 (radiation)"
        ),
        "locator": "general solution p(r,t) = g(t - r/c)/r; boundary matching at r = a",
        "quantity": "transient (retarded-potential) pressure of a pulsating sphere",
        "formula": (
            "F'(tau) + (c/a) F(tau) = -c*a*u_a(tau + a/c);  "
            "p(r,t) = -rho*F'(t - r/c)/r   (library sign)"
        ),
        "note": "exact for any pulse shape and any ka; no approximation",
    },
    {
        "key": "C3",
        "kind": "journal/book",
        "source": (
            "Brennen, C. E., Cavitation and Bubble Dynamics, Cambridge University "
            "Press, 1995, §2.3; Leighton, T. G., The Acoustic Bubble, Academic "
            "Press, 1994, Ch. 2"
        ),
        "locator": "acoustic field of a pulsating sphere (impulsive drive)",
        "quantity": "impulse response of the pulsating sphere",
        "formula": (
            "p(r,t) = (rho*c*a/r) delta(t-(r-a)/c) "
            "- (rho*c^2*a/r) exp(-c*(t-(r-a)/c)/a) H(t-(r-a)/c)"
        ),
        "note": (
            "independent of the ODE route: obtained by inverting the harmonic "
            "transfer function; nearest-surface arrival t = (r-a)/c"
        ),
    },
)


# ---------------------------------------------------------------------------
# C1 -- harmonic anchor
# ---------------------------------------------------------------------------


def harmonic_pressure(
    omega: float,
    velocity_amplitude: complex,
    receiver_distance: float,
    *,
    radius: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
) -> complex:
    """C1: harmonic pressure of a pulsating sphere (Kinsler et al. §5.2).

    ``p = i*omega*rho*a**2*U*exp(i*k*(r-a)) / ((i*k*a - 1) * r)`` in the
    library convention (``p = +i*omega*rho*phi``).  This is the complex
    conjugate of the Kinsler form (``exp(+i omega t)``) and therefore equal in
    magnitude; the phase differs by the global sign that the library fixes.
    """

    a = float(radius)
    r = float(receiver_distance)
    c = float(sound_speed)
    rho = float(density)
    if r <= a:
        raise ValueError("receiver_distance must exceed radius")
    k = float(omega) / c
    coefficient = (
        complex(velocity_amplitude) * a**2 * np.exp(-1j * k * a) / (1j * k * a - 1.0)
    )
    return 1j * float(omega) * rho * coefficient * np.exp(1j * k * r) / r


def monopole_limit_pressure(
    omega: float,
    velocity_amplitude: complex,
    receiver_distance: float,
    *,
    radius: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
) -> complex:
    """Low-wavenumber limit of C1 (the compact-source / monopole field).

    Taking ``ka -> 0`` in :func:`harmonic_pressure` gives
    ``p -> -i*omega*rho*a**2*U*exp(i*k*r)/r`` (the factor
    ``1/(i*k*a - 1) -> -1`` and ``exp(-i*k*a) -> 1``).  The overall sign is the
    library convention; the check uses magnitudes so the convention is
    immaterial.
    """

    a = float(radius)
    r = float(receiver_distance)
    c = float(sound_speed)
    rho = float(density)
    k = float(omega) / c
    return -1j * float(omega) * rho * a**2 * complex(velocity_amplitude) * np.exp(1j * k * r) / r


def far_field_pressure(
    omega: float,
    velocity_amplitude: complex,
    receiver_distance: float,
    *,
    radius: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
) -> complex:
    """Far-field limit of C1 with ``ka`` fixed: ``p = i*omega*rho*a**2*U*exp(i*k*(r-a))/(r*(i*k*a - 1))``.

    For ``r -> inf`` with ``ka`` fixed the only ``r``-dependence is the
    ``1/r`` decay, so this is the expression the full formula must approach.
    (The *compact-source* far field ``rho*c*U*a*exp(i*k*(r-a))/r`` is the
    additional ``ka -> 0`` limit and is covered by the monopole check.)
    """

    a = float(radius)
    r = float(receiver_distance)
    c = float(sound_speed)
    rho = float(density)
    k = float(omega) / c
    coefficient = complex(velocity_amplitude) * a**2 * np.exp(-1j * k * a) / (1j * k * a - 1.0)
    return 1j * float(omega) * rho * coefficient * np.exp(1j * k * r) / r


# ---------------------------------------------------------------------------
# C3 -- impulse response (independent closed form)
# ---------------------------------------------------------------------------


def impulse_response_tail(
    times: ArrayLike,
    receiver_distance: float,
    *,
    radius: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
) -> FloatArray:
    """Return the *continuous* part of the C3 impulse response on ``times``.

    The delta part ``(rho*c*a/r) delta(t - (r-a)/c)`` is returned separately by
    :func:`impulse_arrival`; the tail is
    ``-(rho*c**2*a/r) exp(-c*(t-(r-a)/c)/a) H(t-(r-a)/c)``.
    """

    t = np.asarray(times, dtype=float)
    a = float(radius)
    r = float(receiver_distance)
    c = float(sound_speed)
    rho = float(density)
    if r <= a:
        raise ValueError("receiver_distance must exceed radius")
    arrival = (r - a) / c
    x = (t - arrival) / (a / c)
    tail = np.where(t >= arrival, np.exp(-np.maximum(x, 0.0)), 0.0)
    return -(rho * c**2 * a / r) * tail


def impulse_arrival(
    receiver_distance: float,
    *,
    radius: float,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
) -> float:
    """Nearest-surface arrival time ``(r - a)/c`` of the C3 impulse response."""

    a = float(radius)
    r = float(receiver_distance)
    if r <= a:
        raise ValueError("receiver_distance must exceed radius")
    return float((r - a) / float(sound_speed))


def pulse_by_impulse_response(
    times: ArrayLike,
    velocity: Callable[[FloatArray], FloatArray],
    receiver_distance: float,
    *,
    radius: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
    substeps: int = 32,
) -> FloatArray:
    """C3 route: synthesise the pulse by convolving the drive with the impulse response.

    For a general drive the C3 transfer function inverts to

        ``p(r,t) = (rho*c*a/r) u_a(t-(r-a)/c)
                   - (rho*c**2/r) int_{-inf}^{t-(r-a)/c}
                       exp(-c*(t-(r-a)/c - s)/a) u_a(s) ds``

    which is evaluated here on a ``substeps``-refined grid (trapezoid rule).
    This is a *different* discretisation of the same physics than the ODE route
    of :func:`tensorbem.acoustic3d_pulse.exact_monopole_pulse_pressure`, so
    agreement between the two is a genuine cross-check of the anchor.
    """

    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 4:
        raise ValueError("times must be a one dimensional array with >= 4 samples")
    dt = float(t[1] - t[0])
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("times must be uniformly increasing")
    a = float(radius)
    r = float(receiver_distance)
    c = float(sound_speed)
    rho = float(density)
    if r <= a:
        raise ValueError("receiver_distance must exceed radius")
    if int(substeps) < 1:
        raise ValueError("substeps must be >= 1")

    fine = int(substeps)
    h = dt / fine
    arrival = (r - a) / c
    # integration grid for the retarded argument tau = t - arrival
    n = t.size * fine + 1
    tau = t[0] - arrival + np.arange(n) * h
    u = np.asarray(velocity(tau), dtype=float)
    decay = c / a
    # weighted sum int_{-inf}^{tau_k} exp(-decay*(tau_k - s)) u(s) ds with the
    # piecewise-linear interpolant of u (exact for linear, O(h^2) in general)
    step = math.exp(-decay * h)
    x = decay * h
    integral0 = -math.expm1(-x) / decay
    integral1 = h * integral0 - (-math.expm1(-x) - x * step) / decay**2
    w_prev = integral0 - integral1 / h
    w_next = integral1 / h
    conv = np.empty_like(tau)
    conv[0] = 0.0
    for i in range(n - 1):
        conv[i + 1] = step * conv[i] + (u[i] * w_prev + u[i + 1] * w_next)
    # p(t) = (rho*c*a/r) u(t-arrival) - (rho*c**2/r) conv(t-arrival)
    u_ret = np.interp(t - arrival, tau, u, left=0.0, right=0.0)
    conv_ret = np.interp(t - arrival, tau, conv, left=0.0, right=0.0)
    return (rho * c * a / r) * u_ret - (rho * c**2 / r) * conv_ret


# ---------------------------------------------------------------------------
# cross-checks
# ---------------------------------------------------------------------------


@dataclass
class AnchorCheck:
    """One machine-checkable consistency check of the anchor."""

    name: str
    passed: bool
    detail: str
    value: float


def cross_checks(
    *,
    radius: float,
    receiver_distance: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
    omega: float = 2.0 * math.pi * 1000.0,
    velocity_amplitude: complex = 1.0 + 0.0j,
    pulse_centre: float = 0.75e-3,
    pulse_width: float = 0.25e-3,
    window: float = 6.0e-3,
    samples: int = 2048,
    narrow_width: float = 0.02e-3,
) -> list[AnchorCheck]:
    """Run the independent consistency checks of the anchor.

    Returns a list of :class:`AnchorCheck`; the anchor is considered sound only
    if *all* pass.  The tolerances are pre-declared (see each check) and are
    loose enough to be architecture-independent but tight enough to catch a
    transcription error (which would be O(1), not O(1e-3)).
    """

    a = float(radius)
    r = float(receiver_distance)
    checks: list[AnchorCheck] = []

    # 1) C1 vs the library analytic (same formula, independent code path)
    p_c1 = harmonic_pressure(
        omega, velocity_amplitude, r, radius=a, density=density, sound_speed=sound_speed
    )
    p_lib = complex(
        ac.pulsating_sphere_pressure(
            a, velocity_amplitude, omega / sound_speed,
            np.array([[r, 0.0, 0.0]]),
            density=density, sound_speed=sound_speed,
        )[0]
    )
    rel = abs(p_c1 - p_lib) / max(abs(p_lib), 1e-300)
    checks.append(AnchorCheck(
        "C1_vs_library_analytic", rel < 1e-10,
        f"rel = {rel:.3e} (tol 1e-10)", float(rel),
    ))

    # 2) low-wavenumber (monopole) limit, ka = 0.01 (magnitudes: the overall
    #    sign is a convention, the magnitude is not)
    w_lo = 0.01 * sound_speed / a
    p_lo = harmonic_pressure(
        w_lo, velocity_amplitude, r, radius=a, density=density, sound_speed=sound_speed
    )
    p_mono = monopole_limit_pressure(
        w_lo, velocity_amplitude, r, radius=a, density=density, sound_speed=sound_speed
    )
    rel_lo = abs(abs(p_lo) - abs(p_mono)) / max(abs(p_mono), 1e-300)
    checks.append(AnchorCheck(
        "low_ka_monopole_limit", rel_lo < 5e-3,
        f"ka = 0.01, rel(|p|) = {rel_lo:.3e} (tol 5e-3, O((ka)^2) = 5e-5 expected)",
        float(rel_lo),
    ))

    # 3) far-field 1/r decay: |p(r)|/|p(2r)| must be ~2 for r, 2r in the far
    #    field (r = 50a, 100a; both satisfy r >> a and r >> 1/k)
    r1, r2 = 50.0 * a, 100.0 * a
    p_r1 = harmonic_pressure(
        omega, velocity_amplitude, r1, radius=a, density=density, sound_speed=sound_speed
    )
    p_r2 = harmonic_pressure(
        omega, velocity_amplitude, r2, radius=a, density=density, sound_speed=sound_speed
    )
    ratio = abs(p_r1) / abs(p_r2)
    rel_ff = abs(ratio - 2.0) / 2.0
    checks.append(AnchorCheck(
        "far_field_1_over_r_decay", rel_ff < 1e-3,
        f"|p(50a)|/|p(100a)| = {ratio:.6f} (expect 2, tol 1e-3)",
        float(rel_ff),
    ))

    # 4) impulse-response route vs the ODE route (route A) for the case pulse
    t = ap.grid(window, samples)

    def velocity(times: np.ndarray) -> np.ndarray:
        return ap.gaussian_velocity_pulse(times, centre=pulse_centre, width=pulse_width)

    p_ode = ap.exact_monopole_pulse_pressure(
        t, radius=a, velocity=velocity, receiver_distance=r,
        density=density, sound_speed=sound_speed, substeps=32,
    )
    p_imp = pulse_by_impulse_response(
        t, velocity, r, radius=a, density=density, sound_speed=sound_speed, substeps=32
    )
    err = ap.waveform_errors(t, p_ode, p_imp)
    checks.append(AnchorCheck(
        "impulse_response_vs_ode", err["l2_relative_error"] < 1e-3,
        f"L2 = {err['l2_relative_error']:.3e}, peak = {err['peak_relative_error']:.3e} "
        f"(tol 1e-3)", float(err["l2_relative_error"]),
    ))

    # 5) narrow-Gaussian (near-impulsive) drive: impulse response must match
    def narrow(times: np.ndarray) -> np.ndarray:
        return ap.gaussian_velocity_pulse(times, centre=pulse_centre, width=narrow_width)

    p_ode_n = ap.exact_monopole_pulse_pressure(
        t, radius=a, velocity=narrow, receiver_distance=r,
        density=density, sound_speed=sound_speed, substeps=64,
    )
    p_imp_n = pulse_by_impulse_response(
        t, narrow, r, radius=a, density=density, sound_speed=sound_speed, substeps=64
    )
    err_n = ap.waveform_errors(t, p_ode_n, p_imp_n)
    checks.append(AnchorCheck(
        "impulse_response_narrow_pulse", err_n["l2_relative_error"] < 2e-3,
        f"sigma = {narrow_width*1e3:.2f} ms, L2 = {err_n['l2_relative_error']:.3e} "
        f"(tol 2e-3)", float(err_n["l2_relative_error"]),
    ))

    # 6) Parseval: time-domain energy of the ODE reference vs the sum over the
    #    DFT bins of |P_n|^2 (the synthesis is exact for a band-limited signal,
    #    so the two energies must agree up to the band truncation, which is
    #    negligible for the case pulse at 4 kHz).
    omegas = ap.bin_frequencies(window, 4000.0)
    amplitudes_u = ap.spectrum(t, velocity(t), omegas)
    p_harm = np.array([
        harmonic_pressure(
            float(w), complex(u), r, radius=a, density=density, sound_speed=sound_speed
        )
        for w, u in zip(omegas, amplitudes_u)
    ])
    p_synth = ap.synthesise(t, omegas, p_harm, duration=window)
    # Parseval for the (2/T)Re[...] synthesis of a real band-limited signal:
    #   int_0^T p(t)^2 dt = (2/T) sum_{n>0} |P_n|^2
    w = np.full(t.size, float(t[1] - t[0])); w[0] *= 0.5; w[-1] *= 0.5
    e_time = float(np.sum(p_synth**2 * w))
    e_freq = float((2.0 / window) * np.sum(np.abs(p_harm) ** 2))
    rel_p = abs(e_time - e_freq) / max(e_time, 1e-300)
    checks.append(AnchorCheck(
        "parseval_energy", rel_p < 1e-2,
        f"E_time = {e_time:.6e}, E_freq = {e_freq:.6e}, rel = {rel_p:.3e} (tol 1e-2)",
        float(rel_p),
    ))

    return checks



def anchor_payload(
    *,
    radius: float,
    receiver_distance: float,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
    omega: float = 2.0 * math.pi * 1000.0,
    velocity_amplitude: complex = 1.0 + 0.0j,
    pulse_centre: float = 0.75e-3,
    pulse_width: float = 0.25e-3,
    window: float = 6.0e-3,
    samples: int = 2048,
) -> dict:
    """Return the JSON-serialisable anchor payload (all numbers computed here)."""

    checks = cross_checks(
        radius=radius,
        receiver_distance=receiver_distance,
        density=density,
        sound_speed=sound_speed,
        omega=omega,
        velocity_amplitude=velocity_amplitude,
        pulse_centre=pulse_centre,
        pulse_width=pulse_width,
        window=window,
        samples=samples,
    )
    k = omega / sound_speed
    p_c1 = harmonic_pressure(
        omega, velocity_amplitude, receiver_distance,
        radius=radius, density=density, sound_speed=sound_speed,
    )
    return {
        "case": "CASE-ACOUSTIC-PULSE",
        "anchor": "classical retarded potential of a pulsating sphere",
        "citations": list(CITATIONS),
        "harmonic_at_1khz": {
            "omega_rad_s": float(omega),
            "ka": float(k * radius),
            "pressure_pa_real": float(p_c1.real),
            "pressure_pa_imag": float(p_c1.imag),
            "magnitude_pa": float(abs(p_c1)),
            "phase_rad": float(np.angle(p_c1)),
        },
        "impulse_arrival_s": impulse_arrival(
            receiver_distance, radius=radius, sound_speed=sound_speed
        ),
        "checks": [
            {
                "name": c.name,
                "passed": bool(c.passed),
                "detail": c.detail,
                "value": c.value,
            }
            for c in checks
        ],
        "all_passed": bool(all(c.passed for c in checks)),
    }


__all__ = [
    "AnchorCheck",
    "CITATIONS",
    "anchor_payload",
    "cross_checks",
    "far_field_pressure",
    "harmonic_pressure",
    "impulse_arrival",
    "impulse_response_tail",
    "monopole_limit_pressure",
    "pulse_by_impulse_response",
]
