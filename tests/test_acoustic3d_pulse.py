"""Verification of the transient acoustic pulse route (``CASE-ACOUSTIC-PULSE``).

The frequency-domain Helmholtz BEM (:mod:`tensorbem.acoustic3d`) is checked here
against the *closed-form transient* solution of a pulsating sphere, i.e. against
a time-domain reference that involves no far-field or ``ka << 1`` approximation:

* route A -- retarded potential, first order ODE in retarded time
  (:func:`tensorbem.acoustic3d_pulse.exact_monopole_pulse_pressure`);
* route B -- exact harmonic amplitude synthesised on the DFT bins (validates the
  synthesis machinery, no BEM);
* route C -- BEM harmonic solves synthesised on the same bins (capability under
  test, acceptance ``L2 < 8 %``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import acoustic3d as ac
from tensorbem import acoustic3d_pulse as ap
from tensorbem.geometry3d import icosphere

RADIUS = 0.1
SOUND_SPEED = 1500.0
DENSITY = 1025.0
RECEIVER_RANGE = 2.0
PULSE_WIDTH = 0.25e-3
PULSE_CENTRE = 0.75e-3
WINDOW = 6.0e-3
SAMPLES = 2048
BAND = 4000.0


def _velocity(times: np.ndarray) -> np.ndarray:
    return ap.gaussian_velocity_pulse(times, centre=PULSE_CENTRE, width=PULSE_WIDTH)


def _reference(times: np.ndarray, *, substeps: int = 16) -> np.ndarray:
    return ap.exact_monopole_pulse_pressure(
        times,
        radius=RADIUS,
        velocity=_velocity,
        receiver_distance=RECEIVER_RANGE,
        density=DENSITY,
        sound_speed=SOUND_SPEED,
        substeps=substeps,
    )


# ---------------------------------------------------------------------------
# route A: the exact reference itself
# ---------------------------------------------------------------------------


def test_exact_reference_converges_with_substeps() -> None:
    """The ODE quadrature is converged: substeps 16 vs 64 differ by < 1e-6."""

    t = ap.grid(WINDOW, SAMPLES)
    coarse = _reference(t, substeps=16)
    fine = _reference(t, substeps=64)
    errors = ap.waveform_errors(t, fine, coarse)
    assert errors["l2_relative_error"] < 1e-6
    assert errors["peak_relative_error"] < 1e-6
    assert errors["arrival_shift"] == 0.0


def test_exact_reference_is_resolved_by_the_output_grid() -> None:
    """Halving the time step changes the waveform by < 1e-3 (grid independence)."""

    t = ap.grid(WINDOW, SAMPLES)
    t_fine = ap.grid(WINDOW, 4 * SAMPLES)
    fine = _reference(t_fine)
    coarse = _reference(t)
    errors = ap.waveform_errors(t, np.interp(t, t_fine, fine), coarse)
    assert errors["l2_relative_error"] < 1e-3


def test_exact_reference_arrival_time_is_retarded() -> None:
    """The pulse arrives near ``t_centre + r/c`` (retarded propagation)."""

    t = ap.grid(WINDOW, SAMPLES)
    pressure = _reference(t)
    expected = PULSE_CENTRE + RECEIVER_RANGE / SOUND_SPEED
    assert abs(ap.arrival_time(t, pressure) - expected) < 5.0e-4


def test_exact_reference_decays_as_one_over_range() -> None:
    """The monopole field obeys ``p ~ 1/r``: doubling r halves the peak."""

    t = ap.grid(WINDOW, SAMPLES)
    near = ap.exact_monopole_pulse_pressure(
        t, radius=RADIUS, velocity=_velocity, receiver_distance=1.0,
        density=DENSITY, sound_speed=SOUND_SPEED,
    )
    far = ap.exact_monopole_pulse_pressure(
        t, radius=RADIUS, velocity=_velocity, receiver_distance=2.0,
        density=DENSITY, sound_speed=SOUND_SPEED,
    )
    ratio = np.max(np.abs(far)) / np.max(np.abs(near))
    assert abs(ratio - 0.5) < 1e-3


def test_exact_reference_rejects_bad_inputs() -> None:
    t = ap.grid(WINDOW, 64)
    with pytest.raises(ValueError):
        ap.exact_monopole_pulse_pressure(
            t, radius=-0.1, velocity=_velocity, receiver_distance=1.0
        )
    with pytest.raises(ValueError):
        ap.exact_monopole_pulse_pressure(
            t, radius=0.1, velocity=_velocity, receiver_distance=0.05
        )
    with pytest.raises(ValueError):
        ap.exact_monopole_pulse_pressure(
            t, radius=0.1, velocity=_velocity, receiver_distance=1.0, substeps=0
        )
    with pytest.raises(ValueError):
        ap.exact_monopole_pulse_pressure(
            np.array([0.0, 0.1, 0.05, 0.2]), radius=0.1,
            velocity=_velocity, receiver_distance=1.0,
        )


# ---------------------------------------------------------------------------
# harmonic amplitude: consistency with the frequency-domain module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ka", [0.1, 0.5, 1.0, 1.676])
def test_exact_pulse_amplitude_matches_library_analytic(ka: float) -> None:
    """``exact_pulse_amplitude`` reproduces ``acoustic3d``'s exact sphere field."""

    k = ka / RADIUS
    omega = k * SOUND_SPEED
    points = np.array([[RECEIVER_RANGE, 0.0, 0.0]])
    reference = ac.pulsating_sphere_pressure(
        RADIUS, 1.0, k, points, density=DENSITY, sound_speed=SOUND_SPEED
    )[0]
    candidate = ap.exact_pulse_amplitude(
        omega,
        radius=RADIUS,
        velocity_amplitude=1.0,
        receiver_distance=RECEIVER_RANGE,
        density=DENSITY,
        sound_speed=SOUND_SPEED,
    )
    assert abs(candidate - reference) / abs(reference) < 1e-12


# ---------------------------------------------------------------------------
# route B: synthesis on the DFT bins
# ---------------------------------------------------------------------------


def test_synthesis_of_exact_harmonics_reproduces_the_transient() -> None:
    """Route B vs route A: the bin synthesis is lossless to < 1e-3."""

    t = ap.grid(WINDOW, SAMPLES)
    reference = _reference(t)
    omegas = ap.bin_frequencies(WINDOW, BAND)
    amplitudes_u = ap.spectrum(t, _velocity(t), omegas)
    amplitudes = np.array(
        [
            ap.exact_pulse_amplitude(
                float(w), radius=RADIUS, velocity_amplitude=complex(a),
                receiver_distance=RECEIVER_RANGE, density=DENSITY,
                sound_speed=SOUND_SPEED,
            )
            for w, a in zip(omegas, amplitudes_u)
        ]
    )
    pressure = ap.synthesise(t, omegas, amplitudes, duration=WINDOW)
    errors = ap.waveform_errors(t, reference, pressure)
    assert errors["l2_relative_error"] < 1e-3
    assert errors["peak_relative_error"] < 1e-3
    assert errors["correlation"] > 0.999


def test_spectrum_of_a_harmonic_is_a_single_line() -> None:
    """A pure tone lands on its own bin with the analytic amplitude ``A*T/2``."""

    duration = 8.0e-3
    t = ap.grid(duration, 1024)
    tone = np.cos(2.0 * np.pi * 5.0 * t / duration)  # 5 periods in the window
    omegas = ap.bin_frequencies(duration, 5.0 / duration)
    spectrum = ap.spectrum(t, tone, omegas)
    assert spectrum.size == 5
    assert abs(spectrum[4] - duration / 2.0) / (duration / 2.0) < 1e-2
    assert np.max(np.abs(spectrum[:4])) < 1e-2 * (duration / 2.0)


def test_synthesise_rejects_shape_mismatch() -> None:
    t = ap.grid(WINDOW, 64)
    with pytest.raises(ValueError):
        ap.synthesise(t, np.array([1.0, 2.0]), np.array([1.0 + 0j]))


# ---------------------------------------------------------------------------
# route C: the BEM bridge (acceptance)
# ---------------------------------------------------------------------------


def test_bem_pulse_matches_exact_transient_within_tolerance() -> None:
    """``CASE-ACOUSTIC-PULSE``: L2 waveform error < 8 % on 80 panels."""

    t = ap.grid(WINDOW, SAMPLES)
    reference = _reference(t)
    omegas = ap.bin_frequencies(WINDOW, BAND)
    mesh = icosphere(1, radius=RADIUS)
    solution = ap.synthesise_bem_pulse(
        mesh, t, radius=RADIUS, velocity=_velocity,
        receiver=np.array([RECEIVER_RANGE, 0.0, 0.0]), omegas=omegas,
        density=DENSITY, sound_speed=SOUND_SPEED,
    )
    errors = ap.waveform_errors(t, reference, solution.pressure)
    assert errors["l2_relative_error"] < 0.08
    assert errors["peak_relative_error"] < 0.08
    assert abs(errors["arrival_shift"]) < 1e-5
    assert errors["correlation"] > 0.99
    # the harmonic error table is archived, one row per bin
    table = solution.harmonic_error_table()
    assert len(table) == omegas.size
    assert set(table[0]) == {
        "bin", "frequency_hz", "bem_amplitude", "exact_amplitude", "relative_error",
    }
    assert solution.max_per_bin_error < 0.08


def test_bem_pulse_rejects_receiver_inside_the_sphere() -> None:
    t = ap.grid(WINDOW, 64)
    omegas = ap.bin_frequencies(WINDOW, 1000.0)
    mesh = icosphere(0, radius=RADIUS)
    with pytest.raises(ValueError):
        ap.synthesise_bem_pulse(
            mesh, t, radius=RADIUS, velocity=_velocity,
            receiver=np.array([0.05, 0.0, 0.0]), omegas=omegas,
        )
    with pytest.raises(ValueError):
        ap.synthesise_bem_pulse(
            mesh, t, radius=RADIUS, velocity=_velocity,
            receiver=np.array([RECEIVER_RANGE, 0.0, 0.0]),
            omegas=np.array([-1.0]),
        )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_waveform_errors_on_known_perturbations() -> None:
    t = ap.grid(WINDOW, 256)
    reference = _reference(t)
    identical = ap.waveform_errors(t, reference, reference)
    assert identical["l2_relative_error"] == 0.0
    assert identical["peak_relative_error"] == 0.0
    assert identical["arrival_shift"] == 0.0
    scaled = ap.waveform_errors(t, reference, 1.1 * reference)
    assert abs(scaled["peak_relative_error"] - 0.1) < 1e-12
    assert abs(scaled["correlation"] - 1.0) < 1e-12
    delayed = ap.waveform_errors(t, reference, np.roll(reference, 3))
    assert delayed["arrival_shift"] > 0.0
    assert delayed["correlation"] < 1.0


def test_waveform_errors_reject_zero_reference() -> None:
    t = ap.grid(WINDOW, 32)
    with pytest.raises(ValueError):
        ap.waveform_errors(t, np.zeros_like(t), np.ones_like(t))


def test_pulse_definition_bounds() -> None:
    t = ap.grid(WINDOW, 128)
    pulse = ap.gaussian_velocity_pulse(t, centre=PULSE_CENTRE, width=PULSE_WIDTH)
    assert pulse.max() == pytest.approx(1.0)
    assert abs(t[int(np.argmax(pulse))] - PULSE_CENTRE) < 2.0 * (WINDOW / 128)
    with pytest.raises(ValueError):
        ap.gaussian_velocity_pulse(t, centre=PULSE_CENTRE, width=0.0)
    with pytest.raises(ValueError):
        ap.gaussian_velocity_pulse(t, centre=math.inf, width=PULSE_WIDTH)