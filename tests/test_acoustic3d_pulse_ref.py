"""Tests for the CASE-ACOUSTIC-PULSE reference anchor (m17210).

The anchor pins the target values of the case to the classical
retarded-potential solution of a pulsating sphere (see
``src/tensorbem/benchmarks/acoustic3d_pulse_ref.py`` for the citations).
These tests verify that the anchor is internally consistent and that every
independent closed form agrees with the ODE-integrated reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import acoustic3d as ac
from tensorbem import acoustic3d_pulse as ap
from tensorbem.benchmarks import acoustic3d_pulse_ref as ref

A = 0.1
R = 2.0
RHO = ac.DENSITY_SEAWATER
C = ac.SOUND_SPEED_SEAWATER
OMEGA = 2.0 * math.pi * 1000.0
U = 1.0 + 0.0j


def test_citations_are_complete():
    keys = {c["key"] for c in ref.CITATIONS}
    assert keys == {"C1", "C2", "C3"}
    for c in ref.CITATIONS:
        assert c["source"].strip()
        assert c["locator"].strip()
        assert c["formula"].strip()


def test_harmonic_anchor_matches_library_analytic():
    p_ref = ref.harmonic_pressure(OMEGA, U, R, radius=A, density=RHO, sound_speed=C)
    p_lib = complex(
        ac.pulsating_sphere_pressure(
            A, U, OMEGA / C, np.array([[R, 0.0, 0.0]]),
            density=RHO, sound_speed=C,
        )[0]
    )
    assert abs(p_ref - p_lib) / abs(p_lib) < 1e-12


@pytest.mark.parametrize("ka", [0.01, 0.1, 0.5, 1.0, 1.676])
def test_harmonic_anchor_matches_library_over_band(ka):
    omega = ka * C / A
    p_ref = ref.harmonic_pressure(omega, U, R, radius=A, density=RHO, sound_speed=C)
    p_lib = complex(
        ac.pulsating_sphere_pressure(
            A, U, omega / C, np.array([[R, 0.0, 0.0]]),
            density=RHO, sound_speed=C,
        )[0]
    )
    assert abs(p_ref - p_lib) / abs(p_lib) < 1e-12


def test_low_ka_monopole_limit_magnitude():
    omega = 0.01 * C / A
    p = ref.harmonic_pressure(omega, U, R, radius=A, density=RHO, sound_speed=C)
    p_mono = ref.monopole_limit_pressure(omega, U, R, radius=A, density=RHO, sound_speed=C)
    assert abs(abs(p) - abs(p_mono)) / abs(p_mono) < 5e-3


def test_far_field_one_over_r_decay():
    r1, r2 = 50.0 * A, 100.0 * A
    p1 = ref.harmonic_pressure(OMEGA, U, r1, radius=A, density=RHO, sound_speed=C)
    p2 = ref.harmonic_pressure(OMEGA, U, r2, radius=A, density=RHO, sound_speed=C)
    assert abs(abs(p1) / abs(p2) - 2.0) < 1e-3


def test_impulse_arrival_is_nearest_surface():
    assert ref.impulse_arrival(R, radius=A, sound_speed=C) == pytest.approx((R - A) / C)


def test_impulse_response_tail_shape():
    t = np.linspace(0.0, 5.0e-3, 2000)
    tail = ref.impulse_response_tail(t, R, radius=A, density=RHO, sound_speed=C)
    arrival = (R - A) / C
    # zero before the arrival
    assert np.all(tail[t < arrival - 1e-12] == 0.0)
    # exponential decay with time constant a/c after the arrival
    after = t > arrival + 1e-9
    x = (t[after] - arrival) / (A / C)
    expected = -(RHO * C**2 * A / R) * np.exp(-x)
    assert np.allclose(tail[after], expected, rtol=1e-12)


def test_impulse_response_route_matches_ode_route():
    t = ap.grid(6.0e-3, 2048)

    def velocity(times: np.ndarray) -> np.ndarray:
        return ap.gaussian_velocity_pulse(times, centre=0.75e-3, width=0.25e-3)

    p_ode = ap.exact_monopole_pulse_pressure(
        t, radius=A, velocity=velocity, receiver_distance=R,
        density=RHO, sound_speed=C, substeps=32,
    )
    p_imp = ref.pulse_by_impulse_response(
        t, velocity, R, radius=A, density=RHO, sound_speed=C, substeps=32
    )
    err = ap.waveform_errors(t, p_ode, p_imp)
    assert err["l2_relative_error"] < 1e-3
    assert err["peak_relative_error"] < 1e-3


def test_impulse_response_narrow_pulse():
    t = ap.grid(6.0e-3, 2048)

    def narrow(times: np.ndarray) -> np.ndarray:
        return ap.gaussian_velocity_pulse(times, centre=0.75e-3, width=0.02e-3)

    p_ode = ap.exact_monopole_pulse_pressure(
        t, radius=A, velocity=narrow, receiver_distance=R,
        density=RHO, sound_speed=C, substeps=64,
    )
    p_imp = ref.pulse_by_impulse_response(
        t, narrow, R, radius=A, density=RHO, sound_speed=C, substeps=64
    )
    err = ap.waveform_errors(t, p_ode, p_imp)
    assert err["l2_relative_error"] < 2e-3


def test_all_cross_checks_pass():
    checks = ref.cross_checks(radius=A, receiver_distance=R, density=RHO, sound_speed=C)
    assert len(checks) == 6
    for c in checks:
        assert c.passed, f"{c.name}: {c.detail}"


def test_anchor_payload_is_json_serialisable():
    import json

    payload = ref.anchor_payload(
        radius=A, receiver_distance=R, density=RHO, sound_speed=C,
        omega=OMEGA, velocity_amplitude=U,
    )
    s = json.dumps(payload)
    assert "CASE-ACOUSTIC-PULSE" in s
    assert payload["all_passed"] is True
    assert len(payload["citations"]) == 3


def test_input_validation():
    with pytest.raises(ValueError):
        ref.harmonic_pressure(OMEGA, U, 0.05, radius=A)  # r < a
    with pytest.raises(ValueError):
        ref.impulse_arrival(0.05, radius=A)
    with pytest.raises(ValueError):
        ref.pulse_by_impulse_response(
            ap.grid(6.0e-3, 2048),
            lambda x: ap.gaussian_velocity_pulse(x, centre=0.75e-3, width=0.25e-3),
            0.05, radius=A,
        )
