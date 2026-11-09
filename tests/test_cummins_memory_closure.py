"""Gates of the Cummins memory-kernel closure campaign.

The numbers here are the ones the campaign report quotes: the round trip on a
deeply submerged sphere, the causality of the kernel it produces, the free
decay against the frequency-domain damping, the zero-damping negative control,
and the geometry precondition. The sweep is built once per session (the
multipole reference is the expensive part); everything else is closed form.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from tensorbem.benchmarks import cummins_memory_closure as campaign
from tensorbem import time_domain as td


@functools.lru_cache(maxsize=1)
def _sweep() -> campaign.RadiationSweep:
    """Heave sweep of the case body, shared by the tests in this module."""
    return campaign.submerged_sphere_sweep(points=90, ka_max=5.0)


# ---------------------------------------------------------------------------
# C5 -- geometry precondition
# ---------------------------------------------------------------------------
def test_geometry_gate_accepts_the_sphere() -> None:
    geometry = campaign.sphere_geometry(1.0)
    gate = campaign.geometry_gate(
        length=geometry["length"],
        beam=geometry["beam"],
        draft=geometry["draft"],
        displacement=geometry["displacement"],
        wetted_surface=geometry["wetted_surface"],
        block_coefficient=geometry["block_coefficient"],
    )
    assert gate["passed"]
    assert gate["cb_from_dims"] == pytest.approx(math.pi / 6.0, rel=1.0e-12)
    assert gate["displacement_consistency_pct"] == pytest.approx(0.0, abs=1.0e-12)
    assert gate["block_coefficient"] == pytest.approx(math.pi / 6.0, rel=1.0e-12)


def test_geometry_gate_rejects_corrupt_input() -> None:
    """A rubber-stamp gate would let the campaign proceed on nonsense geometry."""
    geometry = campaign.sphere_geometry(1.0)
    base = dict(
        length=geometry["length"],
        beam=geometry["beam"],
        draft=geometry["draft"],
    )
    inflated = campaign.geometry_gate(
        displacement=1.25 * geometry["displacement"],
        block_coefficient=geometry["block_coefficient"],
        **base,
    )
    assert not inflated["passed"]
    assert not inflated["checks"]["displacement_consistency"]["passed"]
    assert inflated["displacement_consistency_pct"] == pytest.approx(20.0, abs=1.0e-9)

    impossible = campaign.geometry_gate(
        displacement=geometry["displacement"],
        block_coefficient=1.2,
        **base,
    )
    assert not impossible["passed"]
    assert not impossible["checks"]["block_coefficient"]["passed"]

    not_a_block = campaign.geometry_gate(
        displacement=geometry["displacement"],
        block_coefficient=0.65,
        **base,
    )
    assert not not_a_block["passed"]

    zero = campaign.geometry_gate(
        length=0.0, beam=1.0, draft=1.0, displacement=1.0
    )
    assert not zero["passed"]


def test_criterion_geometry_passes_and_proves_the_gate_bites() -> None:
    verdict = campaign.criterion_geometry(_sweep())
    assert verdict["passed"], verdict["values"]["gate"]
    assert verdict["values"]["gate"]["block_coefficient"] == pytest.approx(
        math.pi / 6.0, rel=1.0e-12
    )


# ---------------------------------------------------------------------------
# C1 -- round trip
# ---------------------------------------------------------------------------
def test_round_trip_closes_within_two_percent() -> None:
    verdict = campaign.criterion_round_trip(_sweep(), dt=0.02)
    values = verdict["values"]
    assert verdict["passed"], values
    assert values["damping_relative_error_max"] < 1.0e-3
    assert values["added_mass_relative_error_max"] <= 0.02


def test_round_trip_error_grows_when_the_band_is_truncated() -> None:
    """Negative control on the transform: a narrow band must close worse."""
    sweep = _sweep()
    wide = campaign.criterion_round_trip(sweep, dt=0.02)
    narrow = campaign.criterion_round_trip(
        sweep, dt=0.02, tau_max=0.5 * math.pi / float(np.min(np.diff(sweep.omega)))
    )
    assert (
        narrow["values"]["added_mass_relative_error_max"]
        > wide["values"]["added_mass_relative_error_max"]
    )


# ---------------------------------------------------------------------------
# C2 -- causality
# ---------------------------------------------------------------------------
def test_kernel_is_causal_and_jumps_upward() -> None:
    verdict = campaign.criterion_causality(_sweep(), dt=0.02)
    values = verdict["values"]
    assert verdict["passed"], values
    assert values["kernel_at_negative_times_max_abs"] == 0.0
    assert values["kernel_at_zero"] > 0.0
    assert values["kernel_at_zero_is_global_maximum"]
    assert values["kernel_at_zero_relative_error"] < 1.0e-3
    assert abs(values["kernel_relative_slope_at_zero"]) <= values["kernel_slope_grid_bound"]


def test_kernel_is_even_about_tau_zero() -> None:
    sweep = _sweep()
    kernel = td.radiation_memory_kernel(
        sweep.omega, sweep.damping, np.arange(0.0, 20.0, 0.02)
    )
    tau, values = td.two_sided_kernel(kernel)
    scale = float(np.max(np.abs(values)))
    assert float(np.max(np.abs(values - values[::-1])) / scale) < 1.0e-12
    assert np.all(kernel(np.array([-1.0e-9, -0.02, -5.0])) == 0.0)


# ---------------------------------------------------------------------------
# C3 / C4 -- free decay and the negative control
# ---------------------------------------------------------------------------
def test_free_decay_matches_the_frequency_domain_damping() -> None:
    verdict = campaign.criterion_free_decay(
        _sweep(), dt_list=(0.02, 0.01, 0.005), duration=40.0
    )
    values = verdict["values"]
    assert verdict["passed"], values
    assert values["zeta_decay_vs_energy_pct"] <= 5.0
    # First-order convergence of the trapezoidal memory integral: the measured
    # ratio is ~2 per halving, so the Richardson limit is the right estimator.
    assert 1.5 < values["convergence_ratio"] < 2.5
    assert values["zeta_decay_vs_half_power_pct"] <= 5.0


def test_zero_damping_gives_zero_decay() -> None:
    decay = campaign.criterion_free_decay(
        _sweep(), dt_list=(0.02, 0.01, 0.005), duration=40.0
    )
    control = campaign.criterion_negative_control(
        _sweep(),
        dt=0.01,
        duration=40.0,
        reference_decay_rate=decay["values"]["decay_rate_extrapolated"],
    )
    values = control["values"]
    assert control["passed"], values
    assert values["kernel_is_identically_zero"]
    assert values["decay_rate_ratio"] <= 0.005


# ---------------------------------------------------------------------------
# the constant-added-mass envelope
# ---------------------------------------------------------------------------
def test_half_power_damping_resolves_light_damping() -> None:
    """The global scan under-resolves a light resonance; the local one does not."""
    sweep = _sweep()
    mass = 0.1 * sweep.displaced_mass
    omega_p = sweep.damping_peak_omega
    restoring = (mass + sweep.added_mass_inf) * omega_p**2
    legacy = td.resonance_properties(
        sweep.omega, sweep.added_mass, sweep.damping, mass=mass, restoring=restoring
    )
    resolved = td.half_power_damping(
        sweep.omega,
        sweep.added_mass,
        sweep.damping,
        mass=mass,
        restoring=restoring,
        omega_n=omega_p,
    )
    assert resolved["half_power_ratio"] == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-6)
    assert abs(resolved["zeta"] - resolved["zeta_energy"]) / resolved["zeta_energy"] < 0.05
    # The legacy geometric scan reports a value biased low by the scan spacing.
    assert legacy["zeta_bandwidth"] < 0.95 * resolved["zeta"]


def test_envelope_is_exact_at_the_damping_peak_and_grows_away_from_it() -> None:
    sweep = _sweep()
    envelope = campaign.error_envelope(
        sweep, ratios=(0.5, 0.75, 1.0, 1.5, 2.0), samples=2001
    )
    rows = {round(row["omega_p_over_omega_n"], 6): row for row in envelope["rows"]}
    peak = rows[1.0]
    assert abs(peak["peak_amplitude_error"]) < 5.0e-3
    assert abs(peak["phase_error_degrees"]) < 10.0
    amplitudes = [abs(rows[r]["peak_amplitude_error"]) for r in (0.5, 0.75, 1.0, 1.5, 2.0)]
    assert amplitudes[2] == min(amplitudes)
    phases = [abs(rows[r]["phase_error_degrees"]) for r in (0.5, 0.75, 1.0, 1.5, 2.0)]
    assert phases[2] == min(phases)
    assert phases[-1] > 40.0 and phases[0] > 40.0
    # The damping ratio itself is what the constant-A model gets closest to.
    for row in envelope["rows"]:
        assert row["damping_ratio_relative_error"] < 0.05


def test_envelope_rejects_nonsense_ratios() -> None:
    sweep = _sweep()
    with pytest.raises(ValueError, match="positive"):
        td.constant_added_mass_error_envelope(
            sweep.omega,
            sweep.added_mass,
            sweep.damping,
            mass=100.0,
            ratios=(1.0, 0.0),
        )


# ---------------------------------------------------------------------------
# the whole campaign
# ---------------------------------------------------------------------------
def test_campaign_runs_every_gate_in_order() -> None:
    record = campaign.run_campaign(points=90, ka_max=5.0)
    assert [gate["id"] for gate in record["gates"]] == ["C5", "C1", "C2", "C3", "C4"]
    assert all(gate["passed"] for gate in record["gates"]), [
        (gate["id"], gate["values"]) for gate in record["gates"] if not gate["passed"]
    ]
    assert record["case"]["sweep"]["far_field_damping_error_max"] < 1.0e-6
    assert record["case"]["sweep"]["added_mass_inf_anchor_error_pct"] < 5.0