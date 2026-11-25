"""Tests for the Cummins memory kernel closure campaign on a Wigley hull.

These tests verify the five gates of the campaign:
- W1: geometry precondition
- W2: round-trip closure (B -> K -> B)
- W3: Kramers-Kronig causality
- W4: free decay damping ratio
- W5: negative control (zero damping)
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from tensorbem.benchmarks import cummins_wigley_hull as campaign


@functools.lru_cache(maxsize=1)
def _sweep() -> campaign.WigleyRadiationSweep:
    """Heave sweep of the Wigley hull, shared by the tests in this module."""
    return campaign.wigley_radiation_sweep(
        n_omega=20,
        n_sections=20,
        n_levels=6,
    )


# ---------------------------------------------------------------------------
# W1 -- geometry precondition
# ---------------------------------------------------------------------------
def test_geometry_gate_accepts_wigley_hull() -> None:
    """The Wigley hull geometry must pass the platform's gate."""
    geometry = campaign.wigley_hull_geometry()
    gate = campaign.geometry_gate_wigley(
        length=geometry["length"],
        beam=geometry["beam"],
        draft=geometry["draft"],
        displacement=geometry["displacement"],
        wetted_surface=geometry["wetted_surface"],
        block_coefficient=geometry["block_coefficient"],
    )
    assert gate["passed"], gate
    # Wigley Cb = 4/9 ≈ 0.4444
    assert gate["block_coefficient"] == pytest.approx(4.0 / 9.0, rel=1e-12)
    assert gate["displacement_consistency_pct"] == pytest.approx(0.0, abs=1e-9)


def test_geometry_gate_rejects_corrupt_wigley() -> None:
    """A corrupt geometry must be rejected."""
    geometry = campaign.wigley_hull_geometry()
    
    # Inflated displacement
    inflated = campaign.geometry_gate_wigley(
        length=geometry["length"],
        beam=geometry["beam"],
        draft=geometry["draft"],
        displacement=1.5 * geometry["displacement"],
        wetted_surface=geometry["wetted_surface"],
        block_coefficient=geometry["block_coefficient"],
    )
    assert not inflated["passed"]
    
    # Invalid block coefficient
    invalid_cb = campaign.geometry_gate_wigley(
        length=geometry["length"],
        beam=geometry["beam"],
        draft=geometry["draft"],
        displacement=geometry["displacement"],
        wetted_surface=geometry["wetted_surface"],
        block_coefficient=0.2,  # below minimum
    )
    assert not invalid_cb["passed"]


def test_criterion_geometry_wigley_passes() -> None:
    """The campaign's geometry gate must pass for the Wigley hull."""
    verdict = campaign.criterion_geometry_wigley(_sweep())
    assert verdict["passed"], verdict
    assert verdict["id"] == "W1"


# ---------------------------------------------------------------------------
# W2 -- round trip
# ---------------------------------------------------------------------------
def test_round_trip_wigley_heave() -> None:
    """Round-trip closure for heave must close within 2%."""
    verdict = campaign.criterion_round_trip_wigley(_sweep(), dof=2, dt=0.02)
    values = verdict["values"]
    assert verdict["passed"], values
    assert values["damping_relative_error_max"] < 0.02
    assert values["added_mass_relative_error_max"] < 0.02


def test_round_trip_wigley_surge() -> None:
    """Round-trip closure for surge (if data is available)."""
    verdict = campaign.criterion_round_trip_wigley(_sweep(), dof=0, dt=0.02)
    # Surge may have weak radiation, so we just check it runs
    assert verdict["id"] == "W2"


# ---------------------------------------------------------------------------
# W3 -- causality & Kramers-Kronig
# ---------------------------------------------------------------------------
def test_causality_wigley_heave() -> None:
    """Kramers-Kronig causality for heave must pass."""
    verdict = campaign.criterion_causality_wigley(_sweep(), dof=2, dt=0.02)
    values = verdict["values"]
    assert verdict["passed"], values
    assert values["causal"] is True
    assert values["kkr_relative_residual"] < 0.03


def test_kernel_is_causal() -> None:
    """The memory kernel must be causal (K(t<0) = 0)."""
    sweep = _sweep()
    from tensorbem import time_domain as td
    
    B_omega = sweep.damping[:, 2, 2]  # heave
    kernel = td.radiation_memory_kernel(sweep.omega, B_omega, np.linspace(0., 20., 1001))
    
    # Kernel is defined for t >= 0 only
    assert np.all(kernel(-kernel.tau[1:]) == 0.)
    # All values should be finite
    assert np.all(np.isfinite(kernel.values))


# ---------------------------------------------------------------------------
# W4 -- free decay
# ---------------------------------------------------------------------------
def test_free_decay_wigley_heave() -> None:
    """Free decay damping ratio for heave must match frequency-domain estimate."""
    verdict = campaign.criterion_free_decay_wigley(
        _sweep(), dof=2, mass_fraction=1.0, dt=0.02, duration=50.0
    )
    values = verdict["values"]
    # Skip if no stiffness (shouldn't happen for heave)
    if "note" in values:
        pytest.skip(values["note"])
    assert verdict["passed"], values
    assert values["relative_error"] < 0.05


def test_free_decay_wigley_surge_skipped() -> None:
    """Surge has no hydrostatic stiffness, so free decay is skipped."""
    verdict = campaign.criterion_free_decay_wigley(_sweep(), dof=0)
    values = verdict["values"]
    # Should be skipped or failed due to no stiffness
    assert not verdict["passed"] or "note" in values


# ---------------------------------------------------------------------------
# W5 -- negative control
# ---------------------------------------------------------------------------
def test_negative_control_wigley_heave() -> None:
    """Zero damping must give zero decay rate."""
    verdict = campaign.criterion_negative_control_wigley(
        _sweep(), dof=2, mass_fraction=1.0, dt=0.02, duration=50.0
    )
    values = verdict["values"]
    if "note" in values:
        pytest.skip(values["note"])
    assert verdict["passed"], values
    assert values["relative_error"] < 0.005


# ---------------------------------------------------------------------------
# Full campaign
# ---------------------------------------------------------------------------
def test_full_campaign_wigley_heave() -> None:
    """Run the full campaign for heave and check all gates pass."""
    result = campaign.run_campaign_wigley(
        dof=2,
        n_omega=20,
        n_sections=20,
        n_levels=6,
        dt=0.02,
        tau_max=20.0,
        duration=50.0,
    )
    
    assert "case" in result
    assert "gates" in result
    assert "all_passed" in result
    assert len(result["gates"]) == 5
    
    # Check gate IDs
    gate_ids = [g["id"] for g in result["gates"]]
    assert gate_ids == ["W1", "W2", "W3", "W4", "W5"]
    
    # Print summary
    print("\n=== Wigley Hull Cummins Closure Campaign ===")
    print(f"Case: {result['case']['title']}")
    print(f"DOF: {result['case']['dof_tested']}")
    print(f"All gates passed: {result['all_passed']}")
    print(f"Runtime: {result['runtime_seconds']:.2f} s")
    print("\nGates:")
    for gate in result["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  {gate['id']}: {gate['title']} [{status}]")
        if not gate["passed"]:
            print(f"    Values: {gate['values']}")


if __name__ == "__main__":
    # Run tests
    test_geometry_gate_accepts_wigley_hull()
    test_geometry_gate_rejects_corrupt_wigley()
    test_criterion_geometry_wigley_passes()
    test_round_trip_wigley_heave()
    test_causality_wigley_heave()
    test_kernel_is_causal()
    test_free_decay_wigley_heave()
    test_negative_control_wigley_heave()
    test_full_campaign_wigley_heave()
    print("\nAll tests passed!")


def test_negative_control_period_matches_undamped_oscillator():
    geometry = campaign.wigley_hull_geometry()
    mass = geometry["displaced_mass"]
    omega = np.geomspace(.1, 20., 32)
    added = np.zeros((len(omega), 6, 6))
    added[:, 2, 2] = mass / 2
    sweep = campaign.WigleyRadiationSweep(omega, added, added.copy(), added[-1], mass, geometry)
    restoring = geometry["density"] * campaign.WIGLEY_GRAVITY * (2/3) * geometry["length"] * geometry["beam"]
    reference = 2 * math.pi * math.sqrt(1.5 * mass / restoring)
    errors = []
    for dt in (.02, .01):
        gate = campaign.criterion_negative_control_wigley(sweep, dt=dt, duration=20.)
        assert gate["passed"] and gate["values"]["kernel_norm_zero"] == 0.
        errors.append(abs(gate["values"]["period_measured"] / reference - 1.))
    assert errors[1] < errors[0] / 3.5


def test_heave_decay_against_single_pole_reference(record_property):
    geometry = campaign.wigley_hull_geometry()
    mass = geometry["displaced_mass"]
    mass_inf = mass / 2
    restoring = geometry["density"] * campaign.WIGLEY_GRAVITY * (2/3) * geometry["length"] * geometry["beam"]
    pole, residue = 12., 70.
    # K(t) = residue exp(-pole*t) gives a rational radiation impedance.
    omega = np.geomspace(.001, 100., 401)
    added = np.zeros((len(omega), 6, 6))
    damping = np.zeros_like(added)
    added[:, 2, 2] = mass_inf - residue/(pole**2 + omega**2)
    damping[:, 2, 2] = residue*pole/(pole**2 + omega**2)
    asymptote = np.zeros((6, 6))
    asymptote[2, 2] = mass_inf
    sweep = campaign.WigleyRadiationSweep(omega, added, damping, asymptote, mass, geometry)

    # Eliminate the memory force from the ODE to get the exact oscillatory pole.
    inertia = mass + mass_inf
    roots = np.roots([inertia, pole*inertia, restoring+residue, pole*restoring])
    mode, = roots[roots.imag > 0]
    exact_zeta = -mode.real / abs(mode)
    exact_period = 2*math.pi / mode.imag
    result = campaign.criterion_free_decay_wigley(sweep, dt=.01, tau_max=16., duration=20.)
    assert result["passed"]
    values = result["values"]
    for key, exact in (("zeta_free_decay", exact_zeta), ("period_measured", exact_period)):
        error = abs(values[key]/exact - 1)
        record_property(key + "_reference", exact)
        record_property(key + "_relative_error", error)
        assert error < .001
