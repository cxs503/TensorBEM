"""Tests for the 6x6 matrix-level audit of the Cummins-Wigley case sweep.

The audit is deliberately independent of the boundary-element solver: it reads
the committed ``sweep_*.npz``.  These tests therefore check the *statements*
the audit makes about that artifact, plus the internal consistency of the
gates themselves (a gate whose negative control does not fire is not a gate).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import cummins_wigley_matrix_audit as audit
from tensorbem.benchmarks.case_bem_cummins_wigley import DOF_NAMES, load_sweep

ARTIFACT = Path(__file__).resolve().parents[1] / audit.ARTIFACT
SWEEP = ARTIFACT / "sweep_medium.npz"

pytestmark = pytest.mark.skipif(
    not SWEEP.exists(), reason=f"committed sweep missing: {SWEEP}"
)


@pytest.fixture(scope="module")
def record():
    return audit.run_audit(artifact_dir=ARTIFACT, label="medium", write=False)


# ---------------------------------------------------------------------------
# selection rules are derived from the mirror planes, not hard-coded
# ---------------------------------------------------------------------------
def test_parities_follow_from_the_reflection_matrices():
    rules = audit.mode_parities()
    # x-mirror: surge odd (polar), sway/heave even, roll even (axial, det=-1),
    # pitch/yaw odd.  y-mirror is the swap.
    assert rules["parities"]["x_mirror"] == [1, 0, 0, 0, 1, 1]
    assert rules["parities"]["y_mirror"] == [0, 1, 0, 1, 0, 1]


def test_allowed_pairs_are_the_two_hydrodynamic_couplings():
    rules = audit.mode_parities()
    def key(pair):
        return tuple(sorted(pair, key=DOF_NAMES.index))

    allowed = {key(pair) for pair in rules["allowed_pairs"]}
    expected = {(DOF_NAMES[i], DOF_NAMES[i]) for i in range(6)}
    expected |= {("surge", "pitch"), ("sway", "roll")}
    assert allowed == expected
    # every forbidden pair is off-diagonal, and there are 13 of them
    assert len(rules["forbidden_pairs"]) == 13
    assert all(pair[0] != pair[1] for pair in rules["forbidden_pairs"])


# ---------------------------------------------------------------------------
# the artifact statements
# ---------------------------------------------------------------------------
def test_audit_passes_every_gate_on_the_committed_sweep(record):
    assert record["verdict"]["all_passed"], record["verdict"]
    assert record["verdict"]["n_gates"] == 6


def test_negative_controls_all_fire(record):
    controls = record["gates"]["M5"]["values"]["controls"]
    assert len(controls) == 3
    for name, row in controls.items():
        assert not row["passed_under_corruption"], name
        assert row["measured"] > row["limit"], name


def test_mesh_is_symmetric_enough_for_the_selection_rules(record):
    values = record["gates"]["M0"]["values"]
    for key in ("x_mirror", "y_mirror"):
        assert values[key]["relative_to_length"] <= 1e-9


def test_forbidden_entries_sit_at_the_solve_residual_floor(record):
    values = record["gates"]["M2"]["values"]
    assert values["n_forbidden_entries"] == 26
    assert values["forbidden_worst_relative_peak"] <= values["limit"]
    # not a trivial pass: the allowed couplings must stand well clear of the
    # solve-noise floor (measured ~28x on surge-pitch)
    def key(pair):
        return tuple(sorted(pair, key=DOF_NAMES.index))

    allowed = {key(row["pair"]): row for row in values["allowed"]}
    surge_pitch = max(
        allowed[("surge", "pitch")]["added_mass_relative_peak"],
        allowed[("surge", "pitch")]["damping_relative_peak"],
    )
    assert surge_pitch > 1e1 * values["forbidden_worst_relative_peak"]


def test_every_gated_entry_closes_the_cummins_round_trip(record):
    entries = record["gates"]["M3"]["values"]["entries"]
    assert len(entries) == 36
    gated = [row for row in entries if row["allowed"]]
    assert len(gated) == 10
    for row in gated:
        assert row["damping_relative_error_max"] <= audit.ROUND_TRIP_TOLERANCE, row[
            "pair"
        ]
        assert row["kernel_causal"] is True
        assert row["kernel_even_parity_relative_error"] <= 1e-12


def test_cross_entry_kernel_keeps_its_sign(record):
    """``K_ij(0+)`` must follow the sign of ``B_ij``, not of ``|B_ij|``."""
    def key(pair):
        return tuple(sorted(pair, key=DOF_NAMES.index))

    entries = {
        key(row["pair"]): row
        for row in record["gates"]["M3"]["values"]["entries"]
    }
    sweep = load_sweep(SWEEP)
    for pair in (("surge", "pitch"), ("sway", "roll")):
        i, j = DOF_NAMES.index(pair[0]), DOF_NAMES.index(pair[1])
        spectrum = sweep.damping[:, i, j]
        row = entries[pair]
        assert row["impulse_relative_error"] <= audit.ROUND_TRIP_TOLERANCE
        assert math.copysign(1.0, row["kernel_at_zero"]) == math.copysign(
            1.0, float(np.sum(spectrum))
        )


def test_reciprocity_gate_reports_its_own_blind_spot(record):
    """The resolution section must state the break M1 cannot see."""
    values = record["gates"]["M1"]["values"]
    resolution = values["resolution"]
    assert resolution["resolvable_absolute_asymmetry"] > 0.0
    # the cross entries carry a few percent of the matrix scale, so a small
    # relative break there is genuinely below a matrix-scaled resolution
    assert resolution["smallest_resolvable_relative_break_on_surge_pitch"] > 0.01
    assert "per_entry" in values and len(values["per_entry"]) == 21


# ---------------------------------------------------------------------------
# gate internals: each leg must reject what it claims to reject
# ---------------------------------------------------------------------------
def test_round_trip_helper_refuses_a_sign_changing_spectrum():
    omega = np.linspace(0.2, 12.0, 41)
    spectrum = np.sin(omega)  # changes sign inside the band -> not passive
    row = audit._entry_round_trip(
        omega,
        spectrum,
        np.zeros_like(omega),
        np.arange(0.0, 1.0, 0.01),
    )
    assert "round_trip_skipped" in row


def test_round_trip_helper_normalises_a_negative_cross_spectrum():
    omega = np.linspace(0.2, 12.0, 41)
    spectrum = -(1.0 + 0.1 * omega)
    row = audit._entry_round_trip(
        omega,
        spectrum,
        np.zeros_like(omega),
        np.arange(0.0, 25.0, 0.002),
    )
    assert row["sign_applied"] == -1.0
    assert row["damping_relative_error_max"] <= audit.ROUND_TRIP_TOLERANCE


def test_zero_pattern_gate_rejects_a_filled_forbidden_entry():
    sweep = load_sweep(SWEEP)
    residual = float(np.max(sweep.solver_relative_residual))
    clean = audit.zero_pattern_gate(sweep, solver_residual=residual)
    dirty = audit.zero_pattern_gate(sweep, solver_residual=residual, corrupt=True)
    assert clean["passed"] is True
    assert dirty["passed"] is False


def test_markdown_render_mentions_every_gate(record):
    text = audit.render_markdown(record)
    for key in audit.GATE_ORDER:
        assert key in text
    assert "WITHHELD" in text  # the missing external anchor stays declared