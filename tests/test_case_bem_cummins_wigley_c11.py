"""Gate C11 (state-space reduction) of CASE-BEM-CUMMINS-WIGLEY.

The gate implementation was dead code until m23122: it called
``validate_state_space`` without its two required arguments and used two
``tensorbem.time_domain`` entry points that do not exist, so the first call
raised and every later call would have returned ``passed=False`` with an API
error string -- a gate that reports a physics FAIL for an API mismatch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import case_bem_cummins_wigley as case

ARTIFACT = Path("benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz")


@pytest.fixture(scope="module")
def sweep():
    return case.load_sweep(ARTIFACT)


@pytest.fixture(scope="module")
def heave_gate(sweep):
    return case.state_space_gate(sweep, dof=2)


def test_c11_runs_and_decides_heave(heave_gate):
    """The gate must be *executable* and score the committed heave spectrum."""
    values = heave_gate["values"]
    assert values["status"] == "decided", values["status"]
    assert values["fit_error"] <= values["fit_target_error"]
    assert values["all_poles_stable"] is True
    assert values["damping_nonnegative_in_band"] is True
    assert heave_gate["passed"] is True
    # the ROM route and the direct convolution must agree to the declared bar
    assert values["zeta_relative_error"] <= case.STATE_SPACE_BAR
    assert values["period_relative_error"] <= case.STATE_SPACE_BAR


def test_c11_scores_against_the_measured_spectrum_not_its_own_fit(heave_gate):
    values = heave_gate["values"]
    assert values["fit_error"] >= values["fit_error_self"] - 1e-12
    assert values["zeta_convolution"] > 0.0


def test_c11_negative_control_fails(sweep):
    """A gate that cannot fail is not a gate: corrupt the spectrum, it must fail."""
    control = case.state_space_gate(sweep, dof=2, corrupt=True)
    assert control["values"]["negative_control"] is True
    assert control["passed"] is False
    assert control["values"]["fit_error"] > control["values"]["fit_target_error"]
    assert control["id"] == "C11_control"


def test_c11_reports_an_under_resolved_rom_as_undecidable(sweep):
    """An order-limited ROM is not a physics failure, and must say so."""
    gate = case.state_space_gate(sweep, dof=3, max_poles=4, target_error=0.03)
    values = gate["values"]
    assert gate["passed"] is False
    assert values["status"].startswith("undecidable:")
    assert "reduction order" in values["status"]
    assert values["fit_error"] > values["fit_target_error"]


def test_c11_refuses_modes_without_hydrostatic_restoring(sweep):
    with pytest.raises(ValueError, match="hydrostatic restoring"):
        case.state_space_gate(sweep, dof=5)


def test_c11_audit_covers_every_restoring_mode(sweep):
    audit = case.state_space_audit(sweep, max_poles=4)
    assert set(audit["per_mode"]) == {"heave", "roll", "pitch"}
    assert audit["negative_control"]["passed"] is False
    assert audit["scored_dof"] == "heave"
    assert audit["bar"] == case.STATE_SPACE_BAR
    json.dumps(audit)  # must be JSON-serialisable for the case artifact


def test_c11_is_not_folded_into_the_published_verdict_by_default():
    """The published 16-gate verdict must not change silently."""
    replay = case.gates_from_artifact(ARTIFACT, include_external_anchor=False,
                                      include_decay_modes=False,
                                      corrupted_control=False)
    assert "C11" not in replay["gates"]
    assert replay["state_space_reduction"] is None
    replay_c11 = case.gates_from_artifact(
        ARTIFACT, include_external_anchor=False, include_decay_modes=False,
        corrupted_control=False, include_state_space=True,
    )
    assert replay_c11["gates"]["C11"]["passed"] is True
    assert replay_c11["state_space_reduction"]["scored_dof"] == "heave"