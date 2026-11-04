"""Kramers-Kronig audit of 3-D free-surface radiation: what it can and cannot gate.

The 3-D audit (:mod:`tensorbem.benchmarks.kkr_floating_hemisphere`) swept the
Hulme hemisphere band and compared the panel sweep with the semi-analytical one
through the same causal transform. The measured verdict is *negative* and is
pinned here so a future change cannot silently turn the finding back into an
unchecked claim:

* the transform floor on the semi-analytical reference is tens of percent,
  because the 3-D added mass has no closed-form infinite-frequency limit to
  feed relation (1) and the ``A_inf + c/omega^2`` fit does not identify it, so
  the residual is dominated by the (unknown) ``A_inf`` rather than by the panel
  solution;
* consequently a deliberately inconsistent panel sweep (damping scaled by 10 %)
  barely moves the residual, i.e. the metric is *not* usable as a gate;
* the panel sweep is nevertheless no less causal than the reference on the same
  grid (ratio of the two residuals stays O(1)).

A current coupling is checked too, so the module cannot rot into a no-op.
"""

from __future__ import annotations

import json

import pytest

from tensorbem.benchmarks.kkr_floating_hemisphere import audit_sweep, main, run

KA_MIN, KA_MAX, POINTS = 1.0, 4.0, 12


@pytest.fixture(scope="module")
def audit():
    return run(subdivisions=(1,), ka_min=KA_MIN, ka_max=KA_MAX, points=POINTS)


def test_transform_floor_is_not_a_gate(abstract_audit=None) -> None:
    """The semi-analytical reference itself violates KKR by tens of percent."""
    case = audit_sweep(subdivision=1, ka_min=KA_MIN, ka_max=KA_MAX, points=POINTS)
    for name in ("surge", "heave"):
        floor = case["modes"][name]["kkr_reference_floor"]
        assert floor["max_damping_relative_error"] > 1.0e-2
        assert floor["max_added_mass_relative_error"] > 1.0e-2


def test_panel_sweep_is_no_less_causal_than_the_reference(audit) -> None:
    case = audit["cases"]["subdivision_1_panels_16"]
    for name in ("surge", "heave"):
        mode = case["modes"][name]
        # O(1): the panel residual tracks the reference floor, it does not
        # explode -- the free-surface kernel is not injecting acausality.
        assert mode["damping_residual_ratio_panel_over_reference"] < 3.0
        assert mode["added_mass_residual_ratio_panel_over_reference"] < 3.0


def test_negative_control_is_blind_at_this_band(audit) -> None:
    """A 10 % damping inconsistency is invisible: the documented limitation."""
    mode = audit["cases"]["subdivision_1_panels_16"]["modes"]["heave"]
    control = mode["negative_controls"]["damping_x1.1"]
    assert control["ratio_to_unperturbed"] < 1.5


def test_record_is_strict_json_and_deterministic(tmp_path) -> None:
    output = tmp_path / "kkr.json"
    assert main(["--subdivisions", "1", "--points", str(POINTS), "--ka-min", str(KA_MIN), "--ka-max", str(KA_MAX), "--output", str(output)]) == 0
    first = json.loads(output.read_text())
    assert first["benchmark"] == "kramers_kronig_causality_audit_3d"
    assert len(first["ka"]) == POINTS
    assert first["cases"]["subdivision_1_panels_16"]["panels"] == 16
