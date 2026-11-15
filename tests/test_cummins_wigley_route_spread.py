"""Invariants of the route-spread audit on CASE-BEM-CUMMINS-WIGLEY.

The point of these tests is not that the audit ran: it is that each of its three
claims is checked against something independent.

* the symmetry-block partition of the 6x6 radiation problem is measured, not
  assumed (``block_leakage``);
* the complex-resonance route is verified against a closed-form limit where the
  answer is known (constant coefficients: ``omega_n = sqrt(C/(m+A))``,
  ``zeta = B/(2 omega_n (m+A))``);
* the sampled-kernel route is *pinned as unusable*, so a later reader cannot
  quietly start quoting it (``kernel_root.usable is False`` with the span drift
  recorded);
* the published numbers are a function of the committed artifacts.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.benchmarks import cummins_wigley_route_spread as spread

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
AUDIT = POOL / "route_spread_audit.json"


@pytest.fixture(scope="module")
def sweep():
    return case.load_sweep(POOL / "sweep_medium.npz")


@pytest.fixture(scope="module")
def published():
    return json.loads(AUDIT.read_text())


def test_block_partition_is_measured_not_assumed(sweep):
    leakage = spread.block_leakage(sweep)
    # Pairs the hull's mirror symmetries forbid must sit at BEM noise level...
    assert leakage["blocks_are_exact"] is True
    assert leakage["worst_forbidden_relative"] < 1.0e-2
    # ...while the in-block coupling the per-DOF treatment drops is *not* small:
    # quoting "6DOF" while solving one DOF per mode is only defensible if the
    # size of what is dropped is measured, and here it is large.
    assert leakage["worst_allowed_in_block_relative"] > 0.3
    pairs = {(row["dofs"][0], row["dofs"][1]): row for row in leakage["allowed_in_block_pairs"]}
    assert pairs[("surge", "pitch")]["added_mass"] > 0.1
    assert pairs[("sway", "roll")]["added_mass"] > 0.05


def test_complex_resonance_reproduces_the_constant_coefficient_limit(sweep):
    """With A and B frozen, the resonance root *is* the quadratic formula."""
    dof = 2
    oscillator = case.oscillator_properties(dof, label=sweep.label)
    omega_n = math.sqrt(oscillator["stiffness"] / (oscillator["inertia"] + 20.0))
    damping = 2.0 * 0.1 * omega_n * (oscillator["inertia"] + 20.0)
    added = np.repeat(np.array([[[20.0]]]), sweep.omega.size, axis=0)
    damper = np.repeat(np.array([[[damping]]]), sweep.omega.size, axis=0)
    frozen = dataclasses.replace(sweep, added_mass=np.repeat(
        np.eye(6)[None, :, :] * 0.0, sweep.omega.size, axis=0))
    # place the constants on the (2,2) entry of a diagonal 6x6 sweep
    added6 = np.zeros_like(sweep.added_mass)
    damp6 = np.zeros_like(sweep.damping)
    added6[:, dof, dof] = added[:, 0, 0]
    damp6[:, dof, dof] = damper[:, 0, 0]
    frozen = dataclasses.replace(sweep, added_mass=added6, damping=damp6)
    root = spread.complex_resonance(frozen, dof, guess=omega_n, window=0.2)
    assert root["found"] is True
    # the solver reports the damped frequency (Re Omega = omega_n sqrt(1-zeta^2))
    assert root["omega"] == pytest.approx(omega_n * math.sqrt(1.0 - 0.1**2), rel=1e-6)
    assert root["zeta"] == pytest.approx(0.1, rel=1e-6)
    quad = spread.quadratic_zeta(frozen, dof, omega_n)
    assert quad["zeta"] == pytest.approx(0.1, rel=1e-6)


def test_kernel_root_route_is_falsified(published):
    """A root that moves with the kernel span is reporting the truncation."""
    for row in published["modes"].values():
        kernel = row["kernel_root"]
        assert kernel["usable"] is False
        assert kernel["spread_over_span"] > 0.05
        spans = [entry["span_s"] for entry in kernel["rows"]]
        assert spans == sorted(spans)


def test_single_frequency_reference_sensitivity_is_recorded(published):
    for row in published["modes"].values():
        dispersion = row["dispersion"]
        # the estimator is steep: more than one percent of zeta per percent of
        # omega on the flanks of the resonance
        assert abs(dispersion["d_log_zeta_d_log_omega"]) > 1.0
        assert dispersion["relative_range"] > 0.05


def test_published_numbers_are_a_function_of_the_committed_artifacts(sweep, published):
    decay = json.loads((POOL / "decay_dof_audit.json").read_text())
    bar = published["gate"]["values"]["route_spread_bar"]
    for name, published_row in published["modes"].items():
        dof = case.DOF_NAMES.index(name)
        row = spread.route_spread(sweep, dof, decay_row=decay["modes"][name]["values"], bar=bar)
        assert row["zeta_decay"] == pytest.approx(published_row["zeta_decay"], rel=1e-12)
        assert row["routes"]["complex_resonance"] == pytest.approx(
            published_row["routes"]["complex_resonance"], rel=1e-10
        )
        assert row["route_spread"] == pytest.approx(published_row["route_spread"], rel=1e-10)


def test_gate_scores_only_what_it_can_decide(published):
    gate = published["gate"]
    assert gate["passed"] is True
    assert gate["values"]["scored_modes"] == ["heave", "pitch"]
    assert gate["values"]["reported_modes"] == ["roll"]
    roll = published["modes"]["roll"]
    assert roll["scored"] is False
    assert roll["grade"] == "route_spread_beyond_bar"
    # the roll number is reported, not hidden: its residual is larger than the
    # case tolerance, which is exactly why it may not be scored
    assert roll["residual_decay_vs_resonance"] > roll["case_tolerance"]


def test_coupled_block_modes_matches_the_per_dof_roots(published):
    """The 47% in-plane coupling must not move the heave/pitch roots."""
    vertical = published["coupled_block_modes"]["vertical"]
    assert vertical["usable"] is True
    block = {round(row["omega"], 2): row for row in vertical["roots"]}
    single = {
        round(published["modes"][name]["omega_resonance"], 2): published["modes"][name]
        for name in ("heave", "pitch")
    }
    for omega, row in single.items():
        nearest = min(block, key=lambda candidate: abs(candidate - omega))
        assert abs(nearest - omega) < 0.05
        assert block[nearest]["zeta"] == pytest.approx(row["routes"]["complex_resonance"], rel=2e-2)


def test_blocks_with_two_free_modes_are_marked_unusable(published):
    assert published["coupled_block_modes"]["horizontal"]["usable"] is False