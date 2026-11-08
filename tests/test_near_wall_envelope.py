"""Near-wall accuracy envelope: conservative lookup + measured convergence (m6445).

The tests split into two halves, on purpose:

* **unit** tests on :mod:`tensorbem.near_wall_envelope` with synthetic data -- the
  envelope must never be optimistic, its inverse must round-trip, and a query
  below the measured range must be refused rather than extrapolated;
* **measured** tests that re-run a small slice of the benchmark and compare with
  the committed ``docs/near_wall_envelope/study.json``.  The point of the
  measured half is the claim correction: at *fixed physical clearance* the
  near-wall velocity converges, which the merged
  ``docs/near_surface_error_budget.md`` §5 says it does not.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorbem import near_wall_envelope as nwe
from tensorbem.benchmarks import near_wall_envelope_study as study
from tensorbem.geometry3d import icosphere

STUDY_PATH = pathlib.Path("docs/near_wall_envelope/study.json")


# --------------------------------------------------------------------------
# unit: the lookup
# --------------------------------------------------------------------------


def test_envelope_is_monotone_and_never_optimistic() -> None:
    ratios = [0.1, 0.2, 0.3, 0.6, 1.2]
    # deliberately noisy: the 0.2 sample is larger than the 0.1 sample
    errors = [0.5, 0.7, 0.09, 0.01, 0.002]
    envelope = nwe.build_envelope(ratios, errors)
    assert np.all(np.diff(envelope.clearance_ratio) > 0.0)
    assert np.all(np.diff(envelope.relative_error) <= 0.0)
    # no sample is reported better than it was measured
    for ratio, error in zip(ratios, errors):
        assert envelope.bound(ratio) >= error - 1e-15


def test_bound_steps_down_to_the_measured_ratio() -> None:
    envelope = nwe.build_envelope([0.3, 0.6], [0.1, 0.01])
    assert envelope.bound(0.6) == pytest.approx(0.01)
    # a query between two rungs uses the *smaller* rung (conservative step)
    assert envelope.bound(0.59) == pytest.approx(0.1)
    assert envelope.bound(5.0) == pytest.approx(0.01)


def test_below_the_measured_range_is_refused_not_extrapolated() -> None:
    envelope = nwe.build_envelope([0.3, 0.6], [0.1, 0.01])
    assert math.isinf(envelope.bound(0.05))
    with pytest.raises(ValueError):
        envelope.bound(0.0)
    with pytest.raises(ValueError):
        envelope.required_clearance_ratio(1e-9)
    with pytest.raises(ValueError):
        nwe.build_envelope([0.3], [0.1, 0.2])
    with pytest.raises(ValueError):
        nwe.NearWallEnvelope(
            clearance_ratio=np.array([0.5, 0.3]), relative_error=np.array([0.1, 0.2])
        )


def test_required_clearance_inverts_the_envelope_in_metres() -> None:
    envelope = nwe.build_envelope([0.3, 0.45, 0.6, 0.9], [0.12, 0.04, 0.014, 0.004])
    assert envelope.required_clearance_ratio(0.03) == pytest.approx(0.6)
    assert nwe.required_clearance(0.03, 0.25, envelope) == pytest.approx(0.15)
    ok = envelope.check(0.6 * 0.25, 0.25, 0.03)
    assert ok["ok"] is True and ok["bound_relative_error"] <= 0.03
    bad = envelope.check(0.3 * 0.25, 0.25, 0.03)
    assert bad["ok"] is False and bad["required_d_over_h"] == pytest.approx(0.6)


def test_envelope_round_trips_through_json(tmp_path) -> None:
    envelope = nwe.build_envelope([0.3, 0.6], [0.1, 0.01])
    payload = {"envelope": envelope.as_dict()}
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = nwe.load_envelope(path)
    assert np.array_equal(loaded.clearance_ratio, envelope.clearance_ratio)
    assert np.array_equal(loaded.relative_error, envelope.relative_error)
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text(json.dumps({"other": 1}), encoding="utf-8")
    with pytest.raises(KeyError):
        nwe.load_envelope(unrelated)
    with pytest.raises(ValueError):
        nwe.NearWallEnvelope.from_dict({"clearance_ratio": [], "relative_error": []})


# --------------------------------------------------------------------------
# measured: fixed physical clearance converges
# --------------------------------------------------------------------------


def _error(subdivisions: int, direction: str, clearance: float) -> float:
    mesh = icosphere(subdivisions=subdivisions, radius=1.0)
    return study.measure(mesh, direction, clearance)["relative_error"]


def test_fixed_physical_clearance_converges_under_refinement() -> None:
    """Refinement at fixed ``d`` must improve the near-wall velocity (equator, worst probe)."""
    clearance = 0.05
    coarse = _error(2, "equator", clearance)
    fine = _error(3, "equator", clearance)
    assert coarse > 0.2, coarse  # the honest FAIL magnitude is reproduced
    assert fine < coarse / 4.0, (coarse, fine)
    order = math.log(coarse / fine) / math.log(2.0)
    assert order > study.MIN_FIXED_CLEARANCE_ORDER, order


def test_fixed_relative_clearance_is_scale_invariant() -> None:
    """The same experiment at fixed ``d / h_max`` must *not* improve -- both halves hold."""
    ratio = 0.3
    errors = []
    for subdivisions in (2, 3):
        mesh = icosphere(subdivisions=subdivisions, radius=1.0)
        errors.append(study.measure(mesh, "equator", ratio * mesh.h_max)["relative_error"])
    assert errors[0] > 0.08 and errors[1] > 0.08, errors
    assert max(errors) / min(errors) < study.SCALE_INVARIANCE_FACTOR, errors


def test_measured_slice_certifies_the_three_percent_clearance() -> None:
    """At ``d = 0.6 h_max`` the measured worst probe is inside the 3 % bar."""
    for subdivisions in (2, 3):
        mesh = icosphere(subdivisions=subdivisions, radius=1.0)
        for direction in ("equator", "pole", "oblique"):
            error = study.measure(mesh, direction, 0.6 * mesh.h_max)["relative_error"]
            assert error < study.TARGET_ERROR, (subdivisions, direction, error)


# --------------------------------------------------------------------------
# committed artefact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads(STUDY_PATH.read_text(encoding="utf-8"))


def test_published_study_records_the_gates(published: dict) -> None:
    verdict = published["verdict"]
    assert verdict["fixed_clearance_gate"]["pass"] is True
    assert verdict["fixed_clearance_gate"]["order_two_finest"] > study.MIN_FIXED_CLEARANCE_ORDER
    assert verdict["scale_invariance_pass"] is True
    assert verdict["envelope_certifies_3pct"] is True
    assert verdict["refinement_is_a_lever"] is True


def test_published_orders_match_the_published_rows(published: dict) -> None:
    """Every order in the JSON must be recomputable from the JSON's own error ladder."""
    rows = published["fixed_clearance"]["rows"]
    for order in published["fixed_clearance"]["orders"]:
        ladder = [
            row
            for row in rows
            if row["direction"] == order["direction"]
            and math.isclose(row["clearance"], order["clearance"], rel_tol=1e-12)
        ]
        ladder.sort(key=lambda row: row["h_max"], reverse=True)
        recomputed = study.observed_order(
            [row["h_max"] for row in ladder], [row["relative_error"] for row in ladder]
        )
        assert recomputed["order_two_finest"] == pytest.approx(order["order_two_finest"], rel=1e-9)
        assert recomputed["order_full_ladder"] == pytest.approx(order["order_full_ladder"], rel=1e-9)


def test_published_envelope_is_consistent_with_the_sweep(published: dict) -> None:
    envelope = nwe.NearWallEnvelope.from_dict(published["envelope"])
    rebuilt = study.envelope_from_sweep(published["ratio_sweep"])
    assert np.allclose(rebuilt.clearance_ratio, envelope.clearance_ratio)
    assert np.allclose(rebuilt.relative_error, envelope.relative_error)
    # no published sweep sample may exceed the envelope at its own ratio
    for row in published["ratio_sweep"]["rows"]:
        assert envelope.bound(row["d_over_h"]) >= row["relative_error"] - 1e-12
    # the certified 3 % clearance must agree with the requirement table
    requirement = next(
        row
        for row in published["clearance_requirements"]
        if math.isclose(row["target_relative_error"], study.TARGET_ERROR)
    )
    assert requirement["certified"] is True
    assert requirement["required_d_over_h"] == pytest.approx(
        envelope.required_clearance_ratio(study.TARGET_ERROR)
    )


def test_driver_writes_a_summary_from_its_own_payload(tmp_path) -> None:
    payload = study.audit(
        subdivisions=(2, 3),
        envelope_subdivisions=(2,),
        clearances=(0.05,),
        ratios=(0.6, 0.3),
        directions=("equator",),
        targets=(0.03,),
    )
    out = tmp_path / "summary.md"
    study.write_summary(payload, out)
    text = out.read_text(encoding="utf-8")
    assert "## 2. Fixed physical clearance" in text
    assert "order (2 finest)" in text
    assert "certified" in text
    verdict = payload["verdict"]
    assert verdict["fixed_clearance_gate"]["clearance"] == 0.05
    # a two-mesh, equator-only slice still converges (never a stall)
    assert verdict["fixed_clearance_gate"]["improvement_factor"] > 4.0