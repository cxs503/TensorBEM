"""Invariants of the Hulme-hemisphere referee-robustness bundle.

These are not "the script ran" checks: each assertion binds a claim made in
``referee_robustness.md`` to the frozen numbers in ``referee_robustness.json``, so
an accidental regeneration that destroys the finding fails the suite.

The finding under test: the ``ka = 2.58`` cell that the earlier producer left as
``referee_unresolved`` is a *referee* defect (an irregular frequency of
capytaine's panel system), removable by closing a lid, and the lid-closed referee
reproduces the Hulme (1982) series within 3 % over the whole band.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BUNDLE = (Path(__file__).resolve().parents[1] / "benchmarks" / "verified"
          / "casepool_bem_hulme_hemisphere" / "referee_robustness.json")
MAGNITUDES = ("A11", "B11", "A33", "B33")


@pytest.fixture(scope="module")
def bundle() -> dict:
    if not BUNDLE.exists():
        pytest.skip(f"frozen referee bundle missing: {BUNDLE}")
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def test_pole_is_a_sign_crossing(bundle):
    """The lidless referee's B33 error must change sign inside the probe window."""
    rows = bundle["probes"]["sweep"]
    values = [(r["ka"], r["sub5"]["rel_error"]["B33"]) for r in rows]
    signs = [1 if v > 0 else -1 for _, v in values]
    flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    assert flips == 1, f"expected exactly one pole crossing, got {flips}: {values}"


def test_pole_blowup_and_far_field_recovery(bundle):
    """Peak |error| at the pole must be an order above the far-field error."""
    rows = bundle["probes"]["sweep"]
    peak = max(abs(r["sub5"]["rel_error"]["B33"]) for r in rows)
    far = max(abs(r["sub5"]["rel_error"]["B33"]) for r in rows if abs(r["ka"] - 2.58) > 0.2)
    assert peak > 1.0, peak
    assert far < 0.10, far


def test_lid_removes_the_pole(bundle):
    """With the lid closed the B33 error at the pole cells drops below 5 %."""
    rows = bundle["probes"]["lid"]
    pole = [r for r in rows if abs(r["ka"] - 2.58) <= 0.05]
    assert pole, "no pole cell in the lid arm"
    for r in pole:
        assert abs(r["sub5_lid"]["rel_error"]["B33"]) < 0.05, r["ka"]
        # ... and it really was broken without the lid.
        assert abs(r["sub5_open"]["rel_error"]["B33"]) > 0.30, r["ka"]


def test_lid_referee_matches_hulme_over_the_band(bundle):
    """The lid-closed referee must reproduce the series on all four magnitudes."""
    rows = bundle["probes"]["lid"]
    assert len(rows) >= 8
    worst = max(abs(r["sub5_lid"]["rel_error"][m]) for r in rows for m in MAGNITUDES)
    assert worst < 0.03, worst


def test_declared_fail_gates_stay_failed(bundle):
    """Honesty invariant: the pre-registered gates that failed are not softened."""
    gates = bundle["gates"]
    assert gates["R1_irregular_localised"]["verdict"] == "FAIL"
    assert gates["R2_irregular_mesh_dependent"]["verdict"] == "FAIL"
    assert gates["R4_in_band_three_way"]["verdict"] == "PASS"
    assert bundle["verdict"] == "FAIL"
    prov = bundle["gate_provenance"]
    assert "L2_lid_referee_accuracy" in prov["post_hoc"]
    assert "R1_irregular_localised" in prov["pre_registered"]
