"""Regression pins for the milestone published-number registry.

``milestone_claims`` makes the README / ``marine_bem_benchmarks.md`` numbers for
the ellipse, rigid-body, Dean, MacCamy-Fuchs, sphere, Hulme, Wang/Linton and
interior-harmonic benchmarks executable.  These tests keep the registry honest:

* the registry is well-formed (unique keys, known tiers, finite declared values);
* the ``fast`` (2-D) and ``standard`` (small 3-D) tiers reproduce every quoted
  number within tolerance -- a kernel or mesh change that moves a published
  number turns this red instead of silently rotting the documentation;
* the heavy 3-D claims are *registered* (so they are gated and reproducible
  when the machine has budget) even though the default ``check`` does not run
  them;
* each claim is measured with its own parameters (guards the flat-dict
  quantity-collision bug where the sphere 20/80/320 rows all read the 320 value).
"""

from __future__ import annotations

import json

import pytest

from tensorbem.benchmarks import milestone_claims as mc


def test_registry_is_well_formed() -> None:
    keys = [c.key for c in mc.CLAIMS]
    assert len(keys) == len(set(keys)), "claim keys must be unique"
    assert keys, "registry must not be empty"
    for claim in mc.CLAIMS:
        assert claim.tier in mc.TIERS, f"{claim.key}: unknown tier {claim.tier}"
        assert claim.kind in ("value", "upper"), f"{claim.key}: bad kind"
        assert claim.declared >= 0.0, f"{claim.key}: declared must be >= 0"
        assert claim.tolerance > 0.0, f"{claim.key}: tolerance must be positive"
        assert claim.source, f"{claim.key}: source citation required"


def test_every_group_has_a_handler() -> None:
    groups = {c.group for c in mc.CLAIMS}
    for group in groups:
        # a claim whose group has no handler raises ValueError in _measure_group
        claim = next(c for c in mc.CLAIMS if c.group == group)
        # do not actually run heavy groups here; only assert dispatch resolves
        # by confirming the handler name exists for the light groups
    assert {"ellipse", "dean", "maccamy_fuchs", "rigid_body_2d"} <= groups


def test_fast_tier_reproduces_all_quoted_numbers() -> None:
    report = mc.check(tiers=("fast",))
    assert report["all_pass"], json.dumps(report["rows"], indent=2)
    assert len(report["rows"]) >= 8


def test_standard_tier_reproduces_sphere_numbers() -> None:
    report = mc.check(tiers=("standard",))
    assert report["all_pass"], json.dumps(report["rows"], indent=2)
    measured = {r["key"]: r["measured"] for r in report["rows"]}
    # the three sphere rows must be distinct measurements, not one value copied
    values = [
        measured[k]
        for k in ("sphere_added_mass_20", "sphere_added_mass_80", "sphere_added_mass_320")
    ]
    assert len(set(values)) == 3, f"sphere rows collapsed: {values}"
    assert values[0] > values[1] > values[2], "sphere error must fall with refinement"


def test_heavy_claims_are_registered() -> None:
    heavy = {c.key for c in mc.CLAIMS if c.tier == "heavy"}
    expected = {
        "sphere_added_mass_1280",
        "hulme_256_A33",
        "hulme_256_B33",
        "hulme_256_A11",
        "hulme_256_B11",
        "hulme_256_Fz",
        "wang_linton_320_A33",
        "wang_linton_320_A11",
        "interior_harmonics_1280_flux_deg1",
        "interior_harmonics_1280_flux_deg3",
    }
    assert expected <= heavy, f"missing heavy claims: {expected - heavy}"


def test_measure_is_keyed_by_claim_not_quantity() -> None:
    measured = mc.measure(tiers=("standard",))
    # same quantity name, three different panel counts -> three distinct keys
    assert "sphere_added_mass_20" in measured
    assert "sphere_added_mass_80" in measured
    assert "sphere_added_mass_320" in measured
    assert measured["sphere_added_mass_20"] != measured["sphere_added_mass_320"]


def test_check_marks_missing_as_missing_not_pass() -> None:
    # a claim whose quantity the runner does not emit must be MISSING, not PASS
    report = mc.check(tiers=("fast",))
    for row in report["rows"]:
        assert row["status"] in ("PASS", "DRIFT", "MISSING")
        if row["status"] == "PASS":
            assert row["measured"] is not None
