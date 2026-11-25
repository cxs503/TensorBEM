"""Tests for the case-pool record auditor (:mod:`tensorbem.case_pool_verdict`).

The auditor exists because a committed ``results.json`` is a certificate: its
numbers must support its own verdict.  Every test below either pins one reading
of that claim or -- more importantly -- proves that the auditor *can* fail, so a
clean report means something.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from tensorbem import case_pool_verdict as audit


def _record(**overrides):
    payload = {
        "case_id": "CASE-TEST-DUMMY",
        "schema": "tensorbem/case_test/1",
        "overall": "PASS",
        "criteria": [
            {
                "name": "velocity_relative_l2",
                "value": 1.0e-4,
                "threshold": "< 0.03",
                "ok": True,
                "note": "",
            },
            {
                "name": "guard_is_armed",
                "value": True,
                "threshold": "True on every level",
                "ok": True,
                "note": "prose only",
            },
        ],
        "acceptance": {
            "passed": True,
            "failed_checks": [],
            "tolerance": 0.03,
            "n_checks": 2,
        },
        "routes": {
            "dirichlet": {
                "levels": [
                    {"panels": 32, "h_mean": 1.0 / 32.0, "velocity_relative_l2": 4.0e-3},
                    {"panels": 64, "h_mean": 1.0 / 64.0, "velocity_relative_l2": 1.0e-3},
                ],
                "observed_order_velocity": [2.0],
                "finest": {"panels": 64, "velocity_relative_l2": 1.0e-3},
            }
        },
    }
    payload.update(overrides)
    return payload


def test_threshold_parsing_covers_the_shapes_the_case_modules_publish():
    assert audit.parse_threshold("< 0.03") == ("<", 0.03)
    assert audit.parse_threshold("<= 1e-12 a") == ("<=", 1e-12)
    assert audit.parse_threshold(">= 1.5") == (">=", 1.5)
    assert audit.parse_threshold("> 0") == (">", 0.0)
    # prose-only thresholds must not be mistaken for machine-readable ones
    assert audit.parse_threshold("in [0.5, 2.0]") is None
    assert audit.parse_threshold("True on every level") is None
    assert audit.parse_threshold(None) is None


def test_a_value_that_straddles_its_own_threshold_is_not_a_pass():
    assert audit.compare(0.03, "<", 0.03) is False
    assert audit.compare(0.03, "<=", 0.03) is True
    assert audit.compare(float("nan"), "<", 0.03) is False
    with pytest.raises(TypeError):
        audit.compare("0.01", "<", 0.03)


def test_clean_record_audits_clean():
    report = audit.audit_record(_record())
    assert report["ok"] is True
    assert report["criteria"]["n_agree"] == 1
    assert report["criteria"]["unverified"] == ["guard_is_armed"]
    assert report["acceptance"]["problems"] == []
    assert report["order_mismatches"] == []


def test_recorded_verdict_that_contradicts_its_numbers_is_blocking():
    payload = _record()
    payload["criteria"][0]["value"] = 0.5  # published threshold is "< 0.03"
    report = audit.audit_record(payload)
    assert report["blocking"] is True
    assert [row["name"] for row in report["criteria"]["disagreements"]] == [
        "velocity_relative_l2"
    ]


def test_acceptance_block_must_summarise_the_criteria_list():
    payload = _record()
    payload["criteria"][0]["value"] = 0.5
    payload["criteria"][0]["ok"] = True
    payload["acceptance"] = {
        "passed": True,
        "failed_checks": [],
        "tolerance": 0.03,
        "n_checks": 2,
    }
    report = audit.audit_record(payload)
    assert report["blocking"] is True

    payload = _record()
    payload["criteria"][0]["ok"] = False
    payload["criteria"][0]["value"] = 0.5
    payload["overall"] = "PASS"
    report = audit.audit_record(payload)
    assert any("failed_checks" in problem for problem in report["acceptance"]["problems"])
    assert any("passed=" in problem for problem in report["acceptance"]["problems"])


def test_declared_tolerance_looser_than_the_pool_tolerance_is_flagged():
    payload = _record()
    payload["acceptance"]["tolerance"] = 0.2
    report = audit.audit_record(payload)
    assert report["blocking"] is True
    assert any("looser" in problem for problem in report["acceptance"]["problems"])


def test_order_that_does_not_follow_from_the_published_ladder_is_blocking():
    payload = _record()
    # 4e-3 -> 1e-3 with the spacing halved is an observed order of 2, not 1.
    payload["routes"]["dirichlet"]["observed_order_velocity"] = [1.0]
    report = audit.audit_record(payload)
    assert report["blocking"] is True
    mismatch = report["order_mismatches"][0]
    assert mismatch["key"] == "observed_order_velocity"
    assert mismatch["mismatches"][0]["recomputed"] == pytest.approx(2.0)


def test_finest_row_must_repeat_the_last_rung():
    payload = _record()
    payload["routes"]["dirichlet"]["finest"]["velocity_relative_l2"] = 5.0e-4
    report = audit.audit_record(payload)
    assert report["blocking"] is True
    assert any(
        finding["key"] == "finest.velocity_relative_l2"
        for finding in report["order_mismatches"]
    )


def test_ladder_criterion_is_rederived_from_its_rungs():
    payload = _record()
    payload["criteria"].append(
        {
            "name": "observed_order_over_ladder",
            "value": [2.0, 0.3],
            "threshold": ">= 1.5",
            "ok": True,
            "note": "the worst rung is below the bar but the first is not",
        }
    )
    payload["acceptance"]["n_checks"] = 3
    report = audit.audit_record(payload)
    # Some reading (the first rung, or the last-rung convention) reproduces the
    # recorded verdict, but the two readings disagree -- report which one does.
    assert report["blocking"] is False
    assert report["criteria"]["aggregation_ambiguous"] == [
        {"name": "observed_order_over_ladder", "satisfied": ["first", "max"]}
    ]


def test_non_case_record_is_reported_unsupported_rather_than_failed():
    payload = {"case_id": "X", "criteria": ["a", "b"], "notes": "older schema"}
    report = audit.audit_record(payload)
    assert report["supported"] is False
    assert report["ok"] is True


def test_cli_exit_status_follows_the_audit(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_record()))
    assert audit.main([str(good), "--quiet"]) == 0
    bad = _record()
    bad["criteria"][0]["value"] = 0.5
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad))
    assert audit.main([str(bad_path), "--quiet"]) == 1


def test_committed_cylinder_record_re_derives_cleanly():
    """The anchor's committed certificate must survive its own numbers.

    This is the cheap half of the anchor's verification debt: the solver is not
    re-run here, but every published threshold is re-applied to the published
    value, the acceptance block is matched against the criteria list, and the
    convergence orders are recomputed from the ladder that supports them.
    """
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "benchmarks/verified/case_bem_laplace_cylinder/results.json").read_text()
    )
    report = audit.audit_record(payload)
    assert report["supported"] is True
    assert report["criteria"]["n_criteria"] == len(payload["criteria"])
    assert report["criteria"]["disagreements"] == []
    assert report["criteria"]["non_numeric"] == []
    assert report["acceptance"]["problems"] == []
    assert report["order_mismatches"] == []
    assert report["blocking"] is False, json.dumps(report, indent=1, default=str)
    # the audit must be doing more than echoing the verdict
    assert report["criteria"]["n_agree"] >= 20
    assert math.isfinite(float(report["acceptance"]["declared_tolerance"]))

def test_structured_near_wall_gate_rechecks_each_mesh():
    criterion = {"name": "near_wall_window_is_azimuth_independent",
                 "value": {"worst_over_levels": .006, "levels": [{"worst_error": .006}, {"worst_error": .018}]},
                 "threshold": "<= 0.01 on every level", "ok": True}
    report = audit.audit_criteria([criterion])
    assert report["disagreements"][0]["name"] == criterion["name"]
    del criterion["value"]["levels"][1]["worst_error"]
    assert audit.audit_criteria([criterion])["non_numeric"]
