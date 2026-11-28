"""Tests for the W32 RAO peak & m0 truncation-convergence artifact.

The W32 study (mission 28172) re-derives the W25 production-mesh RAO band
statistics in terms of the RAO peak and the response m0, with the stricter
acceptance gates of the seakeeping-truncation research plan (peak <= 2 %,
m0 <= 0.5 %).  These tests pin the artifact's machine-readable verdicts and
cross-check its numbers against the W25 artifact (m0 = sigma^2).
"""

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
W32 = ROOT / "papers/data/w32-2026-11-16/seakeeping/truncation_convergence.json"
W25 = (
    ROOT
    / "papers/artifacts/w25/seakeeping-truncation/journee_rao_truncation.json"
)


def test_w32_artifact_exists_and_schema():
    d = json.loads(W32.read_text())
    assert d["schema"] == "w32-truncation-convergence/v1"
    assert d["task"] == "paper-exp-seakeeping-truncation-7"
    assert d["mission_id"] == 28172
    assert "axis_A_truncation_frequency" in d
    assert "axis_B_truncation_radius_long_wave" in d
    assert "criteria" in d


def test_w32_m0_matches_w25_sigma_squared():
    """The band m0 must equal the W25 band sigma squared (same integrand).

    Only the levels shared by both ladders (3.0 / 6.0 / 12.0 rad/s) are
    cross-checked; the W32-only level 8.0 has no W25 counterpart.
    """
    w32 = json.loads(W32.read_text())
    w25 = json.loads(W25.read_text())
    shared = {3.0, 6.0, 12.0}
    for r32 in w32["axis_A_truncation_frequency"]["rows"]:
        if r32["omega_max"] not in shared:
            continue
        r25 = next(
            r
            for r in w25["axis_A_truncation_frequency"]["rows"]
            if r["sea_state"] == r32["sea_state"]
            and r["omega_max"] == r32["omega_max"]
        )
        assert r32["m0_heave"] == pytest.approx(r25["sigma_heave"] ** 2, rel=1e-12)
        assert r32["m0_pitch"] == pytest.approx(r25["sigma_pitch"] ** 2, rel=1e-12)


def test_w32_criteria_pass():
    """Both acceptance criteria must be PASS (converged within the ladder)."""
    d = json.loads(W32.read_text())
    c1 = d["criteria"]["criterion_1_rao_peak_relative_change"]
    c2 = d["criteria"]["criterion_2_m0_relative_change"]
    assert c1["gate_pct"] == 2.0
    assert c2["gate_pct"] == 0.5
    assert c1["all_pass"] is True
    assert c2["all_pass"] is True
    # every per-sea-state / per-quantity verdict is PASS
    for axis in ("axis_A_truncation_frequency", "axis_B_truncation_radius_long_wave"):
        for v in d[axis]["convergence_verdicts"].values():
            assert v["pass"] is True
            assert v["convergence_level"] is not None


def test_w32_m0_convergence_level_short_period():
    """T1=1.0 s (omega_p=4.85) needs omega_max >= 8 for m0 < 0.5%."""
    d = json.loads(W32.read_text())
    v = d["axis_A_truncation_frequency"]["convergence_verdicts"][
        "ss_T1_1p00:m0_heave"
    ]
    assert v["convergence_level"] == 8.0
    # at omega_max=6 the m0 is still > 0.5% short (physical tail, not a bug)
    assert v["trunc_error_by_level"]["6.0"] > 0.5


def test_w32_peak_convergence_level_pitch():
    """Pitch peak (omega=4.24) needs omega_max >= 6 to be inside the band."""
    d = json.loads(W32.read_text())
    v = d["axis_A_truncation_frequency"]["convergence_verdicts"][
        "ss_T1_1p00:peak_pitch"
    ]
    assert v["convergence_level"] == 6.0
    assert v["trunc_error_by_level"]["3.0"] > 2.0  # peak outside the band
