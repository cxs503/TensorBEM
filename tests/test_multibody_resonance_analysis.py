"""Tests for the m9161 re-analysis of the multibody frequency-dependence study.

Fast checks only: the analytic figures of merit are exercised on synthetic data
whose answer is known, and the recorded study is *pinned* (the resonance band,
the preserved original verdicts, the wavenumber-gated far field) rather than
recomputed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.multibody_frequency_dependence_3d import (
    KD_FAR_FIELD_MIN,
    RESONANCE_PEAK_RATIO,
    detect_resonance_bands,
    kk_asymptote_from_series,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks" / "multibody_frequency_dependence_3d" / "results.json"


def _row(label, ratio, heave_a, heave_b):
    """One scan row whose *heave* mode carries the interesting series.

    Sway and pitch are held flat so the detector is asked about exactly one
    mode; a row that repeats the same series in all three modes would (rightly)
    report three bands.
    """
    return {
        "label": label,
        "omega_over_omega_n": ratio,
        "added_mass": [[100.0, 0.0, 0.0], [0.0, heave_a, 0.0], [0.0, 0.0, 50.0]],
        "damping": [[10.0, 0.0, 0.0], [0.0, heave_b, 0.0], [0.0, 0.0, 5.0]],
        "resolved": True,
    }


# ---------------------------------------------------------------------------
# unit: resonance detector
# ---------------------------------------------------------------------------


def test_detector_finds_a_narrow_damping_peak_with_an_added_mass_sign_change():
    ratios = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    damping = [100.0, 130.0, 600.0, 140.0, 90.0, 60.0, 40.0]
    added = [200.0, 150.0, -80.0, 20.0, 60.0, 90.0, 110.0]
    rows = [
        _row("catamaran", ratio, a, b)
        for ratio, a, b in zip(ratios, added, damping)
    ]
    bands = detect_resonance_bands(rows)
    assert len(bands) == 1
    band = bands[0]
    assert band["mode"] == "heave"
    assert band["omega_over_omega_n"] == pytest.approx(1.1)
    assert band["omega_over_omega_n_range"] == [1.05, 1.15]
    assert band["b_peak_over_median"] >= RESONANCE_PEAK_RATIO
    assert band["added_mass_sign_change"] is True


def test_detector_ignores_a_smooth_series():
    ratios = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    rows = [
        _row("catamaran", ratio, 200.0 - 50.0 * ratio, 100.0 * math.exp(-ratio))
        for ratio in ratios
    ]
    assert detect_resonance_bands(rows) == []


def test_detector_requires_enough_points():
    rows = [_row("catamaran", 1.0 + 0.1 * i, 100.0, 100.0) for i in range(4)]
    assert detect_resonance_bands(rows) == []


# ---------------------------------------------------------------------------
# unit: Kramers-Kronig asymptote from a series
# ---------------------------------------------------------------------------


def test_kk_asymptote_recovers_a_lorentzian_moment():
    """One resonance: B = B0 * g^2 / ((w-w0)^2+g^2) has an analytic moment."""
    omegas = np.linspace(0.05, 25.0, 4001)
    w0, gamma, b0 = 4.0, 0.6, 50.0
    damping = b0 * gamma**2 / ((omegas - w0) ** 2 + gamma**2)
    exact = float((2.0 / math.pi) * np.trapezoid(damping / omegas**2, omegas))
    a_zero = 100.0
    entry = kk_asymptote_from_series(omegas, damping, a_zero)
    assert entry["a_inf"] == pytest.approx(a_zero - exact, rel=5e-3)
    assert entry["band_moment"] > 0.0


def test_kk_asymptote_needs_four_positive_points():
    entry = kk_asymptote_from_series([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 10.0)
    assert math.isnan(entry["a_inf"])
    assert entry["resolved_rows"] == 3


# ---------------------------------------------------------------------------
# record pins
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not RECORD.exists():  # pragma: no cover - the record ships with the study
        pytest.skip("study record absent")
    return json.loads(RECORD.read_text())


def test_record_records_a_resonance_band_of_the_pair(record):
    bands = record["criteria"]["resonance_bands"]["value"]
    assert bands, "the fine scan must record at least one narrow-band resonance"
    heave = [b for b in bands if b["label"] == "catamaran" and b["mode"] == "heave"]
    assert heave, "the pair resonance is the heave/pitch one"
    band = heave[0]
    assert 1.0 < band["omega_over_omega_n"] < 1.4
    assert band["b_peak_over_median"] >= RESONANCE_PEAK_RATIO
    assert band["added_mass_sign_change"] is True


def test_record_keeps_the_original_verdicts_verbatim(record):
    criteria = record["criteria"]
    original = criteria["gated_original"]
    # the four honest FAILs of the first pass are not overwritten by the new work
    assert "infinite_frequency_added_mass" in original
    assert "distance_decay_far_field" in original
    assert criteria["all_pass_original"] in (True, False)
    # and the fine-scan resonance is what spoils the pair extrapolation
    assert criteria["resonance_bands"]["pass"] is True


def test_record_far_field_is_gated_by_kd(record):
    entry = record["criteria"]["distance_decay_far_field_kd_gated"]
    points = entry["kd_points"]
    assert points
    assert all(point["kd"] > 0.0 for point in points)
    gated = [p for p in points if p["kd"] >= KD_FAR_FIELD_MIN]
    assert entry["points"] == len(gated)
    # the un-gated criterion (|sigma-1| for every d/L > 2) is still recorded
    assert record["criteria"]["distance_decay_far_field"]["limit"] == 0.05


def test_record_separates_the_omega_limit_from_the_mesh_error(record):
    """The 3 % heave residual at 0.1 omega_n is the O(omega^2) correction.

    Refining the matched pair of meshes must *not* collapse it (if it were a mesh
    difference it would), while walking the frequency towards zero at fixed mesh
    must (the record carries both statements).
    """
    low = record["low_frequency_limit"]
    ladder = [p for p in low["pairs"] if abs(p["omega_over_omega_n"] - 0.1) < 1e-12]
    assert len(ladder) >= 2, "the matched mesh ladder at 0.1 omega_n must be recorded"
    coarse, fine = min(ladder, key=lambda p: p["nx"]), max(ladder, key=lambda p: p["nx"])
    assert fine["nx"] > coarse["nx"]
    assert max(fine["residual"]) > 0.5 * max(coarse["residual"])

    trend = sorted(low["trend"], key=lambda t: t["omega_over_omega_n"])
    assert len(trend) >= 2
    # smallest frequency first: the residual must shrink as omega -> 0
    assert trend[0]["omega_over_omega_n"] < trend[-1]["omega_over_omega_n"]
    assert trend[0]["worst_residual"] < trend[-1]["worst_residual"]