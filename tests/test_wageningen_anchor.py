"""Adjudication of the Wageningen B-series anchor used by the W25 study.

The anchor is the judge of every BEM number in
``benchmarks/w25/b4_70_error_surface``, so the judge itself is put on trial
here:

1. **Integrity** -- the artefact must still carry 39 ``KT`` terms and 47 ``KQ``
   terms and the stored coefficient-table sha256 must match a recomputation.
2. **Cross-implementation** -- the loader must reproduce, to 1e-12, the values
   the independent reference implementation produced at the probe points stored
   inside the artefact (this catches a botched transcription: the coefficients
   were moved by machine from that implementation, so a faithful copy has to
   evaluate identically).
3. **Physics anchors** -- properties any faithful transcription of Oosterveld &
   van Oossanen (1975) must have, *independent* of the source code:
   ``KT(J=0)`` grows monotonically with P/D and sits in the published band for
   the B4-70 member, the zero-thrust advance ratio satisfies
   ``J0/(P/D) = 1.04..1.10`` (geometric-pitch neighbourhood), peak efficiency
   is 0.5..0.8 and monotone in P/D, and ``KQ`` decreases monotonically with J
   over the usable range.
"""

from __future__ import annotations

import math

import pytest

from tensorbem.wageningen import (
    ANCHOR_PATH,
    B4Series,
    anchor_payload,
    anchor_sha256,
    b4_70,
    table_fingerprints,
    verify_tables,
)


# ---------------------------------------------------------------------------
# 1. artefact integrity
# ---------------------------------------------------------------------------


def test_anchor_artefact_exists_and_is_hashed():
    assert ANCHOR_PATH.exists(), f"anchor artefact missing: {ANCHOR_PATH}"
    digest = anchor_sha256()
    assert len(digest) == 64
    payload = anchor_payload()
    assert payload["anchor_id"] == "AN-WAGENINGEN-BSERIES-1975"
    assert "Oosterveld" in payload["citation"]["authors"]
    assert "479" in payload["citation"]["report"]


def test_term_counts_follow_the_published_regression():
    payload = anchor_payload()
    assert len(payload["kt_terms"]) == 39
    assert len(payload["kq_terms"]) == 47
    report = verify_tables()
    assert report["n_kt_terms"] == 39 and report["n_kq_terms"] == 47
    assert report["fingerprint_match"], report["mismatched_keys"]


def test_stored_fingerprints_are_recomputable():
    payload = anchor_payload()
    stored = payload["fingerprints"]
    actual = table_fingerprints(payload)
    for key in ("kt_table_sha256", "kq_table_sha256"):
        assert stored[key] == actual[key]


def test_exponents_are_physical():
    payload = anchor_payload()
    for table in (payload["kt_terms"], payload["kq_terms"]):
        for coefficient, s_j, t_pd, u_ae, v_z in table:
            assert abs(coefficient) < 1.0
            assert 0 <= s_j <= 3
            assert 0 <= t_pd <= 6
            assert 0 <= u_ae <= 3
            assert 0 <= v_z <= 2


# ---------------------------------------------------------------------------
# 2. cross-implementation reproduction (the anti-retyping check)
# ---------------------------------------------------------------------------


def test_loader_reproduces_source_implementation_probes():
    payload = anchor_payload()
    probes = payload["source_implementation_probes"]["rows"]
    assert len(probes) >= 10
    worst = 0.0
    for row in probes:
        series = B4Series(
            blades=row["Z"],
            expanded_area_ratio=row["Ae_Ao"],
            pitch_ratio=row["P_D"],
        )
        worst = max(
            worst,
            abs(series.kt(row["J"]) - row["KT"]),
            abs(series.kq(row["J"]) - row["KQ"]),
        )
    assert worst < 1e-12, f"cross-implementation mismatch {worst:.3e}"


# ---------------------------------------------------------------------------
# 3. physics anchors of the regression itself
# ---------------------------------------------------------------------------


def test_bollard_pull_thrust_grows_with_pitch():
    values = [b4_70(pd).kt(0.0) for pd in (0.6, 0.8, 1.0, 1.2, 1.4)]
    assert values == pytest.approx(sorted(values))
    # published band for the B4-70 member (O&O 1975): KT(J=0) ~ 0.25 -> 0.62
    assert 0.24 < values[0] < 0.27
    assert 0.60 < values[-1] < 0.64


def test_zero_thrust_advance_ratio_tracks_pitch():
    ratios = []
    for pd in (0.6, 0.8, 1.0, 1.2, 1.4):
        series = b4_70(pd)
        ratios.append(series.j0 / pd)
    assert all(1.04 < ratio < 1.10 for ratio in ratios), ratios


def test_peak_efficiency_band_and_monotonicity():
    peaks = []
    for pd in (0.6, 0.8, 1.0, 1.2, 1.4):
        series = b4_70(pd)
        grid = [0.02 * i for i in range(0, 61)]
        eta = [series.eta0(j) for j in grid if not math.isnan(series.eta0(j))]
        peaks.append(max(eta))
    assert all(0.50 < peak < 0.80 for peak in peaks), peaks
    assert peaks == pytest.approx(sorted(peaks))


def test_torque_decreases_with_advance_over_usable_range():
    series = b4_70(1.0)
    limits = series.scoring_range()
    assert 1.0 < limits["j_zero_thrust"] < 1.15
    grid = [0.02 * i for i in range(0, 51)]  # J <= 1.0, well inside KQ > 0
    kq_values = [series.kq(j) for j in grid]
    assert all(b < a for a, b in zip(kq_values, kq_values[1:]))


def test_b4_70_curve_in_artefact_matches_evaluator():
    payload = anchor_payload()
    rows = payload["b4_70_pd1_curve"]["rows"]
    series = b4_70(1.0)
    worst = max(abs(series.kt(row["J"]) - row["KT"]) for row in rows)
    assert worst < 1e-12
    assert payload["b4_70_pd1_curve"]["zero_thrust_advance_ratio"] == pytest.approx(
        series.j0, abs=1e-9
    )


def test_scoring_range_is_inside_published_domain():
    series = b4_70(1.0)
    limits = series.scoring_range()
    assert 0.5 <= 1.0 <= 1.4  # P/D inside the regression domain
    assert limits["usable_max_j"] == pytest.approx(limits["j_zero_thrust"])