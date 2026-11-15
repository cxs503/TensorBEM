"""Tests for the KK-gate error-injection calibration (``C6b`` certification).

The claims under test are the ones the certificate makes about the *detector*:
the exact pair really is exact, the reported residual really does grow with the
injected corruption, the two normalisations differ by exactly the documented
factor, and the case-level audit reproduces the per-DOF dilution from the
committed artifact.  A test that only re-ran the code and compared floats to
themselves would not be worth the bytes; each test below names the invariant it
binds.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorbem.benchmarks import kk_gate_calibration as kc

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"


def _grid() -> np.ndarray:
    return kc.calibration_grid()


def test_pair_is_kk_exact_so_any_residual_is_the_estimators() -> None:
    """The calibration pair must satisfy the KK relations *identically*.

    If it did not, a residual measured on it would mix estimator error with pair
    error and the whole certificate would be meaningless.  The floor is the
    estimator's own bias and must sit far below the 10 % criterion.
    """
    grid = _grid()
    pair = kc.CALIBRATION_PAIR
    result = kc.closure_residual(grid, pair.added_mass(grid), pair.damping(grid))
    assert result["max_relative_residual"] < 0.02
    assert result["amplitude_normalised_residual"] < 0.03
    # the pair is a *causal* radiation pair: positive damping, hump inside the band
    damping = pair.damping(grid)
    assert np.all(damping > 0.0)
    hump = pair.damping_peak_omega()
    assert kc.CALIBRATION_BAND[0] < hump < kc.CALIBRATION_BAND[1]
    # B -> 0 as omega -> 0 (otherwise the KK integral of A diverges there)
    tiny = float(pair.damping(np.array([1e-6]))[0])
    assert tiny < 1e-6 * float(np.max(damping))


def test_closed_form_damping_matches_the_kernel_definition() -> None:
    """``B = 240 w^2/((w^2+4)(w^2+64))``: checked against a numeric integral.

    The closed form is quoted in the module docstring; recomputing it from the
    kernel definition (``int_0^inf K(s) cos(ws) ds``) is what makes the claim a
    fact rather than a comment.
    """
    pair = kc.CALIBRATION_PAIR
    for omega in (0.3, 1.0, 4.0, 11.0):
        s = np.linspace(0.0, 200.0, 400001)
        kernel = sum(
            c * np.exp(-a * s) for c, a in zip(pair.coefficients, pair.rates)
        )
        numeric = np.trapezoid(kernel * np.cos(omega * s), s)
        closed = float(pair.damping(np.array([omega]))[0])
        assert abs(numeric - closed) <= 2e-3 * max(abs(closed), 1e-12), omega


def test_reported_residual_grows_with_the_injected_corruption() -> None:
    """Detection power: residual(δ) must increase once it clears the floor.

    Below ~5 % the injected error is smaller than the estimator bias, which is why
    the gate's monotonicity claim starts there instead of at δ = 0.01.
    """
    curve = kc.calibration_curve(
        "damping_step", (0.05, 0.1, 0.2, 0.3, 0.6, 0.9)
    )
    points = [p for p in curve["points"] if p["delta"] >= 0.05]
    reported = [p["reported_residual"] for p in points]
    assert all(b > a for a, b in zip(reported, reported[1:]))
    amplitude = [p["amplitude_normalised_residual"] for p in points]
    assert all(b > a for a, b in zip(amplitude, amplitude[1:]))
    # the corruption C6b injects is visible on the invariant statistic
    at_injection = next(
        p for p in curve["points"] if abs(p["delta"] - kc.C6B_INJECTED_STEP) < 1e-12
    )
    assert at_injection["amplitude_normalised_residual"] > kc.KK_BAR


def test_amplitude_normalisation_is_exactly_the_dilution_factor() -> None:
    """The two statistics must differ by ``max|A| / max|A - A_inf|``, no more.

    This is the mechanism behind the audit finding: the declared bar is a
    fraction of ``max|A|``, which contains the arbitrary ``A_inf``, so the same
    bar is a *different* fraction of the added-mass amplitude per DOF.
    """
    curve = kc.calibration_curve("damping_step", (0.1, 0.3, 0.9))
    for point in curve["points"]:
        dilution = point["dilution_amplitude_over_scale"]
        ratio = point["amplitude_normalised_residual"] / point["reported_residual"]
        assert ratio == pytest.approx(1.0 / dilution, rel=1e-9)
        assert 0.0 < dilution < 1.0


def test_localised_damping_corruption_is_harder_to_detect_than_a_band_step() -> None:
    """The documented blind spot: a *local* shape error is attenuated by the KK integral.

    ``C6b`` breaks the damping above 1 rad/s, i.e. over most of the band.  A
    corruption confined to one bump of the same relative size is reported more
    weakly and needs a larger ``delta`` to reach the bar, because the KK transform
    integrates over frequency.  The numerical size of that attenuation is
    published in the certificate's ``localisation`` table; a +100 % error confined
    to ``d(log omega) = 0.15`` is measured to *pass* the 10 % bar, which is the
    honest limit on what gate C3 can be read as saying.
    """
    grid = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0)
    step = kc.calibration_curve("damping_step", grid)
    local = kc.calibration_curve("damping_bump", grid)
    step_point = next(p for p in step["points"] if abs(p["delta"] - 0.3) < 1e-12)
    local_point = next(p for p in local["points"] if abs(p["delta"] - 0.3) < 1e-12)
    assert local_point["reported_residual"] < step_point["reported_residual"]
    step_threshold = kc.detection_threshold(step)
    local_threshold = kc.detection_threshold(local)
    # the local channel needs a larger injected size to reach the same bar
    assert local_threshold["bracket"][1] > step_point["delta"]
    assert step_threshold["delta"] < local_threshold["delta"]


def test_localisation_sweep_finds_a_passing_large_corruption() -> None:
    """Bind the strongest negative claim: a big narrow error passes the gate.

    If this ever stops being true the blind-spot paragraph in the docs must be
    re-read (it is a claim about the estimator, not a preference).
    """
    sweep = kc.localisation_sweep()
    worst = sweep["worst_passing_corruption"]
    assert worst is not None
    assert worst["passes_bar"] is True
    assert worst["delta"] >= 1.0  # a >=100 % damping error
    assert worst["log_width"] <= 0.25  # confined to a narrow window
    # the same corruption is *not* invisible on the invariant statistic: the two
    # normalisations disagree by ~max|A|/amplitude, as documented
    assert worst["amplitude_normalised_residual"] > worst["reported_residual"]
    # and a wide corruption of the same size is caught
    wide = [
        row
        for row in sweep["rows"]
        if row["log_width"] == max(r["log_width"] for r in sweep["rows"])
        and abs(row["delta"] - 2.0) < 1e-12
    ]
    assert wide and wide[0]["passes_bar"] is False


def test_zero_mean_injection_is_reported_more_strongly_than_one_sided() -> None:
    """The closure's free ``A_inf`` absorbs the mean of the error it is judged on.

    A one-signed bump carries a mean, so part of the injected error lands in the
    offset instead of the residual; a zero-mean bump is reported with the larger
    slope.  That is the quantification of "C3 spends one free parameter".
    """
    one_sided = kc.calibration_curve("added_mass_bump", (0.1, 0.4))
    zero_mean = kc.calibration_curve("added_mass_bump_zero_mean", (0.1, 0.4))
    slope_one = kc.recovery_slope(one_sided)["slope"]
    slope_zero = kc.recovery_slope(zero_mean)["slope"]
    assert slope_zero > slope_one > 0.0


def test_detection_threshold_reports_infinity_when_the_bar_is_never_crossed() -> None:
    """No extrapolation: an unreachable bar reads as ``inf``, not as a guess."""
    curve = kc.calibration_curve("damping_step", (0.01, 0.05), bar=0.95)
    threshold = kc.detection_threshold(curve)
    assert math.isinf(threshold["delta"])
    assert threshold["bracket"] is None


def test_gate_claims_hold_and_are_recorded() -> None:
    payload = kc.certificate()
    gate = kc.calibration_gate(payload)
    assert gate["id"] == "C6b_cal"
    assert gate["passed"] is True, gate["values"]["claims"]
    claims = gate["values"]["claims"]
    assert claims == {"floor_ok": True, "responsive_ok": True, "detection_ok": True}
    # the certificate must predict a *below-bar* reported residual at the step
    # C6b injects while the invariant statistic clears the bar: that pair of
    # facts is the whole point (the declared statistic dilutes the error).
    assert gate["values"]["predicted_residual_at_c6b_step"] < kc.KK_BAR
    assert (
        gate["values"]["predicted_residual_at_c6b_step_amplitude_normalised"]
        > kc.KK_BAR
    )


def test_estimator_runs_in_the_case_configuration() -> None:
    """The pair's tail must be rejected by the power-tail rule, as on the case sweep.

    Otherwise the calibration would be measuring an estimator that has a tail
    model the case does not have, and the numbers would not transfer.
    """
    curve = kc.calibration_curve("damping_step", (0.3,))
    assert curve["damping_tail_valid"] is False
    assert curve["damping_tail_exponent"] < 2.05
    assert curve["band"] == [kc.CALIBRATION_BAND[0], kc.CALIBRATION_BAND[1]]
    assert curve["n_omega"] == kc.CALIBRATION_N_OMEGA


@pytest.mark.skipif(
    not (CASE_DIR / "case_result.json").exists(),
    reason="committed case artifact not available",
)
def test_scale_audit_reproduces_the_committed_dilution() -> None:
    """Read-only audit of the committed verdict: the bar is not scale-invariant.

    The published artifact is *not* modified.  What is asserted is that the
    documented observation follows from the committed numbers: every DOF's
    residual is diluted by ``max|A - A_inf| / max|A|`` (0.33 .. 0.84 here), and on
    the invariant statistic the anchored closure C3a exceeds the same 10 % number
    on at least one DOF even though the gate passes as declared.  If a future
    rerun changes that, this test fails and the finding must be re-read, not
    silently kept.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "calibrate_kk_gate", ROOT / "scripts" / "calibrate_kk_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    case_result = json.loads((CASE_DIR / "case_result.json").read_text())
    audit = module.scale_audit(case_result)

    for gate_id in ("C3", "C3a", "C6b"):
        rows = audit["gates"][gate_id]["rows"]
        assert len(rows) == 6
        for row in rows:
            assert 0.2 < row["dilution_amplitude_over_scale"] < 0.95
            assert row["residual_amplitude_normalised"] == pytest.approx(
                row["residual_reported"] / row["dilution_amplitude_over_scale"],
                rel=1e-9,
            )
    # C3a: the anchored closure's own worst DOF passes as declared ...
    assert audit["gates"]["C3a"]["passed_as_declared"] is True
    assert audit["gates"]["C3a"]["worst_residual_amplitude_normalised"] > kc.KK_BAR
    assert set(audit["gates"]["C3a"]["dofs_above_bar_in_amplitude_units"]) >= {
        "yaw",
        "sway",
    }


def test_main_writes_certificate_markdown_and_gate(tmp_path) -> None:
    code = kc.main(["--out", str(tmp_path)])
    assert code == 0
    payload = json.loads((tmp_path / "kk_gate_calibration.json").read_text())
    assert payload["gate"]["id"] == "C6b_cal"
    assert payload["certificate"]["channels"]["damping_step"]["points"]
    markdown = (tmp_path / "kk_gate_calibration.md").read_text()
    assert "KK gate calibration" in markdown
    assert "detection" in markdown.lower()