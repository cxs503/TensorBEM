"""Analytic limits, hydrodynamic conventions and held-speed data isolation."""

import csv
from pathlib import Path

import numpy as np
import pytest

from tensorbem.section_radiation import (
    double_ended_strip_radiation,
    solve_section_heave,
)
from tensorbem.stf_fullspeed import SectionCoefficients, stf_speed_correction_terms


def semicircle(n):
    theta = np.linspace(-np.pi, 0, n + 1)
    vertices = np.column_stack((np.cos(theta), np.sin(theta)))
    vertices[[0, -1], 1] = 0
    return vertices


def test_semicircle_analytic_limits_and_wave_energy():
    contour = semicircle(32)
    low = solve_section_heave(contour, np.sqrt(0.001), density=1.0, gravity=1.0)
    # In the long-wave limit heave radiates the waterline-width flux on each side.
    assert low.damping / np.sqrt(0.001) == pytest.approx(4.0, rel=0.02)
    # The Dirichlet free-surface limit is the submerged half of the circle dipole.
    high = solve_section_heave(contour, np.sqrt(20.0), density=1.0, gravity=1.0)
    assert high.added_mass == pytest.approx(np.pi / 2, rel=0.025)
    coarse, fine = [
        solve_section_heave(semicircle(n), 1.0, density=1.0, gravity=1.0)
        for n in (16, 32)
    ]
    assert abs(fine.damping - fine.energy_damping) < abs(
        coarse.damping - coarse.energy_damping
    )
    assert fine.damping == pytest.approx(fine.energy_damping, rel=0.015)
    assert fine.relative_residual < 1e-12


def test_section_scaling_and_waterline_boundary():
    contour = semicircle(16)
    a = solve_section_heave(contour, 1.0, density=1.0, gravity=1.0)
    b = solve_section_heave(3 * contour, 1 / np.sqrt(3), density=2.0, gravity=1.0)
    assert b.added_mass == pytest.approx(18 * a.added_mass)
    assert b.damping == pytest.approx(18 / np.sqrt(3) * a.damping)
    with pytest.raises(ValueError):
        solve_section_heave(contour - [0, 0.1], 1.0)


def test_strip_cross_terms_use_correct_coefficient_sign_frequency_and_pitch_origin():
    x = np.array([-1.0, 0.0, 1.0])
    w = np.array([0.5, 1.0, 0.5])
    a = np.array([0.0, 4.0, 0.0])
    b = np.array([0.0, 9.0, 0.0])
    # U=2, encounter frequency=3, total A33=4, B33=9
    A, B = double_ended_strip_radiation(x, w, a, b, 3.0, 2.0)
    np.testing.assert_allclose(A, [[4.0, -2.0], [2.0, 16 / 9]])
    np.testing.assert_allclose(B, [[9.0, 8.0], [-8.0, 4.0]])
    # Conventional section-coefficients API must use the same corrections.
    sections = [
        SectionCoefficients(s, aa, 0, 0, 0, bb, 0, 0, 0, 3)
        for s, aa, bb in zip(x, a, b)
    ]
    da, db, ea, eb = stf_speed_correction_terms(sections, x, 1.0, 2.0, gravity=1.0)
    np.testing.assert_allclose(da + ea, A - np.diag([4.0, 0.0]))
    np.testing.assert_allclose(db + eb, B - np.diag([9.0, 0.0]))
    A0, B0 = double_ended_strip_radiation(x, w, a, b, 3.0, 0.0)
    np.testing.assert_allclose(A0, np.diag([4.0, 0.0]))
    np.testing.assert_allclose(B0, np.diag([9.0, 0.0]))
    t = np.array([[1.0, 0.4], [0.0, 1.0]])
    ap, bp = double_ended_strip_radiation(x - 0.4, w, a, b, 3.0, 2.0)
    np.testing.assert_allclose(ap, t.T @ A @ t)
    np.testing.assert_allclose(bp, t.T @ B @ t)


def test_endpoint_predictions_are_independent_of_held_speed_inputs(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    from benchmarks.journee_wigley3_forward_speed import radiation_study as study

    for file in ("tensorbem.csv", "strip_coefficients.csv"):
        (tmp_path / file).symlink_to(study.ROOT / file)
    for file in ("loads.csv", "radiation.csv"):
        with (study.ROOT / file).open() as f, (tmp_path / file).open("w") as out:
            data = csv.DictReader(f)
            writer = csv.DictWriter(out, fieldnames=data.fieldnames)
            writer.writeheader()
            writer.writerows(r for r in data if float(r["Fn"]) != 0.3)
    original, _, _, _ = study.workflows()
    withheld, sources, _, _ = study.workflows(tmp_path)
    for name in ("endpoint_radiation", "endpoint_radiation_load"):
        for field in withheld[name].dtype.names:
            np.testing.assert_array_equal(withheld[name][field], original[name][field])
    assert sources["endpoint_radiation_load", 0.3, 0.5] == (
        "native_outside_band",
        "endpoint_tank",
    )
    assert sources["endpoint_radiation_load", 0.4, 2.367] == (
        "endpoint_tank",
        "native_outside_band",
    )
    scores = study.compare(predictions_by_solver=withheld)[0]
    chosen = {
        r[3]: r for r in scores if r[0] == "endpoint_radiation_load" and r[2] == 0.3
    }
    assert chosen["pitch"][-1] == "PASS"
    assert chosen["heave"][-1] == "FAIL"
    assert chosen["heave"][7] == pytest.approx(0.210566978, abs=1e-7)


def test_real_section_prediction_reproduces_archived_matrix(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    from benchmarks.journee_wigley3_forward_speed import strip

    with (strip.ROOT / "strip_coefficients.csv").open() as f:
        row = next(
            r
            for r in csv.DictReader(f)
            if r["stations"] == "8" and r["kind"] == "motion" and r["Fn"] == "0.3"
        )
    live = strip.solve_frequency(
        ((8, 12), float(row["omega_e"]), 0.3 * np.sqrt(strip.GRAVITY * strip.LENGTH))
    )
    np.testing.assert_allclose(
        live[:8], [float(row[k]) for k in strip.COEFFICIENTS], rtol=1e-8, atol=1e-9
    )


def test_forced_baseline_covers_every_trial_without_frequency_interpolation(
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    from benchmarks.journee_wigley3_forward_speed import radiation_study as study

    _, _, strip, training = study.workflows()
    rows = study.forced_comparisons(study.ROOT, strip, training)
    assert len(rows) == 416
    assert len({(r[0], r[1]) for r in rows}) == 67
    assert len({(r[0], r[2], r[3], r[4]) for r in rows}) == 104
    assert all(np.isfinite(r[6:12]).all() for r in rows)
    for r in rows:
        assert int(r[5][-1]) == r[2]
