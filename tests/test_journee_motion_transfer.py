"""Check wave-load missingness, original trial identity, complex response and transfer coverage."""

import csv
from pathlib import Path

import numpy as np
import pytest

from tensorbem.rao import dynamic_stiffness

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/journee_wigley3_forward_speed"


def test_complex_response_and_restoring_convention(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    from benchmarks.journee_wigley3_forward_speed import impedance_audit as audit

    m = np.diag([78, 43.8988875])
    a = np.array([[14.0, 0.3], [-0.4, 13.0]])
    b = np.array([[125.0, 34.0], [-40.0, 60.0]])
    copied_c = np.diag([6093.65, 2819.92])
    w = 7.0
    z = dynamic_stiffness(w, m, a, b, audit.REPORT_C)
    np.testing.assert_allclose(
        z, dynamic_stiffness(w, m, a + (copied_c - audit.REPORT_C) / w**2, b, copied_c)
    )
    f = np.array([150 + 200j, 300 - 80j])
    solution = np.linalg.solve(z, f)
    for i in (0, 1):
        eff, imp, left = audit.schur_response(z, f, i)
        assert eff / imp == pytest.approx(solution[i])
        assert 0 <= left <= 1
    curves = audit.paired_parity(ROOT)
    pair, old = curves["report_paired"], curves["previous"]
    outside = pair[(pair["Fn"] == 0.4) & (pair["lambda_L"] == 0.5)][0]
    assert outside["C55"] == pytest.approx(old["C55"][0])
    worst = pair[(pair["Fn"] == 0.4) & (pair["lambda_L"] == 1.384)][0]
    assert worst["C55"] == audit.REPORT_C[1, 1]
    assert worst["pitch"] == pytest.approx(1.517775407, abs=1e-8)


def test_partial_missing_force_phase_and_single_level_radiation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    from benchmarks.journee_wigley3_forward_speed import cross_hull_motions as study

    loads = study.read_hulls(ROOT / "transfer_loads.csv")
    i = loads[loads["hull"] == "I"]
    r = i[(i["Fn"] == 0.4) & (i["lambda_L"] == 0.75)]
    assert len(r) == 2 and np.isnan(r["force3_deg"]).all()
    assert np.isfinite(r["force3"][1]) and np.isfinite(r["moment5_deg"]).all()
    f = study.endpoint_components(
        i, 0.3, 0.75, 2 * np.pi / 2.25, study.HULLS["I"].restoring
    )
    assert np.isnan(f[0]) and np.isfinite(f[1])
    rad = study.read_hulls(ROOT / "additional_forced_hulls.csv")
    iv = rad[rad["hull"] == "IV"]
    assert study.endpoint_matrices(iv, 0.3, 7.0) is None
    ab = study.endpoint_matrices(iv, 0.3, 7.0, mass=156, upper_levels=(1,))
    assert np.isfinite(ab).all()
    corrupted = iv.copy()
    corrupted["A_self"][iv["Fn"] == 0.3] = np.nan
    np.testing.assert_array_equal(
        ab, study.endpoint_matrices(corrupted, 0.3, 7.0, mass=156, upper_levels=(1,))
    )
    for hull, volume in (("I", 0.0946), ("IV", 0.1560)):
        assert study.hull_mesh(hull).volume == pytest.approx(volume, rel=0.01)


def test_original_trials_and_missing_components_survive_transfer(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    from benchmarks.journee_wigley3_forward_speed import cross_hull_motions as study
    from benchmarks.journee_wigley3_forward_speed import impedance_audit as audit

    for n in ("transfer_loads", "transfer_motions", "additional_forced_hulls"):
        (tmp_path / (n + ".csv")).symlink_to(ROOT / (n + ".csv"))
    scores = study.assess(tmp_path)
    assert [(s[0], s[2], s[3], s[-1]) for s in scores] == [
        ("I", "heave", 11, "FAIL"),
        ("I", "pitch", 11, "FAIL"),
        ("IV", "heave", 17, "FAIL"),
        ("IV", "pitch", 17, "PASS"),
    ]
    for file in ("transfer_scores.csv", "transfer_coefficients.csv"):
        with (tmp_path / file).open() as new, (ROOT / file).open() as saved:
            for actual, expected in zip(
                csv.DictReader(new), csv.DictReader(saved), strict=True
            ):
                for key in actual:
                    if key in {"native_residual", "native_condition"} and actual[key]:
                        bound = 1e-10 if key == "native_residual" else 100
                        assert 0 < float(actual[key]) < bound
                    elif (
                        key
                        in {
                            "hull",
                            "quantity",
                            "verdict",
                            "radiation",
                            "F3_source",
                            "F5_source",
                        }
                        or not expected[key]
                    ):
                        assert actual[key] == expected[key]
                    else:
                        assert float(actual[key]) == pytest.approx(
                            float(expected[key]), rel=1e-8, abs=1e-8
                        )
    with (tmp_path / "transfer_trials.csv").open() as stream:
        errors = list(csv.DictReader(stream))
    assert (
        len(
            {
                (r["hull"], r["quantity"], r["source_line"], r["source_group"])
                for r in errors
            }
        )
        == 56
    )
    assert {(r["radiation"], r["F3_source"], r["F5_source"]) for r in errors} == {
        ("endpoint", "endpoint", "endpoint"),
        ("endpoint", "native", "endpoint"),
        ("native", "endpoint", "endpoint"),
    }
    rows, coverage = audit.phase_substitutions(ROOT)
    assert len(rows) == 58 * 2 * 4 and len(coverage) == 37
    assert (
        sum(bool(rad) and load != "unavailable" for _, _, rad, load in coverage) == 33
    )
    for r in rows:
        assert (r[11] - (r[15] - r[17]) + 180) % 360 - 180 == pytest.approx(
            0.0, abs=1e-10
        )
