"""Numerical identities and towing-tank record integrity for forward-speed heave/pitch."""

import csv
from pathlib import Path
import runpy

import numpy as np
import pytest

from tensorbem.forward_speed import solve_forward_speed_heave_pitch
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
ROOT = BENCH / "journee_wigley3_forward_speed"
assessment = runpy.run_path(str(ROOT / "assess.py"))


def test_zero_current_reduces_to_independent_zero_speed_assembly():
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    old = runpy.run_path(str(BENCH / "journee_wigley3_zero_speed/run.py"))
    expected = old["tensorbem_coefficients"](mesh, 4.0)
    result = solve_forward_speed_heave_pitch(
        mesh, 4.0, 0.0, density=old["RHO"], gravity=old["G"]
    )
    for got, ref in zip(
        (result.added_mass, result.damping, result.excitation), expected
    ):
        np.testing.assert_allclose(got, ref, atol=2e-10)
    assert result.encounter_omega == result.wave_omega


def test_pitch_origin_shift_transforms_both_coupling_terms_and_excitation():
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    dx = 0.23
    transform = np.array([[1.0, dx], [0.0, 1.0]])
    base = solve_forward_speed_heave_pitch(mesh, 4.0, 0.9)
    shifted = solve_forward_speed_heave_pitch(mesh, 4.0, 0.9, center=(dx, 0, 0))
    for field in ("added_mass", "damping"):
        np.testing.assert_allclose(
            getattr(shifted, field),
            transform.T @ getattr(base, field) @ transform,
            atol=1e-10,
        )
    np.testing.assert_allclose(
        shifted.excitation, transform.T @ base.excitation, atol=1e-10
    )
    assert base.encounter_omega == pytest.approx(4.0 + 16.0 * 0.9 / 9.80665)
    assert base.relative_residual < 1e-10
    assert abs(base.damping[1, 0] - base.damping[0, 1]) > 1.0


@pytest.mark.parametrize(
    "omega,speed", [(0.0, 1.0), (np.nan, 0.0), (3.0, -0.2), (3.0, np.inf)]
)
def test_invalid_frequency_or_speed(omega, speed):
    with pytest.raises(ValueError):
        solve_forward_speed_heave_pitch(
            wigley3_wetted_hull_mesh(nx=6, nz=4), omega, speed
        )


def test_finite_speed_archived_coefficients_reproduce_live_solve():
    data = assessment["table"](ROOT, "tensorbem")
    row = data[
        (data["panels"] == 264) & (data["Fn"] == 0.3) & (data["lambda_L"] == 1.25)
    ][0]
    mesh = wigley3_wetted_hull_mesh(nx=12, nz=6)
    result = solve_forward_speed_heave_pitch(
        mesh, row["omega"], row["U"], density=1000.0, gravity=9.81
    )
    for prefix, matrix in (("A", result.added_mass), ("B", result.damping)):
        np.testing.assert_allclose(
            matrix, [[row[prefix + i + j] for j in "35"] for i in "35"], rtol=1e-8
        )
    np.testing.assert_allclose(
        result.excitation,
        [complex(row["F" + d + "_re"], row["F" + d + "_im"]) for d in "35"],
        rtol=1e-8,
    )


def test_trial_and_score_integrity():
    table = assessment["table"]
    for name, count in (("motions", 64), ("loads", 42), ("radiation", 104)):
        assert len(table(ROOT, name)) == count
    scores, errors = assessment["compare"]()
    assert len(errors) == 2 * 3 * (64 + 42) * 2
    with (ROOT / "scores.csv").open() as f:
        archived = list(csv.DictReader(f))
    assert len(scores) == len(archived) == 72
    for r, saved in zip(scores, archived):
        assert r[-1] == saved["verdict"]
        assert r[6] == pytest.approx(float(saved["rms"]), rel=1e-8)
        assert r[8] == pytest.approx(float(saved["phase_rms_deg"]), rel=1e-8)
    columns = assessment["coefficient_errors"]()
    assert len(columns) == 2 * 34 * 8
    rows, excluded = assessment["load_substitutions"]()
    assert len(rows) == 19 * 6 * 2
    assert {(r[0], r[1]) for r in excluded} == {(0.3, 0.5), (0.4, 0.5)}
    native = [r for r in rows if r[3] == "native"]
    assert len(native) == 38
    for fn, ratio, _, _, name, _, _, value, _ in native:
        data = table(ROOT, "tensorbem")
        pred = data[
            (data["panels"] == 760) & (data["Fn"] == fn) & (data["lambda_L"] == ratio)
        ]
        assert value == pytest.approx(pred[name][0], rel=1e-7)


def test_measured_radiation_order_scaling_and_bounds():
    table = assessment["table"](ROOT, "radiation")
    rows = table[table["Fn"] == 0.2]
    w_nd = 2.8
    factor = np.sqrt(9.81 / 3)
    a, b = assessment["measured_matrices"](rows, w_nd * factor)
    for col, mode in enumerate((3, 5)):
        for row in (0, 1):
            term = "self" if row == col else "cross"
            for var, matrix, unit in (("A", a, 1), ("B", b, factor)):
                values = []
                for group in (1, 2):
                    series = rows[
                        (rows["driven"] == mode) & (rows["source_group"] == group)
                    ]
                    values.append(
                        np.interp(w_nd, series["w_nd"], series[var + "_" + term])
                    )
                expected = np.mean(values) * 78 * 3 ** (row + col) * unit
                assert matrix[row, col] == pytest.approx(expected)
    assert assessment["measured_matrices"](rows, 30.0) is None


def test_failed_sweep_preserves_last_complete_output(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH.parent))
    from benchmarks.journee_wigley3_forward_speed import run

    output = tmp_path / "predictions.csv"
    output.write_text("previous complete sweep")

    def fail_after_header(solver, cases, stream):
        stream.write("partial")
        raise ArithmeticError("singular system")

    monkeypatch.setattr(run, "_write_sweep", fail_after_header)
    with pytest.raises(ArithmeticError):
        run.calculate("tensorbem", output)
    assert output.read_text() == "previous complete sweep"
    assert list(tmp_path.iterdir()) == [output]
