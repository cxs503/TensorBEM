"""Recompute scores from the measured replicates and reproduce one native solve."""

import csv
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/journee_wigley3_zero_speed"


def test_all_measurements_and_grids_reproduce_archived_verdict():
    measured = np.genfromtxt(ROOT / "table10.csv", delimiter=",", names=True)
    assert len(measured) == 13 and len(np.unique(measured["lambda_L"])) == 11
    assert sorted(measured[measured["lambda_L"] == 1.25]["amplitude_cm"]) == [
        2.34,
        4.86,
        6.50,
    ]
    scores, errors = runpy.run_path(str(ROOT / "assess.py"))["score"](ROOT)
    archived = json.loads((ROOT / "scores.json").read_text())
    assert scores.keys() == archived.keys()
    for solver, grids in scores.items():
        assert grids.keys() == archived[solver].keys()
        for panels, dofs in grids.items():
            for dof, result in dofs.items():
                expected = archived[solver][panels][dof]
                assert result["metrics"] == pytest.approx(
                    expected["metrics"], abs=1e-12
                )
                assert {k: v for k, v in result.items() if k != "metrics"} == {
                    k: v for k, v in expected.items() if k != "metrics"
                }
    with (ROOT / "errors.csv").open() as file:
        recorded = list(csv.reader(file))[1:]
    assert len(errors) == len(recorded) == 2 * 3 * 2 * 13
    for stored, regenerated in zip(recorded, errors):
        assert stored[:6] == list(map(str, regenerated[:6]))
        np.testing.assert_allclose(np.asarray(stored[6:], dtype=float), regenerated[6:])
    for solver in ("tensorbem", "capytaine"):
        assert set(scores[solver]) == {"264", "480", "760"}
        assert all(scores[solver]["760"]["heave"]["passed"].values())
        assert not scores[solver]["760"]["pitch"]["passed"]["amplitude_max"]
        assert not scores[solver]["760"]["pitch"]["passed"]["amplitude_rms"]


def test_coarse_short_wave_radiation_and_excitation_from_live_solver():
    driver = runpy.run_path(str(ROOT / "run.py"))
    mesh = driver["wigley3_wetted_hull_mesh"](nx=12, nz=6)
    records = np.genfromtxt(ROOT / "tensorbem.csv", delimiter=",", names=True)
    (row,) = records[(records["panels"] == 264) & (records["lambda_L"] == 0.5)]
    a, b, f = driver["tensorbem_coefficients"](mesh, row["omega"])
    for prefix, matrix in (("A", a), ("B", b)):
        expected = [row[prefix + dofs] for dofs in ("33", "35", "53", "55")]
        np.testing.assert_allclose(matrix.ravel(), expected, rtol=1e-6, atol=1e-8)
    for i, mode in enumerate(("3", "5")):
        assert f[i] == pytest.approx(
            complex(row[f"F{mode}_re"], row[f"F{mode}_im"]), rel=1e-7
        )
