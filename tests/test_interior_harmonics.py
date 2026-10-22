"""Tests for the interior solid-harmonic benchmark and the classical study."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem.benchmarks.classical_study import run_study
from tensorbem.benchmarks.interior_harmonics import run, run_case


def test_degree_one_reproduces_the_linear_field_case():
    # The x harmonic is the case the existing 3-D study already reports; the
    # numbers must agree, which guards against a sign or axis mix-up.
    row = run_case("x", 2)
    assert math.isclose(row["normal_derivative_relative_l2_error"], 1.2325e-2, rel_tol=2.0e-2)
    assert math.isclose(row["field_rmse"], 9.2055e-4, rel_tol=2.0e-2)
    assert row["degree"] == 1


def test_higher_degree_harmonics_converge_under_refinement():
    for name in ("x2-y2", "x3-3xy2"):
        coarse = run_case(name, 0)
        fine = run_case(name, 1)
        assert fine["normal_derivative_relative_l2_error"] < coarse[
            "normal_derivative_relative_l2_error"
        ]
        assert fine["field_rmse"] < coarse["field_rmse"]


def test_flux_error_grows_with_the_harmonic_degree():
    # Same mesh, stronger angular variation: the constant-panel error must grow.
    errors = [
        run_case(name, 1)["normal_derivative_relative_l2_error"]
        for name in ("x", "x2-y2", "x3-3xy2")
    ]
    assert errors[0] < errors[1] < errors[2]


def test_study_reports_orders_and_checks_for_every_case():
    study = run([0, 1], (1, 2))
    assert study["study"] == "tensorbem_interior_solid_harmonics"
    assert set(study["cases"]) == {"x", "y", "z", "x2-y2", "2xy", "2z2-x2-y2"}
    for name, case in study["cases"].items():
        assert case["checks"]["normal_derivative_relative_l2_error_decreases"], name
        assert case["checks"]["field_rmse_decreases"], name
        order = case["results"][-1]["normal_derivative_relative_l2_error_order"]
        assert order is None or math.isfinite(order)


def test_study_rejects_invalid_levels_and_degrees():
    with pytest.raises(ValueError, match="unique"):
        run([1, 1], (1,))
    with pytest.raises(ValueError, match="non-negative"):
        run([-1, 1], (1,))
    with pytest.raises(ValueError, match="non-negative"):
        run([1, 2.5], (1,))
    with pytest.raises(ValueError, match="radius"):
        run([1, 2], (1,), radius=0.0)
    with pytest.raises(ValueError):
        run([1, 2], (4,))


def test_classical_study_passes_all_checks():
    study = run_study(
        panel_counts=(16, 32),
        subdivisions=(0, 1),
        harmonic_levels=(1,),
        semi_major=1.5,
        semi_minor=0.75,
        semi_axes=(1.5, 1.0, 0.75),
    )
    assert study["study"] == "tensorbem_classical_benchmarks"
    assert all(study["checks"].values()), study["checks"]
    assert study["all_outputs_finite"] is True
    assert set(study["ellipse"]["axes"]) == {"x", "y"}
    assert set(study["ellipsoid"]["axes"]) == {"x", "y", "z"}
    # Observed orders must be reported, not silently dropped.
    for axis in ("x", "y"):
        orders = study["ellipse"]["observed_orders"][axis]["added_mass"]
        assert len(orders) == 1 and orders[0] > 1.5
    for axis in ("x", "y", "z"):
        orders = study["ellipsoid"]["observed_orders"][axis]
        assert len(orders) == 1 and orders[0] > 1.5


def test_classical_study_rejects_invalid_settings():
    with pytest.raises(ValueError, match="two panel counts"):
        run_study(panel_counts=(16,))
    with pytest.raises(ValueError, match="unique"):
        run_study(panel_counts=(16, 16))
    with pytest.raises(ValueError, match="two subdivision levels"):
        run_study(panel_counts=(16, 32), subdivisions=(0,))


def test_classical_study_is_a_strict_json_document(tmp_path):
    output = tmp_path / "classical.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.classical_study",
            "--panels",
            "16",
            "32",
            "--subdivisions",
            "0",
            "1",
            "--harmonic-levels",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert all(report["checks"].values())
    assert report["environment"]["numpy"] == np.__version__
