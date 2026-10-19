import json
import subprocess
import sys

import numpy as np
import pytest

from tensorbem.benchmarks.analytical_suite import (
    CASES,
    exact_field,
    interior_points,
    make_mesh,
    plot_convergence,
    run_case,
    run_study,
)


@pytest.mark.parametrize("case", CASES)
def test_analytical_gradients_and_laplacian(case):
    points = np.array([[0.23, 0.31], [-0.34, 0.21], [-0.31, -0.27]])
    _, gradient = exact_field(case, points)
    step = 1.0e-5
    laplacian = np.zeros(len(points))
    for axis in range(2):
        offset = np.zeros_like(points)
        offset[:, axis] = step
        plus, grad_plus = exact_field(case, points + offset)
        minus, grad_minus = exact_field(case, points - offset)
        assert np.allclose((plus - minus) / (2 * step), gradient[:, axis], atol=1.0e-8)
        laplacian += (grad_plus[:, axis] - grad_minus[:, axis]) / (2 * step)
    assert np.max(np.abs(laplacian)) < 1.0e-7


@pytest.mark.parametrize("case", CASES)
def test_mesh_geometry_and_fixed_interior_probes(case):
    for count in (8, 16, 32):
        mesh = make_mesh(case, count)
        assert mesh.panel_count == count
        assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1)
        points = interior_points(case)
        # A nonzero winding angle verifies all probes are inside the actual mesh.
        start = mesh.vertices[None, :, :] - points[:, None, :]
        end = mesh.ends[None, :, :] - points[:, None, :]
        cross = start[:, :, 0] * end[:, :, 1] - start[:, :, 1] * end[:, :, 0]
        angles = np.arctan2(cross, np.sum(start * end, axis=2))
        assert np.allclose(np.sum(angles, axis=1), 2 * np.pi)
    assert np.array_equal(points, interior_points(case))


@pytest.mark.parametrize("case,flux_limit,field_limit", [
    ("disk_cubic", 0.025, 0.001),
    ("ellipse_quadratic", 0.02, 0.002),
    ("square_separable", 0.07, 0.0011),
    ("l_shape_singular", 0.12, 0.0045),
])
def test_analytical_cases_converge(case, flux_limit, field_limit):
    coarse = run_case(case, 16)
    fine = run_case(case, 32)
    for metric in ("normal_derivative_relative_l2_error", "field_rmse"):
        assert 0 < fine[metric] < coarse[metric]
    assert fine["relative_residual_l2"] < 1.0e-12
    assert fine["normal_derivative_relative_l2_error"] < flux_limit
    assert fine["field_rmse"] < field_limit
    assert fine["normalized_flux_imbalance"] < 0.05
    error = np.array(fine["probe_numerical"]) - np.array(fine["probe_exact"])
    assert fine["field_rmse"] == pytest.approx(np.sqrt(np.mean(error**2)))


def test_square_boundary_data_and_corner_singularity():
    x = np.linspace(0.1, 0.9, 5)
    bottom = np.column_stack((x, np.zeros_like(x)))
    top = np.column_stack((x, np.ones_like(x)))
    assert np.allclose(exact_field("square_separable", bottom)[0], 0)
    assert np.allclose(exact_field("square_separable", top)[0], np.sin(np.pi * x))
    with pytest.raises(ValueError, match="undefined"):
        exact_field("l_shape_singular", np.array([[0.0, 0.0]]))
    points = np.array([[0.1, 0.1], [0.2, 0.2]])
    value, gradient = exact_field("l_shape_singular", points)
    assert value[1] / value[0] == pytest.approx(2 ** (2 / 3))
    assert np.linalg.norm(gradient[1]) / np.linalg.norm(gradient[0]) == pytest.approx(
        2 ** (-1 / 3)
    )


@pytest.mark.parametrize("counts", [[], [16], [16, 16], [8, 12], [0, 16], [16, 24.0], [True, 16]])
def test_study_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        run_study(counts)


def test_unknown_cases_are_rejected():
    for function in (make_mesh, interior_points, exact_field):
        with pytest.raises(ValueError, match="unknown"):
            if function is make_mesh:
                function("unknown", 16)
            elif function is exact_field:
                function("unknown", np.ones((2, 2)))
            else:
                function("unknown")


def test_study_orders_and_strict_json():
    study = run_study([16, 8])
    assert study["panel_counts"] == [8, 16]
    assert json.loads(json.dumps(study, allow_nan=False)) == study
    for data in study["cases"].values():
        coarse, fine = data["results"]
        for metric in ("normal_derivative_relative_l2_error", "field_rmse"):
            assert coarse[metric + "_order"] is None
            expected = np.log(coarse[metric] / fine[metric]) / np.log(
                coarse["h_max"] / fine["h_max"]
            )
            assert fine[metric + "_order"] == pytest.approx(expected)
        assert all(data["checks"].values())


def test_cli_writes_reproducible_json(tmp_path):
    output = tmp_path / "nested" / "study.json"
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.analytical_suite",
         "--panels", "8", "16", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout) == json.loads(output.read_text())


def test_optional_convergence_plot(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    output = tmp_path / "figures" / "convergence.svg"
    plot_convergence(run_study([8, 16]), output)
    assert "<svg" in output.read_text()
