"""Tests for the translating-ellipsoid added-mass benchmark."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tensorbem.benchmarks.ellipsoid_added_mass import run
from tensorbem.geometry3d import ellipsoid_mesh, icosphere, project_to_ellipsoid

AXES = (1.5, 1.0, 0.75)


def test_added_mass_converges_along_all_three_axes():
    coarse = run(0, AXES, 1.0, 1.0)
    medium = run(1, AXES, 1.0, 1.0)
    for name in ("x", "y", "z"):
        rows = [
            next(item for item in report["axes"] if item["motion_axis"] == name)
            for report in (coarse, medium)
        ]
        assert rows[1]["added_mass_relative_error"] < rows[0]["added_mass_relative_error"]
        assert rows[1]["added_mass_relative_error"] < 0.1
        assert rows[1]["relative_residual_l2"] < 1.0e-12


def test_sphere_limit_returns_half_the_displaced_mass():
    density = 1025.0
    report = run(1, (1.0, 1.0, 1.0), 1.0, density)
    volume = 4.0 / 3.0 * math.pi
    for row in report["axes"]:
        assert math.isclose(row["exact_added_mass"], 0.5 * density * volume, rel_tol=1.0e-10)
        assert row["added_mass_relative_error"] < 0.1
    assert np.allclose(report["shape_factors"], 2.0 / 3.0, atol=1.0e-10)


def test_anisotropy_is_reported_and_ordered_by_shape_factor():
    report = run(1, AXES, 1.0, 1.0)
    factors = report["shape_factors"]
    assert math.isclose(report["shape_factor_sum"], 2.0, abs_tol=1.0e-9)
    # The short axis (z) has the largest shape factor and the largest added mass.
    assert factors[0] < factors[1] < factors[2]
    mass = [row["exact_added_mass"] for row in report["axes"]]
    assert mass[0] < mass[1] < mass[2]


def test_mesh_geometry_and_volume_ratio_are_reported():
    report = run(2, AXES, 1.0, 1.0)
    assert report["panels"] == 320
    assert report["vertices"] == 162
    assert 0.9 < report["volume_ratio"] < 1.0
    assert report["surface_area"] > 0.0
    # Refinement must increase the enclosed volume towards the ellipsoid.
    coarse = run(0, AXES, 1.0, 1.0)
    medium = run(1, AXES, 1.0, 1.0)
    assert coarse["volume_ratio"] < medium["volume_ratio"] < report["volume_ratio"]


def test_prolate_spheroid_reproduces_the_slender_body_trend():
    # Motion along the long axis must have much less added mass than across it.
    axes = (2.0, 0.8, 0.8)
    report = run(1, axes, 1.0, 1.0)
    axial = report["axes"][0]["exact_added_mass"]
    transverse = report["axes"][1]["exact_added_mass"]
    assert axial < 0.25 * transverse


def test_benchmark_is_a_strict_json_document():
    report = run(0, AXES, 1.0, 1025.0)
    rendered = json.dumps(report, sort_keys=True, allow_nan=False)
    assert json.loads(rendered) == report
    assert len(report["axes"]) == 3


def test_benchmark_rejects_invalid_input():
    with pytest.raises(ValueError, match="semi_axes"):
        run(0, (1.5, 1.0))
    with pytest.raises(ValueError, match="positive"):
        run(0, (1.5, 0.0, 0.75))
    with pytest.raises(ValueError, match="speed"):
        run(0, AXES, speed=-1.0)
    with pytest.raises(ValueError, match="density"):
        run(0, AXES, density=0.0)


def test_ellipsoid_mesh_inherits_the_icosphere_topology():
    mesh = ellipsoid_mesh(1, AXES)
    sphere = icosphere(1)
    assert mesh.panel_count == sphere.panel_count
    assert mesh.vertex_count == sphere.vertex_count
    assert np.array_equal(mesh.faces, sphere.faces)
    on_surface = (
        (mesh.vertices[:, 0] / AXES[0]) ** 2
        + (mesh.vertices[:, 1] / AXES[1]) ** 2
        + (mesh.vertices[:, 2] / AXES[2]) ** 2
    )
    assert np.allclose(on_surface, 1.0, atol=1.0e-14)
    assert mesh.volume > 0.0


def test_project_to_ellipsoid_keeps_the_ray_direction():
    points = np.array([[1.0, 0.0, 0.0], [0.3, -0.2, 0.5], [-0.1, 0.4, -0.9]])
    projected = project_to_ellipsoid(points, AXES)
    radii = np.linalg.norm(projected, axis=1)
    assert np.allclose(projected / radii[:, None], points / np.linalg.norm(points, axis=1)[:, None])
    on_surface = (
        (projected[:, 0] / AXES[0]) ** 2
        + (projected[:, 1] / AXES[1]) ** 2
        + (projected[:, 2] / AXES[2]) ** 2
    )
    assert np.allclose(on_surface, 1.0, atol=1.0e-14)
    with pytest.raises(ValueError, match="origin"):
        project_to_ellipsoid(np.zeros((1, 3)), AXES)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        project_to_ellipsoid(points, (1.0, 1.0))


def test_command_line_writes_a_strict_json_report(tmp_path):
    output = tmp_path / "ellipsoid.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.ellipsoid_added_mass",
            "--subdivisions",
            "0",
            "--semi-axes",
            "1.5",
            "1.0",
            "0.75",
            "--density",
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
    assert report["benchmark"] == "three_dimensional_translating_ellipsoid"
    assert report["panels"] == 20
    assert len(report["axes"]) == 3
