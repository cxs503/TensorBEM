"""Zero-speed passive-body drift, fixed-body compatibility and tank coverage."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tensorbem.drift import far_field_mean_drift_3d
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "journee_added_resistance"


def test_superposed_radiation_uses_complex_velocity():
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    k = 2.0
    kw = dict(operators=assemble_free_surface_operators_3d(mesh, k))
    rad = solve_radiation_3d(mesh, k, modes=[2, 4], **kw)
    diff = solve_diffraction_3d(mesh, k, wave_angle=np.pi, **kw)
    opts = dict(n_azimuth=36)
    fixed = far_field_mean_drift_3d(mesh, diff, **opts)
    zero = far_field_mean_drift_3d(mesh, diff, radiation=rad, motion=[0, 0], **opts)
    np.testing.assert_allclose(zero["force"], fixed["force"])
    xi = np.array([0.3 + 0.11j, -0.21j])
    # Compare with separate per-mode Kochin transforms and complex recombination.
    from tensorbem.free_surface_solver3d import kochin_function_3d

    expected = fixed["kochin"].copy()
    for j in range(2):
        expected += (-1j * diff.omega * xi[j]) * kochin_function_3d(
            mesh, rad.potentials[:, j], rad.normal_velocities[:, j], k, fixed["azimuth"]
        )
    result = far_field_mean_drift_3d(mesh, diff, radiation=rad, motion=xi, **opts)
    np.testing.assert_allclose(result["kochin"], expected, atol=1e-12)
    for radiation, motion in (
        (None, xi),
        (rad, None),
        (rad, [1]),
        (rad, [np.nan, 0]),
        (replace(rad, wavenumber=k * 2), xi),
    ):
        with pytest.raises(ValueError):
            far_field_mean_drift_3d(
                mesh, diff, radiation=radiation, motion=motion, **opts
            )


def test_drift_grids_and_independent_solver(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH.parents[1]))
    from benchmarks.journee_added_resistance.zero_speed_drift import calculate

    def read(solver):
        with (BENCH / ("drift_" + solver + ".csv")).open() as stream:
            return {
                (r["hull"], float(r["lambda_L"]), int(r["panels"])): float(r["free"])
                for r in csv.DictReader(stream)
            }

    native, ext = read("tensorbem"), read("capytaine")
    assert native.keys() == ext.keys() and len(native) == 54
    for hull, scale in (("III", 3.73), ("IV", 5.73)):
        assert (
            max(
                abs(native[key] - ext[key])
                for key in native
                if key[0] == hull and key[2] == 760
            )
            < 0.05 * scale
        )
    assert calculate(("tensorbem", "III", 1.25, 12, 6))["free"] == pytest.approx(
        native["III", 1.25, 264], rel=1e-8
    )
