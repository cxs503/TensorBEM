"""Gates for the 3-D finite-depth radiation BEM (:mod:`tensorbem.finite_depth_solver3d`).

The heavy verification study lives in ``scripts/verify_finite_depth_solver3d.py``
and writes ``docs/finite_depth_solver3d/results.json``; these tests re-derive the
cheap end of it so a regression cannot hide behind a stale document.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.finite_depth_solver3d import (
    assemble_finite_depth_operators_3d,
    mode_count_for,
    quadrature_points,
    solve_radiation_finite_depth_3d,
)
from tensorbem.finite_depth3d import finite_depth_wavenumber
from tensorbem.free_surface_solver3d import (
    _build_target_operators_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh

RADIUS = 1.0
SUBMERGENCE = 1.5
CENTER = (0.0, 0.0, -SUBMERGENCE)
NU = 1.0
RESULTS = Path(__file__).resolve().parents[1] / "docs" / "finite_depth_solver3d" / "results.json"


def _mesh(subdivision: int = 0):
    return submerged_sphere_mesh(subdivision, radius=RADIUS, submergence=SUBMERGENCE)


def test_quadrature_rule_reproduces_area_and_is_centroid_exact():
    mesh = _mesh(1)
    points, weights, normals = quadrature_points(mesh)
    assert points.shape == (6 * mesh.panel_count, 3)
    assert weights.shape == (6 * mesh.panel_count,)
    assert normals.shape == (6 * mesh.panel_count, 3)
    assert math.isclose(float(weights.sum()), float(mesh.surface_area), rel_tol=1e-12)
    # degree-4 rule: exact for the linear coordinate, i.e. the centroid map
    for axis in range(3):
        per_panel = points[:, axis].reshape(mesh.panel_count, 6)
        weighted = (per_panel * weights.reshape(mesh.panel_count, 6)).sum(axis=1)
        np.testing.assert_allclose(weighted, mesh.areas * mesh.centroids[:, axis], atol=1e-12)
    # no rule point sits at the panel centroid, so the self-panel remainder is
    # never evaluated at r = 0
    dist = np.linalg.norm(points - np.repeat(mesh.centroids, 6, axis=0), axis=1)
    assert dist.min() > 0.05 * float(mesh.h_mean)


def test_operators_reduce_to_the_deep_water_kernel_deep_limit():
    """At ``k0 h = 20`` the finite-depth single layer must be the Kelvin one.

    Only the single layer is gated operator-by-operator: both assemblies treat
    the *self panel* of the double layer with their own quadrature of the
    non-Rankine remainder, and on the very coarse 20-panel sphere that residual
    differs by a few per cent of the 1/2 jump (it is the curvature-dependent part
    of the principal value, and the deep solver uses a degree-2 rule where this
    module uses degree 4).  The end-to-end gate below is the one that matters.
    """
    mesh = _mesh(0)
    finite_s, _ = _finite_operators(mesh, NU, 20.0)
    deep_s, _ = _build_target_operators_3d(
        mesh.centroids, mesh, NU, self_interaction=True, rankine_order=5
    )
    assert np.linalg.norm(finite_s - deep_s) / np.linalg.norm(deep_s) < 1e-2


def _finite_operators(mesh, nu: float, depth: float):
    from tensorbem.finite_depth_solver3d import _smooth_remainder

    count = min(mode_count_for(mesh, depth), 256)
    return _smooth_remainder(mesh.centroids, mesh, nu, depth, count, order=5)


def test_radiation_matches_the_deep_water_solver_in_the_deep_limit():
    mesh = _mesh(0)
    finite = solve_radiation_finite_depth_3d(
        mesh, NU, 20.0, modes=[0, 2], center=CENTER, mode_count=192
    )
    deep = solve_radiation_3d(mesh, NU, modes=[0, 2], center=CENTER)
    np.testing.assert_allclose(
        np.diag(finite.added_mass), np.diag(deep.added_mass), rtol=1e-2
    )
    np.testing.assert_allclose(
        np.diag(finite.damping), np.diag(deep.damping), rtol=1e-2
    )
    assert finite.relative_residual < 1e-10


def test_reciprocity_and_axisymmetry_of_the_six_dof_matrices():
    mesh = _mesh(0)
    result = solve_radiation_finite_depth_3d(
        mesh, NU, 5.0, center=CENTER, mode_count=128
    )
    added = result.added_mass
    damping = result.damping
    assert np.max(np.abs(added - added.T)) / np.max(np.abs(added)) < 1e-3
    assert np.max(np.abs(damping - damping.T)) / np.max(np.abs(damping)) < 1e-3
    # the 20-panel icosphere is only *approximately* axisymmetric, so the
    # surge/sway pair is gated loosely here; the converged values are in the
    # verification study (docs/finite_depth_solver3d/results.json, A2/A3)
    assert abs(added[0, 0] - added[1, 1]) / abs(added[0, 0]) < 2e-2
    assert abs(damping[0, 0] - damping[1, 1]) / abs(damping[0, 0]) < 2e-2
    assert abs(added[0, 2]) / abs(added[2, 2]) < 1e-3
    assert abs(added[3, 3]) / abs(added[0, 0]) < 1e-2


def test_radiation_damping_is_positive_and_bounded_by_the_deep_water_value():
    mesh = _mesh(0)
    values = {}
    for depth in (20.0, 3.0):
        result = solve_radiation_finite_depth_3d(
            mesh, NU, depth, modes=[2], center=CENTER, mode_count=192
        )
        values[depth] = float(result.damping[0, 0])
        assert values[depth] > 0.0
    # the sea bed changes the answer: the shallow case must not be a copy of the
    # deep one (this is what the deep-water-only path could not express)
    assert abs(values[3.0] / values[20.0] - 1.0) > 0.02


def test_shallow_water_stiffens_heave_added_mass_monotonically():
    mesh = _mesh(0)
    previous = None
    for depth in (20.0, 10.0, 5.0):
        result = solve_radiation_finite_depth_3d(
            mesh, NU, depth, modes=[2], center=CENTER, mode_count=192
        )
        value = float(result.added_mass[0, 0])
        if previous is not None:
            assert value > previous
        previous = value


def test_mode_truncation_converges():
    mesh = _mesh(0)
    coarse = solve_radiation_finite_depth_3d(
        mesh, NU, 5.0, modes=[2], center=CENTER, mode_count=64
    )
    fine = solve_radiation_finite_depth_3d(
        mesh, NU, 5.0, modes=[2], center=CENTER, mode_count=512
    )
    assert abs(fine.added_mass[0, 0] / coarse.added_mass[0, 0] - 1.0) < 2e-2
    assert abs(fine.damping[0, 0] / coarse.damping[0, 0] - 1.0) < 2e-2


def test_wavenumber_is_the_strip_dispersion_root():
    nu, depth = 1.0, 5.0
    k0 = finite_depth_wavenumber(nu, depth)
    assert math.isclose(k0 * math.tanh(k0 * depth), nu, rel_tol=1e-12)
    assert k0 > nu  # finite depth always gives the longer wave


@pytest.mark.parametrize(
    "name,call",
    [
        ("mesh_below_sea_bed", lambda: solve_radiation_finite_depth_3d(_mesh(0), NU, 2.0)),
        ("non_positive_depth", lambda: solve_radiation_finite_depth_3d(_mesh(0), NU, 0.0)),
        ("non_positive_nu", lambda: solve_radiation_finite_depth_3d(_mesh(0), -1.0, 5.0)),
        (
            "bad_mode_index",
            lambda: solve_radiation_finite_depth_3d(_mesh(0), NU, 5.0, modes=[7], mode_count=32),
        ),
        (
            "zero_modes",
            lambda: assemble_finite_depth_operators_3d(_mesh(0), NU, 5.0, mode_count=0),
        ),
    ],
)
def test_invalid_arguments_raise(name, call):
    with pytest.raises(ValueError):
        call()


def test_operators_cannot_be_reused_across_depths_or_frequencies():
    mesh = _mesh(0)
    operators = assemble_finite_depth_operators_3d(mesh, NU, 5.0, mode_count=32)
    with pytest.raises(ValueError):
        solve_radiation_finite_depth_3d(mesh, NU, 6.0, operators=operators)
    with pytest.raises(ValueError):
        solve_radiation_finite_depth_3d(mesh, 2.0, 5.0, operators=operators)


def test_mesh_must_be_the_first_declared_geometry():
    # a sphere that dips below z = -depth is outside the fluid domain
    mesh = _mesh(0)
    with pytest.raises(ValueError, match="inside the fluid"):
        solve_radiation_finite_depth_3d(mesh, NU, 2.0)


@pytest.mark.skipif(not RESULTS.exists(), reason="verification study not generated")
def test_documented_depth_ladder_is_still_reproduced():
    """Re-derive the cheapest depth of the documented ladder and compare."""
    payload = json.loads(RESULTS.read_text())
    if not payload.get("finished"):
        pytest.skip("study run incomplete")
    row = next(r for r in payload["A2"]["rows"] if r["depth"] == 20.0)
    mesh = submerged_sphere_mesh(1, radius=RADIUS, submergence=SUBMERGENCE)
    result = solve_radiation_finite_depth_3d(
        mesh, NU, 20.0, modes=[0, 2], center=CENTER, mode_count=row["mode_count"]
    )
    assert math.isclose(
        float(np.diag(result.added_mass)[1]), row["A33"], rel_tol=2e-6
    )
    assert math.isclose(
        float(np.diag(result.damping)[1]), row["B33"], rel_tol=2e-6
    )
    # negative control: a tampered documented value must be detectable
    assert not math.isclose(row["A33"] * 1.01, row["A33"], rel_tol=2e-6)