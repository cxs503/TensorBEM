"""Exact surface measure for flat-panel Kelvin BEM on a curved hull.

The flat-panel discretisation of a smooth body integrates over the chords of the
polyhedron, so the discrete wetted-surface measure is short of the true one
(``1 - A_poly/(4 pi a^2) = 1.9 %`` for the 320-panel inscribed icosphere used by
anchor ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK``).  That deficit propagates straight
into the radiation coefficients -- ``err(A33') ~ 1.0 x deficit``,
``err(B33') ~ 2.9 x deficit`` -- and is what pushes the 320-panel economy mesh
over the 5 % pool gate while the 1280-panel mesh passes at 1.4 %.

``tensorbem.geometry3d.spherical_patch_areas`` returns the exact spherical-patch
area of every panel; passing it as ``panel_areas`` to
``assemble_free_surface_operators_3d`` / ``solve_radiation_3d`` replaces the
integration measure without touching the panels, their normals, the collocation
points or any quadrature rule.  These tests pin

* the measure itself (sum rule, per-panel ordering, refinement behaviour);
* the argument validation of the new keyword;
* the default path being bit-identical to the explicit flat measure;
* the measured effect on the anchor (one 80-panel and one 320-panel solve).

The full study, with controls and five predeclared tiers, is
``benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import spherical_patch_areas, submerged_sphere_mesh

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665


def _patch(subdivisions: int, submergence: float = SUBMERGENCE):
    mesh = submerged_sphere_mesh(subdivisions, radius=RADIUS, submergence=submergence)
    center = np.array([0.0, 0.0, -submergence])
    areas = spherical_patch_areas(
        mesh.vertices, mesh.faces, center=center, radius=RADIUS
    )
    return mesh, center, areas


@pytest.mark.parametrize("subdivisions", [1, 2, 3])
def test_patch_areas_sum_to_the_sphere_area(subdivisions):
    mesh, _, areas = _patch(subdivisions)
    assert areas.shape == (mesh.panel_count,)
    assert np.all(np.isfinite(areas))
    exact = 4.0 * math.pi * RADIUS ** 2
    assert abs(areas.sum() / exact - 1.0) < 1.0e-12


@pytest.mark.parametrize("subdivisions", [1, 2, 3])
def test_every_patch_area_exceeds_its_flat_chord(subdivisions):
    mesh, _, areas = _patch(subdivisions)
    ratio = areas / mesh.areas
    assert np.all(ratio > 1.0)
    assert ratio.max() < 1.10  # a geodesic sphere is nearly uniform
    assert (mesh.areas.sum() / areas.sum() - 1.0) < 0.0


def test_area_deficit_shrinks_under_refinement():
    deficits = []
    for subdivisions in (1, 2, 3):
        mesh, _, areas = _patch(subdivisions)
        deficits.append(1.0 - mesh.surface_area / areas.sum())
    assert deficits[0] > deficits[1] > deficits[2] > 0.0
    # second-order deficit decay: each refinement cuts it by ~4
    assert deficits[0] / deficits[1] > 3.0
    assert deficits[1] / deficits[2] > 3.0


def test_validation_rejects_bad_patch_inputs():
    mesh = submerged_sphere_mesh(1, radius=RADIUS, submergence=SUBMERGENCE)
    center = (0.0, 0.0, -SUBMERGENCE)
    with pytest.raises(ValueError):
        spherical_patch_areas(mesh.vertices, mesh.faces, center=center, radius=0.0)
    with pytest.raises(ValueError):
        spherical_patch_areas(mesh.vertices, mesh.faces, center=(0.0, 0.0), radius=RADIUS)
    shifted = mesh.vertices.copy()
    shifted[:, 2] += 1.0e-3
    with pytest.raises(ValueError):
        spherical_patch_areas(shifted, mesh.faces, center=center, radius=RADIUS)


def test_solver_rejects_a_bad_panel_measure():
    mesh = submerged_sphere_mesh(1, radius=RADIUS, submergence=SUBMERGENCE)
    with pytest.raises(ValueError):
        solve_radiation_3d(mesh, 0.5, panel_areas=np.ones(mesh.panel_count + 1))
    bad = np.ones(mesh.panel_count)
    bad[3] = 0.0
    with pytest.raises(ValueError):
        solve_radiation_3d(mesh, 0.5, panel_areas=bad)
    with pytest.raises(ValueError):
        assemble_free_surface_operators_3d(mesh, 0.5, panel_areas=-np.ones(mesh.panel_count))


def test_default_measure_is_bit_identical_to_the_flat_measure():
    mesh = submerged_sphere_mesh(1, radius=RADIUS, submergence=SUBMERGENCE)
    default = assemble_free_surface_operators_3d(mesh, 0.5)
    explicit = assemble_free_surface_operators_3d(mesh, 0.5, panel_areas=mesh.areas)
    assert np.array_equal(default.single_layer, explicit.single_layer)
    assert np.array_equal(default.double_layer, explicit.double_layer)
    assert np.array_equal(default.system_matrix, explicit.system_matrix)


def _errors(subdivisions: int, ka: float) -> dict[str, dict[str, float]]:
    mesh, center, patch = _patch(subdivisions)
    nu = ka / RADIUS
    ref = submerged_sphere_wave_multipole(
        nu, radius=RADIUS, submergence=SUBMERGENCE, density=DENSITY, gravity=GRAVITY
    )
    vol = ref.displaced_volume
    omega = ref.omega
    out: dict[str, dict[str, float]] = {}
    for name, measure in (("flat", None), ("patch", patch)):
        rad = solve_radiation_3d(
            mesh, nu, center=center.tolist(), density=DENSITY, gravity=GRAVITY,
            panel_areas=measure,
        )
        got = {
            "a33": float(rad.added_mass[2, 2] / (DENSITY * vol)),
            "b33": float(rad.damping[2, 2] / (DENSITY * omega * vol)),
            "a11": float(rad.added_mass[0, 0] / (DENSITY * vol)),
            "b11": float(rad.damping[0, 0] / (DENSITY * omega * vol)),
        }
        refv = {
            "a33": ref.heave_added_mass_dimless,
            "b33": ref.heave_damping_dimless,
            "a11": ref.surge_added_mass_dimless,
            "b11": ref.surge_damping_dimless,
        }
        out[name] = {c: abs(got[c] - refv[c]) / abs(refv[c]) for c in refv}
    return out


def test_exact_measure_removes_the_coarse_mesh_bias_at_80_panels():
    """Damping (the gated quantity) gains ~3.7x; added mass gains little.

    At 80 panels the residual O(h^2) facet-normal error dominates A33', so the
    measure correction is required to *not regress* it, while B33' -- the
    coefficient that fails the pool gate -- must improve by more than 2x.
    """
    err = _errors(1, 1.0)
    assert err["flat"]["b33"] > 0.15  # documented: 20 % band-max on B33'
    assert err["patch"]["b33"] < 0.08
    assert err["flat"]["b33"] / err["patch"]["b33"] > 2.0
    for coefficient in ("a33", "a11", "b11"):
        assert err["patch"][coefficient] <= 1.05 * err["flat"][coefficient]


def test_exact_measure_brings_the_320_panel_economy_mesh_inside_the_gate():
    err = _errors(2, 1.0)
    assert err["flat"]["b33"] > 0.05  # the documented economy-tier failure
    assert err["patch"]["b33"] < 0.02
    assert err["patch"]["b11"] < 0.02
    assert err["patch"]["a33"] < 0.02
    assert err["patch"]["a11"] < 0.02