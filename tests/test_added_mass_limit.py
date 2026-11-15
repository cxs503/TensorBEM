"""Tests for the two spectral added-mass limits (:mod:`tensorbem.added_mass_limit`).

The anchor is an *exact* case: for a half-immersed sphere the antisymmetric
image of the hull closes into a full sphere, and in heave the ``omega ->
infinity`` (Dirichlet calm-water) problem of the half body is exactly the rigid
translation of that full sphere.  The infinite-fluid added mass of a sphere in
heave is ``m_full / 2``, so the half body must give ``m_full / 4`` with no
approximation beyond the panel discretisation -- which the mesh ladder here
measures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.added_mass_limit import (
    added_mass_limit,
    doubled_body_mesh,
    generalized_normals_6dof,
)
from tensorbem.geometry3d import WettedHullMesh
from tensorbem.wigley3_hull import wigley3_wetted_hull_mesh
from tensorbem.benchmarks.cummins_wigley_kk_causality import geometry_gate

DENSITY = 1025.0
CENTER = (0.0, 0.0, -0.09375)


def hemisphere_mesh(nlat: int = 6, nlon: int = 18, radius: float = 1.0) -> WettedHullMesh:
    """Lower hemisphere with its waterline on the equator (analytic test body)."""
    verts: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for i in range(nlat):
        theta = 0.5 * math.pi * i / nlat
        z = -radius * math.sin(theta)
        r = radius * math.cos(theta)
        row = []
        for j in range(nlon):
            phi = 2.0 * math.pi * j / nlon
            row.append(len(verts))
            verts.append((r * math.cos(phi), r * math.sin(phi), z))
        rings.append(row)
    pole = len(verts)
    verts.append((0.0, 0.0, -radius))
    faces: list[tuple[int, int, int]] = []
    for i in range(nlat - 1):
        for j in range(nlon):
            j2 = (j + 1) % nlon
            a, b = rings[i][j], rings[i][j2]
            c, d = rings[i + 1][j2], rings[i + 1][j]
            faces.append((a, b, c))
            faces.append((a, c, d))
    for j in range(nlon):
        j2 = (j + 1) % nlon
        faces.append((rings[-1][j], rings[-1][j2], pole))
    return WettedHullMesh(np.asarray(verts, float),
                          np.asarray(faces, np.int64)[:, ::-1])


def test_doubled_mesh_is_closed_and_has_twice_the_volume() -> None:
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    doubled, lower = doubled_body_mesh(mesh)
    assert doubled.panel_count == 2 * mesh.panel_count
    assert lower.size == mesh.panel_count
    assert doubled.volume == pytest.approx(2.0 * mesh.volume, rel=1e-12)
    # waterline vertices are welded, not duplicated
    assert doubled.vertices.shape[0] < 2 * mesh.vertices.shape[0]


def test_generalized_normals_match_the_hull_mesh() -> None:
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    normals = generalized_normals_6dof(mesh, center=CENTER)
    assert normals.shape == (mesh.panel_count, 6)
    np.testing.assert_allclose(normals[:, :3], mesh.normals, atol=1e-12)
    rel = mesh.centroids - np.asarray(CENTER)[None, :]
    np.testing.assert_allclose(normals[:, 3:], np.cross(rel, mesh.normals), atol=1e-12)


@pytest.mark.parametrize("limit,parity", [("infinite_frequency", -1.0),
                                          ("zero_frequency", 1.0)])
def test_limit_flags_and_matrix_properties(limit: str, parity: float) -> None:
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    result = added_mass_limit(mesh, limit, center=CENTER, density=DENSITY)
    assert result.parity == parity
    assert result.limit == limit
    assert result.max_relative_residual_l2 < 1e-10
    assert result.double_layer_row_sum_error < 1e-10
    assert result.symmetry_error == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(result.added_mass_matrix, result.added_mass_matrix.T,
                               atol=1e-8)
    diag = result.diagonal()
    assert np.all(diag > 0.0), diag
    # heave added mass exceeds heave mass fraction plausibility and both limits differ
    assert diag[2] > 0.0


def test_infinite_frequency_rejects_bad_input() -> None:
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    with pytest.raises(ValueError):
        added_mass_limit(mesh, "high_frequency")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        added_mass_limit(mesh, "infinite_frequency", density=-1.0)
    with pytest.raises(ValueError):
        added_mass_limit(mesh, "infinite_frequency", modes=[9])
    with pytest.raises(TypeError):
        added_mass_limit(mesh.vertices, "infinite_frequency")  # type: ignore[arg-type]


def test_half_immersed_sphere_matches_the_exact_image_value() -> None:
    """``A_inf(heave) = m_full / 4`` exactly for a half-immersed sphere.

    Two rungs of the ladder: the finer one must be closer and the error must be
    small compared with the (large) mesh coarseness, which is the end-to-end
    check of the normals, the jump sign, the half-body integral and the parity
    rule of the image.
    """
    exact = DENSITY * 4.0 / 3.0 * math.pi / 4.0
    errors = []
    for nlat, nlon in ((4, 12), (8, 24)):
        result = added_mass_limit(hemisphere_mesh(nlat, nlon), "infinite_frequency",
                                  density=DENSITY)
        errors.append(abs(result.diagonal()[2] - exact) / exact)
    assert errors[0] > errors[1]
    assert errors[1] < 0.03, errors
    # a sphere cannot radiate a moment about its own centre
    roll_pitch_yaw = result.diagonal()[3:]
    assert np.max(np.abs(roll_pitch_yaw)) < 1e-2 * result.diagonal()[2]


def test_wigley_both_limits_are_ordered_heave() -> None:
    """Rigid-wall (``omega -> 0``) heave added mass exceeds the Dirichlet limit."""
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    inf = added_mass_limit(mesh, "infinite_frequency", center=CENTER, density=DENSITY)
    zero = added_mass_limit(mesh, "zero_frequency", center=CENTER, density=DENSITY)
    assert zero.diagonal()[2] > inf.diagonal()[2] > 0.0


def test_wigley_geometry_gate_precedes_any_number() -> None:
    gate = geometry_gate()
    assert gate["passed"] is True
    assert abs(gate["Cb_from_volume"] - gate["Cb_declared"]) < 1e-12


def test_infinite_frequency_differs_from_the_free_float_added_mass() -> None:
    """``A_inf`` must be well below the low-frequency sweep value (physics floor).

    The stored coarse sweep starts at ``omega = 0.2`` rad/s, where the heave
    added mass is still an order-1 fraction above its high-frequency limit; a
    number at or above that value would mean the parity/limit convention is
    inverted.
    """
    payload = np.load("docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz")
    low = float(np.asarray(payload["added_mass"], dtype=float)[0, 2, 2])
    mesh = wigley3_wetted_hull_mesh(nx=8, nz=4)
    inf = added_mass_limit(mesh, "infinite_frequency", center=CENTER, density=DENSITY)
    assert inf.diagonal()[2] < low