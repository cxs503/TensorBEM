"""Verification of the y = 0 symmetry reduction for the 3-D Kelvin BEM.

The acceptance criterion is algebraic rather than approximate: the folded
half-hull system is the *same* boundary integral equation as the full-hull system
with rows/columns restricted by parity, so half and full results must agree to
round-off (plus the single-layer quadrature noise of the shared rule, which is
measured separately in :func:`test_folded_operators_match_full_matrix`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.free_surface_solver3d import (
    _auto_chief_points_3d,
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh, icosphere
from tensorbem.symmetry3d import (
    MODE_PARITY,
    PanelPatch,
    assemble_symmetric_blocks,
    generalized_normals_3d,
    mirror_patch,
    operators_for_parity,
    solve_diffraction_3d_symmetric,
    solve_radiation_3d_symmetric,
    split_patch,
    verify_generalized_normal_parity,
)

NU = 1.0
PANELS = 2  # 64 panels on the full hemisphere, 32 on the retained half
PARITY = np.array([MODE_PARITY[k] for k in range(6)], dtype=float)
SAME_PARITY = np.equal.outer(PARITY, PARITY)


@pytest.fixture(scope="module")
def case():
    mesh = hemisphere_hull_mesh(PANELS)
    upper, lower = split_patch(mesh)
    return mesh, upper, lower


# ---------------------------------------------------------------- geometry
def test_split_halves_are_equal_and_seam_is_exact(case):
    mesh, upper, lower = case
    assert upper.panel_count + lower.panel_count == mesh.panel_count
    assert upper.panel_count == lower.panel_count
    assert upper.surface_area == pytest.approx(lower.surface_area, rel=1e-14)
    # the seam must be snapped exactly onto y = 0, otherwise the folded system is
    # only *nearly* the restricted full system
    for patch in (upper, lower):
        seam = np.abs(patch.vertices[:, 1]) <= 1e-12
        assert np.all(patch.vertices[seam, 1] == 0.0)
    assert np.all(upper.centroids[:, 1] > 0.0)
    assert np.all(lower.centroids[:, 1] < 0.0)


def test_mirror_patch_preserves_area_and_flips_normal(case):
    _, upper, _ = case
    mirrored = mirror_patch(upper)
    assert mirrored.panel_count == upper.panel_count
    np.testing.assert_allclose(mirrored.areas, upper.areas, rtol=1e-15, atol=0.0)
    expected = upper.normals * np.array([1.0, -1.0, 1.0])[None, :]
    np.testing.assert_allclose(mirrored.normals, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(mirrored.centroids[:, 1], -upper.centroids[:, 1], atol=1e-15)


def test_generalized_normal_parity_is_exact(case):
    _, upper, _ = case
    err = verify_generalized_normal_parity(upper)
    assert np.all(err < 1e-14), err


def test_straddling_panels_are_rejected():
    # an icosphere has triangles crossing y = 0, so it cannot be cut in half
    sphere = icosphere(1, radius=1.0)
    with pytest.raises(ValueError, match="straddle"):
        split_patch(sphere)


def test_patch_rejects_degenerate_and_offset_center(case):
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        PanelPatch(verts, np.array([[0, 0, 1]]))
    _, upper, _ = case
    with pytest.raises(ValueError, match="symmetry plane"):
        solve_radiation_3d_symmetric(upper, NU, center=(0.0, 0.1, 0.0))


# --------------------------------------------------------------- operator
def test_folded_operators_match_full_matrix(case):
    """The folded blocks must reproduce the restricted full-mesh operators."""
    mesh, upper, _ = case
    mirrored = mirror_patch(upper)

    def index_map(target, reference):
        dist = np.linalg.norm(target[:, None, :] - reference[None, :, :], axis=2)
        idx = np.argmin(dist, axis=1)
        assert dist[np.arange(target.shape[0]), idx].max() < 1e-12
        return idx

    idx_u = index_map(upper.centroids, mesh.centroids)
    idx_m = index_map(mirrored.centroids, mesh.centroids)
    full = assemble_free_surface_operators_3d(mesh, NU)
    blocks = assemble_symmetric_blocks(upper, NU)
    assert blocks.assembly_entries() == 2 * upper.panel_count**2
    assert blocks.assembly_entries() * 2 == mesh.panel_count**2

    for parity in (1.0, -1.0):
        ops = operators_for_parity(blocks, parity)
        ref_s = full.single_layer[np.ix_(idx_u, idx_u)] + parity * full.single_layer[
            np.ix_(idx_u, idx_m)
        ]
        ref_h = full.double_layer[np.ix_(idx_u, idx_u)] + parity * full.double_layer[
            np.ix_(idx_u, idx_m)
        ]
        # double layer is exact (analytic solid angles): round-off only
        assert np.abs(ops.double_layer - ref_h).max() <= 1e-12 * np.abs(ref_h).max() + 1e-15
        # single layer differs only through the Duffy rule applied to a folded pair
        assert np.abs(ops.single_layer - ref_s).max() <= 1e-5 * np.abs(ref_s).max()


def test_operators_reject_bad_parity_and_mismatched_blocks(case):
    _, upper, _ = case
    blocks = assemble_symmetric_blocks(upper, NU)
    with pytest.raises(ValueError):
        operators_for_parity(blocks, 0.5)
    with pytest.raises(ValueError):
        assemble_symmetric_blocks(upper, -1.0)


# -------------------------------------------------------------- radiation
def test_radiation_matches_full_mesh(case):
    mesh, upper, _ = case
    full = solve_radiation_3d(mesh, NU)
    half = solve_radiation_3d_symmetric(upper, NU)
    scale_a = np.abs(full.added_mass).max()
    scale_b = np.abs(full.damping).max()
    for name, red, ref, scale in (
        ("added mass", half.added_mass, full.added_mass, scale_a),
        ("damping", half.damping, full.damping, scale_b),
    ):
        same = SAME_PARITY
        assert np.abs(red[same] - ref[same]).max() <= 1e-6 * scale, name
        # couplings between modes of different parity vanish identically when the
        # parity is enforced, while the full solve leaves cancellation residue
        assert np.all(red[~same] == 0.0), name
        assert np.abs(ref[~same]).max() <= 1e-6 * scale, name


def test_radiation_single_mode_and_subset(case):
    mesh, upper, _ = case
    for mode in (1, 2, 3):
        full = solve_radiation_3d(mesh, NU, modes=[mode])
        half = solve_radiation_3d_symmetric(upper, NU, modes=[mode])
        assert half.added_mass.shape == (1, 1)
        assert half.added_mass[0, 0] == pytest.approx(full.added_mass[0, 0], rel=1e-6)
        assert half.damping[0, 0] == pytest.approx(full.damping[0, 0], rel=1e-6)


def test_radiation_uses_two_small_systems(case):
    _, upper, _ = case
    blocks = assemble_symmetric_blocks(upper, NU)
    half = solve_radiation_3d_symmetric(upper, NU, blocks=blocks)
    assert half.potentials.shape == (upper.panel_count, 6)
    assert half.relative_residual < 1e-10
    assert np.all(np.isfinite(half.added_mass))
    assert np.all(np.isfinite(half.damping))


def test_radiation_chief_consistency(case):
    """At an irregular frequency the folded CHIEF rows must reproduce the full ones."""
    mesh, upper, _ = case
    nu_irregular = 2.58
    chief = _auto_chief_points_3d(mesh)
    full = solve_radiation_3d(mesh, nu_irregular, modes=[2], chief_points=chief)
    half = solve_radiation_3d_symmetric(upper, nu_irregular, modes=[2], chief_points=chief)
    assert np.all(full.chief_points_used.shape == chief.shape)
    assert full.damping[0, 0] > 0.0 and half.damping[0, 0] > 0.0
    # At an irregular frequency the CHIEF least-squares solution amplifies the
    # (measured) 1e-5 relative single-layer quadrature difference between the
    # folded and full systems; 64 panels is the coarsest admissible mesh, where
    # the observed split is 1.6e-3 in A_33 (see docs/symmetry3d.md).
    assert half.damping[0, 0] == pytest.approx(full.damping[0, 0], rel=1e-2)
    assert half.added_mass[0, 0] == pytest.approx(full.added_mass[0, 0], rel=1e-2)


# ------------------------------------------------------------- diffraction
@pytest.mark.parametrize("beta", [0.0, 0.6, 1.2, math.pi / 2])
def test_diffraction_matches_full_mesh(case, beta):
    mesh, upper, _ = case
    full = solve_diffraction_3d(mesh, NU, wave_angle=beta)
    half = solve_diffraction_3d_symmetric(upper, NU, wave_angle=beta)
    scale = np.abs(full.excitation_force).max()
    assert np.abs(half.excitation_force - full.excitation_force).max() <= 1e-6 * scale
    # incident normal velocity on the retained half is unchanged by the reduction
    np.testing.assert_allclose(
        half.incident_normal_velocity,
        solve_diffraction_3d(mesh, NU, wave_angle=beta).incident_normal_velocity[
            _index_of(upper, mesh)
        ],
        atol=1e-12,
    )


def test_diffraction_head_seas_sway_force_is_zero(case):
    mesh, upper, _ = case
    half = solve_diffraction_3d_symmetric(upper, NU, wave_angle=0.0)
    full = solve_diffraction_3d(mesh, NU, wave_angle=0.0)
    scale = np.abs(full.excitation_force).max()
    assert abs(half.excitation_force[1]) <= 1e-6 * scale  # beam-transverse load
    assert abs(full.excitation_force[1]) <= 1e-6 * scale


def test_diffraction_chief_consistency(case):
    mesh, upper, _ = case
    nu_irregular = 2.58
    chief = _auto_chief_points_3d(mesh)
    full = solve_diffraction_3d(mesh, nu_irregular, wave_angle=0.6, chief_points=chief)
    half = solve_diffraction_3d_symmetric(
        upper, nu_irregular, wave_angle=0.6, chief_points=chief
    )
    scale = np.abs(full.excitation_force).max()
    # observed 2.5e-3 at this coarse mesh (2.8e-4 at 256 panels); the CHIEF rows
    # are rank-deficient by construction, so the least-squares solution is
    # sensitive at the residual level (1%), not the round-off level.
    assert np.abs(half.excitation_force - full.excitation_force).max() <= 1e-2 * scale


def _index_of(patch, mesh) -> np.ndarray:
    dist = np.linalg.norm(patch.centroids[:, None, :] - mesh.centroids[None, :, :], axis=2)
    idx = np.argmin(dist, axis=1)
    assert dist[np.arange(patch.centroids.shape[0]), idx].max() < 1e-12
    return idx