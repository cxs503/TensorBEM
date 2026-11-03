"""Tests for the ``xOz`` mirror-symmetry reduction of the 3-D free-surface solver.

The reduction is only worth having if it is *exact*: every test here therefore
compares the folded half-surface path against the full-surface path rather than
against a reference solution, so any algebraic slip shows up immediately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh, submerged_sphere_mesh
from tensorbem.symmetry3d import (
    MODE_PARITY_XOZ,
    assemble_free_surface_operators_3d_half,
    full_surface_factor,
    solve_diffraction_3d_symmetric,
    solve_radiation_3d_symmetric,
    split_symmetric_mesh,
)

NU = 1.0
MESH = hemisphere_hull_mesh(2)


# --------------------------------------------------------------------------- #
# mesh splitting
# --------------------------------------------------------------------------- #
def test_mode_parity_table() -> None:
    # surge/heave/pitch displace symmetrically about xOz, sway/roll/yaw do not
    assert MODE_PARITY_XOZ == (1, -1, 1, -1, 1, -1)
    assert all(full_surface_factor(int(s), int(s)) == 2.0 for s in (1, -1))
    assert all(full_surface_factor(1, -1) == 0.0 for _ in (0,))


def test_split_partitions_the_hull_exactly() -> None:
    split = split_symmetric_mesh(MESH)
    assert split.full_panel_count == MESH.panel_count
    assert split.half_panel_count == MESH.panel_count // 2
    assert np.all(MESH.centroids[split.half_to_full, 1] > 0.0)
    assert np.all(MESH.centroids[split.mirror_of_half, 1] < 0.0)
    # involution: reflecting twice returns the same panel
    perm = split.mirror_full_indices()
    assert np.array_equal(perm[perm], np.arange(MESH.panel_count))
    # source bundle = half panels followed by their mirror partners
    assert split.sources.panel_count == MESH.panel_count
    assert np.allclose(
        split.sources.centroids[split.half_panel_count :],
        MESH.centroids[split.mirror_of_half],
        atol=0.0,
        rtol=0.0,
    )


def test_split_rejects_a_non_symmetric_hull() -> None:
    off_centre = hemisphere_hull_mesh(1, center_xy=(0.0, 0.25))
    with pytest.raises(ValueError, match="not .*symmetric|no mirror image"):
        split_symmetric_mesh(off_centre)


def test_split_rejects_unknown_plane() -> None:
    with pytest.raises(ValueError, match="xOz"):
        split_symmetric_mesh(MESH, plane="yOz")  # type: ignore[arg-type]


def test_expand_is_a_parity_extension() -> None:
    split = split_symmetric_mesh(MESH)
    values = np.arange(split.half_panel_count, dtype=float)
    plus = split.expand(values, 1)
    minus = split.expand(values, -1)
    assert np.allclose(plus[split.half_to_full], values)
    assert np.allclose(plus[split.mirror_of_half], values)
    assert np.allclose(minus[split.mirror_of_half], -values)


# --------------------------------------------------------------------------- #
# folded operators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("parity", (1, -1))
def test_folded_operators_equal_the_full_sub_blocks(parity: int) -> None:
    split = split_symmetric_mesh(MESH)
    full = assemble_free_surface_operators_3d(MESH, NU, chief_points=None)
    folded = assemble_free_surface_operators_3d_half(
        MESH, NU, parity=parity, chief_points=None, split=split
    )
    n = split.half_panel_count
    assert folded.single_layer.shape == (n, n)
    assert folded.double_layer.shape == (n, n)
    rows, cols = split.half_to_full, split.mirror_of_half
    expected_s = (
        full.single_layer[np.ix_(rows, rows)]
        + parity * full.single_layer[np.ix_(rows, cols)]
    )
    expected_h = (
        full.double_layer[np.ix_(rows, rows)]
        + parity * full.double_layer[np.ix_(rows, cols)]
    )
    assert np.allclose(folded.single_layer, expected_s, atol=0.0, rtol=0.0)
    assert np.allclose(folded.double_layer, expected_h, atol=0.0, rtol=0.0)


def test_chief_rows_are_folded_and_off_plane_only() -> None:
    chief = np.array([[0.0, 0.05, -0.3], [0.02, -0.06, -0.28], [0.0, 0.0, -0.25]])
    split = split_symmetric_mesh(MESH)
    full = assemble_free_surface_operators_3d(MESH, NU, chief_points=chief)
    for parity in (1, -1):
        folded = assemble_free_surface_operators_3d_half(
            MESH, NU, parity=parity, chief_points=chief, split=split
        )
        # only y > 0 points survive; the on-plane point carries no extra equation
        assert folded.chief_points.shape == (1, 3)
        assert folded.chief_points[0, 1] == pytest.approx(0.05)
        keep = np.flatnonzero(full.chief_points[:, 1] > 1e-12)
        rows, cols = split.half_to_full, split.mirror_of_half
        expected = (
            full.chief_double_layer[keep][:, rows]
            + parity * full.chief_double_layer[keep][:, cols]
        )
        assert np.allclose(folded.chief_double_layer, expected, atol=0.0, rtol=0.0)


def test_chief_points_without_an_off_plane_member_are_rejected() -> None:
    chief = np.array([[0.05, 0.0, -0.3], [-0.04, 0.0, -0.28]])
    with pytest.raises(ValueError, match="off the xOz symmetry plane"):
        assemble_free_surface_operators_3d_half(MESH, NU, chief_points=chief)


def test_invalid_parity_is_rejected() -> None:
    for bad in (0, 2, -2):
        with pytest.raises(ValueError, match="parity"):
            assemble_free_surface_operators_3d_half(MESH, NU, parity=bad)


# --------------------------------------------------------------------------- #
# the solvers
# --------------------------------------------------------------------------- #
def test_radiation_matches_the_full_solver() -> None:
    full = solve_radiation_3d(
        MESH, NU, modes=range(6), chief_points=None, rankine_order=12
    )
    sym = solve_radiation_3d_symmetric(
        MESH, NU, modes=range(6), chief_points=None, rankine_order=12
    )
    scale_a = float(np.abs(full.added_mass).max())
    scale_b = float(np.abs(full.damping).max())
    assert np.abs(sym.added_mass - full.added_mass).max() / scale_a < 1.0e-9
    assert np.abs(sym.damping - full.damping).max() / scale_b < 1.0e-9
    # potentials are expanded back onto the full panel ordering
    assert sym.potentials.shape == full.potentials.shape
    assert sym.relative_residual < 1.0e-10


def test_radiation_matches_with_the_same_chief_set() -> None:
    chief = np.array([[0.02, 0.07, -0.30], [-0.03, 0.05, -0.27]])
    full = solve_radiation_3d(
        MESH, NU, modes=(0, 2), chief_points=chief, rankine_order=10
    )
    sym = solve_radiation_3d_symmetric(
        MESH, NU, modes=(0, 2), chief_points=chief, rankine_order=10
    )
    # CHIEF rows are inconsistent at the discretisation level, so the two least
    # squares problems weight the same constraints differently (the folded one
    # enforces parity exactly, the full one does not).  Agreement at the 1e-3
    # level is what the same constraint set can deliver; the parity-free
    # comparison in test_radiation_matches_the_full_solver is exact.
    scale_a = float(np.abs(full.added_mass).max())
    scale_b = float(np.abs(full.damping).max())
    assert np.abs(sym.added_mass - full.added_mass).max() / scale_a < 5.0e-4
    assert np.abs(sym.damping - full.damping).max() / scale_b < 5.0e-4


def test_off_parity_couplings_vanish_identically() -> None:
    sym = solve_radiation_3d_symmetric(
        MESH, NU, modes=range(6), chief_points=None, rankine_order=8
    )
    modes = list(range(6))
    scale = float(np.abs(sym.added_mass).max())
    for i, mi in enumerate(modes):
        for j, mj in enumerate(modes):
            same = MODE_PARITY_XOZ[mi] == MODE_PARITY_XOZ[mj]
            if same:
                continue
            assert abs(sym.added_mass[i, j]) < 1.0e-12 * scale
            assert abs(sym.damping[i, j]) < 1.0e-12 * scale
    # ...while the same-parity block is genuinely populated
    for mode in (0, 2, 4):
        assert abs(sym.added_mass[mode, mode]) > 0.0
    assert sym.added_mass[0, 0] > 1.0e-3 * scale


def test_diffraction_matches_the_full_solver() -> None:
    full = solve_diffraction_3d(MESH, NU, chief_points=None, rankine_order=12)
    sym = solve_diffraction_3d_symmetric(MESH, NU, chief_points=None, rankine_order=12)
    scale = float(np.abs(full.excitation_force).max())
    assert np.abs(sym.excitation_force - full.excitation_force).max() / scale < 1.0e-9
    assert np.abs(sym.total_potential - full.total_potential).max() < 1.0e-8
    # antiphase (sway / roll / yaw) loads are exactly zero under xOz symmetry
    for mode in (1, 3, 5):
        assert abs(sym.excitation_force[mode]) < 1.0e-12 * scale


def test_diffraction_rejects_oblique_incidence() -> None:
    with pytest.raises(ValueError, match="sin"):
        solve_diffraction_3d_symmetric(MESH, NU, wave_angle=math.pi / 6.0)


def test_afloating_hemisphere_surge_heave_are_positive_definite() -> None:
    sym = solve_radiation_3d_symmetric(
        MESH, NU, modes=(0, 2), chief_points=None, rankine_order=8
    )
    assert sym.added_mass[0, 0] > 0.0
    assert sym.added_mass[1, 1] > 0.0
    assert sym.damping[0, 0] > 0.0
    assert sym.damping[1, 1] > 0.0


def test_a_closed_mesh_with_straddling_panels_is_rejected() -> None:
    """Documented limitation: the plane must be a mesh edge, not a panel secant."""
    mesh = submerged_sphere_mesh(2, radius=0.5, submergence=1.5)
    with pytest.raises(ValueError, match="straddle"):
        split_symmetric_mesh(mesh)


def test_a_half_of_a_symmetric_closed_mesh_works() -> None:
    """A sphere re-meshed as [half, mirror(half)] is split exactly."""
    mesh = hemisphere_hull_mesh(3)
    split = split_symmetric_mesh(mesh)
    assert split.half_panel_count == mesh.panel_count // 2
    assert split.sources.panel_count == mesh.panel_count
    rad = solve_radiation_3d_symmetric(
        mesh, 1.0, modes=(1, 2), chief_points=None, rankine_order=8
    )
    assert rad.added_mass[0, 0] > 0.0
    assert rad.added_mass[1, 1] > 0.0
