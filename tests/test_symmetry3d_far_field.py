"""Far-field post-processing of the ``y = 0`` symmetry reduction (folding tests).

The half-surface Kochin function is the one far-field quantity that folds as
``K_full(theta) = K_half(theta) + s * K_half(-theta)``.  Two levels are asserted:

* **algebra** -- with *symmetrised* full-surface data the folded integral must
  reproduce the full-surface one to round-off (a statement about the formula);
* **end to end** -- a folded value taken from the actual half solve must match
  the full solve to the half/full solve difference (a statement about the solver,
  bounded here at 1e-4 of the largest Kochin amplitude anywhere).

Every bound below is measured by
``python -m tensorbem.benchmarks.symmetry_far_field`` (``--subdivisions 2``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.free_surface_solver3d import (
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    kochin_function_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.symmetry3d import (
    MODE_PARITY,
    far_field_damping_matrix_3d_symmetric,
    folded_kochin_function_3d,
    haskind_excitation_force_3d_symmetric,
    solve_radiation_3d_symmetric,
    split_patch,
)

AZIMUTHS = np.linspace(0.0, 2.0 * math.pi, 37, endpoint=False)
PARITY = np.array([MODE_PARITY[k] for k in range(6)], dtype=float)


@pytest.fixture(scope="module")
def hemisphere():
    mesh = hemisphere_hull_mesh(2)  # 64 full panels, 32 retained
    upper, _ = split_patch(mesh)
    full = solve_radiation_3d(mesh, 1.0)
    half = solve_radiation_3d_symmetric(upper, 1.0)
    return mesh, upper, full, half


def _mirror_index(mesh):
    out = np.empty(mesh.panel_count, dtype=np.int64)
    for idx, centroid in enumerate(mesh.centroids):
        mirrored = centroid * np.array([1.0, -1.0, 1.0])
        out[idx] = int(np.argmin(np.linalg.norm(mesh.centroids - mirrored, axis=1)))
    return out


def _retained_index(mesh, upper):
    out = np.empty(upper.panel_count, dtype=np.int64)
    for idx, centroid in enumerate(upper.centroids):
        out[idx] = int(np.argmin(np.linalg.norm(mesh.centroids - centroid, axis=1)))
    return out


def test_folding_identity_is_round_off(hemisphere):
    """Symmetrised full-surface data: the folded integral *is* the full integral."""
    mesh, upper, full, _ = hemisphere
    idx_upper = _retained_index(mesh, upper)
    idx_mirror = _mirror_index(mesh)
    worst = 0.0
    scale = 0.0
    for i, mode in enumerate(full.modes):
        s = PARITY[int(mode)]
        phi = 0.5 * (full.potentials[:, i] + s * full.potentials[idx_mirror, i])
        vn = 0.5 * (
            full.normal_velocities[:, i] + s * full.normal_velocities[idx_mirror, i]
        )
        k_full = kochin_function_3d(mesh, phi, vn, 1.0, AZIMUTHS)
        k_fold = folded_kochin_function_3d(
            upper, phi[idx_upper], vn[idx_upper], 1.0, AZIMUTHS, parity=s
        )
        worst = max(worst, float(np.abs(k_full - k_fold).max()))
        scale = max(scale, float(np.abs(k_full).max()))
    assert worst <= 1.0e-11 * scale, (worst, scale)


def test_folded_kochin_is_even_or_odd_in_azimuth(hemisphere):
    """``s = +1`` data gives ``K(theta) = K(-theta)``, ``s = -1`` gives the antisymmetric."""
    _, upper, full, half = hemisphere
    for mode in (2, 1):  # heave (even), sway (odd)
        i = int(np.nonzero(full.modes == mode)[0][0])
        s = PARITY[mode]
        k_theta = folded_kochin_function_3d(
            upper,
            half.potentials[:, i],
            half.normal_velocities[:, i],
            1.0,
            AZIMUTHS,
            parity=s,
        )
        k_minus = folded_kochin_function_3d(
            upper,
            half.potentials[:, i],
            half.normal_velocities[:, i],
            1.0,
            -AZIMUTHS,
            parity=s,
        )
        np.testing.assert_allclose(k_theta, s * k_minus, atol=1.0e-10)


def test_folded_kochin_matches_full_mesh_solve(hemisphere):
    mesh, upper, full, half = hemisphere
    worst = 0.0
    scale = 0.0
    for i, mode in enumerate(full.modes):
        k_full = kochin_function_3d(
            mesh, full.potentials[:, i], full.normal_velocities[:, i], 1.0, AZIMUTHS
        )
        k_fold = folded_kochin_function_3d(
            upper,
            half.potentials[:, i],
            half.normal_velocities[:, i],
            1.0,
            AZIMUTHS,
            parity=PARITY[int(mode)],
        )
        worst = max(worst, float(np.abs(k_full - k_fold).max()))
        scale = max(scale, float(np.abs(k_full).max()))
    assert worst <= 1.0e-4 * scale, (worst, scale)


def test_far_field_damping_matches_full_mesh(hemisphere):
    mesh, upper, full, half = hemisphere
    b_full = far_field_damping_matrix_3d(mesh, full)
    b_half = far_field_damping_matrix_3d_symmetric(upper, half)
    scale = float(np.abs(b_full).max())
    assert np.abs(b_full - b_half).max() <= 1.0e-6 * scale
    # opposite-parity couplings are zero by symmetry; the folded matrix realizes it
    cross = PARITY[full.modes][:, None] != PARITY[full.modes][None, :]
    assert np.abs(b_half[cross]).max() <= 1.0e-6 * scale


def test_far_field_damping_shape_follows_modes():
    mesh = hemisphere_hull_mesh(1)
    upper, _ = split_patch(mesh)
    half = solve_radiation_3d_symmetric(upper, 1.0, modes=[2, 1])
    b = far_field_damping_matrix_3d_symmetric(upper, half)
    assert b.shape == (2, 2)


def test_haskind_matches_full_mesh(hemisphere):
    mesh, upper, full, half = hemisphere
    h_full = haskind_excitation_force_3d(mesh, full)
    h_half = haskind_excitation_force_3d_symmetric(upper, half)
    scale = float(np.abs(h_full["kochin_haskind"]).max())
    for key in ("boundary_haskind", "kochin_haskind"):
        assert np.abs(h_full[key] - h_half[key]).max() <= 1.0e-6 * scale, key


def test_haskind_near_and_far_field_agree_on_the_half_surface(hemisphere):
    _, upper, _, half = hemisphere
    h = haskind_excitation_force_3d_symmetric(upper, half)
    scale = float(np.abs(h["kochin_haskind"]).max())
    assert np.abs(h["boundary_haskind"] - h["kochin_haskind"]).max() <= 1.0e-2 * scale


def test_folded_kochin_rejects_bad_inputs(hemisphere):
    _, upper, full, half = hemisphere
    with pytest.raises(ValueError):
        folded_kochin_function_3d(
            upper, half.potentials[:, 0], half.normal_velocities[:, 0], 1.0, AZIMUTHS,
            parity=0.5,
        )
    with pytest.raises(ValueError):
        folded_kochin_function_3d(
            upper, half.potentials[:, 0][:-1], half.normal_velocities[:, 0], 1.0,
            AZIMUTHS, parity=1.0,
        )
    with pytest.raises(ValueError):
        folded_kochin_function_3d(
            upper, half.potentials[:, 0], half.normal_velocities[:, 0], -1.0, AZIMUTHS,
            parity=1.0,
        )
    with pytest.raises(ValueError):
        folded_kochin_function_3d(
            object(), half.potentials[:, 0], half.normal_velocities[:, 0], 1.0,
            AZIMUTHS, parity=1.0,
        )
    with pytest.raises(ValueError):
        far_field_damping_matrix_3d_symmetric(upper, full)  # wrong patch for the solve
    with pytest.raises(ValueError):
        far_field_damping_matrix_3d_symmetric(upper, half, n_azimuth=8)
    with pytest.raises(ValueError):
        haskind_excitation_force_3d_symmetric(upper, half, gravity=-9.81)
