"""Tests for the symmetry-plane (half-model) reduction in ``tensorbem.symmetry``.

The decisive assertion is the *involution*: the half model and the ordinary
full model, run on the same triangulation (the half surface mirrored and
welded by :meth:`HalfSurface.closed_mesh`), must agree to roundoff.  A parity
or image-sign mistake cannot survive it.  The remaining assertions pin the
closed-form accuracy, the exact zeros of the opposite-parity coupling, the
double-layer row-sum identity that fixes the image sign, and the validation
rules.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import ellipsoid_added_mass_matrix
from tensorbem.solver3d import compute_added_mass_matrix_3d
from tensorbem.symmetry import (
    MODE_PARITIES,
    HalfSurface,
    assemble_half_surface_matrices,
    compute_added_mass_matrix_3d_half,
    half_ellipsoid_surface,
)

AXES = (1.0, 0.6, 0.4)
COARSE = (4, 8)
REFERENCE = (8, 16)


def _trimmed_relative_error(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(value) - np.asarray(reference)) / np.maximum(
        np.abs(reference), np.finfo(float).tiny
    )


@pytest.fixture(scope="module")
def ellipsoid_study() -> dict:
    """Half and full results for the ellipsoid at two refinement levels."""
    exact = np.diag(ellipsoid_added_mass_matrix(np.array(AXES)))
    study: dict = {"exact": exact, "levels": {}}
    for level in (COARSE, REFERENCE):
        surface = half_ellipsoid_surface(AXES, *level)
        closed = surface.closed_mesh()
        half = compute_added_mass_matrix_3d_half(surface)
        full = compute_added_mass_matrix_3d(closed)
        study["levels"][level] = {
            "surface": surface,
            "closed": closed,
            "half": half,
            "full": full,
        }
    return study


def test_half_panel_count_formula() -> None:
    for n_phi, n_theta in (COARSE, REFERENCE):
        surface = half_ellipsoid_surface(AXES, n_phi, n_theta)
        assert surface.panel_count == 2 * n_theta * (n_phi - 1)
        assert surface.vertex_count == 2 + (n_phi - 1) * (n_theta + 1)


def test_closed_mesh_is_valid_and_welds_the_seam() -> None:
    surface = half_ellipsoid_surface(AXES, *REFERENCE)
    closed = surface.closed_mesh()
    assert closed.panel_count == 2 * surface.panel_count
    # The symmetry-plane vertices (two poles plus both seams) are shared.
    assert closed.vertex_count < 2 * surface.vertex_count
    volume = float(
        np.sum(np.einsum("ij,ij->i", closed.v0, np.cross(closed.v1, closed.v2))) / 6.0
    )
    exact_volume = 4.0 / 3.0 * math.pi * float(np.prod(AXES))
    # A 448-panel inscribed triangulation of this body carries a few percent of
    # volume deficit; the mesh is accepted by ``TriangleMesh`` (closedness,
    # orientation and positive volume are validated there).
    assert abs(volume - exact_volume) / exact_volume < 0.06


def test_mirrored_surface_keeps_outward_normals() -> None:
    surface = half_ellipsoid_surface(AXES, *COARSE)
    image = surface.mirrored()
    assert image.side == -1
    assert np.all(image.centroids[:, 0] <= 1.0e-12)
    assert np.all(np.einsum("ij,ij->i", image.normals, image.centroids) > 0.0)


def test_involution_matches_full_model_to_roundoff(ellipsoid_study: dict) -> None:
    for level, record in ellipsoid_study["levels"].items():
        half = record["half"].added_mass_matrix
        full = record["full"].added_mass_matrix
        difference = np.max(np.abs(half - full)) / np.max(np.abs(full))
        assert difference < 1.0e-12, f"level {level}: half model differs from full by {difference:.2e}"


def test_opposite_parity_couplings_are_exact_zeros(ellipsoid_study: dict) -> None:
    parities = np.asarray(MODE_PARITIES)
    off_parity = np.outer(parities, parities) < 0.0
    # Surge/pitch/yaw are odd, sway/heave/roll are even: nine odd-even pairs,
    # i.e. eighteen signed entries of the (6, 6) tensor, must vanish identically.
    assert off_parity.sum() == 18
    for record in ellipsoid_study["levels"].values():
        matrix = record["half"].added_mass_matrix
        assert np.all(matrix[off_parity] == 0.0)
        assert record["half"].off_parity_max == 0.0


def test_double_layer_row_sum_identity_fixes_the_image_sign() -> None:
    surface = half_ellipsoid_surface(AXES, *COARSE)
    _, even_double = assemble_half_surface_matrices(surface, 1.0)
    # The folded even-parity double layer is the full closed-surface operator, so
    # its row sums must still be -1/2; a wrong image sign would leave a gap of 1.
    assert np.max(np.abs(even_double.sum(axis=1) + 0.5)) < 1.0e-12
    _, odd_double = assemble_half_surface_matrices(surface, -1.0)
    assert np.max(np.abs(odd_double.sum(axis=1) + 0.5)) > 0.1


def test_solve_diagnostics(ellipsoid_study: dict) -> None:
    for record in ellipsoid_study["levels"].values():
        result = record["half"]
        assert result.max_relative_residual_l2 < 1.0e-12
        assert result.half_panel_count * 2 == result.full_panel_count
        assert result.condition_number_even > 1.0
        assert result.condition_number_odd > 1.0
        assert result.symmetry_error / np.max(np.abs(result.added_mass_matrix)) < 5.0e-3
        assert np.all(np.diag(result.added_mass_matrix)[:3] > 0.0)


def test_translation_added_mass_against_closed_form(ellipsoid_study: dict) -> None:
    exact = ellipsoid_study["exact"]
    coarse = ellipsoid_study["levels"][COARSE]["half"].added_mass_matrix
    reference = ellipsoid_study["levels"][REFERENCE]["half"].added_mass_matrix
    coarse_error = _trimmed_relative_error(np.diag(coarse)[:3], exact[:3])
    reference_error = _trimmed_relative_error(np.diag(reference)[:3], exact[:3])
    # Measured at the reference level: 2.7 % / 2.5 % / 1.8 % for surge/sway/heave;
    # the benchmark module refines further (0.7 % / 0.6 % / 0.5 % at 960 panels).
    assert np.all(reference_error < 0.04)
    assert np.all(reference_error < coarse_error)


def test_observed_order_is_first_order(ellipsoid_study: dict) -> None:
    exact = ellipsoid_study["exact"]
    panels = [
        ellipsoid_study["levels"][level]["surface"].panel_count
        for level in (COARSE, REFERENCE)
    ]
    errors = [
        _trimmed_relative_error(
            np.diag(ellipsoid_study["levels"][level]["half"].added_mass_matrix)[0], exact[:1]
        )[0]
        for level in (COARSE, REFERENCE)
    ]
    order = math.log(errors[0] / errors[1]) / math.log(panels[1] / panels[0])
    assert 0.3 < order < 1.6


def test_sphere_rotational_added_mass_vanishes() -> None:
    surface = half_ellipsoid_surface((1.0, 1.0, 1.0), *REFERENCE)
    result = compute_added_mass_matrix_3d_half(surface)
    diagonal = np.diag(result.added_mass_matrix)
    # Inviscid sphere: zero added inertia about the centre, 2/3*pi*a^3 translating.
    assert abs(diagonal[3]) / diagonal[0] < 5.0e-3
    assert abs(diagonal[0] / (2.0 * math.pi / 3.0) - 1.0) < 0.04


def test_center_must_lie_in_the_symmetry_plane() -> None:
    surface = half_ellipsoid_surface(AXES, *COARSE)
    with pytest.raises(ValueError, match="symmetry plane"):
        compute_added_mass_matrix_3d_half(surface, center=(0.1, 0.0, 0.0))


def test_argument_validation() -> None:
    surface = half_ellipsoid_surface(AXES, *COARSE)
    with pytest.raises(ValueError, match="parity"):
        assemble_half_surface_matrices(surface, 0.0)
    with pytest.raises(ValueError, match="quadrature_order"):
        assemble_half_surface_matrices(surface, 1.0, quadrature_order=0)
    with pytest.raises(TypeError, match="HalfSurface"):
        compute_added_mass_matrix_3d_half(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="density"):
        compute_added_mass_matrix_3d_half(surface, density=-1.0)
    with pytest.raises(ValueError, match="semi_axes"):
        half_ellipsoid_surface((1.0, 0.0, 1.0), *COARSE)


def test_half_surface_rejects_wrong_side() -> None:
    surface = half_ellipsoid_surface(AXES, *COARSE)
    with pytest.raises(ValueError, match="declared side"):
        HalfSurface(surface.vertices, surface.faces, axis=0, side=-1)


def test_half_surface_rejects_closed_surface() -> None:
    closed = half_ellipsoid_surface(AXES, *COARSE).closed_mesh()
    with pytest.raises(ValueError):
        HalfSurface(closed.vertices, closed.faces, axis=0, side=1)


def test_half_surface_rejects_boundary_off_the_plane() -> None:
    vertices = np.array(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    with pytest.raises(ValueError, match="symmetry plane"):
        HalfSurface(vertices, faces, axis=0, side=1)