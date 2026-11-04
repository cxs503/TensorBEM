"""Guards for the 3-D mutation survivors of ``mutation_audit_3d`` (mission m3471).

The 2026-10-24 mutation audit (``docs/mutation_audit_tensorbem.md``) only
sampled the *2-D* modules.  Mission m3471 extended the same sweep to the 3-D
BEM core (``tensorbem.kernels3d``, ``tensorbem.free_surface3d``,
``tensorbem.solver3d``): 24 mutants, 17 killed, 7 survived.  Three of the seven
are *equivalent* mutants (non-discriminating by construction, see
``docs/m3471/mutation_audit_3d.md`` §5) and must not get a test.  The four real
gaps are closed here:

==========================================================  ============================
survivor                                                     closed here by
==========================================================  ============================
``kernels3d:107`` ``order: int = 8`` -> ``-8``               default-order contract
``solver3d:269`` ``residuals / rhs_norms`` -> ``*``          residual is a *ratio*
``solver3d:272`` ``A - A.T`` -> ``A + A.T``                  symmetry + coupling
``solver3d:310`` ``. / edge_length_squared`` -> ``*``        point-to-surface distance
==========================================================  ============================

Every reference below is independent of the formula it polices: an exact
integral over a triangle, a dense barycentric sampling of the surface, or a
quantity reconstructed from a *different* function (``assemble_surface_matrices``)
than the one that reports it.  None of them re-states an implementation formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.geometry3d import TriangleMesh, icosphere
from tensorbem.kernels3d import duffy_quadrature, duffy_quadrature_batch
from tensorbem.solver3d import (
    _minimum_distance_to_mesh,
    assemble_surface_matrices,
    compute_added_mass_matrix_3d,
)

# Two triangles with different shapes and orientations, so a default-order bug
# cannot hide behind a single "nice" geometry.
BATCH_V0 = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]])
BATCH_V1 = np.array([[3.0, 0.0, 0.0], [2.0, 1.5, 0.5]])
BATCH_V2 = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 1.5]])


def test_duffy_quadrature_batch_honours_the_documented_default_order():
    """``duffy_quadrature_batch(v0, v1, v2)`` must mean ``order=8``.

    The reference is an exact integral: the Duffy map sends the square onto the
    triangle with Jacobian ``2 * area * u``, so *any* Gauss rule reproduces the
    area and the first geometric moments of a linear integrand exactly.  A
    wrong default (``-8``) makes the call raise, which is what the mutant did
    while every test passed.
    """
    defaults, default_weights = duffy_quadrature_batch(BATCH_V0, BATCH_V1, BATCH_V2)
    explicit, explicit_weights = duffy_quadrature_batch(
        BATCH_V0, BATCH_V1, BATCH_V2, order=8
    )

    # 8x8 Gauss-Legendre in the Duffy square: the documented default family.
    assert defaults.shape == (2, 64, 3)
    assert default_weights.shape == (2, 64)
    assert np.array_equal(defaults, explicit)
    assert np.array_equal(default_weights, explicit_weights)

    areas = 0.5 * np.linalg.norm(
        np.cross(BATCH_V1 - BATCH_V0, BATCH_V2 - BATCH_V0), axis=1
    )
    centroids = (BATCH_V0 + BATCH_V1 + BATCH_V2) / 3.0
    assert np.allclose(default_weights.sum(axis=1), areas, rtol=0.0, atol=1.0e-12)
    moments = np.einsum("fp,fpc->fc", default_weights, defaults)
    assert np.allclose(moments, areas[:, None] * centroids, rtol=0.0, atol=1.0e-12)

    # The scalar routine is the same rule for one triangle.
    single_points, single_weights = duffy_quadrature(
        np.vstack((BATCH_V0[1], BATCH_V1[1], BATCH_V2[1]))
    )
    assert np.allclose(default_weights[1], single_weights, rtol=0.0, atol=1.0e-12)
    assert np.allclose(moments[1], np.einsum("p,pc->c", single_weights, single_points), atol=1.0e-12)


def _sheared_icosphere(subdivisions: int = 1) -> TriangleMesh:
    """A centro-symmetric but anisotropic body: a sheared, stretched sphere.

    The linear map has positive determinant, so closure and outward orientation
    survive; the shape has no coordinate plane of symmetry, which is what makes
    the off-diagonal added-mass entries non-zero and the symmetry assertion
    non-vacuous.
    """
    base = icosphere(subdivisions)
    vertices = np.column_stack(
        (
            base.vertices[:, 0] + 0.35 * base.vertices[:, 2],
            base.vertices[:, 1] - 0.20 * base.vertices[:, 2],
            1.30 * base.vertices[:, 2],
        )
    )
    return TriangleMesh(vertices, base.faces)


def test_added_mass_matrix_is_symmetric_where_it_must_be():
    """``A - A.T`` (a diagnostic) must not become ``A + A.T`` (a defect).

    Physics reference: the added-mass tensor of a body in an unbounded
    inviscid fluid is symmetric, so ``A == A.T`` for *any* coupling that is
    actually present.  The guard therefore also asserts that a non-zero
    off-diagonal coupling exists -- otherwise a matrix of zeros would satisfy
    symmetry and the test would be worthless.
    """
    mesh = _sheared_icosphere(1)
    result = compute_added_mass_matrix_3d(mesh, density=1025.0)
    matrix = result.added_mass_matrix
    assert matrix.shape == (6, 6)

    scale = float(np.max(np.abs(matrix)))
    assert scale > 0.0
    off_diagonal = float(np.max(np.abs(matrix[~np.eye(6, dtype=bool)])))
    assert off_diagonal > 1.0e-2 * scale, (
        "the anisotropic body must produce a non-trivial coupling term; "
        f"max|off-diagonal| = {off_diagonal:.6e} vs scale {scale:.6e}"
    )

    # A collocation BEM matrix is symmetric only up to discretisation error
    # (measured: 6.6e-6 relative on this body), so the tolerance is set two
    # decades above it -- and still seven decades below the mutant, whose
    # antisymmetrised matrix carries 2*|A_13| = 1.5e3 absolute (0.5 relative).
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    assert asymmetry <= 1.0e-4 * scale
    # The reported diagnostic has to agree with the independent measurement.
    assert result.symmetry_error == pytest.approx(asymmetry, rel=1.0e-9, abs=1.0e-12)


def test_reported_relative_residual_is_a_ratio_not_a_product():
    """``max_relative_residual_l2`` must equal ``||Ax-b|| / ||b||``.

    The mesh is scaled by 1e3 so that ``||b||`` is far from 1: a *product*
    metric then differs from the ratio by many orders of magnitude, whereas on
    a unit-radius test body both are ``1e-16``-ish and indistinguishable.  The
    reference residual is rebuilt from ``assemble_surface_matrices`` (the
    operator definition) and the returned potentials -- never from the metric
    formula under test.
    """
    base = icosphere(1)
    scaled = TriangleMesh(base.vertices * 1.0e3, base.faces)
    result = compute_added_mass_matrix_3d(scaled, density=1.0)

    single, double = assemble_surface_matrices(scaled)
    panel_count = scaled.panel_count
    system = 0.5 * np.eye(panel_count) - double
    rhs = single @ (-result.mode_normals)
    residuals = np.linalg.norm(system @ result.mode_potentials - rhs, axis=0)
    rhs_norms = np.linalg.norm(rhs, axis=0)
    independent = float(np.max(np.where(rhs_norms > 0.0, residuals / rhs_norms, residuals)))

    # Non-vacuity: the rhs magnitude is nowhere near unity, so ``r / ||b||`` and
    # ``r * ||b||`` cannot coincide numerically.
    assert float(np.max(rhs_norms)) > 1.0e3
    assert independent < 1.0e-10
    assert result.max_relative_residual_l2 == pytest.approx(
        independent, rel=1.0e-6, abs=0.0
    )


def _dense_barycentric_samples(mesh: TriangleMesh, order: int = 48) -> np.ndarray:
    """Points of the surface on a per-triangle barycentric lattice."""
    u = np.linspace(0.0, 1.0, order)
    grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
    keep = (grid_u + grid_v) <= 1.0
    a = grid_u[keep][:, None]
    b = grid_v[keep][:, None]
    v0, v1, v2 = mesh.v0, mesh.v1, mesh.v2
    samples = (
        v0[:, None, :]
        + a[None, :, :] * (v1 - v0)[:, None, :]
        + b[None, :, :] * (v2 - v0)[:, None, :]
    )
    return samples.reshape(-1, 3)


def test_minimum_distance_to_mesh_matches_dense_triangle_sampling():
    """The point-to-surface distance must use the true foot point, not a vertex.

    A dense barycentric lattice of the same surface is an independent upper
    bound: its minimum distance can only exceed the true distance, and it
    converges to it.  The probe point sits radially outside an edge midpoint,
    where the nearest feature is an edge *interior*, so a wrong projection
    parameter is displaced by an order-one fraction of an edge length.
    """
    mesh = icosphere(1)
    edge_midpoint = 0.5 * (mesh.v0[0] + mesh.v1[0])
    point = 1.15 * edge_midpoint

    reported = _minimum_distance_to_mesh(mesh, point)
    samples = _dense_barycentric_samples(mesh, order=48)
    sampled = float(np.min(np.linalg.norm(samples - point, axis=1)))

    assert reported <= sampled + 1.0e-12, "reported distance exceeds a surface point"
    assert sampled - reported <= 5.0e-3, (
        f"reported {reported:.6e} vs sampled {sampled:.6e}: "
        "the distance is being measured to a vertex, not to the foot point"
    )
    assert reported > 0.0