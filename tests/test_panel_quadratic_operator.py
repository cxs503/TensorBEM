"""Operator-level verification of the quadratic (6-node) panel BEM operators.

The single-layer ``S`` and double-layer ``D`` operators of
:mod:`tensorbem.panel_quadratic` are tested against *closed-form* identities
rather than against the solver's own output:

* ``S 1 = 1`` on the unit sphere — the single-layer potential of unit density
  is exactly the sphere radius on and inside the surface.
* ``D 1 = -1/2`` — principal value of the double-layer kernel over a closed
  smooth surface (solid-angle identity); ``(1/2 I - D) 1 = 1`` follows.
* Polyhedral solid angles: on a flat closed surface the row sums of ``D`` are
  exactly ``-1/8``, ``-1/4`` and ``-1/2`` at cube vertices, edge midpoints and
  face-interior nodes.
* Flat-panel double-layer self block is identically zero because
  ``(x - y) . n = 0`` for a target inside the plane of a flat element.

These tests pin down the two conventions that are easy to get wrong
(``dG/dn_y = +n.(x-y)/(4*pi*r**3)`` and the ``x-y`` displacement direction) and
the isoparametric Jacobian of the edge-midpoint basis functions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.panel_quadratic import (
    NODES_PER_ELEMENT,
    NODE_BARYCENTRIC,
    QuadraticSurfaceMesh,
    QuadraticTriangle,
    assemble_quadratic_matrices,
    ellipsoid_quadratic,
    icosphere_quadratic,
    quadratic_shape_functions_array,
    quadratic_shape_gradients_ref,
    quadratic_shape_gradients_ref_array,
    solve_laplace_dirichlet_quadratic,
    solve_neumann_from_dirichlet_quadratic,
    triangle_apex_quadrature,
)

FOUR_PI = 4.0 * math.pi
SOURCE = np.array([0.0, 0.0, 0.5])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _flat_reference_triangle() -> QuadraticTriangle:
    """Unit right triangle with all six nodes, midpoints on the straight edges."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # v0
            [1.0, 0.0, 0.0],  # v1
            [0.0, 1.0, 0.0],  # v2
            [0.5, 0.0, 0.0],  # mid01
            [0.5, 0.5, 0.0],  # mid12
            [0.0, 0.5, 0.0],  # mid20
        ]
    )
    return QuadraticTriangle(nodes)


def _cube_mesh() -> QuadraticSurfaceMesh:
    """Closed flat quadratic mesh of the cube [-1, 1]^3 with outward normals."""
    verts = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )

    def vid(x, y, z):
        return int(np.flatnonzero((verts == [x, y, z]).all(axis=1))[0])

    quads = [
        ([-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1], (-1, 0, 0)),
        ([1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1], (1, 0, 0)),
        ([-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1], (0, -1, 0)),
        ([-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1], (0, 1, 0)),
        ([-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], (0, 0, -1)),
        ([-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1], (0, 0, 1)),
    ]
    faces = []
    for a, b, c, d, normal in quads:
        ia, ib, ic, id_ = (vid(*p) for p in (a, b, c, d))
        for tri in ((ia, ib, ic), (ia, ic, id_)):
            p0, p1, p2 = verts[list(tri)]
            if np.dot(np.cross(p1 - p0, p2 - p0), normal) < 0.0:
                tri = (tri[0], tri[2], tri[1])
            faces.append(tri)
    return QuadraticSurfaceMesh.from_flat_triangle_mesh(verts, np.array(faces))


@pytest.fixture(scope="module")
def sphere_meshes():
    return {sub: icosphere_quadratic(subdivisions=sub) for sub in (0, 1)}


@pytest.fixture(scope="module")
def sphere_operators(sphere_meshes):
    return {
        sub: assemble_quadratic_matrices(mesh) for sub, mesh in sphere_meshes.items()
    }


# ---------------------------------------------------------------------------
# Reference-triangle geometry (guards the isoparametric Jacobian)
# ---------------------------------------------------------------------------
def test_shape_gradients_sum_to_zero():
    """Partition of unity differentiated: sum_i dN_i/d(L1, L2) == 0."""
    rng = np.random.default_rng(20261109)
    L1 = rng.uniform(0.0, 1.0, 25)
    L2 = rng.uniform(0.0, 1.0, 25)
    keep = L1 + L2 <= 1.0
    for a, b in zip(L1[keep], L2[keep]):
        grads = quadratic_shape_gradients_ref(float(a), float(b))
        assert np.allclose(grads.sum(axis=0), 0.0, atol=1e-12)


def test_scalar_and_vectorised_gradients_agree():
    L = np.array([[0.15, 0.25], [0.4, 0.05], [0.1, 0.6]])
    batch = quadratic_shape_gradients_ref_array(L)
    for q in range(L.shape[0]):
        single = quadratic_shape_gradients_ref(float(L[q, 0]), float(L[q, 1]))
        assert np.allclose(batch[q], single, atol=1e-14)


def test_flat_panel_metric_is_unity():
    """A flat quadratic triangle must have det J == 1 and normal +z everywhere."""
    elem = _flat_reference_triangle()
    for L1, L2 in ((0.2, 0.3), (0.1, 0.1), (0.6, 0.2), (1.0 / 3.0, 1.0 / 3.0)):
        cross, det = elem.jacobian(L1, L2)
        assert det == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(cross / det, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_apex_quadrature_integrates_low_order_polynomials():
    """The singularity-cancelling rule must be exact for 1, L1 and L2."""
    for node in range(NODES_PER_ELEMENT):
        L, w = triangle_apex_quadrature(NODE_BARYCENTRIC[node], order=4)
        assert w.sum() == pytest.approx(0.5, abs=1e-14)  # reference-triangle area
        assert (w * L[:, 0]).sum() == pytest.approx(1.0 / 6.0, abs=1e-14)
        assert (w * L[:, 1]).sum() == pytest.approx(1.0 / 6.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Singular self-panel quadrature
# ---------------------------------------------------------------------------
def test_flat_panel_double_layer_self_block_vanishes():
    """For a flat element (x - y) . n == 0, so the D self block is exactly 0."""
    elem = _flat_reference_triangle()
    _S_self, D_self = elem.node_self_matrices(order=16)
    assert D_self.shape == (NODES_PER_ELEMENT, NODES_PER_ELEMENT)
    assert np.max(np.abs(D_self)) < 1e-14


def test_flat_panel_self_single_layer_diagonal_positive():
    """The self single-layer kernel is positive, so its diagonal entries are."""
    elem = _flat_reference_triangle()
    S_self, _D_self = elem.node_self_matrices(order=16)
    assert np.all(np.diag(S_self) > 0.0)
    # sum over the six shape functions is the exact integral of G > 0
    assert np.all(S_self.sum(axis=1) > 0.0)


def test_edge_midpoint_self_quadrature_converges():
    """Edge-node self integrals must converge with the Duffy order."""
    elem = _flat_reference_triangle()
    for node in range(NODES_PER_ELEMENT):
        ref = elem.integrate_single_layer_at_node(node, order=64)
        for order in (12, 20, 32):
            val = elem.integrate_single_layer_at_node(node, order=order)
            assert np.max(np.abs(val - ref)) / np.max(np.abs(ref)) < 1e-4


# ---------------------------------------------------------------------------
# Non-singular kernels versus an independent (very fine) quadrature
# ---------------------------------------------------------------------------
def test_off_panel_kernels_match_fine_quadrature():
    elem = _flat_reference_triangle()
    target = np.array([0.3, 0.3, 1.0])
    s8 = elem.integrate_single_layer(target, order=8)
    d8 = elem.integrate_double_layer(target, order=8)
    s_ref = elem.integrate_single_layer(target, order=60)
    d_ref = elem.integrate_double_layer(target, order=60)
    assert np.max(np.abs(s8 - s_ref)) / np.max(np.abs(s_ref)) < 1e-6
    assert np.max(np.abs(d8 - d_ref)) / np.max(np.abs(d_ref)) < 1e-6


def test_kernel_sign_convention_matches_analytic_point_source():
    """dG/dn_y is positive for a target inside the source half-space.

    With a flat panel in the z=0 plane (normal +z) and the collocation point at
    z=+1, ``(x - y) . n = +1`` for every y on the panel, so the double-layer
    kernel and therefore every integral entry is positive.
    """
    elem = _flat_reference_triangle()
    above = float(elem.integrate_double_layer(np.array([0.3, 0.3, 1.0]), order=16).sum())
    below = float(elem.integrate_double_layer(np.array([0.3, 0.3, -1.0]), order=16).sum())
    assert above > 0.0
    assert below == pytest.approx(-above, rel=1e-10)


# ---------------------------------------------------------------------------
# Closed-form identities on a closed surface
# ---------------------------------------------------------------------------
def test_polyhedral_solid_angles_on_cube():
    """Row sums of D on a flat closed surface equal -Omega_i / (4 pi) exactly."""
    mesh = _cube_mesh()
    _S, D = assemble_quadratic_matrices(mesh, quadrature_order=16, self_order=16)
    row_sums = D.sum(axis=1)
    # node classification by distance from the cube centre
    radius = np.linalg.norm(mesh.vertices, axis=1)
    n_vertex = np.count_nonzero(np.isclose(radius, np.sqrt(3.0)))
    n_edge = np.count_nonzero(np.isclose(radius, np.sqrt(2.0)))
    n_face = np.count_nonzero(np.isclose(radius, 1.0))
    assert n_vertex == 8 and n_edge == 12 and n_face == 6

    assert np.allclose(row_sums[np.isclose(radius, np.sqrt(3.0))], -1.0 / 8.0, atol=1e-10)
    assert np.allclose(row_sums[np.isclose(radius, np.sqrt(2.0))], -1.0 / 4.0, atol=1e-10)
    assert np.allclose(row_sums[np.isclose(radius, 1.0)], -1.0 / 2.0, atol=1e-10)


def test_sphere_solid_angle_identity_and_convergence(sphere_operators):
    """|D 1 + 1/2| must be small and halve faster than h on refinement."""
    errors = {}
    for sub, (S, D) in sphere_operators.items():
        ones = np.ones(S.shape[0])
        errors[sub] = float(np.max(np.abs(D @ ones + 0.5)))
    assert errors[0] < 5e-2
    assert errors[1] < 1e-2
    assert errors[0] / errors[1] > 4.0  # ~O(h^3): h ratio is 1.81, so >= ~3


def test_ellipsoid_solid_angle_identity():
    """|D 1 + 1/2| must stay small on a non-spherical closed surface.

    The solid-angle identity ``D 1 = -1/2`` holds for *any* closed smooth
    surface, not only the sphere.  The ellipsoid here is the affine image of
    the icosphere (a genuinely curved, non-spherical isoparametric panel set),
    so this pins the isoparametric Jacobian and the singular self-panel
    quadrature off-sphere, where a sign or chain-rule slip would not be masked
    by spherical symmetry.
    """
    for sub, tol in ((0, 1e-1), (1, 2e-2)):
        mesh = ellipsoid_quadratic(sub, axes=(1.0, 1.6, 0.7))
        _S, D = assemble_quadratic_matrices(mesh)
        ones = np.ones(D.shape[0])
        err = float(np.max(np.abs(D @ ones + 0.5)))
        assert err < tol, f"sub {sub}: |D 1 + 1/2|_inf = {err} >= {tol}"

def test_operators_are_translation_invariant():
    """Rigid translation leaves S and D unchanged (kernel sees x - y only).

    The Laplace kernel ``G(x, y) = 1/(4 pi |x - y|)`` and the isoparametric
    normals depend only on *relative* positions, so translating the whole
    mesh must leave every entry of ``S`` and ``D`` unchanged up to round-off.
    A position-dependent slip in the kernel (e.g. an absolute coordinate
    sneaking into the displacement or the normal) can survive the row-sum
    identities yet break this entry-wise check.
    """
    axes = (1.0, 1.6, 0.7)
    shift = np.array([3.0, -2.0, 1.5])
    base = ellipsoid_quadratic(1, axes=axes)
    moved = QuadraticSurfaceMesh(base.vertices + shift, base.faces.copy())
    S0, D0 = assemble_quadratic_matrices(base)
    S1, D1 = assemble_quadratic_matrices(moved)
    assert np.max(np.abs(S1 - S0)) <= 1e-10 * np.max(np.abs(S0)), (
        "single-layer matrix changed under translation"
    )
    assert np.max(np.abs(D1 - D0)) <= 1e-10 * np.max(np.abs(D0)), (
        "double-layer matrix changed under translation"
    )
    # the solid-angle identity must hold at the new location as well
    ones = np.ones(D1.shape[0])
    assert float(np.max(np.abs(D1 @ ones + 0.5))) < 2e-2


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Proper rotation (Rodrigues) about a unit axis."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def test_operators_are_rotation_invariant():
    """Rigid rotation leaves S and D unchanged (kernel is SE(3)-invariant).

    The Laplace kernel ``G(x, y) = 1/(4 pi |x - y|)`` satisfies
    ``G(R x, R y) = G(x, y)`` for every proper rotation ``R``, and the
    isoparametric normals rotate covariantly (``n -> R n``), so rotating the
    whole mesh must leave every entry of ``S`` and ``D`` unchanged up to
    round-off.  This completes the rigid group: the translation pin
    (``test_operators_are_translation_invariant``) cannot catch a slip that
    references a fixed global axis (e.g. a normal or a Duffy sub-coordinate
    built from absolute coordinates), while this entry-wise check does.
    """
    R = _rotation_matrix(np.array([1.0, 2.0, 3.0]), 0.7)
    assert abs(float(np.linalg.det(R)) - 1.0) < 1e-14
    base = ellipsoid_quadratic(1, axes=(1.0, 1.6, 0.7))
    rotated = QuadraticSurfaceMesh((R @ base.vertices.T).T, base.faces.copy())
    S0, D0 = assemble_quadratic_matrices(base)
    S1, D1 = assemble_quadratic_matrices(rotated)
    assert np.max(np.abs(S1 - S0)) <= 1e-10 * np.max(np.abs(S0)), (
        "single-layer matrix changed under rotation"
    )
    assert np.max(np.abs(D1 - D0)) <= 1e-10 * np.max(np.abs(D0)), (
        "double-layer matrix changed under rotation"
    )
    # the solid-angle identity must hold on the rotated surface as well
    ones = np.ones(D1.shape[0])
    assert float(np.max(np.abs(D1 @ ones + 0.5))) < 2e-2



def test_operators_are_scale_invariant():
    """Uniform scaling: S scales as 1/s, D is invariant (kernel is scale-covariant).

    The Laplace kernel G(x, y) = 1/(4 pi |x - y|) satisfies
    G(s x, s y) = G(x, y) / s for every uniform scale s (the
    single-layer potential of a unit density on a surface of radius R is
    R, so S must scale as 1/s), while the double-layer operator
    D is *scale-invariant* because it depends only on the solid angle
    (D 1 = -1/2 on any closed smooth surface, independent of size).
    A slip that hard-codes the 1/|x-y| denominator or that fails to
    rescale the quadrature Jacobian under a uniform dilation breaks this
    entry-wise check even though the rigid-body pins (translation + rotation)
    pass.
    """
    scale = 2.7
    base = ellipsoid_quadratic(1, axes=(1.0, 1.6, 0.7))
    scaled = ellipsoid_quadratic(1, axes=(1.0 * scale, 1.6 * scale, 0.7 * scale))
    S0, D0 = assemble_quadratic_matrices(base)
    S1, D1 = assemble_quadratic_matrices(scaled)
    # S scales as 1/scale (kernel G(x, y) = 1/(4 pi |x - y|))
    assert np.max(np.abs(S1 / scale - S0)) <= 1e-10 * np.max(np.abs(S0)), (
        "single-layer matrix does not scale as 1/scale"
    )
    # D is scale-invariant (solid angle is size-independent)
    assert np.max(np.abs(D1 - D0)) <= 1e-10 * np.max(np.abs(D0)), (
        "double-layer matrix changed under uniform scaling"
    )
    # the solid-angle identity must hold on the scaled surface as well
    ones = np.ones(D1.shape[0])
    assert float(np.max(np.abs(D1 @ ones + 0.5))) < 2e-2


def test_operators_are_reflection_invariant():
    """Mirror reflection leaves S and D unchanged (kernel is O(3)-invariant).

    The Laplace kernel G(x, y) = 1/(4 pi |x - y|) satisfies
    G(M x, M y) = G(x, y) for every orthogonal M (det M = +/- 1), and the
    outward normal of a reflected closed surface is M n, so reflecting the
    whole mesh must leave every entry of S and D unchanged up to round-off.
    This completes the full orthogonal group: the rigid-body pins
    (translation + rotation) and the scale pin all use *proper* transforms
    (det = +1) and cannot catch a handedness slip -- e.g. a normal built
    from a cross product with an assumed right-handed parameterisation --
    which a reflection (det = -1) exposes as D -> -D while S stays put.
    """
    M = np.diag([1.0, -1.0, 1.0])
    assert abs(float(np.linalg.det(M)) + 1.0) < 1e-14
    base = ellipsoid_quadratic(1, axes=(1.0, 1.6, 0.7))
    reflected = QuadraticSurfaceMesh((M @ base.vertices.T).T, base.faces.copy())
    S0, D0 = assemble_quadratic_matrices(base)
    S1, D1 = assemble_quadratic_matrices(reflected)
    assert np.max(np.abs(S1 - S0)) <= 1e-10 * np.max(np.abs(S0)), (
        "single-layer matrix changed under reflection"
    )
    assert np.max(np.abs(D1 - D0)) <= 1e-10 * np.max(np.abs(D0)), (
        "double-layer matrix changed under reflection"
    )
    # the solid-angle identity must hold on the reflected surface as well
    ones = np.ones(D1.shape[0])
    assert float(np.max(np.abs(D1 @ ones + 0.5))) < 2e-2


def test_inward_wound_mesh_is_rewound_outward():
    """A mesh wound entirely inward is re-wound to the outward convention.

    White-box companion of the reflection pin: reversing every face of a
    valid mesh must be undone by the constructor, so the stored faces equal
    the original outward winding exactly.  Without the re-wind, the
    reflected mesh would assemble D with the opposite sign (D 1 = +1/2)
    and the recovered flux would be silently wrong.
    """
    base = ellipsoid_quadratic(1, axes=(1.0, 1.6, 0.7))
    inward = QuadraticSurfaceMesh(
        base.vertices.copy(), base.faces[:, [0, 2, 1, 5, 4, 3]]
    )
    assert np.array_equal(inward.faces, base.faces), (
        "inward-wound faces were not re-wound to the outward convention"
    )


def test_single_layer_constant_density_identity(sphere_operators):
    """S 1 == 1 on the unit sphere (single-layer potential of unit density)."""
    for sub, (S, D) in sphere_operators.items():
        ones = np.ones(S.shape[0])
        assert float(np.max(np.abs(S @ ones - 1.0))) < 2e-2
        assert np.linalg.norm(D) > 0.0  # double layer is actually assembled


def test_second_kind_identity(sphere_operators):
    """(1/2 I - D) 1 == 1 for the exterior second-kind operator."""
    for sub, (S, D) in sphere_operators.items():
        ones = np.ones(S.shape[0])
        second_kind = 0.5 * np.eye(S.shape[0]) - D
        assert float(np.max(np.abs(second_kind @ ones - 1.0))) < 5e-2


def test_far_field_identities(sphere_meshes):
    """D 1 == -1 inside and 0 outside; S 1 == 1 inside the unit sphere."""
    mesh = sphere_meshes[0]
    for target, expected_d, expected_s in (
        (np.array([0.0, 0.0, 0.0]), -1.0, 1.0),
        (np.array([0.0, 0.0, 3.0]), 0.0, 1.0 / 3.0),
    ):
        d_total = 0.0
        s_total = 0.0
        for e in range(mesh.element_count):
            elem = mesh.element(e)
            s_total += float(np.sum(elem.integrate_single_layer(target, order=24)))
            d_total += float(np.sum(elem.integrate_double_layer(target, order=24)))
        assert d_total == pytest.approx(expected_d, abs=2e-2)
        assert s_total == pytest.approx(expected_s, rel=2e-2)


# ---------------------------------------------------------------------------
# Assembly consistency
# ---------------------------------------------------------------------------
def test_assembly_matches_direct_row_summation(sphere_meshes, sphere_operators):
    """Each assembled row equals the sum over elements of its kernel block.

    Catches double counting of the singular self panel and mis-indexed column
    scatter.
    """
    mesh = sphere_meshes[0]
    S, D = sphere_operators[0]
    for node in (0, 7, 20):
        row_s = np.zeros(mesh.node_count)
        row_d = np.zeros(mesh.node_count)
        target = mesh.vertices[node]
        for e in range(mesh.element_count):
            elem = mesh.element(e)
            ids = mesh.faces[e]
            if node in ids:
                local = int(np.flatnonzero(ids == node)[0])
                row_s[ids] += elem.integrate_single_layer_at_node(local, order=40)
                row_d[ids] += elem.integrate_double_layer_at_node(local, order=40)
            else:
                row_s[ids] += elem.integrate_single_layer(target, order=40)
                row_d[ids] += elem.integrate_double_layer(target, order=40)
        assert np.linalg.norm(S[node] - row_s) / np.linalg.norm(row_s) < 1e-6
        assert np.linalg.norm(D[node] - row_d) / np.linalg.norm(row_d) < 1e-6


# ---------------------------------------------------------------------------
# Physical acceptance: direct second-kind flux recovery
# ---------------------------------------------------------------------------
def _exact_flux_at_centroids(mesh):
    shape = quadratic_shape_functions_array(np.array([[1.0 / 3.0, 1.0 / 3.0]]))[0]
    flux = np.empty(mesh.element_count)
    for e in range(mesh.element_count):
        elem = mesh.element(e)
        _X, _N, _det, nrm = elem._geometry_at(np.array([[1.0 / 3.0, 1.0 / 3.0]]))
        centre = shape @ mesh.vertices[mesh.faces[e]]
        delta = centre - SOURCE
        r = np.linalg.norm(delta)
        grad = -delta / (FOUR_PI * r ** 3)
        flux[e] = grad @ nrm[0]
    return flux


def _boundary_values(mesh):
    r = np.linalg.norm(mesh.vertices - SOURCE, axis=1)
    return 1.0 / (FOUR_PI * r)


def test_direct_neumann_flux_within_5_percent(sphere_meshes):
    """Direct second-kind BIE reproduces the closed-form flux (T=5% at 162 nodes)."""
    mesh = sphere_meshes[1]
    u = _boundary_values(mesh)
    result = solve_neumann_from_dirichlet_quadratic(mesh, u)
    assert result.relative_residual_l2 < 1e-10
    shape = quadratic_shape_functions_array(np.array([[1.0 / 3.0, 1.0 / 3.0]]))[0]
    q_nodal_to_elem = np.array(
        [shape @ result.normal_derivative[mesh.faces[e]] for e in range(mesh.element_count)]
    )
    q_exact = _exact_flux_at_centroids(mesh)
    rel = np.linalg.norm(q_nodal_to_elem - q_exact) / np.linalg.norm(q_exact)
    assert rel < 0.05, f"flux relative error {rel:.4e} exceeds 5%"


def test_direct_neumann_flux_converges_faster_than_h2(sphere_meshes):
    """Observed flux convergence order must be >= 2 between the two meshes."""
    errors = {}
    for sub, mesh in sphere_meshes.items():
        u = _boundary_values(mesh)
        result = solve_neumann_from_dirichlet_quadratic(mesh, u)
        shape = quadratic_shape_functions_array(np.array([[1.0 / 3.0, 1.0 / 3.0]]))[0]
        q = np.array(
            [
                shape @ result.normal_derivative[mesh.faces[e]]
                for e in range(mesh.element_count)
            ]
        )
        q_exact = _exact_flux_at_centroids(mesh)
        errors[sub] = np.linalg.norm(q - q_exact) / np.linalg.norm(q_exact)

    h = {
        sub: math.sqrt(
            np.mean([mesh.element(e).vertex_area for e in range(mesh.element_count)])
        )
        for sub, mesh in sphere_meshes.items()
    }
    order = math.log(errors[0] / errors[1]) / math.log(h[0] / h[1])
    assert order > 2.0, f"observed flux order {order:.2f} < 2"


def test_equivalent_source_density_is_not_the_flux(sphere_meshes):
    """The indirect single-layer density sigma is NOT du/dn (documented pitfall).

    ``S sigma = u`` yields the *equivalent source density*; the physical flux
    requires the second-kind equation.  The two differ at O(1) level, so this
    test keeps the distinction visible in the code base.
    """
    mesh = sphere_meshes[1]
    u = _boundary_values(mesh)
    indirect = solve_laplace_dirichlet_quadratic(mesh, u)
    direct = solve_neumann_from_dirichlet_quadratic(mesh, u)
    sigma = indirect.nodal_values
    q = direct.normal_derivative
    diff = np.linalg.norm(sigma - q) / np.linalg.norm(q)
    assert diff > 0.1, f"unexpected: sigma and q agree to {diff:.3e}"

# ---------------------------------------------------------------------------
# Geometry-generic verification (non-spherical surface, similarity invariance)
# ---------------------------------------------------------------------------
def test_ellipsoid_quadratic_nodes_lie_on_the_surface():
    """Every node of the ellipsoid mesh satisfies |diag(axes)^-1 (x - c)| = 1."""
    axes = np.array([1.0, 1.6, 0.7])
    centre = np.array([0.3, -1.0, 2.0])
    mesh = ellipsoid_quadratic(1, axes=axes, centre=centre)
    scaled = (mesh.vertices - centre) / axes
    radial = np.linalg.norm(scaled, axis=1)
    assert np.max(np.abs(radial - 1.0)) < 1e-12, "nodes are not on the ellipsoid"
    # Closed surface: every edge of every face is shared by exactly two faces,
    # every vertex node is used, and the node count is V + E (vertices + edge
    # midpoints) as required by the 6-node element topology.
    edges: dict[tuple[int, int], int] = {}
    vertex_nodes: set[int] = set()
    for face in mesh.faces:
        vertex_nodes.update(int(v) for v in face[:3])  # columns 0-2 are the vertices
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edges[key] = edges.get(key, 0) + 1
    assert set(edges.values()) == {2}, "quadratic mesh is not watertight"
    assert mesh.node_count == len(vertex_nodes) + len(edges)
    # Euler characteristic of a sphere: V - E + F = 2.
    assert len(vertex_nodes) - len(edges) + mesh.element_count == 2


def test_ellipsoid_quadratic_rejects_bad_axes():
    with pytest.raises(ValueError):
        ellipsoid_quadratic(0, axes=(1.0, 0.0, 1.0))


def test_flux_convergence_on_ellipsoid():
    """The O(h**2) flux recovery survives on a surface with varying curvature.

    The point-source study lives on a sphere, where every curvature direction is
    degenerate; this test repeats the recovery on an ellipsoid whose three
    semi-axes differ, so accuracy cannot come from spherical symmetry.
    """
    from tensorbem.benchmarks.quadratic_panel_verification import (
        GEOMETRY_ORDER_FLOOR,
        FLUX_ACCEPTANCE,
        loglog_slope,
        run_flux_study,
    )

    errors, hs = [], []
    for sub in (0, 1, 2):
        mesh = ellipsoid_quadratic(sub, axes=(1.0, 1.6, 0.7))
        record = run_flux_study(mesh, source=np.zeros(3))
        assert record["relative_residual_l2"] < 1e-10
        errors.append(record["flux_rel_error"])
        hs.append(record["h"])
    order = loglog_slope(np.array(hs), np.array(errors))
    assert order >= GEOMETRY_ORDER_FLOOR, f"ellipsoid flux order {order:.2f} too low"
    assert errors[-1] <= FLUX_ACCEPTANCE, f"ellipsoid flux error {errors[-1]:.3e} > 3%"


def test_similarity_invariance_of_flux_error():
    """Uniform scaling + translation must not change the *relative* flux error."""
    from tensorbem.benchmarks.quadratic_panel_verification import run_similarity_study

    result = run_similarity_study(level=1)
    assert result["relative_deviation"] < 1e-10, (
        f"similarity deviation {result['relative_deviation']:.3e} "
        "-> the operator has a hidden absolute length scale"
    )


def test_committed_study_json_matches_its_artifacts():
    """The committed study record must be self-consistent and hash-verified.

    Guards against hand-edited numbers: the acceptance flags must follow from the
    level table, and the provenance hashes must match the committed figures.
    """
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    doc_dir = root / "docs" / "quadratic_panel_verification"
    study = json.loads((doc_dir / "study.json").read_text())

    assert study["schema"] == "tensorbem.quadratic_panel_verification/v2"
    levels = study["levels"]
    acceptance = study["acceptance"]
    assert acceptance["flux_rel_error_finest"] == levels[-1]["flux_rel_error"]
    assert acceptance["passed_3pct"] == (
        levels[-1]["flux_rel_error"] <= acceptance["threshold"]
    )
    assert acceptance["passed"] == (
        acceptance["passed_3pct"]
        and acceptance["geometry_passed"]
        and acceptance["similarity_passed"]
    )

    geometry = study["geometry"]
    assert len(geometry["geometries"]) >= 2
    names = {g["name"] for g in geometry["geometries"]}
    assert "sphere" in names and any(n.startswith("ellipsoid") for n in names)
    for geom in geometry["geometries"]:
        finest = geom["levels"][-1]["flux_rel_error"]
        assert finest == geom["flux_rel_error_finest"]
        assert geom["observed_order"] == pytest.approx(geom["observed_order"])
        assert geom["acceptance"]["passed"] == (
            finest <= geom["acceptance"]["threshold"]
            and geom["observed_order"] >= geom["acceptance"]["order_floor"]
        )
    assert study["similarity"]["passed"] is True
    assert study["similarity"]["relative_deviation"] <= study["similarity"]["tolerance"]

    for entry in study["provenance"].values():
        path = root / entry["path"]
        assert path.exists(), f"missing artifact {entry['path']}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"sha256 mismatch for {entry['path']}"


def test_benchmark_gate_exits_nonzero_on_failure(tmp_path):
    """The benchmark is a *gate*: a failing acceptance must not exit 0."""
    from tensorbem.benchmarks.quadratic_panel_verification import main

    out_json = tmp_path / "study.json"
    argv = [
        "--levels",
        "0",
        "--json",
        str(out_json),
        "--figure",
        "",
        "--geometry-figure",
        "",
        "--skip-geometry",
    ]
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert "acceptance gate failed" in str(excinfo.value.code)
    assert out_json.exists()


def test_benchmark_gate_can_report_without_failing(tmp_path):
    """``--allow-fail`` records the failing flags but returns normally."""
    from tensorbem.benchmarks.quadratic_panel_verification import main

    study = main(
        [
            "--levels",
            "0",
            "--json",
            str(tmp_path / "study.json"),
            "--figure",
            "",
            "--geometry-figure",
            "",
            "--skip-geometry",
            "--allow-fail",
        ]
    )
    assert study["acceptance"]["passed_3pct"] is False
    assert study["acceptance"]["passed"] is False
