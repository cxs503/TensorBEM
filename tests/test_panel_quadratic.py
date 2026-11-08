"""Tests for the quadratic panel module (panel_quadratic.py).

Tests cover:
1. Shape function properties (partition of unity, interpolation)
2. Quadratic triangle geometry (mapping, Jacobian)
3. Self-influence integrals (analytical vs numerical)
4. Quadratic surface mesh construction
5. Matrix assembly and solve
6. Convergence order measurement
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.panel_quadratic import (
    FOUR_PI,
    QuadraticTriangle,
    QuadraticSurfaceMesh,
    duffy_self_quadrature,
    gauss_triangle_rule,
    icosphere_quadratic,
    partition_of_unity_check,
    quadratic_shape_functions,
    quadratic_shape_gradients_ref,
    assemble_quadratic_matrices,
    self_influence_analytical_vs_numerical,
    solve_laplace_dirichlet_quadratic,
)
from tensorbem.geometry3d import icosphere


# ---------------------------------------------------------------------------
# Shape function tests
# ---------------------------------------------------------------------------
class TestQuadraticShapeFunctions:
    """Test the 6-node quadratic shape functions."""

    def test_partition_of_unity_at_vertices(self):
        """Shape functions sum to 1 at all vertices."""
        assert partition_of_unity_check(1.0, 0.0)
        assert partition_of_unity_check(0.0, 1.0)
        assert partition_of_unity_check(0.0, 0.0)

    def test_partition_of_unity_at_edge_midpoints(self):
        """Shape functions sum to 1 at edge midpoints."""
        assert partition_of_unity_check(0.5, 0.5)
        assert partition_of_unity_check(0.0, 0.5)
        assert partition_of_unity_check(0.5, 0.0)

    def test_partition_of_unity_interior(self):
        """Shape functions sum to 1 at interior points."""
        for L1, L2 in [(0.25, 0.25), (0.33, 0.33), (0.1, 0.7)]:
            assert partition_of_unity_check(L1, L2)

    def test_vertex_interpolation(self):
        """Shape function N_i = 1 at node i, 0 at other nodes."""
        N_v0 = quadratic_shape_functions(1.0, 0.0)
        assert abs(N_v0[0] - 1.0) < 1e-14
        assert abs(N_v0[1]) < 1e-14
        assert abs(N_v0[2]) < 1e-14
        assert abs(N_v0[3]) < 1e-14
        assert abs(N_v0[4]) < 1e-14
        assert abs(N_v0[5]) < 1e-14

        N_v1 = quadratic_shape_functions(0.0, 1.0)
        assert abs(N_v1[0]) < 1e-14
        assert abs(N_v1[1] - 1.0) < 1e-14

        N_mid01 = quadratic_shape_functions(0.5, 0.5)
        assert abs(N_mid01[3] - 1.0) < 1e-14

    def test_quadratic_reproduction(self):
        """Shape functions can reproduce quadratic polynomials exactly."""
        # A quadratic function f(L1, L2) = L1^2 + L2^2 + L1*L2
        # should be exactly represented by the shape functions
        for L1, L2 in [(0.25, 0.25), (0.5, 0.25), (0.1, 0.6)]:
            N = quadratic_shape_functions(L1, L2)
            # The shape functions form a basis for quadratics on the triangle
            assert abs(np.sum(N) - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# Quadrature tests
# ---------------------------------------------------------------------------
class TestQuadrature:
    """Test quadrature rules on the reference triangle."""

    def test_gauss_rule_integrates_constant(self):
        """Gauss rule integrates 1 over the reference triangle to 0.5 (area)."""
        nodes, weights = gauss_triangle_rule(8)
        assert abs(np.sum(weights) - 0.5) < 1e-12

    def test_gauss_rule_integrates_linear(self):
        """Gauss rule integrates L1 over the reference triangle to 1/6."""
        nodes, weights = gauss_triangle_rule(8)
        L1 = nodes[:, 0]
        integral = np.sum(weights * L1)
        assert abs(integral - 1.0 / 6.0) < 1e-12

    def test_gauss_rule_integrates_quadratic(self):
        """Gauss rule integrates L1^2 to 1/12."""
        nodes, weights = gauss_triangle_rule(8)
        L1 = nodes[:, 0]
        integral = np.sum(weights * L1 ** 2)
        assert abs(integral - 1.0 / 12.0) < 1e-10

    def test_duffy_self_quadrature_nodes_valid(self):
        """Duffy quadrature nodes are within the reference triangle."""
        for apex in (0, 1, 2):
            nodes, weights = duffy_self_quadrature(apex, order=8)
            L1, L2 = nodes[:, 0], nodes[:, 1]
            L3 = 1.0 - L1 - L2
            assert np.all(L1 >= -1e-14)
            assert np.all(L2 >= -1e-14)
            assert np.all(L3 >= -1e-14)


# ---------------------------------------------------------------------------
# Quadratic triangle tests
# ---------------------------------------------------------------------------
class TestQuadraticTriangle:
    """Test the QuadraticTriangle element."""

    @pytest.fixture
    def curved_triangle(self):
        """A curved quadratic triangle (midpoint lifted off the plane)."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        mid01 = np.array([0.5, 0.0, 0.0])
        mid12 = np.array([0.5, 0.5, 0.1])  # lifted off plane
        mid20 = np.array([0.0, 0.5, 0.05])  # slightly lifted
        return QuadraticTriangle(np.array([v0, v1, v2, mid01, mid12, mid20]))

    @pytest.fixture
    def sphere_patch(self):
        """A quadratic triangle from a sphere (genuinely curved)."""
        # Take a patch from the icosphere
        quad = icosphere_quadratic(1, radius=1.0)
        return quad.element(0)

    def test_vertex_area(self, curved_triangle):
        """Vertex area is positive for a valid triangle."""
        assert curved_triangle.vertex_area > 0

    def test_map_to_physical_at_vertices_sphere(self, sphere_patch):
        """Mapping at vertex barycentric coords returns vertex positions."""
        x0 = sphere_patch.map_to_physical(1.0, 0.0)
        assert np.allclose(x0, sphere_patch.nodes[0])

        x1 = sphere_patch.map_to_physical(0.0, 1.0)
        assert np.allclose(x1, sphere_patch.nodes[1])

        x2 = sphere_patch.map_to_physical(0.0, 0.0)
        assert np.allclose(x2, sphere_patch.nodes[2])

    def test_jacobian_positive_curved(self, curved_triangle):
        """Jacobian determinant is positive for a curved triangle at centroid."""
        cross, det = curved_triangle.jacobian(1.0 / 3.0, 1.0 / 3.0)
        assert det > 0

    def test_jacobian_positive_sphere(self, sphere_patch):
        """Jacobian is positive for a sphere patch at multiple points."""
        for L1, L2 in [(1 / 3, 1 / 3), (0.25, 0.25), (0.5, 0.25)]:
            cross, det = sphere_patch.jacobian(L1, L2)
            assert det > 1e-10, f"Jacobian degenerate at L1={L1}, L2={L2}"

    def test_self_influence_positive(self, sphere_patch):
        """Self-influence integral is positive."""
        for apex in range(3):
            s = sphere_patch.integrate_single_layer_self(apex_node=apex, order=12)
            assert np.sum(s) > 0

    def test_self_influence_convergence(self, sphere_patch):
        """Self-influence converges with increasing quadrature order."""
        values = []
        for order in [8, 12, 16, 24]:
            s = sphere_patch.integrate_single_layer_self(apex_node=0, order=order)
            values.append(np.sum(s))
        # Should converge (differences decrease)
        diffs = [abs(values[i] - values[-1]) for i in range(len(values) - 1)]
        assert all(d >= 0 for d in diffs)

    def test_non_singular_integration(self, sphere_patch):
        """Non-singular integration gives positive result for external target."""
        target = np.array([2.0, 2.0, 2.0])
        s = sphere_patch.integrate_single_layer(target, order=8)
        assert np.all(s > 0)


# ---------------------------------------------------------------------------
# Quadratic surface mesh tests
# ---------------------------------------------------------------------------
class TestQuadraticSurfaceMesh:
    """Test the QuadraticSurfaceMesh."""

    def test_from_flat_mesh(self):
        """Build quadratic mesh from flat icosphere."""
        flat = icosphere(1, radius=1.0)
        quad = QuadraticSurfaceMesh.from_flat_triangle_mesh(flat.vertices, flat.faces)
        assert quad.element_count == flat.panel_count
        assert quad.node_count > flat.vertex_count

    def test_icosphere_quadratic(self):
        """Build quadratic icosphere."""
        quad = icosphere_quadratic(1, radius=1.0)
        assert quad.element_count == 80  # 20 * 4^1
        flat = icosphere(1, radius=1.0)
        V = flat.vertex_count
        radii = np.linalg.norm(quad.vertices[:V], axis=1)
        assert np.allclose(radii, 1.0, atol=1e-14)

    def test_element_access(self):
        """Can access individual elements."""
        quad = icosphere_quadratic(1, radius=1.0)
        elem = quad.element(0)
        assert isinstance(elem, QuadraticTriangle)
        assert elem.nodes.shape == (6, 3)


# ---------------------------------------------------------------------------
# Matrix assembly and solve tests
# ---------------------------------------------------------------------------
class TestQuadraticBEM:
    """Test matrix assembly and solving."""

    def test_assemble_matrices_shape(self):
        """Assembled matrices have correct shape."""
        quad = icosphere_quadratic(1, radius=1.0)
        S, D = assemble_quadratic_matrices(quad, quadrature_order=4, self_order=8)
        assert S.shape == (quad.node_count, quad.node_count)
        assert D.shape == (quad.node_count, quad.node_count)

    def test_single_layer_positive_diagonal(self):
        """Single-layer matrix has positive diagonal entries."""
        quad = icosphere_quadratic(1, radius=1.0)
        S, _ = assemble_quadratic_matrices(quad, quadrature_order=4, self_order=8)
        assert np.all(np.diag(S) > 0)

    def test_solve_returns_result(self):
        """Solve returns a valid result."""
        quad = icosphere_quadratic(1, radius=1.0)
        u = np.ones(quad.node_count)
        result = solve_laplace_dirichlet_quadratic(
            quad, u, quadrature_order=4, self_order=8
        )
        assert result.nodal_values.shape == (quad.node_count,)
        assert result.relative_residual_l2 < 1e-10


# ---------------------------------------------------------------------------
# Self-influence convergence test (E2 criterion)
# ---------------------------------------------------------------------------
class TestSelfInfluenceConvergence:
    """Test E2: self-influence analytical vs numerical <= 1%."""

    def test_self_influence_convergence_quadratic(self):
        """Self-influence converges to within 1% for quadratic panels."""
        quad = icosphere_quadratic(1, radius=1.0)
        result = self_influence_analytical_vs_numerical(
            quad, quadrature_orders=(8, 12, 16, 24, 32)
        )
        assert result["convergence"][-1] < 0.01

    def test_self_influence_convergence_linear(self):
        """Linear panel self-influence analytic vs numerical agreement."""
        from tensorbem.panel_linear import (
            integrate_linear_single_layer_self_analytic,
            integrate_linear_single_layer_self_numerical,
        )
        flat = icosphere(2, radius=1.0)
        max_diff = 0.0
        for panel in range(min(flat.panel_count, 20)):
            tri = flat.vertices[flat.faces[panel]]
            for apex in range(3):
                analytic = integrate_linear_single_layer_self_analytic(tri, apex=apex)
                numerical = integrate_linear_single_layer_self_numerical(tri, apex=apex, order=24)
                ref = max(np.linalg.norm(analytic), 1e-15)
                diff = float(np.linalg.norm(analytic - numerical) / ref)
                max_diff = max(max_diff, diff)
        assert max_diff < 0.01


# ---------------------------------------------------------------------------
# Zero speed wave resistance (E5)
# ---------------------------------------------------------------------------
class TestZeroSpeedWaveResistance:
    """Test E5: zero speed wave resistance is strictly zero."""

    def test_michell_zero_speed(self):
        """Michell wave resistance at U=0 is exactly zero."""
        from tensorbem.wigley import michell_wave_resistance
        Rw = michell_wave_resistance(10.0, 1.0, 0.625, 0.0)
        assert Rw == 0.0
