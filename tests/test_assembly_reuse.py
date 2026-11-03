"""Assembly reuse in the 3-D solver, and the cost model it unlocks.

The dense surface matrices of the constant-panel 3-D solver depend on the mesh
and the quadrature order only, never on the boundary data. Before 2026-11-03
the translating-ellipsoid benchmark nevertheless assembled them six times per
run (once per translation axis for the solve, and again for each axis inside
the condition-number diagnostic), and rebuilt the Gauss-Legendre nodes of the
self-panel quadrature once per panel. Together the two redundancies made the
1280-panel verification case cost tens of minutes, so those declared numbers
were parked in a ``heavy`` tier that no test run reached.

These tests pin the three claims the fix rests on:

* the memoised nodes are the same numbers ``leggauss`` returns;
* a reused assembly gives bitwise the same results as a fresh one, and a
  mismatched assembly is rejected rather than silently used;
* the ``heavy`` declared numbers of the ellipsoid now reproduce inside a
  budget a single core can pay -- gated, because it is still ~35 s of solves.
"""

import os

import numpy as np
import pytest

from tensorbem.benchmarks import ellipsoid_added_mass
from tensorbem.benchmarks.declared_numbers import check
from tensorbem.geometry3d import ellipsoid_mesh
from tensorbem.kernels3d import gauss_legendre_nodes
from tensorbem.solver3d import (
    AssembledSurface,
    assembled_surface,
    solve_laplace_surface,
)

HEAVY = pytest.mark.skipif(
    not os.environ.get("TENSORBEM_HEAVY_TESTS"),
    reason="set TENSORBEM_HEAVY_TESTS=1 to run the 1280-panel verification",
)

AXES = (1.5, 1.0, 0.75)


def _mesh():
    return ellipsoid_mesh(1, AXES)


# --------------------------------------------------------------------------
# Memoised Gauss-Legendre nodes
# --------------------------------------------------------------------------


def test_cached_nodes_match_leggauss():
    from numpy.polynomial.legendre import leggauss

    for order in (2, 6, 8, 64):
        nodes, weights = gauss_legendre_nodes(order)
        reference_nodes, reference_weights = leggauss(order)
        assert np.array_equal(nodes, reference_nodes)
        assert np.array_equal(weights, reference_weights)


def test_cached_nodes_are_reused_and_read_only():
    first = gauss_legendre_nodes(8)
    second = gauss_legendre_nodes(8)
    assert first[0] is second[0], "the node table is rebuilt on every call"
    assert first[1] is second[1], "the weight table is rebuilt on every call"
    assert not first[0].flags.writeable
    assert not first[1].flags.writeable
    with pytest.raises(ValueError):
        first[0][0] = 0.0


# --------------------------------------------------------------------------
# Assembled matrices are reusable, and tied to their mesh
# --------------------------------------------------------------------------


def test_reused_assembly_is_bitwise_identical_to_a_fresh_one():
    mesh = _mesh()
    boundary = mesh.centroids[:, 0].copy()
    reuse = assembled_surface(mesh)
    fresh = solve_laplace_surface(mesh, boundary, domain="exterior")
    reused = solve_laplace_surface(
        mesh, boundary, domain="exterior", assembled=reuse
    )
    assert np.array_equal(reused.normal_derivative, fresh.normal_derivative)
    assert np.array_equal(reused.single_layer_matrix, fresh.single_layer_matrix)
    assert reused.condition_number == fresh.condition_number
    assert reused.relative_residual_l2 == fresh.relative_residual_l2
    assert np.array_equal(reused.single_layer_matrix, reuse.single)


def test_interior_and_exterior_share_one_assembly():
    mesh = _mesh()
    boundary = mesh.centroids[:, 1].copy()
    reuse = assembled_surface(mesh)
    for domain in ("interior", "exterior"):
        fresh = solve_laplace_surface(mesh, boundary, domain=domain)
        reused = solve_laplace_surface(
            mesh, boundary, domain=domain, assembled=reuse
        )
        assert np.array_equal(reused.normal_derivative, fresh.normal_derivative)


def test_assembly_for_another_mesh_is_rejected():
    mesh = _mesh()
    other = ellipsoid_mesh(2, AXES)
    with pytest.raises(ValueError, match="panels"):
        solve_laplace_surface(
            mesh,
            mesh.centroids[:, 0].copy(),
            domain="exterior",
            assembled=assembled_surface(other),
        )


def test_assembly_at_another_order_is_rejected():
    mesh = _mesh()
    with pytest.raises(ValueError, match="quadrature_order"):
        solve_laplace_surface(
            mesh,
            mesh.centroids[:, 0].copy(),
            domain="exterior",
            quadrature_order=6,
            assembled=assembled_surface(mesh, quadrature_order=8),
        )


def test_assembled_surface_type_is_checked():
    mesh = _mesh()
    with pytest.raises(TypeError):
        solve_laplace_surface(
            mesh,
            mesh.centroids[:, 0].copy(),
            domain="exterior",
            assembled=(np.eye(3), np.eye(3)),  # type: ignore[arg-type]
        )


def test_shapes_are_checked_even_when_the_panel_count_matches():
    mesh = _mesh()
    bad = AssembledSurface(
        single=np.zeros((3, 3)),
        double=np.zeros((3, 3)),
        panel_count=mesh.panel_count,
        quadrature_order=8,
    )
    with pytest.raises(ValueError, match="shape"):
        solve_laplace_surface(
            mesh, mesh.centroids[:, 0].copy(), domain="exterior", assembled=bad
        )


# --------------------------------------------------------------------------
# The benchmark: reuse changes the cost, not the numbers
# --------------------------------------------------------------------------


def test_ellipsoid_benchmark_reports_identical_numbers_with_and_without_reuse():
    reused = ellipsoid_added_mass.run(1, AXES, reuse_assembly=True)
    fresh = ellipsoid_added_mass.run(1, AXES, reuse_assembly=False)
    for reused_row, fresh_row in zip(reused["axes"], fresh["axes"]):
        assert reused_row["motion_axis"] == fresh_row["motion_axis"]
        assert reused_row["added_mass"] == fresh_row["added_mass"]
        assert (
            reused_row["normal_derivative_relative_l2_error"]
            == fresh_row["normal_derivative_relative_l2_error"]
        )
        assert (
            reused_row["single_layer_condition"]
            == fresh_row["single_layer_condition"]
        )
    assert reused["panels"] == fresh["panels"]


# --------------------------------------------------------------------------
# The heavy tier is now reachable
# --------------------------------------------------------------------------


@HEAVY
def test_ellipsoid_heavy_tier_reproduces_within_tolerance():
    """1280-panel ellipsoid: three declared numbers, never checked before.

    Measured 2026-11-03: 35 s single-threaded, relative deviation from the
    documented values 8.7e-6 / 4.8e-6 / 6.5e-6 for the long, intermediate and
    short semi-axis. Before the assembly reuse and the node cache the same run
    exhausted 16 CPU-minutes without finishing, which is why the entries sat in
    the registry unconsumed.
    """
    report = check(["heavy"])
    ellipsoid_rows = [
        row
        for row in report["checks"]
        if row["key"].startswith("ellipsoid_added_mass_1280")
    ]
    assert len(ellipsoid_rows) == 3
    for row in ellipsoid_rows:
        assert row["verdict"] == "PASS", row
        assert row["relative_deviation"] < 1.5e-2