"""Panel-mesh geometry validation tests (m8580).

Hardens the unit-test surface of the panel mesh geometry module
(``tensorbem.geometry3d``) with three predeclared assertion groups:

1. **Watertightness** -- every undirected edge of a closed manifold mesh is
   shared by exactly two panels and the mesh has zero isolated (boundary)
   edges; the same incidence check detects the three boundary edges of an
   open mesh.
2. **Orientation** -- adjacent panels traverse their shared edge in opposite
   directions, and the outward-normal statistics are positive (per-panel
   ``normal . centroid > 0`` for star-shaped bodies, positive signed volume).
3. **Area convergence** -- the flat-faceted unit sphere (icosphere) area
   approaches ``4*pi`` with relative error below the predeclared 1% gate at
   1280 panels, the error decreases monotonically under refinement, and the
   error ratio between consecutive refinements stays in the second-order
   band ``(3, 5)``.

All numbers are produced by the code under test; nothing is hand-transcribed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.geometry3d import TriangleMesh, icosphere, ellipsoid_mesh

SPHERE_LEVELS = [0, 1, 2, 3]
EXACT_SPHERE_AREA = 4.0 * np.pi
AREA_ERROR_GATE = 0.01  # predeclared acceptance threshold (task m8580)


# ---------------------------------------------------------------------------
# incidence helpers (operate on raw face arrays so open meshes can be
# inspected without going through the TriangleMesh constructor, which
# rejects non-closed meshes by design)
# ---------------------------------------------------------------------------
def undirected_edge_incidence(faces) -> dict[tuple[int, int], list[int]]:
    """Map each undirected edge ``(i, j)`` with ``i < j`` to incident faces."""
    incidence: dict[tuple[int, int], list[int]] = {}
    for f, (a, b, c) in enumerate(np.asarray(faces)):
        for x, y in ((a, b), (b, c), (c, a)):
            key = (int(x), int(y)) if x < y else (int(y), int(x))
            incidence.setdefault(key, []).append(f)
    return incidence


def directed_edge_owner(faces) -> dict[tuple[int, int], int]:
    """Map each directed edge ``(i, j)`` to the face that traverses it."""
    owner: dict[tuple[int, int], int] = {}
    for f, (a, b, c) in enumerate(np.asarray(faces)):
        for x, y in ((a, b), (b, c), (c, a)):
            owner[(int(x), int(y))] = f
    return owner


# ---------------------------------------------------------------------------
# 1. watertightness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level", SPHERE_LEVELS)
def test_watertightness_every_edge_shared_by_exactly_two_panels(level):
    mesh = icosphere(level)
    incidence = undirected_edge_incidence(mesh.faces)
    counts = np.array([len(f) for f in incidence.values()])
    # A closed triangulation has E = 3F/2 undirected edges.
    assert counts.size == 3 * mesh.panel_count // 2
    # Every edge is shared by exactly two panels.
    assert np.all(counts == 2)
    # Zero isolated (boundary) edges on a closed manifold mesh.
    assert int(np.sum(counts == 1)) == 0
    # Euler characteristic of a closed genus-0 surface: V - E + F = 2.
    assert mesh.vertex_count - counts.size + mesh.panel_count == 2


def test_watertightness_check_detects_open_mesh():
    """The same incidence check that passes on a closed mesh must flag the
    three boundary edges left behind when one panel is removed."""
    mesh = icosphere(1)
    open_faces = mesh.faces[1:]
    incidence = undirected_edge_incidence(open_faces)
    counts = np.array([len(f) for f in incidence.values()])
    isolated = int(np.sum(counts == 1))
    assert isolated == 3  # the removed triangle's three edges
    assert int(np.sum(counts == 2)) == counts.size - 3
    # The constructor's watertightness gate rejects the open mesh.
    with pytest.raises(ValueError, match="closed|directed"):
        TriangleMesh(mesh.vertices, open_faces)


# ---------------------------------------------------------------------------
# 2. orientation / normal consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level", SPHERE_LEVELS)
def test_adjacent_panels_traverse_shared_edge_in_opposite_directions(level):
    mesh = icosphere(level)
    owner = directed_edge_owner(mesh.faces)
    for f, (a, b, c) in enumerate(mesh.faces):
        for x, y in ((a, b), (b, c), (c, a)):
            partner = owner[(int(y), int(x))]
            # The reverse traversal exists and belongs to a *different* panel:
            # adjacent panels share the edge in opposite directions.
            assert partner != f


@pytest.mark.parametrize("level", SPHERE_LEVELS)
def test_outward_normal_statistics_positive(level):
    mesh = icosphere(level)
    # The unit sphere is star-shaped about the origin: every outward normal
    # has a positive projection onto the panel centroid direction.
    dot = np.einsum("ij,ij->i", mesh.normals, mesh.centroids)
    assert np.all(dot > 0.0)
    assert float(np.mean(dot)) > 0.0
    assert float(np.min(dot)) > 0.0
    # Positive signed volume is the global outward-orientation statistic.
    assert mesh.volume > 0.0


def test_outward_normal_statistics_positive_ellipsoid():
    axes = np.array([1.5, 1.0, 0.75])
    mesh = ellipsoid_mesh(1, axes)
    # grad(x^2/a^2 + y^2/b^2 + z^2/c^2) points outward from the ellipsoid.
    gradient = mesh.centroids / (axes * axes)
    dot = np.einsum("ij,ij->i", mesh.normals, gradient)
    assert np.all(dot > 0.0)
    assert mesh.volume > 0.0


# ---------------------------------------------------------------------------
# 3. unit-sphere area convergence
# ---------------------------------------------------------------------------
def test_sphere_area_convergence_below_one_percent_and_monotone():
    errors = []
    for level in SPHERE_LEVELS:
        mesh = icosphere(level)
        area = mesh.surface_area
        # Flat panels of the inscribed polyhedron lie inside the sphere.
        assert area < EXACT_SPHERE_AREA
        errors.append((EXACT_SPHERE_AREA - area) / EXACT_SPHERE_AREA)
    errors = np.asarray(errors)
    # Predeclared gate: finest level (1280 panels) within 1% of 4*pi.
    assert errors[-1] < AREA_ERROR_GATE
    # Monotone decrease under refinement.
    assert np.all(np.diff(errors) < 0.0)
    # Second-order convergence: halving h quarters the error (ratio in (3, 5)).
    ratios = errors[:-1] / errors[1:]
    assert np.all(ratios > 3.0)
    assert np.all(ratios < 5.0)


def test_sphere_area_convergence_table_is_deterministic():
    """Rebuilding the meshes reproduces the convergence table bit-for-bit."""
    table = []
    for _ in range(2):
        for level in SPHERE_LEVELS:
            mesh = icosphere(level)
            table.append((mesh.panel_count, mesh.surface_area, mesh.volume))
    first, second = table[: len(SPHERE_LEVELS)], table[len(SPHERE_LEVELS):]
    assert first == second
