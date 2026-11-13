"""Tests for the multi-level FMM (m17506).

The three translation operators are checked against the kernel itself, and the
operator as a whole is checked against the dense single-layer sum.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.fmm import direct_laplace_matvec
from tensorbem.fmm_multilevel import (
    MultiLevelFMM,
    _degrees,
    _sr_matrix,
    _translation_tables,
    regular_basis,
    singular_basis,
)


def sphere_panels(subdivisions: int):
    """Centroids, areas and vertices of a subdivided icosphere (unit sphere)."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    points = [np.asarray(v, dtype=float) / np.linalg.norm(v) for v in verts]
    cache: dict = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = points[a] + points[b]
            cache[key] = len(points)
            points.append(m / np.linalg.norm(m))
        return cache[key]

    for _ in range(subdivisions):
        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    vertices = np.asarray(points)
    centroids = np.asarray([(vertices[a] + vertices[b] + vertices[c]) / 3.0 for a, b, c in faces])
    return centroids, np.full(len(faces), 1.0 / len(faces)), vertices


def test_regular_translation_is_exact():
    """R_lm(y + d) = sum T R_l'm'(y): polynomial identity, quadrature is exact."""
    p = 6
    rr, _ = _translation_tables(p)
    dirs = np.random.default_rng(7).normal(size=(20, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    y = 0.5 * dirs
    for sign in ((1, 1, -1), (1, -1, -1)):
        d = np.asarray(sign, dtype=float) / math.sqrt(3.0)
        lhs = regular_basis(p, y + d[None, :])
        rhs = regular_basis(p, y) @ rr[sign].T
        assert np.max(np.abs(lhs - rhs)) <= 1e-12 * max(np.max(np.abs(lhs)), 1.0)


#: truncation budget of the S|R translation, ``O((rho/d)**(p+1))`` with
#: ``rho/d <= sqrt(3)/4 = 0.433`` for the worst interaction-list offset.  The test
#: catches *structural* errors (which are O(1)), so the bound is the measured
#: truncation level, not machine precision.
_SR_TOLERANCE = {4: 2e-1, 6: 5e-2, 8: 5e-3}


@pytest.mark.parametrize("p", [4, 6, 8])
def test_singular_translation_matches_the_kernel(p):
    """M2L on a physical pair: multipole of one cell evaluated in a far cell."""
    deg = _degrees(p)
    offsets = ((2, 0, 0), (2, 1, 0), (3, 3, 3))
    rng = np.random.default_rng(11)
    for offset in offsets:
        table = _sr_matrix(p, offset)
        lam = math.sqrt(float(sum(c * c for c in offset)))
        scaled = table * np.power(lam, -(deg + 1.0))[:, None] * np.power(lam, -deg)[None, :]
        weight = 1.0 / (2.0 * deg + 1.0)
        for _ in range(5):
            charges = rng.normal(size=20)
            centre_s = np.asarray(offset, dtype=float)
            sources = rng.uniform(-0.5, 0.5, size=(20, 3)) + centre_s
            targets = rng.uniform(-0.5, 0.5, size=(10, 3))
            multipole = (regular_basis(p, sources - centre_s) * charges[:, None]).sum(axis=0)
            local = scaled.T @ (multipole * weight)
            approx = regular_basis(p, targets) @ local
            exact = (
                charges[None, :]
                / (4.0 * math.pi * np.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=2))
            ).sum(axis=1)
            assert np.max(np.abs(approx - exact)) <= _SR_TOLERANCE[p] * np.max(np.abs(exact))


def test_multipole_to_multipole_matches_direct_multipole():
    """The upward pass must reproduce the multipole of the same charges."""
    centroids, areas, _ = sphere_panels(2)
    operator = MultiLevelFMM(p=6, max_leaf=8)
    operator.setup(centroids)
    operator._geometry_for(centroids)
    multipoles = operator._upward(operator._p2m(areas), operator._geometry_for(centroids))
    for level, cells in enumerate(operator._src_levels):
        rel = centroids - cells["centers"][cells["inv"]]
        direct = np.zeros_like(multipoles[level])
        basis = regular_basis(6, rel) * areas[:, None]
        order = np.argsort(cells["inv"], kind="stable")
        counts = np.bincount(cells["inv"], minlength=len(cells["keys"]))
        starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
        direct = np.add.reduceat(basis[order], starts, axis=0)
        scale = np.maximum(np.abs(direct).max(axis=1), 1e-300)
        assert np.max(np.abs(multipoles[level] - direct).max(axis=1) / scale) <= 1e-12


def test_interaction_lists_cover_every_source_exactly_once():
    """Near sums + M2L interaction lists must partition the sources (no gaps, no
    double counting) -- this is what makes the multilevel scheme consistent."""
    centroids, areas, _ = sphere_panels(2)
    operator = MultiLevelFMM(p=6, max_leaf=8)
    operator.setup(centroids)
    geometry = operator._geometry_for(centroids)
    depth = operator.depth
    leaf = geometry["tgt_levels"][depth]
    sources = operator._points
    for cell in range(len(leaf["keys"])):
        multiplicity = np.zeros(len(sources))
        multiplicity[geometry["leaf_near"][cell]] += 1.0
        coord = leaf["coord"][cell]
        for level in range(1, depth + 1):
            m = 1 << level
            ancestor = coord >> (depth - level)
            key = (ancestor[0] * m + ancestor[1]) * m + ancestor[2]
            keys = geometry["tgt_levels"][level]["keys"]
            slot = int(np.searchsorted(keys, key))
            if keys[slot] != key:
                continue
            for _, t_idx, s_idx in geometry["m2l"][level]:
                for j in np.flatnonzero(t_idx == slot):
                    multiplicity[operator._src_levels[level]["inv"] == s_idx[j]] += 1.0
        assert np.all(multiplicity == 1.0), "source coverage must be a partition"


@pytest.mark.parametrize("order", [4, 6])
def test_operator_matches_dense_single_layer(order):
    centroids, areas, vertices = sphere_panels(2)
    operator = MultiLevelFMM(p=order, max_leaf=8)
    operator.setup(centroids)
    computed = operator.matvec(vertices, centroids, areas)
    reference = direct_laplace_matvec(vertices, centroids, areas)
    assert np.max(np.abs(computed - reference)) <= 1e-3 * np.max(np.abs(reference))


def test_operator_is_linear_in_the_charges():
    centroids, areas, vertices = sphere_panels(2)
    operator = MultiLevelFMM(p=6, max_leaf=8)
    operator.setup(centroids)
    rng = np.random.default_rng(3)
    q1 = rng.normal(size=len(centroids))
    q2 = rng.normal(size=len(centroids))
    combined = operator.matvec(vertices, centroids, 2.0 * q1 - 3.0 * q2)
    expected = 2.0 * operator.matvec(vertices, centroids, q1) - 3.0 * operator.matvec(vertices, centroids, q2)
    assert np.max(np.abs(combined - expected)) <= 1e-12 * max(np.max(np.abs(expected)), 1.0)


def test_targets_outside_the_source_box_regrid_instead_of_crashing():
    centroids, areas, vertices = sphere_panels(2)
    operator = MultiLevelFMM(p=6, max_leaf=8)
    operator.setup(centroids)
    outside = vertices * 3.0  # well outside the source cloud
    operator.matvec(outside, centroids, areas)  # must not raise
    computed = operator.matvec(vertices, centroids, areas)
    reference = direct_laplace_matvec(vertices, centroids, areas)
    assert np.max(np.abs(computed - reference)) <= 1e-3 * np.max(np.abs(reference))


def test_guards():
    centroids, areas, vertices = sphere_panels(1)
    operator = MultiLevelFMM(p=4, max_leaf=8)
    with pytest.raises(RuntimeError):
        operator.matvec(vertices, centroids, areas)
    operator.setup(centroids)
    with pytest.raises(ValueError):
        operator.matvec(vertices, centroids[:-1], areas[:-1])
    with pytest.raises(ValueError):
        operator.matvec(vertices, centroids, areas[:-1])
    with pytest.raises(ValueError):
        MultiLevelFMM(p=4).setup(np.zeros((0, 3)))
