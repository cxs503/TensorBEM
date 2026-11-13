"""Multi-level fast multipole method for the 3-D Laplace single-layer kernel.

Adds a *hierarchical* (multi-level) FMM next to the single-level implementation
in :mod:`tensorbem.fmm`.  The single-level path keeps one octree level: every
target sees one multipole per source leaf, so with a fixed leaf capacity the
number of leaves -- and therefore the work -- still grows like
``O(N**2 / max_leaf)``.  The scheme here is the classical three-pass one

    P2M (leaf multipoles) -> M2M (upward) -> M2L (interaction lists)
                          -> L2L (downward) -> L2P (leaf locals) + near sums

and its cost per target is ``O((p+1)**2)`` up to a logarithmic number of levels.

Conventions (identical to :mod:`tensorbem.fmm`)
-----------------------------------------------
With the Schmidt semi-normalised real harmonics ``Y`` of
``all_real_spherical_harmonics`` (``Y_00 = 1`` and
``sum_m Y_lm(a) Y_lm(b) = (2l+1) P_l(a.b)``) the kernel splits as

    1 / |x - y| = sum_{l,m} S_lm(x - c) R_lm(y - c) / (2 l + 1)

with the regular solid harmonic ``R_lm(v) = |v|**l Y_lm(v/|v|)`` and the
singular one ``S_lm(v) = Y_lm(v/|v|) / |v|**(l+1)``.  Everything below uses the
*unnormalised* multipole basis ``m_lm = sum_j q_j R_lm(y_j - c)`` so that
``phi(x) = sum_{l,m} m_lm S_lm(x - c) / (2l+1)``.

The translation operators are derived from the kernel itself by quadrature
projection on a sphere (``_translation_tables``) rather than by typing in
tabulated spherical-harmonic translation coefficients: the projection is exact
for the polynomial (R|R) case and geometrically convergent for the singular
(S|R) case.  Both are verified against direct evaluation in
``tests/test_fmm_multilevel.py``.

Scope
-----
The tree is built over the sources; targets must lie inside the source
bounding cube plus a small dilation (the BEM panel sets satisfy this).
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .fmm import all_real_spherical_harmonics, direct_laplace_matvec

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi
_SQRT3_HALF = math.sqrt(3.0) / 2.0

#: candidate interaction offsets of a uniform octree: every cell offset with
#: components in ``[-3, 3]`` that is not one of the 27 near cells.  Which of
#: them belongs to the interaction list of a cell depends on the cell's position
#: inside its parent (see ``_build_geometry``): the pair rule is "the parents are
#: neighbours and the cells themselves are not", which is the classical
#: well-separatedness criterion and makes the multilevel cover *exactly once*.
_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    o for o in product(range(-3, 4), repeat=3)
    if max(abs(o[0]), abs(o[1]), abs(o[2])) > 1
)
_CHILD_SIGNS: tuple[tuple[int, int, int], ...] = tuple(product((-1, 1), repeat=3))


def _degrees(p: int) -> FloatArray:
    """Spherical-harmonic degree of every column of the ``(p+1)**2`` basis."""
    return np.repeat(np.arange(p + 1, dtype=float), 2 * np.arange(p + 1) + 1)


def _sphere_nodes(n_theta: int, n_phi: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre x uniform rule on the unit sphere (weights sum to 4 pi)."""
    x, w = np.polynomial.legendre.leggauss(int(n_theta))
    phis = _TWO_PI * np.arange(int(n_phi)) / float(n_phi)
    cos_t = np.repeat(x, int(n_phi))
    sin_t = np.sqrt(np.maximum(1.0 - cos_t * cos_t, 0.0))
    phi = np.tile(phis, int(n_theta))
    dirs = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
    weights = np.repeat(w, int(n_phi)) * (_TWO_PI / float(n_phi))
    return dirs, weights


def _angles(vectors: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    r = np.linalg.norm(vectors, axis=1)
    safe = np.maximum(r, 1e-300)
    cos_t = np.clip(vectors[:, 2] / safe, -1.0, 1.0)
    return np.arccos(cos_t), np.arctan2(vectors[:, 1], vectors[:, 0]), r


def regular_basis(p: int, vectors: FloatArray) -> FloatArray:
    """``R_lm(v) = |v|**l Y_lm(v)`` for every basis column."""
    theta, phi, r = _angles(vectors)
    Y = all_real_spherical_harmonics(p, theta, phi)
    return Y * np.power(np.maximum(r, 0.0)[:, None], _degrees(p)[None, :])


def singular_basis(p: int, vectors: FloatArray) -> FloatArray:
    """``S_lm(v) = Y_lm(v) / |v|**(l+1)`` for every basis column."""
    theta, phi, r = _angles(vectors)
    Y = all_real_spherical_harmonics(p, theta, phi)
    return Y * np.power(np.maximum(r, 1e-300)[:, None], -(_degrees(p) + 1.0)[None, :])


@lru_cache(maxsize=None)
def _translation_tables(p: int) -> tuple[dict, dict]:
    """Translation matrices of order ``p``, built from the kernel itself.

    Returns ``(rr, sr)`` with

    * ``rr[(sx, sy, sz)]`` -- R|R matrix for the child offset direction
      ``(sx, sy, sz)/sqrt(3)`` of unit length,
    * ``sr[offset]`` -- S|R matrix for the integer cell offset ``offset``, unit
      displacement and quadrature radius ``sqrt(3)/2 / |offset|`` (i.e. the
      largest target-cell radius over the source-target distance).

    Both are indexed ``[(l, m), (l', m')]`` and act on blocked coefficient
    vectors.  ``rr`` is the coefficient matrix of the exact polynomial identity
    ``R_lm(y + d) = sum_{l',m'} T R_l'm'(y)``: projecting both sides onto ``Y``
    on a sphere gives an integral that the Gauss-Legendre rule integrates
    exactly.  ``sr`` is the same projection applied to the analytic function
    ``S_lm(y + d)`` inside its ball of convergence.
    """
    deg = _degrees(p)

    # ---- R|R (polynomial: quadrature is exact) ---------------------------
    dirs, weights = _sphere_nodes(p + 2, 2 * (p + 2))
    Y_dir = all_real_spherical_harmonics(p, *_angles(dirs)[:2])
    rr: dict[tuple[int, int, int], FloatArray] = {}
    for sign in _CHILD_SIGNS:
        d = np.asarray(sign, dtype=float) / math.sqrt(3.0)
        lhs = regular_basis(p, dirs + d[None, :])
        # the harmonics are orthonormal on the sphere (int Y_lm Y_l'm' = delta),
        # so on the unit sphere the projection is the coefficient matrix itself
        gram = Y_dir.T @ (weights[:, None] * lhs)
        rr[sign] = gram.T

    # ---- S|R (analytic: quadrature converges geometrically) --------------
    # built lazily: only the offsets that appear in an interaction list are
    # needed (see ``_sr_matrix``)
    return rr, {}


@lru_cache(maxsize=None)
def _sr_matrix(p: int, offset: tuple[int, int, int]) -> FloatArray:
    """S|R translation matrix for integer cell *offset* (unit displacement)."""
    deg = _degrees(p)
    n_theta = 2 * (p + 2)
    dirs, weights = _sphere_nodes(n_theta, 2 * n_theta)
    Y_dir = all_real_spherical_harmonics(p, *_angles(dirs)[:2])
    norm = math.sqrt(float(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2))
    # S_lm(y + dd) with dd the unit vector *from the target cell towards the
    # source cell*, |y| <= sqrt(3)/2 / |offset|.
    dd = -np.asarray(offset, dtype=float) / norm
    kappa = _SQRT3_HALF / norm
    gram = Y_dir.T @ (weights[:, None] * singular_basis(p, kappa * dirs + dd[None, :]))
    scale = 1.0 / np.power(kappa, deg)  # R_l'm'(a y) = a**l' Y_l'm'(y)
    return (gram * scale[:, None]).T


class MultiLevelFMM:
    """Hierarchical FMM operator for ``phi = S q`` (Laplace single layer)."""

    def __init__(self, p: int = 8, max_leaf: int = 32, max_level: int = 9) -> None:
        self.p = int(p)
        self.max_leaf = int(max_leaf)
        self.max_level = int(max_level)
        self._points: FloatArray | None = None
        self._charges: FloatArray | None = None
        self._cache_targets: FloatArray | None = None
        self._geometry: dict | None = None

    @property
    def depth(self) -> int:
        """Depth of the uniform octree built by :meth:`setup`."""
        return self._depth

    # -- tree ------------------------------------------------------------
    def _cells(self, points: FloatArray, level: int) -> tuple[IntArray, IntArray, FloatArray]:
        """Uniform-grid cells of *points* at *level* (cell keys, assignment, centres)."""
        m = 1 << level
        h = self._root_side / m
        base = self._root_center - 0.5 * self._root_side
        coord = np.clip(np.floor((points - base) / h).astype(np.int64), 0, m - 1)
        keys = (coord[:, 0] * m + coord[:, 1]) * m + coord[:, 2]
        uniq, inv = np.unique(keys, return_inverse=True)
        ci = uniq // (m * m)
        cj = (uniq // m) % m
        ck = uniq % m
        centers = base + (np.stack([ci, cj, ck], axis=1) + 0.5) * h
        return uniq, inv.ravel(), centers

    def _cell_coords(self, keys: IntArray, level: int) -> IntArray:
        m = 1 << level
        return np.stack([keys // (m * m), (keys // m) % m, keys % m], axis=1)

    def setup(self, points: FloatArray, charges: FloatArray | None = None) -> "MultiLevelFMM":
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if len(points) == 0:
            raise ValueError("cannot build a multi-level FMM over an empty point set")
        self._points = points
        self._charges = None if charges is None else np.asarray(charges, dtype=float)
        self._set_grid(points)
        self._cache_targets = None
        self._geometry = None
        # one-off, order-only cost: build the translation tables now so that
        # repeated matvecs (iterative solvers) do not pay for them
        _translation_tables(self.p)
        return self

    def _set_grid(self, extra: FloatArray | None) -> None:
        """(Re)build the uniform octree over the sources, optionally including *extra*."""
        points = self._points
        cloud = points if extra is None or len(extra) == 0 else np.vstack([points, extra])
        lo = cloud.min(axis=0)
        hi = cloud.max(axis=0)
        side = float(np.max(hi - lo))
        side = max(side, 1e-12) * (1.0 + 1e-6)
        self._root_center = 0.5 * (lo + hi)
        self._root_side = side
        depth = self.max_level
        for level in range(self.max_level + 1):
            _, inv, _ = self._cells(cloud, level)
            if int(np.bincount(inv).max()) <= self.max_leaf:
                depth = level
                break
        self._depth = depth
        self._src_levels = []
        for level in range(depth + 1):
            uniq, inv, centers = self._cells(points, level)
            self._src_levels.append(
                {"keys": uniq, "inv": inv, "centers": centers,
                 "coord": self._cell_coords(uniq, level)}
            )
        # leaf groups of the sources (ascending point index inside a cell)
        inv_leaf = self._src_levels[depth]["inv"]
        order = np.argsort(inv_leaf, kind="stable")
        counts = np.bincount(inv_leaf, minlength=len(self._src_levels[depth]["keys"]))
        starts = np.concatenate([[0], np.cumsum(counts)])
        self._leaf_order = order
        self._leaf_starts = starts

    # -- geometry (cached for a target set) --------------------------------
    def _shift_pairs(
        self, dst: dict, level: int, offset: tuple, src: dict
    ) -> tuple[IntArray, IntArray]:
        """Cells of *src* at ``dst_cell + offset`` inside the grid at *level*."""
        m = 1 << level
        ncoord = dst["coord"] + np.asarray(offset, dtype=np.int64)
        inside = np.all((ncoord >= 0) & (ncoord < m), axis=1)
        if not np.any(inside):
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        nkeys = (ncoord[:, 0] * m + ncoord[:, 1]) * m + ncoord[:, 2]
        keys = src["keys"]
        pos = np.searchsorted(keys, nkeys[inside])
        pos = np.minimum(pos, len(keys) - 1)
        hit = keys[pos] == nkeys[inside]
        return np.flatnonzero(inside)[hit], pos[hit]

    def _parent_pairs(
        self, child: dict, parent: dict, level: int
    ) -> list[tuple[tuple[int, int, int], IntArray, IntArray]]:
        """Group children of *level* by their sign triple under the *level-1* parent."""
        m_parent = 1 << (level - 1)
        pc = child["coord"] // 2
        pkeys = (pc[:, 0] * m_parent + pc[:, 1]) * m_parent + pc[:, 2]
        keys = parent["keys"]
        pos = np.searchsorted(keys, pkeys)
        pos = np.minimum(pos, len(keys) - 1)
        if not np.array_equal(keys[pos], pkeys):
            raise RuntimeError("internal error: octree parent cell missing")
        sign = 2 * (child["coord"] - 2 * pc) - 1
        groups = []
        for sx, sy, sz in _CHILD_SIGNS:
            mask = (sign[:, 0] == sx) & (sign[:, 1] == sy) & (sign[:, 2] == sz)
            if np.any(mask):
                groups.append(((sx, sy, sz), np.flatnonzero(mask), pos[mask]))
        return groups

    def _build_geometry(self, targets: FloatArray) -> dict:
        depth = self._depth
        tgt_levels = []
        for level in range(depth + 1):
            keys, inv, centers = self._cells(targets, level)
            tgt_levels.append(
                {"keys": keys, "inv": inv, "centers": centers,
                 "coord": self._cell_coords(keys, level)}
            )
        m2l: list[list[tuple[tuple[int, int, int], IntArray, IntArray]]] = [[] for _ in range(depth + 1)]
        for level in range(1, depth + 1):
            tgt = tgt_levels[level]
            for offset in _OFFSETS:
                t_idx, s_idx = self._shift_pairs(tgt, level, offset, self._src_levels[level])
                if len(t_idx) == 0:
                    continue
                # classical well-separatedness: parents are neighbours, the cells
                # themselves are not (guarantees level-by-level exact coverage)
                src = tgt["coord"][t_idx] + np.asarray(offset, dtype=np.int64)
                keep = np.all(np.abs(tgt["coord"][t_idx] // 2 - src // 2) <= 1, axis=1)
                if not np.any(keep):
                    continue
                m2l[level].append((offset, t_idx[keep], s_idx[keep]))
        l2l = [[] for _ in range(depth + 1)]
        upward = [[] for _ in range(depth + 1)]
        for level in range(1, depth + 1):
            l2l[level] = self._parent_pairs(tgt_levels[level], tgt_levels[level - 1], level)
            upward[level] = self._parent_pairs(
                self._src_levels[level], self._src_levels[level - 1], level
            )
        # near field: 27-cell neighbourhood of every target leaf
        inv_leaf = tgt_levels[depth]["inv"]
        n_tgt_leaf = len(tgt_levels[depth]["keys"])
        tgt_order = np.argsort(inv_leaf, kind="stable")
        counts = np.bincount(inv_leaf, minlength=n_tgt_leaf)
        starts = np.concatenate([[0], np.cumsum(counts)])
        leaf_targets = [tgt_order[starts[k]:starts[k + 1]] for k in range(n_tgt_leaf)]
        near_cells = [[] for _ in range(n_tgt_leaf)]
        for offset in product(range(-1, 2), repeat=3):
            t_idx, s_idx = self._shift_pairs(
                tgt_levels[depth], depth, offset, self._src_levels[depth]
            )
            for t_cell, s_cell in zip(t_idx, s_idx):
                near_cells[int(t_cell)].append(int(s_cell))
        leaf_near = []
        for s_cells in near_cells:
            if s_cells:
                parts = [
                    self._leaf_order[self._leaf_starts[c]:self._leaf_starts[c + 1]]
                    for c in s_cells
                ]
                leaf_near.append(np.concatenate(parts))
            else:
                leaf_near.append(np.zeros(0, dtype=np.int64))
        return {
            "targets": targets,
            "tgt_levels": tgt_levels,
            "m2l": m2l,
            "l2l": l2l,
            "upward": upward,
            "leaf_targets": leaf_targets,
            "leaf_near": leaf_near,
        }

    def _covers(self, targets: FloatArray) -> bool:
        half = 0.5 * self._root_side
        slack = 1e-9 * half
        return bool(np.all(np.abs(targets - self._root_center) <= half + slack))

    def _geometry_for(self, targets: FloatArray) -> dict:
        if self._geometry is None or self._cache_targets is None or not np.array_equal(
            self._cache_targets, targets
        ):
            self._geometry = self._build_geometry(targets)
            self._cache_targets = np.array(targets, dtype=float, copy=True)
        return self._geometry

    # -- passes ----------------------------------------------------------
    def _p2m(self, charges: FloatArray) -> FloatArray:
        """Leaf multipoles ``m_lm = sum_j q_j R_lm(y_j - c_leaf)``."""
        depth = self._depth
        leaf = self._src_levels[depth]
        rel = self._points - leaf["centers"][leaf["inv"]]
        basis = regular_basis(self.p, rel) * charges[:, None]
        order = np.argsort(leaf["inv"], kind="stable")
        starts = self._leaf_starts[:-1]
        return np.add.reduceat(basis[order], starts, axis=0)

    def _upward(self, leaf_multipoles: FloatArray, geometry: dict) -> list[FloatArray]:
        p = self.p
        rr, _ = _translation_tables(p)
        deg = _degrees(p)
        depth = self._depth
        multipoles: list[FloatArray] = [np.zeros(0)] * (depth + 1)
        multipoles[depth] = leaf_multipoles
        for level in range(depth, 0, -1):
            child = multipoles[level]
            parent = np.zeros((len(self._src_levels[level - 1]["keys"]), child.shape[1]))
            lam = _SQRT3_HALF * (self._root_side / (1 << level))
            for sign, c_idx, p_idx in geometry["upward"][level]:
                trans = rr[sign] * np.power(lam, deg)[:, None] / np.power(lam, deg)[None, :]
                np.add.at(parent, p_idx, child[c_idx] @ trans.T)
            multipoles[level - 1] = parent
        return multipoles

    def _transfer(
        self, multipoles: list[FloatArray], geometry: dict
    ) -> list[FloatArray]:
        """M2L into per-level locals, then L2L down to the leaves."""
        p = self.p
        deg = _degrees(p)
        weight = 1.0 / (2.0 * deg + 1.0)
        depth = self._depth
        locals_: list[FloatArray] = []
        for level in range(depth + 1):
            n_cell = len(geometry["tgt_levels"][level]["keys"])
            locals_.append(np.zeros((n_cell, deg.size)))
        for level in range(1, depth + 1):
            acc = locals_[level]
            lam_cell = self._root_side / (1 << level)
            for offset, t_idx, s_idx in geometry["m2l"][level]:
                lam = lam_cell * math.sqrt(float(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2))
                trans = (
                    _sr_matrix(p, offset)
                    * np.power(lam, -(deg + 1.0))[:, None]
                    * np.power(lam, -deg)[None, :]
                )
                np.add.at(acc, t_idx, (multipoles[level][s_idx] * weight[None, :]) @ trans)
        # top-down: the parent must be complete before it is translated down
        for level in range(1, depth + 1):
            child = locals_[level]
            parent = locals_[level - 1]
            lam = _SQRT3_HALF * (self._root_side / (1 << level))
            for sign, c_idx, p_idx in geometry["l2l"][level]:
                trans = _translation_tables(p)[0][sign]
                trans = trans * np.power(lam, deg)[:, None] / np.power(lam, deg)[None, :]
                child[c_idx] += parent[p_idx] @ trans
        return locals_

    def _l2p(self, targets: FloatArray, leaf_locals: FloatArray, geometry: dict) -> FloatArray:
        depth = self._depth
        leaf = geometry["tgt_levels"][depth]
        rel = targets - leaf["centers"][leaf["inv"]]
        basis = regular_basis(self.p, rel)
        return np.einsum("ij,ij->i", basis, leaf_locals[leaf["inv"]])

    def _near(
        self,
        targets: FloatArray,
        charges: FloatArray,
        geometry: dict,
        max_pairs: int = 262_144,
    ) -> FloatArray:
        out = np.zeros(len(targets))
        sources = self._points
        for slot, t_idx in enumerate(geometry["leaf_targets"]):
            if len(t_idx) == 0:
                continue
            s_idx = geometry["leaf_near"][slot]
            if len(s_idx) == 0:
                continue
            step = max(1, int(max_pairs // max(len(s_idx), 1)))
            for start in range(0, len(t_idx), step):
                blk = t_idx[start:start + step]
                diff = targets[blk][:, None, :] - sources[s_idx][None, :, :]
                r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
                r = np.maximum(r, 1e-15)
                out[blk] += np.sum(charges[s_idx][None, :] / (_FOUR_PI * r), axis=1)
        return out

    # -- public ----------------------------------------------------------
    def matvec(
        self,
        targets: FloatArray | None = None,
        sources: FloatArray | None = None,
        charges: FloatArray | None = None,
    ) -> FloatArray:
        if self._points is None:
            raise RuntimeError("MultiLevelFMM.matvec() called before setup()")
        if sources is None:
            sources = self._points
        else:
            sources = np.asarray(sources, dtype=float)
            if sources.shape != self._points.shape or not np.array_equal(sources, self._points):
                raise ValueError(
                    "sources differ from the point set passed to setup(); the tree, "
                    "cell indices and multipole caches belong to that geometry"
                )
        if charges is None:
            if self._charges is None:
                raise ValueError("charges must be supplied (setup() got no charges)")
            charges = self._charges
        charges = np.asarray(charges, dtype=float)
        if charges.shape != (len(self._points),):
            raise ValueError("charges must have shape (N,)")
        if targets is None:
            targets = self._points
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("targets must have shape (M, 3)")
        if len(targets) == 0:
            return np.zeros(0)
        if not self._covers(targets):
            # the uniform grid is grown to the union of sources and targets, so a
            # target set that sticks out of the source bounding cube simply costs
            # one extra regrid (cached for repeated matvecs with that target set)
            self._set_grid(targets)
            self._geometry = None
            self._cache_targets = None

        geometry = self._geometry_for(targets)
        multipoles = self._upward(self._p2m(charges), geometry)
        locals_ = self._transfer(multipoles, geometry)
        far = self._l2p(targets, locals_[self._depth], geometry)
        near = self._near(targets, charges, geometry)
        return far + near


def fmm_multilevel_laplace_matvec(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
    p: int = 8,
    max_leaf: int = 32,
    max_level: int = 9,
) -> FloatArray:
    """Convenience wrapper: multi-level FMM matvec for the Laplace kernel."""
    operator = MultiLevelFMM(p=p, max_leaf=max_leaf, max_level=max_level)
    operator.setup(sources)
    return operator.matvec(targets, sources, charges)
