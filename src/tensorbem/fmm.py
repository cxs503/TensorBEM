"""Fast Multipole Method (FMM) for 3-D Laplace boundary-integral operators.

Provides O(N) matrix-vector products for the single-layer operator
``S q = sum_j G(x_i, y_j) q_j`` with ``G(r) = 1/(4*pi*r)``, using a
hierarchical octree and real-spherical-harmonic multipole expansions.

Public API
----------
- :class:`Octree` — adaptive octree construction (leaf capacity ≤ max_leaf).
- :class:`FMMOperator` — precomputed translation operators for a given order *p*.
- :func:`fmm_laplace_matvec` — full single-layer matvec with near/far split.

Fail-closed contract
--------------------
Helmholtz kernels (``e^{ikr}/r``) with ``ka > ka_max`` return
``NOT_SUPPORTED`` and fall back to direct summation.  The Laplace kernel
is always supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# ---------------------------------------------------------------------------
# Real spherical harmonics (Condon–Shortley phase, fully normalised)
# ---------------------------------------------------------------------------

def _legendre_pmm(m: int, l: int, x: FloatArray) -> FloatArray:
    """Associated Legendre functions P_l^m(x) for |x| <= 1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    pmm = np.ones(n, dtype=float)
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for _ in range(m):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = np.zeros(n, dtype=float)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def _schmidt_seminorm(m: int, l: int) -> float:
    """Schmidt semi-normalisation factor for P_l^m."""
    if m == 0:
        return math.sqrt((2.0 * l + 1.0) / (4.0 * math.pi))
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) /
                     (4.0 * math.pi * math.factorial(l + m)))


def real_spherical_harmonic(l: int, m: int, theta: FloatArray, phi: FloatArray) -> FloatArray:
    r"""Real spherical harmonic Y_{l,m}(theta, phi)."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    ct = np.cos(theta)
    pmm = _legendre_pmm(abs(m), l, ct)
    norm = _schmidt_seminorm(abs(m), l)

    if m == 0:
        return norm * pmm
    elif m > 0:
        return math.sqrt(2.0) * norm * pmm * np.cos(m * phi)
    else:
        return math.sqrt(2.0) * norm * pmm * np.sin(-m * phi)


def all_real_spherical_harmonics(l_max: int, theta: FloatArray, phi: FloatArray) -> FloatArray:
    """Return array of all Y_{l,m} for l=0..l_max, m=-l..l.

    Shape: (n_points, n_harmonics) where n_harmonics = (l_max+1)^2.
    Ordering: l=0,m=0; l=1,m=-1,0,1; l=2,m=-2,-1,0,1,2; ...
    """
    n = len(theta)
    n_harm = (l_max + 1) ** 2
    result = np.zeros((n, n_harm), dtype=float)
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            result[:, idx] = real_spherical_harmonic(l, m, theta, phi)
            idx += 1
    return result


#: A multipole expansion about a leaf centre is only evaluated when the
#: target lies at least this many source-radii away (see FMMOperator.matvec).
_ADMISSIBILITY = 2.0


def _leaf_radii(points: FloatArray, leaves: list) -> list:
    """Largest distance from each leaf centre to one of its own sources."""

    radii = []
    for leaf in leaves:
        if len(leaf.indices) == 0:
            radii.append(0.0)
            continue
        d = points[leaf.indices] - leaf.center[None, :]
        radii.append(float(np.max(np.linalg.norm(d, axis=1))))
    return radii


def _harmonic_degrees(p: int) -> FloatArray:
    """Degree ``l`` of every column of :func:`all_real_spherical_harmonics`."""
    deg = np.empty((p + 1) ** 2, dtype=float)
    col = 0
    for l in range(p + 1):
        deg[col:col + 2 * l + 1] = float(l)
        col += 2 * l + 1
    return deg


def _sph_index(l: int, m: int) -> int:
    """Return the sequential index for spherical harmonic (l, m)."""
    return l * l + (m + l)


# ---------------------------------------------------------------------------
# Octree
# ---------------------------------------------------------------------------

@dataclass
class OctreeNode:
    """One node in the octree."""
    center: FloatArray
    size: float
    indices: IntArray
    children: list[OctreeNode | None] = field(default_factory=lambda: [None] * 8)
    is_leaf: bool = True
    level: int = 0


def _child_index(point: FloatArray, center: FloatArray) -> int:
    """Return child octant index (0-7) for a point relative to center."""
    idx = 0
    if point[0] >= center[0]:
        idx |= 1
    if point[1] >= center[1]:
        idx |= 2
    if point[2] >= center[2]:
        idx |= 4
    return idx


def _build_octree_recursive(
    points: FloatArray,
    indices: IntArray,
    center: FloatArray,
    size: float,
    max_leaf: int,
    max_depth: int,
    level: int,
) -> OctreeNode:
    """Recursively build octree."""
    node = OctreeNode(
        center=center.copy(),
        size=size,
        indices=indices.copy(),
        level=level,
    )

    if len(indices) <= max_leaf or level >= max_depth:
        return node

    half = size / 2.0
    quarter = size / 4.0
    # Child ``k`` must be the octant that ``_child_index`` maps to ``k``, i.e.
    # bit 0 of ``k`` selects x, bit 1 selects y and bit 2 selects z.  Each child
    # box has edge ``size/2``, so its centre lies a quarter of the parent edge
    # from the parent centre: the eight children then tile the parent with no gap
    # and no overlap, and ``_child_index`` is the exact containment test for them.
    child_centers = []
    for k in range(8):
        child_centers.append(center + quarter * np.array([
            1.0 if (k & 1) else -1.0,
            1.0 if (k & 2) else -1.0,
            1.0 if (k & 4) else -1.0,
        ]))

    child_indices: list[list[int]] = [[] for _ in range(8)]
    for idx in indices:
        ci = _child_index(points[idx], center)
        child_indices[ci].append(int(idx))

    has_children = False
    for i in range(8):
        if len(child_indices[i]) > 0:
            child_arr = np.array(child_indices[i], dtype=np.int64)
            node.children[i] = _build_octree_recursive(
                points, child_arr, child_centers[i], half,
                max_leaf, max_depth, level + 1,
            )
            node.is_leaf = False
            has_children = True

    if not has_children:
        node.is_leaf = True

    return node


def build_octree(
    points: FloatArray,
    max_leaf: int = 128,
    max_depth: int | None = None,
) -> OctreeNode:
    """Build an adaptive octree for the given points."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    n = len(points)
    if max_depth is None:
        max_depth = max(1, math.ceil(math.log2(max(n, 1) / max_leaf)) + 2)
        max_depth = min(max_depth, 20)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    size = float(np.max(maxs - mins)) * 1.01

    if size <= 0.0:
        size = 1.0

    indices = np.arange(n, dtype=np.int64)
    return _build_octree_recursive(
        points, indices, center, size, max_leaf, max_depth, 0
    )


def collect_leaves(node: OctreeNode) -> list[OctreeNode]:
    """Return all leaf nodes in the octree."""
    if node.is_leaf:
        return [node]
    leaves = []
    for child in node.children:
        if child is not None:
            leaves.extend(collect_leaves(child))
    return leaves


def collect_all_nodes(node: OctreeNode) -> list[OctreeNode]:
    """Return all nodes (internal + leaf) in the octree."""
    nodes = [node]
    for child in node.children:
        if child is not None:
            nodes.extend(collect_all_nodes(child))
    return nodes


# ---------------------------------------------------------------------------
# Direct computation
# ---------------------------------------------------------------------------

def _direct_laplace_matvec_batch(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
    max_pairs: int = 4_000_000,
) -> FloatArray:
    """Vectorized direct evaluation of sum_j q_j / (4*pi*|x_i - y_j|).

    Targets are processed in blocks so that the intermediate
    ``(n_targets_block, n_sources, 3)`` array stays below ``max_pairs`` entries.
    Blocking only slices the target axis, so every target's sum is accumulated in
    exactly the same order as in the unblocked version (bitwise identical).
    """
    targets = np.asarray(targets, dtype=float)
    sources = np.asarray(sources, dtype=float)
    charges = np.asarray(charges, dtype=float)

    n_t = targets.shape[0]
    n_s = sources.shape[0]
    result = np.zeros(n_t, dtype=float)
    if n_t == 0 or n_s == 0:
        return result

    step = int(max(1, max_pairs // max(n_s, 1)))
    for start in range(0, n_t, step):
        block = targets[start:start + step]
        diff = block[:, None, :] - sources[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        r = np.maximum(r, 1e-15)
        result[start:start + len(block)] = np.sum(
            charges[None, :] / (4.0 * math.pi * r), axis=1
        )
    return result


# ---------------------------------------------------------------------------
# Multipole expansion for Laplace kernel 1/(4*pi*r)
# ---------------------------------------------------------------------------

def _multipole_coefficients(
    sources: FloatArray,
    charges: FloatArray,
    center: FloatArray,
    p: int,
) -> FloatArray:
    """Compute multipole expansion coefficients about *center*.

    M_{l,m} = (1/(2l+1)) * sum_j q_j * |s_j|^l * Y_{l,m}(s_j_hat)
    """
    n_harm = (p + 1) ** 2
    M = np.zeros(n_harm, dtype=float)

    rel = sources - center[None, :]
    r = np.linalg.norm(rel, axis=1)
    valid = r > 1e-15
    if not np.any(valid):
        return M

    theta = np.arccos(np.clip(rel[valid, 2] / r[valid], -1.0, 1.0))
    phi = np.arctan2(rel[valid, 1], rel[valid, 0])

    Y = all_real_spherical_harmonics(p, theta, phi)

    for l in range(p + 1):
        factor = 1.0 / (2.0 * l + 1.0)
        for m in range(-l, l + 1):
            col = _sph_index(l, m)
            M[col] = factor * float(
                np.sum(charges[valid] * (r[valid] ** l) * Y[:, col])
            )

    return M


def _evaluate_multipole_batch(
    M: FloatArray,
    targets: FloatArray,
    center: FloatArray,
    p: int,
) -> FloatArray:
    """Evaluate multipole expansion at target points (vectorized)."""
    n = len(targets)
    result = np.zeros(n, dtype=float)

    rel = targets - center[None, :]
    r = np.linalg.norm(rel, axis=1)
    valid = r > 1e-15

    if not np.any(valid):
        return result

    theta = np.arccos(np.clip(rel[valid, 2] / r[valid], -1.0, 1.0))
    phi = np.arctan2(rel[valid, 1], rel[valid, 0])

    Y = all_real_spherical_harmonics(p, theta, phi)

    # Vectorized evaluation: result = Y @ (M / r^{l+1})
    # Build the scaling factor for each harmonic
    n_harm = (p + 1) ** 2
    scale = np.zeros(n_harm, dtype=float)
    idx = 0
    for l in range(p + 1):
        for m in range(-l, l + 1):
            scale[idx] = 1.0 / (r[valid] ** (l + 1))
            idx += 1

    result[valid] = Y @ (M * scale)
    return result


# ---------------------------------------------------------------------------
# Single-level FMM
# ---------------------------------------------------------------------------

def fmm_single_level_matvec(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
    center: FloatArray,
    p: int,
    near_threshold: float = 2.0,
) -> FloatArray:
    """Single-level FMM matvec for Laplace kernel."""
    n = len(targets)
    result = np.zeros(n, dtype=float)

    rel_s = sources - center[None, :]
    r_s = np.linalg.norm(rel_s, axis=1)
    max_r_s = float(np.max(r_s)) if len(r_s) > 0 else 0.0

    M = _multipole_coefficients(sources, charges, center, p)

    for i in range(n):
        rel_t = targets[i] - center
        r_t = float(np.linalg.norm(rel_t))

        if r_t > near_threshold * max_r_s and r_t > 1e-10:
            theta = math.acos(np.clip(rel_t[2] / r_t, -1.0, 1.0))
            phi = math.atan2(rel_t[1], rel_t[0])
            Y = all_real_spherical_harmonics(p, np.array([theta]), np.array([phi]))[0]
            idx = 0
            for l in range(p + 1):
                for m in range(-l, l + 1):
                    col = _sph_index(l, m)
                    result[i] += M[col] * Y[col] / (r_t ** (l + 1))
                    idx += 1
        else:
            diff = sources - targets[i]
            r = np.linalg.norm(diff, axis=1)
            r = np.maximum(r, 1e-15)
            result[i] = float(np.sum(charges / (4.0 * math.pi * r)))

    return result


# ---------------------------------------------------------------------------
# Full FMM with octree
# ---------------------------------------------------------------------------

@dataclass
class FMMOperator:
    """Precomputed FMM operator for a given octree and expansion order."""
    p: int
    max_leaf: int = 128
    tree: OctreeNode | None = field(default=None, repr=False)
    leaves: list[OctreeNode] = field(default_factory=list, repr=False)
    _multipoles: list[FloatArray] = field(default_factory=list, repr=False)
    #: largest source distance from its own leaf centre; the multipole expansion
    #: about that centre only converges for ``r_t > _ADMISSIBILITY * radius``.
    _radii: list[float] = field(default_factory=list, repr=False)
    #: flat (n_leaves, 3) leaf centres / (n_leaves,) leaf box sizes
    _leaf_centers: FloatArray | None = field(default=None, repr=False)
    _leaf_sizes: FloatArray | None = field(default=None, repr=False)
    #: (n_leaves,) source radii
    _radii_vec: FloatArray | None = field(default=None, repr=False)
    #: (n_leaves,) True where the leaf holds at least one source
    _has_sources: NDArray[np.bool_] | None = field(default=None, repr=False)
    #: one source-index array per leaf
    _leaf_indices: list[IntArray] = field(default_factory=list, repr=False)
    #: (n_leaves, (p+1)**2) multipole coefficients, one row per leaf
    _multipole_matrix: FloatArray | None = field(default=None, repr=False)
    #: sources/charges the cached ``_multipole_matrix`` was built from.  The
    #: multipole coefficients are a linear functional of the charges, so the
    #: cache is only valid for the very same charge vector (m16000).
    _cache_points: FloatArray | None = field(default=None, repr=False)
    _cache_charges: FloatArray | None = field(default=None, repr=False)
    #: (n_leaves, n_leaves) "leaf boxes are near" mask used by ``_is_near``
    _leaf_near: NDArray[np.bool_] | None = field(default=None, repr=False)
    #: the point set the octree / leaf indices / radii were built from (m16265)
    _setup_points: FloatArray | None = field(default=None, repr=False)
    #: target block size; bounds the (block, n_leaves, 3) intermediate
    block_size: int = 2048

    def setup(self, points: FloatArray, charges: FloatArray | None = None) -> None:
        """Build the octree and precompute multipole coefficients."""
        points = np.asarray(points, dtype=float)
        self.tree = build_octree(points, max_leaf=self.max_leaf)
        self.leaves = collect_leaves(self.tree)
        n_harm = (self.p + 1) ** 2

        self._multipoles = []
        self._radii = _leaf_radii(points, self.leaves)

        centers = np.array([leaf.center for leaf in self.leaves], dtype=float)
        sizes = np.array([leaf.size for leaf in self.leaves], dtype=float)
        indices = [np.asarray(leaf.indices, dtype=np.int64) for leaf in self.leaves]

        multipoles = np.zeros((len(self.leaves), n_harm), dtype=float)
        self._cache_points = None
        self._cache_charges = None
        if charges is not None:
            charges = np.asarray(charges, dtype=float)
            for k, leaf in enumerate(self.leaves):
                idx = indices[k]
                M = _multipole_coefficients(points[idx], charges[idx], leaf.center, self.p)
                multipoles[k] = M
                self._multipoles.append(M)
            self._multipole_matrix = multipoles
            self._cache_points = np.array(points, dtype=float, copy=True)
            self._cache_charges = np.array(charges, dtype=float, copy=True)
        else:
            # a re-``setup`` must not inherit the previous point set's multipoles
            self._multipole_matrix = None

        self._leaf_centers = centers
        self._leaf_sizes = sizes
        self._leaf_indices = indices
        self._radii_vec = np.asarray(self._radii, dtype=float)
        self._has_sources = np.array([len(idx) > 0 for idx in indices], dtype=bool)
        # same predicate as ``_is_near``: centre distance <= 2*max(size_t, size_s)
        leaf_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        self._leaf_near = leaf_dist <= 2.0 * np.maximum(sizes[:, None], sizes[None, :])
        # ``matvec`` trusts this geometry: leaf centres, leaf indices and leaf
        # radii are all derived here, so a source set that differs from it must
        # be rejected instead of silently mixed with the setup partition (m16265).
        self._setup_points = np.array(points, dtype=float, copy=True)

    def _require_setup(self, method: str) -> None:
        """Fail closed: public entry points need a built octree (m16109).

        Before the fix, ``matvec`` on a never-set-up operator returned an
        all-zero vector and ``far_fraction`` reported 0.0 -- both
        indistinguishable from legitimate values, so a forgotten ``setup``
        produced silent wrong answers (m16000 limitation #2).
        """
        if self.tree is None or not self.leaves:
            raise RuntimeError(
                f"FMMOperator.{method}() called before setup(): "
                "call setup(points[, charges]) first"
            )

    def _require_setup_geometry(self, sources: FloatArray) -> None:
        """Fail closed when ``sources`` is not the point set passed to :meth:`setup`.

        Leaf centres, leaf indices and leaf radii are all built from the setup
        point set, so the far field (multipoles about a leaf centre) and the
        near field (``sources[leaf_index]``) are only meaningful for that same
        geometry.  Before m16265 the mismatch was fail-open: a shifted or
        rescaled point set of the same length silently produced answers
        3.7\%--6.8\% away from the dense sum (versus 1.1e-6 for the correct
        geometry), and a point set of a different length raised a bare
        ``IndexError`` from deep inside the near-field loop.
        """
        setup_points = self._setup_points
        if setup_points is None:
            raise RuntimeError(
                "FMMOperator.matvec() called before setup(): call setup(points[, charges]) first"
            )
        if sources.shape != setup_points.shape:
            raise ValueError(
                "sources shape %r does not match the setup point set %r; call "
                "setup() again for a different geometry"
                % (sources.shape, setup_points.shape)
            )
        if not np.array_equal(sources, setup_points):
            raise ValueError(
                "sources differ from the point set passed to setup(); the octree, "
                "leaf indices and leaf radii belong to the setup geometry, so the "
                "operator is only defined for that geometry (call setup() again)"
            )

    def _build_multipoles(
        self, points: FloatArray, charges: FloatArray
    ) -> FloatArray:
        """Multipole coefficients of ``(points, charges)`` about every leaf centre."""
        points = np.asarray(points, dtype=float)
        charges = np.asarray(charges, dtype=float)
        multipoles = np.zeros((len(self.leaves), (self.p + 1) ** 2), dtype=float)
        self._multipoles = []
        for k, leaf in enumerate(self.leaves):
            idx = self._leaf_indices[k]
            M = _multipole_coefficients(points[idx], charges[idx], leaf.center, self.p)
            multipoles[k] = M
            self._multipoles.append(M)
        self._cache_points = np.array(points, dtype=float, copy=True)
        self._cache_charges = np.array(charges, dtype=float, copy=True)
        return multipoles

    def _multipole_cache_valid(
        self, sources: FloatArray, charges: FloatArray
    ) -> bool:
        """Whether the cached multipole coefficients belong to *(sources, charges)*."""
        if self._multipole_matrix is None:
            return False
        cached_points = self._cache_points
        cached_charges = self._cache_charges
        if cached_points is None or cached_charges is None:
            return False
        if cached_points.shape != sources.shape:
            return False
        if cached_charges.shape != charges.shape:
            return False
        return bool(
            np.array_equal(cached_points, sources)
            and np.array_equal(cached_charges, charges)
        )

    # -- routing ---------------------------------------------------------
    def _route(self, targets: FloatArray) -> IntArray:
        """Nearest-leaf-centre routing (same approximation as the scalar path)."""
        centers = self._leaf_centers
        leaf_of_target = np.empty(len(targets), dtype=np.int64)
        step = max(self.block_size, 1)
        for start in range(0, len(targets), step):
            block = targets[start:start + step]
            d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            leaf_of_target[start:start + len(block)] = np.argmin(d2, axis=1)
        return leaf_of_target

    def _near_mask(self, block: FloatArray, leaf_of_target: IntArray) -> tuple:
        """Return ``(rel, r, near)`` for one target block.

        ``rel``/``r`` are the (block, n_leaves, 3) / (block, n_leaves)
        target-to-leaf-centre offsets; ``near`` is the (block, n_leaves) mask of
        target-leaf pairs that must be summed directly, i.e. exactly
        ``not admissible or _is_near(tleaf, sleaf)`` on non-empty leaves.
        """
        centers = self._leaf_centers
        rel = block[:, None, :] - centers[None, :, :]
        r = np.linalg.norm(rel, axis=2)
        admissible = r > _ADMISSIBILITY * np.maximum(self._radii_vec, 1e-15)[None, :]
        near = (~admissible) | self._leaf_near[leaf_of_target]
        near &= self._has_sources[None, :]
        return rel, r, near

    # -- far field -------------------------------------------------------
    def _far_contributions(
        self,
        rel: FloatArray,
        r: FloatArray,
        t_pair: IntArray,
        s_pair: IntArray,
        n_targets: int,
    ) -> FloatArray:
        """Sum every admissible multipole pair into a per-target vector."""
        out = np.zeros(n_targets, dtype=float)
        if len(t_pair) == 0:
            return out

        deg = _harmonic_degrees(self.p)
        scale_exp = -(deg + 1.0)
        multipoles = self._multipole_matrix

        # keep the (pairs, (p+1)**2) intermediates bounded
        chunk = int(max(1, 262_144 // max(len(deg), 1)))
        for start in range(0, len(t_pair), chunk):
            sl = slice(start, start + chunk)
            ii = t_pair[sl]
            ss = s_pair[sl]
            rr = r[ii, ss]
            relf = rel[ii, ss]
            theta = np.arccos(np.clip(relf[:, 2] / rr, -1.0, 1.0))
            phi = np.arctan2(relf[:, 1], relf[:, 0])
            Y = all_real_spherical_harmonics(self.p, theta, phi)
            Y *= np.power(rr[:, None], scale_exp[None, :])
            Y *= multipoles[ss]
            out += np.bincount(ii, weights=Y.sum(axis=1), minlength=n_targets)
        return out

    def matvec(
        self,
        targets: FloatArray,
        sources: FloatArray,
        charges: FloatArray,
    ) -> FloatArray:
        """Compute FMM-accelerated matvec: phi = S @ charges.

        Both the near-field direct sums and the far-field multipole evaluations
        are evaluated in blocks of ``block_size`` targets with NumPy array
        operations; the near/far decision per (target, source-leaf) pair is the
        one documented above (``_ADMISSIBILITY`` times the leaf's own source
        radius, plus :meth:`_is_near`).  Results agree with the scalar reference
        loop up to floating-point summation order.
        """
        targets = np.asarray(targets, dtype=float)
        sources = np.asarray(sources, dtype=float)
        charges = np.asarray(charges, dtype=float)

        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("targets must have shape (N, 3)")
        if sources.ndim != 2 or sources.shape[1] != 3:
            raise ValueError("sources must have shape (M, 3)")
        if charges.shape != (sources.shape[0],):
            raise ValueError("charges must have shape (M,)")
        self._require_setup("matvec")
        self._require_setup_geometry(sources)
        n = len(targets)
        result = np.zeros(n, dtype=float)
        if n == 0:
            return result
        if not self._multipole_cache_valid(sources, charges):
            # ``setup`` may be called without charges (geometry only) and the
            # caller may hand a *different* charge vector to every ``matvec``
            # (multiple right-hand sides, iterative solvers).  The multipole
            # coefficients are linear in the charges, so reusing them for another
            # charge vector silently evaluates the far field of the previous
            # right-hand side -- the operator is then no longer linear (m16000).
            self._multipole_matrix = self._build_multipoles(sources, charges)

        leaf_of_target = self._route(targets)
        step = max(self.block_size, 1)

        for start in range(0, n, step):
            tin = targets[start:start + step]
            nb = len(tin)
            rel, r, near = self._near_mask(tin, leaf_of_target[start:start + step])
            far = ~near
            if far.any():
                t_pair, s_pair = np.nonzero(far)
                result[start:start + nb] += self._far_contributions(
                    rel, r, t_pair, s_pair, nb
                )
            if near.any():
                result[start:start + nb] += self._near_contributions(
                    tin, sources, charges, near
                )
        return result

    def _near_contributions(
        self,
        block: FloatArray,
        sources: FloatArray,
        charges: FloatArray,
        near: NDArray[np.bool_],
        max_pairs: int = 65_536,
    ) -> FloatArray:
        """Direct near-field sums, one flat pass over all near (target, source) pairs.

        Pairs are enumerated leaf-major (ascending leaf index, source order
        inside the leaf preserved) and accumulated per target with
        ``np.bincount``, so each target's near-field sum is accumulated in the
        same order as the scalar reference loop.
        """
        nb = block.shape[0]
        out = np.zeros(nb, dtype=float)
        if nb == 0:
            return out

        # Bound the flat pair arrays: split the target block and recurse.  Only
        # whole targets are separated, so each target's near-field sum is still
        # accumulated leaf-major over its own pairs (order unchanged).
        leaf_sizes = np.array([len(idx) for idx in self._leaf_indices], dtype=np.int64)
        pair_count = int((near.sum(axis=0) * leaf_sizes).sum())
        if pair_count > max_pairs and nb > 1:
            mid = nb // 2
            out[:mid] = self._near_contributions(
                block[:mid], sources, charges, near[:mid], max_pairs
            )
            out[mid:] = self._near_contributions(
                block[mid:], sources, charges, near[mid:], max_pairs
            )
            return out

        tgt_parts: list[IntArray] = []
        src_parts: list[IntArray] = []
        for sleaf in np.flatnonzero(near.any(axis=0)):
            idx = self._leaf_indices[sleaf]
            if len(idx) == 0:
                continue
            rows = np.flatnonzero(near[:, sleaf])
            if len(rows) == 0:
                continue
            tgt_parts.append(np.repeat(rows, len(idx)))
            src_parts.append(np.tile(idx, len(rows)))
        if not tgt_parts:
            return out

        tgt = np.concatenate(tgt_parts)
        src = np.concatenate(src_parts)
        diff = block[tgt] - sources[src]
        r = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        r = np.maximum(r, 1e-15)
        vals = charges[src] / (4.0 * math.pi * r)
        return np.bincount(tgt, weights=vals, minlength=nb)

    def far_fraction(self, targets: FloatArray) -> float:
        """Fraction of target-leaf pairs evaluated with the multipole expansion.

        Requires :meth:`setup`; an unset operator raises ``RuntimeError``
        instead of silently reporting 0.0 (m16109).
        """
        targets = np.asarray(targets, dtype=float)
        self._require_setup("far_fraction")
        if len(targets) == 0:
            return 0.0
        leaf_of_target = self._route(targets)
        near_pairs = 0
        total_pairs = 0
        step = max(self.block_size, 1)
        for start in range(0, len(targets), step):
            tin = targets[start:start + step]
            _, _, near = self._near_mask(tin, leaf_of_target[start:start + step])
            near_pairs += int(near.sum())
            total_pairs += tin.shape[0] * int(self._has_sources.sum())
        if total_pairs == 0:
            return 0.0
        return 1.0 - near_pairs / total_pairs

    def _is_near(self, tleaf: OctreeNode, sleaf: OctreeNode) -> bool:
        """Determine if two leaves are in the near-field."""
        dist = float(np.linalg.norm(tleaf.center - sleaf.center))
        theta = 2.0
        return dist <= theta * max(tleaf.size, sleaf.size)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def fmm_laplace_matvec(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
    p: int = 4,
    max_leaf: int = 128,
) -> FloatArray:
    """FMM-accelerated single-layer matvec for the Laplace kernel."""
    targets = np.asarray(targets, dtype=float)
    sources = np.asarray(sources, dtype=float)
    charges = np.asarray(charges, dtype=float)

    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets must have shape (N, 3)")
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("sources must have shape (M, 3)")
    if charges.shape != (sources.shape[0],):
        raise ValueError("charges must have shape (M,)")

    op = FMMOperator(p=p, max_leaf=max_leaf)
    op.setup(sources, charges)
    return op.matvec(targets, sources, charges)


def direct_laplace_matvec(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
) -> FloatArray:
    """Direct (dense) single-layer matvec for the Laplace kernel."""
    targets = np.asarray(targets, dtype=float)
    sources = np.asarray(sources, dtype=float)
    charges = np.asarray(charges, dtype=float)
    return _direct_laplace_matvec_batch(targets, sources, charges)


# ---------------------------------------------------------------------------
# Helmholtz kernel support (fail-closed)
# ---------------------------------------------------------------------------

def helmholtz_supported(ka: float, ka_max: float = 10.0) -> bool:
    """Check if Helmholtz FMM is supported for the given ka."""
    return ka <= ka_max


def fmm_helmholtz_matvec(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
    k: float,
    p: int = 4,
    ka_max: float = 10.0,
) -> tuple[FloatArray, Literal["fmm", "direct", "not_supported"]]:
    """FMM-accelerated matvec for Helmholtz kernel e^{ikr}/(4*pi*r).

    Fail-closed: when ka > ka_max, returns NOT_SUPPORTED.
    """
    targets = np.asarray(targets, dtype=float)
    sources = np.asarray(sources, dtype=float)
    charges = np.asarray(charges, dtype=float)

    extent = float(np.max(np.max(sources, axis=0) - np.min(sources, axis=0)))
    ka = k * extent

    if not helmholtz_supported(ka, ka_max):
        pot = _helmholtz_direct(targets, sources, charges, k)
        return pot, "not_supported"

    if ka < 0.1:
        pot = fmm_laplace_matvec(targets, sources, charges, p=p)
        return pot, "fmm"

    pot = _helmholtz_direct(targets, sources, charges, k)
    return pot, "direct"


def _helmholtz_direct(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
    k: float,
) -> FloatArray:
    """Direct Helmholtz matvec: sum_j q_j * e^{ikr} / (4*pi*r)."""
    diff = targets[:, None, :] - sources[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.maximum(r, 1e-15)
    phase = np.exp(1j * k * r)
    result = np.sum(
        charges[None, :] * (phase / (4.0 * math.pi * r)).real, axis=1
    )
    return result


def fmm_degenerate_matvec(
    targets: FloatArray,
    sources: FloatArray,
    charges: FloatArray,
) -> FloatArray:
    """FMM with p=0 (monopole only)."""
    return fmm_laplace_matvec(targets, sources, charges, p=0)
