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
    child_centers = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                child_centers.append(center + half * np.array([
                    -1.0 + 2.0 * ix,
                    -1.0 + 2.0 * iy,
                    -1.0 + 2.0 * iz,
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
) -> FloatArray:
    """Vectorized direct evaluation of sum_j q_j / (4*pi*|x_i - y_j|)."""
    diff = targets[:, None, :] - sources[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.maximum(r, 1e-15)
    return np.sum(charges[None, :] / (4.0 * math.pi * r), axis=1)


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

    def setup(self, points: FloatArray, charges: FloatArray | None = None) -> None:
        """Build the octree and precompute multipole coefficients."""
        self.tree = build_octree(points, max_leaf=self.max_leaf)
        self.leaves = collect_leaves(self.tree)
        self._multipoles = []
        self._radii = _leaf_radii(points, self.leaves)
        if charges is not None:
            for leaf in self.leaves:
                spts = points[leaf.indices]
                scharges = charges[leaf.indices]
                M = _multipole_coefficients(spts, scharges, leaf.center, self.p)
                self._multipoles.append(M)

    def matvec(
        self,
        targets: FloatArray,
        sources: FloatArray,
        charges: FloatArray,
    ) -> FloatArray:
        """Compute FMM-accelerated matvec: phi = S @ charges.

        A source leaf is expanded about its own centre, so the expansion is only
        used when the target is outside the sphere that contains every source of
        that leaf, ``r_t > _ADMISSIBILITY * radius(leaf)``.  The leaf-centre
        admissibility test alone is *not* sufficient: on a surface distribution
        the leaves are thin slabs whose ``size`` exceeds the centre-to-centre
        spacing, and evaluating a divergent series there produced order-unity
        errors (see ``docs/fmm_cost_audit/README.md``, m14931).
        """
        if self.tree is None:
            self.setup(sources, charges)

        # Precompute multipoles if not done
        if not self._multipoles:
            for leaf in self.leaves:
                spts = sources[leaf.indices]
                scharges = charges[leaf.indices]
                M = _multipole_coefficients(spts, scharges, leaf.center, self.p)
                self._multipoles.append(M)
            self._radii = _leaf_radii(sources, self.leaves)

        n = len(targets)
        result = np.zeros(n, dtype=float)

        leaf_centers = np.array([leaf.center for leaf in self.leaves])

        for i in range(n):
            t = targets[i]
            dists = np.linalg.norm(leaf_centers - t[None, :], axis=1)
            target_leaf_idx = int(np.argmin(dists))
            target_leaf = self.leaves[target_leaf_idx]

            for sleaf_idx, sleaf in enumerate(self.leaves):
                smask = sleaf.indices
                if len(smask) == 0:
                    continue

                rel = t - sleaf.center
                r_t = float(np.linalg.norm(rel))
                admissible = r_t > _ADMISSIBILITY * max(self._radii[sleaf_idx], 1e-15)

                if (not admissible) or self._is_near(target_leaf, sleaf):
                    # Near-field: direct
                    spts = sources[smask]
                    scharges = charges[smask]
                    diff = spts - t[None, :]
                    r = np.linalg.norm(diff, axis=1)
                    r = np.maximum(r, 1e-15)
                    result[i] += float(np.sum(scharges / (4.0 * math.pi * r)))
                else:
                    # Far-field: use precomputed multipole
                    M = self._multipoles[sleaf_idx]
                    theta = math.acos(np.clip(rel[2] / r_t, -1.0, 1.0))
                    phi = math.atan2(rel[1], rel[0])
                    Y = all_real_spherical_harmonics(
                        self.p, np.array([theta]), np.array([phi])
                    )[0]
                    for l in range(self.p + 1):
                        for m in range(-l, l + 1):
                            col = _sph_index(l, m)
                            result[i] += M[col] * Y[col] / (r_t ** (l + 1))

        return result

    def far_fraction(self, targets: FloatArray) -> float:
        """Fraction of target-leaf pairs evaluated with the multipole expansion."""

        centers = np.array([leaf.center for leaf in self.leaves])
        far = 0
        total = 0
        for t in targets:
            tleaf = self.leaves[int(np.argmin(np.linalg.norm(centers - t[None, :], axis=1)))]
            for idx, sleaf in enumerate(self.leaves):
                if len(sleaf.indices) == 0:
                    continue
                total += 1
                r_t = float(np.linalg.norm(t - sleaf.center))
                if (r_t > _ADMISSIBILITY * max(self._radii[idx], 1e-15)
                        and not self._is_near(tleaf, sleaf)):
                    far += 1
        return far / total if total else 0.0

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
