"""Near-singular panel quadrature for field evaluation close to the surface.

Why this module exists
----------------------
:func:`tensorbem.solver3d.evaluate_surface_field` evaluates the Green
representation

    u(x) = sum_j [ q_j * int_{T_j} G dS + u_j * int_{T_j} (dG/dn) dS ]

with a *fixed-order* rule on every panel: Duffy-transformed Gauss quadrature for
the weakly singular single-layer kernel ``G = 1/(4*pi*r)`` and the closed-form
signed solid angle for the double-layer kernel.  Both are spectrally accurate
while the target is far from the panel, but the Duffy rule cannot resolve the
peak of ``1/r`` once the target approaches the panel, so the public API
*rejects* targets closer than ``minimum_clearance_factor * h_max``
(default ``0.05``).  That exclusion removes exactly the near-wall quantities a
ship-hydrodynamics user needs: off-body pressure, wave elevation just above the
free surface, near-field velocity.

The fix implemented here is an h-refined, target-adaptive rule.  Each panel is
bisected by its midpoints into four similar sub-triangles, recursively, until a
sub-triangle is *well separated* from the target

    distance(target, sub-triangle) >= split_ratio * size(sub-triangle),

at which point the ordinary Gauss-Duffy rule is spectrally accurate on it, or
until the refinement budget ``max_level`` is spent.  Only the shrinking chain of
sub-triangles that actually contains the singular peak is refined, so the work
per target grows logarithmically in ``h/d`` instead of linearly, and -- this is
the point -- the quadrature error stops depending on the clearance.

The double-layer term never enters this module: its constant-density integral is
the closed-form signed solid angle, which is already exact at arbitrarily small
clearance (only its *gradient* is hypersingular, see
:func:`tensorbem.solver3d.evaluate_surface_velocity_near_surface`).

Both the potential kernel and its gradient are provided:

* :func:`single_layer_integrals`   -- ``int_{T_j} G dS``
* :func:`single_layer_gradients`   -- ``int_{T_j} grad_x G dS``

Both return one value per input triangle, so a caller can accumulate them
against its own density vector.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .kernels3d import duffy_quadrature_batch

FloatArray = NDArray[np.float64]

_FOUR_PI = 4.0 * np.pi


def _as_triangles(v0: FloatArray, v1: FloatArray, v2: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if v0.ndim != 2 or v0.shape[1] != 3 or v1.shape != v0.shape or v2.shape != v0.shape:
        raise ValueError("v0, v1, and v2 must all have shape (F, 3)")
    if not (np.isfinite(v0).all() and np.isfinite(v1).all() and np.isfinite(v2).all()):
        raise ValueError("triangle vertices must all be finite")
    if np.any(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) <= 0.0):
        raise ValueError("triangles must be non-degenerate")
    return v0, v1, v2


def _as_target(target: FloatArray) -> FloatArray:
    x = np.asarray(target, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("target must be a finite array of shape (3,)")
    return x


def _validate_order(order: int) -> int:
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise TypeError("order must be an integer")
    if order < 2:
        raise ValueError("order must be at least 2")
    return int(order)


def _validate_split_ratio(split_ratio: float) -> float:
    ratio = float(split_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("split_ratio must be a positive finite number")
    return ratio


def _validate_max_level(max_level: int) -> int:
    if isinstance(max_level, (bool, np.bool_)) or not isinstance(max_level, (int, np.integer)):
        raise TypeError("max_level must be an integer")
    if max_level < 0:
        raise ValueError("max_level must be non-negative")
    return int(max_level)


def _segment_distances(x: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    """Distance from one point ``x`` to each segment ``a[i] -> b[i]``."""

    edge = b - a
    denominator = np.einsum("ij,ij->i", edge, edge)
    parameter = np.einsum("ij,ij->i", x - a, edge) / denominator
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = a + parameter[:, None] * edge
    return np.linalg.norm(x - closest, axis=1)


def triangle_distances_and_sizes(
    target: FloatArray, v0: FloatArray, v1: FloatArray, v2: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Distance from ``target`` to each triangle, and each triangle's size.

    The size is the longest edge, which is what the refinement criterion needs
    (it bounds the diameter of the sub-triangle up to a factor two).  The
    distance is the true point-to-triangle distance: the plane distance when the
    projection falls inside the triangle, otherwise the distance to the nearest
    edge.
    """

    x = _as_target(target)
    v0, v1, v2 = _as_triangles(v0, v1, v2)

    edges = np.stack([v1 - v0, v2 - v1, v0 - v2], axis=1)
    sizes = np.linalg.norm(edges, axis=2).max(axis=1)

    normal = np.cross(v1 - v0, v2 - v0)
    normal_norm = np.linalg.norm(normal, axis=1)
    normal_hat = normal / normal_norm[:, None]
    plane_distance = np.abs(np.einsum("ij,ij->i", x[None, :] - v0, normal_hat))

    # Barycentric coordinates of the projection of the target onto the plane,
    # via the dot-product form (robust for the sliver-free meshes used here).
    w = x[None, :] - v0
    d00 = np.einsum("ij,ij->i", v1 - v0, v1 - v0)
    d01 = np.einsum("ij,ij->i", v1 - v0, v2 - v0)
    d11 = np.einsum("ij,ij->i", v2 - v0, v2 - v0)
    d20 = np.einsum("ij,ij->i", w, v1 - v0)
    d21 = np.einsum("ij,ij->i", w, v2 - v0)
    denominator = d00 * d11 - d01 * d01
    beta = (d11 * d20 - d01 * d21) / denominator
    gamma = (d00 * d21 - d01 * d20) / denominator
    alpha = 1.0 - beta - gamma
    inside = (alpha >= 0.0) & (beta >= 0.0) & (gamma >= 0.0)

    edge_distance = np.minimum.reduce(
        [
            _segment_distances(x, v0, v1),
            _segment_distances(x, v1, v2),
            _segment_distances(x, v2, v0),
        ]
    )
    return np.where(inside, plane_distance, edge_distance), sizes


def _bisect(
    v0: FloatArray, v1: FloatArray, v2: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Split each triangle into four by its midpoints.

    The children are stacked *blockwise*: all first children, then all second
    children, and so on.  Callers must map a child back to its parent with
    ``np.tile(parents, 4)``, not ``np.repeat``.
    """

    m01 = 0.5 * (v0 + v1)
    m12 = 0.5 * (v1 + v2)
    m20 = 0.5 * (v2 + v0)
    children_v0 = np.concatenate([v0, m01, m20, m12], axis=0)
    children_v1 = np.concatenate([m01, v1, m12, m20], axis=0)
    children_v2 = np.concatenate([m20, m12, v2, m01], axis=0)
    return children_v0, children_v1, children_v2


def _adaptive_accumulate(
    target: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    order: int,
    split_ratio: float,
    max_level: int,
    kernel: str,
) -> FloatArray:
    """Core adapt-until-well-separated loop shared by both kernels."""

    x = _as_target(target)
    v0, v1, v2 = _as_triangles(v0, v1, v2)
    order = _validate_order(order)
    ratio = _validate_split_ratio(split_ratio)
    level_budget = _validate_max_level(max_level)
    if kernel == "potential":
        shape = ()
    elif kernel == "gradient":
        shape = (3,)
    else:  # pragma: no cover - internal invariant
        raise ValueError("kernel must be 'potential' or 'gradient'")

    total = np.zeros((v0.shape[0],) + shape, dtype=float)
    parents = np.arange(v0.shape[0])
    level = 0
    while parents.size:
        distances, sizes = triangle_distances_and_sizes(x, v0, v1, v2)
        refine = (distances < ratio * sizes) & (level < level_budget)
        if np.any(~refine):
            points, weights = duffy_quadrature_batch(
                v0[~refine], v1[~refine], v2[~refine], order=order
            )
            delta = x[None, None, :] - points
            radius = np.linalg.norm(delta, axis=2)
            if kernel == "potential":
                contribution = np.sum(weights / radius, axis=1) / _FOUR_PI
                np.add.at(total, parents[~refine], contribution)
            else:
                contribution = (
                    -np.sum((weights / radius**3)[:, :, None] * delta, axis=1)
                    / _FOUR_PI
                )
                np.add.at(total, parents[~refine], contribution)
        if not np.any(refine):
            break
        c0, c1, c2 = _bisect(v0[refine], v1[refine], v2[refine])
        parents = np.tile(parents[refine], 4)
        v0, v1, v2 = c0, c1, c2
        level += 1
    return total


def single_layer_integrals(
    target: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    order: int = 8,
    split_ratio: float = 1.0,
    max_level: int = 24,
) -> FloatArray:
    """``int_{T_j} G dS`` with ``G = 1/(4*pi*|x-y|)``, valid down to the surface.

    Returns an array of shape ``(F,)``, one value per input triangle.  Panels
    that are well separated from ``target`` (``distance >= split_ratio * size``)
    are integrated with the ordinary Duffy-Gauss rule of order ``order``; the
    rest are bisected recursively, so the error no longer degrades as the target
    approaches the surface.
    """

    return _adaptive_accumulate(
        target, v0, v1, v2, order, split_ratio, max_level, "potential"
    )


def single_layer_gradients(
    target: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    order: int = 8,
    split_ratio: float = 1.0,
    max_level: int = 24,
) -> FloatArray:
    """``int_{T_j} grad_x G dS`` with ``grad_x G = -(x-y)/(4*pi*|x-y|**3)``.

    Returns an array of shape ``(F, 3)``.  Same adaptive refinement as
    :func:`single_layer_integrals`; this is the kernel that the near-surface
    velocity needs, and it is only weakly singular (the ``1/r**2`` peak is
    integrable because the area element carries ``r dr``).
    """

    return _adaptive_accumulate(
        target, v0, v1, v2, order, split_ratio, max_level, "gradient"
    )