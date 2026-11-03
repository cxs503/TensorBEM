r"""Exact single-layer panel integrals for near-surface field evaluation in 3-D.

Motivation
----------
The constant-panel 3-D solver stores one source strength per flat triangle.  To
recover the potential (or its gradient) at a field point ``x`` it therefore
needs the panel integrals

    I(x) = \int_{T} G(x,y) dS_y        = (1/4\pi) \int_T dS_y/|x-y|
    J(x) = \int_{T} grad_x G(x,y) dS_y  = -(1/4\pi) \int_T (x-y)/|x-y|^3 dS_y

for every panel ``T``.  A fixed-order Duffy rule resolves the singularity only
at the panel vertex it is attached to, so it silently loses accuracy when
``x`` approaches a panel: measured on a panel of size ``h``, an 8x8 Duffy rule
is off by ``3.8e-3`` relative at clearance ``0.24 h``, ``1.9e-2`` at
``0.09 h`` and ``9.8e-2`` at ``0.005 h`` (see
``docs/near_singular_field_3d.md``).  For that reason the field evaluators used
to refuse points closer than ``0.05 h_max``.

On a *flat* panel the same integrals have an exact one-dimensional form.  Let
``P`` be the orthogonal projection of ``x`` onto the panel plane,
``h = |x - P|`` the clearance, ``rho_vec = y - P`` the in-plane radius vector
(with ``R = |x-y| = sqrt(rho^2 + h^2)``), and ``n'_e`` the outward in-plane unit
normal of panel edge ``e``.  The two-dimensional divergence theorem in the
panel plane gives

    \int_T dS/R           = \sum_e \int_e (rho_vec . n'_e) / (R + h) ds
    \int_T rho_vec/R^3 dS = -\sum_e \int_e n'_e / R ds
    \int_T dS/R^3         = (1/h) \sum_e \int_e (rho_vec . n'_e)/(R (R+h)) ds

(from ``div'(rho_vec (R-h)/rho^2) = 1/R``, ``grad'(1/R) = -rho_vec/R^3`` and
``lap'(ln(R+h)) = h/R^3``).  Every edge integrand is smooth -- the apparent
``1/rho`` factors cancel because ``rho_vec . n'_e`` vanishes linearly where
``rho -> 0`` -- so Gauss-Legendre along the three edges converges spectrally.
The three integrals combine into the 3-D kernels as

    \int_T (y-x)/R^3 dS = \int_T rho_vec/R^3 dS + (P - x) \int_T dS/R^3 ,

and ``J = -(1/4\pi) \int_T (y-x)/R^3 dS`` follows by sign.

This yields machine-precision panel integrals at any clearance for the cost of
``3 * order`` kernel evaluations per panel (``order = 24`` by default), the same
order of work as the 8x8 Duffy rule it replaces.

Conventions match :mod:`tensorbem.kernels3d`: ``G = 1/(4 pi r)``,
``grad_x G = -(x-y)/(4 pi r^3)``, and panel vertices are ordered
counter-clockwise about the outward normal.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_FOUR_PI = 4.0 * np.pi
# Clearance (absolute) below which the normal component of the gradient kernel
# is dropped: that component carries a factor ``h`` in front of ``\int dS/R^3``,
# so the product is well defined at ``h = 0``; this only avoids 0/0 in floating
# point.
_H_EPS = 1.0e-14


def _validate_order(order: int) -> int:
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be an integer >= 2")
    if order < 2:
        raise ValueError("order must be an integer >= 2")
    return int(order)


def panel_edge_frame(
    v0: FloatArray, v1: FloatArray, v2: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    r"""Return ``(starts, ends, outward, lengths, normal)`` per panel.

    ``starts``, ``ends``, ``outward`` and ``lengths`` have shape ``(F, 3, ...)``
    -- one entry per panel edge, ordered ``v0->v1``, ``v1->v2``, ``v2->v0``.
    ``outward`` is the unit in-plane edge normal ``m x n`` pointing out of the
    triangle, where ``m`` is the unit edge direction and ``n`` the panel normal.
    """
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if v0.ndim != 2 or v0.shape[1] != 3 or v1.shape != v0.shape or v2.shape != v0.shape:
        raise ValueError("v0, v1, and v2 must all have shape (F, 3)")
    if not (np.isfinite(v0).all() and np.isfinite(v1).all() and np.isfinite(v2).all()):
        raise ValueError("triangle vertices must all be finite")
    cross = np.cross(v1 - v0, v2 - v0)
    double_areas = np.linalg.norm(cross, axis=1)
    if np.any(double_areas <= 0.0):
        raise ValueError("triangle must be non-degenerate")
    normal = cross / double_areas[:, None]
    starts = np.stack((v0, v1, v2), axis=1)
    ends = np.stack((v1, v2, v0), axis=1)
    edge = ends - starts
    lengths = np.linalg.norm(edge, axis=2)
    direction = edge / lengths[:, :, None]
    outward = np.cross(direction, normal[:, None, :])
    return starts, ends, outward, lengths, normal


def integrate_single_layer_flat_triangles(
    target: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    order: int = 24,
) -> tuple[FloatArray, FloatArray]:
    r"""Exact single-layer panel integrals of flat triangles seen from ``target``.

    Returns ``(potential, gradient)`` with ``potential[j] = \int_{T_j} G dS`` and
    ``gradient[j] = \int_{T_j} grad_x G dS`` (shapes ``(F,)`` and ``(F, 3)``).
    The results hold at any clearance, including points arbitrarily close to or
    coplanar with the panel, provided ``target`` does not lie exactly on a panel
    edge (where ``\int rho_vec/R^3`` has a logarithmic endpoint singularity).
    """
    x = np.asarray(target, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("target must be a finite array of shape (3,)")
    starts, ends, outward, lengths, normal = panel_edge_frame(v0, v1, v2)
    order = _validate_order(order)
    nodes, weights = leggauss(order)
    s = 0.5 * (nodes + 1.0)
    w = 0.5 * weights

    h_signed = np.einsum("ij,ij->i", x[None, :] - starts[:, 0, :], normal)
    projection = x[None, :] - h_signed[:, None] * normal
    h = np.abs(h_signed)
    h_vector = projection - x[None, :]

    points = starts[:, :, None, :] + s[None, None, :, None] * (ends - starts)[:, :, None, :]
    rho = points - projection[:, None, None, :]
    rho_n = np.einsum("fekj,fekj->fek", rho, outward[:, :, None, :])
    r2 = np.einsum("fekj,fekj->fek", rho, rho)
    radius = np.sqrt(r2 + (h * h)[:, None, None])
    scale = lengths[:, :, None] * w[None, None, :]

    potential = np.sum(scale * rho_n / (radius + h[:, None, None]), axis=(1, 2))
    gradient_in_plane = -np.sum(
        scale[:, :, :, None] * outward[:, :, None, :] / radius[:, :, :, None],
        axis=(1, 2),
    )
    normal_measure = np.sum(
        scale * rho_n / (radius * (radius + h[:, None, None])), axis=(1, 2)
    )
    normal_component = np.where(
        h[:, None] > _H_EPS, h_vector * (normal_measure / np.maximum(h, _H_EPS))[:, None], 0.0
    )
    # grad_x G = (y-x)/(4 pi R^3), so the two pieces add directly.
    gradient = gradient_in_plane + normal_component
    return potential / _FOUR_PI, gradient / _FOUR_PI


def integrate_single_layer_flat_triangles_batch(
    targets: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    order: int = 24,
) -> tuple[FloatArray, FloatArray]:
    """Same as :func:`integrate_single_layer_flat_triangles` for ``M`` targets.

    Returns ``(potential, gradient)`` with shapes ``(M, F)`` and ``(M, F, 3)``.
    """
    points = np.asarray(targets, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("targets must have shape (M, 3)")
    if not np.isfinite(points).all():
        raise ValueError("targets must all be finite")
    panel_count = np.asarray(v0).shape[0]
    potentials = np.empty((points.shape[0], panel_count), dtype=float)
    gradients = np.empty((points.shape[0], panel_count, 3), dtype=float)
    for index, point in enumerate(points):
        potentials[index], gradients[index] = integrate_single_layer_flat_triangles(
            point, v0, v1, v2, order=order
        )
    return potentials, gradients


def polar_reference_integrals(
    target: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
    angular_order: int = 200,
) -> tuple[FloatArray, FloatArray]:
    r"""Independent (slow) reference for the flat-panel single-layer integrals.

    Used only to cross-validate :func:`integrate_single_layer_flat_triangles`;
    the mathematics is different, so a shared convention error cannot cancel.
    The triangle is split into the three signed subtriangles
    ``(P, v_k, v_{k+1})`` about the projection ``P`` of ``target``; each is a
    convex sector in which a ray from ``P`` meets exactly one edge, and the
    radial integrals are analytic,

        \int_0^{t} rho d rho / sqrt(rho^2+h^2)       = sqrt(t^2+h^2) - h,
        \int_0^{t} rho d rho / (rho^2+h^2)^{3/2}     = 1/h - 1/sqrt(t^2+h^2),

    so only the angular integral is quadratured (Gauss-Legendre per subtriangle
    sector).  Signed sectors make the identity valid for projections outside
    the triangle as well.

    ``angular_order`` nodes per sector give roughly ten correct digits for a
    well-conditioned geometry; the routine is meant for tests, not production.
    """
    x = np.asarray(target, dtype=float)
    starts, _, _, _, normal = panel_edge_frame(v0, v1, v2)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("target must be a finite array of shape (3,)")
    order = _validate_order(angular_order)
    nodes, weights = leggauss(order)
    s_nodes = 0.5 * (nodes + 1.0)
    s_weights = 0.5 * weights
    panel_count = starts.shape[0]
    potentials = np.empty(panel_count, dtype=float)
    gradients = np.empty((panel_count, 3), dtype=float)
    vertices = np.stack((v0, v1, v2), axis=1)
    for index in range(panel_count):
        a = vertices[index, 0]
        h_signed = float(np.dot(x - a, normal[index]))
        h = abs(h_signed)
        projection = x - h_signed * normal[index]
        e1 = vertices[index, 1] - a
        e1 = e1 - np.dot(e1, normal[index]) * normal[index]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal[index], e1)
        potential = 0.0
        radial = np.zeros(2)
        normal_measure = 0.0
        for k in range(3):
            p1 = vertices[index, k] - projection
            p2 = vertices[index, (k + 1) % 3] - projection
            q1 = np.array([p1 @ e1, p1 @ e2])
            q2 = np.array([p2 @ e1, p2 @ e2])
            theta1 = np.arctan2(q1[1], q1[0])
            delta = np.arctan2(q1[0] * q2[1] - q1[1] * q2[0], q1 @ q2)
            if abs(delta) <= 1.0e-14:
                continue
            thetas = theta1 + delta * s_nodes
            directions = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)
            # ray / segment intersection: P + t*d = q1 + s*(q2-q1), t > 0, s in [0,1]
            edge = q2 - q1
            denominator = directions[:, 0] * edge[1] - directions[:, 1] * edge[0]
            t = (q1[0] * edge[1] - q1[1] * edge[0]) / denominator
            inner = np.sqrt(t * t + h * h)
            potential += delta * np.sum(s_weights * (inner - h))
            if h > 0.0:
                measure = 1.0 / h - 1.0 / inner
                radial_measure = np.arcsinh(t / h) - t / inner
                radial += delta * np.array(
                    [np.sum(s_weights * radial_measure * np.cos(thetas)),
                     np.sum(s_weights * radial_measure * np.sin(thetas))]
                )
                normal_measure += delta * np.sum(s_weights * measure)
        potentials[index] = potential / _FOUR_PI
        if h > 0.0:
            in_plane = radial[0] * e1 + radial[1] * e2
            gradients[index] = (in_plane + (projection - x) * normal_measure) / _FOUR_PI
        else:
            gradients[index] = np.nan
    return potentials, gradients
