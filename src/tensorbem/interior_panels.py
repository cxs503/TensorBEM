"""Interior (waterplane-lid) panel sets for 3-D irregular-frequency suppression.

Motivation
----------
The exterior Neumann boundary-integral equation of
:mod:`tensorbem.free_surface_solver3d` loses uniqueness at discrete *irregular
frequencies* ``nu_k`` of a surface-piercing hull (John 1950). At those
wavenumbers the interior Dirichlet problem of the hull interior has a
non-trivial solution, the assembled matrix becomes nearly singular, the
radiation damping can change sign, and a frequency sweep of the RAO shows
unphysical spikes.

Green's second identity in the *exterior* fluid domain ``Omega`` gives, for any
field point ``x`` that lies strictly **inside** the hull (and therefore outside
``Omega``), the null-field relation

    0 = int_{Gamma_B} [ phi dG/dn - G dphi/dn ] dS
      = -H_nu(x, Gamma_B) phi + S_nu(x, Gamma_B) q_e,

with ``G`` the free-surface Green function of
:mod:`tensorbem.free_surface3d`.  Note that no lid integral appears: the
interior waterplane ``z = 0`` inside the hull is *not* part of ``dOmega``, so
the identity holds for interior points at any depth.  Stacking these exact rows
onto the body rows over-determines the system, and the least-squares solution
is unique again.

What this module adds
---------------------
The classical remedy for ships (Ohmatsu 1994; Zhu 1993) discretises the
**interior free surface** -- a "lid" over the waterplane inside the hull -- and
uses that panel set to regularise the equation.  This module produces that
interior panel set: a concentric-ring discretisation of the interior
waterplane, shrunk away from the waterline and lowered slightly below ``z = 0``
so every point is strictly inside the hull.  The centroids of those interior
panels are the null-field collocation points handed to the solver via the
``interior_panels=`` keyword, which turns the scattered 5-point CHIEF set into a
*distributed* lid that covers the whole interior waterplane and therefore damps
the interior resonances of several modes at once instead of relying on a lucky
point placement.

The panel set is validated geometrically (:func:`interior_panel_points` raises
when a point escapes the hull), so a lid can never silently be placed in the
fluid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry3d import SurfaceMesh3D, TriangleMesh, WettedHullMesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_DEFAULT_RINGS = 3
_DEFAULT_AZIMUTH = 8
_DEFAULT_SHRINK = 0.75
_DEFAULT_DEPTH_FRACTION = 0.05


@dataclass(frozen=True)
class InteriorPanelSet:
    """Interior waterplane lid used as a null-field collocation surface."""

    points: FloatArray
    """``(K, 3)`` interior collocation points (panel centroids), ``z < 0``."""

    areas: FloatArray
    """``(K,)`` interior panel areas (diagnostic only; rows are unweighted)."""

    waterline_radius: float
    """Radius of the waterline polygon used as the lid boundary (m)."""

    shrink: float
    """Radial shrink factor applied so that no point touches the hull."""

    depth: float
    """Depth of the lid below the waterplane, positive (m)."""

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


def _waterline_circle(mesh: SurfaceMesh3D) -> tuple[FloatArray, float]:
    """Return ``(centre_xy, inscribed_radius)`` of the hull waterplane polygon.

    The lid is a circular disc inscribed in the waterplane polygon, so a
    non-circular (e.g. elliptical) waterplane is covered by the largest disc
    that stays strictly inside it.  Requiring the lid to fit inside the polygon
    is what keeps every interior point inside the hull.
    """
    if not isinstance(mesh, WettedHullMesh):
        raise ValueError(
            "interior panels require a WettedHullMesh (a hull that pierces z = 0); "
            "a closed TriangleMesh has no interior waterplane"
        )
    edges = mesh.waterline_edges
    if edges.shape[0] < 3:
        raise ValueError("waterline has fewer than three edges")
    idx = np.unique(edges.ravel())
    xy = mesh.vertices[idx][:, :2]
    centre = xy.mean(axis=0)
    delta = xy - centre
    radius = np.hypot(delta[:, 0], delta[:, 1])
    if not np.isfinite(radius).all() or float(radius.max()) <= 0.0:
        raise ValueError("degenerate waterline polygon")
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    # A star-shaped waterplane is required; reject a polygon whose angular
    # coverage has a gap wider than pi about its centroid.
    ordered = np.sort(angles)
    gaps = np.diff(np.concatenate((ordered, [ordered[0] + 2.0 * math.pi])))
    if float(np.max(gaps)) > math.pi:
        raise ValueError("waterline polygon is not star-shaped about its centroid")
    inscribed = float(np.min(radius))
    if inscribed <= 0.0:
        raise ValueError("degenerate waterline polygon")
    return centre, inscribed


def _point_inside_hull(points: FloatArray, mesh: SurfaceMesh3D) -> NDArray[np.bool_]:
    """Ray-cast point-in-solid test against the hull closed by its waterplane mirror.

    Three generic ray directions are used and the majority vote is returned, so
    a ray that grazes an edge or a vertex cannot flip the classification.
    """
    refl = np.array([1.0, 1.0, -1.0])
    # The wetted hull is open at z = 0; mirroring it about the waterplane closes
    # the body, which makes the crossing parity well defined below the waterline.
    v0 = np.vstack((mesh.v0, mesh.v0 * refl))
    v1 = np.vstack((mesh.v1, mesh.v2 * refl))
    v2 = np.vstack((mesh.v2, mesh.v1 * refl))
    e1 = v1 - v0
    e2 = v2 - v0

    directions = np.array(
        [
            [0.5773502691896258, 0.4472135954999579, 0.6793900969003611],
            [-0.3015113445777636, 0.9045340337332909, 0.3015113445777636],
            [0.2672612419124244, -0.5345224838248488, 0.8017837257372732],
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    votes = np.zeros(points.shape[0], dtype=np.int64)
    for direction in directions:
        h = np.cross(np.broadcast_to(direction, e2.shape), e2)
        a = np.einsum("pk,pk->p", e1, h)
        ok = np.abs(a) > 1.0e-14
        f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        for idx, p in enumerate(points):
            # Moller-Trumbore, counting crossings along ``direction``.
            s = p[None, :] - v0
            u = f * np.einsum("pk,pk->p", s, h)
            q = np.cross(s, e1)
            v = f * np.einsum("pk,k->p", q, direction)
            t = f * np.einsum("pk,pk->p", e2, q)
            tol = 1.0e-12
            hits = (
                ok
                & (u >= -tol)
                & (u <= 1.0 + tol)
                & (v >= -tol)
                & (u + v <= 1.0 + tol)
                & (t > 1.0e-12)
            )
            if int(np.count_nonzero(hits)) % 2 == 1:
                votes[idx] += 1
    return votes >= 2


def interior_panel_points(
    mesh: SurfaceMesh3D,
    *,
    rings: int = _DEFAULT_RINGS,
    azimuth: int = _DEFAULT_AZIMUTH,
    shrink: float = _DEFAULT_SHRINK,
    depth_fraction: float = _DEFAULT_DEPTH_FRACTION,
    min_depth_panels: float = 1.5,
    validate: bool = True,
) -> InteriorPanelSet:
    """Return the interior waterplane lid panel set of a surface-piercing hull.

    Parameters
    ----------
    mesh :
        Wetted (``z <= 0``) hull that pierces ``z = 0``.
    rings :
        Number of concentric rings of interior panels (>= 1).  Ring ``k`` holds
        ``max(4, round(azimuth * k / rings))`` panels, so the interior spacing
        stays roughly uniform and the lid resolves the low-order interior modes.
    azimuth :
        Number of panels on the outermost ring.
    shrink :
        Fraction of the inscribed waterline radius kept by the outermost ring;
        the value must be in ``(0, 1)`` so no lid point touches the hull.
    depth_fraction :
        Lid depth below ``z = 0`` as a fraction of ``shrink * radius``.
    min_depth_panels :
        Lower bound on the lid depth in units of the mean panel height
        ``h_mean``; keeps the null-field rows well conditioned as the mesh is
        refined.
    validate :
        Run the point-in-solid test and raise if a lid point escaped the hull.
        Enabling it costs one ray cast per point (cheap, default on).

    Returns
    -------
    InteriorPanelSet
        With ``count`` centroids on the interior waterplane, all at ``z < 0``.
    """
    if isinstance(rings, (bool, np.bool_)) or int(rings) != rings or int(rings) < 1:
        raise ValueError("rings must be an integer >= 1")
    if (
        isinstance(azimuth, (bool, np.bool_))
        or int(azimuth) != azimuth
        or int(azimuth) < 4
    ):
        raise ValueError("azimuth must be an integer >= 4")
    shrink_value = float(shrink)
    if not (0.0 < shrink_value < 1.0):
        raise ValueError("shrink must lie strictly between 0 and 1")
    depth_frac = float(depth_fraction)
    if not (0.0 < depth_frac < 1.0):
        raise ValueError("depth_fraction must lie strictly between 0 and 1")

    centre, radius = _waterline_circle(mesh)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("waterline radius must be positive")

    # The lid must sit at least ~1.5 panel heights below the waterplane, otherwise
    # its centroid rows are nearly singular with respect to the waterline panels
    # and the least-squares fit degrades as the mesh is refined.
    if not (0.0 <= float(min_depth_panels) < 100.0):
        raise ValueError("min_depth_panels must be a non-negative number < 100")
    geometric_cap = radius * math.sqrt(
        max(0.0, 1.0 - min(1.0, (1.15 * shrink_value) ** 2))
    )
    depth = min(
        max(depth_frac * shrink_value * radius, float(min_depth_panels) * float(mesh.h_mean)),
        0.8 * geometric_cap,
    )
    if depth <= 0.0:
        raise ValueError("could not place the lid below the waterplane; reduce shrink")
    n_rings = int(rings)
    n_az = int(azimuth)

    points: list[tuple[float, float, float]] = []
    areas: list[float] = []
    # Panel area estimate per ring: annulus area divided by the ring panel count.
    for k in range(1, n_rings + 1):
        r_out = shrink_value * radius * k / n_rings
        r_in = shrink_value * radius * (k - 1) / n_rings
        m = max(4, int(round(n_az * k / n_rings)))
        ring_area = math.pi * (r_out**2 - r_in**2) / m
        for j in range(m):
            theta = 2.0 * math.pi * j / m
            points.append(
                (centre[0] + r_out * math.cos(theta), centre[1] + r_out * math.sin(theta), -depth)
            )
            areas.append(ring_area)
    # Central panel.
    points.append((centre[0], centre[1], -depth))
    areas.append(math.pi * (shrink_value * radius / n_rings) ** 2)

    pts = np.asarray(points, dtype=float)
    if not np.isfinite(pts).all():
        raise ValueError("interior panel points must be finite")
    if np.any(pts[:, 2] >= 0.0):
        raise ValueError("interior panel points must lie strictly below z = 0")

    if validate:
        inside = _point_inside_hull(pts, mesh)
        if not bool(np.all(inside)):
            # The 3-D ray cast can report a false negative when a ray grazes a
            # triangle edge.  Re-test the rejected points a hair further inside
            # (1e-3 of the way to the centroid); only points that fail both
            # tests are genuinely outside the hull.
            rejected = pts[~inside]
            nudged = rejected.copy()
            nudged[:, :2] += 1.0e-3 * (centre[None, :] - rejected[:, :2])
            still_out = ~_point_inside_hull(nudged, mesh)
            inside = inside.copy()
            inside[~inside] = ~still_out
        if not bool(np.all(inside)):
            bad = int(np.count_nonzero(~inside))
            raise ValueError(
                f"{bad} of {pts.shape[0]} interior lid points fall outside the hull; "
                "reduce shrink or depth_fraction"
            )
    return InteriorPanelSet(
        points=pts,
        areas=np.asarray(areas, dtype=float),
        waterline_radius=radius,
        shrink=shrink_value,
        depth=depth,
    )

def waterline_polygon(mesh: SurfaceMesh3D) -> FloatArray:
    """Return the waterplane polygon of a surface-piercing hull, CCW ordered.

    Only hulls whose waterplane is star-shaped with respect to its centroid are
    supported: the vertices at ``z = 0`` are ordered by polar angle about their
    centroid.  A rectangular barge (and any convex/ship-like waterplane) meets
    that condition; the previous :func:`interior_panel_points` assumed a
    *circular* waterplane and raised for every non-axisymmetric hull.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    ring = verts[np.abs(verts[:, 2]) <= 1.0e-12]
    if ring.shape[0] < 3:
        raise ValueError("hull has fewer than three waterline vertices")
    xy = np.unique(np.round(ring[:, :2], 12), axis=0)
    centre = xy.mean(axis=0)
    angles = np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
    return xy[np.argsort(angles)]


def _polygon_area(poly: FloatArray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def waterplane_lid_points(
    mesh: SurfaceMesh3D,
    *,
    rings: int = 3,
    shrink: float = 0.6,
    depth_fraction: float = _DEFAULT_DEPTH_FRACTION,
    min_depth_panels: float = 1.5,
    validate: bool = True,
) -> InteriorPanelSet:
    """Interior waterplane lid for a *non-axisymmetric* waterplane.

    The concentric-ring construction of :func:`interior_panel_points` is
    generalised from a disk to the actual waterplane polygon: ring ``k`` is the
    polygon scaled by ``shrink * k / rings`` about the centroid, so the lid
    follows a rectangular (barge) or tapered (ship) waterplane instead of
    spilling outside a circular bound.  Every point is placed ``depth`` below
    ``z = 0`` and the point-in-solid test rejects any point that escaped the
    hull.
    """
    if isinstance(rings, (bool, np.bool_)) or int(rings) != rings or int(rings) < 1:
        raise ValueError("rings must be an integer >= 1")
    shrink_value = float(shrink)
    if not (0.0 < shrink_value < 1.0):
        raise ValueError("shrink must lie strictly between 0 and 1")
    depth_frac = float(depth_fraction)
    if not (0.0 < depth_frac < 1.0):
        raise ValueError("depth_fraction must lie strictly between 0 and 1")

    poly = waterline_polygon(mesh)
    centre = poly.mean(axis=0)
    radii = np.linalg.norm(poly - centre, axis=1)
    radius = float(radii.max())
    if radius <= 0.0:
        raise ValueError("waterline polygon has zero radius")

    depth = max(
        depth_frac * shrink_value * radius,
        float(min_depth_panels) * float(mesh.h_mean),
    )
    # The lid must stay inside the hull: never deeper than the local draft.
    draft = -float(np.min(np.asarray(mesh.vertices, dtype=float)[:, 2]))
    if draft <= 0.0:
        raise ValueError("hull does not extend below the waterline (draft <= 0)")
    depth = min(depth, 0.8 * draft)
    if depth <= 0.0:
        raise ValueError("could not place the lid below the waterplane")

    n_rings = int(rings)
    points: list[tuple[float, float, float]] = []
    ring_scales: list[float] = []
    for k in range(1, n_rings + 1):
        scale = shrink_value * k / n_rings
        ring_scales.append(scale)
        for vertex in poly:
            xy = centre + scale * (vertex - centre)
            points.append((float(xy[0]), float(xy[1]), -depth))
    pts = np.asarray(points, dtype=float)
    if not np.isfinite(pts).all():
        raise ValueError("interior panel points must be finite")

    # Each ring contributes the annulus between consecutive scaled polygons;
    # equal-area weights are enough (the rows are unweighted).
    total_area = abs(_polygon_area(poly)) * shrink_value**2
    areas = np.full(pts.shape[0], total_area / float(pts.shape[0]))

    if validate:
        inside = _point_inside_hull(pts, mesh)
        if not bool(np.all(inside)):
            # The 3-D ray cast can report a false negative when a ray grazes a
            # triangle edge.  Re-test the rejected points a hair further inside
            # (1e-3 of the way to the centroid); only points that fail both
            # tests are genuinely outside the hull.
            rejected = pts[~inside]
            nudged = rejected.copy()
            nudged[:, :2] += 1.0e-3 * (centre[None, :] - rejected[:, :2])
            still_out = ~_point_inside_hull(nudged, mesh)
            inside = inside.copy()
            inside[~inside] = ~still_out
        if not bool(np.all(inside)):
            bad = int(np.count_nonzero(~inside))
            raise ValueError(
                f"{bad} of {pts.shape[0]} waterplane lid points fall outside the hull; "
                "the waterplane is not star-shaped about its centroid"
            )
    return InteriorPanelSet(
        points=pts,
        areas=areas,
        waterline_radius=radius,
        shrink=shrink_value,
        depth=depth,
    )
