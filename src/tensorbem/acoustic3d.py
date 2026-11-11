"""Three-dimensional unbaffled exterior Helmholtz acoustic BEM (radiation + scattering).

Motivation
----------
:mod:`tensorbem.kernels3d` provides the *real* Laplacian layer potentials and
:mod:`tensorbem.helmholtz` the *two-dimensional* Helmholtz diffraction problem.
Neither answers the ship-acoustics question this module exists for: what sound
pressure does a curved, closed three-dimensional hull radiate into an unbounded
fluid when its structure vibrates, and what does the hull itself scatter back
(self-scattering)?  The classical ship-noise paraphernalia -- Rayleigh's
integral over an *infinite rigid baffle*, or a planar array of monopoles --
answers a different problem: it replaces the real hull by a flat piston, hides
the back side of the radiator, cannot represent the acoustic shadow of the hull
and cannot reproduce the fore-aft asymmetry of a real radiator.  This module
removes the baffle: it solves the exterior Helmholtz problem on the closed
wetted surface.

Conventions
-----------
Same as :mod:`tensorbem.free_surface3d`: fields carry the harmonic factor
``Re{phi(x) * exp(-i*omega*t)}``, so the outgoing free-space Green function of
``(Laplacian + k**2) u = 0`` is

    G_k(r) = exp(i*k*r) / (4*pi*r),        (Laplacian + k**2) G_k = -delta,

and the double-layer kernel (derivative with respect to the *source* normal
``n_y``, with ``n`` pointing out of the body into the fluid) is

    dG_k/dn_y = (x - y).n_y * (1 - i*k*r) * exp(i*k*r) / (4*pi*r**3),
    r = |x - y|.

Acoustic pressure follows from the linearised Euler equation,
``p = i*omega*rho*phi``, and the level convention is the underwater one,
``SPL = 20*log10(p_rms / p_ref)`` with ``p_ref = 1e-6 Pa`` (1 uPa).

Boundary integral equations
---------------------------
Green's second identity in the *exterior* domain, with the surface layers of
the physical solution ``(phi, q)``, ``q = dphi/dn = u_n`` (normal velocity),

    phi(x) = int_S [ phi dG_k/dn - G_k q ] dS,          x in the fluid,

gives, for a collocation point on the (smooth) surface,

    A phi = -S q,        A = 1/2 I - D,                 (surface equation)

with ``S`` the single-layer and ``D`` the double-layer matrix.  ``D`` is
assembled from the *signed solid angle* of every flat panel, which integrates
the Laplace part of the kernel exactly, plus a quadrature of the regular
remainder ``dG_k/dn - dG_0/dn = O(k**2 h**2)`` times the Laplace kernel; the
collocation panel itself contributes zero to the principal value.  At interior
points ``x_m`` the null-field identity ``int_S [phi dG_k/dn - G_k q] dS = 0``
supplies the CHIEF rows (Schenck 1968; Jones 1974) that remove the irregular
frequencies of the exterior Neumann problem, i.e. the interior Dirichlet
eigenvalues of the closed body.

Radiating and scattering bodies
-------------------------------
* radiation: prescribe ``q`` and solve ``A phi = -S q``;
* scattering by a rigid body: the total field is ``phi = phi_inc + phi_s`` with
  ``dphi/dn = 0``, hence ``q_s = -dphi_inc/dn`` and the same system for
  ``phi_s``.

Far field
---------
With ``x = R d``, ``R -> infinity``, ``phi_s -> f(d) exp(i*k*R)/R`` and

    f(d) = (1/(4*pi)) int_S exp(-i*k d.y) [ -i*k (d.n) phi - q ] dS,

so the scattering cross-section is ``sigma = int |f|^2 dOmega`` and the
directivity is ``|f(d)|^2``.

Shared upstream surface with the water-wave solvers
---------------------------------------------------
The Laplace ``k -> 0`` limit of this module is the exterior Neumann problem of
:mod:`tensorbem.kernels3d`, so both share the flat-panel solid-angle assembly
and :func:`tensorbem.kernels3d.integrate_single_layer_triangle_self`; the
Helmholtz extension is the singularity-extraction correction on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import leggauss, legval
from scipy.special import spherical_jn, spherical_yn

from .geometry3d import TriangleMesh, icosphere
from .kernels3d import duffy_quadrature, duffy_quadrature_batch

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_FOUR_PI = 4.0 * math.pi

#: Acoustics of sea water (documented defaults, all overridable).
SOUND_SPEED_SEAWATER = 1500.0  # m/s
DENSITY_SEAWATER = 1025.0  # kg/m^3
REFERENCE_PRESSURE = 1.0e-6  # Pa (1 uPa): ICES 209 / ISO 18405 underwater level

#: Nominal ICES 209 / IEC 61260 one-third-octave centre frequencies (Hz).
ICES209_THIRD_OCTAVE_CENTERS: tuple[float, ...] = (
    10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0,
    125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
    1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
    10000.0,
)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def helmholtz_green_3d(distance: FloatArray | float, wavenumber: float) -> ComplexArray | complex:
    r"""Return ``G_k(r) = exp(i*k*r)/(4*pi*r)`` for positive ``r``."""
    r = np.asarray(distance, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("distance must be positive; handle the singular panel separately")
    k = float(wavenumber)
    if not np.isfinite(k):
        raise ValueError("wavenumber must be finite")
    values = np.exp(1j * k * r) / (_FOUR_PI * r)
    if values.ndim == 0:
        return complex(values)
    return values


def helmholtz_double_layer_kernel_3d(
    target: FloatArray, source: FloatArray, source_normal: FloatArray, wavenumber: float
) -> complex:
    r"""Return ``dG_k/dn_y = (x-y).n_y (1-i*k*r) exp(i*k*r)/(4*pi*r**3)``."""
    x = np.asarray(target, dtype=float)
    y = np.asarray(source, dtype=float)
    n = np.asarray(source_normal, dtype=float)
    if x.shape != (3,) or y.shape != (3,) or n.shape != (3,):
        raise ValueError("target, source and source_normal must have shape (3,)")
    delta = x - y
    r2 = float(np.dot(delta, delta))
    if r2 <= 0.0:
        raise ValueError("double-layer point kernel is singular at coincident points")
    r = math.sqrt(r2)
    k = float(wavenumber)
    return complex(
        np.dot(delta, n) * (1.0 - 1j * k * r) * np.exp(1j * k * r) / (_FOUR_PI * r2 * r)
    )


def double_layer_remainder(distance: FloatArray, wavenumber: float) -> ComplexArray:
    r"""Return ``(1-i*k*r)*exp(i*k*r) - 1``, the regular double-layer remainder."""
    r = np.asarray(distance, dtype=float)
    k = float(wavenumber)
    return (1.0 - 1j * k * r) * np.exp(1j * k * r) - 1.0


def wavenumber_from_frequency(
    frequency: FloatArray | float, sound_speed: float = SOUND_SPEED_SEAWATER
) -> FloatArray | float:
    """Return the acoustic wavenumber ``k = 2*pi*f/c``."""
    f = np.asarray(frequency, dtype=float)
    if np.any(f < 0.0):
        raise ValueError("frequency must be non-negative")
    c = float(sound_speed)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("sound_speed must be positive and finite")
    value = 2.0 * math.pi * f / c
    return float(value) if value.ndim == 0 else value


def frequency_from_wavenumber(
    wavenumber: FloatArray | float, sound_speed: float = SOUND_SPEED_SEAWATER
) -> FloatArray | float:
    """Return ``f = k*c/(2*pi)``."""
    k = np.asarray(wavenumber, dtype=float)
    value = k * float(sound_speed) / (2.0 * math.pi)
    return float(value) if value.ndim == 0 else value


# ---------------------------------------------------------------------------
# operator assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HelmholtzLayerMatrices:
    """Discretised single- and double-layer potentials of the Helmholtz kernel.

    ``single_layer[i, j]`` is ``int_{S_j} G_k(x_i, y) dS_y`` and
    ``double_layer[i, j]`` the principal-value ``int_{S_j} dG_k/dn_y dS_y``.
    For collocation on the surface (``targets=None``) the diagonal of
    ``double_layer`` is the flat-panel principal value, i.e. exactly zero.
    """

    wavenumber: float
    single_layer: ComplexArray
    double_layer: ComplexArray
    targets: FloatArray
    on_surface: bool
    panel_count: int
    laplace_row_sums: FloatArray  # surface rows sum to -1/2, interior rows to -1

    @property
    def matrix_size(self) -> int:
        return int(self.single_layer.shape[0])


def _panel_scales(areas: FloatArray) -> FloatArray:
    """Return a characteristic panel length ``h = sqrt(2*area)``."""
    return np.sqrt(2.0 * np.asarray(areas, dtype=float))


def _signed_solid_angles(
    targets: FloatArray, v0: FloatArray, v1: FloatArray, v2: FloatArray
) -> FloatArray:
    """Vectorised signed solid angle of ``F`` triangles seen from ``T`` targets.

    Returns an array of shape ``(T, F)`` and reproduces, batched over targets,
    the arithmetic of :func:`tensorbem.kernels3d.solid_angle_triangle`.
    """
    a = v0[None, :, :] - targets[:, None, :]
    b = v1[None, :, :] - targets[:, None, :]
    c = v2[None, :, :] - targets[:, None, :]
    determinant = np.einsum("tij,tij->ti", a, np.cross(b, c))
    norm_a = np.linalg.norm(a, axis=2)
    norm_b = np.linalg.norm(b, axis=2)
    norm_c = np.linalg.norm(c, axis=2)
    denominator = (
        norm_a * norm_b * norm_c
        + np.einsum("tij,tij->ti", a, b) * norm_c
        + np.einsum("tij,tij->ti", b, c) * norm_a
        + np.einsum("tij,tij->ti", c, a) * norm_b
    )
    return 2.0 * np.arctan2(determinant, denominator)


def single_layer_triangle_self_laplace(vertices: FloatArray) -> FloatArray | float:
    r"""Exact ``int_T dS/(4*pi*|x-y|)`` at the centroid ``x`` of each triangle.

    In polar coordinates about the centroid the kernel reduces to
    ``int_0^{2*pi} R(theta) dtheta`` with ``R`` the radial extent of the
    triangle.  For the sector bounded by the edge ``[p, q]`` the extent is
    ``h/cos(theta - phi)`` with ``h`` the perpendicular distance from the
    centroid to the edge line, so the sector integral is the closed form

        h * (asinh(s_q/h) - asinh(s_p/h)),

    with ``s_p``, ``s_q`` the signed distances of the edge endpoints from the
    foot of the perpendicular.  The panel is convex about its centroid, so the
    three sectors partition ``[0, 2*pi)`` exactly and the sum is analytic (no
    quadrature).  Verified against
    :func:`tensorbem.kernels3d.integrate_single_layer_triangle_self` to 1e-14.
    """
    tri = np.asarray(vertices, dtype=float)
    if tri.ndim == 2:
        tri = tri[None, :, :]
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("vertices must have shape (T, 3, 3)")
    centroid = tri.mean(axis=1)
    total = np.zeros(tri.shape[0], dtype=float)
    for i in range(3):
        p = tri[:, i, :]
        q = tri[:, (i + 1) % 3, :]
        edge = q - p
        length = np.linalg.norm(edge, axis=1)
        unit = edge / length[:, None]
        projection = np.einsum("tj,tj->t", centroid - p, unit)
        foot = p + projection[:, None] * unit
        height = np.linalg.norm(foot - centroid, axis=1)
        if np.any(height <= 0.0):
            raise ValueError("degenerate triangle: centroid on an edge line")
        s_p = -projection
        s_q = length - projection
        total += height * (np.arcsinh(s_q / height) - np.arcsinh(s_p / height))
    result = total / _FOUR_PI
    return float(result[0]) if result.shape[0] == 1 else result


def assemble_helmholtz_layer_matrices(
    mesh: TriangleMesh,
    wavenumber: float,
    *,
    targets: FloatArray | None = None,
    near_factor: float = 2.5,
    quad_order: int = 8,
    far_order: int = 3,
    block: int = 48,
    pair_chunk: int = 4096,
) -> HelmholtzLayerMatrices:
    """Assemble ``S_k`` and ``D_k`` of the Helmholtz kernel on a triangular mesh.

    Singularity extraction keeps every quadrature regular:

    * the double-layer Laplace part is *exact* on flat panels through the signed
      solid angle (so the ``k -> 0`` limit is the Laplace layer of
      :mod:`tensorbem.kernels3d` to machine precision), and the Helmholtz
      remainder ``(x-y).n/(4*pi*r**3) * [(1-i*k*r)exp(i*k*r) - 1] = O(k**2)``
      is integrated with a panel quadrature;
    * the single-layer Laplace part is analytic on the collocation panel
      (:func:`single_layer_triangle_self_laplace`) and quadrature-integrated
      elsewhere, the bounded remainder ``(exp(i*k*r)-1)/(4*pi*r)`` sharing the
      same points.

    Parameters
    ----------
    targets:
        Collocation points ``(T, 3)``.  ``None`` means the panel centroids,
        i.e. the surface equation; any other set (interior CHIEF points, field
        points) evaluates the *same* kernels elsewhere.
    near_factor:
        Panels whose centroid distance is below ``near_factor * h_panel`` are
        integrated with the ``quad_order`` Duffy rule instead of the
        ``far_order`` panel rule.
    """

    k = float(wavenumber)
    if not np.isfinite(k) or k < 0.0:
        raise ValueError("wavenumber must be finite and non-negative")

    areas = np.asarray(mesh.areas, dtype=float)
    centroids = np.asarray(mesh.centroids, dtype=float)
    normals = np.asarray(mesh.normals, dtype=float)
    v0 = np.asarray(mesh.v0, dtype=float)
    v1 = np.asarray(mesh.v1, dtype=float)
    v2 = np.asarray(mesh.v2, dtype=float)
    panel_count = areas.shape[0]
    scales = _panel_scales(areas)
    vertices_of = np.stack([v0, v1, v2], axis=1)  # (F, 3, 3)

    on_surface = targets is None
    if on_surface:
        points = centroids.copy()
    else:
        points = np.asarray(targets, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("targets must have shape (T, 3)")
        if not np.isfinite(points).all():
            raise ValueError("targets must be finite")

    # Panel quadrature points for the regular / far integrands.
    far_points, far_weights = _panel_rule(vertices_of, far_order)  # (F, Q0, 3), (F, Q0)
    near_points, near_weights = _panel_rule(vertices_of, quad_order)

    single = np.zeros((points.shape[0], panel_count), dtype=np.complex128)
    double = np.zeros((points.shape[0], panel_count), dtype=np.complex128)
    row_sums = np.zeros(points.shape[0], dtype=float)

    for start in range(0, points.shape[0], block):
        stop = min(start + block, points.shape[0])
        chunk = points[start:stop]
        delta = chunk[:, None, :] - centroids[None, :, :]
        r = np.linalg.norm(delta, axis=2)

        # --- exact Laplace double layer through signed solid angles ----------
        omega = _signed_solid_angles(chunk, v0, v1, v2)
        laplace_double = -omega / _FOUR_PI

        # --- regular part of both layers with the panel rule -----------------
        diff = chunk[:, None, None, :] - far_points[None, :, :, :]  # (B, F, Q, 3)
        dist = np.linalg.norm(diff, axis=3)
        safe = np.where(dist > 0.0, dist, 1.0)
        single_chunk = np.einsum(
            "tfq,fq->tf",
            np.where(dist > 0.0, np.exp(1j * k * dist) / (_FOUR_PI * safe), 0.0),
            far_weights,
        ).astype(np.complex128)
        correction = np.einsum(
            "tfq,fq->tf",
            np.einsum("tfqj,fj->tfq", diff, normals)
            * np.where(dist > 0.0, double_layer_remainder(dist, k), 0.0)
            / (_FOUR_PI * safe**3),
            far_weights,
        )
        double_chunk = laplace_double + correction
        del diff, dist, safe

        coincident = r <= 1.0e-9 * scales[None, :]
        if not on_surface and np.any(coincident):
            raise ValueError(
                "a target point coincides with a panel centroid; evaluate the surface "
                "equation with targets=None instead"
            )

        # --- collocation panels: principal value + exact single layer --------
        if on_surface:
            rows = np.arange(stop - start)
            panels = start + rows
            laplace_self = single_layer_triangle_self_laplace(vertices_of[panels])
            q_points = near_points[panels]  # (B, Q, 3)
            q_weights = near_weights[panels]
            rr = np.linalg.norm(chunk[:, None, :] - q_points, axis=2)
            remainder = np.einsum(
                "bq,bq->b",
                q_weights,
                (np.exp(1j * k * rr) - 1.0) / (_FOUR_PI * rr),
            )
            single_chunk[rows, panels] = laplace_self + remainder
            # flat-panel principal value of the double layer is zero, and its
            # Helmholtz remainder vanishes identically: the target is coplanar
            # with the panel, so (x-y).n = 0.
            double_chunk[rows, panels] = 0.0
            laplace_double[rows, panels] = 0.0
            coincident[rows, panels] = False

        row_sums[start:stop] = laplace_double.sum(axis=1)

        # --- near panels: accurate rule on the full kernel -------------------
        near = (r < near_factor * scales[None, :]) & (~coincident)
        if on_surface:
            near[np.arange(stop - start), start + np.arange(stop - start)] = False
        if np.any(near):
            near_rows, near_cols = np.nonzero(near)
            for offset in range(0, near_rows.shape[0], pair_chunk):
                row_ids = near_rows[offset : offset + pair_chunk]
                col_ids = near_cols[offset : offset + pair_chunk]
                q_points = near_points[col_ids]
                q_weights = near_weights[col_ids]
                diff = chunk[row_ids][:, None, :] - q_points
                dist = np.linalg.norm(diff, axis=2)
                single_chunk[row_ids, col_ids] = np.einsum(
                    "pq,pq->p",
                    q_weights,
                    np.exp(1j * k * dist) / (_FOUR_PI * dist),
                )
                correction = np.einsum(
                    "pq,pq->p",
                    q_weights,
                    np.einsum("pqj,pj->pq", diff, normals[col_ids])
                    * double_layer_remainder(dist, k)
                    / (_FOUR_PI * dist**3),
                )
                double_chunk[row_ids, col_ids] = (
                    laplace_double[row_ids, col_ids] + correction
                )

        single[start:stop] = single_chunk
        double[start:stop] = double_chunk

    return HelmholtzLayerMatrices(
        wavenumber=k,
        single_layer=single,
        double_layer=double,
        targets=points,
        on_surface=on_surface,
        panel_count=panel_count,
        laplace_row_sums=row_sums,
    )


def _panel_rule(
    vertices_of: FloatArray, order: int
) -> tuple[FloatArray, FloatArray]:
    """Return ``(F, order**2, 3)`` quadrature points and weights for ``F`` panels."""
    points, weights = duffy_quadrature_batch(
        vertices_of[:, 0, :], vertices_of[:, 1, :], vertices_of[:, 2, :], order=order
    )
    return points, weights


# ---------------------------------------------------------------------------
# closed-body construction (wetted hull + waterplane lid)
# ---------------------------------------------------------------------------


def close_surface_with_plane_lid(
    surface: object, plane_z: float = 0.0, *, area_tolerance: float = 1.0e-9
) -> TriangleMesh:
    """Close an open, outward-oriented surface with a planar lid.

    A ship's wetted-surface mesh (:class:`tensorbem.geometry3d.WettedHullMesh`)
    is deliberately open along the waterline: the waterplane lid is not part of
    the hydrodynamic boundary.  The acoustic exterior problem, in contrast,
    needs a *closed* body, otherwise the boundary-integral equation of
    :func:`solve_exterior_neumann` is missing the solid angle of the opening and
    the CHIEF rows have nowhere to live.  This function finds the boundary
    loop(s) of the face set, orders each loop, and fans it from its centroid,
    producing a :class:`tensorbem.geometry3d.TriangleMesh` whose closure and
    outward orientation are validated by the ``TriangleMesh`` constructor.

    The lid is a *rigid* (Neumann) cap: the ship is modelled as a closed body in
    an unbounded fluid.  That removes the infinite-baffle assumption of the
    Rayleigh integral but not the free surface -- see the limitation section of
    ``docs/acoustic3d_helmholtz.md``.
    """
    vertices = np.asarray(surface.vertices, dtype=float)  # type: ignore[attr-defined]
    faces = np.asarray(surface.faces, dtype=np.int64)  # type: ignore[attr-defined]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("surface.vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("surface.faces must have shape (F, 3)")

    boundary = _boundary_edges(faces)
    if boundary.shape[0] == 0:
        return TriangleMesh(vertices, faces)

    loops = _order_boundary_loop(boundary)
    new_vertices = [vertices]
    new_faces = [faces]
    offset = vertices.shape[0]
    for loop in loops:
        points = vertices[loop]
        if np.ptp(points[:, 2]) > 1.0e-6 * max(1.0, float(np.abs(points[:, 2]).max())):
            raise ValueError("boundary loop is not planar at constant z")
        centroid = points.mean(axis=0)
        centroid[2] = float(plane_z)
        signed_area = 0.5 * float(
            np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
        )
        if abs(signed_area) <= area_tolerance:
            raise ValueError("boundary loop encloses no area")
        ordered = loop if signed_area > 0.0 else loop[::-1]
        lid_centre = offset
        new_vertices.append(centroid[None, :])
        offset += 1
        for index in range(ordered.shape[0]):
            a = int(ordered[index])
            b = int(ordered[(index + 1) % ordered.shape[0]])
            new_faces.append(np.array([[lid_centre, a, b]], dtype=np.int64))

    return TriangleMesh(np.vstack(new_vertices), np.vstack(new_faces))


def _boundary_edges(faces: IntArray) -> IntArray:
    """Return the undirected edges that appear exactly once (the open boundary)."""
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys = np.sort(directed, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys = keys[order]
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if boundary.shape[0] == 0:
        if np.any(counts > 2):
            raise ValueError("non-manifold surface: an edge is shared by more than two faces")
        return np.zeros((0, 2), dtype=np.int64)
    if np.any(counts > 2):
        raise ValueError("non-manifold surface: an edge is shared by more than two faces")
    return boundary


def _order_boundary_loop(boundary: IntArray) -> list[IntArray]:
    """Order undirected boundary edges into closed loops of vertex indices."""
    neighbours: dict[int, list[int]] = {}
    for a, b in np.asarray(boundary, dtype=np.int64):
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))
    for vertex, links in neighbours.items():
        if len(links) != 2:
            raise ValueError(
                f"boundary vertex {vertex} has degree {len(links)}; the open boundary "
                "must be a disjoint union of simple closed loops"
            )
    loops: list[IntArray] = []
    remaining = set(neighbours)
    while remaining:
        start = min(remaining)
        loop = [start]
        previous = None
        current = start
        while True:
            candidates = [v for v in neighbours[current] if v != previous]
            if not candidates:
                raise ValueError("boundary loop terminates early")
            nxt = candidates[0]
            if nxt == start:
                break
            loop.append(nxt)
            previous, current = current, nxt
            if len(loop) > len(neighbours):
                raise ValueError("boundary loop does not close")
        remaining.difference_update(loop)
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


# ---------------------------------------------------------------------------
# CHIEF points and the exterior Neumann solve
# ---------------------------------------------------------------------------


def interior_chief_points(
    mesh: TriangleMesh, count: int = 5, fraction: float = 0.4
) -> FloatArray:
    """Return ``count`` interior CHIEF points distributed over the closed body.

    The points are taken along the rays from the vertex centroid towards
    evenly spaced panels, at ``fraction`` of the way to the surface.  This is
    deterministic, independent of panel numbering, and strictly interior for
    convex or star-shaped bodies (a ship hull closed by its waterplane is
    star-shaped about a point amidships).  For strongly non-convex bodies pass
    explicit ``chief_points=`` to :func:`solve_exterior_neumann`.
    """
    if count < 1:
        raise ValueError("count must be at least one")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one")
    centroids = np.asarray(mesh.centroids, dtype=float)
    centre = centroids.mean(axis=0)
    indices = np.linspace(0, centroids.shape[0] - 1, count).astype(int)
    points = centre[None, :] + fraction * (centroids[indices] - centre[None, :])
    return points


@dataclass(frozen=True)
class HelmholtzSolution:
    """Solution of the exterior Neumann (radiating body) problem.

    ``phi`` is the surface velocity potential and ``q`` the prescribed normal
    velocity (``dphi/dn``); the acoustic pressure anywhere in the fluid is
    ``p = i*omega*rho*phi`` with ``phi`` from :func:`field_potential`.
    """

    wavenumber: float
    frequency: float
    omega: float
    phi: ComplexArray
    normal_velocity: ComplexArray
    chief_points: FloatArray
    chief_used: int
    condition_number: float
    boundary_relative_residual: float
    chief_relative_residual: float
    surface_area: float
    panels_per_wavelength: float


def _system_matrix(layers: HelmholtzLayerMatrices) -> ComplexArray:
    size = layers.panel_count
    return 0.5 * np.eye(size, dtype=np.complex128) - layers.double_layer


def _condition_estimate_1norm(matrix: ComplexArray) -> float:
    """Return the 1-norm condition estimate ``||A||_1 * ||A^{-1}||_1``.

    A cheap stand-in for the SVD condition number: it costs one extra LU
    factorisation instead of a full singular-value decomposition, which matters
    because an exhaustive frequency sweep solves hundreds of systems.
    """
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:  # pragma: no cover - exactly singular matrix
        return float("inf")
    return float(np.linalg.norm(matrix, 1) * np.linalg.norm(inverse, 1))


def solve_exterior_neumann(
    mesh: TriangleMesh,
    wavenumber: float,
    normal_velocity: FloatArray | complex,
    *,
    chief_points: FloatArray | None = None,
    n_chief: int = 5,
    density: float = DENSITY_SEAWATER,
    sound_speed: float = SOUND_SPEED_SEAWATER,
    diagnostics: bool = False,
    **assembly_kwargs: object,
) -> HelmholtzSolution:
    """Solve ``(1/2 I - D_k) phi = -S_k q`` for a radiating closed body.

    Parameters
    ----------
    normal_velocity:
        Complex normal velocity amplitude ``q = dphi/dn`` on every panel
        (scalar, or an array of length ``mesh.panel_count``).
    chief_points:
        Explicit interior CHIEF points; ``None`` auto-generates ``n_chief`` of
        them with :func:`interior_chief_points`.
    diagnostics:
        Compute the condition number (SVD, ``O(N**3)``) and the residuals.
    """

    k = float(wavenumber)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("wavenumber must be positive and finite")
    panel_count = int(mesh.panel_count)
    q = np.asarray(normal_velocity, dtype=np.complex128)
    if q.ndim == 0:
        q = np.full(panel_count, complex(q), dtype=np.complex128)
    if q.shape != (panel_count,):
        raise ValueError("normal_velocity must be a scalar or of length panel_count")

    layers = assemble_helmholtz_layer_matrices(mesh, k, **assembly_kwargs)  # type: ignore[arg-type]
    matrix = _system_matrix(layers)
    rhs = -(layers.single_layer @ q)

    if chief_points is None:
        points = interior_chief_points(mesh, n_chief)
    else:
        points = np.asarray(chief_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("chief_points must have shape (K, 3)")
        if not np.isfinite(points).all():
            raise ValueError("chief_points must be finite")
    chief_used = 0
    chief_residual = 0.0
    if points.shape[0] > 0:
        interior = assemble_helmholtz_layer_matrices(mesh, k, targets=points, **assembly_kwargs)  # type: ignore[arg-type]
        chief_rows = interior.double_layer
        chief_rhs = interior.single_layer @ q
        stacked = np.vstack([matrix, chief_rows])
        stacked_rhs = np.concatenate([rhs, chief_rhs])
        solution, *_ = np.linalg.lstsq(stacked, stacked_rhs, rcond=None)
        phi = solution
        chief_used = int(points.shape[0])
        chief_residual = float(
            np.linalg.norm(chief_rows @ phi - chief_rhs)
            / max(np.linalg.norm(chief_rhs), np.finfo(float).tiny)
        )
        boundary_residual = float(
            np.linalg.norm(matrix @ phi - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        )
    else:
        phi = np.linalg.solve(matrix, rhs)
        boundary_residual = float(
            np.linalg.norm(matrix @ phi - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        )

    condition = _condition_estimate_1norm(matrix) if diagnostics else float("nan")
    omega = k * float(sound_speed)
    wavelength = 2.0 * math.pi / k
    panels_per_wavelength = float(
        math.sqrt(float(mesh.surface_area) / panel_count) / wavelength
    )
    return HelmholtzSolution(
        wavenumber=k,
        frequency=frequency_from_wavenumber(k, sound_speed),
        omega=omega,
        phi=np.asarray(phi, dtype=np.complex128),
        normal_velocity=q,
        chief_points=points,
        chief_used=chief_used,
        condition_number=condition,
        boundary_relative_residual=boundary_residual,
        chief_relative_residual=chief_residual,
        surface_area=float(mesh.surface_area),
        panels_per_wavelength=panels_per_wavelength,
    )


def incident_plane_wave(
    wavenumber: float,
    direction: FloatArray,
    points: FloatArray,
    *,
    amplitude: complex = 1.0,
) -> ComplexArray:
    """Return the velocity potential of a unit plane wave travelling along ``direction``."""
    k = float(wavenumber)
    d = np.asarray(direction, dtype=float)
    if d.shape != (3,):
        raise ValueError("direction must have shape (3,)")
    norm = float(np.linalg.norm(d))
    if norm <= 0.0:
        raise ValueError("direction must be non-zero")
    d = d / norm
    x = np.asarray(points, dtype=float)
    return complex(amplitude) * np.exp(1j * k * (x @ d))


def solve_rigid_scattering(
    mesh: TriangleMesh,
    wavenumber: float,
    direction: FloatArray,
    *,
    chief_points: FloatArray | None = None,
    n_chief: int = 5,
    diagnostics: bool = False,
    **assembly_kwargs: object,
) -> HelmholtzSolution:
    """Solve the scattering of a unit plane wave by a rigid closed body.

    The returned ``phi``/``normal_velocity`` are the *scattered* fields; the
    total field is obtained by adding :func:`incident_plane_wave` evaluated at
    the same points.  The rigid boundary condition ``d(phi_inc+phi_s)/dn = 0``
    gives ``q_s = -dphi_inc/dn = -i*k*(d.n)*exp(i*k*d.y)``.
    """

    k = float(wavenumber)
    d = np.asarray(direction, dtype=float)
    if d.shape != (3,):
        raise ValueError("direction must have shape (3,)")
    norm = float(np.linalg.norm(d))
    if norm <= 0.0:
        raise ValueError("direction must be non-zero")
    d = d / norm
    centroids = np.asarray(mesh.centroids, dtype=float)
    normals = np.asarray(mesh.normals, dtype=float)
    phase = np.exp(1j * k * (centroids @ d))
    q_scattered = -1j * k * (normals @ d) * phase
    return solve_exterior_neumann(
        mesh,
        k,
        q_scattered,
        chief_points=chief_points,
        n_chief=n_chief,
        diagnostics=diagnostics,
        **assembly_kwargs,
    )


# ---------------------------------------------------------------------------
# field, pressure, far field
# ---------------------------------------------------------------------------


def field_potential(
    mesh: TriangleMesh,
    wavenumber: float,
    phi: ComplexArray,
    normal_velocity: ComplexArray,
    points: FloatArray,
    **assembly_kwargs: object,
) -> ComplexArray:
    """Return ``phi(x) = int_S [phi dG/dn - G q] dS`` at exterior points ``x``."""
    field_points = np.asarray(points, dtype=float)
    if field_points.ndim == 1:
        field_points = field_points[None, :]
    if field_points.ndim != 2 or field_points.shape[1] != 3:
        raise ValueError("points must have shape (T, 3)")
    layers = assemble_helmholtz_layer_matrices(
        mesh, float(wavenumber), targets=field_points, **assembly_kwargs  # type: ignore[arg-type]
    )
    return layers.double_layer @ np.asarray(phi, dtype=np.complex128) - layers.single_layer @ np.asarray(
        normal_velocity, dtype=np.complex128
    )


def acoustic_pressure(
    potential: ComplexArray,
    wavenumber: float,
    *,
    density: float = DENSITY_SEAWATER,
    sound_speed: float = SOUND_SPEED_SEAWATER,
) -> ComplexArray:
    """Return the complex pressure amplitude ``p = i*omega*rho*phi``."""
    omega = float(wavenumber) * float(sound_speed)
    return 1j * omega * float(density) * np.asarray(potential, dtype=np.complex128)


def sound_pressure_level(
    pressure_rms: FloatArray | float, reference_pressure: float = REFERENCE_PRESSURE
) -> FloatArray | float:
    """Return ``20*log10(p_rms/p_ref)`` in dB re 1 uPa."""
    p = np.abs(np.asarray(pressure_rms, dtype=float))
    level = 20.0 * np.log10(np.maximum(p, np.finfo(float).tiny) / float(reference_pressure))
    return float(level) if level.ndim == 0 else level


def rms_from_amplitude(amplitude: ComplexArray | float) -> FloatArray | float:
    """Return ``|p|/sqrt(2)``, the RMS of a harmonic amplitude."""
    return np.abs(np.asarray(amplitude)) / math.sqrt(2.0)


def far_field_amplitude(
    mesh: TriangleMesh,
    wavenumber: float,
    phi: ComplexArray,
    normal_velocity: ComplexArray,
    directions: FloatArray,
) -> ComplexArray:
    """Return the far-field pattern ``f(d)`` with ``phi ~ f(d) exp(i*k*R)/R``."""
    k = float(wavenumber)
    d = np.asarray(directions, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] != 3:
        raise ValueError("directions must have shape (D, 3)")
    centroids = np.asarray(mesh.centroids, dtype=float)
    normals = np.asarray(mesh.normals, dtype=float)
    areas = np.asarray(mesh.areas, dtype=float)
    phase = np.exp(-1j * k * (d @ centroids.T))  # (D, F)
    integrand = (
        -1j * k * (d @ normals.T) * np.asarray(phi, dtype=np.complex128)[None, :]
        - np.asarray(normal_velocity, dtype=np.complex128)[None, :]
    )
    return (phase * integrand * areas[None, :]).sum(axis=1) / _FOUR_PI


def sphere_directions(n_theta: int = 64, n_phi: int = 64) -> tuple[FloatArray, FloatArray]:
    """Return Gauss-Legendre directions on the unit sphere and their weights."""
    nodes, weights = leggauss(n_theta)
    phi_azimuth = 2.0 * math.pi * np.arange(n_phi) / n_phi
    cos_theta = np.repeat(nodes, n_phi)
    sin_theta = np.repeat(np.sqrt(np.clip(1.0 - nodes**2, 0.0, None)), n_phi)
    azimuth = np.tile(phi_azimuth, n_theta)
    directions = np.stack(
        [sin_theta * np.cos(azimuth), sin_theta * np.sin(azimuth), cos_theta], axis=1
    )
    solid_angle_weights = np.repeat(weights, n_phi) * (2.0 * math.pi / n_phi)
    return directions, solid_angle_weights


def scattering_cross_section(
    mesh: TriangleMesh,
    wavenumber: float,
    phi_scattered: ComplexArray,
    normal_velocity_scattered: ComplexArray,
    *,
    n_theta: int = 64,
    n_phi: int = 64,
) -> float:
    """Return ``sigma = int |f(d)|^2 dOmega`` for a unit-amplitude incident wave."""
    directions, weights = sphere_directions(n_theta, n_phi)
    pattern = far_field_amplitude(
        mesh, wavenumber, phi_scattered, normal_velocity_scattered, directions
    )
    return float(np.sum(weights * np.abs(pattern) ** 2))


def horizontal_directivity(
    mesh: TriangleMesh,
    wavenumber: float,
    phi: ComplexArray,
    normal_velocity: ComplexArray,
    *,
    radius: float,
    z: float = 0.0,
    azimuths_deg: Sequence[float] | None = None,
    density: float = DENSITY_SEAWATER,
    sound_speed: float = SOUND_SPEED_SEAWATER,
) -> dict[str, FloatArray]:
    """Return the SPL on a horizontal circle of ``radius`` at depth ``z``.

    This is the ICES 209 directivity cut: level versus bearing in the
    horizontal plane through (or below) the source.
    """
    if azimuths_deg is None:
        azimuths = np.arange(0.0, 360.0, 10.0)
    else:
        azimuths = np.asarray(azimuths_deg, dtype=float)
    radians = np.deg2rad(azimuths)
    points = np.stack(
        [radius * np.cos(radians), radius * np.sin(radians), np.full_like(radians, z)], axis=1
    )
    potential = field_potential(mesh, wavenumber, phi, normal_velocity, points)
    pressure = acoustic_pressure(potential, wavenumber, density=density, sound_speed=sound_speed)
    return {
        "azimuth_deg": azimuths,
        "points": points,
        "pressure": pressure,
        "spl": np.asarray(sound_pressure_level(rms_from_amplitude(pressure))),
    }


def monopole_superposition_pressure(
    mesh: TriangleMesh,
    wavenumber: float,
    normal_velocity: ComplexArray,
    points: FloatArray,
    *,
    density: float = DENSITY_SEAWATER,
    sound_speed: float = SOUND_SPEED_SEAWATER,
) -> ComplexArray:
    """Return the pressure of the *no-scattering* (Kirchhoff) surrogate model.

    Every panel radiates as a free-field simple source of volume velocity
    ``q_j * A_j``, with no interaction between panels::

        p(x) = -i*omega*rho * sum_j q_j A_j G_k(x, y_j).

    The sign is the one of the free-space simple source: a point volume
    velocity ``Q`` satisfies ``(Laplacian + k**2) phi = Q delta`` and therefore
    carries the potential ``phi = -Q G_k`` (verified in the test suite: for a
    uniformly pulsating sphere the surrogate agrees with the full BEM solution,
    which is the same monopole).  This is the model the classical ship-noise
    chain uses (and what an infinite-baffle Rayleigh integral degenerates to, up
    to the factor two on the radiating face).  It is returned here only to
    *quantify* what self-scattering changes; it is not a reference solution.
    """
    centroids = np.asarray(mesh.centroids, dtype=float)
    areas = np.asarray(mesh.areas, dtype=float)
    q = np.asarray(normal_velocity, dtype=np.complex128)
    k = float(wavenumber)
    field_points = np.asarray(points, dtype=float)
    if field_points.ndim == 1:
        field_points = field_points[None, :]
    r = np.linalg.norm(field_points[:, None, :] - centroids[None, :, :], axis=2)
    volume_velocity = q * areas
    potential = -(np.exp(1j * k * r) / (_FOUR_PI * r)) @ volume_velocity
    return acoustic_pressure(potential, k, density=density, sound_speed=sound_speed)


def radiated_power_surface(
    mesh: TriangleMesh,
    wavenumber: float,
    potential: ComplexArray,
    normal_velocity: ComplexArray,
    *,
    density: float = DENSITY_SEAWATER,
    sound_speed: float = SOUND_SPEED_SEAWATER,
) -> float:
    """Return the time-averaged radiated power ``1/2 Re int p v_n^* dS`` (W)."""
    areas = np.asarray(mesh.areas, dtype=float)
    pressure = acoustic_pressure(potential, wavenumber, density=density, sound_speed=sound_speed)
    intensity = 0.5 * np.real(pressure * np.conj(np.asarray(normal_velocity, dtype=np.complex128)))
    return float(np.sum(intensity * areas))


def radiated_power_far_field(
    mesh: TriangleMesh,
    wavenumber: float,
    potential: ComplexArray,
    normal_velocity: ComplexArray,
    *,
    n_theta: int = 96,
    n_phi: int = 96,
    density: float = DENSITY_SEAWATER,
    sound_speed: float = SOUND_SPEED_SEAWATER,
) -> float:
    """Return the radiated power integrated from the far-field pattern (W).

    ``P = omega**2*rho/(2*c) * int |f|**2 dOmega`` with ``phi ~ f exp(i*k*R)/R``.
    Compared with :func:`radiated_power_surface` inside the same solution this
    is a closed-surface energy balance of the BEM (a conservation-law check
    independent of any analytic solution).
    """
    directions, weights = sphere_directions(n_theta, n_phi)
    pattern = far_field_amplitude(mesh, wavenumber, potential, normal_velocity, directions)
    omega = float(wavenumber) * float(sound_speed)
    return float(
        omega**2 * float(density) / (2.0 * float(sound_speed)) * np.sum(weights * np.abs(pattern) ** 2)
    )


# ---------------------------------------------------------------------------
# one-third-octave (ICES 209) band machinery
# ---------------------------------------------------------------------------


def third_octave_centres(
    f_min: float = ICES209_THIRD_OCTAVE_CENTERS[0],
    f_max: float = ICES209_THIRD_OCTAVE_CENTERS[-1],
) -> FloatArray:
    """Return the ICES 209 nominal one-third-octave centres inside ``[f_min, f_max]``."""
    lo, hi = sorted((float(f_min), float(f_max)))
    if lo <= 0.0:
        raise ValueError("f_min must be positive")
    selected = [f for f in ICES209_THIRD_OCTAVE_CENTERS if lo - 1e-9 <= f <= hi + 1e-9]
    if not selected:
        raise ValueError("no nominal one-third-octave centre lies inside the range")
    return np.asarray(selected, dtype=float)


def third_octave_edges(centre: FloatArray | float) -> tuple[FloatArray, FloatArray]:
    """Return the band edges ``f_c * 2**(-1/6)`` and ``f_c * 2**(+1/6)``."""
    fc = np.asarray(centre, dtype=float)
    lower = fc * 2.0 ** (-1.0 / 6.0)
    upper = fc * 2.0 ** (1.0 / 6.0)
    return lower, upper


def third_octave_band_levels(
    centre_frequencies: FloatArray,
    pressure_amplitudes: ComplexArray | FloatArray,
    *,
    reference_pressure: float = REFERENCE_PRESSURE,
) -> FloatArray:
    """Return the band SPL of a line spectrum placed at the band centres.

    ``L_band = 10*log10( sum_i p_rms_i**2 / p_ref**2 )`` over the lines inside
    the band ``[f_c*2**(-1/6), f_c*2**(+1/6)]``; with one line per band this
    equals ``20*log10(p_rms/p_ref)``, the ICES 209 convention for a
    third-octave band level.
    """
    fc = np.asarray(centre_frequencies, dtype=float)
    amplitudes = np.asarray(pressure_amplitudes)
    if fc.shape != amplitudes.shape:
        raise ValueError("centre_frequencies and pressure_amplitudes must have the same shape")
    rms = np.asarray(rms_from_amplitude(amplitudes), dtype=float)
    mean_square = rms**2 / float(reference_pressure) ** 2
    return 10.0 * np.log10(np.maximum(mean_square, np.finfo(float).tiny))


# ---------------------------------------------------------------------------
# analytic references (machine-checkable verdicts)
# ---------------------------------------------------------------------------


def pulsating_sphere_potential(
    radius: float,
    radial_velocity: complex,
    wavenumber: float,
    points: FloatArray,
) -> ComplexArray:
    """Exact velocity potential of a radially pulsating sphere.

    ``phi(r) = C exp(i*k*r)/r`` with ``C = v*a**2*exp(-i*k*a)/(i*k*a - 1)`` so
    that ``dphi/dr = v`` at ``r = a``.
    """
    a = float(radius)
    k = float(wavenumber)
    if a <= 0.0 or k < 0.0:
        raise ValueError("radius must be positive and wavenumber non-negative")
    coefficient = complex(radial_velocity) * a**2 * np.exp(-1j * k * a) / (1j * k * a - 1.0)
    x = np.asarray(points, dtype=float)
    r = np.linalg.norm(x, axis=1)
    if np.any(r < a * (1.0 - 1e-12)):
        raise ValueError("evaluation points must not lie inside the sphere")
    return coefficient * np.exp(1j * k * r) / r


def pulsating_sphere_pressure(
    radius: float,
    radial_velocity: complex,
    wavenumber: float,
    points: FloatArray,
    *,
    density: float = DENSITY_SEAWATER,
    sound_speed: float = SOUND_SPEED_SEAWATER,
) -> ComplexArray:
    """Exact acoustic pressure of a radially pulsating sphere."""
    potential = pulsating_sphere_potential(radius, radial_velocity, wavenumber, points)
    return acoustic_pressure(potential, wavenumber, density=density, sound_speed=sound_speed)


def _sphere_hankel(order: int, x: float) -> complex:
    return complex(spherical_jn(order, x) + 1j * spherical_yn(order, x))


def _sphere_hankel_derivative(order: int, x: float) -> complex:
    dj = float(spherical_jn(order, x, derivative=True))
    dy = float(spherical_yn(order, x, derivative=True))
    return complex(dj + 1j * dy)


def rigid_sphere_mie_coefficients(
    wavenumber: float, radius: float, n_terms: int | None = None
) -> ComplexArray:
    """Return the Mie coefficients ``A_n = -j_n'(ka)/h_n^{(1)'}(ka)`` of a rigid sphere."""
    k = float(wavenumber)
    a = float(radius)
    if a <= 0.0 or k <= 0.0:
        raise ValueError("radius and wavenumber must be positive")
    ka = k * a
    if n_terms is None:
        n_terms = int(math.ceil(ka + 4.0 * ka ** (1.0 / 3.0) + 10.0))
    n_terms = int(max(1, n_terms))
    orders = np.arange(n_terms + 1)
    coefficients = np.array(
        [
            -float(spherical_jn(int(n), ka, derivative=True)) / _sphere_hankel_derivative(int(n), ka)
            for n in orders
        ],
        dtype=np.complex128,
    )
    return coefficients


def rigid_sphere_scattering_cross_section(
    wavenumber: float, radius: float, n_terms: int | None = None
) -> float:
    """Return the exact scattering cross-section of a rigid sphere (Mie series).

    ``sigma = (4*pi/k**2) * sum_n (2n+1) |A_n|**2`` for a unit-amplitude
    incident plane wave.
    """
    k = float(wavenumber)
    coefficients = rigid_sphere_mie_coefficients(k, radius, n_terms)
    orders = np.arange(coefficients.shape[0])
    return float(
        4.0 * math.pi / k**2 * np.sum((2.0 * orders + 1.0) * np.abs(coefficients) ** 2)
    )


def rigid_sphere_scattered_potential(
    wavenumber: float,
    radius: float,
    direction: FloatArray,
    points: FloatArray,
    n_terms: int | None = None,
) -> ComplexArray:
    """Exact scattered potential of a rigid sphere in a unit plane wave (Mie series).

    ``phi_s = sum_n (2n+1) i**n A_n h_n^{(1)}(k r) P_n(cos theta)`` with
    ``theta`` measured from the direction of incidence ``d``.
    """
    k = float(wavenumber)
    a = float(radius)
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm <= 0.0:
        raise ValueError("direction must be non-zero")
    d = d / norm
    x = np.asarray(points, dtype=float)
    r = np.linalg.norm(x, axis=1)
    if np.any(r < a * (1.0 - 1e-9)):
        raise ValueError("evaluation points must not lie inside the sphere")
    cos_theta = np.clip((x @ d) / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0)
    coefficients = rigid_sphere_mie_coefficients(k, a, n_terms)
    total = np.zeros(x.shape[0], dtype=np.complex128)
    for n, coefficient in enumerate(coefficients):
        if abs(coefficient) <= 1e-14:
            continue
        legendre = legval(cos_theta, [0.0] * n + [1.0])
        total += (
            (2 * n + 1)
            * (1j**n)
            * coefficient
            * _sphere_hankel(n, k * r)
            * legendre
        )
    return total


def sphere_mode_wavenumbers(radius: float, n_modes: int = 8) -> FloatArray:
    """Return the irregular (interior Dirichlet) wavenumbers of a sphere ``k*a``.

    ``k*a`` equals a zero of the spherical Bessel function ``j_n``; these are
    the frequencies at which the exterior Neumann boundary-integral equation
    loses uniqueness and CHIEF points are needed.
    """
    a = float(radius)
    if a <= 0.0:
        raise ValueError("radius must be positive")
    zeros: list[float] = []
    n = 0
    while len(zeros) < n_modes:
        grid = np.linspace(1e-6, (n + 1) * math.pi + 4.0, 4000 * (n + 1))
        values = spherical_jn(n, grid)
        sign_change = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0.0)[0]
        for index in sign_change:
            lo, hi = grid[index], grid[index + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if np.sign(spherical_jn(n, lo)) * np.sign(spherical_jn(n, mid)) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            zeros.append(0.5 * (lo + hi))
        n += 1
    return np.asarray(sorted(zeros)[:n_modes], dtype=float)


# ---------------------------------------------------------------------------
# resolution diagnostics (honest validity flags for a frequency sweep)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshAcousticResolution:
    """Acoustic mesh adequacy of one frequency on one mesh."""

    frequency: float
    wavenumber: float
    wavelength: float
    representative_panel_size: float
    panels_per_wavelength: float
    radial_far_field_ok: bool

    @property
    def adequate(self) -> bool:
        """Six panels per wavelength, the usual low-order BEM working rule."""
        return self.panels_per_wavelength >= 1.0 / 6.0


def acoustic_resolution(
    mesh: TriangleMesh,
    frequency: float,
    *,
    sound_speed: float = SOUND_SPEED_SEAWATER,
    panels_per_wavelength_target: float = 1.0 / 6.0,
) -> MeshAcousticResolution:
    """Return the panel-per-wavelength diagnostic of ``mesh`` at ``frequency``."""
    k = float(wavenumber_from_frequency(frequency, sound_speed))
    wavelength = 2.0 * math.pi / k
    scale = math.sqrt(float(mesh.surface_area) / int(mesh.panel_count))
    return MeshAcousticResolution(
        frequency=float(frequency),
        wavenumber=k,
        wavelength=wavelength,
        representative_panel_size=scale,
        panels_per_wavelength=scale / wavelength,
        radial_far_field_ok=scale / wavelength >= panels_per_wavelength_target,
    )


def icosphere_for_acoustic(
    wavenumber: float,
    radius: float,
    *,
    panels_per_wavelength: float = 6.0,
    max_subdivisions: int = 4,
    min_subdivisions: int = 2,
) -> TriangleMesh:
    """Return the coarsest icosphere that resolves ``wavenumber`` on ``radius``.

    ``min_subdivisions`` defaults to 2 (320 panels) rather than 1: an
    icosphere(1) inscribes only ``0.955`` of the sphere area, and that 4.5 %
    volume deficit biases the *low*-frequency monopole level by ``-0.65 dB``
    (measured in ``docs/acoustic3d_ices209/criteria.json``) even though the
    boundary-integral solve itself is converged there.  The default therefore
    keeps the geometry error below the acoustic one.
    """
    k = float(wavenumber)
    if k <= 0.0:
        raise ValueError("wavenumber must be positive")
    wavelength = 2.0 * math.pi / k
    target_edge = wavelength / float(panels_per_wavelength)
    subdivisions = min_subdivisions
    while subdivisions < max_subdivisions:
        mesh = icosphere(subdivisions, radius)
        edge = math.sqrt(2.0 * float(mesh.surface_area) / int(mesh.panel_count))
        if edge <= target_edge:
            break
        subdivisions += 1
    return icosphere(subdivisions, radius)