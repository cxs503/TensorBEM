"""Hydrostatic restoring matrix for floating bodies.

The hydrostatic restoring matrix C (6×6) relates small rigid-body displacements
to the resulting restoring forces/moments. For a floating body in equilibrium:

    C[2,2] = ρ g A_wp          (heave restoring)
    C[3,3] = ρ g ∇ GM_T        (roll restoring)
    C[4,4] = ρ g ∇ GM_L        (pitch restoring)

where:
    A_wp  = waterplane area
    ∇     = displaced volume
    GM_T  = transverse metacentric height (roll stability)
    GM_L  = longitudinal metacentric height (pitch stability)

The metacentric heights are:
    GM_T = KB + BM_T - KG
    GM_L = KB + BM_L - KG

where:
    KB    = height of center of buoyancy above keel
    BM_T  = I_T / ∇ (transverse metacentric radius)
    BM_L  = I_L / ∇ (longitudinal metacentric radius)
    I_T   = second moment of waterplane area about transverse axis (y-axis)
    I_L   = second moment of waterplane area about longitudinal axis (x-axis)
    KG    = height of center of gravity above keel

For small-angle linearization, all off-diagonal terms are zero (no cross-coupling
between heave/roll/pitch, and surge/sway/yaw have no hydrostatic restoring).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .geometry3d import WettedHullMesh

FloatArray = NDArray[np.float64]

_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0


@dataclass(frozen=True)
class HydrostaticProperties:
    """Hydrostatic properties of a floating body.

    Attributes
    ----------
    displaced_volume :
        Volume of fluid displaced by the submerged hull (m³).
    center_of_buoyancy :
        Coordinates of the center of buoyancy (m).
    waterplane_area :
        Area of the waterplane cross-section at z=0 (m²).
    waterplane_center :
        Coordinates of the waterplane centroid (m).
    waterplane_moments :
        Second moments of waterplane area [I_xx, I_yy, I_xy] (m⁴).
        I_xx = ∫∫ y² dA (transverse moment, for roll)
        I_yy = ∫∫ x² dA (longitudinal moment, for pitch)
        I_xy = ∫∫ xy dA (product of inertia)
    restoring_matrix :
        6×6 hydrostatic restoring matrix C (N/m or N·m/rad).
    """

    displaced_volume: float
    center_of_buoyancy: FloatArray
    waterplane_area: float
    waterplane_center: FloatArray
    waterplane_moments: FloatArray
    restoring_matrix: FloatArray


def compute_waterplane_properties(
    hull: WettedHullMesh,
) -> tuple[float, FloatArray, FloatArray]:
    """Compute waterplane area, centroid, and second moments.

    The waterplane is the intersection of the hull with z=0. We extract the
    boundary polygon from the WettedHullMesh and compute its properties using
    the divergence theorem (Green's theorem in 2D).

    Parameters
    ----------
    hull :
        Wetted hull mesh with waterplane boundary.

    Returns
    -------
    area :
        Waterplane area (m²).
    centroid :
        Waterplane centroid coordinates (x_c, y_c) (m).
    moments :
        Second moments [I_xx, I_yy, I_xy] (m⁴).
    """
    vertices = hull.vertices
    edges = hull.waterline_edges

    if len(edges) == 0:
        # No waterplane (fully submerged or no intersection)
        return 0.0, np.zeros(2), np.zeros(3)

    # Extract waterplane polygon vertices in order
    # Build adjacency to trace the boundary
    edge_list = [(int(e[0]), int(e[1])) for e in edges]
    adj: dict[int, list[int]] = {}
    for i, j in edge_list:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    # Trace the boundary polygon
    if not adj:
        return 0.0, np.zeros(2), np.zeros(3)

    start = next(iter(adj))
    ordered = [start]
    visited = {start}
    current = start

    while True:
        neighbors = adj[current]
        next_v = None
        for n in neighbors:
            if n not in visited:
                next_v = n
                break
        if next_v is None:
            # Try to close the loop
            for n in neighbors:
                if n == start and len(ordered) > 2:
                    break
            break
        ordered.append(next_v)
        visited.add(next_v)
        current = next_v

    if len(ordered) < 3:
        return 0.0, np.zeros(2), np.zeros(3)

    # Get (x, y) coordinates of waterplane vertices
    xy = np.array([[vertices[i, 0], vertices[i, 1]] for i in ordered])

    # Compute area and moments using the shoelace formula
    # A = (1/2) Σ (x_i * y_{i+1} - x_{i+1} * y_i)
    # I_xx = (1/12) Σ (y_i² + y_i*y_{i+1} + y_{i+1}²) * (x_i*y_{i+1} - x_{i+1}*y_i)
    # I_yy = (1/12) Σ (x_i² + x_i*x_{i+1} + x_{i+1}²) * (x_i*y_{i+1} - x_{i+1}*y_i)
    # I_xy = (1/24) Σ (x_i*y_{i+1} + 2*x_i*y_i + 2*x_{i+1}*y_{i+1} + x_{i+1}*y_i) * (x_i*y_{i+1} - x_{i+1}*y_i)

    n = len(xy)
    x = xy[:, 0]
    y = xy[:, 1]

    # Shoelace terms
    cross = np.zeros(n)
    for i in range(n):
        j = (i + 1) % n
        cross[i] = x[i] * y[j] - x[j] * y[i]

    area = 0.5 * np.sum(cross)

    if abs(area) < 1e-14:
        return 0.0, np.zeros(2), np.zeros(3)

    # Centroid
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
    centroid = np.array([cx, cy])

    # Second moments about origin
    i_xx_orig = 0.0
    i_yy_orig = 0.0
    i_xy_orig = 0.0

    for i in range(n):
        j = (i + 1) % n
        c = cross[i]
        i_xx_orig += (y[i] ** 2 + y[i] * y[j] + y[j] ** 2) * c
        i_yy_orig += (x[i] ** 2 + x[i] * x[j] + x[j] ** 2) * c
        i_xy_orig += (x[i] * y[j] + 2 * x[i] * y[i] + 2 * x[j] * y[j] + x[j] * y[i]) * c

    i_xx_orig /= 12.0
    i_yy_orig /= 12.0
    i_xy_orig /= 24.0

    # Shift to centroid using parallel axis theorem
    # I_xx = I_xx_orig - A * cy²
    # I_yy = I_yy_orig - A * cx²
    # I_xy = I_xy_orig - A * cx * cy
    i_xx = i_xx_orig - area * cy**2
    i_yy = i_yy_orig - area * cx**2
    i_xy = i_xy_orig - area * cx * cy

    # Ensure moments are positive (they should be by definition)
    i_xx = abs(i_xx)
    i_yy = abs(i_yy)

    moments = np.array([i_xx, i_yy, i_xy])

    return abs(area), centroid, moments


def compute_displaced_volume_and_buoyancy(
    hull: WettedHullMesh,
) -> tuple[float, FloatArray]:
    """Compute displaced volume and center of buoyancy.

    Uses the divergence theorem: V = (1/3) ∫∫ r·n dS over the submerged hull.
    The center of buoyancy is the volume-weighted average of the centroid of
    each tetrahedron formed by the origin and each surface triangle.

    Parameters
    ----------
    hull :
        Wetted hull mesh.

    Returns
    -------
    volume :
        Displaced volume (m³).
    center :
        Center of buoyancy coordinates (m).
    """
    vertices = hull.vertices
    faces = hull.faces

    if len(faces) == 0:
        return 0.0, np.zeros(3)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Signed volume of tetrahedra formed with origin
    # V_tet = (1/6) * v0 · (v1 × v2)
    cross = np.cross(v1, v2)
    signed_volumes = np.sum(v0 * cross, axis=1) / 6.0

    total_volume = np.sum(signed_volumes)

    if abs(total_volume) < 1e-14:
        return 0.0, np.zeros(3)

    # Center of buoyancy: volume-weighted centroid of tetrahedra
    # Centroid of tetrahedron = (v0 + v1 + v2 + 0) / 4
    centroids = (v0 + v1 + v2) / 4.0
    center = np.sum(signed_volumes[:, np.newaxis] * centroids, axis=0) / total_volume

    return abs(total_volume), center


def compute_hydrostatic_restoring_matrix(
    hull: WettedHullMesh,
    mass: float,
    center_of_gravity: FloatArray,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> HydrostaticProperties:
    """Compute the 6×6 hydrostatic restoring matrix C.

    For a floating body in equilibrium, the restoring matrix relates small
    displacements to restoring forces/moments:

        C[2,2] = ρ g A_wp          (heave)
        C[3,3] = ρ g ∇ GM_T        (roll)
        C[4,4] = ρ g ∇ GM_L        (pitch)

    All other terms are zero under the small-angle assumption.

    Parameters
    ----------
    hull :
        Wetted hull mesh (intersection of floating body with free surface).
    mass :
        Mass of the floating body (kg). For equilibrium, mass = ρ ∇.
    center_of_gravity :
        Coordinates of center of gravity (m).
    density :
        Fluid density (kg/m³). Default: 1025.0 (seawater).
    gravity :
        Gravitational acceleration (m/s²). Default: 9.80665.

    Returns
    -------
    properties :
        Hydrostatic properties including the restoring matrix.

    Notes
    -----
    The metacentric heights are computed as:
        GM_T = KB + BM_T - KG
        GM_L = KB + BM_L - KG

    where KB is the vertical position of the center of buoyancy, BM_T = I_T/∇
    is the transverse metacentric radius, and KG is the vertical position of
    the center of gravity.

    For a stable floating body, GM_T > 0 and GM_L > 0.
    """
    center_of_gravity = np.asarray(center_of_gravity, dtype=float)

    # Compute waterplane properties
    wp_area, wp_center, wp_moments = compute_waterplane_properties(hull)

    # Compute displaced volume and center of buoyancy
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)

    # Metacentric radii
    # I_xx is the transverse moment (about x-axis, for roll)
    # I_yy is the longitudinal moment (about y-axis, for pitch)
    i_xx = wp_moments[0]  # transverse moment for roll
    i_yy = wp_moments[1]  # longitudinal moment for pitch

    if volume > 1e-14:
        bm_t = i_xx / volume  # transverse metacentric radius
        bm_l = i_yy / volume  # longitudinal metacentric radius
    else:
        bm_t = 0.0
        bm_l = 0.0

    # Vertical positions (z-coordinate, z=0 is free surface, z<0 is submerged)
    # For metacentric height, we need distances from keel (lowest point)
    # Find the keel (minimum z-coordinate of the hull)
    z_min = np.min(hull.vertices[:, 2])
    
    # Heights above keel
    kb = center_of_buoyancy[2] - z_min  # height of buoyancy center above keel
    kg = center_of_gravity[2] - z_min  # height of gravity center above keel

    # Metacentric heights
    gm_t = kb + bm_t - kg  # transverse metacentric height (roll)
    gm_l = kb + bm_l - kg  # longitudinal metacentric height (pitch)

    # Build restoring matrix
    c = np.zeros((6, 6), dtype=float)

    # Heave restoring
    c[2, 2] = density * gravity * wp_area

    # Roll restoring
    c[3, 3] = density * gravity * volume * gm_t

    # Pitch restoring
    c[4, 4] = density * gravity * volume * gm_l

    return HydrostaticProperties(
        displaced_volume=volume,
        center_of_buoyancy=center_of_buoyancy,
        waterplane_area=wp_area,
        waterplane_center=wp_center,
        waterplane_moments=wp_moments,
        restoring_matrix=c,
    )


def compute_mass_matrix(
    mass: float,
    center_of_gravity: FloatArray,
    inertia: FloatArray | None = None,
) -> FloatArray:
    """Compute the 6×6 rigid-body mass matrix M.

    The mass matrix relates accelerations to inertial forces:

        M = [[m*I_3,    -m*S^T],
             [-m*S,  J + m*S*S^T]]

    where m is the mass, S is the skew-symmetric matrix of the center of
    gravity coordinates, and J is the inertia tensor about the origin.

    For simplicity, if inertia is not provided, we use a diagonal approximation
    with the moments of inertia computed from the mass and CG.

    Parameters
    ----------
    mass :
        Body mass (kg).
    center_of_gravity :
        Center of gravity coordinates (m).
    inertia :
        Inertia tensor about the origin (3×3). If None, a diagonal approximation
        is used.

    Returns
    -------
    M :
        6×6 mass matrix (kg or kg·m²).
    """
    cg = np.asarray(center_of_gravity, dtype=float)

    # Skew-symmetric matrix of CG
    # S = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    x, y, z = cg
    s = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    # Mass matrix blocks
    m11 = mass * np.eye(3)
    m12 = -mass * s.T
    m21 = -mass * s

    if inertia is not None:
        j = np.asarray(inertia, dtype=float)
    else:
        # Default approximation: assume body has radius of gyration k = 0.4 * characteristic_length
        # where characteristic_length is estimated from CG distance to origin
        # This provides a more realistic inertia than point-mass approximation
        r = np.linalg.norm(cg)
        if r > 1e-10:
            # Use radius of gyration k = 0.4 * r (typical for ship-like bodies)
            k = 0.4 * r
            # Inertia about CG: J_cg = m * k^2 * I
            # Inertia about origin: J = J_cg + m * (|cg|^2 * I - cg⊗cg)
            j_cg = mass * k**2 * np.eye(3)
            j = j_cg + mass * (r**2 * np.eye(3) - np.outer(cg, cg))
        else:
            # CG at origin: use a default radius of gyration
            k = 0.4  # meters
            j = mass * k**2 * np.eye(3)

    m22 = j + mass * (s @ s.T)

    # Assemble full 6×6 matrix
    m = np.zeros((6, 6), dtype=float)
    m[0:3, 0:3] = m11
    m[0:3, 3:6] = m12
    m[3:6, 0:3] = m21
    m[3:6, 3:6] = m22

    return m
