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

For small-angle linearization the heave/roll/pitch block is

    C[2,2] = rho g A_wp            C[3,3] = rho g grad GM_T
    C[4,4] = rho g grad GM_L       C[4,5] = C[5,4] = -rho g I_xy
    C[2,3] = C[3,2] =  rho g A_wp  y_f
    C[2,4] = C[4,2] = -rho g A_wp  x_f

with ``(x_f, y_f)`` the waterplane centroid, so heave is coupled to roll when
the waterplane centroid is off the centreline and to pitch when it is off the
pitch axis (the usual case for a real hull, whose longitudinal centre of
flotation is not amidships). The coupling vanishes only in the special frame
whose origin sits at the waterplane centroid; it is not a second-order effect
and it is frame dependent by construction (a moment taken about an axis at
``x = 0`` differs from one taken at ``x = x_0`` by ``x_0`` times the heave
force). Surge, sway and yaw have no hydrostatic restoring. The equations of
motion therefore use the full small-angle matrix, not its diagonal.

Derivation: the waterplane rise of a point ``(x, y)`` under the displacement
``(xi_3, xi_4, xi_5)`` is ``zeta = xi_3 + xi_4 y - xi_5 x`` (z up, roll positive
to starboard, pitch positive bow up), and the hydrostatic potential energy is

    E = 1/2 rho g ∬_A zeta^2 dA + 1/2 rho g grad (z_B - z_G) (xi_4^2 + xi_5^2)

whose Hessian is ``C``. The first term gives every coupling entry above; the
second gives the ``KB - KG`` part of the roll and pitch diagonals.
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


def _trace_waterline_loops(edges: NDArray[np.integer] | list[tuple[int, int]]) -> list[list[int]]:
    """Trace every disjoint closed waterline loop from an undirected edge list."""
    adjacency: dict[int, list[int]] = {}
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    for vertex, neighbours in adjacency.items():
        if len(neighbours) != 2:
            raise ValueError(
                "waterline edges must form closed loops: vertex "
                f"{vertex} has {len(neighbours)} incident waterline edges"
            )
    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        previous, current = -1, start
        for _ in range(len(edges) + 1):
            a, b = adjacency[current]
            step = b if a == previous else a
            if step == start:
                break
            if step in visited:
                raise ValueError("waterline edges do not form disjoint loops")
            loop.append(step)
            visited.add(step)
            previous, current = current, step
        else:
            raise ValueError("waterline loop did not close")
        loops.append(loop)
    return loops


def _point_in_polygon(point: FloatArray, polygon: FloatArray) -> bool:
    """Return whether a 2-D point is inside a polygon using ray crossing."""
    px, py = float(point[0]), float(point[1])
    inside = False
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        if (y1 > py) != (y0 > py):
            x_cross = (x0 - x1) * (py - y1) / (y0 - y1) + x1
            if px < x_cross:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def compute_waterplane_properties(
    hull: WettedHullMesh,
) -> tuple[float, FloatArray, FloatArray]:
    """Compute waterplane area, centroid, and second moments.

    The waterplane is the intersection of the hull with z=0. Every closed
    waterline loop of the ``WettedHullMesh`` is traced and the reported
    properties are those of the union of the loops, so a single hull (one loop)
    and a multi-hull arrangement such as a catamaran (one loop per hull) are
    both handled. Loop properties are computed with Green's theorem in 2D and
    combined through the parallel axis theorem about the pooled centroid.

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

    # The waterline must be a union of simple closed loops: a single-hull
    # waterplane has one loop, while a twin-hull (catamaran) waterplane has one
    # loop per hull. Every waterline vertex therefore has exactly two incident
    # waterline edges; anything else makes the enclosed area ambiguous.
    for vertex, neighbours in adj.items():
        if len(neighbours) != 2:
            raise ValueError(
                "waterline edges must form closed loops: vertex "
                f"{vertex} has {len(neighbours)} incident waterline edges"
            )

    # Trace every loop in the waterline graph.
    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        previous = -1
        current = start
        for _ in range(len(edge_list) + 1):
            a, b = adj[current]
            step = b if a == previous else a
            if step == start:
                break
            if step in visited:
                raise ValueError("waterline edges do not form disjoint loops")
            loop.append(step)
            visited.add(step)
            previous, current = current, step
        loops.append(loop)

    # Nested loops represent holes; disjoint outer loops (e.g. twin hulls)
    # remain additive. Determine nesting independently of each loop's winding.
    polygons = [vertices[loop][:, :2] for loop in loops if len(loop) >= 3]
    valid_loops = [loop for loop in loops if len(loop) >= 3]
    nesting_signs: list[float] = []
    for i, loop in enumerate(valid_loops):
        point = vertices[loop[0], :2]
        depth = sum(
            1 for j, polygon in enumerate(polygons)
            if i != j and _point_in_polygon(point, polygon)
        )
        nesting_signs.append(-1.0 if depth % 2 else 1.0)

    total_area = 0.0
    x_moment = 0.0
    y_moment = 0.0
    i_xx_origin = 0.0
    i_yy_origin = 0.0
    i_xy_origin = 0.0

    for loop, nesting_sign in zip(valid_loops, nesting_signs):
        xy = vertices[loop][:, :2]
        x = xy[:, 0]
        y = xy[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        area = 0.5 * np.sum(cross)
        if abs(area) < 1.0e-14:
            continue
        sign = (1.0 if area > 0.0 else -1.0) * nesting_sign
        total_area += nesting_sign * abs(area)
        x_moment += sign * np.sum((x + x_next) * cross) / 6.0
        y_moment += sign * np.sum((y + y_next) * cross) / 6.0
        i_xx_origin += (
            sign
            * np.sum((y**2 + y * y_next + y_next**2) * cross)
            / 12.0
        )
        i_yy_origin += (
            sign
            * np.sum((x**2 + x * x_next + x_next**2) * cross)
            / 12.0
        )
        i_xy_origin += (
            sign
            * np.sum((x * y_next + 2.0 * x * y + 2.0 * x_next * y_next + x_next * y) * cross)
            / 24.0
        )

    if total_area < 1.0e-14:
        return 0.0, np.zeros(2), np.zeros(3)

    centroid = np.array([x_moment / total_area, y_moment / total_area])

    # Shift to the pooled centroid using the parallel axis theorem
    # I_xx = I_xx_origin - A * cy², I_yy = I_yy_origin - A * cx²,
    # I_xy = I_xy_origin - A * cx * cy.
    i_xx = i_xx_origin - total_area * centroid[1] ** 2
    i_yy = i_yy_origin - total_area * centroid[0] ** 2
    i_xy = i_xy_origin - total_area * centroid[0] * centroid[1]

    # Ensure moments are positive (they should be by definition)
    i_xx = abs(i_xx)
    i_yy = abs(i_yy)

    moments = np.array([i_xx, i_yy, i_xy])

    return total_area, centroid, moments


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
        C[2,3] = C[3,2] =  ρ g A_wp y_f   (heave-roll coupling)
        C[2,4] = C[4,2] = -ρ g A_wp x_f   (heave-pitch coupling)
        C[3,4] = C[4,3] = -ρ g I_xy       (roll-pitch coupling)

    where ``(x_f, y_f)`` is the waterplane centroid and ``I_xy`` the waterplane
    product of inertia, both in the frame of ``hull`` (origin on the calm
    waterplane). All other terms are zero: surge, sway and yaw are not
    restored by hydrostatics, and there is no heave-surge or heave-sway
    coupling.

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

    For a stable floating body, GM_T > 0 and GM_L > 0. The coupling entries are
    expressed in the frame of the mesh: if the pitch reference axis is moved to
    ``x = x_0`` the entry becomes ``C[2,4] + ρ g A_wp x_0`` and if the roll axis
    is moved to ``y = y_0`` it becomes ``C[2,3] - ρ g A_wp y_0`` (a moment about
    a shifted axis picks up ``x_0`` times the heave force), so the matrix is
    deliberately frame dependent -- compare matrices only in the same frame. The diagonal uses ``mass = ρ ∇`` (the equilibrium condition); a body
    whose mass does not equal the displaced mass is not in hydrostatic
    equilibrium and its ``KB - KG`` term would need the actual weight.
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
    z_g = float(center_of_gravity[2])  # centre of gravity, waterplane frame

    # Metacentric heights (centroid-frame, i.e. the classical BM = I_c / grad)
    gm_t = kb + bm_t - kg  # transverse metacentric height (roll)
    gm_l = kb + bm_l - kg  # longitudinal metacentric height (pitch)

    # Build restoring matrix (heave, roll, pitch block; surge/sway/yaw are
    # unrestored by hydrostatics).
    rho_g = density * gravity
    c = np.zeros((6, 6), dtype=float)

    # Heave restoring
    c[2, 2] = rho_g * wp_area

    # Waterplane first moments and second moments about the *origin* of the
    # mesh. The displacement field zeta = xi_3 + xi_4 y - xi_5 x is taken about
    # that origin, so the Hessian needs I_xx^origin = I_xx^centroid + A y_f^2
    # (parallel axis), not the centroidal moment that BM_T uses.
    x_f = float(wp_center[0])
    y_f = float(wp_center[1])
    i_xx_origin = float(wp_moments[0]) + wp_area * y_f**2
    i_yy_origin = float(wp_moments[1]) + wp_area * x_f**2
    i_xy_origin = float(wp_moments[2]) + wp_area * x_f * y_f

    # Roll / pitch restoring: rho g (I_origin + grad (z_B - z_G)). In the frame
    # whose origin is the waterplane centroid this is exactly rho g grad GM_T
    # with the classical BM = I_centroid / grad.
    c[3, 3] = rho_g * (i_xx_origin + volume * (center_of_buoyancy[2] - z_g))
    c[4, 4] = rho_g * (i_yy_origin + volume * (center_of_buoyancy[2] - z_g))

    # Waterplane coupling terms: the waterplane rise zeta = xi_3 + xi_4 y
    # - xi_5 x gives a heave-roll term from the y first moment, a heave-pitch
    # term from the x first moment, and a roll-pitch term from I_xy.
    c[2, 3] = c[3, 2] = rho_g * wp_area * y_f
    c[2, 4] = c[4, 2] = -rho_g * wp_area * x_f
    c[3, 4] = c[4, 3] = -rho_g * i_xy_origin

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
