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

For small-angle linearization the matrix is symmetric and surge/sway/yaw carry no
hydrostatic restoring diagonal, but the off-diagonal terms are **not** zero in
general: they are the waterplane first moments (``C[2,3]``, ``C[2,4]``) and the
waterplane product moment (``C[3,4]``), so a hull whose waterplane centroid is
displaced from the origin (``LCF``/``TCF`` != 0) couples heave into roll and
pitch. See :func:`compute_hydrostatic_restoring_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _waterplane_integrals_about_origin(
    wp_area: float,
    wp_center: FloatArray,
    wp_moments: FloatArray,
) -> tuple[float, float, float, float, float]:
    """Return ``(S_x, S_y, I_xx, I_yy, I_xy)``: waterplane integrals about the origin.

    :func:`compute_waterplane_properties` reports the area, the centroid and the
    second moments **about the centroid**. The restoring matrix needs the same
    integrals about the coordinate origin, obtained with the parallel-axis
    theorem::

        S_x  = A x_c,          S_y  = A y_c
        I_xx = I_xx^c + A y_c^2
        I_yy = I_yy^c + A x_c^2
        I_xy = I_xy^c + A x_c y_c

    Maintaining one source of truth (the centroidal properties) keeps the
    waterline-integration code single-purpose; the shift is exact algebra, so no
    accuracy is lost.
    """
    area = float(wp_area)
    x_c = float(wp_center[0])
    y_c = float(wp_center[1])
    i_xx_c = float(wp_moments[0])
    i_yy_c = float(wp_moments[1])
    i_xy_c = float(wp_moments[2])

    s_x = area * x_c
    s_y = area * y_c
    i_xx = i_xx_c + area * y_c**2
    i_yy = i_yy_c + area * x_c**2
    i_xy = i_xy_c + area * x_c * y_c
    return s_x, s_y, i_xx, i_yy, i_xy


def compute_hydrostatic_restoring_matrix(
    hull: WettedHullMesh,
    mass: float,
    center_of_gravity: FloatArray,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> HydrostaticProperties:
    """Compute the 6×6 hydrostatic restoring matrix ``C``.

    The restoring matrix is the Hessian of the hydrostatic potential energy in
    the six rigid-body modes ``(surge, sway, heave, roll, pitch, yaw)`` at
    equilibrium flotation. With the waterplane integrals taken *about the
    coordinate origin*

        ``A = ∫∫ dA``, ``S_x = ∫∫ x dA``, ``S_y = ∫∫ y dA``,
        ``I_xx = ∫∫ y² dA``, ``I_yy = ∫∫ x² dA``, ``I_xy = ∫∫ x y dA``,

    and ``∇`` the displaced volume, ``(x_B, y_B, z_B)`` the centre of buoyancy and
    ``(x_G, y_G, z_G)`` the centre of gravity, the non-zero entries are
    (Journée & Massie, *Offshore Hydromechanics*, §3.2.2)::

        C[2,2]          = ρ g A                            heave
        C[2,3] = C[3,2] = +ρ g S_y                          heave-roll
        C[2,4] = C[4,2] = -ρ g S_x                          heave-pitch
        C[3,3]          = ρ g I_xx + ρ g ∇ (z_B - z_G)      roll
        C[3,4] = C[4,3] = -ρ g I_xy                         roll-pitch
        C[4,4]          = ρ g I_yy + ρ g ∇ (z_B - z_G)      pitch

    The opposite signs of the two heave couplings are not a typo: the vertical
    displacement of a point on the waterplane is ``w = ξ3 + y ξ4 - x ξ5`` for
    rotations about the positive x and y axes, so the heave-roll cross term of
    ``½ρg ∫∫ w² dA`` carries ``+S_y`` while the heave-pitch term carries ``-S_x``.
    Both are confirmed against an independent numerical Hessian of the potential
    energy (``tensorbem.benchmarks.hydrostatic_energy_reference``).

    Surge, sway and yaw carry no hydrostatic restoring at all: the hydrostatic
    force and the body weight are vertical, and a vertical force exerts no moment
    about a vertical axis, so all entries in the surge, sway and yaw rows and
    columns vanish (verified as ~1e-9 of the largest entry by the same energy
    Hessian). The matrix is symmetric because it is a Hessian. The diagonal roll/pitch entries are the
    familiar metacentric stiffness: ``C[3,3] = ρ g ∇ GM_T`` with
    ``GM_T = I_T^centroid / ∇ + z_B - z_G`` and ``I_T = I_xx``.

    The off-diagonal entries are not zero in general. ``-C[2,3] / (ρ g)`` and
    ``-C[2,4] / (ρ g)`` are the beamwise and lengthwise first moments of the
    waterplane, so a hull whose waterplane centroid is displaced from the origin
    (``TCF``/``LCF`` != 0 — i.e. almost every real ship) couples heave into roll
    and pitch; ``C[3,4]`` couples roll and pitch through the waterplane product
    moment. A body whose waterplane centroid sits on the origin (a symmetric
    barge, a floating hemisphere) and whose waterplane has a symmetry axis gets a
    strictly diagonal heave/roll/pitch block, which is the case every earlier
    validation case in this repository exercised.

    Parameters
    ----------
    hull :
        Wetted hull mesh (intersection of floating body with free surface).
    mass :
        Mass of the floating body (kg). For equilibrium, ``mass = ρ ∇``; see the
        notes below.
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
    The gravity terms are evaluated at the equilibrium mass ``m = ρ ∇``. This is
    the same assumption the metacentric form ``GM = KB + BM - KG`` makes, and it
    keeps the result independent of a mass supplied out of equilibrium (a
    floating body out of equilibrium is not a restoring problem — it
    accelerates). ``mass`` is therefore accepted for API compatibility and is
    not consumed; the equilibrium condition is what fixes `m`.

    Fully submerged bodies (no waterplane) have no heave, roll or pitch restoring
    from the free surface: their ``C`` block is exactly zero.
    """
    center_of_gravity = np.asarray(center_of_gravity, dtype=float)

    # Waterplane geometry and submerged volume
    wp_area, wp_center, wp_moments = compute_waterplane_properties(hull)
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)

    s_x, s_y, i_xx, i_yy, i_xy = _waterplane_integrals_about_origin(
        wp_area, wp_center, wp_moments
    )

    z_g = float(center_of_gravity[2])
    z_b = float(center_of_buoyancy[2])

    rho_g = density * gravity
    # Equilibrium mass (m = ρ ∇): potential energy of the body weight.
    weight = rho_g * volume

    c = np.zeros((6, 6), dtype=float)

    # Heave-roll and heave-pitch: the waterplane first moments. The signs differ
    # because w = xi3 + y*xi4 - x*xi5 (rotations about +x and +y move a point
    # vertically in opposite directions).
    c[2, 2] = rho_g * wp_area
    c[2, 3] = c[3, 2] = rho_g * s_y
    c[2, 4] = c[4, 2] = -rho_g * s_x

    # Roll: waterplane transverse inertia (about the origin) plus the
    # buoyancy/weight couple through the vertical separation of B and G.
    vertical_separation = weight * (z_b - z_g)
    c[3, 3] = rho_g * i_xx + vertical_separation
    c[4, 4] = rho_g * i_yy + vertical_separation

    # Roll-pitch: waterplane product of inertia about the origin.
    c[3, 4] = c[4, 3] = -rho_g * i_xy

    # Surge, sway and yaw stay identically zero: the hydrostatic force and the
    # weight are vertical, and a vertical force has no moment about a vertical
    # axis, so no yaw entry can be non-zero.

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
