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

Linearising about the static equilibrium of a body whose reference frame has its
origin ``O`` on the calm waterplane ``z = 0``, the *complete* matrix also carries
the heave-roll-pitch cross-coupling terms that arise whenever the waterplane is
eccentric or has a non-zero product of inertia about ``O``:

    C[2,3] = C[3,2] =  rho g A_wp y_c      (heave <-> roll)
    C[2,4] = C[4,2] = -rho g A_wp x_c      (heave <-> pitch)
    C[3,4] = C[4,3] = -rho g I_xy_O        (roll  <-> pitch)

and the diagonal entries are referred to the roll/pitch axes *through O*, i.e.
``C[3,3] = rho g (I_xx_O + volume*z_B) - mass*g*z_G`` (and likewise for pitch),
where ``I_xx_O`` is the waterplane second moment about the x-axis through ``O``.

:func:`compute_hydrostatic_restoring_matrix` returns only the diagonal reference
form (its historical contract); :func:`compute_hydrostatic_restoring_matrix_full`
returns the complete matrix and reports how far the body is from the diagonal
domain.  Surge, sway and yaw carry no hydrostatic restoring: the potential energy
depends on the vertical position of the hull only, so those three rows/columns
are identically zero.
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

    # Shoelace terms are accumulated over every loop. Each loop is oriented
    # individually (clockwise or counter-clockwise) and normalised to a positive
    # area by its own sign, so the pooled properties are the union of the loops.
    total_area = 0.0
    x_moment = 0.0
    y_moment = 0.0
    i_xx_origin = 0.0
    i_yy_origin = 0.0
    i_xy_origin = 0.0

    for loop in loops:
        if len(loop) < 3:
            continue
        xy = vertices[loop][:, :2]
        x = xy[:, 0]
        y = xy[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        area = 0.5 * np.sum(cross)
        if abs(area) < 1.0e-14:
            continue
        sign = 1.0 if area > 0.0 else -1.0
        total_area += abs(area)
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


#: Physical source of every non-zero entry of the complete 6x6 hydrostatic
#: restoring matrix, keyed by ``(row, column)`` in the ordering
#: ``(surge, sway, heave, roll, pitch, yaw)``.  The symbols are: ``A`` waterplane
#: area, ``x_c, y_c`` waterplane centroid, ``I_xx_O, I_yy_O, I_xy_O`` waterplane
#: second moments about the axes through the reference origin ``O``,
#: ``volume`` displaced volume, ``z_B`` / ``z_G`` buoyancy / gravity centre
#: heights (``z = 0`` at the calm waterplane, negative below it), ``mass`` body
#: mass, ``rho`` density and ``g`` gravity.
RESTORING_MATRIX_TERM_SOURCES: tuple[tuple[int, int, str, str], ...] = (
    (
        2,
        2,
        "rho g A",
        "change of displaced volume in heave (waterplane area times the vertical "
        "sinkage); pure buoyancy term",
    ),
    (
        2,
        3,
        "rho g A y_c",
        "heave-roll coupling: the buoyancy lost/gained in heave acts through the "
        "waterplane centroid, which is offset by y_c from the roll axis through O",
    ),
    (
        2,
        4,
        "-rho g A x_c",
        "heave-pitch coupling: the buoyancy lost/gained in heave acts through the "
        "waterplane centroid, which is offset by x_c from the pitch axis through O",
    ),
    (
        3,
        3,
        "rho g (I_xx_O + volume z_B) - mass g z_G",
        "roll restoring = waterplane transverse second moment about the x-axis "
        "through O (BM term) + rotation of the buoyancy vector about the centre of "
        "buoyancy (KB term) - vertical offset of weight (KG term)",
    ),
    (
        3,
        4,
        "-rho g I_xy_O",
        "roll-pitch coupling: a roll rotates the waterplane about x, a pitch does "
        "the same about y, and the mixed second moment of the waterplane couples "
        "the two changes of the displaced-volume moment (non-zero only when the "
        "waterplane product of inertia about O is non-zero)",
    ),
    (
        4,
        4,
        "rho g (I_yy_O + volume z_B) - mass g z_G",
        "pitch restoring: longitudinal waterplane second moment about the y-axis "
        "through O (BM_L) + KB term - KG term",
    ),
)


@dataclass(frozen=True)
class FullRestoringProperties:
    """Complete 6x6 hydrostatic restoring matrix and its validity diagnostics.

    Attributes
    ----------
    displaced_volume, center_of_buoyancy, waterplane_area, waterplane_center,
    waterplane_moments :
        The same hydrostatic integrals that
        :func:`compute_hydrostatic_restoring_matrix` reports; the waterplane
        moments are referenced to the waterplane centroid.
    restoring_matrix :
        Complete 6x6 matrix ``C`` in the ordering
        ``(surge, sway, heave, roll, pitch, yaw)`` with the sign convention
        ``F_restoring = -C xi``.
    diagonal_matrix :
        The historical diagonal-only matrix, identical (bit for bit) to
        ``compute_hydrostatic_restoring_matrix(...).restoring_matrix``.
    coupling_terms :
        ``restoring_matrix - diagonal_matrix``: every entry contributed by the
        coupling / parallel-axis corrections.
    waterplane_moments_about_origin :
        ``[I_xx_O, I_yy_O, I_xy_O]``, the waterplane second moments about the
        reference origin ``O`` (m⁴).
    equilibrium_relative_mismatch :
        ``|mass - rho*volume| / (rho*volume)``.  The linearisation assumes the
        body floats in equilibrium; a non-zero value means the weight terms are
        not balanced by buoyancy.
    diagonal_adequate :
        ``True`` when every coupling term is negligible relative to the diagonal
        restoring magnitude, i.e. when the diagonal-only matrix is justified.
    notes :
        Human-readable findings, one per detected deviation from the diagonal
        domain.
    """

    displaced_volume: float
    center_of_buoyancy: FloatArray
    waterplane_area: float
    waterplane_center: FloatArray
    waterplane_moments: FloatArray
    restoring_matrix: FloatArray
    diagonal_matrix: FloatArray
    coupling_terms: FloatArray
    waterplane_moments_about_origin: FloatArray
    equilibrium_relative_mismatch: float
    diagonal_adequate: bool
    notes: tuple[str, ...]

    def as_hydrostatic_properties(self) -> HydrostaticProperties:
        """Wrap the complete matrix so it can be fed to the RAO chain."""
        return HydrostaticProperties(
            displaced_volume=self.displaced_volume,
            center_of_buoyancy=self.center_of_buoyancy,
            waterplane_area=self.waterplane_area,
            waterplane_center=self.waterplane_center,
            waterplane_moments=self.waterplane_moments,
            restoring_matrix=self.restoring_matrix,
        )


def waterplane_second_moments_about_origin(
    waterplane_area: float,
    waterplane_center: FloatArray,
    waterplane_moments: FloatArray,
) -> FloatArray:
    """Transfer the waterplane second moments from the centroid to the origin.

    ``compute_waterplane_properties`` reports the moments about axes through the
    waterplane *centroid*.  The restoring matrix, however, must be referred to
    the roll/pitch axes through the body reference origin ``O`` (the point about
    which the rigid-body degrees of freedom are defined).  The parallel-axis
    theorem gives::

        I_xx_O = I_xx_c + A y_c^2
        I_yy_O = I_yy_c + A x_c^2
        I_xy_O = I_xy_c + A x_c y_c

    Parameters
    ----------
    waterplane_area :
        Waterplane area ``A`` (m²).
    waterplane_center :
        Waterplane centroid ``(x_c, y_c)`` (m).
    waterplane_moments :
        Centroidal moments ``[I_xx_c, I_yy_c, I_xy_c]`` (m⁴).

    Returns
    -------
    moments :
        ``[I_xx_O, I_yy_O, I_xy_O]`` about the origin (m⁴).
    """
    area = float(waterplane_area)
    center = np.asarray(waterplane_center, dtype=float)
    moments = np.asarray(waterplane_moments, dtype=float)
    x_c = float(center[0])
    y_c = float(center[1])
    return np.array(
        [
            float(moments[0]) + area * y_c * y_c,
            float(moments[1]) + area * x_c * x_c,
            float(moments[2]) + area * x_c * y_c,
        ]
    )


def compute_hydrostatic_restoring_matrix_full(
    hull: WettedHullMesh,
    mass: float,
    center_of_gravity: FloatArray,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    relative_tolerance: float = 1.0e-6,
) -> FullRestoringProperties:
    """Compute the complete 6x6 hydrostatic restoring matrix, coupling included.

    The matrix is the Hessian of the potential energy
    (weight + displaced-fluid) with respect to the six rigid-body modes, taken
    about the static equilibrium and linearised for small displacements.  With
    the body reference origin ``O`` on the calm waterplane ``z = 0``::

        C = [[0,   0,   0,        0,         0,         0],
             [0,   0,   0,        0,         0,         0],
             [0,   0,   rho g A,  rho g A y_c, -rho g A x_c, 0],
             [0,   0,   rho g A y_c, S_44,    -rho g I_xy_O, 0],
             [0,   0,  -rho g A x_c, -rho g I_xy_O, S_55,    0],
             [0,   0,   0,        0,         0,         0]]

        S_44 = rho g (I_xx_O + volume z_B) - mass g z_G
        S_55 = rho g (I_yy_O + volume z_B) - mass g z_G

    Derivation (each term's physical source is tagged in
    :data:`RESTORING_MATRIX_TERM_SOURCES`):

    * The vertical coordinate of a body point ``r`` after a small motion is
      ``Z(r) = xi_3 - xi_5 x + xi_4 y + z (1 - (xi_4^2 + xi_5^2)/2)``; the
      quadratic bracket is what produces the KB/KG terms below.
    * Weight: ``V_w = mass g Z(r_G)``; its Hessian contributes ``-mass g z_G``
      to both C[3,3] and C[4,4] and nothing else (a vertical force at the centre
      of gravity produces moments only about the horizontal axes).
    * Buoyancy: ``V_b = -rho g int_{Z<=0} Z dV``.  Expanding the displaced
      region to second order gives ``rho g/2 int_{A} delta^2 dA`` with
      ``delta = xi_3 + xi_4 y - xi_5 x``, plus ``rho g (xi_4^2 + xi_5^2)/2
      * volume z_B`` from the rotation of the region.  The first term yields
      ``rho g A``, ``rho g A y_c``, ``-rho g A x_c``, ``rho g I_xx_O``,
      ``-rho g I_xy_O`` and ``rho g I_yy_O``; the second yields the ``volume z_B``
      (KB) contributions.
    * Surge, sway and yaw do not appear: the potential energy depends on the
      vertical position only, so those rows and columns vanish identically.

    Parameters
    ----------
    hull :
        Wetted hull mesh (submerged part of the floating body, ``z <= 0``).
    mass :
        Body mass (kg).  ``mass = rho*volume`` for a body in equilibrium; any
        mismatch is reported in ``equilibrium_relative_mismatch``.
    center_of_gravity :
        Centre of gravity ``(x_G, y_G, z_G)`` (m) in the body frame.
    density :
        Fluid density (kg/m³).  Default: 1025.0 (seawater).
    gravity :
        Gravitational acceleration (m/s²).  Default: 9.80665.
    relative_tolerance :
        Threshold used to decide whether the coupling terms are negligible
        (``diagonal_adequate``) and whether the body floats in equilibrium.

    Returns
    -------
    properties :
        :class:`FullRestoringProperties` with the complete matrix, the diagonal
        reference matrix, the coupling corrections and validity notes.

    Notes
    -----
    The diagonal entries are assembled as
    ``diagonal_reference + correction`` so that a hull which satisfies the
    diagonal domain (waterplane centroid on the reference vertical axis, zero
    product of inertia, mass equal to the displaced mass) reproduces
    :func:`compute_hydrostatic_restoring_matrix` bit for bit.  This makes the
    coupling corrections directly readable as additive deltas.
    """
    center_of_gravity = np.asarray(center_of_gravity, dtype=float)

    wp_area, wp_center, wp_moments = compute_waterplane_properties(hull)
    volume, center_of_buoyancy = compute_displaced_volume_and_buoyancy(hull)

    i_xx = wp_moments[0]
    i_yy = wp_moments[1]

    if volume > 1e-14:
        bm_t = i_xx / volume
        bm_l = i_yy / volume
    else:
        bm_t = 0.0
        bm_l = 0.0

    z_min = np.min(hull.vertices[:, 2])
    kb = center_of_buoyancy[2] - z_min
    kg = center_of_gravity[2] - z_min

    gm_t = kb + bm_t - kg
    gm_l = kb + bm_l - kg

    # --- diagonal reference matrix (identical to the historical contract) ---
    diagonal = np.zeros((6, 6), dtype=float)
    diagonal[2, 2] = density * gravity * wp_area
    diagonal[3, 3] = density * gravity * volume * gm_t
    diagonal[4, 4] = density * gravity * volume * gm_l

    # --- corrections: parallel-axis transfer of the roll/pitch reference axis
    # (waterplane centroid -> reference origin) and the flotation mismatch of
    # the weight term.  Both vanish identically for a hull inside the diagonal
    # domain, so the sum below is bit-for-bit the diagonal reference there. ---
    moments_about_origin = waterplane_second_moments_about_origin(
        wp_area, wp_center, wp_moments
    )
    displaced_mass = density * volume
    vertical_mismatch = (displaced_mass - mass) * gravity * center_of_gravity[2]
    roll_axis_correction = gravity * (
        density * wp_area * wp_center[1] * wp_center[1]
    )
    pitch_axis_correction = gravity * (
        density * wp_area * wp_center[0] * wp_center[0]
    )

    meter = density * gravity
    full = diagonal.copy()
    full[3, 3] = diagonal[3, 3] + roll_axis_correction + vertical_mismatch
    full[4, 4] = diagonal[4, 4] + pitch_axis_correction + vertical_mismatch
    # heave-roll: buoyancy increment of a heave acts through the waterplane
    # centroid, offset y_c from the roll axis through O.
    full[2, 3] += meter * wp_area * wp_center[1]
    full[3, 2] += meter * wp_area * wp_center[1]
    # heave-pitch: same mechanism about the pitch axis (sign from xi_5 = -theta
    # acting on the x-offset of the waterplane centroid).
    full[2, 4] += -meter * wp_area * wp_center[0]
    full[4, 2] += -meter * wp_area * wp_center[0]
    # roll-pitch: mixed waterplane second moment about the origin.
    full[3, 4] += -meter * moments_about_origin[2]
    full[4, 3] += -meter * moments_about_origin[2]

    coupling = full - diagonal

    # --- validity diagnostics ------------------------------------------------
    scale = max(abs(full[2, 2]), abs(full[3, 3]), abs(full[4, 4]), 1.0e-30)
    coupling_scale = float(np.max(np.abs(coupling)))
    equilibrium_relative_mismatch = (
        abs(displaced_mass - mass) / displaced_mass if displaced_mass > 0.0 else 0.0
    )

    notes: list[str] = []
    waterplane_scale = max(
        math.sqrt(wp_area) if wp_area > 0.0 else 0.0,
        float(np.max(np.abs(wp_center))) if wp_area > 0.0 else 0.0,
        1.0,
    )
    if abs(wp_center[1]) > relative_tolerance * waterplane_scale:
        notes.append(
            "waterplane centroid is offset in y from the roll axis through the "
            f"origin (y_c = {wp_center[1]:.6g} m): heave-roll coupling "
            f"C[2,3] = {full[2, 3]:.6g} N/rad and a parallel-axis correction of "
            f"{roll_axis_correction:.6g} on C[3,3] are required"
        )
    if abs(wp_center[0]) > relative_tolerance * waterplane_scale:
        notes.append(
            "waterplane centroid is offset in x from the pitch axis through the "
            f"origin (x_c = {wp_center[0]:.6g} m): heave-pitch coupling "
            f"C[2,4] = {full[2, 4]:.6g} N/rad and a parallel-axis correction of "
            f"{pitch_axis_correction:.6g} on C[4,4] are required"
        )
    inertia_scale = max(
        abs(moments_about_origin[0]), abs(moments_about_origin[1]), 1.0e-30
    )
    if abs(moments_about_origin[2]) > relative_tolerance * inertia_scale:
        notes.append(
            "waterplane product of inertia about the origin is non-zero "
            f"(I_xy_O = {moments_about_origin[2]:.6g} m^4): roll-pitch coupling "
            f"C[3,4] = {full[3, 4]:.6g} N/rad is required"
        )
    if equilibrium_relative_mismatch > relative_tolerance:
        notes.append(
            "body is not in flotation equilibrium: mass exceeds/falls short of "
            f"the displaced mass by {equilibrium_relative_mismatch:.3%}, so the "
            f"weight terms leave a residual {-vertical_mismatch:.6g} N*m that is "
            "attributed to the diagonal entries by this linearisation"
        )

    diagonal_adequate = not notes

    return FullRestoringProperties(
        displaced_volume=volume,
        center_of_buoyancy=center_of_buoyancy,
        waterplane_area=wp_area,
        waterplane_center=wp_center,
        waterplane_moments=wp_moments,
        restoring_matrix=full,
        diagonal_matrix=diagonal,
        coupling_terms=coupling,
        waterplane_moments_about_origin=moments_about_origin,
        equilibrium_relative_mismatch=equilibrium_relative_mismatch,
        diagonal_adequate=diagonal_adequate,
        notes=tuple(notes),
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
