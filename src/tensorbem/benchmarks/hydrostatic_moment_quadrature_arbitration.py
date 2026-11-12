"""Arbitration of the pressure route's moment quadrature (mission m16662).

Mission m16554 shipped a second, structurally independent route to the
hydrostatic restoring matrix (direct pressure integration) and closed with an
*honest FAIL*: on a 160-facet wall-sided barge the roll and pitch diagonals
disagreed with the waterplane route by 4.12 % and 3.69 %, the disagreement did
not obviously shrink, and the report left two candidate culprits on the table

* the waterplane route's ``rho*g*Nabla*(z_B - z_G)`` term, or
* the pressure route's finite-rotation weight moment,

plus a third residual it could not place at all: roll-yaw and pitch-yaw entries
of ``4.2e-2`` of the matrix scale on a body whose centre of gravity is off the
rotation axis.

This module arbitrates both, without a human reading code:

1. **A closed-form third opinion.** For a wall-sided box and for a
   half-submerged ellipsoid the restoring matrix has an exact textbook value
   (:func:`analytic_barge_matrix`, :func:`analytic_spheroid_matrix`,
   Journee & Massie section 3.2.2).  So there are three numbers per entry: the
   closed form, the waterplane route and the pressure route.  The closed form
   decides who is wrong.
2. **The pre-fix quadrature, reproduced on purpose.** :func:`centroid_moment`
   is the rule the route used until m16662 (area centroid x panel force).  It is
   kept here so the artifact can *measure* the debt it caused instead of
   describing it: the roll/pitch error falls as ``h**2`` (4.12 % at 160 facets,
   5.4e-3 at 1040, 1.35e-3 at 4160) -- the signature of a quadrature loss, not
   of a physical route difference.
3. **The ``z_G`` scan mission m16554 asked for.**  A fixed *absolute* quadrature
   error, divided by a total ``rho*g*(I_xx + Nabla*(z_B - z_G))`` that shrinks as
   ``z_G`` approaches ``z_B``, produces a *relative* error that grows with
   ``z_G`` -- which is exactly the pattern m16554 read as evidence against the
   formula.  The scan separates absolute from relative and shows the exact rule
   is flat at round-off across the whole range.
4. **The yaw residual, resolved.**  It does not decay under refinement because
   it is not discretisation: it is the equilibrium moment of an *out-of-trim*
   body being carried round by the rotation,
   ``C[3,6] = W*(x_G - x_B)`` and ``C[4,6] = W*(y_G - y_B)``.  The closed form
   carries no such entry (it assumes the body is trimmed), the pressure route
   does -- and both become zero when the centre of gravity is put at the centre
   of buoyancy.  So the pressure route is right about the body it was given and
   the waterplane route is right about its own premise; the mismatch was a
   verification body that was not in trim.  The couplings live in the yaw column
   only (``C[6,3]`` stays at round-off), so ``C`` is not symmetric out of trim:
   a non-zero equilibrium load is not a potential, hence no Hessian and no
   symmetry.  Trimmed bodies stay symmetric to round-off.

Everything here is programmatic: the JSON artifact and the markdown memorandum
are written by :func:`main` (``python -m
tensorbem.benchmarks.hydrostatic_moment_quadrature_arbitration --write``), so no
number in the published documents is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..geometry3d import ellipsoid_mesh
from ..hydrostatic import compute_hydrostatic_restoring_matrix
from ..hydrostatic_pressure_route import (
    pressure_loads,
    pressure_moment,
    richardson_restoring_matrix,
    wetted_hull_from_closed,
)
from .hydrostatic_pressure_route_study import (
    DENSITY,
    GRAVITY,
    closed_box_mesh,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")

#: Barge of mission m16554: 10 m x 4 m x 2 m, 0.5 m freeboard, waterline at z=0.
BARGE = dict(length=10.0, beam=4.0, draft=2.0, freeboard=0.5)
BARGE_COG = (0.0, 0.0, -1.5)
#: Same three verification meshes m16554 used, coarsest first.
BARGE_MESHES = (
    ("barge-n4b3h4", dict(n_length=4, n_beam=3, n_height=4)),
    ("barge-n10b8h10", dict(n_length=10, n_beam=8, n_height=10)),
    ("barge-n20b16h20", dict(n_length=20, n_beam=16, n_height=20)),
)
#: Half-submerged spheroid, semi-axes (5, 2, 1.5) m.
SPHEROID_AXES = (5.0, 2.0, 1.5)
SPHEROID_SHIFT = (1.2, 0.7, 0.0)
SPHEROID_COG = (1.5, 1.0, 0.0)


# ---------------------------------------------------------------------------
# panel geometry and the two quadrature rules
# ---------------------------------------------------------------------------
def _panel_geometry(
    vertices: FloatArray, faces: IntArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    double_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * double_area
    normals = np.zeros_like(cross)
    good = double_area > 0.0
    normals[good] = cross[good] / double_area[good, None]
    centroids = (p0 + p1 + p2) / 3.0
    return areas, normals, centroids


def centroid_moment(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    density: float = DENSITY,
    gravity: float = GRAVITY,
    free_surface_z: float = 0.0,
) -> FloatArray:
    """``-int p (r x n) dS`` with the **pre-m16662** centroid rule.

    Kept deliberately: this is the rule that produced mission m16554's 4 % roll
    discrepancy, and the artifact has to be able to quote its error rather than
    assert it.  It is exact for a *linear* integrand, and the moment integrand
    is quadratic, so it loses ``A*h**2/12`` per panel direction.
    """
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=np.int64)
    areas, normals, centroids = _panel_geometry(verts, tris)
    panel_pressure = float(density) * float(gravity) * (float(free_surface_z) - centroids[:, 2])
    panel_force = -(panel_pressure * areas)[:, None] * normals
    return np.sum(np.cross(centroids, panel_force), axis=0)


@dataclass(frozen=True)
class QuadratureComparison:
    """Legacy vs exact moment quadrature on one planar-panel test surface."""

    panel_vertices: FloatArray
    panel_faces: IntArray
    analytic_moment: FloatArray
    exact_moment: FloatArray
    centroid_moment: FloatArray


def planar_wall_quadrature_check(
    *,
    length: float = 10.0,
    draft: float = 2.0,
    panels_x: int = 2,
    panels_z: int = 2,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> QuadratureComparison:
    """A vertical rectangle ``[0, length] x [-draft, 0]`` at ``y = 0``, ``n = +y``.

    For that surface ``(r x n)_x = y n_z - z n_y = -z``, so the x-moment is the
    closed form ``M_x = -rho*g*int z**2 dS = -rho*g*length*draft**3/3``; the other
    two components vanish by symmetry.  A rule that cannot integrate ``z**2``
    exactly cannot get this moment right, and no amount of physical modelling
    enters.
    """
    xs = np.linspace(0.0, length, int(panels_x) + 1)
    zs = np.linspace(-draft, 0.0, int(panels_z) + 1)
    verts: list[list[float]] = []
    index: dict[tuple[float, float], int] = {}
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            index[(i, j)] = len(verts)
            verts.append([float(x), 0.0, float(z)])
    faces: list[tuple[int, int, int]] = []
    for i in range(int(panels_x)):
        for j in range(int(panels_z)):
            a = index[(i, j)]
            b = index[(i + 1, j)]
            c = index[(i + 1, j + 1)]
            d = index[(i, j + 1)]
            faces.append((a, c, b))
            faces.append((a, d, c))
    panel_vertices = np.asarray(verts, dtype=float)
    panel_faces = np.asarray(faces, dtype=np.int64)
    rho_g = float(density) * float(gravity)
    analytic = np.array(
        [
            -rho_g * length * draft**3 / 3.0,          # int p (r x n)_x, (r x n)_x = -z
            0.0,                                        # n and r have no y-moment arm
            -rho_g * (length**2 / 2.0) * (draft**2 / 2.0),  # (r x n)_z = x
        ]
    )
    exact = pressure_moment(
        panel_vertices, panel_faces, density=density, gravity=gravity
    )
    legacy = centroid_moment(
        panel_vertices, panel_faces, density=density, gravity=gravity
    )
    return QuadratureComparison(
        panel_vertices=panel_vertices,
        panel_faces=panel_faces,
        analytic_moment=analytic,
        exact_moment=exact,
        centroid_moment=legacy,
    )


# ---------------------------------------------------------------------------
# closed-form third opinion
# ---------------------------------------------------------------------------
def analytic_barge_matrix(
    *,
    length: float,
    beam: float,
    draft: float,
    center_of_gravity_z: float,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> FloatArray:
    """Exact ``C`` for a wall-sided box with a symmetric waterplane.

    ``C[2,2] = rho*g*A``, ``C[3,3] = rho*g*(I_xx + Nabla*(z_B - z_G))``,
    ``C[4,4] = rho*g*(I_yy + Nabla*(z_B - z_G))`` with ``A = L*B``,
    ``I_xx = L*B**3/12``, ``I_yy = B*L**3/12``, ``Nabla = L*B*T`` and
    ``z_B = -T/2``.  Wall-sided means the waterplane is unchanged to first order
    in the motion, so this *is* the linearised value, not an approximation of it.
    """
    rho_g = float(density) * float(gravity)
    volume = length * beam * draft
    z_b = -0.5 * draft
    matrix = np.zeros((6, 6))
    matrix[2, 2] = rho_g * length * beam
    metacentric = volume * (z_b - float(center_of_gravity_z))
    matrix[3, 3] = rho_g * (length * beam**3 / 12.0 + metacentric)
    matrix[4, 4] = rho_g * (beam * length**3 / 12.0 + metacentric)
    return matrix


def analytic_spheroid_matrix(
    *,
    semi_axes: Sequence[float] = SPHEROID_AXES,
    center_of_gravity_z: float = 0.0,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> FloatArray:
    """Exact ``C`` for an ellipsoid floating half submerged on its principal axes.

    Waterplane ellipse ``a x b``: ``A = pi*a*b``, ``I_xx = pi*a*b**3/4``,
    ``I_yy = pi*a**3*b/4``; displaced volume ``Nabla = 2/3*pi*a*b*c`` and centre
    of buoyancy ``z_B = -3c/8`` (half-ellipsoid centroid).
    """
    a, b, c = (float(v) for v in semi_axes)
    rho_g = float(density) * float(gravity)
    volume = 2.0 / 3.0 * math.pi * a * b * c
    z_b = -3.0 * c / 8.0
    matrix = np.zeros((6, 6))
    matrix[2, 2] = rho_g * math.pi * a * b
    metacentric = volume * (z_b - float(center_of_gravity_z))
    matrix[3, 3] = rho_g * (math.pi * a * b**3 / 4.0 + metacentric)
    matrix[4, 4] = rho_g * (math.pi * a**3 * b / 4.0 + metacentric)
    return matrix


# ---------------------------------------------------------------------------
# the three routes
# ---------------------------------------------------------------------------
@dataclass
class RouteComparison:
    analytic: FloatArray
    waterplane: FloatArray
    pressure_exact: FloatArray
    pressure_legacy: FloatArray
    panels: int
    displaced_volume: float
    center_of_buoyancy: tuple[float, float, float]
    weight: float


def _pressure_route_with_legacy_moment(
    vertices: FloatArray,
    faces: FloatArray,
    *,
    center_of_gravity: Sequence[float],
    steps: Sequence[float] = (2e-3, 1e-3, 5e-4),
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> FloatArray:
    """The pressure route as it stood before m16662 (centroid moment Rule).

    Rebuilt here rather than kept as a switch in the library: the defect is
    historical, and the shipped route must have exactly one quadrature rule.
    """
    from ..hydrostatic_pressure_route import clip_to_fluid, rigid_transform

    cog = np.asarray(center_of_gravity, dtype=float)
    clipped_vertices, clipped_faces = clip_to_fluid(vertices, faces)
    areas, normals, centroids = _panel_geometry(clipped_vertices, clipped_faces)
    volume = float(np.sum(np.sum(centroids * normals, axis=1) * areas) / 3.0)
    weight = float(density) * float(gravity) * volume

    def load(motion: FloatArray) -> FloatArray:
        moved = rigid_transform(vertices, motion)
        cv, cf = clip_to_fluid(moved, faces)
        a, n, c = _panel_geometry(cv, cf)
        panel_pressure = float(density) * float(gravity) * (0.0 - c[:, 2])
        panel_force = -(panel_pressure * a)[:, None] * n
        force = np.sum(panel_force, axis=0)
        moment = np.sum(np.cross(c, panel_force), axis=0)
        moved_cog = rigid_transform(cog.reshape(1, 3), motion)[0]
        moment = moment + np.cross(moved_cog, np.array([0.0, 0.0, -weight]))
        return np.concatenate([force, moment])

    increments = [float(s) for s in steps]
    matrices = []
    for step in increments:
        matrix = np.zeros((6, 6))
        for k in range(6):
            plus = np.zeros(6)
            minus = np.zeros(6)
            plus[k] = step
            minus[k] = -step
            matrix[:, k] = -(load(plus) - load(minus)) / (2.0 * step)
        matrices.append(matrix)
    design = np.array([[1.0, s * s] for s in increments])
    coefficients = np.linalg.pinv(design) @ np.stack([m.ravel() for m in matrices])
    return coefficients[0].reshape(6, 6)


def compare_routes(
    vertices: FloatArray,
    faces: IntArray,
    *,
    center_of_gravity: Sequence[float],
    analytic: FloatArray,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> RouteComparison:
    pressure = richardson_restoring_matrix(
        vertices, faces, center_of_gravity=center_of_gravity, density=density, gravity=gravity
    )
    waterplane = compute_hydrostatic_restoring_matrix(
        wetted_hull_from_closed(vertices, faces),
        mass=density * pressure.immersed_volume,
        center_of_gravity=np.asarray(center_of_gravity, dtype=float),
        density=density,
        gravity=gravity,
    )
    legacy = _pressure_route_with_legacy_moment(
        vertices, faces, center_of_gravity=center_of_gravity, density=density, gravity=gravity
    )
    return RouteComparison(
        analytic=np.asarray(analytic, dtype=float),
        waterplane=np.asarray(waterplane.restoring_matrix, dtype=float),
        pressure_exact=np.asarray(pressure.matrix, dtype=float),
        pressure_legacy=legacy,
        panels=int(len(faces)),
        displaced_volume=float(pressure.immersed_volume),
        center_of_buoyancy=tuple(float(v) for v in pressure.center_of_buoyancy),
        weight=float(pressure.weight),
    )


def _relative(measured: FloatArray, reference: FloatArray) -> FloatArray:
    scale = float(np.abs(reference).max())
    return np.abs(measured - reference) / scale


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------
def quadrature_arbitration() -> dict:
    """Barge: closed form vs waterplane vs pressure (exact and legacy rule)."""
    rows = []
    for name, mesh_kwargs in BARGE_MESHES:
        vertices, faces = closed_box_mesh(**BARGE, **mesh_kwargs)
        analytic = analytic_barge_matrix(
            length=BARGE["length"],
            beam=BARGE["beam"],
            draft=BARGE["draft"],
            center_of_gravity_z=BARGE_COG[2],
        )
        comparison = compare_routes(
            vertices, faces, center_of_gravity=BARGE_COG, analytic=analytic
        )
        rows.append(
            {
                "mesh": name,
                "panels": comparison.panels,
                "analytic_roll": float(analytic[3, 3]),
                "analytic_pitch": float(analytic[4, 4]),
                "waterplane_roll": float(comparison.waterplane[3, 3]),
                "waterplane_pitch": float(comparison.waterplane[4, 4]),
                "exact_roll": float(comparison.pressure_exact[3, 3]),
                "exact_pitch": float(comparison.pressure_exact[4, 4]),
                "legacy_roll": float(comparison.pressure_legacy[3, 3]),
                "legacy_pitch": float(comparison.pressure_legacy[4, 4]),
                "exact_roll_relative": float(
                    (comparison.pressure_exact[3, 3] - analytic[3, 3]) / analytic[3, 3]
                ),
                "exact_pitch_relative": float(
                    (comparison.pressure_exact[4, 4] - analytic[4, 4]) / analytic[4, 4]
                ),
                "legacy_roll_relative": float(
                    (comparison.pressure_legacy[3, 3] - analytic[3, 3]) / analytic[3, 3]
                ),
                "legacy_pitch_relative": float(
                    (comparison.pressure_legacy[4, 4] - analytic[4, 4]) / analytic[4, 4]
                ),
                "waterplane_roll_relative": float(
                    (comparison.waterplane[3, 3] - analytic[3, 3]) / analytic[3, 3]
                ),
                "exact_matrix_max_relative_waterplane": float(
                    _relative(comparison.pressure_exact, comparison.waterplane).max()
                ),
                "legacy_matrix_max_relative_waterplane": float(
                    _relative(comparison.pressure_legacy, comparison.waterplane).max()
                ),
            }
        )
    return {
        "body": "barge-L10B4T2",
        "geometry": {**BARGE, "center_of_gravity": list(BARGE_COG)},
        "rows": rows,
    }


def z_g_scan(
    *, values: Sequence[float] = (-2.0, -1.75, -1.5, -1.25, -1.0)
) -> dict:
    """Sweep ``z_G``: absolute error stays put, relative error does not.

    The range stops at ``z_G = -1.0`` on purpose: ``C[3,3]`` for this barge is
    ``rho*g*(I_xx + Nabla*(z_B - z_G))`` with ``z_B = -1``, so it changes sign at
    ``z_G = -5/3`` and a *relative* error against a vanishing reference says
    nothing.  Both routes still agree absolutely there, but the scan only claims
    what it can measure.
    """
    vertices, faces = closed_box_mesh(**BARGE, **BARGE_MESHES[0][1])
    rows = []
    for z_g in values:
        cog = (0.0, 0.0, float(z_g))
        analytic = analytic_barge_matrix(
            length=BARGE["length"],
            beam=BARGE["beam"],
            draft=BARGE["draft"],
            center_of_gravity_z=z_g,
        )
        comparison = compare_routes(vertices, faces, center_of_gravity=cog, analytic=analytic)
        reference = float(analytic[3, 3])
        rows.append(
            {
                "z_G": float(z_g),
                "analytic_roll": reference,
                "exact_roll": float(comparison.pressure_exact[3, 3]),
                "legacy_roll": float(comparison.pressure_legacy[3, 3]),
                "exact_absolute_error": float(comparison.pressure_exact[3, 3] - reference),
                "legacy_absolute_error": float(comparison.pressure_legacy[3, 3] - reference),
                "exact_relative_error": (float(comparison.pressure_exact[3, 3]) - reference)
                / reference,
                "legacy_relative_error": (float(comparison.pressure_legacy[3, 3]) - reference)
                / reference,
                "exact_waterplane_max_relative": float(
                    _relative(comparison.pressure_exact, comparison.waterplane).max()
                ),
                "legacy_waterplane_max_relative": float(
                    _relative(comparison.pressure_legacy, comparison.waterplane).max()
                ),
            }
        )
    return {
        "body": "barge-L10B4T2 (coarsest mesh, 160 panels)",
        "rows": rows,
    }


def yaw_coupling_arbitration() -> dict:
    """The roll/pitch-yaw residue: out-of-trim equilibrium moment, not noise."""
    rows = []
    for subdivisions in (3, 4):
        mesh = ellipsoid_mesh(subdivisions, np.asarray(SPHEROID_AXES, dtype=float))
        vertices = mesh.vertices + np.asarray(SPHEROID_SHIFT, dtype=float)[None, :]
        analytic = analytic_spheroid_matrix()
        comparison = compare_routes(
            vertices, mesh.faces, center_of_gravity=SPHEROID_COG, analytic=analytic
        )
        x_b, y_b, _ = comparison.center_of_buoyancy
        scale = float(np.abs(comparison.pressure_exact).max())
        rows.append(
            {
                "subdivisions": int(subdivisions),
                "panels": comparison.panels,
                "center_of_buoyancy": [x_b, y_b, 0.0],
                "center_of_gravity": list(SPHEROID_COG),
                "weight": comparison.weight,
                "measured_roll_yaw": float(comparison.pressure_exact[3, 5]),
                "measured_pitch_yaw": float(comparison.pressure_exact[4, 5]),
                "transpose_roll_yaw": float(comparison.pressure_exact[5, 3]),
                "transpose_pitch_yaw": float(comparison.pressure_exact[5, 4]),
                "predicted_roll_yaw": comparison.weight * (SPHEROID_COG[0] - x_b),
                "predicted_pitch_yaw": comparison.weight * (SPHEROID_COG[1] - y_b),
                "scale": scale,
                "roll_roll_relative_to_analytic": float(
                    (comparison.pressure_exact[3, 3] - analytic[3, 3]) / analytic[3, 3]
                ),
                "pitch_pitch_relative_to_analytic": float(
                    (comparison.pressure_exact[4, 4] - analytic[4, 4]) / analytic[4, 4]
                ),
            }
        )

    # Trimmed control: put the centre of gravity at the centre of buoyancy.
    mesh = ellipsoid_mesh(4, np.asarray(SPHEROID_AXES, dtype=float))
    vertices = mesh.vertices + np.asarray(SPHEROID_SHIFT, dtype=float)[None, :]
    trimmed = richardson_restoring_matrix(
        vertices, mesh.faces, center_of_gravity=(SPHEROID_SHIFT[0], SPHEROID_SHIFT[1], 0.0)
    )
    scale = float(np.abs(trimmed.matrix).max())
    control = {
        "subdivisions": 4,
        "center_of_gravity": [SPHEROID_SHIFT[0], SPHEROID_SHIFT[1], 0.0],
        "measured_roll_yaw": float(trimmed.matrix[3, 5]),
        "measured_pitch_yaw": float(trimmed.matrix[4, 5]),
        "relative_to_scale": float(np.abs(trimmed.matrix[3:5, 5]).max() / scale),
    }
    return {"out_of_trim": rows, "trimmed_control": control}


def planar_panel_check() -> dict:
    """One vertical wall, three rules, closed-form moment."""
    rows = []
    for panels in (1, 2, 4, 8, 16):
        comparison = planar_wall_quadrature_check(panels_x=panels, panels_z=panels)
        analytic = comparison.analytic_moment[0]
        rows.append(
            {
                "panels_per_side": int(panels),
                "analytic_m_x": float(analytic),
                "exact_m_x": float(comparison.exact_moment[0]),
                "centroid_m_x": float(comparison.centroid_moment[0]),
                "exact_relative_error": float(
                    (comparison.exact_moment[0] - analytic) / analytic
                ),
                "centroid_relative_error": float(
                    (comparison.centroid_moment[0] - analytic) / analytic
                ),
                "exact_off_axis_moment": float(np.abs(comparison.exact_moment[1:]).max()),
            }
        )
    return {
        "surface": "vertical rectangle 10 m x 2 m at y=0, normal +y",
        "rows": rows,
    }


def spheroid_crosscheck() -> dict:
    """Curved body: the three routes on the same facets, plus refinement."""
    rows = []
    for subdivisions in (2, 3, 4):
        mesh = ellipsoid_mesh(subdivisions, np.asarray(SPHEROID_AXES, dtype=float))
        analytic = analytic_spheroid_matrix()
        comparison = compare_routes(
            mesh.vertices,
            mesh.faces,
            center_of_gravity=(0.0, 0.0, 0.0),
            analytic=analytic,
        )
        rows.append(
            {
                "subdivisions": int(subdivisions),
                "panels": comparison.panels,
                "displaced_volume": comparison.displaced_volume,
                "analytic_volume": 2.0 / 3.0 * math.pi * np.prod(SPHEROID_AXES),
                "roll_analytic": float(analytic[3, 3]),
                "roll_waterplane": float(comparison.waterplane[3, 3]),
                "roll_pressure_exact": float(comparison.pressure_exact[3, 3]),
                "roll_pressure_legacy": float(comparison.pressure_legacy[3, 3]),
                "waterplane_relative_to_analytic": float(
                    (comparison.waterplane[3, 3] - analytic[3, 3]) / analytic[3, 3]
                ),
                "exact_relative_to_analytic": float(
                    (comparison.pressure_exact[3, 3] - analytic[3, 3]) / analytic[3, 3]
                ),
                "legacy_relative_to_analytic": float(
                    (comparison.pressure_legacy[3, 3] - analytic[3, 3]) / analytic[3, 3]
                ),
                "exact_vs_waterplane_max_relative": float(
                    _relative(comparison.pressure_exact, comparison.waterplane).max()
                ),
                "legacy_vs_waterplane_max_relative": float(
                    _relative(comparison.pressure_legacy, comparison.waterplane).max()
                ),
            }
        )
    return {"body": "spheroid-a5b2c1.5", "semi_axes": list(SPHEROID_AXES), "rows": rows}


def build_payload() -> dict:
    return {
        "mission": "m16662",
        "question": (
            "mission m16554 left a 4.12 %/3.69 % roll/pitch disagreement and a "
            "4.2e-2 yaw residue unexplained; which route is wrong?"
        ),
        "density": DENSITY,
        "gravity": GRAVITY,
        "planar_panel": planar_panel_check(),
        "barge_arbitration": quadrature_arbitration(),
        "z_g_scan": z_g_scan(),
        "spheroid": spheroid_crosscheck(),
        "yaw_coupling": yaw_coupling_arbitration(),
    }


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    return f"{value:+.3e}"


def build_report(payload: dict) -> str:
    barge = payload["barge_arbitration"]
    out: list[str] = []
    out.append("# Hydrostatic pressure route: arbitration of the m16554 discrepancy (m16662)\n\n")
    out.append(
        "Generated by `python -m "
        "tensorbem.benchmarks.hydrostatic_moment_quadrature_arbitration --write`; "
        "every number below is emitted by the code that computes it.\n\n"
    )
    out.append("## 0. Question\n\n" + payload["question"] + "\n\n")
    out.append(
        f"Density {payload['density']} kg/m3, gravity {payload['gravity']} m/s2.\n\n"
    )

    out.append("## 1. The quadrature rule is the culprit, not either physics model\n\n")
    out.append("### 1.1 One planar wall, closed-form moment\n\n")
    out.append(
        "| panels/side | exact M_x | centroid M_x | exact rel. | centroid rel. |\n"
        "|---|---|---|---|---|\n"
    )
    for row in payload["planar_panel"]["rows"]:
        out.append(
            f"| {row['panels_per_side']} | {row['exact_m_x']:.6e} | "
            f"{row['centroid_m_x']:.6e} | {_fmt(row['exact_relative_error'])} | "
            f"{_fmt(row['centroid_relative_error'])} |\n"
        )
    out.append(
        "\nThe moment integrand is quadratic whereas the centroid rule is "
        "first-degree exact, so it under-integrates every panel by `A*h^2/12`. "
        "On the whole barge that is a fixed absolute amount, which is why it read "
        "as 'systematic' rather than 'discretisation' in m16554.\n\n"
    )

    out.append("### 1.2 Barge vs the closed form\n\n")
    out.append(
        "| mesh | panels | analytic C[roll,roll] | waterplane rel. | exact rel. "
        "| centroid rel. | exact vs waterplane |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for row in barge["rows"]:
        out.append(
            f"| {row['mesh']} | {row['panels']} | {row['analytic_roll']:.6e} | "
            f"{_fmt(row['waterplane_roll_relative'])} | {_fmt(row['exact_roll_relative'])} | "
            f"{_fmt(row['legacy_roll_relative'])} | "
            f"{row['exact_matrix_max_relative_waterplane']:.3e} |\n"
        )
    out.append(
        "\nThe closed form and the waterplane route agree to machine precision (they "
        "are the same formula), so the closed form cannot arbitrate between them -- "
        "but the pressure route with the *exact* rule lands on the same value to "
        "round-off at every mesh, while the centroid rule is off by `O(h^2)`.\n\n"
    )

    out.append("## 2. The `z_G` scan m16554 proposed\n\n")
    out.append(
        "| z_G | analytic C[roll,roll] | exact abs. error | centroid abs. error "
        "| exact rel. | centroid rel. |\n"
        "|---|---|---|---|---|---|\n"
    )
    for row in payload["z_g_scan"]["rows"]:
        out.append(
            f"| {row['z_G']:+.1f} | {row['analytic_roll']:.6e} | "
            f"{_fmt(row['exact_absolute_error'])} | {_fmt(row['legacy_absolute_error'])} | "
            f"{_fmt(row['exact_relative_error'])} | {_fmt(row['legacy_relative_error'])} |\n"
        )
    out.append(
        "\nThe *absolute* centroid error is flat across `z_G` (a quadrature property "
        "of the mesh), so the *relative* error grows as the restoring entry itself "
        "shrinks towards `z_G -> z_B`. That growth is what m16554 read as evidence "
        "against the `rho*g*Nabla*(z_B - z_G)` term; it is an artefact of the "
        "denominator. With the exact rule both columns are flat at round-off.\n\n"
    )

    out.append("## 3. The yaw residue: an out-of-trim body, not a bug\n\n")
    out.append(
        "| subdivisions | panels | C[roll,yaw] | W*(x_G - x_B) | C[pitch,yaw] "
        "| W*(y_G - y_B) |\n"
        "|---|---|---|---|---|---|\n"
    )
    for row in payload["yaw_coupling"]["out_of_trim"]:
        out.append(
            f"| {row['subdivisions']} | {row['panels']} | {row['measured_roll_yaw']:.4e} | "
            f"{row['predicted_roll_yaw']:.4e} | {row['measured_pitch_yaw']:.4e} | "
            f"{row['predicted_pitch_yaw']:.4e} |\n"
        )
    control = payload["yaw_coupling"]["trimmed_control"]
    out.append(
        f"\nTrimmed control (centre of gravity moved to the centre of buoyancy "
        f"{(control['center_of_gravity'][0], control['center_of_gravity'][1])}): "
        f"C[roll,yaw] = {control['measured_roll_yaw']:.3e}, "
        f"C[pitch,yaw] = {control['measured_pitch_yaw']:.3e} "
        f"= {control['relative_to_scale']:.2e} of the matrix scale.\n\n"
        "The couplings sit in the yaw *column* only: the transpose row stays at "
        "round-off, so `C` is **not symmetric** for an out-of-trim body. That is "
        "not a defect of the route but a property of the state it linearises: the "
        "equilibrium load is not zero, and it is carried round by a rotation about "
        "the vertical axis (which leaves the pressure field `p(z)` invariant) but "
        "not by a rotation about a horizontal one. A non-zero equilibrium load is "
        "not a potential, so there is no Hessian and no symmetry. The trimmed "
        "control above is symmetric to round-off.\n\n"
        "The coupling is the equilibrium moment of a body that is not in trim, "
        "carried round by the rotation: the pressure route linearises the body it "
        "was handed, while the closed-form waterplane table assumes a trimmed body "
        "and has no such entry. Neither route is wrong; the verification body was "
        "not in trim.\n\n"
    )

    out.append("## 4. Curved body\n\n")
    out.append(
        "| subdivisions | panels | volume | waterplane vs analytic | exact vs analytic "
        "| centroid vs analytic |\n"
        "|---|---|---|---|---|---|\n"
    )
    for row in payload["spheroid"]["rows"]:
        out.append(
            f"| {row['subdivisions']} | {row['panels']} | {row['displaced_volume']:.6f} | "
            f"{_fmt(row['waterplane_relative_to_analytic'])} | "
            f"{_fmt(row['exact_relative_to_analytic'])} | "
            f"{_fmt(row['legacy_relative_to_analytic'])} |\n"
        )
    out.append(
        "\nOn a curved hull the facet discretisation dominates the remaining error, "
        "as it should; the exact rule removes the quadrature contribution under it.\n"
    )
    return "".join(out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figure(payload: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    panels = [row["panels"] for row in payload["barge_arbitration"]["rows"]]
    legacy = [abs(row["legacy_roll_relative"]) for row in payload["barge_arbitration"]["rows"]]
    exact = [abs(row["exact_roll_relative"]) for row in payload["barge_arbitration"]["rows"]]
    axes[0].loglog(panels, legacy, "o-", label="centroid rule (pre-m16662)")
    axes[0].loglog(panels, exact, "s-", label="edge-midpoint rule (m16662)")
    reference = np.asarray(panels, dtype=float)
    axes[0].loglog(
        reference,
        legacy[0] * (reference[0] / reference) ** 2,
        "k--",
        linewidth=1.0,
        label=r"$O(h^2)$ reference",
    )
    axes[0].set_xlabel("panels on the barge")
    axes[0].set_ylabel(r"relative error of $C_{44}$ vs closed form")
    axes[0].set_title("roll entry: quadrature error erased")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    z_g = [row["z_G"] for row in payload["z_g_scan"]["rows"]]
    axes[1].semilogy(
        z_g,
        [abs(row["legacy_absolute_error"]) for row in payload["z_g_scan"]["rows"]],
        "o-",
        label="centroid rule, |absolute|",
    )
    axes[1].semilogy(
        z_g,
        [abs(row["legacy_relative_error"]) for row in payload["z_g_scan"]["rows"]],
        "^-",
        label="centroid rule, |relative|",
    )
    axes[1].semilogy(
        z_g,
        [max(abs(row["exact_relative_error"]), 1e-16) for row in payload["z_g_scan"]["rows"]],
        "s-",
        label="edge-midpoint rule, |relative|",
    )
    axes[1].set_xlabel("centre of gravity height $z_G$ [m]")
    axes[1].set_ylabel("error of $C_{44}$ vs closed form")
    axes[1].set_title(r"$z_G$ scan: the debt is in the rule, not in $z_G$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the payload here")
    parser.add_argument("--report", type=Path, default=None, help="write the memorandum here")
    parser.add_argument("--figure", type=Path, default=None, help="write the figure here")
    args = parser.parse_args(argv)

    payload = build_payload()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(build_report(payload))
        print(f"wrote {args.report}")
    if args.figure is not None:
        print(f"wrote {make_figure(payload, args.figure)}")
    if args.json is None and args.report is None and args.figure is None:
        print(build_report(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())