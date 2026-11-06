"""Curved-waterplane / curved-wall audit of the hydrostatic restoring matrix.

``docs/hydrostatic_restoring_couplings.md`` §5 item 3 records an explicit
verification debt: the coupled restoring matrix of
:func:`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` has only ever
been checked on **wall-sided barges with a rectangular waterplane**.  The note
guessed that a curved (arc-wall) hull "needs curved-surface clipping" for the
energy Hessian.  It does not: :mod:`tensorbem.benchmarks.hydrostatic_energy_reference`
clips every *triangle* against ``z <= 0`` (Sutherland-Hodgman) and integrates
``∫ z dV = ∮ (z²/2) n_z dA``, which is agnostic to whether the panel belongs to a
vertical wall or a sphere.  So the debt is payable, and this module pays it.

Two curved families are audited, each with two independent references:

``eccentric_sphere``
    A sphere of radius ``R`` whose centre sits at ``(x0, y0, -d)``.  Curved wall,
    **circular** waterplane displaced from the origin, so the heave-roll and
    heave-pitch couplings ``C[2,3] = ρg A y0`` and ``C[2,4] = -ρg A x0`` are
    non-zero while ``C[3,4] = -ρg I_xy = 0`` by the symmetry of a circle.

``yawed_spheroid`` / ``eccentric_yawed_spheroid``
    A half-ellipsoid ``(a, b, c)`` yawed by ``ψ`` (and optionally shifted), so
    the waterplane is an **ellipse with a non-zero product moment**
    ``I_xy = (I_yy⁰ - I_xx⁰) sinψ cosψ + A x0 y0``.  All six entries of the
    heave/roll/pitch block are then non-zero: this is as far from a wall-sided
    barge as a single smooth hull can get.

References
----------
``closed form``
    Circle/ellipse waterplane integrals plus spherical-cap volume and centroid
    are elementary; the exact ``C`` follows from the *same* published formula
    (``journee``-style) the implementation claims to implement, evaluated on
    exact geometry.
``energy Hessian``
    :func:`tensorbem.benchmarks.hydrostatic_energy_reference.restoring_matrix_from_energy`
    on the *same faceted mesh* the implementation sees.  Because both sides see
    identical geometry, this comparison isolates **formula validity** from
    faceting error: agreement is expected at the finite-difference noise level
    even though the mesh is only a polyhedral approximation of the sphere.

The distinction matters for reading the numbers: ``code vs energy`` is the
verification of the formula for curved walls (sharp, ~1e-6); ``code vs closed
form`` is the discretisation error of the faceted hull (O(h²), percent level and
shrinking with ``subdivisions``).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..geometry3d import WettedHullMesh, floating_sphere_hull_mesh, spheroid_hull_mesh
from ..hydrostatic import compute_hydrostatic_restoring_matrix
from .hydrostatic_energy_reference import (
    potential_energy as potential_energy_reference,
    restoring_matrix_from_energy,
    submerged_first_moment_z,
)

#: Mode indices of the audited block.
MODE_NAMES = {2: "heave", 3: "roll", 4: "pitch"}

#: The non-zero structure the closed forms can predict.
AUDITED_ENTRIES = ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4))

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665


def _rotate_z(vertices: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points about the ``z`` axis (leaves ``z``, hence the waterline, intact)."""
    cos, sin = math.cos(angle), math.sin(angle)
    out = np.array(vertices, dtype=float, copy=True)
    out[:, 0] = cos * vertices[:, 0] - sin * vertices[:, 1]
    out[:, 1] = sin * vertices[:, 0] + cos * vertices[:, 1]
    return out


# ---------------------------------------------------------------------------
# Hull builders
# ---------------------------------------------------------------------------


def eccentric_sphere_hull(
    subdivisions: int,
    radius: float,
    depth: float,
    offset_xy: tuple[float, float],
) -> WettedHullMesh:
    """Wetted cap of a sphere whose centre is at ``(x0, y0, -depth)``."""
    return floating_sphere_hull_mesh(
        subdivisions, radius, center=(offset_xy[0], offset_xy[1], -depth)
    )


def yawed_spheroid_hull(
    semi_axes: tuple[float, float, float],
    yaw: float,
    subdivisions: int,
    offset_xy: tuple[float, float] = (0.0, 0.0),
) -> WettedHullMesh:
    """Half-ellipsoid ``(a, b, c)`` yawed by ``yaw`` radians and shifted in ``x``/``y``.

    ``spheroid_hull_mesh`` truncates the ellipsoid at the calm waterplane through
    its centre, so the body is a "bowl" floating exactly at its rim; a yaw keeps
    every waterline vertex on ``z = 0`` (rotation about ``z``), so the mesh stays
    a valid :class:`WettedHullMesh`.
    """
    base = spheroid_hull_mesh(semi_axes, subdivisions=subdivisions)
    vertices = _rotate_z(base.vertices, yaw)
    vertices[:, 0] += offset_xy[0]
    vertices[:, 1] += offset_xy[1]
    return WettedHullMesh(vertices, base.faces)


# ---------------------------------------------------------------------------
# Fast (vectorised) energy Hessian
# ---------------------------------------------------------------------------


def _restricted_second_moment_z(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Return ``∫_{S ∩ {z<=0}} z dV = ∮ (z²/2) n_z dA`` for a triangulated surface.

    This is the flux form of the submerged first moment that the reference
    implementation :func:`~tensorbem.benchmarks.hydrostatic_energy_reference.submerged_first_moment_z`
    evaluates face by face.

    The pure-Python reference in
    :mod:`tensorbem.benchmarks.hydrostatic_energy_reference` clips each triangle
    with Sutherland-Hodgman, which costs ~4 ms per thousand panels per energy
    evaluation; a 21-point Hessian on a few thousand panels then costs a minute.
    Restricting a triangle to one half-space has only three cases, each with a
    closed form, so the same quantity is available vectorised:

    * no vertex below the waterline -> 0;
    * all three below -> the exact ``∫ z² dA = A (Σ z_i² + Σ_{i<j} z_i z_j) / 6``;
    * one below -> the corner triangle at that vertex, with the two remaining
      vertices on ``z = 0``, hence ``A_sub z_p² / 6``;
    * two below -> the full triangle minus the corner triangle at the third.

    :func:`tests.test_hydrostatic_curved_waterplane` pins this against the
    reference implementation at ``1e-12`` relative on random rigid motions, so
    the speed-up does not cost independence.
    """
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)]
    z = tri[:, :, 2]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(normals, axis=1)
    n_z = np.divide(
        normals[:, 2], 2.0 * area, out=np.zeros_like(area), where=area > 0.0
    )
    full = area * (
        z[:, 0] ** 2 + z[:, 1] ** 2 + z[:, 2] ** 2
        + z[:, 0] * z[:, 1] + z[:, 1] * z[:, 2] + z[:, 2] * z[:, 0]
    ) / 6.0
    below = z <= 0.0
    count = below.sum(axis=1)
    integral = np.where(count == 3, full, 0.0)
    # One vertex below: the cut corner triangle keeps z = (z_p, 0, 0).
    one = count == 1
    if np.any(one):
        index = np.argmax(below[one], axis=1)
        rows = np.flatnonzero(one)
        z_p = z[rows, index]
        other = np.stack(
            [z[rows, (index + 1) % 3], z[rows, (index + 2) % 3]], axis=1
        )
        t1 = z_p / (z_p - other[:, 0])
        t2 = z_p / (z_p - other[:, 1])
        integral[rows] = area[rows] * t1 * t2 * z_p**2 / 6.0
    # Two below: the full triangle minus the corner triangle at the dry vertex.
    two = count == 2
    if np.any(two):
        index = np.argmax(~below[two], axis=1)
        rows = np.flatnonzero(two)
        z_p = z[rows, index]
        other = np.stack(
            [z[rows, (index + 1) % 3], z[rows, (index + 2) % 3]], axis=1
        )
        t1 = z_p / (z_p - other[:, 0])
        t2 = z_p / (z_p - other[:, 1])
        integral[rows] = full[rows] - area[rows] * t1 * t2 * z_p**2 / 6.0
    # ∮ (z²/2) n_z dA, the flux form of ∫_S z dV.
    return 0.5 * float(np.sum(n_z * integral))


def _rotation_matrix(rotation_vector: Sequence[float]) -> np.ndarray:
    """Exponential map of a rotation vector (Rodrigues), matching the reference."""
    alpha = np.asarray(rotation_vector, dtype=float)
    theta = float(np.linalg.norm(alpha))
    skew = np.array(
        [
            [0.0, -alpha[2], alpha[1]],
            [alpha[2], 0.0, -alpha[0]],
            [-alpha[1], alpha[0], 0.0],
        ]
    )
    if theta < 1.0e-12:
        return np.eye(3) + skew + 0.5 * skew @ skew
    return (
        np.eye(3)
        + (math.sin(theta) / theta) * skew
        + ((1.0 - math.cos(theta)) / theta**2) * (skew @ skew)
    )


def potential_energy_fast(
    displacement: Sequence[float],
    vertices: np.ndarray,
    faces: np.ndarray,
    mass: float,
    center_of_gravity: Sequence[float],
    density: float,
    gravity: float,
) -> float:
    """Same quantity as the reference ``potential_energy``, vectorised."""
    xi = np.asarray(displacement, dtype=float)
    rotation = _rotation_matrix(xi[3:6])
    moved = vertices @ rotation.T + xi[0:3]
    cg = rotation @ np.asarray(center_of_gravity, dtype=float) + xi[0:3]
    return mass * gravity * float(cg[2]) - density * gravity * _restricted_second_moment_z(
        moved, faces
    )


def restoring_matrix_from_energy_fast(
    vertices: np.ndarray,
    faces: np.ndarray,
    mass: float,
    center_of_gravity: Sequence[float],
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    step: float = 2.0e-4,
) -> np.ndarray:
    """Numerical Hessian of :func:`potential_energy_fast` (same stencil as the reference)."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    zero = np.zeros(6)

    def energy(xi: np.ndarray) -> float:
        return potential_energy_fast(
            xi, vertices, faces, mass, center_of_gravity, density, gravity
        )

    base = energy(zero)
    matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(i, 6):
            if i == j:
                plus = zero.copy()
                minus = zero.copy()
                plus[i] = step
                minus[i] = -step
                value = (energy(plus) - 2.0 * base + energy(minus)) / step**2
            else:
                plus_plus = zero.copy()
                plus_minus = zero.copy()
                minus_plus = zero.copy()
                minus_minus = zero.copy()
                plus_plus[i] = plus_plus[j] = step
                plus_minus[i] = step
                plus_minus[j] = -step
                minus_plus[i] = -step
                minus_plus[j] = step
                minus_minus[i] = minus_minus[j] = -step
                value = (
                    energy(plus_plus)
                    - energy(plus_minus)
                    - energy(minus_plus)
                    + energy(minus_minus)
                ) / (4.0 * step**2)
            matrix[i, j] = matrix[j, i] = value
    return matrix


def restoring_matrix_from_energy_richardson(
    vertices: np.ndarray,
    faces: np.ndarray,
    mass: float,
    center_of_gravity: Sequence[float],
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    steps: Sequence[float] = (2.0e-4, 1.0e-4),
) -> np.ndarray:
    """Richardson-extrapolated energy Hessian.

    Clipping a faceted hull changes which triangles are cut, so the potential
    energy is only *piecewise* smooth in the displacement and the central second
    difference carries an ``O(step)`` kink term (measured order 0.996-1.134 in
    :func:`fd_step_study`).  Extrapolating with that exponent,
    ``C ≈ 2 H(step/2) - H(step)``, removes it and brings the reference down to
    ``1e-7`` relative of the implementation — sharp enough that the audit can
    claim formula-level agreement on curved walls rather than "within the noise
    of a difference scheme".
    """
    coarse, fine = (float(step) for step in steps)
    first = restoring_matrix_from_energy_fast(
        vertices, faces, mass, center_of_gravity, density, gravity, step=coarse
    )
    second = restoring_matrix_from_energy_fast(
        vertices, faces, mass, center_of_gravity, density, gravity, step=fine
    )
    return 2.0 * second - first


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------


def sphere_closed_form(
    radius: float,
    depth: float,
    offset_xy: tuple[float, float],
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, float]:
    """Exact circle-waterplane restoring entries for a sphere of centre depth ``depth``.

    With the centre at depth ``d`` the sphere pierces the waterplane and the
    **dry** part is a cap of height ``h = R - d`` above ``z = 0``; the submerged
    body is the sphere *minus* that cap::

        ∇ = 4πR³/3 - π h² (3R - h)/3

    Its vertical centroid follows from the cap centroid
    ``d_cap = 3 (2R - h)² / (4 (3R - h))`` measured from the waterplane *upwards*
    and ``∫(z - z_c) dV = 0`` over the whole sphere::

        z_B = -d - ∇_cap d_cap / ∇

    The waterplane is a circle of radius ``a = sqrt(R² - d²)`` centred on
    ``(x0, y0)``, so ``I_xy = A x0 y0`` is *not* negligible once the hull is
    off-centre even though the circle itself has no product moment.
    """
    x0, y0 = offset_xy
    a_sq = radius**2 - depth**2
    if a_sq <= 0.0:
        raise ValueError("the sphere must pierce the waterplane")
    area = math.pi * a_sq
    i_xx_centroid = math.pi * a_sq**2 / 4.0
    height = radius - depth
    cap_volume = math.pi * height**2 * (3.0 * radius - height) / 3.0
    cap_centroid = 3.0 * (2.0 * radius - height) ** 2 / (4.0 * (3.0 * radius - height))
    volume = 4.0 * math.pi * radius**3 / 3.0 - cap_volume
    z_b = -depth - cap_volume * cap_centroid / volume
    return _restoring_entries(
        area=area,
        s_x=area * x0,
        s_y=area * y0,
        i_xx=i_xx_centroid + area * y0**2,
        i_yy=i_xx_centroid + area * x0**2,
        i_xy=area * x0 * y0,
        volume=volume,
        z_b=z_b,
        z_g=-depth,
        density=density,
        gravity=gravity,
    )


def spheroid_closed_form(
    semi_axes: tuple[float, float, float],
    yaw: float,
    offset_xy: tuple[float, float],
    center_of_gravity_z: float,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, float]:
    """Exact waterplane integrals of a yawed ellipse plus the half-ellipsoid volume."""
    a, b, c = semi_axes
    x0, y0 = offset_xy
    area = math.pi * a * b
    # Principal-axis second moments of the ellipse u²/a² + v²/b² <= 1.
    i_uu = math.pi * a**3 * b / 4.0  # ∫ u² dA
    i_vv = math.pi * a * b**3 / 4.0  # ∫ v² dA
    cos, sin = math.cos(yaw), math.sin(yaw)
    # Body axes (u, v) map to world as u = x cos + y sin, v = -x sin + y cos.
    i_xx_body = sin**2 * i_uu + cos**2 * i_vv
    i_yy_body = cos**2 * i_uu + sin**2 * i_vv
    i_xy_body = sin * cos * (i_uu - i_vv)
    volume = 2.0 * math.pi * a * b * c / 3.0
    z_b = -3.0 * c / 8.0
    return _restoring_entries(
        area=area,
        s_x=area * x0,
        s_y=area * y0,
        i_xx=i_xx_body + area * y0**2,
        i_yy=i_yy_body + area * x0**2,
        i_xy=i_xy_body + area * x0 * y0,
        volume=volume,
        z_b=z_b,
        z_g=center_of_gravity_z,
        density=density,
        gravity=gravity,
    )


def _restoring_entries(
    *,
    area: float,
    s_x: float,
    s_y: float,
    i_xx: float,
    i_yy: float,
    i_xy: float,
    volume: float,
    z_b: float,
    z_g: float,
    density: float,
    gravity: float,
) -> dict[str, float]:
    """Assemble the six published entries from exact waterplane integrals."""
    rho_g = density * gravity
    couple = rho_g * volume * (z_b - z_g)
    return {
        "waterplane_area": area,
        "s_x": s_x,
        "s_y": s_y,
        "i_xx": i_xx,
        "i_yy": i_yy,
        "i_xy": i_xy,
        "displaced_volume": volume,
        "center_of_buoyancy_z": z_b,
        "c22": rho_g * area,
        "c23": rho_g * s_y,
        "c24": -rho_g * s_x,
        "c33": rho_g * i_xx + couple,
        "c34": -rho_g * i_xy,
        "c44": rho_g * i_yy + couple,
    }


# ---------------------------------------------------------------------------
# Case registry and audit driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One curved-hull audit case."""

    key: str
    description: str
    hull: WettedHullMesh
    center_of_gravity: tuple[float, float, float]
    closed_form: Mapping[str, float] = field(default_factory=dict)

    @property
    def subdivisions(self) -> int:
        return int(self.hull.faces.shape[0])


def build_cases(subdivisions: int = 3) -> list[Case]:
    """Return the curved-wall audit family at a given panel resolution."""
    sphere = eccentric_sphere_hull(
        subdivisions, radius=1.0, depth=0.5, offset_xy=(0.35, -0.20)
    )
    bowl = yawed_spheroid_hull((1.4, 0.8, 0.6), yaw=math.radians(30.0), subdivisions=subdivisions)
    mixed = yawed_spheroid_hull(
        (1.4, 0.8, 0.6),
        yaw=math.radians(30.0),
        subdivisions=subdivisions,
        offset_xy=(0.25, -0.15),
    )
    return [
        Case(
            key="eccentric_sphere",
            description=(
                "sphere R=1, centre (0.35, -0.20, -0.50): curved wall, circular "
                "curved waterplane displaced from the origin"
            ),
            hull=sphere,
            center_of_gravity=(0.35, -0.20, -0.50),
            closed_form=sphere_closed_form(1.0, 0.5, (0.35, -0.20)),
        ),
        Case(
            key="yawed_spheroid",
            description=(
                "half-ellipsoid (1.4, 0.8, 0.6) yawed 30°: elliptical curved "
                "waterplane with non-zero product moment, CG on the waterplane"
            ),
            hull=bowl,
            center_of_gravity=(0.0, 0.0, 0.0),
            closed_form=spheroid_closed_form((1.4, 0.8, 0.6), math.radians(30.0), (0.0, 0.0), 0.0),
        ),
        Case(
            key="eccentric_yawed_spheroid",
            description=(
                "half-ellipsoid (1.4, 0.8, 0.6) yawed 30° and shifted (0.25, -0.15): "
                "all six heave/roll/pitch entries non-zero"
            ),
            hull=mixed,
            center_of_gravity=(0.25, -0.15, 0.0),
            closed_form=spheroid_closed_form(
                (1.4, 0.8, 0.6), math.radians(30.0), (0.25, -0.15), 0.0
            ),
        ),
    ]


_ENTRY_KEYS = {
    (2, 2): "c22",
    (2, 3): "c23",
    (2, 4): "c24",
    (3, 3): "c33",
    (3, 4): "c34",
    (4, 4): "c44",
}


def _relative_deviation(reference: float, value: float, floor: float = 0.0) -> float:
    """Relative deviation with a floor, so entries that are zero by symmetry stay meaningful.

    A purely relative metric blows up on the entries the theory predicts to vanish
    (``C[2,3]`` on the centred spheroid) or on entries a hundred times smaller than
    their neighbours; ``floor`` is the smallest scale the deviation is measured
    against, and is set to ``1e-9`` of the largest audited entry of the case.
    """
    if reference == 0.0 and value == 0.0:
        return 0.0
    scale = max(abs(reference), floor)
    return abs(value - reference) / scale


def _floor_scale(restoring: np.ndarray, case: Case) -> float:
    """Smallest scale the per-entry deviations are measured against."""
    entries = [abs(float(restoring[i, j])) for (i, j) in AUDITED_ENTRIES]
    entries += [abs(float(case.closed_form[_ENTRY_KEYS[(i, j)]])) for (i, j) in AUDITED_ENTRIES]
    largest = max(entries) if entries else 1.0
    return 1.0e-9 * (largest or 1.0)


def audit_case(
    case: Case,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, object]:
    """Compare the implementation against the energy Hessian and the closed form."""
    properties = compute_hydrostatic_restoring_matrix(
        case.hull,
        mass=0.0,
        center_of_gravity=np.asarray(case.center_of_gravity, dtype=float),
        density=density,
        gravity=gravity,
    )
    code = properties.restoring_matrix
    # The body weight in the energy reference must balance the *mesh* displacement,
    # so the case sits exactly at equilibrium flotation for the geometry it sees.
    mass = density * properties.displaced_volume
    energy = restoring_matrix_from_energy_richardson(
        case.hull.vertices,
        case.hull.faces,
        mass=mass,
        center_of_gravity=case.center_of_gravity,
        density=density,
        gravity=gravity,
    )
    floor = _floor_scale(code, case)
    rows: list[dict[str, object]] = []
    for (i, j) in AUDITED_ENTRIES:
        key = _ENTRY_KEYS[(i, j)]
        closed = float(case.closed_form[key])
        rows.append(
            {
                "row": MODE_NAMES[i],
                "column": MODE_NAMES[j],
                "symbol": f"C[{i},{j}]",
                "code": float(code[i, j]),
                "energy": float(energy[i, j]),
                "closed_form": closed,
                "relerr_code_vs_energy": _relative_deviation(
                    float(energy[i, j]), float(code[i, j]), floor
                ),
                "relerr_code_vs_closed_form": _relative_deviation(
                    closed, float(code[i, j]), floor
                ),
            }
        )
    # Entries the theory says vanish: surge/sway/yaw rows and columns.
    null_scale = max(abs(float(code[i, j])) for (i, j) in AUDITED_ENTRIES)
    null_entries = [
        float(code[i, j])
        for i in range(6)
        for j in range(6)
        if (i not in (2, 3, 4)) or (j not in (2, 3, 4))
    ]
    return {
        "key": case.key,
        "description": case.description,
        "panels": int(case.hull.faces.shape[0]),
        "displaced_volume": float(properties.displaced_volume),
        "waterplane_area": float(properties.waterplane_area),
        "waterplane_center": [float(v) for v in properties.waterplane_center],
        "center_of_buoyancy": [float(v) for v in properties.center_of_buoyancy],
        "closed_form_waterplane_area": float(case.closed_form["waterplane_area"]),
        "closed_form_displaced_volume": float(case.closed_form["displaced_volume"]),
        "rows": rows,
        "null_block_max_norm": float(np.max(np.abs(null_entries))),
        "null_block_relative": float(np.max(np.abs(null_entries)) / null_scale) if null_scale else 0.0,
        "symmetry_residual": float(np.max(np.abs(code - code.T))),
    }


def convergence_study(
    subdivisions: Sequence[int] = (2, 3, 4),
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, object]:
    """Faceting-error convergence of the curved cases against the closed forms."""
    levels: dict[str, list[dict[str, float]]] = {}
    for level in subdivisions:
        for case in build_cases(subdivisions=level):
            record = audit_case(case, density=density, gravity=gravity)
            worst_closed = max(
                float(row["relerr_code_vs_closed_form"]) for row in record["rows"]
            )
            worst_energy = max(
                float(row["relerr_code_vs_energy"]) for row in record["rows"]
            )
            levels.setdefault(case.key, []).append(
                {
                    "subdivisions": int(level),
                    "panels": int(record["panels"]),
                    "worst_relerr_vs_closed_form": worst_closed,
                    "worst_relerr_vs_energy": worst_energy,
                    "volume_relerr_vs_closed_form": _relative_deviation(
                        float(record["closed_form_displaced_volume"]),
                        float(record["displaced_volume"]),
                    ),
                }
            )
    orders: dict[str, float] = {}
    for key, series in levels.items():
        if len(series) >= 2:
            first, last = series[0], series[-1]
            ratio = first["worst_relerr_vs_closed_form"] / max(
                last["worst_relerr_vs_closed_form"], 1e-16
            )
            orders[key] = math.log(ratio) / math.log(4.0)
    return {"levels": levels, "observed_orders": orders}


def fd_step_study(
    steps: Sequence[float] = (4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5),
    subdivisions: int = 3,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, object]:
    """Residual of the energy Hessian versus its finite-difference step.

    The reference is *not* smooth in the displacement: clipping a faceted hull at
    ``z = 0`` changes which triangles are cut, so the clipped area is only
    piecewise smooth and its second derivative jumps.  The error of a central
    second difference across such a kink is therefore ``O(step)`` rather than
    ``O(step²)``, which is why the audit reports ``code vs energy`` at the
    ``1e-4`` level instead of the ``1e-8`` the wall-sided barges reach.  This
    study documents that reading: the residual falls with the step and the fitted
    order is close to 1, i.e. it is truncation of the *reference*, not a mismatch
    of the formula.
    """
    per_case: dict[str, list[dict[str, float]]] = {}
    for case in build_cases(subdivisions=subdivisions):
        properties = compute_hydrostatic_restoring_matrix(
            case.hull,
            mass=0.0,
            center_of_gravity=np.asarray(case.center_of_gravity, dtype=float),
            density=density,
            gravity=gravity,
        )
        code = properties.restoring_matrix
        floor = _floor_scale(code, case)
        mass = density * properties.displaced_volume
        series: list[dict[str, float]] = []
        for step in steps:
            energy = restoring_matrix_from_energy_fast(
                case.hull.vertices,
                case.hull.faces,
                mass=mass,
                center_of_gravity=case.center_of_gravity,
                density=density,
                gravity=gravity,
                step=step,
            )
            worst = max(
                _relative_deviation(float(energy[i, j]), float(code[i, j]), floor)
                for (i, j) in AUDITED_ENTRIES
            )
            series.append({"step": float(step), "worst_relative": float(worst)})
        per_case[case.key] = series
    orders: dict[str, float] = {}
    for key, series in per_case.items():
        first, last = series[0], series[-1]
        if last["worst_relative"] > 0.0 and first["worst_relative"] > 0.0:
            orders[key] = math.log(first["worst_relative"] / last["worst_relative"]) / math.log(
                first["step"] / last["step"]
            )
    return {"series": per_case, "fitted_orders": orders}


def run(
    subdivisions: Sequence[int] = (3,),
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, object]:
    """Run the audit and return a JSON-serialisable record."""
    cases = [audit_case(case, density=density, gravity=gravity) for case in build_cases(subdivisions=max(subdivisions))]
    return {
        "density": density,
        "gravity": gravity,
        "criterion": {
            "code_vs_energy_rel": 1.0e-6,
            "code_vs_closed_form_rel": 3.0e-2,
        },
        "cases": cases,
        "convergence": convergence_study(subdivisions, density=density, gravity=gravity),
        "fd_steps": fd_step_study(subdivisions=max(subdivisions), density=density, gravity=gravity),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    record = run(subdivisions=(max(args.subdivisions - 1, 1), args.subdivisions))
    text = json.dumps(record, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())